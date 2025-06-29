#!/usr/bin/env python3
"""
Claim Analyzer Agent for MCP Server
Integrates with existing Neo4j/Qdrant hybrid database system for claim analysis and fact-checking
"""

import asyncio
import logging
import json
import re
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import yaml
from pathlib import Path

import aiohttp
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import AsyncGraphDatabase
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Claim:
    """Represents a claim to be fact-checked"""
    claim_id: str
    text: str
    source: str
    domain: str
    timestamp: datetime
    confidence: float = 0.0
    context: str = ""
    entities: List[str] = None
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.md5(f"{self.text}{self.source}".encode()).hexdigest()
        if self.entities is None:
            self.entities = []

@dataclass
class Evidence:
    """Represents evidence for or against a claim"""
    evidence_id: str
    text: str
    source_url: str
    credibility_score: float
    stance: str  # "supports", "refutes", "neutral"
    domain: str
    timestamp: datetime
    vector_embedding: Optional[np.ndarray] = None

@dataclass
class FactCheckResult:
    """Represents the result of a fact-check"""
    claim: Claim
    verdict: str  # "True", "False", "Partially True", "Unverified", "Opinion"
    confidence: float
    evidence_list: List[Evidence]
    reasoning: str
    sources: List[str]
    cross_domain_patterns: List[str]
    timestamp: datetime
    graph_node_id: Optional[str] = None

class DatabaseConnector:
    """Manages connections to Neo4j, Qdrant, and Redis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neo4j_driver = None
        self.qdrant_client = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize all database connections"""
        # Neo4j connection
        neo4j_config = self.config.get('database', {}).get('neo4j', {})
        self.neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_config.get('uri', 'bolt://localhost:7687'),
            auth=(
                neo4j_config.get('user', 'neo4j'),
                neo4j_config.get('password', 'password')
            ),
            max_connection_pool_size=neo4j_config.get('max_pool_size', 20)
        )
        
        # Qdrant connection
        qdrant_config = self.config.get('database', {}).get('qdrant', {})
        self.qdrant_client = qdrant_client.QdrantClient(
            host=qdrant_config.get('host', 'localhost'),
            port=qdrant_config.get('port', 6333),
            timeout=qdrant_config.get('timeout', 30)
        )
        
        # Redis connection
        redis_config = self.config.get('database', {}).get('redis', {})
        self.redis_client = redis.from_url(
            redis_config.get('url', 'redis://localhost:6379'),
            max_connections=redis_config.get('max_connections', 50)
        )
        
        # Initialize collections if they don't exist
        await self._initialize_qdrant_collections()
        
        logger.info("Database connections initialized successfully")
    
    async def _initialize_qdrant_collections(self):
        """Initialize Qdrant collections for claims and evidence"""
        collections = [
            {
                'name': 'claims',
                'vector_size': 384,  # all-MiniLM-L6-v2 embedding size
                'distance': Distance.COSINE
            },
            {
                'name': 'evidence',
                'vector_size': 384,
                'distance': Distance.COSINE
            }
        ]
        
        for collection in collections:
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection['name'],
                    vectors_config=VectorParams(
                        size=collection['vector_size'],
                        distance=collection['distance']
                    )
                )
                logger.info(f"Created Qdrant collection: {collection['name']}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Qdrant collection already exists: {collection['name']}")
                else:
                    logger.error(f"Error creating collection {collection['name']}: {e}")
    
    async def close(self):
        """Close all database connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.redis_client:
            await self.redis_client.close()

class ClaimExtractor:
    """Extracts claims from various text sources using NLP"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Claim detection patterns
        self.claim_patterns = [
            r"(?:is|are|was|were|will be|has been|have been)\s+(?:a|an|the)?\s*\w+",
            r"(?:claims?|states?|argues?|believes?|says?)\s+that\s+.*",
            r"(?:according to|research shows|studies indicate|experts say)\s+.*",
            r"(?:it is|this is|that is)\s+(?:true|false|correct|incorrect|a fact)\s+that\s+.*"
        ]
        
        # Domain classification keywords
        self.domain_keywords = {
            'science': ['experiment', 'study', 'research', 'theory', 'hypothesis', 'data', 'evidence'],
            'math': ['theorem', 'proof', 'equation', 'formula', 'calculate', 'number', 'geometry'],
            'religion': ['god', 'faith', 'belief', 'scripture', 'church', 'prayer', 'divine'],
            'history': ['ancient', 'war', 'civilization', 'empire', 'century', 'historical', 'period'],
            'literature': ['novel', 'poem', 'author', 'book', 'story', 'character', 'literary'],
            'philosophy': ['ethics', 'logic', 'metaphysics', 'epistemology', 'moral', 'existence']
        }
    
    async def extract_claims(self, text: str, source: str = "unknown", domain: str = "general") -> List[Claim]:
        """Extract verifiable claims from text"""
        if not text.strip():
            return []
        
        claims = []
        
        # Auto-detect domain if not specified
        if domain == "general":
            domain = self._classify_domain(text)
        
        if self.nlp:
            claims = await self._extract_with_spacy(text, source, domain)
        else:
            claims = await self._extract_with_patterns(text, source, domain)
        
        # Store claims in Neo4j and Qdrant
        for claim in claims:
            await self._store_claim(claim)
        
        return claims
    
    def _classify_domain(self, text: str) -> str:
        """Classify text into one of the six domains"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if all scores are 0
        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    async def _extract_with_spacy(self, text: str, source: str, domain: str) -> List[Claim]:
        """Extract claims using spaCy NLP"""
        claims = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Skip questions, commands, and very short sentences
            if len(sent_text) < 10 or sent_text.endswith('?') or sent_text.startswith('/'):
                continue
            
            # Check if sentence contains claim indicators
            if self._is_claim_sentence(sent_text):
                confidence = self._calculate_claim_confidence(sent_text, sent)
                
                if confidence > 0.3:  # Threshold for claim detection
                    # Extract entities
                    entities = [ent.text for ent in sent.ents]
                    
                    claim = Claim(
                        claim_id="",
                        text=sent_text,
                        source=source,
                        domain=domain,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        context=self._get_context(sent, doc),
                        entities=entities
                    )
                    claims.append(claim)
        
        return claims
    
    async def _extract_with_patterns(self, text: str, source: str, domain: str) -> List[Claim]:
        """Fallback pattern-based claim extraction"""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            if self._is_claim_sentence(sentence):
                claim = Claim(
                    claim_id="",
                    text=sentence,
                    source=source,
                    domain=domain,
                    timestamp=datetime.now(),
                    confidence=0.5,  # Default confidence for pattern-based
                    context=sentence,
                    entities=[]
                )
                claims.append(claim)
        
        return claims
    
    def _is_claim_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a verifiable claim"""
        sentence_lower = sentence.lower()
        
        # Check for claim patterns
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # Check for factual statements
        factual_indicators = [
            'fact', 'truth', 'evidence', 'study', 'research', 'data',
            'statistics', 'proven', 'confirmed', 'verified', 'documented'
        ]
        
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
        
        # Check for absolute statements
        absolute_terms = ['always', 'never', 'all', 'none', 'every', 'no one', 'everyone']
        if any(term in sentence_lower for term in absolute_terms):
            return True
        
        return False
    
    def _calculate_claim_confidence(self, sentence: str, spacy_sent) -> float:
        """Calculate confidence that this sentence contains a verifiable claim"""
        confidence = 0.0
        
        # Check for verbs indicating factual statements
        factual_verbs = {'be', 'have', 'do', 'say', 'state', 'claim', 'prove', 'show'}
        for token in spacy_sent:
            if token.lemma_ in factual_verbs and token.pos_ == 'VERB':
                confidence += 0.3
        
        # Check for named entities (more likely to be factual)
        if len([ent for ent in spacy_sent.ents]) > 0:
            confidence += 0.2
        
        # Check for numbers/dates (often factual)
        if any(token.like_num or token.ent_type_ in ['DATE', 'TIME', 'PERCENT', 'MONEY'] 
               for token in spacy_sent):
            confidence += 0.2
        
        # Check sentence structure
        if len(sentence.split()) > 5:  # Substantial sentences more likely to contain claims
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_context(self, sentence, doc) -> str:
        """Get surrounding context for a claim"""
        sent_start = sentence.start
        sent_end = sentence.end
        
        # Get previous and next sentence if available
        context_tokens = []
        
        # Previous sentence
        prev_sent_tokens = [token for token in doc if token.i < sent_start]
        if prev_sent_tokens:
            context_tokens.extend(prev_sent_tokens[-20:])  # Last 20 tokens
        
        # Current sentence
        context_tokens.extend([token for token in sentence])
        
        # Next sentence
        next_sent_tokens = [token for token in doc if token.i >= sent_end]
        if next_sent_tokens:
            context_tokens.extend(next_sent_tokens[:20])  # First 20 tokens
        
        return ' '.join([token.text for token in context_tokens])
    
    async def _store_claim(self, claim: Claim):
        """Store claim in Neo4j graph and Qdrant vector database"""
        try:
            # Generate embedding for the claim
            embedding = self.sentence_model.encode(claim.text)
            
            # Store in Neo4j
            async with self.db_connector.neo4j_driver.session() as session:
                query = """
                CREATE (c:Claim {
                    id: $claim_id,
                    text: $text,
                    source: $source,
                    domain: $domain,
                    timestamp: $timestamp,
                    confidence: $confidence,
                    context: $context,
                    entities: $entities
                })
                RETURN c.id as node_id
                """
                
                result = await session.run(query, {
                    'claim_id': claim.claim_id,
                    'text': claim.text,
                    'source': claim.source,
                    'domain': claim.domain,
                    'timestamp': claim.timestamp.isoformat(),
                    'confidence': claim.confidence,
                    'context': claim.context,
                    'entities': claim.entities
                })
                
                record = await result.single()
                if record:
                    logger.info(f"Stored claim in Neo4j: {claim.claim_id}")
            
            # Store in Qdrant
            point = PointStruct(
                id=claim.claim_id,
                vector=embedding.tolist(),
                payload={
                    'text': claim.text,
                    'source': claim.source,
                    'domain': claim.domain,
                    'timestamp': claim.timestamp.isoformat(),
                    'confidence': claim.confidence,
                    'entities': claim.entities
                }
            )
            
            self.db_connector.qdrant_client.upsert(
                collection_name='claims',
                points=[point]
            )
            logger.info(f"Stored claim in Qdrant: {claim.claim_id}")
            
            # Cache in Redis for quick access
            cache_key = f"claim:{claim.claim_id}"
            await self.db_connector.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(asdict(claim), default=str)
            )
            
        except Exception as e:
            logger.error(f"Error storing claim {claim.claim_id}: {e}")

class FactChecker:
    """Fact-checks claims using hybrid database search and external APIs"""
    
    def __init__(self, db_connector: DatabaseConnector, config: Dict[str, Any]):
        self.db_connector = db_connector
        self.config = config
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Source credibility scores
        self.source_credibility = config.get('source_credibility', {
            'wikipedia.org': 0.8,
            'snopes.com': 0.9,
            'factcheck.org': 0.9,
            'politifact.com': 0.9,
            'reuters.com': 0.9,
            'bbc.com': 0.8,
            'nasa.gov': 0.95,
            'cdc.gov': 0.95,
        })
        
        # Rate limiting
        self.last_api_call = {}
        self.api_call_intervals = {
            'web_search': 1.0,
            'fact_check_api': 2.0,
        }
    
    async def fact_check_claim(self, claim: Claim) -> FactCheckResult:
        """Perform comprehensive fact-checking on a claim"""
        logger.info(f"Fact-checking claim: {claim.text[:100]}...")
        
        try:
            # Search for similar claims in the database
            similar_claims = await self._search_similar_claims(claim)
            
            # Search for evidence across domains
            evidence_list = await self._search_evidence(claim)
            
            # Cross-domain pattern analysis
            patterns = await self._analyze_cross_domain_patterns(claim)
            
            # Analyze all evidence to determine verdict
            verdict, confidence, reasoning = self._analyze_evidence(claim, evidence_list, similar_claims)
            
            # Extract source URLs
            sources = [evidence.source_url for evidence in evidence_list]
            
            result = FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence_list=evidence_list,
                reasoning=reasoning,
                sources=sources,
                cross_domain_patterns=patterns,
                timestamp=datetime.now()
            )
            
            # Store result in the graph
            await self._store_fact_check_result(result)
            
            logger.info(f"Fact-check complete: {verdict} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error fact-checking claim: {str(e)}")
            return FactCheckResult(
                claim=claim,
                verdict="Error",
                confidence=0.0,
                evidence_list=[],
                reasoning=f"Error during fact-checking: {str(e)}",
                sources=[],
                cross_domain_patterns=[],
                timestamp=datetime.now()
            )
    
    async def _search_similar_claims(self, claim: Claim) -> List[Dict]:
        """Search for similar claims in Qdrant using vector similarity"""
        try:
            # Generate embedding for the claim
            query_embedding = self.sentence_model.encode(claim.text)
            
            # Search in Qdrant
            search_results = self.db_connector.qdrant_client.search(
                collection_name='claims',
                query_vector=query_embedding.tolist(),
                limit=10,
                score_threshold=0.7  # Similarity threshold
            )
            
            similar_claims = []
            for result in search_results:
                similar_claims.append({
                    'claim_id': result.id,
                    'text': result.payload.get('text', ''),
                    'similarity_score': result.score,
                    'domain': result.payload.get('domain', ''),
                    'source': result.payload.get('source', '')
                })
            
            return similar_claims
            
        except Exception as e:
            logger.error(f"Error searching similar claims: {e}")
            return []
    
    async def _search_evidence(self, claim: Claim) -> List[Evidence]:
        """Search for evidence across multiple sources and domains"""
        evidence_list = []
        
        # Search strategies
        strategies = [
            self._search_graph_evidence,
            self._search_vector_evidence,
            self._search_external_apis
        ]
        
        for strategy in strategies:
            try:
                results = await strategy(claim)
                evidence_list.extend(results)
            except Exception as e:
                logger.warning(f"Evidence search strategy failed: {str(e)}")
        
        # Remove duplicates and limit results
        unique_evidence = []
        seen_urls = set()
        
        for evidence in evidence_list:
            if evidence.source_url not in seen_urls:
                unique_evidence.append(evidence)
                seen_urls.add(evidence.source_url)
        
        return unique_evidence[:self.config.get('max_results', 10)]
    
    async def _search_graph_evidence(self, claim: Claim) -> List[Evidence]:
        """Search for evidence using Neo4j graph relationships"""
        evidence_list = []
        
        try:
            async with self.db_connector.neo4j_driver.session() as session:
                # Search for related documents, concepts, and entities
                query = """
                MATCH (c:Claim {id: $claim_id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document)
                WHERE d.domain = $domain OR d.domain IN ['general', 'cross-domain']
                OPTIONAL MATCH (d)-[:CONTAINS]->(concept:Concept)
                OPTIONAL MATCH (concept)-[:RELATED_TO]->(related_concept:Concept)
                OPTIONAL MATCH (related_concept)<-[:CONTAINS]-(evidence_doc:Document)
                RETURN DISTINCT evidence_doc.url as url, evidence_doc.title as title, 
                       evidence_doc.content as content, evidence_doc.credibility as credibility,
                       evidence_doc.domain as domain
                LIMIT 20
                """
                
                result = await session.run(query, {
                    'claim_id': claim.claim_id,
                    'domain': claim.domain
                })
                
                async for record in result:
                    if record['url']:
                        evidence = Evidence(
                            evidence_id=hashlib.md5(record['url'].encode()).hexdigest(),
                            text=record['content'] or record['title'] or '',
                            source_url=record['url'],
                            credibility_score=record['credibility'] or 0.7,
                            stance='neutral',  # Will be determined by analysis
                            domain=record['domain'] or claim.domain,
                            timestamp=datetime.now()
                        )
                        evidence_list.append(evidence)
        
        except Exception as e:
            logger.error(f"Error searching graph evidence: {e}")
        
        return evidence_list
    
    async def _search_vector_evidence(self, claim: Claim) -> List[Evidence]:
        """Search for evidence using Qdrant vector similarity"""
        evidence_list = []
        
        try:
            # Generate embedding for the claim
            query_embedding = self.sentence_model.encode(claim.text)
            
            # Search in evidence collection
            search_results = self.db_connector.qdrant_client.search(
                collection_name='evidence',
                query_vector=query_embedding.tolist(),
                limit=15,
                score_threshold=0.6
            )
            
            for result in search_results:
                evidence = Evidence(
                    evidence_id=result.id,
                    text=result.payload.get('text', ''),
                    source_url=result.payload.get('source_url', ''),
                    credibility_score=result.payload.get('credibility_score', 0.7),
                    stance=result.payload.get('stance', 'neutral'),
                    domain=result.payload.get('domain', claim.domain),
                    timestamp=datetime.now()
                )
                evidence_list.append(evidence)
        
        except Exception as e:
            logger.error(f"Error searching vector evidence: {e}")
        
        return evidence_list
    
    async def _search_external_apis(self, claim: Claim) -> List[Evidence]:
        """Search external fact-checking APIs"""
        evidence_list = []
        
        # Rate limiting
        await self._rate_limit('fact_check_api')
        
        # Mock external API responses based on common claims
        claim_lower = claim.text.lower()
        
        if 'moon landing' in claim_lower and any(term in claim_lower for term in ['fake', 'hoax', 'staged']):
            evidence = Evidence(
                evidence_id='nasa_moon_landing_evidence',
                text='NASA provides extensive documentation of Apollo moon landings including lunar rocks, photos, and independent verification by multiple countries.',
                source_url='https://www.nasa.gov/mission_pages/apollo/apollo11.html',
                credibility_score=0.95,
                stance='refutes',
                domain='science',
                timestamp=datetime.now()
            )
            evidence_list.append(evidence)
        
        elif 'earth' in claim_lower and 'flat' in claim_lower:
            evidence = Evidence(
                evidence_id='earth_shape_evidence',
                text='Scientific evidence from satellite imagery, physics, and multiple observation methods confirms Earth is an oblate spheroid.',
                source_url='https://example.com/earth-shape-evidence',
                credibility_score=0.9,
                stance='refutes',
                domain='science',
                timestamp=datetime.now()
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    async def _analyze_cross_domain_patterns(self, claim: Claim) -> List[str]:
        """Analyze cross-domain patterns related to the claim"""
        patterns = []
        
        try:
            async with self.db_connector.neo4j_driver.session() as session:
                # Look for cross-domain concept relationships
                query = """
                MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
                MATCH (e)<-[:MENTIONS]-(d1:Document {domain: $domain})
                MATCH (e)<-[:MENTIONS]-(d2:Document)
                WHERE d2.domain <> $domain
                MATCH (d1)-[:CONTAINS]->(concept1:Concept)
                MATCH (d2)-[:CONTAINS]->(concept2:Concept)
                WHERE concept1.name = concept2.name OR 
                      EXISTS((concept1)-[:SIMILAR_TO]-(concept2))
                RETURN DISTINCT concept1.name as pattern, 
                       collect(DISTINCT d2.domain) as other_domains
                LIMIT 10
                """
                
                result = await session.run(query, {
                    'claim_id': claim.claim_id,
                    'domain': claim.domain
                })
                
                async for record in result:
                    pattern_desc = f"{record['pattern']} appears across {claim.domain} and {', '.join(record['other_domains'])}"
                    patterns.append(pattern_desc)
        
        except Exception as e:
            logger.error(f"Error analyzing cross-domain patterns: {e}")
        
        return patterns
    
    def _analyze_evidence(self, claim: Claim, evidence_list: List[Evidence], similar_claims: List[Dict]) -> Tuple[str, float, str]:
        """Analyze collected evidence to determine verdict"""
        if not evidence_list and not similar_claims:
            return "Unverified", 0.0, "No evidence found to verify this claim."
        
        # Analyze evidence stances
        support_score = 0.0
        refute_score = 0.0
        total_credibility = 0.0
        
        reasoning_parts = []
        
        # Analyze evidence
        for evidence in evidence_list:
            credibility = evidence.credibility_score
            total_credibility += credibility
            
            # Determine stance if not already set
            if evidence.stance == 'neutral':
                evidence.stance = self._determine_evidence_stance(claim.text, evidence.text)
            
            if evidence.stance == 'supports':
                support_score += credibility
            elif evidence.stance == 'refutes':
                refute_score += credibility
            
            reasoning_parts.append(f"Source: {evidence.source_url[:50]}... - {evidence.stance} (credibility: {credibility:.2f})")
        
        # Consider similar claims
        for similar_claim in similar_claims:
            similarity_weight = similar_claim['similarity_score'] * 0.5  # Reduced weight for similar claims
            # This would ideally look up the verdict of similar claims
            # For now, assume neutral impact
        
        # Determine overall verdict
        if not evidence_list:
            verdict = "Unverified"
            confidence = 0.0
        elif refute_score > support_score * 1.5:  # Strong refutation threshold
            verdict = "False"
            confidence = min(refute_score / (support_score + refute_score + 0.1), 0.95)
        elif support_score > refute_score * 1.5:  # Strong support threshold
            verdict = "True"
            confidence = min(support_score / (support_score + refute_score + 0.1), 0.95)
        elif abs(support_score - refute_score) < 0.3:
            verdict = "Partially True"
            confidence = 0.6
        else:
            verdict = "Unverified"
            confidence = 0.3
        
        # Adjust confidence based on evidence quality
        if len(evidence_list) < 2:
            confidence *= 0.7  # Reduce confidence with limited evidence
        
        avg_credibility = total_credibility / len(evidence_list) if evidence_list else 0.0
        confidence = min(confidence * avg_credibility, 1.0)
        
        reasoning = f"Based on {len(evidence_list)} evidence sources and {len(similar_claims)} similar claims. " + "; ".join(reasoning_parts[:3])
        
        return verdict, confidence, reasoning
    
    def _determine_evidence_stance(self, claim_text: str, evidence_text: str) -> str:
        """Determine if evidence supports, refutes, or is neutral to the claim"""
        evidence_lower = evidence_text.lower()
        claim_lower = claim_text.lower()
        
        # Simple keyword-based stance detection
        support_indicators = ['true', 'correct', 'accurate', 'confirmed', 'verified', 'proven']
        refute_indicators = ['false', 'incorrect', 'debunked', 'disproven', 'myth', 'hoax']
        
        support_count = sum(1 for indicator in support_indicators if indicator in evidence_lower)
        refute_count = sum(1 for indicator in refute_indicators if indicator in evidence_lower)
        
        if refute_count > support_count:
            return 'refutes'
        elif support_count > refute_count:
            return 'supports'
        else:
            return 'neutral'
    
    async def _store_fact_check_result(self, result: FactCheckResult):
        """Store fact-check result in Neo4j"""
        try:
            async with self.db_connector.neo4j_driver.session() as session:
                query = """
                MATCH (c:Claim {id: $claim_id})
                CREATE (fr:FactCheckResult {
                    id: $result_id,
                    verdict: $verdict,
                    confidence: $confidence,
                    reasoning: $reasoning,
                    timestamp: $timestamp
                })
                CREATE (c)-[:HAS_FACT_CHECK]->(fr)
                RETURN fr.id as result_id
                """
                
                result_id = hashlib.md5(f"{result.claim.claim_id}{result.timestamp}".encode()).hexdigest()
                
                await session.run(query, {
                    'claim_id': result.claim.claim_id,
                    'result_id': result_id,
                    'verdict': result.verdict,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'timestamp': result.timestamp.isoformat()
                })
                
                result.graph_node_id = result_id
                logger.info(f"Stored fact-check result: {result_id}")
        
        except Exception as e:
            logger.error(f"Error storing fact-check result: {e}")
    
    async def _rate_limit(self, api_type: str):
        """Implement rate limiting for API calls"""
        if api_type in self.last_api_call:
            time_since_last = time.time() - self.last_api_call[api_type]
            required_interval = self.api_call_intervals.get(api_type, 1.0)
            
            if time_since_last < required_interval:
                sleep_time = required_interval - time_since_last
                await asyncio.sleep(sleep_time)
        
        self.last_api_call[api_type] = time.time()

class ClaimAnalyzerAgent:
    """Main Claim Analyzer Agent for MCP Server"""
    
    def __init__(self, config_path: str = "agents/claim_analyzer/config.yaml"):
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector(self.config)
        self.claim_extractor = None
        self.fact_checker = None
        
        # Agent state
        self.is_running = False
        self.processed_claims = 0
        self.fact_checks_performed = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        default_config = {
            'database': {
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'user': 'neo4j',
                    'password': 'password',
                    'max_pool_size': 20
                },
                'qdrant': {
                    'host': 'localhost',
                    'port': 6333,
                    'timeout': 30
                },
                'redis': {
                    'url': 'redis://localhost:6379',
                    'max_connections': 50
                }
            },
            'agent': {
                'max_results': 10,
                'confidence_threshold': 0.5,
                'batch_size': 50,
                'processing_interval': 300  # 5 minutes
            },
            'source_credibility': {
                'wikipedia.org': 0.8,
                'snopes.com': 0.9,
                'factcheck.org': 0.9,
                'politifact.com': 0.9,
                'reuters.com': 0.9,
                'bbc.com': 0.8,
                'nasa.gov': 0.95,
                'cdc.gov': 0.95,
            }
        }
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Deep merge with defaults
                def deep_merge(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            deep_merge(default[key], value)
                        else:
                            default[key] = value
                    return default
                
                return deep_merge(default_config, user_config)
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    async def initialize(self):
        """Initialize the agent and its components"""
        logger.info("Initializing Claim Analyzer Agent...")
        
        # Initialize database connections
        await self.db_connector.initialize()
        
        # Initialize components
        self.claim_extractor = ClaimExtractor(self.db_connector)
        self.fact_checker = FactChecker(self.db_connector, self.config)
        
        self.is_running = True
        logger.info("Claim Analyzer Agent initialized successfully")
    
    async def process_text(self, text: str, source: str = "unknown", domain: str = "general") -> Dict[str, Any]:
        """Process text to extract and fact-check claims"""
        logger.info(f"Processing text from source: {source}")
        
        # Extract claims
        claims = await self.claim_extractor.extract_claims(text, source, domain)
        self.processed_claims += len(claims)
        
        # Fact-check each claim
        results = []
        for claim in claims:
            result = await self.fact_checker.fact_check_claim(claim)
            results.append(asdict(result))
            self.fact_checks_performed += 1
        
        return {
            'total_claims': len(claims),
            'fact_check_results': results,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    async def fact_check_single_claim(self, claim_text: str, source: str = "manual", domain: str = "general") -> Dict[str, Any]:
        """Fact-check a single claim"""
        claim = Claim(
            claim_id="",
            text=claim_text,
            source=source,
            domain=domain,
            timestamp=datetime.now()
        )
        
        result = await self.fact_checker.fact_check_claim(claim)
        self.fact_checks_performed += 1
        
        return asdict(result)
    
    async def get_similar_claims(self, claim_text: str, limit: int = 5) -> List[Dict]:
        """Get similar claims from the database"""
        try:
            # Generate embedding
            embedding = self.claim_extractor.sentence_model.encode(claim_text)
            
            # Search in Qdrant
            search_results = self.db_connector.qdrant_client.search(
                collection_name='claims',
                query_vector=embedding.tolist(),
                limit=limit,
                score_threshold=0.6
            )
            
            similar_claims = []
            for result in search_results:
                similar_claims.append({
                    'claim_id': result.id,
                    'text': result.payload.get('text', ''),
                    'similarity_score': result.score,
                    'domain': result.payload.get('domain', ''),
                    'source': result.payload.get('source', ''),
                    'timestamp': result.payload.get('timestamp', '')
                })
            
            return similar_claims
        
        except Exception as e:
            logger.error(f"Error getting similar claims: {e}")
            return []
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'is_running': self.is_running,
            'processed_claims': self.processed_claims,
            'fact_checks_performed': self.fact_checks_performed,
            'database_status': await self._check_database_status()
        }
    
    async def _check_database_status(self) -> Dict[str, bool]:
        """Check status of all database connections"""
        status = {}
        
        try:
            # Check Neo4j
            async with self.db_connector.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            status['neo4j'] = True
        except Exception:
            status['neo4j'] = False
        
        try:
            # Check Qdrant
            collections = self.db_connector.qdrant_client.get_collections()
            status['qdrant'] = True
        except Exception:
            status['qdrant'] = False
        
        try:
            # Check Redis
            await self.db_connector.redis_client.ping()
            status['redis'] = True
        except Exception:
            status['redis'] = False
        
        return status
    
    async def shutdown(self):
        """Shutdown the agent gracefully"""
        logger.info("Shutting down Claim Analyzer Agent...")
        self.is_running = False
        await self.db_connector.close()
        logger.info("Claim Analyzer Agent shutdown complete")

# Example usage and integration with MCP server
async def main():
    """Example usage of the Claim Analyzer Agent"""
    agent = ClaimAnalyzerAgent()
    
    try:
        await agent.initialize()
        
        # Example 1: Process text with multiple claims
        sample_text = """
        The Earth is flat and NASA has been hiding this truth from us.
        Climate change is a natural phenomenon that has nothing to do with human activity.
        Vaccines are completely safe and have eliminated many deadly diseases.
        The Great Wall of China is visible from space with the naked eye.
        """
        
        print("Processing sample text...")
        results = await agent.process_text(sample_text, "sample_document", "science")
        print(f"Found {results['total_claims']} claims")
        
        for i, result in enumerate(results['fact_check_results'], 1):
            print(f"\nClaim {i}: {result['claim']['text'][:80]}...")
            print(f"Verdict: {result['verdict']} (confidence: {result['confidence']:.2f})")
            print(f"Evidence sources: {len(result['evidence_list'])}")
            if result['cross_domain_patterns']:
                print(f"Cross-domain patterns: {', '.join(result['cross_domain_patterns'])}")
        
        # Example 2: Fact-check single claim
        print("\n" + "="*80)
        print("Fact-checking single claim...")
        single_result = await agent.fact_check_single_claim(
            "The moon landing was filmed in a Hollywood studio",
            "user_input",
            "history"
        )
        
        print(f"Verdict: {single_result['verdict']}")
        print(f"Confidence: {single_result['confidence']:.2f}")
        print(f"Reasoning: {single_result['reasoning'][:200]}...")
        
        # Example 3: Find similar claims
        print("\n" + "="*80)
        print("Finding similar claims...")
        similar = await agent.get_similar_claims("The Earth is not round", limit=3)
        print(f"Found {len(similar)} similar claims:")
        for claim in similar:
            print(f"- {claim['text'][:60]}... (similarity: {claim['similarity_score']:.2f})")
        
        # Agent statistics
        print("\n" + "="*80)
        stats = await agent.get_agent_stats()
        print("Agent Statistics:")
        print(f"Claims processed: {stats['processed_claims']}")
        print(f"Fact-checks performed: {stats['fact_checks_performed']}")
        print(f"Database status: {stats['database_status']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
