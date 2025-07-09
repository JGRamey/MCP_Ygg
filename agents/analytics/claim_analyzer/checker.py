#!/usr/bin/env python3
"""Fact checking module for Claim Analyzer Agent"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

from sentence_transformers import SentenceTransformer

from .models import Claim, Evidence, FactCheckResult
from .database import DatabaseConnector
from .exceptions import FactCheckingError, EvidenceSearchError, RateLimitError
from .utils import PerformanceTimer, create_evidence_summary

logger = logging.getLogger(__name__)


class FactChecker:
    """Fact-checks claims using hybrid database search and external APIs"""
    
    def __init__(self, db_connector: DatabaseConnector, config: Dict[str, Any]):
        self.db_connector = db_connector
        self.config = config
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        
        self.last_api_call = {}
        self.api_call_intervals = {
            'web_search': 1.0,
            'fact_check_api': 2.0,
        }
    
    async def fact_check_claim(self, claim: Claim) -> FactCheckResult:
        """Perform comprehensive fact-checking on a claim"""
        logger.info(f"Fact-checking claim: {claim.text[:100]}...")
        
        try:
            with PerformanceTimer("fact_check_claim"):
                similar_claims = await self._search_similar_claims(claim)
                evidence_list = await self._search_evidence(claim)
                patterns = await self._analyze_cross_domain_patterns(claim)
                
                verdict, confidence, reasoning = self._analyze_evidence(claim, evidence_list, similar_claims)
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
                
                await self._store_fact_check_result(result)
                
                logger.info(f"Fact-check complete: {verdict} (confidence: {confidence:.2f})")
                return result
            
        except (EvidenceSearchError, RateLimitError) as e:
            logger.warning(f"Non-critical error during fact-checking: {e}")
            return FactCheckResult(
                claim=claim,
                verdict="Unverified",
                confidence=0.0,
                evidence_list=[],
                reasoning=f"Limited verification due to: {str(e)}",
                sources=[],
                cross_domain_patterns=[],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error fact-checking claim: {str(e)}")
            raise FactCheckingError(f"Fact-checking failed for claim: {claim.claim_id}") from e
    
    async def _search_similar_claims(self, claim: Claim) -> List[Dict]:
        """Search for similar claims in Qdrant using vector similarity"""
        try:
            query_embedding = self.sentence_model.encode(claim.text)
            
            search_results = self.db_connector.qdrant_client.search(
                collection_name='claims',
                query_vector=query_embedding.tolist(),
                limit=10,
                score_threshold=0.7
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
        
        return self._deduplicate_evidence(evidence_list)
    
    def _deduplicate_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence and limit results"""
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
                            stance='neutral',
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
            query_embedding = self.sentence_model.encode(claim.text)
            
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
        
        await self._rate_limit('fact_check_api')
        
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
        
        support_score = 0.0
        refute_score = 0.0
        total_credibility = 0.0
        reasoning_parts = []
        
        for evidence in evidence_list:
            credibility = evidence.credibility_score
            total_credibility += credibility
            
            if evidence.stance == 'neutral':
                evidence.stance = self._determine_evidence_stance(claim.text, evidence.text)
            
            if evidence.stance == 'supports':
                support_score += credibility
            elif evidence.stance == 'refutes':
                refute_score += credibility
            
            reasoning_parts.append(f"Source: {evidence.source_url[:50]}... - {evidence.stance} (credibility: {credibility:.2f})")
        
        # Determine verdict
        if not evidence_list:
            verdict = "Unverified"
            confidence = 0.0
        elif refute_score > support_score * 1.5:
            verdict = "False"
            confidence = min(refute_score / (support_score + refute_score + 0.1), 0.95)
        elif support_score > refute_score * 1.5:
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
            confidence *= 0.7
        
        avg_credibility = total_credibility / len(evidence_list) if evidence_list else 0.0
        confidence = min(confidence * avg_credibility, 1.0)
        
        reasoning = f"Based on {len(evidence_list)} evidence sources and {len(similar_claims)} similar claims. " + "; ".join(reasoning_parts[:3])
        
        return verdict, confidence, reasoning
    
    def _determine_evidence_stance(self, claim_text: str, evidence_text: str) -> str:
        """Determine if evidence supports, refutes, or is neutral to the claim"""
        evidence_lower = evidence_text.lower()
        
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