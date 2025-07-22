#!/usr/bin/env python3
"""
Enhanced Fact Checker with Multi-Source Verification and Explainability
Part of Phase 2 Enhanced AI Agents for MCP Yggdrasil
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sentence_transformers import SentenceTransformer

from .models import Claim, Evidence, FactCheckResult
from .database import DatabaseConnector
from .exceptions import FactCheckingError, EvidenceSearchError
from .utils import PerformanceTimer

logger = logging.getLogger(__name__)


@dataclass
class VerificationSource:
    """Represents a verification source with metadata"""
    name: str
    url: str
    credibility: float
    domain: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0


@dataclass
class ExplanationStep:
    """Represents a step in the fact-checking explanation"""
    step_number: int
    action: str
    source: str
    finding: str
    confidence_impact: float
    reasoning: str


@dataclass
class EnhancedFactCheckResult(FactCheckResult):
    """Extended fact-check result with explainability features"""
    explanation_steps: List[ExplanationStep]
    source_breakdown: Dict[str, float]
    confidence_breakdown: Dict[str, float]
    alternative_interpretations: List[str]
    methodology_used: List[str]
    quality_indicators: Dict[str, Any]


class EnhancedFactChecker:
    """
    Enhanced fact checker with multi-source verification and explainability.
    
    Features:
    - Multi-source cross-verification
    - Explainable AI decisions
    - Dynamic confidence scoring
    - Bias detection
    - Source credibility analysis
    """
    
    def __init__(self, db_connector: DatabaseConnector, config: Dict[str, Any]):
        self.db_connector = db_connector
        self.config = config
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize verification sources
        self.verification_sources = self._load_verification_sources()
        
        # Enhanced source credibility with bias indicators
        self.source_credibility = self._load_enhanced_credibility()
        
        # Explanation tracking
        self.explanation_steps: List[ExplanationStep] = []
        self.methodology_used: List[str] = []
        
        # Performance tracking
        self.verification_stats = {
            'sources_consulted': 0,
            'cross_references_found': 0,
            'consensus_level': 0.0,
            'processing_time': 0.0
        }
        
    def _load_verification_sources(self) -> List[VerificationSource]:
        """Load configured verification sources"""
        sources = [
            VerificationSource(
                name="Snopes",
                url="https://www.snopes.com",
                credibility=0.95,
                domain="general",
                rate_limit=2.0
            ),
            VerificationSource(
                name="FactCheck.org",
                url="https://www.factcheck.org",
                credibility=0.95,
                domain="politics",
                rate_limit=1.5
            ),
            VerificationSource(
                name="Wikipedia",
                url="https://en.wikipedia.org",
                credibility=0.80,
                domain="general",
                rate_limit=0.5
            ),
            VerificationSource(
                name="PubMed",
                url="https://pubmed.ncbi.nlm.nih.gov",
                credibility=0.90,
                domain="science",
                rate_limit=1.0
            ),
            VerificationSource(
                name="NASA Fact Sheets",
                url="https://www.nasa.gov",
                credibility=0.95,
                domain="science",
                rate_limit=2.0
            )
        ]
        
        return sources
    
    def _load_enhanced_credibility(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced source credibility with bias indicators"""
        return {
            'snopes.com': {
                'credibility': 0.95,
                'bias_score': 0.1,  # Low bias
                'expertise_domains': ['general', 'politics', 'health'],
                'verification_method': 'investigative_journalism',
                'transparency_score': 0.9
            },
            'factcheck.org': {
                'credibility': 0.95,
                'bias_score': 0.05,
                'expertise_domains': ['politics', 'policy'],
                'verification_method': 'academic_research',
                'transparency_score': 0.95
            },
            'wikipedia.org': {
                'credibility': 0.80,
                'bias_score': 0.2,
                'expertise_domains': ['general', 'history', 'science'],
                'verification_method': 'crowd_sourced',
                'transparency_score': 1.0
            },
            'nasa.gov': {
                'credibility': 0.95,
                'bias_score': 0.05,
                'expertise_domains': ['science', 'space', 'climate'],
                'verification_method': 'scientific_research',
                'transparency_score': 0.9
            },
            'cdc.gov': {
                'credibility': 0.95,
                'bias_score': 0.1,
                'expertise_domains': ['health', 'medicine', 'epidemiology'],
                'verification_method': 'scientific_research',
                'transparency_score': 0.85
            }
        }
    
    async def enhanced_fact_check(self, claim: Claim) -> EnhancedFactCheckResult:
        """
        Perform enhanced fact-checking with multi-source verification and explainability
        """
        logger.info(f"Enhanced fact-checking claim: {claim.text[:100]}...")
        
        # Initialize explanation tracking
        self.explanation_steps = []
        self.methodology_used = []
        
        try:
            with PerformanceTimer("enhanced_fact_check"):
                
                # Step 1: Multi-source evidence gathering
                self._add_explanation_step(
                    1, "evidence_gathering", "multiple_sources", 
                    "Collecting evidence from diverse sources", 0.0,
                    "Starting comprehensive evidence collection from verified sources"
                )
                
                evidence_collection = await self._multi_source_evidence_collection(claim)
                
                # Step 2: Cross-source verification
                self._add_explanation_step(
                    2, "cross_verification", "source_comparison",
                    f"Cross-referencing {len(evidence_collection)} sources", 0.2,
                    "Comparing findings across different sources for consensus"
                )
                
                cross_verification_result = await self._cross_source_verification(
                    claim, evidence_collection
                )
                
                # Step 3: Enhanced confidence calculation
                confidence_analysis = self._enhanced_confidence_analysis(
                    claim, evidence_collection, cross_verification_result
                )
                
                # Step 4: Generate explainable verdict
                verdict_analysis = self._generate_explainable_verdict(
                    claim, evidence_collection, confidence_analysis
                )
                
                # Step 5: Quality assessment
                quality_indicators = self._assess_verification_quality(
                    evidence_collection, cross_verification_result
                )
                
                # Create enhanced result
                result = EnhancedFactCheckResult(
                    claim=claim,
                    verdict=verdict_analysis['verdict'],
                    confidence=confidence_analysis['final_confidence'],
                    evidence_list=evidence_collection,
                    reasoning=verdict_analysis['detailed_reasoning'],
                    sources=[e.source_url for e in evidence_collection],
                    cross_domain_patterns=cross_verification_result.get('patterns', []),
                    timestamp=datetime.now(),
                    explanation_steps=self.explanation_steps.copy(),
                    source_breakdown=confidence_analysis['source_breakdown'],
                    confidence_breakdown=confidence_analysis['confidence_breakdown'],
                    alternative_interpretations=verdict_analysis.get('alternatives', []),
                    methodology_used=self.methodology_used.copy(),
                    quality_indicators=quality_indicators
                )
                
                await self._store_enhanced_result(result)
                
                logger.info(f"Enhanced fact-check complete: {result.verdict} "
                          f"(confidence: {result.confidence:.2f})")
                
                return result
                
        except Exception as e:
            logger.error(f"Enhanced fact-checking failed: {e}")
            raise FactCheckingError(f"Enhanced fact-check failed: {e}")
    
    async def _multi_source_evidence_collection(self, claim: Claim) -> List[Evidence]:
        """Collect evidence from multiple diverse sources"""
        evidence_collection = []
        
        # Strategy 1: Database evidence (existing approach enhanced)
        db_evidence = await self._collect_database_evidence(claim)
        evidence_collection.extend(db_evidence)
        self.methodology_used.append("database_search")
        
        # Strategy 2: Domain-specific expert sources
        expert_evidence = await self._collect_expert_source_evidence(claim)
        evidence_collection.extend(expert_evidence)
        self.methodology_used.append("expert_sources")
        
        # Strategy 3: Cross-domain verification
        cross_domain_evidence = await self._collect_cross_domain_evidence(claim)
        evidence_collection.extend(cross_domain_evidence)
        self.methodology_used.append("cross_domain_analysis")
        
        # Strategy 4: Historical claim analysis
        historical_evidence = await self._collect_historical_evidence(claim)
        evidence_collection.extend(historical_evidence)
        self.methodology_used.append("historical_analysis")
        
        # Strategy 5: Semantic similarity with known facts
        semantic_evidence = await self._collect_semantic_evidence(claim)
        evidence_collection.extend(semantic_evidence)
        self.methodology_used.append("semantic_analysis")
        
        self._add_explanation_step(
            len(self.explanation_steps) + 1, 
            "evidence_collection_complete",
            f"{len(self.verification_sources)} sources",
            f"Collected {len(evidence_collection)} pieces of evidence",
            0.3,
            f"Successfully gathered evidence using {len(self.methodology_used)} different methods"
        )
        
        return self._deduplicate_and_rank_evidence(evidence_collection)
    
    async def _collect_database_evidence(self, claim: Claim) -> List[Evidence]:
        """Enhanced database evidence collection with explainability"""
        evidence_list = []
        
        try:
            # Neo4j graph-based evidence
            async with self.db_connector.neo4j_driver.session() as session:
                # Enhanced query with explanation tracking
                query = """
                MATCH (c:Claim {id: $claim_id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document)
                WHERE d.credibility > 0.7
                OPTIONAL MATCH (d)-[:VERIFIED_BY]->(source:Source)
                OPTIONAL MATCH (d)-[:CONTRADICTS|SUPPORTS]->(related_claim:Claim)
                RETURN DISTINCT 
                    d.url as url, 
                    d.title as title,
                    d.content as content, 
                    d.credibility as credibility,
                    d.domain as domain,
                    source.name as source_name,
                    count(related_claim) as related_claims_count,
                    collect(related_claim.text) as related_claims
                ORDER BY d.credibility DESC
                LIMIT 20
                """
                
                result = await session.run(query, {'claim_id': claim.claim_id})
                
                async for record in result:
                    if record['url']:
                        # Enhanced evidence with explanation metadata
                        evidence = Evidence(
                            evidence_id=hashlib.md5(record['url'].encode()).hexdigest(),
                            text=record['content'] or record['title'] or '',
                            source_url=record['url'],
                            credibility_score=record['credibility'] or 0.7,
                            stance='neutral',  # Will be determined later
                            domain=record['domain'] or claim.domain,
                            timestamp=datetime.now()
                        )
                        
                        # Add explanation metadata
                        evidence.text += f"\n[Source: {record.get('source_name', 'Unknown')}]"
                        if record.get('related_claims_count', 0) > 0:
                            evidence.text += f"\n[Related claims: {record['related_claims_count']}]"
                        
                        evidence_list.append(evidence)
                        
            # Vector database evidence with enhanced similarity
            vector_evidence = await self._enhanced_vector_search(claim)
            evidence_list.extend(vector_evidence)
            
        except Exception as e:
            logger.warning(f"Database evidence collection error: {e}")
        
        return evidence_list
    
    async def _enhanced_vector_search(self, claim: Claim) -> List[Evidence]:
        """Enhanced vector search with multiple embedding strategies"""
        evidence_list = []
        
        try:
            # Multiple query strategies for better coverage
            query_strategies = [
                claim.text,  # Original claim
                self._reformulate_claim_question(claim.text),  # As question
                self._extract_key_concepts(claim.text)  # Key concepts only
            ]
            
            for strategy_text in query_strategies:
                query_embedding = self.sentence_model.encode(strategy_text)
                
                search_results = self.db_connector.qdrant_client.search(
                    collection_name='evidence',
                    query_vector=query_embedding.tolist(),
                    limit=10,
                    score_threshold=0.65
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
                    
                    # Enhance with similarity metadata
                    evidence.text += f"\n[Similarity: {result.score:.2f}]"
                    evidence.text += f"\n[Search strategy: {strategy_text[:50]}...]"
                    
                    evidence_list.append(evidence)
                    
        except Exception as e:
            logger.warning(f"Enhanced vector search error: {e}")
        
        return evidence_list
    
    def _reformulate_claim_question(self, claim_text: str) -> str:
        """Convert claim to question format for better search"""
        # Simple heuristic reformulation
        if claim_text.lower().startswith(('the ', 'a ', 'an ')):
            return f"Is it true that {claim_text.lower()}?"
        else:
            return f"Is {claim_text.lower()} true?"
    
    def _extract_key_concepts(self, claim_text: str) -> str:
        """Extract key concepts from claim for focused search"""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Remove common words and extract key terms
        words = claim_text.lower().split()
        stopwords = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Look for proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', claim_text)
        
        return ' '.join(key_words + proper_nouns)
    
    async def _collect_expert_source_evidence(self, claim: Claim) -> List[Evidence]:
        """Collect evidence from domain-specific expert sources"""
        evidence_list = []
        
        # Select sources based on claim domain
        relevant_sources = [s for s in self.verification_sources 
                          if s.domain == claim.domain or s.domain == 'general']
        
        for source in relevant_sources[:3]:  # Limit to top 3 most relevant
            try:
                # Simulate expert source consultation
                # In a real implementation, this would call actual APIs
                evidence = await self._consult_expert_source(claim, source)
                if evidence:
                    evidence_list.append(evidence)
                    
                # Rate limiting
                await asyncio.sleep(source.rate_limit)
                
            except Exception as e:
                logger.warning(f"Expert source {source.name} consultation failed: {e}")
        
        return evidence_list
    
    async def _consult_expert_source(self, claim: Claim, source: VerificationSource) -> Optional[Evidence]:
        """Consult a specific expert source (simulation for demo)"""
        
        # This is a simulation - in production this would make actual API calls
        claim_lower = claim.text.lower()
        
        if source.name == "NASA Fact Sheets" and any(term in claim_lower for term in ['earth', 'flat', 'space', 'moon', 'landing']):
            return Evidence(
                evidence_id=f"nasa_consultation_{hashlib.md5(claim.text.encode()).hexdigest()}",
                text=f"NASA official position: Scientific evidence conclusively shows that Earth is an oblate spheroid. "
                     f"Multiple independent observations from satellites, physics, and space missions confirm this. "
                     f"[Source: {source.name}] [Credibility: {source.credibility}] [Domain expertise: space science]",
                source_url=f"{source.url}/fact-sheets/earth-shape",
                credibility_score=source.credibility,
                stance='refutes' if 'flat' in claim_lower else 'supports',
                domain=source.domain,
                timestamp=datetime.now()
            )
        
        elif source.name == "Snopes" and any(term in claim_lower for term in ['hoax', 'conspiracy', 'fake', 'staged']):
            return Evidence(
                evidence_id=f"snopes_consultation_{hashlib.md5(claim.text.encode()).hexdigest()}",
                text=f"Snopes investigation: Extensive fact-checking reveals this claim has been thoroughly debunked. "
                     f"Multiple credible sources and expert analysis contradict the main assertions. "
                     f"[Source: {source.name}] [Methodology: Investigative journalism] [Credibility: {source.credibility}]",
                source_url=f"{source.url}/fact-check/{claim.claim_id}",
                credibility_score=source.credibility,
                stance='refutes',
                domain=source.domain,
                timestamp=datetime.now()
            )
        
        return None
    
    async def _collect_cross_domain_evidence(self, claim: Claim) -> List[Evidence]:
        """Collect evidence from related domains for broader context"""
        evidence_list = []
        
        try:
            # Find cross-domain patterns in Neo4j
            async with self.db_connector.neo4j_driver.session() as session:
                query = """
                MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
                MATCH (e)<-[:MENTIONS]-(d1:Document {domain: $domain})
                MATCH (e)<-[:MENTIONS]-(d2:Document)
                WHERE d2.domain <> $domain AND d2.credibility > 0.6
                MATCH (d2)-[:CONTAINS]->(concept:Concept)
                RETURN DISTINCT 
                    d2.url as url,
                    d2.content as content,
                    d2.domain as domain,
                    d2.credibility as credibility,
                    concept.name as concept_name,
                    e.name as shared_entity
                ORDER BY d2.credibility DESC
                LIMIT 15
                """
                
                result = await session.run(query, {
                    'claim_id': claim.claim_id,
                    'domain': claim.domain
                })
                
                async for record in result:
                    if record['url']:
                        evidence = Evidence(
                            evidence_id=hashlib.md5(record['url'].encode()).hexdigest(),
                            text=f"{record['content']} "
                                 f"[Cross-domain: {claim.domain} → {record['domain']}] "
                                 f"[Shared entity: {record['shared_entity']}] "
                                 f"[Concept: {record['concept_name']}]",
                            source_url=record['url'],
                            credibility_score=record['credibility'],
                            stance='neutral',
                            domain=record['domain'],
                            timestamp=datetime.now()
                        )
                        evidence_list.append(evidence)
                        
        except Exception as e:
            logger.warning(f"Cross-domain evidence collection error: {e}")
        
        return evidence_list
    
    async def _collect_historical_evidence(self, claim: Claim) -> List[Evidence]:
        """Collect evidence from historical claim analysis"""
        evidence_list = []
        
        try:
            # Find similar historical claims and their outcomes
            async with self.db_connector.neo4j_driver.session() as session:
                query = """
                MATCH (historical_claim:Claim)
                WHERE historical_claim.id <> $claim_id 
                  AND EXISTS((historical_claim)-[:HAS_FACT_CHECK]->(:FactCheckResult))
                WITH historical_claim, 
                     size([(historical_claim)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(current:Claim {id: $claim_id}) | e]) as shared_entities
                WHERE shared_entities > 0
                MATCH (historical_claim)-[:HAS_FACT_CHECK]->(result:FactCheckResult)
                RETURN 
                    historical_claim.text as claim_text,
                    result.verdict as verdict,
                    result.confidence as confidence,
                    result.reasoning as reasoning,
                    shared_entities,
                    historical_claim.domain as domain
                ORDER BY shared_entities DESC, result.confidence DESC
                LIMIT 10
                """
                
                result = await session.run(query, {'claim_id': claim.claim_id})
                
                historical_patterns = []
                async for record in result:
                    historical_patterns.append({
                        'claim': record['claim_text'],
                        'verdict': record['verdict'],
                        'confidence': record['confidence'],
                        'shared_entities': record['shared_entities']
                    })
                
                if historical_patterns:
                    # Create composite evidence from historical patterns
                    pattern_analysis = self._analyze_historical_patterns(historical_patterns)
                    
                    evidence = Evidence(
                        evidence_id=f"historical_analysis_{claim.claim_id}",
                        text=f"Historical analysis: {pattern_analysis['summary']} "
                             f"[Analyzed {len(historical_patterns)} similar claims] "
                             f"[Pattern confidence: {pattern_analysis['pattern_confidence']:.2f}] "
                             f"[Verdict distribution: {pattern_analysis['verdict_distribution']}]",
                        source_url="internal://historical_analysis",
                        credibility_score=pattern_analysis['pattern_confidence'],
                        stance=pattern_analysis['stance'],
                        domain=claim.domain,
                        timestamp=datetime.now()
                    )
                    evidence_list.append(evidence)
                    
        except Exception as e:
            logger.warning(f"Historical evidence collection error: {e}")
        
        return evidence_list
    
    def _analyze_historical_patterns(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze historical patterns to extract insights"""
        if not patterns:
            return {
                'summary': 'No historical patterns found',
                'pattern_confidence': 0.0,
                'stance': 'neutral',
                'verdict_distribution': {}
            }
        
        # Count verdicts
        verdict_counts = {}
        total_confidence = 0.0
        
        for pattern in patterns:
            verdict = pattern['verdict']
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            total_confidence += pattern['confidence']
        
        avg_confidence = total_confidence / len(patterns)
        most_common_verdict = max(verdict_counts, key=verdict_counts.get)
        
        # Determine stance based on most common verdict
        stance_map = {
            'True': 'supports',
            'False': 'refutes',
            'Partially True': 'neutral',
            'Unverified': 'neutral'
        }
        
        stance = stance_map.get(most_common_verdict, 'neutral')
        
        summary = f"Similar claims historically tend to be '{most_common_verdict}' " \
                 f"({verdict_counts[most_common_verdict]}/{len(patterns)} cases)"
        
        return {
            'summary': summary,
            'pattern_confidence': min(avg_confidence * 0.8, 0.9),  # Slightly discounted
            'stance': stance,
            'verdict_distribution': verdict_counts
        }
    
    async def _collect_semantic_evidence(self, claim: Claim) -> List[Evidence]:
        """Collect evidence using semantic similarity with established facts"""
        evidence_list = []
        
        try:
            # Search for semantically similar established facts
            query_embedding = self.sentence_model.encode(claim.text)
            
            # Search in facts collection (assuming it exists)
            search_results = self.db_connector.qdrant_client.search(
                collection_name='verified_facts',  # Hypothetical collection
                query_vector=query_embedding.tolist(),
                limit=8,
                score_threshold=0.7
            )
            
            for result in search_results:
                evidence = Evidence(
                    evidence_id=f"semantic_{result.id}",
                    text=f"Established fact: {result.payload.get('text', '')} "
                         f"[Semantic similarity: {result.score:.2f}] "
                         f"[Verification status: {result.payload.get('verification_status', 'verified')}]",
                    source_url=result.payload.get('source_url', 'internal://verified_facts'),
                    credibility_score=min(result.score, 0.9),  # Cap at 0.9 for semantic matches
                    stance=self._determine_semantic_stance(claim.text, result.payload.get('text', '')),
                    domain=result.payload.get('domain', claim.domain),
                    timestamp=datetime.now()
                )
                evidence_list.append(evidence)
                
        except Exception as e:
            logger.warning(f"Semantic evidence collection error: {e}")
        
        return evidence_list
    
    def _determine_semantic_stance(self, claim_text: str, fact_text: str) -> str:
        """Determine if semantic fact supports, refutes, or is neutral to claim"""
        
        # Simple heuristic - in production this would use more sophisticated NLP
        claim_lower = claim_text.lower()
        fact_lower = fact_text.lower()
        
        # Look for negation patterns
        if any(neg in claim_lower for neg in ['not', 'no', 'never', 'false', 'fake', 'hoax']):
            if any(neg in fact_lower for neg in ['not', 'no', 'never', 'false']):
                return 'supports'  # Double negative
            else:
                return 'refutes'
        else:
            if any(neg in fact_lower for neg in ['not', 'no', 'never', 'false']):
                return 'refutes'
            else:
                return 'supports'
    
    def _deduplicate_and_rank_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicates and rank evidence by relevance and credibility"""
        
        # Deduplicate by URL
        seen_urls = set()
        unique_evidence = []
        
        for evidence in evidence_list:
            if evidence.source_url not in seen_urls:
                unique_evidence.append(evidence)
                seen_urls.add(evidence.source_url)
        
        # Rank by credibility score and relevance
        ranked_evidence = sorted(
            unique_evidence, 
            key=lambda e: (e.credibility_score, len(e.text)), 
            reverse=True
        )
        
        # Limit to top results
        max_evidence = self.config.get('agent', {}).get('evidence_limit', 15)
        return ranked_evidence[:max_evidence]
    
    async def _cross_source_verification(self, claim: Claim, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """Perform cross-source verification analysis"""
        
        verification_result = {
            'consensus_level': 0.0,
            'conflicting_sources': [],
            'supporting_sources': [],
            'patterns': [],
            'source_reliability': {}
        }
        
        if len(evidence_list) < 2:
            verification_result['consensus_level'] = 0.3  # Low confidence with single source
            return verification_result
        
        # Analyze stance distribution
        stance_counts = {'supports': 0, 'refutes': 0, 'neutral': 0}
        weighted_stance_scores = {'supports': 0.0, 'refutes': 0.0, 'neutral': 0.0}
        
        for evidence in evidence_list:
            # Determine stance if not already set
            if evidence.stance == 'neutral':
                evidence.stance = self._determine_evidence_stance_advanced(claim.text, evidence.text)
            
            stance_counts[evidence.stance] += 1
            weighted_stance_scores[evidence.stance] += evidence.credibility_score
        
        # Calculate consensus
        total_sources = len(evidence_list)
        max_stance_count = max(stance_counts.values())
        consensus_level = max_stance_count / total_sources
        
        # Weight by credibility
        total_weighted_score = sum(weighted_stance_scores.values())
        if total_weighted_score > 0:
            weighted_consensus = max(weighted_stance_scores.values()) / total_weighted_score
            consensus_level = (consensus_level + weighted_consensus) / 2
        
        verification_result['consensus_level'] = consensus_level
        
        # Identify conflicting vs supporting sources
        dominant_stance = max(stance_counts, key=stance_counts.get)
        
        for evidence in evidence_list:
            source_info = {
                'url': evidence.source_url,
                'credibility': evidence.credibility_score,
                'stance': evidence.stance,
                'domain': evidence.domain
            }
            
            if evidence.stance == dominant_stance:
                verification_result['supporting_sources'].append(source_info)
            else:
                verification_result['conflicting_sources'].append(source_info)
        
        # Identify patterns
        verification_result['patterns'] = self._identify_verification_patterns(evidence_list)
        
        # Source reliability analysis
        verification_result['source_reliability'] = self._analyze_source_reliability(evidence_list)
        
        self._add_explanation_step(
            len(self.explanation_steps) + 1,
            "cross_verification_complete",
            f"{len(evidence_list)} sources",
            f"Consensus level: {consensus_level:.2f}, Conflicting sources: {len(verification_result['conflicting_sources'])}",
            0.4,
            f"Cross-source verification shows {consensus_level:.0%} agreement among sources"
        )
        
        return verification_result
    
    def _determine_evidence_stance_advanced(self, claim_text: str, evidence_text: str) -> str:
        """Advanced stance determination with multiple indicators"""
        
        evidence_lower = evidence_text.lower()
        claim_lower = claim_text.lower()
        
        # Strong support indicators
        strong_support = ['confirmed', 'verified', 'proven', 'true', 'accurate', 'correct', 'established fact']
        # Strong refute indicators
        strong_refute = ['false', 'incorrect', 'debunked', 'disproven', 'myth', 'hoax', 'fabricated', 'misleading']
        # Neutral/uncertain indicators
        neutral_indicators = ['unclear', 'uncertain', 'mixed evidence', 'requires further study', 'inconclusive']
        
        # Calculate scores
        support_score = sum(1 for indicator in strong_support if indicator in evidence_lower)
        refute_score = sum(1 for indicator in strong_refute if indicator in evidence_lower)
        neutral_score = sum(1 for indicator in neutral_indicators if indicator in evidence_lower)
        
        # Check for negation patterns
        negation_patterns = ['does not', 'did not', 'is not', 'are not', 'cannot', 'will not']
        has_negation = any(pattern in evidence_lower for pattern in negation_patterns)
        
        if has_negation:
            # Flip support/refute for negated statements
            support_score, refute_score = refute_score, support_score
        
        # Determine final stance
        if refute_score > support_score and refute_score > neutral_score:
            return 'refutes'
        elif support_score > refute_score and support_score > neutral_score:
            return 'supports'
        else:
            return 'neutral'
    
    def _identify_verification_patterns(self, evidence_list: List[Evidence]) -> List[str]:
        """Identify patterns in the verification evidence"""
        patterns = []
        
        # Domain distribution pattern
        domains = [e.domain for e in evidence_list]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if len(domain_counts) > 1:
            patterns.append(f"Multi-domain evidence: {', '.join(f'{d}({c})' for d, c in domain_counts.items())}")
        
        # Credibility distribution pattern
        high_cred = sum(1 for e in evidence_list if e.credibility_score > 0.8)
        med_cred = sum(1 for e in evidence_list if 0.6 <= e.credibility_score <= 0.8)
        low_cred = sum(1 for e in evidence_list if e.credibility_score < 0.6)
        
        patterns.append(f"Source credibility: {high_cred} high, {med_cred} medium, {low_cred} low")
        
        # Temporal pattern (if timestamps vary significantly)
        timestamps = [e.timestamp for e in evidence_list if e.timestamp]
        if len(timestamps) > 1:
            time_span = max(timestamps) - min(timestamps)
            patterns.append(f"Evidence span: {time_span.days} days")
        
        return patterns
    
    def _analyze_source_reliability(self, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """Analyze the reliability of sources used"""
        
        reliability_analysis = {
            'high_reliability_count': 0,
            'medium_reliability_count': 0,
            'low_reliability_count': 0,
            'unknown_reliability_count': 0,
            'bias_indicators': [],
            'methodology_diversity': []
        }
        
        for evidence in evidence_list:
            # Extract domain from URL for reliability lookup
            domain = self._extract_domain_from_url(evidence.source_url)
            
            if domain in self.source_credibility:
                source_info = self.source_credibility[domain]
                
                if source_info['credibility'] >= 0.8:
                    reliability_analysis['high_reliability_count'] += 1
                elif source_info['credibility'] >= 0.6:
                    reliability_analysis['medium_reliability_count'] += 1
                else:
                    reliability_analysis['low_reliability_count'] += 1
                
                # Check for bias indicators
                if source_info.get('bias_score', 0.5) > 0.3:
                    reliability_analysis['bias_indicators'].append(domain)
                
                # Track methodology diversity
                method = source_info.get('verification_method', 'unknown')
                if method not in reliability_analysis['methodology_diversity']:
                    reliability_analysis['methodology_diversity'].append(method)
            else:
                reliability_analysis['unknown_reliability_count'] += 1
        
        return reliability_analysis
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL for source analysis"""
        import re
        
        if not url or url.startswith('internal://'):
            return 'internal'
        
        pattern = r'https?://(?:www\.)?([^/]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else 'unknown'
    
    def _enhanced_confidence_analysis(self, claim: Claim, evidence_list: List[Evidence], 
                                   cross_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced confidence calculation with detailed breakdown"""
        
        confidence_analysis = {
            'final_confidence': 0.0,
            'source_breakdown': {},
            'confidence_breakdown': {},
            'adjustment_factors': {}
        }
        
        if not evidence_list:
            confidence_analysis['final_confidence'] = 0.1
            confidence_analysis['confidence_breakdown']['no_evidence'] = 0.1
            return confidence_analysis
        
        # Base confidence from evidence quality
        base_confidence = sum(e.credibility_score for e in evidence_list) / len(evidence_list)
        confidence_analysis['confidence_breakdown']['evidence_quality'] = base_confidence
        
        # Consensus adjustment
        consensus_bonus = cross_verification['consensus_level'] * 0.3
        confidence_analysis['confidence_breakdown']['consensus'] = consensus_bonus
        
        # Source diversity bonus
        unique_domains = len(set(e.domain for e in evidence_list))
        diversity_bonus = min(unique_domains * 0.05, 0.2)
        confidence_analysis['confidence_breakdown']['source_diversity'] = diversity_bonus
        
        # Volume adjustment
        volume_factor = min(len(evidence_list) * 0.02, 0.15)
        confidence_analysis['confidence_breakdown']['evidence_volume'] = volume_factor
        
        # Cross-domain pattern bonus
        if cross_verification.get('patterns'):
            pattern_bonus = 0.1
            confidence_analysis['confidence_breakdown']['cross_domain_patterns'] = pattern_bonus
        else:
            pattern_bonus = 0.0
        
        # Reliability adjustment
        reliability = cross_verification.get('source_reliability', {})
        high_rel = reliability.get('high_reliability_count', 0)
        total_sources = len(evidence_list)
        reliability_factor = (high_rel / total_sources) * 0.1 if total_sources > 0 else 0.0
        confidence_analysis['confidence_breakdown']['source_reliability'] = reliability_factor
        
        # Calculate final confidence
        final_confidence = min(
            base_confidence + consensus_bonus + diversity_bonus + 
            volume_factor + pattern_bonus + reliability_factor,
            0.95  # Cap at 95%
        )
        
        # Apply penalties
        penalties = 0.0
        
        # Conflicting sources penalty
        conflicting_count = len(cross_verification.get('conflicting_sources', []))
        if conflicting_count > 0:
            conflict_penalty = min(conflicting_count * 0.05, 0.2)
            penalties += conflict_penalty
            confidence_analysis['adjustment_factors']['conflict_penalty'] = -conflict_penalty
        
        # Low consensus penalty
        if cross_verification['consensus_level'] < 0.6:
            low_consensus_penalty = (0.6 - cross_verification['consensus_level']) * 0.3
            penalties += low_consensus_penalty
            confidence_analysis['adjustment_factors']['low_consensus_penalty'] = -low_consensus_penalty
        
        final_confidence = max(final_confidence - penalties, 0.1)
        confidence_analysis['final_confidence'] = final_confidence
        
        # Source breakdown
        for evidence in evidence_list:
            domain = self._extract_domain_from_url(evidence.source_url)
            if domain not in confidence_analysis['source_breakdown']:
                confidence_analysis['source_breakdown'][domain] = []
            
            confidence_analysis['source_breakdown'][domain].append({
                'credibility': evidence.credibility_score,
                'stance': evidence.stance
            })
        
        self._add_explanation_step(
            len(self.explanation_steps) + 1,
            "confidence_calculation",
            "statistical_analysis",
            f"Final confidence: {final_confidence:.2f}",
            final_confidence,
            f"Confidence based on: evidence quality ({base_confidence:.2f}), "
            f"consensus ({consensus_bonus:.2f}), diversity ({diversity_bonus:.2f})"
        )
        
        return confidence_analysis
    
    def _generate_explainable_verdict(self, claim: Claim, evidence_list: List[Evidence], 
                                   confidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate verdict with detailed explanation"""
        
        if not evidence_list:
            return {
                'verdict': 'Unverified',
                'detailed_reasoning': 'No evidence found to verify this claim.',
                'alternatives': ['Requires more research', 'Claim may be too specific or recent']
            }
        
        # Count stances with credibility weighting
        stance_scores = {'supports': 0.0, 'refutes': 0.0, 'neutral': 0.0}
        
        for evidence in evidence_list:
            stance_scores[evidence.stance] += evidence.credibility_score
        
        total_score = sum(stance_scores.values())
        if total_score == 0:
            return {
                'verdict': 'Unverified',
                'detailed_reasoning': 'Evidence available but inconclusive.',
                'alternatives': ['Requires expert review', 'May be opinion-based claim']
            }
        
        # Determine verdict based on weighted scores
        support_ratio = stance_scores['supports'] / total_score
        refute_ratio = stance_scores['refutes'] / total_score
        confidence = confidence_analysis['final_confidence']
        
        # Verdict logic with confidence thresholds
        if refute_ratio > 0.6 and confidence > 0.7:
            verdict = 'False'
            reasoning_focus = 'refuting'
        elif support_ratio > 0.6 and confidence > 0.7:
            verdict = 'True'
            reasoning_focus = 'supporting'
        elif confidence > 0.5 and abs(support_ratio - refute_ratio) < 0.3:
            verdict = 'Partially True'
            reasoning_focus = 'mixed'
        else:
            verdict = 'Unverified'
            reasoning_focus = 'insufficient'
        
        # Generate detailed reasoning
        detailed_reasoning = self._construct_detailed_reasoning(
            claim, evidence_list, confidence_analysis, reasoning_focus, verdict
        )
        
        # Generate alternative interpretations
        alternatives = self._generate_alternative_interpretations(
            claim, evidence_list, verdict, confidence
        )
        
        self._add_explanation_step(
            len(self.explanation_steps) + 1,
            "verdict_determination",
            "logical_analysis",
            f"Verdict: {verdict}",
            confidence,
            f"Based on {reasoning_focus} evidence pattern with {confidence:.0%} confidence"
        )
        
        return {
            'verdict': verdict,
            'detailed_reasoning': detailed_reasoning,
            'alternatives': alternatives
        }
    
    def _construct_detailed_reasoning(self, claim: Claim, evidence_list: List[Evidence],
                                   confidence_analysis: Dict[str, Any], focus: str, verdict: str) -> str:
        """Construct detailed, explainable reasoning"""
        
        reasoning_parts = []
        
        # Evidence summary
        evidence_count = len(evidence_list)
        high_cred_count = sum(1 for e in evidence_list if e.credibility_score > 0.8)
        
        reasoning_parts.append(
            f"Analysis of {evidence_count} evidence sources "
            f"({high_cred_count} high-credibility) yields verdict: {verdict}."
        )
        
        # Stance breakdown
        stance_counts = {'supports': 0, 'refutes': 0, 'neutral': 0}
        for evidence in evidence_list:
            stance_counts[evidence.stance] += 1
        
        reasoning_parts.append(
            f"Evidence distribution: {stance_counts['supports']} supporting, "
            f"{stance_counts['refutes']} refuting, {stance_counts['neutral']} neutral."
        )
        
        # Confidence explanation
        confidence = confidence_analysis['final_confidence']
        confidence_factors = confidence_analysis['confidence_breakdown']
        
        main_factors = []
        for factor, value in confidence_factors.items():
            if value > 0.1:
                main_factors.append(f"{factor.replace('_', ' ')} ({value:.2f})")
        
        reasoning_parts.append(
            f"Confidence level {confidence:.0%} based on: {', '.join(main_factors[:3])}."
        )
        
        # Source quality
        reliability = confidence_analysis.get('source_reliability', {})
        if reliability:
            reasoning_parts.append(
                f"Source reliability: {reliability.get('high_reliability_count', 0)} high, "
                f"{reliability.get('medium_reliability_count', 0)} medium quality sources."
            )
        
        # Key evidence examples (top 2)
        top_evidence = sorted(evidence_list, key=lambda e: e.credibility_score, reverse=True)[:2]
        for i, evidence in enumerate(top_evidence, 1):
            domain = self._extract_domain_from_url(evidence.source_url)
            reasoning_parts.append(
                f"Key source {i}: {domain} ({evidence.stance}, credibility: {evidence.credibility_score:.2f})."
            )
        
        return " ".join(reasoning_parts)
    
    def _generate_alternative_interpretations(self, claim: Claim, evidence_list: List[Evidence],
                                           verdict: str, confidence: float) -> List[str]:
        """Generate alternative interpretations of the claim"""
        
        alternatives = []
        
        # Based on confidence level
        if confidence < 0.5:
            alternatives.extend([
                "Claim requires additional verification from authoritative sources",
                "Evidence is insufficient for definitive conclusion",
                "May be partially true but needs more specific context"
            ])
        
        # Based on evidence patterns
        domains = set(e.domain for e in evidence_list)
        if len(domains) == 1:
            alternatives.append(f"Analysis limited to {list(domains)[0]} domain - cross-domain verification needed")
        
        conflicting_evidence = sum(1 for e in evidence_list if e.stance in ['refutes', 'neutral'])
        if conflicting_evidence > 0:
            alternatives.append("Some evidence contradicts claim - interpretation may depend on context")
        
        # Based on claim characteristics
        if any(term in claim.text.lower() for term in ['always', 'never', 'all', 'none']):
            alternatives.append("Absolute claims rarely hold universally - exceptions may exist")
        
        if any(term in claim.text.lower() for term in ['recent', 'new', 'latest']):
            alternatives.append("Recent claims may lack sufficient long-term verification")
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _assess_verification_quality(self, evidence_list: List[Evidence],
                                   cross_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the overall quality of the verification process"""
        
        quality_indicators = {
            'evidence_sufficiency': 'adequate' if len(evidence_list) >= 3 else 'limited',
            'source_diversity': len(set(e.domain for e in evidence_list)),
            'credibility_average': sum(e.credibility_score for e in evidence_list) / len(evidence_list) if evidence_list else 0.0,
            'consensus_strength': cross_verification.get('consensus_level', 0.0),
            'methodology_diversity': len(cross_verification.get('source_reliability', {}).get('methodology_diversity', [])),
            'potential_bias_sources': len(cross_verification.get('source_reliability', {}).get('bias_indicators', [])),
            'verification_completeness': self._calculate_completeness_score(evidence_list, cross_verification)
        }
        
        # Overall quality score
        quality_score = (
            (1.0 if quality_indicators['evidence_sufficiency'] == 'adequate' else 0.5) * 0.2 +
            min(quality_indicators['source_diversity'] / 3, 1.0) * 0.2 +
            quality_indicators['credibility_average'] * 0.3 +
            quality_indicators['consensus_strength'] * 0.2 +
            min(quality_indicators['methodology_diversity'] / 3, 1.0) * 0.1
        )
        
        quality_indicators['overall_quality_score'] = quality_score
        quality_indicators['quality_grade'] = (
            'Excellent' if quality_score > 0.8 else
            'Good' if quality_score > 0.6 else
            'Fair' if quality_score > 0.4 else
            'Poor'
        )
        
        return quality_indicators
    
    def _calculate_completeness_score(self, evidence_list: List[Evidence],
                                    cross_verification: Dict[str, Any]) -> float:
        """Calculate how complete the verification process was"""
        
        completeness_factors = []
        
        # Evidence collection completeness
        evidence_factor = min(len(evidence_list) / 10, 1.0)  # Target: 10 sources
        completeness_factors.append(evidence_factor)
        
        # Methodology completeness
        methods_used = len(self.methodology_used)
        method_factor = min(methods_used / 5, 1.0)  # Target: 5 methods
        completeness_factors.append(method_factor)
        
        # Cross-verification completeness
        cross_factor = 1.0 if cross_verification.get('consensus_level', 0) > 0 else 0.5
        completeness_factors.append(cross_factor)
        
        # Source diversity completeness
        domains = set(e.domain for e in evidence_list)
        diversity_factor = min(len(domains) / 3, 1.0)  # Target: 3 domains
        completeness_factors.append(diversity_factor)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    async def _store_enhanced_result(self, result: EnhancedFactCheckResult):
        """Store enhanced fact-check result with full explainability data"""
        
        try:
            async with self.db_connector.neo4j_driver.session() as session:
                # Store main result
                query = """
                MATCH (c:Claim {id: $claim_id})
                CREATE (fr:EnhancedFactCheckResult {
                    id: $result_id,
                    verdict: $verdict,
                    confidence: $confidence,
                    reasoning: $reasoning,
                    timestamp: $timestamp,
                    explanation_steps: $explanation_steps,
                    source_breakdown: $source_breakdown,
                    confidence_breakdown: $confidence_breakdown,
                    methodology_used: $methodology_used,
                    quality_score: $quality_score,
                    quality_grade: $quality_grade
                })
                CREATE (c)-[:HAS_ENHANCED_FACT_CHECK]->(fr)
                RETURN fr.id as result_id
                """
                
                result_id = hashlib.md5(f"enhanced_{result.claim.claim_id}_{result.timestamp}".encode()).hexdigest()
                
                await session.run(query, {
                    'claim_id': result.claim.claim_id,
                    'result_id': result_id,
                    'verdict': result.verdict,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'timestamp': result.timestamp.isoformat(),
                    'explanation_steps': [asdict(step) for step in result.explanation_steps],
                    'source_breakdown': result.source_breakdown,
                    'confidence_breakdown': result.confidence_breakdown,
                    'methodology_used': result.methodology_used,
                    'quality_score': result.quality_indicators.get('overall_quality_score', 0.0),
                    'quality_grade': result.quality_indicators.get('quality_grade', 'Unknown')
                })
                
                result.graph_node_id = result_id
                logger.info(f"Stored enhanced fact-check result: {result_id}")
                
        except Exception as e:
            logger.error(f"Error storing enhanced result: {e}")
    
    def _add_explanation_step(self, step_number: int, action: str, source: str,
                           finding: str, confidence_impact: float, reasoning: str):
        """Add step to explanation trail"""
        
        step = ExplanationStep(
            step_number=step_number,
            action=action,
            source=source,
            finding=finding,
            confidence_impact=confidence_impact,
            reasoning=reasoning
        )
        
        self.explanation_steps.append(step)
        logger.debug(f"Explanation step {step_number}: {action} - {finding}")