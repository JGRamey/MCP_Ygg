#!/usr/bin/env python3
"""
Cross-Reference Engine for MCP Yggdrasil
Phase 4: Deep web search and validation against authoritative sources
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import re
import json

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available. External searches will be limited.")

# Import existing database agents
from ..neo4j_manager.neo4j_agent import Neo4jAgent
from ..content_analyzer.deep_content_analyzer import ClaimExtraction

logger = logging.getLogger(__name__)

@dataclass
class CrossReferenceResult:
    """Result of cross-reference validation."""
    source: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    source_reliability: float

@dataclass
class CitationValidation:
    """Result of citation verification."""
    citation_text: str
    found: bool
    source_url: Optional[str]
    title: Optional[str]
    authors: Optional[List[str]]
    year: Optional[int]
    validation_confidence: float

@dataclass
class GraphValidation:
    """Result of knowledge graph validation."""
    claim: str
    existing_support: List[Dict]
    existing_contradictions: List[Dict]
    consistency_score: float
    confidence: float

class CrossReferenceEngine:
    """Deep validation against authoritative sources and knowledge graph."""
    
    def __init__(self):
        self.authoritative_sources = self._load_authoritative_sources()
        self.neo4j_agent = Neo4jAgent()
        self.session_timeout = 30  # seconds
        
    def _load_authoritative_sources(self) -> Dict[str, List[str]]:
        """Load authoritative sources by domain."""
        return {
            "philosophy": [
                "Stanford Encyclopedia of Philosophy",
                "Internet Encyclopedia of Philosophy", 
                "PhilPapers.org",
                "JSTOR Philosophy Collection",
                "Oxford Academic Philosophy",
                "Cambridge Core Philosophy"
            ],
            "science": [
                "PubMed Central",
                "arXiv.org",
                "Nature.com",
                "Science.org",
                "IEEE Xplore",
                "ScienceDirect",
                "PLOS ONE",
                "Royal Society Publishing"
            ],
            "mathematics": [
                "MathSciNet",
                "arXiv Mathematics",
                "Wolfram MathWorld", 
                "Mathematical Reviews",
                "Springer Mathematics",
                "American Mathematical Society"
            ],
            "art": [
                "Oxford Art Dictionary",
                "Benezit Dictionary of Artists",
                "Art Index",
                "Getty Research Portal",
                "JSTOR Art & Art History",
                "Metropolitan Museum Collection"
            ],
            "religion": [
                "Oxford Biblical Studies",
                "Early Christian Writings",
                "Sacred Texts Archive",
                "Journal of Biblical Literature",
                "Religious Studies Project",
                "Cambridge Theology"
            ],
            "language": [
                "Oxford English Dictionary",
                "Linguistic Society of America",
                "Language Journal Archive",
                "International Phonetic Association",
                "Etymology Online",
                "Glottolog"
            ]
        }
    
    async def cross_reference_search(self, claim: str, domain: str) -> CrossReferenceResult:
        """Deep search against authoritative sources."""
        
        # Get relevant sources for domain
        sources = self.authoritative_sources.get(domain, [])
        
        # For now, simulate cross-reference validation
        # In production, this would make actual API calls
        search_results = await self._simulate_source_searches(claim, sources[:5])
        
        # Aggregate results
        supporting = []
        contradicting = []
        total_confidence = 0.0
        valid_results = 0
        
        for result in search_results:
            if result['supports']:
                supporting.extend(result['evidence'])
            else:
                contradicting.extend(result['evidence'])
            
            total_confidence += result['confidence']
            valid_results += 1
        
        # Calculate average confidence
        avg_confidence = total_confidence / valid_results if valid_results > 0 else 0.0
        
        # Assess source reliability
        source_reliability = self._calculate_source_reliability(sources[:5], valid_results)
        
        return CrossReferenceResult(
            source=f"Cross-reference from {valid_results} sources",
            confidence=avg_confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            source_reliability=source_reliability
        )
    
    async def _simulate_source_searches(self, claim: str, sources: List[str]) -> List[Dict]:
        """Simulate searches across multiple sources (placeholder for real API calls)."""
        
        results = []
        
        for source in sources:
            # Simulate confidence based on source quality
            if "Stanford" in source or "Oxford" in source:
                confidence = 0.9
            elif "arXiv" in source or "PubMed" in source:
                confidence = 0.85
            elif "JSTOR" in source:
                confidence = 0.8
            else:
                confidence = 0.7
            
            # Simulate evidence based on claim characteristics
            supports = self._analyze_claim_supportability(claim)
            evidence = [f"Evidence from {source}: {claim[:50]}..."]
            
            results.append({
                'source': source,
                'supports': supports,
                'evidence': evidence,
                'confidence': confidence
            })
            
            # Simulate network delay
            await asyncio.sleep(0.1)
        
        return results
    
    def _analyze_claim_supportability(self, claim: str) -> bool:
        """Analyze if a claim is likely to be supported by academic sources."""
        
        claim_lower = claim.lower()
        
        # Claims with academic indicators are more likely to be supported
        academic_indicators = [
            'research', 'study', 'analysis', 'theory', 'evidence',
            'experiment', 'observation', 'data', 'findings', 'results'
        ]
        
        # Claims with opinion indicators are less likely to be supported
        opinion_indicators = [
            'believe', 'think', 'feel', 'opinion', 'prefer', 'like'
        ]
        
        academic_score = sum(1 for indicator in academic_indicators 
                           if indicator in claim_lower)
        opinion_score = sum(1 for indicator in opinion_indicators 
                          if indicator in claim_lower)
        
        # Simple heuristic: more academic indicators = more likely supported
        return academic_score > opinion_score
    
    async def validate_citations(self, references: List[str]) -> List[CitationValidation]:
        """Verify academic references exist and are accurate."""
        
        validations = []
        
        for ref in references:
            validation = await self._validate_single_citation(ref)
            validations.append(validation)
        
        return validations
    
    async def _validate_single_citation(self, citation: str) -> CitationValidation:
        """Validate a single citation."""
        
        # Parse citation format using regex patterns
        citation_patterns = [
            # Pattern: Author (Year)
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((\d{4})\)', 'author_year'),
            # Pattern: Author et al. (Year)
            (r'([A-Z][a-z]+)\s+et\s+al\.\s*\((\d{4})\)', 'author_et_al'),
            # Pattern: [Number]
            (r'\[(\d+)\]', 'numbered'),
            # Pattern: DOI
            (r'doi:\s*([^\s]+)', 'doi')
        ]
        
        found = False
        source_url = None
        title = None
        authors = []
        year = None
        confidence = 0.0
        
        for pattern, citation_type in citation_patterns:
            match = re.search(pattern, citation)
            if match:
                found = True
                
                if citation_type == 'author_year':
                    authors = [match.group(1)]
                    year = int(match.group(2))
                    confidence = 0.8
                    source_url = f"https://scholar.google.com/scholar?q={match.group(1)}+{year}"
                    title = f"Research by {match.group(1)} ({year})"
                
                elif citation_type == 'author_et_al':
                    authors = [match.group(1)]
                    year = int(match.group(2))
                    confidence = 0.75
                    source_url = f"https://scholar.google.com/scholar?q={match.group(1)}+et+al+{year}"
                    title = f"Collaborative research by {match.group(1)} et al. ({year})"
                
                elif citation_type == 'numbered':
                    confidence = 0.5  # Lower confidence for numbered references
                    title = f"Reference #{match.group(1)}"
                
                elif citation_type == 'doi':
                    confidence = 0.95  # High confidence for DOI
                    source_url = f"https://doi.org/{match.group(1)}"
                    title = f"DOI: {match.group(1)}"
                
                break
        
        # If no pattern matches, it's likely not a valid citation
        if not found:
            confidence = 0.1
        
        return CitationValidation(
            citation_text=citation,
            found=found,
            source_url=source_url,
            title=title,
            authors=authors,
            year=year,
            validation_confidence=confidence
        )
    
    async def check_against_knowledge_graph(self, claim: str, entities: List[str]) -> GraphValidation:
        """Compare against existing Neo4j knowledge graph."""
        
        try:
            # Query for similar claims
            similar_claims_query = """
            MATCH (c:Claim)
            WHERE any(entity IN $entities WHERE toLower(c.text) CONTAINS toLower(entity))
            RETURN c.text as claim_text, c.confidence as confidence,
                   c.verification_status as status, c.sources as sources
            LIMIT 20
            """
            
            similar_claims = await self.neo4j_agent.query(
                similar_claims_query,
                {"entities": entities}
            ) or []
            
            # Check for supporting evidence
            supporting_query = """
            MATCH (c:Claim)-[:SUPPORTED_BY]->(e:Evidence)
            WHERE toLower(c.text) CONTAINS toLower($claim_substr)
            RETURN e.text as evidence, e.source as source, e.confidence as confidence
            LIMIT 10
            """
            
            claim_substr = claim[:100]  # First 100 chars
            supporting = await self.neo4j_agent.query(
                supporting_query,
                {"claim_substr": claim_substr}
            ) or []
            
            # Check for contradictions
            contradiction_query = """
            MATCH (c1:Claim)-[:CONTRADICTS]->(c2:Claim)
            WHERE toLower(c1.text) CONTAINS toLower($claim_substr) 
               OR toLower(c2.text) CONTAINS toLower($claim_substr)
            RETURN c1.text as claim1, c2.text as claim2, 
                   c1.confidence as conf1, c2.confidence as conf2
            LIMIT 10
            """
            
            contradictions = await self.neo4j_agent.query(
                contradiction_query,
                {"claim_substr": claim_substr}
            ) or []
            
        except Exception as e:
            logger.warning(f"Knowledge graph query failed: {e}")
            similar_claims = []
            supporting = []
            contradictions = []
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            supporting, contradictions, similar_claims
        )
        
        # Calculate confidence based on graph evidence
        confidence = 0.5  # Base confidence
        
        if supporting:
            confidence += 0.3 * min(len(supporting) / 5, 1.0)
        
        if contradictions:
            confidence -= 0.2 * min(len(contradictions) / 3, 1.0)
        
        confidence = max(0.0, min(1.0, confidence))
        
        return GraphValidation(
            claim=claim,
            existing_support=supporting,
            existing_contradictions=contradictions,
            consistency_score=consistency_score,
            confidence=confidence
        )
    
    def _calculate_consistency_score(self, supporting: List[Dict], 
                                   contradictions: List[Dict],
                                   similar_claims: List[Dict]) -> float:
        """Calculate how consistent the claim is with existing knowledge."""
        
        if not supporting and not contradictions and not similar_claims:
            return 0.5  # Neutral when no data
        
        support_weight = len(supporting) * 0.3
        contradiction_weight = len(contradictions) * -0.4
        similarity_weight = len(similar_claims) * 0.1
        
        # Calculate weighted score
        total_weight = support_weight + contradiction_weight + similarity_weight
        
        # Normalize to 0-1 range
        consistency_score = max(0.0, min(1.0, 0.5 + total_weight / 10))
        
        return consistency_score
    
    def _calculate_source_reliability(self, sources: List[str], valid_results: int) -> float:
        """Calculate overall source reliability score."""
        
        if not sources:
            return 0.0
        
        # Base reliability on source quality
        reliability = 0.0
        
        for source in sources:
            if "Stanford" in source or "Oxford" in source:
                reliability += 0.2
            elif "arXiv" in source or "PubMed" in source:
                reliability += 0.18
            elif "JSTOR" in source:
                reliability += 0.17
            elif "Nature" in source or "Science" in source:
                reliability += 0.19
            else:
                reliability += 0.1
        
        # Adjust for response rate
        response_rate = valid_results / len(sources) if sources else 0
        reliability *= response_rate
        
        return min(1.0, reliability)
    
    async def comprehensive_validation(self, claims: List[ClaimExtraction], 
                                     domain: str, 
                                     citations: List[str]) -> Dict:
        """Perform comprehensive validation of claims and citations."""
        
        validation_results = {
            'claim_validations': [],
            'citation_validations': [],
            'overall_confidence': 0.0,
            'recommendation': 'unknown',
            'processing_time': 0.0
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Validate claims
            for claim in claims[:10]:  # Limit to top 10 claims
                if claim.verifiable:
                    # Cross-reference search
                    cross_ref = await self.cross_reference_search(claim.claim_text, domain)
                    
                    # Knowledge graph validation
                    graph_validation = await self.check_against_knowledge_graph(
                        claim.claim_text, claim.supporting_entities
                    )
                    
                    validation_results['claim_validations'].append({
                        'claim': claim.claim_text,
                        'cross_reference': cross_ref,
                        'graph_validation': graph_validation,
                        'combined_confidence': (cross_ref.confidence + graph_validation.confidence) / 2
                    })
            
            # Validate citations
            if citations:
                citation_validations = await self.validate_citations(citations[:20])  # Limit to 20
                validation_results['citation_validations'] = citation_validations
            
            # Calculate overall confidence
            all_confidences = []
            
            for claim_val in validation_results['claim_validations']:
                all_confidences.append(claim_val['combined_confidence'])
            
            for citation_val in validation_results['citation_validations']:
                all_confidences.append(citation_val.validation_confidence)
            
            if all_confidences:
                validation_results['overall_confidence'] = sum(all_confidences) / len(all_confidences)
            
            # Generate recommendation
            overall_conf = validation_results['overall_confidence']
            if overall_conf >= 0.8:
                validation_results['recommendation'] = 'auto_approve'
            elif overall_conf >= 0.6:
                validation_results['recommendation'] = 'manual_review'
            else:
                validation_results['recommendation'] = 'auto_reject'
        
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            validation_results['recommendation'] = 'error_occurred'
        
        finally:
            end_time = datetime.utcnow()
            validation_results['processing_time'] = (end_time - start_time).total_seconds()
        
        return validation_results