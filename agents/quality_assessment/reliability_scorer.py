#!/usr/bin/env python3
"""
Reliability Scorer for MCP Yggdrasil
Phase 4: Comprehensive reliability scoring algorithm for academic content
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Import Phase 4 components
from ..scraper.intelligent_scraper_agent import ScrapedDocument
from ..content_analyzer.deep_content_analyzer import ContentAnalysis
from ..fact_verifier.cross_reference_engine import CrossReferenceResult, CitationValidation, GraphValidation

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for content quality."""
    HIGH = "high"        # 0.8-1.0: Auto-approve
    MEDIUM = "medium"    # 0.6-0.8: Manual review  
    LOW = "low"          # <0.6: Auto-reject

@dataclass
class ReliabilityScore:
    """Complete reliability assessment."""
    overall_score: float
    component_scores: Dict[str, float]
    confidence_level: ConfidenceLevel
    recommendation: str
    detailed_analysis: Dict
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]

class ReliabilityScorer:
    """Comprehensive reliability scoring algorithm."""
    
    def __init__(self):
        # Scoring weights (must sum to 1.0)
        self.weights = {
            'source_authority': 0.25,
            'cross_reference_support': 0.30,
            'citation_quality': 0.20,
            'expert_consensus': 0.15,
            'academic_rigor': 0.10
        }
        
        # Quality thresholds
        self.thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.0
        }
        
        # Academic rigor indicators
        self.academic_indicators = {
            'has_abstract': ['abstract', 'summary', 'overview'],
            'has_methodology': ['methodology', 'method', 'approach', 'framework', 'design'],
            'has_results': ['results', 'findings', 'data', 'analysis', 'evidence'],
            'has_conclusion': ['conclusion', 'summary', 'in conclusion', 'to conclude', 'findings'],
            'has_references': ['references', 'bibliography', 'works cited', 'sources'],
            'formal_language': ['furthermore', 'moreover', 'however', 'therefore', 'consequently']
        }
    
    def calculate_reliability_score(
        self,
        scraped_doc: ScrapedDocument,
        content_analysis: ContentAnalysis,
        cross_ref_results: CrossReferenceResult,
        citation_validations: List[CitationValidation],
        graph_validation: GraphValidation
    ) -> ReliabilityScore:
        """Calculate comprehensive reliability score."""
        
        # Calculate component scores
        scores = {
            'source_authority': self._score_source_authority(scraped_doc),
            'cross_reference_support': self._score_cross_reference(cross_ref_results),
            'citation_quality': self._score_citations(citation_validations),
            'expert_consensus': self._score_expert_consensus(graph_validation),
            'academic_rigor': self._score_academic_rigor(content_analysis, scraped_doc)
        }
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[component] * self.weights[component]
            for component in scores
        )
        
        # Ensure score is within bounds
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(confidence_level, scores, content_analysis)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(scores)
        weaknesses = self._identify_weaknesses(scores)
        
        # Generate improvement suggestions
        improvements = self._generate_improvements(scores, scraped_doc, content_analysis)
        
        # Detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            scores, scraped_doc, content_analysis, cross_ref_results
        )
        
        return ReliabilityScore(
            overall_score=overall_score,
            component_scores=scores,
            confidence_level=confidence_level,
            recommendation=recommendation,
            detailed_analysis=detailed_analysis,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements
        )
    
    def _score_source_authority(self, scraped_doc: ScrapedDocument) -> float:
        """Score based on source authority and content type."""
        
        # Start with metadata authority score
        base_score = scraped_doc.metadata.authority_score
        
        # Content type multipliers
        from ..scraper.intelligent_scraper_agent import ContentType
        content_type_multipliers = {
            ContentType.ACADEMIC_PAPER: 1.2,
            ContentType.ENCYCLOPEDIA: 1.1,
            ContentType.TECHNICAL_DOCS: 1.05,
            ContentType.HISTORICAL_RECORD: 1.0,
            ContentType.RELIGIOUS_TEXT: 0.95,
            ContentType.NEWS_ARTICLE: 0.9,
            ContentType.BOOK_EXCERPT: 0.85,
            ContentType.MANUSCRIPT: 0.8,
            ContentType.BLOG_POST: 0.7,
            ContentType.FORUM_DISCUSSION: 0.6
        }
        
        multiplier = content_type_multipliers.get(
            scraped_doc.metadata.content_type, 1.0
        )
        
        # Adjust based on publication date (recent academic work might be more current)
        pub_date_bonus = 0.0
        if scraped_doc.metadata.publication_date:
            import datetime
            years_old = (datetime.datetime.now() - scraped_doc.metadata.publication_date).days / 365
            if years_old < 5:  # Recent work gets small bonus
                pub_date_bonus = 0.05
            elif years_old > 50:  # Very old work might be foundational
                pub_date_bonus = 0.03
        
        # Word count factor (substantial content preferred)
        word_count_factor = 1.0
        if scraped_doc.metadata.word_count < 200:
            word_count_factor = 0.8  # Too short
        elif scraped_doc.metadata.word_count > 10000:
            word_count_factor = 0.95  # Possibly too long
        
        final_score = base_score * multiplier * word_count_factor + pub_date_bonus
        return min(1.0, final_score)
    
    def _score_cross_reference(self, cross_ref_results: CrossReferenceResult) -> float:
        """Score based on cross-reference support."""
        
        # Base score from cross-reference confidence
        base_score = cross_ref_results.confidence
        
        # Evidence balance factor
        supporting_count = len(cross_ref_results.supporting_evidence)
        contradicting_count = len(cross_ref_results.contradicting_evidence)
        
        if supporting_count + contradicting_count > 0:
            support_ratio = supporting_count / (supporting_count + contradicting_count)
            # Apply sigmoid-like transformation for balance
            balance_factor = 2 * support_ratio * (1 - support_ratio) + 0.5 * support_ratio
        else:
            balance_factor = 0.5  # Neutral when no evidence
        
        # Source reliability multiplier
        reliability_factor = cross_ref_results.source_reliability
        
        # Volume bonus (more evidence is better, up to a point)
        volume_bonus = min(0.1, (supporting_count + contradicting_count) * 0.02)
        
        final_score = base_score * balance_factor * reliability_factor + volume_bonus
        return min(1.0, final_score)
    
    def _score_citations(self, citation_validations: List[CitationValidation]) -> float:
        """Score based on citation quality and validation."""
        
        if not citation_validations:
            return 0.4  # Below neutral for lack of citations
        
        # Calculate validation statistics
        total_citations = len(citation_validations)
        valid_citations = sum(1 for c in citation_validations if c.found)
        high_confidence_citations = sum(
            1 for c in citation_validations 
            if c.validation_confidence >= 0.8
        )
        
        # Base validation rate
        validation_rate = valid_citations / total_citations if total_citations > 0 else 0
        
        # Quality factor (high confidence citations)
        quality_factor = high_confidence_citations / total_citations if total_citations > 0 else 0
        
        # Volume factor (having citations is important)
        volume_factor = min(1.0, total_citations / 10)  # Optimal around 10 citations
        
        # Average confidence of valid citations
        avg_confidence = 0.0
        if valid_citations > 0:
            avg_confidence = sum(
                c.validation_confidence for c in citation_validations if c.found
            ) / valid_citations
        
        # Combine factors
        final_score = (
            validation_rate * 0.4 + 
            quality_factor * 0.3 + 
            volume_factor * 0.2 + 
            avg_confidence * 0.1
        )
        
        return min(1.0, final_score)
    
    def _score_expert_consensus(self, graph_validation: GraphValidation) -> float:
        """Score based on expert consensus in knowledge graph."""
        
        # Start with graph validation confidence
        base_score = graph_validation.confidence
        
        # Consistency score from knowledge graph
        consistency_bonus = graph_validation.consistency_score * 0.2
        
        # Support vs contradiction analysis
        support_count = len(graph_validation.existing_support)
        contradiction_count = len(graph_validation.existing_contradictions)
        
        # More support increases score
        support_bonus = min(0.15, support_count * 0.03)
        
        # Contradictions decrease score, but less severely if support is strong
        contradiction_penalty = 0.0
        if contradiction_count > 0:
            penalty_strength = 0.2 if support_count < contradiction_count else 0.1
            contradiction_penalty = min(penalty_strength, contradiction_count * 0.05)
        
        final_score = base_score + consistency_bonus + support_bonus - contradiction_penalty
        return max(0.0, min(1.0, final_score))
    
    def _score_academic_rigor(
        self,
        content_analysis: ContentAnalysis,
        scraped_doc: ScrapedDocument
    ) -> float:
        """Score based on academic rigor and structure."""
        
        content = scraped_doc.content.lower()
        score = 0.0
        
        # Check for academic structure indicators
        structure_scores = {}
        for indicator, keywords in self.academic_indicators.items():
            found = any(keyword in content for keyword in keywords)
            structure_scores[indicator] = found
        
        # Weight different indicators
        indicator_weights = {
            'has_abstract': 0.15,
            'has_methodology': 0.20,
            'has_results': 0.20,
            'has_conclusion': 0.15,
            'has_references': 0.15,
            'formal_language': 0.15
        }
        
        for indicator, weight in indicator_weights.items():
            if structure_scores.get(indicator, False):
                score += weight
        
        # Content analysis bonuses
        # Domain specificity (focused content is better)
        domain_specificity = max(content_analysis.domain_mapping.values()) if content_analysis.domain_mapping else 0
        score += domain_specificity * 0.1
        
        # Sentiment analysis (academic tone preferred)
        if content_analysis.sentiment_analysis.get('tone') == 'academic':
            score += 0.1
        elif content_analysis.sentiment_analysis.get('tone') == 'formal':
            score += 0.05
        
        # Verifiable claims bonus
        verifiable_claims = sum(1 for claim in content_analysis.claims if claim.verifiable)
        total_claims = len(content_analysis.claims)
        if total_claims > 0:
            verifiable_ratio = verifiable_claims / total_claims
            score += verifiable_ratio * 0.1
        
        # Entity richness (good entity extraction indicates structured content)
        entity_density = len(content_analysis.entities) / max(1, scraped_doc.metadata.word_count / 100)
        entity_bonus = min(0.05, entity_density * 0.01)
        score += entity_bonus
        
        return min(1.0, score)
    
    def _determine_confidence_level(self, overall_score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        
        if overall_score >= self.thresholds['high']:
            return ConfidenceLevel.HIGH
        elif overall_score >= self.thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_recommendation(
        self,
        confidence_level: ConfidenceLevel,
        scores: Dict[str, float],
        content_analysis: ContentAnalysis
    ) -> str:
        """Generate actionable recommendation."""
        
        if confidence_level == ConfidenceLevel.HIGH:
            primary_domain = max(content_analysis.domain_mapping.items(), key=lambda x: x[1])[0] if content_analysis.domain_mapping else 'general'
            return f"AUTO-APPROVE: High-quality {primary_domain} content ready for knowledge graph integration."
        
        elif confidence_level == ConfidenceLevel.MEDIUM:
            # Identify specific weak areas
            weak_areas = [
                component.replace('_', ' ') for component, score in scores.items()
                if score < 0.6
            ]
            
            if weak_areas:
                return f"MANUAL REVIEW REQUIRED: Review needed for {', '.join(weak_areas)}. Consider additional validation."
            else:
                return "MANUAL REVIEW REQUIRED: Overall quality is borderline. Human judgment recommended."
        
        else:  # LOW
            # Identify critical failures
            critical_failures = [
                component.replace('_', ' ') for component, score in scores.items()
                if score < 0.3
            ]
            
            if critical_failures:
                return f"AUTO-REJECT: Critical quality issues in {', '.join(critical_failures)}. Not suitable for knowledge graph."
            else:
                return "AUTO-REJECT: Overall quality too low for inclusion in knowledge base."
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify strong points in the content."""
        strengths = []
        
        for component, score in scores.items():
            if score >= 0.8:
                component_name = component.replace('_', ' ').title()
                strengths.append(f"Excellent {component_name} (score: {score:.2f})")
            elif score >= 0.7:
                component_name = component.replace('_', ' ').title()
                strengths.append(f"Strong {component_name} (score: {score:.2f})")
        
        return strengths
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """Identify weak points in the content."""
        weaknesses = []
        
        for component, score in scores.items():
            if score < 0.4:
                component_name = component.replace('_', ' ').title()
                weaknesses.append(f"Poor {component_name} (score: {score:.2f})")
            elif score < 0.6:
                component_name = component.replace('_', ' ').title()
                weaknesses.append(f"Weak {component_name} (score: {score:.2f})")
        
        return weaknesses
    
    def _generate_improvements(self, scores: Dict[str, float], 
                             scraped_doc: ScrapedDocument,
                             content_analysis: ContentAnalysis) -> List[str]:
        """Generate specific improvement suggestions."""
        
        suggestions = []
        
        if scores['source_authority'] < 0.6:
            suggestions.append("Consider sourcing from more authoritative domains (.edu, .gov, established publishers)")
        
        if scores['citation_quality'] < 0.6:
            suggestions.append("Add properly formatted academic citations with verifiable sources")
        
        if scores['academic_rigor'] < 0.6:
            suggestions.append("Improve content structure with methodology, results, and conclusions sections")
        
        if scores['cross_reference_support'] < 0.6:
            suggestions.append("Verify claims against multiple authoritative sources")
        
        if scores['expert_consensus'] < 0.6:
            suggestions.append("Check consistency with established knowledge in the field")
        
        # Content-specific suggestions
        if scraped_doc.metadata.word_count < 500:
            suggestions.append("Content may be too brief for comprehensive analysis")
        
        if not content_analysis.claims:
            suggestions.append("Content lacks verifiable claims for fact-checking")
        
        if len(content_analysis.entities) < 3:
            suggestions.append("Content could benefit from more specific entities and terminology")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _generate_detailed_analysis(
        self,
        scores: Dict[str, float],
        scraped_doc: ScrapedDocument,
        content_analysis: ContentAnalysis,
        cross_ref_results: CrossReferenceResult
    ) -> Dict:
        """Generate detailed analysis for review."""
        
        primary_domain = 'general'
        if content_analysis.domain_mapping:
            primary_domain = max(content_analysis.domain_mapping.items(), key=lambda x: x[1])[0]
        
        return {
            'content_metrics': {
                'word_count': scraped_doc.metadata.word_count,
                'reading_time': scraped_doc.metadata.reading_time_minutes,
                'language': scraped_doc.metadata.language,
                'content_type': scraped_doc.metadata.content_type.value
            },
            'analysis_summary': {
                'primary_domain': primary_domain,
                'domain_confidence': max(content_analysis.domain_mapping.values()) if content_analysis.domain_mapping else 0,
                'entities_found': len(content_analysis.entities),
                'concepts_identified': len(content_analysis.concepts),
                'verifiable_claims': sum(1 for claim in content_analysis.claims if claim.verifiable),
                'total_claims': len(content_analysis.claims)
            },
            'validation_summary': {
                'supporting_sources': len(cross_ref_results.supporting_evidence),
                'contradicting_sources': len(cross_ref_results.contradicting_evidence),
                'source_reliability': cross_ref_results.source_reliability,
                'cross_reference_confidence': cross_ref_results.confidence
            },
            'component_breakdown': {
                component: {
                    'score': score,
                    'weight': self.weights[component],
                    'contribution': score * self.weights[component]
                }
                for component, score in scores.items()
            }
        }