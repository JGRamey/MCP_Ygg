#!/usr/bin/env python3
"""
Embedding Quality Checker for Enhanced Vector Indexer
Assesses and monitors the quality of generated embeddings
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .models import EmbeddingQuality
from .utils import (
    TRANSFORMERS_AVAILABLE,
    get_quality_thresholds,
    validate_embedding
)

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingQualityChecker:
    """Assesses the quality of generated embeddings"""
    
    def __init__(self):
        self.quality_thresholds = get_quality_thresholds()
        self.reference_embeddings = {}  # Store reference embeddings for comparison
        self.domain_statistics = {}     # Track domain-specific stats
    
    def assess_quality(
        self, 
        embedding: np.ndarray, 
        content: Dict[str, Any], 
        model_used: str
    ) -> EmbeddingQuality:
        """Comprehensive quality assessment of an embedding"""
        doc_id = content.get('doc_id', 'unknown')
        
        # Basic validation
        if not validate_embedding(embedding):
            return self._create_failed_assessment(doc_id, model_used, "Invalid embedding vector")
        
        # Initialize quality metrics
        quality_score = 1.0
        recommendations = []
        
        # Perform quality checks
        norm_quality, norm_recommendations = self._assess_vector_norm(embedding)
        quality_score *= norm_quality
        recommendations.extend(norm_recommendations)
        
        # Consistency assessment
        consistency_score = self._assess_consistency(embedding, content, model_used)
        
        # Semantic density assessment
        semantic_density = self._calculate_semantic_density(embedding, content)
        
        # Outlier detection
        outlier_score = self._detect_outliers(embedding, content.get('domain', 'general'))
        
        # Overall quality adjustment based on consistency and density
        quality_score = min(1.0, max(0.0, 
            quality_score * 0.6 + 
            consistency_score * 0.25 + 
            semantic_density * 0.15
        ))
        
        # Apply outlier penalty
        if outlier_score > self.quality_thresholds['outlier_threshold']:
            quality_score *= 0.9
            recommendations.append("Embedding appears to be an outlier - content may be unusual")
        
        # Generate recommendations based on scores
        recommendations.extend(self._generate_recommendations(
            quality_score, consistency_score, semantic_density, outlier_score
        ))
        
        return EmbeddingQuality(
            doc_id=doc_id,
            model_used=model_used,
            quality_score=quality_score,
            consistency_score=consistency_score,
            semantic_density=semantic_density,
            outlier_score=outlier_score,
            recommendations=recommendations
        )
    
    def _create_failed_assessment(self, doc_id: str, model_used: str, reason: str) -> EmbeddingQuality:
        """Create assessment for failed embedding"""
        return EmbeddingQuality(
            doc_id=doc_id,
            model_used=model_used,
            quality_score=0.0,
            consistency_score=0.0,
            semantic_density=0.0,
            outlier_score=0.0,
            recommendations=[f"Failed assessment: {reason}"]
        )
    
    def _assess_vector_norm(self, embedding: np.ndarray) -> tuple[float, List[str]]:
        """Assess vector norm quality"""
        embedding_norm = np.linalg.norm(embedding)
        quality_score = 1.0
        recommendations = []
        
        if embedding_norm < self.quality_thresholds['minimum_norm']:
            quality_score = 0.3
            recommendations.append("Embedding norm too low - may indicate poor text quality or processing issues")
        elif embedding_norm > self.quality_thresholds['maximum_norm']:
            quality_score = 0.7
            recommendations.append("Embedding norm too high - may indicate outlier content or model issues")
        elif embedding_norm < 0.5:
            quality_score = 0.8
            recommendations.append("Embedding norm lower than typical - consider text preprocessing")
        
        return quality_score, recommendations
    
    def _assess_consistency(self, embedding: np.ndarray, content: Dict[str, Any], model_used: str) -> float:
        """Assess embedding consistency with similar content"""
        domain = content.get('domain', 'general')
        reference_key = f"{domain}_{model_used}"
        
        if reference_key not in self.reference_embeddings:
            self.reference_embeddings[reference_key] = []
        
        references = self.reference_embeddings[reference_key]
        
        if len(references) < 3:
            # Not enough references yet - store and return neutral score
            references.append(embedding.copy())
            return 0.8  # Assume decent consistency
        
        # Calculate similarity with recent references
        similarities = []
        if TRANSFORMERS_AVAILABLE:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                for ref_embedding in references[-10:]:  # Use last 10 references
                    similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
                    similarities.append(similarity)
            except Exception as e:
                logger.debug(f"Error calculating similarity: {e}")
                similarities = [0.7]  # Fallback
        else:
            # Simple dot product similarity as fallback
            for ref_embedding in references[-5:]:
                # Normalize vectors
                norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
                norm_ref = ref_embedding / (np.linalg.norm(ref_embedding) + 1e-8)
                similarity = np.dot(norm_emb, norm_ref)
                similarities.append(similarity)
        
        if similarities:
            consistency = np.mean(similarities)
            
            # Add current embedding to references
            references.append(embedding.copy())
            
            # Keep only recent references to prevent memory issues
            if len(references) > 100:
                references[:] = references[-50:]
            
            return max(0.0, min(1.0, consistency))
        
        return 0.7  # Default consistency
    
    def _calculate_semantic_density(self, embedding: np.ndarray, content: Dict[str, Any]) -> float:
        """Calculate how much semantic information is captured"""
        text = content.get('text', '')
        text_length = len(text.split()) if text else 0
        
        if text_length == 0:
            return 0.0
        
        # Calculate embedding statistics
        embedding_norm = np.linalg.norm(embedding)
        embedding_std = np.std(embedding)
        
        # Heuristic: good embeddings should have reasonable norm and variance
        # relative to text complexity
        complexity_factor = min(1.0, np.log(text_length + 1) / 10.0)
        norm_factor = min(1.0, embedding_norm / 5.0)  # Normalize around expected norm
        variance_factor = min(1.0, embedding_std * 10)  # Good embeddings have some variance
        
        density = (complexity_factor * 0.4 + norm_factor * 0.4 + variance_factor * 0.2)
        
        return max(0.0, min(1.0, density))
    
    def _detect_outliers(self, embedding: np.ndarray, domain: str) -> float:
        """Detect if embedding is an outlier for the domain"""
        reference_key = f"outlier_{domain}"
        
        if reference_key not in self.reference_embeddings:
            self.reference_embeddings[reference_key] = []
        
        domain_embeddings = self.reference_embeddings[reference_key]
        
        if len(domain_embeddings) < 10:
            # Not enough data for reliable outlier detection
            domain_embeddings.append(embedding.copy())
            return 0.5  # Neutral outlier score
        
        try:
            # Calculate distance from domain centroid
            centroid = np.mean(domain_embeddings, axis=0)
            distance = np.linalg.norm(embedding - centroid)
            
            # Calculate standard deviation of distances
            distances = [np.linalg.norm(emb - centroid) for emb in domain_embeddings]
            std_distance = np.std(distances)
            
            if std_distance < 1e-8:  # Avoid division by zero
                outlier_score = 1.0
            else:
                # Outlier score based on how many standard deviations away
                outlier_score = distance / std_distance
            
            # Add to domain embeddings
            domain_embeddings.append(embedding.copy())
            
            # Keep manageable size
            if len(domain_embeddings) > 200:
                domain_embeddings[:] = domain_embeddings[-100:]
            
            return min(5.0, outlier_score)  # Cap at 5 standard deviations
            
        except Exception as e:
            logger.debug(f"Error in outlier detection: {e}")
            return 1.0  # Neutral score on error
    
    def _generate_recommendations(
        self, 
        quality_score: float, 
        consistency_score: float, 
        semantic_density: float, 
        outlier_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on quality metrics"""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Consider trying a different embedding model for better quality")
        
        if consistency_score < 0.6:
            recommendations.append("Content may be too noisy or ambiguous - consider text preprocessing")
        
        if semantic_density < 0.4:
            recommendations.append("Content may lack semantic richness - ensure sufficient meaningful text")
        
        if outlier_score > 2.5:
            recommendations.append("Content appears unusual for its domain - verify content appropriateness")
        
        if quality_score < 0.4:
            recommendations.append("Consider reprocessing with different parameters or manual review")
        
        # Positive recommendations
        if quality_score > 0.85 and consistency_score > 0.8:
            recommendations.append("High-quality embedding - suitable for production use")
        
        return recommendations
    
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get quality statistics for a domain"""
        reference_key = f"outlier_{domain}"
        
        if reference_key not in self.reference_embeddings:
            return {"error": f"No statistics available for domain: {domain}"}
        
        embeddings = self.reference_embeddings[reference_key]
        
        if len(embeddings) < 5:
            return {"error": f"Insufficient data for domain statistics: {domain}"}
        
        try:
            # Calculate domain statistics
            embedding_array = np.array(embeddings)
            centroid = np.mean(embedding_array, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            
            return {
                "domain": domain,
                "sample_count": len(embeddings),
                "average_distance_from_centroid": np.mean(distances),
                "std_distance": np.std(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "centroid_norm": np.linalg.norm(centroid)
            }
            
        except Exception as e:
            logger.error(f"Error calculating domain statistics: {e}")
            return {"error": f"Failed to calculate statistics: {str(e)}"}
    
    def update_quality_thresholds(self, domain: str, feedback_data: List[Dict[str, Any]]) -> None:
        """Update quality thresholds based on user feedback"""
        if not feedback_data:
            return
        
        # Analyze feedback to adjust thresholds
        quality_scores = []
        user_ratings = []
        
        for feedback in feedback_data:
            if 'quality_score' in feedback and 'user_rating' in feedback:
                quality_scores.append(feedback['quality_score'])
                user_ratings.append(feedback['user_rating'])
        
        if len(quality_scores) >= 10:  # Need sufficient data
            # Simple threshold adjustment based on correlation
            correlation = np.corrcoef(quality_scores, user_ratings)[0, 1]
            
            if correlation > 0.5:  # Good correlation
                # Find threshold that maximizes user satisfaction
                sorted_pairs = sorted(zip(quality_scores, user_ratings))
                best_threshold = self.quality_thresholds['minimum_quality_score']
                
                for quality, rating in sorted_pairs:
                    if rating >= 0.7:  # Good user rating
                        best_threshold = max(best_threshold, quality * 0.9)
                        break
                
                # Update threshold (with conservative adjustment)
                old_threshold = self.quality_thresholds['minimum_quality_score']
                new_threshold = old_threshold * 0.8 + best_threshold * 0.2
                self.quality_thresholds['minimum_quality_score'] = new_threshold
                
                logger.info(f"Updated quality threshold for {domain}: {old_threshold:.3f} -> {new_threshold:.3f}")
    
    def clear_references(self, domain: Optional[str] = None) -> None:
        """Clear reference embeddings for memory management"""
        if domain:
            keys_to_clear = [k for k in self.reference_embeddings.keys() if domain in k]
            for key in keys_to_clear:
                del self.reference_embeddings[key]
            logger.info(f"Cleared reference embeddings for domain: {domain}")
        else:
            self.reference_embeddings.clear()
            logger.info("Cleared all reference embeddings")