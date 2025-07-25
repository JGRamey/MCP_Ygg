#!/usr/bin/env python3
"""
Enhanced Vector Indexer Data Models
Data classes and schemas for vector indexing operations
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


@dataclass
class ModelPerformance:
    """Model performance metrics for embedding models"""

    model_name: str
    accuracy: float
    speed: float  # embeddings per second
    quality_score: float
    memory_usage: float  # MB
    vector_size: int
    supported_languages: List[str]
    last_evaluated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "speed": self.speed,
            "quality_score": self.quality_score,
            "memory_usage": self.memory_usage,
            "vector_size": self.vector_size,
            "supported_languages": self.supported_languages,
            "last_evaluated": self.last_evaluated.isoformat(),
        }


@dataclass
class EmbeddingQuality:
    """Quality assessment results for generated embeddings"""

    doc_id: str
    model_used: str
    quality_score: float  # 0-1 scale overall quality
    consistency_score: float  # Embedding consistency with similar content
    semantic_density: float  # Information density measure
    outlier_score: float  # How much it differs from typical embeddings
    recommendations: List[str]  # Improvement suggestions

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if embedding meets quality threshold"""
        return self.quality_score >= threshold

    def needs_reprocessing(self, threshold: float = 0.4) -> bool:
        """Check if embedding should be reprocessed"""
        return self.quality_score < threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "doc_id": self.doc_id,
            "model_used": self.model_used,
            "quality_score": self.quality_score,
            "consistency_score": self.consistency_score,
            "semantic_density": self.semantic_density,
            "outlier_score": self.outlier_score,
            "recommendations": self.recommendations,
        }


@dataclass
class VectorIndexResult:
    """Enhanced result from vector indexing operation"""

    vector_id: str
    embedding: np.ndarray
    model_used: str
    quality_assessment: EmbeddingQuality
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float

    def is_successful(self) -> bool:
        """Check if indexing was successful"""
        return (
            self.embedding is not None
            and self.quality_assessment.quality_score > 0.5
            and self.confidence_score > 0.6
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of indexing result"""
        return {
            "vector_id": self.vector_id,
            "model_used": self.model_used,
            "quality_score": self.quality_assessment.quality_score,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "successful": self.is_successful(),
        }


@dataclass
class ModelConfig:
    """Configuration for embedding models"""

    name: str
    model_path: str
    strengths: List[str]
    languages: List[str]
    vector_size: int
    memory_requirement: str  # 'low', 'medium', 'high'
    preferred_domains: List[str]

    def supports_language(self, language: str) -> bool:
        """Check if model supports given language"""
        return language.lower() in [lang.lower() for lang in self.languages]

    def is_good_for_domain(self, domain: str) -> bool:
        """Check if model is suitable for domain"""
        return "all" in self.preferred_domains or domain.lower() in [
            d.lower() for d in self.preferred_domains
        ]
