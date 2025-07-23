#!/usr/bin/env python3
"""
Enhanced Vector Indexer Utilities
Constants, helper functions, and utility classes
"""

import logging
from typing import Dict, Any, List
import hashlib
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_DOMAINS = ['mathematics', 'science', 'philosophy', 'literature', 'history', 'art']
DEFAULT_VECTOR_SIZE = 384
DEFAULT_QUALITY_THRESHOLD = 0.6
DEFAULT_CONSISTENCY_THRESHOLD = 0.7

# Model availability flags
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Advanced ML libraries not available - using basic functionality")

# Base indexer availability
try:
    from .vector_indexer import VectorIndexer, VectorDocument, SearchResult, QdrantManager, RedisCache
    BASE_INDEXER_AVAILABLE = True
except ImportError:
    BASE_INDEXER_AVAILABLE = False
    logger.warning("Base vector indexer not found - implementing standalone")


def generate_cache_key(*args) -> str:
    """Generate MD5 cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


def normalize_language_code(language: str) -> str:
    """Normalize language code to standard format"""
    return language.lower().strip()[:2]  # First 2 characters, lowercase


def calculate_processing_speed(start_time: float, item_count: int) -> float:
    """Calculate processing speed (items per second)"""
    processing_time = time.time() - start_time
    return item_count / processing_time if processing_time > 0 else 0


def validate_embedding(embedding: np.ndarray) -> bool:
    """Basic validation of embedding vector"""
    if embedding is None or len(embedding) == 0:
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False
    
    # Check norm
    norm = np.linalg.norm(embedding)
    return 0.01 < norm < 100.0  # Reasonable norm range


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get standard model configurations"""
    return {
        'general': {
            'name': 'all-MiniLM-L6-v2',
            'strengths': ['speed', 'general_purpose', 'efficiency'],
            'languages': ['en', 'es', 'fr', 'de', 'it'],
            'vector_size': 384,
            'memory_requirement': 'low',
            'preferred_domains': ['general', 'history', 'literature']
        },
        'multilingual': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'strengths': ['multilingual', 'semantic_quality', 'cross_lingual'],
            'languages': ['en', 'zh', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'ja', 'ar'],
            'vector_size': 384,
            'memory_requirement': 'medium',
            'preferred_domains': ['all']
        },
        'semantic': {
            'name': 'all-mpnet-base-v2',
            'strengths': ['semantic_quality', 'academic', 'detailed_analysis'],
            'languages': ['en'],
            'vector_size': 768,
            'memory_requirement': 'high',
            'preferred_domains': ['philosophy', 'literature', 'science']
        },
        'academic': {
            'name': 'allenai/scibert_scivocab_uncased',
            'strengths': ['academic', 'scientific', 'technical'],
            'languages': ['en'],
            'vector_size': 768,
            'memory_requirement': 'high',
            'preferred_domains': ['science', 'mathematics', 'technology']
        },
        'fast': {
            'name': 'all-MiniLM-L12-v2',
            'strengths': ['speed', 'efficiency', 'real_time'],
            'languages': ['en'],
            'vector_size': 384,
            'memory_requirement': 'low',
            'preferred_domains': ['general']
        }
    }


def get_selection_criteria_weights() -> Dict[str, float]:
    """Get model selection criteria weights"""
    return {
        'language_support': 3.0,
        'domain_expertise': 2.0,
        'quality_score': 1.5,
        'processing_speed': 1.0,
        'memory_efficiency': 0.5
    }


def get_quality_thresholds() -> Dict[str, float]:
    """Get quality assessment thresholds"""
    return {
        'minimum_norm': 0.1,
        'maximum_norm': 10.0,
        'consistency_threshold': 0.7,
        'outlier_threshold': 2.0,
        'minimum_quality_score': 0.6,
        'reindex_threshold': 0.4,
        'semantic_density_threshold': 0.4
    }


def get_domain_specific_thresholds() -> Dict[str, Dict[str, Any]]:
    """Get domain-specific quality requirements"""
    return {
        'mathematics': {
            'preferred_models': ['academic', 'semantic'],
            'quality_threshold': 0.7,
            'consistency_requirement': 'high'
        },
        'science': {
            'preferred_models': ['academic', 'semantic'],
            'quality_threshold': 0.7,
            'consistency_requirement': 'high'
        },
        'philosophy': {
            'preferred_models': ['semantic', 'multilingual'],
            'quality_threshold': 0.65,
            'consistency_requirement': 'medium'
        },
        'literature': {
            'preferred_models': ['semantic', 'multilingual'],
            'quality_threshold': 0.6,
            'consistency_requirement': 'medium'
        },
        'history': {
            'preferred_models': ['general', 'multilingual'],
            'quality_threshold': 0.6,
            'consistency_requirement': 'medium'
        },
        'art': {
            'preferred_models': ['semantic', 'general'],
            'quality_threshold': 0.55,
            'consistency_requirement': 'low'
        }
    }


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.debug(f"{self.operation_name} completed in {duration:.3f}s")
    
    def get_duration(self) -> float:
        """Get duration of timed operation"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def create_mock_embedding(size: int = DEFAULT_VECTOR_SIZE) -> np.ndarray:
    """Create mock embedding for testing"""
    return np.random.rand(size)