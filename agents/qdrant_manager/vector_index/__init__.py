#!/usr/bin/env python3
"""
Enhanced Vector Index Package
Modular vector indexing with dynamic model selection and quality checking

This package provides:
- Dynamic embedding model selection
- Comprehensive quality assessment
- Enhanced vector operations
- Performance monitoring and optimization
"""

from .enhanced_indexer import EnhancedVectorIndexer

# Import main components
from .model_manager import ModelManager

# Import core data models
from .models import EmbeddingQuality, ModelConfig, ModelPerformance, VectorIndexResult
from .quality_checker import EmbeddingQualityChecker

# Import utilities
from .utils import (
    BASE_INDEXER_AVAILABLE,
    SUPPORTED_DOMAINS,
    TRANSFORMERS_AVAILABLE,
    generate_cache_key,
    get_model_configs,
    get_quality_thresholds,
    validate_embedding,
)

# Version info
__version__ = "2.0.0"
__author__ = "MCP Yggdrasil Team"

# Package-level exports
__all__ = [
    # Data Models
    "ModelPerformance",
    "EmbeddingQuality",
    "VectorIndexResult",
    "ModelConfig",
    # Core Components
    "ModelManager",
    "EmbeddingQualityChecker",
    "EnhancedVectorIndexer",
    # Utilities
    "TRANSFORMERS_AVAILABLE",
    "BASE_INDEXER_AVAILABLE",
    "SUPPORTED_DOMAINS",
    "generate_cache_key",
    "validate_embedding",
    "get_model_configs",
    "get_quality_thresholds",
]


# Convenience factory function
def create_enhanced_indexer(config_path: str = None) -> EnhancedVectorIndexer:
    """
    Factory function to create EnhancedVectorIndexer instance

    Args:
        config_path: Optional path to configuration file

    Returns:
        EnhancedVectorIndexer instance
    """
    if config_path:
        return EnhancedVectorIndexer(config_path)
    else:
        return EnhancedVectorIndexer()
