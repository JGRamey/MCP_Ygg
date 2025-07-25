#!/usr/bin/env python3
"""
Enhanced Text Processor Package
Modular text processing with multilingual support and transformers
"""

# Import main classes for easy access
from agents.text_processor.enhanced_text_processor import EnhancedTextProcessor
from agents.text_processor.entity_linker import EntityLinker
from agents.text_processor.models import LinkedEntity, ProcessedText
from agents.text_processor.multilingual_processor import MultilingualProcessor
from agents.text_processor.utils import load_config, load_processed_document

# Import base text processor components (maintain backward compatibility)
try:
    from agents.text_processor.text_processor import (
        EmbeddingGenerator,
        Entity,
        ProcessedDocument,
        TextChunk,
        TextChunker,
        TextCleaner,
        TextProcessor,
    )
except ImportError:
    # Graceful degradation if base text processor not available
    pass

__all__ = [
    "EnhancedTextProcessor",
    "ProcessedText",
    "LinkedEntity",
    "EntityLinker",
    "MultilingualProcessor",
    "load_config",
    "load_processed_document",
]

__version__ = "2.0.0"
__author__ = "MCP Yggdrasil Team"
