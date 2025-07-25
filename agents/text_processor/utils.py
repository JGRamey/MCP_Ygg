#!/usr/bin/env python3
"""
Text Processor Utility Functions
Configuration loading and helper utilities for text processing
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file with fallback defaults"""
    try:
        with open(config_path, "r") as f:
            import yaml

            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        # Default configuration
        return {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_summary_length": 150,
            "min_summary_length": 50,
            "enable_caching": True,
            "max_cache_size": 1000,
            "batch_size": 16,
            "enable_gpu": torch.cuda.is_available(),
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_summary_length": 150,
            "min_summary_length": 50,
            "enable_caching": True,
            "max_cache_size": 1000,
            "batch_size": 16,
            "enable_gpu": torch.cuda.is_available(),
        }


def load_processed_document(
    doc_id: str, base_dir: str = "data/processed"
) -> Optional[Dict]:
    """Load a processed document from storage"""
    doc_file = Path(base_dir) / f"{doc_id}.json"
    embeddings_file = Path(base_dir) / f"{doc_id}_embeddings.pkl"

    try:
        # Load main document data
        with open(doc_file, "r") as f:
            doc_data = json.load(f)

        # Load embeddings if available
        if embeddings_file.exists():
            with open(embeddings_file, "rb") as f:
                embeddings_data = pickle.load(f)
                doc_data["embeddings"] = embeddings_data["doc_embedding"]
                doc_data["chunk_embeddings"] = embeddings_data["chunk_embeddings"]

        return doc_data

    except Exception as e:
        logger.error(f"Error loading document {doc_id}: {e}")
        return None
