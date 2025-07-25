#!/usr/bin/env python3
"""
Text Processor Utility Functions
Essential utility functions for text processing (unused code removed)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    print("Text processor utilities loaded successfully")
