#!/usr/bin/env python3
"""
Enhanced Text Processor Data Models
Data classes and schemas for text processing with multilingual support
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ProcessedText:
    """Enhanced processed text with additional fields"""
    original_text: str
    language: str
    language_confidence: float
    entities: List[Dict]
    concepts: List[Dict]
    summary: str
    sentiment: Dict
    key_phrases: List[str]
    linked_entities: List[Dict]  # Linked to knowledge graph
    embeddings: np.ndarray
    processing_metadata: Dict


@dataclass
class LinkedEntity:
    """Entity linked to knowledge graph"""
    text: str
    label: str
    kb_id: Optional[str]
    kb_type: Optional[str]
    confidence: float
    properties: Dict