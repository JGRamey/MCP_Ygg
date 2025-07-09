#!/usr/bin/env python3
"""Data models for Claim Analyzer Agent"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union


@dataclass
class Claim:
    """
    Represents a claim to be fact-checked.
    
    Attributes:
        claim_id: Unique identifier for the claim
        text: The actual claim text
        source: Where the claim originated from
        domain: Domain classification (science, math, religion, etc.)
        timestamp: When the claim was processed
        confidence: Confidence score for claim extraction (0.0-1.0)
        context: Surrounding context text
        entities: Named entities found in the claim
    """
    claim_id: str
    text: str
    source: str
    domain: str
    timestamp: datetime
    confidence: float = 0.0
    context: str = ""
    entities: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not self.claim_id:
            self.claim_id = hashlib.md5(f"{self.text}{self.source}".encode()).hexdigest()
        if self.entities is None:
            self.entities = []


@dataclass
class Evidence:
    """
    Represents evidence for or against a claim.
    
    Attributes:
        evidence_id: Unique identifier for the evidence
        text: The evidence text/content
        source_url: URL of the evidence source
        credibility_score: Source credibility rating (0.0-1.0)
        stance: Relationship to claim ("supports", "refutes", "neutral")
        domain: Domain of the evidence
        timestamp: When evidence was collected
        vector_embedding: Optional vector representation
    """
    evidence_id: str
    text: str
    source_url: str
    credibility_score: float
    stance: str  # "supports", "refutes", "neutral"
    domain: str
    timestamp: datetime
    vector_embedding: Optional[np.ndarray] = None


@dataclass
class FactCheckResult:
    """
    Represents the result of a fact-check operation.
    
    Attributes:
        claim: The original claim that was fact-checked
        verdict: Final verdict ("True", "False", "Partially True", "Unverified", "Opinion")
        confidence: Confidence in the verdict (0.0-1.0)
        evidence_list: List of evidence used in the fact-check
        reasoning: Textual explanation of the verdict
        sources: List of source URLs used
        cross_domain_patterns: Patterns found across different domains
        timestamp: When the fact-check was performed
        graph_node_id: Optional Neo4j node ID for the result
    """
    claim: Claim
    verdict: str  # "True", "False", "Partially True", "Unverified", "Opinion"
    confidence: float
    evidence_list: List[Evidence]
    reasoning: str
    sources: List[str]
    cross_domain_patterns: List[str]
    timestamp: datetime
    graph_node_id: Optional[str] = None