#!/usr/bin/env python3
"""
Connection Analyzer
Analyzes and discovers connections between concepts
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStrength:
    """Represents the strength of a connection between concepts"""

    source_concept: str
    target_concept: str
    strength: float
    connection_type: str
    evidence: List[str]


class ConnectionAnalyzer:
    """Analyzes connections between concepts"""

    def __init__(self):
        self.connection_graph = nx.Graph()

    async def analyze_connections(
        self, concepts: List[Dict[str, Any]]
    ) -> List[ConnectionStrength]:
        """Analyze connections between concepts"""
        connections = []

        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i + 1 :], i + 1):
                strength = self._calculate_connection_strength(concept1, concept2)
                if strength > 0.5:  # Threshold for meaningful connections
                    connection = ConnectionStrength(
                        source_concept=concept1.get("name", f"concept_{i}"),
                        target_concept=concept2.get("name", f"concept_{j}"),
                        strength=strength,
                        connection_type="semantic_similarity",
                        evidence=[f"Similarity score: {strength:.3f}"],
                    )
                    connections.append(connection)

        return connections

    def _calculate_connection_strength(
        self, concept1: Dict[str, Any], concept2: Dict[str, Any]
    ) -> float:
        """Calculate connection strength between two concepts"""
        # Simplified connection strength calculation
        name1 = concept1.get("name", "").lower()
        name2 = concept2.get("name", "").lower()

        # Basic word overlap check
        words1 = set(name1.split())
        words2 = set(name2.split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0
