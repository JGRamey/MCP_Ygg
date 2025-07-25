"""
Data models for visualization components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class VisualizationType(Enum):
    """Types of visualizations that can be generated."""

    YGGDRASIL_TREE = "yggdrasil_tree"
    NETWORK_GRAPH = "network_graph"
    TIMELINE = "timeline"
    DOMAIN_CLUSTER = "domain_cluster"
    AUTHORITY_MAP = "authority_map"
    CONCEPT_MAP = "concept_map"
    RELATIONSHIP_FLOW = "relationship_flow"


class NodeType(Enum):
    """Node types with corresponding colors."""

    DOCUMENT = "document"
    CONCEPT = "concept"
    PERSON = "person"
    EVENT = "event"
    PATTERN = "pattern"
    ROOT = "root"
    DOMAIN = "domain"


@dataclass
class VisualizationNode:
    """Represents a node in the visualization."""

    id: str
    label: str
    title: str
    node_type: NodeType
    domain: Optional[str]
    date: Optional[str]
    level: int
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[float] = None
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationEdge:
    """Represents an edge in the visualization."""

    id: str
    source: str
    target: str
    relationship_type: str
    weight: float = 1.0
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationData:
    """Complete visualization data structure."""

    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any]
    layout_type: str
    filters: Dict[str, Any]
