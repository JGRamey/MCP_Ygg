"""Data models for network analysis."""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

class AnalysisType(Enum):
    """Types of network analysis that can be performed."""
    CENTRALITY = "centrality"
    COMMUNITY_DETECTION = "community_detection"
    INFLUENCE_PROPAGATION = "influence_propagation"
    KNOWLEDGE_FLOW = "knowledge_flow"
    BRIDGE_NODES = "bridge_nodes"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    PATH_ANALYSIS = "path_analysis"


class CentralityMeasure(Enum):
    """Types of centrality measures."""
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    DEGREE = "degree"
    KATZ = "katz"
    HARMONIC = "harmonic"


class CommunityAlgorithm(Enum):
    """Community detection algorithms."""
    GIRVAN_NEWMAN = "girvan_newman"
    GREEDY_MODULARITY = "greedy_modularity"
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    LEIDEN = "leiden"
    FLUID_COMMUNITIES = "fluid_communities"


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    centrality_scores: Dict[str, float]
    community_id: Optional[int]
    clustering_coefficient: float
    degree: int
    metadata: Dict[str, Any]


@dataclass
class CommunityInfo:
    """Information about a detected community."""
    community_id: int
    nodes: List[str]
    size: int
    internal_edges: int
    external_edges: int
    modularity_contribution: float
    description: str
    metadata: Dict[str, Any]


@dataclass
class NetworkAnalysisResult:
    """Complete network analysis result."""
    analysis_type: AnalysisType
    graph_metrics: Dict[str, float]
    node_metrics: List[NodeMetrics]
    communities: List[CommunityInfo]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    execution_time: float
