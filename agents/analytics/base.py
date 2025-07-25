"""Base classes and interfaces for analytics components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx
from neo4j import AsyncSession


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


class CommunityAlgorithm(Enum):
    """Community detection algorithms."""

    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    GREEDY_MODULARITY = "greedy_modularity"
    GIRVAN_NEWMAN = "girvan_newman"
    LABEL_PROPAGATION = "label_propagation"


@dataclass
class AnalysisConfig:
    """Configuration for network analysis."""

    min_node_degree: int = 1
    min_edge_weight: float = 0.1
    community_resolution: float = 1.0
    pattern_confidence: float = 0.7
    use_cache: bool = True
    cache_ttl: int = 3600
    max_nodes: int = 10000
    max_edges: int = 50000
    centrality_top_k: int = 10
    community_min_size: int = 3
    enable_metrics: bool = True
    parallel_processing: bool = True


@dataclass
class AnalysisResult:
    """Base result structure for analysis operations."""

    analysis_type: str
    timestamp: str
    node_count: int
    edge_count: int
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    data: Dict[str, Any] = None


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._graph: Optional[nx.Graph] = None
        self._directed_graph: Optional[nx.DiGraph] = None
        self._session: Optional[AsyncSession] = None

    @abstractmethod
    async def analyze(self, session: AsyncSession) -> AnalysisResult:
        """Perform analysis and return results."""
        pass

    @property
    def graph(self) -> nx.Graph:
        """Get or create undirected graph instance."""
        if self._graph is None:
            raise ValueError("Graph not initialized. Call load_graph first.")
        return self._graph

    @property
    def directed_graph(self) -> nx.DiGraph:
        """Get or create directed graph instance."""
        if self._directed_graph is None:
            raise ValueError("Directed graph not initialized. Call load_graph first.")
        return self._directed_graph

    async def load_graph(self, session: AsyncSession, directed: bool = False) -> None:
        """Load graph from Neo4j."""
        self._session = session

        # Query to get all concepts and their relationships
        query = """
        MATCH (n:Concept)
        OPTIONAL MATCH (n)-[r:RELATES_TO]-(m:Concept)
        RETURN n, r, m
        """

        result = await session.run(query)

        # Initialize graphs
        self._graph = nx.Graph()
        self._directed_graph = nx.DiGraph()

        async for record in result:
            node = record["n"]
            if not node:
                continue

            node_id = node.get("id", str(node.element_id))
            node_attrs = dict(node)

            # Add node to both graphs
            self._graph.add_node(node_id, **node_attrs)
            self._directed_graph.add_node(node_id, **node_attrs)

            # Add relationship if exists
            if record["r"] and record["m"]:
                rel = record["r"]
                other = record["m"]
                other_id = other.get("id", str(other.element_id))

                edge_attrs = {
                    "weight": rel.get("weight", 1.0),
                    "type": rel.get("type", "RELATES_TO"),
                    "strength": rel.get("strength", 1.0),
                }

                # Add edge to undirected graph
                self._graph.add_edge(node_id, other_id, **edge_attrs)

                # Add edge to directed graph (direction based on relationship)
                if directed:
                    self._directed_graph.add_edge(node_id, other_id, **edge_attrs)
                else:
                    self._directed_graph.add_edge(node_id, other_id, **edge_attrs)
                    self._directed_graph.add_edge(other_id, node_id, **edge_attrs)

    async def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Get attributes for a specific node."""
        if self._graph and node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return {}

    async def get_edge_attributes(self, source: str, target: str) -> Dict[str, Any]:
        """Get attributes for a specific edge."""
        if self._graph and self._graph.has_edge(source, target):
            return dict(self._graph[source][target])
        return {}

    def _validate_graph(self) -> bool:
        """Validate graph meets minimum requirements."""
        if not self._graph:
            return False

        node_count = self._graph.number_of_nodes()
        edge_count = self._graph.number_of_edges()

        # Check minimum requirements
        if node_count < 1:
            return False

        # Check maximum limits
        if node_count > self.config.max_nodes:
            return False

        if edge_count > self.config.max_edges:
            return False

        return True

    def _filter_nodes_by_degree(self, min_degree: int = None) -> List[str]:
        """Filter nodes by minimum degree."""
        min_degree = min_degree or self.config.min_node_degree
        return [node for node, degree in self._graph.degree() if degree >= min_degree]

    def _filter_edges_by_weight(self, min_weight: float = None) -> List[tuple]:
        """Filter edges by minimum weight."""
        min_weight = min_weight or self.config.min_edge_weight
        return [
            (u, v, data)
            for u, v, data in self._graph.edges(data=True)
            if data.get("weight", 1.0) >= min_weight
        ]

    def _create_subgraph(self, nodes: List[str]) -> nx.Graph:
        """Create subgraph from selected nodes."""
        return self._graph.subgraph(nodes).copy()

    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic graph metrics."""
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
            "is_connected": nx.is_connected(self._graph),
            "number_of_components": nx.number_connected_components(self._graph),
            "average_degree": (
                sum(dict(self._graph.degree()).values()) / self._graph.number_of_nodes()
                if self._graph.number_of_nodes() > 0
                else 0
            ),
            "clustering_coefficient": nx.average_clustering(self._graph),
            "transitivity": nx.transitivity(self._graph),
        }

    async def cleanup(self):
        """Clean up resources."""
        self._graph = None
        self._directed_graph = None
        self._session = None


class GraphMetrics:
    """Utility class for graph metric calculations."""

    @staticmethod
    def calculate_centrality(
        graph: nx.Graph, measure: CentralityMeasure, top_k: int = 10
    ) -> Dict[str, float]:
        """Calculate centrality measure for graph."""
        if measure == CentralityMeasure.PAGERANK:
            centrality = nx.pagerank(graph)
        elif measure == CentralityMeasure.BETWEENNESS:
            centrality = nx.betweenness_centrality(graph)
        elif measure == CentralityMeasure.CLOSENESS:
            centrality = nx.closeness_centrality(graph)
        elif measure == CentralityMeasure.EIGENVECTOR:
            try:
                centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                centrality = {}
        elif measure == CentralityMeasure.DEGREE:
            centrality = dict(nx.degree_centrality(graph))
        else:
            return {}

        # Return top K nodes
        return dict(
            sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

    @staticmethod
    def detect_communities(graph: nx.Graph, algorithm: CommunityAlgorithm) -> List[set]:
        """Detect communities using specified algorithm."""
        if algorithm == CommunityAlgorithm.LOUVAIN:
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(graph)
                communities = defaultdict(set)
                for node, comm_id in partition.items():
                    communities[comm_id].add(node)
                return list(communities.values())
            except ImportError:
                # Fallback to greedy modularity
                return list(nx.community.greedy_modularity_communities(graph))

        elif algorithm == CommunityAlgorithm.GREEDY_MODULARITY:
            return list(nx.community.greedy_modularity_communities(graph))

        elif algorithm == CommunityAlgorithm.GIRVAN_NEWMAN:
            communities = nx.community.girvan_newman(graph)
            return list(next(communities))

        elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
            return list(nx.community.label_propagation_communities(graph))

        else:
            return []

    @staticmethod
    def calculate_shortest_paths(
        graph: nx.Graph, source: str, targets: List[str] = None
    ) -> Dict[str, int]:
        """Calculate shortest paths from source to targets."""
        if targets is None:
            return dict(nx.shortest_path_length(graph, source))
        else:
            paths = {}
            for target in targets:
                try:
                    paths[target] = nx.shortest_path_length(graph, source, target)
                except nx.NetworkXNoPath:
                    paths[target] = float("inf")
            return paths
