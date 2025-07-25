"""
Centralized graph utilities for network analysis.
Eliminates redundant graph operations across modules.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncio
import networkx as nx
import numpy as np
from neo4j import AsyncDriver, AsyncGraphDatabase
from scipy import stats
from sklearn.cluster import DBSCAN

from .config import NetworkConfig
from .models import CommunityInfo, NodeMetrics


class GraphLoader:
    """Handles loading graphs from Neo4j with optimized queries."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.logger = logging.getLogger("graph_loader")

    async def load_graph(
        self,
        neo4j_driver: AsyncDriver,
        domain_scope: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        min_connections: int = 1,
    ) -> nx.Graph:
        """Load graph data from Neo4j with specified filters."""

        async with neo4j_driver.session() as session:
            # Build node filter
            node_filters = []
            params = {}

            if domain_scope:
                node_filters.append("n.domain IN $domains")
                params["domains"] = domain_scope

            if node_types:
                label_conditions = " OR ".join([f"n:{nt}" for nt in node_types])
                node_filters.append(f"({label_conditions})")

            node_where = "WHERE " + " AND ".join(node_filters) if node_filters else ""

            # Get nodes
            nodes_query = f"""
            MATCH (n)
            {node_where}
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as connections
            WHERE connections >= $min_connections
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                n.title as title,
                n.domain as domain,
                n.date as date,
                n.author as author,
                connections
            LIMIT $max_nodes
            """

            params.update(
                {"min_connections": min_connections, "max_nodes": self.config.max_nodes}
            )

            result = await session.run(nodes_query, params)

            # Create NetworkX graph
            G = nx.Graph()
            node_data = {}

            async for record in result:
                node_id = str(record["node_id"])
                node_info = {
                    "title": record["title"],
                    "domain": record["domain"],
                    "date": record["date"],
                    "author": record["author"],
                    "labels": record["labels"],
                    "connections": record["connections"],
                }
                G.add_node(node_id, **node_info)
                node_data[node_id] = node_info

            if len(G.nodes) == 0:
                return G

            # Get relationships
            rel_filters = []
            if relationship_types:
                rel_filters.append("type(r) IN $rel_types")
                params["rel_types"] = relationship_types

            rel_where = "AND " + " AND ".join(rel_filters) if rel_filters else ""

            relationships_query = f"""
            MATCH (n1)-[r]-(n2)
            WHERE id(n1) IN $node_ids AND id(n2) IN $node_ids
            {rel_where}
            RETURN 
                id(n1) as source,
                id(n2) as target,
                type(r) as rel_type,
                r.weight as weight
            LIMIT $max_edges
            """

            params.update(
                {"node_ids": list(node_data.keys()), "max_edges": self.config.max_edges}
            )

            result = await session.run(relationships_query, params)

            async for record in result:
                source = str(record["source"])
                target = str(record["target"])
                weight = record.get("weight", 1.0)
                rel_type = record["rel_type"]

                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, weight=weight, rel_type=rel_type)

            return G


class CentralityCalculator:
    """Centralized centrality calculations with error handling."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.logger = logging.getLogger("centrality_calculator")

    def calculate_all_centralities(
        self, graph: nx.Graph
    ) -> Dict[str, Dict[str, float]]:
        """Calculate all centrality measures for the graph."""

        centrality_measures = {}

        # PageRank
        centrality_measures["pagerank"] = self._safe_pagerank(graph)

        # Betweenness centrality
        centrality_measures["betweenness"] = self._safe_betweenness(graph)

        # Closeness centrality
        centrality_measures["closeness"] = self._safe_closeness(graph)

        # Eigenvector centrality
        centrality_measures["eigenvector"] = self._safe_eigenvector(graph)

        # Degree centrality
        centrality_measures["degree"] = nx.degree_centrality(graph)

        return centrality_measures

    def _safe_pagerank(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate PageRank with error handling."""
        try:
            return nx.pagerank(
                graph,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol,
            )
        except Exception as e:
            self.logger.warning(f"PageRank calculation failed: {e}")
            return {node: 0.0 for node in graph.nodes}

    def _safe_betweenness(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate betweenness centrality with error handling."""
        try:
            return nx.betweenness_centrality(
                graph, normalized=self.config.normalize_centrality
            )
        except Exception as e:
            self.logger.warning(f"Betweenness centrality calculation failed: {e}")
            return {node: 0.0 for node in graph.nodes}

    def _safe_closeness(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate closeness centrality with error handling."""
        try:
            return nx.closeness_centrality(
                graph, normalized=self.config.normalize_centrality
            )
        except Exception as e:
            self.logger.warning(f"Closeness centrality calculation failed: {e}")
            return {node: 0.0 for node in graph.nodes}

    def _safe_eigenvector(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate eigenvector centrality with error handling."""
        try:
            return nx.eigenvector_centrality(graph, max_iter=1000)
        except Exception as e:
            self.logger.warning(f"Eigenvector centrality calculation failed: {e}")
            return {node: 0.0 for node in graph.nodes}


class ClusteringAnalyzer:
    """Centralized clustering calculations."""

    def __init__(self):
        self.logger = logging.getLogger("clustering_analyzer")

    def calculate_clustering_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics."""

        metrics = {}

        # Basic clustering metrics
        metrics["node_clustering"] = nx.clustering(graph)
        metrics["average_clustering"] = nx.average_clustering(graph)
        metrics["transitivity"] = nx.transitivity(graph)

        # Triangle analysis
        metrics["triangles"] = nx.triangles(graph)

        # Square clustering (for 4-cycles)
        try:
            metrics["square_clustering"] = nx.square_clustering(graph)
        except Exception:
            metrics["square_clustering"] = {node: 0.0 for node in graph.nodes}

        return metrics


class ConnectivityAnalyzer:
    """Centralized connectivity and path analysis."""

    def __init__(self):
        self.logger = logging.getLogger("connectivity_analyzer")

    def analyze_connectivity(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze graph connectivity properties."""

        metrics = {
            "is_connected": nx.is_connected(graph),
            "num_components": nx.number_connected_components(graph),
        }

        if metrics["is_connected"]:
            metrics["diameter"] = nx.diameter(graph)
            metrics["radius"] = nx.radius(graph)
            metrics["avg_path_length"] = nx.average_shortest_path_length(graph)
            metrics["center_nodes"] = list(nx.center(graph))
        else:
            # Analyze largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)

            if len(largest_cc) > 1:
                metrics["diameter"] = nx.diameter(largest_subgraph)
                metrics["radius"] = nx.radius(largest_subgraph)
                metrics["avg_path_length"] = nx.average_shortest_path_length(
                    largest_subgraph
                )
                metrics["center_nodes"] = list(nx.center(largest_subgraph))
            else:
                metrics["diameter"] = 0
                metrics["radius"] = 0
                metrics["avg_path_length"] = 0
                metrics["center_nodes"] = []

        return metrics

    def calculate_efficiency_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate graph efficiency metrics."""

        metrics = {}

        try:
            # Global efficiency
            metrics["global_efficiency"] = nx.global_efficiency(graph)

            # Local efficiency
            metrics["local_efficiency"] = nx.local_efficiency(graph)

            # Average efficiency
            if nx.is_connected(graph):
                all_pairs_shortest = dict(nx.all_pairs_shortest_path_length(graph))
                all_lengths = []
                for source, targets in all_pairs_shortest.items():
                    for target, length in targets.items():
                        if source != target:
                            all_lengths.append(length)

                if all_lengths:
                    metrics["average_efficiency"] = np.mean(
                        [1.0 / length for length in all_lengths]
                    )
                else:
                    metrics["average_efficiency"] = 0.0
            else:
                metrics["average_efficiency"] = 0.0

        except Exception as e:
            self.logger.warning(f"Efficiency calculation failed: {e}")
            metrics["global_efficiency"] = 0.0
            metrics["local_efficiency"] = 0.0
            metrics["average_efficiency"] = 0.0

        return metrics


class GraphStatistics:
    """Centralized basic graph statistics."""

    @staticmethod
    def calculate_basic_stats(graph: nx.Graph) -> Dict[str, Any]:
        """Calculate basic graph statistics."""

        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
        }

        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        if degrees:
            stats["avg_degree"] = np.mean(degrees)
            stats["degree_std"] = np.std(degrees)
            stats["max_degree"] = max(degrees)
            stats["min_degree"] = min(degrees)
            stats["degree_distribution"] = Counter(degrees)
        else:
            stats["avg_degree"] = 0
            stats["degree_std"] = 0
            stats["max_degree"] = 0
            stats["min_degree"] = 0
            stats["degree_distribution"] = {}

        # Assortativity
        try:
            stats["assortativity"] = nx.degree_assortativity_coefficient(graph)
        except Exception:
            stats["assortativity"] = 0.0

        return stats


class GraphValidator:
    """Graph validation utilities."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.logger = logging.getLogger("graph_validator")

    def validate_graph(self, graph: nx.Graph) -> Tuple[bool, List[str]]:
        """Validate graph meets analysis requirements."""

        issues = []

        # Check size constraints
        if len(graph.nodes) == 0:
            issues.append("Graph has no nodes")
        elif len(graph.nodes) > self.config.max_nodes:
            issues.append(
                f"Graph too large: {len(graph.nodes)} > {self.config.max_nodes}"
            )

        if len(graph.edges) > self.config.max_edges:
            issues.append(
                f"Too many edges: {len(graph.edges)} > {self.config.max_edges}"
            )

        # Check connectivity
        if len(graph.nodes) > 1 and len(graph.edges) == 0:
            issues.append("Graph has nodes but no edges")

        # Check for self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            issues.append(f"Graph contains {len(self_loops)} self-loops")

        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if len(isolated) > len(graph.nodes) * 0.5:
            issues.append(f"Too many isolated nodes: {len(isolated)}")

        return len(issues) == 0, issues


class TemporalGraphUtils:
    """Utilities for temporal graph analysis."""

    @staticmethod
    def create_temporal_directed_graph(graph: nx.Graph) -> nx.DiGraph:
        """Create directed graph based on temporal information."""

        directed_graph = nx.DiGraph()

        # Add all nodes
        for node, data in graph.nodes(data=True):
            directed_graph.add_node(node, **data)

        # Add directed edges based on dates
        for u, v, data in graph.edges(data=True):
            u_date = graph.nodes[u].get("date")
            v_date = graph.nodes[v].get("date")

            if u_date and v_date:
                try:
                    # Convert to datetime if needed
                    if isinstance(u_date, str):
                        u_datetime = datetime.fromisoformat(
                            u_date.replace("Z", "+00:00")
                        )
                    else:
                        u_datetime = u_date

                    if isinstance(v_date, str):
                        v_datetime = datetime.fromisoformat(
                            v_date.replace("Z", "+00:00")
                        )
                    else:
                        v_datetime = v_date

                    # Add edge from older to newer
                    if u_datetime < v_datetime:
                        directed_graph.add_edge(u, v, **data)
                    elif v_datetime < u_datetime:
                        directed_graph.add_edge(v, u, **data)
                    else:
                        # Same date, add both directions
                        directed_graph.add_edge(u, v, **data)
                        directed_graph.add_edge(v, u, **data)

                except Exception:
                    # If date parsing fails, add both directions
                    directed_graph.add_edge(u, v, **data)
                    directed_graph.add_edge(v, u, **data)
            else:
                # No date information, add both directions
                directed_graph.add_edge(u, v, **data)
                directed_graph.add_edge(v, u, **data)

        return directed_graph


class GraphMetricsAggregator:
    """Aggregates various graph metrics."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.centrality_calc = CentralityCalculator(config)
        self.clustering_analyzer = ClusteringAnalyzer()
        self.connectivity_analyzer = ConnectivityAnalyzer()

    def calculate_comprehensive_metrics(
        self,
        graph: nx.Graph,
        centrality_measures: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics."""

        metrics = {}

        # Basic statistics
        metrics.update(GraphStatistics.calculate_basic_stats(graph))

        # Clustering metrics
        clustering_metrics = self.clustering_analyzer.calculate_clustering_metrics(
            graph
        )
        metrics["avg_clustering"] = clustering_metrics["average_clustering"]
        metrics["transitivity"] = clustering_metrics["transitivity"]

        # Connectivity metrics
        connectivity_metrics = self.connectivity_analyzer.analyze_connectivity(graph)
        metrics.update(connectivity_metrics)

        # Efficiency metrics
        efficiency_metrics = self.connectivity_analyzer.calculate_efficiency_metrics(
            graph
        )
        metrics.update(efficiency_metrics)

        # Add centrality statistics if provided
        if centrality_measures:
            for measure_name, scores in centrality_measures.items():
                if scores:
                    values = list(scores.values())
                    metrics[f"{measure_name}_mean"] = np.mean(values)
                    metrics[f"{measure_name}_std"] = np.std(values)
                    metrics[f"{measure_name}_max"] = np.max(values)

        return metrics
