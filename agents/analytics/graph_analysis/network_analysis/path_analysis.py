"""
Path Analysis Module for Network Analysis
Analyzes path structures, distances, and network connectivity patterns.
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from ..graph_utils import ConnectivityAnalyzer, GraphLoader, GraphStatistics
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class PathAnalyzer:
    """Analyzes path structures and distances in networks."""

    def __init__(self, config=None):
        """Initialize the path analyzer."""
        self.config = config
        self.logger = self._setup_logging()

        # Initialize utility classes
        self.graph_loader = GraphLoader(config)
        self.connectivity_analyzer = ConnectivityAnalyzer()
        self.graph_statistics = GraphStatistics()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("path_analyzer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    async def analyze_paths(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze path structures and distances in the network."""

        self.logger.info("Analyzing path structures...")

        try:
            # Check if graph is connected
            if not nx.is_connected(graph):
                # Work with largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                working_graph = graph.subgraph(largest_cc).copy()
                self.logger.info(
                    f"Working with largest connected component ({len(largest_cc)} nodes)"
                )
            else:
                working_graph = graph

            num_nodes = working_graph.number_of_nodes()

            if num_nodes < 2:
                # Not enough nodes for path analysis
                return self._create_empty_result()

            # Calculate path metrics
            path_metrics = self._calculate_path_metrics(working_graph)

            # Create node-level path metrics
            node_metrics = self._create_node_metrics(graph, working_graph, path_metrics)

            # Compile graph metrics
            graph_metrics = self._compile_graph_metrics(path_metrics, num_nodes)

            # Generate insights and recommendations
            insights = self._generate_insights(graph_metrics, path_metrics)
            recommendations = self._generate_recommendations(graph_metrics)

            return NetworkAnalysisResult(
                analysis_type=AnalysisType.PATH_ANALYSIS,
                graph_metrics=graph_metrics,
                node_metrics=node_metrics,
                communities=[],
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.now(timezone.utc),
                execution_time=0.0,
            )

        except Exception as e:
            self.logger.error(f"Error in path analysis: {e}")
            raise

    def _calculate_path_metrics(self, working_graph: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive path metrics for the network."""

        num_nodes = working_graph.number_of_nodes()
        path_metrics = {}

        try:
            if num_nodes <= 1000:  # Full calculation for smaller graphs
                # All pairs shortest paths
                path_lengths = dict(nx.all_pairs_shortest_path_length(working_graph))

                # Extract all path lengths
                all_lengths = []
                for source, targets in path_lengths.items():
                    for target, length in targets.items():
                        if source != target:
                            all_lengths.append(length)

                path_metrics["all_lengths"] = all_lengths
                path_metrics["path_lengths"] = path_lengths

                # Diameter, radius, and center
                eccentricities = nx.eccentricity(working_graph)
                path_metrics["eccentricities"] = eccentricities
                path_metrics["diameter"] = (
                    max(eccentricities.values()) if eccentricities else 0
                )
                path_metrics["radius"] = (
                    min(eccentricities.values()) if eccentricities else 0
                )
                path_metrics["center_nodes"] = [
                    node
                    for node, ecc in eccentricities.items()
                    if ecc == path_metrics["radius"]
                ]

            else:
                # Sample-based analysis for large graphs
                sample_nodes = list(working_graph.nodes)[:100]
                path_lengths = {}
                all_lengths = []

                for node in sample_nodes:
                    lengths = nx.single_source_shortest_path_length(working_graph, node)
                    path_lengths[node] = lengths
                    for target, length in lengths.items():
                        if node != target:
                            all_lengths.append(length)

                path_metrics["all_lengths"] = all_lengths
                path_metrics["path_lengths"] = path_lengths
                path_metrics["diameter"] = max(all_lengths) if all_lengths else 0
                path_metrics["radius"] = 0  # Not calculated for large graphs
                path_metrics["center_nodes"] = []
                path_metrics["eccentricities"] = {}

            # Calculate derived metrics
            if all_lengths:
                path_metrics["avg_path_length"] = np.mean(all_lengths)
                path_metrics["max_path_length"] = max(all_lengths)
                path_metrics["path_length_std"] = np.std(all_lengths)
                path_metrics["efficiency"] = np.mean(
                    [1.0 / length for length in all_lengths]
                )

                # Path length distribution
                length_distribution = Counter(all_lengths)
                path_metrics["length_distribution"] = length_distribution
                path_metrics["most_common_length"] = length_distribution.most_common(1)[
                    0
                ][0]
            else:
                path_metrics.update(
                    {
                        "avg_path_length": 0,
                        "max_path_length": 0,
                        "path_length_std": 0,
                        "efficiency": 0,
                        "length_distribution": {},
                        "most_common_length": 0,
                    }
                )

        except Exception as e:
            self.logger.warning(f"Error calculating path metrics: {e}")
            # Return minimal metrics
            path_metrics = {
                "all_lengths": [],
                "avg_path_length": 0,
                "diameter": 0,
                "radius": 0,
                "efficiency": 0,
                "path_length_std": 0,
                "center_nodes": [],
                "eccentricities": {},
                "length_distribution": {},
                "most_common_length": 0,
            }

        return path_metrics

    def _create_node_metrics(
        self, graph: nx.Graph, working_graph: nx.Graph, path_metrics: Dict[str, Any]
    ) -> List[NodeMetrics]:
        """Create node-level path metrics."""

        node_metrics = []
        eccentricities = path_metrics.get("eccentricities", {})
        center_nodes = path_metrics.get("center_nodes", [])

        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Calculate closeness centrality as path-based measure
            try:
                closeness = (
                    nx.closeness_centrality(working_graph, node)
                    if node in working_graph
                    else 0
                )
            except Exception:
                closeness = 0

            # Eccentricity
            eccentricity = eccentricities.get(node, 0)

            # Additional path-based metrics
            try:
                # Average shortest path length from this node
                if node in path_metrics.get("path_lengths", {}):
                    node_path_lengths = list(
                        path_metrics["path_lengths"][node].values()
                    )
                    # Exclude self (distance 0)
                    node_path_lengths = [
                        length for length in node_path_lengths if length > 0
                    ]
                    avg_distance_from_node = (
                        np.mean(node_path_lengths) if node_path_lengths else 0
                    )
                    max_distance_from_node = (
                        max(node_path_lengths) if node_path_lengths else 0
                    )
                else:
                    avg_distance_from_node = 0
                    max_distance_from_node = 0
            except Exception:
                avg_distance_from_node = 0
                max_distance_from_node = 0

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    "closeness": closeness,
                    "eccentricity": eccentricity,
                    "is_center": node in center_nodes,
                    "avg_distance_from_node": avg_distance_from_node,
                    "max_distance_from_node": max_distance_from_node,
                },
                community_id=None,
                clustering_coefficient=0.0,  # Not relevant for path analysis
                degree=graph.degree(node),
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _compile_graph_metrics(
        self, path_metrics: Dict[str, Any], num_nodes: int
    ) -> Dict[str, float]:
        """Compile graph-level path metrics."""

        graph_metrics = {
            "avg_path_length": path_metrics.get("avg_path_length", 0),
            "diameter": path_metrics.get("diameter", 0),
            "radius": path_metrics.get("radius", 0),
            "efficiency": path_metrics.get("efficiency", 0),
            "path_length_std": path_metrics.get("path_length_std", 0),
            "num_center_nodes": len(path_metrics.get("center_nodes", [])),
            "connected_component_size": num_nodes,
            "path_length_range": (
                path_metrics.get("max_path_length", 0) - 1
                if path_metrics.get("max_path_length", 0) > 0
                else 0
            ),
        }

        # Add path length distribution metrics
        length_distribution = path_metrics.get("length_distribution", {})
        if length_distribution:
            graph_metrics.update(
                {
                    "most_common_path_length": path_metrics.get(
                        "most_common_length", 0
                    ),
                    "path_length_diversity": len(length_distribution),
                    "path_length_skewness": self._calculate_skewness(
                        path_metrics.get("all_lengths", [])
                    ),
                }
            )

        # Network compactness (inverse of average path length)
        if graph_metrics["avg_path_length"] > 0:
            graph_metrics["compactness"] = 1.0 / graph_metrics["avg_path_length"]
        else:
            graph_metrics["compactness"] = 0

        return graph_metrics

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0

        try:
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0

            skewness = np.mean([((x - mean) / std) ** 3 for x in values])
            return skewness
        except Exception:
            return 0

    def _generate_insights(
        self, graph_metrics: Dict[str, float], path_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from path analysis."""

        insights = [
            f"Average shortest path length: {graph_metrics['avg_path_length']:.2f}",
            f"Network diameter: {graph_metrics['diameter']}",
            f"Network radius: {graph_metrics['radius']}",
            f"Network efficiency: {graph_metrics['efficiency']:.4f}",
            f"Path length standard deviation: {graph_metrics['path_length_std']:.2f}",
        ]

        # Center nodes analysis
        center_nodes = path_metrics.get("center_nodes", [])
        if center_nodes:
            insights.append(f"Network has {len(center_nodes)} center node(s)")
            if len(center_nodes) <= 3:
                # Note: We would need access to the original graph to get titles
                insights.append(
                    f"Center nodes are strategically positioned for optimal reach"
                )

        # Path length distribution analysis
        if path_metrics.get("most_common_length"):
            insights.append(
                f"Most common path length: {path_metrics['most_common_length']}"
            )

        # Network compactness analysis
        compactness = graph_metrics.get("compactness", 0)
        if compactness > 0.5:
            insights.append("Network is highly compact with short average distances")
        elif compactness > 0.25:
            insights.append("Network shows moderate compactness")
        else:
            insights.append("Network is dispersed with longer average distances")

        # Efficiency analysis
        efficiency = graph_metrics.get("efficiency", 0)
        if efficiency > 0.3:
            insights.append("High network efficiency indicates good connectivity")
        elif efficiency > 0.15:
            insights.append("Moderate network efficiency")
        else:
            insights.append("Low network efficiency suggests connectivity challenges")

        # Diameter vs radius analysis
        diameter = graph_metrics.get("diameter", 0)
        radius = graph_metrics.get("radius", 0)
        if diameter > 0 and radius > 0:
            diameter_ratio = diameter / radius
            if diameter_ratio > 1.8:
                insights.append(
                    "Large diameter-to-radius ratio suggests elongated network structure"
                )
            elif diameter_ratio < 1.2:
                insights.append(
                    "Small diameter-to-radius ratio suggests compact, centralized structure"
                )

        return insights

    def _generate_recommendations(self, graph_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations from path analysis."""

        recommendations = [
            "Monitor diameter changes to track network expansion",
            "Use center nodes for optimal information placement",
            "Consider efficiency metrics for network optimization",
            "Short average path length indicates good connectivity",
        ]

        # Specific recommendations based on metrics
        avg_path_length = graph_metrics.get("avg_path_length", 0)
        if avg_path_length > 4:
            recommendations.append("Consider adding shortcuts to reduce path lengths")
        elif avg_path_length < 2:
            recommendations.append(
                "Very short paths suggest high density - monitor for overcrowding"
            )

        efficiency = graph_metrics.get("efficiency", 0)
        if efficiency < 0.2:
            recommendations.append(
                "Low efficiency suggests need for better connectivity"
            )

        num_center_nodes = graph_metrics.get("num_center_nodes", 0)
        if num_center_nodes == 1:
            recommendations.append(
                "Single center node creates vulnerability - consider backup paths"
            )
        elif num_center_nodes > 5:
            recommendations.append(
                "Multiple center nodes provide redundancy and robustness"
            )

        diameter = graph_metrics.get("diameter", 0)
        if diameter > 8:
            recommendations.append("Large diameter may slow information propagation")

        return recommendations

    def _create_empty_result(self) -> NetworkAnalysisResult:
        """Create empty result for cases with insufficient data."""

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.PATH_ANALYSIS,
            graph_metrics={},
            node_metrics=[],
            communities=[],
            insights=["Insufficient data for path analysis"],
            recommendations=[],
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )


# Factory function for easy integration
def create_path_analyzer(config=None) -> PathAnalyzer:
    """Create and return a PathAnalyzer instance."""
    return PathAnalyzer(config)


# Async wrapper for compatibility
async def analyze_paths(graph: nx.Graph, config=None) -> NetworkAnalysisResult:
    """Analyze path structures in a network graph."""
    analyzer = create_path_analyzer(config)
    return await analyzer.analyze_paths(graph)
