"""
Clustering Analysis Module - Focused clustering patterns and triangular structures analysis.
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ..config import NetworkConfig
from ..graph_utils import ClusteringAnalyzer as BaseClusteringAnalyzer
from ..graph_utils import (
    GraphMetricsAggregator,
)
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class ClusteringAnalyzer:
    """Specialized analyzer for clustering patterns in networks."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.base_clustering_analyzer = BaseClusteringAnalyzer()
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("clustering_analyzer")

    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze clustering patterns in the network."""

        self.logger.info("Analyzing clustering patterns...")

        # Calculate comprehensive clustering metrics
        clustering_metrics = self.base_clustering_analyzer.calculate_clustering_metrics(
            graph
        )

        # Analyze clustering distribution
        clustering_distribution = self._analyze_clustering_distribution(
            graph, clustering_metrics
        )

        # Analyze triangular structures
        triangle_analysis = self._analyze_triangular_structures(
            graph, clustering_metrics
        )

        # Identify clustering patterns
        clustering_patterns = self._identify_clustering_patterns(
            graph, clustering_metrics
        )

        # Create node metrics
        node_metrics = self._create_clustering_node_metrics(
            graph, clustering_metrics, clustering_distribution, clustering_patterns
        )

        # Calculate graph metrics
        graph_metrics = self._calculate_clustering_graph_metrics(
            graph, clustering_metrics, clustering_distribution, triangle_analysis
        )

        # Generate insights
        insights = self._generate_clustering_insights(
            graph, clustering_metrics, clustering_distribution, triangle_analysis
        )

        # Generate recommendations
        recommendations = self._generate_clustering_recommendations(
            clustering_metrics, clustering_distribution, clustering_patterns
        )

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.CLUSTERING_ANALYSIS,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )

    def _analyze_clustering_distribution(
        self, graph: nx.Graph, clustering_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Analyze the distribution of clustering coefficients."""

        clustering_dict = clustering_metrics["node_clustering"]
        clustering_values = list(clustering_dict.values())

        if not clustering_values:
            return {}

        # Statistical analysis
        distribution_stats = {
            "mean": np.mean(clustering_values),
            "median": np.median(clustering_values),
            "std": np.std(clustering_values),
            "min": np.min(clustering_values),
            "max": np.max(clustering_values),
            "range": np.max(clustering_values) - np.min(clustering_values),
            "q25": np.percentile(clustering_values, 25),
            "q75": np.percentile(clustering_values, 75),
            "iqr": np.percentile(clustering_values, 75)
            - np.percentile(clustering_values, 25),
        }

        # Clustering categories
        high_clustering_threshold = np.percentile(clustering_values, 90)
        medium_clustering_threshold = np.percentile(clustering_values, 50)

        high_clustering_nodes = [
            node
            for node, coeff in clustering_dict.items()
            if coeff > high_clustering_threshold and coeff > 0.5
        ]
        medium_clustering_nodes = [
            node
            for node, coeff in clustering_dict.items()
            if medium_clustering_threshold <= coeff <= high_clustering_threshold
        ]
        low_clustering_nodes = [
            node
            for node, coeff in clustering_dict.items()
            if coeff < medium_clustering_threshold
        ]

        # Zero clustering analysis
        zero_clustering_nodes = [
            node for node, coeff in clustering_dict.items() if coeff == 0
        ]

        return {
            "statistics": distribution_stats,
            "high_clustering_nodes": high_clustering_nodes,
            "medium_clustering_nodes": medium_clustering_nodes,
            "low_clustering_nodes": low_clustering_nodes,
            "zero_clustering_nodes": zero_clustering_nodes,
            "high_threshold": high_clustering_threshold,
            "medium_threshold": medium_clustering_threshold,
            "clustering_heterogeneity": (
                distribution_stats["std"] / distribution_stats["mean"]
                if distribution_stats["mean"] > 0
                else 0
            ),
        }

    def _analyze_triangular_structures(
        self, graph: nx.Graph, clustering_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Analyze triangular structures and their properties."""

        triangles_dict = clustering_metrics["triangles"]

        # Total triangles in network (each triangle counted once)
        total_triangles = sum(triangles_dict.values()) // 3

        # Triangle distribution analysis
        triangle_counts = list(triangles_dict.values())

        analysis = {
            "total_triangles": total_triangles,
            "avg_triangles_per_node": (
                np.mean(triangle_counts) if triangle_counts else 0
            ),
            "max_triangles_per_node": max(triangle_counts) if triangle_counts else 0,
            "triangle_distribution": Counter(triangle_counts),
        }

        # Identify nodes with high triangle participation
        if triangle_counts:
            triangle_threshold = np.percentile(triangle_counts, 90)
            high_triangle_nodes = [
                node
                for node, count in triangles_dict.items()
                if count > triangle_threshold and count > 0
            ]
            analysis["high_triangle_nodes"] = high_triangle_nodes
            analysis["triangle_threshold"] = triangle_threshold
        else:
            analysis["high_triangle_nodes"] = []
            analysis["triangle_threshold"] = 0

        # Triangle density (ratio of triangles to possible triangles)
        num_nodes = graph.number_of_nodes()
        if num_nodes >= 3:
            max_possible_triangles = num_nodes * (num_nodes - 1) * (num_nodes - 2) // 6
            analysis["triangle_density"] = (
                total_triangles / max_possible_triangles
                if max_possible_triangles > 0
                else 0
            )
        else:
            analysis["triangle_density"] = 0

        return analysis

    def _identify_clustering_patterns(
        self, graph: nx.Graph, clustering_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Identify specific clustering patterns in the network."""

        patterns = {}
        clustering_dict = clustering_metrics["node_clustering"]

        # Find cliques (complete subgraphs)
        try:
            # Find cliques of size 3 and larger
            cliques = list(nx.find_cliques(graph))

            # Categorize cliques by size
            clique_sizes = [len(clique) for clique in cliques]
            clique_size_distribution = Counter(clique_sizes)

            # Find largest cliques
            if cliques:
                max_clique_size = max(clique_sizes)
                largest_cliques = [
                    clique for clique in cliques if len(clique) == max_clique_size
                ]
            else:
                max_clique_size = 0
                largest_cliques = []

            patterns.update(
                {
                    "cliques": cliques,
                    "clique_count": len(cliques),
                    "clique_size_distribution": dict(clique_size_distribution),
                    "max_clique_size": max_clique_size,
                    "largest_cliques": largest_cliques[:5],  # Limit to first 5
                }
            )

        except Exception as e:
            self.logger.warning(f"Clique analysis failed: {e}")
            patterns.update(
                {
                    "cliques": [],
                    "clique_count": 0,
                    "clique_size_distribution": {},
                    "max_clique_size": 0,
                    "largest_cliques": [],
                }
            )

        # Find k-cores with high clustering
        try:
            k_core = nx.core_number(graph)

            # Identify nodes in high k-cores with high clustering
            high_k_core_clustered = []
            for node in graph.nodes:
                core_num = k_core.get(node, 0)
                clustering_coeff = clustering_dict.get(node, 0)

                if core_num >= 3 and clustering_coeff > 0.5:
                    high_k_core_clustered.append(node)

            patterns["high_k_core_clustered"] = high_k_core_clustered

        except Exception as e:
            self.logger.warning(f"K-core clustering analysis failed: {e}")
            patterns["high_k_core_clustered"] = []

        # Identify clustering hubs (high degree + high clustering)
        clustering_hubs = []
        for node in graph.nodes:
            degree = graph.degree(node)
            clustering_coeff = clustering_dict.get(node, 0)

            # High degree (top 20%) and high clustering (>0.5)
            if (
                degree > np.percentile([graph.degree(n) for n in graph.nodes], 80)
                and clustering_coeff > 0.5
            ):
                clustering_hubs.append(node)

        patterns["clustering_hubs"] = clustering_hubs

        return patterns

    def _create_clustering_node_metrics(
        self,
        graph: nx.Graph,
        clustering_metrics: Dict[str, any],
        clustering_distribution: Dict[str, any],
        clustering_patterns: Dict[str, any],
    ) -> List[NodeMetrics]:
        """Create node metrics for clustering analysis."""

        node_metrics = []
        clustering_dict = clustering_metrics["node_clustering"]
        triangles_dict = clustering_metrics["triangles"]
        square_clustering_dict = clustering_metrics["square_clustering"]

        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Compile clustering-related scores
            centrality_scores = {
                "clustering": clustering_dict.get(node, 0),
                "triangles": triangles_dict.get(node, 0),
                "square_clustering": square_clustering_dict.get(node, 0),
                "is_high_clustering": node
                in clustering_distribution.get("high_clustering_nodes", []),
                "is_clustering_hub": node
                in clustering_patterns.get("clustering_hubs", []),
                "is_in_large_clique": False,  # Will be set below
                "max_clique_participation": 0,  # Will be calculated below
            }

            # Check clique participation
            cliques = clustering_patterns.get("cliques", [])
            max_clique_size = 0
            for clique in cliques:
                if node in clique:
                    max_clique_size = max(max_clique_size, len(clique))
                    if len(clique) >= 4:  # Large clique threshold
                        centrality_scores["is_in_large_clique"] = True

            centrality_scores["max_clique_participation"] = max_clique_size

            # Calculate clustering importance score
            clustering_coeff = clustering_dict.get(node, 0)
            triangle_count = triangles_dict.get(node, 0)
            degree = graph.degree(node)

            clustering_importance = (
                clustering_coeff * 0.4
                + (triangle_count / 10) * 0.3
                + (max_clique_size / 10) * 0.3
            )
            centrality_scores["clustering_importance"] = clustering_importance

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,
                clustering_coefficient=clustering_coeff,
                degree=degree,
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _calculate_clustering_graph_metrics(
        self,
        graph: nx.Graph,
        clustering_metrics: Dict[str, any],
        clustering_distribution: Dict[str, any],
        triangle_analysis: Dict[str, any],
    ) -> Dict[str, any]:
        """Calculate graph-level metrics for clustering analysis."""

        metrics = {}

        # Basic clustering metrics
        metrics.update(
            {
                "avg_clustering": clustering_metrics["average_clustering"],
                "transitivity": clustering_metrics["transitivity"],
                "total_triangles": triangle_analysis["total_triangles"],
                "triangle_density": triangle_analysis["triangle_density"],
            }
        )

        # Distribution metrics
        if "statistics" in clustering_distribution:
            stats = clustering_distribution["statistics"]
            metrics.update(
                {
                    "clustering_std": stats["std"],
                    "clustering_range": stats["range"],
                    "clustering_heterogeneity": clustering_distribution[
                        "clustering_heterogeneity"
                    ],
                    "clustering_median": stats["median"],
                    "clustering_iqr": stats["iqr"],
                }
            )

        # Categorical metrics
        metrics.update(
            {
                "num_high_clustering": len(
                    clustering_distribution.get("high_clustering_nodes", [])
                ),
                "num_zero_clustering": len(
                    clustering_distribution.get("zero_clustering_nodes", [])
                ),
                "high_clustering_ratio": (
                    len(clustering_distribution.get("high_clustering_nodes", []))
                    / len(graph.nodes)
                    if graph.nodes
                    else 0
                ),
            }
        )

        # Triangle metrics
        metrics.update(
            {
                "avg_triangles_per_node": triangle_analysis["avg_triangles_per_node"],
                "max_triangles_per_node": triangle_analysis["max_triangles_per_node"],
            }
        )

        # Compare to random graph
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            if num_edges > 0:
                # Expected clustering in random graph
                density = nx.density(graph)
                random_clustering = density  # Approximation for random graph

                clustering_enhancement = (
                    metrics["avg_clustering"] / random_clustering
                    if random_clustering > 0
                    else 0
                )
                metrics["clustering_enhancement"] = clustering_enhancement
            else:
                metrics["clustering_enhancement"] = 0

        except Exception:
            metrics["clustering_enhancement"] = 0

        # Add basic graph metrics
        metrics.update(self.metrics_aggregator.calculate_comprehensive_metrics(graph))

        return metrics

    def _generate_clustering_insights(
        self,
        graph: nx.Graph,
        clustering_metrics: Dict[str, any],
        clustering_distribution: Dict[str, any],
        triangle_analysis: Dict[str, any],
    ) -> List[str]:
        """Generate insights from clustering analysis."""

        insights = []

        # Basic clustering insights
        avg_clustering = clustering_metrics["average_clustering"]
        transitivity = clustering_metrics["transitivity"]
        total_triangles = triangle_analysis["total_triangles"]

        insights.append(f"Average clustering coefficient: {avg_clustering:.4f}")
        insights.append(f"Network transitivity: {transitivity:.4f}")
        insights.append(f"Total triangles in network: {total_triangles}")

        # Clustering distribution insights
        if "statistics" in clustering_distribution:
            heterogeneity = clustering_distribution["clustering_heterogeneity"]
            insights.append(f"Clustering heterogeneity: {heterogeneity:.4f}")

            if heterogeneity > 1.0:
                insights.append(
                    "High clustering heterogeneity indicates diverse local structures"
                )
            elif heterogeneity < 0.3:
                insights.append(
                    "Low clustering heterogeneity suggests uniform local connectivity"
                )

        # Compare to random graph
        try:
            density = nx.density(graph)
            if density > 0:
                random_clustering = density
                enhancement = (
                    avg_clustering / random_clustering if random_clustering > 0 else 0
                )
                insights.append(
                    f"Clustering enhancement over random: {enhancement:.2f}x"
                )

                if enhancement > 3:
                    insights.append(
                        "Strong clustering enhancement indicates structured local organization"
                    )
        except Exception:
            pass

        # High clustering nodes
        high_clustering_nodes = clustering_distribution.get("high_clustering_nodes", [])
        if high_clustering_nodes:
            insights.append(
                f"Identified {len(high_clustering_nodes)} highly clustered nodes"
            )

            # Show top clustered nodes
            clustering_dict = clustering_metrics["node_clustering"]
            top_clustered = sorted(
                [(node, clustering_dict[node]) for node in high_clustering_nodes],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            if top_clustered:
                insights.append("Most clustered nodes:")
                for node, coeff in top_clustered:
                    node_title = graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: clustering = {coeff:.4f}")

        # Triangle analysis insights
        triangle_density = triangle_analysis["triangle_density"]
        if triangle_density > 0.01:
            insights.append(
                f"Triangle density: {triangle_density:.4f} - rich triangular structure"
            )
        elif triangle_density < 0.001:
            insights.append("Low triangle density - sparse triangular structures")

        # Zero clustering analysis
        zero_clustering_nodes = clustering_distribution.get("zero_clustering_nodes", [])
        if zero_clustering_nodes:
            zero_ratio = (
                len(zero_clustering_nodes) / len(graph.nodes) if graph.nodes else 0
            )
            insights.append(
                f"{len(zero_clustering_nodes)} nodes ({zero_ratio:.1%}) have zero clustering"
            )

            if zero_ratio > 0.3:
                insights.append(
                    "High proportion of zero-clustering nodes suggests tree-like or star structures"
                )

        return insights

    def _generate_clustering_recommendations(
        self,
        clustering_metrics: Dict[str, any],
        clustering_distribution: Dict[str, any],
        clustering_patterns: Dict[str, any],
    ) -> List[str]:
        """Generate recommendations from clustering analysis."""

        recommendations = [
            "High clustering indicates strong local cohesion",
            "Monitor clustering changes to detect community formation",
            "Use triangular structures for robust information propagation",
        ]

        # Clustering level recommendations
        avg_clustering = clustering_metrics["average_clustering"]

        if avg_clustering > 0.5:
            recommendations.append(
                "High average clustering enables localized strategies"
            )
        elif avg_clustering < 0.1:
            recommendations.append(
                "Low clustering suggests focusing on global rather than local approaches"
            )

        # Heterogeneity recommendations
        heterogeneity = clustering_distribution.get("clustering_heterogeneity", 0)

        if heterogeneity > 1.0:
            recommendations.append(
                "High clustering heterogeneity requires differentiated local strategies"
            )
        elif heterogeneity < 0.3:
            recommendations.append(
                "Uniform clustering enables standardized local interventions"
            )

        # High clustering nodes recommendations
        high_clustering_nodes = clustering_distribution.get("high_clustering_nodes", [])
        if high_clustering_nodes:
            recommendations.append(
                "Consider highly clustered nodes for local influence strategies"
            )
            recommendations.append(
                "Leverage clustering hubs for community-based interventions"
            )

        # Triangle-based recommendations
        total_triangles = clustering_metrics.get("triangles", {})
        if total_triangles:
            avg_triangles = np.mean(list(total_triangles.values()))
            if avg_triangles > 5:
                recommendations.append(
                    "Rich triangular structure supports trust-based mechanisms"
                )

        # Zero clustering recommendations
        zero_clustering_nodes = clustering_distribution.get("zero_clustering_nodes", [])
        if len(zero_clustering_nodes) > len(
            clustering_distribution.get("high_clustering_nodes", [])
        ):
            recommendations.append(
                "Many nodes lack local clustering - consider strategic link formation"
            )

        return recommendations

    def identify_clustering_leaders(
        self,
        clustering_metrics: Dict[str, any],
        clustering_patterns: Dict[str, any],
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        """Identify top nodes for different clustering roles."""

        leaders = {}

        # Top clustered nodes
        clustering_dict = clustering_metrics["node_clustering"]
        top_clustered = sorted(
            clustering_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        leaders["most_clustered"] = [node for node, coeff in top_clustered if coeff > 0]

        # Clustering hubs (high degree + high clustering)
        clustering_hubs = clustering_patterns.get("clustering_hubs", [])
        leaders["clustering_hubs"] = clustering_hubs[:top_k]

        # Triangle leaders (most triangles)
        triangles_dict = clustering_metrics["triangles"]
        top_triangle_nodes = sorted(
            triangles_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        leaders["triangle_leaders"] = [
            node for node, count in top_triangle_nodes if count > 0
        ]

        return leaders
