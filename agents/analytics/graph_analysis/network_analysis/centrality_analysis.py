"""
Centrality Analysis Module - Focused centrality calculations and analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ..config import NetworkConfig
from ..graph_utils import CentralityCalculator, GraphMetricsAggregator
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class CentralityAnalyzer:
    """Specialized analyzer for centrality measures."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.centrality_calc = CentralityCalculator(config)
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("centrality_analyzer")

    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Perform comprehensive centrality analysis."""

        self.logger.info("Computing centrality measures...")

        # Calculate all centrality measures
        centrality_measures = self.centrality_calc.calculate_all_centralities(graph)

        # Create node metrics
        node_metrics = self._create_node_metrics(graph, centrality_measures)

        # Calculate graph-level metrics
        graph_metrics = self.metrics_aggregator.calculate_comprehensive_metrics(
            graph, centrality_measures
        )

        # Generate insights
        insights = self._generate_centrality_insights(
            graph, centrality_measures, node_metrics
        )

        # Generate recommendations
        recommendations = self._generate_centrality_recommendations(
            centrality_measures, node_metrics
        )

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.CENTRALITY,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )

    def _create_node_metrics(
        self, graph: nx.Graph, centrality_measures: Dict[str, Dict[str, float]]
    ) -> List[NodeMetrics]:
        """Create node metrics from centrality calculations."""

        node_metrics = []

        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Compile centrality scores for this node
            centrality_scores = {}
            for measure_name, scores in centrality_measures.items():
                centrality_scores[measure_name] = scores.get(node, 0.0)

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,  # Not relevant for centrality analysis
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _generate_centrality_insights(
        self,
        graph: nx.Graph,
        centrality_measures: Dict[str, Dict[str, float]],
        node_metrics: List[NodeMetrics],
    ) -> List[str]:
        """Generate insights from centrality analysis."""

        insights = []

        # General network insights
        insights.append(
            f"Network has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

        # Centrality insights for each measure
        for measure_name, scores in centrality_measures.items():
            if scores:
                top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"Top {measure_name} nodes:")
                for node, score in top_nodes:
                    node_title = graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: {score:.4f}")

        # Centrality distribution insights
        for measure_name, scores in centrality_measures.items():
            if scores:
                values = list(scores.values())
                if len(values) > 1:
                    concentration = (
                        np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    )
                    if concentration > 1.0:
                        insights.append(
                            f"High {measure_name} concentration suggests hierarchical structure"
                        )
                    elif concentration < 0.5:
                        insights.append(
                            f"Low {measure_name} concentration suggests egalitarian structure"
                        )

        # Centrality correlation insights
        if len(centrality_measures) >= 2:
            measure_names = list(centrality_measures.keys())
            for i in range(len(measure_names)):
                for j in range(i + 1, len(measure_names)):
                    measure1, measure2 = measure_names[i], measure_names[j]
                    values1 = list(centrality_measures[measure1].values())
                    values2 = list(centrality_measures[measure2].values())

                    if len(values1) == len(values2) and len(values1) > 1:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(correlation):
                            if correlation > 0.7:
                                insights.append(
                                    f"Strong positive correlation between {measure1} and {measure2} ({correlation:.3f})"
                                )
                            elif correlation < -0.7:
                                insights.append(
                                    f"Strong negative correlation between {measure1} and {measure2} ({correlation:.3f})"
                                )

        # Identify key node roles
        if "pagerank" in centrality_measures and "betweenness" in centrality_measures:
            # Find nodes high in both PageRank and betweenness (influential bridges)
            pr_scores = centrality_measures["pagerank"]
            bt_scores = centrality_measures["betweenness"]

            influential_bridges = []
            for node in graph.nodes:
                pr_score = pr_scores.get(node, 0)
                bt_score = bt_scores.get(node, 0)
                if pr_score > np.percentile(
                    list(pr_scores.values()), 80
                ) and bt_score > np.percentile(list(bt_scores.values()), 80):
                    influential_bridges.append(node)

            if influential_bridges:
                insights.append(
                    f"Identified {len(influential_bridges)} influential bridge nodes"
                )
                if len(influential_bridges) <= 3:
                    for node in influential_bridges:
                        node_title = graph.nodes[node].get("title", f"Node {node}")
                        insights.append(f"  • {node_title}")

        return insights

    def _generate_centrality_recommendations(
        self,
        centrality_measures: Dict[str, Dict[str, float]],
        node_metrics: List[NodeMetrics],
    ) -> List[str]:
        """Generate recommendations from centrality analysis."""

        recommendations = [
            "Monitor high PageRank nodes for influence and authority",
            "Use high betweenness nodes for information flow control",
            "Target high closeness nodes for rapid information dissemination",
            "Consider eigenvector centrality for identifying prestige networks",
        ]

        # Add specific recommendations based on centrality distribution
        if "pagerank" in centrality_measures:
            pagerank_values = list(centrality_measures["pagerank"].values())
            if pagerank_values:
                pagerank_concentration = (
                    np.std(pagerank_values) / np.mean(pagerank_values)
                    if np.mean(pagerank_values) > 0
                    else 0
                )

                if pagerank_concentration > 1.0:
                    recommendations.append(
                        "High PageRank concentration suggests focusing on top-tier nodes for maximum impact"
                    )
                else:
                    recommendations.append(
                        "Distributed PageRank suggests multi-node strategies for broad influence"
                    )

        if "betweenness" in centrality_measures:
            betweenness_values = list(centrality_measures["betweenness"].values())
            if betweenness_values:
                max_betweenness = max(betweenness_values)
                if max_betweenness > 0.1:  # High betweenness
                    recommendations.append(
                        "High betweenness centrality nodes are critical for network connectivity"
                    )
                else:
                    recommendations.append(
                        "Low betweenness values suggest robust, redundant connectivity"
                    )

        if "degree" in centrality_measures:
            degree_values = list(centrality_measures["degree"].values())
            if degree_values:
                avg_degree = np.mean(degree_values)
                if avg_degree < 0.1:  # Low connectivity
                    recommendations.append(
                        "Low average degree suggests opportunities for strategic link formation"
                    )
                elif avg_degree > 0.5:  # High connectivity
                    recommendations.append(
                        "High connectivity enables rapid information diffusion"
                    )

        return recommendations

    def calculate_centrality_rankings(
        self, centrality_measures: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[tuple]]:
        """Calculate rankings for each centrality measure."""

        rankings = {}

        for measure_name, scores in centrality_measures.items():
            if scores:
                # Sort nodes by centrality score in descending order
                sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                rankings[measure_name] = sorted_nodes

        return rankings

    def identify_central_nodes(
        self, centrality_measures: Dict[str, Dict[str, float]], top_k: int = 10
    ) -> Dict[str, List[str]]:
        """Identify top-k central nodes for each measure."""

        central_nodes = {}

        for measure_name, scores in centrality_measures.items():
            if scores:
                # Get top-k nodes for this measure
                top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                    :top_k
                ]
                central_nodes[measure_name] = [node for node, score in top_nodes]

        return central_nodes

    def calculate_centrality_statistics(
        self, centrality_measures: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary for each centrality measure."""

        statistics = {}

        for measure_name, scores in centrality_measures.items():
            if scores:
                values = list(scores.values())
                stats = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "range": np.max(values) - np.min(values),
                    "concentration": (
                        np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    ),
                }
                statistics[measure_name] = stats

        return statistics
