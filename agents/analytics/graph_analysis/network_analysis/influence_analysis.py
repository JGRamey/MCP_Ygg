"""
Influence Analysis Module - Focused influence propagation and reach analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

from ..config import NetworkConfig
from ..graph_utils import GraphMetricsAggregator
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class InfluenceAnalyzer:
    """Specialized analyzer for influence propagation patterns."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("influence_analyzer")

    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze influence propagation patterns."""

        self.logger.info("Analyzing influence propagation...")

        # Calculate influence-related metrics
        influence_metrics = self._calculate_influence_metrics(graph)

        # Calculate reach metrics
        reach_metrics = self._calculate_reach_metrics(graph)

        # Calculate k-core metrics
        k_core_metrics = self._calculate_k_core_metrics(graph)

        # Create node metrics
        node_metrics = self._create_influence_node_metrics(
            graph, influence_metrics, reach_metrics, k_core_metrics
        )

        # Calculate graph metrics
        graph_metrics = self._calculate_influence_graph_metrics(
            graph, influence_metrics, reach_metrics, k_core_metrics
        )

        # Generate insights
        insights = self._generate_influence_insights(
            graph, influence_metrics, reach_metrics, k_core_metrics
        )

        # Generate recommendations
        recommendations = self._generate_influence_recommendations(
            influence_metrics, reach_metrics, k_core_metrics
        )

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.INFLUENCE_PROPAGATION,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )

    def _calculate_influence_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate various influence-related metrics."""

        metrics = {}

        # K-core decomposition to find influential cores
        try:
            k_core = nx.core_number(graph)
            metrics["k_core"] = k_core
        except Exception as e:
            self.logger.warning(f"K-core analysis failed: {e}")
            metrics["k_core"] = {node: 0 for node in graph.nodes}

        # K-shell analysis
        try:
            k_shell = nx.k_shell(graph)
            k_shell_dict = {node: 0 for node in graph.nodes}
            for node in k_shell.nodes:
                k_shell_dict[node] = 1
            metrics["k_shell"] = k_shell_dict
        except Exception as e:
            self.logger.warning(f"K-shell analysis failed: {e}")
            metrics["k_shell"] = {node: 0 for node in graph.nodes}

        # Degree centrality (local influence)
        metrics["degree_centrality"] = nx.degree_centrality(graph)

        return metrics

    def _calculate_reach_metrics(self, graph: nx.Graph) -> Dict[str, Dict[str, int]]:
        """Calculate reach metrics for nodes."""

        reach_metrics = {"reach_1hop": {}, "reach_2hop": {}, "reach_3hop": {}}

        # Sample nodes for performance (analyze top-degree nodes and random sample)
        degree_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, degree in degree_sorted[:50]]

        # Add random sample if graph is large
        if len(graph.nodes) > 100:
            import random

            remaining_nodes = list(set(graph.nodes) - set(top_nodes))
            sample_size = min(50, len(remaining_nodes))
            random_nodes = random.sample(remaining_nodes, sample_size)
            nodes_to_analyze = top_nodes + random_nodes
        else:
            nodes_to_analyze = list(graph.nodes)

        for node in nodes_to_analyze:
            # Calculate reach within different hop distances
            for hops in range(1, 4):
                try:
                    reachable = set([node])
                    current_level = {node}

                    for _ in range(hops):
                        next_level = set()
                        for n in current_level:
                            next_level.update(graph.neighbors(n))
                        next_level -= reachable
                        reachable.update(next_level)
                        current_level = next_level
                        if not current_level:
                            break

                    reach_metrics[f"reach_{hops}hop"][node] = (
                        len(reachable) - 1
                    )  # Exclude self
                except Exception:
                    reach_metrics[f"reach_{hops}hop"][node] = 0

        return reach_metrics

    def _calculate_k_core_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate k-core related metrics."""

        try:
            k_core = nx.core_number(graph)

            # Calculate k-core statistics
            k_values = list(k_core.values())

            metrics = {
                "k_core_numbers": k_core,
                "max_k_core": max(k_values) if k_values else 0,
                "avg_k_core": np.mean(k_values) if k_values else 0,
                "k_core_distribution": {k: k_values.count(k) for k in set(k_values)},
            }

            # Identify k-core shells
            k_shells = {}
            max_k = max(k_values) if k_values else 0

            for k in range(max_k + 1):
                k_shell_nodes = [
                    node for node, core_num in k_core.items() if core_num == k
                ]
                if k_shell_nodes:
                    k_shells[k] = k_shell_nodes

            metrics["k_shells"] = k_shells

            return metrics

        except Exception as e:
            self.logger.warning(f"K-core metrics calculation failed: {e}")
            return {
                "k_core_numbers": {node: 0 for node in graph.nodes},
                "max_k_core": 0,
                "avg_k_core": 0,
                "k_core_distribution": {0: len(graph.nodes)},
                "k_shells": {0: list(graph.nodes)},
            }

    def _create_influence_node_metrics(
        self,
        graph: nx.Graph,
        influence_metrics: Dict[str, any],
        reach_metrics: Dict[str, Dict[str, int]],
        k_core_metrics: Dict[str, any],
    ) -> List[NodeMetrics]:
        """Create node metrics for influence analysis."""

        node_metrics = []

        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Compile influence scores
            centrality_scores = {
                "k_core": influence_metrics["k_core"].get(node, 0),
                "k_shell": influence_metrics["k_shell"].get(node, 0),
                "degree_centrality": influence_metrics["degree_centrality"].get(
                    node, 0
                ),
            }

            # Add reach metrics if available
            for reach_type, reach_dict in reach_metrics.items():
                if node in reach_dict:
                    centrality_scores[reach_type] = reach_dict[node]
                else:
                    centrality_scores[reach_type] = 0

            # Calculate influence potential
            degree = graph.degree(node)
            k_core_value = influence_metrics["k_core"].get(node, 0)
            reach_2hop = reach_metrics["reach_2hop"].get(node, 0)

            # Simple influence potential formula
            influence_potential = degree * 0.3 + k_core_value * 0.4 + reach_2hop * 0.3
            centrality_scores["influence_potential"] = influence_potential

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,
                clustering_coefficient=nx.clustering(graph, node),
                degree=degree,
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _calculate_influence_graph_metrics(
        self,
        graph: nx.Graph,
        influence_metrics: Dict[str, any],
        reach_metrics: Dict[str, Dict[str, int]],
        k_core_metrics: Dict[str, any],
    ) -> Dict[str, any]:
        """Calculate graph-level metrics for influence analysis."""

        metrics = {}

        # Basic graph metrics
        metrics.update(self.metrics_aggregator.calculate_comprehensive_metrics(graph))

        # K-core metrics
        metrics.update(
            {
                "max_k_core": k_core_metrics["max_k_core"],
                "avg_k_core": k_core_metrics["avg_k_core"],
                "k_core_distribution": k_core_metrics["k_core_distribution"],
            }
        )

        # Reach metrics (averages)
        for reach_type, reach_dict in reach_metrics.items():
            if reach_dict:
                metrics[f"avg_{reach_type}"] = np.mean(list(reach_dict.values()))
                metrics[f"max_{reach_type}"] = max(reach_dict.values())
            else:
                metrics[f"avg_{reach_type}"] = 0
                metrics[f"max_{reach_type}"] = 0

        # Influence concentration
        k_core_values = list(influence_metrics["k_core"].values())
        if k_core_values:
            metrics["influence_concentration"] = (
                np.std(k_core_values) / np.mean(k_core_values)
                if np.mean(k_core_values) > 0
                else 0
            )

        # Influence hierarchy depth
        metrics["influence_hierarchy_depth"] = k_core_metrics["max_k_core"]

        return metrics

    def _generate_influence_insights(
        self,
        graph: nx.Graph,
        influence_metrics: Dict[str, any],
        reach_metrics: Dict[str, Dict[str, int]],
        k_core_metrics: Dict[str, any],
    ) -> List[str]:
        """Generate insights from influence analysis."""

        insights = []

        # K-core insights
        max_k_core = k_core_metrics["max_k_core"]
        avg_k_core = k_core_metrics["avg_k_core"]

        insights.append(f"Maximum k-core value: {max_k_core}")
        insights.append(f"Average k-core value: {avg_k_core:.2f}")

        if max_k_core > 5:
            insights.append(
                "High k-core values indicate presence of dense, influential subgroups"
            )
        elif max_k_core < 2:
            insights.append(
                "Low k-core values suggest sparse, tree-like structure with limited influence cores"
            )

        # Reach insights
        if "reach_2hop" in reach_metrics:
            reach_values = list(reach_metrics["reach_2hop"].values())
            if reach_values:
                avg_reach = np.mean(reach_values)
                max_reach = max(reach_values)
                insights.append(f"Average 2-hop reach: {avg_reach:.1f} nodes")
                insights.append(f"Maximum 2-hop reach: {max_reach} nodes")

                if max_reach > len(graph.nodes) * 0.3:
                    insights.append(
                        "Some nodes can reach >30% of network within 2 hops - high influence potential"
                    )

        # Identify most influential nodes
        k_core_numbers = influence_metrics["k_core"]
        top_k_core_nodes = sorted(
            k_core_numbers.items(), key=lambda x: x[1], reverse=True
        )[:5]

        if top_k_core_nodes and top_k_core_nodes[0][1] > 0:
            insights.append("Top influential nodes (by k-core):")
            for node, score in top_k_core_nodes:
                node_title = graph.nodes[node].get("title", f"Node {node}")
                insights.append(f"  • {node_title}: k-core = {score}")

        # K-core distribution analysis
        k_core_dist = k_core_metrics["k_core_distribution"]
        if len(k_core_dist) > 1:
            insights.append(f"K-core distribution spans {len(k_core_dist)} levels")

            # Find the most populated k-core level
            most_populated_k = max(k_core_dist.items(), key=lambda x: x[1])
            insights.append(
                f"Most nodes ({most_populated_k[1]}) are in k-core level {most_populated_k[0]}"
            )

        # Influence hierarchy insights
        if max_k_core > 3:
            insights.append(
                "Multi-level influence hierarchy detected - consider tier-based strategies"
            )

        return insights

    def _generate_influence_recommendations(
        self,
        influence_metrics: Dict[str, any],
        reach_metrics: Dict[str, Dict[str, int]],
        k_core_metrics: Dict[str, any],
    ) -> List[str]:
        """Generate recommendations from influence analysis."""

        recommendations = [
            "Target high k-core nodes for maximum influence propagation",
            "Monitor k-core changes to detect shifting influence patterns",
            "Use reach metrics to optimize information dissemination strategies",
        ]

        # K-core based recommendations
        max_k_core = k_core_metrics["max_k_core"]
        if max_k_core > 3:
            recommendations.append(
                "Multi-level k-core structure suggests layered influence strategies"
            )
        elif max_k_core < 2:
            recommendations.append(
                "Low k-core structure suggests focusing on high-degree nodes for influence"
            )

        # Reach based recommendations
        if "reach_2hop" in reach_metrics:
            reach_values = list(reach_metrics["reach_2hop"].values())
            if reach_values:
                max_reach = max(reach_values)
                avg_reach = np.mean(reach_values)

                if max_reach > avg_reach * 3:
                    recommendations.append(
                        "High reach variance suggests identifying and leveraging super-influencers"
                    )

                if avg_reach < 5:
                    recommendations.append(
                        "Low average reach suggests need for multiple seed nodes for broad influence"
                    )

        # Concentration based recommendations
        k_core_values = list(influence_metrics["k_core"].values())
        if k_core_values:
            concentration = (
                np.std(k_core_values) / np.mean(k_core_values)
                if np.mean(k_core_values) > 0
                else 0
            )

            if concentration > 1.0:
                recommendations.append(
                    "High influence concentration enables targeted interventions on key nodes"
                )
            else:
                recommendations.append(
                    "Distributed influence suggests broad-based engagement strategies"
                )

        return recommendations

    def identify_influence_leaders(
        self,
        influence_metrics: Dict[str, any],
        reach_metrics: Dict[str, Dict[str, int]],
        top_k: int = 10,
    ) -> List[str]:
        """Identify top influence leaders based on multiple metrics."""

        # Combine k-core and reach for influence score
        influence_scores = {}

        for node in influence_metrics["k_core"].keys():
            k_core_score = influence_metrics["k_core"][node]
            reach_score = reach_metrics["reach_2hop"].get(node, 0)

            # Weighted combination (k-core is more important for influence)
            combined_score = k_core_score * 0.6 + reach_score * 0.4
            influence_scores[node] = combined_score

        # Return top-k nodes
        top_nodes = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return [node for node, score in top_nodes]
