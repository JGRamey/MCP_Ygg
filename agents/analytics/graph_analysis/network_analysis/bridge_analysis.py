"""
Bridge Analysis Module - Focused bridge nodes and structural holes analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..config import NetworkConfig
from ..graph_utils import GraphMetricsAggregator
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class BridgeAnalyzer:
    """Specialized analyzer for bridge nodes and structural holes."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("bridge_analyzer")

    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze bridge nodes and structural holes."""

        self.logger.info("Analyzing bridge nodes...")

        # Calculate betweenness centrality (key for bridge identification)
        betweenness_metrics = self._calculate_betweenness_metrics(graph)

        # Calculate edge betweenness
        edge_betweenness_metrics = self._calculate_edge_betweenness_metrics(graph)

        # Calculate structural holes metrics
        structural_holes_metrics = self._calculate_structural_holes_metrics(graph)

        # Identify bridge nodes
        bridge_identification = self._identify_bridge_nodes(graph, betweenness_metrics)

        # Create node metrics
        node_metrics = self._create_bridge_node_metrics(
            graph, betweenness_metrics, structural_holes_metrics, bridge_identification
        )

        # Calculate graph metrics
        graph_metrics = self._calculate_bridge_graph_metrics(
            graph,
            betweenness_metrics,
            edge_betweenness_metrics,
            structural_holes_metrics,
            bridge_identification,
        )

        # Generate insights
        insights = self._generate_bridge_insights(
            graph, betweenness_metrics, structural_holes_metrics, bridge_identification
        )

        # Generate recommendations
        recommendations = self._generate_bridge_recommendations(
            betweenness_metrics, structural_holes_metrics, bridge_identification
        )

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.BRIDGE_NODES,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )

    def _calculate_betweenness_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate betweenness centrality and related metrics."""

        try:
            betweenness = nx.betweenness_centrality(graph)

            # Calculate normalized betweenness if requested
            if self.config.normalize_centrality:
                normalized_betweenness = betweenness
            else:
                # Calculate unnormalized betweenness
                n = len(graph.nodes)
                normalization_factor = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
                normalized_betweenness = {
                    node: score / normalization_factor
                    for node, score in betweenness.items()
                }

            return {
                "betweenness": betweenness,
                "normalized_betweenness": normalized_betweenness,
            }

        except Exception as e:
            self.logger.warning(f"Betweenness calculation failed: {e}")
            return {
                "betweenness": {node: 0.0 for node in graph.nodes},
                "normalized_betweenness": {node: 0.0 for node in graph.nodes},
            }

    def _calculate_edge_betweenness_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate edge betweenness centrality."""

        try:
            edge_betweenness = nx.edge_betweenness_centrality(graph)

            # Find edges with highest betweenness (critical bridges)
            critical_edges = sorted(
                edge_betweenness.items(), key=lambda x: x[1], reverse=True
            )

            return {
                "edge_betweenness": edge_betweenness,
                "critical_edges": critical_edges[:10],  # Top 10 critical edges
            }

        except Exception as e:
            self.logger.warning(f"Edge betweenness calculation failed: {e}")
            return {"edge_betweenness": {}, "critical_edges": []}

    def _calculate_structural_holes_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate structural holes metrics (effective size, constraint)."""

        effective_size = {}
        constraint = {}
        redundancy = {}

        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))

            if len(neighbors) <= 1:
                effective_size[node] = 0
                constraint[node] = 1.0 if neighbors else 0.0
                redundancy[node] = 0.0
                continue

            # Calculate effective size (diversity of connections)
            neighbor_connections = 0
            total_possible_connections = len(neighbors) * (len(neighbors) - 1) / 2

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if graph.has_edge(neighbors[i], neighbors[j]):
                        neighbor_connections += 1

            # Effective size calculation
            if total_possible_connections > 0:
                redundancy_ratio = neighbor_connections / total_possible_connections
                effective_size[node] = len(neighbors) * (1 - redundancy_ratio)
                redundancy[node] = redundancy_ratio
            else:
                effective_size[node] = len(neighbors)
                redundancy[node] = 0.0

            # Calculate constraint (inverse of structural holes)
            # Simplified constraint calculation
            constraint[node] = 1.0 / (effective_size[node] + 1)

        return {
            "effective_size": effective_size,
            "constraint": constraint,
            "redundancy": redundancy,
        }

    def _identify_bridge_nodes(
        self, graph: nx.Graph, betweenness_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Identify bridge nodes using multiple criteria."""

        betweenness = betweenness_metrics["betweenness"]

        # Define bridge threshold (90th percentile)
        betweenness_values = list(betweenness.values())
        if betweenness_values:
            bridge_threshold = np.percentile(betweenness_values, 90)
        else:
            bridge_threshold = 0

        # Identify bridge nodes
        bridge_nodes = [
            node
            for node, score in betweenness.items()
            if score > bridge_threshold and score > 0
        ]

        # Identify articulation points (structural bridges)
        articulation_points = list(nx.articulation_points(graph))

        # Identify bridges (edges whose removal increases connected components)
        bridge_edges = list(nx.bridges(graph))

        # Nodes that are endpoints of bridge edges
        bridge_edge_nodes = set()
        for edge in bridge_edges:
            bridge_edge_nodes.update(edge)

        return {
            "bridge_nodes": bridge_nodes,
            "articulation_points": articulation_points,
            "bridge_edges": bridge_edges,
            "bridge_edge_nodes": list(bridge_edge_nodes),
            "bridge_threshold": bridge_threshold,
        }

    def _create_bridge_node_metrics(
        self,
        graph: nx.Graph,
        betweenness_metrics: Dict[str, any],
        structural_holes_metrics: Dict[str, any],
        bridge_identification: Dict[str, any],
    ) -> List[NodeMetrics]:
        """Create node metrics for bridge analysis."""

        node_metrics = []

        for node in graph.nodes:
            node_data = graph.nodes[node]

            # Compile bridge-related scores
            centrality_scores = {
                "betweenness": betweenness_metrics["betweenness"].get(node, 0),
                "normalized_betweenness": betweenness_metrics[
                    "normalized_betweenness"
                ].get(node, 0),
                "effective_size": structural_holes_metrics["effective_size"].get(
                    node, 0
                ),
                "constraint": structural_holes_metrics["constraint"].get(node, 0),
                "redundancy": structural_holes_metrics["redundancy"].get(node, 0),
                "is_bridge": node in bridge_identification["bridge_nodes"],
                "is_articulation_point": node
                in bridge_identification["articulation_points"],
                "is_bridge_edge_node": node
                in bridge_identification["bridge_edge_nodes"],
            }

            # Calculate bridge importance score
            betweenness_score = betweenness_metrics["betweenness"].get(node, 0)
            effective_size_score = structural_holes_metrics["effective_size"].get(
                node, 0
            )
            is_articulation = (
                1.0 if node in bridge_identification["articulation_points"] else 0.0
            )

            bridge_importance = (
                betweenness_score * 0.4
                + (effective_size_score / 10) * 0.4
                + is_articulation * 0.2
            )
            centrality_scores["bridge_importance"] = bridge_importance

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _calculate_bridge_graph_metrics(
        self,
        graph: nx.Graph,
        betweenness_metrics: Dict[str, any],
        edge_betweenness_metrics: Dict[str, any],
        structural_holes_metrics: Dict[str, any],
        bridge_identification: Dict[str, any],
    ) -> Dict[str, any]:
        """Calculate graph-level metrics for bridge analysis."""

        metrics = {}

        # Basic graph metrics
        metrics.update(self.metrics_aggregator.calculate_comprehensive_metrics(graph))

        # Bridge-specific metrics
        betweenness_values = list(betweenness_metrics["betweenness"].values())
        if betweenness_values:
            metrics.update(
                {
                    "num_bridge_nodes": len(bridge_identification["bridge_nodes"]),
                    "num_articulation_points": len(
                        bridge_identification["articulation_points"]
                    ),
                    "num_bridge_edges": len(bridge_identification["bridge_edges"]),
                    "avg_betweenness": np.mean(betweenness_values),
                    "max_betweenness": max(betweenness_values),
                    "betweenness_std": np.std(betweenness_values),
                    "bridge_concentration": (
                        len(bridge_identification["bridge_nodes"]) / len(graph.nodes)
                        if graph.nodes
                        else 0
                    ),
                }
            )

        # Structural holes metrics
        effective_size_values = list(
            structural_holes_metrics["effective_size"].values()
        )
        constraint_values = list(structural_holes_metrics["constraint"].values())

        if effective_size_values:
            metrics.update(
                {
                    "avg_effective_size": np.mean(effective_size_values),
                    "max_effective_size": max(effective_size_values),
                    "avg_constraint": np.mean(constraint_values),
                    "min_constraint": min(constraint_values),
                }
            )

        # Network vulnerability metrics
        metrics["structural_vulnerability"] = (
            len(bridge_identification["articulation_points"]) / len(graph.nodes)
            if graph.nodes
            else 0
        )

        # Edge criticality
        if edge_betweenness_metrics["edge_betweenness"]:
            edge_bet_values = list(
                edge_betweenness_metrics["edge_betweenness"].values()
            )
            metrics["avg_edge_betweenness"] = np.mean(edge_bet_values)
            metrics["max_edge_betweenness"] = max(edge_bet_values)

        return metrics

    def _generate_bridge_insights(
        self,
        graph: nx.Graph,
        betweenness_metrics: Dict[str, any],
        structural_holes_metrics: Dict[str, any],
        bridge_identification: Dict[str, any],
    ) -> List[str]:
        """Generate insights from bridge analysis."""

        insights = []

        # Bridge node insights
        num_bridges = len(bridge_identification["bridge_nodes"])
        total_nodes = len(graph.nodes)
        bridge_percentage = (num_bridges / total_nodes * 100) if total_nodes > 0 else 0

        insights.append(
            f"Identified {num_bridges} bridge nodes ({bridge_percentage:.1f}% of network)"
        )

        # Betweenness insights
        betweenness = betweenness_metrics["betweenness"]
        if betweenness:
            avg_betweenness = np.mean(list(betweenness.values()))
            max_betweenness = max(betweenness.values())

            insights.append(f"Average betweenness centrality: {avg_betweenness:.4f}")
            insights.append(f"Maximum betweenness centrality: {max_betweenness:.4f}")

        # Structural vulnerability
        num_articulation_points = len(bridge_identification["articulation_points"])
        if num_articulation_points > 0:
            vulnerability = (
                num_articulation_points / total_nodes if total_nodes > 0 else 0
            )
            insights.append(
                f"Network has {num_articulation_points} articulation points ({vulnerability:.1%} vulnerability)"
            )

            if vulnerability > 0.1:
                insights.append(
                    "High structural vulnerability - network depends heavily on key nodes"
                )
            elif vulnerability < 0.05:
                insights.append(
                    "Low structural vulnerability - network has redundant paths"
                )

        # Bridge edges
        num_bridge_edges = len(bridge_identification["bridge_edges"])
        if num_bridge_edges > 0:
            insights.append(f"Network has {num_bridge_edges} critical bridge edges")

        # Structural holes insights
        effective_size = structural_holes_metrics["effective_size"]
        if effective_size:
            avg_effective_size = np.mean(list(effective_size.values()))
            insights.append(f"Average effective size: {avg_effective_size:.2f}")

            # Find nodes with most structural holes
            top_structural_holes = sorted(
                effective_size.items(), key=lambda x: x[1], reverse=True
            )[:3]
            if top_structural_holes and top_structural_holes[0][1] > 0:
                insights.append("Nodes with most structural holes:")
                for node, score in top_structural_holes:
                    node_title = graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: effective size = {score:.2f}")

        # Top bridge nodes by betweenness
        if betweenness:
            top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            if top_bridges and top_bridges[0][1] > 0:
                insights.append("Top bridge nodes by betweenness:")
                for node, score in top_bridges:
                    node_title = graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: betweenness = {score:.4f}")

        return insights

    def _generate_bridge_recommendations(
        self,
        betweenness_metrics: Dict[str, any],
        structural_holes_metrics: Dict[str, any],
        bridge_identification: Dict[str, any],
    ) -> List[str]:
        """Generate recommendations from bridge analysis."""

        recommendations = [
            "Monitor bridge nodes as they are critical for network connectivity",
            "Consider bridge nodes for strategic interventions or communications",
            "Strengthen connections around high-betweenness nodes to reduce vulnerability",
        ]

        # Vulnerability-based recommendations
        num_articulation_points = len(bridge_identification["articulation_points"])
        num_bridge_edges = len(bridge_identification["bridge_edges"])

        if num_articulation_points > 0:
            recommendations.append(
                "Create redundant paths around articulation points to improve robustness"
            )

        if num_bridge_edges > 0:
            recommendations.append(
                "Consider strengthening or creating alternative paths for critical bridge edges"
            )

        # Structural holes recommendations
        effective_size = structural_holes_metrics["effective_size"]
        if effective_size:
            max_effective_size = (
                max(effective_size.values()) if effective_size.values() else 0
            )

            if max_effective_size > 5:
                recommendations.append(
                    "Leverage nodes with structural holes for brokerage opportunities"
                )
                recommendations.append(
                    "Consider nodes with high effective size for cross-group coordination"
                )

        # Betweenness-based recommendations
        betweenness = betweenness_metrics["betweenness"]
        if betweenness:
            max_betweenness = max(betweenness.values()) if betweenness.values() else 0

            if max_betweenness > 0.1:
                recommendations.append(
                    "High betweenness nodes are bottlenecks - consider load balancing strategies"
                )

            # Concentration analysis
            betweenness_values = list(betweenness.values())
            if len(betweenness_values) > 1:
                concentration = (
                    np.std(betweenness_values) / np.mean(betweenness_values)
                    if np.mean(betweenness_values) > 0
                    else 0
                )

                if concentration > 1.0:
                    recommendations.append(
                        "High betweenness concentration suggests centralized control points"
                    )
                else:
                    recommendations.append(
                        "Distributed betweenness suggests decentralized bridge structure"
                    )

        return recommendations

    def identify_critical_nodes(
        self,
        betweenness_metrics: Dict[str, any],
        structural_holes_metrics: Dict[str, any],
        bridge_identification: Dict[str, any],
    ) -> List[str]:
        """Identify the most critical nodes for network connectivity."""

        # Combine multiple measures to identify critical nodes
        criticality_scores = {}

        betweenness = betweenness_metrics["betweenness"]
        effective_size = structural_holes_metrics["effective_size"]
        articulation_points = set(bridge_identification["articulation_points"])
        bridge_edge_nodes = set(bridge_identification["bridge_edge_nodes"])

        for node in betweenness.keys():
            score = 0

            # Betweenness contribution (40%)
            score += betweenness.get(node, 0) * 0.4

            # Effective size contribution (30%)
            score += (effective_size.get(node, 0) / 10) * 0.3

            # Articulation point bonus (20%)
            if node in articulation_points:
                score += 0.2

            # Bridge edge node bonus (10%)
            if node in bridge_edge_nodes:
                score += 0.1

            criticality_scores[node] = score

        # Return top 10 most critical nodes
        critical_nodes = sorted(
            criticality_scores.items(), key=lambda x: x[1], reverse=True
        )[:10]
        return [node for node, score in critical_nodes]
