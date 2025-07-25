"""
Flow Analysis Module - Knowledge flow and information propagation analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from ..config import NetworkConfig
from ..graph_utils import GraphMetricsAggregator, TemporalGraphUtils
from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics


class FlowAnalyzer:
    """Specialized analyzer for knowledge flow patterns."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("flow_analyzer")

    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze knowledge flow patterns."""

        self.logger.info("Analyzing knowledge flow...")

        # Create directed graph for flow analysis
        directed_graph = self._prepare_directed_graph(graph)

        # Calculate flow-related metrics
        flow_metrics = self._calculate_flow_metrics(directed_graph)

        # Calculate HITS scores (authorities and hubs)
        hits_metrics = self._calculate_hits_metrics(directed_graph)

        # Analyze flow patterns
        flow_patterns = self._analyze_flow_patterns(directed_graph, flow_metrics)

        # Create node metrics
        node_metrics = self._create_flow_node_metrics(
            directed_graph, flow_metrics, hits_metrics, flow_patterns
        )

        # Calculate graph metrics
        graph_metrics = self._calculate_flow_graph_metrics(
            directed_graph, flow_metrics, hits_metrics, flow_patterns
        )

        # Generate insights
        insights = self._generate_flow_insights(
            directed_graph, flow_metrics, hits_metrics, flow_patterns
        )

        # Generate recommendations
        recommendations = self._generate_flow_recommendations(
            flow_metrics, hits_metrics, flow_patterns
        )

        return NetworkAnalysisResult(
            analysis_type=AnalysisType.KNOWLEDGE_FLOW,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0,
        )

    def _prepare_directed_graph(self, graph: nx.Graph) -> nx.DiGraph:
        """Prepare directed graph for flow analysis."""

        # If already directed, use as is
        if graph.is_directed():
            return graph

        # Try to create temporal directed graph first
        try:
            directed_graph = TemporalGraphUtils.create_temporal_directed_graph(graph)
            if directed_graph.number_of_edges() > 0:
                return directed_graph
        except Exception as e:
            self.logger.warning(f"Temporal graph creation failed: {e}")

        # Fallback: convert undirected to directed by adding both directions
        directed_graph = nx.DiGraph()

        # Add all nodes
        for node, data in graph.nodes(data=True):
            directed_graph.add_node(node, **data)

        # Add directed edges (both directions for undirected edges)
        for u, v, data in graph.edges(data=True):
            directed_graph.add_edge(u, v, **data)
            directed_graph.add_edge(v, u, **data)

        return directed_graph

    def _calculate_flow_metrics(self, directed_graph: nx.DiGraph) -> Dict[str, any]:
        """Calculate flow-related metrics."""

        metrics = {}

        # In-degree and out-degree
        metrics["in_degree"] = dict(directed_graph.in_degree())
        metrics["out_degree"] = dict(directed_graph.out_degree())

        # Flow ratios
        flow_ratios = {}
        for node in directed_graph.nodes:
            in_deg = metrics["in_degree"].get(node, 0)
            out_deg = metrics["out_degree"].get(node, 0)
            total_deg = in_deg + out_deg

            if total_deg > 0:
                flow_ratios[node] = out_deg / total_deg  # Higher = more source-like
            else:
                flow_ratios[node] = 0.5

        metrics["flow_ratio"] = flow_ratios

        # Reciprocity
        try:
            metrics["reciprocity"] = nx.reciprocity(directed_graph)
        except Exception:
            metrics["reciprocity"] = 0.0

        return metrics

    def _calculate_hits_metrics(self, directed_graph: nx.DiGraph) -> Dict[str, any]:
        """Calculate HITS algorithm scores (authorities and hubs)."""

        try:
            hubs, authorities = nx.hits(directed_graph, max_iter=100, normalized=True)

            return {"hubs": hubs, "authorities": authorities}

        except Exception as e:
            self.logger.warning(f"HITS algorithm failed: {e}")
            return {
                "hubs": {node: 0.0 for node in directed_graph.nodes},
                "authorities": {node: 0.0 for node in directed_graph.nodes},
            }

    def _analyze_flow_patterns(
        self, directed_graph: nx.DiGraph, flow_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Analyze specific flow patterns in the network."""

        patterns = {}

        # Identify sources and sinks
        sources = []
        sinks = []
        bridges = []

        for node in directed_graph.nodes:
            flow_ratio = flow_metrics["flow_ratio"].get(node, 0.5)
            degree = directed_graph.degree(node)

            if degree > 2:  # Only consider nodes with meaningful connections
                if flow_ratio > 0.8:
                    sources.append(node)
                elif flow_ratio < 0.2:
                    sinks.append(node)
                elif 0.4 <= flow_ratio <= 0.6:
                    bridges.append(node)

        patterns["sources"] = sources
        patterns["sinks"] = sinks
        patterns["bridges"] = bridges

        # Identify strongly connected components
        try:
            scc = list(nx.strongly_connected_components(directed_graph))
            patterns["strongly_connected_components"] = scc
            patterns["num_scc"] = len(scc)

            # Find largest SCC
            if scc:
                largest_scc = max(scc, key=len)
                patterns["largest_scc_size"] = len(largest_scc)
                patterns["largest_scc_nodes"] = list(largest_scc)

        except Exception as e:
            self.logger.warning(f"SCC analysis failed: {e}")
            patterns["strongly_connected_components"] = []
            patterns["num_scc"] = 0
            patterns["largest_scc_size"] = 0
            patterns["largest_scc_nodes"] = []

        # Analyze flow bottlenecks (high in-degree, low out-degree)
        bottlenecks = []
        for node in directed_graph.nodes:
            in_deg = flow_metrics["in_degree"].get(node, 0)
            out_deg = flow_metrics["out_degree"].get(node, 0)

            if in_deg > 5 and out_deg < in_deg * 0.3:  # High input, low output
                bottlenecks.append(node)

        patterns["bottlenecks"] = bottlenecks

        return patterns

    def _create_flow_node_metrics(
        self,
        directed_graph: nx.DiGraph,
        flow_metrics: Dict[str, any],
        hits_metrics: Dict[str, any],
        flow_patterns: Dict[str, any],
    ) -> List[NodeMetrics]:
        """Create node metrics for flow analysis."""

        node_metrics = []

        for node in directed_graph.nodes:
            node_data = directed_graph.nodes[node]

            # Compile flow-related scores
            centrality_scores = {
                "in_degree": flow_metrics["in_degree"].get(node, 0),
                "out_degree": flow_metrics["out_degree"].get(node, 0),
                "flow_ratio": flow_metrics["flow_ratio"].get(node, 0.5),
                "hub_score": hits_metrics["hubs"].get(node, 0),
                "authority_score": hits_metrics["authorities"].get(node, 0),
                "is_source": node in flow_patterns["sources"],
                "is_sink": node in flow_patterns["sinks"],
                "is_bridge": node in flow_patterns["bridges"],
                "is_bottleneck": node in flow_patterns["bottlenecks"],
            }

            # Calculate flow importance score
            hub_score = hits_metrics["hubs"].get(node, 0)
            authority_score = hits_metrics["authorities"].get(node, 0)
            in_degree = flow_metrics["in_degree"].get(node, 0)
            out_degree = flow_metrics["out_degree"].get(node, 0)

            flow_importance = (
                hub_score * 0.3
                + authority_score * 0.3
                + (in_degree / 10) * 0.2
                + (out_degree / 10) * 0.2
            )
            centrality_scores["flow_importance"] = flow_importance

            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,
                clustering_coefficient=0.0,  # Not applicable for directed flow
                degree=directed_graph.degree(node),
                metadata=node_data,
            )
            node_metrics.append(metrics)

        return node_metrics

    def _calculate_flow_graph_metrics(
        self,
        directed_graph: nx.DiGraph,
        flow_metrics: Dict[str, any],
        hits_metrics: Dict[str, any],
        flow_patterns: Dict[str, any],
    ) -> Dict[str, any]:
        """Calculate graph-level metrics for flow analysis."""

        metrics = {}

        # Basic directed graph metrics
        metrics["num_nodes"] = directed_graph.number_of_nodes()
        metrics["num_edges"] = directed_graph.number_of_edges()
        metrics["density"] = nx.density(directed_graph)
        metrics["reciprocity"] = flow_metrics["reciprocity"]

        # Degree statistics
        in_degrees = list(flow_metrics["in_degree"].values())
        out_degrees = list(flow_metrics["out_degree"].values())

        if in_degrees:
            metrics.update(
                {
                    "avg_in_degree": np.mean(in_degrees),
                    "max_in_degree": max(in_degrees),
                    "std_in_degree": np.std(in_degrees),
                    "avg_out_degree": np.mean(out_degrees),
                    "max_out_degree": max(out_degrees),
                    "std_out_degree": np.std(out_degrees),
                }
            )

        # Flow pattern metrics
        metrics.update(
            {
                "num_sources": len(flow_patterns["sources"]),
                "num_sinks": len(flow_patterns["sinks"]),
                "num_bridges": len(flow_patterns["bridges"]),
                "num_bottlenecks": len(flow_patterns["bottlenecks"]),
                "num_strongly_connected_components": flow_patterns["num_scc"],
                "largest_scc_size": flow_patterns["largest_scc_size"],
            }
        )

        # Flow ratios statistics
        flow_ratios = list(flow_metrics["flow_ratio"].values())
        if flow_ratios:
            metrics.update(
                {
                    "avg_flow_ratio": np.mean(flow_ratios),
                    "flow_ratio_std": np.std(flow_ratios),
                }
            )

        # HITS statistics
        hub_scores = list(hits_metrics["hubs"].values())
        authority_scores = list(hits_metrics["authorities"].values())

        if hub_scores:
            metrics.update(
                {
                    "avg_hub_score": np.mean(hub_scores),
                    "max_hub_score": max(hub_scores),
                    "avg_authority_score": np.mean(authority_scores),
                    "max_authority_score": max(authority_scores),
                }
            )

        return metrics

    def _generate_flow_insights(
        self,
        directed_graph: nx.DiGraph,
        flow_metrics: Dict[str, any],
        hits_metrics: Dict[str, any],
        flow_patterns: Dict[str, any],
    ) -> List[str]:
        """Generate insights from flow analysis."""

        insights = []

        # Basic flow insights
        reciprocity = flow_metrics["reciprocity"]
        insights.append(f"Network reciprocity: {reciprocity:.3f}")

        if reciprocity > 0.5:
            insights.append(
                "High reciprocity indicates bidirectional knowledge exchange"
            )
        elif reciprocity < 0.2:
            insights.append("Low reciprocity suggests unidirectional knowledge flow")

        # Degree insights
        in_degrees = list(flow_metrics["in_degree"].values())
        out_degrees = list(flow_metrics["out_degree"].values())

        if in_degrees:
            avg_in = np.mean(in_degrees)
            avg_out = np.mean(out_degrees)
            insights.append(
                f"Average in-degree: {avg_in:.2f}, out-degree: {avg_out:.2f}"
            )

        # Flow pattern insights
        num_sources = len(flow_patterns["sources"])
        num_sinks = len(flow_patterns["sinks"])
        num_bridges = len(flow_patterns["bridges"])
        num_bottlenecks = len(flow_patterns["bottlenecks"])

        insights.append(
            f"Identified {num_sources} knowledge sources, {num_sinks} sinks, {num_bridges} bridges"
        )

        if num_bottlenecks > 0:
            insights.append(
                f"Found {num_bottlenecks} potential bottlenecks in knowledge flow"
            )

        # Strongly connected components
        num_scc = flow_patterns["num_scc"]
        largest_scc_size = flow_patterns["largest_scc_size"]

        if num_scc > 1:
            insights.append(f"Network has {num_scc} strongly connected components")
            insights.append(f"Largest component has {largest_scc_size} nodes")

        # Top authorities and hubs
        authorities = hits_metrics["authorities"]
        hubs = hits_metrics["hubs"]

        if authorities:
            top_authorities = sorted(
                authorities.items(), key=lambda x: x[1], reverse=True
            )[:3]
            if top_authorities[0][1] > 0:
                insights.append("Top knowledge authorities:")
                for node, score in top_authorities:
                    node_title = directed_graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: {score:.3f}")

        if hubs:
            top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_hubs[0][1] > 0:
                insights.append("Top knowledge hubs:")
                for node, score in top_hubs:
                    node_title = directed_graph.nodes[node].get("title", f"Node {node}")
                    insights.append(f"  • {node_title}: {score:.3f}")

        return insights

    def _generate_flow_recommendations(
        self,
        flow_metrics: Dict[str, any],
        hits_metrics: Dict[str, any],
        flow_patterns: Dict[str, any],
    ) -> List[str]:
        """Generate recommendations from flow analysis."""

        recommendations = [
            "Focus on high-authority nodes for knowledge consolidation",
            "Leverage high-hub nodes for knowledge distribution",
            "Monitor reciprocity changes to understand collaboration patterns",
        ]

        # Source/sink recommendations
        num_sources = len(flow_patterns["sources"])
        num_sinks = len(flow_patterns["sinks"])

        if num_sources > 0 and num_sinks > 0:
            recommendations.append(
                "Connect knowledge sources to sinks to improve flow efficiency"
            )

        if num_sources < 3:
            recommendations.append(
                "Consider developing additional knowledge sources for redundancy"
            )

        # Bottleneck recommendations
        num_bottlenecks = len(flow_patterns["bottlenecks"])
        if num_bottlenecks > 0:
            recommendations.append(
                "Address bottlenecks to improve knowledge flow throughput"
            )
            recommendations.append("Create alternative paths around bottleneck nodes")

        # Reciprocity recommendations
        reciprocity = flow_metrics["reciprocity"]
        if reciprocity < 0.3:
            recommendations.append(
                "Encourage bidirectional exchanges to increase knowledge reciprocity"
            )
        elif reciprocity > 0.8:
            recommendations.append(
                "High reciprocity enables collaborative knowledge development"
            )

        # Connectivity recommendations
        num_scc = flow_patterns["num_scc"]
        if num_scc > 5:
            recommendations.append(
                "Multiple components suggest opportunities for cross-component bridging"
            )

        return recommendations

    def identify_flow_leaders(
        self,
        hits_metrics: Dict[str, any],
        flow_metrics: Dict[str, any],
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        """Identify top nodes for different flow roles."""

        leaders = {}

        # Top authorities (knowledge destinations)
        authorities = hits_metrics["authorities"]
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        leaders["authorities"] = [node for node, score in top_authorities]

        # Top hubs (knowledge distributors)
        hubs = hits_metrics["hubs"]
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        leaders["hubs"] = [node for node, score in top_hubs]

        # Top sources (high out-degree, low in-degree)
        out_degrees = flow_metrics["out_degree"]
        in_degrees = flow_metrics["in_degree"]

        source_scores = {}
        for node in out_degrees.keys():
            out_deg = out_degrees.get(node, 0)
            in_deg = in_degrees.get(node, 0)
            total_deg = out_deg + in_deg

            if total_deg > 2:  # Minimum activity threshold
                source_score = out_deg - in_deg  # Favor high output
                source_scores[node] = source_score

        top_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        leaders["sources"] = [node for node, score in top_sources if score > 0]

        return leaders
