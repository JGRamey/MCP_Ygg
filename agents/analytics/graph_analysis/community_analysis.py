"""Community analysis module for network analysis."""

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set

import networkx as nx
import numpy as np

from .base import AnalysisConfig, AnalysisResult, BaseAnalyzer, CommunityAlgorithm


class CommunityAnalyzer(BaseAnalyzer):
    """Analyze communities and their properties."""

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.community_cache = {}

    async def analyze(self, session) -> AnalysisResult:
        """Perform community analysis."""
        start_time = time.time()

        try:
            await self.load_graph(session)

            if not self._validate_graph():
                return AnalysisResult(
                    analysis_type="community_analysis",
                    timestamp=datetime.now().isoformat(),
                    node_count=0,
                    edge_count=0,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Graph validation failed",
                )

            # Detect communities using multiple algorithms
            communities = {
                "louvain": self._louvain_communities(),
                "greedy_modularity": self._greedy_modularity_communities(),
                "label_propagation": self._label_propagation_communities(),
                "leiden": self._leiden_communities(),
            }

            # Analyze each set of communities
            analysis = {}
            for method, comm_list in communities.items():
                if comm_list:
                    analysis[method] = {
                        "communities": self._analyze_communities(comm_list),
                        "modularity": self._calculate_modularity(comm_list),
                        "coverage": self._calculate_coverage(comm_list),
                        "performance": self._calculate_performance(comm_list),
                        "summary": self._create_community_summary(comm_list),
                    }
                else:
                    analysis[method] = {
                        "error": "Failed to detect communities",
                        "communities": [],
                        "modularity": 0.0,
                        "coverage": 0.0,
                    }

            # Find best performing algorithm
            best_method = self._find_best_algorithm(analysis)

            return AnalysisResult(
                analysis_type="community_analysis",
                timestamp=datetime.now().isoformat(),
                node_count=self.graph.number_of_nodes(),
                edge_count=self.graph.number_of_edges(),
                execution_time=time.time() - start_time,
                success=True,
                data={
                    "algorithms": analysis,
                    "best_algorithm": best_method,
                    "comparison": self._compare_algorithms(analysis),
                },
            )

        except Exception as e:
            return AnalysisResult(
                analysis_type="community_analysis",
                timestamp=datetime.now().isoformat(),
                node_count=0,
                edge_count=0,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def _louvain_communities(self) -> List[Set[str]]:
        """Detect communities using Louvain algorithm."""
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(
                self.graph, resolution=self.config.community_resolution
            )

            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)

            return list(communities.values())
        except ImportError:
            return []
        except Exception:
            return []

    def _greedy_modularity_communities(self) -> List[Set[str]]:
        """Detect communities using greedy modularity."""
        try:
            return list(
                nx.community.greedy_modularity_communities(
                    self.graph, resolution=self.config.community_resolution
                )
            )
        except Exception:
            return []

    def _label_propagation_communities(self) -> List[Set[str]]:
        """Detect communities using label propagation."""
        try:
            return list(nx.community.label_propagation_communities(self.graph))
        except Exception:
            return []

    def _leiden_communities(self) -> List[Set[str]]:
        """Detect communities using Leiden algorithm."""
        try:
            import igraph as ig
            import leidenalg

            # Convert NetworkX graph to igraph
            edge_list = list(self.graph.edges())
            g = ig.Graph(edges=edge_list)

            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=self.config.community_resolution,
            )

            # Convert back to NetworkX format
            communities = []
            node_list = list(self.graph.nodes())

            for community in partition:
                comm_nodes = set()
                for node_idx in community:
                    comm_nodes.add(node_list[node_idx])
                communities.append(comm_nodes)

            return communities
        except ImportError:
            return []
        except Exception:
            return []

    def _analyze_communities(self, communities: List[Set[str]]) -> List[Dict[str, Any]]:
        """Analyze properties of detected communities."""
        analyzed = []

        for i, community in enumerate(communities):
            if len(community) < self.config.community_min_size:
                continue

            subgraph = self.graph.subgraph(community)

            # Basic properties
            size = len(community)
            density = nx.density(subgraph)

            # Domain analysis
            domains = [self.graph.nodes[n].get("domain", "unknown") for n in community]
            domain_counts = {d: domains.count(d) for d in set(domains)}
            primary_domain = max(domain_counts.keys(), key=domain_counts.get)

            # Connectivity analysis
            internal_edges = subgraph.number_of_edges()
            external_edges = sum(
                1
                for n in community
                for neighbor in self.graph.neighbors(n)
                if neighbor not in community
            )

            # Centrality analysis
            key_members = self._get_key_members(community)

            # Cohesion metrics
            avg_clustering = nx.average_clustering(subgraph)
            transitivity = nx.transitivity(subgraph)

            # Expansion and conductance
            total_degree = sum(self.graph.degree(n) for n in community)
            conductance = external_edges / total_degree if total_degree > 0 else 0
            expansion = external_edges / size if size > 0 else 0

            analyzed.append(
                {
                    "community_id": i,
                    "size": size,
                    "density": density,
                    "domains": domain_counts,
                    "primary_domain": primary_domain,
                    "domain_diversity": len(domain_counts),
                    "internal_edges": internal_edges,
                    "external_edges": external_edges,
                    "conductance": conductance,
                    "expansion": expansion,
                    "avg_clustering": avg_clustering,
                    "transitivity": transitivity,
                    "key_members": key_members,
                    "cohesion_score": self._calculate_cohesion_score(
                        density, size, len(domain_counts), avg_clustering
                    ),
                    "sample_nodes": list(community)[:5],  # Sample for display
                }
            )

        analyzed.sort(key=lambda x: x["cohesion_score"], reverse=True)
        return analyzed[:25]  # Return top 25 communities

    def _get_key_members(self, community: Set[str]) -> List[Dict[str, Any]]:
        """Get most important members of a community."""
        subgraph = self.graph.subgraph(community)

        key_members = []

        # Use multiple centrality measures
        try:
            pagerank = nx.pagerank(subgraph)
            betweenness = nx.betweenness_centrality(subgraph)
            degree_centrality = nx.degree_centrality(subgraph)

            # Composite score
            composite_scores = {}
            for node in community:
                score = (
                    0.4 * pagerank.get(node, 0)
                    + 0.3 * betweenness.get(node, 0)
                    + 0.3 * degree_centrality.get(node, 0)
                )
                composite_scores[node] = score

            # Get top members
            top_members = sorted(
                composite_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for node, score in top_members:
                key_members.append(
                    {
                        "node": node,
                        "name": self.graph.nodes[node].get("name", node),
                        "domain": self.graph.nodes[node].get("domain", "unknown"),
                        "importance_score": score,
                        "degree": subgraph.degree(node),
                        "pagerank": pagerank.get(node, 0),
                        "betweenness": betweenness.get(node, 0),
                    }
                )

        except Exception:
            # Fallback to degree centrality
            degrees = dict(subgraph.degree())
            top_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]

            for node, degree in top_by_degree:
                key_members.append(
                    {
                        "node": node,
                        "name": self.graph.nodes[node].get("name", node),
                        "domain": self.graph.nodes[node].get("domain", "unknown"),
                        "degree": degree,
                        "importance_score": degree / len(community),
                    }
                )

        return key_members

    def _calculate_cohesion_score(
        self, density: float, size: int, domain_diversity: int, avg_clustering: float
    ) -> float:
        """Calculate a composite cohesion score for a community."""
        # Weight factors
        density_weight = 0.4
        size_weight = 0.2
        domain_weight = 0.2
        clustering_weight = 0.2

        # Normalize size (log scale for large communities)
        size_score = np.log(size + 1) / np.log(self.graph.number_of_nodes() + 1)

        # Domain diversity penalty (prefer homogeneous communities)
        domain_score = 1.0 / (1.0 + domain_diversity - 1)

        # Composite score
        cohesion = (
            density_weight * density
            + size_weight * size_score
            + domain_weight * domain_score
            + clustering_weight * avg_clustering
        )

        return cohesion

    def _calculate_modularity(self, communities: List[Set[str]]) -> float:
        """Calculate modularity score."""
        try:
            return nx.community.modularity(self.graph, communities)
        except Exception:
            return 0.0

    def _calculate_coverage(self, communities: List[Set[str]]) -> float:
        """Calculate fraction of nodes covered by communities."""
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)

        return (
            len(all_nodes) / self.graph.number_of_nodes()
            if self.graph.number_of_nodes() > 0
            else 0.0
        )

    def _calculate_performance(self, communities: List[Set[str]]) -> float:
        """Calculate performance metric (internal density vs external sparsity)."""
        try:
            return nx.community.performance(self.graph, communities)
        except Exception:
            return 0.0

    def _create_community_summary(self, communities: List[Set[str]]) -> Dict[str, Any]:
        """Create summary statistics for communities."""
        if not communities:
            return {}

        sizes = [len(comm) for comm in communities]

        # Domain analysis
        domain_distribution = defaultdict(int)
        for community in communities:
            domains = [self.graph.nodes[n].get("domain", "unknown") for n in community]
            primary_domain = max(set(domains), key=domains.count)
            domain_distribution[primary_domain] += 1

        return {
            "total_communities": len(communities),
            "size_distribution": {
                "mean": np.mean(sizes),
                "median": np.median(sizes),
                "std": np.std(sizes),
                "min": min(sizes),
                "max": max(sizes),
            },
            "domain_distribution": dict(domain_distribution),
            "largest_community_size": max(sizes),
            "smallest_community_size": min(sizes),
            "communities_with_min_size": sum(
                1 for s in sizes if s >= self.config.community_min_size
            ),
        }

    def _find_best_algorithm(self, analysis: Dict[str, Any]) -> str:
        """Find the best performing algorithm based on multiple criteria."""
        scores = {}

        for method, data in analysis.items():
            if "modularity" not in data:
                continue

            # Composite score based on modularity, coverage, and performance
            modularity = data.get("modularity", 0)
            coverage = data.get("coverage", 0)
            performance = data.get("performance", 0)

            # Weight the metrics
            composite_score = 0.5 * modularity + 0.3 * coverage + 0.2 * performance

            scores[method] = composite_score

        if scores:
            return max(scores.keys(), key=lambda x: scores[x])
        return "greedy_modularity"  # Default fallback

    def _compare_algorithms(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different community detection algorithms."""
        comparison = {}

        metrics = ["modularity", "coverage", "performance"]

        for metric in metrics:
            metric_values = {}
            for method, data in analysis.items():
                if metric in data:
                    metric_values[method] = data[metric]

            if metric_values:
                best_method = max(metric_values.keys(), key=lambda x: metric_values[x])
                comparison[metric] = {
                    "values": metric_values,
                    "best": best_method,
                    "best_value": metric_values[best_method],
                }

        # Community count comparison
        community_counts = {}
        for method, data in analysis.items():
            if "communities" in data:
                community_counts[method] = len(data["communities"])

        comparison["community_counts"] = community_counts

        return comparison

    async def analyze_community_evolution(
        self, session, time_windows: List[str]
    ) -> Dict[str, Any]:
        """Analyze how communities evolve over time."""
        evolution_data = {}

        for window in time_windows:
            # Query for concepts in specific time window
            query = f"""
            MATCH (c:Concept)
            WHERE c.time_period = $window
            OPTIONAL MATCH (c)-[r:RELATES_TO]-(other:Concept)
            WHERE other.time_period = $window
            RETURN c, r, other
            """

            result = await session.run(query, window=window)

            # Build temporal graph
            temporal_graph = nx.Graph()
            async for record in result:
                node = record["c"]
                if node:
                    node_id = node.get("id", str(node.element_id))
                    temporal_graph.add_node(node_id, **dict(node))

                    if record["r"] and record["other"]:
                        other = record["other"]
                        other_id = other.get("id", str(other.element_id))
                        temporal_graph.add_edge(node_id, other_id)

            # Detect communities in temporal graph
            if temporal_graph.number_of_nodes() > 0:
                communities = self._greedy_modularity_communities_for_graph(
                    temporal_graph
                )
                evolution_data[window] = {
                    "node_count": temporal_graph.number_of_nodes(),
                    "edge_count": temporal_graph.number_of_edges(),
                    "community_count": len(communities),
                    "modularity": self._calculate_modularity_for_graph(
                        temporal_graph, communities
                    ),
                    "largest_community_size": (
                        max(len(c) for c in communities) if communities else 0
                    ),
                }

        return evolution_data

    def _greedy_modularity_communities_for_graph(
        self, graph: nx.Graph
    ) -> List[Set[str]]:
        """Helper method to detect communities for a specific graph."""
        try:
            return list(nx.community.greedy_modularity_communities(graph))
        except Exception:
            return []

    def _calculate_modularity_for_graph(
        self, graph: nx.Graph, communities: List[Set[str]]
    ) -> float:
        """Helper method to calculate modularity for a specific graph."""
        try:
            return nx.community.modularity(graph, communities)
        except Exception:
            return 0.0
