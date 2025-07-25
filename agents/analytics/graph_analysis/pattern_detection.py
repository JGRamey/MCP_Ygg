"""Pattern detection algorithms for network analysis."""

import itertools
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from .base import AnalysisConfig, AnalysisResult, BaseAnalyzer


class PatternDetector(BaseAnalyzer):
    """Detect various patterns in the knowledge graph."""

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.pattern_cache = {}

    async def analyze(self, session) -> AnalysisResult:
        """Detect patterns in the graph."""
        start_time = time.time()

        try:
            await self.load_graph(session)

            if not self._validate_graph():
                return AnalysisResult(
                    analysis_type="pattern_detection",
                    timestamp=datetime.now().isoformat(),
                    node_count=0,
                    edge_count=0,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Graph validation failed",
                )

            patterns = {
                "triadic_patterns": self._detect_triadic_patterns(),
                "cross_domain_bridges": self._find_cross_domain_bridges(),
                "knowledge_chains": self._find_knowledge_chains(),
                "concept_clusters": self._identify_concept_clusters(),
                "temporal_patterns": await self._detect_temporal_patterns(session),
                "structural_motifs": self._detect_structural_motifs(),
                "hub_patterns": self._detect_hub_patterns(),
                "bridge_patterns": self._detect_bridge_patterns(),
            }

            return AnalysisResult(
                analysis_type="pattern_detection",
                timestamp=datetime.now().isoformat(),
                node_count=self.graph.number_of_nodes(),
                edge_count=self.graph.number_of_edges(),
                execution_time=time.time() - start_time,
                success=True,
                data=patterns,
            )

        except Exception as e:
            return AnalysisResult(
                analysis_type="pattern_detection",
                timestamp=datetime.now().isoformat(),
                node_count=0,
                edge_count=0,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def _detect_triadic_patterns(self) -> List[Dict[str, Any]]:
        """Detect common triadic patterns (triangles, etc.)."""
        triangles = []

        # Find all triangles in the graph
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    if self.graph.has_edge(n1, n2):
                        # Found a triangle
                        triangle_nodes = [node, n1, n2]
                        domains = [
                            self.graph.nodes[n].get("domain", "unknown")
                            for n in triangle_nodes
                        ]

                        # Calculate triangle properties
                        edge_weights = []
                        for u, v in itertools.combinations(triangle_nodes, 2):
                            weight = self.graph[u][v].get("weight", 1.0)
                            edge_weights.append(weight)

                        triangles.append(
                            {
                                "nodes": triangle_nodes,
                                "domains": domains,
                                "cross_domain": len(set(domains)) > 1,
                                "avg_weight": np.mean(edge_weights),
                                "min_weight": min(edge_weights),
                                "max_weight": max(edge_weights),
                                "triangle_strength": np.prod(edge_weights),
                                "unique_domains": len(set(domains)),
                            }
                        )

        # Sort by triangle strength and return top patterns
        triangles.sort(key=lambda x: x["triangle_strength"], reverse=True)
        return triangles[:20]

    def _find_cross_domain_bridges(self) -> List[Dict[str, Any]]:
        """Find nodes that bridge different domains."""
        bridges = []

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                continue

            # Get domains of neighbors
            neighbor_domains = [
                self.graph.nodes[n].get("domain", "unknown") for n in neighbors
            ]
            unique_domains = set(neighbor_domains)

            if len(unique_domains) > 1:
                node_domain = self.graph.nodes[node].get("domain", "unknown")
                domain_counts = {d: neighbor_domains.count(d) for d in unique_domains}

                # Calculate bridge properties
                bridge_score = len(unique_domains) * len(neighbors)
                domain_diversity = len(unique_domains) / len(neighbors)

                bridges.append(
                    {
                        "node": node,
                        "node_name": self.graph.nodes[node].get("name", node),
                        "node_domain": node_domain,
                        "connected_domains": list(unique_domains),
                        "domain_counts": domain_counts,
                        "bridge_score": bridge_score,
                        "domain_diversity": domain_diversity,
                        "neighbor_count": len(neighbors),
                        "is_cross_domain_hub": len(unique_domains) >= 3,
                    }
                )

        bridges.sort(key=lambda x: x["bridge_score"], reverse=True)
        return bridges[:15]

    def _find_knowledge_chains(self) -> List[Dict[str, Any]]:
        """Find meaningful chains of connected concepts."""
        chains = []
        important_nodes = self._get_important_nodes()

        # Find paths between important nodes
        for i, start in enumerate(important_nodes[:10]):
            for end in important_nodes[i + 1 : 20]:
                try:
                    # Find multiple simple paths
                    paths = list(nx.all_simple_paths(self.graph, start, end, cutoff=6))[
                        :5
                    ]  # Limit to 5 paths

                    for path in paths:
                        if len(path) >= 3:  # Meaningful chains
                            # Calculate chain properties
                            chain_domains = []
                            chain_weights = []

                            for node in path:
                                domain = self.graph.nodes[node].get("domain", "unknown")
                                chain_domains.append(domain)

                            for i in range(len(path) - 1):
                                weight = self.graph[path[i]][path[i + 1]].get(
                                    "weight", 1.0
                                )
                                chain_weights.append(weight)

                            chains.append(
                                {
                                    "path": path,
                                    "path_names": [
                                        self.graph.nodes[node].get("name", node)
                                        for node in path
                                    ],
                                    "domains": chain_domains,
                                    "length": len(path),
                                    "avg_weight": np.mean(chain_weights),
                                    "min_weight": min(chain_weights),
                                    "domain_transitions": len(set(chain_domains)),
                                    "chain_strength": np.prod(chain_weights),
                                }
                            )
                except nx.NetworkXNoPath:
                    continue

        # Sort by chain strength and return top chains
        chains.sort(key=lambda x: x["chain_strength"], reverse=True)
        return chains[:25]

    def _get_important_nodes(self) -> List[str]:
        """Get nodes with high importance scores."""
        importance_scores = {}

        # Calculate composite importance score
        try:
            pagerank_scores = nx.pagerank(self.graph)
            betweenness_scores = nx.betweenness_centrality(self.graph)

            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                pagerank = pagerank_scores.get(node, 0)
                betweenness = betweenness_scores.get(node, 0)

                # Composite importance score
                importance = (
                    0.4 * pagerank
                    + 0.4 * betweenness
                    + 0.2 * degree / self.graph.number_of_nodes()
                )
                importance_scores[node] = importance

        except Exception:
            # Fallback to degree centrality
            for node in self.graph.nodes():
                importance_scores[node] = self.graph.degree(node)

        return sorted(
            importance_scores.keys(), key=lambda x: importance_scores[x], reverse=True
        )

    def _identify_concept_clusters(self) -> List[Dict[str, Any]]:
        """Identify tightly connected concept clusters."""
        try:
            # Use Louvain community detection
            import community as community_louvain

            partition = community_louvain.best_partition(self.graph)

            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)

        except ImportError:
            # Fallback to greedy modularity communities
            communities_list = list(
                nx.community.greedy_modularity_communities(
                    self.graph, resolution=self.config.community_resolution
                )
            )
            communities = {i: list(comm) for i, comm in enumerate(communities_list)}

        clusters = []
        for comm_id, nodes in communities.items():
            if len(nodes) < self.config.community_min_size:
                continue

            subgraph = self.graph.subgraph(nodes)

            # Analyze cluster properties
            domains = [self.graph.nodes[n].get("domain", "unknown") for n in nodes]
            domain_dist = {d: domains.count(d) for d in set(domains)}

            # Calculate cluster metrics
            density = nx.density(subgraph)
            avg_clustering = nx.average_clustering(subgraph)

            # Find central nodes in cluster
            cluster_centrality = nx.degree_centrality(subgraph)
            central_nodes = sorted(
                cluster_centrality.items(), key=lambda x: x[1], reverse=True
            )[:3]

            clusters.append(
                {
                    "cluster_id": comm_id,
                    "size": len(nodes),
                    "nodes": nodes[:10],  # Sample for large clusters
                    "node_names": [
                        self.graph.nodes[n].get("name", n) for n in nodes[:10]
                    ],
                    "density": density,
                    "avg_clustering": avg_clustering,
                    "domain_distribution": domain_dist,
                    "primary_domain": max(domain_dist.keys(), key=domain_dist.get),
                    "domain_diversity": len(domain_dist),
                    "cohesion_score": density * len(nodes),
                    "central_nodes": [
                        {
                            "node": node,
                            "name": self.graph.nodes[node].get("name", node),
                            "centrality": score,
                        }
                        for node, score in central_nodes
                    ],
                }
            )

        clusters.sort(key=lambda x: x["cohesion_score"], reverse=True)
        return clusters[:20]

    async def _detect_temporal_patterns(self, session) -> List[Dict[str, Any]]:
        """Detect temporal patterns in concept evolution."""
        query = """
        MATCH (c:Concept)
        WHERE c.time_period IS NOT NULL
        OPTIONAL MATCH (c)-[r:RELATES_TO]-(other:Concept)
        WHERE other.time_period IS NOT NULL
        RETURN c.id as concept, c.time_period as period, 
               collect(distinct other.time_period) as related_periods,
               c.domain as domain
        ORDER BY c.time_period
        """

        result = await session.run(query)
        temporal_data = []

        async for record in result:
            temporal_data.append(
                {
                    "concept": record["concept"],
                    "period": record["period"],
                    "domain": record["domain"],
                    "related_periods": record["related_periods"],
                }
            )

        # Analyze temporal patterns
        patterns = []

        # Group by time period
        period_groups = defaultdict(list)
        for item in temporal_data:
            period_groups[item["period"]].append(item)

        # Find significant periods and their characteristics
        for period, concepts in period_groups.items():
            if len(concepts) >= 3:  # Significant period
                domains = [c["domain"] for c in concepts]
                domain_counts = {d: domains.count(d) for d in set(domains)}

                # Calculate cross-period connections
                cross_connections = 0
                for concept in concepts:
                    cross_connections += len(concept["related_periods"])

                patterns.append(
                    {
                        "period": period,
                        "concept_count": len(concepts),
                        "domain_distribution": domain_counts,
                        "dominant_domain": max(
                            domain_counts.keys(), key=domain_counts.get
                        ),
                        "cross_period_connections": cross_connections,
                        "avg_connections_per_concept": cross_connections
                        / len(concepts),
                        "sample_concepts": [c["concept"] for c in concepts[:5]],
                    }
                )

        patterns.sort(key=lambda x: x["concept_count"], reverse=True)
        return patterns[:15]

    def _detect_structural_motifs(self) -> List[Dict[str, Any]]:
        """Detect structural motifs in the graph."""
        motifs = []

        # Star motifs (nodes with high degree)
        degrees = dict(self.graph.degree())
        avg_degree = np.mean(list(degrees.values()))

        star_nodes = [
            node for node, degree in degrees.items() if degree >= 3 * avg_degree
        ]

        for node in star_nodes:
            neighbors = list(self.graph.neighbors(node))
            node_domain = self.graph.nodes[node].get("domain", "unknown")

            motifs.append(
                {
                    "type": "star",
                    "center_node": node,
                    "center_name": self.graph.nodes[node].get("name", node),
                    "center_domain": node_domain,
                    "degree": len(neighbors),
                    "connected_domains": len(
                        set(
                            self.graph.nodes[n].get("domain", "unknown")
                            for n in neighbors
                        )
                    ),
                }
            )

        return motifs

    def _detect_hub_patterns(self) -> List[Dict[str, Any]]:
        """Detect hub patterns in the network."""
        hubs = []

        # Calculate various centrality measures
        try:
            pagerank = nx.pagerank(self.graph)
            betweenness = nx.betweenness_centrality(self.graph)
            degree_centrality = nx.degree_centrality(self.graph)

            # Identify nodes that are hubs in multiple measures
            for node in self.graph.nodes():
                pr_score = pagerank.get(node, 0)
                bc_score = betweenness.get(node, 0)
                dc_score = degree_centrality.get(node, 0)

                # Composite hub score
                hub_score = (pr_score + bc_score + dc_score) / 3

                if hub_score > 0.1:  # Threshold for hub classification
                    hubs.append(
                        {
                            "node": node,
                            "name": self.graph.nodes[node].get("name", node),
                            "domain": self.graph.nodes[node].get("domain", "unknown"),
                            "hub_score": hub_score,
                            "pagerank": pr_score,
                            "betweenness": bc_score,
                            "degree_centrality": dc_score,
                            "degree": self.graph.degree(node),
                        }
                    )
        except Exception:
            # Fallback to degree-based hubs
            degrees = dict(self.graph.degree())
            threshold = np.percentile(list(degrees.values()), 90)

            for node, degree in degrees.items():
                if degree >= threshold:
                    hubs.append(
                        {
                            "node": node,
                            "name": self.graph.nodes[node].get("name", node),
                            "domain": self.graph.nodes[node].get("domain", "unknown"),
                            "hub_score": degree / self.graph.number_of_nodes(),
                            "degree": degree,
                        }
                    )

        hubs.sort(key=lambda x: x["hub_score"], reverse=True)
        return hubs[:15]

    def _detect_bridge_patterns(self) -> List[Dict[str, Any]]:
        """Detect bridge patterns that connect different parts of the network."""
        bridges = []

        # Find articulation points (nodes whose removal increases components)
        try:
            articulation_points = list(nx.articulation_points(self.graph))

            for node in articulation_points:
                # Analyze the bridging role
                neighbors = list(self.graph.neighbors(node))
                neighbor_domains = [
                    self.graph.nodes[n].get("domain", "unknown") for n in neighbors
                ]

                bridges.append(
                    {
                        "node": node,
                        "name": self.graph.nodes[node].get("name", node),
                        "domain": self.graph.nodes[node].get("domain", "unknown"),
                        "type": "articulation_point",
                        "bridge_domains": list(set(neighbor_domains)),
                        "bridge_strength": len(set(neighbor_domains)),
                        "degree": len(neighbors),
                    }
                )
        except Exception:
            pass

        # Find bridges (edges whose removal increases components)
        try:
            bridge_edges = list(nx.bridges(self.graph))

            for u, v in bridge_edges:
                u_domain = self.graph.nodes[u].get("domain", "unknown")
                v_domain = self.graph.nodes[v].get("domain", "unknown")

                if u_domain != v_domain:  # Cross-domain bridge
                    bridges.append(
                        {
                            "edge": (u, v),
                            "nodes": [u, v],
                            "names": [
                                self.graph.nodes[u].get("name", u),
                                self.graph.nodes[v].get("name", v),
                            ],
                            "domains": [u_domain, v_domain],
                            "type": "bridge_edge",
                            "weight": self.graph[u][v].get("weight", 1.0),
                        }
                    )
        except Exception:
            pass

        return bridges
