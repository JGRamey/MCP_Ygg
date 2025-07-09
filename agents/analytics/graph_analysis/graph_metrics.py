"""Graph metrics calculation module."""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import networkx as nx
from datetime import datetime
import time

from .base import BaseAnalyzer, AnalysisConfig, AnalysisResult, CentralityMeasure, GraphMetrics as BaseGraphMetrics
from cache import cache


class GraphMetricsAnalyzer(BaseAnalyzer):
    """Calculate various graph metrics."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.metrics_cache = {}
    
    async def analyze(self, session) -> AnalysisResult:
        """Calculate comprehensive graph metrics."""
        start_time = time.time()
        
        try:
            await self.load_graph(session)
            
            if not self._validate_graph():
                return AnalysisResult(
                    analysis_type="graph_metrics",
                    timestamp=datetime.now().isoformat(),
                    node_count=0,
                    edge_count=0,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Graph validation failed"
                )
            
            # Calculate all metrics
            metrics = {
                "basic_stats": self._calculate_basic_stats(),
                "centrality_measures": await self._calculate_centrality_measures(),
                "connectivity": self._analyze_connectivity(),
                "degree_distribution": self._calculate_degree_distribution(),
                "clustering": self._analyze_clustering(),
                "efficiency": self._calculate_efficiency_metrics(),
                "robustness": self._analyze_robustness()
            }
            
            return AnalysisResult(
                analysis_type="graph_metrics",
                timestamp=datetime.now().isoformat(),
                node_count=self.graph.number_of_nodes(),
                edge_count=self.graph.number_of_edges(),
                execution_time=time.time() - start_time,
                success=True,
                data=metrics
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type="graph_metrics",
                timestamp=datetime.now().isoformat(),
                node_count=0,
                edge_count=0,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        basic_metrics = self._calculate_basic_metrics()
        
        # Add additional statistics
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        basic_metrics.update({
            "max_degree": max(degree_values) if degree_values else 0,
            "min_degree": min(degree_values) if degree_values else 0,
            "std_degree": np.std(degree_values) if degree_values else 0,
            "median_degree": np.median(degree_values) if degree_values else 0,
            "diameter": self._calculate_diameter(),
            "radius": self._calculate_radius(),
            "center": self._calculate_center(),
            "periphery": self._calculate_periphery()
        })
        
        return basic_metrics
    
    async def _calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures."""
        centrality_results = {}
        
        # Calculate each centrality measure
        for measure in CentralityMeasure:
            try:
                centrality = BaseGraphMetrics.calculate_centrality(
                    self.graph, 
                    measure, 
                    self.config.centrality_top_k
                )
                centrality_results[measure.value] = centrality
            except Exception as e:
                centrality_results[measure.value] = {"error": str(e)}
        
        return centrality_results
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity."""
        connectivity_metrics = {}
        
        # Basic connectivity
        connectivity_metrics["is_connected"] = nx.is_connected(self.graph)
        
        # Connected components
        components = list(nx.connected_components(self.graph))
        connectivity_metrics["num_components"] = len(components)
        
        if components:
            component_sizes = [len(comp) for comp in components]
            connectivity_metrics["largest_component_size"] = max(component_sizes)
            connectivity_metrics["smallest_component_size"] = min(component_sizes)
            connectivity_metrics["avg_component_size"] = np.mean(component_sizes)
        
        # Connectivity measures
        if nx.is_connected(self.graph):
            connectivity_metrics["average_shortest_path"] = nx.average_shortest_path_length(self.graph)
            connectivity_metrics["global_efficiency"] = nx.global_efficiency(self.graph)
            connectivity_metrics["wiener_index"] = nx.wiener_index(self.graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_component = max(components, key=len)
            subgraph = self.graph.subgraph(largest_component)
            connectivity_metrics["average_shortest_path"] = nx.average_shortest_path_length(subgraph)
            connectivity_metrics["global_efficiency"] = nx.global_efficiency(subgraph)
        
        # Node connectivity
        connectivity_metrics["node_connectivity"] = nx.node_connectivity(self.graph)
        connectivity_metrics["edge_connectivity"] = nx.edge_connectivity(self.graph)
        
        return connectivity_metrics
    
    def _calculate_degree_distribution(self) -> Dict[str, Any]:
        """Calculate degree distribution."""
        degrees = [d for n, d in self.graph.degree()]
        degree_counts = defaultdict(int)
        
        for d in degrees:
            degree_counts[d] += 1
        
        # Statistical measures
        distribution = {
            "degrees": list(degree_counts.keys()),
            "counts": list(degree_counts.values()),
            "mean": np.mean(degrees),
            "std": np.std(degrees),
            "median": np.median(degrees),
            "max": max(degrees) if degrees else 0,
            "min": min(degrees) if degrees else 0,
            "entropy": self._calculate_degree_entropy(degree_counts)
        }
        
        # Power law test (basic)
        distribution["power_law_exponent"] = self._estimate_power_law_exponent(degrees)
        
        return distribution
    
    def _calculate_degree_entropy(self, degree_counts: Dict[int, int]) -> float:
        """Calculate entropy of degree distribution."""
        total = sum(degree_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in degree_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _estimate_power_law_exponent(self, degrees: List[int]) -> float:
        """Estimate power law exponent using simple method."""
        if not degrees or min(degrees) <= 0:
            return 0.0
        
        # Simple estimation: gamma = 1 + n * sum(ln(xi/xmin))^(-1)
        x_min = min(degrees)
        if x_min <= 0:
            return 0.0
        
        log_ratios = [np.log(d / x_min) for d in degrees if d >= x_min]
        if not log_ratios:
            return 0.0
        
        return 1 + len(log_ratios) / sum(log_ratios)
    
    def _analyze_clustering(self) -> Dict[str, Any]:
        """Analyze clustering in the graph."""
        clustering_metrics = {}
        
        # Global clustering
        clustering_metrics["average_clustering"] = nx.average_clustering(self.graph)
        clustering_metrics["transitivity"] = nx.transitivity(self.graph)
        
        # Local clustering
        local_clustering = nx.clustering(self.graph)
        clustering_values = list(local_clustering.values())
        
        clustering_metrics["local_clustering"] = {
            "mean": np.mean(clustering_values),
            "std": np.std(clustering_values),
            "median": np.median(clustering_values),
            "max": max(clustering_values) if clustering_values else 0,
            "min": min(clustering_values) if clustering_values else 0
        }
        
        # Top clustered nodes
        top_clustered = sorted(local_clustering.items(), key=lambda x: x[1], reverse=True)[:10]
        clustering_metrics["top_clustered_nodes"] = [
            {"node": node, "clustering": score} for node, score in top_clustered
        ]
        
        return clustering_metrics
    
    def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        efficiency_metrics = {}
        
        # Global efficiency
        efficiency_metrics["global_efficiency"] = nx.global_efficiency(self.graph)
        
        # Local efficiency
        efficiency_metrics["local_efficiency"] = nx.local_efficiency(self.graph)
        
        # Economic efficiency (if graph is connected)
        if nx.is_connected(self.graph):
            # Calculate cost-efficiency trade-off
            num_edges = self.graph.number_of_edges()
            max_edges = self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1) // 2
            cost = num_edges / max_edges if max_edges > 0 else 0
            efficiency_metrics["cost_efficiency"] = efficiency_metrics["global_efficiency"] / cost if cost > 0 else 0
        
        return efficiency_metrics
    
    def _analyze_robustness(self) -> Dict[str, Any]:
        """Analyze network robustness."""
        robustness_metrics = {}
        
        # Algebraic connectivity (Fiedler value)
        try:
            robustness_metrics["algebraic_connectivity"] = nx.algebraic_connectivity(self.graph)
        except:
            robustness_metrics["algebraic_connectivity"] = 0.0
        
        # Assortativity
        try:
            robustness_metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(self.graph)
        except:
            robustness_metrics["degree_assortativity"] = 0.0
        
        # Critical nodes (high betweenness centrality)
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            critical_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            robustness_metrics["critical_nodes"] = [
                {"node": node, "betweenness": score} for node, score in critical_nodes
            ]
        except:
            robustness_metrics["critical_nodes"] = []
        
        return robustness_metrics
    
    def _calculate_diameter(self) -> int:
        """Calculate graph diameter."""
        if not nx.is_connected(self.graph):
            return float('inf')
        
        try:
            return nx.diameter(self.graph)
        except:
            return 0
    
    def _calculate_radius(self) -> int:
        """Calculate graph radius."""
        if not nx.is_connected(self.graph):
            return float('inf')
        
        try:
            return nx.radius(self.graph)
        except:
            return 0
    
    def _calculate_center(self) -> List[str]:
        """Calculate graph center."""
        if not nx.is_connected(self.graph):
            return []
        
        try:
            return list(nx.center(self.graph))
        except:
            return []
    
    def _calculate_periphery(self) -> List[str]:
        """Calculate graph periphery."""
        if not nx.is_connected(self.graph):
            return []
        
        try:
            return list(nx.periphery(self.graph))
        except:
            return []
    
    async def get_node_importance(self, node_id: str) -> Dict[str, Any]:
        """Get importance metrics for a specific node."""
        if node_id not in self.graph:
            return {"error": "Node not found"}
        
        node_attrs = await self.get_node_attributes(node_id)
        
        # Calculate various centrality measures for this node
        centrality_metrics = {}
        
        try:
            pagerank = nx.pagerank(self.graph)
            centrality_metrics["pagerank"] = pagerank.get(node_id, 0)
        except:
            centrality_metrics["pagerank"] = 0
        
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            centrality_metrics["betweenness"] = betweenness.get(node_id, 0)
        except:
            centrality_metrics["betweenness"] = 0
        
        try:
            closeness = nx.closeness_centrality(self.graph)
            centrality_metrics["closeness"] = closeness.get(node_id, 0)
        except:
            centrality_metrics["closeness"] = 0
        
        # Node degree and clustering
        degree = self.graph.degree(node_id)
        clustering_coeff = nx.clustering(self.graph, node_id)
        
        # Connected domains
        neighbors = list(self.graph.neighbors(node_id))
        neighbor_domains = []
        for neighbor in neighbors:
            neighbor_attrs = await self.get_node_attributes(neighbor)
            domain = neighbor_attrs.get("domain", "unknown")
            if domain not in neighbor_domains:
                neighbor_domains.append(domain)
        
        return {
            "node_id": node_id,
            "name": node_attrs.get("name", "Unknown"),
            "domain": node_attrs.get("domain", "Unknown"),
            "degree": degree,
            "clustering": clustering_coeff,
            "connected_domains": neighbor_domains,
            "cross_domain_connections": len(neighbor_domains) - 1 if len(neighbor_domains) > 1 else 0,
            "centrality_metrics": centrality_metrics,
            "importance_score": self._calculate_importance_score(centrality_metrics, degree)
        }
    
    def _calculate_importance_score(self, centrality_metrics: Dict[str, float], degree: int) -> float:
        """Calculate composite importance score."""
        weights = {
            "pagerank": 0.3,
            "betweenness": 0.3,
            "closeness": 0.2,
            "degree": 0.2
        }
        
        # Normalize degree (simple normalization)
        normalized_degree = degree / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        
        score = (
            weights["pagerank"] * centrality_metrics.get("pagerank", 0) +
            weights["betweenness"] * centrality_metrics.get("betweenness", 0) +
            weights["closeness"] * centrality_metrics.get("closeness", 0) +
            weights["degree"] * normalized_degree
        )
        
        return score