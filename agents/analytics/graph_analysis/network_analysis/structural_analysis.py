"""
Structural Analysis Module - Overall network structure analysis.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import Counter

import networkx as nx

from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics
from ..config import NetworkConfig
from ..graph_utils import GraphMetricsAggregator, ConnectivityAnalyzer, GraphStatistics


class StructuralAnalyzer:
    """Specialized analyzer for overall network structure."""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.connectivity_analyzer = ConnectivityAnalyzer()
        self.logger = logging.getLogger("structural_analyzer")
    
    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze overall network structure."""
        
        self.logger.info("Analyzing network structure...")
        
        # Calculate basic structural metrics
        basic_metrics = self._calculate_basic_structural_metrics(graph)
        
        # Analyze connectivity properties
        connectivity_metrics = self.connectivity_analyzer.analyze_connectivity(graph)
        
        # Calculate small-world properties
        small_world_metrics = self._calculate_small_world_metrics(graph)
        
        # Analyze degree distribution
        degree_metrics = self._analyze_degree_distribution(graph)
        
        # Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(graph)
        
        # Create node metrics
        node_metrics = self._create_structural_node_metrics(
            graph, basic_metrics, degree_metrics
        )
        
        # Combine all metrics
        graph_metrics = self._combine_structural_metrics(
            basic_metrics, connectivity_metrics, small_world_metrics,
            degree_metrics, robustness_metrics
        )
        
        # Generate insights
        insights = self._generate_structural_insights(
            graph, graph_metrics, connectivity_metrics, small_world_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_structural_recommendations(
            graph_metrics, connectivity_metrics, small_world_metrics
        )
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.STRUCTURAL_ANALYSIS,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    def _calculate_basic_structural_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate basic structural metrics."""
        
        # Use the centralized statistics calculator
        basic_stats = GraphStatistics.calculate_basic_stats(graph)
        
        # Add additional structural metrics
        additional_metrics = {}
        
        # Clustering metrics
        try:
            additional_metrics['avg_clustering'] = nx.average_clustering(graph)
            additional_metrics['transitivity'] = nx.transitivity(graph)
        except Exception as e:
            self.logger.warning(f"Clustering calculation failed: {e}")
            additional_metrics['avg_clustering'] = 0.0
            additional_metrics['transitivity'] = 0.0
        
        # Combine basic stats with additional metrics
        basic_stats.update(additional_metrics)
        
        return basic_stats
    
    def _calculate_small_world_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate small-world properties."""
        
        metrics = {}
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes < 3:
            return {
                'small_world_coeff': 0,
                'clustering_ratio': 0,
                'path_length_ratio': 0,
                'is_small_world': False
            }
        
        try:
            # Calculate actual clustering and path length
            actual_clustering = nx.average_clustering(graph)
            
            if nx.is_connected(graph):
                actual_path_length = nx.average_shortest_path_length(graph)
            else:
                # Use largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                if len(largest_cc) > 1:
                    subgraph = graph.subgraph(largest_cc)
                    actual_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    actual_path_length = 0
            
            # Compare to random graph with same nodes and edges
            if num_edges > 0:
                density = nx.density(graph)
                random_graph = nx.erdos_renyi_graph(num_nodes, density)
                
                random_clustering = nx.average_clustering(random_graph)
                if nx.is_connected(random_graph):
                    random_path_length = nx.average_shortest_path_length(random_graph)
                else:
                    random_path_length = actual_path_length  # Fallback
                
                # Calculate ratios
                clustering_ratio = actual_clustering / random_clustering if random_clustering > 0 else 0
                path_length_ratio = actual_path_length / random_path_length if random_path_length > 0 else 0
                
                # Small world coefficient
                small_world_coeff = clustering_ratio / path_length_ratio if path_length_ratio > 0 else 0
                
                # Determine if it's a small world network
                is_small_world = clustering_ratio > 1 and path_length_ratio <= 2
                
                metrics = {
                    'small_world_coeff': small_world_coeff,
                    'clustering_ratio': clustering_ratio,
                    'path_length_ratio': path_length_ratio,
                    'is_small_world': is_small_world,
                    'actual_clustering': actual_clustering,
                    'actual_path_length': actual_path_length,
                    'random_clustering': random_clustering,
                    'random_path_length': random_path_length
                }
            else:
                metrics = {
                    'small_world_coeff': 0,
                    'clustering_ratio': 0,
                    'path_length_ratio': 0,
                    'is_small_world': False
                }
                
        except Exception as e:
            self.logger.warning(f"Small world calculation failed: {e}")
            metrics = {
                'small_world_coeff': 0,
                'clustering_ratio': 0,
                'path_length_ratio': 0,
                'is_small_world': False
            }
        
        return metrics
    
    def _analyze_degree_distribution(self, graph: nx.Graph) -> Dict[str, any]:
        """Analyze degree distribution and related properties."""
        
        degrees = [d for n, d in graph.degree()]
        
        if not degrees:
            return {}
        
        degree_dist = Counter(degrees)
        unique_degrees = sorted(degree_dist.keys())
        
        metrics = {
            'degree_sequence': degrees,
            'degree_distribution': dict(degree_dist),
            'unique_degrees': unique_degrees,
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'avg_degree': np.mean(degrees),
            'median_degree': np.median(degrees),
            'degree_variance': np.var(degrees),
            'degree_std': np.std(degrees)
        }
        
        # Degree heterogeneity
        if np.mean(degrees) > 0:
            metrics['degree_heterogeneity'] = np.std(degrees) / np.mean(degrees)
        else:
            metrics['degree_heterogeneity'] = 0
        
        # Scale-free analysis (power law fitting)
        try:
            # Simple power law check: look at degree distribution on log-log scale
            degree_counts = list(degree_dist.values())
            degrees_sorted = sorted(degree_dist.keys(), reverse=True)
            
            if len(degrees_sorted) > 3 and max(degrees_sorted) > min(degrees_sorted):
                # Calculate correlation coefficient for log-log plot
                log_degrees = np.log(degrees_sorted)
                log_counts = np.log([degree_dist[d] for d in degrees_sorted])
                
                correlation = np.corrcoef(log_degrees, log_counts)[0, 1]
                metrics['power_law_correlation'] = correlation
                metrics['is_scale_free'] = abs(correlation) > 0.8  # Strong negative correlation
            else:
                metrics['power_law_correlation'] = 0
                metrics['is_scale_free'] = False
                
        except Exception as e:
            self.logger.warning(f"Power law analysis failed: {e}")
            metrics['power_law_correlation'] = 0
            metrics['is_scale_free'] = False
        
        return metrics
    
    def _calculate_robustness_metrics(self, graph: nx.Graph) -> Dict[str, any]:
        """Calculate network robustness metrics."""
        
        metrics = {}
        
        # Connectivity robustness
        metrics['node_connectivity'] = nx.node_connectivity(graph)
        metrics['edge_connectivity'] = nx.edge_connectivity(graph)
        
        # Algebraic connectivity (second smallest eigenvalue of Laplacian)
        try:
            laplacian_eigenvalues = nx.laplacian_spectrum(graph)
            if len(laplacian_eigenvalues) > 1:
                metrics['algebraic_connectivity'] = float(laplacian_eigenvalues[1])
            else:
                metrics['algebraic_connectivity'] = 0.0
        except Exception as e:
            self.logger.warning(f"Algebraic connectivity calculation failed: {e}")
            metrics['algebraic_connectivity'] = 0.0
        
        # Robustness to random failures and targeted attacks
        if len(graph.nodes) > 10:  # Only for larger graphs
            try:
                # Sample robustness analysis
                original_components = nx.number_connected_components(graph)
                
                # Random node removal robustness
                num_nodes = len(graph.nodes)
                sample_size = min(10, num_nodes // 10)
                
                if sample_size > 0:
                    import random
                    test_nodes = random.sample(list(graph.nodes), sample_size)
                    test_graph = graph.copy()
                    test_graph.remove_nodes_from(test_nodes)
                    
                    new_components = nx.number_connected_components(test_graph)
                    metrics['random_failure_impact'] = new_components - original_components
                
                # Targeted attack robustness (remove highest degree nodes)
                degree_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
                high_degree_nodes = [node for node, degree in degree_sorted[:sample_size]]
                
                test_graph = graph.copy()
                test_graph.remove_nodes_from(high_degree_nodes)
                new_components = nx.number_connected_components(test_graph)
                metrics['targeted_attack_impact'] = new_components - original_components
                
            except Exception as e:
                self.logger.warning(f"Robustness analysis failed: {e}")
                metrics['random_failure_impact'] = 0
                metrics['targeted_attack_impact'] = 0
        
        return metrics
    
    def _create_structural_node_metrics(
        self,
        graph: nx.Graph,
        basic_metrics: Dict[str, any],
        degree_metrics: Dict[str, any]
    ) -> List[NodeMetrics]:
        """Create node metrics for structural analysis."""
        
        node_metrics = []
        clustering_dict = nx.clustering(graph)
        
        for node in graph.nodes:
            node_data = graph.nodes[node]
            degree = graph.degree(node)
            
            # Calculate normalized degree
            num_nodes = graph.number_of_nodes()
            normalized_degree = degree / (num_nodes - 1) if num_nodes > 1 else 0
            
            centrality_scores = {
                'degree': degree,
                'normalized_degree': normalized_degree,
                'degree_rank': 0  # Will be calculated below
            }
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores=centrality_scores,
                community_id=None,
                clustering_coefficient=clustering_dict.get(node, 0),
                degree=degree,
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate degree ranks
        sorted_by_degree = sorted(node_metrics, key=lambda x: x.degree, reverse=True)
        for rank, node_metric in enumerate(sorted_by_degree):
            node_metric.centrality_scores['degree_rank'] = rank + 1
        
        return node_metrics
    
    def _combine_structural_metrics(
        self,
        basic_metrics: Dict[str, any],
        connectivity_metrics: Dict[str, any],
        small_world_metrics: Dict[str, any],
        degree_metrics: Dict[str, any],
        robustness_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Combine all structural metrics."""
        
        combined_metrics = {}
        
        # Add all metric dictionaries
        combined_metrics.update(basic_metrics)
        combined_metrics.update(connectivity_metrics)
        combined_metrics.update(small_world_metrics)
        combined_metrics.update(degree_metrics)
        combined_metrics.update(robustness_metrics)
        
        return combined_metrics
    
    def _generate_structural_insights(
        self,
        graph: nx.Graph,
        graph_metrics: Dict[str, any],
        connectivity_metrics: Dict[str, any],
        small_world_metrics: Dict[str, any]
    ) -> List[str]:
        """Generate insights from structural analysis."""
        
        insights = []
        
        # Basic structure insights
        num_nodes = graph_metrics.get('num_nodes', 0)
        num_edges = graph_metrics.get('num_edges', 0)
        density = graph_metrics.get('density', 0)
        avg_degree = graph_metrics.get('avg_degree', 0)
        
        insights.append(f"Network has {num_nodes} nodes and {num_edges} edges")
        insights.append(f"Network density: {density:.4f}")
        insights.append(f"Average degree: {avg_degree:.2f}")
        
        # Clustering insights
        avg_clustering = graph_metrics.get('avg_clustering', 0)
        transitivity = graph_metrics.get('transitivity', 0)
        
        insights.append(f"Average clustering coefficient: {avg_clustering:.4f}")
        insights.append(f"Network transitivity: {transitivity:.4f}")
        
        # Connectivity insights
        is_connected = connectivity_metrics.get('is_connected', False)
        num_components = connectivity_metrics.get('num_components', 0)
        diameter = connectivity_metrics.get('diameter', 0)
        avg_path_length = connectivity_metrics.get('avg_path_length', 0)
        
        if is_connected:
            insights.append(f"Network is connected with diameter {diameter}")
            insights.append(f"Average shortest path length: {avg_path_length:.2f}")
        else:
            insights.append(f"Network has {num_components} connected components")
        
        # Small world insights
        small_world_coeff = small_world_metrics.get('small_world_coeff', 0)
        is_small_world = small_world_metrics.get('is_small_world', False)
        
        if is_small_world:
            insights.append(f"Network exhibits small-world properties (coefficient: {small_world_coeff:.2f})")
        elif small_world_coeff > 1:
            insights.append(f"Network shows small-world tendencies (coefficient: {small_world_coeff:.2f})")
        
        # Assortativity insights
        assortativity = graph_metrics.get('assortativity', 0)
        if assortativity > 0.1:
            insights.append("Network shows assortative mixing (similar nodes connect)")
        elif assortativity < -0.1:
            insights.append("Network shows disassortative mixing (dissimilar nodes connect)")
        else:
            insights.append("Network shows neutral mixing patterns")
        
        # Degree distribution insights
        degree_heterogeneity = graph_metrics.get('degree_heterogeneity', 0)
        is_scale_free = graph_metrics.get('is_scale_free', False)
        
        if is_scale_free:
            insights.append("Degree distribution suggests scale-free network properties")
        
        if degree_heterogeneity > 1.0:
            insights.append("High degree heterogeneity indicates presence of hubs")
        elif degree_heterogeneity < 0.5:
            insights.append("Low degree heterogeneity suggests relatively uniform connectivity")
        
        # Robustness insights
        algebraic_connectivity = graph_metrics.get('algebraic_connectivity', 0)
        if algebraic_connectivity > 0.1:
            insights.append("High algebraic connectivity indicates robust network structure")
        elif algebraic_connectivity < 0.01:
            insights.append("Low algebraic connectivity suggests structural vulnerability")
        
        return insights
    
    def _generate_structural_recommendations(
        self,
        graph_metrics: Dict[str, any],
        connectivity_metrics: Dict[str, any],
        small_world_metrics: Dict[str, any]
    ) -> List[str]:
        """Generate recommendations from structural analysis."""
        
        recommendations = [
            "Monitor network density changes to track growth patterns",
            "Use clustering metrics to identify cohesive subgroups",
            "Consider path length for information diffusion strategies"
        ]
        
        # Connectivity recommendations
        is_connected = connectivity_metrics.get('is_connected', False)
        if not is_connected:
            num_components = connectivity_metrics.get('num_components', 0)
            recommendations.append(f"Bridge {num_components} components to improve connectivity")
        
        # Small world recommendations
        is_small_world = small_world_metrics.get('is_small_world', False)
        if is_small_world:
            recommendations.append("Leverage small-world properties for efficient information spread")
        else:
            clustering_ratio = small_world_metrics.get('clustering_ratio', 0)
            if clustering_ratio < 1:
                recommendations.append("Consider increasing local clustering to improve information retention")
        
        # Density recommendations
        density = graph_metrics.get('density', 0)
        if density < 0.05:
            recommendations.append("Low density suggests opportunities for strategic link formation")
        elif density > 0.5:
            recommendations.append("High density enables rapid information diffusion")
        
        # Assortativity recommendations
        assortativity = graph_metrics.get('assortativity', 0)
        recommendations.append("Leverage assortativity patterns for targeted interventions")
        
        # Robustness recommendations
        algebraic_connectivity = graph_metrics.get('algebraic_connectivity', 0)
        if algebraic_connectivity < 0.01:
            recommendations.append("Improve network robustness by strengthening weak connections")
        
        return recommendations