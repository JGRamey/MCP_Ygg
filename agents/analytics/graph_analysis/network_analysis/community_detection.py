"""
Community Detection Module - Focused community detection and analysis.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import Counter

import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman

from ..models import AnalysisType, NetworkAnalysisResult, NodeMetrics, CommunityInfo
from ..config import NetworkConfig
from ..graph_utils import GraphMetricsAggregator


class CommunityDetector:
    """Specialized analyzer for community detection."""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.metrics_aggregator = GraphMetricsAggregator(config)
        self.logger = logging.getLogger("community_detector")
    
    async def analyze(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Perform comprehensive community detection analysis."""
        
        self.logger.info("Detecting communities...")
        
        # Detect communities using multiple algorithms
        communities_results = self._detect_communities_multi_algorithm(graph)
        
        # Select best community detection result
        best_algorithm, communities_info = self._select_best_communities(
            graph, communities_results
        )
        
        # Update node metrics with community assignments
        node_metrics = self._create_node_metrics_with_communities(graph, communities_info)
        
        # Calculate graph metrics
        graph_metrics = self._calculate_community_graph_metrics(
            graph, communities_info, best_algorithm
        )
        
        # Generate insights
        insights = self._generate_community_insights(graph, communities_info, graph_metrics)
        
        # Generate recommendations
        recommendations = self._generate_community_recommendations(communities_info, graph_metrics)
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.COMMUNITY_DETECTION,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=communities_info,
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    def _detect_communities_multi_algorithm(self, graph: nx.Graph) -> Dict[str, any]:
        """Detect communities using multiple algorithms."""
        
        results = {}
        
        # Greedy modularity optimization
        try:
            communities = greedy_modularity_communities(
                graph, 
                resolution=self.config.resolution_parameter
            )
            communities_list = [list(community) for community in communities]
            modularity = community.modularity(graph, communities)
            
            results['greedy_modularity'] = {
                'communities': communities_list,
                'modularity': modularity,
                'algorithm': 'greedy_modularity'
            }
        except Exception as e:
            self.logger.warning(f"Greedy modularity failed: {e}")
        
        # Label propagation
        try:
            communities = community.label_propagation_communities(graph)
            communities_list = [list(community) for community in communities]
            modularity = community.modularity(graph, communities)
            
            results['label_propagation'] = {
                'communities': communities_list,
                'modularity': modularity,
                'algorithm': 'label_propagation'
            }
        except Exception as e:
            self.logger.warning(f"Label propagation failed: {e}")
        
        # Louvain method (if available)
        try:
            communities = community.louvain_communities(graph, resolution=self.config.resolution_parameter)
            communities_list = [list(community) for community in communities]
            modularity = community.modularity(graph, communities)
            
            results['louvain'] = {
                'communities': communities_list,
                'modularity': modularity,
                'algorithm': 'louvain'
            }
        except Exception as e:
            self.logger.warning(f"Louvain method failed: {e}")
        
        # Fluid communities (for smaller graphs)
        if len(graph.nodes) <= 1000:
            try:
                k = min(self.config.max_communities, len(graph.nodes) // 10)
                if k > 1:
                    communities = community.asyn_fluidc(graph, k)
                    communities_list = [list(community) for community in communities]
                    modularity = community.modularity(graph, communities)
                    
                    results['fluid'] = {
                        'communities': communities_list,
                        'modularity': modularity,
                        'algorithm': 'fluid'
                    }
            except Exception as e:
                self.logger.warning(f"Fluid communities failed: {e}")
        
        return results
    
    def _select_best_communities(
        self, 
        graph: nx.Graph, 
        results: Dict[str, any]
    ) -> tuple:
        """Select the best community detection result based on modularity."""
        
        if not results:
            return None, []
        
        # Find algorithm with highest modularity
        best_algorithm = None
        best_modularity = -1
        best_result = None
        
        for algorithm, result in results.items():
            if result['modularity'] > best_modularity:
                best_modularity = result['modularity']
                best_algorithm = algorithm
                best_result = result
        
        if best_result is None:
            return None, []
        
        # Convert to CommunityInfo objects
        communities_info = []
        for i, comm_nodes in enumerate(best_result['communities']):
            if len(comm_nodes) >= self.config.min_community_size:
                community_info = self._create_community_info(graph, i, comm_nodes)
                communities_info.append(community_info)
        
        return best_algorithm, communities_info
    
    def _create_community_info(
        self, 
        graph: nx.Graph, 
        community_id: int, 
        nodes: List[str]
    ) -> CommunityInfo:
        """Create CommunityInfo object for a detected community."""
        
        subgraph = graph.subgraph(nodes)
        internal_edges = subgraph.number_of_edges()
        
        # Count external edges
        external_edges = 0
        for node in nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes:
                    external_edges += 1
        external_edges //= 2  # Avoid double counting
        
        # Generate community description
        description = self._generate_community_description(graph, nodes)
        
        # Calculate additional metadata
        metadata = {
            'algorithm': 'multi_algorithm_best',
            'avg_degree': sum(graph.degree(node) for node in nodes) / len(nodes),
            'density': nx.density(subgraph),
            'avg_clustering': nx.average_clustering(subgraph) if len(nodes) > 2 else 0,
            'diameter': nx.diameter(subgraph) if nx.is_connected(subgraph) and len(nodes) > 1 else 0
        }
        
        return CommunityInfo(
            community_id=community_id,
            nodes=nodes,
            size=len(nodes),
            internal_edges=internal_edges,
            external_edges=external_edges,
            modularity_contribution=0.0,  # Would need detailed calculation
            description=description,
            metadata=metadata
        )
    
    def _generate_community_description(self, graph: nx.Graph, community_nodes: List[str]) -> str:
        """Generate a description for a community based on its nodes."""
        
        # Analyze domains in the community
        domains = [graph.nodes[node].get('domain') for node in community_nodes]
        domain_counts = Counter([d for d in domains if d])
        
        if domain_counts:
            dominant_domain = domain_counts.most_common(1)[0][0]
            domain_purity = domain_counts[dominant_domain] / len(community_nodes)
            
            if domain_purity > 0.8:
                description = f"Primarily {dominant_domain} community"
            elif domain_purity > 0.5:
                description = f"Mostly {dominant_domain} community with mixed domains"
            else:
                top_domains = [domain for domain, _ in domain_counts.most_common(2)]
                description = f"Mixed community ({', '.join(top_domains)})"
        else:
            description = "Mixed domain community"
        
        # Add size information
        size_desc = "large" if len(community_nodes) > 20 else "medium" if len(community_nodes) > 10 else "small"
        description = f"{size_desc.capitalize()} {description.lower()}"
        
        # Add connectivity information
        subgraph = graph.subgraph(community_nodes)
        if nx.is_connected(subgraph):
            description += " (well-connected)"
        else:
            description += " (fragmented)"
        
        return description
    
    def _create_node_metrics_with_communities(
        self, 
        graph: nx.Graph, 
        communities_info: List[CommunityInfo]
    ) -> List[NodeMetrics]:
        """Create node metrics with community assignments."""
        
        # Create community assignment mapping
        community_assignment = {}
        for comm_info in communities_info:
            for node in comm_info.nodes:
                community_assignment[node] = comm_info.community_id
        
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={},  # Not calculated in community analysis
                community_id=community_assignment.get(node),
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        return node_metrics
    
    def _calculate_community_graph_metrics(
        self, 
        graph: nx.Graph, 
        communities: List[CommunityInfo], 
        algorithm: str
    ) -> Dict[str, any]:
        """Calculate graph-level metrics for community analysis."""
        
        # Basic community metrics
        metrics = {
            'num_communities': len(communities),
            'algorithm_used': algorithm,
        }
        
        if communities:
            sizes = [c.size for c in communities]
            metrics.update({
                'avg_community_size': np.mean(sizes),
                'median_community_size': np.median(sizes),
                'std_community_size': np.std(sizes),
                'largest_community_size': max(sizes),
                'smallest_community_size': min(sizes),
            })
            
            # Calculate modularity if possible
            try:
                # Reconstruct communities for modularity calculation
                communities_sets = []
                for comm in communities:
                    communities_sets.append(set(comm.nodes))
                modularity = community.modularity(graph, communities_sets)
                metrics['modularity'] = modularity
            except Exception:
                metrics['modularity'] = 0.0
            
            # Community quality metrics
            internal_edges = [c.internal_edges for c in communities]
            external_edges = [c.external_edges for c in communities]
            
            metrics.update({
                'avg_internal_edges': np.mean(internal_edges),
                'avg_external_edges': np.mean(external_edges),
                'total_internal_edges': sum(internal_edges),
                'total_external_edges': sum(external_edges),
            })
            
            # Coverage and performance
            total_edges = graph.number_of_edges()
            if total_edges > 0:
                metrics['coverage'] = sum(internal_edges) / total_edges
                
                # Performance metric (ratio of internal to total edges)
                total_possible_internal = sum(internal_edges) + sum(external_edges)
                if total_possible_internal > 0:
                    metrics['performance'] = sum(internal_edges) / total_possible_internal
        
        # Add basic graph metrics
        metrics.update(self.metrics_aggregator.calculate_comprehensive_metrics(graph))
        
        return metrics
    
    def _generate_community_insights(
        self,
        graph: nx.Graph,
        communities: List[CommunityInfo],
        graph_metrics: Dict[str, any]
    ) -> List[str]:
        """Generate insights from community analysis."""
        
        insights = []
        
        insights.append(f"Detected {len(communities)} communities using {graph_metrics.get('algorithm_used', 'unknown')} algorithm")
        
        if 'modularity' in graph_metrics:
            modularity = graph_metrics['modularity']
            insights.append(f"Modularity score: {modularity:.3f}")
            
            if modularity > 0.3:
                insights.append("High modularity indicates well-defined community structure")
            elif modularity < 0.1:
                insights.append("Low modularity suggests weak community structure")
        
        if communities:
            sizes = [c.size for c in communities]
            insights.append(f"Average community size: {np.mean(sizes):.1f} nodes")
            insights.append(f"Largest community: {max(sizes)} nodes")
            insights.append(f"Smallest community: {min(sizes)} nodes")
            
            # Community quality insights
            avg_internal = graph_metrics.get('avg_internal_edges', 0)
            avg_external = graph_metrics.get('avg_external_edges', 0)
            
            if avg_internal > avg_external:
                insights.append("Communities show strong internal cohesion")
            else:
                insights.append("Communities have high external connectivity")
            
            # Coverage insights
            if 'coverage' in graph_metrics:
                coverage = graph_metrics['coverage']
                insights.append(f"Community coverage: {coverage:.1%} of all edges are within communities")
            
            # Domain analysis
            domain_analysis = self._analyze_community_domains(graph, communities)
            if domain_analysis:
                insights.extend(domain_analysis)
        
        return insights
    
    def _analyze_community_domains(
        self, 
        graph: nx.Graph, 
        communities: List[CommunityInfo]
    ) -> List[str]:
        """Analyze domain distribution within communities."""
        
        insights = []
        
        domain_communities = {}
        mixed_communities = 0
        
        for comm in communities:
            domains = [graph.nodes[node].get('domain') for node in comm.nodes]
            domain_counts = Counter([d for d in domains if d])
            
            if domain_counts:
                dominant_domain = domain_counts.most_common(1)[0][0]
                domain_purity = domain_counts[dominant_domain] / len(comm.nodes)
                
                if domain_purity > 0.8:
                    if dominant_domain not in domain_communities:
                        domain_communities[dominant_domain] = 0
                    domain_communities[dominant_domain] += 1
                else:
                    mixed_communities += 1
        
        if domain_communities:
            insights.append("Domain-specific communities detected:")
            for domain, count in domain_communities.items():
                insights.append(f"  • {count} communities primarily in {domain}")
        
        if mixed_communities > 0:
            insights.append(f"{mixed_communities} communities span multiple domains")
        
        return insights
    
    def _generate_community_recommendations(
        self,
        communities: List[CommunityInfo],
        graph_metrics: Dict[str, any]
    ) -> List[str]:
        """Generate recommendations from community analysis."""
        
        recommendations = [
            "Use community structure for targeted interventions",
            "Monitor community evolution over time",
            "Consider inter-community bridges for integration strategies"
        ]
        
        modularity = graph_metrics.get('modularity', 0)
        if modularity > 0.3:
            recommendations.append("High modularity suggests opportunities for specialized approaches per community")
        elif modularity < 0.1:
            recommendations.append("Low modularity suggests focusing on global rather than community-specific strategies")
        
        if communities:
            # Size-based recommendations
            sizes = [c.size for c in communities]
            size_variance = np.var(sizes)
            
            if size_variance > np.mean(sizes) ** 2:
                recommendations.append("High community size variance suggests different management strategies needed")
            
            # Quality-based recommendations
            if 'performance' in graph_metrics:
                performance = graph_metrics['performance']
                if performance > 0.7:
                    recommendations.append("High community performance enables localized interventions")
                else:
                    recommendations.append("Low community performance requires cross-community coordination")
        
        return recommendations