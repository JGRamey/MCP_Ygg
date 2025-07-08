# MCP Yggdrasil - Code Refactoring Examples
# Breaking down large monolithic files into maintainable modules

# ============================================================================
# REFACTORING analytics/network_analyzer.py (1,711 lines)
# ============================================================================

# NEW STRUCTURE:
# analytics/
# ├── __init__.py
# ├── base.py              # Base classes and interfaces
# ├── graph_metrics.py     # Graph metric calculations
# ├── pattern_detection.py # Pattern detection algorithms
# ├── community_analysis.py # Community detection
# ├── visualization.py     # Visualization utilities
# └── network_analyzer.py  # Main orchestrator (now ~200 lines)

# analytics/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import networkx as nx
from neo4j import AsyncSession


@dataclass
class AnalysisConfig:
    """Configuration for network analysis."""
    min_node_degree: int = 1
    min_edge_weight: float = 0.1
    community_resolution: float = 1.0
    pattern_confidence: float = 0.7
    use_cache: bool = True
    cache_ttl: int = 3600


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._graph: Optional[nx.Graph] = None
    
    @abstractmethod
    async def analyze(self, session: AsyncSession) -> Dict[str, Any]:
        """Perform analysis and return results."""
        pass
    
    @property
    def graph(self) -> nx.Graph:
        """Get or create graph instance."""
        if self._graph is None:
            raise ValueError("Graph not initialized. Call load_graph first.")
        return self._graph
    
    async def load_graph(self, session: AsyncSession) -> None:
        """Load graph from Neo4j."""
        query = """
        MATCH (n:Concept)
        OPTIONAL MATCH (n)-[r:RELATES_TO]-(m:Concept)
        RETURN n, r, m
        """
        result = await session.run(query)
        
        self._graph = nx.Graph()
        async for record in result:
            node = record["n"]
            self._graph.add_node(node["id"], **node)
            
            if record["r"] and record["m"]:
                rel = record["r"]
                other = record["m"]
                self._graph.add_edge(
                    node["id"], 
                    other["id"], 
                    weight=rel.get("weight", 1.0)
                )


# analytics/graph_metrics.py
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from .base import BaseAnalyzer, AnalysisConfig
from cache.cache_manager import CacheManager

cache = CacheManager()


class GraphMetricsAnalyzer(BaseAnalyzer):
    """Calculate various graph metrics."""
    
    @cache.cached(ttl=3600, key_prefix="graph_metrics")
    async def analyze(self, session) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics."""
        await self.load_graph(session)
        
        metrics = {
            "basic_stats": self._calculate_basic_stats(),
            "centrality_measures": self._calculate_centrality(),
            "connectivity": self._analyze_connectivity(),
            "degree_distribution": self._calculate_degree_distribution(),
            "clustering": self._analyze_clustering()
        }
        
        return metrics
    
    def _calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "average_degree": np.mean([d for n, d in self.graph.degree()]),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else -1
        }
    
    def _calculate_centrality(self) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate various centrality measures."""
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Betweenness centrality (expensive, so limit to subgraph if needed)
        if self.graph.number_of_nodes() > 1000:
            # Sample for large graphs
            sample_nodes = list(self.graph.nodes())[:500]
            subgraph = self.graph.subgraph(sample_nodes)
            between_cent = nx.betweenness_centrality(subgraph)
        else:
            between_cent = nx.betweenness_centrality(self.graph)
        top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Eigenvector centrality
        try:
            eigen_cent = nx.eigenvector_centrality(self.graph, max_iter=1000)
            top_eigen = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        except:
            top_eigen = []
        
        return {
            "degree": top_degree,
            "betweenness": top_between,
            "eigenvector": top_eigen
        }
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity."""
        components = list(nx.connected_components(self.graph))
        
        return {
            "is_connected": nx.is_connected(self.graph),
            "num_components": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0,
            "avg_shortest_path": self._calculate_avg_shortest_path()
        }
    
    def _calculate_avg_shortest_path(self) -> float:
        """Calculate average shortest path length."""
        if nx.is_connected(self.graph):
            return nx.average_shortest_path_length(self.graph)
        else:
            # Calculate for largest component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph)
    
    def _calculate_degree_distribution(self) -> Dict[str, List[int]]:
        """Calculate degree distribution."""
        degrees = [d for n, d in self.graph.degree()]
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
        
        return {
            "degrees": list(degree_counts.keys()),
            "counts": list(degree_counts.values())
        }
    
    def _analyze_clustering(self) -> Dict[str, float]:
        """Analyze clustering in the graph."""
        return {
            "average_clustering": nx.average_clustering(self.graph),
            "transitivity": nx.transitivity(self.graph)
        }


# analytics/pattern_detection.py
from typing import List, Dict, Set, Tuple
import itertools
from .base import BaseAnalyzer
from cache.cache_manager import CacheManager

cache = CacheManager()


class PatternDetector(BaseAnalyzer):
    """Detect various patterns in the knowledge graph."""
    
    @cache.cached(ttl=7200, key_prefix="patterns")
    async def analyze(self, session) -> Dict[str, Any]:
        """Detect patterns in the graph."""
        await self.load_graph(session)
        
        patterns = {
            "triadic_patterns": self._detect_triadic_patterns(),
            "cross_domain_bridges": self._find_cross_domain_bridges(),
            "knowledge_chains": self._find_knowledge_chains(),
            "concept_clusters": self._identify_concept_clusters(),
            "temporal_patterns": await self._detect_temporal_patterns(session)
        }
        
        return patterns
    
    def _detect_triadic_patterns(self) -> List[Dict[str, Any]]:
        """Detect common triadic patterns (triangles, etc.)."""
        triangles = []
        
        # Find all triangles
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if self.graph.has_edge(n1, n2):
                        # Found a triangle
                        triangle_nodes = [node, n1, n2]
                        domains = [self.graph.nodes[n].get('domain', 'unknown') 
                                 for n in triangle_nodes]
                        
                        triangles.append({
                            'nodes': triangle_nodes,
                            'domains': domains,
                            'cross_domain': len(set(domains)) > 1,
                            'avg_weight': np.mean([
                                self.graph[u][v].get('weight', 1.0)
                                for u, v in itertools.combinations(triangle_nodes, 2)
                            ])
                        })
        
        # Return top patterns by weight
        return sorted(triangles, key=lambda x: x['avg_weight'], reverse=True)[:20]
    
    def _find_cross_domain_bridges(self) -> List[Dict[str, Any]]:
        """Find nodes that bridge different domains."""
        bridges = []
        
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                continue
            
            # Get domains of neighbors
            neighbor_domains = [
                self.graph.nodes[n].get('domain', 'unknown') 
                for n in neighbors
            ]
            unique_domains = set(neighbor_domains)
            
            if len(unique_domains) > 1:
                domain_counts = {d: neighbor_domains.count(d) for d in unique_domains}
                
                bridges.append({
                    'node': node,
                    'node_domain': self.graph.nodes[node].get('domain', 'unknown'),
                    'connected_domains': list(unique_domains),
                    'domain_counts': domain_counts,
                    'bridge_score': len(unique_domains) * len(neighbors)
                })
        
        return sorted(bridges, key=lambda x: x['bridge_score'], reverse=True)[:10]
    
    def _find_knowledge_chains(self) -> List[List[str]]:
        """Find meaningful chains of connected concepts."""
        chains = []
        
        # Find paths between important nodes
        important_nodes = self._get_important_nodes()
        
        for i, start in enumerate(important_nodes[:5]):
            for end in important_nodes[i+1:10]:
                try:
                    # Find multiple paths
                    paths = list(nx.all_simple_paths(
                        self.graph, start, end, cutoff=5
                    ))[:3]
                    
                    for path in paths:
                        if len(path) >= 3:  # Meaningful chains
                            chains.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return chains[:20]
    
    def _get_important_nodes(self) -> List[str]:
        """Get nodes with high importance scores."""
        importance_scores = {}
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            degree = self.graph.degree(node)
            importance = node_data.get('importance', 0.5) * degree
            importance_scores[node] = importance
        
        return sorted(importance_scores.keys(), 
                     key=lambda x: importance_scores[x], 
                     reverse=True)
    
    def _identify_concept_clusters(self) -> List[Dict[str, Any]]:
        """Identify tightly connected concept clusters."""
        # Use Louvain community detection
        communities = nx.community.louvain_communities(
            self.graph, 
            resolution=self.config.community_resolution
        )
        
        clusters = []
        for community in communities:
            if len(community) < 3:  # Skip tiny clusters
                continue
            
            subgraph = self.graph.subgraph(community)
            
            # Analyze cluster
            domains = [self.graph.nodes[n].get('domain', 'unknown') 
                       for n in community]
            domain_dist = {d: domains.count(d) for d in set(domains)}
            
            clusters.append({
                'size': len(community),
                'nodes': list(community)[:10],  # Sample for large clusters
                'density': nx.density(subgraph),
                'domain_distribution': domain_dist,
                'primary_domain': max(domain_dist.keys(), key=domain_dist.get),
                'cohesion_score': nx.density(subgraph) * len(community)
            })
        
        return sorted(clusters, key=lambda x: x['cohesion_score'], reverse=True)[:15]
    
    async def _detect_temporal_patterns(self, session) -> List[Dict[str, Any]]:
        """Detect temporal patterns in concept evolution."""
        query = """
        MATCH (c:Concept)
        WHERE c.time_period IS NOT NULL
        OPTIONAL MATCH (c)-[r:RELATES_TO]-(other:Concept)
        WHERE other.time_period IS NOT NULL
        RETURN c.id as concept, c.time_period as period, 
               collect(distinct other.time_period) as related_periods
        ORDER BY c.time_period
        """
        
        result = await session.run(query)
        temporal_data = []
        
        async for record in result:
            temporal_data.append({
                'concept': record['concept'],
                'period': record['period'],
                'related_periods': record['related_periods']
            })
        
        # Analyze temporal patterns
        patterns = []
        
        # Group by time period
        period_groups = defaultdict(list)
        for item in temporal_data:
            period_groups[item['period']].append(item)
        
        # Find patterns
        for period, concepts in period_groups.items():
            if len(concepts) > 5:  # Significant period
                patterns.append({
                    'period': period,
                    'concept_count': len(concepts),
                    'cross_period_connections': sum(
                        len(c['related_periods']) for c in concepts
                    ),
                    'sample_concepts': [c['concept'] for c in concepts[:5]]
                })
        
        return sorted(patterns, key=lambda x: x['concept_count'], reverse=True)


# analytics/community_analysis.py
from typing import Dict, List, Set
import numpy as np
from .base import BaseAnalyzer
from sklearn.metrics.pairwise import cosine_similarity


class CommunityAnalyzer(BaseAnalyzer):
    """Analyze communities and their properties."""
    
    async def analyze(self, session) -> Dict[str, Any]:
        """Perform community analysis."""
        await self.load_graph(session)
        
        # Detect communities using multiple algorithms
        communities = {
            "louvain": self._louvain_communities(),
            "label_propagation": self._label_propagation_communities(),
            "k_clique": self._k_clique_communities()
        }
        
        # Analyze each set of communities
        analysis = {}
        for method, comm_list in communities.items():
            analysis[method] = {
                "communities": self._analyze_communities(comm_list),
                "modularity": self._calculate_modularity(comm_list),
                "coverage": self._calculate_coverage(comm_list)
            }
        
        return analysis
    
    def _louvain_communities(self) -> List[Set[str]]:
        """Detect communities using Louvain algorithm."""
        return list(nx.community.louvain_communities(self.graph))
    
    def _label_propagation_communities(self) -> List[Set[str]]:
        """Detect communities using label propagation."""
        return list(nx.community.label_propagation_communities(self.graph))
    
    def _k_clique_communities(self) -> List[Set[str]]:
        """Detect communities using k-clique percolation."""
        # Convert to undirected for k-clique
        return list(nx.community.k_clique_communities(self.graph, k=3))
    
    def _analyze_communities(self, communities: List[Set[str]]) -> List[Dict]:
        """Analyze properties of detected communities."""
        analyzed = []
        
        for i, community in enumerate(communities):
            if len(community) < 3:  # Skip tiny communities
                continue
            
            subgraph = self.graph.subgraph(community)
            
            # Get domain distribution
            domains = [self.graph.nodes[n].get('domain', 'unknown') 
                      for n in community]
            domain_counts = {d: domains.count(d) for d in set(domains)}
            
            # Calculate internal vs external connections
            internal_edges = subgraph.number_of_edges()
            external_edges = sum(
                1 for n in community 
                for neighbor in self.graph.neighbors(n) 
                if neighbor not in community
            )
            
            analyzed.append({
                'id': i,
                'size': len(community),
                'density': nx.density(subgraph),
                'domains': domain_counts,
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'conductance': external_edges / (internal_edges + external_edges) if internal_edges > 0 else 1.0,
                'key_members': self._get_key_members(community)
            })
        
        return sorted(analyzed, key=lambda x: x['size'], reverse=True)[:20]
    
    def _get_key_members(self, community: Set[str]) -> List[str]:
        """Get most important members of a community."""
        subgraph = self.graph.subgraph(community)
        
        # Use PageRank within community
        try:
            pagerank = nx.pagerank(subgraph)
            return sorted(pagerank.keys(), key=pagerank.get, reverse=True)[:5]
        except:
            # Fallback to degree
            degrees = dict(subgraph.degree())
            return sorted(degrees.keys(), key=degrees.get, reverse=True)[:5]
    
    def _calculate_modularity(self, communities: List[Set[str]]) -> float:
        """Calculate modularity score."""
        return nx.community.modularity(self.graph, communities)
    
    def _calculate_coverage(self, communities: List[Set[str]]) -> float:
        """Calculate fraction of nodes covered by communities."""
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        
        return len(all_nodes) / self.graph.number_of_nodes()


# analytics/network_analyzer.py (REFACTORED - now only ~200 lines)
from typing import Dict, Any, Optional
from .base import AnalysisConfig
from .graph_metrics import GraphMetricsAnalyzer
from .pattern_detection import PatternDetector
from .community_analysis import CommunityAnalyzer
from neo4j import AsyncGraphDatabase
import asyncio


class NetworkAnalyzer:
    """Main network analyzer orchestrating all analysis components."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 config: Optional[AnalysisConfig] = None):
        self.driver = AsyncGraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        self.config = config or AnalysisConfig()
        
        # Initialize analyzers
        self.metrics_analyzer = GraphMetricsAnalyzer(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.community_analyzer = CommunityAnalyzer(self.config)
    
    async def close(self):
        """Close database connection."""
        await self.driver.close()
    
    async def analyze_network(self, include_all: bool = True) -> Dict[str, Any]:
        """Perform comprehensive network analysis."""
        async with self.driver.session() as session:
            tasks = []
            
            # Always include basic metrics
            tasks.append(('metrics', self.metrics_analyzer.analyze(session)))
            
            if include_all:
                tasks.append(('patterns', self.pattern_detector.analyze(session)))
                tasks.append(('communities', self.community_analyzer.analyze(session)))
            
            # Run analyses in parallel
            results = {}
            for name, task in tasks:
                try:
                    results[name] = await task
                except Exception as e:
                    results[name] = {'error': str(e)}
            
            # Add summary
            results['summary'] = self._create_summary(results)
            
            return results
    
    async def get_node_importance(self, node_id: str) -> Dict[str, Any]:
        """Get importance metrics for a specific node."""
        async with self.driver.session() as session:
            query = """
            MATCH (n:Concept {id: $node_id})
            OPTIONAL MATCH (n)-[r:RELATES_TO]-(m:Concept)
            RETURN n,
                   count(distinct m) as degree,
                   avg(r.weight) as avg_weight,
                   collect(distinct m.domain) as connected_domains
            """
            
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            
            if not record:
                return {'error': 'Node not found'}
            
            node = record['n']
            
            return {
                'id': node_id,
                'name': node.get('name', 'Unknown'),
                'domain': node.get('domain', 'Unknown'),
                'metrics': {
                    'degree': record['degree'],
                    'avg_connection_weight': record['avg_weight'],
                    'cross_domain_connections': len(record['connected_domains']) - 1,
                    'importance_score': node.get('importance', 0.5) * record['degree']
                }
            }
    
    async def find_shortest_path(self, start_id: str, end_id: str) -> Dict[str, Any]:
        """Find shortest path between two concepts."""
        async with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (start:Concept {id: $start_id})-[*]-(end:Concept {id: $end_id})
            )
            RETURN [n in nodes(path) | {id: n.id, name: n.name, domain: n.domain}] as path,
                   length(path) as distance
            """
            
            result = await session.run(query, start_id=start_id, end_id=end_id)
            record = await result.single()
            
            if not record:
                return {'error': 'No path found'}
            
            return {
                'path': record['path'],
                'distance': record['distance']
            }
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of analysis results."""
        summary = {
            'network_health': 'good',  # Default
            'key_insights': [],
            'recommendations': []
        }
        
        # Check metrics
        if 'metrics' in results and 'basic_stats' in results['metrics']:
            stats = results['metrics']['basic_stats']
            
            # Health checks
            if stats['density'] < 0.01:
                summary['network_health'] = 'sparse'
                summary['recommendations'].append(
                    "Network is very sparse. Consider adding more connections."
                )
            
            if stats['num_nodes'] > 10000 and stats['average_degree'] < 5:
                summary['recommendations'].append(
                    "Large network with low connectivity. Consider pruning or enriching."
                )
        
        # Pattern insights
        if 'patterns' in results and 'cross_domain_bridges' in results['patterns']:
            bridges = results['patterns']['cross_domain_bridges']
            if bridges:
                summary['key_insights'].append(
                    f"Found {len(bridges)} important cross-domain bridge concepts"
                )
        
        # Community insights
        if 'communities' in results:
            for method, data in results['communities'].items():
                if 'communities' in data:
                    summary['key_insights'].append(
                        f"{method} detected {len(data['communities'])} communities"
                    )
        
        return summary


# ============================================================================
# USAGE EXAMPLE FOR AI CODING ASSISTANT
# ============================================================================
"""
To implement this refactoring:

1. Create the new directory structure:
   ```bash
   mkdir -p analytics/{base,graph_metrics,pattern_detection,community_analysis,visualization}
   touch analytics/__init__.py
   touch analytics/base.py
   touch analytics/graph_metrics.py
   touch analytics/pattern_detection.py
   touch analytics/community_analysis.py
   ```

2. Copy the respective code sections to each file

3. Update imports in files that use NetworkAnalyzer:
   ```python
   # Old import
   from analytics.network_analyzer import NetworkAnalyzer
   
   # New import (same interface, refactored internals)
   from analytics.network_analyzer import NetworkAnalyzer
   ```

4. Test the refactored code:
   ```bash
   pytest tests/analytics/test_network_analyzer.py -v
   ```

5. Update any documentation referencing the old structure

The same pattern can be applied to other large files in the project.
"""