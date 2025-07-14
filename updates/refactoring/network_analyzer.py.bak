"""
Network Analysis Agent for MCP Server
Performs graph-based analytics using PageRank, community detection, and centrality measures.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from collections import defaultdict, Counter
import pickle

import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
from networkx.algorithms.centrality import pagerank, betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms.cluster import clustering, transitivity
from networkx.algorithms.shortest_paths import shortest_path_length, average_shortest_path_length
from networkx.algorithms.components import connected_components, strongly_connected_components
import matplotlib.pyplot as plt
import seaborn as sns

from neo4j import AsyncGraphDatabase, AsyncDriver


class AnalysisType(Enum):
    """Types of network analysis that can be performed."""
    CENTRALITY = "centrality"
    COMMUNITY_DETECTION = "community_detection"
    INFLUENCE_PROPAGATION = "influence_propagation"
    KNOWLEDGE_FLOW = "knowledge_flow"
    BRIDGE_NODES = "bridge_nodes"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    PATH_ANALYSIS = "path_analysis"


class CentralityMeasure(Enum):
    """Types of centrality measures."""
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    DEGREE = "degree"
    KATZ = "katz"
    HARMONIC = "harmonic"


class CommunityAlgorithm(Enum):
    """Community detection algorithms."""
    GIRVAN_NEWMAN = "girvan_newman"
    GREEDY_MODULARITY = "greedy_modularity"
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    LEIDEN = "leiden"
    FLUID_COMMUNITIES = "fluid_communities"


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    centrality_scores: Dict[str, float]
    community_id: Optional[int]
    clustering_coefficient: float
    degree: int
    metadata: Dict[str, Any]


@dataclass
class CommunityInfo:
    """Information about a detected community."""
    community_id: int
    nodes: List[str]
    size: int
    internal_edges: int
    external_edges: int
    modularity_contribution: float
    description: str
    metadata: Dict[str, Any]


@dataclass
class NetworkAnalysisResult:
    """Complete network analysis result."""
    analysis_type: AnalysisType
    graph_metrics: Dict[str, float]
    node_metrics: List[NodeMetrics]
    communities: List[CommunityInfo]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    execution_time: float


class NetworkConfig:
    """Configuration for network analysis."""
    
    def __init__(self, config_path: str = "analytics/config.py"):
        # Database connection
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        
        # Analysis parameters
        self.max_nodes = 10000
        self.max_edges = 50000
        self.pagerank_alpha = 0.85
        self.pagerank_max_iter = 100
        self.pagerank_tol = 1e-6
        
        # Community detection parameters
        self.min_community_size = 3
        self.max_communities = 50
        self.resolution_parameter = 1.0
        
        # Centrality calculation parameters
        self.normalize_centrality = True
        self.centrality_k = None  # For k-path centralities
        
        # Performance settings
        self.parallel_processing = True
        self.cache_results = True
        self.cache_duration_hours = 24
        
        # Visualization settings
        self.generate_plots = True
        self.plot_dir = "analytics/plots"
        self.figure_size = (12, 8)
        self.node_size_multiplier = 300
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file if it exists."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")


class NetworkAnalyzer:
    """Main network analysis engine."""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize the network analyzer."""
        self.config = config or NetworkConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        
        # Cached data
        self.cached_graphs: Dict[str, nx.Graph] = {}
        self.cached_metrics: Dict[str, Dict] = {}
        
        # Storage
        self.plot_dir = Path(self.config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("network_analyzer")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self.logger.info("Network analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize network analyzer: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        self.logger.info("Network analyzer closed")
    
    async def analyze_network(
        self,
        analysis_type: AnalysisType,
        domain_scope: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        min_connections: int = 1
    ) -> NetworkAnalysisResult:
        """Perform comprehensive network analysis."""
        
        start_time = datetime.now()
        
        try:
            # Load graph from Neo4j
            graph = await self._load_graph(domain_scope, node_types, relationship_types, min_connections)
            
            if len(graph.nodes) == 0:
                raise ValueError("No nodes found matching the specified criteria")
            
            self.logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.CENTRALITY:
                result = await self._analyze_centrality(graph)
            elif analysis_type == AnalysisType.COMMUNITY_DETECTION:
                result = await self._analyze_communities(graph)
            elif analysis_type == AnalysisType.INFLUENCE_PROPAGATION:
                result = await self._analyze_influence_propagation(graph)
            elif analysis_type == AnalysisType.KNOWLEDGE_FLOW:
                result = await self._analyze_knowledge_flow(graph)
            elif analysis_type == AnalysisType.BRIDGE_NODES:
                result = await self._analyze_bridge_nodes(graph)
            elif analysis_type == AnalysisType.STRUCTURAL_ANALYSIS:
                result = await self._analyze_structure(graph)
            elif analysis_type == AnalysisType.CLUSTERING_ANALYSIS:
                result = await self._analyze_clustering(graph)
            elif analysis_type == AnalysisType.PATH_ANALYSIS:
                result = await self._analyze_paths(graph)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.generated_at = datetime.now(timezone.utc)
            
            # Generate visualization if configured
            if self.config.generate_plots:
                await self._generate_network_visualization(graph, result)
            
            self.logger.info(f"Network analysis completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in network analysis: {e}")
            raise
    
    async def _load_graph(
        self,
        domain_scope: Optional[List[str]],
        node_types: Optional[List[str]],
        relationship_types: Optional[List[str]],
        min_connections: int
    ) -> nx.Graph:
        """Load graph data from Neo4j."""
        
        async with self.neo4j_driver.session() as session:
            # Build node filter
            node_filters = []
            params = {}
            
            if domain_scope:
                node_filters.append("n.domain IN $domains")
                params['domains'] = domain_scope
            
            if node_types:
                label_conditions = " OR ".join([f"n:{nt}" for nt in node_types])
                node_filters.append(f"({label_conditions})")
            
            node_where = "WHERE " + " AND ".join(node_filters) if node_filters else ""
            
            # Get nodes
            nodes_query = f"""
            MATCH (n)
            {node_where}
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as connections
            WHERE connections >= $min_connections
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                n.title as title,
                n.domain as domain,
                n.date as date,
                n.author as author,
                connections
            LIMIT $max_nodes
            """
            
            params.update({
                'min_connections': min_connections,
                'max_nodes': self.config.max_nodes
            })
            
            result = await session.run(nodes_query, params)
            
            # Create NetworkX graph
            G = nx.Graph()
            node_data = {}
            
            async for record in result:
                node_id = str(record['node_id'])
                node_info = {
                    'title': record['title'],
                    'domain': record['domain'],
                    'date': record['date'],
                    'author': record['author'],
                    'labels': record['labels'],
                    'connections': record['connections']
                }
                G.add_node(node_id, **node_info)
                node_data[node_id] = node_info
            
            if len(G.nodes) == 0:
                return G
            
            # Get relationships
            rel_filters = []
            if relationship_types:
                rel_filters.append("type(r) IN $rel_types")
                params['rel_types'] = relationship_types
            
            rel_where = "AND " + " AND ".join(rel_filters) if rel_filters else ""
            
            relationships_query = f"""
            MATCH (n1)-[r]-(n2)
            WHERE id(n1) IN $node_ids AND id(n2) IN $node_ids
            {rel_where}
            RETURN 
                id(n1) as source,
                id(n2) as target,
                type(r) as rel_type,
                r.weight as weight
            LIMIT $max_edges
            """
            
            params.update({
                'node_ids': list(node_data.keys()),
                'max_edges': self.config.max_edges
            })
            
            result = await session.run(relationships_query, params)
            
            async for record in result:
                source = str(record['source'])
                target = str(record['target'])
                weight = record.get('weight', 1.0)
                rel_type = record['rel_type']
                
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, weight=weight, rel_type=rel_type)
            
            return G
    
    async def _analyze_centrality(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze node centrality measures."""
        
        self.logger.info("Computing centrality measures...")
        
        # Calculate different centrality measures
        centrality_measures = {}
        
        # PageRank
        try:
            pagerank_scores = nx.pagerank(
                graph,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol
            )
            centrality_measures['pagerank'] = pagerank_scores
        except Exception as e:
            self.logger.warning(f"PageRank calculation failed: {e}")
            centrality_measures['pagerank'] = {node: 0.0 for node in graph.nodes}
        
        # Betweenness centrality
        try:
            betweenness_scores = nx.betweenness_centrality(
                graph, 
                normalized=self.config.normalize_centrality
            )
            centrality_measures['betweenness'] = betweenness_scores
        except Exception as e:
            self.logger.warning(f"Betweenness centrality calculation failed: {e}")
            centrality_measures['betweenness'] = {node: 0.0 for node in graph.nodes}
        
        # Closeness centrality
        try:
            closeness_scores = nx.closeness_centrality(
                graph,
                normalized=self.config.normalize_centrality
            )
            centrality_measures['closeness'] = closeness_scores
        except Exception as e:
            self.logger.warning(f"Closeness centrality calculation failed: {e}")
            centrality_measures['closeness'] = {node: 0.0 for node in graph.nodes}
        
        # Eigenvector centrality
        try:
            eigenvector_scores = nx.eigenvector_centrality(
                graph,
                max_iter=1000
            )
            centrality_measures['eigenvector'] = eigenvector_scores
        except Exception as e:
            self.logger.warning(f"Eigenvector centrality calculation failed: {e}")
            centrality_measures['eigenvector'] = {node: 0.0 for node in graph.nodes}
        
        # Degree centrality
        degree_scores = nx.degree_centrality(graph)
        centrality_measures['degree'] = degree_scores
        
        # Create node metrics
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    measure: scores.get(node, 0.0)
                    for measure, scores in centrality_measures.items()
                },
                community_id=None,  # Will be filled by community detection
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph-level metrics
        graph_metrics = self._calculate_graph_metrics(graph, centrality_measures)
        
        # Generate insights
        insights = self._generate_centrality_insights(graph, centrality_measures, node_metrics)
        
        # Generate recommendations
        recommendations = self._generate_centrality_recommendations(centrality_measures, node_metrics)
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.CENTRALITY,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _analyze_communities(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze community structure in the network."""
        
        self.logger.info("Detecting communities...")
        
        communities_info = []
        
        # Try multiple community detection algorithms
        try:
            # Greedy modularity optimization
            communities = greedy_modularity_communities(graph, resolution=self.config.resolution_parameter)
            communities_list = [list(community) for community in communities]
            
            # Calculate modularity
            modularity = community.modularity(graph, communities)
            
            # Create community info
            for i, comm_nodes in enumerate(communities_list):
                if len(comm_nodes) >= self.config.min_community_size:
                    subgraph = graph.subgraph(comm_nodes)
                    internal_edges = subgraph.number_of_edges()
                    
                    # Count external edges
                    external_edges = 0
                    for node in comm_nodes:
                        for neighbor in graph.neighbors(node):
                            if neighbor not in comm_nodes:
                                external_edges += 1
                    external_edges //= 2  # Avoid double counting
                    
                    # Generate community description
                    description = self._generate_community_description(graph, comm_nodes)
                    
                    community_info = CommunityInfo(
                        community_id=i,
                        nodes=comm_nodes,
                        size=len(comm_nodes),
                        internal_edges=internal_edges,
                        external_edges=external_edges,
                        modularity_contribution=0.0,  # Would need detailed calculation
                        description=description,
                        metadata={
                            'algorithm': 'greedy_modularity',
                            'avg_degree': sum(graph.degree(node) for node in comm_nodes) / len(comm_nodes),
                            'density': nx.density(subgraph)
                        }
                    )
                    communities_info.append(community_info)
            
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
        
        # Update node metrics with community assignments
        node_metrics = []
        community_assignment = {}
        for i, comm_info in enumerate(communities_info):
            for node in comm_info.nodes:
                community_assignment[node] = i
        
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={},  # Would be filled by centrality analysis
                community_id=community_assignment.get(node),
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph metrics
        graph_metrics = {
            'num_communities': len(communities_info),
            'modularity': modularity if 'modularity' in locals() else 0.0,
            'avg_community_size': np.mean([c.size for c in communities_info]) if communities_info else 0,
            'largest_community_size': max([c.size for c in communities_info]) if communities_info else 0,
            'community_size_std': np.std([c.size for c in communities_info]) if communities_info else 0
        }
        
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
    
    async def _analyze_influence_propagation(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze influence propagation patterns."""
        
        self.logger.info("Analyzing influence propagation...")
        
        # Calculate influence-related metrics
        influence_metrics = {}
        
        # K-core decomposition to find influential cores
        try:
            k_core = nx.core_number(graph)
            influence_metrics['k_core'] = k_core
        except Exception as e:
            self.logger.warning(f"K-core analysis failed: {e}")
            influence_metrics['k_core'] = {node: 0 for node in graph.nodes}
        
        # Calculate reach (how many nodes can be reached within k steps)
        reach_metrics = {}
        for node in list(graph.nodes)[:min(100, len(graph.nodes))]:  # Sample for performance
            try:
                # Calculate reach within 2 hops
                reachable = set([node])
                current_level = {node}
                for _ in range(2):
                    next_level = set()
                    for n in current_level:
                        next_level.update(graph.neighbors(n))
                    next_level -= reachable
                    reachable.update(next_level)
                    current_level = next_level
                    if not current_level:
                        break
                
                reach_metrics[node] = len(reachable) - 1  # Exclude self
            except Exception:
                reach_metrics[node] = 0
        
        # Create node metrics
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'k_core': influence_metrics['k_core'].get(node, 0),
                    'reach_2hop': reach_metrics.get(node, 0)
                },
                community_id=None,
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph metrics
        graph_metrics = {
            'max_k_core': max(influence_metrics['k_core'].values()) if influence_metrics['k_core'] else 0,
            'avg_k_core': np.mean(list(influence_metrics['k_core'].values())) if influence_metrics['k_core'] else 0,
            'avg_reach_2hop': np.mean(list(reach_metrics.values())) if reach_metrics else 0,
            'influence_concentration': np.std(list(influence_metrics['k_core'].values())) if influence_metrics['k_core'] else 0
        }
        
        # Generate insights
        insights = [
            f"Maximum k-core value: {graph_metrics['max_k_core']}",
            f"Average k-core value: {graph_metrics['avg_k_core']:.2f}",
            f"Average 2-hop reach: {graph_metrics['avg_reach_2hop']:.1f}",
        ]
        
        # Find most influential nodes
        top_k_core_nodes = sorted(
            influence_metrics['k_core'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        if top_k_core_nodes:
            insights.append("Top influential nodes (by k-core):")
            for node, score in top_k_core_nodes:
                node_title = graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: k-core = {score}")
        
        recommendations = [
            "Target high k-core nodes for maximum influence propagation",
            "Consider nodes with high 2-hop reach for information dissemination",
            "Monitor k-core changes to detect shifting influence patterns"
        ]
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.INFLUENCE_PROPAGATION,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _analyze_knowledge_flow(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze knowledge flow patterns."""
        
        self.logger.info("Analyzing knowledge flow...")
        
        # For knowledge flow, we need a directed graph
        if not graph.is_directed():
            # Convert to directed based on temporal information
            directed_graph = self._create_temporal_directed_graph(graph)
        else:
            directed_graph = graph
        
        # Calculate flow-related metrics
        flow_metrics = {}
        
        # In-degree and out-degree
        in_degree = dict(directed_graph.in_degree())
        out_degree = dict(directed_graph.out_degree())
        
        # Authority and hub scores (HITS algorithm)
        try:
            hubs, authorities = nx.hits(directed_graph, max_iter=100)
            flow_metrics['hubs'] = hubs
            flow_metrics['authorities'] = authorities
        except Exception as e:
            self.logger.warning(f"HITS algorithm failed: {e}")
            flow_metrics['hubs'] = {node: 0.0 for node in directed_graph.nodes}
            flow_metrics['authorities'] = {node: 0.0 for node in directed_graph.nodes}
        
        # Create node metrics
        node_metrics = []
        for node in directed_graph.nodes:
            node_data = directed_graph.nodes[node]
            
            # Calculate knowledge flow ratio
            in_deg = in_degree.get(node, 0)
            out_deg = out_degree.get(node, 0)
            total_deg = in_deg + out_deg
            
            if total_deg > 0:
                flow_ratio = out_deg / total_deg  # Higher = more source-like
            else:
                flow_ratio = 0.5
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'in_degree': in_deg,
                    'out_degree': out_deg,
                    'hub_score': flow_metrics['hubs'].get(node, 0),
                    'authority_score': flow_metrics['authorities'].get(node, 0),
                    'flow_ratio': flow_ratio
                },
                community_id=None,
                clustering_coefficient=0.0,  # Not applicable for directed
                degree=total_deg,
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph metrics
        graph_metrics = {
            'avg_in_degree': np.mean(list(in_degree.values())),
            'avg_out_degree': np.mean(list(out_degree.values())),
            'max_in_degree': max(in_degree.values()) if in_degree else 0,
            'max_out_degree': max(out_degree.values()) if out_degree else 0,
            'reciprocity': nx.reciprocity(directed_graph) if directed_graph.is_directed() else 0.0
        }
        
        # Identify knowledge sources and sinks
        sources = [node for node, metrics in zip(directed_graph.nodes, node_metrics) 
                  if metrics.centrality_scores['flow_ratio'] > 0.8 and metrics.degree > 2]
        sinks = [node for node, metrics in zip(directed_graph.nodes, node_metrics) 
                if metrics.centrality_scores['flow_ratio'] < 0.2 and metrics.degree > 2]
        
        # Generate insights
        insights = [
            f"Average in-degree: {graph_metrics['avg_in_degree']:.2f}",
            f"Average out-degree: {graph_metrics['avg_out_degree']:.2f}",
            f"Network reciprocity: {graph_metrics['reciprocity']:.3f}",
            f"Identified {len(sources)} knowledge sources",
            f"Identified {len(sinks)} knowledge sinks"
        ]
        
        # Top authorities and hubs
        top_authorities = sorted(
            flow_metrics['authorities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        top_hubs = sorted(
            flow_metrics['hubs'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_authorities:
            insights.append("Top knowledge authorities:")
            for node, score in top_authorities:
                node_title = directed_graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: authority = {score:.3f}")
        
        if top_hubs:
            insights.append("Top knowledge hubs:")
            for node, score in top_hubs:
                node_title = directed_graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: hub = {score:.3f}")
        
        recommendations = [
            "Focus on high-authority nodes for knowledge consolidation",
            "Leverage high-hub nodes for knowledge distribution",
            "Monitor reciprocity changes to understand collaboration patterns",
            "Connect knowledge sources to sinks to improve flow"
        ]
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.KNOWLEDGE_FLOW,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _analyze_bridge_nodes(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze bridge nodes and structural holes."""
        
        self.logger.info("Analyzing bridge nodes...")
        
        # Calculate betweenness centrality (key for bridge identification)
        betweenness = nx.betweenness_centrality(graph)
        
        # Calculate edge betweenness
        edge_betweenness = nx.edge_betweenness_centrality(graph)
        
        # Find structural holes using effective size
        effective_size = {}
        constraint = {}
        
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) <= 1:
                effective_size[node] = 0
                constraint[node] = 1.0 if neighbors else 0.0
                continue
            
            # Calculate effective size (diversity of connections)
            redundancy = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if graph.has_edge(neighbors[i], neighbors[j]):
                        redundancy += 1
            
            max_redundancy = len(neighbors) * (len(neighbors) - 1) / 2
            effective_size[node] = len(neighbors) - (redundancy / max_redundancy if max_redundancy > 0 else 0)
            
            # Calculate constraint (inverse of structural holes)
            constraint[node] = 1.0 / (effective_size[node] + 1)
        
        # Identify bridge nodes
        bridge_threshold = np.percentile(list(betweenness.values()), 90)
        bridge_nodes = [node for node, score in betweenness.items() if score > bridge_threshold]
        
        # Create node metrics
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'betweenness': betweenness.get(node, 0),
                    'effective_size': effective_size.get(node, 0),
                    'constraint': constraint.get(node, 0),
                    'is_bridge': node in bridge_nodes
                },
                community_id=None,
                clustering_coefficient=nx.clustering(graph, node),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph metrics
        graph_metrics = {
            'num_bridge_nodes': len(bridge_nodes),
            'avg_betweenness': np.mean(list(betweenness.values())),
            'max_betweenness': max(betweenness.values()) if betweenness else 0,
            'avg_effective_size': np.mean(list(effective_size.values())),
            'bridge_concentration': len(bridge_nodes) / len(graph.nodes) if graph.nodes else 0
        }
        
        # Generate insights
        insights = [
            f"Identified {len(bridge_nodes)} bridge nodes ({graph_metrics['bridge_concentration']:.1%} of network)",
            f"Average betweenness centrality: {graph_metrics['avg_betweenness']:.4f}",
            f"Maximum betweenness centrality: {graph_metrics['max_betweenness']:.4f}",
            f"Average effective size: {graph_metrics['avg_effective_size']:.2f}"
        ]
        
        # Top bridge nodes
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_bridges:
            insights.append("Top bridge nodes:")
            for node, score in top_bridges:
                node_title = graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: betweenness = {score:.4f}")
        
        # Structural holes analysis
        top_structural_holes = sorted(effective_size.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_structural_holes:
            insights.append("Nodes with most structural holes:")
            for node, score in top_structural_holes:
                node_title = graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: effective size = {score:.2f}")
        
        recommendations = [
            "Monitor bridge nodes as they are critical for network connectivity",
            "Consider bridge nodes for strategic interventions or communications",
            "Strengthen connections around high-betweenness nodes to reduce vulnerability",
            "Leverage nodes with structural holes for brokerage opportunities"
        ]
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.BRIDGE_NODES,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _analyze_structure(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze overall network structure."""
        
        self.logger.info("Analyzing network structure...")
        
        # Basic structural metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        
        # Connectivity analysis
        is_connected = nx.is_connected(graph)
        num_components = nx.number_connected_components(graph)
        
        if is_connected:
            diameter = nx.diameter(graph)
            avg_path_length = nx.average_shortest_path_length(graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            diameter = nx.diameter(largest_subgraph) if len(largest_cc) > 1 else 0
            avg_path_length = nx.average_shortest_path_length(largest_subgraph) if len(largest_cc) > 1 else 0
        
        # Clustering metrics
        avg_clustering = nx.average_clustering(graph)
        transitivity = nx.transitivity(graph)
        
        # Degree distribution
        degrees = [d for n, d in graph.degree()]
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        max_degree = max(degrees) if degrees else 0
        
        # Small world properties
        # Compare clustering and path length to random graph
        try:
            random_graph = nx.erdos_renyi_graph(num_nodes, density)
            random_clustering = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph) if nx.is_connected(random_graph) else 0
            
            clustering_ratio = avg_clustering / random_clustering if random_clustering > 0 else 0
            path_length_ratio = avg_path_length / random_path_length if random_path_length > 0 else 0
            
            # Small world coefficient
            small_world_coeff = clustering_ratio / path_length_ratio if path_length_ratio > 0 else 0
        except Exception:
            clustering_ratio = 0
            path_length_ratio = 0
            small_world_coeff = 0
        
        # Assortativity (degree correlation)
        try:
            assortativity = nx.degree_assortativity_coefficient(graph)
        except Exception:
            assortativity = 0
        
        # Create basic node metrics
        node_metrics = []
        clustering_dict = nx.clustering(graph)
        
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'degree': graph.degree(node),
                    'normalized_degree': graph.degree(node) / (num_nodes - 1) if num_nodes > 1 else 0
                },
                community_id=None,
                clustering_coefficient=clustering_dict.get(node, 0),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Compile graph metrics
        graph_metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'is_connected': is_connected,
            'num_components': num_components,
            'diameter': diameter,
            'avg_path_length': avg_path_length,
            'avg_clustering': avg_clustering,
            'transitivity': transitivity,
            'avg_degree': avg_degree,
            'degree_std': degree_std,
            'max_degree': max_degree,
            'assortativity': assortativity,
            'small_world_coeff': small_world_coeff,
            'clustering_ratio': clustering_ratio,
            'path_length_ratio': path_length_ratio
        }
        
        # Generate insights
        insights = [
            f"Network has {num_nodes} nodes and {num_edges} edges",
            f"Network density: {density:.4f}",
            f"Average degree: {avg_degree:.2f}",
            f"Average clustering coefficient: {avg_clustering:.4f}",
            f"Network transitivity: {transitivity:.4f}",
        ]
        
        if is_connected:
            insights.append(f"Network is connected with diameter {diameter}")
            insights.append(f"Average shortest path length: {avg_path_length:.2f}")
        else:
            insights.append(f"Network has {num_components} connected components")
        
        # Small world analysis
        if small_world_coeff > 1:
            insights.append(f"Network exhibits small-world properties (coefficient: {small_world_coeff:.2f})")
        
        # Assortativity analysis
        if assortativity > 0.1:
            insights.append("Network shows assortative mixing (similar nodes connect)")
        elif assortativity < -0.1:
            insights.append("Network shows disassortative mixing (dissimilar nodes connect)")
        else:
            insights.append("Network shows neutral mixing patterns")
        
        recommendations = [
            "Monitor network density changes to track growth patterns",
            "Consider the small-world properties for information diffusion strategies",
            "Use clustering metrics to identify cohesive subgroups",
            "Leverage assortativity patterns for targeted interventions"
        ]
        
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
    
    async def _analyze_clustering(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze clustering patterns in the network."""
        
        self.logger.info("Analyzing clustering patterns...")
        
        # Calculate clustering coefficients
        clustering_dict = nx.clustering(graph)
        avg_clustering = nx.average_clustering(graph)
        transitivity = nx.transitivity(graph)
        
        # Square clustering (for 4-cycles)
        try:
            square_clustering = nx.square_clustering(graph)
        except Exception:
            square_clustering = {node: 0.0 for node in graph.nodes}
        
        # Find triangles
        triangles = nx.triangles(graph)
        
        # Clustering distribution analysis
        clustering_values = list(clustering_dict.values())
        clustering_stats = {
            'mean': np.mean(clustering_values),
            'std': np.std(clustering_values),
            'min': np.min(clustering_values),
            'max': np.max(clustering_values),
            'median': np.median(clustering_values)
        }
        
        # Identify highly clustered nodes
        high_clustering_threshold = np.percentile(clustering_values, 90)
        highly_clustered_nodes = [
            node for node, coeff in clustering_dict.items() 
            if coeff > high_clustering_threshold and coeff > 0.5
        ]
        
        # Create node metrics
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'clustering': clustering_dict.get(node, 0),
                    'square_clustering': square_clustering.get(node, 0),
                    'triangles': triangles.get(node, 0)
                },
                community_id=None,
                clustering_coefficient=clustering_dict.get(node, 0),
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Calculate graph metrics
        graph_metrics = {
            'avg_clustering': avg_clustering,
            'transitivity': transitivity,
            'clustering_std': clustering_stats['std'],
            'clustering_range': clustering_stats['max'] - clustering_stats['min'],
            'num_highly_clustered': len(highly_clustered_nodes),
            'total_triangles': sum(triangles.values()) // 3,  # Each triangle counted 3 times
            'clustering_heterogeneity': clustering_stats['std'] / clustering_stats['mean'] if clustering_stats['mean'] > 0 else 0
        }
        
        # Generate insights
        insights = [
            f"Average clustering coefficient: {avg_clustering:.4f}",
            f"Network transitivity: {transitivity:.4f}",
            f"Clustering heterogeneity: {graph_metrics['clustering_heterogeneity']:.4f}",
            f"Total triangles in network: {graph_metrics['total_triangles']}",
            f"Highly clustered nodes: {len(highly_clustered_nodes)}"
        ]
        
        # Compare to random graph
        try:
            random_clustering = 2 * graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
            clustering_enhancement = avg_clustering / random_clustering if random_clustering > 0 else 0
            insights.append(f"Clustering enhancement over random: {clustering_enhancement:.2f}x")
        except Exception:
            pass
        
        # Top clustered nodes
        top_clustered = sorted(clustering_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_clustered and top_clustered[0][1] > 0:
            insights.append("Most clustered nodes:")
            for node, coeff in top_clustered:
                node_title = graph.nodes[node].get('title', f'Node {node}')
                insights.append(f"  • {node_title}: clustering = {coeff:.4f}")
        
        recommendations = [
            "High clustering indicates strong local cohesion",
            "Monitor clustering changes to detect community formation",
            "Use triangular structures for robust information propagation",
            "Consider clustered nodes for local influence strategies"
        ]
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.CLUSTERING_ANALYSIS,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _analyze_paths(self, graph: nx.Graph) -> NetworkAnalysisResult:
        """Analyze path structures and distances in the network."""
        
        self.logger.info("Analyzing path structures...")
        
        # Check if graph is connected
        if not nx.is_connected(graph):
            # Work with largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            working_graph = graph.subgraph(largest_cc).copy()
        else:
            working_graph = graph
        
        num_nodes = working_graph.number_of_nodes()
        
        if num_nodes < 2:
            # Not enough nodes for path analysis
            return self._create_empty_result(AnalysisType.PATH_ANALYSIS)
        
        # Calculate shortest path metrics
        if num_nodes <= 1000:  # Limit for performance
            # All pairs shortest paths
            path_lengths = dict(nx.all_pairs_shortest_path_length(working_graph))
            
            # Extract all path lengths
            all_lengths = []
            for source, targets in path_lengths.items():
                for target, length in targets.items():
                    if source != target:
                        all_lengths.append(length)
            
            avg_path_length = np.mean(all_lengths) if all_lengths else 0
            max_path_length = max(all_lengths) if all_lengths else 0
            path_length_std = np.std(all_lengths) if all_lengths else 0
            
            # Diameter
            diameter = max_path_length
            
            # Radius and center
            eccentricities = nx.eccentricity(working_graph)
            radius = min(eccentricities.values()) if eccentricities else 0
            center_nodes = [node for node, ecc in eccentricities.items() if ecc == radius]
            
        else:
            # Sample-based analysis for large graphs
            sample_nodes = list(working_graph.nodes)[:100]
            path_lengths = {}
            all_lengths = []
            
            for node in sample_nodes:
                lengths = nx.single_source_shortest_path_length(working_graph, node)
                path_lengths[node] = lengths
                for target, length in lengths.items():
                    if node != target:
                        all_lengths.append(length)
            
            avg_path_length = np.mean(all_lengths) if all_lengths else 0
            max_path_length = max(all_lengths) if all_lengths else 0
            path_length_std = np.std(all_lengths) if all_lengths else 0
            diameter = max_path_length  # Approximation
            radius = 0  # Not calculated for large graphs
            center_nodes = []
            eccentricities = {}
        
        # Calculate efficiency (inverse of path length)
        if all_lengths:
            efficiency = np.mean([1.0 / length for length in all_lengths])
        else:
            efficiency = 0
        
        # Node-level path metrics
        node_metrics = []
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            # Calculate closeness centrality as path-based measure
            try:
                closeness = nx.closeness_centrality(working_graph, node) if node in working_graph else 0
            except Exception:
                closeness = 0
            
            # Eccentricity
            eccentricity = eccentricities.get(node, 0)
            
            metrics = NodeMetrics(
                node_id=node,
                centrality_scores={
                    'closeness': closeness,
                    'eccentricity': eccentricity,
                    'is_center': node in center_nodes
                },
                community_id=None,
                clustering_coefficient=0.0,  # Not relevant for path analysis
                degree=graph.degree(node),
                metadata=node_data
            )
            node_metrics.append(metrics)
        
        # Compile graph metrics
        graph_metrics = {
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'radius': radius,
            'efficiency': efficiency,
            'path_length_std': path_length_std,
            'num_center_nodes': len(center_nodes),
            'connected_component_size': num_nodes,
            'path_length_range': max_path_length - 1 if max_path_length > 0 else 0
        }
        
        # Generate insights
        insights = [
            f"Average shortest path length: {avg_path_length:.2f}",
            f"Network diameter: {diameter}",
            f"Network radius: {radius}",
            f"Network efficiency: {efficiency:.4f}",
            f"Path length standard deviation: {path_length_std:.2f}"
        ]
        
        if center_nodes:
            insights.append(f"Network has {len(center_nodes)} center node(s)")
            if len(center_nodes) <= 3:
                center_titles = []
                for node in center_nodes:
                    title = graph.nodes[node].get('title', f'Node {node}')
                    center_titles.append(title)
                insights.append(f"Center nodes: {', '.join(center_titles)}")
        
        # Path length distribution analysis
        if all_lengths:
            length_distribution = Counter(all_lengths)
            most_common_length = length_distribution.most_common(1)[0][0]
            insights.append(f"Most common path length: {most_common_length}")
        
        recommendations = [
            "Monitor diameter changes to track network expansion",
            "Use center nodes for optimal information placement",
            "Consider efficiency metrics for network optimization",
            "Short average path length indicates good connectivity"
        ]
        
        return NetworkAnalysisResult(
            analysis_type=AnalysisType.PATH_ANALYSIS,
            graph_metrics=graph_metrics,
            node_metrics=node_metrics,
            communities=[],
            insights=insights,
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    def _create_temporal_directed_graph(self, graph: nx.Graph) -> nx.DiGraph:
        """Create directed graph based on temporal information."""
        
        directed_graph = nx.DiGraph()
        
        # Add all nodes
        for node, data in graph.nodes(data=True):
            directed_graph.add_node(node, **data)
        
        # Add directed edges based on dates
        for u, v, data in graph.edges(data=True):
            u_date = graph.nodes[u].get('date')
            v_date = graph.nodes[v].get('date')
            
            if u_date and v_date:
                try:
                    # Convert to datetime if needed
                    if isinstance(u_date, str):
                        u_datetime = datetime.fromisoformat(u_date.replace('Z', '+00:00'))
                    else:
                        u_datetime = u_date
                    
                    if isinstance(v_date, str):
                        v_datetime = datetime.fromisoformat(v_date.replace('Z', '+00:00'))
                    else:
                        v_datetime = v_date
                    
                    # Add edge from older to newer
                    if u_datetime < v_datetime:
                        directed_graph.add_edge(u, v, **data)
                    elif v_datetime < u_datetime:
                        directed_graph.add_edge(v, u, **data)
                    else:
                        # Same date, add both directions
                        directed_graph.add_edge(u, v, **data)
                        directed_graph.add_edge(v, u, **data)
                        
                except Exception:
                    # If date parsing fails, add both directions
                    directed_graph.add_edge(u, v, **data)
                    directed_graph.add_edge(v, u, **data)
            else:
                # No date information, add both directions
                directed_graph.add_edge(u, v, **data)
                directed_graph.add_edge(v, u, **data)
        
        return directed_graph
    
    def _calculate_graph_metrics(
        self,
        graph: nx.Graph,
        centrality_measures: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate graph-level metrics."""
        
        metrics = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph),
            'transitivity': nx.transitivity(graph)
        }
        
        # Add centrality statistics
        for measure_name, scores in centrality_measures.items():
            if scores:
                values = list(scores.values())
                metrics[f'{measure_name}_mean'] = np.mean(values)
                metrics[f'{measure_name}_std'] = np.std(values)
                metrics[f'{measure_name}_max'] = np.max(values)
        
        return metrics
    
    def _generate_centrality_insights(
        self,
        graph: nx.Graph,
        centrality_measures: Dict[str, Dict[str, float]],
        node_metrics: List[NodeMetrics]
    ) -> List[str]:
        """Generate insights from centrality analysis."""
        
        insights = []
        
        # General network insights
        insights.append(f"Network has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Centrality insights
        for measure_name, scores in centrality_measures.items():
            if scores:
                top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"Top {measure_name} nodes:")
                for node, score in top_nodes:
                    node_title = graph.nodes[node].get('title', f'Node {node}')
                    insights.append(f"  • {node_title}: {score:.4f}")
        
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
                                insights.append(f"Strong positive correlation between {measure1} and {measure2} ({correlation:.3f})")
                            elif correlation < -0.7:
                                insights.append(f"Strong negative correlation between {measure1} and {measure2} ({correlation:.3f})")
        
        return insights
    
    def _generate_centrality_recommendations(
        self,
        centrality_measures: Dict[str, Dict[str, float]],
        node_metrics: List[NodeMetrics]
    ) -> List[str]:
        """Generate recommendations from centrality analysis."""
        
        recommendations = [
            "Monitor high PageRank nodes for influence and authority",
            "Use high betweenness nodes for information flow control",
            "Target high closeness nodes for rapid information dissemination",
            "Consider eigenvector centrality for identifying prestige networks"
        ]
        
        # Add specific recommendations based on centrality distribution
        if 'pagerank' in centrality_measures:
            pagerank_values = list(centrality_measures['pagerank'].values())
            pagerank_concentration = np.std(pagerank_values) / np.mean(pagerank_values) if np.mean(pagerank_values) > 0 else 0
            
            if pagerank_concentration > 1.0:
                recommendations.append("High PageRank concentration suggests hierarchical structure")
            else:
                recommendations.append("Distributed PageRank suggests egalitarian structure")
        
        return recommendations
    
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
        
        return description
    
    def _generate_community_insights(
        self,
        graph: nx.Graph,
        communities: List[CommunityInfo],
        graph_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate insights from community analysis."""
        
        insights = []
        
        insights.append(f"Detected {len(communities)} communities")
        insights.append(f"Modularity score: {graph_metrics.get('modularity', 0):.3f}")
        
        if communities:
            sizes = [c.size for c in communities]
            insights.append(f"Average community size: {np.mean(sizes):.1f}")
            insights.append(f"Largest community size: {max(sizes)}")
            insights.append(f"Smallest community size: {min(sizes)}")
            
            # Community quality insights
            avg_internal_edges = np.mean([c.internal_edges for c in communities])
            avg_external_edges = np.mean([c.external_edges for c in communities])
            
            if avg_internal_edges > avg_external_edges:
                insights.append("Communities show strong internal cohesion")
            else:
                insights.append("Communities show weak internal cohesion")
        
        return insights
    
    def _generate_community_recommendations(
        self,
        communities: List[CommunityInfo],
        graph_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations from community analysis."""
        
        recommendations = [
            "Use community structure for targeted interventions",
            "Monitor community evolution over time",
            "Consider inter-community bridges for integration strategies"
        ]
        
        modularity = graph_metrics.get('modularity', 0)
        if modularity > 0.3:
            recommendations.append("High modularity suggests well-defined communities")
        elif modularity < 0.1:
            recommendations.append("Low modularity suggests weak community structure")
        
        return recommendations
    
    def _create_empty_result(self, analysis_type: AnalysisType) -> NetworkAnalysisResult:
        """Create empty result for cases with insufficient data."""
        
        return NetworkAnalysisResult(
            analysis_type=analysis_type,
            graph_metrics={},
            node_metrics=[],
            communities=[],
            insights=["Insufficient data for analysis"],
            recommendations=[],
            generated_at=datetime.now(timezone.utc),
            execution_time=0.0
        )
    
    async def _generate_network_visualization(
        self,
        graph: nx.Graph,
        analysis_result: NetworkAnalysisResult
    ) -> str:
        """Generate network visualization."""
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Limit nodes for visualization performance
            if len(graph.nodes) > 500:
                # Sample nodes for visualization
                sampled_nodes = list(graph.nodes)[:500]
                vis_graph = graph.subgraph(sampled_nodes)
            else:
                vis_graph = graph
            
            # Calculate layout
            if len(vis_graph.nodes) > 100:
                pos = nx.spring_layout(vis_graph, k=0.5, iterations=20)
            else:
                pos = nx.spring_layout(vis_graph, k=1, iterations=50)
            
            # Node colors based on analysis type
            if analysis_result.analysis_type == AnalysisType.CENTRALITY and analysis_result.node_metrics:
                # Color by PageRank or first centrality measure
                node_colors = []
                centrality_key = 'pagerank' if 'pagerank' in analysis_result.node_metrics[0].centrality_scores else list(analysis_result.node_metrics[0].centrality_scores.keys())[0]
                
                for node in vis_graph.nodes:
                    node_metric = next((m for m in analysis_result.node_metrics if m.node_id == node), None)
                    if node_metric:
                        score = node_metric.centrality_scores.get(centrality_key, 0)
                        node_colors.append(score)
                    else:
                        node_colors.append(0)
            
            elif analysis_result.analysis_type == AnalysisType.COMMUNITY_DETECTION:
                # Color by community
                node_colors = []
                for node in vis_graph.nodes:
                    node_metric = next((m for m in analysis_result.node_metrics if m.node_id == node), None)
                    if node_metric and node_metric.community_id is not None:
                        node_colors.append(node_metric.community_id)
                    else:
                        node_colors.append(-1)
            
            else:
                # Default coloring by degree
                node_colors = [vis_graph.degree(node) for node in vis_graph.nodes]
            
            # Node sizes based on degree
            node_sizes = [self.config.node_size_multiplier * (1 + vis_graph.degree(node) / 10) for node in vis_graph.nodes]
            
            # Draw the network
            nx.draw_networkx_nodes(
                vis_graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.7,
                cmap=plt.cm.viridis
            )
            
            nx.draw_networkx_edges(
                vis_graph, pos,
                alpha=0.3,
                width=0.5
            )
            
            # Add labels for high-degree nodes
            high_degree_nodes = [node for node in vis_graph.nodes if vis_graph.degree(node) > np.percentile([vis_graph.degree(n) for n in vis_graph.nodes], 90)]
            high_degree_labels = {node: vis_graph.nodes[node].get('title', str(node))[:10] for node in high_degree_nodes}
            
            nx.draw_networkx_labels(
                vis_graph, pos,
                labels=high_degree_labels,
                font_size=8,
                font_weight='bold'
            )
            
            ax.set_title(f"Network Analysis: {analysis_result.analysis_type.value.replace('_', ' ').title()}")
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"network_{analysis_result.analysis_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Network visualization saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error generating network visualization: {e}")
            return ""


# CLI Interface
async def main():
    """Main CLI interface for network analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Network Analyzer")
    parser.add_argument("--analysis-type", choices=[t.value for t in AnalysisType], 
                       required=True, help="Type of analysis to perform")
    parser.add_argument("--domains", nargs='+', help="Domain scope for analysis")
    parser.add_argument("--node-types", nargs='+', help="Node types to include")
    parser.add_argument("--relationship-types", nargs='+', help="Relationship types to include")
    parser.add_argument("--min-connections", type=int, default=1, help="Minimum connections per node")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    config = NetworkConfig(args.config) if args.config else NetworkConfig()
    analyzer = NetworkAnalyzer(config)
    
    await analyzer.initialize()
    
    try:
        # Run analysis
        analysis_type = AnalysisType(args.analysis_type)
        result = await analyzer.analyze_network(
            analysis_type,
            args.domains,
            args.node_types,
            args.relationship_types,
            args.min_connections
        )
        
        # Display results
        print(f"\n=== Network Analysis: {analysis_type.value} ===")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        print(f"\nGraph Metrics:")
        for key, value in result.graph_metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nInsights:")
        for insight in result.insights:
            print(f"  • {insight}")
        
        print(f"\nRecommendations:")
        for recommendation in result.recommendations:
            print(f"  • {recommendation}")
        
        if result.communities:
            print(f"\nCommunities ({len(result.communities)}):")
            for community in result.communities[:5]:  # Show first 5
                print(f"  Community {community.community_id}: {community.size} nodes - {community.description}")
        
        print(f"\nTop nodes by centrality/importance:")
        sorted_nodes = sorted(result.node_metrics, key=lambda x: x.degree, reverse=True)
        for node in sorted_nodes[:5]:
            print(f"  {node.node_id}: degree={node.degree}, clustering={node.clustering_coefficient:.3f}")
    
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())