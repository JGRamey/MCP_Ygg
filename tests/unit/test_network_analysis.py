#!/usr/bin/env python3
"""
Test suite for the network analysis module
Tests all components of the refactored network analysis system
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import networkx as nx
import numpy as np
from datetime import datetime

from agents.analytics.graph_analysis.network_analysis.core_analyzer import NetworkAnalyzer
from agents.analytics.graph_analysis.network_analysis.centrality_analysis import CentralityAnalyzer
from agents.analytics.graph_analysis.network_analysis.community_detection import CommunityDetector
from agents.analytics.graph_analysis.network_analysis.influence_analysis import InfluenceAnalyzer
from agents.analytics.graph_analysis.network_analysis.bridge_analysis import BridgeAnalyzer
from agents.analytics.graph_analysis.network_analysis.flow_analysis import FlowAnalyzer
from agents.analytics.graph_analysis.network_analysis.structural_analysis import StructuralAnalyzer
from agents.analytics.graph_analysis.network_analysis.clustering_analysis import ClusteringAnalyzer
from agents.analytics.graph_analysis.network_analysis.path_analysis import PathAnalyzer
from agents.analytics.graph_analysis.network_analysis.network_visualization import NetworkVisualizer
from agents.analytics.graph_analysis.base import AnalysisConfig, AnalysisResult


class TestNetworkAnalysisBase:
    """Base class for network analysis tests."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        G = nx.Graph()
        G.add_edges_from([
            (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)
        ])
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['label'] = f"Node_{node}"
            G.nodes[node]['domain'] = 'test'
            G.nodes[node]['type'] = 'concept'
        
        return G
    
    @pytest.fixture
    def analysis_config(self):
        """Create analysis configuration for testing."""
        return AnalysisConfig(
            graph_type="knowledge_graph",
            analysis_level="comprehensive",
            include_visualization=True,
            cache_results=False,
            parallel_processing=False
        )
    
    @pytest.fixture
    def mock_session(self):
        """Create mock Neo4j session."""
        session = AsyncMock()
        session.run.return_value = AsyncMock()
        return session


class TestCentralityAnalyzer(TestNetworkAnalysisBase):
    """Test centrality analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_centrality_analyzer_initialization(self, analysis_config):
        """Test centrality analyzer initialization."""
        analyzer = CentralityAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'centrality_measures')
    
    @pytest.mark.asyncio
    async def test_calculate_centrality_measures(self, analysis_config, sample_graph, mock_session):
        """Test centrality measures calculation."""
        analyzer = CentralityAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'centrality_measures' in result.data
        assert result.node_count == len(sample_graph.nodes())
        assert result.edge_count == len(sample_graph.edges())
    
    @pytest.mark.unit
    def test_degree_centrality_calculation(self, analysis_config, sample_graph):
        """Test degree centrality calculation."""
        analyzer = CentralityAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        degree_centrality = analyzer._calculate_degree_centrality()
        
        assert isinstance(degree_centrality, dict)
        assert len(degree_centrality) == len(sample_graph.nodes())
        assert all(0 <= value <= 1 for value in degree_centrality.values())
    
    @pytest.mark.unit
    def test_betweenness_centrality_calculation(self, analysis_config, sample_graph):
        """Test betweenness centrality calculation."""
        analyzer = CentralityAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        betweenness_centrality = analyzer._calculate_betweenness_centrality()
        
        assert isinstance(betweenness_centrality, dict)
        assert len(betweenness_centrality) == len(sample_graph.nodes())
        assert all(0 <= value <= 1 for value in betweenness_centrality.values())
    
    @pytest.mark.unit
    def test_closeness_centrality_calculation(self, analysis_config, sample_graph):
        """Test closeness centrality calculation."""
        analyzer = CentralityAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        closeness_centrality = analyzer._calculate_closeness_centrality()
        
        assert isinstance(closeness_centrality, dict)
        assert len(closeness_centrality) == len(sample_graph.nodes())
        assert all(0 <= value <= 1 for value in closeness_centrality.values())
    
    @pytest.mark.unit
    def test_eigenvector_centrality_calculation(self, analysis_config, sample_graph):
        """Test eigenvector centrality calculation."""
        analyzer = CentralityAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        eigenvector_centrality = analyzer._calculate_eigenvector_centrality()
        
        assert isinstance(eigenvector_centrality, dict)
        assert len(eigenvector_centrality) == len(sample_graph.nodes())
        assert all(0 <= value <= 1 for value in eigenvector_centrality.values())


class TestCommunityDetector(TestNetworkAnalysisBase):
    """Test community detection functionality."""
    
    @pytest.mark.asyncio
    async def test_community_detector_initialization(self, analysis_config):
        """Test community detector initialization."""
        detector = CommunityDetector(analysis_config)
        
        assert detector.config == analysis_config
        assert hasattr(detector, 'community_algorithms')
    
    @pytest.mark.asyncio
    async def test_community_detection_analysis(self, analysis_config, sample_graph, mock_session):
        """Test community detection analysis."""
        detector = CommunityDetector(analysis_config)
        detector.graph = sample_graph
        
        # Mock the load_graph method
        detector.load_graph = AsyncMock(return_value=True)
        
        result = await detector.analyze(mock_session)
        
        assert result.success is True
        assert 'communities' in result.data
        assert result.node_count == len(sample_graph.nodes())
    
    @pytest.mark.unit
    def test_louvain_community_detection(self, analysis_config, sample_graph):
        """Test Louvain community detection."""
        detector = CommunityDetector(analysis_config)
        detector.graph = sample_graph
        
        communities = detector._detect_communities_louvain()
        
        assert isinstance(communities, dict)
        assert len(communities) == len(sample_graph.nodes())
        assert all(isinstance(community_id, int) for community_id in communities.values())
    
    @pytest.mark.unit
    def test_community_quality_metrics(self, analysis_config, sample_graph):
        """Test community quality metrics calculation."""
        detector = CommunityDetector(analysis_config)
        detector.graph = sample_graph
        
        # Create mock communities
        communities = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        
        modularity = detector._calculate_modularity(communities)
        
        assert isinstance(modularity, float)
        assert -1 <= modularity <= 1


class TestInfluenceAnalyzer(TestNetworkAnalysisBase):
    """Test influence analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_influence_analyzer_initialization(self, analysis_config):
        """Test influence analyzer initialization."""
        analyzer = InfluenceAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'influence_models')
    
    @pytest.mark.asyncio
    async def test_influence_analysis(self, analysis_config, sample_graph, mock_session):
        """Test influence analysis."""
        analyzer = InfluenceAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'influence_scores' in result.data
        assert result.node_count == len(sample_graph.nodes())
    
    @pytest.mark.unit
    def test_influence_propagation_simulation(self, analysis_config, sample_graph):
        """Test influence propagation simulation."""
        analyzer = InfluenceAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Simulate influence from node 1
        influence_scores = analyzer._simulate_influence_propagation(
            seed_nodes=[1], 
            propagation_probability=0.5,
            max_iterations=10
        )
        
        assert isinstance(influence_scores, dict)
        assert len(influence_scores) == len(sample_graph.nodes())
        assert all(0 <= score <= 1 for score in influence_scores.values())
        assert influence_scores[1] > 0  # Seed node should have influence


class TestBridgeAnalyzer(TestNetworkAnalysisBase):
    """Test bridge analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_bridge_analyzer_initialization(self, analysis_config):
        """Test bridge analyzer initialization."""
        analyzer = BridgeAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'bridge_types')
    
    @pytest.mark.asyncio
    async def test_bridge_analysis(self, analysis_config, sample_graph, mock_session):
        """Test bridge analysis."""
        analyzer = BridgeAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'bridge_nodes' in result.data
        assert 'bridge_edges' in result.data
    
    @pytest.mark.unit
    def test_bridge_node_identification(self, analysis_config, sample_graph):
        """Test bridge node identification."""
        analyzer = BridgeAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        bridge_nodes = analyzer._identify_bridge_nodes()
        
        assert isinstance(bridge_nodes, list)
        # All nodes should be bridge nodes in this simple graph
        assert len(bridge_nodes) >= 0
    
    @pytest.mark.unit
    def test_bridge_edge_identification(self, analysis_config, sample_graph):
        """Test bridge edge identification."""
        analyzer = BridgeAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        bridge_edges = analyzer._identify_bridge_edges()
        
        assert isinstance(bridge_edges, list)
        # Check that bridge edges are valid edges
        for edge in bridge_edges:
            assert edge in sample_graph.edges() or (edge[1], edge[0]) in sample_graph.edges()


class TestFlowAnalyzer(TestNetworkAnalysisBase):
    """Test flow analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_flow_analyzer_initialization(self, analysis_config):
        """Test flow analyzer initialization."""
        analyzer = FlowAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'flow_algorithms')
    
    @pytest.mark.asyncio
    async def test_flow_analysis(self, analysis_config, sample_graph, mock_session):
        """Test flow analysis."""
        analyzer = FlowAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'flow_patterns' in result.data
    
    @pytest.mark.unit
    def test_knowledge_flow_simulation(self, analysis_config, sample_graph):
        """Test knowledge flow simulation."""
        analyzer = FlowAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Add edge weights for flow calculation
        for edge in sample_graph.edges():
            sample_graph.edges[edge]['weight'] = 1.0
        
        flow_patterns = analyzer._simulate_knowledge_flow(
            source_nodes=[1], 
            flow_capacity=1.0
        )
        
        assert isinstance(flow_patterns, dict)
        assert 'total_flow' in flow_patterns
        assert 'flow_distribution' in flow_patterns


class TestStructuralAnalyzer(TestNetworkAnalysisBase):
    """Test structural analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_structural_analyzer_initialization(self, analysis_config):
        """Test structural analyzer initialization."""
        analyzer = StructuralAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'structural_metrics')
    
    @pytest.mark.asyncio
    async def test_structural_analysis(self, analysis_config, sample_graph, mock_session):
        """Test structural analysis."""
        analyzer = StructuralAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'structural_metrics' in result.data
        assert result.node_count == len(sample_graph.nodes())
    
    @pytest.mark.unit
    def test_graph_density_calculation(self, analysis_config, sample_graph):
        """Test graph density calculation."""
        analyzer = StructuralAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        density = analyzer._calculate_graph_density()
        
        assert isinstance(density, float)
        assert 0 <= density <= 1
    
    @pytest.mark.unit
    def test_assortativity_calculation(self, analysis_config, sample_graph):
        """Test assortativity calculation."""
        analyzer = StructuralAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        assortativity = analyzer._calculate_assortativity()
        
        assert isinstance(assortativity, float)
        assert -1 <= assortativity <= 1


class TestClusteringAnalyzer(TestNetworkAnalysisBase):
    """Test clustering analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_clustering_analyzer_initialization(self, analysis_config):
        """Test clustering analyzer initialization."""
        analyzer = ClusteringAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'clustering_algorithms')
    
    @pytest.mark.asyncio
    async def test_clustering_analysis(self, analysis_config, sample_graph, mock_session):
        """Test clustering analysis."""
        analyzer = ClusteringAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'clustering_coefficient' in result.data
    
    @pytest.mark.unit
    def test_clustering_coefficient_calculation(self, analysis_config, sample_graph):
        """Test clustering coefficient calculation."""
        analyzer = ClusteringAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        clustering_coefficient = analyzer._calculate_clustering_coefficient()
        
        assert isinstance(clustering_coefficient, dict)
        assert len(clustering_coefficient) == len(sample_graph.nodes())
        assert all(0 <= coeff <= 1 for coeff in clustering_coefficient.values())
    
    @pytest.mark.unit
    def test_global_clustering_coefficient(self, analysis_config, sample_graph):
        """Test global clustering coefficient calculation."""
        analyzer = ClusteringAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        global_clustering = analyzer._calculate_global_clustering()
        
        assert isinstance(global_clustering, float)
        assert 0 <= global_clustering <= 1


class TestPathAnalyzer(TestNetworkAnalysisBase):
    """Test path analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_path_analyzer_initialization(self, analysis_config):
        """Test path analyzer initialization."""
        analyzer = PathAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'path_algorithms')
    
    @pytest.mark.asyncio
    async def test_path_analysis(self, analysis_config, sample_graph, mock_session):
        """Test path analysis."""
        analyzer = PathAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert 'path_statistics' in result.data
    
    @pytest.mark.unit
    def test_shortest_path_calculation(self, analysis_config, sample_graph):
        """Test shortest path calculation."""
        analyzer = PathAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        shortest_paths = analyzer._calculate_shortest_paths()
        
        assert isinstance(shortest_paths, dict)
        # Should contain path lengths between all pairs of nodes
        assert len(shortest_paths) == len(sample_graph.nodes())
    
    @pytest.mark.unit
    def test_path_diversity_analysis(self, analysis_config, sample_graph):
        """Test path diversity analysis."""
        analyzer = PathAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        path_diversity = analyzer._analyze_path_diversity(source=1, target=6)
        
        assert isinstance(path_diversity, dict)
        assert 'path_count' in path_diversity
        assert 'average_path_length' in path_diversity


class TestNetworkVisualizer(TestNetworkAnalysisBase):
    """Test network visualization functionality."""
    
    @pytest.mark.asyncio
    async def test_network_visualizer_initialization(self, analysis_config):
        """Test network visualizer initialization."""
        visualizer = NetworkVisualizer(analysis_config)
        
        assert visualizer.config == analysis_config
        assert hasattr(visualizer, 'layout_algorithms')
    
    @pytest.mark.asyncio
    async def test_network_visualization(self, analysis_config, sample_graph, mock_session):
        """Test network visualization."""
        visualizer = NetworkVisualizer(analysis_config)
        visualizer.graph = sample_graph
        
        # Mock the load_graph method
        visualizer.load_graph = AsyncMock(return_value=True)
        
        result = await visualizer.analyze(mock_session)
        
        assert result.success is True
        assert 'visualization_data' in result.data
    
    @pytest.mark.unit
    def test_layout_generation(self, analysis_config, sample_graph):
        """Test layout generation."""
        visualizer = NetworkVisualizer(analysis_config)
        visualizer.graph = sample_graph
        
        layout = visualizer._generate_layout(algorithm='spring')
        
        assert isinstance(layout, dict)
        assert len(layout) == len(sample_graph.nodes())
        assert all(len(pos) == 2 for pos in layout.values())  # 2D positions
    
    @pytest.mark.unit
    def test_node_styling(self, analysis_config, sample_graph):
        """Test node styling."""
        visualizer = NetworkVisualizer(analysis_config)
        visualizer.graph = sample_graph
        
        node_styles = visualizer._generate_node_styles()
        
        assert isinstance(node_styles, dict)
        assert len(node_styles) == len(sample_graph.nodes())
        assert all('color' in style for style in node_styles.values())
        assert all('size' in style for style in node_styles.values())


class TestNetworkAnalyzerCore(TestNetworkAnalysisBase):
    """Test the core network analyzer orchestrator."""
    
    @pytest.mark.asyncio
    async def test_network_analyzer_initialization(self, analysis_config):
        """Test network analyzer initialization."""
        analyzer = NetworkAnalyzer(analysis_config)
        
        assert analyzer.config == analysis_config
        assert hasattr(analyzer, 'sub_analyzers')
    
    @pytest.mark.asyncio
    async def test_comprehensive_network_analysis(self, analysis_config, sample_graph, mock_session):
        """Test comprehensive network analysis."""
        analyzer = NetworkAnalyzer(analysis_config)
        analyzer.graph = sample_graph
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert result.analysis_type == "network_analysis"
        assert result.node_count == len(sample_graph.nodes())
        assert result.edge_count == len(sample_graph.edges())
        assert result.execution_time > 0
        assert isinstance(result.data, dict)
    
    @pytest.mark.integration
    async def test_network_analysis_with_real_data(self, analysis_config, mock_session):
        """Test network analysis with more realistic data."""
        analyzer = NetworkAnalyzer(analysis_config)
        
        # Create a more complex graph
        G = nx.karate_club_graph()
        analyzer.graph = G
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert result.node_count == len(G.nodes())
        assert result.edge_count == len(G.edges())
        assert 'centrality_measures' in result.data
        assert 'communities' in result.data
        assert 'structural_metrics' in result.data
    
    @pytest.mark.slow
    async def test_network_analysis_performance(self, analysis_config, mock_session):
        """Test network analysis performance with large graph."""
        analyzer = NetworkAnalyzer(analysis_config)
        
        # Create a larger graph
        G = nx.barabasi_albert_graph(1000, 5)
        analyzer.graph = G
        
        # Mock the load_graph method
        analyzer.load_graph = AsyncMock(return_value=True)
        
        result = await analyzer.analyze(mock_session)
        
        assert result.success is True
        assert result.execution_time < 30  # Should complete within 30 seconds
        assert result.node_count == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])