"""
Core Network Analyzer - Main orchestrator for network analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import networkx as nx
from neo4j import AsyncGraphDatabase, AsyncDriver

from ..models import AnalysisType, NetworkAnalysisResult
from ..config import NetworkConfig
from ..graph_utils import GraphLoader, GraphValidator, GraphMetricsAggregator

from .centrality_analysis import CentralityAnalyzer
from .community_detection import CommunityDetector
from .influence_analysis import InfluenceAnalyzer
from .bridge_analysis import BridgeAnalyzer
from .flow_analysis import FlowAnalyzer
from .structural_analysis import StructuralAnalyzer
from .clustering_analysis import ClusteringAnalyzer
from .path_analysis import PathAnalyzer
from .network_visualization import NetworkVisualizer


class NetworkAnalyzer:
    """Main network analysis orchestrator."""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize the network analyzer."""
        self.config = config or NetworkConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        
        # Initialize utilities
        self.graph_loader = GraphLoader(self.config)
        self.graph_validator = GraphValidator(self.config)
        self.metrics_aggregator = GraphMetricsAggregator(self.config)
        
        # Initialize specialized analyzers
        self.centrality_analyzer = CentralityAnalyzer(self.config)
        self.community_detector = CommunityDetector(self.config)
        self.influence_analyzer = InfluenceAnalyzer(self.config)
        self.bridge_analyzer = BridgeAnalyzer(self.config)
        self.flow_analyzer = FlowAnalyzer(self.config)
        self.structural_analyzer = StructuralAnalyzer(self.config)
        self.clustering_analyzer = ClusteringAnalyzer(self.config)
        self.path_analyzer = PathAnalyzer(self.config)
        self.visualizer = NetworkVisualizer(self.config)
        
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
        
        if not logger.handlers:
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
            graph = await self.graph_loader.load_graph(
                self.neo4j_driver,
                domain_scope,
                node_types,
                relationship_types,
                min_connections
            )
            
            # Validate graph
            is_valid, issues = self.graph_validator.validate_graph(graph)
            if not is_valid:
                raise ValueError(f"Graph validation failed: {'; '.join(issues)}")
            
            self.logger.info(f"Loaded and validated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Route to appropriate analyzer
            if analysis_type == AnalysisType.CENTRALITY:
                result = await self.centrality_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.COMMUNITY_DETECTION:
                result = await self.community_detector.analyze(graph)
            elif analysis_type == AnalysisType.INFLUENCE_PROPAGATION:
                result = await self.influence_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.KNOWLEDGE_FLOW:
                result = await self.flow_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.BRIDGE_NODES:
                result = await self.bridge_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.STRUCTURAL_ANALYSIS:
                result = await self.structural_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.CLUSTERING_ANALYSIS:
                result = await self.clustering_analyzer.analyze(graph)
            elif analysis_type == AnalysisType.PATH_ANALYSIS:
                result = await self.path_analyzer.analyze(graph)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.generated_at = datetime.now(timezone.utc)
            
            # Generate visualization if configured
            if self.config.generate_plots:
                await self.visualizer.generate_network_visualization(graph, result)
            
            self.logger.info(f"Network analysis completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in network analysis: {e}")
            raise
    
    async def analyze_multiple_types(
        self,
        analysis_types: List[AnalysisType],
        domain_scope: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        min_connections: int = 1
    ) -> Dict[AnalysisType, NetworkAnalysisResult]:
        """Perform multiple analysis types on the same graph."""
        
        results = {}
        
        # Load graph once for all analyses
        graph = await self.graph_loader.load_graph(
            self.neo4j_driver,
            domain_scope,
            node_types,
            relationship_types,
            min_connections
        )
        
        # Validate graph
        is_valid, issues = self.graph_validator.validate_graph(graph)
        if not is_valid:
            raise ValueError(f"Graph validation failed: {'; '.join(issues)}")
        
        # Run each analysis
        for analysis_type in analysis_types:
            try:
                start_time = datetime.now()
                
                if analysis_type == AnalysisType.CENTRALITY:
                    result = await self.centrality_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.COMMUNITY_DETECTION:
                    result = await self.community_detector.analyze(graph)
                elif analysis_type == AnalysisType.INFLUENCE_PROPAGATION:
                    result = await self.influence_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.KNOWLEDGE_FLOW:
                    result = await self.flow_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.BRIDGE_NODES:
                    result = await self.bridge_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.STRUCTURAL_ANALYSIS:
                    result = await self.structural_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.CLUSTERING_ANALYSIS:
                    result = await self.clustering_analyzer.analyze(graph)
                elif analysis_type == AnalysisType.PATH_ANALYSIS:
                    result = await self.path_analyzer.analyze(graph)
                else:
                    self.logger.warning(f"Unsupported analysis type: {analysis_type}")
                    continue
                
                # Set timing information
                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time = execution_time
                result.generated_at = datetime.now(timezone.utc)
                
                results[analysis_type] = result
                
            except Exception as e:
                self.logger.error(f"Error in {analysis_type.value} analysis: {e}")
                continue
        
        # Generate comprehensive visualization if configured
        if self.config.generate_plots and results:
            await self.visualizer.generate_multi_analysis_visualization(graph, results)
        
        return results
    
    def get_cached_graph(self, cache_key: str) -> Optional[nx.Graph]:
        """Get cached graph if available."""
        return self.cached_graphs.get(cache_key)
    
    def cache_graph(self, cache_key: str, graph: nx.Graph) -> None:
        """Cache graph for future use."""
        self.cached_graphs[cache_key] = graph
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cached_graphs.clear()
        self.cached_metrics.clear()
        self.logger.info("Cache cleared")