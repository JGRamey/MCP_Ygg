"""
Main chart generator orchestrator for MCP Yggdrasil visualizations.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List

from neo4j import AsyncGraphDatabase, AsyncDriver

from .config import VisualizationConfig
from .models import VisualizationType
from ..processors.yggdrasil_processor import YggdrasilProcessor
from ..processors.network_processor import NetworkProcessor
from ..layouts.yggdrasil_layout import YggdrasilLayout
from ..layouts.force_layout import ForceLayout
from ..exporters.html_exporter import HTMLExporter


class ChartGenerator:
    """Main chart generation orchestrator."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the chart generator."""
        self.config = config or VisualizationConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        
        # Initialize components
        self.html_exporter = HTMLExporter(self.config)
        self.yggdrasil_layout = YggdrasilLayout(self.config)
        self.force_layout = ForceLayout(self.config)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("chart_generator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
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
            
            self.logger.info("Chart generator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chart generator: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        self.logger.info("Chart generator closed")
    
    async def generate_yggdrasil_chart(
        self,
        title: str = "Yggdrasil Knowledge Tree",
        domain_filter: Optional[str] = None,
        max_depth: int = 5,
        export_format: str = "html"
    ) -> str:
        """Generate the main Yggdrasil tree visualization."""
        
        try:
            # Get data using processor
            processor = YggdrasilProcessor(self.config, self.neo4j_driver)
            viz_data = await processor.get_data(domain_filter, max_depth)
            
            # Apply layout
            self.yggdrasil_layout.apply_layout(viz_data)
            layout_params = self.yggdrasil_layout.get_layout_parameters()
            
            # Export based on format
            if export_format == "html":
                output_file = self.html_exporter.export_html(
                    viz_data, 
                    title, 
                    VisualizationType.YGGDRASIL_TREE,
                    layout_params
                )
            elif export_format == "svg":
                output_file = self.html_exporter.export_svg(
                    viz_data, 
                    title, 
                    VisualizationType.YGGDRASIL_TREE
                )
            elif export_format == "png":
                output_file = self.html_exporter.export_png(
                    viz_data, 
                    title, 
                    VisualizationType.YGGDRASIL_TREE
                )
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            self.logger.info(f"Generated Yggdrasil chart: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating Yggdrasil chart: {e}")
            raise
    
    async def generate_network_graph(
        self,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        title: str = "Knowledge Network",
        export_format: str = "html"
    ) -> str:
        """Generate a general network graph visualization."""
        
        try:
            # Get data using processor
            processor = NetworkProcessor(self.config, self.neo4j_driver)
            viz_data = await processor.get_data(node_types, relationship_types)
            
            # Apply layout
            self.force_layout.apply_layout(viz_data)
            layout_params = self.force_layout.get_layout_parameters()
            
            # Export based on format
            if export_format == "html":
                output_file = self.html_exporter.export_html(
                    viz_data, 
                    title, 
                    VisualizationType.NETWORK_GRAPH,
                    layout_params
                )
            elif export_format == "svg":
                output_file = self.html_exporter.export_svg(
                    viz_data, 
                    title, 
                    VisualizationType.NETWORK_GRAPH
                )
            elif export_format == "png":
                output_file = self.html_exporter.export_png(
                    viz_data, 
                    title, 
                    VisualizationType.NETWORK_GRAPH
                )
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            self.logger.info(f"Generated network graph: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating network graph: {e}")
            raise