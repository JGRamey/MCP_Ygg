"""
Network Visualization Module for Graph Analysis
Provides comprehensive visualization capabilities for network analysis results.
"""

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from ..models import NetworkAnalysisResult, NodeMetrics, AnalysisType


class NetworkVisualizer:
    """Comprehensive network visualization engine."""
    
    def __init__(self, config=None):
        """Initialize the network visualizer."""
        self.config = config
        self.logger = self._setup_logging()
        
        # Visualization settings
        self.default_figure_size = (12, 8)
        self.default_node_size_multiplier = 300
        self.default_plot_dir = Path("analytics/plots")
        
        # Create plot directory
        self.plot_dir = Path(getattr(config, 'plot_dir', self.default_plot_dir))
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup matplotlib/seaborn
        self._setup_plotting_style()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("network_visualizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _setup_plotting_style(self) -> None:
        """Setup plotting style and suppress warnings."""
        try:
            # Suppress matplotlib warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            
            # Try to use seaborn style
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    # Fallback to default style
                    plt.style.use('default')
                    self.logger.info("Using default matplotlib style")
            
            # Set color palette
            sns.set_palette("viridis")
            
        except Exception as e:
            self.logger.warning(f"Could not setup plotting style: {e}")
    
    async def generate_network_visualization(
        self,
        graph: nx.Graph,
        analysis_result: NetworkAnalysisResult,
        layout_algorithm: str = "spring",
        save_plot: bool = True,
        show_labels: bool = True,
        color_scheme: str = "analysis_based"
    ) -> str:
        """Generate comprehensive network visualization."""
        
        try:
            self.logger.info(f"Generating visualization for {analysis_result.analysis_type.value}")
            
            # Get figure size
            figure_size = getattr(self.config, 'figure_size', self.default_figure_size)
            
            fig, ax = plt.subplots(figsize=figure_size)
            
            # Prepare graph for visualization
            vis_graph = self._prepare_graph_for_visualization(graph)
            
            # Calculate layout
            pos = self._calculate_layout(vis_graph, layout_algorithm)
            
            # Determine node colors and sizes
            node_colors = self._get_node_colors(vis_graph, analysis_result, color_scheme)
            node_sizes = self._get_node_sizes(vis_graph)
            
            # Draw the network
            self._draw_network_components(
                vis_graph, pos, node_colors, node_sizes, 
                show_labels, analysis_result, ax
            )
            
            # Add title and formatting
            title = self._generate_plot_title(analysis_result)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save plot if requested
            filepath = ""
            if save_plot:
                filepath = self._save_plot(fig, analysis_result)
            
            plt.close()
            
            self.logger.info(f"Network visualization completed")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error generating network visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return ""
    
    def generate_multiple_layouts(
        self,
        graph: nx.Graph,
        analysis_result: NetworkAnalysisResult,
        layouts: List[str] = None
    ) -> List[str]:
        """Generate visualizations with multiple layout algorithms."""
        
        if layouts is None:
            layouts = ["spring", "circular", "kamada_kawai", "random"]
        
        filepaths = []
        
        for layout in layouts:
            try:
                filepath = self.generate_network_visualization(
                    graph, analysis_result, layout_algorithm=layout
                )
                if filepath:
                    filepaths.append(filepath)
            except Exception as e:
                self.logger.warning(f"Failed to generate {layout} layout: {e}")
        
        return filepaths
    
    def generate_subgraph_visualization(
        self,
        graph: nx.Graph,
        nodes_subset: List[str],
        analysis_result: NetworkAnalysisResult,
        title_suffix: str = "subgraph"
    ) -> str:
        """Generate visualization for a subgraph."""
        
        try:
            subgraph = graph.subgraph(nodes_subset).copy()
            
            # Create modified analysis result for subgraph
            subgraph_result = self._create_subgraph_result(analysis_result, nodes_subset)
            
            return self.generate_network_visualization(
                subgraph, subgraph_result, save_plot=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating subgraph visualization: {e}")
            return ""
    
    def _prepare_graph_for_visualization(self, graph: nx.Graph) -> nx.Graph:
        """Prepare graph for visualization by sampling if necessary."""
        
        # Limit nodes for visualization performance
        max_nodes = getattr(self.config, 'max_visualization_nodes', 500)
        
        if len(graph.nodes) > max_nodes:
            self.logger.info(f"Sampling {max_nodes} nodes from {len(graph.nodes)} for visualization")
            
            # Prioritize high-degree nodes for sampling
            degree_dict = dict(graph.degree())
            sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
            sampled_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            
            return graph.subgraph(sampled_nodes).copy()
        else:
            return graph
    
    def _calculate_layout(self, graph: nx.Graph, algorithm: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using specified layout algorithm."""
        
        try:
            if algorithm == "spring":
                if len(graph.nodes) > 100:
                    pos = nx.spring_layout(graph, k=0.5, iterations=20)
                else:
                    pos = nx.spring_layout(graph, k=1, iterations=50)
            
            elif algorithm == "circular":
                pos = nx.circular_layout(graph)
            
            elif algorithm == "kamada_kawai":
                if len(graph.nodes) <= 200:  # K-K is expensive for large graphs
                    pos = nx.kamada_kawai_layout(graph)
                else:
                    # Fallback to spring layout
                    pos = nx.spring_layout(graph, k=0.5, iterations=20)
            
            elif algorithm == "random":
                pos = nx.random_layout(graph)
            
            elif algorithm == "shell":
                pos = nx.shell_layout(graph)
            
            elif algorithm == "spectral":
                try:
                    pos = nx.spectral_layout(graph)
                except Exception:
                    # Fallback if spectral fails
                    pos = nx.spring_layout(graph)
            
            else:
                # Default to spring layout
                pos = nx.spring_layout(graph)
            
            return pos
            
        except Exception as e:
            self.logger.warning(f"Layout algorithm {algorithm} failed: {e}, using spring layout")
            return nx.spring_layout(graph)
    
    def _get_node_colors(
        self, 
        graph: nx.Graph, 
        analysis_result: NetworkAnalysisResult, 
        color_scheme: str
    ) -> List[float]:
        """Determine node colors based on analysis type and scheme."""
        
        node_colors = []
        
        if color_scheme == "analysis_based" and analysis_result.node_metrics:
            
            if analysis_result.analysis_type == AnalysisType.CENTRALITY:
                # Color by PageRank or first centrality measure
                centrality_key = self._get_primary_centrality_key(analysis_result.node_metrics[0])
                
                for node in graph.nodes:
                    node_metric = next(
                        (m for m in analysis_result.node_metrics if m.node_id == node), 
                        None
                    )
                    if node_metric:
                        score = node_metric.centrality_scores.get(centrality_key, 0)
                        node_colors.append(score)
                    else:
                        node_colors.append(0)
            
            elif analysis_result.analysis_type == AnalysisType.COMMUNITY_DETECTION:
                # Color by community
                for node in graph.nodes:
                    node_metric = next(
                        (m for m in analysis_result.node_metrics if m.node_id == node), 
                        None
                    )
                    if node_metric and node_metric.community_id is not None:
                        node_colors.append(node_metric.community_id)
                    else:
                        node_colors.append(-1)
            
            elif analysis_result.analysis_type == AnalysisType.CLUSTERING_ANALYSIS:
                # Color by clustering coefficient
                for node in graph.nodes:
                    node_metric = next(
                        (m for m in analysis_result.node_metrics if m.node_id == node), 
                        None
                    )
                    if node_metric:
                        node_colors.append(node_metric.clustering_coefficient)
                    else:
                        node_colors.append(0)
            
            else:
                # Default coloring by degree
                node_colors = [graph.degree(node) for node in graph.nodes]
        
        elif color_scheme == "degree":
            node_colors = [graph.degree(node) for node in graph.nodes]
        
        elif color_scheme == "domain":
            # Color by domain if available
            domain_colors = {}
            color_counter = 0
            
            for node in graph.nodes:
                domain = graph.nodes[node].get('domain', 'unknown')
                if domain not in domain_colors:
                    domain_colors[domain] = color_counter
                    color_counter += 1
                node_colors.append(domain_colors[domain])
        
        else:
            # Default: uniform coloring
            node_colors = [1.0] * len(graph.nodes)
        
        return node_colors
    
    def _get_primary_centrality_key(self, node_metric: NodeMetrics) -> str:
        """Get the primary centrality measure for coloring."""
        
        centrality_priority = ['pagerank', 'betweenness', 'closeness', 'eigenvector', 'degree']
        
        for key in centrality_priority:
            if key in node_metric.centrality_scores:
                return key
        
        # Return first available key
        if node_metric.centrality_scores:
            return list(node_metric.centrality_scores.keys())[0]
        
        return 'degree'
    
    def _get_node_sizes(self, graph: nx.Graph) -> List[float]:
        """Calculate node sizes based on degree."""
        
        node_size_multiplier = getattr(self.config, 'node_size_multiplier', self.default_node_size_multiplier)
        
        node_sizes = [
            node_size_multiplier * (1 + graph.degree(node) / 10) 
            for node in graph.nodes
        ]
        
        return node_sizes
    
    def _draw_network_components(
        self,
        graph: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        node_colors: List[float],
        node_sizes: List[float],
        show_labels: bool,
        analysis_result: NetworkAnalysisResult,
        ax
    ) -> None:
        """Draw all network components (nodes, edges, labels)."""
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            cmap=plt.cm.viridis,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.3,
            width=0.5,
            edge_color='gray',
            ax=ax
        )
        
        # Add labels if requested
        if show_labels:
            self._add_node_labels(graph, pos, ax)
    
    def _add_node_labels(
        self, 
        graph: nx.Graph, 
        pos: Dict[str, Tuple[float, float]], 
        ax
    ) -> None:
        """Add labels for high-degree nodes."""
        
        try:
            degrees = [graph.degree(node) for node in graph.nodes]
            if degrees:
                degree_threshold = np.percentile(degrees, 90)
                
                high_degree_nodes = [
                    node for node in graph.nodes 
                    if graph.degree(node) > degree_threshold
                ]
                
                # Limit number of labels for readability
                high_degree_nodes = high_degree_nodes[:20]
                
                high_degree_labels = {
                    node: graph.nodes[node].get('title', str(node))[:10] 
                    for node in high_degree_nodes
                }
                
                nx.draw_networkx_labels(
                    graph, pos,
                    labels=high_degree_labels,
                    font_size=8,
                    font_weight='bold',
                    ax=ax
                )
        except Exception as e:
            self.logger.warning(f"Could not add labels: {e}")
    
    def _generate_plot_title(self, analysis_result: NetworkAnalysisResult) -> str:
        """Generate appropriate plot title."""
        
        analysis_name = analysis_result.analysis_type.value.replace('_', ' ').title()
        timestamp = analysis_result.generated_at.strftime("%Y-%m-%d %H:%M")
        
        return f"Network Analysis: {analysis_name} ({timestamp})"
    
    def _save_plot(self, fig, analysis_result: NetworkAnalysisResult) -> str:
        """Save the plot to file."""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"network_{analysis_result.analysis_type.value}_{timestamp}.png"
            filepath = self.plot_dir / filename
            
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            
            self.logger.info(f"Network visualization saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")
            return ""
    
    def _create_subgraph_result(
        self, 
        analysis_result: NetworkAnalysisResult, 
        nodes_subset: List[str]
    ) -> NetworkAnalysisResult:
        """Create a modified analysis result for subgraph visualization."""
        
        # Filter node metrics for subgraph
        subgraph_node_metrics = [
            metric for metric in analysis_result.node_metrics 
            if metric.node_id in nodes_subset
        ]
        
        # Create new result with filtered metrics
        subgraph_result = NetworkAnalysisResult(
            analysis_type=analysis_result.analysis_type,
            graph_metrics=analysis_result.graph_metrics,
            node_metrics=subgraph_node_metrics,
            communities=analysis_result.communities,
            insights=analysis_result.insights,
            recommendations=analysis_result.recommendations,
            generated_at=analysis_result.generated_at,
            execution_time=analysis_result.execution_time
        )
        
        return subgraph_result
    
    def create_comparison_visualization(
        self,
        graphs: List[nx.Graph],
        titles: List[str],
        save_plot: bool = True
    ) -> str:
        """Create side-by-side comparison of multiple graphs."""
        
        try:
            num_graphs = len(graphs)
            fig, axes = plt.subplots(1, num_graphs, figsize=(6 * num_graphs, 6))
            
            if num_graphs == 1:
                axes = [axes]
            
            for i, (graph, title) in enumerate(zip(graphs, titles)):
                ax = axes[i]
                
                # Prepare graph
                vis_graph = self._prepare_graph_for_visualization(graph)
                pos = self._calculate_layout(vis_graph, "spring")
                node_colors = [vis_graph.degree(node) for node in vis_graph.nodes]
                node_sizes = self._get_node_sizes(vis_graph)
                
                # Draw
                nx.draw_networkx_nodes(
                    vis_graph, pos, node_color=node_colors,
                    node_size=node_sizes, alpha=0.7, cmap=plt.cm.viridis, ax=ax
                )
                nx.draw_networkx_edges(
                    vis_graph, pos, alpha=0.3, width=0.5, ax=ax
                )
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
            
            plt.tight_layout()
            
            filepath = ""
            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"network_comparison_{timestamp}.png"
                filepath = self.plot_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                self.logger.info(f"Comparison visualization saved to {filepath}")
            
            plt.close()
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating comparison visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return ""


# Factory function for easy integration
def create_network_visualizer(config=None) -> NetworkVisualizer:
    """Create and return a NetworkVisualizer instance."""
    return NetworkVisualizer(config)


# Async wrapper for compatibility
async def generate_network_visualization(
    graph: nx.Graph,
    analysis_result: NetworkAnalysisResult,
    config=None,
    **kwargs
) -> str:
    """Generate network visualization."""
    visualizer = create_network_visualizer(config)
    return await visualizer.generate_network_visualization(graph, analysis_result, **kwargs)