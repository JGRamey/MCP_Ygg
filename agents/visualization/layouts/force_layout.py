"""
Force-directed layout engine for network visualization.
"""

import logging
from typing import Any, Dict

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ..core.config import VisualizationConfig
from ..core.models import VisualizationData


class ForceLayout:
    """Force-directed layout engine using NetworkX."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_layout(self, viz_data: VisualizationData) -> None:
        """Apply force-directed layout using NetworkX."""

        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available, using fallback layout")
            self._apply_fallback_layout(viz_data)
            return

        try:
            # Create NetworkX graph
            G = nx.Graph()

            # Add nodes
            for node in viz_data.nodes:
                G.add_node(node.id)

            # Add edges
            for edge in viz_data.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)

            # Calculate layout
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, k=200, iterations=50)

                # Apply positions
                for node in viz_data.nodes:
                    if node.id in pos:
                        node.x = pos[node.id][0] * 1000  # Scale up
                        node.y = pos[node.id][1] * 1000

            # Apply styling
            self._apply_styling(viz_data)

        except Exception as e:
            self.logger.warning(f"Error in force layout: {e}")
            # Fallback to simple layout
            self._apply_fallback_layout(viz_data)

    def _apply_styling(self, viz_data: VisualizationData) -> None:
        """Apply colors and sizes to nodes and edges."""

        # Apply node styling
        for node in viz_data.nodes:
            node.color = self.config.get_node_color(node.node_type)
            node.size = self.config.node_sizes["min_size"]

        # Apply edge styling
        for edge in viz_data.edges:
            edge.color = self.config.get_edge_color(edge.relationship_type)

    def _apply_fallback_layout(self, viz_data: VisualizationData) -> None:
        """Fallback to simple grid layout."""
        for i, node in enumerate(viz_data.nodes):
            node.x = (i % 10) * 100
            node.y = (i // 10) * 100
            node.color = self.config.get_node_color(node.node_type)
            node.size = self.config.node_sizes["min_size"]

    def get_layout_parameters(self) -> Dict[str, Any]:
        """Get layout-specific parameters for visualization."""
        return {"layout_type": "force", "physics_enabled": self.config.enable_physics}
