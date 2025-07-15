"""
Yggdrasil tree layout engine for hierarchical visualization.
"""

from typing import Dict, List
from ..core.models import VisualizationData, VisualizationNode
from ..core.config import VisualizationConfig


class YggdrasilLayout:
    """Layout engine for Yggdrasil tree structure."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def apply_layout(self, viz_data: VisualizationData) -> None:
        """Apply Yggdrasil tree layout to nodes."""
        
        # Group nodes by level and domain
        level_groups = {}
        for node in viz_data.nodes:
            level = node.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes in tree structure
        base_y = 0
        level_height = self.config.tree_spacing["level_distance"]
        
        for level in sorted(level_groups.keys()):
            nodes_at_level = level_groups[level]
            
            if level == 0:  # Root node
                for node in nodes_at_level:
                    node.x = self.config.tree_spacing["root_x"]
                    node.y = self.config.tree_spacing["root_y"]
            else:
                # Distribute nodes horizontally
                if len(nodes_at_level) == 1:
                    nodes_at_level[0].x = 0
                else:
                    total_width = (len(nodes_at_level) - 1) * self.config.tree_spacing["node_distance"]
                    start_x = -total_width / 2
                    
                    for i, node in enumerate(nodes_at_level):
                        node.x = start_x + i * self.config.tree_spacing["node_distance"]
                
                # Set Y position based on level
                for node in nodes_at_level:
                    node.y = base_y - level * level_height
        
        # Apply colors and sizes
        self._apply_styling(viz_data)
    
    def _apply_styling(self, viz_data: VisualizationData) -> None:
        """Apply colors and sizes to nodes and edges."""
        
        # Apply node styling
        for node in viz_data.nodes:
            node.color = self.config.get_node_color(node.node_type)
            
            # Size based on importance (could be PageRank, degree, etc.)
            base_size = self.config.node_sizes["min_size"]
            if node.metadata and node.metadata.get('word_count'):
                # Size based on word count
                word_count = node.metadata['word_count']
                size_multiplier = min(word_count / 1000, 5)  # Cap at 5x
                node.size = base_size + size_multiplier * self.config.node_sizes["size_factor"]
            else:
                node.size = base_size
            
            # Ensure size is within bounds
            node.size = max(
                self.config.node_sizes["min_size"],
                min(node.size or base_size, self.config.node_sizes["max_size"])
            )
        
        # Apply edge styling
        for edge in viz_data.edges:
            edge.color = self.config.get_edge_color(edge.relationship_type)
    
    def get_layout_parameters(self) -> Dict[str, any]:
        """Get layout-specific parameters for visualization."""
        return {
            'layout_type': 'hierarchical',
            'node_spacing': self.config.tree_spacing["node_distance"],
            'level_separation': self.config.tree_spacing["level_distance"],
            'tree_spacing': self.config.tree_spacing["node_distance"]
        }