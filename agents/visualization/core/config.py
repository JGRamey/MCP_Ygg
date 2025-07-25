"""
Configuration management for visualization components.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .models import NodeType


class VisualizationConfig:
    """Configuration for visualization generation."""

    def __init__(self, config_path: str = "config/visualization.yaml"):
        # Database connection
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"

        # Color mappings (as specified in the project plan)
        self.node_colors = {
            NodeType.DOCUMENT: "#3498db",  # Blue
            NodeType.CONCEPT: "#e74c3c",  # Red
            NodeType.PERSON: "#2ecc71",  # Green
            NodeType.EVENT: "#f1c40f",  # Yellow
            NodeType.PATTERN: "#9b59b6",  # Purple
            NodeType.ROOT: "#34495e",  # Dark gray
            NodeType.DOMAIN: "#e67e22",  # Orange
        }

        # Edge colors
        self.edge_colors = {
            "DERIVED_FROM": "#95a5a6",
            "REFERENCES": "#3498db",
            "INFLUENCED_BY": "#e74c3c",
            "CONTAINS_CONCEPT": "#2ecc71",
            "SIMILAR_CONCEPT": "#9b59b6",
            "default": "#bdc3c7",
        }

        # Layout parameters
        self.tree_spacing = {
            "node_distance": 100,
            "level_distance": 200,
            "root_x": 0,
            "root_y": 0,
        }

        # Size parameters
        self.node_sizes = {"min_size": 10, "max_size": 50, "size_factor": 1.5}

        # Visualization parameters
        self.max_nodes = 1000
        self.max_edges = 2000
        self.enable_physics = True
        self.show_labels = True
        self.enable_zoom = True
        self.enable_pan = True

        # Template settings
        self.template_dir = "agents/visualization/templates"
        self.output_dir = "agents/visualization/output"

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from file if it exists."""
        try:
            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

    def get_node_color(self, node_type: NodeType) -> str:
        """Get color for a node type."""
        return self.node_colors.get(node_type, "#999999")

    def get_edge_color(self, relationship_type: str) -> str:
        """Get color for an edge relationship type."""
        return self.edge_colors.get(relationship_type, self.edge_colors["default"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "node_colors": {nt.value: color for nt, color in self.node_colors.items()},
            "edge_colors": self.edge_colors,
            "tree_spacing": self.tree_spacing,
            "node_sizes": self.node_sizes,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "enable_physics": self.enable_physics,
            "show_labels": self.show_labels,
            "enable_zoom": self.enable_zoom,
            "enable_pan": self.enable_pan,
            "template_dir": self.template_dir,
            "output_dir": self.output_dir,
        }
