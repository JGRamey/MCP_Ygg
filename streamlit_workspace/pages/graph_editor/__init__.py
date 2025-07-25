"""
Graph Editor Module

Modular Graph Editor interface providing interactive knowledge graph visualization
and editing capabilities with Neo4j database integration and CSV fallback.

Components:
- models.py: Data structures and configuration
- neo4j_connector.py: Database connection and data management
- graph_visualizer.py: Graph creation and visualization engine
- ui_components.py: UI elements and interface components
- main.py: Main orchestrator (under 200 lines)
"""

from .graph_visualizer import GraphVisualizer
from .models import (
    GraphFilters,
    GraphMode,
    GraphSettings,
    LayoutType,
    RelationshipFilters,
)
from .neo4j_connector import DataSourceManager, Neo4jConnector
from .ui_components import GraphEditorUI, GraphReportGenerator

__all__ = [
    "GraphMode",
    "LayoutType",
    "GraphFilters",
    "RelationshipFilters",
    "GraphSettings",
    "Neo4jConnector",
    "DataSourceManager",
    "GraphVisualizer",
    "GraphEditorUI",
    "GraphReportGenerator",
]
