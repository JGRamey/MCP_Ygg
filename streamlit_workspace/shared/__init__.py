"""
Shared Components for MCP Yggdrasil Streamlit Workspace

This module provides reusable components extracted from both the sophisticated pages
and the modular components to create a unified, production-ready architecture.

Key Features:
- Unified UI components (headers, sidebars, cards, forms)
- Shared data processing utilities
- Common search and query operations
- Configuration and state management
- Performance optimization utilities

Architecture:
- shared/ui/: Reusable UI components and styling
- shared/data/: Data processing and pipeline utilities
- shared/search/: Search and query operations
- shared/config/: Configuration and state management

Usage:
    from shared.ui import create_page_header, create_metric_card
    from shared.data import process_uploaded_file, validate_content
    from shared.search import perform_text_search, execute_graph_query
    from shared.config import get_session_state, initialize_agents
"""

# UI Components
from .ui.headers import create_page_header, create_section_header
from .ui.sidebars import create_navigation_sidebar, create_filter_sidebar
from .ui.cards import create_metric_card, create_data_card, create_concept_card
from .ui.forms import create_search_form, create_upload_form, create_concept_form
from .ui.styling import apply_custom_css, get_theme_colors

# Data Operations
from .data.processors import process_uploaded_file, process_web_content, validate_content
from .data.validators import validate_concept_data, validate_relationship_data
from .data.exporters import export_to_csv, export_to_json, export_to_pdf
from .data.pipelines import run_content_pipeline, run_analysis_pipeline

# Search Operations
from .search.text_search import perform_text_search, search_concepts_by_domain
from .search.semantic_search import perform_semantic_search, find_similar_concepts
from .search.graph_queries import execute_graph_query, get_concept_relationships

# Configuration Management
from .config.state_management import get_session_state, update_session_state, clear_session_cache
from .config.agent_management import initialize_agents, get_agent_status, refresh_agents
from .config.performance import setup_caching, get_performance_metrics, optimize_queries

__all__ = [
    # UI Components
    'create_page_header', 'create_section_header',
    'create_navigation_sidebar', 'create_filter_sidebar', 
    'create_metric_card', 'create_data_card', 'create_concept_card',
    'create_search_form', 'create_upload_form', 'create_concept_form',
    'apply_custom_css', 'get_theme_colors',
    
    # Data Operations
    'process_uploaded_file', 'process_web_content', 'validate_content',
    'validate_concept_data', 'validate_relationship_data',
    'export_to_csv', 'export_to_json', 'export_to_pdf',
    'run_content_pipeline', 'run_analysis_pipeline',
    
    # Search Operations
    'perform_text_search', 'search_concepts_by_domain',
    'perform_semantic_search', 'find_similar_concepts',
    'execute_graph_query', 'get_concept_relationships',
    
    # Configuration
    'get_session_state', 'update_session_state', 'clear_session_cache',
    'initialize_agents', 'get_agent_status', 'refresh_agents',
    'setup_caching', 'get_performance_metrics', 'optimize_queries'
]

__version__ = "1.0.0"
__author__ = "MCP Yggdrasil Team"
__description__ = "Shared components for production-ready Streamlit workspace"