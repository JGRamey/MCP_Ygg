"""
Streamlit Dashboard Components

Modular components for the MCP Yggdrasil Streamlit dashboard interface.
This module breaks down the monolithic dashboard into focused, reusable components
following the established modular architecture patterns.

Components:
- page_renderers.py: All dashboard page rendering functions
- ui_components.py: Reusable UI elements (header, sidebar, forms)
- data_operations.py: Data processing and pipeline operations
- visualization_components.py: Chart and graph generation components
- config_management.py: Configuration and state management

Architecture:
- Single responsibility principle: Each component has a clear, focused purpose
- Consistent error handling with module-specific logging
- Factory functions for easy instantiation
- Streamlit-optimized caching and session management

Usage:
    from components import PageRenderers, UIComponents, DataOperations
    
    # Initialize components
    page_renderer = PageRenderers()
    ui_components = UIComponents()
    
    # Render dashboard pages
    page_renderer.render_overview_page()
    ui_components.render_header()
"""

from .page_renderers import (
    PageRenderers,
    render_overview_page,
    render_data_input_page,
    render_query_page,
    render_visualizations_page,
    render_maintenance_page,
    render_analytics_page,
    render_anomalies_page,
    render_recommendations_page
)

from .ui_components import (
    UIComponents,
    render_header,
    render_sidebar,
    create_metric_card,
    create_data_card,
    apply_custom_styling
)

from .data_operations import (
    DataOperations,
    process_uploaded_files,
    start_web_scraping,
    add_manual_document,
    import_batch_data,
    run_full_pipeline
)

from .visualization_components import (
    VisualizationComponents,
    generate_yggdrasil_visualization,
    create_performance_charts,
    create_analytics_dashboard,
    create_anomaly_visualizations
)

from .config_management import (
    DashboardConfig,
    DashboardState,
    get_dashboard_state,
    initialize_dashboard_config
)

# Search and query operations
from .search_operations import (
    SearchOperations,
    perform_text_search,
    perform_semantic_search,
    execute_graph_query
)

# Analytics and ML operations
from .analytics_operations import (
    AnalyticsOperations,
    run_anomaly_detection,
    get_recommendations,
    generate_analytics_insights
)

__all__ = [
    # Main component classes
    'PageRenderers',
    'UIComponents', 
    'DataOperations',
    'VisualizationComponents',
    'DashboardConfig',
    'DashboardState',
    'SearchOperations',
    'AnalyticsOperations',
    
    # Page rendering functions
    'render_overview_page',
    'render_data_input_page',
    'render_query_page',
    'render_visualizations_page',
    'render_maintenance_page',
    'render_analytics_page',
    'render_anomalies_page',
    'render_recommendations_page',
    
    # UI component functions
    'render_header',
    'render_sidebar',
    'create_metric_card',
    'create_data_card',
    'apply_custom_styling',
    
    # Data operation functions
    'process_uploaded_files',
    'start_web_scraping', 
    'add_manual_document',
    'import_batch_data',
    'run_full_pipeline',
    
    # Visualization functions
    'generate_yggdrasil_visualization',
    'create_performance_charts',
    'create_analytics_dashboard',
    'create_anomaly_visualizations',
    
    # Configuration functions
    'get_dashboard_state',
    'initialize_dashboard_config',
    
    # Search functions
    'perform_text_search',
    'perform_semantic_search',
    'execute_graph_query',
    
    # Analytics functions
    'run_anomaly_detection',
    'get_recommendations',
    'generate_analytics_insights'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "MCP Yggdrasil Team"
__description__ = "Modular Streamlit dashboard components for knowledge management"