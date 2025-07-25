"""
Refactored Streamlit Dashboard for MCP Server
Modular dashboard using component-based architecture.

This refactored version uses modular components following the established
architecture patterns from the graph analysis refactoring.
"""

import logging
from pathlib import Path

import streamlit as st

# Import modular components
from components.config_management import DashboardConfig, get_dashboard_state
from components.page_renderers import (
    render_analytics_page,
    render_anomalies_page,
    render_data_input_page,
    render_maintenance_page,
    render_overview_page,
    render_query_page,
    render_recommendations_page,
    render_visualizations_page,
)
from components.ui_components import render_header, render_sidebar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MCP Server Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_dashboard() -> bool:
    """
    Initialize the dashboard with configuration and state management.

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Get dashboard state (creates if needed)
        dashboard_state = get_dashboard_state()

        # Initialize agents if not already done
        if not st.session_state.get("initialized", False):
            logger.info("Initializing dashboard agents and state")

            # Initialize agents
            dashboard_state.agents = dashboard_state.initialize_agents()

            # Set initialization flag
            st.session_state.initialized = True

            logger.info("Dashboard initialization completed successfully")

        return True

    except Exception as e:
        logger.error(f"Dashboard initialization failed: {e}")
        st.error(f"Failed to initialize dashboard: {str(e)}")
        return False


def main():
    """Main application entry point with modular architecture."""
    try:
        # Initialize dashboard
        if not initialize_dashboard():
            st.stop()

        # Render header with system status
        render_header()

        # Render sidebar and get selected page
        selected_page = render_sidebar()

        # Route to appropriate page renderer
        page_routing = {
            "Overview": render_overview_page,
            "Data Input": render_data_input_page,
            "Query & Search": render_query_page,
            "Visualizations": render_visualizations_page,
            "Maintenance": render_maintenance_page,
            "Analytics": render_analytics_page,
            "Anomalies": render_anomalies_page,
            "Recommendations": render_recommendations_page,
        }

        # Render selected page
        page_function = page_routing.get(selected_page)
        if page_function:
            try:
                page_function()
            except Exception as page_error:
                logger.error(f"Error rendering page '{selected_page}': {page_error}")
                st.error(
                    f"Error loading {selected_page} page. Please try refreshing or contact support."
                )
        else:
            st.error(f"Unknown page: {selected_page}")
            logger.warning(f"Attempted to access unknown page: {selected_page}")

        # Handle session state triggers
        handle_session_triggers()

    except Exception as e:
        logger.error(f"Critical error in main dashboard: {e}")
        st.error(
            "A critical error occurred. Please refresh the page or contact support."
        )


def handle_session_triggers():
    """Handle various session state triggers from UI interactions."""
    try:
        # Handle pipeline trigger
        if st.session_state.get("trigger_pipeline", False):
            from components.data_operations import run_full_pipeline

            run_full_pipeline()
            st.session_state.trigger_pipeline = False

        # Handle export modal
        if st.session_state.get("show_export_modal", False):
            show_export_modal()

        # Handle chart modal
        if st.session_state.get("show_chart_modal", False):
            show_chart_modal()

        # Handle search modal
        if st.session_state.get("show_search_modal", False):
            show_search_modal()

    except Exception as e:
        logger.error(f"Error handling session triggers: {e}")


def show_export_modal():
    """Show data export modal."""
    with st.expander("📤 Export Data", expanded=True):
        st.write("Export functionality would be implemented here")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Close", key="close_export"):
                st.session_state.show_export_modal = False
                st.rerun()
        with col2:
            if st.button("Export", key="do_export"):
                st.success("Export initiated!")
                st.session_state.show_export_modal = False


def show_chart_modal():
    """Show chart generation modal."""
    with st.expander("📊 Generate Chart", expanded=True):
        st.write("Chart generation options would be implemented here")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Close", key="close_chart"):
                st.session_state.show_chart_modal = False
                st.rerun()
        with col2:
            if st.button("Generate", key="do_generate"):
                st.success("Chart generated!")
                st.session_state.show_chart_modal = False


def show_search_modal():
    """Show quick search modal."""
    with st.expander("🔍 Quick Search", expanded=True):
        search_query = st.text_input("Search query:", key="quick_search_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Close", key="close_search"):
                st.session_state.show_search_modal = False
                st.rerun()
        with col2:
            if st.button("Search", key="do_search") and search_query:
                from components.search_operations import perform_text_search

                perform_text_search(search_query)
                st.session_state.show_search_modal = False


if __name__ == "__main__":
    main()
