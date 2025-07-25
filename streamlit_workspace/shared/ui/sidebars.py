"""
Sidebar components for navigation and filtering

Provides reusable sidebar components extracted from existing pages to ensure
consistent navigation and filtering interfaces across the workspace.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st


def create_navigation_sidebar(
    current_page: str, pages: List[Dict[str, str]], show_metrics: bool = True
) -> str:
    """
    Create a standard navigation sidebar with page selection and quick metrics.

    Args:
        current_page: Currently selected page
        pages: List of page dictionaries with 'name', 'icon', and optional 'description'
        show_metrics: Whether to show quick metrics

    Returns:
        Selected page name
    """
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        # Page selection
        page_options = [f"{page['icon']} {page['name']}" for page in pages]
        current_option = f"{next(p['icon'] for p in pages if p['name'] == current_page)} {current_page}"

        try:
            current_index = page_options.index(current_option)
        except ValueError:
            current_index = 0

        selected = st.selectbox(
            "Select Page",
            page_options,
            index=current_index,
            label_visibility="collapsed",
        )

        # Extract page name from selection
        selected_page = selected.split(" ", 1)[1] if " " in selected else selected

        st.markdown("---")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", key="nav_refresh"):
                st.rerun()

        with col2:
            if st.button("🏠 Home", key="nav_home"):
                st.session_state.selected_page = "Overview"
                st.rerun()

        # Quick metrics if enabled
        if show_metrics:
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")

            # Get basic metrics from session state or defaults
            metrics = st.session_state.get(
                "quick_metrics",
                {"concepts": "N/A", "relationships": "N/A", "last_update": "N/A"},
            )

            st.metric("Concepts", metrics.get("concepts", "N/A"))
            st.metric("Relationships", metrics.get("relationships", "N/A"))

            if metrics.get("last_update") != "N/A":
                st.caption(f"Updated: {metrics['last_update']}")

    return selected_page


def create_filter_sidebar(
    filters: Dict[str, Any], title: str = "🗃️ Filters"
) -> Dict[str, Any]:
    """
    Create a filtering sidebar with various filter types.

    Args:
        filters: Dictionary defining filter configurations
        title: Sidebar section title

    Returns:
        Dictionary of selected filter values
    """
    with st.sidebar:
        st.markdown(f"### {title}")

        filter_values = {}

        for filter_key, filter_config in filters.items():
            filter_type = filter_config.get("type", "text")
            label = filter_config.get("label", filter_key.title())
            options = filter_config.get("options", [])
            default = filter_config.get("default", None)
            help_text = filter_config.get("help", None)

            if filter_type == "multiselect":
                filter_values[filter_key] = st.multiselect(
                    label, options, default=default, help=help_text
                )

            elif filter_type == "selectbox":
                filter_values[filter_key] = st.selectbox(
                    label,
                    options,
                    index=options.index(default) if default in options else 0,
                    help=help_text,
                )

            elif filter_type == "text":
                filter_values[filter_key] = st.text_input(
                    label, value=default or "", help=help_text
                )

            elif filter_type == "number":
                min_val = filter_config.get("min", 0)
                max_val = filter_config.get("max", 100)
                filter_values[filter_key] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default or min_val,
                    help=help_text,
                )

            elif filter_type == "slider":
                min_val = filter_config.get("min", 0)
                max_val = filter_config.get("max", 100)
                step = filter_config.get("step", 1)
                filter_values[filter_key] = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default or min_val,
                    step=step,
                    help=help_text,
                )

            elif filter_type == "date":
                filter_values[filter_key] = st.date_input(
                    label, value=default, help=help_text
                )

            elif filter_type == "checkbox":
                filter_values[filter_key] = st.checkbox(
                    label, value=default or False, help=help_text
                )

    return filter_values


def create_domain_filter_sidebar() -> Dict[str, Any]:
    """
    Create a specialized domain filter sidebar for MCP Yggdrasil.

    Returns:
        Dictionary of selected domain filter values
    """
    domain_filters = {
        "domains": {
            "type": "multiselect",
            "label": "Filter by Domain",
            "options": [
                "🎨 Art",
                "🗣️ Language",
                "🔢 Mathematics",
                "🤔 Philosophy",
                "🔬 Science",
                "💻 Technology",
            ],
            "default": [],
            "help": "Select one or more domains to filter content",
        },
        "date_range": {
            "type": "selectbox",
            "label": "Date Range",
            "options": ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
            "default": "All Time",
            "help": "Filter by creation/modification date",
        },
        "content_type": {
            "type": "multiselect",
            "label": "Content Type",
            "options": ["Concept", "Text", "Document", "Scraped", "Manual"],
            "default": [],
            "help": "Filter by type of content",
        },
        "quality_score": {
            "type": "slider",
            "label": "Min Quality Score",
            "min": 0,
            "max": 100,
            "default": 0,
            "step": 5,
            "help": "Minimum quality score threshold",
        },
    }

    return create_filter_sidebar(domain_filters, "🎯 Domain Filters")


def create_operations_sidebar() -> Dict[str, Any]:
    """
    Create an operations sidebar with database controls and status.

    Returns:
        Dictionary of operation results
    """
    operations = {}

    with st.sidebar:
        st.markdown("### 🔧 Operations")

        # Database operations
        st.markdown("**Database**")
        col1, col2 = st.columns(2)

        with col1:
            operations["refresh_db"] = st.button("🔄 Refresh", key="ops_refresh_db")

        with col2:
            operations["backup_db"] = st.button("💾 Backup", key="ops_backup_db")

        # Cache operations
        st.markdown("**Cache**")
        col1, col2 = st.columns(2)

        with col1:
            operations["clear_cache"] = st.button("🗑️ Clear", key="ops_clear_cache")

        with col2:
            operations["cache_stats"] = st.button("📊 Stats", key="ops_cache_stats")

        # Agent operations
        st.markdown("**Agents**")
        col1, col2 = st.columns(2)

        with col1:
            operations["restart_agents"] = st.button(
                "🔄 Restart", key="ops_restart_agents"
            )

        with col2:
            operations["agent_status"] = st.button("📈 Status", key="ops_agent_status")

        st.markdown("---")

        # System information
        st.markdown("### 📊 System Info")

        # Get system info from session state or defaults
        system_info = st.session_state.get(
            "system_info",
            {
                "memory_usage": "N/A",
                "cpu_usage": "N/A",
                "disk_space": "N/A",
                "uptime": "N/A",
            },
        )

        st.metric("Memory", system_info.get("memory_usage", "N/A"))
        st.metric("CPU", system_info.get("cpu_usage", "N/A"))
        st.metric("Disk", system_info.get("disk_space", "N/A"))

        if system_info.get("uptime") != "N/A":
            st.caption(f"Uptime: {system_info['uptime']}")

    return operations


def create_search_sidebar() -> Dict[str, Any]:
    """
    Create a search sidebar with different search options.

    Returns:
        Dictionary of search parameters
    """
    search_params = {}

    with st.sidebar:
        st.markdown("### 🔍 Search Options")

        # Search type
        search_params["search_type"] = st.radio(
            "Search Type",
            ["Text Search", "Semantic Search", "Graph Query"],
            help="Choose the type of search to perform",
        )

        # Search query
        search_params["query"] = st.text_input(
            "Search Query",
            placeholder="Enter your search terms...",
            help="Enter keywords or phrases to search for",
        )

        # Search filters
        if search_params["search_type"] == "Text Search":
            search_params["case_sensitive"] = st.checkbox("Case Sensitive")
            search_params["whole_words"] = st.checkbox("Whole Words Only")

        elif search_params["search_type"] == "Semantic Search":
            search_params["similarity_threshold"] = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum similarity score for results",
            )
            search_params["max_results"] = st.number_input(
                "Max Results", min_value=1, max_value=100, value=10
            )

        elif search_params["search_type"] == "Graph Query":
            search_params["query_type"] = st.selectbox(
                "Query Type",
                ["Find Paths", "Get Neighbors", "Find Communities", "Custom Cypher"],
            )
            search_params["max_depth"] = st.number_input(
                "Max Depth", min_value=1, max_value=10, value=3
            )

        # Execute search
        search_params["execute"] = st.button("🔍 Search", type="primary")

        # Clear search
        if st.button("🗑️ Clear"):
            for key in list(search_params.keys()):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    return search_params


def create_analytics_sidebar() -> Dict[str, Any]:
    """
    Create an analytics sidebar with metric controls and filters.

    Returns:
        Dictionary of analytics parameters
    """
    analytics_params = {}

    with st.sidebar:
        st.markdown("### 📈 Analytics Controls")

        # Time range
        analytics_params["time_range"] = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
            index=2,
        )

        # Metrics to display
        analytics_params["metrics"] = st.multiselect(
            "Metrics to Display",
            [
                "Concept Count",
                "Relationship Count",
                "Query Performance",
                "Cache Hit Rate",
                "Error Rate",
                "User Activity",
            ],
            default=["Concept Count", "Relationship Count", "Query Performance"],
        )

        # Chart types
        analytics_params["chart_type"] = st.radio(
            "Chart Type", ["Line Chart", "Bar Chart", "Pie Chart", "Heatmap"], index=0
        )

        # Refresh rate
        analytics_params["auto_refresh"] = st.checkbox("Auto Refresh")
        if analytics_params["auto_refresh"]:
            analytics_params["refresh_interval"] = st.selectbox(
                "Refresh Interval",
                ["30 seconds", "1 minute", "5 minutes", "15 minutes"],
                index=1,
            )

        # Generate report
        analytics_params["generate_report"] = st.button(
            "📊 Generate Report", type="primary"
        )

        # Export options
        st.markdown("**Export Options**")
        col1, col2 = st.columns(2)

        with col1:
            analytics_params["export_csv"] = st.button("📄 CSV")

        with col2:
            analytics_params["export_pdf"] = st.button("📑 PDF")

    return analytics_params
