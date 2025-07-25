"""
Header components for Streamlit pages

Provides reusable header components extracted from existing pages to ensure
consistent page headers and section headers across the workspace.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


def create_page_header(
    title: str,
    description: str,
    icon: str = "🌳",
    show_status: bool = True,
    status_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create a standardized page header with title, description, and optional status indicators.

    Args:
        title: Page title
        description: Page description
        icon: Page icon (default: 🌳)
        show_status: Whether to show system status
        status_info: Dictionary containing status information
    """
    # Main header
    st.markdown(
        f"""
    <div class="main-header">
        {icon} {title}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Description
    st.markdown(f"**{description}**")

    # Status indicators if requested
    if show_status and status_info:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            neo4j_status = status_info.get("neo4j", "Unknown")
            status_class = (
                "status-online" if neo4j_status == "Connected" else "status-offline"
            )
            st.markdown(
                f'<span class="{status_class}">Neo4j: {neo4j_status}</span>',
                unsafe_allow_html=True,
            )

        with col2:
            qdrant_status = status_info.get("qdrant", "Unknown")
            status_class = (
                "status-online" if qdrant_status == "Connected" else "status-offline"
            )
            st.markdown(
                f'<span class="{status_class}">Qdrant: {qdrant_status}</span>',
                unsafe_allow_html=True,
            )

        with col3:
            redis_status = status_info.get("redis", "Unknown")
            status_class = (
                "status-online" if redis_status == "Connected" else "status-offline"
            )
            st.markdown(
                f'<span class="{status_class}">Redis: {redis_status}</span>',
                unsafe_allow_html=True,
            )

        with col4:
            last_updated = status_info.get("last_updated", datetime.now())
            if isinstance(last_updated, str):
                st.markdown(f"Updated: {last_updated}")
            else:
                st.markdown(f"Updated: {last_updated.strftime('%H:%M:%S')}")

    st.markdown("---")


def create_section_header(
    title: str,
    description: Optional[str] = None,
    expandable: bool = False,
    expanded: bool = True,
) -> Any:
    """
    Create a section header within a page.

    Args:
        title: Section title
        description: Optional section description
        expandable: Whether the section should be expandable
        expanded: Default expanded state (if expandable)

    Returns:
        Streamlit container or expander object
    """
    if expandable:
        container = st.expander(f"### {title}", expanded=expanded)
        if description:
            container.markdown(description)
        return container
    else:
        st.markdown(
            f"""
        <div class="section-header">
            {title}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if description:
            st.markdown(description)

        return st.container()


def create_subsection_header(title: str, level: int = 4) -> None:
    """
    Create a subsection header with appropriate heading level.

    Args:
        title: Subsection title
        level: Heading level (3-6)
    """
    if level < 3:
        level = 3
    elif level > 6:
        level = 6

    header_markdown = "#" * level + f" {title}"
    st.markdown(header_markdown)


def create_page_navigation(
    current_page: str, available_pages: List[Dict[str, str]]
) -> str:
    """
    Create page navigation breadcrumbs.

    Args:
        current_page: Current page name
        available_pages: List of dicts with 'name' and 'icon' keys

    Returns:
        Selected page name
    """
    # Create breadcrumb navigation
    nav_items = []
    for page in available_pages:
        if page["name"] == current_page:
            nav_items.append(f"**{page['icon']} {page['name']}**")
        else:
            nav_items.append(f"{page['icon']} {page['name']}")

    st.markdown(" → ".join(nav_items))

    # Create page selector
    page_options = [f"{page['icon']} {page['name']}" for page in available_pages]
    current_option = f"{next(p['icon'] for p in available_pages if p['name'] == current_page)} {current_page}"

    selected = st.selectbox(
        "Navigate to:",
        page_options,
        index=page_options.index(current_option),
        label_visibility="collapsed",
    )

    # Extract page name from selection
    selected_page = selected.split(" ", 1)[1] if " " in selected else selected
    return selected_page


def create_action_header(title: str, actions: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Create a header with action buttons.

    Args:
        title: Header title
        actions: List of action dictionaries with 'label', 'key', and optional 'type'

    Returns:
        Dictionary mapping action keys to their clicked state
    """
    col1, *action_cols = st.columns([3] + [1] * len(actions))

    with col1:
        st.markdown(f"### {title}")

    results = {}
    for i, action in enumerate(actions):
        with action_cols[i]:
            button_type = action.get("type", "secondary")
            disabled = action.get("disabled", False)
            help_text = action.get("help", None)

            if button_type == "primary":
                results[action["key"]] = st.button(
                    action["label"],
                    key=action["key"],
                    disabled=disabled,
                    help=help_text,
                    type="primary",
                )
            else:
                results[action["key"]] = st.button(
                    action["label"],
                    key=action["key"],
                    disabled=disabled,
                    help=help_text,
                )

    return results


def create_status_header(
    title: str,
    status: str,
    last_update: Optional[datetime] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create a header with status information and metrics.

    Args:
        title: Header title
        status: Current status ('online', 'offline', 'warning')
        last_update: Last update timestamp
        metrics: Optional metrics to display
    """
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"### {title}")

    with col2:
        status_class = {
            "online": "status-online",
            "offline": "status-offline",
            "warning": "status-warning",
        }.get(status.lower(), "status-offline")

        st.markdown(
            f'<span class="{status_class}">Status: {status.title()}</span>',
            unsafe_allow_html=True,
        )

    with col3:
        if last_update:
            if isinstance(last_update, str):
                st.markdown(f"Updated: {last_update}")
            else:
                st.markdown(f"Updated: {last_update.strftime('%H:%M:%S')}")

    # Display metrics if provided
    if metrics:
        metric_cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(key, value)
