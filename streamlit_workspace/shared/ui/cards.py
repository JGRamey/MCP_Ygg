"""
Card components for displaying data

Provides reusable card components extracted from existing pages including
metric cards, data cards, and concept cards with consistent styling.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st


def create_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """
    Create a metric card with enhanced styling.

    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator (optional)
        delta_color: Color for delta ('positive', 'negative', 'normal')
        help_text: Optional help text
    """
    delta_class = {
        "positive": "metric-positive",
        "negative": "metric-negative",
        "normal": "metric-neutral",
    }.get(delta_color, "metric-neutral")

    delta_html = (
        f'<div class="metric-change {delta_class}">{delta}</div>' if delta else ""
    )
    help_html = (
        f'<div class="metric-help" title="{help_text}">ⓘ</div>' if help_text else ""
    )

    card_html = f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
        {help_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def create_data_card(
    title: str,
    content: Dict[str, Any],
    actions: Optional[List[Dict[str, str]]] = None,
    card_type: str = "default",
) -> Dict[str, bool]:
    """
    Create a data display card with optional actions.

    Args:
        title: Card title
        content: Dictionary of data to display
        actions: Optional list of action buttons
        card_type: Card styling type ('default', 'concept', 'submission')

    Returns:
        Dictionary mapping action keys to their clicked state
    """
    card_class = {
        "default": "data-card",
        "concept": "concept-card",
        "submission": "submission-card",
    }.get(card_type, "data-card")

    # Start card container
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

    # Card header
    if card_type == "concept":
        st.markdown(
            f'<div class="concept-header">{title}</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(f"**{title}**")

    # Card content
    for key, value in content.items():
        if isinstance(value, dict):
            st.markdown(f"**{key}:**")
            for sub_key, sub_value in value.items():
                st.markdown(f"  - {sub_key}: {sub_value}")
        elif isinstance(value, list):
            st.markdown(f"**{key}:** {', '.join(map(str, value))}")
        else:
            if card_type == "concept" and key in ["domain", "type", "created"]:
                st.markdown(
                    f'<div class="concept-meta">{key}: {value}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{key}:** {value}")

    # Action buttons
    action_results = {}
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                button_type = action.get("type", "secondary")
                disabled = action.get("disabled", False)

                if button_type == "primary":
                    action_results[action["key"]] = st.button(
                        action["label"],
                        key=f"{action['key']}_{title}",
                        disabled=disabled,
                        type="primary",
                    )
                else:
                    action_results[action["key"]] = st.button(
                        action["label"],
                        key=f"{action['key']}_{title}",
                        disabled=disabled,
                    )

    # End card container
    st.markdown("</div>", unsafe_allow_html=True)

    return action_results


def create_concept_card(
    concept: Dict[str, Any], show_relationships: bool = False, editable: bool = True
) -> Dict[str, bool]:
    """
    Create a specialized card for displaying concept information.

    Args:
        concept: Concept data dictionary
        show_relationships: Whether to show relationship information
        editable: Whether to show edit actions

    Returns:
        Dictionary mapping action keys to their clicked state
    """
    concept_id = concept.get("id", "unknown")
    title = concept.get("title", "Untitled Concept")
    domain = concept.get("domain", "Unknown")
    concept_type = concept.get("type", "Unknown")
    description = concept.get("description", "No description available")

    # Build content dictionary
    content = {
        "ID": concept_id,
        "Domain": domain,
        "Type": concept_type,
        "Description": description,
    }

    # Add optional fields
    if "tags" in concept:
        content["Tags"] = concept["tags"]

    if "created" in concept:
        content["Created"] = concept["created"]

    if "last_modified" in concept:
        content["Modified"] = concept["last_modified"]

    if show_relationships and "relationships" in concept:
        content["Relationships"] = f"{len(concept['relationships'])} connections"

    # Define actions
    actions = []
    if editable:
        actions.extend(
            [
                {"label": "✏️ Edit", "key": "edit", "type": "primary"},
                {"label": "🗑️ Delete", "key": "delete", "type": "secondary"},
                {"label": "🔗 Relations", "key": "relations", "type": "secondary"},
            ]
        )
    else:
        actions.append({"label": "👁️ View", "key": "view", "type": "primary"})

    return create_data_card(title, content, actions, card_type="concept")


def create_stats_grid(
    stats: Dict[str, Union[str, int, float]], columns: int = 4
) -> None:
    """
    Create a grid of statistic cards.

    Args:
        stats: Dictionary of statistics to display
        columns: Number of columns in the grid
    """
    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)

    cols = st.columns(columns)
    for i, (label, value) in enumerate(stats.items()):
        with cols[i % columns]:
            create_metric_card(label, value)

    st.markdown("</div>", unsafe_allow_html=True)


def create_submission_card(submission: Dict[str, Any]) -> Dict[str, bool]:
    """
    Create a card for content submission display.

    Args:
        submission: Submission data dictionary

    Returns:
        Dictionary mapping action keys to their clicked state
    """
    submission_id = submission.get("id", "unknown")
    title = submission.get("title", "Untitled Submission")
    source_type = submission.get("source_type", "Unknown")
    status = submission.get("status", "Unknown")
    timestamp = submission.get("timestamp", "Unknown")

    # Build content
    content = {
        "ID": submission_id,
        "Source Type": source_type,
        "Status": status,
        "Submitted": timestamp,
    }

    if "domain" in submission:
        content["Domain"] = submission["domain"]

    if "source_url" in submission and submission["source_url"]:
        content["Source URL"] = submission["source_url"]

    if "content_length" in submission:
        content["Content Length"] = f"{submission['content_length']} characters"

    # Status-based actions
    actions = []
    if status == "submitted":
        actions.extend(
            [
                {"label": "✅ Approve", "key": "approve", "type": "primary"},
                {"label": "❌ Reject", "key": "reject", "type": "secondary"},
                {"label": "👁️ Preview", "key": "preview", "type": "secondary"},
            ]
        )
    elif status == "approved":
        actions.extend(
            [
                {"label": "🚀 Process", "key": "process", "type": "primary"},
                {"label": "👁️ View", "key": "view", "type": "secondary"},
            ]
        )
    else:
        actions.append({"label": "👁️ View", "key": "view", "type": "secondary"})

    return create_data_card(title, content, actions, card_type="submission")


def create_performance_card(metrics: Dict[str, Any]) -> None:
    """
    Create a performance metrics card.

    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown('<div class="analytics-container">', unsafe_allow_html=True)

    st.markdown("### 📊 Performance Metrics")

    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        response_time = metrics.get("avg_response_time", "N/A")
        create_metric_card("Avg Response Time", f"{response_time}ms")

    with col2:
        cache_hit_rate = metrics.get("cache_hit_rate", "N/A")
        create_metric_card("Cache Hit Rate", f"{cache_hit_rate}%")

    with col3:
        active_connections = metrics.get("active_connections", "N/A")
        create_metric_card("Active Connections", active_connections)

    with col4:
        memory_usage = metrics.get("memory_usage", "N/A")
        create_metric_card("Memory Usage", f"{memory_usage}MB")

    st.markdown("</div>", unsafe_allow_html=True)


def create_insight_card(title: str, content: str, insight_type: str = "info") -> None:
    """
    Create an insight or recommendation card.

    Args:
        title: Insight title
        content: Insight content
        insight_type: Type of insight ('info', 'warning', 'success', 'error')
    """
    icon_map = {"info": "ℹ️", "warning": "⚠️", "success": "✅", "error": "❌"}

    icon = icon_map.get(insight_type, "ℹ️")

    insight_html = f"""
    <div class="insight-card">
        <div class="insight-header">
            {icon} {title}
        </div>
        <div>{content}</div>
    </div>
    """

    st.markdown(insight_html, unsafe_allow_html=True)
