"""
UI Components - Reusable UI elements for file manager
Provides consistent UI patterns and components
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def render_data_table(
    df: pd.DataFrame, key: str, selection_mode: str = "multiple", height: int = 400
) -> Optional[Dict]:
    """
    Render a data table with selection capabilities

    Args:
        df: DataFrame to display
        key: Unique key for the table
        selection_mode: "single", "multiple", or "none"
        height: Table height in pixels

    Returns:
        Selected row(s) data or None
    """
    if df.empty:
        st.info("No data to display")
        return None

    # Display dataframe with selection
    if selection_mode == "none":
        st.dataframe(df, height=height, use_container_width=True, key=key)
        return None

    # For selection modes, use data_editor
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=height,
        key=key,
        disabled=list(df.columns),  # Make columns read-only
        hide_index=True,
    )

    # Return selection (simplified - actual implementation would depend on Streamlit version)
    return None


def render_metrics_row(metrics: List[Tuple[str, Any]], columns: int = None):
    """
    Render a row of metrics with consistent styling

    Args:
        metrics: List of (label, value) tuples
        columns: Number of columns (defaults to len(metrics))
    """
    if not metrics:
        return

    num_cols = columns or len(metrics)
    cols = st.columns(num_cols)

    for i, (label, value) in enumerate(metrics):
        if i < len(cols):
            with cols[i]:
                st.metric(label, value)


def render_search_filters(
    domains: List[str], content_types: List[str] = None, key_prefix: str = "filter"
) -> Dict[str, Any]:
    """
    Render standardized search/filter controls

    Args:
        domains: List of available domains
        content_types: List of available content types
        key_prefix: Prefix for widget keys

    Returns:
        Dictionary of filter values
    """
    filters = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        filters["domain"] = st.selectbox(
            "Domain", ["All"] + domains, key=f"{key_prefix}_domain"
        )

    with col2:
        if content_types:
            filters["content_type"] = st.selectbox(
                "Content Type",
                ["All"] + content_types,
                key=f"{key_prefix}_content_type",
            )

    with col3:
        filters["limit"] = st.number_input(
            "Results Limit",
            min_value=10,
            max_value=500,
            value=50,
            key=f"{key_prefix}_limit",
        )

    return filters


def render_file_upload_area(
    accepted_types: List[str], max_file_size_mb: int = 10, key: str = "file_upload"
) -> Optional[Any]:
    """
    Render file upload area with validation

    Args:
        accepted_types: List of accepted file extensions
        max_file_size_mb: Maximum file size in MB
        key: Unique key for the uploader

    Returns:
        Uploaded file object or None
    """
    uploaded_file = st.file_uploader(
        f"Upload file ({', '.join(accepted_types)})",
        type=accepted_types,
        key=key,
        help=f"Maximum file size: {max_file_size_mb}MB",
    )

    if uploaded_file:
        # Validate file size
        if uploaded_file.size > max_file_size_mb * 1024 * 1024:
            st.error(f"File too large! Maximum size: {max_file_size_mb}MB")
            return None

        # Show file info
        st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

    return uploaded_file


def render_confirmation_dialog(
    title: str,
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    key: str = "confirm_dialog",
) -> bool:
    """
    Render a confirmation dialog

    Args:
        title: Dialog title
        message: Confirmation message
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button
        key: Unique key for the dialog

    Returns:
        True if confirmed, False otherwise
    """
    st.subheader(title)
    st.write(message)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button(confirm_text, type="primary", key=f"{key}_confirm"):
            return True

    with col2:
        if st.button(cancel_text, key=f"{key}_cancel"):
            return False

    return False


def render_progress_indicator(
    current: int, total: int, message: str = "", key: str = "progress"
):
    """
    Render progress indicator with message

    Args:
        current: Current progress value
        total: Total progress value
        message: Optional progress message
        key: Unique key for the progress bar
    """
    if total <= 0:
        return

    progress = current / total
    progress_bar = st.progress(progress, key=f"{key}_bar")

    if message:
        st.text(f"{message} ({current}/{total})")
    else:
        st.text(f"Progress: {current}/{total} ({progress:.1%})")


def render_status_badge(status: str, key: str = None) -> None:
    """
    Render status badge with appropriate styling

    Args:
        status: Status text
        key: Optional unique key
    """
    status_colors = {
        "success": "🟢",
        "warning": "🟡",
        "error": "🔴",
        "info": "🔵",
        "pending": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "failed": "❌",
    }

    icon = status_colors.get(status.lower(), "⚪")
    st.write(f"{icon} {status.title()}")


def render_expandable_json(
    data: Dict[str, Any],
    title: str = "Details",
    expanded: bool = False,
    key: str = "json_view",
):
    """
    Render JSON data in an expandable section

    Args:
        data: Dictionary data to display
        title: Section title
        expanded: Whether to start expanded
        key: Unique key for the expander
    """
    with st.expander(title, expanded=expanded, key=key):
        st.json(data)


def render_action_buttons(
    actions: List[Tuple[str, str, str]], key_prefix: str = "action"
) -> Optional[str]:
    """
    Render a row of action buttons

    Args:
        actions: List of (label, icon, action_id) tuples
        key_prefix: Prefix for button keys

    Returns:
        Action ID of clicked button or None
    """
    if not actions:
        return None

    cols = st.columns(len(actions))

    for i, (label, icon, action_id) in enumerate(actions):
        with cols[i]:
            if st.button(f"{icon} {label}", key=f"{key_prefix}_{action_id}"):
                return action_id

    return None


def render_data_preview(
    data: Any, title: str = "Data Preview", max_rows: int = 5, key: str = "preview"
):
    """
    Render a preview of data (DataFrame, list, dict, etc.)

    Args:
        data: Data to preview
        title: Preview section title
        max_rows: Maximum rows to show for DataFrames
        key: Unique key for the preview
    """
    with st.expander(title, key=key):
        if isinstance(data, pd.DataFrame):
            if len(data) > max_rows:
                st.write(f"Showing first {max_rows} of {len(data)} rows:")
                st.dataframe(data.head(max_rows), use_container_width=True)
            else:
                st.dataframe(data, use_container_width=True)
        elif isinstance(data, (list, tuple)):
            if len(data) > max_rows:
                st.write(f"Showing first {max_rows} of {len(data)} items:")
                for i, item in enumerate(data[:max_rows]):
                    st.write(f"{i+1}. {item}")
                st.write("...")
            else:
                for i, item in enumerate(data):
                    st.write(f"{i+1}. {item}")
        elif isinstance(data, dict):
            st.json(data)
        else:
            st.write(data)


def render_search_box(
    placeholder: str = "Search...", key: str = "search", help_text: str = None
) -> str:
    """
    Render a search input box with standardized styling

    Args:
        placeholder: Placeholder text
        key: Unique key for the input
        help_text: Optional help text

    Returns:
        Search query string
    """
    return st.text_input(
        "🔍 Search",
        placeholder=placeholder,
        key=key,
        help=help_text,
        label_visibility="collapsed",
    )


def render_loading_spinner(message: str = "Loading..."):
    """
    Render a loading spinner with message

    Args:
        message: Loading message
    """
    with st.spinner(message):
        return st.empty()  # Return placeholder for content
