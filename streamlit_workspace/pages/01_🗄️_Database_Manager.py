"""
Database Manager - Complete CRUD Operations

API-FIRST IMPLEMENTATION: Uses FastAPI backend via API client
Eliminates direct database calls for true separation of concerns.

Features:
- Complete CRUD operations via API endpoints
- Real-time API connectivity status
- Async operations with progress indicators
- Comprehensive error handling and user feedback
- Domain statistics and relationship management
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add utils to path for API client
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import api_client, run_async


def main():
    """Main Database Manager interface"""

    st.set_page_config(
        page_title="Database Manager - MCP Yggdrasil", page_icon="🗄️", layout="wide"
    )

    # Apply custom CSS
    st.markdown(
        """
    <style>
    .concept-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .concept-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2E8B57;
        margin-bottom: 0.5rem;
    }
    
    .concept-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E8B57;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("# 🗄️ Database Manager")
    st.markdown("**Complete CRUD operations for concepts, relationships, and domains**")

    # API status check
    show_api_status()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Database Operations")

        operation = st.selectbox(
            "Select Operation",
            [
                "📊 Overview",
                "🔍 Browse & Search",
                "➕ Create Concept",
                "✏️ Edit Concept",
                "🔗 Manage Relationships",
                "📈 Domain Management",
            ],
        )

        st.markdown("---")

        # Quick stats in sidebar
        show_quick_stats_sidebar()

    # Main content based on selection
    if operation == "📊 Overview":
        show_overview()
    elif operation == "🔍 Browse & Search":
        show_browse_search()
    elif operation == "➕ Create Concept":
        show_create_concept()
    elif operation == "✏️ Edit Concept":
        show_edit_concept()
    elif operation == "🔗 Manage Relationships":
        show_manage_relationships()
    elif operation == "📈 Domain Management":
        show_domain_management()


@run_async
async def show_api_status():
    """Show API connection status"""
    try:
        # Simple health check by searching for concepts
        result = await api_client.search_concepts(query="", limit=1)
        if result is not None:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Unavailable")
    except Exception as e:
        st.error(f"🔴 API Error: {str(e)}")


@run_async
async def show_quick_stats_sidebar():
    """Show quick statistics in sidebar via API"""
    st.markdown("### 📊 Quick Stats")

    try:
        # Get stats through API calls
        with st.spinner("Loading stats..."):
            # Get total concepts across all domains
            all_concepts = await api_client.search_concepts(query="", limit=1000)
            total_concepts = len(all_concepts) if all_concepts else 0

            # Get unique domains
            domains = set()
            concept_types = set()
            if all_concepts:
                for concept in all_concepts:
                    if concept.get("domain"):
                        domains.add(concept["domain"])
                    if concept.get("type"):
                        concept_types.add(concept["type"])

            st.metric("Concepts", total_concepts)
            st.metric("Domains", len(domains))
            st.metric("Types", len(concept_types))
            st.metric("API Status", "🟢 Online")

    except Exception as e:
        st.error(f"Could not load stats: {str(e)}")
        st.metric("Concepts", "Error")
        st.metric("Domains", "Error")
        st.metric("Types", "Error")
        st.metric("API Status", "🔴 Offline")


@run_async
async def show_overview():
    """Show database overview via API"""
    st.markdown("## 📊 Database Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Domain Distribution")

        with st.spinner("Loading domain data from API..."):
            # Get all concepts and calculate domain distribution
            all_concepts = await api_client.search_concepts(query="", limit=1000)

            if all_concepts:
                # Calculate domain distribution
                domain_counts = {}
                for concept in all_concepts:
                    domain = concept.get("domain", "Unknown")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Create DataFrame for visualization
                df = pd.DataFrame(
                    [
                        {"domain": domain, "concept_count": count}
                        for domain, count in domain_counts.items()
                    ]
                )

                # Create pie chart
                fig = px.pie(
                    df,
                    values="concept_count",
                    names="domain",
                    title="Concepts by Domain",
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

                # Show table
                st.markdown("### 📋 Domain Details")
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No domain data available from API")

    with col2:
        st.markdown("### 🕒 Recent Activity")
        if "operation_history" in st.session_state:
            recent_ops = st.session_state.operation_history[-5:]
            for op in reversed(recent_ops):
                st.markdown(f"**{op['operation_type']}**: {op['description']}")
                st.caption(op["timestamp"])
        else:
            st.info("No recent activity")


@run_async
async def show_browse_search():
    """Show browse and search interface via API"""
    st.markdown("## 🔍 Browse & Search Concepts")

    # Search controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_term = st.text_input(
            "🔍 Search concepts", placeholder="Enter search term..."
        )

    with col2:
        domain_filter = st.selectbox(
            "Domain Filter",
            [
                "All Domains",
                "Art",
                "Science",
                "Mathematics",
                "Philosophy",
                "Language",
                "Technology",
                "Religion",
            ],
        )

    with col3:
        limit = st.number_input("Results Limit", min_value=10, max_value=500, value=50)

    # Search button
    if st.button("🔍 Search", type="primary") or search_term:
        await perform_search(search_term, domain_filter, limit)


async def perform_search(search_term: str, domain_filter: str, limit: int):
    """Perform search via API"""
    with st.spinner("Searching via API..."):
        domain = None if domain_filter == "All Domains" else domain_filter.lower()

        if search_term:
            concepts = await api_client.search_concepts(
                query=search_term, domain=domain, limit=limit
            )
        else:
            # Get all concepts for the domain
            concepts = await api_client.search_concepts(
                query="", domain=domain, limit=limit
            )

        if concepts:
            st.markdown(f"### Found {len(concepts)} concepts")

            # Display results
            for concept in concepts:
                show_concept_card(concept)
        else:
            st.info("No concepts found matching your criteria")


def show_concept_card(concept: Dict[str, Any]):
    """Display a concept card with actions"""
    with st.container():
        st.markdown(
            f"""
        <div class="concept-card">
            <div class="concept-header">{concept.get('id', 'N/A')}: {concept.get('name', 'Unknown')}</div>
            <div class="concept-meta">Domain: {concept.get('domain', 'N/A')} | Type: {concept.get('type', 'N/A')} | Level: {concept.get('level', 'N/A')}</div>
            <p>{concept.get('description', 'No description available')[:200]}...</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Action buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                f"✏️ Edit", key=f"edit_{concept.get('id')}", use_container_width=True
            ):
                st.session_state.edit_concept_id = concept.get("id")
                st.rerun()

        with col2:
            if st.button(
                f"🔗 Relations",
                key=f"rel_{concept.get('id')}",
                use_container_width=True,
            ):
                st.session_state.view_relationships_id = concept.get("id")
                st.rerun()

        with col3:
            if st.button(
                f"📋 Details", key=f"view_{concept.get('id')}", use_container_width=True
            ):
                show_concept_details(concept.get("id"))

        with col4:
            if st.button(
                f"🗑️ Delete",
                key=f"del_{concept.get('id')}",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.delete_concept_id = concept.get("id")
                st.rerun()


@run_async
async def show_concept_details(concept_id: str):
    """Show detailed concept information via API"""
    try:
        # Search for the specific concept
        results = await api_client.search_concepts(query=concept_id, limit=1)

        if results:
            concept = results[0]
            st.json(concept)
        else:
            st.warning(f"Could not find concept {concept_id}")

    except Exception as e:
        st.error(f"Error loading concept details: {str(e)}")


def show_create_concept():
    """Show create concept form with API integration"""
    st.markdown("## ➕ Create New Concept")

    with st.form("create_concept_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            concept_id = st.text_input("Concept ID*", placeholder="e.g., ART0001")
            concept_name = st.text_input(
                "Concept Name*", placeholder="e.g., Renaissance_Art"
            )
            domain = st.selectbox(
                "Domain*",
                [
                    "Art",
                    "Science",
                    "Mathematics",
                    "Philosophy",
                    "Language",
                    "Technology",
                    "Religion",
                ],
            )

        with col2:
            concept_type = st.selectbox("Type*", ["root", "sub_root", "branch", "leaf"])
            level = st.number_input("Level*", min_value=1, max_value=10, value=1)
            location = st.text_input("Location", placeholder="Optional")

        description = st.text_area(
            "Description", placeholder="Detailed description of the concept..."
        )

        # Optional metadata
        with st.expander("📅 Additional Metadata"):
            col1, col2 = st.columns(2)
            with col1:
                earliest_date = st.number_input(
                    "Earliest Evidence Date",
                    value=None,
                    placeholder="Year (e.g., 1400)",
                )
                latest_date = st.number_input(
                    "Latest Evidence Date", value=None, placeholder="Year (e.g., 1600)"
                )
            with col2:
                certainty_level = st.selectbox(
                    "Certainty Level", ["High", "Medium", "Low", "Unknown"]
                )
                cultural_context = st.text_input(
                    "Cultural Context", placeholder="e.g., European, Ancient Greek"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("➕ Create Concept", type="primary")

        if submitted:
            if not all([concept_id, concept_name, domain, concept_type]):
                st.error("Please fill in all required fields (marked with *)")
            else:
                create_concept_via_api(
                    concept_id,
                    concept_name,
                    domain,
                    concept_type,
                    level,
                    description,
                    location,
                    certainty_level,
                    cultural_context,
                    earliest_date,
                    latest_date,
                )


@run_async
async def create_concept_via_api(
    concept_id: str,
    concept_name: str,
    domain: str,
    concept_type: str,
    level: int,
    description: str,
    location: str,
    certainty_level: str,
    cultural_context: str,
    earliest_date: Optional[int],
    latest_date: Optional[int],
):
    """Create concept via API"""
    concept_data = {
        "id": concept_id,
        "name": concept_name,
        "domain": domain,
        "type": concept_type,
        "level": level,
        "description": description,
        "location": location,
        "certainty_level": certainty_level,
        "cultural_context": cultural_context,
    }

    # Add optional dates if provided
    if earliest_date:
        concept_data["earliest_evidence_date"] = int(earliest_date)
    if latest_date:
        concept_data["latest_evidence_date"] = int(latest_date)

    try:
        with st.spinner("Creating concept via API..."):
            result = await api_client.manage_database("create", concept_data)

            if result:
                st.success(f"✅ Concept {concept_id} created successfully")
                add_to_history(
                    "CREATE", f"Created concept {concept_id}: {concept_name}"
                )
            else:
                st.error("❌ Failed to create concept")

    except Exception as e:
        st.error(f"❌ Error creating concept: {str(e)}")


def show_edit_concept():
    """Show edit concept interface with API integration"""
    st.markdown("## ✏️ Edit Concept")

    # Concept selection
    if "edit_concept_id" not in st.session_state:
        st.info(
            "Select a concept to edit from the Browse & Search page, or enter a concept ID below:"
        )

        concept_id = st.text_input("Concept ID", placeholder="e.g., ART0001")
        if st.button("Load Concept") and concept_id:
            st.session_state.edit_concept_id = concept_id
            st.rerun()
    else:
        concept_id = st.session_state.edit_concept_id
        load_and_edit_concept(concept_id)


@run_async
async def load_and_edit_concept(concept_id: str):
    """Load and show edit form for concept via API"""
    try:
        # Get concept details via API
        with st.spinner("Loading concept from API..."):
            results = await api_client.search_concepts(query=concept_id, limit=1)

            if not results:
                st.error(f"Concept {concept_id} not found")
                if st.button("Clear Selection"):
                    del st.session_state.edit_concept_id
                    st.rerun()
                return

            concept = results[0]
            st.info(f"Editing: **{concept_id}** - {concept.get('name', 'Unknown')}")

            if st.button("🔄 Select Different Concept"):
                del st.session_state.edit_concept_id
                st.rerun()

            # Edit form
            with st.form("edit_concept_form"):
                st.markdown('<div class="form-section">', unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    new_name = st.text_input(
                        "Concept Name", value=concept.get("name", "")
                    )
                    new_domain = st.selectbox(
                        "Domain",
                        [
                            "Art",
                            "Science",
                            "Mathematics",
                            "Philosophy",
                            "Language",
                            "Technology",
                            "Religion",
                        ],
                        index=(
                            [
                                "Art",
                                "Science",
                                "Mathematics",
                                "Philosophy",
                                "Language",
                                "Technology",
                                "Religion",
                            ].index(concept.get("domain", "Art"))
                            if concept.get("domain")
                            in [
                                "Art",
                                "Science",
                                "Mathematics",
                                "Philosophy",
                                "Language",
                                "Technology",
                                "Religion",
                            ]
                            else 0
                        ),
                    )

                with col2:
                    new_type = st.selectbox(
                        "Type",
                        ["root", "sub_root", "branch", "leaf"],
                        index=(
                            ["root", "sub_root", "branch", "leaf"].index(
                                concept.get("type", "leaf")
                            )
                            if concept.get("type")
                            in ["root", "sub_root", "branch", "leaf"]
                            else 3
                        ),
                    )
                    new_level = st.number_input(
                        "Level",
                        min_value=1,
                        max_value=10,
                        value=concept.get("level", 1),
                    )

                new_description = st.text_area(
                    "Description", value=concept.get("description", "")
                )
                new_location = st.text_input(
                    "Location", value=concept.get("location", "")
                )

                st.markdown("</div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("💾 Save Changes", type="primary")
                with col2:
                    if st.form_submit_button("🗑️ Delete Concept", type="secondary"):
                        st.session_state.confirm_delete = concept_id

                if submitted:
                    await update_concept_via_api(
                        concept_id,
                        new_name,
                        new_domain,
                        new_type,
                        new_level,
                        new_description,
                        new_location,
                    )

    except Exception as e:
        st.error(f"Error loading concept: {str(e)}")


@run_async
async def update_concept_via_api(
    concept_id: str,
    new_name: str,
    new_domain: str,
    new_type: str,
    new_level: int,
    new_description: str,
    new_location: str,
):
    """Update concept via API"""
    updates = {
        "id": concept_id,
        "name": new_name,
        "domain": new_domain,
        "type": new_type,
        "level": new_level,
        "description": new_description,
        "location": new_location,
    }

    try:
        with st.spinner("Updating concept via API..."):
            result = await api_client.manage_database("update", updates)

            if result:
                st.success(f"✅ Concept {concept_id} updated successfully")
                add_to_history("UPDATE", f"Updated concept {concept_id}: {new_name}")
            else:
                st.error("❌ Failed to update concept")

    except Exception as e:
        st.error(f"❌ Error updating concept: {str(e)}")


def show_manage_relationships():
    """Show relationship management interface"""
    st.markdown("## 🔗 Manage Relationships")

    # Show relationships for selected concept
    if "view_relationships_id" in st.session_state:
        concept_id = st.session_state.view_relationships_id
        show_concept_relationships(concept_id)

        if st.button("🔄 Select Different Concept"):
            del st.session_state.view_relationships_id
            st.rerun()
    else:
        st.info(
            "Select a concept to view its relationships from the Browse & Search page, or enter a concept ID below:"
        )

        concept_id = st.text_input("Concept ID", placeholder="e.g., ART0001")
        if st.button("Load Relationships") and concept_id:
            st.session_state.view_relationships_id = concept_id
            st.rerun()


@run_async
async def show_concept_relationships(concept_id: str):
    """Show relationships for a concept via API"""
    try:
        # Get concept details
        with st.spinner("Loading concept relationships from API..."):
            results = await api_client.search_concepts(query=concept_id, limit=1)

            if results:
                concept = results[0]
                st.info(
                    f"Viewing relationships for: **{concept_id}** - {concept.get('name', 'Unknown')}"
                )

                # Note: Relationship data would need specific API endpoints
                st.info(
                    "🚧 Relationship data will be available when relationship endpoints are implemented in the API"
                )

                # Placeholder for relationship visualization
                st.markdown("### 🌐 Relationship Visualization")
                st.info("API endpoint needed: /api/relationships/{concept_id}")

            else:
                st.warning(f"Could not find concept {concept_id}")

    except Exception as e:
        st.error(f"Error loading relationships: {str(e)}")


@run_async
async def show_domain_management():
    """Show domain management interface via API"""
    st.markdown("## 📈 Domain Management")

    with st.spinner("Loading domain data from API..."):
        # Get all concepts and calculate domain statistics
        all_concepts = await api_client.search_concepts(query="", limit=1000)

        if all_concepts:
            # Calculate domain distribution
            domain_counts = {}
            for concept in all_concepts:
                domain = concept.get("domain", "Unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Create domain data
            domains_data = [
                {"domain": domain, "concept_count": count}
                for domain, count in domain_counts.items()
            ]

            st.markdown("### 📊 Current Domains")

            # Enhanced domain display
            for domain_info in domains_data:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"**{domain_info['domain']}**")
                        st.caption(f"{domain_info['concept_count']} concepts")

                    with col2:
                        if st.button(
                            f"📋 View Concepts",
                            key=f"view_domain_{domain_info['domain']}",
                        ):
                            # Set domain filter and switch to browse page
                            st.session_state.browse_domain = domain_info["domain"]
                            st.info(
                                f"Browse {domain_info['domain']} concepts in the Browse & Search tab"
                            )

                    with col3:
                        if st.button(
                            f"📊 Analytics", key=f"analytics_{domain_info['domain']}"
                        ):
                            st.info(
                                f"Analytics for {domain_info['domain']} - Enhanced analytics coming soon!"
                            )

            # Domain statistics chart
            st.markdown("### 📈 Domain Statistics")

            df = pd.DataFrame(domains_data)
            fig = px.bar(
                df,
                x="domain",
                y="concept_count",
                title="Concepts per Domain",
                color="concept_count",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No domain data available from API")


def add_to_history(operation_type: str, description: str):
    """Add operation to history"""
    if "operation_history" not in st.session_state:
        st.session_state.operation_history = []

    from datetime import datetime

    entry = {
        "operation_type": operation_type,
        "description": description,
        "timestamp": datetime.now().isoformat(),
    }

    st.session_state.operation_history.append(entry)


# Handle delete confirmation
if "confirm_delete" in st.session_state:
    concept_id = st.session_state.confirm_delete

    st.warning(
        f"⚠️ Are you sure you want to delete concept **{concept_id}**? This action cannot be undone."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Delete", type="primary"):
            delete_concept_via_api(concept_id)

    with col2:
        if st.button("❌ Cancel"):
            del st.session_state.confirm_delete
            st.rerun()


@run_async
async def delete_concept_via_api(concept_id: str):
    """Delete concept via API"""
    try:
        with st.spinner("Deleting concept via API..."):
            result = await api_client.manage_database("delete", {"id": concept_id})

            if result:
                st.success(f"✅ Concept {concept_id} deleted successfully")
                add_to_history("DELETE", f"Deleted concept {concept_id}")
            else:
                st.error("❌ Failed to delete concept")

            del st.session_state.confirm_delete
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error deleting concept: {str(e)}")


if __name__ == "__main__":
    main()
