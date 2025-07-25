"""
Database Browser - Neo4j and Qdrant interface via API
Provides browsing and management interface for database content
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .api_integration import FileManagerAPI
from .ui_components import render_data_table, render_metrics_row, render_search_filters


class DatabaseBrowser:
    """Database browsing interface using API calls"""

    def __init__(self, api: FileManagerAPI):
        self.api = api
        self.domains = [
            "mathematics",
            "science",
            "philosophy",
            "religion",
            "art",
            "language",
            "general",
        ]

    def render_neo4j(self):
        """Render Neo4j database browser"""
        st.subheader("🗄️ Neo4j Knowledge Graph")

        # Overview metrics
        self._render_neo4j_overview()

        # Concept browser
        self._render_concept_browser()

        # Concept management
        with st.expander("➕ Concept Management"):
            self._render_concept_management()

    def render_qdrant(self):
        """Render Qdrant vector database browser"""
        st.subheader("🔍 Qdrant Vector Collections")

        # Collections overview
        self._render_collections_overview()

        # Vector search
        self._render_vector_search()

        # Collection management
        with st.expander("⚙️ Collection Management"):
            self._render_collection_management()

    def _render_neo4j_overview(self):
        """Display Neo4j database overview"""
        try:
            overview = self.api.get_neo4j_overview()

            if overview:
                st.subheader("📊 Database Overview")

                # Extract metrics from overview
                concepts_count = overview.get("concepts_count", 0)
                relationships_count = overview.get("relationships_count", 0)
                domains_count = overview.get("domains_count", 0)

                render_metrics_row(
                    [
                        ("Concepts", concepts_count),
                        ("Relationships", relationships_count),
                        ("Domains", domains_count),
                    ]
                )

                # Domain distribution
                if "domain_distribution" in overview:
                    st.subheader("📈 Domain Distribution")
                    domain_df = pd.DataFrame(overview["domain_distribution"])
                    st.bar_chart(domain_df.set_index("domain")["count"])
            else:
                st.warning("Unable to fetch Neo4j overview")

        except Exception as e:
            self.api.handle_api_error("Neo4j Overview", e)

    def _render_concept_browser(self):
        """Render concept browsing interface"""
        st.subheader("🔍 Browse Concepts")

        # Search filters
        col1, col2, col3 = st.columns(3)

        with col1:
            domain_filter = st.selectbox(
                "Domain Filter", ["All"] + self.domains, key="neo4j_domain_filter"
            )

        with col2:
            limit = st.number_input(
                "Results Limit",
                min_value=10,
                max_value=500,
                value=50,
                key="neo4j_limit",
            )

        with col3:
            if st.button("🔍 Search Concepts", key="search_concepts"):
                self._load_concepts(domain_filter, limit)

        # Display results
        if "neo4j_concepts" in st.session_state:
            concepts = st.session_state.neo4j_concepts

            if concepts:
                # Convert to DataFrame for display
                concepts_df = pd.DataFrame(concepts)

                # Display table with selection
                selected_concept = render_data_table(
                    concepts_df, key="concepts_table", selection_mode="single"
                )

                # Show selected concept details
                if selected_concept:
                    self._show_concept_details(selected_concept)
            else:
                st.info("No concepts found")

    def _render_concept_management(self):
        """Render concept creation/editing interface"""
        tab1, tab2 = st.tabs(["➕ Create Concept", "✏️ Edit Concept"])

        with tab1:
            self._render_create_concept_form()

        with tab2:
            self._render_edit_concept_form()

    def _render_create_concept_form(self):
        """Render form for creating new concepts"""
        with st.form("create_concept_form"):
            st.subheader("Create New Concept")

            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Concept Name*", key="new_concept_name")
                domain = st.selectbox("Domain*", self.domains, key="new_concept_domain")
                concept_type = st.selectbox(
                    "Type",
                    ["Entity", "Idea", "Work", "Event", "Person"],
                    key="new_concept_type",
                )

            with col2:
                description = st.text_area("Description", key="new_concept_description")
                tags = st.text_input("Tags (comma-separated)", key="new_concept_tags")
                source = st.text_input("Source", key="new_concept_source")

            submitted = st.form_submit_button("➕ Create Concept")

            if submitted and name and domain:
                concept_data = {
                    "name": name,
                    "domain": domain,
                    "type": concept_type,
                    "description": description,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                    "source": source,
                }

                try:
                    result = self.api.create_concept(concept_data)
                    if result:
                        st.success(f"✅ Concept '{name}' created successfully!")
                        # Refresh concepts list
                        if "neo4j_concepts" in st.session_state:
                            del st.session_state.neo4j_concepts
                    else:
                        st.error("❌ Failed to create concept")
                except Exception as e:
                    self.api.handle_api_error("Create Concept", e)

    def _render_edit_concept_form(self):
        """Render form for editing existing concepts"""
        if "selected_concept_id" not in st.session_state:
            st.info("Select a concept from the browser above to edit")
            return

        concept_id = st.session_state.selected_concept_id
        st.write(f"Editing concept: {concept_id}")

        # Implementation would load concept details and show edit form
        st.info("Edit functionality - implementation pending")

    def _render_collections_overview(self):
        """Display Qdrant collections overview"""
        try:
            collections = self.api.get_qdrant_collections()

            if collections:
                st.subheader("📊 Vector Collections")

                collections_data = []
                for collection in collections:
                    collections_data.append(
                        {
                            "Collection": collection.get("name", "Unknown"),
                            "Vectors": collection.get("vectors_count", 0),
                            "Dimension": collection.get("config", {})
                            .get("params", {})
                            .get("vectors", {})
                            .get("size", "Unknown"),
                            "Status": collection.get("status", "Unknown"),
                        }
                    )

                if collections_data:
                    collections_df = pd.DataFrame(collections_data)
                    st.dataframe(collections_df, use_container_width=True)
                else:
                    st.info("No collections found")
            else:
                st.warning("Unable to fetch Qdrant collections")

        except Exception as e:
            self.api.handle_api_error("Qdrant Collections", e)

    def _render_vector_search(self):
        """Render vector search interface"""
        st.subheader("🔍 Vector Search")

        col1, col2 = st.columns([2, 1])

        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter text to search for similar vectors...",
                key="vector_search_query",
            )

        with col2:
            collection_name = st.selectbox(
                "Collection",
                ["concepts", "documents", "relationships"],  # Default collections
                key="vector_search_collection",
            )

        if st.button("🔍 Search Vectors") and search_query:
            try:
                results = self.api.search_vectors(
                    collection_name, search_query, limit=20
                )

                if results:
                    st.subheader("🎯 Search Results")

                    for i, result in enumerate(results):
                        with st.expander(
                            f"Result {i+1} (Score: {result.get('score', 0):.4f})"
                        ):
                            st.json(result.get("payload", {}))
                else:
                    st.info("No similar vectors found")
            except Exception as e:
                self.api.handle_api_error("Vector Search", e)

    def _render_collection_management(self):
        """Render collection management interface"""
        st.subheader("⚙️ Collection Operations")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Collection Info**")
            selected_collection = st.selectbox(
                "Select Collection",
                ["concepts", "documents", "relationships"],
                key="collection_mgmt_select",
            )

            if st.button("📊 Get Collection Info"):
                try:
                    info = self.api.get_qdrant_collection_info(selected_collection)
                    if info:
                        st.json(info)
                    else:
                        st.warning("Unable to fetch collection info")
                except Exception as e:
                    self.api.handle_api_error("Collection Info", e)

        with col2:
            st.write("**Operations**")
            st.info(
                "Collection operations will be implemented based on API availability"
            )

    def _load_concepts(self, domain_filter: str, limit: int):
        """Load concepts from Neo4j via API"""
        domain = None if domain_filter == "All" else domain_filter.lower()

        try:
            concepts = self.api.get_neo4j_concepts(domain=domain, limit=limit)
            st.session_state.neo4j_concepts = concepts or []

            if concepts:
                st.success(f"✅ Loaded {len(concepts)} concepts")
            else:
                st.warning("No concepts found matching the criteria")
        except Exception as e:
            self.api.handle_api_error("Load Concepts", e)
            st.session_state.neo4j_concepts = []

    def _show_concept_details(self, concept: Dict):
        """Display detailed information about selected concept"""
        st.subheader("📝 Concept Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {concept.get('name', 'Unknown')}")
            st.write(f"**Domain:** {concept.get('domain', 'Unknown')}")
            st.write(f"**Type:** {concept.get('type', 'Unknown')}")

        with col2:
            st.write(f"**ID:** {concept.get('id', 'Unknown')}")
            st.write(f"**Created:** {concept.get('created_at', 'Unknown')}")
            st.write(f"**Updated:** {concept.get('updated_at', 'Unknown')}")

        if concept.get("description"):
            st.write(f"**Description:** {concept['description']}")

        if concept.get("tags"):
            st.write(f"**Tags:** {', '.join(concept['tags'])}")

        # Store selected concept for editing
        if st.button("✏️ Edit This Concept"):
            st.session_state.selected_concept_id = concept.get("id")
            st.rerun()
