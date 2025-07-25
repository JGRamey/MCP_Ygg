"""
Graph Editor UI Components

This module contains all Streamlit UI components, filters, controls,
and interface elements for the Graph Editor.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .models import (
    DOMAIN_OPTIONS,
    NODE_TYPE_OPTIONS,
    GraphFilters,
    GraphMode,
    LayoutType,
    RelationshipFilters,
)
from .neo4j_connector import DataSourceManager, Neo4jConnector


class GraphEditorUI:
    """Main UI manager for Graph Editor interface"""

    def __init__(self):
        self.neo4j_connector = Neo4jConnector()
        self.data_source_manager = DataSourceManager()

    def render_page_header(self):
        """
        Page header and title rendering

        Purpose: Displays the main page title and description
        Use: Called once at the top of the main interface
        """
        st.markdown("# 📊 Graph Editor")
        st.markdown("**Interactive knowledge graph visualization and editing**")

    def render_sidebar_controls(self) -> tuple[str, str, int, int]:
        """
        Sidebar control panel rendering

        Purpose: Creates all sidebar controls for graph settings and filters
        Use: Called once to set up the sidebar interface
        Returns: Tuple of (graph_mode, layout_type, node_size, edge_width)
        """
        with st.sidebar:
            st.markdown("### 🎛️ Graph Controls")

            # Graph mode selection
            graph_mode = st.selectbox(
                "Graph Mode",
                [mode.value for mode in GraphMode],
                help="Choose the type of graph visualization",
            )

            st.markdown("---")

            # Layout controls section
            st.markdown("### 📐 Layout Settings")
            layout_type = st.selectbox(
                "Layout Algorithm",
                [layout.value for layout in LayoutType],
                help="Choose how nodes are positioned",
            )

            node_size = st.slider("Node Size", 10, 50, 20, help="Size of concept nodes")
            edge_width = st.slider(
                "Edge Width", 1, 5, 2, help="Width of relationship lines"
            )

            st.markdown("---")

            # Filters section
            self.render_graph_filters()

            return graph_mode, layout_type, node_size, edge_width

    def render_graph_filters(self):
        """
        Filter controls rendering

        Purpose: Creates filtering options for domains, types, and relationships
        Use: Called from sidebar to provide data filtering capabilities
        """
        st.markdown("### 🔍 Filters")

        # Domain filter
        selected_domains = st.multiselect(
            "Domains",
            DOMAIN_OPTIONS,
            default=["All Domains"],
            help="Filter concepts by knowledge domain",
        )

        # Type filter
        selected_types = st.multiselect(
            "Node Types",
            NODE_TYPE_OPTIONS,
            default=["All Types"],
            help="Filter by concept hierarchy level",
        )

        # Level filter
        level_range = st.slider(
            "Level Range", 1, 10, (1, 5), help="Filter by concept depth in hierarchy"
        )

        # Store filters in session state
        st.session_state.graph_filters = {
            "domains": selected_domains,
            "types": selected_types,
            "level_range": level_range,
        }

        # Relationship filters subsection
        st.markdown("#### 🔗 Relationships")
        show_relationships = st.checkbox("Show Relationships", value=True)
        min_relationship_strength = st.slider(
            "Min Relationship Strength", 0.0, 1.0, 0.0, help="Filter weak relationships"
        )

        st.session_state.relationship_filters = {
            "show": show_relationships,
            "min_strength": min_relationship_strength,
        }

    def render_connection_status(self) -> bool:
        """
        Neo4j connection status display

        Purpose: Shows database connection status and provides startup instructions
        Use: Called at the top of graph views to inform users about data source
        Returns: True if connected, False otherwise
        """
        neo4j_status = self.neo4j_connector.check_connection()

        if not neo4j_status.connected:
            st.warning(
                f"""
            ⚠️ **Neo4j Database is not connected**
            
            {neo4j_status.message}
            
            **To start Neo4j:**
            ```bash
            # Using Docker Compose (recommended)
            docker-compose up -d neo4j
            
            # Or start all services
            docker-compose up -d
            ```
            
            Currently showing data from CSV files or demo data.
            """
            )
            return False
        return True

    def render_graph_controls(self) -> Dict[str, Any]:
        """
        Graph control buttons rendering

        Purpose: Creates action buttons for graph operations (refresh, save, report)
        Use: Called above the main graph to provide user controls
        Returns: Dictionary indicating which actions were triggered
        """
        control_col1, control_col2, control_col3 = st.columns(3)
        actions = {}

        with control_col1:
            if st.button("🔄 Refresh Graph", use_container_width=True):
                st.cache_data.clear()
                actions["refresh"] = True

        with control_col2:
            if st.button("💾 Save Layout", use_container_width=True):
                st.success("Layout saved to session")
                actions["save"] = True

        with control_col3:
            if st.button("📊 Generate Report", use_container_width=True):
                actions["report"] = True

        return actions

    def render_data_source_indicator(self, data_source_type: str):
        """
        Data source status indicator

        Purpose: Shows users what data source is currently being used
        Use: Called when displaying graphs to indicate data origin
        """
        if data_source_type == "csv":
            st.info("📂 Showing data from CSV files")
        elif data_source_type == "demo":
            st.info("🎭 Showing demonstration data")
        elif data_source_type == "neo4j":
            st.success("🗄️ Connected to Neo4j database")

    def render_graph_metrics(self, concepts: List[Dict[str, Any]]):
        """
        Graph metrics sidebar panel

        Purpose: Displays statistics about the current graph visualization
        Use: Called in sidebar to show graph information
        """
        st.markdown("### 📊 Graph Metrics")

        if not concepts:
            st.info("No data to display")
            return

        # Basic metrics
        st.metric("Total Concepts", len(concepts))

        # Domain distribution
        domain_counts = {}
        for concept in concepts:
            domain = concept["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        st.markdown("#### Domain Distribution")
        for domain, count in domain_counts.items():
            st.metric(domain.title(), count)

    def render_concept_search(self) -> Optional[str]:
        """
        Concept search interface

        Purpose: Provides search functionality for finding specific concepts
        Use: Called in focused view mode to allow concept selection
        Returns: Selected concept ID or None
        """
        concept_search = st.text_input(
            "🔍 Search for concept to focus on",
            placeholder="Enter concept name or ID...",
            help="Search for a specific concept to center the view on",
        )

        if concept_search:
            # Search for matching concepts
            try:
                # Get some concepts to search through
                concepts, _ = self.data_source_manager.get_concepts_with_fallback(
                    limit=100
                )

                matching_concepts = [
                    c
                    for c in concepts
                    if concept_search.lower() in c["name"].lower()
                    or concept_search.lower() in c["id"].lower()
                ]

                if matching_concepts:
                    concept_options = [
                        f"{c['id']}: {c['name']}" for c in matching_concepts
                    ]
                    selected_concept = st.selectbox("Select Concept", concept_options)

                    if selected_concept:
                        return selected_concept.split(":")[0]
                else:
                    st.info("No concepts found matching your search")
            except Exception as e:
                st.error(f"Search error: {str(e)}")

        return None

    def render_concept_details(self, concept: Dict[str, Any]):
        """
        Concept detail panel rendering

        Purpose: Shows detailed information about a selected concept
        Use: Called when user selects or focuses on a specific concept
        """
        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <h4>{concept['id']}: {concept['name']}</h4>
            <p><strong>Domain:</strong> {concept['domain']}</p>
            <p><strong>Type:</strong> {concept['type']}</p>
            <p><strong>Level:</strong> {concept.get('level', 'N/A')}</p>
            <p><strong>Description:</strong> {concept.get('description', 'No description available')}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "✏️ Edit", key=f"edit_graph_{concept['id']}", use_container_width=True
            ):
                st.session_state.edit_concept_id = concept["id"]
                st.switch_page("pages/01_🗄️_Database_Manager.py")

        with col2:
            if st.button(
                "🎯 Focus", key=f"focus_graph_{concept['id']}", use_container_width=True
            ):
                st.session_state.focus_concept_id = concept["id"]
                st.rerun()

        with col3:
            if st.button(
                "🔗 Relations",
                key=f"rel_graph_{concept['id']}",
                use_container_width=True,
            ):
                st.session_state.view_relationships_id = concept["id"]
                st.switch_page("pages/01_🗄️_Database_Manager.py")

    def render_relationship_analysis(
        self, concepts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Relationship analysis interface

        Purpose: Analyzes and displays relationship patterns in the knowledge graph
        Use: Called in relationship analysis mode to show network statistics
        Returns: Analysis results dictionary
        """
        try:
            analysis = {
                "total_concepts": len(concepts),
                "total_relationships": 0,
                "type_distribution": {},
                "top_connected": [],
                "cross_domain_connections": [],
                "avg_connections": 0,
                "density": 0,
            }

            # Count connections for each concept (limited for performance)
            concept_connections = {}
            relationship_types = {}

            for concept in concepts[:100]:  # Limit for performance
                relationships = self.neo4j_connector.get_concept_relationships(
                    concept["id"]
                )
                concept_connections[concept["id"]] = {
                    "name": concept["name"],
                    "count": len(relationships),
                }
                analysis["total_relationships"] += len(relationships)

                for rel in relationships:
                    rel_type = rel["type"]
                    relationship_types[rel_type] = (
                        relationship_types.get(rel_type, 0) + 1
                    )

            analysis["type_distribution"] = relationship_types

            # Top connected concepts
            top_connected = sorted(
                concept_connections.items(), key=lambda x: x[1]["count"], reverse=True
            )[:10]
            analysis["top_connected"] = [
                [k, v["name"], v["count"]] for k, v in top_connected
            ]

            # Calculate averages
            if concept_connections:
                total_connections = sum(
                    c["count"] for c in concept_connections.values()
                )
                analysis["avg_connections"] = total_connections / len(
                    concept_connections
                )

                # Simple density calculation
                max_possible = len(concepts) * (len(concepts) - 1) / 2
                analysis["density"] = (
                    analysis["total_relationships"] / max_possible
                    if max_possible > 0
                    else 0
                )

            return analysis

        except Exception as e:
            st.error(f"Error in relationship analysis: {str(e)}")
            return {}

    def render_domain_statistics(self, domain: str, concepts: List[Dict[str, Any]]):
        """
        Domain-specific statistics display

        Purpose: Shows detailed statistics for a specific knowledge domain
        Use: Called in domain explorer mode to provide domain insights
        """
        st.markdown(f"### 📊 {domain.title()} Domain Statistics")

        # Calculate type and level distributions
        type_counts = {}
        level_counts = {}

        for concept in concepts:
            concept_type = concept["type"]
            concept_level = concept.get("level", 1)

            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
            level_counts[concept_level] = level_counts.get(concept_level, 0) + 1

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Type Distribution")
            for concept_type, count in type_counts.items():
                st.metric(concept_type.title(), count)

        with col2:
            st.markdown("#### Level Distribution")
            for level, count in sorted(level_counts.items()):
                st.metric(f"Level {level}", count)


class GraphReportGenerator:
    """Generates comprehensive reports about the graph"""

    @staticmethod
    def generate_graph_report(concepts: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """
        Graph analysis report generation

        Purpose: Creates comprehensive analysis report of the current graph
        Use: Called when user requests detailed graph analysis
        """
        st.success("📊 Graph report generated!")

        # This would generate a comprehensive report
        st.markdown("### 📈 Graph Analysis Report")
        st.markdown(f"**Total Concepts:** {len(concepts)}")
        st.markdown(
            f"**Total Relationships:** {analysis.get('total_relationships', 0)}"
        )
        st.markdown(
            f"**Average Connections:** {analysis.get('avg_connections', 0):.2f}"
        )
        st.markdown(f"**Network Density:** {analysis.get('density', 0):.4f}")

        # Add timestamp
        from datetime import datetime

        st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
