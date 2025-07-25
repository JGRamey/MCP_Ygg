"""
Graph Editor - Interactive Knowledge Graph Visualization & Editing

API-FIRST IMPLEMENTATION: Uses FastAPI backend via API client
Eliminates direct database calls for true separation of concerns.

Features:
- Interactive knowledge graph visualization with multiple modes
- API-based graph data fetching with error handling
- Advanced filtering and search capabilities
- Focused concept exploration and relationship analysis
- Real-time connection status through API health checks
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add utils to path for API client
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import api_client, run_async


def main():
    """
    Main entry point for Graph Editor page.

    API-first implementation that delegates all data operations
    to the FastAPI backend via the unified API client.
    """
    st.set_page_config(
        page_title="Graph Editor - MCP Yggdrasil", page_icon="📊", layout="wide"
    )

    # Apply custom CSS styling
    st.markdown(
        """
    <style>
    .graph-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .concept-card {
        background: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .concept-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2E8B57;
        margin-bottom: 0.5rem;
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
    st.markdown("# 📊 Graph Editor")
    st.markdown("**Interactive Knowledge Graph Visualization & Editing**")

    # API health check
    show_api_status()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Graph Controls")

        # Graph mode selection
        graph_mode = st.selectbox(
            "Visualization Mode",
            [
                "🌐 Full Network",
                "🎯 Focused View",
                "🔍 Domain Explorer",
                "📈 Relationship Analysis",
            ],
            help="Select how to visualize the knowledge graph",
        )

        # Domain filter
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
            help="Filter concepts by domain",
        )

        # Depth control for focused views
        if "Focused" in graph_mode:
            depth = st.slider(
                "Connection Depth",
                1,
                4,
                2,
                help="How many levels of connections to show",
            )
        else:
            depth = 2

        # Node limit
        node_limit = st.slider(
            "Max Nodes", 10, 200, 50, help="Maximum number of nodes to display"
        )

        st.markdown("---")

        # Search functionality
        st.markdown("### 🔍 Search")
        search_query = st.text_input(
            "Search Concepts", placeholder="Enter concept name or keyword..."
        )

        if st.button("🔍 Search", use_container_width=True):
            perform_concept_search(search_query, domain_filter)

    # Main content area
    if graph_mode == "🌐 Full Network":
        show_full_network_view(domain_filter, node_limit)
    elif graph_mode == "🎯 Focused View":
        show_focused_view(depth, domain_filter)
    elif graph_mode == "🔍 Domain Explorer":
        show_domain_explorer(domain_filter)
    elif graph_mode == "📈 Relationship Analysis":
        show_relationship_analysis()


@run_async
async def show_api_status():
    """Show API connection status"""
    try:
        # Simple health check by trying to get graph overview
        result = await api_client.get_graph_data()
        if result:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Unavailable")
    except Exception as e:
        st.error(f"🔴 API Error: {str(e)}")


@run_async
async def show_full_network_view(domain_filter: str, node_limit: int):
    """Show full network visualization via API"""
    st.markdown("## 🌐 Full Network View")

    with st.spinner("Loading graph data from API..."):
        # Get graph data via API
        params = {"limit": node_limit}
        if domain_filter != "All Domains":
            params["domain"] = domain_filter.lower()

        graph_data = await api_client.get_graph_data(**params)

        if graph_data and graph_data.get("nodes"):
            # Create network visualization
            create_network_visualization(graph_data, "Full Network")

            # Show graph statistics
            show_graph_statistics(graph_data)
        else:
            st.warning("No graph data available from API")


@run_async
async def show_focused_view(depth: int, domain_filter: str):
    """Show focused concept view via API"""
    st.markdown("## 🎯 Focused View")

    # Concept selection
    col1, col2 = st.columns([2, 1])

    with col1:
        concept_id = st.text_input("Focus Concept ID", placeholder="e.g., SCI0001")

    with col2:
        if st.button("🎯 Focus", use_container_width=True) and concept_id:
            with st.spinner("Loading focused view from API..."):
                # Get focused graph data via API
                graph_data = await api_client.get_graph_data(
                    concept_id=concept_id, depth=depth
                )

                if graph_data and graph_data.get("nodes"):
                    create_network_visualization(graph_data, f"Focused on {concept_id}")
                    show_concept_details_panel(concept_id)
                else:
                    st.warning(f"No data found for concept {concept_id}")


@run_async
async def show_domain_explorer(domain_filter: str):
    """Show domain exploration interface via API"""
    st.markdown("## 🔍 Domain Explorer")

    if domain_filter == "All Domains":
        st.info("Please select a specific domain from the sidebar to explore")
        return

    with st.spinner(f"Exploring {domain_filter} domain via API..."):
        # Search concepts in domain via API
        concepts = await api_client.search_concepts(
            query="", domain=domain_filter.lower(), limit=100
        )

        if concepts:
            # Show domain statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Concepts", len(concepts))

            with col2:
                concept_types = [c.get("type", "unknown") for c in concepts]
                most_common_type = max(set(concept_types), key=concept_types.count)
                st.metric("Most Common Type", most_common_type)

            with col3:
                levels = [c.get("level", 0) for c in concepts if c.get("level")]
                avg_level = sum(levels) / len(levels) if levels else 0
                st.metric("Average Level", f"{avg_level:.1f}")

            # Create domain-specific visualization
            domain_graph_data = {
                "nodes": concepts,
                "edges": [],  # Would need relationship data from API
            }
            create_network_visualization(domain_graph_data, f"{domain_filter} Domain")

            # Show concept list
            show_domain_concept_list(concepts)
        else:
            st.warning(f"No concepts found in {domain_filter} domain")


def show_relationship_analysis():
    """Show relationship analysis interface"""
    st.markdown("## 📈 Relationship Analysis")

    st.info(
        "🚧 Relationship analysis will be implemented when relationship endpoints are available in the API"
    )

    # Placeholder for relationship analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔗 Relationship Types")
        st.info("API endpoint needed: /api/relationships/types")

    with col2:
        st.markdown("### 📊 Connection Patterns")
        st.info("API endpoint needed: /api/relationships/patterns")


@run_async
async def perform_concept_search(query: str, domain_filter: str):
    """Perform concept search via API"""
    if not query.strip():
        st.warning("Please enter a search query")
        return

    with st.spinner("Searching concepts via API..."):
        domain = None if domain_filter == "All Domains" else domain_filter.lower()

        results = await api_client.search_concepts(query=query, domain=domain, limit=50)

        if results:
            st.success(f"Found {len(results)} matching concepts")

            # Display search results
            st.markdown("### 🔍 Search Results")
            for concept in results:
                show_concept_card(concept)
        else:
            st.warning("No matching concepts found")


def show_concept_card(concept: Dict[str, Any]):
    """Display a concept card"""
    with st.container():
        st.markdown(
            f"""
        <div class="concept-card">
            <div class="concept-header">{concept.get('id', 'N/A')}: {concept.get('name', 'Unknown')}</div>
            <p><strong>Domain:</strong> {concept.get('domain', 'N/A')} | 
               <strong>Type:</strong> {concept.get('type', 'N/A')} | 
               <strong>Level:</strong> {concept.get('level', 'N/A')}</p>
            <p>{concept.get('description', 'No description available')[:200]}...</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                f"🎯 Focus", key=f"focus_{concept.get('id')}", use_container_width=True
            ):
                st.session_state.focus_concept = concept.get("id")
                st.rerun()

        with col2:
            if st.button(
                f"🔗 Relations",
                key=f"rel_{concept.get('id')}",
                use_container_width=True,
            ):
                st.session_state.view_relations = concept.get("id")
                st.rerun()

        with col3:
            if st.button(
                f"📋 Details",
                key=f"details_{concept.get('id')}",
                use_container_width=True,
            ):
                st.session_state.show_details = concept.get("id")
                st.rerun()


def create_network_visualization(graph_data: Dict[str, Any], title: str):
    """Create interactive network visualization using Plotly"""
    try:
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            st.warning("No nodes to visualize")
            return

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in nodes:
            node_id = node.get("id", str(node.get("name", "")))
            G.add_node(node_id, **node)

        # Add edges if available
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                G.add_edge(source, target, **edge)

        # Position nodes
        if len(G.nodes()) > 1:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = {list(G.nodes())[0]: (0, 0)}

        # Create edges trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create nodes trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        domain_colors = {
            "art": "#FF6B6B",
            "science": "#4ECDC4",
            "mathematics": "#45B7D1",
            "philosophy": "#96CEB4",
            "language": "#FFEAA7",
            "technology": "#DDA0DD",
            "religion": "#98D8C8",
        }

        for node_id in G.nodes():
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)

                node_data = G.nodes[node_id]
                node_text.append(f"{node_id}<br>{node_data.get('name', '')}")

                # Color by domain
                domain = node_data.get("domain", "").lower()
                color = domain_colors.get(domain, "#888")
                node_color.append(color)

                # Size by level/importance
                level = node_data.get("level", 1)
                size = max(10, min(30, level * 5))
                node_size.append(size)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            hoverinfo="text",
            marker=dict(
                size=node_size, color=node_color, line=dict(width=2, color="white")
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Drag to pan, scroll to zoom",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            ),
        )

        st.plotly_chart(fig, use_container_width=True, height=600)

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")


def show_graph_statistics(graph_data: Dict[str, Any]):
    """Show graph statistics"""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    st.markdown("### 📊 Graph Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Nodes", len(nodes))

    with col2:
        st.metric("Total Edges", len(edges))

    with col3:
        domains = [n.get("domain") for n in nodes if n.get("domain")]
        unique_domains = len(set(domains))
        st.metric("Domains", unique_domains)

    with col4:
        if edges:
            density = (
                len(edges) / (len(nodes) * (len(nodes) - 1) / 2)
                if len(nodes) > 1
                else 0
            )
            st.metric("Density", f"{density:.3f}")
        else:
            st.metric("Density", "0.000")


@run_async
async def show_concept_details_panel(concept_id: str):
    """Show detailed concept information via API"""
    st.markdown("### 📋 Concept Details")

    # Search for the specific concept
    results = await api_client.search_concepts(query=concept_id, limit=1)

    if results:
        concept = results[0]

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"""
            **ID:** {concept.get('id', 'N/A')}  
            **Name:** {concept.get('name', 'Unknown')}  
            **Domain:** {concept.get('domain', 'N/A')}  
            **Type:** {concept.get('type', 'N/A')}  
            **Level:** {concept.get('level', 'N/A')}
            """
            )

        with col2:
            st.text_area(
                "Description",
                value=concept.get("description", "No description available"),
                height=150,
                disabled=True,
            )
    else:
        st.warning(f"Could not load details for concept {concept_id}")


def show_domain_concept_list(concepts: List[Dict[str, Any]]):
    """Show list of concepts in domain"""
    st.markdown("### 📋 Domain Concepts")

    # Create DataFrame for better display
    df_data = []
    for concept in concepts:
        df_data.append(
            {
                "ID": concept.get("id", "N/A"),
                "Name": concept.get("name", "Unknown"),
                "Type": concept.get("type", "N/A"),
                "Level": concept.get("level", "N/A"),
                "Description": (
                    concept.get("description", "")[:100] + "..."
                    if concept.get("description")
                    else "No description"
                ),
            }
        )

    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
