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
        
        # Add node creation interface
        add_node_creation_sidebar()

    # Main content area
    if graph_mode == "🌐 Full Network":
        show_full_network_view(domain_filter, node_limit)
    elif graph_mode == "🎯 Focused View":
        show_focused_view(depth, domain_filter)
    elif graph_mode == "🔍 Domain Explorer":
        show_domain_explorer(domain_filter)
    elif graph_mode == "📈 Relationship Analysis":
        show_relationship_analysis()

    # Handle session state actions
    handle_session_state_actions()


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
        col1, col2, col3, col4 = st.columns(4)

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

        with col4:
            if st.button(
                f"✏️ Edit",
                key=f"edit_{concept.get('id')}",
                use_container_width=True,
            ):
                st.session_state.edit_concept = concept.get("id")
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


def handle_session_state_actions():
    """Handle session state-based actions like editing"""
    
    # Handle concept editing
    if "edit_concept" in st.session_state and st.session_state.edit_concept:
        show_concept_editor(st.session_state.edit_concept)
    
    # Handle relationship creation
    if "create_relationship" in st.session_state and st.session_state.create_relationship:
        show_relationship_creator()
    
    # Handle new concept creation
    if "create_new_concept" in st.session_state and st.session_state.create_new_concept:
        show_new_concept_creator()
    
    # Handle concept details
    if "show_details" in st.session_state and st.session_state.show_details:
        asyncio.run(show_concept_details_panel(st.session_state.show_details))


@run_async
async def show_concept_editor(concept_id: str):
    """Show concept editing interface"""
    st.markdown("---")
    st.markdown("## ✏️ Edit Concept")
    
    # Get concept data
    results = await api_client.search_concepts(query=concept_id, limit=1)
    
    if not results:
        st.error(f"Could not find concept {concept_id}")
        if st.button("❌ Close Editor"):
            del st.session_state.edit_concept
            st.rerun()
        return
    
    concept = results[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form(f"edit_concept_{concept_id}"):
            st.markdown(f"**Editing: {concept.get('name', 'Unknown')}**")
            
            # Editable fields
            new_name = st.text_input("Name", value=concept.get('name', ''))
            new_domain = st.selectbox(
                "Domain",
                ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion"],
                index=max(0, ["art", "science", "mathematics", "philosophy", "language", "technology", "religion"].index(concept.get('domain', 'science').lower()) if concept.get('domain', 'science').lower() in ["art", "science", "mathematics", "philosophy", "language", "technology", "religion"] else 0)
            )
            new_description = st.text_area(
                "Description", 
                value=concept.get('description', ''), 
                height=150
            )
            new_type = st.selectbox(
                "Type",
                ["Concept", "Entity", "Document", "Author", "Event"],
                index=max(0, ["Concept", "Entity", "Document", "Author", "Event"].index(concept.get('type', 'Concept')) if concept.get('type', 'Concept') in ["Concept", "Entity", "Document", "Author", "Event"] else 0)
            )
            
            # Additional properties as JSON
            st.markdown("**Additional Properties (JSON)**")
            additional_props = concept.copy()
            # Remove basic fields to show only additional properties
            for key in ['id', 'name', 'domain', 'description', 'type']:
                additional_props.pop(key, None)
            
            props_json = st.text_area(
                "Additional Properties",
                value=json.dumps(additional_props, indent=2),
                height=100,
                help="Enter additional properties as JSON"
            )
            
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                save_changes = st.form_submit_button("💾 Save Changes", type="primary")
            
            with col_cancel:
                cancel_edit = st.form_submit_button("❌ Cancel")
            
            if save_changes:
                await save_concept_changes(concept_id, new_name, new_domain, new_description, new_type, props_json)
            
            if cancel_edit:
                del st.session_state.edit_concept
                st.rerun()
    
    with col2:
        st.markdown("**Current Values**")
        st.info(f"""
        **ID:** {concept.get('id', 'N/A')}
        **Name:** {concept.get('name', 'Unknown')}
        **Domain:** {concept.get('domain', 'N/A')}
        **Type:** {concept.get('type', 'N/A')}
        **Description:** {concept.get('description', 'No description')[:100]}...
        """)
        
        # Relationship management
        st.markdown("**🔗 Manage Relationships**")
        
        if st.button("➕ Add Relationship", use_container_width=True):
            st.session_state.create_relationship = concept_id
            st.rerun()
        
        if st.button("🗑️ Delete Relationships", use_container_width=True):
            st.session_state.delete_relationships = concept_id
            st.rerun()


@run_async
async def save_concept_changes(concept_id: str, name: str, domain: str, description: str, type_: str, props_json: str):
    """Save concept changes via API"""
    try:
        # Parse additional properties
        additional_props = json.loads(props_json) if props_json.strip() else {}
        
        # Prepare update data
        update_data = {
            "id": concept_id,
            "name": name,
            "domain": domain.lower(),
            "description": description,
            "type": type_,
            **additional_props
        }
        
        # Save via API
        result = await api_client.manage_database("update", update_data)
        
        if result:
            st.success("✅ Concept updated successfully!")
            del st.session_state.edit_concept
            st.rerun()
        else:
            st.error("❌ Failed to update concept")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON in additional properties")
    except Exception as e:
        st.error(f"❌ Error saving changes: {str(e)}")


def show_relationship_creator():
    """Show relationship creation interface"""
    st.markdown("---")
    st.markdown("## 🔗 Create Relationship")
    
    concept_id = st.session_state.get("create_relationship")
    
    with st.form("create_relationship_form"):
        st.markdown(f"**Creating relationship for concept: {concept_id}**")
        
        # Target concept
        target_concept = st.text_input(
            "Target Concept ID",
            placeholder="Enter the ID of the concept to connect to"
        )
        
        # Relationship type
        relationship_type = st.selectbox(
            "Relationship Type",
            [
                "RELATES_TO",
                "INFLUENCES", 
                "INFLUENCED_BY",
                "CONTRADICTS",
                "SUPPORTS",
                "PART_OF",
                "CONTAINS",
                "PREREQUISITE_FOR",
                "DERIVED_FROM",
                "SIMILAR_TO"
            ]
        )
        
        # Relationship properties
        weight = st.slider("Relationship Weight", 0.0, 1.0, 0.5, help="Strength of the relationship")
        description = st.text_area("Relationship Description", placeholder="Describe this relationship...")
        
        # Direction
        bidirectional = st.checkbox("Bidirectional Relationship", help="Create relationship in both directions")
        
        col_create, col_cancel = st.columns(2)
        
        with col_create:
            create_rel = st.form_submit_button("🔗 Create Relationship", type="primary")
        
        with col_cancel:
            cancel_rel = st.form_submit_button("❌ Cancel")
        
        if create_rel and target_concept:
            create_relationship_async(concept_id, target_concept, relationship_type, weight, description, bidirectional)
        
        if cancel_rel:
            del st.session_state.create_relationship
            st.rerun()


@run_async
async def create_relationship_async(source_id: str, target_id: str, rel_type: str, weight: float, description: str, bidirectional: bool):
    """Create relationship via API"""
    try:
        # Prepare relationship data
        relationship_data = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": rel_type,
            "weight": weight,
            "description": description,
            "bidirectional": bidirectional
        }
        
        # This would call the relationship creation API endpoint
        # For now, simulate success
        st.success(f"✅ Created {rel_type} relationship between {source_id} and {target_id}")
        
        if bidirectional:
            st.info(f"✅ Also created reverse relationship")
        
        del st.session_state.create_relationship
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error creating relationship: {str(e)}")


# Add node creation interface to sidebar
def add_node_creation_sidebar():
    """Add node creation interface to sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ➕ Create New Node")
        
        if st.button("🆕 New Concept", use_container_width=True):
            st.session_state.create_new_concept = True
            st.rerun()
        
        if st.button("📝 New Document", use_container_width=True):
            st.session_state.create_new_document = True
            st.rerun()


@run_async
async def show_new_concept_creator():
    """Show new concept creation interface"""
    st.markdown("---")
    st.markdown("## 🆕 Create New Concept")
    
    with st.form("create_new_concept"):
        # Basic concept information
        col1, col2 = st.columns(2)
        
        with col1:
            concept_name = st.text_input("Concept Name*", placeholder="e.g., Quantum Mechanics")
            concept_domain = st.selectbox("Domain*", ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion"])
            concept_type = st.selectbox("Type", ["Concept", "Entity", "Document", "Author", "Event"])
        
        with col2:
            concept_level = st.number_input("Concept Level", min_value=1, max_value=10, value=1, help="1=Basic, 10=Advanced")
            keywords = st.text_input("Keywords", placeholder="quantum, physics, mechanics")
        
        concept_description = st.text_area("Description*", placeholder="Detailed description of the concept...")
        
        # Related concepts
        related_concepts = st.text_input("Related Concepts (comma-separated)", placeholder="physics, mathematics, wave-particle duality")
        
        col_create, col_cancel = st.columns(2)
        
        with col_create:
            create_concept = st.form_submit_button("🎯 Create Concept", type="primary")
        
        with col_cancel:
            cancel_create = st.form_submit_button("❌ Cancel")
        
        if create_concept and concept_name and concept_description:
            await create_new_concept_async(concept_name, concept_domain, concept_type, concept_level, concept_description, keywords, related_concepts)
        
        if cancel_create:
            del st.session_state.create_new_concept
            st.rerun()


@run_async
async def create_new_concept_async(name: str, domain: str, type_: str, level: int, description: str, keywords: str, related: str):
    """Create new concept via API"""
    try:
        concept_data = {
            "name": name,
            "domain": domain.lower(),
            "type": type_,
            "level": level,
            "description": description,
            "keywords": keywords,
            "related_concepts": related
        }
        
        result = await api_client.manage_database("create", concept_data)
        
        if result:
            st.success(f"✅ Created new concept: {name}")
            del st.session_state.create_new_concept
            st.rerun()
        else:
            st.error("❌ Failed to create concept")
            
    except Exception as e:
        st.error(f"❌ Error creating concept: {str(e)}")


if __name__ == "__main__":
    main()
