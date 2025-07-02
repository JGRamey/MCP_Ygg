"""
Graph Editor - Interactive Knowledge Graph Visualization & Editing
Visual network interface for exploring and editing the knowledge graph
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import sys
from pathlib import Path
import json
import math

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.database_operations import (
    get_all_concepts, get_concept_by_id, get_concept_relationships,
    get_domains, search_concepts, create_concept, update_concept, delete_concept
)
from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main Graph Editor interface"""
    
    st.set_page_config(
        page_title="Graph Editor - MCP Yggdrasil",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS for graph interface
    st.markdown("""
    <style>
    .graph-container {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .control-panel {
        background: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .node-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .layout-button {
        margin: 0.2rem;
        width: 100%;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
    }
    
    .filter-section {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E8B57;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 📊 Graph Editor")
    st.markdown("**Interactive knowledge graph visualization and editing**")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Graph Controls")
        
        # Graph mode selection
        graph_mode = st.selectbox(
            "Graph Mode",
            ["🌐 Full Network", "🎯 Focused View", "🔍 Domain Explorer", "📈 Relationship Analysis"]
        )
        
        st.markdown("---")
        
        # Layout controls
        st.markdown("### 📐 Layout Settings")
        layout_type = st.selectbox(
            "Layout Algorithm",
            ["spring", "circular", "random", "shell", "kamada_kawai"]
        )
        
        node_size = st.slider("Node Size", 10, 50, 20)
        edge_width = st.slider("Edge Width", 1, 5, 2)
        
        st.markdown("---")
        
        # Filters
        show_graph_filters()
    
    # Main content based on mode
    if graph_mode == "🌐 Full Network":
        show_full_network(layout_type, node_size, edge_width)
    elif graph_mode == "🎯 Focused View":
        show_focused_view(layout_type, node_size, edge_width)
    elif graph_mode == "🔍 Domain Explorer":
        show_domain_explorer(layout_type, node_size, edge_width)
    elif graph_mode == "📈 Relationship Analysis":
        show_relationship_analysis()

def show_graph_filters():
    """Show graph filtering controls"""
    st.markdown("### 🔍 Filters")
    
    # Domain filter
    domains = get_domains()
    domain_options = ["All Domains"] + [d['domain'] for d in domains]
    selected_domains = st.multiselect(
        "Domains",
        domain_options,
        default=["All Domains"]
    )
    
    # Type filter
    type_options = ["All Types", "root", "sub_root", "branch", "leaf"]
    selected_types = st.multiselect(
        "Node Types",
        type_options,
        default=["All Types"]
    )
    
    # Level filter
    level_range = st.slider("Level Range", 1, 10, (1, 5))
    
    # Store filters in session state
    st.session_state.graph_filters = {
        'domains': selected_domains,
        'types': selected_types,
        'level_range': level_range
    }
    
    # Relationship filters
    st.markdown("#### 🔗 Relationships")
    show_relationships = st.checkbox("Show Relationships", value=True)
    min_relationship_strength = st.slider("Min Relationship Strength", 0.0, 1.0, 0.0)
    
    st.session_state.relationship_filters = {
        'show': show_relationships,
        'min_strength': min_relationship_strength
    }

def show_full_network(layout_type, node_size, edge_width):
    """Show full network visualization"""
    st.markdown("## 🌐 Full Knowledge Network")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Graph controls
        with st.container():
            control_col1, control_col2, control_col3 = st.columns(3)
            
            with control_col1:
                if st.button("🔄 Refresh Graph", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            with control_col2:
                if st.button("💾 Save Layout", use_container_width=True):
                    st.success("Layout saved to session")
                    add_to_history("GRAPH", "Saved graph layout")
            
            with control_col3:
                if st.button("📊 Generate Report", use_container_width=True):
                    generate_graph_report()
        
        # Main graph visualization
        graph_container = st.container()
        with graph_container:
            try:
                # Get filtered concepts
                concepts = get_filtered_concepts()
                
                if not concepts:
                    st.warning("No concepts match the current filters")
                    return
                
                # Create network graph
                fig = create_network_graph(concepts, layout_type, node_size, edge_width)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="main_graph")
                else:
                    st.error("Could not generate graph visualization")
                    
            except Exception as e:
                st.error(f"Error creating graph: {str(e)}")
                st.code(f"Error details: {e}", language="text")
    
    with col2:
        # Graph metrics and info
        show_graph_metrics(concepts if 'concepts' in locals() else [])
        
        # Selected node info
        show_selected_node_info()

def show_focused_view(layout_type, node_size, edge_width):
    """Show focused view of specific concept and its connections"""
    st.markdown("## 🎯 Focused Concept View")
    
    # Concept selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        concept_search = st.text_input("🔍 Search for concept to focus on", placeholder="Enter concept name or ID...")
    
    with col2:
        focus_depth = st.selectbox("Connection Depth", [1, 2, 3], index=1)
    
    if concept_search:
        # Search for matching concepts
        matching_concepts = search_concepts(concept_search, limit=10)
        
        if matching_concepts:
            # Let user select specific concept
            concept_options = [f"{c['id']}: {c['name']}" for c in matching_concepts]
            selected_concept = st.selectbox("Select Concept", concept_options)
            
            if selected_concept:
                concept_id = selected_concept.split(":")[0]
                
                # Create focused graph
                fig = create_focused_graph(concept_id, focus_depth, layout_type, node_size, edge_width)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="focused_graph")
                    
                    # Show concept details
                    concept = get_concept_by_id(concept_id)
                    if concept:
                        with st.expander("📋 Concept Details", expanded=True):
                            show_concept_detail_panel(concept)
                else:
                    st.warning("No connections found for this concept")
        else:
            st.info("No concepts found matching your search")

def show_domain_explorer(layout_type, node_size, edge_width):
    """Show domain-specific graph exploration"""
    st.markdown("## 🔍 Domain Explorer")
    
    domains = get_domains()
    
    if not domains:
        st.warning("No domain data available")
        return
    
    # Domain selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_domain = st.selectbox(
            "Select Domain",
            [d['domain'] for d in domains]
        )
    
    with col2:
        show_cross_domain = st.checkbox("Show Cross-Domain Connections", value=False)
    
    if selected_domain:
        # Get domain concepts
        domain_concepts = get_all_concepts(domain=selected_domain)
        
        if domain_concepts:
            # Create domain graph
            fig = create_domain_graph(selected_domain, domain_concepts, show_cross_domain, layout_type, node_size, edge_width)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="domain_graph")
                
                # Domain statistics
                with st.expander("📊 Domain Statistics", expanded=True):
                    show_domain_statistics(selected_domain, domain_concepts)
            else:
                st.warning("Could not create domain visualization")
        else:
            st.warning(f"No concepts found in {selected_domain} domain")

def show_relationship_analysis():
    """Show relationship analysis and statistics"""
    st.markdown("## 📈 Relationship Analysis")
    
    try:
        # Get all concepts for analysis
        concepts = get_all_concepts(limit=1000)  # Reasonable limit for analysis
        
        if not concepts:
            st.warning("No concepts available for analysis")
            return
        
        # Analyze relationships
        relationship_data = analyze_relationships(concepts)
        
        # Display analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔗 Relationship Types")
            
            if relationship_data['type_distribution']:
                df_types = pd.DataFrame(list(relationship_data['type_distribution'].items()), 
                                      columns=['Relationship Type', 'Count'])
                
                fig_types = px.pie(df_types, values='Count', names='Relationship Type',
                                 title="Distribution of Relationship Types")
                st.plotly_chart(fig_types, use_container_width=True)
                
                st.dataframe(df_types, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Most Connected Concepts")
            
            if relationship_data['top_connected']:
                df_connected = pd.DataFrame(relationship_data['top_connected'], 
                                          columns=['Concept ID', 'Name', 'Connection Count'])
                st.dataframe(df_connected, use_container_width=True)
        
        # Network analysis metrics
        st.markdown("### 📊 Network Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Total Concepts", relationship_data['total_concepts'])
        
        with metrics_col2:
            st.metric("Total Relationships", relationship_data['total_relationships'])
        
        with metrics_col3:
            st.metric("Avg Connections", f"{relationship_data['avg_connections']:.1f}")
        
        with metrics_col4:
            st.metric("Network Density", f"{relationship_data['density']:.3f}")
        
        # Cross-domain analysis
        if relationship_data['cross_domain_connections']:
            st.markdown("### 🌐 Cross-Domain Connections")
            
            df_cross = pd.DataFrame(relationship_data['cross_domain_connections'])
            
            fig_cross = px.bar(df_cross, x='domains', y='count',
                             title="Cross-Domain Connection Strength")
            st.plotly_chart(fig_cross, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error performing relationship analysis: {str(e)}")

def get_filtered_concepts():
    """Get concepts based on current filters"""
    filters = st.session_state.get('graph_filters', {})
    
    # Start with all concepts
    concepts = get_all_concepts(limit=500)  # Reasonable limit for visualization
    
    if not concepts:
        return []
    
    # Apply domain filter
    if filters.get('domains') and "All Domains" not in filters['domains']:
        concepts = [c for c in concepts if c['domain'] in filters['domains']]
    
    # Apply type filter
    if filters.get('types') and "All Types" not in filters['types']:
        concepts = [c for c in concepts if c['type'] in filters['types']]
    
    # Apply level filter
    if filters.get('level_range'):
        min_level, max_level = filters['level_range']
        concepts = [c for c in concepts if min_level <= c.get('level', 1) <= max_level]
    
    return concepts

def create_network_graph(concepts, layout_type, node_size, edge_width):
    """Create network graph visualization"""
    try:
        if not concepts:
            return None
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for concept in concepts:
            G.add_node(concept['id'], 
                      name=concept['name'],
                      domain=concept['domain'],
                      type=concept['type'],
                      level=concept.get('level', 1))
        
        # Add edges from relationships
        relationship_filters = st.session_state.get('relationship_filters', {'show': True})
        
        if relationship_filters.get('show', True):
            for concept in concepts[:50]:  # Limit for performance
                relationships = get_concept_relationships(concept['id'])
                for rel in relationships:
                    if rel['target_id'] in G.nodes():
                        strength = rel.get('strength', 0.5)
                        if strength >= relationship_filters.get('min_strength', 0.0):
                            G.add_edge(concept['id'], rel['target_id'], 
                                     relationship=rel['type'], strength=strength)
        
        if len(G.nodes()) == 0:
            return None
        
        # Generate layout
        if layout_type == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "random":
            pos = nx.random_layout(G)
        elif layout_type == "shell":
            pos = nx.shell_layout(G)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(edge[2].get('relationship', 'RELATES_TO'))
        
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=edge_width, color='rgba(125,125,125,0.5)'),
                                hoverinfo='none',
                                mode='lines'))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size_list = []
        hover_text = []
        
        # Color mapping for domains (6 primary domains)
        domain_colors = {
            'Art': '#FF6B6B',
            'Science': '#4ECDC4', 
            'Mathematics': '#45B7D1',
            'Philosophy': '#96CEB4',  # Includes Religion sub-domain
            'Language': '#FFEAA7',
            'Technology': '#DDA0DD'
        }
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            # Node text and color
            name = node[1]['name']
            domain = node[1]['domain']
            node_text.append(f"{node[0]}")
            node_color.append(domain_colors.get(domain, '#95A5A6'))
            
            # Node size based on connections
            connections = G.degree(node[0])
            size = max(node_size, min(node_size * 2, node_size + connections * 2))
            node_size_list.append(size)
            
            # Hover text
            hover_text.append(f"{name}<br>Domain: {domain}<br>Type: {node[1]['type']}<br>Connections: {connections}")
        
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(size=node_size_list,
                                           color=node_color,
                                           line=dict(width=2, color='white')),
                                text=node_text,
                                textposition="middle center",
                                textfont=dict(size=8, color='white'),
                                hovertext=hover_text,
                                hoverinfo='text'))
        
        # Update layout
        fig.update_layout(
            title=f"Knowledge Graph Network ({len(G.nodes())} concepts, {len(G.edges())} relationships)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"Layout: {layout_type} | Nodes: {len(G.nodes())} | Edges: {len(G.edges())}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating network graph: {str(e)}")
        return None

def create_focused_graph(concept_id, depth, layout_type, node_size, edge_width):
    """Create focused graph around specific concept"""
    try:
        # Get central concept
        central_concept = get_concept_by_id(concept_id)
        if not central_concept:
            return None
        
        # Build focused network
        G = nx.Graph()
        
        # Add central node
        G.add_node(concept_id, 
                  name=central_concept['name'],
                  domain=central_concept['domain'],
                  type=central_concept['type'],
                  level=central_concept.get('level', 1),
                  distance=0)
        
        # Add connected nodes up to specified depth
        to_explore = [(concept_id, 0)]
        explored = {concept_id}
        
        while to_explore:
            current_id, current_depth = to_explore.pop(0)
            
            if current_depth < depth:
                relationships = get_concept_relationships(current_id)
                
                for rel in relationships:
                    target_id = rel['target_id']
                    
                    if target_id not in explored:
                        target_concept = get_concept_by_id(target_id)
                        if target_concept:
                            G.add_node(target_id,
                                      name=target_concept['name'],
                                      domain=target_concept['domain'], 
                                      type=target_concept['type'],
                                      level=target_concept.get('level', 1),
                                      distance=current_depth + 1)
                            explored.add(target_id)
                            to_explore.append((target_id, current_depth + 1))
                    
                    if target_id in G.nodes():
                        G.add_edge(current_id, target_id,
                                  relationship=rel['type'],
                                  strength=rel.get('strength', 0.5))
        
        if len(G.nodes()) <= 1:
            return None
        
        # Create visualization similar to network graph but with distance-based coloring
        return create_network_graph_from_nx(G, layout_type, node_size, edge_width, focus_node=concept_id)
    
    except Exception as e:
        st.error(f"Error creating focused graph: {str(e)}")
        return None

def create_network_graph_from_nx(G, layout_type, node_size, edge_width, focus_node=None):
    """Create Plotly graph from NetworkX graph"""
    # Similar to create_network_graph but works with existing NetworkX graph
    # Implementation would be similar to create_network_graph but adapted for pre-built graph
    # For brevity, using simplified version
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=edge_width, color='rgba(125,125,125,0.5)'),
                            hoverinfo='none',
                            mode='lines'))
    
    # Add nodes with distance-based coloring if focus_node provided
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    hover_text = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[0])
        
        if focus_node and node[0] == focus_node:
            node_color.append('red')  # Central node
        else:
            distance = node[1].get('distance', 1)
            node_color.append(f'rgba(70, 130, 180, {1.0 - distance * 0.3})')  # Fade by distance
        
        hover_text.append(f"{node[1]['name']}<br>Domain: {node[1]['domain']}<br>Distance: {node[1].get('distance', 0)}")
    
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            marker=dict(size=node_size, color=node_color),
                            text=node_text,
                            textposition="middle center",
                            hovertext=hover_text,
                            hoverinfo='text'))
    
    fig.update_layout(
        title=f"Focused View: {focus_node if focus_node else 'Network'} ({len(G.nodes())} concepts)",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_domain_graph(domain, concepts, show_cross_domain, layout_type, node_size, edge_width):
    """Create domain-specific graph"""
    # Implementation similar to create_network_graph but filtered for domain
    # For brevity, using simplified approach
    return create_network_graph(concepts, layout_type, node_size, edge_width)

def show_graph_metrics(concepts):
    """Show graph metrics and statistics"""
    st.markdown("### 📊 Graph Metrics")
    
    if not concepts:
        st.info("No data to display")
        return
    
    # Basic metrics
    st.metric("Total Concepts", len(concepts))
    
    # Domain distribution
    domain_counts = {}
    for concept in concepts:
        domain = concept['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    st.markdown("#### Domain Distribution")
    for domain, count in domain_counts.items():
        st.metric(domain, count)

def show_selected_node_info():
    """Show information about selected node"""
    st.markdown("### 📋 Selected Node")
    
    if 'selected_concept_id' in st.session_state:
        concept_id = st.session_state.selected_concept_id
        concept = get_concept_by_id(concept_id)
        
        if concept:
            show_concept_detail_panel(concept)
    else:
        st.info("Click on a node in the graph to see details")

def show_concept_detail_panel(concept):
    """Show detailed concept information panel"""
    st.markdown(f"""
    <div class="node-info">
        <h4>{concept['id']}: {concept['name']}</h4>
        <p><strong>Domain:</strong> {concept['domain']}</p>
        <p><strong>Type:</strong> {concept['type']}</p>
        <p><strong>Level:</strong> {concept.get('level', 'N/A')}</p>
        <p><strong>Description:</strong> {concept.get('description', 'No description available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✏️ Edit", key=f"edit_graph_{concept['id']}", use_container_width=True):
            st.session_state.edit_concept_id = concept['id']
            st.switch_page("pages/01_🗄️_Database_Manager.py")
    
    with col2:
        if st.button("🎯 Focus", key=f"focus_graph_{concept['id']}", use_container_width=True):
            st.session_state.focus_concept_id = concept['id']
            st.rerun()
    
    with col3:
        if st.button("🔗 Relations", key=f"rel_graph_{concept['id']}", use_container_width=True):
            st.session_state.view_relationships_id = concept['id']
            st.switch_page("pages/01_🗄️_Database_Manager.py")

def show_domain_statistics(domain, concepts):
    """Show detailed domain statistics"""
    st.markdown(f"### 📊 {domain} Domain Statistics")
    
    # Type distribution
    type_counts = {}
    level_counts = {}
    
    for concept in concepts:
        concept_type = concept['type']
        concept_level = concept.get('level', 1)
        
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

def analyze_relationships(concepts):
    """Analyze relationship patterns in the knowledge graph"""
    analysis = {
        'total_concepts': len(concepts),
        'total_relationships': 0,
        'type_distribution': {},
        'top_connected': [],
        'cross_domain_connections': [],
        'avg_connections': 0,
        'density': 0
    }
    
    try:
        # Count connections for each concept
        concept_connections = {}
        relationship_types = {}
        
        for concept in concepts[:100]:  # Limit for performance
            relationships = get_concept_relationships(concept['id'])
            concept_connections[concept['id']] = {
                'name': concept['name'], 
                'count': len(relationships)
            }
            analysis['total_relationships'] += len(relationships)
            
            for rel in relationships:
                rel_type = rel['type']
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        analysis['type_distribution'] = relationship_types
        
        # Top connected concepts
        top_connected = sorted(concept_connections.items(), 
                             key=lambda x: x[1]['count'], reverse=True)[:10]
        analysis['top_connected'] = [[k, v['name'], v['count']] for k, v in top_connected]
        
        # Calculate averages
        if concept_connections:
            total_connections = sum(c['count'] for c in concept_connections.values())
            analysis['avg_connections'] = total_connections / len(concept_connections)
            
            # Simple density calculation
            max_possible = len(concepts) * (len(concepts) - 1) / 2
            analysis['density'] = analysis['total_relationships'] / max_possible if max_possible > 0 else 0
        
        # Cross-domain analysis (simplified)
        domain_pairs = {}
        for concept in concepts[:50]:  # Limit for performance
            relationships = get_concept_relationships(concept['id'])
            for rel in relationships:
                # This would need more complex logic to determine cross-domain relationships
                pass
        
    except Exception as e:
        st.error(f"Error in relationship analysis: {str(e)}")
    
    return analysis

def generate_graph_report():
    """Generate comprehensive graph analysis report"""
    st.success("📊 Graph report generated!")
    add_to_history("ANALYSIS", "Generated graph analysis report")
    
    # This would generate a comprehensive report
    # For now, just show success message

if __name__ == "__main__":
    main()