"""
Database Manager - Complete CRUD Operations
Comprehensive interface for managing concepts, relationships, and domains
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.database_operations import (
    get_all_concepts, get_concept_by_id, create_concept, update_concept, 
    delete_concept, search_concepts, get_domains, get_concept_relationships
)
from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main Database Manager interface"""
    
    st.set_page_config(
        page_title="Database Manager - MCP Yggdrasil",
        page_icon="🗄️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
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
    
    .action-button {
        margin: 0.2rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E8B57;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🗄️ Database Manager")
    st.markdown("**Complete CRUD operations for concepts, relationships, and domains**")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Database Operations")
        
        operation = st.selectbox(
            "Select Operation",
            ["📊 Overview", "🔍 Browse & Search", "➕ Create Concept", "✏️ Edit Concept", "🔗 Manage Relationships", "📈 Domain Management"]
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

def show_quick_stats_sidebar():
    """Show quick statistics in sidebar"""
    try:
        from utils.database_operations import get_quick_stats
        stats = get_quick_stats()
        
        st.markdown("### 📊 Quick Stats")
        st.metric("Concepts", stats.get('concepts', 'N/A'))
        st.metric("Relationships", stats.get('relationships', 'N/A'))
        st.metric("Domains", stats.get('domains', 'N/A'))
        st.metric("Vectors", stats.get('vectors', 'N/A'))
        
    except Exception as e:
        st.error(f"Could not load stats: {str(e)}")

def show_overview():
    """Show database overview"""
    st.markdown("## 📊 Database Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Domain distribution
        st.markdown("### 📈 Domain Distribution")
        domains = get_domains()
        
        if domains:
            df = pd.DataFrame(domains)
            
            # Create pie chart
            import plotly.express as px
            fig = px.pie(df, values='concept_count', names='domain', 
                        title="Concepts by Domain")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.markdown("### 📋 Domain Details")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No domain data available")
    
    with col2:
        # Recent activity (placeholder)
        st.markdown("### 🕒 Recent Activity")
        if 'operation_history' in st.session_state:
            recent_ops = st.session_state.operation_history[-5:]
            for op in reversed(recent_ops):
                st.markdown(f"**{op['operation_type']}**: {op['description']}")
                st.caption(op['timestamp'])
        else:
            st.info("No recent activity")

def show_browse_search():
    """Show browse and search interface"""
    st.markdown("## 🔍 Browse & Search Concepts")
    
    # Search controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search concepts", placeholder="Enter search term...")
    
    with col2:
        domains = get_domains()
        domain_options = ["All Domains"] + [d['domain'] for d in domains]
        selected_domain = st.selectbox("Domain Filter", domain_options)
    
    with col3:
        limit = st.number_input("Results Limit", min_value=10, max_value=500, value=50)
    
    # Search button
    if st.button("🔍 Search", type="primary") or search_term:
        search_domain = None if selected_domain == "All Domains" else selected_domain
        
        if search_term:
            concepts = search_concepts(search_term, domain=search_domain, limit=limit)
        else:
            concepts = get_all_concepts(limit=limit, domain=search_domain)
        
        st.markdown(f"### Found {len(concepts)} concepts")
        
        # Display results
        if concepts:
            for concept in concepts:
                show_concept_card(concept)
        else:
            st.info("No concepts found matching your criteria")

def show_concept_card(concept):
    """Display a concept card with actions"""
    with st.container():
        st.markdown(f"""
        <div class="concept-card">
            <div class="concept-header">{concept['id']}: {concept['name']}</div>
            <div class="concept-meta">Domain: {concept['domain']} | Type: {concept['type']} | Level: {concept['level']}</div>
            <p>{concept.get('description', 'No description available')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"✏️ Edit", key=f"edit_{concept['id']}", use_container_width=True):
                st.session_state.edit_concept_id = concept['id']
                st.rerun()
        
        with col2:
            if st.button(f"🔗 Relations", key=f"rel_{concept['id']}", use_container_width=True):
                st.session_state.view_relationships_id = concept['id']
                st.rerun()
        
        with col3:
            if st.button(f"📋 Details", key=f"view_{concept['id']}", use_container_width=True):
                show_concept_details(concept['id'])
        
        with col4:
            if st.button(f"🗑️ Delete", key=f"del_{concept['id']}", type="secondary", use_container_width=True):
                st.session_state.delete_concept_id = concept['id']
                st.rerun()

def show_concept_details(concept_id):
    """Show detailed concept information"""
    concept = get_concept_by_id(concept_id)
    if concept:
        st.json(concept)
        
        # Show relationships
        relationships = get_concept_relationships(concept_id)
        if relationships:
            st.markdown("### 🔗 Relationships")
            df = pd.DataFrame(relationships)
            st.dataframe(df, use_container_width=True)

def show_create_concept():
    """Show create concept form"""
    st.markdown("## ➕ Create New Concept")
    
    with st.form("create_concept_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            concept_id = st.text_input("Concept ID*", placeholder="e.g., ART0001")
            concept_name = st.text_input("Concept Name*", placeholder="e.g., Renaissance_Art")
            domain = st.selectbox("Domain*", ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"])
        
        with col2:
            concept_type = st.selectbox("Type*", ["root", "sub_root", "branch", "leaf"])
            level = st.number_input("Level*", min_value=1, max_value=10, value=1)
            location = st.text_input("Location", placeholder="Optional")
        
        description = st.text_area("Description", placeholder="Detailed description of the concept...")
        
        # Optional metadata
        with st.expander("📅 Additional Metadata"):
            col1, col2 = st.columns(2)
            with col1:
                earliest_date = st.number_input("Earliest Evidence Date", value=None, placeholder="Year (e.g., 1400)")
                latest_date = st.number_input("Latest Evidence Date", value=None, placeholder="Year (e.g., 1600)")
            with col2:
                certainty_level = st.selectbox("Certainty Level", ["High", "Medium", "Low", "Unknown"])
                cultural_context = st.text_input("Cultural Context", placeholder="e.g., European, Ancient Greek")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("➕ Create Concept", type="primary")
        
        if submitted:
            if not all([concept_id, concept_name, domain, concept_type]):
                st.error("Please fill in all required fields (marked with *)")
            else:
                # Prepare concept data
                concept_data = {
                    'id': concept_id,
                    'name': concept_name,
                    'domain': domain,
                    'type': concept_type,
                    'level': level,
                    'description': description,
                    'location': location,
                    'certainty_level': certainty_level,
                    'cultural_context': cultural_context
                }
                
                # Add optional dates if provided
                if earliest_date:
                    concept_data['earliest_evidence_date'] = int(earliest_date)
                if latest_date:
                    concept_data['latest_evidence_date'] = int(latest_date)
                
                # Create concept
                success, message = create_concept(concept_data)
                
                if success:
                    st.success(f"✅ {message}")
                    add_to_history("CREATE", f"Created concept {concept_id}: {concept_name}")
                    mark_unsaved_changes(False)
                else:
                    st.error(f"❌ {message}")

def show_edit_concept():
    """Show edit concept interface"""
    st.markdown("## ✏️ Edit Concept")
    
    # Concept selection
    if 'edit_concept_id' not in st.session_state:
        st.info("Select a concept to edit from the Browse & Search page, or enter a concept ID below:")
        
        concept_id = st.text_input("Concept ID", placeholder="e.g., ART0001")
        if st.button("Load Concept") and concept_id:
            st.session_state.edit_concept_id = concept_id
            st.rerun()
    else:
        concept_id = st.session_state.edit_concept_id
        concept = get_concept_by_id(concept_id)
        
        if not concept:
            st.error(f"Concept {concept_id} not found")
            if st.button("Clear Selection"):
                del st.session_state.edit_concept_id
                st.rerun()
            return
        
        st.info(f"Editing: **{concept_id}** - {concept.get('name', 'Unknown')}")
        
        if st.button("🔄 Select Different Concept"):
            del st.session_state.edit_concept_id
            st.rerun()
        
        # Edit form
        with st.form("edit_concept_form"):
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Concept Name", value=concept.get('name', ''))
                new_domain = st.selectbox("Domain", 
                    ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"],
                    index=["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"].index(concept.get('domain', 'Art'))
                )
            
            with col2:
                new_type = st.selectbox("Type", 
                    ["root", "sub_root", "branch", "leaf"],
                    index=["root", "sub_root", "branch", "leaf"].index(concept.get('type', 'leaf'))
                )
                new_level = st.number_input("Level", min_value=1, max_value=10, value=concept.get('level', 1))
            
            new_description = st.text_area("Description", value=concept.get('description', ''))
            new_location = st.text_input("Location", value=concept.get('location', ''))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("💾 Save Changes", type="primary")
            with col2:
                if st.form_submit_button("🗑️ Delete Concept", type="secondary"):
                    st.session_state.confirm_delete = concept_id
            
            if submitted:
                updates = {
                    'name': new_name,
                    'domain': new_domain,
                    'type': new_type,
                    'level': new_level,
                    'description': new_description,
                    'location': new_location
                }
                
                success, message = update_concept(concept_id, updates)
                
                if success:
                    st.success(f"✅ {message}")
                    add_to_history("UPDATE", f"Updated concept {concept_id}: {new_name}")
                    mark_unsaved_changes(False)
                else:
                    st.error(f"❌ {message}")

def show_manage_relationships():
    """Show relationship management interface"""
    st.markdown("## 🔗 Manage Relationships")
    
    # Show relationships for selected concept
    if 'view_relationships_id' in st.session_state:
        concept_id = st.session_state.view_relationships_id
        concept = get_concept_by_id(concept_id)
        
        if concept:
            st.info(f"Viewing relationships for: **{concept_id}** - {concept.get('name', 'Unknown')}")
            
            relationships = get_concept_relationships(concept_id)
            
            if relationships:
                df = pd.DataFrame(relationships)
                st.dataframe(df, use_container_width=True)
                
                # Visualize relationships
                st.markdown("### 🌐 Relationship Visualization")
                try:
                    import plotly.graph_objects as go
                    import networkx as nx
                    
                    # Create network graph
                    G = nx.Graph()
                    G.add_node(concept_id, name=concept.get('name', concept_id), type='center')
                    
                    for rel in relationships:
                        target_id = rel['target_id']
                        target_name = rel['target_name']
                        G.add_node(target_id, name=target_name, type='related')
                        G.add_edge(concept_id, target_id, relationship=rel['type'])
                    
                    # Position nodes
                    pos = nx.spring_layout(G)
                    
                    # Create plotly figure
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                          line=dict(width=2, color='#888'),
                                          hoverinfo='none',
                                          mode='lines')
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(G.nodes[node]['name'])
                        node_color.append('red' if G.nodes[node]['type'] == 'center' else 'blue')
                    
                    node_trace = go.Scatter(x=node_x, y=node_y,
                                          mode='markers+text',
                                          text=node_text,
                                          textposition="middle center",
                                          hoverinfo='text',
                                          marker=dict(size=20, color=node_color))
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title=f'Relationships for {concept_id}',
                                      titlefont_size=16,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      annotations=[ dict(
                                          text="Red = Selected concept, Blue = Related concepts",
                                          showarrow=False,
                                          xref="paper", yref="paper",
                                          x=0.005, y=-0.002,
                                          xanchor="left", yanchor="bottom",
                                          font=dict(size=12)
                                      )],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create visualization: {e}")
            else:
                st.info("No relationships found for this concept")
        
        if st.button("🔄 Select Different Concept"):
            del st.session_state.view_relationships_id
            st.rerun()
    else:
        st.info("Select a concept to view its relationships from the Browse & Search page, or enter a concept ID below:")
        
        concept_id = st.text_input("Concept ID", placeholder="e.g., ART0001")
        if st.button("Load Relationships") and concept_id:
            st.session_state.view_relationships_id = concept_id
            st.rerun()

def show_domain_management():
    """Show domain management interface"""
    st.markdown("## 📈 Domain Management")
    
    domains = get_domains()
    
    if domains:
        st.markdown("### 📊 Current Domains")
        
        df = pd.DataFrame(domains)
        
        # Enhanced domain display
        for _, domain in df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{domain['domain']}**")
                    st.caption(f"{domain['concept_count']} concepts")
                
                with col2:
                    if st.button(f"📋 View Concepts", key=f"view_domain_{domain['domain']}"):
                        st.session_state.browse_domain = domain['domain']
                        st.switch_page("pages/01_🗄️_Database_Manager.py")
                
                with col3:
                    if st.button(f"📊 Analytics", key=f"analytics_{domain['domain']}"):
                        st.info(f"Analytics for {domain['domain']} - Coming soon!")
        
        # Domain statistics
        st.markdown("### 📈 Domain Statistics")
        
        import plotly.express as px
        fig = px.bar(df, x='domain', y='concept_count', 
                    title="Concepts per Domain",
                    color='concept_count',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No domain data available")

# Handle delete confirmation
if 'confirm_delete' in st.session_state:
    concept_id = st.session_state.confirm_delete
    
    st.warning(f"⚠️ Are you sure you want to delete concept **{concept_id}**? This action cannot be undone.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Delete", type="primary"):
            success, message = delete_concept(concept_id)
            if success:
                st.success(f"✅ {message}")
                add_to_history("DELETE", f"Deleted concept {concept_id}")
            else:
                st.error(f"❌ {message}")
            del st.session_state.confirm_delete
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel"):
            del st.session_state.confirm_delete
            st.rerun()

if __name__ == "__main__":
    main()