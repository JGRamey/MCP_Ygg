"""
Graph Editor Main Interface

Main orchestrator for the modular Graph Editor interface.
Coordinates between UI components, data sources, and visualization engine.
"""

import streamlit as st
from typing import Dict, Any

from .models import GraphMode, GraphSettings
from .neo4j_connector import DataSourceManager
from .graph_visualizer import GraphVisualizer
from .ui_components import GraphEditorUI, GraphReportGenerator


class GraphEditorOrchestrator:
    """
    Main orchestrator for Graph Editor interface
    
    Purpose: Coordinates between all graph editor components and manages the main workflow
    Use: Single entry point that delegates to specialized modules for clean separation
    """
    
    def __init__(self):
        """Initialize all component managers"""
        self.ui = GraphEditorUI()
        self.data_manager = DataSourceManager()
        self.visualizer = GraphVisualizer()
        self.report_generator = GraphReportGenerator()
    
    def run(self):
        """
        Main application entry point
        
        Purpose: Orchestrates the entire Graph Editor interface workflow
        Use: Called from the Streamlit page to render the complete interface
        """
        # Page configuration
        st.set_page_config(
            page_title="Graph Editor - MCP Yggdrasil",
            page_icon="📊",
            layout="wide"
        )
        
        # Apply custom CSS styling
        self._apply_custom_css()
        
        # Render page header
        self.ui.render_page_header()
        
        # Render sidebar and get settings
        graph_mode, layout_type, node_size, edge_width = self.ui.render_sidebar_controls()
        
        # Create settings object
        settings = GraphSettings(
            layout_type=layout_type,
            node_size=node_size,
            edge_width=edge_width
        )
        
        # Route to appropriate view based on mode
        if graph_mode == GraphMode.FULL_NETWORK.value:
            self._render_full_network_view(settings)
        elif graph_mode == GraphMode.FOCUSED_VIEW.value:
            self._render_focused_view(settings)
        elif graph_mode == GraphMode.DOMAIN_EXPLORER.value:
            self._render_domain_explorer_view(settings)
        elif graph_mode == GraphMode.RELATIONSHIP_ANALYSIS.value:
            self._render_relationship_analysis_view()
        elif graph_mode == GraphMode.CROSS_CULTURAL.value:
            self._render_cross_cultural_view(settings)
    
    def _render_full_network_view(self, settings: GraphSettings):
        """
        Full network visualization mode
        
        Purpose: Displays the complete knowledge graph with all concepts and relationships
        Use: Main graph view showing entire network structure
        """
        st.markdown("## 🌐 Full Knowledge Network")
        
        # Check connection status
        self.ui.render_connection_status()
        
        # Main content columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Graph controls
            actions = self.ui.render_graph_controls()
            
            # Handle refresh action
            if actions.get('refresh'):
                st.rerun()
            
            # Get filtered concepts and create visualization
            try:
                filters = st.session_state.get('graph_filters', {})
                concepts, data_source = self.data_manager.get_concepts_with_fallback(filters)
                
                if not concepts:
                    st.warning("No concepts match the current filters")
                    return
                
                # Show data source indicator
                self.ui.render_data_source_indicator(data_source.type)
                
                # Create and display graph
                relationship_filters = st.session_state.get('relationship_filters', {'show': True})
                fig = self.visualizer.create_network_graph(concepts, settings, relationship_filters)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="main_graph")
                else:
                    st.error("Could not generate graph visualization")
                    
            except Exception as e:
                st.error(f"Error creating graph: {str(e)}")
        
        with col2:
            # Graph metrics and selected node info
            self.ui.render_graph_metrics(concepts if 'concepts' in locals() else [])
            
            # Show selected node information
            if 'selected_concept_id' in st.session_state:
                concept_id = st.session_state.selected_concept_id
                concept = self.data_manager.neo4j_connector.get_concept_by_id(concept_id)
                if concept:
                    self.ui.render_concept_details(concept)
            else:
                st.info("Click on a node in the graph to see details")
    
    def _render_focused_view(self, settings: GraphSettings):
        """
        Focused concept view mode
        
        Purpose: Shows a specific concept and its immediate connections up to specified depth
        Use: Detailed view of individual concepts and their local network
        """
        st.markdown("## 🎯 Focused Concept View")
        
        # Concept selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            concept_id = self.ui.render_concept_search()
        
        with col2:
            focus_depth = st.selectbox("Connection Depth", [1, 2, 3], index=1)
        
        # Render focused graph if concept selected
        if concept_id:
            fig = self.visualizer.create_focused_graph(concept_id, focus_depth, settings)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="focused_graph")
                
                # Show concept details
                concept = self.data_manager.neo4j_connector.get_concept_by_id(concept_id)
                if concept:
                    with st.expander("📋 Concept Details", expanded=True):
                        self.ui.render_concept_details(concept)
            else:
                st.warning("No connections found for this concept")
    
    def _render_domain_explorer_view(self, settings: GraphSettings):
        """
        Domain-specific graph exploration mode
        
        Purpose: Explores concepts within specific knowledge domains
        Use: Domain-focused analysis and visualization
        """
        st.markdown("## 🔍 Domain Explorer")
        
        # Get available domains
        domains = self.data_manager.neo4j_connector.get_domains()
        
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
            domain_concepts, _ = self.data_manager.get_concepts_with_fallback(
                {'domains': [selected_domain]}, limit=200
            )
            
            if domain_concepts:
                # Create domain graph
                relationship_filters = st.session_state.get('relationship_filters', {'show': True})
                fig = self.visualizer.create_network_graph(domain_concepts, settings, relationship_filters)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="domain_graph")
                    
                    # Domain statistics
                    with st.expander("📊 Domain Statistics", expanded=True):
                        self.ui.render_domain_statistics(selected_domain, domain_concepts)
                else:
                    st.warning("Could not create domain visualization")
            else:
                st.warning(f"No concepts found in {selected_domain} domain")
    
    def _render_relationship_analysis_view(self):
        """
        Relationship analysis and statistics mode
        
        Purpose: Analyzes and displays relationship patterns and network statistics
        Use: Network analysis and relationship insights
        """
        st.markdown("## 📈 Relationship Analysis")
        
        try:
            # Get concepts for analysis
            concepts, _ = self.data_manager.get_concepts_with_fallback(limit=1000)
            
            if not concepts:
                st.warning("No concepts available for analysis")
                return
            
            # Perform relationship analysis
            analysis = self.ui.render_relationship_analysis(concepts)
            
            # Display analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔗 Relationship Types")
                
                if analysis.get('type_distribution'):
                    import pandas as pd
                    import plotly.express as px
                    
                    df_types = pd.DataFrame(
                        list(analysis['type_distribution'].items()), 
                        columns=['Relationship Type', 'Count']
                    )
                    
                    fig_types = px.pie(
                        df_types, values='Count', names='Relationship Type',
                        title="Distribution of Relationship Types"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
                    st.dataframe(df_types, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Most Connected Concepts")
                
                if analysis.get('top_connected'):
                    df_connected = pd.DataFrame(
                        analysis['top_connected'], 
                        columns=['Concept ID', 'Name', 'Connection Count']
                    )
                    st.dataframe(df_connected, use_container_width=True)
            
            # Network metrics
            st.markdown("### 📊 Network Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Total Concepts", analysis.get('total_concepts', 0))
            
            with metrics_col2:
                st.metric("Total Relationships", analysis.get('total_relationships', 0))
            
            with metrics_col3:
                st.metric("Avg Connections", f"{analysis.get('avg_connections', 0):.1f}")
            
            with metrics_col4:
                st.metric("Network Density", f"{analysis.get('density', 0):.3f}")
                
        except Exception as e:
            st.error(f"Error performing relationship analysis: {str(e)}")
    
    def _render_cross_cultural_view(self, settings: GraphSettings):
        """
        Cross-cultural connections visualization mode
        
        Purpose: Shows concepts and relationships that span multiple domains/cultures
        Use: Explore cross-domain knowledge connections and universal concepts
        """
        st.markdown("## 🌍 Cross-Cultural Connections")
        st.markdown("**Discover concepts and relationships that transcend domain boundaries**")
        
        try:
            # Get cross-cultural data from the data manager
            cross_concepts, cross_relationships, data_source = self.data_manager.get_cross_cultural_data()
            
            # Show data source indicator
            self.ui.render_data_source_indicator(data_source.type)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🔗 Multi-Domain Concepts", "🌉 Cross-Domain Relationships", "📊 Analysis"])
            
            with tab1:
                st.subheader("Concepts Appearing Across Multiple Domains")
                
                if cross_concepts:
                    # Display cross-cultural concepts
                    for concept in cross_concepts:
                        with st.expander(f"🌟 {concept['concept_name']} ({concept['domain_count']} domains)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Domains**: {', '.join(concept['domains'])}")
                                if concept.get('description1'):
                                    st.write(f"**Description**: {concept['description1']}")
                            
                            with col2:
                                # Create a simple domain distribution chart
                                domain_data = {domain: 1 for domain in concept['domains']}
                                import pandas as pd
                                df_domains = pd.DataFrame(list(domain_data.items()), columns=['Domain', 'Count'])
                                st.bar_chart(df_domains.set_index('Domain'))
                    
                    # Summary statistics
                    st.subheader("📈 Cross-Cultural Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Cross-Cultural Concepts", len(cross_concepts))
                    
                    with col2:
                        avg_domains = sum(c['domain_count'] for c in cross_concepts) / len(cross_concepts) if cross_concepts else 0
                        st.metric("Avg Domains per Concept", f"{avg_domains:.1f}")
                    
                    with col3:
                        max_domains = max((c['domain_count'] for c in cross_concepts), default=0)
                        st.metric("Max Domains", max_domains)
                
                else:
                    st.info("No cross-cultural concepts found. This may indicate:")
                    st.markdown("- Limited cross-domain data in the knowledge base")
                    st.markdown("- Concepts are not yet linked across domains")
                    st.markdown("- Neo4j database is not available (showing demo data)")
            
            with tab2:
                st.subheader("Relationships Crossing Domain Boundaries")
                
                if cross_relationships:
                    # Display cross-domain relationships
                    for rel in cross_relationships[:20]:  # Limit to first 20
                        with st.expander(f"🔗 {rel['source_concept']} → {rel['target_concept']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Source**: {rel['source_concept']} ({rel['source_domain']})")
                                if rel.get('source_description'):
                                    st.write(f"*{rel['source_description'][:100]}...*")
                            
                            with col2:
                                st.write(f"**Target**: {rel['target_concept']} ({rel['target_domain']})")
                                if rel.get('target_description'):
                                    st.write(f"*{rel['target_description'][:100]}...*")
                            
                            st.write(f"**Relationship**: {rel['relationship_type']}")
                    
                    # Relationship analysis
                    st.subheader("🔍 Relationship Analysis")
                    
                    # Count relationships by type
                    rel_types = {}
                    domain_pairs = {}
                    
                    for rel in cross_relationships:
                        rel_type = rel['relationship_type']
                        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                        
                        domain_pair = f"{rel['source_domain']} → {rel['target_domain']}"
                        domain_pairs[domain_pair] = domain_pairs.get(domain_pair, 0) + 1
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Relationship Types**")
                        rel_df = pd.DataFrame(list(rel_types.items()), columns=['Type', 'Count'])
                        st.dataframe(rel_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Most Common Domain Connections**")
                        domain_df = pd.DataFrame(list(domain_pairs.items()), columns=['Connection', 'Count'])
                        domain_df = domain_df.sort_values('Count', ascending=False).head(10)
                        st.dataframe(domain_df, use_container_width=True)
                
                else:
                    st.info("No cross-domain relationships found. Consider:")
                    st.markdown("- Adding relationships between concepts in different domains")
                    st.markdown("- Checking Neo4j database connectivity")
                    st.markdown("- Expanding the knowledge base with cross-references")
            
            with tab3:
                st.subheader("🎯 Cross-Cultural Analysis")
                
                # Universal concepts analysis
                if cross_concepts:
                    st.write("### 🌟 Most Universal Concepts")
                    
                    # Sort by domain count
                    universal_concepts = sorted(cross_concepts, key=lambda x: x['domain_count'], reverse=True)[:10]
                    
                    for i, concept in enumerate(universal_concepts, 1):
                        st.write(f"{i}. **{concept['concept_name']}** - {concept['domain_count']} domains: {', '.join(concept['domains'])}")
                
                # Domain interaction matrix
                if cross_relationships:
                    st.write("### 🔄 Domain Interaction Map") 
                    
                    # Create domain interaction matrix
                    domains = set()
                    for rel in cross_relationships:
                        domains.add(rel['source_domain'])
                        domains.add(rel['target_domain'])
                    
                    domains = sorted(list(domains))
                    interaction_matrix = {source: {target: 0 for target in domains} for source in domains}
                    
                    for rel in cross_relationships:
                        interaction_matrix[rel['source_domain']][rel['target_domain']] += 1
                    
                    # Convert to DataFrame for display
                    matrix_df = pd.DataFrame(interaction_matrix).fillna(0)
                    st.dataframe(matrix_df, use_container_width=True)
                
                # Insights and recommendations
                st.write("### 💡 Insights & Recommendations")
                
                if cross_concepts or cross_relationships:
                    insights = []
                    
                    if cross_concepts:
                        most_universal = max(cross_concepts, key=lambda x: x['domain_count'])
                        insights.append(f"🏆 Most universal concept: **{most_universal['concept_name']}** appears in {most_universal['domain_count']} domains")
                    
                    if cross_relationships:
                        insights.append(f"🔗 Found {len(cross_relationships)} cross-domain relationships")
                        
                        # Find most connected domain pair
                        domain_connections = {}
                        for rel in cross_relationships:
                            key = tuple(sorted([rel['source_domain'], rel['target_domain']]))
                            domain_connections[key] = domain_connections.get(key, 0) + 1
                        
                        if domain_connections:
                            most_connected = max(domain_connections.items(), key=lambda x: x[1])
                            insights.append(f"🤝 Strongest domain connection: **{most_connected[0][0]}** ↔ **{most_connected[0][1]}** ({most_connected[1]} connections)")
                    
                    for insight in insights:
                        st.success(insight)
                
                else:
                    st.warning("Limited cross-cultural data available. Consider:")
                    st.markdown("- 🔗 Adding more relationships between concepts in different domains")
                    st.markdown("- 📚 Expanding the knowledge base with interdisciplinary content")
                    st.markdown("- 🌐 Ensuring Neo4j database has cross-domain relationships")
        
        except Exception as e:
            st.error(f"Error rendering cross-cultural view: {str(e)}")
            st.markdown("**Possible solutions:**")
            st.markdown("- Check Neo4j database connection")
            st.markdown("- Verify cross-domain relationships exist in the database")
            st.markdown("- Ensure proper data loading from CSV files")
    
    def _apply_custom_css(self):
        """
        Custom CSS styling application
        
        Purpose: Applies custom styles for better visual presentation
        Use: Called once at startup to enhance UI appearance
        """
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
        
        .node-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.5rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)


def main():
    """
    Main entry point for Graph Editor page
    
    Purpose: Creates and runs the Graph Editor orchestrator
    Use: Called by Streamlit when the page is accessed
    """
    orchestrator = GraphEditorOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()