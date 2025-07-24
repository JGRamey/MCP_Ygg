"""
Neo4j Database Manager for MCP Yggdrasil
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .models import Neo4jNodeInfo, DatabaseFilters

try:
    from py2neo import Graph, Node, Relationship
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jManager:
    """Manager for Neo4j database operations."""
    
    def __init__(self):
        """Initialize Neo4j manager."""
        self.graph = None
        if NEO4J_AVAILABLE:
            try:
                self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "yggdrasil"))
            except Exception as e:
                st.warning(f"Neo4j connection failed: {e}")
    
    def render_interface(self, filters: DatabaseFilters):
        """Render Neo4j database management interface."""
        st.markdown("## 🗄️ Neo4j Database")
        st.markdown("**Knowledge graph nodes and relationships**")
        
        if not NEO4J_AVAILABLE:
            st.error("Neo4j client not available. Install py2neo: `pip install py2neo`")
            return
        
        if not self.graph:
            st.error("❌ Could not connect to Neo4j database")
            st.markdown("**Check that:**")
            st.markdown("- Neo4j is running on localhost:7687")
            st.markdown("- Username: neo4j, Password: yggdrasil")
            return
        
        # Database status
        try:
            self.graph.run("MATCH (n) RETURN count(n) as count LIMIT 1").data()
            st.success("✅ Connected to Neo4j database")
        except Exception as e:
            st.error(f"❌ Database query failed: {e}")
            return
        
        # Management tabs
        tabs = st.tabs(["📄 Concepts", "🔗 Relationships", "📊 Statistics", "🔍 Query"])
        
        with tabs[0]:
            self._show_concepts(filters)
        
        with tabs[1]:
            self._show_relationships(filters)
        
        with tabs[2]:
            self._show_statistics()
        
        with tabs[3]:
            self._show_query_interface()
    
    def _show_concepts(self, filters: DatabaseFilters):
        """Show Neo4j concepts with filtering."""
        st.subheader("📄 Concept Nodes")
        
        # Build query with filters
        where_clauses = []
        if filters.domains:
            domain_filter = " OR ".join([f"c.domain = '{d}'" for d in filters.domains])
            where_clauses.append(f"({domain_filter})")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        MATCH (c:Concept)
        WHERE {where_clause}
        RETURN c.id as id, c.name as name, c.domain as domain, 
               c.description as description, labels(c) as labels
        ORDER BY c.name
        LIMIT 100
        """
        
        try:
            result = self.graph.run(query).data()
            
            if result:
                # Display as table
                concepts_df = pd.DataFrame(result)
                
                # Add selection
                selected_indices = st.dataframe(
                    concepts_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )
                
                # Show selected concept details
                if selected_indices and len(selected_indices.selection['rows']) > 0:
                    selected_idx = selected_indices.selection['rows'][0]
                    selected_concept = result[selected_idx]
                    self._show_concept_details(selected_concept)
            else:
                st.info("No concepts found matching filters")
                
        except Exception as e:
            st.error(f"Error querying concepts: {e}")
    
    def _show_relationships(self, filters: DatabaseFilters):
        """Show Neo4j relationships."""
        st.subheader("🔗 Relationships")
        
        # Build query
        query = """
        MATCH (a)-[r]->(b)
        RETURN a.name as source, type(r) as relationship_type, 
               b.name as target, a.domain as source_domain, 
               b.domain as target_domain
        ORDER BY relationship_type, a.name
        LIMIT 100
        """
        
        try:
            result = self.graph.run(query).data()
            
            if result:
                rels_df = pd.DataFrame(result)
                
                # Filter by domain if specified
                if filters.domains:
                    mask = (rels_df['source_domain'].isin(filters.domains)) | \
                           (rels_df['target_domain'].isin(filters.domains))
                    rels_df = rels_df[mask]
                
                st.dataframe(rels_df, use_container_width=True)
                
                # Relationship type statistics
                st.subheader("Relationship Type Distribution")
                rel_counts = rels_df['relationship_type'].value_counts()
                st.bar_chart(rel_counts)
            else:
                st.info("No relationships found")
                
        except Exception as e:
            st.error(f"Error querying relationships: {e}")
    
    def _show_statistics(self):
        """Show Neo4j database statistics."""
        st.subheader("📊 Database Statistics")
        
        try:
            # Node counts by label
            node_stats = self.graph.run("""
            MATCH (n)
            WITH labels(n) as labels
            UNWIND labels as label
            RETURN label, count(*) as count
            ORDER BY count DESC
            """).data()
            
            if node_stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Node Counts by Type**")
                    nodes_df = pd.DataFrame(node_stats)
                    st.dataframe(nodes_df, use_container_width=True)
                
                with col2:
                    st.write("**Node Distribution**")
                    st.bar_chart(nodes_df.set_index('label')['count'])
            
            # Relationship statistics
            rel_stats = self.graph.run("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(*) as count
            ORDER BY count DESC
            """).data()
            
            if rel_stats:
                st.write("**Relationship Counts**")
                rels_df = pd.DataFrame(rel_stats)
                st.dataframe(rels_df, use_container_width=True)
            
            # Overall statistics
            overall_stats = self.graph.run("""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as total_nodes, count(r) as total_relationships
            """).data()[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Nodes", f"{overall_stats['total_nodes']:,}")
            
            with col2:
                st.metric("Total Relationships", f"{overall_stats['total_relationships']:,}")
                
        except Exception as e:
            st.error(f"Error getting statistics: {e}")
    
    def _show_query_interface(self):
        """Show Cypher query interface."""
        st.subheader("🔍 Cypher Query Interface")
        
        # Predefined queries
        predefined_queries = {
            "All Concepts": "MATCH (c:Concept) RETURN c.name, c.domain LIMIT 20",
            "Cross-Domain Relationships": """
                MATCH (a:Concept)-[r]-(b:Concept)
                WHERE a.domain <> b.domain
                RETURN a.name, a.domain, type(r), b.name, b.domain
                LIMIT 20
            """,
            "Node Degree Distribution": """
                MATCH (n:Concept)
                WITH n, size((n)--()) as degree
                RETURN degree, count(*) as nodes
                ORDER BY degree DESC
            """,
            "Domain Statistics": """
                MATCH (c:Concept)
                RETURN c.domain, count(*) as concept_count
                ORDER BY concept_count DESC
            """
        }
        
        query_type = st.selectbox("Query Type", ["Custom"] + list(predefined_queries.keys()))
        
        if query_type == "Custom":
            query = st.text_area("Cypher Query", height=100, placeholder="MATCH (n) RETURN n LIMIT 10")
        else:
            query = predefined_queries[query_type]
            st.code(query, language="cypher")
        
        if st.button("Execute Query") and query:
            try:
                result = self.graph.run(query).data()
                
                if result:
                    result_df = pd.DataFrame(result)
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download option
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "neo4j_query_results.csv",
                        "text/csv"
                    )
                else:
                    st.info("Query returned no results")
                    
            except Exception as e:
                st.error(f"Query error: {e}")
    
    def _show_concept_details(self, concept: Dict[str, Any]):
        """Show detailed information for a selected concept."""
        st.subheader(f"Concept Details: {concept['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Domain:** {concept['domain']}")
            st.write(f"**ID:** {concept['id']}")
            st.write(f"**Labels:** {', '.join(concept['labels'])}")
        
        with col2:
            if concept['description']:
                st.write("**Description:**")
                st.write(concept['description'])
        
        # Find related concepts
        try:
            related_query = f"""
            MATCH (c:Concept {{id: '{concept['id']}'}})-[r]->(related:Concept)
            RETURN related.name as name, type(r) as relationship, related.domain as domain
            UNION
            MATCH (c:Concept {{id: '{concept['id']}'}})<-[r]-(related:Concept)
            RETURN related.name as name, type(r) as relationship, related.domain as domain
            LIMIT 10
            """
            
            related_result = self.graph.run(related_query).data()
            
            if related_result:
                st.subheader("Related Concepts")
                related_df = pd.DataFrame(related_result)
                st.dataframe(related_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error finding related concepts: {e}")