"""
Database Manager - Neo4j and Qdrant Database Content Management
Focus on actual database content, not file system files
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import json
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main Database Manager interface"""
    
    st.set_page_config(
        page_title="Database Manager - MCP Yggdrasil",
        page_icon="📁",
        layout="wide"
    )
    
    # Header
    st.markdown("# 🗄️ Database Content Manager")
    st.markdown("**Neo4j and Qdrant database content management**")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Database Operations")
        
        operation = st.selectbox(
            "Select Operation",
            ["🗄️ Neo4j Database", "🔍 Qdrant Vectors", "📝 Scraped Content", "📥 Import to Database", "💾 Database Backup"]
        )
        
        st.markdown("---")
        
        # Database filters
        st.markdown("### 🗃️ Database Filters")
        
        # Domain filter
        selected_domains = st.multiselect(
            "Filter by Domain",
            ["🎨 Art", "🗣️ Language", "🔢 Mathematics", "🤔 Philosophy", "🔬 Science", "💻 Technology"],
            default=[]
        )
        
        # Content type filter
        content_types = st.multiselect(
            "Content Type",
            ["📄 Concepts", "🔗 Relationships", "📰 Articles", "📚 Books", "🖼️ Images", "🎬 Videos"],
            default=["📄 Concepts", "🔗 Relationships"]
        )
        
        # Store filters in session state
        st.session_state.database_filters = {
            'domains': [d.split(' ', 1)[1].lower() for d in selected_domains],  # Remove emoji and lowercase
            'content_types': content_types
        }
    
    # Main content based on operation
    if operation == "🗄️ Neo4j Database":
        show_neo4j_database()
    elif operation == "🔍 Qdrant Vectors":
        show_qdrant_vectors()
    elif operation == "📝 Scraped Content":
        show_scraped_content()
    elif operation == "📥 Import to Database":
        show_database_import()
    elif operation == "💾 Database Backup":
        show_database_backup()

def show_neo4j_database():
    """Show Neo4j database content"""
    st.markdown("## 🗄️ Neo4j Database Content")
    
    # Get database filters
    filters = st.session_state.get('database_filters', {})
    
    # Database connection status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            from utils.database_operations import test_connections
            connections = test_connections()
            if connections.get('neo4j'):
                st.success("✅ Neo4j Connected")
            else:
                st.error("❌ Neo4j Disconnected")
        except:
            st.warning("⚠️ Connection status unknown")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("➕ Add Concept", use_container_width=True):
            st.session_state.show_add_concept = True
    
    # Show database content
    tab1, tab2, tab3 = st.tabs(["📄 Concepts", "🔗 Relationships", "📊 Statistics"])
    
    with tab1:
        show_neo4j_concepts(filters)
    
    with tab2:
        show_neo4j_relationships(filters)
    
    with tab3:
        show_neo4j_statistics()

def show_neo4j_concepts(filters):
    """Show concepts from Neo4j database"""
    try:
        from utils.database_operations import get_all_concepts, delete_concept
        
        # Get concepts with filters
        domain_filter = filters.get('domains', [])
        if domain_filter:
            concepts = []
            for domain in domain_filter:
                concepts.extend(get_all_concepts(domain=domain, limit=100))
        else:
            concepts = get_all_concepts(limit=200)
        
        if not concepts:
            st.info("📭 No concepts found in Neo4j database")
            st.markdown("**💡 Suggestions:**")
            st.markdown("- Import CSV data to Neo4j")
            st.markdown("- Add concepts manually")
            st.markdown("- Check database connection")
            return
        
        # Display concepts in a table
        st.markdown(f"### 📄 Concepts ({len(concepts)} found)")
        
        # Search within concepts
        search_term = st.text_input("🔍 Search concepts", placeholder="Enter concept name or description...")
        
        if search_term:
            concepts = [c for c in concepts if search_term.lower() in c['name'].lower() or 
                       search_term.lower() in str(c.get('description', '')).lower()]
        
        # Display concepts
        for i, concept in enumerate(concepts[:50]):  # Limit display
            with st.expander(f"📄 {concept['name']} ({concept['domain']})", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**ID:** {concept['id']}")
                    st.markdown(f"**Domain:** {concept['domain']}")
                    st.markdown(f"**Type:** {concept.get('type', 'Unknown')}")
                    st.markdown(f"**Level:** {concept.get('level', 'Unknown')}")
                    st.markdown(f"**Description:** {concept.get('description', 'No description')}")
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_concept_{i}"):
                        st.session_state.edit_concept_id = concept['id']
                    
                    if st.button("🗑️ Delete", key=f"delete_concept_{i}"):
                        if st.button(f"⚠️ Confirm Delete {concept['name']}", key=f"confirm_delete_{i}"):
                            success, message = delete_concept(concept['id'])
                            if success:
                                st.success(f"✅ Deleted {concept['name']}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
    
    except Exception as e:
        st.error(f"Error loading concepts: {str(e)}")

def show_neo4j_relationships(filters):
    """Show relationships from Neo4j database"""
    try:
        from utils.database_operations import get_all_concepts, get_concept_relationships
        
        st.markdown("### 🔗 Relationships")
        
        # Get some concepts to show their relationships
        concepts = get_all_concepts(limit=20)
        
        if not concepts:
            st.info("📭 No concepts found to show relationships")
            return
        
        # Select a concept to view relationships
        concept_names = [f"{c['id']}: {c['name']}" for c in concepts]
        selected = st.selectbox("Select concept to view relationships", concept_names)
        
        if selected:
            concept_id = selected.split(':')[0]
            relationships = get_concept_relationships(concept_id)
            
            if relationships:
                st.markdown(f"#### Relationships for {selected}")
                
                for rel in relationships:
                    direction_icon = "➡️" if rel['direction'] == 'outgoing' else "⬅️"
                    st.markdown(f"{direction_icon} **{rel['type']}** → {rel['target_name']} ({rel['target_domain']})")
            else:
                st.info("No relationships found for this concept")
    
    except Exception as e:
        st.error(f"Error loading relationships: {str(e)}")

def show_neo4j_statistics():
    """Show Neo4j database statistics"""
    try:
        from utils.database_operations import get_quick_stats
        
        stats = get_quick_stats()
        
        if 'error' in stats:
            st.error(f"Database error: {stats['error']}")
            return
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Total Concepts", stats.get('concepts', 0))
        
        with col2:
            st.metric("🔗 Total Relationships", stats.get('relationships', 0))
        
        with col3:
            st.metric("🗃️ Domains", stats.get('domains', 0))
        
        # Additional statistics
        if stats.get('concepts', 0) > 0:
            from utils.database_operations import get_domains
            domains = get_domains()
            
            if domains:
                st.markdown("#### 📊 Domain Distribution")
                for domain in domains:
                    st.markdown(f"- **{domain['domain']}**: {domain['concept_count']} concepts")
    
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def show_qdrant_vectors():
    """Show Qdrant vector database content"""
    st.markdown("## 🔍 Qdrant Vector Database")
    
    # Connection status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            import requests
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                st.success("✅ Qdrant Connected")
            else:
                st.error("❌ Qdrant Disconnected")
        except:
            st.warning("⚠️ Qdrant not accessible")
    
    with col2:
        if st.button("🔄 Refresh Vectors", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("➕ Add Vector", use_container_width=True):
            st.session_state.show_add_vector = True
    
    # Show vector collections and data
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        
        if response.status_code == 200:
            collections = response.json()
            
            if collections.get('result', {}).get('collections'):
                st.markdown("### 📊 Vector Collections")
                
                for collection in collections['result']['collections']:
                    with st.expander(f"📦 {collection['name']}", expanded=False):
                        # Get collection info
                        info_response = requests.get(f"http://localhost:6333/collections/{collection['name']}")
                        if info_response.status_code == 200:
                            info = info_response.json()
                            result = info.get('result', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Vectors", result.get('vectors_count', 0))
                            with col2:
                                st.metric("Points", result.get('points_count', 0))
                            with col3:
                                st.metric("Status", result.get('status', 'Unknown'))
            else:
                st.info("📭 No vector collections found")
        else:
            st.error("Failed to connect to Qdrant")
    
    except Exception as e:
        st.error(f"Error loading Qdrant data: {str(e)}")

def show_scraped_content():
    """Show scraped content from content scraper"""
    st.markdown("## 📝 Scraped Content")
    
    # Try to load scraped content
    try:
        project_root = Path(__file__).parent.parent.parent
        submissions_file = project_root / "data" / "submissions.json"
        
        if submissions_file.exists():
            with open(submissions_file, 'r') as f:
                submissions = json.load(f)
            
            if submissions:
                st.markdown(f"### 📋 Scraped Content ({len(submissions)} items)")
                
                for sub_id, submission in submissions.items():
                    with st.expander(f"📄 {submission.get('title', 'Unknown')} - {submission.get('domain', 'Unknown')}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**ID:** {sub_id}")
                            st.markdown(f"**Title:** {submission.get('title', 'Unknown')}")
                            st.markdown(f"**Domain:** {submission.get('domain', 'Unknown')}")
                            st.markdown(f"**Source:** {submission.get('source_url', 'Unknown')}")
                            st.markdown(f"**Length:** {submission.get('content_length', 0)} characters")
                            st.markdown(f"**Status:** {submission.get('status', 'Unknown')}")
                            st.markdown(f"**Date:** {submission.get('timestamp', 'Unknown')}")
                            
                            if submission.get('content'):
                                st.text_area("Content Preview", submission['content'], height=100)
                        
                        with col2:
                            if st.button("💾 Save to Neo4j", key=f"save_{sub_id}"):
                                st.info("Feature coming soon!")
                            
                            if st.button("🗑️ Delete", key=f"delete_{sub_id}"):
                                del submissions[sub_id]
                                with open(submissions_file, 'w') as f:
                                    json.dump(submissions, f, indent=2)
                                st.success("✅ Deleted!")
                                st.rerun()
            else:
                st.info("📭 No scraped content found")
        else:
            st.info("📭 No scraped content found")
    
    except Exception as e:
        st.error(f"Error loading scraped content: {str(e)}")

def show_database_import():
    """Show database import interface"""
    st.markdown("## 📥 Import to Database")
    
    st.info("🚧 Database import functionality coming soon!")
    st.markdown("**Planned features:**")
    st.markdown("- Import CSV files to Neo4j")
    st.markdown("- Generate vectors for Qdrant")
    st.markdown("- Process scraped content")
    st.markdown("- Bulk data operations")

def show_database_backup():
    """Show database backup interface"""
    st.markdown("## 💾 Database Backup")
    
    st.info("🚧 Database backup functionality coming soon!")
    st.markdown("**Planned features:**")
    st.markdown("- Export Neo4j data")
    st.markdown("- Backup Qdrant vectors")
    st.markdown("- Schedule automated backups")
    st.markdown("- Restore from backups")

if __name__ == "__main__":
    main()