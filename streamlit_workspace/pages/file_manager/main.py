"""
File Manager Main Orchestrator
Coordinates all file management components in a modular architecture
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.session_management import add_to_history, mark_unsaved_changes
from .models import DatabaseFilters
from .csv_manager import CSVManager
from .neo4j_manager import Neo4jManager
from .qdrant_manager import QdrantManager
from .backup_manager import BackupManager


class FileManagerApp:
    """
    Main File Manager application orchestrator.
    
    Coordinates between CSV management, Neo4j operations, Qdrant management,
    and backup operations to provide a unified database file management interface.
    """
    
    def __init__(self):
        """Initialize the file manager application."""
        # Get project paths
        project_root = Path(__file__).parent.parent.parent.parent
        csv_root = project_root / "CSV"
        
        # Initialize managers
        self.csv_manager = CSVManager(csv_root)
        self.neo4j_manager = Neo4jManager()
        self.qdrant_manager = QdrantManager()
        self.backup_manager = BackupManager(csv_root)
        
        # Initialize session state
        if 'file_manager_initialized' not in st.session_state:
            st.session_state.file_manager_initialized = True
            st.session_state.database_filters = DatabaseFilters(domains=[], content_types=[])
    
    def render_page(self):
        """Render the main file manager page."""
        st.set_page_config(
            page_title="Database File Manager - MCP Yggdrasil",
            page_icon="📁",
            layout="wide"
        )
        
        # Page header
        st.markdown("# 🗄️ Database Content Manager")
        st.markdown("**Neo4j and Qdrant database content management**")
        
        # Render sidebar with navigation and filters
        self._render_sidebar()
        
        # Get current filters from session state
        filters = DatabaseFilters(
            domains=st.session_state.database_filters.get('domains', []),
            content_types=st.session_state.database_filters.get('content_types', [])
        )
        
        # Main content based on selected operation
        operation = st.session_state.get('selected_operation', '📊 CSV Database Files')
        
        if operation == "📊 CSV Database Files":
            self.csv_manager.render_csv_files_interface(filters)
        elif operation == "🗄️ Neo4j Database":
            self.neo4j_manager.render_neo4j_interface(filters)
        elif operation == "🔍 Qdrant Vectors":
            self.qdrant_manager.render_qdrant_interface(filters)
        elif operation == "📝 Scraped Content":
            self._render_scraped_content_interface(filters)
        elif operation == "📥 Import to Database":
            self._render_database_import_interface(filters)
        elif operation == "💾 Database Backup":
            self.backup_manager.render_backup_interface(filters)
    
    def _render_sidebar(self):
        """Render sidebar with navigation and filters."""
        with st.sidebar:
            st.markdown("### 🧭 Database Operations")
            
            # Main operation selection
            operation = st.selectbox(
                "Select Operation",
                [
                    "📊 CSV Database Files", 
                    "🗄️ Neo4j Database", 
                    "🔍 Qdrant Vectors", 
                    "📝 Scraped Content", 
                    "📥 Import to Database", 
                    "💾 Database Backup"
                ],
                key="operation_selector"
            )
            
            # Store selected operation in session state
            st.session_state.selected_operation = operation
            
            st.markdown("---")
            
            # Database filters section
            st.markdown("### 🗃️ Database Filters")
            
            # Domain filter
            selected_domains = st.multiselect(
                "Filter by Domain",
                ["🎨 Art", "🗣️ Language", "🔢 Mathematics", "🤔 Philosophy", "🔬 Science", "💻 Technology"],
                default=[],
                help="Filter content by domain"
            )
            
            # Content type filter
            content_types = st.multiselect(
                "Content Type",
                ["📄 Concepts", "🔗 Relationships", "📰 Articles", "📚 Books", "🖼️ Images", "🎬 Videos"],
                default=["📄 Concepts", "🔗 Relationships"],
                help="Filter by content type"
            )
            
            # Store filters in session state
            st.session_state.database_filters = {
                'domains': [d.split(' ', 1)[1].lower() for d in selected_domains],  # Remove emoji and lowercase
                'content_types': content_types
            }
            
            # Quick actions section
            st.markdown("---")
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            with col2:
                if st.button("📊 Stats", use_container_width=True):
                    self._show_quick_stats()
            
            # System status section
            st.markdown("---")
            st.markdown("### 🔌 System Status")
            
            # Check database connections
            neo4j_status = "🟢 Connected" if self.neo4j_manager.is_connected() else "🔴 Offline"
            st.write(f"**Neo4j**: {neo4j_status}")
            
            qdrant_status = "🟢 Connected" if self.qdrant_manager.is_connected() else "🔴 Offline"
            st.write(f"**Qdrant**: {qdrant_status}")
            
            # CSV files status
            csv_files = self.csv_manager.get_csv_files(DatabaseFilters(domains=[], content_types=[]))
            st.write(f"**CSV Files**: {len(csv_files)} found")
    
    def _show_quick_stats(self):
        """Show quick database statistics."""
        st.subheader("📊 Quick Database Statistics")
        
        # CSV stats
        csv_files = self.csv_manager.get_csv_files(DatabaseFilters(domains=[], content_types=[]))
        total_csv_records = sum(f.row_count for f in csv_files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CSV Files", len(csv_files))
            st.metric("CSV Records", f"{total_csv_records:,}")
        
        with col2:
            # Neo4j stats (if connected)
            if self.neo4j_manager.is_connected():
                try:
                    result = self.neo4j_manager.graph.run("MATCH (n) RETURN count(n) as count").data()
                    node_count = result[0]['count'] if result else 0
                    st.metric("Neo4j Nodes", f"{node_count:,}")
                    
                    rel_result = self.neo4j_manager.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()
                    rel_count = rel_result[0]['count'] if rel_result else 0
                    st.metric("Neo4j Relationships", f"{rel_count:,}")
                except:
                    st.metric("Neo4j Nodes", "Error")
                    st.metric("Neo4j Relationships", "Error")
            else:
                st.metric("Neo4j Nodes", "Offline")
                st.metric("Neo4j Relationships", "Offline")
        
        with col3:
            # Qdrant stats (if connected)
            if self.qdrant_manager.is_connected():
                collections = self.qdrant_manager.get_collections()
                total_vectors = sum(c.vectors_count for c in collections)
                st.metric("Qdrant Collections", len(collections))
                st.metric("Total Vectors", f"{total_vectors:,}")
            else:
                st.metric("Qdrant Collections", "Offline")
                st.metric("Total Vectors", "Offline")
    
    def _render_scraped_content_interface(self, filters: DatabaseFilters):
        """Render scraped content management interface."""
        st.markdown("## 📝 Scraped Content")
        st.markdown("**Manage scraped content awaiting approval**")
        
        # Mock scraped content for demonstration
        st.info("📋 Scraped content management would integrate with the staging system from Phase 4.")
        
        # Content status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pending Review", "42")
        
        with col2:
            st.metric("Approved", "128")
        
        with col3:
            st.metric("Rejected", "15")
        
        with col4:
            st.metric("Processing", "7")
        
        # Content management tabs
        content_tabs = st.tabs(["⏳ Pending", "✅ Approved", "❌ Rejected", "🔄 Processing"])
        
        with content_tabs[0]:
            st.info("Pending content review interface would be implemented here.")
        
        with content_tabs[1]:
            st.info("Approved content ready for import would be shown here.")
        
        with content_tabs[2]:
            st.info("Rejected content with reasons would be listed here.")
        
        with content_tabs[3]:
            st.info("Currently processing content would be displayed here.")
    
    def _render_database_import_interface(self, filters: DatabaseFilters):
        """Render database import interface."""
        st.markdown("## 📥 Import to Database")
        st.markdown("**Import data directly to Neo4j and Qdrant databases**")
        
        # Import type selection
        import_type = st.selectbox(
            "Import Type",
            [
                "CSV to Neo4j Nodes",
                "CSV to Neo4j Relationships", 
                "Documents to Qdrant Vectors",
                "Approved Staging Content",
                "Bulk Data Import"
            ]
        )
        
        if import_type == "CSV to Neo4j Nodes":
            self._render_csv_to_neo4j_import()
        elif import_type == "CSV to Neo4j Relationships":
            self._render_csv_to_relationships_import()
        elif import_type == "Documents to Qdrant Vectors":
            self._render_documents_to_qdrant_import()
        elif import_type == "Approved Staging Content":
            st.info("This would integrate with the validation pipeline from Phase 4.")
        else:
            st.info(f"{import_type} functionality would be implemented here.")
    
    def _render_csv_to_neo4j_import(self):
        """Render CSV to Neo4j import interface."""
        st.subheader("📄 Import CSV as Neo4j Nodes")
        
        # File selection from CSV manager
        csv_files = self.csv_manager.get_csv_files(DatabaseFilters(domains=[], content_types=[]))
        
        if not csv_files:
            st.warning("No CSV files found. Please add CSV files first.")
            return
        
        # CSV file selection
        selected_csv = st.selectbox(
            "Select CSV File",
            [f"{f.domain}/{f.name}" for f in csv_files],
            help="Choose CSV file to import as Neo4j nodes"
        )
        
        if selected_csv:
            # Get selected file info
            domain, filename = selected_csv.split('/', 1)
            selected_file = next(f for f in csv_files if f.domain == domain and f.name == filename)
            
            # Import configuration
            col1, col2 = st.columns(2)
            
            with col1:
                node_label = st.text_input("Neo4j Node Label", value="Concept")
                batch_size = st.number_input("Batch Size", value=100, min_value=10, max_value=1000)
            
            with col2:
                create_relationships = st.checkbox("Create relationships", value=False)
                skip_duplicates = st.checkbox("Skip duplicates", value=True)
            
            # Preview
            st.subheader("📋 Import Preview")
            st.write(f"**File**: {selected_file.name}")
            st.write(f"**Records**: {selected_file.row_count:,}")
            st.write(f"**Target Label**: {node_label}")
            
            if st.button("🚀 Start Import", type="primary"):
                self._execute_csv_import(selected_file, node_label, batch_size, skip_duplicates)
    
    def _render_csv_to_relationships_import(self):
        """Render CSV to Neo4j relationships import interface."""
        st.subheader("🔗 Import CSV as Neo4j Relationships")
        st.info("Relationship import functionality would be implemented here.")
    
    def _render_documents_to_qdrant_import(self):
        """Render documents to Qdrant import interface."""
        st.subheader("🔍 Import Documents to Qdrant Vectors")
        st.info("Document vectorization and Qdrant import would be implemented here.")
    
    def _execute_csv_import(self, csv_file, node_label: str, batch_size: int, skip_duplicates: bool):
        """Execute CSV import to Neo4j."""
        if not self.neo4j_manager.is_connected():
            st.error("❌ Neo4j is not connected. Cannot import.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            import pandas as pd
            
            # Load CSV file
            df = pd.read_csv(csv_file.path)
            total_rows = len(df)
            
            status_text.text(f"Importing {total_rows} records...")
            
            # Process in batches
            imported_count = 0
            
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_df = df.iloc[i:batch_end]
                
                # Create Cypher query for batch
                for _, row in batch_df.iterrows():
                    try:
                        # Create node properties
                        properties = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                        
                        # Build Cypher query
                        if skip_duplicates:
                            query = f"""
                            MERGE (n:{node_label} {{name: $name}})
                            SET n += $properties
                            RETURN n
                            """
                        else:
                            query = f"""
                            CREATE (n:{node_label})
                            SET n += $properties
                            RETURN n
                            """
                        
                        # Execute query
                        self.neo4j_manager.graph.run(query, name=properties.get('name', ''), properties=properties)
                        imported_count += 1
                        
                    except Exception as e:
                        st.warning(f"Error importing row {i + imported_count}: {str(e)}")
                
                # Update progress
                progress_bar.progress((batch_end) / total_rows)
                status_text.text(f"Imported {batch_end}/{total_rows} records...")
            
            st.success(f"✅ Successfully imported {imported_count} records as {node_label} nodes!")
            
            # Add to history
            add_to_history(f"Imported {imported_count} records from {csv_file.name} to Neo4j")
            
        except Exception as e:
            st.error(f"Import failed: {str(e)}")


def main():
    """Main function to render the file manager application."""
    app = FileManagerApp()
    app.render_page()


if __name__ == "__main__":
    main()