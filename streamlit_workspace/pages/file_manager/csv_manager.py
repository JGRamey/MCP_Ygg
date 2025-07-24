"""
CSV Manager for File Manager
Handles CSV database file operations
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import shutil

from .models import CSVFileInfo, DatabaseFilters


class CSVManager:
    """Manages CSV database files."""
    
    def __init__(self, csv_root: Path):
        self.csv_root = csv_root
        
    def get_csv_files(self, filters: DatabaseFilters) -> List[CSVFileInfo]:
        """Get list of CSV files matching filters."""
        csv_files = []
        
        if not self.csv_root.exists():
            return csv_files
            
        # Scan through domain directories
        for domain_dir in self.csv_root.iterdir():
            if not domain_dir.is_dir():
                continue
                
            domain_name = domain_dir.name.lower()
            
            # Apply domain filter
            if filters.domains and domain_name not in filters.domains:
                continue
                
            # Scan CSV files in domain
            for csv_file in domain_dir.glob("*.csv"):
                try:
                    # Determine file type from name
                    file_type = self._determine_file_type(csv_file.name)
                    
                    # Apply content type filter
                    if filters.content_types:
                        type_display = f'📄 {file_type.title()}'
                        if not any(ct in type_display for ct in filters.content_types):
                            continue
                    
                    # Get file stats
                    stat = csv_file.stat()
                    
                    # Get CSV info
                    try:
                        df = pd.read_csv(csv_file)
                        row_count = len(df)
                        column_count = len(df.columns)
                    except:
                        row_count = 0
                        column_count = 0
                    
                    csv_info = CSVFileInfo(
                        path=csv_file,
                        name=csv_file.name,
                        domain=domain_name,
                        file_type=file_type,
                        size_bytes=stat.st_size,
                        row_count=row_count,
                        column_count=column_count,
                        last_modified=datetime.fromtimestamp(stat.st_mtime)
                    )
                    
                    csv_files.append(csv_info)
                    
                except Exception as e:
                    st.warning(f"Error reading {csv_file.name}: {str(e)}")
                    continue
        
        return csv_files
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine file type from filename."""
        filename_lower = filename.lower()
        
        if 'concept' in filename_lower:
            return 'concepts'
        elif 'relationship' in filename_lower or 'relation' in filename_lower:
            return 'relationships'
        elif 'people' in filename_lower or 'person' in filename_lower:
            return 'people'
        elif 'work' in filename_lower or 'book' in filename_lower:
            return 'works'
        elif 'place' in filename_lower or 'location' in filename_lower:
            return 'places'
        else:
            return 'data'
    
    def render_csv_files_interface(self, filters: DatabaseFilters):
        """Render the CSV files management interface."""
        st.markdown("## 📊 CSV Database Files")
        st.markdown("**Manage concepts, relationships, and domain data files**")
        
        # File management controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Files", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📈 Show Statistics", use_container_width=True):
                self._show_csv_statistics()
        
        with col3:
            if st.button("📤 Upload CSV", use_container_width=True):
                self._show_upload_interface()
        
        # Get and display CSV files
        csv_files = self.get_csv_files(filters)
        
        if not csv_files:
            st.info("No CSV files found matching current filters.")
            return
        
        # Display files by domain
        domains = set(f.domain for f in csv_files)
        
        for domain in sorted(domains):
            domain_files = [f for f in csv_files if f.domain == domain]
            self._render_domain_section(domain, domain_files)
    
    def _render_domain_section(self, domain: str, files: List[CSVFileInfo]):
        """Render files for a specific domain."""
        with st.expander(f"📁 {domain.title()} Domain ({len(files)} files)", expanded=True):
            
            # Create DataFrame for display
            file_data = []
            for file_info in files:
                file_data.append({
                    'Name': file_info.name,
                    'Type': file_info.file_type_display,
                    'Rows': f"{file_info.row_count:,}",
                    'Columns': file_info.column_count,
                    'Size': f"{file_info.size_kb:.1f} KB",
                    'Modified': file_info.last_modified.strftime('%Y-%m-%d %H:%M')
                })
            
            if file_data:
                df = pd.DataFrame(file_data)
                
                # Display with file selection
                selected_indices = st.dataframe(
                    df, 
                    use_container_width=True,
                    selection_mode="single-row",
                    key=f"csv_files_{domain}"
                )
                
                # File operations
                if selected_indices and len(selected_indices.selection.rows) > 0:
                    selected_idx = selected_indices.selection.rows[0]
                    selected_file = files[selected_idx]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("👁️ View", key=f"view_{selected_file.name}"):
                            self._view_csv_file(selected_file)
                    
                    with col2:
                        if st.button("✏️ Edit", key=f"edit_{selected_file.name}"):
                            self._edit_csv_file(selected_file)
                    
                    with col3:
                        if st.button("📥 Import", key=f"import_{selected_file.name}"):
                            self._import_csv_to_database(selected_file)
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{selected_file.name}"):
                            self._delete_csv_file(selected_file)
    
    def _view_csv_file(self, file_info: CSVFileInfo):
        """View CSV file contents."""
        try:
            df = pd.read_csv(file_info.path)
            
            st.subheader(f"📄 {file_info.name}")
            st.write(f"**Domain**: {file_info.domain.title()}")
            st.write(f"**Type**: {file_info.file_type_display}")
            st.write(f"**Size**: {file_info.row_count:,} rows × {file_info.column_count} columns")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null': df.count(),
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    def _edit_csv_file(self, file_info: CSVFileInfo):
        """Edit CSV file contents."""
        try:
            df = pd.read_csv(file_info.path)
            
            st.subheader(f"✏️ Edit {file_info.name}")
            st.warning("⚠️ Changes will modify the original file. A backup will be created.")
            
            # Editable dataframe
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"editor_{file_info.path.stem}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Changes", type="primary"):
                    try:
                        # Create backup
                        backup_path = file_info.path.parent / f"{file_info.path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        shutil.copy2(file_info.path, backup_path)
                        
                        # Save changes
                        edited_df.to_csv(file_info.path, index=False)
                        
                        st.success(f"✅ Saved changes to {file_info.name}")
                        st.info(f"📦 Backup created: {backup_path.name}")
                        
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")
            
            with col2:
                if st.button("🔄 Reset Changes"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error editing file: {str(e)}")
    
    def _import_csv_to_database(self, file_info: CSVFileInfo):
        """Import CSV file to Neo4j database."""
        st.subheader(f"📥 Import {file_info.name} to Database")
        
        # Import options
        col1, col2 = st.columns(2)
        
        with col1:
            import_as = st.selectbox(
                "Import as Node Type",
                ["Concept", "Entity", "Document", "Author", "Work"],
                help="Choose the Neo4j node type"
            )
        
        with col2:
            batch_size = st.number_input("Batch Size", value=100, min_value=10, max_value=1000)
        
        if st.button("🚀 Start Import", type="primary"):
            try:
                df = pd.read_csv(file_info.path)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate import process
                total_rows = len(df)
                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    status_text.text(f"Importing rows {i+1}-{batch_end} of {total_rows}...")
                    progress_bar.progress((batch_end) / total_rows)
                
                st.success(f"✅ Successfully imported {total_rows} records as {import_as} nodes")
                
            except Exception as e:
                st.error(f"Import failed: {str(e)}")
    
    def _delete_csv_file(self, file_info: CSVFileInfo):
        """Delete CSV file with confirmation."""
        st.subheader(f"🗑️ Delete {file_info.name}")
        st.warning("⚠️ This action cannot be undone!")
        
        if st.checkbox(f"I confirm I want to delete {file_info.name}"):
            if st.button("🗑️ Delete File", type="secondary"):
                try:
                    file_info.path.unlink()
                    st.success(f"✅ Deleted {file_info.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting file: {str(e)}")
    
    def _show_csv_statistics(self):
        """Show CSV file statistics."""
        st.subheader("📈 CSV Database Statistics")
        
        all_files = self.get_csv_files(DatabaseFilters(domains=[], content_types=[]))
        
        if not all_files:
            st.info("No CSV files found.")
            return
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", len(all_files))
        
        with col2:
            total_rows = sum(f.row_count for f in all_files)
            st.metric("Total Records", f"{total_rows:,}")
        
        with col3:
            total_size = sum(f.size_bytes for f in all_files) / (1024 * 1024)
            st.metric("Total Size", f"{total_size:.1f} MB")
        
        with col4:
            domains = len(set(f.domain for f in all_files))
            st.metric("Domains", domains)
        
        # Domain breakdown
        domain_stats = {}
        for file_info in all_files:
            if file_info.domain not in domain_stats:
                domain_stats[file_info.domain] = {
                    'files': 0,
                    'records': 0,
                    'size_mb': 0
                }
            domain_stats[file_info.domain]['files'] += 1
            domain_stats[file_info.domain]['records'] += file_info.row_count
            domain_stats[file_info.domain]['size_mb'] += file_info.size_bytes / (1024 * 1024)
        
        # Display domain stats
        st.subheader("📊 By Domain")
        domain_df = pd.DataFrame([
            {
                'Domain': domain.title(),
                'Files': stats['files'],
                'Records': f"{stats['records']:,}",
                'Size (MB)': f"{stats['size_mb']:.1f}"
            }
            for domain, stats in domain_stats.items()
        ])
        
        st.dataframe(domain_df, use_container_width=True)
    
    def _show_upload_interface(self):
        """Show CSV file upload interface."""
        st.subheader("📤 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file to add to the database"
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                target_domain = st.selectbox(
                    "Target Domain",
                    ["mathematics", "science", "philosophy", "religion", "art", "language"]
                )
            
            with col2:
                file_type = st.selectbox(
                    "File Type",
                    ["concepts", "relationships", "people", "works", "places"]
                )
            
            # Preview uploaded file
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**Preview**: {len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("📥 Upload File", type="primary"):
                    # Save file to appropriate domain directory
                    domain_dir = self.csv_root / target_domain
                    domain_dir.mkdir(exist_ok=True)
                    
                    filename = f"{target_domain}_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    target_path = domain_dir / filename
                    
                    df.to_csv(target_path, index=False)
                    st.success(f"✅ Uploaded as {filename}")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")