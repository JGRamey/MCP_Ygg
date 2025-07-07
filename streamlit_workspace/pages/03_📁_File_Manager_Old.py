"""
File Manager - Project File and Configuration Management
Complete interface for editing CSV files, configurations, and project structure
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
import json
import yaml
import subprocess
import tempfile
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main File Manager interface"""
    
    st.set_page_config(
        page_title="File Manager - MCP Yggdrasil",
        page_icon="📁",
        layout="wide"
    )
    
    # Custom CSS for file manager
    st.markdown("""
    <style>
    .file-browser {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .file-item {
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .file-item:hover {
        background-color: #f8f9fa;
    }
    
    .file-icon {
        font-size: 1.2rem;
        width: 20px;
    }
    
    .file-name {
        flex: 1;
        font-family: 'Courier New', monospace;
    }
    
    .file-size {
        color: #666;
        font-size: 0.9rem;
    }
    
    .editor-container {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .code-editor {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .action-bar {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    .git-status {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E8B57;
    }
    
    .backup-panel {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 📁 Data File Manager")
    st.markdown("**Database content and knowledge data management**")
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 File Operations")
        
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
        
        # Date range filter
        st.markdown("#### 📅 Date Range")
        date_filter = st.date_input("Filter by date range", value=None)
        
        # Store filters in session state
        st.session_state.database_filters = {
            'domains': [d.split(' ', 1)[1].lower() for d in selected_domains],  # Remove emoji and lowercase
            'content_types': content_types,
            'date_filter': date_filter
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
    
    # Path navigation
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.text_input("📂 Current Path", value=str(current_path), disabled=True)
    
    with col2:
        if st.button("⬆️ Parent", use_container_width=True):
            if current_path.parent != current_path:
                st.session_state.browse_path = current_path.parent
                st.rerun()
    
    with col3:
        if st.button("🏠 Root", use_container_width=True):
            st.session_state.browse_path = project_root
            st.rerun()
    
    # File listing
    try:
        if current_path.exists() and current_path.is_dir():
            items = list(current_path.iterdir())
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            # Create file browser
            st.markdown('<div class="file-browser">', unsafe_allow_html=True)
            
            for item in items:
                if item.name.startswith('.'):
                    continue  # Skip hidden files
                
                # File icon
                if item.is_dir():
                    icon = "📁"
                elif item.suffix in ['.py']:
                    icon = "🐍"
                elif item.suffix in ['.csv']:
                    icon = "📊"
                elif item.suffix in ['.yaml', '.yml']:
                    icon = "⚙️"
                elif item.suffix in ['.json']:
                    icon = "📋"
                elif item.suffix in ['.md']:
                    icon = "📝"
                else:
                    icon = "📄"
                
                # File size
                if item.is_file():
                    size = format_file_size(item.stat().st_size)
                else:
                    size = ""
                
                # Create clickable file item
                col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
                
                with col1:
                    st.markdown(f'<div class="file-icon">{icon}</div>', unsafe_allow_html=True)
                
                with col2:
                    if item.is_dir():
                        if st.button(item.name, key=f"dir_{item.name}", use_container_width=True):
                            st.session_state.browse_path = item
                            st.rerun()
                    else:
                        if st.button(item.name, key=f"file_{item.name}", use_container_width=True):
                            st.session_state.selected_file = item
                            st.rerun()
                
                with col3:
                    st.text(size)
                
                with col4:
                    if item.is_file() and st.button("✏️", key=f"edit_{item.name}"):
                        st.session_state.edit_file = item
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Path does not exist: {current_path}")
    
    except Exception as e:
        st.error(f"Error browsing files: {str(e)}")
    
    # File preview/editor
    if 'selected_file' in st.session_state:
        show_file_preview(st.session_state.selected_file)

def show_file_preview(file_path):
    """Show file preview and basic editing"""
    st.markdown("---")
    st.markdown(f"### 📄 File: {file_path.name}")
    
    try:
        if file_path.suffix in ['.py', '.md', '.txt', '.yaml', '.yml', '.json', '.csv']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Size", format_file_size(file_path.stat().st_size))
            with col2:
                st.metric("Type", file_path.suffix or "No extension")
            with col3:
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                st.metric("Modified", mod_time.strftime("%Y-%m-%d %H:%M"))
            
            # Content preview
            if file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    st.markdown("#### 📊 CSV Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 rows of {len(df)} total rows")
                except Exception as e:
                    st.warning(f"Could not parse CSV: {e}")
                    st.code(content[:1000] + "..." if len(content) > 1000 else content)
            else:
                st.markdown("#### 📝 Content Preview")
                preview_length = 2000
                if len(content) > preview_length:
                    st.code(content[:preview_length] + f"\n... ({len(content) - preview_length} more characters)")
                else:
                    st.code(content)
            
            # Edit button
            if st.button("✏️ Edit File", type="primary"):
                st.session_state.edit_file = file_path
                st.rerun()
        
        else:
            st.info(f"File type {file_path.suffix} not supported for preview")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

def show_csv_editor(project_root):
    """Show CSV file editor"""
    st.markdown("## 📝 CSV File Editor")
    
    # Find CSV files
    csv_files = []
    csv_dir = project_root / "CSV"
    
    if csv_dir.exists():
        for csv_file in csv_dir.rglob("*.csv"):
            csv_files.append(csv_file)
    
    if not csv_files:
        st.warning("No CSV files found in the project")
        return
    
    # CSV file selection
    selected_csv = st.selectbox(
        "📊 Select CSV File",
        csv_files,
        format_func=lambda x: str(x.relative_to(project_root))
    )
    
    if selected_csv:
        try:
            df = pd.read_csv(selected_csv)
            
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", format_file_size(selected_csv.stat().st_size))
            
            # Edit mode selection
            edit_mode = st.radio(
                "Edit Mode",
                ["📊 View & Filter", "✏️ Edit Data", "➕ Add Rows", "🗑️ Delete Rows"],
                horizontal=True
            )
            
            if edit_mode == "📊 View & Filter":
                show_csv_viewer(df, selected_csv)
            elif edit_mode == "✏️ Edit Data":
                show_csv_editor_mode(df, selected_csv)
            elif edit_mode == "➕ Add Rows":
                show_csv_add_rows(df, selected_csv)
            elif edit_mode == "🗑️ Delete Rows":
                show_csv_delete_rows(df, selected_csv)
        
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")

def show_csv_viewer(df, file_path):
    """Show CSV viewer with filtering"""
    st.markdown("### 📊 CSV Viewer")
    
    # Column filters
    if len(df.columns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            columns_to_show = st.multiselect(
                "Columns to Display",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]  # Show first 5 by default
            )
        
        with col2:
            if 'domain' in df.columns:
                domain_filter = st.multiselect(
                    "Filter by Domain",
                    df['domain'].unique() if 'domain' in df.columns else [],
                    default=[]
                )
            else:
                domain_filter = []
        
        # Apply filters
        filtered_df = df.copy()
        if domain_filter:
            filtered_df = filtered_df[filtered_df['domain'].isin(domain_filter)]
        
        if columns_to_show:
            filtered_df = filtered_df[columns_to_show]
        
        # Display data
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Filtered Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv_data,
                    file_name=f"filtered_{file_path.name}",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Generate Summary"):
                show_csv_summary(filtered_df)

def show_csv_editor_mode(df, file_path):
    """Show CSV editing interface"""
    st.markdown("### ✏️ Edit CSV Data")
    
    # Row selection for editing
    if len(df) > 0:
        row_index = st.selectbox("Select Row to Edit", range(len(df)), format_func=lambda x: f"Row {x}: {df.iloc[x, 0] if len(df.columns) > 0 else x}")
        
        if row_index is not None:
            st.markdown("#### Edit Selected Row")
            
            # Create editable form
            with st.form("edit_row_form"):
                edited_values = {}
                
                col1, col2 = st.columns(2)
                columns = df.columns.tolist()
                
                for i, column in enumerate(columns):
                    current_value = df.iloc[row_index, i]
                    
                    with col1 if i % 2 == 0 else col2:
                        if pd.isna(current_value):
                            current_value = ""
                        edited_values[column] = st.text_input(
                            f"{column}",
                            value=str(current_value),
                            key=f"edit_{column}_{row_index}"
                        )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                        try:
                            # Update dataframe
                            for column, value in edited_values.items():
                                df.iloc[row_index, df.columns.get_loc(column)] = value
                            
                            # Save to file
                            df.to_csv(file_path, index=False)
                            st.success("✅ Changes saved successfully!")
                            add_to_history("EDIT", f"Modified CSV file: {file_path.name}")
                            mark_unsaved_changes(False)
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error saving changes: {str(e)}")
                
                with col2:
                    if st.form_submit_button("🔄 Reset", use_container_width=True):
                        st.rerun()

def show_csv_add_rows(df, file_path):
    """Show interface for adding new rows"""
    st.markdown("### ➕ Add New Rows")
    
    with st.form("add_row_form"):
        st.markdown("#### Enter New Row Data")
        
        new_values = {}
        col1, col2 = st.columns(2)
        
        for i, column in enumerate(df.columns):
            with col1 if i % 2 == 0 else col2:
                new_values[column] = st.text_input(f"{column}", key=f"new_{column}")
        
        if st.form_submit_button("➕ Add Row", type="primary"):
            try:
                # Create new row
                new_row = pd.DataFrame([new_values])
                
                # Append to dataframe
                updated_df = pd.concat([df, new_row], ignore_index=True)
                
                # Save to file
                updated_df.to_csv(file_path, index=False)
                st.success("✅ New row added successfully!")
                add_to_history("CREATE", f"Added row to CSV file: {file_path.name}")
                mark_unsaved_changes(False)
                st.rerun()
            
            except Exception as e:
                st.error(f"Error adding row: {str(e)}")

def show_csv_delete_rows(df, file_path):
    """Show interface for deleting rows"""
    st.markdown("### 🗑️ Delete Rows")
    
    if len(df) > 0:
        # Multi-select for rows to delete
        rows_to_delete = st.multiselect(
            "Select Rows to Delete",
            range(len(df)),
            format_func=lambda x: f"Row {x}: {df.iloc[x, 0] if len(df.columns) > 0 else x}"
        )
        
        if rows_to_delete:
            st.warning(f"⚠️ You are about to delete {len(rows_to_delete)} row(s)")
            
            # Show preview of rows to be deleted
            st.markdown("#### Rows to be deleted:")
            st.dataframe(df.iloc[rows_to_delete], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                    try:
                        # Remove selected rows
                        updated_df = df.drop(rows_to_delete).reset_index(drop=True)
                        
                        # Save to file
                        updated_df.to_csv(file_path, index=False)
                        st.success(f"✅ Deleted {len(rows_to_delete)} row(s) successfully!")
                        add_to_history("DELETE", f"Deleted {len(rows_to_delete)} rows from CSV file: {file_path.name}")
                        mark_unsaved_changes(False)
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error deleting rows: {str(e)}")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.rerun()

def show_config_editor(project_root):
    """Show configuration file editor"""
    st.markdown("## ⚙️ Configuration Editor")
    
    # Find config files
    config_files = []
    
    # Common config file patterns
    config_patterns = [
        project_root / ".env",
        project_root / "docker-compose.yml",
        project_root / "config" / "server.yaml",
        project_root / "pyproject.toml",
        project_root / "requirements.txt"
    ]
    
    for pattern in config_patterns:
        if pattern.exists():
            config_files.append(pattern)
    
    # Also find YAML files in config directory
    config_dir = project_root / "config"
    if config_dir.exists():
        for yaml_file in config_dir.glob("*.yaml"):
            if yaml_file not in config_files:
                config_files.append(yaml_file)
        for yml_file in config_dir.glob("*.yml"):
            if yml_file not in config_files:
                config_files.append(yml_file)
    
    if not config_files:
        st.warning("No configuration files found")
        return
    
    # File selection
    selected_config = st.selectbox(
        "⚙️ Select Configuration File",
        config_files,
        format_func=lambda x: str(x.relative_to(project_root))
    )
    
    if selected_config:
        show_config_file_editor(selected_config, project_root)

def show_config_file_editor(file_path, project_root):
    """Show editor for specific config file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # File info
        st.markdown(f"### ⚙️ Editing: {file_path.relative_to(project_root)}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Size", format_file_size(file_path.stat().st_size))
        with col2:
            st.metric("Type", file_path.suffix or "No extension")
        with col3:
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            st.metric("Modified", mod_time.strftime("%Y-%m-%d %H:%M"))
        
        # Editor
        st.markdown("#### 📝 Content Editor")
        
        # Determine editor type based on file extension
        if file_path.suffix in ['.yaml', '.yml']:
            edited_content = st.text_area(
                "YAML Content",
                content,
                height=400,
                help="Edit YAML configuration. Be careful with indentation!"
            )
            
            # YAML validation
            try:
                yaml.safe_load(edited_content)
                st.success("✅ Valid YAML syntax")
            except yaml.YAMLError as e:
                st.error(f"❌ YAML syntax error: {e}")
        
        elif file_path.suffix == '.json':
            edited_content = st.text_area("JSON Content", content, height=400)
            
            # JSON validation
            try:
                json.loads(edited_content)
                st.success("✅ Valid JSON syntax")
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON syntax error: {e}")
        
        else:
            edited_content = st.text_area("File Content", content, height=400)
        
        # Save controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                try:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Save new content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(edited_content)
                    
                    st.success("✅ File saved successfully!")
                    add_to_history("EDIT", f"Modified config file: {file_path.name}")
                    mark_unsaved_changes(False)
                
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("📥 Download", use_container_width=True):
                st.download_button(
                    "💾 Download File",
                    edited_content,
                    file_name=file_path.name,
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"Error reading configuration file: {str(e)}")

def show_git_management(project_root):
    """Show Git management interface"""
    st.markdown("## 🐙 Git Management")
    
    try:
        # Check if it's a git repository
        result = subprocess.run(['git', 'status'], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.warning("This is not a Git repository or Git is not installed")
            return
        
        # Git status
        st.markdown("### 📊 Repository Status")
        st.markdown('<div class="git-status">', unsafe_allow_html=True)
        st.code(result.stdout, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Git operations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📥 Pull Changes")
            if st.button("🔄 Pull Latest", use_container_width=True):
                pull_result = subprocess.run(['git', 'pull'], cwd=project_root, capture_output=True, text=True)
                if pull_result.returncode == 0:
                    st.success("✅ Pull successful")
                    st.code(pull_result.stdout)
                else:
                    st.error("❌ Pull failed")
                    st.code(pull_result.stderr)
        
        with col2:
            st.markdown("#### 📤 Commit Changes")
            commit_message = st.text_input("Commit Message", placeholder="Describe your changes...")
            if st.button("💾 Commit", use_container_width=True) and commit_message:
                # Add all changes
                add_result = subprocess.run(['git', 'add', '.'], cwd=project_root, capture_output=True, text=True)
                
                # Commit
                commit_result = subprocess.run(['git', 'commit', '-m', commit_message], cwd=project_root, capture_output=True, text=True)
                
                if commit_result.returncode == 0:
                    st.success("✅ Commit successful")
                    add_to_history("GIT", f"Committed changes: {commit_message}")
                else:
                    st.warning(commit_result.stdout or commit_result.stderr)
        
        with col3:
            st.markdown("#### 📡 Push Changes")
            if st.button("🚀 Push", use_container_width=True):
                push_result = subprocess.run(['git', 'push'], cwd=project_root, capture_output=True, text=True)
                if push_result.returncode == 0:
                    st.success("✅ Push successful")
                else:
                    st.error("❌ Push failed")
                    st.code(push_result.stderr)
        
        # Recent commits
        st.markdown("### 📜 Recent Commits")
        log_result = subprocess.run(['git', 'log', '--oneline', '-10'], cwd=project_root, capture_output=True, text=True)
        if log_result.returncode == 0:
            st.code(log_result.stdout, language="bash")
    
    except Exception as e:
        st.error(f"Error with Git operations: {str(e)}")

def show_backup_manager(project_root):
    """Show backup management interface"""
    st.markdown("## 💾 Backup Manager")
    
    backup_dir = project_root / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Create backup section
    st.markdown("### 📦 Create New Backup")
    st.markdown('<div class="backup-panel">', unsafe_allow_html=True)
    
    backup_name = st.text_input("Backup Name", value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    include_options = st.multiselect(
        "Include in Backup",
        ["CSV Files", "Configuration Files", "Agent Scripts", "Documentation"],
        default=["CSV Files", "Configuration Files"]
    )
    
    if st.button("📦 Create Backup", type="primary"):
        create_project_backup(project_root, backup_dir, backup_name, include_options)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # List existing backups
    st.markdown("### 📋 Existing Backups")
    
    if backup_dir.exists():
        backups = list(backup_dir.glob("*.tar.gz"))
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if backups:
            for backup in backups:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.text(backup.name)
                
                with col2:
                    st.text(format_file_size(backup.stat().st_size))
                
                with col3:
                    mod_time = datetime.fromtimestamp(backup.stat().st_mtime)
                    st.text(mod_time.strftime("%Y-%m-%d %H:%M"))
                
                with col4:
                    if st.button("📥", key=f"download_{backup.name}"):
                        with open(backup, 'rb') as f:
                            st.download_button(
                                "💾 Download",
                                f.read(),
                                file_name=backup.name,
                                mime="application/gzip",
                                key=f"dl_{backup.name}"
                            )
        else:
            st.info("No backups found")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def show_csv_summary(df):
    """Show CSV summary statistics"""
    st.markdown("#### 📊 Data Summary")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Column info
    st.markdown("#### Column Information")
    column_info = []
    for col in df.columns:
        column_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique()
        })
    
    info_df = pd.DataFrame(column_info)
    st.dataframe(info_df, use_container_width=True)

def show_data_import(project_root):
    """Show data import interface"""
    st.markdown("## 📥 Import Data")
    st.markdown("Import new data files into the knowledge base")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a data file", 
        type=['csv', 'json', 'txt', 'tsv'],
        help="Upload CSV, JSON, TXT, or TSV files containing knowledge data"
    )
    
    if uploaded_file is not None:
        # Domain selection
        domain = st.selectbox(
            "Select Domain",
            ["art", "language", "mathematics", "philosophy", "science", "technology", "shared"]
        )
        
        # Preview uploaded data
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Import button
            if st.button("📥 Import to Database", type="primary"):
                try:
                    # Save to appropriate domain folder
                    target_dir = project_root / "CSV" / domain
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_path = target_dir / uploaded_file.name
                    df.to_csv(target_path, index=False)
                    
                    st.success(f"✅ File imported to {domain} domain successfully!")
                    add_to_history("IMPORT", f"Imported {uploaded_file.name} to {domain}")
                
                except Exception as e:
                    st.error(f"Error importing file: {str(e)}")

def show_data_search(project_root):
    """Show data search interface"""
    st.markdown("## 🔍 Search Data")
    st.markdown("Search across all knowledge data files")
    
    # Search input
    search_term = st.text_input("🔍 Search Term", placeholder="Enter concept, name, or keyword...")
    
    if search_term:
        # Search options
        col1, col2 = st.columns(2)
        
        with col1:
            search_domains = st.multiselect(
                "Search in Domains",
                ["art", "language", "mathematics", "philosophy", "science", "technology", "shared"],
                default=["art", "language", "mathematics", "philosophy", "science", "technology", "shared"]
            )
        
        with col2:
            file_types = st.multiselect(
                "File Types",
                [".csv", ".json", ".txt"],
                default=[".csv"]
            )
        
        if st.button("🔍 Search", type="primary"):
            results = search_in_files(project_root, search_term, search_domains, file_types)
            
            if results:
                st.markdown("### 📋 Search Results")
                for result in results:
                    with st.expander(f"📄 {result['file']} - {result['matches']} matches"):
                        st.markdown(f"**Domain:** {result['domain']}")
                        st.markdown(f"**File:** {result['file']}")
                        
                        if result['preview']:
                            st.markdown("**Preview:**")
                            st.code(result['preview'])
                        
                        if st.button(f"📂 Open File", key=f"open_{result['file']}"):
                            st.session_state.browse_path = Path(result['path']).parent
                            st.session_state.selected_file = Path(result['path'])
                            st.rerun()
            else:
                st.info("No matches found")

def show_data_backup(project_root):
    """Show data backup interface"""
    st.markdown("## 💾 Data Backup")
    st.markdown("Backup and restore knowledge data")
    
    # Create backup section
    st.markdown("### 📦 Create Data Backup")
    
    backup_name = st.text_input("Backup Name", value=f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    domains_to_backup = st.multiselect(
        "Select Domains to Backup",
        ["art", "language", "mathematics", "philosophy", "science", "technology", "shared"],
        default=["art", "language", "mathematics", "philosophy", "science", "technology", "shared"]
    )
    
    if st.button("📦 Create Data Backup", type="primary"):
        create_data_backup(project_root, backup_name, domains_to_backup)

def search_in_files(project_root, search_term, domains, file_types):
    """Search for term in data files"""
    results = []
    csv_dir = project_root / "CSV"
    
    for domain in domains:
        domain_dir = csv_dir / domain
        if not domain_dir.exists():
            continue
            
        for file_type in file_types:
            for file_path in domain_dir.rglob(f"*{file_type}"):
                try:
                    if file_type == ".csv":
                        df = pd.read_csv(file_path)
                        matches = 0
                        preview_lines = []
                        
                        for idx, row in df.iterrows():
                            row_text = " ".join(str(val) for val in row.values)
                            if search_term.lower() in row_text.lower():
                                matches += 1
                                preview_lines.append(f"Row {idx}: {row_text[:100]}...")
                                if len(preview_lines) >= 3:
                                    break
                        
                        if matches > 0:
                            results.append({
                                'domain': domain,
                                'file': file_path.name,
                                'path': str(file_path),
                                'matches': matches,
                                'preview': "\n".join(preview_lines)
                            })
                    
                    else:  # text files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if search_term.lower() in content.lower():
                                lines = content.split('\n')
                                preview_lines = [line for line in lines if search_term.lower() in line.lower()][:3]
                                
                                results.append({
                                    'domain': domain,
                                    'file': file_path.name,
                                    'path': str(file_path),
                                    'matches': len(preview_lines),
                                    'preview': "\n".join(preview_lines)
                                })
                
                except Exception as e:
                    continue  # Skip files that can't be read
    
    return results

def create_data_backup(project_root, backup_name, domains):
    """Create backup of data files"""
    try:
        import tarfile
        
        backup_dir = project_root / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{backup_name}.tar.gz"
        
        with tarfile.open(backup_path, "w:gz") as tar:
            csv_dir = project_root / "CSV"
            
            for domain in domains:
                domain_dir = csv_dir / domain
                if domain_dir.exists():
                    tar.add(domain_dir, arcname=f"CSV/{domain}")
        
        st.success(f"✅ Data backup created: {backup_name}.tar.gz")
        add_to_history("BACKUP", f"Created data backup: {backup_name}")
    
    except Exception as e:
        st.error(f"Error creating data backup: {str(e)}")

def create_project_backup(project_root, backup_dir, backup_name, include_options):
    """Create a project backup"""
    try:
        import tarfile
        
        backup_path = backup_dir / f"{backup_name}.tar.gz"
        
        with tarfile.open(backup_path, "w:gz") as tar:
            if "CSV Files" in include_options:
                csv_dir = project_root / "CSV"
                if csv_dir.exists():
                    tar.add(csv_dir, arcname="CSV")
            
            if "Configuration Files" in include_options:
                config_files = [
                    project_root / ".env",
                    project_root / "docker-compose.yml",
                    project_root / "config",
                    project_root / "pyproject.toml"
                ]
                for config_file in config_files:
                    if config_file.exists():
                        tar.add(config_file, arcname=config_file.name)
            
            if "Agent Scripts" in include_options:
                agents_dir = project_root / "agents"
                if agents_dir.exists():
                    tar.add(agents_dir, arcname="agents")
            
            if "Documentation" in include_options:
                doc_files = [
                    project_root / "README.md",
                    project_root / "UIplan.md",
                    project_root / "plan.md",
                    project_root / "docs"
                ]
                for doc_file in doc_files:
                    if doc_file.exists():
                        tar.add(doc_file, arcname=doc_file.name)
        
        st.success(f"✅ Backup created: {backup_name}.tar.gz")
        add_to_history("BACKUP", f"Created project backup: {backup_name}")
    
    except Exception as e:
        st.error(f"Error creating backup: {str(e)}")

if __name__ == "__main__":
    main()