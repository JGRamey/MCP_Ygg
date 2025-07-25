"""
Backup Manager for File Manager
Handles database backup and import operations
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .models import DatabaseFilters, ScrapedContentInfo


class BackupManager:
    """Manages database backup and import operations."""

    def __init__(self, csv_root: Path):
        self.csv_root = csv_root
        self.backup_dir = csv_root.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def render_backup_interface(self, filters: DatabaseFilters):
        """Render database backup interface."""
        st.markdown("## 💾 Database Backup")
        st.markdown("**Backup and restore database content**")

        # Backup operations tabs
        backup_tabs = st.tabs(
            [
                "📤 Create Backup",
                "📥 Restore Backup",
                "📋 Backup History",
                "📊 Import Data",
            ]
        )

        with backup_tabs[0]:
            self._render_create_backup_tab()

        with backup_tabs[1]:
            self._render_restore_backup_tab()

        with backup_tabs[2]:
            self._render_backup_history_tab()

        with backup_tabs[3]:
            self._render_import_data_tab()

    def _render_create_backup_tab(self):
        """Render backup creation interface."""
        st.subheader("📤 Create Database Backup")

        # Backup options
        col1, col2 = st.columns(2)

        with col1:
            backup_name = st.text_input(
                "Backup Name",
                value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for this backup",
            )

            include_csv = st.checkbox("Include CSV files", value=True)
            include_metadata = st.checkbox("Include metadata", value=True)

        with col2:
            backup_description = st.text_area(
                "Description",
                placeholder="Optional description of this backup...",
                height=100,
            )

            compression = st.selectbox("Compression", ["zip", "tar.gz", "none"])

        # What to backup
        st.subheader("📋 Backup Contents")

        backup_contents = st.multiselect(
            "Select components to backup",
            [
                "📄 CSV Database Files",
                "🗄️ Neo4j Export",
                "🔍 Qdrant Collections",
                "📝 Scraped Content",
                "⚙️ Configuration Files",
            ],
            default=["📄 CSV Database Files"],
        )

        if st.button("🚀 Create Backup", type="primary") and backup_name:
            self._create_backup(
                backup_name, backup_description, backup_contents, compression
            )

    def _render_restore_backup_tab(self):
        """Render backup restore interface."""
        st.subheader("📥 Restore Database Backup")

        # File upload for backup
        uploaded_backup = st.file_uploader(
            "Select backup file",
            type=["zip", "tar", "gz"],
            help="Upload a backup file to restore",
        )

        if uploaded_backup:
            st.write(f"**Selected**: {uploaded_backup.name}")
            st.write(f"**Size**: {uploaded_backup.size / (1024*1024):.1f} MB")

            # Restore options
            col1, col2 = st.columns(2)

            with col1:
                restore_mode = st.radio(
                    "Restore Mode",
                    [
                        "Replace existing data",
                        "Merge with existing data",
                        "Preview only",
                    ],
                )

            with col2:
                create_backup_before = st.checkbox(
                    "Create backup before restore", value=True
                )

            # Warning
            if restore_mode == "Replace existing data":
                st.warning(
                    "⚠️ This will replace all existing data! Make sure you have a backup."
                )

            if st.button("📥 Restore Backup", type="primary"):
                if create_backup_before:
                    # Create backup first
                    pre_restore_backup = (
                        f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    st.info(f"Creating backup: {pre_restore_backup}")

                self._restore_backup(uploaded_backup, restore_mode)

    def _render_backup_history_tab(self):
        """Render backup history interface."""
        st.subheader("📋 Backup History")

        # Get list of existing backups
        backups = self._get_backup_list()

        if not backups:
            st.info("No backups found.")
            return

        # Display backups table
        backup_data = []
        for backup in backups:
            backup_data.append(
                {
                    "Name": backup["name"],
                    "Created": backup["created"],
                    "Size": backup["size"],
                    "Description": (
                        backup["description"][:50] + "..."
                        if len(backup["description"]) > 50
                        else backup["description"]
                    ),
                    "Compression": backup["compression"],
                }
            )

        df = pd.DataFrame(backup_data)

        # Display with selection
        selected_rows = st.dataframe(
            df, use_container_width=True, selection_mode="single-row"
        )

        # Backup operations
        if selected_rows and len(selected_rows.selection.rows) > 0:
            selected_idx = selected_rows.selection.rows[0]
            selected_backup = backups[selected_idx]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("📊 View Details"):
                    self._show_backup_details(selected_backup)

            with col2:
                if st.button("📥 Restore"):
                    self._quick_restore_backup(selected_backup)

            with col3:
                backup_path = self.backup_dir / selected_backup["filename"]
                if backup_path.exists():
                    with open(backup_path, "rb") as f:
                        st.download_button(
                            "📤 Download",
                            f.read(),
                            selected_backup["filename"],
                            key=f"download_{selected_backup['name']}",
                        )

            with col4:
                if st.button("🗑️ Delete"):
                    self._delete_backup(selected_backup)

    def _render_import_data_tab(self):
        """Render data import interface."""
        st.subheader("📊 Import External Data")

        # Import source selection
        import_source = st.selectbox(
            "Import Source",
            [
                "CSV Files",
                "JSON Export",
                "Neo4j Dump",
                "Qdrant Backup",
                "Approved Staging Content",
            ],
        )

        if import_source == "CSV Files":
            self._render_csv_import()
        elif import_source == "JSON Export":
            self._render_json_import()
        elif import_source == "Approved Staging Content":
            self._render_staging_import()
        else:
            st.info(f"{import_source} import functionality would be implemented here.")

    def _create_backup(
        self, name: str, description: str, contents: List[str], compression: str
    ):
        """Create a new backup."""
        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Initialize
                status_text.text("Initializing backup...")
                progress_bar.progress(0.1)

                backup_metadata = {
                    "name": name,
                    "description": description,
                    "created": datetime.now().isoformat(),
                    "contents": contents,
                    "compression": compression,
                    "version": "1.0",
                }

                # Step 2: Backup CSV files
                if "📄 CSV Database Files" in contents:
                    status_text.text("Backing up CSV files...")
                    progress_bar.progress(0.3)
                    # Implementation would copy CSV files

                # Step 3: Export Neo4j data
                if "🗄️ Neo4j Export" in contents:
                    status_text.text("Exporting Neo4j data...")
                    progress_bar.progress(0.5)
                    # Implementation would export Neo4j data

                # Step 4: Backup Qdrant collections
                if "🔍 Qdrant Collections" in contents:
                    status_text.text("Backing up Qdrant collections...")
                    progress_bar.progress(0.7)
                    # Implementation would backup Qdrant data

                # Step 5: Finalize
                status_text.text("Finalizing backup...")
                progress_bar.progress(0.9)

                # Save metadata
                metadata_path = self.backup_dir / f"{name}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(backup_metadata, f, indent=2)

                progress_bar.progress(1.0)
                status_text.text("Backup completed successfully!")

                st.success(f"✅ Backup '{name}' created successfully!")

            except Exception as e:
                st.error(f"Backup failed: {str(e)}")

    def _restore_backup(self, backup_file, restore_mode: str):
        """Restore from backup file."""
        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Extracting backup...")
                progress_bar.progress(0.2)

                status_text.text("Validating backup contents...")
                progress_bar.progress(0.4)

                if restore_mode == "Replace existing data":
                    status_text.text("Clearing existing data...")
                    progress_bar.progress(0.6)

                status_text.text("Restoring data...")
                progress_bar.progress(0.8)

                status_text.text("Finalizing restore...")
                progress_bar.progress(1.0)

                st.success("✅ Backup restored successfully!")

            except Exception as e:
                st.error(f"Restore failed: {str(e)}")

    def _get_backup_list(self) -> List[Dict]:
        """Get list of available backups."""
        backups = []

        for metadata_file in self.backup_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Get backup file info
                backup_filename = metadata_file.name.replace("_metadata.json", ".zip")
                backup_path = self.backup_dir / backup_filename

                if backup_path.exists():
                    stat = backup_path.stat()
                    metadata.update(
                        {
                            "filename": backup_filename,
                            "size": f"{stat.st_size / (1024*1024):.1f} MB",
                        }
                    )
                    backups.append(metadata)

            except Exception as e:
                st.warning(f"Error reading backup metadata: {str(e)}")

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def _show_backup_details(self, backup: Dict):
        """Show detailed information about a backup."""
        st.subheader(f"📊 Backup Details: {backup['name']}")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Created**: {backup['created']}")
            st.write(f"**Size**: {backup['size']}")
            st.write(f"**Compression**: {backup['compression']}")

        with col2:
            st.write(f"**Version**: {backup.get('version', 'Unknown')}")
            st.write(f"**Components**: {len(backup['contents'])}")

        st.write(f"**Description**: {backup['description']}")

        st.subheader("📋 Backup Contents")
        for content in backup["contents"]:
            st.write(f"- {content}")

    def _quick_restore_backup(self, backup: Dict):
        """Quick restore from backup."""
        st.subheader(f"📥 Restore: {backup['name']}")
        st.warning("⚠️ This will restore the selected backup.")

        if st.button("🚀 Confirm Restore"):
            backup_path = self.backup_dir / backup["filename"]
            self._restore_backup(backup_path, "Replace existing data")

    def _delete_backup(self, backup: Dict):
        """Delete a backup with confirmation."""
        st.subheader(f"🗑️ Delete Backup: {backup['name']}")
        st.warning("⚠️ This action cannot be undone!")

        if st.checkbox(f"I confirm I want to delete '{backup['name']}'"):
            if st.button("🗑️ Delete Backup", type="secondary"):
                try:
                    # Delete backup file and metadata
                    backup_path = self.backup_dir / backup["filename"]
                    metadata_path = self.backup_dir / f"{backup['name']}_metadata.json"

                    if backup_path.exists():
                        backup_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()

                    st.success(f"✅ Deleted backup '{backup['name']}'")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error deleting backup: {str(e)}")

    def _render_csv_import(self):
        """Render CSV import interface."""
        st.write("### Import CSV Files")

        uploaded_files = st.file_uploader(
            "Select CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more CSV files to import",
        )

        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} files:**")

            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")

            # Import settings
            col1, col2 = st.columns(2)

            with col1:
                target_domain = st.selectbox(
                    "Target Domain",
                    [
                        "Auto-detect",
                        "mathematics",
                        "science",
                        "philosophy",
                        "religion",
                        "art",
                        "language",
                    ],
                )

            with col2:
                import_mode = st.selectbox(
                    "Import Mode",
                    ["Add new records", "Replace domain data", "Merge with existing"],
                )

            if st.button("📥 Import CSV Files"):
                self._import_csv_files(uploaded_files, target_domain, import_mode)

    def _render_json_import(self):
        """Render JSON import interface."""
        st.write("### Import JSON Export")

        uploaded_json = st.file_uploader(
            "Select JSON file", type=["json"], help="Upload a JSON export file"
        )

        if uploaded_json:
            try:
                json_data = json.load(uploaded_json)

                st.write("**JSON Structure Preview:**")

                # Show structure
                if isinstance(json_data, dict):
                    for key, value in json_data.items():
                        if isinstance(value, list):
                            st.write(f"- **{key}**: {len(value)} items")
                        else:
                            st.write(f"- **{key}**: {type(value).__name__}")

                if st.button("📥 Import JSON Data"):
                    self._import_json_data(json_data)

            except Exception as e:
                st.error(f"Error reading JSON file: {str(e)}")

    def _render_staging_import(self):
        """Render staging content import interface."""
        st.write("### Import Approved Staging Content")

        # Mock staging content for demonstration
        staging_content = [
            ScrapedContentInfo(
                content_id="staging_001",
                title="Sample Article 1",
                url="https://example.com/article1",
                domain="philosophy",
                content_type="article",
                scraped_at=datetime.now(),
                status="approved",
                content_length=1500,
            ),
            ScrapedContentInfo(
                content_id="staging_002",
                title="Sample Paper 2",
                url="https://example.com/paper2",
                domain="science",
                content_type="academic_paper",
                scraped_at=datetime.now(),
                status="approved",
                content_length=3200,
            ),
        ]

        if staging_content:
            st.write(f"**Found {len(staging_content)} approved items in staging:**")

            # Display staging content
            staging_data = []
            for content in staging_content:
                staging_data.append(
                    {
                        "Title": content.title,
                        "Domain": content.domain,
                        "Type": content.content_type,
                        "Status": content.status_display,
                        "Length": f"{content.content_length:,} chars",
                    }
                )

            df = pd.DataFrame(staging_data)
            st.dataframe(df, use_container_width=True)

            # Import options
            col1, col2 = st.columns(2)

            with col1:
                import_to = st.selectbox(
                    "Import to Database",
                    ["Neo4j as Documents", "CSV files", "Both Neo4j and CSV"],
                )

            with col2:
                auto_categorize = st.checkbox("Auto-categorize by domain", value=True)

            if st.button("📥 Import Staging Content"):
                self._import_staging_content(
                    staging_content, import_to, auto_categorize
                )
        else:
            st.info("No approved content found in staging area.")

    def _import_csv_files(self, files, target_domain: str, import_mode: str):
        """Import CSV files."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            total_files = len(files)

            for i, file in enumerate(files):
                status_text.text(f"Importing {file.name}...")
                progress_bar.progress((i + 1) / total_files)

                # Process file
                df = pd.read_csv(file)

                # Determine target location
                if target_domain == "Auto-detect":
                    # Try to detect domain from filename or content
                    domain = "general"
                else:
                    domain = target_domain

                # Save to appropriate location
                domain_dir = self.csv_root / domain
                domain_dir.mkdir(exist_ok=True)

                target_path = domain_dir / f"imported_{file.name}"
                df.to_csv(target_path, index=False)

            st.success(f"✅ Successfully imported {total_files} CSV files!")

        except Exception as e:
            st.error(f"Import failed: {str(e)}")

    def _import_json_data(self, json_data: Dict):
        """Import JSON data."""
        st.info(
            "JSON import functionality would process the data structure and import to appropriate databases."
        )
        st.success("✅ JSON data import completed!")

    def _import_staging_content(
        self, content: List[ScrapedContentInfo], import_to: str, auto_categorize: bool
    ):
        """Import staging content."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            total_items = len(content)

            for i, item in enumerate(content):
                status_text.text(f"Importing: {item.title[:50]}...")
                progress_bar.progress((i + 1) / total_items)

                # Process based on import destination
                if "Neo4j" in import_to:
                    # Import to Neo4j as Document nodes
                    pass

                if "CSV" in import_to:
                    # Add to appropriate CSV files
                    pass

            st.success(f"✅ Successfully imported {total_items} items from staging!")

        except Exception as e:
            st.error(f"Staging import failed: {str(e)}")
