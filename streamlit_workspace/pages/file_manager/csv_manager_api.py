"""
CSV Manager - API-based CSV file management
Handles CSV operations through FastAPI backend
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .api_integration import FileManagerAPI
from .ui_components import (
    render_action_buttons,
    render_confirmation_dialog,
    render_data_preview,
    render_data_table,
    render_file_upload_area,
    render_metrics_row,
)


class CSVManager:
    """Manages CSV database files via API."""

    def __init__(self, api: FileManagerAPI):
        self.api = api
        self.domains = [
            "mathematics",
            "science",
            "philosophy",
            "religion",
            "art",
            "language",
            "general",
        ]

    def render(self):
        """Main render method for CSV management interface"""
        st.subheader("📊 CSV File Management")

        # CSV files overview
        self._render_csv_overview()

        # File operations tabs
        tab1, tab2, tab3 = st.tabs(["📂 Browse Files", "📤 Upload", "🗑️ Manage"])

        with tab1:
            self._render_file_browser()

        with tab2:
            self._render_file_upload()

        with tab3:
            self._render_file_management()

    def _render_csv_overview(self):
        """Display CSV files overview metrics"""
        try:
            files = self.api.get_csv_files()

            if files:
                # Calculate metrics
                total_files = len(files)
                domain_counts = {}
                total_size = 0

                for file_info in files:
                    domain = file_info.get("domain", "unknown")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    total_size += file_info.get("size", 0)

                # Display metrics
                render_metrics_row(
                    [
                        ("Total Files", total_files),
                        ("Domains", len(domain_counts)),
                        ("Total Size", f"{total_size / (1024*1024):.1f} MB"),
                    ]
                )

                # Domain distribution
                if domain_counts:
                    st.subheader("📈 Files by Domain")
                    domain_df = pd.DataFrame(
                        list(domain_counts.items()), columns=["Domain", "Count"]
                    )
                    st.bar_chart(domain_df.set_index("Domain"))
            else:
                st.info("No CSV files found")

        except Exception as e:
            self.api.handle_api_error("CSV Overview", e)

    def _render_file_browser(self):
        """Render CSV file browser interface"""
        st.subheader("📂 Browse CSV Files")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            domain_filter = st.selectbox(
                "Domain Filter", ["All"] + self.domains, key="csv_domain_filter"
            )

        with col2:
            sort_by = st.selectbox(
                "Sort by", ["Name", "Size", "Modified", "Domain"], key="csv_sort_by"
            )

        with col3:
            if st.button("🔄 Refresh Files", key="refresh_csv_files"):
                self._load_csv_files(domain_filter, sort_by)

        # Load files on first render
        if "csv_files" not in st.session_state:
            self._load_csv_files(domain_filter, sort_by)

        # Display files
        if "csv_files" in st.session_state and st.session_state.csv_files:
            files_df = pd.DataFrame(st.session_state.csv_files)

            # Add selection column
            files_df["Select"] = False

            # Display table
            edited_df = st.data_editor(
                files_df,
                use_container_width=True,
                hide_index=True,
                key="csv_files_table",
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select"),
                    "Size": st.column_config.NumberColumn("Size (KB)", format="%.1f"),
                    "Modified": st.column_config.DatetimeColumn("Modified"),
                },
            )

            # Handle file selection
            selected_files = edited_df[edited_df["Select"]]["Name"].tolist()

            if selected_files:
                st.subheader(f"📋 Selected Files ({len(selected_files)})")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("👁️ View Content"):
                        self._view_selected_files(selected_files)

                with col2:
                    if st.button("📥 Download"):
                        self._download_selected_files(selected_files)

                with col3:
                    if st.button("🗑️ Delete", type="secondary"):
                        self._delete_selected_files(selected_files)
        else:
            st.info("No CSV files available")

    def _render_file_upload(self):
        """Render file upload interface"""
        st.subheader("📤 Upload CSV Files")

        # Upload area
        uploaded_file = render_file_upload_area(
            accepted_types=["csv"], max_file_size_mb=50, key="csv_upload"
        )

        if uploaded_file:
            # File metadata form
            with st.form("csv_upload_form"):
                col1, col2 = st.columns(2)

                with col1:
                    domain = st.selectbox("Domain*", self.domains, key="upload_domain")

                    file_type = st.selectbox(
                        "File Type",
                        ["concepts", "relationships", "entities", "works", "events"],
                        key="upload_type",
                    )

                with col2:
                    description = st.text_area("Description", key="upload_description")

                    tags = st.text_input("Tags (comma-separated)", key="upload_tags")

                # Data preview
                try:
                    df = pd.read_csv(uploaded_file)
                    render_data_preview(df, "File Preview", max_rows=5)

                    st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                    st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    df = None

                # Upload button
                submitted = st.form_submit_button("📤 Upload File")

                if submitted and df is not None:
                    self._upload_csv_file(
                        uploaded_file, domain, file_type, description, tags
                    )

    def _render_file_management(self):
        """Render file management operations"""
        st.subheader("🗑️ File Management")

        # Bulk operations
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cleanup Operations**")

            if st.button("🧹 Clean Empty Files"):
                self._cleanup_empty_files()

            if st.button("🔄 Rebuild Index"):
                self._rebuild_file_index()

        with col2:
            st.write("**Maintenance**")

            if st.button("📊 Validate All Files"):
                self._validate_all_files()

            if st.button("📈 Generate Report"):
                self._generate_files_report()

        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("Dangerous operations - use with caution!")

            if render_confirmation_dialog(
                "🗑️ Delete All Files",
                "This will permanently delete ALL CSV files. Are you sure?",
                key="delete_all_confirm",
            ):
                self._delete_all_files()

    def _load_csv_files(self, domain_filter: str, sort_by: str):
        """Load CSV files from API"""
        try:
            directory = "CSV"
            if domain_filter != "All":
                directory = f"CSV/{domain_filter.lower()}"

            files = self.api.get_csv_files(directory)

            if files:
                # Process and sort files
                processed_files = []
                for file_info in files:
                    processed_files.append(
                        {
                            "Name": file_info.get("name", "Unknown"),
                            "Domain": file_info.get("domain", "Unknown"),
                            "Size": file_info.get("size", 0) / 1024,  # Convert to KB
                            "Modified": pd.to_datetime(file_info.get("modified", "")),
                            "Type": file_info.get("type", "CSV"),
                        }
                    )

                # Sort files
                if sort_by == "Name":
                    processed_files.sort(key=lambda x: x["Name"])
                elif sort_by == "Size":
                    processed_files.sort(key=lambda x: x["Size"], reverse=True)
                elif sort_by == "Modified":
                    processed_files.sort(key=lambda x: x["Modified"], reverse=True)
                elif sort_by == "Domain":
                    processed_files.sort(key=lambda x: x["Domain"])

                st.session_state.csv_files = processed_files
                st.success(f"✅ Loaded {len(processed_files)} CSV files")
            else:
                st.session_state.csv_files = []
                st.info("No CSV files found")

        except Exception as e:
            self.api.handle_api_error("Load CSV Files", e)
            st.session_state.csv_files = []

    def _upload_csv_file(
        self, uploaded_file, domain: str, file_type: str, description: str, tags: str
    ):
        """Upload CSV file via API"""
        try:
            # Read file data
            file_data = uploaded_file.read()

            # Upload via API
            result = self.api.upload_csv_file(file_data, uploaded_file.name)

            if result and result.get("success"):
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

                # Clear the cached files to force refresh
                if "csv_files" in st.session_state:
                    del st.session_state.csv_files

                # Store upload metadata if API supports it
                # This would typically be handled by the backend

            else:
                st.error("❌ Failed to upload file")

        except Exception as e:
            self.api.handle_api_error("Upload CSV File", e)

    def _view_selected_files(self, selected_files: List[str]):
        """View content of selected files"""
        for filename in selected_files:
            try:
                content = self.api.get_csv_content(filename)

                if content:
                    st.subheader(f"📄 {filename}")

                    # Convert to DataFrame if possible
                    if "data" in content:
                        df = pd.DataFrame(content["data"])
                        render_data_preview(df, f"Content Preview - {filename}")
                    else:
                        st.json(content)
                else:
                    st.warning(f"Unable to load content for {filename}")

            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")

    def _download_selected_files(self, selected_files: List[str]):
        """Download selected files"""
        st.info("Download functionality - implementation pending")
        # Would implement file download via API

    def _delete_selected_files(self, selected_files: List[str]):
        """Delete selected files"""
        if render_confirmation_dialog(
            "🗑️ Delete Files",
            f"Delete {len(selected_files)} selected files?",
            key="delete_selected_confirm",
        ):
            for filename in selected_files:
                try:
                    result = self.api.delete_csv_file(filename)
                    if result:
                        st.success(f"✅ Deleted {filename}")
                    else:
                        st.error(f"❌ Failed to delete {filename}")
                except Exception as e:
                    st.error(f"Error deleting {filename}: {str(e)}")

            # Refresh file list
            if "csv_files" in st.session_state:
                del st.session_state.csv_files

    def _cleanup_empty_files(self):
        """Clean up empty CSV files"""
        st.info("Cleanup operation - implementation pending")

    def _rebuild_file_index(self):
        """Rebuild file index"""
        st.info("Index rebuild - implementation pending")

    def _validate_all_files(self):
        """Validate all CSV files"""
        st.info("File validation - implementation pending")

    def _generate_files_report(self):
        """Generate files report"""
        st.info("Report generation - implementation pending")

    def _delete_all_files(self):
        """Delete all CSV files"""
        st.error("Bulk delete operation - implementation pending")
