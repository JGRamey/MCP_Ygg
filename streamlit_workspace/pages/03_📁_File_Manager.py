"""
File Manager - Main orchestrator for database file management
Follows Phase 5.5c specification exactly
"""

import streamlit as st
from streamlit_workspace.pages.file_manager.api_integration import FileManagerAPI
from streamlit_workspace.pages.file_manager.csv_manager_api import CSVManager
from streamlit_workspace.pages.file_manager.database_browser import DatabaseBrowser

st.set_page_config(page_title="Database File Manager", page_icon="📁", layout="wide")


class FileManagerUI:
    """Main file manager interface"""

    def __init__(self):
        self.api = FileManagerAPI()
        self.csv_manager = CSVManager(self.api)
        self.db_browser = DatabaseBrowser(self.api)

    def render(self):
        """Main render method"""
        st.title("📁 Database File Manager")
        st.markdown("Manage CSV files and database content")

        # Show API status
        self.api.show_api_status()

        # Tab selection
        tab1, tab2, tab3 = st.tabs(
            ["📊 CSV Files", "🗄️ Neo4j Browser", "🔍 Qdrant Collections"]
        )

        with tab1:
            self.csv_manager.render()

        with tab2:
            self.db_browser.render_neo4j()

        with tab3:
            self.db_browser.render_qdrant()


# Initialize and render
file_manager = FileManagerUI()
file_manager.render()
