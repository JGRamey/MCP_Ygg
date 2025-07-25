"""
File Manager API Integration
Handles all API calls for file management operations
"""

from typing import Any, Dict, List, Optional

import asyncio
import streamlit as st
from streamlit_workspace.utils.api_client import api_client, run_async


class FileManagerAPI:
    """API integration layer for file management operations"""

    def __init__(self):
        self.domains = [
            "mathematics",
            "science",
            "philosophy",
            "religion",
            "art",
            "language",
            "general",
        ]

    @run_async
    async def get_csv_files(self, directory: str = "CSV") -> List[Dict]:
        """Get list of CSV files via API"""
        return await api_client.manage_files(
            operation="list", file_type="csv", data={"directory": directory}
        )

    @run_async
    async def upload_csv_file(self, file_data: bytes, filename: str) -> Dict:
        """Upload CSV file via API"""
        return await api_client.manage_files(
            operation="upload",
            file_type="csv",
            data={"file_data": file_data, "filename": filename},
        )

    @run_async
    async def delete_csv_file(self, filename: str) -> Dict:
        """Delete CSV file via API"""
        return await api_client.manage_files(
            operation="delete", file_type="csv", data={"filename": filename}
        )

    @run_async
    async def get_csv_content(self, filename: str) -> Dict:
        """Get CSV file content via API"""
        return await api_client.manage_files(
            operation="read", file_type="csv", data={"filename": filename}
        )

    @run_async
    async def get_neo4j_overview(self) -> Dict:
        """Get Neo4j database overview via API"""
        return await api_client.get_graph_data()

    @run_async
    async def get_neo4j_concepts(
        self, domain: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Get Neo4j concepts via API"""
        return await api_client.search_concepts(
            query="*" if not domain else f"domain:{domain}", domain=domain, limit=limit
        )

    @run_async
    async def create_concept(self, concept_data: Dict) -> Dict:
        """Create new concept via API"""
        return await api_client.manage_database(operation="create", data=concept_data)

    @run_async
    async def update_concept(self, concept_id: str, concept_data: Dict) -> Dict:
        """Update existing concept via API"""
        return await api_client.manage_database(
            operation="update", data={**concept_data, "id": concept_id}
        )

    @run_async
    async def delete_concept(self, concept_id: str) -> Dict:
        """Delete concept via API"""
        return await api_client.manage_database(
            operation="delete", data={"id": concept_id}
        )

    @run_async
    async def get_qdrant_collections(self) -> List[Dict]:
        """Get Qdrant collections via API"""
        # This would use a dedicated endpoint for Qdrant operations
        # For now, using a placeholder
        return await api_client._get("/api/qdrant/collections")

    @run_async
    async def get_qdrant_collection_info(self, collection_name: str) -> Dict:
        """Get Qdrant collection information via API"""
        return await api_client._get(f"/api/qdrant/collections/{collection_name}")

    @run_async
    async def search_vectors(
        self, collection_name: str, query: str, limit: int = 10
    ) -> List[Dict]:
        """Search vectors in Qdrant collection via API"""
        return await api_client._post(
            f"/api/qdrant/collections/{collection_name}/search",
            {"query": query, "limit": limit},
        )

    @run_async
    async def get_system_health(self) -> Dict:
        """Get system health status via API"""
        return await api_client.get_system_status()

    def show_api_status(self):
        """Display API connectivity status"""
        col1, col2 = st.columns([3, 1])

        with col1:
            if asyncio.run(api_client.health_check()):
                st.success("🟢 API Connected")
            else:
                st.error("🔴 API Disconnected - File operations unavailable")

        with col2:
            if st.button("🔄 Refresh", key="api_status_refresh"):
                st.rerun()

    def handle_api_error(self, operation: str, error: Exception = None):
        """Standardized API error handling"""
        if error:
            st.error(f"❌ {operation} failed: {str(error)}")
        else:
            st.error(f"❌ {operation} failed - please check API connection")

        st.info("💡 Troubleshooting tips:")
        st.markdown(
            """
        - Ensure the FastAPI server is running on port 8000
        - Check your network connection
        - Verify API endpoints are accessible
        """
        )

    def show_operation_progress(self, operation: str, items: List[Any]):
        """Show progress for multi-item operations"""
        if not items:
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, item in enumerate(items):
            status_text.text(f"{operation}: {item}")
            progress_bar.progress((i + 1) / len(items))

        status_text.text(f"✅ {operation} completed for {len(items)} items")
        progress_bar.empty()
        status_text.empty()
