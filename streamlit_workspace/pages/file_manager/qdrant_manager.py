"""
Qdrant Manager for File Manager
Handles Qdrant vector database operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .models import DatabaseFilters, QdrantCollectionInfo

try:
    from qdrant_client import QdrantClient

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantManager:
    """Manages Qdrant vector database operations."""

    def __init__(self):
        self.client = None
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(host="localhost", port=6333)
            except Exception as e:
                st.warning(f"Qdrant connection failed: {str(e)}")

    def is_connected(self) -> bool:
        """Check if Qdrant is connected."""
        if not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except:
            return False

    def get_collections(self) -> List[QdrantCollectionInfo]:
        """Get list of Qdrant collections."""
        if not self.is_connected():
            return []

        try:
            collections = self.client.get_collections().collections
            collection_infos = []

            for collection in collections:
                info = self.client.get_collection(collection.name)

                collection_info = QdrantCollectionInfo(
                    name=collection.name,
                    vectors_count=info.vectors_count or 0,
                    vector_size=info.config.params.vectors.size,
                    distance_metric=str(info.config.params.vectors.distance),
                    status=str(info.status),
                    indexed=info.status.value == "green",
                )

                collection_infos.append(collection_info)

            return collection_infos

        except Exception as e:
            st.error(f"Error getting collections: {str(e)}")
            return []

    def render_qdrant_interface(self, filters: DatabaseFilters):
        """Render the Qdrant database management interface."""
        st.markdown("## 🔍 Qdrant Vector Database")
        st.markdown("**Manage vector collections and semantic search**")

        if not QDRANT_AVAILABLE:
            st.error(
                "❌ qdrant-client not available. Install with: `pip install qdrant-client`"
            )
            return

        if not self.is_connected():
            st.error("❌ Qdrant database not connected")
            st.markdown("**Troubleshooting:**")
            st.markdown("- Ensure Qdrant is running on `localhost:6333`")
            st.markdown("- Start with: `docker-compose up qdrant`")
            return

        # Qdrant operations tabs
        qdrant_tabs = st.tabs(
            ["📊 Collections", "🔍 Search", "📈 Analytics", "⚙️ Management"]
        )

        with qdrant_tabs[0]:
            self._render_collections_tab()

        with qdrant_tabs[1]:
            self._render_search_tab()

        with qdrant_tabs[2]:
            self._render_analytics_tab()

        with qdrant_tabs[3]:
            self._render_management_tab()

    def _render_collections_tab(self):
        """Render collections overview tab."""
        st.subheader("📊 Vector Collections")

        collections = self.get_collections()

        if not collections:
            st.info("No collections found in Qdrant database.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Collections", len(collections))

        with col2:
            total_vectors = sum(c.vectors_count for c in collections)
            st.metric("Total Vectors", f"{total_vectors:,}")

        with col3:
            active_collections = sum(1 for c in collections if c.indexed)
            st.metric("Active Collections", active_collections)

        with col4:
            avg_size = (
                sum(c.vector_size for c in collections) / len(collections)
                if collections
                else 0
            )
            st.metric("Avg Vector Size", f"{avg_size:.0f}")

        # Collections table
        st.subheader("📋 Collection Details")

        collection_data = []
        for collection in collections:
            status_icon = "🟢" if collection.indexed else "🔴"
            collection_data.append(
                {
                    "Name": collection.name,
                    "Status": f"{status_icon} {collection.status}",
                    "Vectors": f"{collection.vectors_count:,}",
                    "Dimension": collection.vector_size,
                    "Distance": collection.distance_metric,
                    "Indexed": "✅" if collection.indexed else "⏳",
                }
            )

        df = pd.DataFrame(collection_data)

        # Display with selection
        selected_rows = st.dataframe(
            df, use_container_width=True, selection_mode="single-row"
        )

        # Collection operations
        if selected_rows and len(selected_rows.selection.rows) > 0:
            selected_idx = selected_rows.selection.rows[0]
            selected_collection = collections[selected_idx]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("📊 View Details"):
                    self._show_collection_details(selected_collection)

            with col2:
                if st.button("🔍 Search Vectors"):
                    self._search_collection(selected_collection)

            with col3:
                if st.button("📈 Show Analytics"):
                    self._show_collection_analytics(selected_collection)

            with col4:
                if st.button("🗑️ Delete Collection"):
                    self._delete_collection(selected_collection)

    def _render_search_tab(self):
        """Render semantic search tab."""
        st.subheader("🔍 Semantic Search")

        collections = self.get_collections()

        if not collections:
            st.info("No collections available for search.")
            return

        # Search configuration
        col1, col2 = st.columns(2)

        with col1:
            selected_collection = st.selectbox(
                "Select Collection",
                [c.name for c in collections],
                help="Choose collection to search in",
            )

        with col2:
            search_limit = st.number_input(
                "Result Limit", value=10, min_value=1, max_value=100
            )

        # Search input
        search_query = st.text_area(
            "Search Query",
            height=100,
            placeholder="Enter text to find semantically similar vectors...",
            help="This would be converted to a vector and used for similarity search",
        )

        search_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.01)

        if st.button("🔍 Search Vectors") and search_query:
            self._perform_semantic_search(
                selected_collection, search_query, search_limit, search_threshold
            )

    def _render_analytics_tab(self):
        """Render analytics tab."""
        st.subheader("📈 Vector Database Analytics")

        collections = self.get_collections()

        if not collections:
            st.info("No collections available for analytics.")
            return

        # Collection size distribution
        st.subheader("📊 Collection Size Distribution")

        size_data = pd.DataFrame(
            [
                {
                    "Collection": c.name,
                    "Vectors": c.vectors_count,
                    "Dimension": c.vector_size,
                }
                for c in collections
            ]
        )

        # Vector count chart
        st.bar_chart(size_data.set_index("Collection")["Vectors"])

        # Dimension analysis
        st.subheader("📐 Vector Dimensions")
        dimension_counts = size_data["Dimension"].value_counts()

        if len(dimension_counts) > 1:
            st.bar_chart(dimension_counts)
        else:
            st.info(
                f"All collections use {dimension_counts.index[0]}-dimensional vectors."
            )

        # Storage analysis
        st.subheader("💾 Storage Analysis")

        total_storage = 0
        storage_data = []

        for collection in collections:
            # Estimate storage (vectors * dimension * 4 bytes per float)
            estimated_mb = (collection.vectors_count * collection.vector_size * 4) / (
                1024 * 1024
            )
            total_storage += estimated_mb

            storage_data.append(
                {
                    "Collection": collection.name,
                    "Estimated Size (MB)": f"{estimated_mb:.1f}",
                }
            )

        storage_df = pd.DataFrame(storage_data)
        st.dataframe(storage_df, use_container_width=True)

        st.metric("Total Estimated Storage", f"{total_storage:.1f} MB")

    def _render_management_tab(self):
        """Render collection management tab."""
        st.subheader("⚙️ Collection Management")

        # Create new collection
        with st.expander("➕ Create New Collection", expanded=False):
            self._render_create_collection_form()

        # Collection operations
        with st.expander("🔧 Collection Operations", expanded=False):
            self._render_collection_operations()

        # Backup and restore
        with st.expander("💾 Backup & Restore", expanded=False):
            self._render_backup_operations()

    def _show_collection_details(self, collection: QdrantCollectionInfo):
        """Show detailed information about a collection."""
        st.subheader(f"📊 Collection: {collection.name}")

        # Basic information
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Vector Count", f"{collection.vectors_count:,}")
            st.metric("Vector Dimension", collection.vector_size)

        with col2:
            st.metric("Distance Metric", collection.distance_metric)
            st.metric("Status", collection.status)

        # Try to get sample vectors
        try:
            # Get a small sample of vectors
            response = self.client.scroll(
                collection_name=collection.name,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )

            if response[0]:  # Has records
                st.subheader("📄 Sample Records")

                sample_data = []
                for point in response[0]:
                    sample_data.append(
                        {
                            "ID": str(point.id),
                            "Payload Keys": (
                                ", ".join(point.payload.keys())
                                if point.payload
                                else "None"
                            ),
                        }
                    )

                if sample_data:
                    sample_df = pd.DataFrame(sample_data)
                    st.dataframe(sample_df, use_container_width=True)
            else:
                st.info("Collection is empty.")

        except Exception as e:
            st.warning(f"Could not retrieve sample data: {str(e)}")

    def _search_collection(self, collection: QdrantCollectionInfo):
        """Search within a specific collection."""
        st.subheader(f"🔍 Search in {collection.name}")
        st.info(
            "Semantic search functionality would be implemented here with actual vector embeddings."
        )

    def _show_collection_analytics(self, collection: QdrantCollectionInfo):
        """Show analytics for a specific collection."""
        st.subheader(f"📈 Analytics: {collection.name}")

        # Basic metrics
        estimated_size_mb = (collection.vectors_count * collection.vector_size * 4) / (
            1024 * 1024
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Vectors", f"{collection.vectors_count:,}")

        with col2:
            st.metric("Dimension", collection.vector_size)

        with col3:
            st.metric("Est. Size", f"{estimated_size_mb:.1f} MB")

        st.info("Detailed analytics would be implemented here.")

    def _delete_collection(self, collection: QdrantCollectionInfo):
        """Delete a collection with confirmation."""
        st.subheader(f"🗑️ Delete Collection: {collection.name}")
        st.warning("⚠️ This will permanently delete all vectors in this collection!")

        if st.checkbox(f"I confirm I want to delete '{collection.name}'"):
            if st.button("🗑️ Delete Collection", type="secondary"):
                try:
                    self.client.delete_collection(collection.name)
                    st.success(f"✅ Deleted collection '{collection.name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting collection: {str(e)}")

    def _perform_semantic_search(
        self, collection_name: str, query: str, limit: int, threshold: float
    ):
        """Perform semantic search in collection."""
        st.subheader("🔍 Search Results")

        # This would require actual vector embedding of the query
        st.info("Semantic search would require:")
        st.markdown("1. Convert query text to vector using embedding model")
        st.markdown("2. Perform similarity search in Qdrant")
        st.markdown("3. Return ranked results with similarity scores")

        # Mock results for demonstration
        mock_results = pd.DataFrame(
            {
                "Score": [0.95, 0.87, 0.82, 0.78],
                "Content": [
                    "Sample result 1 with high similarity",
                    "Another relevant result",
                    "Third matching result",
                    "Fourth result with lower similarity",
                ],
                "Metadata": [
                    "Type: Article",
                    "Type: Book",
                    "Type: Paper",
                    "Type: Note",
                ],
            }
        )

        st.dataframe(mock_results, use_container_width=True)

    def _render_create_collection_form(self):
        """Render form to create new collection."""
        st.write("### Create New Vector Collection")

        col1, col2 = st.columns(2)

        with col1:
            collection_name = st.text_input(
                "Collection Name", placeholder="documents_philosophy"
            )
            vector_size = st.number_input(
                "Vector Dimension", value=384, min_value=1, max_value=4096
            )

        with col2:
            distance_metric = st.selectbox(
                "Distance Metric",
                ["Cosine", "Euclidean", "Dot"],
                help="Similarity measurement method",
            )
            hnsw_ef = st.number_input("HNSW EF", value=128, min_value=16, max_value=512)

        if st.button("➕ Create Collection") and collection_name:
            try:
                from qdrant_client.models import Distance, HnswConfig, VectorParams

                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclidean": Distance.EUCLID,
                    "Dot": Distance.DOT,
                }

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_map[distance_metric],
                        hnsw_config=HnswConfig(ef_construct=hnsw_ef),
                    ),
                )

                st.success(f"✅ Created collection '{collection_name}'")
                st.rerun()

            except Exception as e:
                st.error(f"Error creating collection: {str(e)}")

    def _render_collection_operations(self):
        """Render collection operations."""
        st.write("### Collection Operations")

        collections = self.get_collections()

        if collections:
            selected_collection = st.selectbox(
                "Select Collection for Operations", [c.name for c in collections]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔄 Optimize Collection"):
                    st.info(f"Optimizing {selected_collection}...")

            with col2:
                if st.button("📊 Rebuild Index"):
                    st.info(f"Rebuilding index for {selected_collection}...")

            with col3:
                if st.button("🧹 Clean Collection"):
                    st.info(f"Cleaning {selected_collection}...")
        else:
            st.info("No collections available for operations.")

    def _render_backup_operations(self):
        """Render backup and restore operations."""
        st.write("### Backup & Restore")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Backup Collections**")
            if st.button("💾 Create Backup"):
                st.info("Backup functionality would be implemented here.")

        with col2:
            st.write("**Restore Collections**")
            backup_file = st.file_uploader("Select backup file", type=["json", "zip"])
            if backup_file and st.button("📥 Restore Backup"):
                st.info("Restore functionality would be implemented here.")
