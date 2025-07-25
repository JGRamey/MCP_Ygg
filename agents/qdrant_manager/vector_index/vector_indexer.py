#!/usr/bin/env python3
"""
MCP Server Vector Indexing Agent
Stores embeddings in Qdrant with metadata and Neo4j integration
"""

import hashlib
import json
import logging
import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncio
import numpy as np
import redis
import yaml
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document representation for vector storage"""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]

    def to_qdrant_point(self) -> rest.PointStruct:
        """Convert to Qdrant point structure"""
        return rest.PointStruct(id=self.id, vector=self.vector, payload=self.metadata)


@dataclass
class SearchResult:
    """Search result from vector database"""

    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionConfig:
    """Configuration for Qdrant collection"""

    name: str
    vector_size: int
    distance: str = "Cosine"
    hnsw_config: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,
            }

        if self.quantization_config is None:
            self.quantization_config = {
                "scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}
            }


class QdrantManager:
    """Manages Qdrant vector database operations"""

    def __init__(
        self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None
    ):
        """Initialize Qdrant client"""
        self.host = host
        self.port = port
        self.api_key = api_key
        self.client = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to Qdrant"""
        try:
            self.client = QdrantClient(
                host=self.host, port=self.port, api_key=self.api_key, timeout=60
            )

            # Test connection
            collections = self.client.get_collections()
            logger.info(
                f"Connected to Qdrant successfully. Collections: {len(collections.collections)}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new collection"""
        try:
            collection_config = rest.CreateCollection(
                vectors_config=rest.VectorParams(
                    size=config.vector_size, distance=config.distance
                ),
                hnsw_config=rest.HnswConfigDiff(**config.hnsw_config),
                quantization_config=rest.QuantizationConfig(
                    **config.quantization_config
                ),
            )

            result = self.client.create_collection(
                collection_name=config.name,
                vectors_config=collection_config.vectors_config,
                hnsw_config=collection_config.hnsw_config,
                quantization_config=collection_config.quantization_config,
            )

            logger.info(f"Created collection: {config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {config.name}: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "config": {
                    "distance": info.config.params.vectors.distance,
                    "vector_size": info.config.params.vectors.size,
                },
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def upsert_points(
        self, collection_name: str, points: List[VectorDocument], batch_size: int = 100
    ) -> bool:
        """Insert or update points in collection"""
        try:
            # Convert to Qdrant points
            qdrant_points = [doc.to_qdrant_point() for doc in points]

            # Batch upload
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i : i + batch_size]

                self.client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )

                logger.debug(
                    f"Uploaded batch {i//batch_size + 1}/{(len(qdrant_points)-1)//batch_size + 1}"
                )

            logger.info(
                f"Successfully uploaded {len(points)} points to {collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            search_request = rest.SearchRequest(
                vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                filter=rest.Filter(**filter_conditions) if filter_conditions else None,
                with_payload=True,
                with_vector=False,
            )

            results = self.client.search(
                collection_name=collection_name,
                **search_request.dict(exclude_none=True),
            )

            return [
                SearchResult(
                    id=str(result.id), score=result.score, metadata=result.payload or {}
                )
                for result in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points from collection"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=rest.PointIdsList(points=point_ids),
                wait=True,
            )

            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
                "disk_data_size": info.disk_data_size,
                "ram_data_size": info.ram_data_size,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


class RedisCache:
    """Redis cache for frequent queries"""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", prefix: str = "mcp_vector:"
    ):
        """Initialize Redis cache"""
        self.prefix = prefix
        self.redis_client = None
        self.connect(redis_url)

    def connect(self, redis_url: str) -> None:
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"{self.prefix}{key}")
            return pickle.loads(data) if data else None
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False

        try:
            data = pickle.dumps(value)
            self.redis_client.setex(f"{self.prefix}{key}", ttl, data)
            return True
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis_client:
            return False

        try:
            self.redis_client.delete(f"{self.prefix}{key}")
            return True
        except Exception as e:
            logger.debug(f"Cache delete failed: {e}")
            return False

    def clear_prefix(self, pattern: str = "*") -> int:
        """Clear all keys matching pattern"""
        if not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(f"{self.prefix}{pattern}")
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")
            return 0


class VectorIndexer:
    """Main vector indexing system"""

    DOMAINS = ["math", "science", "religion", "history", "literature", "philosophy"]

    def __init__(self, config_path: str = "agents/vector_index/config.yaml"):
        """Initialize vector indexer"""
        self.load_config(config_path)

        # Initialize components
        self.qdrant = QdrantManager(
            host=self.config["qdrant"]["host"],
            port=self.config["qdrant"]["port"],
            api_key=self.config["qdrant"].get("api_key"),
        )

        self.cache = RedisCache(
            redis_url=self.config["redis"]["url"], prefix=self.config["redis"]["prefix"]
        )

        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 4)
        )

        # Setup collections
        self.setup_collections()

    def load_config(self, config_path: str) -> None:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "qdrant": {"host": "localhost", "port": 6333, "api_key": None},
                "redis": {"url": "redis://localhost:6379", "prefix": "mcp_vector:"},
                "collections": {
                    "vector_size": 384,  # Default for sentence transformers
                    "distance": "Cosine",
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 100,
                        "full_scan_threshold": 10000,
                    },
                },
                "indexing": {"batch_size": 100, "max_workers": 4, "cache_ttl": 3600},
                "search": {
                    "default_limit": 10,
                    "score_threshold": 0.7,
                    "enable_caching": True,
                },
            }

    def setup_collections(self) -> None:
        """Setup Qdrant collections for each domain"""
        logger.info("Setting up Qdrant collections...")

        vector_size = self.config["collections"]["vector_size"]
        distance = self.config["collections"]["distance"]
        hnsw_config = self.config["collections"]["hnsw_config"]

        for domain in self.DOMAINS:
            collection_name = f"documents_{domain}"

            if not self.qdrant.collection_exists(collection_name):
                config = CollectionConfig(
                    name=collection_name,
                    vector_size=vector_size,
                    distance=distance,
                    hnsw_config=hnsw_config,
                )

                if self.qdrant.create_collection(config):
                    logger.info(f"Created collection for domain: {domain}")
                else:
                    logger.error(f"Failed to create collection for domain: {domain}")
            else:
                logger.info(f"Collection already exists for domain: {domain}")

        # Create general collection for cross-domain search
        general_collection = "documents_general"
        if not self.qdrant.collection_exists(general_collection):
            config = CollectionConfig(
                name=general_collection,
                vector_size=vector_size,
                distance=distance,
                hnsw_config=hnsw_config,
            )
            self.qdrant.create_collection(config)

    def index_document(self, processed_doc: Dict) -> bool:
        """Index a processed document with embeddings"""
        try:
            doc_id = processed_doc.get("doc_id", "")
            domain = processed_doc.get("domain", "general")
            embeddings = processed_doc.get("embeddings", [])
            chunk_embeddings = processed_doc.get("chunk_embeddings", [])

            if not embeddings and not chunk_embeddings:
                logger.warning(f"No embeddings found for document: {doc_id}")
                return False

            # Prepare metadata
            metadata = {
                "doc_id": doc_id,
                "title": processed_doc.get("title", ""),
                "author": processed_doc.get("author", ""),
                "domain": domain,
                "subcategory": processed_doc.get("subcategory", ""),
                "language": processed_doc.get("language", ""),
                "date": processed_doc.get("date", ""),
                "source": processed_doc.get("source", ""),
                "word_count": processed_doc.get("word_count", 0),
                "indexed_at": datetime.now().isoformat(),
            }

            # Index document-level embedding
            if embeddings:
                doc_vector = VectorDocument(
                    id=f"doc_{doc_id}",
                    vector=(
                        embeddings
                        if isinstance(embeddings, list)
                        else embeddings.tolist()
                    ),
                    metadata={**metadata, "type": "document"},
                )

                # Index in domain-specific collection
                domain_collection = f"documents_{domain}"
                if not self.qdrant.upsert_points(domain_collection, [doc_vector]):
                    logger.error(f"Failed to index document in {domain_collection}")
                    return False

                # Index in general collection
                if not self.qdrant.upsert_points("documents_general", [doc_vector]):
                    logger.error("Failed to index document in general collection")
                    return False

            # Index chunk embeddings
            if chunk_embeddings:
                chunk_vectors = []
                chunks = processed_doc.get("chunks", [])

                for i, chunk_embedding in enumerate(chunk_embeddings):
                    chunk_text = chunks[i] if i < len(chunks) else ""
                    chunk_metadata = {
                        **metadata,
                        "type": "chunk",
                        "chunk_index": i,
                        "chunk_text": chunk_text[:500],  # Store excerpt
                        "parent_doc_id": doc_id,
                    }

                    chunk_vector = VectorDocument(
                        id=f"chunk_{doc_id}_{i}",
                        vector=(
                            chunk_embedding
                            if isinstance(chunk_embedding, list)
                            else chunk_embedding.tolist()
                        ),
                        metadata=chunk_metadata,
                    )
                    chunk_vectors.append(chunk_vector)

                # Batch index chunks
                batch_size = self.config["indexing"]["batch_size"]
                for i in range(0, len(chunk_vectors), batch_size):
                    batch = chunk_vectors[i : i + batch_size]

                    # Index in domain collection
                    if not self.qdrant.upsert_points(domain_collection, batch):
                        logger.error(
                            f"Failed to index chunk batch in {domain_collection}"
                        )

                    # Index in general collection
                    if not self.qdrant.upsert_points("documents_general", batch):
                        logger.error(
                            "Failed to index chunk batch in general collection"
                        )

            logger.info(f"Successfully indexed document: {doc_id}")

            # Clear related cache entries
            self._clear_document_cache(doc_id)

            return True

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        domain: Optional[str] = None,
        limit: int = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search for similar documents/chunks"""
        try:
            # Use provided parameters or defaults
            limit = limit or self.config["search"]["default_limit"]
            score_threshold = (
                score_threshold or self.config["search"]["score_threshold"]
            )

            # Generate cache key
            cache_key = self._generate_search_cache_key(
                query, domain, limit, score_threshold, filter_metadata
            )

            # Check cache first
            if self.config["search"]["enable_caching"]:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Returning cached search results")
                    return [SearchResult(**result) for result in cached_result]

            # Determine collection
            if domain and domain in self.DOMAINS:
                collection_name = f"documents_{domain}"
            else:
                collection_name = "documents_general"

            # Use provided embedding or generate from query
            if query_embedding is None:
                # Would need to generate embedding here
                # For now, return empty results
                logger.warning(
                    "No query embedding provided and embedding generation not implemented"
                )
                return []

            # Perform search
            results = self.qdrant.search_vectors(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_metadata,
            )

            # Cache results
            if self.config["search"]["enable_caching"] and results:
                cache_data = [asdict(result) for result in results]
                self.cache.set(
                    cache_key, cache_data, self.config["indexing"]["cache_ttl"]
                )

            logger.info(f"Found {len(results)} similar documents/chunks")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_document_id(self, doc_id: str) -> List[SearchResult]:
        """Find documents similar to a specific document"""
        try:
            # Get document embedding from any collection
            for domain in self.DOMAINS + ["general"]:
                collection_name = (
                    f"documents_{domain}"
                    if domain != "general"
                    else "documents_general"
                )

                try:
                    # Search for the document itself to get its embedding
                    filter_condition = {
                        "must": [{"key": "doc_id", "match": {"value": doc_id}}]
                    }

                    doc_results = self.qdrant.search_vectors(
                        collection_name=collection_name,
                        query_vector=[0.0]
                        * self.config["collections"]["vector_size"],  # Dummy vector
                        limit=1,
                        filter_conditions=filter_condition,
                    )

                    if doc_results:
                        # Now search for similar documents
                        # Note: This is simplified - in practice, you'd retrieve the actual vector
                        return self.search(
                            query="",
                            query_embedding=None,  # Would use the document's actual embedding
                            domain=doc_results[0].metadata.get("domain"),
                            limit=10,
                        )

                except Exception:
                    continue

            logger.warning(f"Document not found: {doc_id}")
            return []

        except Exception as e:
            logger.error(f"Error searching by document ID: {e}")
            return []

    def get_domain_statistics(self, domain: str) -> Dict:
        """Get statistics for a domain collection"""
        collection_name = f"documents_{domain}"
        return self.qdrant.get_collection_stats(collection_name)

    def get_all_statistics(self) -> Dict:
        """Get statistics for all collections"""
        stats = {}

        for domain in self.DOMAINS + ["general"]:
            collection_name = (
                f"documents_{domain}" if domain != "general" else "documents_general"
            )
            stats[domain] = self.qdrant.get_collection_stats(collection_name)

        return stats

    def delete_document(self, doc_id: str) -> bool:
        """Delete all vectors for a document"""
        try:
            deleted_count = 0

            # Delete from all collections
            for domain in self.DOMAINS + ["general"]:
                collection_name = (
                    f"documents_{domain}"
                    if domain != "general"
                    else "documents_general"
                )

                # Find all points for this document
                filter_condition = {
                    "must": [{"key": "doc_id", "match": {"value": doc_id}}]
                }

                # Search to get point IDs
                search_results = self.qdrant.search_vectors(
                    collection_name=collection_name,
                    query_vector=[0.0] * self.config["collections"]["vector_size"],
                    limit=1000,  # High limit to get all chunks
                    score_threshold=0.0,
                    filter_conditions=filter_condition,
                )

                if search_results:
                    point_ids = [result.id for result in search_results]
                    if self.qdrant.delete_points(collection_name, point_ids):
                        deleted_count += len(point_ids)

            # Clear cache
            self._clear_document_cache(doc_id)

            logger.info(f"Deleted {deleted_count} vectors for document: {doc_id}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def _generate_search_cache_key(
        self,
        query: str,
        domain: Optional[str],
        limit: int,
        score_threshold: Optional[float],
        filter_metadata: Optional[Dict],
    ) -> str:
        """Generate cache key for search"""
        key_parts = [
            query,
            str(domain),
            str(limit),
            str(score_threshold),
            str(hash(str(sorted(filter_metadata.items()))) if filter_metadata else ""),
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _clear_document_cache(self, doc_id: str) -> None:
        """Clear cache entries related to a document"""
        # Clear search cache (simplified - would need more sophisticated cache invalidation)
        self.cache.clear_prefix(f"search_*")

    def reindex_collection(self, domain: str) -> bool:
        """Reindex an entire domain collection"""
        try:
            collection_name = f"documents_{domain}"

            # This would typically involve:
            # 1. Reading all documents from storage
            # 2. Regenerating embeddings if needed
            # 3. Recreating the collection
            # 4. Reindexing all documents

            logger.info(f"Reindexing collection: {collection_name}")
            # Implementation would go here

            return True

        except Exception as e:
            logger.error(f"Error reindexing collection: {e}")
            return False

    def optimize_collections(self) -> bool:
        """Optimize all collections for better performance"""
        try:
            for domain in self.DOMAINS + ["general"]:
                collection_name = (
                    f"documents_{domain}"
                    if domain != "general"
                    else "documents_general"
                )

                # Qdrant automatically optimizes, but we can trigger manual optimization
                # This is a placeholder for optimization operations
                logger.info(f"Optimizing collection: {collection_name}")

            return True

        except Exception as e:
            logger.error(f"Error optimizing collections: {e}")
            return False

    def backup_collection(self, domain: str, backup_path: str) -> bool:
        """Backup a collection to file"""
        try:
            # This would involve exporting collection data
            # Implementation depends on Qdrant backup capabilities
            logger.info(f"Backing up collection {domain} to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error backing up collection: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


async def main():
    """Example usage of vector indexer"""
    # Example processed document with embeddings
    processed_doc = {
        "doc_id": "sample_doc_123",
        "title": "Sample Philosophical Text",
        "author": "Ancient Philosopher",
        "domain": "philosophy",
        "subcategory": "ethics",
        "language": "english",
        "date": "350 BCE",
        "source": "Ancient Library",
        "word_count": 1500,
        "embeddings": np.random.rand(384).tolist(),  # Mock embedding
        "chunk_embeddings": [
            np.random.rand(384).tolist() for _ in range(5)
        ],  # Mock chunk embeddings
        "chunks": [
            "First chunk text...",
            "Second chunk text...",
            "Third chunk text...",
            "Fourth chunk text...",
            "Fifth chunk text...",
        ],
    }

    indexer = VectorIndexer()

    try:
        # Index document
        success = indexer.index_document(processed_doc)
        print(f"Document indexed: {success}")

        # Search for similar documents
        query_embedding = np.random.rand(384).tolist()  # Mock query embedding
        results = indexer.search(
            query="sample query",
            query_embedding=query_embedding,
            domain="philosophy",
            limit=5,
        )

        print(f"Found {len(results)} similar documents")
        for result in results:
            print(
                f"- {result.metadata.get('title', 'Unknown')} (score: {result.score:.3f})"
            )

        # Get statistics
        stats = indexer.get_all_statistics()
        print(f"Collection statistics: {stats}")

    finally:
        indexer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
