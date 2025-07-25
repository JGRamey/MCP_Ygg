#!/usr/bin/env python3
"""
Cached Qdrant Manager Agent
Extends the base Qdrant agent with comprehensive caching capabilities
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncio
import numpy as np
from cache.integration_manager import cache_integration

from .qdrant_agent import QdrantAgent, SearchQuery, VectorOperationResult, VectorPoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedQdrantAgent(QdrantAgent):
    """Qdrant Agent with integrated caching capabilities."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.cache_manager = cache_integration

    async def initialize(self) -> bool:
        """Initialize the Qdrant connection and cache system."""
        # Initialize base Qdrant agent
        qdrant_init = await super().initialize()

        # Initialize cache system
        try:
            await self.cache_manager.initialize()
            logger.info("Qdrant cache system initialized successfully")
        except Exception as e:
            logger.warning(f"Qdrant cache initialization failed: {e}")
            # Continue without cache

        return qdrant_init

    # Cached search operations
    @cache_integration.cache_vector_search(ttl=600)  # 10 minutes
    async def search_vectors_cached(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> VectorOperationResult:
        """Search vectors with caching."""
        return await super().search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

    @cache_integration.cache_vector_search(ttl=1800)  # 30 minutes
    async def search_by_payload_cached(
        self, collection_name: str, payload_filter: Dict[str, Any], limit: int = 10
    ) -> VectorOperationResult:
        """Search by payload with caching."""
        return await super().search_by_payload(
            collection_name=collection_name, payload_filter=payload_filter, limit=limit
        )

    @cache_integration.cache_vector_search(ttl=3600)  # 1 hour
    async def get_collection_info_cached(
        self, collection_name: str
    ) -> VectorOperationResult:
        """Get collection info with caching."""
        return await super().get_collection_info(collection_name)

    @cache_integration.cache_vector_search(ttl=600)  # 10 minutes
    async def get_point_cached(
        self, collection_name: str, point_id: Union[str, int]
    ) -> VectorOperationResult:
        """Get specific point with caching."""
        return await super().get_point(collection_name, point_id)

    @cache_integration.cache_vector_search(ttl=1800)  # 30 minutes
    async def get_points_by_domain_cached(
        self, collection_name: str, domain: str, limit: int = 100
    ) -> VectorOperationResult:
        """Get points by domain with caching."""
        payload_filter = {"domain": {"match": {"value": domain}}}

        return await self.search_by_payload_cached(
            collection_name=collection_name, payload_filter=payload_filter, limit=limit
        )

    @cache_integration.cache_vector_search(ttl=3600)  # 1 hour
    async def get_collection_statistics_cached(
        self, collection_name: str
    ) -> VectorOperationResult:
        """Get collection statistics with caching."""
        return await super().get_collection_statistics(collection_name)

    @cache_integration.cache_vector_search(ttl=600)  # 10 minutes
    async def semantic_search_concepts_cached(
        self,
        query_text: str,
        domain: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> VectorOperationResult:
        """Semantic search for concepts with caching."""
        # This would integrate with text embedding service
        # For now, return placeholder
        return VectorOperationResult(
            success=True,
            data={"query": query_text, "domain": domain, "results": [], "cached": True},
            execution_time=0.1,
        )

    @cache_integration.cache_vector_search(ttl=1800)  # 30 minutes
    async def find_similar_concepts_cached(
        self,
        concept_id: str,
        collection_name: str = "concepts",
        limit: int = 10,
        score_threshold: float = 0.8,
    ) -> VectorOperationResult:
        """Find similar concepts with caching."""
        # Get the concept vector first
        concept_result = await self.get_point_cached(collection_name, concept_id)

        if not concept_result.success:
            return concept_result

        concept_vector = concept_result.data.get("vector", [])

        # Search for similar vectors
        return await self.search_vectors_cached(
            collection_name=collection_name,
            query_vector=concept_vector,
            limit=limit + 1,  # +1 to exclude the original concept
            score_threshold=score_threshold,
        )

    @cache_integration.cache_vector_search(ttl=3600)  # 1 hour
    async def get_domain_vector_statistics_cached(
        self, domain: str
    ) -> VectorOperationResult:
        """Get vector statistics for a domain with caching."""
        # Get all points for the domain
        domain_points = await self.get_points_by_domain_cached("concepts", domain, 1000)

        if not domain_points.success:
            return domain_points

        points = domain_points.data.get("points", [])

        # Calculate statistics
        vectors = [point.get("vector", []) for point in points if point.get("vector")]

        if not vectors:
            return VectorOperationResult(
                success=True,
                data={"domain": domain, "vector_count": 0, "statistics": {}},
                execution_time=0.1,
            )

        vectors_np = np.array(vectors)

        statistics = {
            "domain": domain,
            "vector_count": len(vectors),
            "vector_dimension": vectors_np.shape[1] if vectors_np.size > 0 else 0,
            "mean_vector": (
                vectors_np.mean(axis=0).tolist() if vectors_np.size > 0 else []
            ),
            "std_vector": (
                vectors_np.std(axis=0).tolist() if vectors_np.size > 0 else []
            ),
            "vector_norms": {
                "mean": (
                    float(np.linalg.norm(vectors_np, axis=1).mean())
                    if vectors_np.size > 0
                    else 0
                ),
                "std": (
                    float(np.linalg.norm(vectors_np, axis=1).std())
                    if vectors_np.size > 0
                    else 0
                ),
                "min": (
                    float(np.linalg.norm(vectors_np, axis=1).min())
                    if vectors_np.size > 0
                    else 0
                ),
                "max": (
                    float(np.linalg.norm(vectors_np, axis=1).max())
                    if vectors_np.size > 0
                    else 0
                ),
            },
        }

        return VectorOperationResult(success=True, data=statistics, execution_time=0.1)

    # Cache invalidation methods
    async def upsert_vector_with_cache_invalidation(
        self, collection_name: str, vector_point: VectorPoint
    ) -> VectorOperationResult:
        """Upsert vector and invalidate relevant cache entries."""
        result = await super().upsert_vector(collection_name, vector_point)

        if result.success:
            # Invalidate relevant caches
            domain = vector_point.payload.get("domain", "")
            await self.cache_manager.invalidate_domain_cache(domain)

            # Clear collection statistics cache
            await self.cache_manager.cache.delete_pattern(
                f"vector_search:*{collection_name}*"
            )

        return result

    async def delete_vector_with_cache_invalidation(
        self, collection_name: str, point_id: Union[str, int]
    ) -> VectorOperationResult:
        """Delete vector and invalidate relevant cache entries."""
        result = await super().delete_vector(collection_name, point_id)

        if result.success:
            # Invalidate all related caches
            await self.cache_manager.cache.delete_pattern(
                f"vector_search:*{collection_name}*"
            )
            await self.cache_manager.cache.delete_pattern(f"vector_search:*{point_id}*")

        return result

    async def batch_upsert_vectors_with_cache_invalidation(
        self, collection_name: str, vector_points: List[VectorPoint]
    ) -> VectorOperationResult:
        """Batch upsert vectors and invalidate relevant cache entries."""
        result = await super().batch_upsert_vectors(collection_name, vector_points)

        if result.success:
            # Invalidate all collection-related caches
            await self.cache_manager.cache.delete_pattern(
                f"vector_search:*{collection_name}*"
            )

            # Invalidate domain caches
            domains = set()
            for point in vector_points:
                domain = point.payload.get("domain", "")
                if domain:
                    domains.add(domain)

            for domain in domains:
                await self.cache_manager.invalidate_domain_cache(domain)

        return result

    # Cache management methods
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and performance metrics."""
        return await self.cache_manager.get_integration_status()

    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Get detailed cache performance report."""
        return await self.cache_manager.get_cache_performance_report()

    async def clear_collection_cache(self, collection_name: str):
        """Clear cache for a specific collection."""
        await self.cache_manager.cache.delete_pattern(
            f"vector_search:*{collection_name}*"
        )

    async def clear_domain_cache(self, domain: str):
        """Clear cache for a specific domain."""
        await self.cache_manager.invalidate_domain_cache(domain)

    async def warm_cache_for_collection(self, collection_name: str):
        """Warm cache for a specific collection."""
        try:
            # Pre-load collection info
            await self.get_collection_info_cached(collection_name)
            await self.get_collection_statistics_cached(collection_name)

            # Pre-load domain statistics for common domains
            common_domains = [
                "philosophy",
                "science",
                "mathematics",
                "art",
                "language",
                "technology",
            ]
            for domain in common_domains:
                await self.get_domain_vector_statistics_cached(domain)

            logger.info(f"Cache warmed for collection: {collection_name}")
        except Exception as e:
            logger.warning(
                f"Cache warming failed for collection {collection_name}: {e}"
            )

    async def warm_cache_for_domain(self, domain: str):
        """Warm cache for a specific domain."""
        try:
            # Pre-load domain-specific data
            await self.get_points_by_domain_cached("concepts", domain, 50)
            await self.get_domain_vector_statistics_cached(domain)

            logger.info(f"Qdrant cache warmed for domain: {domain}")
        except Exception as e:
            logger.warning(f"Qdrant cache warming failed for domain {domain}: {e}")

    async def close(self):
        """Close Qdrant agent and cache system."""
        await super().close()
        await self.cache_manager.close()

    # Enhanced search with caching
    async def hybrid_search_cached(
        self,
        collection_name: str,
        query_vector: List[float],
        payload_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> VectorOperationResult:
        """Hybrid search with vector and payload filtering, cached."""
        # Create cache key based on all parameters
        cache_key = f"hybrid_search:{collection_name}:{hash(str(query_vector))}"
        if payload_filter:
            cache_key += f":{hash(str(payload_filter))}"
        cache_key += f":{limit}:{score_threshold}"

        # Try to get from cache first
        cached_result = await self.cache_manager.cache.get(cache_key)
        if cached_result:
            return cached_result

        # Not in cache, perform search
        result = await super().hybrid_search(
            collection_name=collection_name,
            query_vector=query_vector,
            payload_filter=payload_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Cache the result
        if result.success:
            await self.cache_manager.cache.set(cache_key, result, ttl=600)

        return result


# Global cached Qdrant agent instance
cached_qdrant_agent = CachedQdrantAgent()


# Convenience functions for direct use
async def init_cached_qdrant_agent(config_path: Optional[str] = None) -> bool:
    """Initialize the cached Qdrant agent."""
    if config_path:
        global cached_qdrant_agent
        cached_qdrant_agent = CachedQdrantAgent(config_path)

    return await cached_qdrant_agent.initialize()


async def get_cached_qdrant_agent() -> CachedQdrantAgent:
    """Get the global cached Qdrant agent instance."""
    return cached_qdrant_agent
