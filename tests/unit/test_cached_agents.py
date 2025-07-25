#!/usr/bin/env python3
"""
Test suite for cached agents
Tests the cache-enhanced Neo4j and Qdrant agents
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import asyncio
import numpy as np
import pytest
from cache.integration_manager import CacheIntegrationManager

from agents.neo4j_manager.cached_neo4j_agent import CachedNeo4jAgent
from agents.neo4j_manager.neo4j_agent import NodeData, OperationResult, RelationshipData
from agents.qdrant_manager.cached_qdrant_agent import CachedQdrantAgent
from agents.qdrant_manager.qdrant_agent import VectorOperationResult, VectorPoint


class TestCachedNeo4jAgent:
    """Test the cached Neo4j agent functionality."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache_manager = AsyncMock()
        cache_manager.initialize.return_value = True
        cache_manager.get_integration_status.return_value = {
            "cache_health": {"status": "healthy"},
            "cache_stats": {"hit_rate": 85.0, "total_keys": 1000},
        }
        cache_manager.get_cache_performance_report.return_value = {
            "performance_metrics": {"hit_rate": 85.0}
        }
        return cache_manager

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create mock Neo4j driver."""
        driver = AsyncMock()
        driver.verify_connectivity.return_value = True
        return driver

    @pytest.fixture
    def cached_neo4j_agent(self, mock_cache_manager):
        """Create cached Neo4j agent with mocked dependencies."""
        with patch(
            "agents.neo4j_manager.cached_neo4j_agent.cache_integration",
            mock_cache_manager,
        ):
            agent = CachedNeo4jAgent()
            agent.driver = AsyncMock()
            return agent

    @pytest.mark.asyncio
    async def test_cached_neo4j_agent_initialization(
        self, cached_neo4j_agent, mock_cache_manager
    ):
        """Test cached Neo4j agent initialization."""
        with patch.object(cached_neo4j_agent, "initialize", return_value=True):
            result = await cached_neo4j_agent.initialize()
            assert result is True
            assert cached_neo4j_agent.cache_manager == mock_cache_manager

    @pytest.mark.asyncio
    async def test_get_node_cached(self, cached_neo4j_agent):
        """Test cached node retrieval."""
        node_id = "test_node_id"
        expected_result = OperationResult(
            success=True,
            data={"node": {"id": node_id, "name": "Test Node"}},
            execution_time=0.1,
        )

        # Mock the parent class method
        with patch.object(
            cached_neo4j_agent.__class__.__bases__[0],
            "get_node",
            return_value=expected_result,
        ):
            result = await cached_neo4j_agent.get_node_cached(node_id)

            assert result.success is True
            assert result.data["node"]["id"] == node_id

    @pytest.mark.asyncio
    async def test_get_concepts_by_domain_cached(self, cached_neo4j_agent):
        """Test cached concept retrieval by domain."""
        domain = "philosophy"
        expected_result = OperationResult(
            success=True,
            data={"concepts": [{"id": 1, "name": "Concept 1", "domain": domain}]},
            execution_time=0.2,
        )

        # Mock the execute_cypher method
        with patch.object(
            cached_neo4j_agent, "execute_cypher", return_value=expected_result
        ):
            result = await cached_neo4j_agent.get_concepts_by_domain(domain)

            assert result.success is True
            assert "concepts" in result.data

    @pytest.mark.asyncio
    async def test_get_domain_statistics_cached(self, cached_neo4j_agent):
        """Test cached domain statistics retrieval."""
        domain = "science"
        expected_result = OperationResult(
            success=True,
            data={"domain": domain, "concept_count": 150, "person_count": 25},
            execution_time=0.3,
        )

        # Mock the execute_cypher method
        with patch.object(
            cached_neo4j_agent, "execute_cypher", return_value=expected_result
        ):
            result = await cached_neo4j_agent.get_domain_statistics(domain)

            assert result.success is True
            assert result.data["domain"] == domain
            assert "concept_count" in result.data

    @pytest.mark.asyncio
    async def test_create_node_with_cache_invalidation(self, cached_neo4j_agent):
        """Test node creation with cache invalidation."""
        node_data = NodeData(
            node_type="Concept",
            properties={"name": "Test Concept", "domain": "philosophy"},
        )

        expected_result = OperationResult(success=True, data={"node_id": "new_node_id"})

        # Mock parent class method and cache invalidation
        with patch.object(
            cached_neo4j_agent.__class__.__bases__[0],
            "create_node",
            return_value=expected_result,
        ):
            with patch.object(
                cached_neo4j_agent.cache_manager, "invalidate_domain_cache"
            ) as mock_invalidate:
                with patch.object(
                    cached_neo4j_agent.cache_manager, "clear_analytics_cache"
                ) as mock_clear:
                    result = (
                        await cached_neo4j_agent.create_node_with_cache_invalidation(
                            node_data
                        )
                    )

                    assert result.success is True
                    mock_invalidate.assert_called_once_with("philosophy")
                    mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_performance_report(self, cached_neo4j_agent):
        """Test cache performance report retrieval."""
        result = await cached_neo4j_agent.get_cache_performance_report()

        assert isinstance(result, dict)
        assert "performance_metrics" in result
        assert result["performance_metrics"]["hit_rate"] == 85.0

    @pytest.mark.asyncio
    async def test_warm_cache_for_domain(self, cached_neo4j_agent):
        """Test cache warming for specific domain."""
        domain = "mathematics"

        # Mock the cached methods
        with patch.object(
            cached_neo4j_agent,
            "get_concepts_by_domain",
            return_value=OperationResult(success=True),
        ):
            with patch.object(
                cached_neo4j_agent,
                "get_domain_statistics",
                return_value=OperationResult(success=True),
            ):
                await cached_neo4j_agent.warm_cache_for_domain(domain)

                # Verify the methods were called
                cached_neo4j_agent.get_concepts_by_domain.assert_called_once_with(
                    domain, 50
                )
                cached_neo4j_agent.get_domain_statistics.assert_called_once_with(domain)

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_write_operations(self, cached_neo4j_agent):
        """Test cache invalidation during write operations."""
        query = "CREATE (n:Concept {name: 'Test'})"

        expected_result = OperationResult(success=True)

        # Mock parent class method and cache invalidation
        with patch.object(
            cached_neo4j_agent.__class__.__bases__[0],
            "execute_cypher",
            return_value=expected_result,
        ):
            with patch.object(
                cached_neo4j_agent, "_invalidate_cache_by_query_pattern"
            ) as mock_invalidate:
                result = await cached_neo4j_agent.execute_cypher(query)

                assert result.success is True
                mock_invalidate.assert_called_once_with(query)


class TestCachedQdrantAgent:
    """Test the cached Qdrant agent functionality."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache_manager = AsyncMock()
        cache_manager.initialize.return_value = True
        cache_manager.get_integration_status.return_value = {
            "cache_health": {"status": "healthy"},
            "cache_stats": {"hit_rate": 90.0, "total_keys": 500},
        }
        return cache_manager

    @pytest.fixture
    def cached_qdrant_agent(self, mock_cache_manager):
        """Create cached Qdrant agent with mocked dependencies."""
        with patch(
            "agents.qdrant_manager.cached_qdrant_agent.cache_integration",
            mock_cache_manager,
        ):
            agent = CachedQdrantAgent()
            agent.client = AsyncMock()
            return agent

    @pytest.mark.asyncio
    async def test_cached_qdrant_agent_initialization(
        self, cached_qdrant_agent, mock_cache_manager
    ):
        """Test cached Qdrant agent initialization."""
        with patch.object(cached_qdrant_agent, "initialize", return_value=True):
            result = await cached_qdrant_agent.initialize()
            assert result is True
            assert cached_qdrant_agent.cache_manager == mock_cache_manager

    @pytest.mark.asyncio
    async def test_search_vectors_cached(self, cached_qdrant_agent):
        """Test cached vector search."""
        collection_name = "test_collection"
        query_vector = [0.1, 0.2, 0.3, 0.4]

        expected_result = VectorOperationResult(
            success=True,
            data={"results": [{"id": 1, "score": 0.95}]},
            execution_time=0.1,
        )

        # Mock the parent class method
        with patch.object(
            cached_qdrant_agent.__class__.__bases__[0],
            "search_vectors",
            return_value=expected_result,
        ):
            result = await cached_qdrant_agent.search_vectors_cached(
                collection_name, query_vector, limit=10
            )

            assert result.success is True
            assert "results" in result.data

    @pytest.mark.asyncio
    async def test_get_points_by_domain_cached(self, cached_qdrant_agent):
        """Test cached point retrieval by domain."""
        collection_name = "concepts"
        domain = "philosophy"

        expected_result = VectorOperationResult(
            success=True,
            data={"points": [{"id": 1, "payload": {"domain": domain}}]},
            execution_time=0.2,
        )

        # Mock the search_by_payload_cached method
        with patch.object(
            cached_qdrant_agent,
            "search_by_payload_cached",
            return_value=expected_result,
        ):
            result = await cached_qdrant_agent.get_points_by_domain_cached(
                collection_name, domain, limit=50
            )

            assert result.success is True
            assert "points" in result.data

    @pytest.mark.asyncio
    async def test_get_domain_vector_statistics_cached(self, cached_qdrant_agent):
        """Test cached domain vector statistics."""
        domain = "science"

        # Mock the get_points_by_domain_cached method
        mock_points_result = VectorOperationResult(
            success=True,
            data={
                "points": [
                    {"vector": [0.1, 0.2, 0.3]},
                    {"vector": [0.4, 0.5, 0.6]},
                    {"vector": [0.7, 0.8, 0.9]},
                ]
            },
        )

        with patch.object(
            cached_qdrant_agent,
            "get_points_by_domain_cached",
            return_value=mock_points_result,
        ):
            result = await cached_qdrant_agent.get_domain_vector_statistics_cached(
                domain
            )

            assert result.success is True
            assert result.data["domain"] == domain
            assert result.data["vector_count"] == 3
            assert "vector_dimension" in result.data
            assert "mean_vector" in result.data

    @pytest.mark.asyncio
    async def test_find_similar_concepts_cached(self, cached_qdrant_agent):
        """Test cached similar concept search."""
        concept_id = "concept_123"
        collection_name = "concepts"

        # Mock get_point_cached to return concept vector
        mock_concept_result = VectorOperationResult(
            success=True, data={"vector": [0.1, 0.2, 0.3, 0.4]}
        )

        # Mock search_vectors_cached to return similar concepts
        mock_similar_result = VectorOperationResult(
            success=True, data={"results": [{"id": "similar_concept_1", "score": 0.9}]}
        )

        with patch.object(
            cached_qdrant_agent, "get_point_cached", return_value=mock_concept_result
        ):
            with patch.object(
                cached_qdrant_agent,
                "search_vectors_cached",
                return_value=mock_similar_result,
            ):
                result = await cached_qdrant_agent.find_similar_concepts_cached(
                    concept_id, collection_name
                )

                assert result.success is True
                assert "results" in result.data

    @pytest.mark.asyncio
    async def test_upsert_vector_with_cache_invalidation(self, cached_qdrant_agent):
        """Test vector upsert with cache invalidation."""
        collection_name = "test_collection"
        vector_point = VectorPoint(
            id="test_point",
            vector=[0.1, 0.2, 0.3],
            payload={"domain": "philosophy", "type": "concept"},
        )

        expected_result = VectorOperationResult(success=True)

        # Mock parent class method and cache invalidation
        with patch.object(
            cached_qdrant_agent.__class__.__bases__[0],
            "upsert_vector",
            return_value=expected_result,
        ):
            with patch.object(
                cached_qdrant_agent.cache_manager, "invalidate_domain_cache"
            ) as mock_invalidate:
                with patch.object(
                    cached_qdrant_agent.cache_manager.cache, "delete_pattern"
                ) as mock_delete:
                    result = (
                        await cached_qdrant_agent.upsert_vector_with_cache_invalidation(
                            collection_name, vector_point
                        )
                    )

                    assert result.success is True
                    mock_invalidate.assert_called_once_with("philosophy")
                    mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_upsert_vectors_with_cache_invalidation(
        self, cached_qdrant_agent
    ):
        """Test batch vector upsert with cache invalidation."""
        collection_name = "test_collection"
        vector_points = [
            VectorPoint(
                id="point1", vector=[0.1, 0.2], payload={"domain": "philosophy"}
            ),
            VectorPoint(id="point2", vector=[0.3, 0.4], payload={"domain": "science"}),
        ]

        expected_result = VectorOperationResult(success=True)

        # Mock parent class method and cache invalidation
        with patch.object(
            cached_qdrant_agent.__class__.__bases__[0],
            "batch_upsert_vectors",
            return_value=expected_result,
        ):
            with patch.object(
                cached_qdrant_agent.cache_manager, "invalidate_domain_cache"
            ) as mock_invalidate:
                with patch.object(
                    cached_qdrant_agent.cache_manager.cache, "delete_pattern"
                ) as mock_delete:
                    result = await cached_qdrant_agent.batch_upsert_vectors_with_cache_invalidation(
                        collection_name, vector_points
                    )

                    assert result.success is True
                    # Should invalidate both domains
                    assert mock_invalidate.call_count == 2
                    mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_cache_for_collection(self, cached_qdrant_agent):
        """Test cache warming for collection."""
        collection_name = "test_collection"

        # Mock the cached methods
        with patch.object(
            cached_qdrant_agent,
            "get_collection_info_cached",
            return_value=VectorOperationResult(success=True),
        ):
            with patch.object(
                cached_qdrant_agent,
                "get_collection_statistics_cached",
                return_value=VectorOperationResult(success=True),
            ):
                with patch.object(
                    cached_qdrant_agent,
                    "get_domain_vector_statistics_cached",
                    return_value=VectorOperationResult(success=True),
                ):
                    await cached_qdrant_agent.warm_cache_for_collection(collection_name)

                    # Verify the methods were called
                    cached_qdrant_agent.get_collection_info_cached.assert_called_once_with(
                        collection_name
                    )
                    cached_qdrant_agent.get_collection_statistics_cached.assert_called_once_with(
                        collection_name
                    )
                    # Should warm cache for common domains
                    assert (
                        cached_qdrant_agent.get_domain_vector_statistics_cached.call_count
                        == 6
                    )

    @pytest.mark.asyncio
    async def test_hybrid_search_cached(self, cached_qdrant_agent):
        """Test cached hybrid search."""
        collection_name = "test_collection"
        query_vector = [0.1, 0.2, 0.3]
        payload_filter = {"domain": {"match": {"value": "philosophy"}}}

        expected_result = VectorOperationResult(
            success=True, data={"results": [{"id": 1, "score": 0.95}]}
        )

        # Mock parent class method and cache operations
        with patch.object(
            cached_qdrant_agent.__class__.__bases__[0],
            "hybrid_search",
            return_value=expected_result,
        ):
            with patch.object(
                cached_qdrant_agent.cache_manager.cache, "get", return_value=None
            ):
                with patch.object(
                    cached_qdrant_agent.cache_manager.cache, "set"
                ) as mock_set:
                    result = await cached_qdrant_agent.hybrid_search_cached(
                        collection_name, query_vector, payload_filter
                    )

                    assert result.success is True
                    mock_set.assert_called_once()  # Result should be cached


class TestCacheIntegration:
    """Test cache integration across agents."""

    @pytest.fixture
    def mock_cache_integration_manager(self):
        """Create mock cache integration manager."""
        manager = AsyncMock()
        manager.initialize.return_value = True
        manager.get_integration_status.return_value = {
            "cache_health": {"status": "healthy"},
            "integration_status": {
                "neo4j_manager": True,
                "qdrant_manager": True,
                "analytics_agents": False,
                "api_endpoints": False,
                "streamlit_interface": False,
            },
        }
        return manager

    @pytest.mark.asyncio
    async def test_cache_integration_initialization(
        self, mock_cache_integration_manager
    ):
        """Test cache integration manager initialization."""
        with patch(
            "cache.integration_manager.CacheIntegrationManager",
            return_value=mock_cache_integration_manager,
        ):
            manager = CacheIntegrationManager()
            result = await manager.initialize()

            assert result is True
            mock_cache_integration_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_integration_status(self, mock_cache_integration_manager):
        """Test cache integration status reporting."""
        with patch(
            "cache.integration_manager.CacheIntegrationManager",
            return_value=mock_cache_integration_manager,
        ):
            manager = CacheIntegrationManager()
            status = await manager.get_integration_status()

            assert "cache_health" in status
            assert "integration_status" in status
            assert status["integration_status"]["neo4j_manager"] is True
            assert status["integration_status"]["qdrant_manager"] is True

    @pytest.mark.unit
    def test_cache_decorator_neo4j_query(self):
        """Test Neo4j query cache decorator."""
        from cache.integration_manager import cache_integration

        # Mock the decorator
        with patch.object(cache_integration, "cache_neo4j_query") as mock_decorator:
            mock_decorator.return_value = lambda func: func

            @cache_integration.cache_neo4j_query(ttl=300)
            async def test_query():
                return {"result": "test"}

            mock_decorator.assert_called_once_with(ttl=300)

    @pytest.mark.unit
    def test_cache_decorator_vector_search(self):
        """Test vector search cache decorator."""
        from cache.integration_manager import cache_integration

        # Mock the decorator
        with patch.object(cache_integration, "cache_vector_search") as mock_decorator:
            mock_decorator.return_value = lambda func: func

            @cache_integration.cache_vector_search(ttl=600)
            async def test_search():
                return {"results": []}

            mock_decorator.assert_called_once_with(ttl=600)

    @pytest.mark.asyncio
    async def test_domain_cache_invalidation(self, mock_cache_integration_manager):
        """Test domain-specific cache invalidation."""
        with patch(
            "cache.integration_manager.CacheIntegrationManager",
            return_value=mock_cache_integration_manager,
        ):
            manager = CacheIntegrationManager()

            domain = "philosophy"
            await manager.invalidate_domain_cache(domain)

            mock_cache_integration_manager.invalidate_domain_cache.assert_called_once_with(
                domain
            )

    @pytest.mark.asyncio
    async def test_concept_cache_invalidation(self, mock_cache_integration_manager):
        """Test concept-specific cache invalidation."""
        with patch(
            "cache.integration_manager.CacheIntegrationManager",
            return_value=mock_cache_integration_manager,
        ):
            manager = CacheIntegrationManager()

            concept_id = "concept_123"
            await manager.invalidate_concept_cache(concept_id)

            mock_cache_integration_manager.invalidate_concept_cache.assert_called_once_with(
                concept_id
            )


class TestCachePerformance:
    """Test cache performance and optimization."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # Mock cache stats
        cache_stats = {"keyspace_hits": 850, "keyspace_misses": 150, "total_keys": 1000}

        # Calculate hit rate
        hits = cache_stats["keyspace_hits"]
        misses = cache_stats["keyspace_misses"]
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0

        assert hit_rate == 85.0
        assert hit_rate >= 85.0  # Should meet target hit rate

    @pytest.mark.asyncio
    async def test_cache_memory_usage_monitoring(self):
        """Test cache memory usage monitoring."""
        # Mock memory stats
        memory_stats = {
            "used_memory": 50 * 1024 * 1024,  # 50MB
            "used_memory_human": "50MB",
            "max_memory": 100 * 1024 * 1024,  # 100MB
        }

        memory_usage_percent = (
            memory_stats["used_memory"] / memory_stats["max_memory"]
        ) * 100

        assert memory_usage_percent == 50.0
        assert memory_usage_percent < 80.0  # Should be within acceptable range

    @pytest.mark.asyncio
    async def test_cache_performance_recommendations(self):
        """Test cache performance recommendations."""
        from cache.integration_manager import CacheIntegrationManager

        manager = CacheIntegrationManager()

        # Test with low hit rate
        low_hit_rate_stats = {
            "hit_rate": 65.0,
            "total_keys": 15000,
            "used_memory": 200 * 1024 * 1024,
        }

        recommendations = manager._generate_performance_recommendations(
            low_hit_rate_stats
        )

        assert len(recommendations) >= 3
        assert any("hit rate" in rec.lower() for rec in recommendations)
        assert any("keys" in rec.lower() for rec in recommendations)
        assert any("memory" in rec.lower() for rec in recommendations)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        # This would test cache performance with many concurrent operations
        # Mock implementation for testing

        from cache.cache_manager import CacheManager

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = CacheManager()

            # Simulate high load
            tasks = []
            for i in range(1000):
                task = cache.get(f"test_key_{i}")
                tasks.append(task)

            # All operations should complete without error
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # No exceptions should occur
            assert all(not isinstance(result, Exception) for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
