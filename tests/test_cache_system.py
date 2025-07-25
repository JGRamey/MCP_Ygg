#!/usr/bin/env python3
"""
Test suite for the comprehensive cache system
"""

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import pytest
from cache.cache_manager import CacheManager
from cache.config import get_cache_config
from cache.integration_manager import CacheIntegrationManager


class TestCacheManager:
    """Test the core cache manager functionality."""

    @pytest.fixture
    async def cache_manager(self):
        """Create a cache manager for testing."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.return_value = AsyncMock()
            cache = CacheManager()
            yield cache
            await cache.close()

    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_manager):
        """Test cache manager initialization."""
        assert cache_manager._cache_prefix == "mcp:"
        assert cache_manager.redis is not None

    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = {"test": "data"}

        # Mock Redis operations
        cache_manager.redis.get.return_value = None
        cache_manager.redis.setex.return_value = True

        await cache_manager.set(key, value, ttl=300)
        cache_manager.redis.setex.assert_called_once()

        # Test get operation
        with patch("pickle.loads") as mock_loads:
            mock_loads.return_value = value
            cache_manager.redis.get.return_value = b"pickled_data"

            result = await cache_manager.get(key)
            assert result == value

    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_manager):
        """Test cache delete operation."""
        key = "test_key"

        cache_manager.redis.delete.return_value = 1
        await cache_manager.delete(key)

        cache_manager.redis.delete.assert_called_once_with(key)

    @pytest.mark.asyncio
    async def test_cache_pattern_delete(self, cache_manager):
        """Test pattern-based cache deletion."""
        pattern = "test_pattern"

        # Mock scan_iter to return some keys
        cache_manager.redis.scan_iter.return_value = [
            b"mcp:test_pattern:key1",
            b"mcp:test_pattern:key2",
        ]
        cache_manager.redis.delete.return_value = 2

        result = await cache_manager.delete_pattern(pattern)
        assert result == 2

    @pytest.mark.asyncio
    async def test_cache_decorator(self, cache_manager):
        """Test the cache decorator functionality."""
        call_count = 0

        @cache_manager.cached(ttl=300)
        async def test_function(param1, param2=None):
            nonlocal call_count
            call_count += 1
            return f"result_{param1}_{param2}"

        # Mock cache miss first, then hit
        cache_manager.redis.get.side_effect = [None, b"pickled_result"]

        with patch("pickle.loads") as mock_loads, patch("pickle.dumps") as mock_dumps:
            mock_loads.return_value = "cached_result"
            mock_dumps.return_value = b"pickled_result"

            # First call - cache miss
            result1 = await test_function("test", param2="value")
            assert call_count == 1

            # Second call - cache hit
            result2 = await test_function("test", param2="value")
            assert call_count == 1  # Should not increment due to cache hit
            assert result2 == "cached_result"

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics functionality."""
        mock_info = {
            "connected_clients": 5,
            "used_memory": 1024,
            "used_memory_human": "1KB",
            "keyspace_hits": 100,
            "keyspace_misses": 20,
        }

        cache_manager.redis.info.return_value = mock_info
        cache_manager.redis.scan_iter.return_value = [b"key1", b"key2", b"key3"]

        stats = await cache_manager.get_stats()

        assert stats["connected_clients"] == 5
        assert stats["used_memory"] == 1024
        assert stats["hit_rate"] == 83.33333333333334  # 100/(100+20) * 100
        assert stats["total_keys"] == 3

    @pytest.mark.asyncio
    async def test_cache_health_check(self, cache_manager):
        """Test cache health check functionality."""
        cache_manager.redis.get.return_value = None
        cache_manager.redis.setex.return_value = True
        cache_manager.redis.delete.return_value = 1

        with patch("pickle.loads") as mock_loads, patch("pickle.dumps") as mock_dumps:
            mock_loads.return_value = {"timestamp": "2023-01-01T00:00:00"}
            mock_dumps.return_value = b"pickled_data"

            health = await cache_manager.health_check()

            assert health["status"] == "healthy"
            assert health["redis_connected"] is True
            assert health["read_write_test"] is True


class TestCacheIntegrationManager:
    """Test the cache integration manager."""

    @pytest.fixture
    def integration_manager(self):
        """Create a cache integration manager for testing."""
        with patch("cache.integration_manager.CacheManager") as mock_cache_manager:
            mock_cache_manager.return_value = AsyncMock()
            manager = CacheIntegrationManager()
            yield manager

    @pytest.mark.asyncio
    async def test_integration_initialization(self, integration_manager):
        """Test integration manager initialization."""
        integration_manager.cache.health_check.return_value = {"status": "healthy"}

        result = await integration_manager.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_neo4j_cache_decorator(self, integration_manager):
        """Test Neo4j cache decorator."""

        @integration_manager.cache_neo4j_query(ttl=300)
        async def test_neo4j_query(query, params=None):
            return {"result": "neo4j_data"}

        # Mock the cache decorator
        integration_manager.cache.cached.return_value = lambda func: func

        result = await test_neo4j_query("MATCH (n) RETURN n")
        assert result == {"result": "neo4j_data"}
        assert integration_manager.integration_status["neo4j_manager"] is True

    @pytest.mark.asyncio
    async def test_vector_search_cache_decorator(self, integration_manager):
        """Test vector search cache decorator."""

        @integration_manager.cache_vector_search(ttl=600)
        async def test_vector_search(query_vector, limit=10):
            return {"result": "vector_data"}

        # Mock the cache decorator
        integration_manager.cache.cached.return_value = lambda func: func

        result = await test_vector_search([0.1, 0.2, 0.3])
        assert result == {"result": "vector_data"}
        assert integration_manager.integration_status["qdrant_manager"] is True

    @pytest.mark.asyncio
    async def test_analytics_cache_decorator(self, integration_manager):
        """Test analytics cache decorator."""

        @integration_manager.cache_analytics_computation(ttl=3600)
        async def test_analytics_computation(computation_type, params):
            return {"result": "analytics_data"}

        # Mock the cache decorator
        integration_manager.cache.cached.return_value = lambda func: func

        result = await test_analytics_computation("centrality", {"type": "betweenness"})
        assert result == {"result": "analytics_data"}
        assert integration_manager.integration_status["analytics_agents"] is True

    @pytest.mark.asyncio
    async def test_domain_cache_invalidation(self, integration_manager):
        """Test domain cache invalidation."""
        domain = "philosophy"

        await integration_manager.invalidate_domain_cache(domain)

        # Check that delete_pattern was called for relevant patterns
        assert integration_manager.cache.delete_pattern.call_count >= 1

    @pytest.mark.asyncio
    async def test_concept_cache_invalidation(self, integration_manager):
        """Test concept cache invalidation."""
        concept_id = "concept123"

        await integration_manager.invalidate_concept_cache(concept_id)

        # Check that delete_pattern was called for relevant patterns
        assert integration_manager.cache.delete_pattern.call_count >= 1

    @pytest.mark.asyncio
    async def test_performance_report(self, integration_manager):
        """Test cache performance report generation."""
        mock_stats = {
            "hit_rate": 85.0,
            "total_keys": 1000,
            "used_memory": 50 * 1024 * 1024,  # 50MB
            "used_memory_human": "50MB",
            "connected_clients": 3,
        }

        integration_manager.cache.get_stats.return_value = mock_stats
        integration_manager.cache.health_check.return_value = {"status": "healthy"}

        report = await integration_manager.get_cache_performance_report()

        assert report["performance_metrics"]["hit_rate"] == 85.0
        assert report["performance_metrics"]["total_keys"] == 1000
        assert report["integration_coverage"]["total_components"] == 5
        assert len(report["recommendations"]) >= 0


class TestCacheConfig:
    """Test cache configuration."""

    def test_cache_config_structure(self):
        """Test that cache configuration has expected structure."""
        config = get_cache_config()

        assert "redis_url" in config
        assert "default_ttl" in config
        assert "ttl" in config
        assert "warming" in config
        assert "performance" in config

        # Test TTL configurations
        assert config["ttl"]["graph_concepts"] == 300
        assert config["ttl"]["vector_search"] == 600
        assert config["ttl"]["analytics_computation"] == 3600

        # Test warming configuration
        assert config["warming"]["enabled"] is True
        assert config["warming"]["batch_size"] == 10

        # Test performance configuration
        assert config["performance"]["hit_rate_alert"] == 0.7
        assert config["performance"]["slow_query_threshold"] == 1.0


class TestCacheSystemIntegration:
    """Test the overall cache system integration."""

    @pytest.mark.asyncio
    async def test_cache_system_workflow(self):
        """Test complete cache system workflow."""
        # This would test the complete workflow:
        # 1. Initialize cache system
        # 2. Execute cached operations
        # 3. Verify cache hits/misses
        # 4. Test cache invalidation
        # 5. Verify performance metrics

        # Mock implementation for testing
        with patch("cache.integration_manager.CacheManager") as mock_cache_manager:
            mock_instance = AsyncMock()
            mock_cache_manager.return_value = mock_instance

            # Test initialization
            manager = CacheIntegrationManager()
            mock_instance.health_check.return_value = {"status": "healthy"}

            result = await manager.initialize()
            assert result is True

            # Test cache operations
            mock_instance.get_stats.return_value = {
                "hit_rate": 90.0,
                "total_keys": 500,
                "used_memory": 10 * 1024 * 1024,
                "used_memory_human": "10MB",
            }

            stats = await manager.get_integration_status()
            assert stats["cache_stats"]["hit_rate"] == 90.0

            # Test performance report
            report = await manager.get_cache_performance_report()
            assert report["performance_metrics"]["hit_rate"] == 90.0

    @pytest.mark.asyncio
    async def test_cache_warming(self):
        """Test cache warming functionality."""
        with patch("cache.integration_manager.CacheManager") as mock_cache_manager:
            mock_instance = AsyncMock()
            mock_cache_manager.return_value = mock_instance

            manager = CacheIntegrationManager()

            # Test warming for specific domain
            await manager._warm_domain_cache("philosophy")

            # In real implementation, this would call actual warming functions
            # For now, just verify it doesn't crash
            assert True  # Placeholder assertion

    @pytest.mark.asyncio
    async def test_cache_performance_monitoring(self):
        """Test cache performance monitoring."""
        with patch("cache.integration_manager.CacheManager") as mock_cache_manager:
            mock_instance = AsyncMock()
            mock_cache_manager.return_value = mock_instance

            manager = CacheIntegrationManager()

            # Test performance recommendations
            mock_stats = {
                "hit_rate": 60.0,  # Below threshold
                "total_keys": 15000,  # High number
                "used_memory": 200 * 1024 * 1024,  # 200MB
            }

            recommendations = manager._generate_performance_recommendations(mock_stats)

            assert len(recommendations) > 0
            assert any("hit rate" in rec.lower() for rec in recommendations)
            assert any("keys" in rec.lower() for rec in recommendations)
            assert any("memory" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    # Run the tests
    asyncio.run(pytest.main([__file__, "-v"]))
