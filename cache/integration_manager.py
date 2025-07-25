"""
Cache Integration Manager for MCP Yggdrasil
Provides centralized cache integration across all system components
"""

from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import asyncio

from .cache_manager import CacheManager
from .config import get_cache_config, get_ttl_for_key_type


class CacheIntegrationManager:
    """Manages cache integration across all MCP Yggdrasil components."""

    def __init__(self):
        self.cache = CacheManager()
        self.config = get_cache_config()
        self.integration_status = {
            "neo4j_manager": False,
            "qdrant_manager": False,
            "analytics_agents": False,
            "api_endpoints": False,
            "streamlit_interface": False,
        }

    async def initialize(self):
        """Initialize cache integration system."""
        # Perform health check
        health = await self.cache.health_check()
        if health["status"] != "healthy":
            raise Exception(f"Cache system unhealthy: {health}")

        # Warm cache if enabled
        if self.config["warming"]["enabled"]:
            await self._warm_system_cache()

        return True

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status across all components."""
        stats = await self.cache.get_stats()

        return {
            "cache_health": await self.cache.health_check(),
            "integration_status": self.integration_status,
            "cache_stats": stats,
            "performance_metrics": {
                "hit_rate": stats.get("hit_rate", 0),
                "total_keys": stats.get("total_keys", 0),
                "memory_usage": stats.get("used_memory_human", "0B"),
            },
        }

    # Neo4j Query Caching
    def cache_neo4j_query(self, ttl: Optional[int] = None):
        """Decorator for caching Neo4j queries."""
        cache_ttl = ttl or get_ttl_for_key_type("graph_concepts")

        def decorator(func):
            @self.cache.cached(ttl=cache_ttl, key_prefix="neo4j_query")
            @wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.integration_status["neo4j_manager"] = True
                return result

            return wrapper

        return decorator

    # Qdrant Vector Search Caching
    def cache_vector_search(self, ttl: Optional[int] = None):
        """Decorator for caching Qdrant vector searches."""
        cache_ttl = ttl or get_ttl_for_key_type("vector_search")

        def decorator(func):
            @self.cache.cached(ttl=cache_ttl, key_prefix="vector_search")
            @wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.integration_status["qdrant_manager"] = True
                return result

            return wrapper

        return decorator

    # Analytics Computation Caching
    def cache_analytics_computation(self, ttl: Optional[int] = None):
        """Decorator for caching analytics computations."""
        cache_ttl = ttl or get_ttl_for_key_type("analytics_computation")

        def decorator(func):
            @self.cache.cached(ttl=cache_ttl, key_prefix="analytics_computation")
            @wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.integration_status["analytics_agents"] = True
                return result

            return wrapper

        return decorator

    # API Response Caching
    def cache_api_response(self, ttl: Optional[int] = None):
        """Decorator for caching API responses."""
        cache_ttl = ttl or get_ttl_for_key_type("api_responses")

        def decorator(func):
            @self.cache.cached(ttl=cache_ttl, key_prefix="api_response")
            @wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.integration_status["api_endpoints"] = True
                return result

            return wrapper

        return decorator

    # Streamlit Session Caching
    def cache_streamlit_operation(self, ttl: Optional[int] = None):
        """Decorator for caching Streamlit operations."""
        cache_ttl = ttl or get_ttl_for_key_type("user_sessions")

        def decorator(func):
            @self.cache.cached(ttl=cache_ttl, key_prefix="streamlit_operation")
            @wraps(func)
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.integration_status["streamlit_interface"] = True
                return result

            return wrapper

        return decorator

    # Specialized cache invalidation methods
    async def invalidate_domain_cache(self, domain: str):
        """Invalidate all cache entries for a specific domain."""
        patterns = [
            f"neo4j_query:*{domain}*",
            f"vector_search:*{domain}*",
            f"analytics_computation:*{domain}*",
        ]

        for pattern in patterns:
            await self.cache.delete_pattern(pattern)

    async def invalidate_concept_cache(self, concept_id: str):
        """Invalidate cache entries for a specific concept."""
        patterns = [
            f"neo4j_query:*{concept_id}*",
            f"vector_search:*{concept_id}*",
            f"analytics_computation:*{concept_id}*",
        ]

        for pattern in patterns:
            await self.cache.delete_pattern(pattern)

    async def invalidate_user_session_cache(self, session_id: str):
        """Invalidate cache entries for a specific user session."""
        await self.cache.delete_pattern(f"streamlit_operation:*{session_id}*")

    async def clear_analytics_cache(self):
        """Clear all analytics computation cache."""
        await self.cache.delete_pattern("analytics_computation:*")

    async def clear_search_cache(self):
        """Clear all search-related cache."""
        patterns = ["neo4j_query:*", "vector_search:*"]

        for pattern in patterns:
            await self.cache.delete_pattern(pattern)

    async def _warm_system_cache(self):
        """Warm the cache with commonly accessed data."""
        # Common domains to warm
        common_domains = [
            "philosophy",
            "science",
            "mathematics",
            "art",
            "language",
            "technology",
        ]

        warming_tasks = []
        for domain in common_domains:
            # This would be replaced with actual warming functions
            # warming_tasks.append(self._warm_domain_cache(domain))
            pass

        if warming_tasks:
            await asyncio.gather(*warming_tasks, return_exceptions=True)

    async def _warm_domain_cache(self, domain: str):
        """Warm cache for a specific domain."""
        # This would call actual domain-specific queries
        # Example: await get_concepts_by_domain(domain)
        pass

    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance report."""
        stats = await self.cache.get_stats()
        integration_status = await self.get_integration_status()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cache_health": integration_status["cache_health"],
            "integration_coverage": {
                "components_integrated": sum(self.integration_status.values()),
                "total_components": len(self.integration_status),
                "coverage_percentage": (
                    sum(self.integration_status.values()) / len(self.integration_status)
                )
                * 100,
            },
            "performance_metrics": {
                "hit_rate": stats.get("hit_rate", 0),
                "total_keys": stats.get("total_keys", 0),
                "memory_usage_bytes": stats.get("used_memory", 0),
                "memory_usage_human": stats.get("used_memory_human", "0B"),
                "connected_clients": stats.get("connected_clients", 0),
            },
            "recommendations": self._generate_performance_recommendations(stats),
        }

    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on cache stats."""
        recommendations = []

        hit_rate = stats.get("hit_rate", 0)
        if hit_rate < 70:
            recommendations.append(
                "Cache hit rate is below 70%. Consider increasing TTL values or improving cache warming."
            )

        total_keys = stats.get("total_keys", 0)
        if total_keys > 10000:
            recommendations.append(
                "High number of cache keys. Consider implementing cache eviction policies."
            )

        memory_usage = stats.get("used_memory", 0)
        if memory_usage > 100 * 1024 * 1024:  # 100MB
            recommendations.append(
                "High memory usage. Consider reducing TTL values or implementing memory limits."
            )

        if not any(self.integration_status.values()):
            recommendations.append(
                "No components are using cache. Implement cache integration across system components."
            )

        return recommendations

    async def close(self):
        """Close cache integration manager."""
        await self.cache.close()


# Global cache integration manager
cache_integration = CacheIntegrationManager()


# Convenience functions for direct use
async def init_cache_system():
    """Initialize the cache system."""
    return await cache_integration.initialize()


async def get_cache_status():
    """Get current cache system status."""
    return await cache_integration.get_integration_status()


async def get_cache_report():
    """Get comprehensive cache performance report."""
    return await cache_integration.get_cache_performance_report()


async def shutdown_cache_system():
    """Shutdown the cache system."""
    await cache_integration.close()
