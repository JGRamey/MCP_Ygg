import asyncio
import functools
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Union

import redis.asyncio as redis
from prometheus_client import Counter, Histogram


# Metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits', ['function'])
cache_misses = Counter('cache_misses_total', 'Total cache misses', ['function'])
cache_latency = Histogram('cache_operation_seconds', 'Cache operation latency', ['operation'])


class CacheManager:
    """Advanced caching manager with TTL, invalidation, and monitoring."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self._cache_prefix = "mcp:"
        
    async def close(self):
        """Close Redis connection."""
        await self.redis.close()
        
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a stable hash from arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{self._cache_prefix}{func_name}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with cache_latency.labels('get').time():
            value = await self.redis.get(key)
            if value:
                return pickle.loads(value)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in cache with TTL."""
        with cache_latency.labels('set').time():
            serialized = pickle.dumps(value)
            await self.redis.setex(key, ttl, serialized)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        with cache_latency.labels('delete').time():
            await self.redis.delete(key)
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        with cache_latency.labels('delete_pattern').time():
            keys = []
            async for key in self.redis.scan_iter(match=f"{self._cache_prefix}{pattern}"):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        info = await self.redis.info()
        return {
            'connected_clients': info.get('connected_clients', 0),
            'used_memory': info.get('used_memory', 0),
            'used_memory_human': info.get('used_memory_human', '0B'),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': self._calculate_hit_rate(info),
            'total_keys': await self._count_keys()
        }
    
    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    async def _count_keys(self) -> int:
        """Count total keys with MCP prefix."""
        count = 0
        async for _ in self.redis.scan_iter(match=f"{self._cache_prefix}*"):
            count += 1
        return count
    
    def cached(self, ttl: int = 300, key_prefix: Optional[str] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                func_name = f"{key_prefix or func.__name__}"
                cache_key = self._generate_key(func_name, args, kwargs)
                
                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    cache_hits.labels(func_name).inc()
                    return cached_value
                
                # Cache miss - compute value
                cache_misses.labels(func_name).inc()
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.set(cache_key, result, ttl)
                return result
            
            # Add cache invalidation methods
            wrapper.invalidate = lambda *args, **kwargs: self.delete(
                self._generate_key(key_prefix or func.__name__, args, kwargs)
            )
            wrapper.invalidate_all = lambda: self.delete_pattern(
                f"{key_prefix or func.__name__}:*"
            )
            
            return wrapper
        return decorator
    
    async def warm_cache(self, warm_functions: list):
        """Warm cache with commonly accessed data."""
        for func_info in warm_functions:
            try:
                func = func_info['function']
                args = func_info.get('args', ())
                kwargs = func_info.get('kwargs', {})
                
                # Execute function to warm cache
                await func(*args, **kwargs)
                
            except Exception as e:
                # Log error but continue warming other functions
                print(f"Cache warming failed for {func_info}: {e}")
    
    async def health_check(self) -> dict:
        """Perform health check on cache system."""
        try:
            # Test basic operations
            test_key = f"{self._cache_prefix}health_check"
            test_value = {"timestamp": datetime.utcnow().isoformat()}
            
            await self.set(test_key, test_value, ttl=60)
            retrieved = await self.get(test_key)
            await self.delete(test_key)
            
            return {
                'status': 'healthy',
                'redis_connected': True,
                'read_write_test': retrieved == test_value,
                'stats': await self.get_stats()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'redis_connected': False,
                'error': str(e)
            }


# Global cache instance
cache = CacheManager()


# Usage examples and helper functions
async def get_concepts_by_domain_cached(domain: str, limit: int = 100):
    """Example cached function for Neo4j queries."""
    @cache.cached(ttl=300, key_prefix="graph_concepts")
    async def _get_concepts(domain: str, limit: int):
        # This would be replaced with actual Neo4j query
        return {"domain": domain, "limit": limit, "concepts": []}
    
    return await _get_concepts(domain, limit)


async def semantic_search_cached(query_vector, limit: int = 10):
    """Example cached function for Qdrant searches."""
    @cache.cached(ttl=600, key_prefix="vector_search")
    async def _semantic_search(query_vector, limit: int):
        # This would be replaced with actual Qdrant search
        return {"query": "vector_search", "limit": limit, "results": []}
    
    return await _semantic_search(query_vector, limit)


async def analytics_computation_cached(computation_type: str, parameters: dict):
    """Example cached function for analytics computations."""
    @cache.cached(ttl=3600, key_prefix="analytics_computation")
    async def _compute_analytics(computation_type: str, parameters: dict):
        # This would be replaced with actual analytics computation
        return {"type": computation_type, "parameters": parameters, "result": {}}
    
    return await _compute_analytics(computation_type, parameters)


# Cache invalidation helpers
async def invalidate_domain_cache(domain: str):
    """Invalidate all cache entries for a specific domain."""
    await cache.delete_pattern(f"graph_concepts:*{domain}*")
    await cache.delete_pattern(f"vector_search:*{domain}*")


async def invalidate_all_analytics_cache():
    """Invalidate all analytics cache entries."""
    await cache.delete_pattern("analytics_computation:*")


# Cache warming configuration
CACHE_WARMING_FUNCTIONS = [
    {
        'function': get_concepts_by_domain_cached,
        'args': ('philosophy',),
        'kwargs': {'limit': 50}
    },
    {
        'function': get_concepts_by_domain_cached,
        'args': ('science',),
        'kwargs': {'limit': 50}
    },
    # Add more warming functions as needed
]


async def warm_system_cache():
    """Warm the cache with commonly accessed data."""
    await cache.warm_cache(CACHE_WARMING_FUNCTIONS)