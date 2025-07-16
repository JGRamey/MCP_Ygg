"""
Performance Optimization Module for MCP Server
Implements caching, query optimization, and performance monitoring.
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from pathlib import Path
from collections import defaultdict, OrderedDict
import weakref

import redis.asyncio as redis
import numpy as np
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient


T = TypeVar('T')


class CacheType(Enum):
    """Types of caching strategies."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class CacheEntry:
    """Represents a cached entry."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_name: str
    execution_time: float
    cache_hit: bool
    cache_key: Optional[str]
    timestamp: datetime
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at:
                    del self.cache[key]
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc)
                self._hits += 1
                return entry.value
            
            self._misses += 1
            return None
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in cache."""
        async with self._lock:
            ttl = ttl_seconds or self.ttl_seconds
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl) if ttl > 0 else None
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict oldest entries if over max size
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size
        }


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "mcp:"):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client: Optional[redis.Redis] = None
        self._hits = 0
        self._misses = 0
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is not None:
                value = pickle.loads(data)
                self._hits += 1
                return value
            
            self._misses += 1
            return None
            
        except Exception as e:
            logging.warning(f"Redis cache get error: {e}")
            self._misses += 1
            return None
    
    async def put(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Put value in Redis cache."""
        if not self.redis_client:
            return
        
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            
            if ttl_seconds > 0:
                await self.redis_client.setex(redis_key, ttl_seconds, data)
            else:
                await self.redis_client.set(redis_key, data)
                
        except Exception as e:
            logging.warning(f"Redis cache put error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logging.warning(f"Redis cache delete error: {e}")
            return False
    
    async def clear_prefix(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            pattern_key = self._make_key(pattern)
            keys = await self.redis_client.keys(pattern_key)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logging.warning(f"Redis cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "redis_connected": self.redis_client is not None
        }


class HybridCache:
    """Hybrid cache using both memory and Redis."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        redis_url: str = "redis://localhost:6379",
        l1_ttl: int = 300,  # L1 (memory) TTL
        l2_ttl: int = 3600  # L2 (Redis) TTL
    ):
        """Initialize hybrid cache."""
        self.l1_cache = LRUCache(max_size=memory_cache_size, ttl_seconds=l1_ttl)
        self.l2_cache = RedisCache(redis_url=redis_url)
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
    
    async def initialize(self) -> None:
        """Initialize cache layers."""
        await self.l2_cache.initialize()
    
    async def close(self) -> None:
        """Close cache connections."""
        await self.l2_cache.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache."""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Store in L1 cache for faster access
            await self.l1_cache.put(key, value, self.l1_ttl)
            return value
        
        return None
    
    async def put(self, key: str, value: Any) -> None:
        """Put value in hybrid cache."""
        # Store in both layers
        await self.l1_cache.put(key, value, self.l1_ttl)
        await self.l2_cache.put(key, value, self.l2_ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from both cache layers."""
        l1_result = await self.l1_cache.delete(key)
        l2_result = await self.l2_cache.delete(key)
        return l1_result or l2_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        return {
            "l1_cache": l1_stats,
            "l2_cache": l2_stats
        }


class QueryOptimizer:
    """Optimizes database queries for better performance."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_stats: Dict[str, List[float]] = defaultdict(list)
        self.optimized_queries: Dict[str, str] = {}
        self.index_recommendations: List[str] = []
    
    def optimize_neo4j_query(self, query: str, params: Dict[str, Any]) -> str:
        """Optimize Neo4j Cypher query."""
        optimized = query
        
        # Add query hints and optimizations
        optimizations = [
            self._add_index_hints,
            self._optimize_match_patterns,
            self._optimize_where_clauses,
            self._add_limits_early,
            self._optimize_aggregations
        ]
        
        for optimization in optimizations:
            optimized = optimization(optimized, params)
        
        return optimized
    
    def _add_index_hints(self, query: str, params: Dict[str, Any]) -> str:
        """Add index hints to improve performance."""
        # Look for common patterns that can benefit from indexes
        if "MATCH (n:Document)" in query and "n.domain" in query:
            # Suggest using index on domain
            if "USING INDEX" not in query:
                query = query.replace(
                    "MATCH (n:Document)",
                    "MATCH (n:Document) USING INDEX n:Document(domain)"
                )
        
        if "MATCH (p:Person)" in query and "p.name" in query:
            if "USING INDEX" not in query:
                query = query.replace(
                    "MATCH (p:Person)",
                    "MATCH (p:Person) USING INDEX p:Person(name)"
                )
        
        return query
    
    def _optimize_match_patterns(self, query: str, params: Dict[str, Any]) -> str:
        """Optimize MATCH patterns for better performance."""
        # Reorder MATCH clauses to start with most selective
        lines = query.split('\n')
        match_lines = [line.strip() for line in lines if line.strip().startswith('MATCH')]
        
        if len(match_lines) > 1:
            # Sort by selectivity (simple heuristic)
            def selectivity_score(match_line: str) -> int:
                score = 0
                if 'id(' in match_line:
                    score += 10  # ID lookups are most selective
                if any(prop in match_line for prop in ['domain', 'name', 'title']):
                    score += 5  # Property filters are selective
                if ':' in match_line:
                    score += 2  # Label filters are somewhat selective
                return score
            
            match_lines.sort(key=selectivity_score, reverse=True)
            
            # Rebuild query with optimized order
            non_match_lines = [line for line in lines if not line.strip().startswith('MATCH')]
            optimized_lines = match_lines + non_match_lines
            query = '\n'.join(optimized_lines)
        
        return query
    
    def _optimize_where_clauses(self, query: str, params: Dict[str, Any]) -> str:
        """Optimize WHERE clauses."""
        # Move more selective conditions first
        if 'WHERE' in query:
            # This is a simplified optimization
            # In practice, you'd analyze the query tree more thoroughly
            pass
        
        return query
    
    def _add_limits_early(self, query: str, params: Dict[str, Any]) -> str:
        """Add LIMIT clauses early when possible."""
        # If there's a LIMIT at the end, try to push it up
        if query.strip().endswith(') LIMIT'):
            # Look for opportunities to add WITH ... LIMIT earlier
            pass
        
        return query
    
    def _optimize_aggregations(self, query: str, params: Dict[str, Any]) -> str:
        """Optimize aggregation queries."""
        # Use more efficient aggregation patterns
        if 'count(' in query.lower():
            # Suggest using EXISTS for existence checks
            query = query.replace(
                'WHERE count(',
                'WHERE EXISTS {'
            )
        
        return query
    
    def record_query_performance(self, query_hash: str, execution_time: float) -> None:
        """Record query performance for analysis."""
        self.query_stats[query_hash].append(execution_time)
        
        # Keep only recent measurements
        if len(self.query_stats[query_hash]) > 100:
            self.query_stats[query_hash] = self.query_stats[query_hash][-100:]
    
    def get_slow_queries(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Get queries that are consistently slow."""
        slow_queries = []
        
        for query_hash, times in self.query_stats.items():
            if len(times) >= 5:  # Need enough samples
                avg_time = np.mean(times) * 1000  # Convert to ms
                if avg_time > threshold_ms:
                    slow_queries.append({
                        "query_hash": query_hash,
                        "avg_time_ms": avg_time,
                        "sample_count": len(times),
                        "max_time_ms": max(times) * 1000,
                        "min_time_ms": min(times) * 1000
                    })
        
        return sorted(slow_queries, key=lambda x: x["avg_time_ms"], reverse=True)


class PerformanceMonitor:
    """Monitors and collects performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = max_metrics
        self.operation_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "total_time": 0.0,
            "count": 0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        })
    
    def record_operation(
        self,
        operation_name: str,
        execution_time: float,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ) -> None:
        """Record performance metrics for an operation."""
        
        metric = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            cache_hit=cache_hit,
            cache_key=cache_key,
            timestamp=datetime.now(timezone.utc),
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        self.metrics.append(metric)
        
        # Update operation stats
        stats = self.operation_stats[operation_name]
        stats["total_time"] += execution_time
        stats["count"] += 1
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        
        if cache_hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1
        
        # Trim metrics if too many
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operations."""
        if operation_name:
            if operation_name in self.operation_stats:
                stats = self.operation_stats[operation_name].copy()
                if stats["count"] > 0:
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
                    stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests if total_cache_requests > 0 else 0
                return stats
            return {}
        
        # Return all operation stats
        all_stats = {}
        for op_name, stats in self.operation_stats.items():
            op_stats = stats.copy()
            if op_stats["count"] > 0:
                op_stats["avg_time"] = op_stats["total_time"] / op_stats["count"]
                total_cache_requests = op_stats["cache_hits"] + op_stats["cache_misses"]
                op_stats["cache_hit_rate"] = op_stats["cache_hits"] / total_cache_requests if total_cache_requests > 0 else 0
            all_stats[op_name] = op_stats
        
        return all_stats
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp > cutoff_time]
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file."""
        metrics_data = [asdict(m) for m in self.metrics]
        
        # Convert datetime objects to strings
        for metric in metrics_data:
            metric["timestamp"] = metric["timestamp"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)


def cached(
    cache: Union[LRUCache, RedisCache, HybridCache],
    key_func: Optional[Callable[..., str]] = None,
    ttl_seconds: int = 3600
):
    """Decorator for caching function results."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.put(cache_key, result, ttl_seconds)
            return result
        
        return wrapper
    return decorator


def timed(monitor: PerformanceMonitor, operation_name: Optional[str] = None):
    """Decorator for timing function execution."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitor.record_operation(op_name, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitor.record_operation(f"{op_name}_error", execution_time)
                raise
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(
        self,
        cache_type: CacheType = CacheType.HYBRID,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        redis_url: str = "redis://localhost:6379"
    ):
        """Initialize performance optimizer."""
        self.cache_type = cache_type
        self.optimization_level = optimization_level
        
        # Initialize cache
        if cache_type == CacheType.MEMORY:
            self.cache = LRUCache(max_size=10000)
        elif cache_type == CacheType.REDIS:
            self.cache = RedisCache(redis_url=redis_url)
        else:  # HYBRID
            self.cache = HybridCache(redis_url=redis_url)
        
        self.query_optimizer = QueryOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Optimization settings
        self.query_cache_ttl = self._get_cache_ttl()
        self.batch_sizes = self._get_batch_sizes()
        self.connection_pools = self._get_connection_pool_sizes()
    
    def _get_cache_ttl(self) -> Dict[str, int]:
        """Get cache TTL settings based on optimization level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            return {
                "query_results": 300,
                "embeddings": 3600,
                "graph_metrics": 1800,
                "recommendations": 600
            }
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return {
                "query_results": 600,
                "embeddings": 7200,
                "graph_metrics": 3600,
                "recommendations": 1800
            }
        else:  # EXTREME
            return {
                "query_results": 1800,
                "embeddings": 14400,
                "graph_metrics": 7200,
                "recommendations": 3600
            }
    
    def _get_batch_sizes(self) -> Dict[str, int]:
        """Get batch sizes based on optimization level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            return {
                "document_processing": 10,
                "graph_updates": 50,
                "vector_updates": 100,
                "pattern_analysis": 20
            }
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return {
                "document_processing": 25,
                "graph_updates": 100,
                "vector_updates": 250,
                "pattern_analysis": 50
            }
        else:  # EXTREME
            return {
                "document_processing": 50,
                "graph_updates": 200,
                "vector_updates": 500,
                "pattern_analysis": 100
            }
    
    def _get_connection_pool_sizes(self) -> Dict[str, int]:
        """Get connection pool sizes based on optimization level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            return {
                "neo4j_pool_size": 10,
                "qdrant_pool_size": 5,
                "redis_pool_size": 5
            }
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return {
                "neo4j_pool_size": 20,
                "qdrant_pool_size": 10,
                "redis_pool_size": 10
            }
        else:  # EXTREME
            return {
                "neo4j_pool_size": 50,
                "qdrant_pool_size": 25,
                "redis_pool_size": 20
            }
    
    async def initialize(self) -> None:
        """Initialize performance optimizer."""
        if hasattr(self.cache, 'initialize'):
            await self.cache.initialize()
    
    async def close(self) -> None:
        """Close performance optimizer."""
        if hasattr(self.cache, 'close'):
            await self.cache.close()
    
    def get_cached_query_decorator(self, ttl_seconds: Optional[int] = None):
        """Get decorator for caching query results."""
        ttl = ttl_seconds or self.query_cache_ttl["query_results"]
        
        def key_func(query: str, params: Dict[str, Any]) -> str:
            return hashlib.md5(f"{query}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        return cached(self.cache, key_func, ttl)
    
    def get_timed_decorator(self, operation_name: Optional[str] = None):
        """Get decorator for timing operations."""
        return timed(self.performance_monitor, operation_name)
    
    def optimize_query(self, query: str, params: Dict[str, Any]) -> str:
        """Optimize a database query."""
        return self.query_optimizer.optimize_neo4j_query(query, params)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "cache_stats": self.cache.get_stats(),
            "operation_stats": self.performance_monitor.get_operation_stats(),
            "slow_queries": self.query_optimizer.get_slow_queries(),
            "optimization_settings": {
                "cache_type": self.cache_type.value,
                "optimization_level": self.optimization_level.value,
                "cache_ttl": self.query_cache_ttl,
                "batch_sizes": self.batch_sizes,
                "connection_pools": self.connection_pools
            }
        }
    
    async def warm_cache(self, warm_up_queries: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Warm up cache with commonly used queries."""
        for query, params in warm_up_queries:
            cache_key = hashlib.md5(f"{query}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
            
            # Check if already cached
            cached_result = await self.cache.get(cache_key)
            if cached_result is None:
                # Execute query and cache result (this would be done by the actual query execution)
                # For now, we just log that we would warm this up
                logging.info(f"Would warm cache for query: {query[:50]}...")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for further optimization."""
        recommendations = []
        
        # Analyze cache hit rates
        cache_stats = self.cache.get_stats()
        if isinstance(cache_stats, dict) and "hit_rate" in cache_stats:
            hit_rate = cache_stats["hit_rate"]
            if hit_rate < 0.5:
                recommendations.append("Consider increasing cache TTL or size - low hit rate detected")
        
        # Analyze slow queries
        slow_queries = self.query_optimizer.get_slow_queries()
        if len(slow_queries) > 5:
            recommendations.append("Multiple slow queries detected - consider query optimization or indexing")
        
        # Analyze operation performance
        op_stats = self.performance_monitor.get_operation_stats()
        for op_name, stats in op_stats.items():
            if stats.get("avg_time", 0) > 5.0:  # 5+ seconds average
                recommendations.append(f"Operation '{op_name}' is slow - consider optimization")
        
        if not recommendations:
            recommendations.append("System performance looks good - no major optimizations needed")
        
        return recommendations


# Example usage and testing
async def example_usage():
    """Example of how to use the performance optimization components."""
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        cache_type=CacheType.HYBRID,
        optimization_level=OptimizationLevel.AGGRESSIVE
    )
    
    await optimizer.initialize()
    
    try:
        # Example cached function
        @optimizer.get_cached_query_decorator(ttl_seconds=600)
        async def expensive_query(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Simulate expensive database operation
            await asyncio.sleep(0.1)
            return [{"result": "data"}]
        
        # Example timed function
        @optimizer.get_timed_decorator("document_processing")
        async def process_document(doc_id: str) -> Dict[str, Any]:
            # Simulate document processing
            await asyncio.sleep(0.05)
            return {"processed": doc_id}
        
        # Test caching
        start_time = time.time()
        result1 = await expensive_query("SELECT * FROM documents", {"limit": 10})
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = await expensive_query("SELECT * FROM documents", {"limit": 10})  # Should hit cache
        second_call_time = time.time() - start_time
        
        print(f"First call: {first_call_time:.3f}s")
        print(f"Second call: {second_call_time:.3f}s (cached)")
        print(f"Cache speedup: {first_call_time / second_call_time:.1f}x")
        
        # Test performance monitoring
        for i in range(5):
            await process_document(f"doc_{i}")
        
        # Get performance report
        report = optimizer.get_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2, default=str))
        
        # Get optimization recommendations
        recommendations = optimizer.get_optimization_recommendations()
        print("\nOptimization Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    
    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
