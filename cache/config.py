"""Cache configuration settings."""

import os
from typing import Any, Dict

# Default cache configuration
DEFAULT_CACHE_CONFIG = {
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "default_ttl": 300,  # 5 minutes
    "key_prefix": "mcp:",
    "max_connections": 10,
    "health_check_interval": 30,  # seconds
    "cache_warming_enabled": True,
    "metrics_enabled": True,
}

# TTL configurations for different types of data
CACHE_TTL_CONFIG = {
    "graph_concepts": 300,  # 5 minutes
    "graph_relationships": 600,  # 10 minutes
    "vector_search": 600,  # 10 minutes
    "analytics_computation": 3600,  # 1 hour
    "api_responses": 300,  # 5 minutes
    "user_sessions": 1800,  # 30 minutes
    "system_stats": 60,  # 1 minute
    "file_metadata": 1800,  # 30 minutes
    "scraping_results": 3600,  # 1 hour
    "claim_verification": 7200,  # 2 hours
}

# Cache warming settings
CACHE_WARMING_CONFIG = {
    "enabled": True,
    "startup_delay": 30,  # seconds to wait before warming
    "batch_size": 10,  # number of items to warm at once
    "timeout": 60,  # timeout per warming operation
    "retry_attempts": 3,
    "retry_delay": 5,  # seconds between retries
}

# Performance monitoring settings
PERFORMANCE_CONFIG = {
    "slow_query_threshold": 1.0,  # seconds
    "memory_usage_alert": 0.8,  # 80% memory usage
    "hit_rate_alert": 0.7,  # Alert if hit rate below 70%
    "connection_pool_size": 20,
    "connection_timeout": 30,  # seconds
}


def get_cache_config() -> Dict[str, Any]:
    """Get the complete cache configuration."""
    return {
        **DEFAULT_CACHE_CONFIG,
        "ttl": CACHE_TTL_CONFIG,
        "warming": CACHE_WARMING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
    }


def get_ttl_for_key_type(key_type: str) -> int:
    """Get TTL for a specific key type."""
    return CACHE_TTL_CONFIG.get(key_type, DEFAULT_CACHE_CONFIG["default_ttl"])


def is_cache_warming_enabled() -> bool:
    """Check if cache warming is enabled."""
    return CACHE_WARMING_CONFIG["enabled"]


def get_redis_url() -> str:
    """Get Redis connection URL."""
    return DEFAULT_CACHE_CONFIG["redis_url"]
