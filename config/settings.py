"""
Centralized configuration management for MCP Yggdrasil.
Environment-based settings with validation and feature flags.
"""
from pydantic import BaseSettings, validator, Field
from typing import Optional, Dict, Any, List
import os
from functools import lru_cache
from datetime import timedelta


class Settings(BaseSettings):
    """Main application settings with environment variable support."""
    
    # Application
    app_name: str = "MCP Yggdrasil"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENV")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    api_title: str = "MCP Yggdrasil API"
    api_description: str = "Knowledge management and content processing API"
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE")
    
    # Database - Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    neo4j_max_connection_lifetime: int = Field(default=3600, env="NEO4J_MAX_CONN_LIFETIME")
    neo4j_max_connection_pool_size: int = Field(default=50, env="NEO4J_POOL_SIZE")
    
    # Database - Qdrant
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_https: bool = Field(default=False, env="QDRANT_HTTPS")
    qdrant_timeout: int = Field(default=5, env="QDRANT_TIMEOUT")
    
    # Database - Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    redis_decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")
    
    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    
    # Message Queue - RabbitMQ
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        env="RABBITMQ_URL"
    )
    rabbitmq_exchange: str = Field(default="mcp_events", env="RABBITMQ_EXCHANGE")
    
    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT")
    celery_task_serializer: str = Field(default="json", env="CELERY_SERIALIZER")
    celery_accept_content: List[str] = Field(default=["json"], env="CELERY_ACCEPT")
    celery_timezone: str = Field(default="UTC", env="CELERY_TIMEZONE")
    
    # Performance & Caching
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # seconds
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring & Observability
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    jaeger_host: str = Field(default="localhost", env="JAEGER_HOST")
    jaeger_port: int = Field(default=6831, env="JAEGER_PORT")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    
    # File Storage
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    max_upload_size: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=["pdf", "txt", "md", "json", "csv"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Scraping Configuration
    scraping_timeout: int = Field(default=30, env="SCRAPING_TIMEOUT")
    max_concurrent_scrapers: int = Field(default=5, env="MAX_SCRAPERS")
    user_agent: str = Field(
        default="MCP-Yggdrasil/1.0 (+https://github.com/yggdrasil)",
        env="USER_AGENT"
    )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment is one of allowed values."""
        allowed = ['development', 'staging', 'production', 'testing']
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string if needed."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    @property
    def database_url(self) -> str:
        """Get the primary database URL (Neo4j)."""
        return self.neo4j_uri
    
    @property
    def cache_config(self) -> Dict[str, Any]:
        """Get cache configuration as dict."""
        return {
            'enabled': self.enable_caching,
            'ttl': self.cache_ttl,
            'max_size': self.cache_max_size,
            'redis_url': self.redis_url,
            'redis_password': self.redis_password
        }
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
        # Allow loading from multiple env files based on environment
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


class FeatureFlags:
    """Feature flag management for gradual rollout and A/B testing."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.flags = self._initialize_flags()
    
    def _initialize_flags(self) -> Dict[str, bool]:
        """Initialize feature flags based on environment."""
        base_flags = {
            # Core features
            'multi_llm_support': True,
            'advanced_analytics': True,
            'real_time_sync': True,
            'event_driven_architecture': True,
            
            # Security features
            'mfa_authentication': True,
            'rbac_system': True,
            'data_encryption': True,
            'pii_detection': True,
            
            # API features
            'graphql_api': True,
            'websocket_support': True,
            'api_versioning': True,
            
            # Performance features
            'auto_scaling': False,
            'distributed_caching': True,
            'query_optimization': True,
            
            # Experimental features
            'ai_recommendations': False,
            'voice_interface': False,
            'ar_visualization': False,
            
            # Debug features
            'debug_mode': self.environment == 'development',
            'profiling': self.environment == 'development',
            'verbose_logging': self.environment != 'production'
        }
        
        # Production-specific overrides
        if self.environment == 'production':
            base_flags.update({
                'auto_scaling': True,
                'debug_mode': False,
                'profiling': False
            })
        
        return base_flags
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.flags.get(feature, False)
    
    def enable(self, feature: str):
        """Enable a feature flag."""
        self.flags[feature] = True
    
    def disable(self, feature: str):
        """Disable a feature flag."""
        self.flags[feature] = False
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return self.flags.copy()
    
    def set_flags(self, flags: Dict[str, bool]):
        """Bulk update feature flags."""
        self.flags.update(flags)


# Singleton pattern for settings
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton pattern for feature flags
@lru_cache()
def get_feature_flags() -> FeatureFlags:
    """Get cached feature flags instance."""
    settings = get_settings()
    return FeatureFlags(environment=settings.environment)


# Convenience exports
settings = get_settings()
feature_flags = get_feature_flags()