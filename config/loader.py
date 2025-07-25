"""
Configuration loader utilities for MCP Yggdrasil.
Handles environment-specific configuration loading and validation.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .settings import settings, feature_flags

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and merge configuration from various sources."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.environment = settings.environment
        self._cache = {}
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.config_dir / filename
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f) or {}
                self._cache[filename] = config
                return config
        except Exception as e:
            logger.error(f"Error loading config file {filepath}: {e}")
            return {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files."""
        configs = {}
        yaml_files = self.config_dir.glob("*.yaml")
        
        for yaml_file in yaml_files:
            config_name = yaml_file.stem
            configs[config_name] = self.load_yaml(yaml_file.name)
        
        return configs
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        # Try to load from specific agent config file
        agent_config = self.load_yaml(f"{agent_name}.yaml")
        
        # Fall back to general agent configs
        if not agent_config:
            all_agents = self.load_yaml("database_agents.yaml")
            agent_config = all_agents.get(agent_name, {})
        
        # Apply environment-specific overrides
        if self.environment in agent_config:
            base_config = agent_config.copy()
            base_config.update(agent_config[self.environment])
            return base_config
        
        return agent_config
    
    def get_database_config(self, db_type: str) -> Dict[str, Any]:
        """Get database-specific configuration."""
        if db_type == "neo4j":
            return {
                "uri": settings.neo4j_uri,
                "user": settings.neo4j_user,
                "password": settings.neo4j_password,
                "database": settings.neo4j_database,
                "max_connection_lifetime": settings.neo4j_max_connection_lifetime,
                "max_connection_pool_size": settings.neo4j_max_connection_pool_size
            }
        elif db_type == "qdrant":
            return {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "api_key": settings.qdrant_api_key,
                "https": settings.qdrant_https,
                "timeout": settings.qdrant_timeout
            }
        elif db_type == "redis":
            return {
                "url": settings.redis_url,
                "password": settings.redis_password,
                "max_connections": settings.redis_max_connections,
                "decode_responses": settings.redis_decode_responses
            }
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get all configured API keys."""
        return {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "google": settings.google_api_key,
            "youtube": settings.youtube_api_key
        }
    
    def validate_required_settings(self) -> bool:
        """Validate that all required settings are present."""
        required = [
            ("SECRET_KEY", settings.secret_key),
            ("NEO4J_PASSWORD", settings.neo4j_password)
        ]
        
        missing = []
        for name, value in required:
            if not value:
                missing.append(name)
        
        if missing:
            logger.error(f"Missing required settings: {', '.join(missing)}")
            return False
        
        return True


# Global config loader instance
config_loader = ConfigLoader()


def get_config(config_type: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration by type and optional name.
    
    Args:
        config_type: Type of config (agent, database, api_keys, etc.)
        name: Optional specific name within the type
    
    Returns:
        Configuration dictionary
    """
    if config_type == "agent" and name:
        return config_loader.get_agent_config(name)
    elif config_type == "database" and name:
        return config_loader.get_database_config(name)
    elif config_type == "api_keys":
        return config_loader.get_api_keys()
    elif config_type == "cache":
        return settings.cache_config
    elif config_type == "all":
        return config_loader.load_all_configs()
    else:
        return config_loader.load_yaml(f"{config_type}.yaml")


def validate_environment() -> bool:
    """Validate the current environment configuration."""
    if not config_loader.validate_required_settings():
        return False
    
    # Check feature flag consistency
    if settings.is_production and feature_flags.is_enabled('debug_mode'):
        logger.warning("Debug mode is enabled in production!")
    
    # Check database connectivity settings
    if not settings.neo4j_uri or not settings.redis_url:
        logger.error("Database connection settings are missing")
        return False
    
    logger.info(f"Configuration validated for {settings.environment} environment")
    return True