"""
Configuration and State Management for Streamlit Dashboard

This module handles dashboard configuration, state management, and agent initialization
following the established modular architecture patterns.

Key Features:
- Dashboard configuration management with environment variable support
- Streamlit session state management and initialization
- Agent initialization and lifecycle management
- Performance settings and optimization configuration
- Error handling and logging integration

Author: MCP Yggdrasil Analytics Team
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from agents.anomaly_detector.detector import AnomalyDetector
from agents.knowledge_graph.graph_builder import GraphBuilder
from agents.maintenance.maintainer import DatabaseMaintainer
from agents.pattern_recognition.pattern_analyzer import PatternAnalyzer
from agents.recommendation.recommender import RecommendationEngine

# Import agents
from agents.scraper.scraper import WebScraper
from agents.text_processor.processor import TextProcessor
from agents.vector_index.indexer import VectorIndexer

logger = logging.getLogger(__name__)


class DashboardConfig:
    """
    Configuration management for the Streamlit dashboard.

    Handles database connections, dashboard settings, performance configuration,
    and environment-specific settings with fallback defaults.
    """

    def __init__(self):
        """Initialize dashboard configuration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Database configuration
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.redis_url = "redis://localhost:6379"

        # Dashboard settings
        self.refresh_interval = 30  # seconds
        self.max_display_items = 100
        self.chart_height = 400
        self.enable_real_time = True
        self.auto_save_interval = 300  # 5 minutes

        # Performance settings
        self.cache_ttl = 3600  # 1 hour
        self.max_concurrent_operations = 5
        self.query_timeout = 30  # seconds
        self.batch_size = 1000

        # UI settings
        self.theme = "default"
        self.show_debug_info = False
        self.enable_notifications = True
        self.compact_mode = False

        # Load configuration from files and environment
        self._load_config()

    def _load_config(self):
        """Load configuration from config files and environment variables."""
        try:
            # Try to load from config file
            config_path = project_root / "config" / "dashboard.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    self._apply_config_data(config_data)

            # Override with environment variables
            self._load_from_environment()

            self.logger.info("Dashboard configuration loaded successfully")

        except Exception as e:
            self.logger.warning(f"Error loading dashboard config: {e}, using defaults")

    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data from YAML file."""
        if "database" in config_data:
            db_config = config_data["database"]
            self.neo4j_uri = db_config.get("neo4j_uri", self.neo4j_uri)
            self.neo4j_user = db_config.get("neo4j_user", self.neo4j_user)
            self.neo4j_password = db_config.get("neo4j_password", self.neo4j_password)
            self.qdrant_host = db_config.get("qdrant_host", self.qdrant_host)
            self.qdrant_port = db_config.get("qdrant_port", self.qdrant_port)
            self.redis_url = db_config.get("redis_url", self.redis_url)

        if "dashboard" in config_data:
            dash_config = config_data["dashboard"]
            self.refresh_interval = dash_config.get(
                "refresh_interval", self.refresh_interval
            )
            self.max_display_items = dash_config.get(
                "max_display_items", self.max_display_items
            )
            self.chart_height = dash_config.get("chart_height", self.chart_height)
            self.enable_real_time = dash_config.get(
                "enable_real_time", self.enable_real_time
            )

        if "performance" in config_data:
            perf_config = config_data["performance"]
            self.cache_ttl = perf_config.get("cache_ttl", self.cache_ttl)
            self.max_concurrent_operations = perf_config.get(
                "max_concurrent_operations", self.max_concurrent_operations
            )
            self.query_timeout = perf_config.get("query_timeout", self.query_timeout)
            self.batch_size = perf_config.get("batch_size", self.batch_size)

    def _load_from_environment(self):
        """Load configuration from environment variables."""
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.qdrant_host = os.getenv("QDRANT_HOST", self.qdrant_host)
        self.qdrant_port = int(os.getenv("QDRANT_PORT", str(self.qdrant_port)))
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)

        # Performance settings from environment
        self.cache_ttl = int(os.getenv("DASHBOARD_CACHE_TTL", str(self.cache_ttl)))
        self.query_timeout = int(
            os.getenv("DASHBOARD_QUERY_TIMEOUT", str(self.query_timeout))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "database": {
                "neo4j_uri": self.neo4j_uri,
                "neo4j_user": self.neo4j_user,
                "qdrant_host": self.qdrant_host,
                "qdrant_port": self.qdrant_port,
                "redis_url": self.redis_url,
            },
            "dashboard": {
                "refresh_interval": self.refresh_interval,
                "max_display_items": self.max_display_items,
                "chart_height": self.chart_height,
                "enable_real_time": self.enable_real_time,
                "theme": self.theme,
            },
            "performance": {
                "cache_ttl": self.cache_ttl,
                "max_concurrent_operations": self.max_concurrent_operations,
                "query_timeout": self.query_timeout,
                "batch_size": self.batch_size,
            },
        }


class DashboardState:
    """
    Dashboard state management with agent initialization and lifecycle control.

    Manages Streamlit session state, agent instances, and application state
    across user sessions with proper cleanup and error handling.
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard state management."""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Agent instances
        self.agents = {}
        self.agent_status = {}

        # Application state
        self.initialized = False
        self.last_refresh = None
        self.error_count = 0
        self.performance_metrics = {
            "page_loads": 0,
            "query_count": 0,
            "avg_response_time": 0,
        }

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values."""
        try:
            # Core application state
            if "initialized" not in st.session_state:
                st.session_state.initialized = False

            if "dashboard_config" not in st.session_state:
                st.session_state.dashboard_config = self.config.to_dict()

            if "current_page" not in st.session_state:
                st.session_state.current_page = "Overview"

            # User preferences
            if "user_preferences" not in st.session_state:
                st.session_state.user_preferences = {
                    "theme": self.config.theme,
                    "compact_mode": self.config.compact_mode,
                    "auto_refresh": self.config.enable_real_time,
                    "show_debug": self.config.show_debug_info,
                }

            # Data state
            if "selected_domain" not in st.session_state:
                st.session_state.selected_domain = "all"

            if "query_history" not in st.session_state:
                st.session_state.query_history = []

            if "upload_state" not in st.session_state:
                st.session_state.upload_state = {
                    "files_processed": 0,
                    "current_operation": None,
                    "progress": 0,
                }

            # Performance tracking
            if "performance_metrics" not in st.session_state:
                st.session_state.performance_metrics = self.performance_metrics.copy()

            self.logger.debug("Session state initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing session state: {e}")

    def initialize_agents(self) -> Dict[str, Any]:
        """
        Initialize all dashboard agents with proper error handling.

        Returns:
            Dictionary of initialized agent instances
        """
        try:
            agents = {}

            # Initialize each agent with error handling
            agent_configs = {
                "scraper": (WebScraper, "Web scraping agent"),
                "text_processor": (TextProcessor, "Text processing agent"),
                "graph_builder": (GraphBuilder, "Knowledge graph builder"),
                "vector_indexer": (VectorIndexer, "Vector indexing agent"),
                "pattern_analyzer": (PatternAnalyzer, "Pattern recognition agent"),
                "maintainer": (DatabaseMaintainer, "Database maintenance agent"),
                "anomaly_detector": (AnomalyDetector, "Anomaly detection agent"),
                "recommender": (RecommendationEngine, "Recommendation engine"),
            }

            for agent_name, (agent_class, description) in agent_configs.items():
                try:
                    agents[agent_name] = agent_class()
                    self.agent_status[agent_name] = {
                        "status": "active",
                        "last_used": datetime.now(),
                        "error_count": 0,
                        "description": description,
                    }
                    self.logger.debug(f"Initialized {agent_name}: {description}")

                except Exception as agent_error:
                    self.logger.error(
                        f"Failed to initialize {agent_name}: {agent_error}"
                    )
                    self.agent_status[agent_name] = {
                        "status": "failed",
                        "error": str(agent_error),
                        "error_count": 1,
                        "description": description,
                    }

            self.agents = agents
            self.logger.info(
                f"Agent initialization complete: {len(agents)}/{len(agent_configs)} agents active"
            )

            return agents

        except Exception as e:
            self.logger.error(f"Critical error during agent initialization: {e}")
            return {}

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all agents."""
        return self.agent_status.copy()

    def update_performance_metrics(self, metric_name: str, value: Any):
        """Update performance metrics."""
        try:
            if metric_name in self.performance_metrics:
                if metric_name == "avg_response_time":
                    # Calculate running average
                    current_avg = self.performance_metrics[metric_name]
                    query_count = self.performance_metrics.get("query_count", 1)
                    self.performance_metrics[metric_name] = (
                        current_avg * (query_count - 1) + value
                    ) / query_count
                else:
                    self.performance_metrics[metric_name] = value

                # Update session state
                st.session_state.performance_metrics = self.performance_metrics.copy()

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def cleanup_agents(self):
        """Clean up agent resources."""
        try:
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "cleanup"):
                    agent.cleanup()
                self.agent_status[agent_name]["status"] = "stopped"

            self.logger.info("Agent cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during agent cleanup: {e}")


# Global dashboard state instance
_dashboard_state = None


def get_dashboard_state() -> DashboardState:
    """
    Get or create the global dashboard state instance.

    Returns:
        DashboardState instance
    """
    global _dashboard_state
    if _dashboard_state is None:
        _dashboard_state = DashboardState()
    return _dashboard_state


def initialize_dashboard_config(config_path: Optional[Path] = None) -> DashboardConfig:
    """
    Initialize dashboard configuration from file or defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        DashboardConfig instance
    """
    try:
        config = DashboardConfig()
        if config_path and config_path.exists():
            # Load specific config file if provided
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                config._apply_config_data(config_data)

        return config

    except Exception as e:
        logger.error(f"Error initializing dashboard config: {e}")
        return DashboardConfig()  # Return defaults


# Factory functions for easy instantiation
def create_dashboard_config(config_path: Optional[Path] = None) -> DashboardConfig:
    """Create and configure a DashboardConfig instance."""
    return initialize_dashboard_config(config_path)


def create_dashboard_state(config: Optional[DashboardConfig] = None) -> DashboardState:
    """Create and configure a DashboardState instance."""
    return DashboardState(config)


# Export main classes and functions
__all__ = [
    "DashboardConfig",
    "DashboardState",
    "get_dashboard_state",
    "initialize_dashboard_config",
    "create_dashboard_config",
    "create_dashboard_state",
]
