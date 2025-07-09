"""Configuration management for anomaly detection."""
import logging
from pathlib import Path
from typing import Dict, Any
import yaml


class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    def __init__(self, config_path: str = "agents/anomaly_detector/config.yaml"):
        # Database connections
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        
        # Directories
        self.log_dir = "agents/anomaly_detector/logs"
        self.model_dir = "agents/anomaly_detector/models"
        
        # Detection thresholds
        self.isolation_forest_contamination = 0.1
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5
        self.lof_n_neighbors = 20
        self.temporal_threshold_days = 365 * 100  # 100 years
        self.content_similarity_threshold = 0.95
        self.min_word_count = 10
        self.max_word_count = 1000000
        
        # Processing settings
        self.batch_size = 1000
        self.max_features_tfidf = 10000
        self.pca_components = 50
        self.enable_models = {
            "isolation_forest": True,
            "dbscan": True,
            "lof": True,
            "statistical": True,
            "temporal": True,
            "content": True
        }
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file if it exists."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }