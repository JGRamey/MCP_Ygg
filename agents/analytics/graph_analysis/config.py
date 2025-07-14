"""Configuration for network analysis."""
import logging
from pathlib import Path
import yaml

class NetworkConfig:
    """Configuration for network analysis."""
    
    def __init__(self, config_path: str = "analytics/config.py"):
        # Database connection
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        
        # Analysis parameters
        self.max_nodes = 10000
        self.max_edges = 50000
        self.pagerank_alpha = 0.85
        self.pagerank_max_iter = 100
        self.pagerank_tol = 1e-6
        
        # Community detection parameters
        self.min_community_size = 3
        self.max_communities = 50
        self.resolution_parameter = 1.0
        
        # Centrality calculation parameters
        self.normalize_centrality = True
        self.centrality_k = None  # For k-path centralities
        
        # Performance settings
        self.parallel_processing = True
        self.cache_results = True
        self.cache_duration_hours = 24
        
        # Visualization settings
        self.generate_plots = True
        self.plot_dir = "analytics/plots"
        self.figure_size = (12, 8)
        self.node_size_multiplier = 300
        
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
