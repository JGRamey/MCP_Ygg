
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages application configuration loaded from a YAML file."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from the YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value using dot notation (e.g., 'database.neo4j.host')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value using dot notation."""
        keys = key.split('.')
        config_ref = self._config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                config_ref[k] = value
            else:
                if not isinstance(config_ref.get(k), dict):
                    config_ref[k] = {}
                config_ref = config_ref[k]

    def reload(self) -> None:
        """Reloads the configuration from the file."""
        self._config = self._load_config()

    def save(self, config: Optional[Dict] = None) -> None:
        """Saves the current configuration (or a provided one) back to the YAML file."""
        data_to_save = config if config is not None else self._config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(data_to_save, f, indent=2)
