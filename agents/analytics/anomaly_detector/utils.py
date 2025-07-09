"""Utility functions for anomaly detection."""
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import aiofiles

from .models import Anomaly


class ModelManager:
    """Manages saving and loading of ML models."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("anomaly_detector.model_manager")
    
    async def save_models(self, models: Dict[str, Any]) -> None:
        """Save trained models to disk."""
        for model_name, model in models.items():
            if model is not None:
                file_path = self.model_dir / f'{model_name}.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Saved {model_name} to {file_path}")
    
    async def load_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Load trained models from disk."""
        loaded_models = {}
        
        for model_name in model_names:
            file_path = self.model_dir / f'{model_name}.pkl'
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                    loaded_models[model_name] = model
                self.logger.info(f"Loaded {model_name} from {file_path}")
            else:
                loaded_models[model_name] = None
        
        return loaded_models


class AnomalyLogger:
    """Handles logging of detected anomalies."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("anomaly_detector.anomaly_logger")
    
    async def log_anomalies(self, anomalies: List[Anomaly]) -> None:
        """Log detected anomalies to file."""
        if not anomalies:
            return
        
        log_file = self.log_dir / f"detected_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        anomaly_data = [anomaly.to_dict() for anomaly in anomalies]
        
        async with aiofiles.open(log_file, 'w') as f:
            await f.write(json.dumps(anomaly_data, indent=2, default=str))
        
        self.logger.info(f"Logged {len(anomalies)} anomalies to {log_file}")


class AnomalySummaryGenerator:
    """Generates summaries and statistics for detected anomalies."""
    
    @staticmethod
    def generate_summary(anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        if not anomalies:
            return {"total": 0}
        
        summary = {
            "total": len(anomalies),
            "by_type": {},
            "by_severity": {},
            "recent_count": 0
        }
        
        recent_threshold = datetime.now(timezone.utc).timestamp() - 86400  # 24 hours ago
        
        for anomaly in anomalies:
            # Count by type
            type_name = anomaly.anomaly_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Count by severity
            severity_name = anomaly.severity.value
            summary["by_severity"][severity_name] = summary["by_severity"].get(severity_name, 0) + 1
            
            # Count recent anomalies
            if anomaly.detected_at.timestamp() > recent_threshold:
                summary["recent_count"] += 1
        
        return summary
    
    @staticmethod
    def get_anomalies_by_node(anomalies: List[Anomaly], node_id: str) -> List[Anomaly]:
        """Get all anomalies for a specific node."""
        return [
            anomaly for anomaly in anomalies
            if anomaly.node_id == node_id
        ]
    
    @staticmethod
    def resolve_anomaly(anomalies: List[Anomaly], anomaly_id: str, resolution_notes: str) -> bool:
        """Mark an anomaly as resolved."""
        for anomaly in anomalies:
            if anomaly.id == anomaly_id:
                anomaly.details["resolved"] = True
                anomaly.details["resolution_notes"] = resolution_notes
                anomaly.details["resolved_at"] = datetime.now(timezone.utc).isoformat()
                return True
        return False


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration for anomaly detector."""
    logger = logging.getLogger("anomaly_detector")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = log_dir / f"anomalies_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger