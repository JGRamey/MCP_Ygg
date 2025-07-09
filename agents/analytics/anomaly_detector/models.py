"""Data models for anomaly detection."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TEMPORAL = "temporal"  # Date/time inconsistencies
    METADATA = "metadata"  # Unusual metadata patterns
    CONTENT = "content"  # Content anomalies
    DOMAIN = "domain"  # Domain classification issues
    RELATIONSHIP = "relationship"  # Unusual graph relationships
    VECTOR = "vector"  # Vector embedding anomalies
    STATISTICAL = "statistical"  # Statistical outliers


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    data_source: str
    node_id: Optional[str]
    anomaly_score: float
    details: Dict[str, Any]
    detected_at: datetime
    features: Dict[str, float]
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary for serialization."""
        return {
            'id': self.id,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'data_source': self.data_source,
            'node_id': self.node_id,
            'anomaly_score': self.anomaly_score,
            'details': self.details,
            'detected_at': self.detected_at.isoformat(),
            'features': self.features,
            'suggestions': self.suggestions
        }