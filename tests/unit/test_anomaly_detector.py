"""Test cases for the refactored anomaly detector module."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from agents.anomaly_detector.config import AnomalyConfig
from agents.anomaly_detector.data_fetcher import DataFetcher
from agents.anomaly_detector.detectors import (
    StatisticalAnomalyDetector,
    TemporalAnomalyDetector,
)
from agents.anomaly_detector.models import Anomaly, AnomalySeverity, AnomalyType
from agents.anomaly_detector.utils import AnomalySummaryGenerator


class TestAnomalyConfig:
    """Test suite for AnomalyConfig."""

    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = AnomalyConfig()
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.min_word_count == 10
        assert config.enable_models["temporal"] is True

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = AnomalyConfig()
        config_dict = config.to_dict()
        assert "neo4j_uri" in config_dict
        assert "enable_models" in config_dict


class TestTemporalAnomalyDetector:
    """Test suite for TemporalAnomalyDetector."""

    def test_detect_future_dates(self):
        """Test detection of future dates."""
        detector = TemporalAnomalyDetector()

        # Create test data with future date
        future_date = datetime(2030, 1, 1)
        data = pd.DataFrame(
            {
                "node_id": [1, 2],
                "date": [datetime(2020, 1, 1), future_date],
                "year": [2020, 2030],
            }
        )

        anomalies = detector.detect(data)

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.TEMPORAL
        assert anomalies[0].severity == AnomalySeverity.HIGH
        assert "future date" in anomalies[0].description

    def test_detect_ancient_dates(self):
        """Test detection of extremely ancient dates."""
        detector = TemporalAnomalyDetector()

        # Create test data with very old date
        data = pd.DataFrame(
            {
                "node_id": [1, 2],
                "date": [datetime(2020, 1, 1), datetime.min],
                "year": [2020, -4000],
            }
        )

        anomalies = detector.detect(data)

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.TEMPORAL
        assert anomalies[0].severity == AnomalySeverity.MEDIUM
        assert "ancient date" in anomalies[0].description


class TestStatisticalAnomalyDetector:
    """Test suite for StatisticalAnomalyDetector."""

    def test_detect_outliers(self):
        """Test detection of statistical outliers."""
        detector = StatisticalAnomalyDetector()

        # Create test data with outlier
        data = pd.DataFrame(
            {
                "node_id": [1, 2, 3, 4, 5],
                "word_count": [100, 110, 105, 1000, 95],  # 1000 is outlier
                "relationship_count": [5, 6, 4, 5, 7],
                "label_count": [2, 2, 3, 2, 2],
            }
        )

        anomalies = detector.detect(data)

        assert len(anomalies) >= 1
        outlier_anomaly = next((a for a in anomalies if a.node_id == "4"), None)
        assert outlier_anomaly is not None
        assert outlier_anomaly.anomaly_type == AnomalyType.STATISTICAL
        assert "word_count" in outlier_anomaly.description


class TestDataFetcher:
    """Test suite for DataFetcher."""

    @pytest.fixture
    def mock_drivers(self):
        """Create mock database drivers."""
        neo4j_driver = Mock()
        qdrant_client = AsyncMock()
        return neo4j_driver, qdrant_client

    def test_combine_data(self, mock_drivers):
        """Test data combination functionality."""
        neo4j_driver, qdrant_client = mock_drivers
        fetcher = DataFetcher(neo4j_driver, qdrant_client)

        # Create test graph data
        graph_data = pd.DataFrame(
            {
                "node_id": [1, 2],
                "date": ["2020-01-01", "2021-01-01"],
                "word_count": [100, 200],
                "relationship_count": [5, 10],
                "domain": ["science", "philosophy"],
                "language": ["en", "en"],
                "author": ["Author1", None],
                "title": ["Title1", "Title2"],
                "labels": [["Document"], ["Article"]],
            }
        )

        vector_data = {}

        result = fetcher.combine_data(graph_data, vector_data)

        assert not result.empty
        assert "year" in result.columns
        assert "has_author" in result.columns
        assert "has_title" in result.columns
        assert result["has_author"].iloc[0] == 1  # Has author
        assert result["has_author"].iloc[1] == 0  # No author


class TestAnomalySummaryGenerator:
    """Test suite for AnomalySummaryGenerator."""

    def test_generate_summary(self):
        """Test anomaly summary generation."""
        anomalies = [
            Anomaly(
                id="1",
                anomaly_type=AnomalyType.TEMPORAL,
                severity=AnomalySeverity.HIGH,
                description="Test",
                data_source="test",
                node_id="1",
                anomaly_score=0.9,
                details={},
                detected_at=datetime.now(),
                features={},
                suggestions=[],
            ),
            Anomaly(
                id="2",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Test",
                data_source="test",
                node_id="2",
                anomaly_score=0.7,
                details={},
                detected_at=datetime.now(),
                features={},
                suggestions=[],
            ),
        ]

        summary = AnomalySummaryGenerator.generate_summary(anomalies)

        assert summary["total"] == 2
        assert summary["by_type"]["temporal"] == 1
        assert summary["by_type"]["content"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1

    def test_get_anomalies_by_node(self):
        """Test getting anomalies for specific node."""
        anomalies = [
            Anomaly(
                id="1",
                anomaly_type=AnomalyType.TEMPORAL,
                severity=AnomalySeverity.HIGH,
                description="Test",
                data_source="test",
                node_id="node1",
                anomaly_score=0.9,
                details={},
                detected_at=datetime.now(),
                features={},
                suggestions=[],
            ),
            Anomaly(
                id="2",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Test",
                data_source="test",
                node_id="node2",
                anomaly_score=0.7,
                details={},
                detected_at=datetime.now(),
                features={},
                suggestions=[],
            ),
        ]

        node_anomalies = AnomalySummaryGenerator.get_anomalies_by_node(
            anomalies, "node1"
        )

        assert len(node_anomalies) == 1
        assert node_anomalies[0].node_id == "node1"

    def test_resolve_anomaly(self):
        """Test anomaly resolution."""
        anomalies = [
            Anomaly(
                id="test_anomaly",
                anomaly_type=AnomalyType.TEMPORAL,
                severity=AnomalySeverity.HIGH,
                description="Test",
                data_source="test",
                node_id="1",
                anomaly_score=0.9,
                details={},
                detected_at=datetime.now(),
                features={},
                suggestions=[],
            )
        ]

        success = AnomalySummaryGenerator.resolve_anomaly(
            anomalies, "test_anomaly", "Fixed manually"
        )

        assert success is True
        assert anomalies[0].details["resolved"] is True
        assert anomalies[0].details["resolution_notes"] == "Fixed manually"

        # Test non-existent anomaly
        success = AnomalySummaryGenerator.resolve_anomaly(
            anomalies, "non_existent", "Test"
        )
        assert success is False
