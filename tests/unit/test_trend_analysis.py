#!/usr/bin/env python3
"""
Test suite for the trend analysis module
Tests all components of the refactored trend analysis system
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import asyncio
import numpy as np
import pandas as pd
import pytest

from agents.analytics.graph_analysis.base import AnalysisConfig, AnalysisResult
from agents.analytics.graph_analysis.trend_analysis.core_analyzer import TrendAnalyzer
from agents.analytics.graph_analysis.trend_analysis.data_collectors import DataCollector
from agents.analytics.graph_analysis.trend_analysis.predictor import TrendPredictor
from agents.analytics.graph_analysis.trend_analysis.seasonality_detector import (
    SeasonalityDetector,
)
from agents.analytics.graph_analysis.trend_analysis.statistics_engine import (
    StatisticsEngine,
)
from agents.analytics.graph_analysis.trend_analysis.trend_detector import TrendDetector
from agents.analytics.graph_analysis.trend_analysis.trend_visualization import (
    TrendVisualizer,
)


class TestTrendAnalysisBase:
    """Base class for trend analysis tests."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        values = np.random.randn(len(dates)).cumsum() + 100

        return pd.DataFrame(
            {"date": dates, "value": values, "domain": ["philosophy"] * len(dates)}
        )

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph evolution data."""
        return {
            "nodes": [
                {"id": 1, "created_at": "2023-01-01", "domain": "philosophy"},
                {"id": 2, "created_at": "2023-02-01", "domain": "science"},
                {"id": 3, "created_at": "2023-03-01", "domain": "philosophy"},
                {"id": 4, "created_at": "2023-04-01", "domain": "science"},
                {"id": 5, "created_at": "2023-05-01", "domain": "mathematics"},
            ],
            "edges": [
                {"source": 1, "target": 2, "created_at": "2023-01-15"},
                {"source": 2, "target": 3, "created_at": "2023-02-15"},
                {"source": 3, "target": 4, "created_at": "2023-03-15"},
                {"source": 4, "target": 5, "created_at": "2023-04-15"},
            ],
        }

    @pytest.fixture
    def analysis_config(self):
        """Create analysis configuration for testing."""
        return AnalysisConfig(
            graph_type="knowledge_graph",
            analysis_level="comprehensive",
            include_visualization=True,
            cache_results=False,
            parallel_processing=False,
        )

    @pytest.fixture
    def mock_session(self):
        """Create mock Neo4j session."""
        session = AsyncMock()
        session.run.return_value = AsyncMock()
        return session


class TestDataCollector(TestTrendAnalysisBase):
    """Test data collection functionality."""

    @pytest.mark.asyncio
    async def test_data_collector_initialization(self, analysis_config):
        """Test data collector initialization."""
        collector = DataCollector(analysis_config)

        assert collector.config == analysis_config
        assert hasattr(collector, "collection_strategies")

    @pytest.mark.asyncio
    async def test_collect_time_series_data(self, analysis_config, mock_session):
        """Test time series data collection."""
        collector = DataCollector(analysis_config)

        # Mock Neo4j query results
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"date": "2023-01-01", "count": 10, "domain": "philosophy"},
            {"date": "2023-01-02", "count": 12, "domain": "philosophy"},
            {"date": "2023-01-03", "count": 15, "domain": "philosophy"},
        ]
        mock_session.run.return_value = mock_result

        result = await collector.collect_time_series_data(mock_session, "philosophy")

        assert result.success is True
        assert "time_series" in result.data
        assert isinstance(result.data["time_series"], list)
        assert len(result.data["time_series"]) == 3

    @pytest.mark.asyncio
    async def test_collect_graph_evolution_data(self, analysis_config, mock_session):
        """Test graph evolution data collection."""
        collector = DataCollector(analysis_config)

        # Mock Neo4j query results
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"node_id": 1, "created_at": "2023-01-01", "domain": "philosophy"},
            {"node_id": 2, "created_at": "2023-02-01", "domain": "science"},
        ]
        mock_session.run.return_value = mock_result

        result = await collector.collect_graph_evolution_data(mock_session)

        assert result.success is True
        assert "evolution_data" in result.data
        assert isinstance(result.data["evolution_data"], list)

    @pytest.mark.unit
    def test_data_preprocessing(self, analysis_config, sample_time_series):
        """Test data preprocessing functionality."""
        collector = DataCollector(analysis_config)

        processed_data = collector._preprocess_time_series(sample_time_series)

        assert isinstance(processed_data, pd.DataFrame)
        assert "date" in processed_data.columns
        assert "value" in processed_data.columns
        assert processed_data["date"].dtype == "datetime64[ns]"

    @pytest.mark.unit
    def test_data_aggregation(self, analysis_config, sample_time_series):
        """Test data aggregation functionality."""
        collector = DataCollector(analysis_config)

        aggregated_data = collector._aggregate_by_period(
            sample_time_series, period="M", agg_func="mean"  # Monthly
        )

        assert isinstance(aggregated_data, pd.DataFrame)
        assert len(aggregated_data) <= len(sample_time_series)


class TestTrendDetector(TestTrendAnalysisBase):
    """Test trend detection functionality."""

    @pytest.mark.asyncio
    async def test_trend_detector_initialization(self, analysis_config):
        """Test trend detector initialization."""
        detector = TrendDetector(analysis_config)

        assert detector.config == analysis_config
        assert hasattr(detector, "detection_algorithms")

    @pytest.mark.asyncio
    async def test_trend_detection_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test trend detection analysis."""
        detector = TrendDetector(analysis_config)

        # Mock data loading
        detector._load_time_series_data = AsyncMock(return_value=sample_time_series)

        result = await detector.analyze(mock_session)

        assert result.success is True
        assert "detected_trends" in result.data
        assert isinstance(result.data["detected_trends"], list)

    @pytest.mark.unit
    def test_linear_trend_detection(self, analysis_config, sample_time_series):
        """Test linear trend detection."""
        detector = TrendDetector(analysis_config)

        # Create data with clear linear trend
        x = np.arange(len(sample_time_series))
        y = x * 2 + 10 + np.random.normal(0, 1, len(x))

        trend_info = detector._detect_linear_trend(y)

        assert isinstance(trend_info, dict)
        assert "slope" in trend_info
        assert "intercept" in trend_info
        assert "r_squared" in trend_info
        assert "p_value" in trend_info
        assert trend_info["slope"] > 0  # Should detect positive trend

    @pytest.mark.unit
    def test_change_point_detection(self, analysis_config, sample_time_series):
        """Test change point detection."""
        detector = TrendDetector(analysis_config)

        # Create data with clear change point
        data = np.concatenate(
            [
                np.random.normal(10, 1, 50),  # First segment
                np.random.normal(20, 1, 50),  # Second segment with higher mean
            ]
        )

        change_points = detector._detect_change_points(data)

        assert isinstance(change_points, list)
        assert len(change_points) >= 1
        assert all(isinstance(cp, int) for cp in change_points)

    @pytest.mark.unit
    def test_trend_strength_calculation(self, analysis_config, sample_time_series):
        """Test trend strength calculation."""
        detector = TrendDetector(analysis_config)

        # Create data with strong trend
        x = np.arange(100)
        y = x * 3 + np.random.normal(0, 0.1, 100)

        strength = detector._calculate_trend_strength(y)

        assert isinstance(strength, float)
        assert 0 <= strength <= 1
        assert strength > 0.7  # Should detect strong trend

    @pytest.mark.unit
    def test_trend_direction_detection(self, analysis_config):
        """Test trend direction detection."""
        detector = TrendDetector(analysis_config)

        # Test increasing trend
        increasing_data = np.arange(100) * 2
        direction_inc = detector._detect_trend_direction(increasing_data)
        assert direction_inc == "increasing"

        # Test decreasing trend
        decreasing_data = np.arange(100, 0, -1) * 2
        direction_dec = detector._detect_trend_direction(decreasing_data)
        assert direction_dec == "decreasing"

        # Test stable trend
        stable_data = np.full(100, 50)
        direction_stable = detector._detect_trend_direction(stable_data)
        assert direction_stable == "stable"


class TestTrendPredictor(TestTrendAnalysisBase):
    """Test trend prediction functionality."""

    @pytest.mark.asyncio
    async def test_trend_predictor_initialization(self, analysis_config):
        """Test trend predictor initialization."""
        predictor = TrendPredictor(analysis_config)

        assert predictor.config == analysis_config
        assert hasattr(predictor, "prediction_models")

    @pytest.mark.asyncio
    async def test_trend_prediction_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test trend prediction analysis."""
        predictor = TrendPredictor(analysis_config)

        # Mock data loading
        predictor._load_time_series_data = AsyncMock(return_value=sample_time_series)

        result = await predictor.analyze(mock_session)

        assert result.success is True
        assert "predictions" in result.data
        assert isinstance(result.data["predictions"], dict)

    @pytest.mark.unit
    def test_linear_regression_prediction(self, analysis_config, sample_time_series):
        """Test linear regression prediction."""
        predictor = TrendPredictor(analysis_config)

        # Create simple linear data
        x = np.arange(100)
        y = x * 2 + 10 + np.random.normal(0, 1, 100)

        predictions = predictor._predict_linear_regression(y, steps_ahead=10)

        assert isinstance(predictions, dict)
        assert "predicted_values" in predictions
        assert "confidence_intervals" in predictions
        assert len(predictions["predicted_values"]) == 10

    @pytest.mark.unit
    def test_exponential_smoothing_prediction(
        self, analysis_config, sample_time_series
    ):
        """Test exponential smoothing prediction."""
        predictor = TrendPredictor(analysis_config)

        # Create data with trend
        data = np.cumsum(np.random.randn(100)) + 100

        predictions = predictor._predict_exponential_smoothing(data, steps_ahead=5)

        assert isinstance(predictions, dict)
        assert "predicted_values" in predictions
        assert len(predictions["predicted_values"]) == 5

    @pytest.mark.unit
    def test_prediction_accuracy_metrics(self, analysis_config):
        """Test prediction accuracy metrics."""
        predictor = TrendPredictor(analysis_config)

        # Create actual vs predicted data
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        metrics = predictor._calculate_accuracy_metrics(actual, predicted)

        assert isinstance(metrics, dict)
        assert "mae" in metrics  # Mean Absolute Error
        assert "mse" in metrics  # Mean Squared Error
        assert "rmse" in metrics  # Root Mean Squared Error
        assert "mape" in metrics  # Mean Absolute Percentage Error
        assert all(metric >= 0 for metric in metrics.values())


class TestStatisticsEngine(TestTrendAnalysisBase):
    """Test statistics engine functionality."""

    @pytest.mark.asyncio
    async def test_statistics_engine_initialization(self, analysis_config):
        """Test statistics engine initialization."""
        engine = StatisticsEngine(analysis_config)

        assert engine.config == analysis_config
        assert hasattr(engine, "statistical_tests")

    @pytest.mark.asyncio
    async def test_statistical_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test statistical analysis."""
        engine = StatisticsEngine(analysis_config)

        # Mock data loading
        engine._load_time_series_data = AsyncMock(return_value=sample_time_series)

        result = await engine.analyze(mock_session)

        assert result.success is True
        assert "statistical_summary" in result.data
        assert isinstance(result.data["statistical_summary"], dict)

    @pytest.mark.unit
    def test_descriptive_statistics(self, analysis_config, sample_time_series):
        """Test descriptive statistics calculation."""
        engine = StatisticsEngine(analysis_config)

        stats = engine._calculate_descriptive_stats(sample_time_series["value"])

        assert isinstance(stats, dict)
        required_stats = ["mean", "median", "std", "min", "max", "skewness", "kurtosis"]
        assert all(stat in stats for stat in required_stats)
        assert all(isinstance(stats[stat], (int, float)) for stat in required_stats)

    @pytest.mark.unit
    def test_stationarity_test(self, analysis_config, sample_time_series):
        """Test stationarity testing."""
        engine = StatisticsEngine(analysis_config)

        stationarity_result = engine._test_stationarity(sample_time_series["value"])

        assert isinstance(stationarity_result, dict)
        assert "is_stationary" in stationarity_result
        assert "p_value" in stationarity_result
        assert "test_statistic" in stationarity_result
        assert isinstance(stationarity_result["is_stationary"], bool)

    @pytest.mark.unit
    def test_correlation_analysis(self, analysis_config):
        """Test correlation analysis."""
        engine = StatisticsEngine(analysis_config)

        # Create correlated data
        x = np.random.randn(100)
        y = x * 2 + np.random.randn(100) * 0.1

        correlation_result = engine._calculate_correlation(x, y)

        assert isinstance(correlation_result, dict)
        assert "pearson_correlation" in correlation_result
        assert "spearman_correlation" in correlation_result
        assert "p_value" in correlation_result
        assert -1 <= correlation_result["pearson_correlation"] <= 1

    @pytest.mark.unit
    def test_confidence_intervals(self, analysis_config):
        """Test confidence interval calculation."""
        engine = StatisticsEngine(analysis_config)

        data = np.random.normal(50, 10, 100)

        ci = engine._calculate_confidence_interval(data, confidence_level=0.95)

        assert isinstance(ci, dict)
        assert "lower_bound" in ci
        assert "upper_bound" in ci
        assert "confidence_level" in ci
        assert ci["lower_bound"] < ci["upper_bound"]


class TestSeasonalityDetector(TestTrendAnalysisBase):
    """Test seasonality detection functionality."""

    @pytest.mark.asyncio
    async def test_seasonality_detector_initialization(self, analysis_config):
        """Test seasonality detector initialization."""
        detector = SeasonalityDetector(analysis_config)

        assert detector.config == analysis_config
        assert hasattr(detector, "seasonality_methods")

    @pytest.mark.asyncio
    async def test_seasonality_detection_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test seasonality detection analysis."""
        detector = SeasonalityDetector(analysis_config)

        # Mock data loading
        detector._load_time_series_data = AsyncMock(return_value=sample_time_series)

        result = await detector.analyze(mock_session)

        assert result.success is True
        assert "seasonality_patterns" in result.data
        assert isinstance(result.data["seasonality_patterns"], dict)

    @pytest.mark.unit
    def test_autocorrelation_seasonality(self, analysis_config):
        """Test autocorrelation-based seasonality detection."""
        detector = SeasonalityDetector(analysis_config)

        # Create data with clear seasonality (period=12)
        t = np.arange(120)
        seasonal_data = np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.1, 120)

        seasonality_result = detector._detect_seasonality_autocorr(seasonal_data)

        assert isinstance(seasonality_result, dict)
        assert "has_seasonality" in seasonality_result
        assert "period" in seasonality_result
        assert "strength" in seasonality_result
        assert seasonality_result["has_seasonality"] is True
        assert seasonality_result["period"] == 12

    @pytest.mark.unit
    def test_fft_seasonality(self, analysis_config):
        """Test FFT-based seasonality detection."""
        detector = SeasonalityDetector(analysis_config)

        # Create data with clear seasonality
        t = np.arange(120)
        seasonal_data = np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.1, 120)

        seasonality_result = detector._detect_seasonality_fft(seasonal_data)

        assert isinstance(seasonality_result, dict)
        assert "dominant_frequencies" in seasonality_result
        assert "periods" in seasonality_result
        assert isinstance(seasonality_result["dominant_frequencies"], list)
        assert isinstance(seasonality_result["periods"], list)

    @pytest.mark.unit
    def test_seasonal_decomposition(self, analysis_config):
        """Test seasonal decomposition."""
        detector = SeasonalityDetector(analysis_config)

        # Create data with trend + seasonality
        t = np.arange(120)
        trend = t * 0.1
        seasonal = np.sin(2 * np.pi * t / 12) * 2
        noise = np.random.normal(0, 0.1, 120)
        data = trend + seasonal + noise

        decomposition = detector._seasonal_decomposition(data, period=12)

        assert isinstance(decomposition, dict)
        assert "trend" in decomposition
        assert "seasonal" in decomposition
        assert "residual" in decomposition
        assert len(decomposition["trend"]) == len(data)
        assert len(decomposition["seasonal"]) == len(data)
        assert len(decomposition["residual"]) == len(data)


class TestTrendVisualizer(TestTrendAnalysisBase):
    """Test trend visualization functionality."""

    @pytest.mark.asyncio
    async def test_trend_visualizer_initialization(self, analysis_config):
        """Test trend visualizer initialization."""
        visualizer = TrendVisualizer(analysis_config)

        assert visualizer.config == analysis_config
        assert hasattr(visualizer, "plot_types")

    @pytest.mark.asyncio
    async def test_trend_visualization_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test trend visualization analysis."""
        visualizer = TrendVisualizer(analysis_config)

        # Mock data loading
        visualizer._load_time_series_data = AsyncMock(return_value=sample_time_series)

        result = await visualizer.analyze(mock_session)

        assert result.success is True
        assert "visualizations" in result.data
        assert isinstance(result.data["visualizations"], dict)

    @pytest.mark.unit
    def test_time_series_plot_generation(self, analysis_config, sample_time_series):
        """Test time series plot generation."""
        visualizer = TrendVisualizer(analysis_config)

        plot_data = visualizer._generate_time_series_plot(sample_time_series)

        assert isinstance(plot_data, dict)
        assert "plot_type" in plot_data
        assert "data" in plot_data
        assert "layout" in plot_data
        assert plot_data["plot_type"] == "time_series"

    @pytest.mark.unit
    def test_trend_decomposition_plot(self, analysis_config):
        """Test trend decomposition plot generation."""
        visualizer = TrendVisualizer(analysis_config)

        # Create mock decomposition data
        decomposition_data = {
            "original": np.random.randn(100),
            "trend": np.random.randn(100),
            "seasonal": np.random.randn(100),
            "residual": np.random.randn(100),
        }

        plot_data = visualizer._generate_decomposition_plot(decomposition_data)

        assert isinstance(plot_data, dict)
        assert "plot_type" in plot_data
        assert "subplots" in plot_data
        assert plot_data["plot_type"] == "decomposition"
        assert len(plot_data["subplots"]) == 4

    @pytest.mark.unit
    def test_prediction_plot_generation(self, analysis_config):
        """Test prediction plot generation."""
        visualizer = TrendVisualizer(analysis_config)

        # Create mock prediction data
        historical_data = np.random.randn(100)
        predictions = np.random.randn(10)
        confidence_intervals = {
            "lower": np.random.randn(10),
            "upper": np.random.randn(10),
        }

        plot_data = visualizer._generate_prediction_plot(
            historical_data, predictions, confidence_intervals
        )

        assert isinstance(plot_data, dict)
        assert "plot_type" in plot_data
        assert "historical_data" in plot_data
        assert "predictions" in plot_data
        assert "confidence_intervals" in plot_data


class TestTrendAnalyzerCore(TestTrendAnalysisBase):
    """Test the core trend analyzer orchestrator."""

    @pytest.mark.asyncio
    async def test_trend_analyzer_initialization(self, analysis_config):
        """Test trend analyzer initialization."""
        analyzer = TrendAnalyzer(analysis_config)

        assert analyzer.config == analysis_config
        assert hasattr(analyzer, "sub_analyzers")

    @pytest.mark.asyncio
    async def test_comprehensive_trend_analysis(
        self, analysis_config, sample_time_series, mock_session
    ):
        """Test comprehensive trend analysis."""
        analyzer = TrendAnalyzer(analysis_config)

        # Mock data loading
        analyzer._load_data = AsyncMock(return_value=sample_time_series)

        result = await analyzer.analyze(mock_session)

        assert result.success is True
        assert result.analysis_type == "trend_analysis"
        assert result.execution_time > 0
        assert isinstance(result.data, dict)
        assert "trend_summary" in result.data

    @pytest.mark.integration
    async def test_trend_analysis_with_seasonal_data(
        self, analysis_config, mock_session
    ):
        """Test trend analysis with seasonal data."""
        analyzer = TrendAnalyzer(analysis_config)

        # Create seasonal data
        t = np.arange(365)
        trend = t * 0.01
        seasonal = np.sin(2 * np.pi * t / 30) * 5  # Monthly seasonality
        noise = np.random.normal(0, 1, 365)
        values = trend + seasonal + noise + 100

        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
        seasonal_data = pd.DataFrame(
            {"date": dates, "value": values, "domain": ["test"] * 365}
        )

        # Mock data loading
        analyzer._load_data = AsyncMock(return_value=seasonal_data)

        result = await analyzer.analyze(mock_session)

        assert result.success is True
        assert "trend_summary" in result.data
        assert "seasonality_patterns" in result.data
        assert "predictions" in result.data

    @pytest.mark.slow
    async def test_trend_analysis_performance(self, analysis_config, mock_session):
        """Test trend analysis performance with large dataset."""
        analyzer = TrendAnalyzer(analysis_config)

        # Create large dataset
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        values = np.random.randn(len(dates)).cumsum() + 1000

        large_data = pd.DataFrame(
            {"date": dates, "value": values, "domain": ["test"] * len(dates)}
        )

        # Mock data loading
        analyzer._load_data = AsyncMock(return_value=large_data)

        result = await analyzer.analyze(mock_session)

        assert result.success is True
        assert result.execution_time < 60  # Should complete within 60 seconds
        assert len(large_data) > 1000  # Ensure we're testing with substantial data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
