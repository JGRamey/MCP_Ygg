"""
Trend Analysis Module for MCP Yggdrasil
Modular trend analysis system for temporal pattern detection and prediction.

This module provides:
- Core trend analysis orchestration
- Time series data collection and processing
- Trend direction and strength detection
- Prediction and forecasting capabilities
- Statistical analysis and seasonality detection
- Comprehensive trend visualization

Architecture:
- core_analyzer.py: Main orchestrator and analysis coordination
- data_collectors.py: Time series data collection from various sources
- trend_detector.py: Trend direction and strength analysis
- predictor.py: Prediction and forecasting algorithms
- statistics_engine.py: Statistical calculations and metrics
- seasonality_detector.py: Seasonal pattern analysis
- trend_visualization.py: Comprehensive trend plotting

Usage:
    from trend_analysis import TrendAnalyzer, analyze_trends

    analyzer = TrendAnalyzer(config)
    result = await analyzer.analyze_trends(TrendType.DOCUMENT_GROWTH)
"""

# Import shared models from parent
from ..models import TrendAnalysis, TrendConfig, TrendDirection, TrendPoint, TrendType
from .core_analyzer import TrendAnalyzer, analyze_trends
from .data_collectors import (
    AuthorProductivityCollector,
    CitationNetworksCollector,
    ConceptEmergenceCollector,
    DocumentGrowthCollector,
    DomainActivityCollector,
    PatternFrequencyCollector,
    TimeSeriesDataCollector,
)
from .predictor import TrendPredictor, generate_predictions
from .seasonality_detector import SeasonalityDetector, create_seasonality_detector
from .statistics_engine import StatisticsEngine, create_statistics_engine
from .trend_detector import TrendDetector, detect_trend_direction
from .trend_visualization import (
    TrendVisualizationEngine,
    create_trend_visualization_engine,
)

__all__ = [
    # Main classes
    "TrendAnalyzer",
    "TimeSeriesDataCollector",
    "TrendDetector",
    "TrendPredictor",
    "StatisticsEngine",
    "SeasonalityDetector",
    "TrendVisualizationEngine",
    # Models and enums
    "TrendAnalysis",
    "TrendPoint",
    "TrendType",
    "TrendDirection",
    "TrendConfig",
    # Convenience functions
    "analyze_trends",
    "detect_trend_direction",
    "generate_predictions",
    "create_statistics_engine",
    "create_seasonality_detector",
    "create_trend_visualization_engine",
]

# Module version
__version__ = "1.0.0"

# Module metadata
__author__ = "MCP Yggdrasil Team"
__description__ = "Modular trend analysis system for temporal pattern detection"
