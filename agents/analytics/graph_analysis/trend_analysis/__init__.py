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

from .core_analyzer import TrendAnalyzer, analyze_trends
from .data_collectors import (
    TimeSeriesDataCollector,
    DocumentGrowthCollector,
    ConceptEmergenceCollector,
    PatternFrequencyCollector,
    DomainActivityCollector,
    CitationNetworksCollector,
    AuthorProductivityCollector
)
from .trend_detector import TrendDetector, detect_trend_direction
from .predictor import TrendPredictor, generate_predictions
from .statistics_engine import StatisticsEngine, calculate_trend_statistics
from .seasonality_detector import SeasonalityDetector, detect_seasonality
from .trend_visualization import TrendVisualizer, generate_trend_visualization

# Import shared models from parent
from ..models import TrendAnalysis, TrendPoint, TrendType, TrendDirection, TrendConfig

__all__ = [
    # Main classes
    'TrendAnalyzer',
    'TimeSeriesDataCollector',
    'TrendDetector', 
    'TrendPredictor',
    'StatisticsEngine',
    'SeasonalityDetector',
    'TrendVisualizer',
    
    # Models and enums
    'TrendAnalysis',
    'TrendPoint',
    'TrendType', 
    'TrendDirection',
    'TrendConfig',
    
    # Convenience functions
    'analyze_trends',
    'detect_trend_direction',
    'generate_predictions',
    'calculate_trend_statistics',
    'detect_seasonality',
    'generate_trend_visualization'
]

# Module version
__version__ = "1.0.0"

# Module metadata
__author__ = "MCP Yggdrasil Team"
__description__ = "Modular trend analysis system for temporal pattern detection"