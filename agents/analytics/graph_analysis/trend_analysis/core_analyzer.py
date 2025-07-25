"""
Core Trend Analysis Orchestrator
Main coordinator for all trend analysis operations and module integration.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
from neo4j import AsyncDriver, AsyncGraphDatabase

from ..models import TrendAnalysis, TrendConfig, TrendPoint, TrendType
from .data_collectors import TimeSeriesDataCollector
from .predictor import TrendPredictor
from .seasonality_detector import SeasonalityDetector
from .statistics_engine import StatisticsEngine
from .trend_detector import TrendDetector
from .trend_visualization import TrendVisualizer


class TrendAnalyzer:
    """Main trend analysis orchestrator and coordination engine."""

    def __init__(self, config: Optional[TrendConfig] = None):
        """Initialize the trend analyzer with modular components."""
        self.config = config or TrendConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None

        # Initialize specialized modules
        self.data_collector = TimeSeriesDataCollector(config)
        self.trend_detector = TrendDetector(config)
        self.predictor = TrendPredictor(config)
        self.statistics_engine = StatisticsEngine(config)
        self.seasonality_detector = SeasonalityDetector(config)
        self.visualizer = TrendVisualizer(config)

        # Cached data for performance
        self.cached_data: Dict[str, List[TrendPoint]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Storage
        self.plot_dir = Path(getattr(config, "plot_dir", "analytics/plots"))
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        self.logger.info("TrendAnalyzer initialized with modular architecture")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("trend_analyzer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    async def initialize(self) -> None:
        """Initialize database connections and modules."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()

            # Initialize data collector with driver
            await self.data_collector.initialize(self.neo4j_driver)

            self.logger.info("Trend analyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize trend analyzer: {e}")
            raise

    async def close(self) -> None:
        """Close database connections and cleanup."""
        try:
            if self.neo4j_driver:
                await self.neo4j_driver.close()

            # Cleanup modules if they have close methods
            if hasattr(self.data_collector, "close"):
                await self.data_collector.close()

            self.logger.info("Trend analyzer closed successfully")

        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    async def analyze_trend(
        self,
        trend_type: TrendType,
        domain: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: str = "daily",
        use_cache: bool = True,
    ) -> TrendAnalysis:
        """
        Analyze a specific trend using modular components.

        Args:
            trend_type: Type of trend to analyze
            domain: Optional domain filter
            start_date: Start date for analysis
            end_date: End date for analysis
            granularity: Time granularity (daily, weekly, monthly)
            use_cache: Whether to use cached data

        Returns:
            Complete trend analysis result
        """

        start_time = datetime.now()

        try:
            self.logger.info(f"Starting trend analysis for {trend_type.value}")

            # Get time series data
            data = await self._get_time_series_data(
                trend_type, domain, start_date, end_date, granularity, use_cache
            )

            if len(data) < getattr(self.config, "min_data_points", 10):
                raise ValueError(
                    f"Insufficient data points: {len(data)} < {self.config.min_data_points}"
                )

            # Process data through modular pipeline
            processed_data = await self._process_data_pipeline(data)

            # Detect trend characteristics
            direction, strength = await self.trend_detector.detect_trend_direction(
                processed_data
            )

            # Calculate comprehensive statistics
            statistics = await self.statistics_engine.calculate_statistics(
                processed_data
            )

            # Detect seasonality patterns
            seasonality = await self.seasonality_detector.detect_seasonality(
                processed_data
            )

            # Generate predictions
            predictions = await self.predictor.generate_predictions(processed_data)

            # Extract insights
            insights = self._extract_comprehensive_insights(
                processed_data, direction, strength, statistics, seasonality
            )

            # Calculate confidence score
            confidence = self._calculate_confidence(
                processed_data, statistics, seasonality
            )

            # Create analysis result
            trend_analysis = TrendAnalysis(
                trend_type=trend_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                data_points=processed_data,
                statistics=statistics,
                predictions=predictions,
                insights=insights,
                generated_at=datetime.now(timezone.utc),
                metadata={
                    "domain": domain,
                    "granularity": granularity,
                    "seasonality": seasonality,
                    "data_quality_score": self._assess_data_quality(processed_data),
                },
            )

            # Generate visualization if configured
            if getattr(self.config, "save_plots", True):
                visualization_path = await self.visualizer.generate_trend_visualization(
                    trend_analysis
                )
                trend_analysis.metadata["visualization_path"] = visualization_path

            execution_time = (datetime.now() - start_time).total_seconds()
            trend_analysis.metadata["execution_time"] = execution_time

            self.logger.info(
                f"Trend analysis completed for {trend_type.value} in {execution_time:.2f}s"
            )
            return trend_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing trend {trend_type.value}: {e}")
            raise

    async def analyze_multiple_trends(
        self,
        trend_types: List[TrendType],
        domain: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: str = "daily",
    ) -> Dict[TrendType, TrendAnalysis]:
        """Analyze multiple trends concurrently."""

        tasks = []
        for trend_type in trend_types:
            task = self.analyze_trend(
                trend_type, domain, start_date, end_date, granularity
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        analysis_results = {}
        for trend_type, result in zip(trend_types, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to analyze {trend_type.value}: {result}")
                continue
            analysis_results[trend_type] = result

        return analysis_results

    async def _get_time_series_data(
        self,
        trend_type: TrendType,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str,
        use_cache: bool,
    ) -> List[TrendPoint]:
        """Get time series data with caching support."""

        cache_key = f"{trend_type.value}_{domain}_{start_date}_{end_date}_{granularity}"

        # Check cache if enabled
        if use_cache and cache_key in self.cached_data:
            cache_age = datetime.now() - self.cache_timestamps[cache_key]
            cache_ttl = getattr(self.config, "cache_ttl_minutes", 30)

            if cache_age.total_seconds() < cache_ttl * 60:
                self.logger.info(f"Using cached data for {trend_type.value}")
                return self.cached_data[cache_key]

        # Fetch fresh data
        data = await self.data_collector.get_time_series_data(
            trend_type, domain, start_date, end_date, granularity
        )

        # Cache the data
        if use_cache:
            self.cached_data[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()

        return data

    async def _process_data_pipeline(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Process data through preprocessing pipeline."""

        try:
            # Basic preprocessing
            processed_data = self.data_collector.preprocess_data(data)

            # Additional processing can be added here
            # e.g., outlier detection, smoothing, normalization

            return processed_data

        except Exception as e:
            self.logger.warning(f"Error in data processing pipeline: {e}")
            return data  # Return original data if processing fails

    def _extract_comprehensive_insights(
        self,
        data: List[TrendPoint],
        direction,
        strength: float,
        statistics: Dict[str, float],
        seasonality: Dict[str, float],
    ) -> List[str]:
        """Extract comprehensive insights from all analysis components."""

        insights = []

        # Basic trend insights
        insights.append(f"Trend direction: {direction.value}")
        insights.append(f"Trend strength: {strength:.3f}")

        # Statistical insights
        if statistics:
            mean_value = statistics.get("mean", 0)
            volatility = statistics.get("std", 0)
            insights.append(f"Average value: {mean_value:.2f}")
            insights.append(f"Volatility (std dev): {volatility:.2f}")

            # Growth rate
            if "growth_rate" in statistics:
                growth_rate = statistics["growth_rate"]
                insights.append(f"Overall growth rate: {growth_rate:.2%}")

        # Seasonality insights
        if seasonality and seasonality.get("has_seasonality", False):
            period = seasonality.get("period", "unknown")
            amplitude = seasonality.get("amplitude", 0)
            insights.append(f"Seasonal pattern detected with period: {period}")
            insights.append(f"Seasonal amplitude: {amplitude:.2f}")

        # Data quality insights
        data_quality = self._assess_data_quality(data)
        if data_quality < 0.8:
            insights.append(
                f"Data quality moderate ({data_quality:.2f}) - results may be less reliable"
            )
        elif data_quality > 0.95:
            insights.append(
                f"High data quality ({data_quality:.2f}) - results are highly reliable"
            )

        # Trend significance
        if len(data) > 30:  # Sufficient data for significance testing
            if strength > 0.7:
                insights.append(
                    "Strong trend detected - significant pattern identified"
                )
            elif strength > 0.3:
                insights.append("Moderate trend detected - noticeable pattern present")
            else:
                insights.append(
                    "Weak or no trend detected - data appears relatively stable"
                )

        return insights

    def _calculate_confidence(
        self,
        data: List[TrendPoint],
        statistics: Dict[str, float],
        seasonality: Dict[str, float],
    ) -> float:
        """Calculate overall confidence score for the analysis."""

        confidence_factors = []

        # Data quantity factor
        data_count = len(data)
        if data_count >= 100:
            data_factor = 1.0
        elif data_count >= 50:
            data_factor = 0.8
        elif data_count >= 20:
            data_factor = 0.6
        else:
            data_factor = 0.4
        confidence_factors.append(data_factor)

        # Data quality factor
        quality_factor = self._assess_data_quality(data)
        confidence_factors.append(quality_factor)

        # Statistical significance factor
        r_squared = statistics.get("r_squared", 0)
        stat_factor = min(r_squared * 2, 1.0)  # Scale R² to confidence
        confidence_factors.append(stat_factor)

        # Seasonality consistency factor (if applicable)
        if seasonality and seasonality.get("has_seasonality", False):
            seasonality_strength = seasonality.get("strength", 0)
            seasonal_factor = min(seasonality_strength, 1.0)
            confidence_factors.append(seasonal_factor)

        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)

    def _assess_data_quality(self, data: List[TrendPoint]) -> float:
        """Assess the quality of the data for analysis."""

        if not data:
            return 0.0

        quality_score = 1.0

        # Check for missing values
        missing_count = sum(1 for point in data if point.value is None)
        missing_ratio = missing_count / len(data)
        quality_score *= 1 - missing_ratio

        # Check for data consistency (no extreme outliers)
        values = [point.value for point in data if point.value is not None]
        if len(values) > 5:
            import numpy as np

            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                outlier_threshold = 3 * iqr
                outliers = sum(
                    1 for v in values if abs(v - np.median(values)) > outlier_threshold
                )
                outlier_ratio = outliers / len(values)
                quality_score *= 1 - outlier_ratio * 0.5  # Reduce impact of outliers

        return max(quality_score, 0.0)

    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate cached data."""
        if pattern:
            # Remove cache entries matching pattern
            keys_to_remove = [key for key in self.cached_data if pattern in key]
            for key in keys_to_remove:
                del self.cached_data[key]
                del self.cache_timestamps[key]
        else:
            # Clear all cache
            self.cached_data.clear()
            self.cache_timestamps.clear()

        self.logger.info(f"Cache invalidated (pattern: {pattern})")


# Factory function for easy integration
def create_trend_analyzer(config: Optional[TrendConfig] = None) -> TrendAnalyzer:
    """Create and return a TrendAnalyzer instance."""
    return TrendAnalyzer(config)


# Async wrapper for compatibility
async def analyze_trends(
    trend_type: TrendType, config: Optional[TrendConfig] = None, **kwargs
) -> TrendAnalysis:
    """Analyze trends using the core analyzer."""
    analyzer = create_trend_analyzer(config)
    await analyzer.initialize()
    try:
        return await analyzer.analyze_trend(trend_type, **kwargs)
    finally:
        await analyzer.close()
