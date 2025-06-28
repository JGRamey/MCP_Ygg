"""
Trend Analysis Agent for MCP Server
Analyzes temporal trends in concepts, documents, and knowledge evolution.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from collections import defaultdict, Counter
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

from neo4j import AsyncGraphDatabase, AsyncDriver
import networkx as nx


class TrendType(Enum):
    """Types of trends that can be analyzed."""
    DOCUMENT_GROWTH = "document_growth"
    CONCEPT_EMERGENCE = "concept_emergence"
    PATTERN_FREQUENCY = "pattern_frequency"
    DOMAIN_ACTIVITY = "domain_activity"
    CITATION_NETWORKS = "citation_networks"
    AUTHOR_PRODUCTIVITY = "author_productivity"
    KNOWLEDGE_FLOW = "knowledge_flow"
    SEASONAL_PATTERNS = "seasonal_patterns"


class TrendDirection(Enum):
    """Direction of trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    VOLATILE = "volatile"


@dataclass
class TrendPoint:
    """Represents a single point in a trend."""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrendAnalysis:
    """Complete trend analysis result."""
    trend_type: TrendType
    direction: TrendDirection
    strength: float  # 0-1 where 1 is strongest
    confidence: float  # 0-1 confidence in the analysis
    data_points: List[TrendPoint]
    statistics: Dict[str, float]
    predictions: List[TrendPoint]
    insights: List[str]
    generated_at: datetime


class TrendConfig:
    """Configuration for trend analysis."""
    
    def __init__(self, config_path: str = "analytics/config.yaml"):
        # Database connection
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        
        # Analysis parameters
        self.min_data_points = 10
        self.prediction_horizon_days = 30
        self.smoothing_window = 7  # days
        self.outlier_threshold = 3.0  # standard deviations
        self.significance_level = 0.05
        
        # Trend detection parameters
        self.min_trend_strength = 0.3
        self.volatility_threshold = 0.5
        self.seasonality_periods = [7, 30, 365]  # daily, monthly, yearly
        
        # Visualization
        self.figure_size = (12, 8)
        self.color_palette = "viridis"
        self.save_plots = True
        self.plot_dir = "analytics/plots"
        
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


class TrendAnalyzer:
    """Main trend analysis engine."""
    
    def __init__(self, config: Optional[TrendConfig] = None):
        """Initialize the trend analyzer."""
        self.config = config or TrendConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        
        # Models and cached data
        self.trend_models: Dict[str, Any] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Storage
        self.plot_dir = Path(self.config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("trend_analyzer")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self.logger.info("Trend analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trend analyzer: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        self.logger.info("Trend analyzer closed")
    
    async def analyze_trend(
        self,
        trend_type: TrendType,
        domain: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: str = "daily"
    ) -> TrendAnalysis:
        """Analyze a specific trend."""
        
        try:
            # Get time series data
            data = await self._get_time_series_data(trend_type, domain, start_date, end_date, granularity)
            
            if len(data) < self.config.min_data_points:
                raise ValueError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Detect trend direction and strength
            direction, strength = self._detect_trend_direction(processed_data)
            
            # Calculate statistics
            statistics = self._calculate_statistics(processed_data)
            
            # Generate predictions
            predictions = self._generate_predictions(processed_data)
            
            # Extract insights
            insights = self._extract_insights(processed_data, direction, strength, statistics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(processed_data, statistics)
            
            trend_analysis = TrendAnalysis(
                trend_type=trend_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                data_points=processed_data,
                statistics=statistics,
                predictions=predictions,
                insights=insights,
                generated_at=datetime.now(timezone.utc)
            )
            
            # Generate visualization
            if self.config.save_plots:
                await self._generate_visualization(trend_analysis)
            
            self.logger.info(f"Trend analysis completed for {trend_type.value}")
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend {trend_type.value}: {e}")
            raise
    
    async def _get_time_series_data(
        self,
        trend_type: TrendType,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get time series data for the specified trend type."""
        
        if trend_type == TrendType.DOCUMENT_GROWTH:
            return await self._get_document_growth_data(domain, start_date, end_date, granularity)
        elif trend_type == TrendType.CONCEPT_EMERGENCE:
            return await self._get_concept_emergence_data(domain, start_date, end_date, granularity)
        elif trend_type == TrendType.PATTERN_FREQUENCY:
            return await self._get_pattern_frequency_data(domain, start_date, end_date, granularity)
        elif trend_type == TrendType.DOMAIN_ACTIVITY:
            return await self._get_domain_activity_data(domain, start_date, end_date, granularity)
        elif trend_type == TrendType.CITATION_NETWORKS:
            return await self._get_citation_networks_data(domain, start_date, end_date, granularity)
        elif trend_type == TrendType.AUTHOR_PRODUCTIVITY:
            return await self._get_author_productivity_data(domain, start_date, end_date, granularity)
        else:
            raise ValueError(f"Unsupported trend type: {trend_type}")
    
    async def _get_document_growth_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get document growth time series data."""
        
        async with self.neo4j_driver.session() as session:
            # Build query based on parameters
            domain_clause = "AND d.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND d.date >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND d.date <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            # Determine grouping based on granularity
            if granularity == "daily":
                group_format = "date(d.date)"
            elif granularity == "weekly":
                group_format = "date(d.date) - duration({days: date(d.date).weekday})"
            elif granularity == "monthly":
                group_format = "date({year: d.date.year, month: d.date.month, day: 1})"
            elif granularity == "yearly":
                group_format = "date({year: d.date.year, month: 1, day: 1})"
            else:
                group_format = "date(d.date)"
            
            query = f"""
            MATCH (d:Document)
            WHERE d.date IS NOT NULL {date_clause} {domain_clause}
            WITH {group_format} AS time_period, count(d) AS doc_count
            ORDER BY time_period
            RETURN time_period, doc_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['time_period']))
                value = float(record['doc_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'granularity': granularity, 'domain': domain}
                ))
            
            return data_points
    
    async def _get_concept_emergence_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get concept emergence time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND c.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND c.first_seen >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND c.first_seen <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (c:Concept)
            WHERE c.first_seen IS NOT NULL {date_clause} {domain_clause}
            WITH date(c.first_seen) AS emergence_date, count(c) AS concept_count
            ORDER BY emergence_date
            RETURN emergence_date, concept_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['emergence_date']))
                value = float(record['concept_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'type': 'concept_emergence', 'domain': domain}
                ))
            
            return data_points
    
    async def _get_pattern_frequency_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get pattern frequency time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = ""
            if domain:
                domain_clause = "AND $domain IN p.domains"
            
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND p.detected_at >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND p.detected_at <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (p:Pattern)
            WHERE p.detected_at IS NOT NULL {date_clause} {domain_clause}
            WITH date(p.detected_at) AS detection_date, count(p) AS pattern_count
            ORDER BY detection_date
            RETURN detection_date, pattern_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['detection_date']))
                value = float(record['pattern_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'type': 'pattern_frequency', 'domain': domain}
                ))
            
            return data_points
    
    async def _get_domain_activity_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get domain activity time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND n.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND n.date >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND n.date <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (n)
            WHERE n.date IS NOT NULL {date_clause} {domain_clause}
            OPTIONAL MATCH (n)-[r]-()
            WITH date(n.date) AS activity_date, count(DISTINCT n) AS node_count, count(r) AS relationship_count
            ORDER BY activity_date
            RETURN activity_date, node_count + relationship_count AS activity_score
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['activity_date']))
                value = float(record['activity_score'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'type': 'domain_activity', 'domain': domain}
                ))
            
            return data_points
    
    async def _get_citation_networks_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get citation network growth time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND (d1.domain = $domain OR d2.domain = $domain)" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND r.created_at >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND r.created_at <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (d1:Document)-[r:REFERENCES]->(d2:Document)
            WHERE r.created_at IS NOT NULL {date_clause} {domain_clause}
            WITH date(r.created_at) AS citation_date, count(r) AS citation_count
            ORDER BY citation_date
            RETURN citation_date, citation_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['citation_date']))
                value = float(record['citation_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'type': 'citation_networks', 'domain': domain}
                ))
            
            return data_points
    
    async def _get_author_productivity_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Get author productivity time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND d.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND d.date >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND d.date <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (p:Person)-[:AUTHORED]->(d:Document)
            WHERE d.date IS NOT NULL {date_clause} {domain_clause}
            WITH date(d.date) AS pub_date, count(DISTINCT p) AS active_authors, count(d) AS total_docs
            ORDER BY pub_date
            RETURN pub_date, toFloat(total_docs) / toFloat(active_authors) AS productivity_score
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['pub_date']))
                value = float(record['productivity_score'] or 0)
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'type': 'author_productivity', 'domain': domain}
                ))
            
            return data_points
    
    def _preprocess_data(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Preprocess trend data (smoothing, outlier removal, etc.)."""
        
        if len(data) < 3:
            return data
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([
            {'timestamp': point.timestamp, 'value': point.value, 'metadata': point.metadata}
            for point in data
        ])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove outliers using z-score
        z_scores = np.abs(stats.zscore(df['value']))
        df = df[z_scores < self.config.outlier_threshold]
        
        # Apply smoothing if enough data points
        if len(df) >= self.config.smoothing_window:
            df['smoothed_value'] = df['value'].rolling(
                window=self.config.smoothing_window,
                center=True,
                min_periods=1
            ).mean()
        else:
            df['smoothed_value'] = df['value']
        
        # Convert back to TrendPoint objects
        processed_data = []
        for _, row in df.iterrows():
            processed_data.append(TrendPoint(
                timestamp=row['timestamp'],
                value=row['smoothed_value'],
                metadata=row['metadata']
            ))
        
        return processed_data
    
    def _detect_trend_direction(self, data: List[TrendPoint]) -> Tuple[TrendDirection, float]:
        """Detect trend direction and strength."""
        
        if len(data) < 2:
            return TrendDirection.STABLE, 0.0
        
        # Extract values and timestamps
        values = np.array([point.value for point in data])
        timestamps = np.array([(point.timestamp - data[0].timestamp).total_seconds() 
                              for point in data])
        
        # Fit linear regression
        model = LinearRegression()
        X = timestamps.reshape(-1, 1)
        model.fit(X, values)
        
        slope = model.coef_[0]
        r2 = r2_score(values, model.predict(X))
        
        # Calculate volatility
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Determine direction
        if abs(slope) < 1e-10 or r2 < 0.1:
            direction = TrendDirection.STABLE
        elif volatility > self.config.volatility_threshold:
            direction = TrendDirection.VOLATILE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate strength (combination of R² and slope magnitude)
        strength = min(r2 * (1 - volatility), 1.0)
        
        return direction, strength
    
    def _calculate_statistics(self, data: List[TrendPoint]) -> Dict[str, float]:
        """Calculate statistical measures for the trend."""
        
        values = np.array([point.value for point in data])
        
        statistics = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
            'data_points': len(data)
        }
        
        # Calculate growth rate if applicable
        if len(values) > 1:
            start_value = values[0]
            end_value = values[-1]
            if start_value != 0:
                growth_rate = (end_value - start_value) / start_value
                statistics['growth_rate'] = float(growth_rate)
            
            # Calculate average change per period
            changes = np.diff(values)
            statistics['avg_change'] = float(np.mean(changes))
            statistics['volatility'] = float(np.std(changes))
        
        # Detect seasonality
        seasonality_scores = self._detect_seasonality(data)
        statistics.update(seasonality_scores)
        
        return statistics
    
    def _detect_seasonality(self, data: List[TrendPoint]) -> Dict[str, float]:
        """Detect seasonal patterns in the data."""
        
        seasonality_scores = {}
        
        if len(data) < 30:  # Need sufficient data for seasonality detection
            return seasonality_scores
        
        values = np.array([point.value for point in data])
        
        for period in self.config.seasonality_periods:
            if len(values) >= 2 * period:
                # Simple seasonality detection using autocorrelation
                if len(values) > period:
                    correlation = np.corrcoef(values[:-period], values[period:])[0, 1]
                    seasonality_scores[f'seasonality_{period}d'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return seasonality_scores
    
    def _generate_predictions(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Generate future predictions based on historical data."""
        
        if len(data) < self.config.min_data_points:
            return []
        
        # Prepare data for modeling
        values = np.array([point.value for point in data])
        timestamps = np.array([(point.timestamp - data[0].timestamp).total_seconds() / 86400 
                              for point in data])  # Convert to days
        
        # Fit polynomial regression for better predictions
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(timestamps.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_poly, values)
        
        # Generate future timestamps
        last_timestamp = data[-1].timestamp
        future_timestamps = []
        for i in range(1, self.config.prediction_horizon_days + 1):
            future_timestamp = last_timestamp + timedelta(days=i)
            future_timestamps.append(future_timestamp)
        
        # Generate predictions
        predictions = []
        for future_timestamp in future_timestamps:
            days_from_start = (future_timestamp - data[0].timestamp).total_seconds() / 86400
            X_future = poly_features.transform([[days_from_start]])
            predicted_value = model.predict(X_future)[0]
            
            # Ensure non-negative predictions for count data
            predicted_value = max(0, predicted_value)
            
            predictions.append(TrendPoint(
                timestamp=future_timestamp,
                value=predicted_value,
                metadata={'type': 'prediction', 'model': 'polynomial_regression'}
            ))
        
        return predictions
    
    def _extract_insights(
        self,
        data: List[TrendPoint],
        direction: TrendDirection,
        strength: float,
        statistics: Dict[str, float]
    ) -> List[str]:
        """Extract meaningful insights from the trend analysis."""
        
        insights = []
        
        # Direction insights
        if direction == TrendDirection.INCREASING:
            if strength > 0.7:
                insights.append(f"Strong upward trend detected with {strength:.1%} confidence")
            else:
                insights.append(f"Moderate upward trend detected")
        elif direction == TrendDirection.DECREASING:
            if strength > 0.7:
                insights.append(f"Strong downward trend detected with {strength:.1%} confidence")
            else:
                insights.append(f"Moderate downward trend detected")
        elif direction == TrendDirection.VOLATILE:
            insights.append("High volatility detected - trend direction is unclear")
        else:
            insights.append("Trend appears stable with no clear direction")
        
        # Growth rate insights
        if 'growth_rate' in statistics:
            growth_rate = statistics['growth_rate']
            if growth_rate > 0.5:
                insights.append(f"Significant growth of {growth_rate:.1%} over the analysis period")
            elif growth_rate < -0.5:
                insights.append(f"Significant decline of {abs(growth_rate):.1%} over the analysis period")
        
        # Volatility insights
        if 'volatility' in statistics:
            volatility = statistics['volatility']
            mean_value = statistics['mean']
            cv = volatility / mean_value if mean_value != 0 else 0
            
            if cv > 0.5:
                insights.append("High variability detected - values fluctuate significantly")
            elif cv < 0.1:
                insights.append("Low variability detected - values are relatively stable")
        
        # Seasonality insights
        for key, value in statistics.items():
            if key.startswith('seasonality_') and abs(value) > 0.3:
                period = key.split('_')[1]
                insights.append(f"Seasonal pattern detected with {period} periodicity (correlation: {value:.2f})")
        
        # Data quality insights
        data_points = statistics['data_points']
        if data_points < 20:
            insights.append("Limited data available - analysis may be less reliable")
        elif data_points > 100:
            insights.append("Rich dataset available - analysis is highly reliable")
        
        return insights
    
    def _calculate_confidence(self, data: List[TrendPoint], statistics: Dict[str, float]) -> float:
        """Calculate confidence in the trend analysis."""
        
        # Base confidence on data quantity
        data_confidence = min(len(data) / 50.0, 1.0)  # Full confidence at 50+ data points
        
        # Adjust for volatility
        volatility = statistics.get('volatility', 0)
        mean_value = statistics.get('mean', 1)
        cv = volatility / mean_value if mean_value != 0 else 1
        volatility_confidence = max(0, 1 - cv)
        
        # Combine confidences
        overall_confidence = (data_confidence + volatility_confidence) / 2
        
        return min(max(overall_confidence, 0.0), 1.0)
    
    async def _generate_visualization(self, trend_analysis: TrendAnalysis) -> str:
        """Generate visualization for the trend analysis."""
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)
            
            # Extract data for plotting
            timestamps = [point.timestamp for point in trend_analysis.data_points]
            values = [point.value for point in trend_analysis.data_points]
            
            # Main trend plot
            ax1.plot(timestamps, values, 'o-', linewidth=2, markersize=4, label='Actual')
            
            # Add predictions if available
            if trend_analysis.predictions:
                pred_timestamps = [point.timestamp for point in trend_analysis.predictions]
                pred_values = [point.value for point in trend_analysis.predictions]
                ax1.plot(pred_timestamps, pred_values, '--', alpha=0.7, label='Predicted')
            
            ax1.set_title(f"{trend_analysis.trend_type.value.replace('_', ' ').title()} Trend Analysis")
            ax1.set_ylabel("Value")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Statistics plot
            stats_data = {
                'Mean': trend_analysis.statistics.get('mean', 0),
                'Median': trend_analysis.statistics.get('median', 0),
                'Std': trend_analysis.statistics.get('std', 0),
                'Min': trend_analysis.statistics.get('min', 0),
                'Max': trend_analysis.statistics.get('max', 0)
            }
            
            ax2.bar(stats_data.keys(), stats_data.values(), alpha=0.7)
            ax2.set_title("Statistical Summary")
            ax2.set_ylabel("Value")
            
            plt.tight_layout()
            
            # Save plot
            filename = f"trend_{trend_analysis.trend_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            return ""
    
    async def analyze_multiple_trends(
        self,
        trend_types: List[TrendType],
        domain: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[TrendType, TrendAnalysis]:
        """Analyze multiple trends simultaneously."""
        
        results = {}
        
        for trend_type in trend_types:
            try:
                analysis = await self.analyze_trend(
                    trend_type, domain, start_date, end_date
                )
                results[trend_type] = analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing {trend_type.value}: {e}")
                continue
        
        return results
    
    def compare_trends(
        self,
        analyses: Dict[TrendType, TrendAnalysis]
    ) -> Dict[str, Any]:
        """Compare multiple trend analyses."""
        
        comparison = {
            'strongest_trend': None,
            'most_volatile': None,
            'fastest_growing': None,
            'most_predictable': None,
            'correlations': {}
        }
        
        if not analyses:
            return comparison
        
        # Find strongest trend
        max_strength = 0
        for trend_type, analysis in analyses.items():
            if analysis.strength > max_strength:
                max_strength = analysis.strength
                comparison['strongest_trend'] = trend_type.value
        
        # Find most volatile
        max_volatility = 0
        for trend_type, analysis in analyses.items():
            volatility = analysis.statistics.get('volatility', 0)
            if volatility > max_volatility:
                max_volatility = volatility
                comparison['most_volatile'] = trend_type.value
        
        # Find fastest growing
        max_growth = -float('inf')
        for trend_type, analysis in analyses.items():
            growth_rate = analysis.statistics.get('growth_rate', 0)
            if growth_rate > max_growth:
                max_growth = growth_rate
                comparison['fastest_growing'] = trend_type.value
        
        # Find most predictable (highest confidence)
        max_confidence = 0
        for trend_type, analysis in analyses.items():
            if analysis.confidence > max_confidence:
                max_confidence = analysis.confidence
                comparison['most_predictable'] = trend_type.value
        
        return comparison


# CLI Interface
async def main():
    """Main CLI interface for trend analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Trend Analyzer")
    parser.add_argument("--trend-type", choices=[t.value for t in TrendType], 
                       required=True, help="Type of trend to analyze")
    parser.add_argument("--domain", help="Domain to focus on")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--granularity", choices=["daily", "weekly", "monthly", "yearly"],
                       default="daily", help="Time granularity")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None
    
    # Initialize analyzer
    config = TrendConfig(args.config) if args.config else TrendConfig()
    analyzer = TrendAnalyzer(config)
    
    await analyzer.initialize()
    
    try:
        # Run analysis
        trend_type = TrendType(args.trend_type)
        analysis = await analyzer.analyze_trend(
            trend_type, args.domain, start_date, end_date, args.granularity
        )
        
        # Display results
        print(f"\n=== Trend Analysis: {trend_type.value} ===")
        print(f"Direction: {analysis.direction.value}")
        print(f"Strength: {analysis.strength:.2f}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Data Points: {len(analysis.data_points)}")
        
        print(f"\nStatistics:")
        for key, value in analysis.statistics.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nInsights:")
        for insight in analysis.insights:
            print(f"  • {insight}")
        
        if analysis.predictions:
            print(f"\nPredictions (next {len(analysis.predictions)} periods):")
            for pred in analysis.predictions[:5]:  # Show first 5
                print(f"  {pred.timestamp.strftime('%Y-%m-%d')}: {pred.value:.2f}")
    
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())
