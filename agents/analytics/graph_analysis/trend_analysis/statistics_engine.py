"""
Statistics Engine for Trend Analysis

This module provides comprehensive statistical analysis capabilities for time series data,
including descriptive statistics, distribution analysis, correlation calculations, and
statistical confidence measures.

Key Features:
- Descriptive statistics (mean, median, variance, skewness, kurtosis)
- Growth rate and volatility calculations
- Statistical confidence assessment
- Distribution analysis and outlier detection
- Advanced statistical measures for trend data

Author: MCP Yggdrasil Analytics Team
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta

from ..models import TrendPoint, TrendDirection, StatisticalSummary
from ..config import TrendConfig

logger = logging.getLogger(__name__)


class StatisticsEngine:
    """
    Advanced statistical analysis engine for trend data.
    
    Provides comprehensive statistical measures, distribution analysis,
    and confidence calculations for time series trend data.
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        """Initialize the statistics engine."""
        self.config = config or TrendConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def calculate_comprehensive_statistics(
        self, 
        data: List[TrendPoint]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive statistical measures for trend data.
        
        Args:
            data: List of trend data points
            
        Returns:
            Dictionary containing statistical measures
        """
        try:
            if not data:
                return {}
                
            values = np.array([point.value for point in data])
            
            # Basic descriptive statistics
            statistics = self._calculate_descriptive_stats(values)
            
            # Growth and change metrics
            growth_metrics = self._calculate_growth_metrics(values)
            statistics.update(growth_metrics)
            
            # Distribution characteristics
            distribution_stats = self._analyze_distribution(values)
            statistics.update(distribution_stats)
            
            # Variability measures
            variability_stats = self._calculate_variability_measures(values)
            statistics.update(variability_stats)
            
            # Data quality metrics
            quality_metrics = self._assess_data_quality(data)
            statistics.update(quality_metrics)
            
            self.logger.debug(f"Calculated {len(statistics)} statistical measures")
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def _calculate_descriptive_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate basic descriptive statistics."""
        try:
            stats_dict = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'variance': float(np.var(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'data_points': len(values)
            }
            
            # Percentiles
            percentiles = [5, 10, 25, 75, 90, 95]
            for p in percentiles:
                stats_dict[f'p{p}'] = float(np.percentile(values, p))
            
            # Interquartile range
            stats_dict['iqr'] = stats_dict['p75'] - stats_dict['p25']
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating descriptive stats: {e}")
            return {}
    
    def _calculate_growth_metrics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate growth and change-related metrics."""
        try:
            growth_metrics = {}
            
            if len(values) > 1:
                # Overall growth rate
                start_value = values[0]
                end_value = values[-1]
                if start_value != 0:
                    growth_rate = (end_value - start_value) / start_value
                    growth_metrics['growth_rate'] = float(growth_rate)
                    growth_metrics['growth_rate_percent'] = float(growth_rate * 100)
                
                # Period-over-period changes
                changes = np.diff(values)
                growth_metrics['avg_change'] = float(np.mean(changes))
                growth_metrics['max_change'] = float(np.max(changes))
                growth_metrics['min_change'] = float(np.min(changes))
                
                # Cumulative growth
                if start_value != 0:
                    cumulative_returns = (values - start_value) / start_value
                    growth_metrics['max_drawdown'] = float(np.min(cumulative_returns))
                    growth_metrics['max_gain'] = float(np.max(cumulative_returns))
                
                # Compound Annual Growth Rate (CAGR) approximation
                if start_value > 0 and len(values) > 1:
                    periods = len(values) - 1
                    cagr = (end_value / start_value) ** (1/periods) - 1
                    growth_metrics['cagr'] = float(cagr)
            
            return growth_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating growth metrics: {e}")
            return {}
    
    def _analyze_distribution(self, values: np.ndarray) -> Dict[str, float]:
        """Analyze the distribution characteristics of the data."""
        try:
            distribution_stats = {}
            
            # Shape statistics
            if len(values) > 2:
                distribution_stats['skewness'] = float(stats.skew(values))
                distribution_stats['kurtosis'] = float(stats.kurtosis(values))
                
                # Normality tests
                if len(values) >= 8:  # Minimum for Shapiro-Wilk
                    statistic, p_value = stats.shapiro(values)
                    distribution_stats['normality_statistic'] = float(statistic)
                    distribution_stats['normality_p_value'] = float(p_value)
                    distribution_stats['is_normal'] = float(p_value > 0.05)
            
            # Outlier detection using IQR method
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            distribution_stats['outlier_count'] = len(outliers)
            distribution_stats['outlier_percentage'] = float(len(outliers) / len(values) * 100)
            
            # Distribution spread measures
            distribution_stats['coefficient_of_variation'] = float(
                np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            )
            
            return distribution_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing distribution: {e}")
            return {}
    
    def _calculate_variability_measures(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate various measures of variability and volatility."""
        try:
            variability_stats = {}
            
            if len(values) > 1:
                # Basic volatility (standard deviation of changes)
                changes = np.diff(values)
                variability_stats['volatility'] = float(np.std(changes))
                
                # Relative volatility
                mean_value = np.mean(values)
                if mean_value != 0:
                    variability_stats['relative_volatility'] = float(
                        np.std(changes) / mean_value
                    )
                
                # Average absolute deviation
                variability_stats['mean_absolute_deviation'] = float(
                    np.mean(np.abs(values - np.mean(values)))
                )
                
                # Median absolute deviation
                median_val = np.median(values)
                variability_stats['median_absolute_deviation'] = float(
                    np.median(np.abs(values - median_val))
                )
                
                # Range-based measures
                variability_stats['relative_range'] = float(
                    (np.max(values) - np.min(values)) / np.mean(values) 
                    if np.mean(values) != 0 else 0
                )
                
                # Stability measures
                variability_stats['stability_index'] = float(
                    1 / (1 + variability_stats['coefficient_of_variation'])
                    if 'coefficient_of_variation' in locals() else 1
                )
            
            return variability_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating variability measures: {e}")
            return {}
    
    def _assess_data_quality(self, data: List[TrendPoint]) -> Dict[str, float]:
        """Assess the quality of the trend data."""
        try:
            quality_metrics = {}
            
            values = np.array([point.value for point in data])
            
            # Data completeness
            quality_metrics['data_completeness'] = float(len(data) / max(len(data), 1))
            
            # Missing values (represented as NaN or None)
            nan_count = np.sum(np.isnan(values))
            quality_metrics['missing_value_percentage'] = float(nan_count / len(values) * 100)
            
            # Data consistency (check for duplicate timestamps)
            timestamps = [point.timestamp for point in data]
            unique_timestamps = len(set(timestamps))
            quality_metrics['timestamp_uniqueness'] = float(unique_timestamps / len(timestamps))
            
            # Temporal regularity (check for consistent intervals)
            if len(timestamps) > 2:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                
                if intervals:
                    interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) != 0 else 0
                    quality_metrics['temporal_regularity'] = float(1 / (1 + interval_cv))
            
            # Data reliability score (composite measure)
            reliability_factors = [
                quality_metrics.get('data_completeness', 0),
                1 - quality_metrics.get('missing_value_percentage', 0) / 100,
                quality_metrics.get('timestamp_uniqueness', 0),
                quality_metrics.get('temporal_regularity', 1)
            ]
            quality_metrics['reliability_score'] = float(np.mean(reliability_factors))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return {}
    
    async def calculate_confidence_metrics(
        self, 
        data: List[TrendPoint], 
        statistics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate confidence metrics for trend analysis.
        
        Args:
            data: Trend data points
            statistics: Pre-calculated statistics
            
        Returns:
            Dictionary containing confidence measures
        """
        try:
            confidence_metrics = {}
            
            # Data quantity confidence
            data_confidence = min(len(data) / 50.0, 1.0)  # Full confidence at 50+ points
            confidence_metrics['data_quantity_confidence'] = float(data_confidence)
            
            # Volatility-based confidence
            volatility = statistics.get('volatility', 0)
            mean_value = statistics.get('mean', 1)
            cv = volatility / mean_value if mean_value != 0 else 1
            volatility_confidence = max(0, 1 - cv)
            confidence_metrics['volatility_confidence'] = float(volatility_confidence)
            
            # Data quality confidence
            reliability_score = statistics.get('reliability_score', 0.5)
            confidence_metrics['quality_confidence'] = float(reliability_score)
            
            # Normality confidence (how well data fits normal distribution)
            normality_p = statistics.get('normality_p_value', 0.5)
            normality_confidence = min(normality_p * 2, 1.0)  # Scale p-value to confidence
            confidence_metrics['normality_confidence'] = float(normality_confidence)
            
            # Overall confidence (weighted average)
            weights = {
                'data_quantity_confidence': 0.3,
                'volatility_confidence': 0.3,
                'quality_confidence': 0.3,
                'normality_confidence': 0.1
            }
            
            overall_confidence = sum(
                confidence_metrics[key] * weight 
                for key, weight in weights.items() 
                if key in confidence_metrics
            )
            
            confidence_metrics['overall_confidence'] = float(
                min(max(overall_confidence, 0.0), 1.0)
            )
            
            return confidence_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {e}")
            return {}
    
    async def perform_correlation_analysis(
        self, 
        primary_data: List[TrendPoint],
        comparison_data: Optional[List[TrendPoint]] = None
    ) -> Dict[str, float]:
        """
        Perform correlation analysis between data series.
        
        Args:
            primary_data: Primary trend data
            comparison_data: Optional comparison data for correlation
            
        Returns:
            Dictionary containing correlation measures
        """
        try:
            correlation_metrics = {}
            
            if not comparison_data:
                # Auto-correlation analysis
                values = np.array([point.value for point in primary_data])
                
                # Lag-1 autocorrelation
                if len(values) > 1:
                    lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1]
                    correlation_metrics['lag1_autocorrelation'] = float(
                        lag1_corr if not np.isnan(lag1_corr) else 0
                    )
                
                # Multiple lag autocorrelations
                max_lag = min(10, len(values) // 4)
                for lag in range(2, max_lag + 1):
                    if len(values) > lag:
                        lag_corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                        correlation_metrics[f'lag{lag}_autocorrelation'] = float(
                            lag_corr if not np.isnan(lag_corr) else 0
                        )
            else:
                # Cross-correlation analysis
                primary_values = np.array([point.value for point in primary_data])
                comparison_values = np.array([point.value for point in comparison_data])
                
                # Ensure same length (truncate to shorter)
                min_length = min(len(primary_values), len(comparison_values))
                primary_values = primary_values[:min_length]
                comparison_values = comparison_values[:min_length]
                
                if min_length > 1:
                    # Pearson correlation
                    pearson_corr = np.corrcoef(primary_values, comparison_values)[0, 1]
                    correlation_metrics['pearson_correlation'] = float(
                        pearson_corr if not np.isnan(pearson_corr) else 0
                    )
                    
                    # Spearman rank correlation
                    spearman_corr, _ = stats.spearmanr(primary_values, comparison_values)
                    correlation_metrics['spearman_correlation'] = float(
                        spearman_corr if not np.isnan(spearman_corr) else 0
                    )
                    
                    # Kendall's tau
                    kendall_tau, _ = stats.kendalltau(primary_values, comparison_values)
                    correlation_metrics['kendall_tau'] = float(
                        kendall_tau if not np.isnan(kendall_tau) else 0
                    )
            
            return correlation_metrics
            
        except Exception as e:
            self.logger.error(f"Error performing correlation analysis: {e}")
            return {}
    
    def generate_statistical_summary(
        self, 
        statistics: Dict[str, float], 
        confidence_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Generate human-readable statistical insights.
        
        Args:
            statistics: Statistical measures
            confidence_metrics: Confidence measures
            
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            # Data quality insights
            data_points = statistics.get('data_points', 0)
            reliability_score = statistics.get('reliability_score', 0)
            
            if data_points < 20:
                insights.append("Limited data available - analysis may be less reliable")
            elif data_points > 100:
                insights.append("Rich dataset available - analysis is highly reliable")
            
            if reliability_score > 0.8:
                insights.append("High data quality detected - results are trustworthy")
            elif reliability_score < 0.5:
                insights.append("Data quality issues detected - interpret results cautiously")
            
            # Distribution insights
            skewness = statistics.get('skewness', 0)
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(f"Significant {direction}-skewed distribution detected")
            
            # Variability insights
            cv = statistics.get('coefficient_of_variation', 0)
            if cv > 0.5:
                insights.append("High variability detected - values fluctuate significantly")
            elif cv < 0.1:
                insights.append("Low variability detected - values are relatively stable")
            
            # Growth insights
            growth_rate = statistics.get('growth_rate', 0)
            if abs(growth_rate) > 0.5:
                direction = "growth" if growth_rate > 0 else "decline"
                insights.append(f"Significant {direction} of {abs(growth_rate):.1%} detected")
            
            # Confidence insights
            overall_confidence = confidence_metrics.get('overall_confidence', 0)
            if overall_confidence > 0.8:
                insights.append("High confidence in statistical analysis")
            elif overall_confidence < 0.5:
                insights.append("Lower confidence due to data limitations")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating statistical summary: {e}")
            return []


# Factory function for easy instantiation
def create_statistics_engine(config: Optional[TrendConfig] = None) -> StatisticsEngine:
    """
    Create and configure a StatisticsEngine instance.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured StatisticsEngine instance
    """
    return StatisticsEngine(config)


# Export main classes and functions
__all__ = [
    'StatisticsEngine',
    'create_statistics_engine'
]