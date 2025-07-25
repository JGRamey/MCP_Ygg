"""
Seasonality Detector for Trend Analysis

This module provides advanced seasonality detection and analysis capabilities
for time series data, including multiple detection algorithms, seasonal
decomposition, and pattern recognition.

Key Features:
- Multiple seasonality detection algorithms (autocorrelation, FFT, decomposition)
- Seasonal pattern strength assessment
- Periodic pattern identification
- Seasonal trend decomposition
- Calendar-based seasonality detection

Author: MCP Yggdrasil Analytics Team
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq

from ..config import TrendConfig
from ..models import SeasonalDecomposition, SeasonalityPattern, TrendPoint

logger = logging.getLogger(__name__)


class SeasonalityDetector:
    """
    Advanced seasonality detection and analysis engine.

    Provides multiple algorithms for detecting seasonal patterns in time series data,
    including autocorrelation-based detection, frequency domain analysis, and
    statistical decomposition methods.
    """

    def __init__(self, config: Optional[TrendConfig] = None):
        """Initialize the seasonality detector."""
        self.config = config or TrendConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Default seasonality periods to check (in days)
        self.default_periods = [
            7,
            14,
            30,
            90,
            365,
        ]  # Weekly, bi-weekly, monthly, quarterly, yearly

        # Minimum correlation threshold for seasonality detection
        self.correlation_threshold = 0.3

        # Minimum data points required for reliable detection
        self.min_data_points = 30

    async def detect_seasonality_patterns(
        self, data: List[TrendPoint], custom_periods: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns using multiple algorithms.

        Args:
            data: Time series data points
            custom_periods: Optional custom periods to check (in days)

        Returns:
            Dictionary containing seasonality analysis results
        """
        try:
            if len(data) < self.min_data_points:
                self.logger.warning(
                    f"Insufficient data for seasonality detection: {len(data)} points"
                )
                return {
                    "has_seasonality": False,
                    "confidence": 0.0,
                    "patterns": [],
                    "method": "insufficient_data",
                }

            periods = custom_periods or getattr(
                self.config, "seasonality_periods", self.default_periods
            )

            # Run multiple detection algorithms
            autocorr_results = await self._autocorrelation_detection(data, periods)
            fft_results = await self._frequency_domain_detection(data)
            decomp_results = await self._decomposition_detection(data, periods)
            calendar_results = await self._calendar_based_detection(data)

            # Combine results and determine overall seasonality
            combined_results = self._combine_detection_results(
                autocorr_results, fft_results, decomp_results, calendar_results
            )

            return combined_results

        except Exception as e:
            self.logger.error(f"Error detecting seasonality patterns: {e}")
            return {
                "has_seasonality": False,
                "confidence": 0.0,
                "patterns": [],
                "method": "error",
                "error": str(e),
            }

    async def _autocorrelation_detection(
        self, data: List[TrendPoint], periods: List[int]
    ) -> Dict[str, Any]:
        """Detect seasonality using autocorrelation analysis."""
        try:
            values = np.array([point.value for point in data])
            seasonality_scores = {}
            significant_patterns = []

            for period in periods:
                if len(values) >= 2 * period:
                    # Calculate autocorrelation at the specified lag
                    if len(values) > period:
                        correlation = np.corrcoef(values[:-period], values[period:])[
                            0, 1
                        ]
                        correlation = correlation if not np.isnan(correlation) else 0.0

                        seasonality_scores[f"period_{period}d"] = float(correlation)

                        # Check if correlation is significant
                        if abs(correlation) > self.correlation_threshold:
                            significant_patterns.append(
                                {
                                    "period": period,
                                    "strength": abs(correlation),
                                    "type": (
                                        "positive" if correlation > 0 else "negative"
                                    ),
                                    "method": "autocorrelation",
                                }
                            )

            # Calculate overall seasonality confidence
            max_correlation = max(
                [abs(score) for score in seasonality_scores.values()], default=0
            )
            confidence = min(max_correlation * 1.5, 1.0)  # Scale to 0-1 range

            return {
                "method": "autocorrelation",
                "scores": seasonality_scores,
                "patterns": significant_patterns,
                "confidence": confidence,
                "has_seasonality": len(significant_patterns) > 0,
            }

        except Exception as e:
            self.logger.error(f"Error in autocorrelation detection: {e}")
            return {"method": "autocorrelation", "error": str(e)}

    async def _frequency_domain_detection(
        self, data: List[TrendPoint]
    ) -> Dict[str, Any]:
        """Detect seasonality using frequency domain analysis (FFT)."""
        try:
            values = np.array([point.value for point in data])

            # Remove trend to focus on periodic components
            detrended = signal.detrend(values)

            # Apply FFT
            fft_values = fft(detrended)
            frequencies = fftfreq(len(detrended))

            # Calculate power spectrum
            power_spectrum = np.abs(fft_values) ** 2

            # Find dominant frequencies (excluding DC component)
            positive_freqs = frequencies[1 : len(frequencies) // 2]
            positive_power = power_spectrum[1 : len(power_spectrum) // 2]

            # Identify peaks in power spectrum
            peak_indices, properties = signal.find_peaks(
                positive_power,
                height=np.mean(positive_power) * 2,  # Peaks must be 2x average
                distance=len(positive_power) // 20,  # Minimum distance between peaks
            )

            # Convert frequencies to periods (in data points)
            significant_patterns = []
            if len(peak_indices) > 0:
                peak_frequencies = positive_freqs[peak_indices]
                peak_powers = positive_power[peak_indices]

                for freq, power in zip(peak_frequencies, peak_powers):
                    if freq > 0:
                        period_points = 1 / freq
                        # Convert to approximate days (assuming daily data)
                        period_days = int(round(period_points))

                        if 2 <= period_days <= len(values) // 2:
                            strength = power / np.max(
                                positive_power
                            )  # Normalize by max power

                            significant_patterns.append(
                                {
                                    "period": period_days,
                                    "strength": float(strength),
                                    "frequency": float(freq),
                                    "power": float(power),
                                    "method": "fft",
                                }
                            )

            # Sort by strength
            significant_patterns.sort(key=lambda x: x["strength"], reverse=True)

            # Calculate confidence based on peak strength
            confidence = (
                significant_patterns[0]["strength"] if significant_patterns else 0.0
            )

            return {
                "method": "fft",
                "patterns": significant_patterns[:5],  # Top 5 patterns
                "confidence": confidence,
                "has_seasonality": len(significant_patterns) > 0,
            }

        except Exception as e:
            self.logger.error(f"Error in FFT detection: {e}")
            return {"method": "fft", "error": str(e)}

    async def _decomposition_detection(
        self, data: List[TrendPoint], periods: List[int]
    ) -> Dict[str, Any]:
        """Detect seasonality using seasonal decomposition."""
        try:
            # Convert to pandas Series with datetime index
            timestamps = [point.timestamp for point in data]
            values = [point.value for point in data]

            ts_series = pd.Series(values, index=pd.to_datetime(timestamps))

            # Try seasonal decomposition for different periods
            decomposition_results = []

            for period in periods:
                if len(ts_series) >= 2 * period:
                    try:
                        # Resample to regular frequency if needed
                        freq_map = {7: "W", 30: "M", 90: "Q", 365: "Y"}
                        freq = freq_map.get(period, "D")

                        # Simple moving average decomposition
                        trend = ts_series.rolling(window=period, center=True).mean()
                        detrended = ts_series - trend
                        seasonal = detrended.rolling(window=period).mean()
                        residual = detrended - seasonal

                        # Calculate seasonal strength
                        seasonal_var = seasonal.var()
                        residual_var = residual.var()

                        if residual_var > 0:
                            seasonal_strength = seasonal_var / (
                                seasonal_var + residual_var
                            )
                        else:
                            seasonal_strength = 0

                        if (
                            seasonal_strength > 0.1
                        ):  # Threshold for meaningful seasonality
                            decomposition_results.append(
                                {
                                    "period": period,
                                    "strength": float(seasonal_strength),
                                    "method": "decomposition",
                                    "trend_strength": float(
                                        trend.var() / ts_series.var()
                                    ),
                                    "residual_strength": float(
                                        residual_var / ts_series.var()
                                    ),
                                }
                            )

                    except Exception as period_error:
                        self.logger.debug(
                            f"Decomposition failed for period {period}: {period_error}"
                        )
                        continue

            # Sort by strength
            decomposition_results.sort(key=lambda x: x["strength"], reverse=True)

            # Calculate overall confidence
            confidence = (
                decomposition_results[0]["strength"] if decomposition_results else 0.0
            )

            return {
                "method": "decomposition",
                "patterns": decomposition_results,
                "confidence": confidence,
                "has_seasonality": len(decomposition_results) > 0,
            }

        except Exception as e:
            self.logger.error(f"Error in decomposition detection: {e}")
            return {"method": "decomposition", "error": str(e)}

    async def _calendar_based_detection(self, data: List[TrendPoint]) -> Dict[str, Any]:
        """Detect calendar-based seasonal patterns (day of week, month, etc.)."""
        try:
            # Group data by calendar features
            calendar_patterns = defaultdict(list)

            for point in data:
                # Day of week (0=Monday, 6=Sunday)
                dow = point.timestamp.weekday()
                calendar_patterns["day_of_week"].append((dow, point.value))

                # Month
                month = point.timestamp.month
                calendar_patterns["month"].append((month, point.value))

                # Quarter
                quarter = (point.timestamp.month - 1) // 3 + 1
                calendar_patterns["quarter"].append((quarter, point.value))

                # Day of month
                day = point.timestamp.day
                calendar_patterns["day_of_month"].append((day, point.value))

            # Analyze patterns
            pattern_results = []

            for pattern_type, pattern_data in calendar_patterns.items():
                if len(pattern_data) >= 10:  # Minimum data for analysis
                    # Group by calendar unit and calculate statistics
                    grouped_data = defaultdict(list)
                    for unit, value in pattern_data:
                        grouped_data[unit].append(value)

                    # Calculate variance within vs between groups
                    group_means = []
                    within_group_var = 0
                    total_count = 0

                    for unit, values in grouped_data.items():
                        if len(values) > 1:
                            group_mean = np.mean(values)
                            group_means.append(group_mean)
                            within_group_var += np.var(values) * len(values)
                            total_count += len(values)

                    if total_count > 0 and len(group_means) > 1:
                        within_group_var /= total_count
                        between_group_var = np.var(group_means)

                        # Calculate F-statistic (measure of seasonal effect)
                        if within_group_var > 0:
                            f_statistic = between_group_var / within_group_var
                            seasonal_strength = min(f_statistic / 10, 1.0)  # Normalize

                            if seasonal_strength > 0.2:
                                pattern_results.append(
                                    {
                                        "pattern_type": pattern_type,
                                        "strength": float(seasonal_strength),
                                        "f_statistic": float(f_statistic),
                                        "method": "calendar_based",
                                        "groups": len(grouped_data),
                                        "group_means": {
                                            k: float(np.mean(v))
                                            for k, v in grouped_data.items()
                                        },
                                    }
                                )

            # Sort by strength
            pattern_results.sort(key=lambda x: x["strength"], reverse=True)

            # Calculate confidence
            confidence = pattern_results[0]["strength"] if pattern_results else 0.0

            return {
                "method": "calendar_based",
                "patterns": pattern_results,
                "confidence": confidence,
                "has_seasonality": len(pattern_results) > 0,
            }

        except Exception as e:
            self.logger.error(f"Error in calendar-based detection: {e}")
            return {"method": "calendar_based", "error": str(e)}

    def _combine_detection_results(
        self,
        autocorr_results: Dict[str, Any],
        fft_results: Dict[str, Any],
        decomp_results: Dict[str, Any],
        calendar_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine results from multiple detection methods."""
        try:
            # Collect all patterns
            all_patterns = []

            # Add patterns from each method
            for result in [
                autocorr_results,
                fft_results,
                decomp_results,
                calendar_results,
            ]:
                if "patterns" in result:
                    for pattern in result["patterns"]:
                        pattern["source_method"] = result["method"]
                        all_patterns.append(pattern)

            # Calculate combined confidence
            method_confidences = []
            for result in [
                autocorr_results,
                fft_results,
                decomp_results,
                calendar_results,
            ]:
                if "confidence" in result:
                    method_confidences.append(result["confidence"])

            if method_confidences:
                # Use weighted average with higher weight for methods that found patterns
                weights = [1.0 if conf > 0 else 0.5 for conf in method_confidences]
                combined_confidence = np.average(method_confidences, weights=weights)
            else:
                combined_confidence = 0.0

            # Determine overall seasonality
            has_seasonality = any(
                result.get("has_seasonality", False)
                for result in [
                    autocorr_results,
                    fft_results,
                    decomp_results,
                    calendar_results,
                ]
            )

            # Sort patterns by strength
            all_patterns.sort(key=lambda x: x.get("strength", 0), reverse=True)

            # Generate insights
            insights = self._generate_seasonality_insights(
                all_patterns, combined_confidence
            )

            return {
                "has_seasonality": has_seasonality,
                "confidence": float(combined_confidence),
                "patterns": all_patterns[:10],  # Top 10 patterns
                "insights": insights,
                "method_results": {
                    "autocorrelation": autocorr_results,
                    "fft": fft_results,
                    "decomposition": decomp_results,
                    "calendar_based": calendar_results,
                },
            }

        except Exception as e:
            self.logger.error(f"Error combining detection results: {e}")
            return {
                "has_seasonality": False,
                "confidence": 0.0,
                "patterns": [],
                "error": str(e),
            }

    def _generate_seasonality_insights(
        self, patterns: List[Dict[str, Any]], confidence: float
    ) -> List[str]:
        """Generate human-readable insights about detected seasonality."""
        insights = []

        try:
            if not patterns:
                insights.append("No significant seasonal patterns detected")
                return insights

            # Overall seasonality assessment
            if confidence > 0.7:
                insights.append(
                    "Strong seasonal patterns detected with high confidence"
                )
            elif confidence > 0.4:
                insights.append("Moderate seasonal patterns detected")
            else:
                insights.append(
                    "Weak seasonal patterns detected - interpret cautiously"
                )

            # Pattern-specific insights
            for i, pattern in enumerate(patterns[:3]):  # Top 3 patterns
                method = pattern.get("source_method", "unknown")
                strength = pattern.get("strength", 0)

                if "period" in pattern:
                    period = pattern["period"]
                    period_desc = self._get_period_description(period)
                    insights.append(
                        f"Pattern #{i+1}: {period_desc} cycle detected "
                        f"(strength: {strength:.2f}, method: {method})"
                    )
                elif "pattern_type" in pattern:
                    pattern_type = pattern["pattern_type"]
                    insights.append(
                        f"Pattern #{i+1}: {pattern_type} seasonality detected "
                        f"(strength: {strength:.2f}, method: {method})"
                    )

            # Method consensus
            methods_with_patterns = set(p.get("source_method") for p in patterns)
            if len(methods_with_patterns) > 2:
                insights.append(
                    f"Multiple detection methods confirm seasonality ({len(methods_with_patterns)} methods)"
                )

            return insights

        except Exception as e:
            self.logger.error(f"Error generating seasonality insights: {e}")
            return ["Error generating seasonality insights"]

    def _get_period_description(self, period_days: int) -> str:
        """Convert period in days to human-readable description."""
        if period_days <= 1:
            return "daily"
        elif period_days <= 3:
            return "short-term"
        elif period_days <= 9:
            return "weekly"
        elif period_days <= 16:
            return "bi-weekly"
        elif period_days <= 40:
            return "monthly"
        elif period_days <= 120:
            return "quarterly"
        elif period_days <= 400:
            return "annual"
        else:
            return "long-term"


# Factory function for easy instantiation
def create_seasonality_detector(
    config: Optional[TrendConfig] = None,
) -> SeasonalityDetector:
    """
    Create and configure a SeasonalityDetector instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured SeasonalityDetector instance
    """
    return SeasonalityDetector(config)


# Export main classes and functions
__all__ = ["SeasonalityDetector", "create_seasonality_detector"]
