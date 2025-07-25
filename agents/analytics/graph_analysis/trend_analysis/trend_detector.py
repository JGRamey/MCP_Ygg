"""
Trend Detection Module
Analyzes trend direction, strength, and characteristics in time series data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..models import TrendDirection, TrendPoint


class TrendDetector:
    """Advanced trend detection and analysis engine."""

    def __init__(self, config=None):
        """Initialize the trend detector."""
        self.config = config
        self.logger = self._setup_logging()

        # Configuration parameters
        self.volatility_threshold = (
            getattr(config, "volatility_threshold", 0.1) if config else 0.1
        )
        self.trend_strength_threshold = (
            getattr(config, "trend_strength_threshold", 0.3) if config else 0.3
        )
        self.min_r2_threshold = (
            getattr(config, "min_r2_threshold", 0.1) if config else 0.1
        )

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("trend_detector")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    async def detect_trend_direction(
        self, data: List[TrendPoint]
    ) -> Tuple[TrendDirection, float]:
        """
        Detect trend direction and strength using multiple methods.

        Args:
            data: List of TrendPoint objects

        Returns:
            Tuple of (trend_direction, strength_score)
        """

        if len(data) < 2:
            return TrendDirection.STABLE, 0.0

        try:
            # Extract values and timestamps
            values = np.array(
                [point.value for point in data if point.value is not None]
            )

            if len(values) < 2:
                return TrendDirection.STABLE, 0.0

            # Primary method: Linear regression
            linear_direction, linear_strength = self._detect_linear_trend(data, values)

            # Secondary method: Moving averages
            ma_direction, ma_strength = self._detect_moving_average_trend(values)

            # Tertiary method: Peak and trough analysis
            extrema_direction, extrema_strength = self._detect_extrema_trend(values)

            # Combine methods with weighted average
            combined_strength = (
                linear_strength * 0.5 + ma_strength * 0.3 + extrema_strength * 0.2
            )

            # Determine final direction based on consensus
            directions = [linear_direction, ma_direction, extrema_direction]
            direction_votes = {}
            for direction in directions:
                direction_votes[direction] = direction_votes.get(direction, 0) + 1

            # Get the most voted direction
            final_direction = max(direction_votes, key=direction_votes.get)

            # If there's no clear consensus and strength is low, default to stable
            if (
                len(direction_votes) > 2
                and combined_strength < self.trend_strength_threshold
            ):
                final_direction = TrendDirection.STABLE
                combined_strength = 0.0

            self.logger.info(
                f"Trend detected: {final_direction.value} (strength: {combined_strength:.3f})"
            )

            return final_direction, float(combined_strength)

        except Exception as e:
            self.logger.error(f"Error in trend detection: {e}")
            return TrendDirection.STABLE, 0.0

    def _detect_linear_trend(
        self, data: List[TrendPoint], values: np.ndarray
    ) -> Tuple[TrendDirection, float]:
        """Detect trend using linear regression."""

        try:
            # Create time series in seconds from start
            timestamps = np.array(
                [
                    (point.timestamp - data[0].timestamp).total_seconds()
                    for point in data
                    if point.value is not None
                ]
            )

            if len(timestamps) != len(values):
                # Fallback to simple indexing
                timestamps = np.arange(len(values))

            # Fit linear regression
            model = LinearRegression()
            X = timestamps.reshape(-1, 1)
            model.fit(X, values)

            slope = model.coef_[0]
            r2 = r2_score(values, model.predict(X))

            # Calculate normalized slope (per unit time)
            time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
            normalized_slope = (
                slope * time_span / np.mean(values) if np.mean(values) != 0 else 0
            )

            # Calculate volatility
            if len(values) > 1:
                returns = np.diff(values) / values[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0
            else:
                volatility = 0

            # Determine direction and strength
            strength = r2 * (1 - min(volatility, 1.0))  # Penalize high volatility

            if r2 < self.min_r2_threshold or abs(normalized_slope) < 1e-6:
                direction = TrendDirection.STABLE
                strength = 0.0
            elif volatility > self.volatility_threshold:
                direction = TrendDirection.CYCLICAL
                strength = min(strength, 0.7)  # Cap cyclical strength
            elif normalized_slope > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            return direction, float(strength)

        except Exception as e:
            self.logger.warning(f"Error in linear trend detection: {e}")
            return TrendDirection.STABLE, 0.0

    def _detect_moving_average_trend(
        self, values: np.ndarray, window_size: Optional[int] = None
    ) -> Tuple[TrendDirection, float]:
        """Detect trend using moving averages."""

        try:
            if window_size is None:
                window_size = max(3, len(values) // 5)  # Adaptive window size

            if len(values) < window_size:
                return TrendDirection.STABLE, 0.0

            # Calculate moving averages
            ma_short = (
                self._moving_average(values, window_size // 2)
                if window_size > 2
                else values
            )
            ma_long = self._moving_average(values, window_size)

            # Compare recent MA values
            recent_points = min(5, len(ma_short) // 3)
            if recent_points < 2:
                return TrendDirection.STABLE, 0.0

            short_recent = ma_short[-recent_points:]
            long_recent = ma_long[-recent_points:]

            # Calculate trend in moving averages
            short_trend = (
                (short_recent[-1] - short_recent[0]) / len(short_recent)
                if len(short_recent) > 1
                else 0
            )
            long_trend = (
                (long_recent[-1] - long_recent[0]) / len(long_recent)
                if len(long_recent) > 1
                else 0
            )

            # Normalize trends
            value_range = np.max(values) - np.min(values)
            if value_range > 0:
                short_trend_norm = short_trend / value_range
                long_trend_norm = long_trend / value_range
            else:
                short_trend_norm = long_trend_norm = 0

            # Determine direction based on both MAs
            avg_trend = (short_trend_norm + long_trend_norm) / 2
            strength = min(abs(avg_trend) * 10, 1.0)  # Scale to 0-1

            if abs(avg_trend) < 0.01:
                direction = TrendDirection.STABLE
            elif avg_trend > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            return direction, float(strength)

        except Exception as e:
            self.logger.warning(f"Error in moving average trend detection: {e}")
            return TrendDirection.STABLE, 0.0

    def _detect_extrema_trend(self, values: np.ndarray) -> Tuple[TrendDirection, float]:
        """Detect trend using peak and trough analysis."""

        try:
            if len(values) < 5:  # Need minimum points for peak detection
                return TrendDirection.STABLE, 0.0

            # Find peaks and troughs
            peaks, _ = find_peaks(values, distance=max(1, len(values) // 10))
            troughs, _ = find_peaks(-values, distance=max(1, len(values) // 10))

            if len(peaks) < 2 and len(troughs) < 2:
                return TrendDirection.STABLE, 0.0

            # Analyze peak heights over time
            peak_trend = 0.0
            trough_trend = 0.0

            if len(peaks) >= 2:
                peak_values = values[peaks]
                peak_trend = (peak_values[-1] - peak_values[0]) / len(peak_values)

            if len(troughs) >= 2:
                trough_values = values[troughs]
                trough_trend = (trough_values[-1] - trough_values[0]) / len(
                    trough_values
                )

            # Normalize trends
            value_range = np.max(values) - np.min(values)
            if value_range > 0:
                peak_trend_norm = peak_trend / value_range
                trough_trend_norm = trough_trend / value_range
            else:
                peak_trend_norm = trough_trend_norm = 0

            # Combine peak and trough trends
            overall_trend = (peak_trend_norm + trough_trend_norm) / 2

            # Calculate strength based on consistency
            strength = min(abs(overall_trend) * 5, 1.0)

            # Check for cyclical patterns
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Look for regular spacing between extrema
                peak_intervals = np.diff(peaks)
                trough_intervals = np.diff(troughs)

                peak_regularity = (
                    1 - (np.std(peak_intervals) / np.mean(peak_intervals))
                    if np.mean(peak_intervals) > 0
                    else 0
                )
                trough_regularity = (
                    1 - (np.std(trough_intervals) / np.mean(trough_intervals))
                    if np.mean(trough_intervals) > 0
                    else 0
                )

                if peak_regularity > 0.7 or trough_regularity > 0.7:
                    return TrendDirection.CYCLICAL, float(min(strength * 1.2, 1.0))

            # Determine direction
            if abs(overall_trend) < 0.01:
                direction = TrendDirection.STABLE
            elif overall_trend > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            return direction, float(strength)

        except Exception as e:
            self.logger.warning(f"Error in extrema trend detection: {e}")
            return TrendDirection.STABLE, 0.0

    def _moving_average(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average with the specified window size."""

        if window_size <= 0 or window_size > len(values):
            return values

        # Use convolution for efficient moving average
        kernel = np.ones(window_size) / window_size
        ma = np.convolve(values, kernel, mode="valid")

        # Pad the beginning to maintain length
        pad_size = window_size - 1
        if pad_size > 0:
            initial_values = np.full(pad_size, values[0])
            ma = np.concatenate([initial_values, ma])

        return ma

    def detect_trend_changes(
        self, data: List[TrendPoint], window_size: int = 10
    ) -> List[Dict[str, any]]:
        """
        Detect points where the trend direction changes.

        Args:
            data: List of TrendPoint objects
            window_size: Size of the sliding window for trend detection

        Returns:
            List of change points with metadata
        """

        if len(data) < window_size * 2:
            return []

        change_points = []
        values = np.array([point.value for point in data if point.value is not None])

        try:
            # Sliding window trend detection
            for i in range(window_size, len(values) - window_size):
                # Analyze trend before and after current point
                before_window = values[i - window_size : i]
                after_window = values[i : i + window_size]

                # Calculate trends for both windows
                before_slope = self._calculate_slope(before_window)
                after_slope = self._calculate_slope(after_window)

                # Detect significant change in slope
                if abs(before_slope - after_slope) > 0.1:  # Threshold for significance
                    change_type = self._classify_change(before_slope, after_slope)

                    change_points.append(
                        {
                            "index": i,
                            "timestamp": data[i].timestamp if i < len(data) else None,
                            "value": float(values[i]),
                            "change_type": change_type,
                            "before_slope": float(before_slope),
                            "after_slope": float(after_slope),
                            "magnitude": float(abs(before_slope - after_slope)),
                        }
                    )

            return change_points

        except Exception as e:
            self.logger.error(f"Error detecting trend changes: {e}")
            return []

    def _calculate_slope(self, values: np.ndarray) -> float:
        """Calculate the slope of a value series."""

        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    def _classify_change(self, before_slope: float, after_slope: float) -> str:
        """Classify the type of trend change."""

        threshold = 0.01

        if abs(before_slope) < threshold and abs(after_slope) < threshold:
            return "stable_to_stable"
        elif abs(before_slope) < threshold and after_slope > threshold:
            return "stable_to_increasing"
        elif abs(before_slope) < threshold and after_slope < -threshold:
            return "stable_to_decreasing"
        elif before_slope > threshold and abs(after_slope) < threshold:
            return "increasing_to_stable"
        elif before_slope < -threshold and abs(after_slope) < threshold:
            return "decreasing_to_stable"
        elif before_slope > threshold and after_slope < -threshold:
            return "increasing_to_decreasing"
        elif before_slope < -threshold and after_slope > threshold:
            return "decreasing_to_increasing"
        elif before_slope > threshold and after_slope > threshold:
            return "increasing_to_increasing"
        elif before_slope < -threshold and after_slope < -threshold:
            return "decreasing_to_decreasing"
        else:
            return "unknown"

    def calculate_trend_strength_score(self, data: List[TrendPoint]) -> float:
        """
        Calculate a comprehensive trend strength score.

        Returns:
            Float between 0 and 1 indicating trend strength
        """

        if len(data) < 3:
            return 0.0

        try:
            values = np.array(
                [point.value for point in data if point.value is not None]
            )

            # Multiple strength indicators
            scores = []

            # 1. R-squared from linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            predicted = slope * x + intercept
            r2 = r2_score(values, predicted)
            scores.append(r2)

            # 2. Consistency of direction
            changes = np.diff(values)
            positive_changes = np.sum(changes > 0)
            negative_changes = np.sum(changes < 0)
            total_changes = len(changes)

            if total_changes > 0:
                direction_consistency = (
                    max(positive_changes, negative_changes) / total_changes
                )
                scores.append(direction_consistency)

            # 3. Magnitude of trend relative to noise
            if len(values) > 1:
                trend_magnitude = abs(values[-1] - values[0])
                noise_level = np.std(np.diff(values))
                if noise_level > 0:
                    signal_to_noise = trend_magnitude / (
                        noise_level * np.sqrt(len(values))
                    )
                    scores.append(min(signal_to_noise / 3, 1.0))  # Normalize

            # 4. Monotonicity score
            if len(values) > 2:
                sorted_values = np.sort(values)
                if np.array_equal(values, sorted_values) or np.array_equal(
                    values, sorted_values[::-1]
                ):
                    scores.append(1.0)  # Perfect monotonicity
                else:
                    # Calculate Spearman rank correlation with position
                    from scipy.stats import spearmanr

                    correlation, _ = spearmanr(np.arange(len(values)), values)
                    scores.append(abs(correlation) if not np.isnan(correlation) else 0)

            # Calculate weighted average
            if scores:
                return float(np.mean(scores))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0


# Factory function for easy integration
def create_trend_detector(config=None) -> TrendDetector:
    """Create and return a TrendDetector instance."""
    return TrendDetector(config)


# Async wrapper for compatibility
async def detect_trend_direction(
    data: List[TrendPoint], config=None
) -> Tuple[TrendDirection, float]:
    """Detect trend direction using the trend detector."""
    detector = create_trend_detector(config)
    return await detector.detect_trend_direction(data)
