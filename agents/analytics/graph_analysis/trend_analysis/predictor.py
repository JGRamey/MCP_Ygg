"""
Trend Prediction Module
Generates future predictions based on historical trend data using multiple algorithms.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings

from ..models import TrendPoint


class TrendPredictor:
    """Advanced trend prediction engine with multiple forecasting methods."""
    
    def __init__(self, config=None):
        """Initialize the trend predictor."""
        self.config = config
        self.logger = self._setup_logging()
        
        # Configuration parameters
        self.min_data_points = getattr(config, 'min_data_points', 10) if config else 10
        self.prediction_horizon_days = getattr(config, 'prediction_horizon_days', 30) if config else 30
        self.confidence_intervals = getattr(config, 'confidence_intervals', True) if config else True
        
        # Suppress sklearn warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("trend_predictor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def generate_predictions(
        self,
        data: List[TrendPoint],
        horizon_days: Optional[int] = None,
        methods: Optional[List[str]] = None
    ) -> List[TrendPoint]:
        """
        Generate future predictions using multiple forecasting methods.
        
        Args:
            data: Historical trend data
            horizon_days: Days to predict into the future
            methods: List of prediction methods to use
            
        Returns:
            List of predicted TrendPoint objects
        """
        
        if len(data) < self.min_data_points:
            self.logger.warning(f"Insufficient data points for prediction: {len(data)} < {self.min_data_points}")
            return []
        
        horizon = horizon_days if horizon_days is not None else self.prediction_horizon_days
        methods = methods or ['linear', 'polynomial', 'exponential_smoothing']
        
        try:
            # Prepare data
            values, timestamps = self._prepare_data(data)
            
            if len(values) < 3:
                return []
            
            # Generate predictions with multiple methods
            predictions_by_method = {}
            
            for method in methods:
                try:
                    method_predictions = await self._predict_with_method(
                        values, timestamps, data, horizon, method
                    )
                    if method_predictions:
                        predictions_by_method[method] = method_predictions
                except Exception as e:
                    self.logger.warning(f"Prediction method {method} failed: {e}")
            
            if not predictions_by_method:
                self.logger.warning("All prediction methods failed")
                return []
            
            # Ensemble predictions (average multiple methods)
            ensemble_predictions = self._create_ensemble_predictions(
                predictions_by_method, data[-1].timestamp, horizon
            )
            
            self.logger.info(f"Generated {len(ensemble_predictions)} predictions using {len(predictions_by_method)} methods")
            
            return ensemble_predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return []
    
    def _prepare_data(self, data: List[TrendPoint]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for prediction models."""
        
        # Extract values and timestamps
        values = np.array([point.value for point in data if point.value is not None])
        
        # Convert timestamps to days from start
        timestamps = np.array([
            (point.timestamp - data[0].timestamp).total_seconds() / 86400 
            for point in data if point.value is not None
        ])
        
        return values, timestamps
    
    async def _predict_with_method(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int,
        method: str
    ) -> List[TrendPoint]:
        """Generate predictions using a specific method."""
        
        if method == 'linear':
            return self._linear_prediction(values, timestamps, original_data, horizon)
        elif method == 'polynomial':
            return self._polynomial_prediction(values, timestamps, original_data, horizon)
        elif method == 'exponential_smoothing':
            return self._exponential_smoothing_prediction(values, original_data, horizon)
        elif method == 'trend_decomposition':
            return self._trend_decomposition_prediction(values, timestamps, original_data, horizon)
        elif method == 'moving_average':
            return self._moving_average_prediction(values, original_data, horizon)
        else:
            self.logger.warning(f"Unknown prediction method: {method}")
            return []
    
    def _linear_prediction(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int
    ) -> List[TrendPoint]:
        """Generate predictions using linear regression."""
        
        try:
            # Fit linear model
            model = LinearRegression()
            X = timestamps.reshape(-1, 1)
            model.fit(X, values)
            
            # Calculate model quality metrics
            predictions_train = model.predict(X)
            mse = mean_squared_error(values, predictions_train)
            r2 = model.score(X, values)
            
            # Generate future predictions
            last_timestamp = original_data[-1].timestamp
            predictions = []
            
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                days_from_start = (future_timestamp - original_data[0].timestamp).total_seconds() / 86400
                
                predicted_value = model.predict([[days_from_start]])[0]
                
                # Calculate confidence bounds (simple approach)
                std_error = np.sqrt(mse)
                confidence_95 = 1.96 * std_error
                
                predictions.append(TrendPoint(
                    timestamp=future_timestamp,
                    value=max(0, predicted_value),  # Ensure non-negative
                    metadata={
                        'type': 'prediction',
                        'method': 'linear',
                        'confidence_lower': max(0, predicted_value - confidence_95),
                        'confidence_upper': predicted_value + confidence_95,
                        'model_r2': r2,
                        'model_mse': mse
                    }
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in linear prediction: {e}")
            return []
    
    def _polynomial_prediction(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int,
        degree: int = 2
    ) -> List[TrendPoint]:
        """Generate predictions using polynomial regression."""
        
        try:
            # Prevent overfitting with small datasets
            max_degree = min(degree, len(values) // 3, 3)
            if max_degree < 1:
                max_degree = 1
            
            # Fit polynomial model
            poly_features = PolynomialFeatures(degree=max_degree)
            X_poly = poly_features.fit_transform(timestamps.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(X_poly, values)
            
            # Calculate model quality
            predictions_train = model.predict(X_poly)
            mse = mean_squared_error(values, predictions_train)
            r2 = model.score(X_poly, values)
            
            # Generate future predictions
            last_timestamp = original_data[-1].timestamp
            predictions = []
            
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                days_from_start = (future_timestamp - original_data[0].timestamp).total_seconds() / 86400
                
                X_future = poly_features.transform([[days_from_start]])
                predicted_value = model.predict(X_future)[0]
                
                # Simple confidence estimation
                std_error = np.sqrt(mse)
                confidence_95 = 1.96 * std_error
                
                predictions.append(TrendPoint(
                    timestamp=future_timestamp,
                    value=max(0, predicted_value),
                    metadata={
                        'type': 'prediction',
                        'method': f'polynomial_degree_{max_degree}',
                        'confidence_lower': max(0, predicted_value - confidence_95),
                        'confidence_upper': predicted_value + confidence_95,
                        'model_r2': r2,
                        'model_mse': mse
                    }
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in polynomial prediction: {e}")
            return []
    
    def _exponential_smoothing_prediction(
        self,
        values: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int,
        alpha: float = 0.3
    ) -> List[TrendPoint]:
        """Generate predictions using exponential smoothing."""
        
        try:
            if len(values) < 2:
                return []
            
            # Simple exponential smoothing
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            
            for i in range(1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            
            # Calculate trend component
            if len(values) >= 3:
                trend = (smoothed[-1] - smoothed[-3]) / 2  # Simple trend estimation
            else:
                trend = smoothed[-1] - smoothed[-2] if len(smoothed) >= 2 else 0
            
            # Generate predictions
            last_timestamp = original_data[-1].timestamp
            last_value = smoothed[-1]
            predictions = []
            
            # Calculate variance for confidence intervals
            residuals = values - smoothed
            variance = np.var(residuals) if len(residuals) > 1 else 0
            std_error = np.sqrt(variance)
            
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                predicted_value = last_value + trend * i
                
                # Increasing uncertainty over time
                uncertainty_factor = np.sqrt(i) * std_error
                confidence_95 = 1.96 * uncertainty_factor
                
                predictions.append(TrendPoint(
                    timestamp=future_timestamp,
                    value=max(0, predicted_value),
                    metadata={
                        'type': 'prediction',
                        'method': 'exponential_smoothing',
                        'alpha': alpha,
                        'trend_component': trend,
                        'confidence_lower': max(0, predicted_value - confidence_95),
                        'confidence_upper': predicted_value + confidence_95
                    }
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in exponential smoothing prediction: {e}")
            return []
    
    def _trend_decomposition_prediction(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int
    ) -> List[TrendPoint]:
        """Generate predictions using trend decomposition."""
        
        try:
            if len(values) < 10:  # Need sufficient data for decomposition
                return []
            
            # Simple trend extraction using moving averages
            window_size = max(3, len(values) // 5)
            trend = self._extract_trend(values, window_size)
            
            # Extract seasonal component (if pattern exists)
            detrended = values - trend
            seasonal = self._extract_seasonal_component(detrended)
            
            # Residual
            residual = values - trend - seasonal
            
            # Predict trend continuation
            trend_slope = (trend[-1] - trend[0]) / len(trend) if len(trend) > 1 else 0
            
            # Generate predictions
            last_timestamp = original_data[-1].timestamp
            predictions = []
            
            residual_std = np.std(residual)
            
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                
                # Trend component
                trend_value = trend[-1] + trend_slope * i
                
                # Seasonal component (repeat pattern)
                seasonal_index = (len(values) + i - 1) % len(seasonal) if len(seasonal) > 0 else 0
                seasonal_value = seasonal[seasonal_index] if len(seasonal) > 0 else 0
                
                predicted_value = trend_value + seasonal_value
                
                # Confidence intervals based on residual variance
                confidence_95 = 1.96 * residual_std * np.sqrt(i)  # Increasing uncertainty
                
                predictions.append(TrendPoint(
                    timestamp=future_timestamp,
                    value=max(0, predicted_value),
                    metadata={
                        'type': 'prediction',
                        'method': 'trend_decomposition',
                        'trend_component': trend_value,
                        'seasonal_component': seasonal_value,
                        'confidence_lower': max(0, predicted_value - confidence_95),
                        'confidence_upper': predicted_value + confidence_95
                    }
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in trend decomposition prediction: {e}")
            return []
    
    def _moving_average_prediction(
        self,
        values: np.ndarray,
        original_data: List[TrendPoint],
        horizon: int,
        window_size: Optional[int] = None
    ) -> List[TrendPoint]:
        """Generate predictions using moving average."""
        
        try:
            if window_size is None:
                window_size = min(max(3, len(values) // 4), 10)
            
            if len(values) < window_size:
                window_size = len(values)
            
            # Calculate moving average for recent values
            recent_values = values[-window_size:]
            predicted_value = np.mean(recent_values)
            
            # Calculate volatility
            volatility = np.std(recent_values) if len(recent_values) > 1 else 0
            
            # Generate predictions (constant value with increasing uncertainty)
            last_timestamp = original_data[-1].timestamp
            predictions = []
            
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                
                # Increasing uncertainty over time
                uncertainty = volatility * np.sqrt(i)
                confidence_95 = 1.96 * uncertainty
                
                predictions.append(TrendPoint(
                    timestamp=future_timestamp,
                    value=max(0, predicted_value),
                    metadata={
                        'type': 'prediction',
                        'method': 'moving_average',
                        'window_size': window_size,
                        'base_volatility': volatility,
                        'confidence_lower': max(0, predicted_value - confidence_95),
                        'confidence_upper': predicted_value + confidence_95
                    }
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in moving average prediction: {e}")
            return []
    
    def _extract_trend(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Extract trend component using moving average."""
        
        if window_size >= len(values):
            return np.full_like(values, np.mean(values))
        
        # Calculate centered moving average
        trend = np.zeros_like(values)
        
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            trend[i] = np.mean(values[start_idx:end_idx])
        
        return trend
    
    def _extract_seasonal_component(self, detrended: np.ndarray) -> np.ndarray:
        """Extract seasonal component from detrended data."""
        
        # Simple approach: look for repeating patterns
        # For more sophisticated seasonality, consider using scipy.signal or statsmodels
        
        if len(detrended) < 7:  # Not enough data for seasonal analysis
            return np.zeros_like(detrended)
        
        # Try common seasonal periods (7 days, 30 days, etc.)
        possible_periods = [7, 14, 30, 90]
        best_period = None
        best_correlation = 0
        
        for period in possible_periods:
            if period >= len(detrended):
                continue
            
            # Calculate autocorrelation at this lag
            correlation = np.corrcoef(detrended[:-period], detrended[period:])[0, 1]
            
            if not np.isnan(correlation) and abs(correlation) > best_correlation:
                best_correlation = abs(correlation)
                best_period = period
        
        if best_period and best_correlation > 0.3:
            # Extract seasonal pattern
            seasonal = np.zeros_like(detrended)
            for i in range(len(detrended)):
                seasonal[i] = detrended[i % best_period]
            return seasonal
        else:
            return np.zeros_like(detrended)
    
    def _create_ensemble_predictions(
        self,
        predictions_by_method: Dict[str, List[TrendPoint]],
        last_timestamp: datetime,
        horizon: int
    ) -> List[TrendPoint]:
        """Create ensemble predictions by averaging multiple methods."""
        
        if not predictions_by_method:
            return []
        
        # Determine method weights (can be made configurable)
        method_weights = {
            'linear': 0.3,
            'polynomial': 0.25,
            'exponential_smoothing': 0.25,
            'trend_decomposition': 0.15,
            'moving_average': 0.05
        }
        
        ensemble_predictions = []
        
        for i in range(horizon):
            future_timestamp = last_timestamp + timedelta(days=i + 1)
            
            # Collect predictions for this time point
            values = []
            weights = []
            confidence_lowers = []
            confidence_uppers = []
            
            for method, method_predictions in predictions_by_method.items():
                if i < len(method_predictions):
                    pred = method_predictions[i]
                    values.append(pred.value)
                    weights.append(method_weights.get(method, 0.1))
                    
                    # Collect confidence intervals if available
                    if 'confidence_lower' in pred.metadata:
                        confidence_lowers.append(pred.metadata['confidence_lower'])
                        confidence_uppers.append(pred.metadata['confidence_upper'])
            
            if not values:
                continue
            
            # Calculate weighted average
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            ensemble_value = np.average(values, weights=weights)
            
            # Calculate ensemble confidence intervals
            if confidence_lowers and confidence_uppers:
                ensemble_lower = np.average(confidence_lowers, weights=weights)
                ensemble_upper = np.average(confidence_uppers, weights=weights)
            else:
                # Fallback: use standard deviation of predictions
                pred_std = np.std(values)
                ensemble_lower = max(0, ensemble_value - 1.96 * pred_std)
                ensemble_upper = ensemble_value + 1.96 * pred_std
            
            ensemble_predictions.append(TrendPoint(
                timestamp=future_timestamp,
                value=max(0, ensemble_value),
                metadata={
                    'type': 'prediction',
                    'method': 'ensemble',
                    'methods_used': list(predictions_by_method.keys()),
                    'method_weights': dict(zip(predictions_by_method.keys(), weights)),
                    'confidence_lower': ensemble_lower,
                    'confidence_upper': ensemble_upper,
                    'prediction_variance': np.var(values)
                }
            ))
        
        return ensemble_predictions


# Factory function for easy integration
def create_trend_predictor(config=None) -> TrendPredictor:
    """Create and return a TrendPredictor instance."""
    return TrendPredictor(config)


# Async wrapper for compatibility
async def generate_predictions(
    data: List[TrendPoint],
    config=None,
    **kwargs
) -> List[TrendPoint]:
    """Generate predictions using the trend predictor."""
    predictor = create_trend_predictor(config)
    return await predictor.generate_predictions(data, **kwargs)