#!/usr/bin/env python3
"""
Metrics Middleware for FastAPI
Integrates Prometheus metrics collection with request processing
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from monitoring.metrics import metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    metrics = None

logger = logging.getLogger(__name__)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for all API requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics_enabled = METRICS_AVAILABLE
        
        if self.metrics_enabled:
            logger.info("✅ Metrics middleware initialized with Prometheus support")
        else:
            logger.warning("⚠️ Metrics middleware initialized without Prometheus support")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        
        # Skip metrics collection for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Record request start time
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            status = str(response.status_code)
            
        except Exception as e:
            # Handle exceptions and record error metrics
            if self.metrics_enabled and metrics:
                metrics.record_error("api_exception", "fastapi")
            
            logger.error(f"Request failed: {e}")
            status = "500"
            
            # Re-raise the exception
            raise
        
        finally:
            # Calculate request duration
            duration = time.time() - start_time
            
            # Record metrics if available
            if self.metrics_enabled and metrics:
                method = request.method
                endpoint = self._get_endpoint_pattern(request)
                
                # Record API request metrics
                metrics.record_api_request(method, endpoint, duration, status)
                
                # Update system health based on response time
                health_score = self._calculate_health_score(duration, status)
                metrics.update_system_health("api", health_score)
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Metrics-Enabled"] = str(self.metrics_enabled)
        
        return response
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Extract endpoint pattern from request for consistent labeling"""
        path = request.url.path
        
        # Common endpoint patterns for better grouping
        endpoint_patterns = {
            '/api/health': '/api/health',
            '/api/status': '/api/status',
            '/api/metrics': '/api/metrics',
            '/api/performance': '/api/performance/*',
            '/api/content': '/api/content/*',
            '/api/analysis': '/api/analysis/*',
            '/api/scrape': '/api/scrape/*',
            '/api/database': '/api/database/*',
            '/api/search': '/api/search/*',
        }
        
        # Check for pattern matches
        for pattern, label in endpoint_patterns.items():
            if path.startswith(pattern):
                return label
        
        # For paths with IDs, replace with generic pattern
        if '/api/' in path:
            parts = path.split('/')
            if len(parts) >= 3:
                # Replace potential IDs with {id}
                for i, part in enumerate(parts[3:], 3):
                    if part.isdigit() or len(part) > 20:  # Likely an ID
                        parts[i] = '{id}'
                return '/'.join(parts)
        
        return path
    
    def _calculate_health_score(self, duration: float, status: str) -> float:
        """Calculate health score based on response time and status"""
        # Base score on response time
        if duration < 0.1:  # < 100ms
            time_score = 1.0
        elif duration < 0.5:  # < 500ms
            time_score = 0.9
        elif duration < 1.0:  # < 1s
            time_score = 0.7
        elif duration < 2.0:  # < 2s
            time_score = 0.5
        else:  # >= 2s
            time_score = 0.3
        
        # Adjust based on status code
        if status.startswith('2'):  # 2xx success
            status_score = 1.0
        elif status.startswith('3'):  # 3xx redirect
            status_score = 0.9
        elif status.startswith('4'):  # 4xx client error
            status_score = 0.6
        elif status.startswith('5'):  # 5xx server error
            status_score = 0.2
        else:
            status_score = 0.5
        
        # Combined score (weighted: 70% time, 30% status)
        return (time_score * 0.7) + (status_score * 0.3)

def create_metrics_middleware(app):
    """Factory function to create metrics middleware"""
    return MetricsMiddleware(app)