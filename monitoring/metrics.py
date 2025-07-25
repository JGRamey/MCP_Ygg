#!/usr/bin/env python3
"""
Prometheus Metrics Integration for MCP Yggdrasil
Complete metrics collection and export system
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
from fastapi import Response

# Graceful Prometheus integration
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for graceful degradation
    class MockMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    Counter = Histogram = Gauge = MockMetric
    generate_latest = lambda: b"# Prometheus client not available\n"
    CONTENT_TYPE_LATEST = "text/plain"

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Centralized Prometheus metrics for MCP Yggdrasil"""

    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE

        if self.enabled:
            # API Metrics
            self.api_requests_total = Counter(
                "mcp_api_requests_total",
                "Total API requests",
                ["method", "endpoint", "status"],
            )

            self.api_request_duration = Histogram(
                "mcp_api_request_duration_seconds",
                "API request duration",
                ["method", "endpoint"],
            )

            # Database Metrics
            self.neo4j_queries_total = Counter(
                "mcp_neo4j_queries_total",
                "Total Neo4j queries",
                ["query_type", "status"],
            )

            self.neo4j_query_duration = Histogram(
                "mcp_neo4j_query_duration_seconds",
                "Neo4j query duration",
                ["query_type"],
            )

            self.qdrant_operations_total = Counter(
                "mcp_qdrant_operations_total",
                "Total Qdrant operations",
                ["operation", "collection", "status"],
            )

            self.qdrant_operation_duration = Histogram(
                "mcp_qdrant_operation_duration_seconds",
                "Qdrant operation duration",
                ["operation", "collection"],
            )

            # Cache Metrics
            self.cache_operations_total = Counter(
                "mcp_cache_operations_total",
                "Cache operations",
                ["operation", "result"],
            )

            self.cache_hit_rate = Gauge(
                "mcp_cache_hit_rate", "Cache hit rate percentage"
            )

            # System Metrics
            self.system_cpu_usage = Gauge(
                "mcp_system_cpu_usage_percent", "System CPU usage"
            )

            self.system_memory_usage = Gauge(
                "mcp_system_memory_usage_percent", "System memory usage"
            )

            self.system_disk_usage = Gauge(
                "mcp_system_disk_usage_percent", "System disk usage"
            )

            # Connection Metrics
            self.active_connections = Gauge(
                "mcp_active_connections", "Active connections", ["type"]
            )

            # Content Processing Metrics
            self.documents_processed_total = Counter(
                "mcp_documents_processed_total",
                "Total documents processed",
                ["source_type", "status"],
            )

            self.processing_queue_size = Gauge(
                "mcp_processing_queue_size", "Current processing queue size"
            )

            # Enhanced AI Agent Metrics
            self.ai_agent_operations_total = Counter(
                "mcp_ai_agent_operations_total",
                "Total AI agent operations",
                ["agent_type", "operation", "status"],
            )

            self.ai_agent_processing_duration = Histogram(
                "mcp_ai_agent_processing_duration_seconds",
                "AI agent processing duration",
                ["agent_type", "operation"],
            )

            # Error Metrics
            self.errors_total = Counter(
                "mcp_errors_total", "Total errors", ["error_type", "component"]
            )

            # System Health Score
            self.system_health_score = Gauge(
                "mcp_system_health_score", "System health score (0-1)", ["component"]
            )

            logger.info("✅ Prometheus metrics initialized successfully")
        else:
            logger.warning("⚠️ Prometheus metrics disabled - client not available")
            # Set all metrics to mock objects
            for attr_name in dir(self):
                if not attr_name.startswith("_") and attr_name not in [
                    "enabled",
                    "update_system_metrics",
                    "get_metrics_response",
                ]:
                    setattr(self, attr_name, MockMetric())

    def update_system_metrics(self):
        """Update system resource metrics"""
        if not self.enabled:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.system_disk_usage.set(disk.percent)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def get_metrics_response(self) -> Response:
        """Generate Prometheus metrics HTTP response"""
        if not self.enabled:
            content = "# Prometheus client not available\n# Install with: pip install prometheus-client\n"
            return Response(content=content, media_type="text/plain")

        try:
            # Update system metrics before exporting
            self.update_system_metrics()

            # Generate metrics
            metrics_data = generate_latest()

            return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            error_content = f"# Error generating metrics: {e}\n"
            return Response(content=error_content, media_type="text/plain")

    def record_api_request(
        self, method: str, endpoint: str, duration: float, status: str
    ):
        """Record API request metrics"""
        if self.enabled:
            self.api_requests_total.labels(
                method=method, endpoint=endpoint, status=status
            ).inc()
            self.api_request_duration.labels(method=method, endpoint=endpoint).observe(
                duration
            )

    def record_database_query(
        self, db_type: str, query_type: str, duration: float, status: str
    ):
        """Record database query metrics"""
        if self.enabled:
            if db_type == "neo4j":
                self.neo4j_queries_total.labels(
                    query_type=query_type, status=status
                ).inc()
                self.neo4j_query_duration.labels(query_type=query_type).observe(
                    duration
                )
            elif db_type == "qdrant":
                collection = (
                    query_type.split("_")[0] if "_" in query_type else "unknown"
                )
                operation = (
                    query_type.split("_", 1)[1] if "_" in query_type else query_type
                )
                self.qdrant_operations_total.labels(
                    operation=operation, collection=collection, status=status
                ).inc()
                self.qdrant_operation_duration.labels(
                    operation=operation, collection=collection
                ).observe(duration)

    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics"""
        if self.enabled:
            self.cache_operations_total.labels(operation=operation, result=result).inc()

    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate"""
        if self.enabled:
            self.cache_hit_rate.set(hit_rate)

    def record_document_processing(self, source_type: str, status: str):
        """Record document processing metrics"""
        if self.enabled:
            self.documents_processed_total.labels(
                source_type=source_type, status=status
            ).inc()

    def update_processing_queue_size(self, size: int):
        """Update processing queue size"""
        if self.enabled:
            self.processing_queue_size.set(size)

    def record_ai_agent_operation(
        self, agent_type: str, operation: str, duration: float, status: str
    ):
        """Record AI agent operation metrics"""
        if self.enabled:
            self.ai_agent_operations_total.labels(
                agent_type=agent_type, operation=operation, status=status
            ).inc()
            self.ai_agent_processing_duration.labels(
                agent_type=agent_type, operation=operation
            ).observe(duration)

    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        if self.enabled:
            self.errors_total.labels(error_type=error_type, component=component).inc()

    def update_system_health(self, component: str, score: float):
        """Update system health score (0.0 = unhealthy, 1.0 = perfect)"""
        if self.enabled:
            self.system_health_score.labels(component=component).set(score)

    def update_active_connections(self, connection_type: str, count: int):
        """Update active connections"""
        if self.enabled:
            self.active_connections.labels(type=connection_type).set(count)


# Global metrics instance
metrics = PrometheusMetrics()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics system summary"""
    return {
        "prometheus_enabled": metrics.enabled,
        "metrics_available": metrics.enabled,
        "total_metrics": 17 if metrics.enabled else 0,
        "last_updated": datetime.now().isoformat(),
        "system_status": "operational" if metrics.enabled else "degraded",
    }
