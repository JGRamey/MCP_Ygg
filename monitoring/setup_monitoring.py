#!/usr/bin/env python3
"""
Enhanced Monitoring Setup for MCP Yggdrasil
Phase 2 Completion - Prometheus + Grafana Integration
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️ prometheus_client not available. Install with: pip install prometheus-client")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMonitoring:
    """Enhanced monitoring system for MCP Yggdrasil with Prometheus metrics."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.metrics_enabled = PROMETHEUS_AVAILABLE
        
        if self.metrics_enabled:
            # Performance metrics
            self.api_request_duration = Histogram(
                'api_request_duration_seconds',
                'API request duration',
                ['method', 'endpoint', 'status']
            )
            
            self.cache_operations = Counter(
                'cache_operations_total',
                'Total cache operations',
                ['operation', 'result']
            )
            
            self.scraping_duration = Histogram(
                'scraping_duration_seconds',
                'Web scraping duration',
                ['url', 'status']
            )
            
            self.memory_usage = Gauge(
                'memory_usage_bytes',
                'Current memory usage in bytes'
            )
            
            self.active_connections = Gauge(
                'active_connections',
                'Number of active connections',
                ['type']
            )
            
            # System health metrics
            self.system_health = Gauge(
                'system_health_score',
                'Overall system health score (0-1)',
                ['component']
            )
            
            logger.info("✅ Prometheus metrics initialized")
        else:
            logger.warning("⚠️ Prometheus metrics disabled - client not available")
    
    async def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self.metrics_enabled:
            logger.warning("⚠️ Cannot start metrics server - Prometheus client not available")
            return False
        
        try:
            start_http_server(self.port)
            logger.info(f"✅ Prometheus metrics server started on port {self.port}")
            logger.info(f"📊 Metrics available at: http://localhost:{self.port}/metrics")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")
            return False
    
    def record_api_request(self, method: str, endpoint: str, duration: float, status: str):
        """Record API request metrics."""
        if self.metrics_enabled:
            self.api_request_duration.labels(method=method, endpoint=endpoint, status=status).observe(duration)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics."""
        if self.metrics_enabled:
            self.cache_operations.labels(operation=operation, result=result).inc()
    
    def record_scraping_operation(self, url: str, duration: float, status: str):
        """Record web scraping metrics."""
        if self.metrics_enabled:
            self.scraping_duration.labels(url=url, status=status).observe(duration)
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage metric."""
        if self.metrics_enabled:
            self.memory_usage.set(bytes_used)
    
    def update_active_connections(self, connection_type: str, count: int):
        """Update active connections metric."""
        if self.metrics_enabled:
            self.active_connections.labels(type=connection_type).set(count)
    
    def update_system_health(self, component: str, score: float):
        """Update system health score (0.0 = unhealthy, 1.0 = perfect)."""
        if self.metrics_enabled:
            self.system_health.labels(component=component).set(score)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.metrics_enabled:
            return {"error": "Metrics not available"}
        
        return {
            "metrics_enabled": True,
            "metrics_endpoint": f"http://localhost:{self.port}/metrics",
            "available_metrics": [
                "api_request_duration_seconds",
                "cache_operations_total", 
                "scraping_duration_seconds",
                "memory_usage_bytes",
                "active_connections",
                "system_health_score"
            ]
        }

def check_docker_available() -> bool:
    """Check if Docker is available for Grafana setup."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def setup_grafana_docker() -> bool:
    """Set up Grafana using Docker."""
    if not check_docker_available():
        logger.warning("⚠️ Docker not available - skipping Grafana setup")
        return False
    
    try:
        # Check if Grafana container already exists
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=mcp-grafana'], 
                              capture_output=True, text=True)
        
        if 'mcp-grafana' in result.stdout:
            logger.info("✅ Grafana container already exists")
            # Start if not running
            subprocess.run(['docker', 'start', 'mcp-grafana'], check=True)
        else:
            # Create new Grafana container
            cmd = [
                'docker', 'run', '-d',
                '--name', 'mcp-grafana',
                '-p', '3000:3000',
                '-e', 'GF_SECURITY_ADMIN_PASSWORD=admin',
                'grafana/grafana:latest'
            ]
            subprocess.run(cmd, check=True)
            logger.info("✅ Grafana container created and started")
        
        logger.info("📊 Grafana available at: http://localhost:3000 (admin/admin)")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to set up Grafana: {e}")
        return False

async def run_monitoring_demo():
    """Run a demonstration of the monitoring system."""
    logger.info("🚀 Starting Enhanced Monitoring Demo")
    
    # Initialize monitoring
    monitor = EnhancedMonitoring(port=8001)
    
    # Start metrics server
    server_started = await monitor.start_metrics_server()
    if not server_started:
        logger.error("❌ Could not start metrics server")
        return False
    
    # Simulate some metrics
    logger.info("📊 Generating sample metrics...")
    
    # Simulate API requests
    for i in range(10):
        duration = 0.001 + (i * 0.01)  # Increasing latency
        status = "200" if i < 8 else "500"  # Some errors
        monitor.record_api_request("GET", "/api/health", duration, status)
        await asyncio.sleep(0.1)
    
    # Simulate cache operations
    for operation, result in [("get", "hit"), ("get", "miss"), ("set", "success")] * 5:
        monitor.record_cache_operation(operation, result)
        await asyncio.sleep(0.05)
    
    # Simulate scraping operations
    for i, url in enumerate(["example.com", "httpbin.org", "jsonplaceholder.typicode.com"]):
        duration = 0.2 + (i * 0.1)
        status = "success"
        monitor.record_scraping_operation(url, duration, status)
        await asyncio.sleep(0.1)
    
    # Update system metrics
    import psutil
    process = psutil.Process()
    monitor.update_memory_usage(process.memory_info().rss)
    monitor.update_active_connections("redis", 5)
    monitor.update_active_connections("neo4j", 10)
    monitor.update_system_health("api", 0.95)
    monitor.update_system_health("cache", 0.98)
    monitor.update_system_health("scraper", 0.92)
    
    # Display metrics summary
    summary = monitor.get_metrics_summary()
    logger.info(f"📈 Metrics Summary: {json.dumps(summary, indent=2)}")
    
    logger.info("✅ Monitoring demo completed successfully")
    logger.info("🔗 View metrics at: http://localhost:8001/metrics")
    
    return True

async def main():
    """Main monitoring setup function."""
    print("🚀 MCP Yggdrasil Enhanced Monitoring Setup")
    print("=" * 50)
    
    # Setup monitoring
    success = await run_monitoring_demo()
    
    if success:
        print("\n✅ Enhanced Monitoring Setup Complete!")
        print("\n📊 Available Endpoints:")
        print("   • Prometheus Metrics: http://localhost:8001/metrics")
        print("   • Grafana Dashboard: http://localhost:3000 (if Docker available)")
        
        print("\n🔧 Next Steps:")
        print("   1. Install Prometheus: Download from https://prometheus.io/download/")
        print("   2. Use provided prometheus_config.yml")
        print("   3. Import grafana_dashboard.json to Grafana")
        print("   4. Set up alerting rules from mcp_yggdrasil_rules.yml")
        
        # Optionally set up Grafana
        if check_docker_available():
            setup_grafana = input("\n🐳 Set up Grafana with Docker? (y/n): ").lower().strip() == 'y'
            if setup_grafana:
                setup_grafana_docker()
        
        # Keep server running for demo
        print("\n⏰ Keeping metrics server running for 60 seconds...")
        print("   Visit http://localhost:8001/metrics to see live metrics")
        await asyncio.sleep(60)
        
    else:
        print("\n❌ Monitoring setup failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))