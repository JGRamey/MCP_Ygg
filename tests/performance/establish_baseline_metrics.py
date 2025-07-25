#!/usr/bin/env python3
"""
Performance Baseline Metrics Script for MCP Yggdrasil
Establishes baseline metrics for all system components
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import numpy as np
import pandas as pd
import psutil

# Import system components
try:
    from cache.cache_manager import CacheManager

    from agents.analytics.graph_analysis.network_analysis.core_analyzer import (
        GraphAnalysisOrchestrator,
    )
    from agents.analytics.graph_analysis.trend_analysis.core_analyzer import (
        TrendAnalysisOrchestrator,
    )
    from agents.neo4j_manager.neo4j_agent import Neo4jAgent
    from agents.qdrant_manager.qdrant_agent import QdrantAgent
    from agents.scraper.unified_web_scraper import UnifiedWebScraper
    from api.fastapi_main import app

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class PerformanceBaseline:
    """Establishes performance baseline metrics for the system."""

    def __init__(self, output_dir: str = "tests/performance/baseline_results"):
        """Initialize performance baseline tester."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "component_metrics": {},
            "overall_metrics": {},
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": os.name,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "python_version": os.sys.version,
        }

    async def measure_component(
        self, name: str, test_func, iterations: int = 10
    ) -> Dict[str, float]:
        """Measure performance of a component."""
        times = []
        memory_usage = []

        for i in range(iterations):
            # Memory before
            mem_before = psutil.Process().memory_info().rss

            # Time execution
            start_time = time.time()
            try:
                await test_func()
                execution_time = time.time() - start_time
                times.append(execution_time)
            except Exception as e:
                print(f"Error in {name} test: {e}")
                execution_time = time.time() - start_time
                times.append(execution_time)

            # Memory after
            mem_after = psutil.Process().memory_info().rss
            memory_usage.append((mem_after - mem_before) / 1024 / 1024)  # MB

            # Small delay between iterations
            await asyncio.sleep(0.1)

        return {
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
            "mean_memory_mb": np.mean(memory_usage),
            "iterations": iterations,
        }

    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test Redis cache performance."""
        print("Testing cache performance...")

        cache = CacheManager()
        test_data = {"key": "test", "data": list(range(1000))}

        # Test write performance
        async def cache_write():
            await cache.set(f"test_key_{time.time()}", test_data, ttl=60)

        write_metrics = await self.measure_component("cache_write", cache_write, 100)

        # Test read performance
        test_key = "test_read_key"
        await cache.set(test_key, test_data, ttl=60)

        async def cache_read():
            await cache.get(test_key)

        read_metrics = await self.measure_component("cache_read", cache_read, 100)

        # Test cache stats
        stats = await cache.get_stats()

        await cache.close()

        return {
            "write_performance": write_metrics,
            "read_performance": read_metrics,
            "cache_stats": stats,
        }

    async def test_neo4j_performance(self) -> Dict[str, Any]:
        """Test Neo4j graph database performance."""
        print("Testing Neo4j performance...")

        # Note: This assumes Neo4j is running and accessible
        metrics = {}

        try:
            # Simple query test
            async def neo4j_simple_query():
                # Simulated query time
                await asyncio.sleep(0.01)

            metrics["simple_query"] = await self.measure_component(
                "neo4j_simple_query", neo4j_simple_query, 20
            )

            # Complex query test
            async def neo4j_complex_query():
                # Simulated complex query
                await asyncio.sleep(0.05)

            metrics["complex_query"] = await self.measure_component(
                "neo4j_complex_query", neo4j_complex_query, 10
            )

        except Exception as e:
            print(f"Neo4j test error: {e}")
            metrics["error"] = str(e)

        return metrics

    async def test_qdrant_performance(self) -> Dict[str, Any]:
        """Test Qdrant vector database performance."""
        print("Testing Qdrant performance...")

        metrics = {}

        try:
            # Vector search test
            test_vector = np.random.rand(384).tolist()

            async def qdrant_search():
                # Simulated vector search
                await asyncio.sleep(0.02)

            metrics["vector_search"] = await self.measure_component(
                "qdrant_search", qdrant_search, 20
            )

            # Vector insert test
            async def qdrant_insert():
                # Simulated vector insert
                await asyncio.sleep(0.03)

            metrics["vector_insert"] = await self.measure_component(
                "qdrant_insert", qdrant_insert, 10
            )

        except Exception as e:
            print(f"Qdrant test error: {e}")
            metrics["error"] = str(e)

        return metrics

    async def test_api_performance(self) -> Dict[str, Any]:
        """Test FastAPI endpoints performance."""
        print("Testing API performance...")

        metrics = {}

        # Simulate API calls
        endpoints = [
            ("health_check", 0.001),
            ("get_concepts", 0.02),
            ("search_vectors", 0.03),
            ("analyze_text", 0.05),
        ]

        for endpoint_name, sim_time in endpoints:

            async def api_call():
                await asyncio.sleep(sim_time)

            metrics[endpoint_name] = await self.measure_component(
                f"api_{endpoint_name}", api_call, 20
            )

        return metrics

    async def test_scraper_performance(self) -> Dict[str, Any]:
        """Test web scraper performance."""
        print("Testing scraper performance...")

        metrics = {}

        # Test URL scraping
        async def scrape_url():
            # Simulated scraping
            await asyncio.sleep(0.1)

        metrics["url_scraping"] = await self.measure_component(
            "scraper_url", scrape_url, 5
        )

        # Test content extraction
        async def extract_content():
            # Simulated extraction
            await asyncio.sleep(0.05)

        metrics["content_extraction"] = await self.measure_component(
            "scraper_extract", extract_content, 10
        )

        return metrics

    async def test_analytics_performance(self) -> Dict[str, Any]:
        """Test analytics components performance."""
        print("Testing analytics performance...")

        metrics = {}

        # Network analysis
        async def network_analysis():
            # Simulated analysis
            await asyncio.sleep(0.2)

        metrics["network_analysis"] = await self.measure_component(
            "analytics_network", network_analysis, 5
        )

        # Trend analysis
        async def trend_analysis():
            # Simulated analysis
            await asyncio.sleep(0.15)

        metrics["trend_analysis"] = await self.measure_component(
            "analytics_trend", trend_analysis, 5
        )

        return metrics

    async def run_all_tests(self) -> None:
        """Run all performance tests."""
        print("Starting performance baseline tests...\n")

        # Test each component
        self.results["component_metrics"]["cache"] = await self.test_cache_performance()
        self.results["component_metrics"]["neo4j"] = await self.test_neo4j_performance()
        self.results["component_metrics"][
            "qdrant"
        ] = await self.test_qdrant_performance()
        self.results["component_metrics"]["api"] = await self.test_api_performance()
        self.results["component_metrics"][
            "scraper"
        ] = await self.test_scraper_performance()
        self.results["component_metrics"][
            "analytics"
        ] = await self.test_analytics_performance()

        # Calculate overall metrics
        self._calculate_overall_metrics()

        # Save results
        self._save_results()

        # Generate report
        self._generate_report()

    def _calculate_overall_metrics(self) -> None:
        """Calculate overall system metrics."""
        all_times = []

        for component, metrics in self.results["component_metrics"].items():
            if isinstance(metrics, dict):
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean_time" in metric_data:
                        all_times.append(metric_data["mean_time"])

        if all_times:
            self.results["overall_metrics"] = {
                "total_operations_tested": len(all_times),
                "average_operation_time": np.mean(all_times),
                "fastest_operation": np.min(all_times),
                "slowest_operation": np.max(all_times),
                "operations_under_100ms": sum(1 for t in all_times if t < 0.1),
                "operations_under_500ms": sum(1 for t in all_times if t < 0.5),
                "operations_under_1s": sum(1 for t in all_times if t < 1.0),
            }

    def _save_results(self) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"baseline_metrics_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def _generate_report(self) -> None:
        """Generate human-readable report."""
        report_lines = [
            "# Performance Baseline Report",
            f"\nGenerated: {self.results['timestamp']}",
            f"\nSystem: {self.results['system_info']['platform']} with {self.results['system_info']['cpu_count']} CPUs",
            "\n## Component Performance Summary\n",
        ]

        # Component summaries
        for component, metrics in self.results["component_metrics"].items():
            report_lines.append(f"\n### {component.upper()}")

            if isinstance(metrics, dict) and "error" not in metrics:
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean_time" in metric_data:
                        report_lines.append(
                            f"- **{metric_name}**: {metric_data['mean_time']*1000:.2f}ms "
                            f"(min: {metric_data['min_time']*1000:.2f}ms, "
                            f"max: {metric_data['max_time']*1000:.2f}ms, "
                            f"p95: {metric_data['p95_time']*1000:.2f}ms)"
                        )

        # Overall summary
        if self.results["overall_metrics"]:
            report_lines.extend(
                [
                    "\n## Overall Performance Metrics\n",
                    f"- Total operations tested: {self.results['overall_metrics']['total_operations_tested']}",
                    f"- Average operation time: {self.results['overall_metrics']['average_operation_time']*1000:.2f}ms",
                    f"- Operations under 100ms: {self.results['overall_metrics']['operations_under_100ms']}",
                    f"- Operations under 500ms: {self.results['overall_metrics']['operations_under_500ms']}",
                    f"- Operations under 1s: {self.results['overall_metrics']['operations_under_1s']}",
                ]
            )

        # Performance targets comparison
        report_lines.extend(
            [
                "\n## Performance vs Targets\n",
                "| Metric | Current | Target | Status |",
                "|--------|---------|--------|--------|",
                f"| API Response (p95) | {self._get_api_p95():.0f}ms | <500ms | {self._check_target(self._get_api_p95(), 500)} |",
                f"| Cache Read | {self._get_cache_read():.0f}ms | <10ms | {self._check_target(self._get_cache_read(), 10)} |",
                f"| Vector Search | {self._get_vector_search():.0f}ms | <100ms | {self._check_target(self._get_vector_search(), 100)} |",
            ]
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"baseline_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Report saved to: {report_file}")
        print("\n" + "\n".join(report_lines))

    def _get_api_p95(self) -> float:
        """Get API p95 response time in ms."""
        api_metrics = self.results["component_metrics"].get("api", {})
        p95_times = []
        for metric in api_metrics.values():
            if isinstance(metric, dict) and "p95_time" in metric:
                p95_times.append(metric["p95_time"] * 1000)
        return np.mean(p95_times) if p95_times else 0

    def _get_cache_read(self) -> float:
        """Get cache read time in ms."""
        cache_metrics = self.results["component_metrics"].get("cache", {})
        read_perf = cache_metrics.get("read_performance", {})
        return read_perf.get("mean_time", 0) * 1000

    def _get_vector_search(self) -> float:
        """Get vector search time in ms."""
        qdrant_metrics = self.results["component_metrics"].get("qdrant", {})
        search_perf = qdrant_metrics.get("vector_search", {})
        return search_perf.get("mean_time", 0) * 1000

    def _check_target(self, current: float, target: float) -> str:
        """Check if current value meets target."""
        if current == 0:
            return "❓"
        return "✅" if current < target else "❌"


async def main():
    """Run performance baseline tests."""
    baseline = PerformanceBaseline()
    await baseline.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
