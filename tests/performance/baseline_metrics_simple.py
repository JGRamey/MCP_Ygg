#!/usr/bin/env python3
"""
Simple Performance Baseline Metrics Script for MCP Yggdrasil
Establishes baseline metrics without requiring all components to be running
"""

import asyncio
import time
import json
import psutil
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SimplePerformanceBaseline:
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
            "target_comparison": {}
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": os.name,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "python_version": sys.version.split()[0]
        }
    
    async def measure_operation(self, name: str, operation_func, iterations: int = 10) -> Dict[str, float]:
        """Measure performance of an operation."""
        times = []
        memory_usage = []
        
        print(f"  Measuring {name}... ", end='', flush=True)
        
        for i in range(iterations):
            # Memory before
            mem_before = psutil.Process().memory_info().rss
            
            # Time execution
            start_time = time.perf_counter()
            try:
                await operation_func()
                execution_time = time.perf_counter() - start_time
                times.append(execution_time)
            except Exception as e:
                print(f"\n  Error in {name}: {e}")
                return {"error": str(e)}
            
            # Memory after
            mem_after = psutil.Process().memory_info().rss
            memory_usage.append((mem_after - mem_before) / 1024 / 1024)  # MB
            
            # Small delay between iterations
            await asyncio.sleep(0.01)
        
        metrics = {
            "mean_time_ms": round(np.mean(times) * 1000, 2),
            "median_time_ms": round(np.median(times) * 1000, 2),
            "std_time_ms": round(np.std(times) * 1000, 2),
            "min_time_ms": round(np.min(times) * 1000, 2),
            "max_time_ms": round(np.max(times) * 1000, 2),
            "p95_time_ms": round(np.percentile(times, 95) * 1000, 2),
            "p99_time_ms": round(np.percentile(times, 99) * 1000, 2),
            "mean_memory_mb": round(np.mean(memory_usage), 2),
            "iterations": iterations
        }
        
        print(f"✓ (avg: {metrics['mean_time_ms']}ms)")
        return metrics
    
    async def test_file_operations(self) -> Dict[str, Any]:
        """Test file system operations performance."""
        print("\n📁 Testing file operations...")
        
        test_file = self.output_dir / "test_file.tmp"
        test_data = "x" * 1024 * 100  # 100KB of data
        
        # Write test
        async def file_write():
            with open(test_file, 'w') as f:
                f.write(test_data)
        
        write_metrics = await self.measure_operation("file_write_100kb", file_write, 20)
        
        # Read test
        async def file_read():
            with open(test_file, 'r') as f:
                _ = f.read()
        
        read_metrics = await self.measure_operation("file_read_100kb", file_read, 50)
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        return {
            "write_performance": write_metrics,
            "read_performance": read_metrics
        }
    
    async def test_computation_performance(self) -> Dict[str, Any]:
        """Test computational operations performance."""
        print("\n🔢 Testing computational operations...")
        
        # Matrix operations
        async def matrix_ops():
            a = np.random.rand(100, 100)
            b = np.random.rand(100, 100)
            _ = np.dot(a, b)
        
        matrix_metrics = await self.measure_operation("matrix_multiply_100x100", matrix_ops, 50)
        
        # Vector operations
        async def vector_ops():
            vectors = [np.random.rand(384) for _ in range(100)]
            query_vector = np.random.rand(384)
            _ = [np.dot(query_vector, v) for v in vectors]
        
        vector_metrics = await self.measure_operation("vector_similarity_100x384", vector_ops, 20)
        
        # JSON parsing
        test_json = json.dumps({"data": [{"id": i, "values": list(range(100))} for i in range(100)]})
        
        async def json_ops():
            _ = json.loads(test_json)
        
        json_metrics = await self.measure_operation("json_parse_100_objects", json_ops, 50)
        
        return {
            "matrix_operations": matrix_metrics,
            "vector_operations": vector_metrics,
            "json_operations": json_metrics
        }
    
    async def test_async_operations(self) -> Dict[str, Any]:
        """Test async operation performance."""
        print("\n⚡ Testing async operations...")
        
        # Concurrent tasks
        async def concurrent_tasks():
            tasks = [asyncio.sleep(0.001) for _ in range(10)]
            await asyncio.gather(*tasks)
        
        concurrent_metrics = await self.measure_operation("concurrent_10_tasks", concurrent_tasks, 20)
        
        # Sequential tasks
        async def sequential_tasks():
            for _ in range(10):
                await asyncio.sleep(0.001)
        
        sequential_metrics = await self.measure_operation("sequential_10_tasks", sequential_tasks, 20)
        
        return {
            "concurrent_execution": concurrent_metrics,
            "sequential_execution": sequential_metrics,
            "concurrency_speedup": round(
                sequential_metrics.get("mean_time_ms", 0) / concurrent_metrics.get("mean_time_ms", 1), 2
            ) if concurrent_metrics.get("mean_time_ms", 0) > 0 else 0
        }
    
    async def test_memory_operations(self) -> Dict[str, Any]:
        """Test memory-intensive operations."""
        print("\n💾 Testing memory operations...")
        
        # Large data structure creation
        async def create_large_dict():
            data = {f"key_{i}": {"id": i, "data": list(range(100))} for i in range(1000)}
            _ = len(data)
        
        dict_metrics = await self.measure_operation("create_dict_1000_entries", create_large_dict, 10)
        
        # Memory allocation pattern
        async def memory_pattern():
            chunks = []
            for _ in range(100):
                chunks.append(bytearray(1024 * 10))  # 10KB chunks
            _ = len(chunks)
        
        memory_metrics = await self.measure_operation("allocate_100x10kb", memory_pattern, 10)
        
        return {
            "dict_creation": dict_metrics,
            "memory_allocation": memory_metrics
        }
    
    async def test_simulated_components(self) -> Dict[str, Any]:
        """Test simulated component operations."""
        print("\n🔧 Testing simulated component operations...")
        
        # Simulate cache operations
        cache_data = {}
        
        async def cache_write():
            key = f"key_{time.time()}"
            cache_data[key] = {"data": list(range(100))}
        
        async def cache_read():
            if cache_data:
                key = list(cache_data.keys())[0]
                _ = cache_data.get(key)
        
        cache_write_metrics = await self.measure_operation("simulated_cache_write", cache_write, 100)
        
        # Ensure cache has data for read test
        for _ in range(10):
            await cache_write()
        
        cache_read_metrics = await self.measure_operation("simulated_cache_read", cache_read, 100)
        
        # Simulate API response
        async def api_response():
            # Simulate JSON response creation
            response = {
                "status": "success",
                "data": [{"id": i, "value": i*2} for i in range(50)],
                "timestamp": time.time()
            }
            _ = json.dumps(response)
        
        api_metrics = await self.measure_operation("simulated_api_response", api_response, 50)
        
        # Simulate database query
        async def db_query():
            # Simulate query processing
            await asyncio.sleep(0.01)  # Simulate network/IO delay
            results = [{"id": i, "name": f"Entity_{i}"} for i in range(100)]
            _ = len(results)
        
        db_metrics = await self.measure_operation("simulated_db_query", db_query, 20)
        
        return {
            "cache_write": cache_write_metrics,
            "cache_read": cache_read_metrics,
            "api_response": api_metrics,
            "database_query": db_metrics
        }
    
    def _calculate_target_comparison(self) -> None:
        """Compare results against target metrics."""
        targets = {
            "api_response_ms": 500,
            "cache_read_ms": 10,
            "cache_write_ms": 50,
            "db_query_ms": 200,
            "vector_search_ms": 100,
            "file_read_ms": 50,
            "memory_usage_mb": 1000
        }
        
        # Check simulated API response
        api_time = self.results["component_metrics"].get("simulated_components", {}).get("api_response", {}).get("p95_time_ms", 0)
        self.results["target_comparison"]["api_response"] = {
            "current": api_time,
            "target": targets["api_response_ms"],
            "status": "✅ PASS" if api_time < targets["api_response_ms"] else "❌ FAIL",
            "margin": f"{round((targets['api_response_ms'] - api_time) / targets['api_response_ms'] * 100, 1)}%"
        }
        
        # Check cache performance
        cache_read = self.results["component_metrics"].get("simulated_components", {}).get("cache_read", {}).get("mean_time_ms", 0)
        self.results["target_comparison"]["cache_read"] = {
            "current": cache_read,
            "target": targets["cache_read_ms"],
            "status": "✅ PASS" if cache_read < targets["cache_read_ms"] else "❌ FAIL",
            "margin": f"{round((targets['cache_read_ms'] - cache_read) / targets['cache_read_ms'] * 100, 1)}%"
        }
        
        # Check vector operations (simulated)
        vector_time = self.results["component_metrics"].get("computation", {}).get("vector_operations", {}).get("mean_time_ms", 0)
        self.results["target_comparison"]["vector_operations"] = {
            "current": vector_time,
            "target": targets["vector_search_ms"],
            "status": "✅ PASS" if vector_time < targets["vector_search_ms"] else "❌ FAIL",
            "margin": f"{round((targets['vector_search_ms'] - vector_time) / targets['vector_search_ms'] * 100, 1)}%"
        }
        
        # Memory usage (current process)
        current_memory_mb = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        self.results["target_comparison"]["memory_usage"] = {
            "current": current_memory_mb,
            "target": targets["memory_usage_mb"],
            "status": "✅ PASS" if current_memory_mb < targets["memory_usage_mb"] else "❌ FAIL",
            "margin": f"{round((targets['memory_usage_mb'] - current_memory_mb) / targets['memory_usage_mb'] * 100, 1)}%"
        }
    
    async def run_all_tests(self) -> None:
        """Run all performance tests."""
        print("🚀 Starting Performance Baseline Tests")
        print(f"System: {self.results['system_info']['platform']} | "
              f"CPUs: {self.results['system_info']['cpu_count']} | "
              f"RAM: {self.results['system_info']['memory_total_gb']}GB")
        print("=" * 60)
        
        # Run tests
        self.results["component_metrics"]["file_operations"] = await self.test_file_operations()
        self.results["component_metrics"]["computation"] = await self.test_computation_performance()
        self.results["component_metrics"]["async_operations"] = await self.test_async_operations()
        self.results["component_metrics"]["memory_operations"] = await self.test_memory_operations()
        self.results["component_metrics"]["simulated_components"] = await self.test_simulated_components()
        
        # Calculate overall metrics
        self._calculate_overall_metrics()
        
        # Compare against targets
        self._calculate_target_comparison()
        
        # Save results
        self._save_results()
        
        # Generate report
        self._generate_report()
    
    def _calculate_overall_metrics(self) -> None:
        """Calculate overall system metrics."""
        all_times = []
        all_memory = []
        
        for component, metrics in self.results["component_metrics"].items():
            if isinstance(metrics, dict):
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean_time_ms" in metric_data:
                        all_times.append(metric_data["mean_time_ms"])
                        if "mean_memory_mb" in metric_data:
                            all_memory.append(metric_data["mean_memory_mb"])
        
        if all_times:
            self.results["overall_metrics"] = {
                "total_operations_tested": len(all_times),
                "average_operation_time_ms": round(np.mean(all_times), 2),
                "fastest_operation_ms": round(np.min(all_times), 2),
                "slowest_operation_ms": round(np.max(all_times), 2),
                "operations_under_10ms": sum(1 for t in all_times if t < 10),
                "operations_under_50ms": sum(1 for t in all_times if t < 50),
                "operations_under_100ms": sum(1 for t in all_times if t < 100),
                "operations_under_500ms": sum(1 for t in all_times if t < 500),
                "average_memory_impact_mb": round(np.mean(all_memory), 2) if all_memory else 0
            }
    
    def _save_results(self) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"baseline_metrics_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def _generate_report(self) -> None:
        """Generate human-readable report."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BASELINE REPORT")
        print("=" * 60)
        
        # System info
        print(f"\n🖥️  System Information:")
        print(f"   Platform: {self.results['system_info']['platform']}")
        print(f"   CPUs: {self.results['system_info']['cpu_count']}")
        print(f"   Memory: {self.results['system_info']['memory_total_gb']}GB total, "
              f"{self.results['system_info']['memory_available_gb']}GB available")
        
        # Overall metrics
        if self.results["overall_metrics"]:
            print(f"\n📈 Overall Performance Metrics:")
            om = self.results["overall_metrics"]
            print(f"   Operations tested: {om['total_operations_tested']}")
            print(f"   Average time: {om['average_operation_time_ms']}ms")
            print(f"   Fastest operation: {om['fastest_operation_ms']}ms")
            print(f"   Slowest operation: {om['slowest_operation_ms']}ms")
            print(f"   Operations <10ms: {om['operations_under_10ms']} "
                  f"({om['operations_under_10ms']/om['total_operations_tested']*100:.0f}%)")
            print(f"   Operations <100ms: {om['operations_under_100ms']} "
                  f"({om['operations_under_100ms']/om['total_operations_tested']*100:.0f}%)")
        
        # Target comparison
        print(f"\n🎯 Performance vs Targets:")
        print(f"   {'Metric':<20} {'Current':<10} {'Target':<10} {'Status':<15} {'Margin':<10}")
        print(f"   {'-'*65}")
        
        for metric, comparison in self.results["target_comparison"].items():
            metric_name = metric.replace('_', ' ').title()
            current = f"{comparison['current']}ms" if 'memory' not in metric else f"{comparison['current']}MB"
            target = f"{comparison['target']}ms" if 'memory' not in metric else f"{comparison['target']}MB"
            print(f"   {metric_name:<20} {current:<10} {target:<10} {comparison['status']:<15} {comparison['margin']:<10}")
        
        # Key findings
        print(f"\n🔍 Key Findings:")
        
        # Find fastest operations
        fast_ops = []
        for component, metrics in self.results["component_metrics"].items():
            if isinstance(metrics, dict):
                for op_name, op_data in metrics.items():
                    if isinstance(op_data, dict) and "mean_time_ms" in op_data:
                        if op_data["mean_time_ms"] < 1:
                            fast_ops.append((op_name, op_data["mean_time_ms"]))
        
        if fast_ops:
            fast_ops.sort(key=lambda x: x[1])
            print(f"   ⚡ Fastest operations (<1ms):")
            for op, time in fast_ops[:3]:
                print(f"      - {op}: {time}ms")
        
        # Find slow operations
        slow_ops = []
        for component, metrics in self.results["component_metrics"].items():
            if isinstance(metrics, dict):
                for op_name, op_data in metrics.items():
                    if isinstance(op_data, dict) and "mean_time_ms" in op_data:
                        if op_data["mean_time_ms"] > 50:
                            slow_ops.append((op_name, op_data["mean_time_ms"]))
        
        if slow_ops:
            slow_ops.sort(key=lambda x: x[1], reverse=True)
            print(f"   🐌 Slowest operations (>50ms):")
            for op, time in slow_ops[:3]:
                print(f"      - {op}: {time}ms")
        
        # Concurrency benefit
        async_metrics = self.results["component_metrics"].get("async_operations", {})
        if "concurrency_speedup" in async_metrics:
            print(f"   🔄 Concurrency speedup: {async_metrics['concurrency_speedup']}x")
        
        print("\n" + "=" * 60)
        
        # Save markdown report
        self._save_markdown_report()
    
    def _save_markdown_report(self) -> None:
        """Save detailed markdown report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"baseline_report_{timestamp}.md"
        
        lines = [
            "# MCP Yggdrasil Performance Baseline Report",
            f"\nGenerated: {self.results['timestamp']}",
            "\n## System Information",
            f"- **Platform**: {self.results['system_info']['platform']}",
            f"- **CPUs**: {self.results['system_info']['cpu_count']}",
            f"- **Memory**: {self.results['system_info']['memory_total_gb']}GB total",
            f"- **Python**: {self.results['system_info']['python_version']}",
            "\n## Performance Summary",
            "\n### Target Comparison",
            "| Metric | Current | Target | Status | Margin |",
            "|--------|---------|--------|--------|--------|"
        ]
        
        for metric, comp in self.results["target_comparison"].items():
            unit = "ms" if 'memory' not in metric else "MB"
            lines.append(f"| {metric.replace('_', ' ').title()} | "
                        f"{comp['current']}{unit} | "
                        f"{comp['target']}{unit} | "
                        f"{comp['status']} | "
                        f"{comp['margin']} |")
        
        lines.extend([
            "\n### Detailed Component Metrics",
            "\n#### File Operations",
            "| Operation | Mean | P95 | P99 | Min | Max |",
            "|-----------|------|-----|-----|-----|-----|"
        ])
        
        # Add detailed metrics for each component
        for component, metrics in self.results["component_metrics"].items():
            if isinstance(metrics, dict):
                for op_name, op_data in metrics.items():
                    if isinstance(op_data, dict) and "mean_time_ms" in op_data:
                        lines.append(f"| {op_name} | "
                                   f"{op_data['mean_time_ms']}ms | "
                                   f"{op_data['p95_time_ms']}ms | "
                                   f"{op_data['p99_time_ms']}ms | "
                                   f"{op_data['min_time_ms']}ms | "
                                   f"{op_data['max_time_ms']}ms |")
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))


async def main():
    """Run performance baseline tests."""
    baseline = SimplePerformanceBaseline()
    await baseline.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())