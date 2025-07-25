#!/usr/bin/env python3
"""
Performance Monitoring API Routes
Monitor scraping performance and ensure <10 second targets are met
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# Import performance monitoring components
try:
    from agents.scraper.high_performance_scraper import HighPerformanceScraper
except ImportError as e:
    print(f"Performance scraper import warning: {e}")
    HighPerformanceScraper = None

router = APIRouter(prefix="/api/performance", tags=["performance"])
logger = logging.getLogger(__name__)

# Global performance tracker
performance_history = []
active_scraper = None


# Request models
class PerformanceTestRequest(BaseModel):
    urls: List[HttpUrl]
    max_concurrent: Optional[int] = 10
    timeout: Optional[int] = 8
    cache_enabled: bool = True


class BenchmarkRequest(BaseModel):
    target_urls: Optional[List[HttpUrl]] = None
    iterations: int = 1
    max_concurrent: int = 10


# Response models
class PerformanceResult(BaseModel):
    timestamp: datetime
    test_type: str
    total_urls: int
    successful_scrapes: int
    success_rate: float
    average_time: float
    max_time: float
    min_time: float
    target_met: bool
    metrics: Dict[str, Any]


@router.post("/test-scraping", response_model=PerformanceResult)
async def test_scraping_performance(request: PerformanceTestRequest):
    """
    Test scraping performance on provided URLs
    Target: <10 seconds for standard web pages
    """
    try:
        if not HighPerformanceScraper:
            raise HTTPException(
                status_code=500, detail="High-performance scraper not available"
            )

        test_urls = [str(url) for url in request.urls]

        # Initialize scraper with test parameters
        async with HighPerformanceScraper(
            max_concurrent=request.max_concurrent, timeout=request.timeout
        ) as scraper:

            logger.info(f"🧪 Testing performance on {len(test_urls)} URLs")
            start_time = time.time()

            # Run performance test
            results = await scraper.scrape_urls(test_urls, request.max_concurrent)

            total_time = time.time() - start_time

            # Analyze results
            successful = [r for r in results if r.status == "success"]
            times = [r.processing_time for r in successful]

            if not times:
                avg_time = max_time = min_time = 0
            else:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)

            # Check if target is met
            target_met = max_time < 10.0 if times else False

            # Get detailed metrics
            metrics = scraper.get_performance_metrics()

            result = PerformanceResult(
                timestamp=datetime.now(),
                test_type="custom_urls",
                total_urls=len(test_urls),
                successful_scrapes=len(successful),
                success_rate=len(successful) / len(test_urls) if test_urls else 0,
                average_time=avg_time,
                max_time=max_time,
                min_time=min_time,
                target_met=target_met,
                metrics=metrics,
            )

            # Store in history
            performance_history.append(result.dict())

            logger.info(
                f"✅ Performance test completed: {result.success_rate:.1%} success, {avg_time:.2f}s avg"
            )

            return result

    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Performance test failed: {str(e)}"
        )


@router.post("/benchmark", response_model=Dict[str, Any])
async def run_benchmark(request: BenchmarkRequest):
    """
    Run comprehensive benchmark tests
    """
    try:
        if not HighPerformanceScraper:
            raise HTTPException(
                status_code=500, detail="High-performance scraper not available"
            )

        # Default test URLs if not provided
        test_urls = request.target_urls or [
            "https://example.com",
            "https://httpbin.org/html",
            "https://httpbin.org/status/200",
            "https://jsonplaceholder.typicode.com/posts/1",
        ]

        test_urls = [str(url) for url in test_urls]

        logger.info(
            f"🏁 Running benchmark: {request.iterations} iterations on {len(test_urls)} URLs"
        )

        benchmark_results = []

        for iteration in range(request.iterations):
            async with HighPerformanceScraper(
                max_concurrent=request.max_concurrent, timeout=8
            ) as scraper:

                start_time = time.time()
                results = await scraper.scrape_urls(test_urls, request.max_concurrent)
                iteration_time = time.time() - start_time

                successful = [r for r in results if r.status == "success"]
                times = [r.processing_time for r in successful]

                iteration_result = {
                    "iteration": iteration + 1,
                    "total_time": iteration_time,
                    "successful_scrapes": len(successful),
                    "success_rate": (
                        len(successful) / len(test_urls) if test_urls else 0
                    ),
                    "average_time": sum(times) / len(times) if times else 0,
                    "max_time": max(times) if times else 0,
                    "min_time": min(times) if times else 0,
                    "target_met": max(times) < 10.0 if times else False,
                    "metrics": scraper.get_performance_metrics(),
                }

                benchmark_results.append(iteration_result)

                logger.info(
                    f"Iteration {iteration + 1}: {iteration_result['success_rate']:.1%} success, {iteration_result['average_time']:.2f}s avg"
                )

        # Calculate overall statistics
        all_success_rates = [r["success_rate"] for r in benchmark_results]
        all_avg_times = [
            r["average_time"] for r in benchmark_results if r["average_time"] > 0
        ]
        all_max_times = [r["max_time"] for r in benchmark_results if r["max_time"] > 0]

        overall_stats = {
            "iterations": request.iterations,
            "test_urls_count": len(test_urls),
            "overall_success_rate": (
                sum(all_success_rates) / len(all_success_rates)
                if all_success_rates
                else 0
            ),
            "overall_avg_time": (
                sum(all_avg_times) / len(all_avg_times) if all_avg_times else 0
            ),
            "overall_max_time": max(all_max_times) if all_max_times else 0,
            "target_met_consistently": all(r["target_met"] for r in benchmark_results),
            "iterations_meeting_target": sum(
                1 for r in benchmark_results if r["target_met"]
            ),
        }

        benchmark_report = {
            "benchmark_id": f"benchmark_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "overall_stats": overall_stats,
            "iteration_results": benchmark_results,
            "test_urls": test_urls,
        }

        # Store in history
        performance_history.append(
            {
                "timestamp": datetime.now(),
                "test_type": "benchmark",
                "benchmark_report": benchmark_report,
            }
        )

        logger.info(
            f"🏆 Benchmark completed: {overall_stats['overall_success_rate']:.1%} success rate, {overall_stats['overall_avg_time']:.2f}s avg time"
        )

        return benchmark_report

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/history")
async def get_performance_history(limit: int = 50):
    """
    Get recent performance test history
    """
    try:
        # Return most recent results
        recent_history = performance_history[-limit:] if performance_history else []

        return {
            "total_tests": len(performance_history),
            "recent_tests": len(recent_history),
            "history": recent_history,
        }

    except Exception as e:
        logger.error(f"Error retrieving performance history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/metrics")
async def get_current_metrics():
    """
    Get current performance metrics
    """
    try:
        global active_scraper

        if not active_scraper:
            # Create a temporary scraper to get baseline metrics
            active_scraper = HighPerformanceScraper()
            await active_scraper.start()

        current_metrics = active_scraper.get_performance_metrics()

        # Add system information
        system_info = {
            "high_performance_scraper_available": HighPerformanceScraper is not None,
            "performance_history_count": len(performance_history),
            "last_test_timestamp": (
                performance_history[-1]["timestamp"] if performance_history else None
            ),
        }

        return {
            "current_metrics": current_metrics,
            "system_info": system_info,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/targets")
async def get_performance_targets():
    """
    Get performance targets and current status
    """
    try:
        targets = {
            "scraping_performance": {
                "target": "<10 seconds for standard web pages",
                "metric": "max_processing_time",
                "threshold": 10.0,
                "unit": "seconds",
            },
            "concurrent_operations": {
                "target": "100+ simultaneous scraping requests",
                "metric": "max_concurrent_requests",
                "threshold": 100,
                "unit": "requests",
            },
            "success_rate": {
                "target": ">95% successful scrapes",
                "metric": "success_rate",
                "threshold": 0.95,
                "unit": "percentage",
            },
        }

        # Check recent performance against targets
        target_status = {}
        if performance_history:
            recent_tests = [
                t
                for t in performance_history
                if isinstance(t.get("timestamp"), datetime)
                and t["timestamp"] > datetime.now() - timedelta(hours=24)
            ]

            if recent_tests:
                # Analyze recent performance
                recent_performance_tests = [
                    t for t in recent_tests if t.get("test_type") != "benchmark"
                ]

                if recent_performance_tests:
                    avg_max_time = sum(
                        t.get("max_time", 0) for t in recent_performance_tests
                    ) / len(recent_performance_tests)
                    avg_success_rate = sum(
                        t.get("success_rate", 0) for t in recent_performance_tests
                    ) / len(recent_performance_tests)

                    target_status = {
                        "scraping_performance": {
                            "current_value": avg_max_time,
                            "target_met": avg_max_time < 10.0,
                            "last_24h_average": avg_max_time,
                        },
                        "success_rate": {
                            "current_value": avg_success_rate,
                            "target_met": avg_success_rate > 0.95,
                            "last_24h_average": avg_success_rate,
                        },
                    }

        return {
            "targets": targets,
            "current_status": target_status,
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting performance targets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get targets: {str(e)}")


@router.post("/quick-test")
async def quick_performance_test():
    """
    Quick performance test on standard websites
    """
    try:
        if not HighPerformanceScraper:
            raise HTTPException(
                status_code=500, detail="High-performance scraper not available"
            )

        # Standard test URLs
        test_urls = ["https://example.com", "https://httpbin.org/html"]

        async with HighPerformanceScraper(max_concurrent=5, timeout=8) as scraper:
            start_time = time.time()
            results = await scraper.scrape_urls(test_urls)
            total_time = time.time() - start_time

            successful = [r for r in results if r.status == "success"]
            max_time = max(r.processing_time for r in successful) if successful else 0

            quick_result = {
                "test_timestamp": datetime.now().isoformat(),
                "total_urls": len(test_urls),
                "successful": len(successful),
                "total_time": total_time,
                "max_individual_time": max_time,
                "target_met": max_time < 10.0,
                "status": "PASS" if max_time < 10.0 else "FAIL",
                "details": [
                    {
                        "url": r.url,
                        "time": r.processing_time,
                        "status": r.status,
                        "word_count": r.word_count,
                    }
                    for r in results
                ],
            }

            logger.info(
                f"⚡ Quick test: {quick_result['status']} (max: {max_time:.2f}s)"
            )

            return quick_result

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick test failed: {str(e)}")


@router.get("/health")
async def performance_health_check():
    """
    Health check for performance monitoring system
    """
    try:
        health_status = {
            "status": "healthy",
            "high_performance_scraper": HighPerformanceScraper is not None,
            "performance_history_available": len(performance_history) > 0,
            "last_test": (
                performance_history[-1]["timestamp"].isoformat()
                if performance_history
                else None
            ),
            "total_tests_recorded": len(performance_history),
            "timestamp": datetime.now().isoformat(),
        }

        return health_status

    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# Cleanup function for graceful shutdown
async def cleanup_performance_monitoring():
    """Cleanup performance monitoring resources"""
    global active_scraper
    if active_scraper:
        await active_scraper.close()
        active_scraper = None
