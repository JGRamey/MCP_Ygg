#!/usr/bin/env python3
"""
Scraper Performance Test
Validates the <10 seconds target for standard web pages
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from agents.scraper.high_performance_scraper import HighPerformanceScraper
    print("✅ High-performance scraper imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceValidator:
    """Validates scraping performance against targets"""
    
    def __init__(self):
        self.target_time = 10.0  # seconds
        self.test_results = []
        
    async def test_single_url_performance(self, url: str, iterations: int = 3) -> Dict[str, Any]:
        """Test single URL performance multiple times"""
        
        logger.info(f"🧪 Testing {url} ({iterations} iterations)")
        
        results = []
        
        async with HighPerformanceScraper(max_concurrent=1, timeout=8) as scraper:
            for i in range(iterations):
                start_time = time.time()
                result = await scraper.scrape_url(url, force_refresh=True)  # Force refresh to avoid cache
                end_time = time.time()
                
                iteration_result = {
                    'iteration': i + 1,
                    'processing_time': result.processing_time,
                    'total_time': end_time - start_time,
                    'success': result.status == 'success',
                    'content_length': len(result.content),
                    'word_count': result.word_count,
                    'title': result.title[:50] + "..." if len(result.title) > 50 else result.title
                }
                
                results.append(iteration_result)
                logger.info(f"  Iteration {i+1}: {iteration_result['processing_time']:.2f}s - {'✅' if iteration_result['success'] else '❌'}")
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        if successful_results:
            times = [r['processing_time'] for r in successful_results]
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
        else:
            avg_time = max_time = min_time = 0
        
        test_summary = {
            'url': url,
            'iterations': iterations,
            'successful_iterations': len(successful_results),
            'success_rate': len(successful_results) / iterations,
            'average_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'target_met': max_time < self.target_time,
            'results': results
        }
        
        return test_summary
    
    async def test_concurrent_performance(self, urls: List[str], max_concurrent: int = 10) -> Dict[str, Any]:
        """Test concurrent scraping performance"""
        
        logger.info(f"🚀 Testing concurrent performance: {len(urls)} URLs, {max_concurrent} concurrent")
        
        async with HighPerformanceScraper(max_concurrent=max_concurrent, timeout=8) as scraper:
            start_time = time.time()
            results = await scraper.scrape_urls(urls, max_concurrent)
            total_time = time.time() - start_time
            
            successful = [r for r in results if r.status == 'success']
            times = [r.processing_time for r in successful] if successful else [0]
            
            concurrent_test = {
                'total_urls': len(urls),
                'successful_scrapes': len(successful),
                'success_rate': len(successful) / len(urls) if urls else 0,
                'total_time': total_time,
                'average_time_per_url': sum(times) / len(times) if times else 0,
                'max_individual_time': max(times) if times else 0,
                'min_individual_time': min(times) if times else 0,
                'throughput': len(successful) / total_time if total_time > 0 else 0,
                'target_met': max(times) < self.target_time if times else False,
                'max_concurrent': max_concurrent
            }
            
            return concurrent_test
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance validation"""
        
        logger.info("🏁 Starting comprehensive performance validation")
        
        # Test URLs (public domain and testing sites)
        test_urls = [
            "https://example.com",
            "https://httpbin.org/html",
            "https://httpbin.org/status/200",
            "https://jsonplaceholder.typicode.com/posts/1",
            "https://www.gutenberg.org/cache/epub/8438/pg8438-images.html",  # Aristotle's Nicomachean Ethics
        ]
        
        comprehensive_results = {
            'test_timestamp': time.time(),
            'target_time_seconds': self.target_time,
            'single_url_tests': [],
            'concurrent_tests': [],
            'overall_assessment': {}
        }
        
        # Test each URL individually
        for url in test_urls[:3]:  # Test first 3 URLs individually
            single_test = await self.test_single_url_performance(url, iterations=2)
            comprehensive_results['single_url_tests'].append(single_test)
        
        # Test concurrent performance
        concurrent_test_1 = await self.test_concurrent_performance(test_urls, max_concurrent=5)
        comprehensive_results['concurrent_tests'].append(concurrent_test_1)
        
        concurrent_test_2 = await self.test_concurrent_performance(test_urls, max_concurrent=10)
        comprehensive_results['concurrent_tests'].append(concurrent_test_2)
        
        # Overall assessment
        all_single_tests_pass = all(test['target_met'] for test in comprehensive_results['single_url_tests'])
        all_concurrent_tests_pass = all(test['target_met'] for test in comprehensive_results['concurrent_tests'])
        
        overall_max_time = 0
        total_successful_scrapes = 0
        
        for test in comprehensive_results['single_url_tests']:
            overall_max_time = max(overall_max_time, test['max_time'])
            total_successful_scrapes += test['successful_iterations']
        
        for test in comprehensive_results['concurrent_tests']:
            overall_max_time = max(overall_max_time, test['max_individual_time'])
            total_successful_scrapes += test['successful_scrapes']
        
        comprehensive_results['overall_assessment'] = {
            'overall_target_met': all_single_tests_pass and all_concurrent_tests_pass,
            'overall_max_time': overall_max_time,
            'total_successful_scrapes': total_successful_scrapes,
            'single_url_tests_pass': all_single_tests_pass,
            'concurrent_tests_pass': all_concurrent_tests_pass,
            'performance_grade': self._calculate_performance_grade(comprehensive_results)
        }
        
        return comprehensive_results
    
    def _calculate_performance_grade(self, results: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        
        score = 0
        max_score = 100
        
        # Single URL test scoring (40 points)
        single_tests = results['single_url_tests']
        if single_tests:
            single_pass_rate = sum(1 for test in single_tests if test['target_met']) / len(single_tests)
            score += single_pass_rate * 40
        
        # Concurrent test scoring (40 points)
        concurrent_tests = results['concurrent_tests']
        if concurrent_tests:
            concurrent_pass_rate = sum(1 for test in concurrent_tests if test['target_met']) / len(concurrent_tests)
            score += concurrent_pass_rate * 40
        
        # Success rate scoring (20 points)
        total_attempts = sum(test['iterations'] for test in single_tests)
        total_successes = sum(test['successful_iterations'] for test in single_tests)
        
        for test in concurrent_tests:
            total_attempts += test['total_urls']
            total_successes += test['successful_scrapes']
        
        if total_attempts > 0:
            success_rate = total_successes / total_attempts
            score += success_rate * 20
        
        # Assign grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        
        print("\n" + "="*60)
        print("🏆 SCRAPER PERFORMANCE TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Assessment:")
        assessment = results['overall_assessment']
        print(f"   Target Met: {'✅ YES' if assessment['overall_target_met'] else '❌ NO'}")
        print(f"   Max Time: {assessment['overall_max_time']:.2f}s (target: <{self.target_time}s)")
        print(f"   Performance Grade: {assessment['performance_grade']}")
        print(f"   Total Successful Scrapes: {assessment['total_successful_scrapes']}")
        
        print(f"\n🔍 Single URL Tests:")
        for i, test in enumerate(results['single_url_tests'], 1):
            status = "✅ PASS" if test['target_met'] else "❌ FAIL"
            print(f"   {i}. {test['url'][:50]}...")
            print(f"      Status: {status}")
            print(f"      Max Time: {test['max_time']:.2f}s")
            print(f"      Success Rate: {test['success_rate']:.1%}")
        
        print(f"\n🚀 Concurrent Tests:")
        for i, test in enumerate(results['concurrent_tests'], 1):
            status = "✅ PASS" if test['target_met'] else "❌ FAIL"
            print(f"   {i}. {test['total_urls']} URLs, {test['max_concurrent']} concurrent")
            print(f"      Status: {status}")
            print(f"      Max Individual Time: {test['max_individual_time']:.2f}s")
            print(f"      Throughput: {test['throughput']:.1f} URLs/second")
            print(f"      Success Rate: {test['success_rate']:.1%}")
        
        print("\n" + "="*60)
        
        if assessment['overall_target_met']:
            print("🎉 PERFORMANCE TARGET ACHIEVED!")
            print("   Scraper meets the <10 seconds requirement for standard web pages")
        else:
            print("⚠️  PERFORMANCE TARGET NOT MET")
            print("   Optimization needed to meet <10 seconds requirement")
        
        print("="*60)


async def main():
    """Main test execution"""
    
    print("🧪 Scraper Performance Validation Test")
    print("Target: <10 seconds for standard web pages")
    print("-" * 50)
    
    validator = PerformanceValidator()
    
    try:
        # Run comprehensive test
        results = await validator.run_comprehensive_test()
        
        # Print results
        validator.print_results(results)
        
        # Return appropriate exit code
        if results['overall_assessment']['overall_target_met']:
            print("\n✅ Test PASSED - Performance target achieved!")
            sys.exit(0)
        else:
            print("\n❌ Test FAILED - Performance target not met")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())