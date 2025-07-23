#!/usr/bin/env python3
"""
Aristotle's Nicomachean Ethics - Scraping Performance Test Only
Real-world test case for scraping performance (without concept analysis)
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any
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

class AristotleScrapingTest:
    """Performance test for Aristotle's Nicomachean Ethics scraping only"""
    
    def __init__(self):
        self.test_url = "https://www.gutenberg.org/cache/epub/8438/pg8438-images.html"
        self.document_name = "Aristotle's Nicomachean Ethics"
        self.target_scraping_time = 10.0  # seconds
        
    async def test_scraping_performance(self) -> Dict[str, Any]:
        """Test scraping performance on Aristotle's text"""
        
        logger.info(f"🏛️  Testing scraping performance: {self.document_name}")
        logger.info(f"   URL: {self.test_url}")
        logger.info(f"   Target: <{self.target_scraping_time} seconds")
        
        scraping_results = {}
        
        async with HighPerformanceScraper(max_concurrent=1, timeout=15) as scraper:
            
            # Test 1: Cold scrape (no cache)
            logger.info("   📥 Cold scrape test...")
            start_time = time.time()
            result = await scraper.scrape_url(self.test_url, force_refresh=True)
            cold_scrape_time = time.time() - start_time
            
            # Test 2: Warm scrape (with cache)
            logger.info("   🔥 Warm scrape test...")
            start_time = time.time()
            cached_result = await scraper.scrape_url(self.test_url, force_refresh=False)
            warm_scrape_time = time.time() - start_time
            
            # Test 3: Multiple iterations for consistency
            logger.info("   🔄 Consistency test (3 iterations)...")
            iteration_times = []
            for i in range(3):
                start_time = time.time()
                iter_result = await scraper.scrape_url(self.test_url, force_refresh=True)
                iteration_time = time.time() - start_time
                iteration_times.append(iteration_time)
                logger.info(f"      Iteration {i+1}: {iteration_time:.2f}s")
            
            # Analyze content
            content_analysis = {
                'success': result.status == 'success',
                'content_length': len(result.content),
                'word_count': result.word_count,
                'title': result.title,
                'has_philosophical_content': 'ethics' in result.content.lower() or 'virtue' in result.content.lower()
            }
            
            scraping_results = {
                'document_name': self.document_name,
                'url': self.test_url,
                'cold_scrape_time': cold_scrape_time,
                'warm_scrape_time': warm_scrape_time,
                'iteration_times': iteration_times,
                'average_iteration_time': sum(iteration_times) / len(iteration_times),
                'max_iteration_time': max(iteration_times),
                'min_iteration_time': min(iteration_times),
                'content_analysis': content_analysis,
                'target_met': max(iteration_times) < self.target_scraping_time,
                'performance_metrics': scraper.get_performance_metrics()
            }
            
            logger.info(f"   ✅ Scraping completed:")
            logger.info(f"      Cold scrape: {cold_scrape_time:.2f}s")
            logger.info(f"      Warm scrape: {warm_scrape_time:.2f}s") 
            logger.info(f"      Average: {scraping_results['average_iteration_time']:.2f}s")
            logger.info(f"      Target met: {'✅ YES' if scraping_results['target_met'] else '❌ NO'}")
            
        return scraping_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "="*70)
        print("🏛️  ARISTOTLE'S NICOMACHEAN ETHICS - SCRAPING PERFORMANCE TEST")
        print("="*70)
        
        print(f"\n📖 Document: {results['document_name']}")
        print(f"   URL: {results['url']}")
        print(f"   Description: Aristotle's foundational work on virtue ethics")
        
        # Scraping results
        print(f"\n📥 SCRAPING PERFORMANCE:")
        print(f"   Target: <{self.target_scraping_time} seconds")
        print(f"   Status: {'✅ PASS' if results.get('target_met') else '❌ FAIL'}")
        print(f"   Cold scrape: {results.get('cold_scrape_time', 0):.2f}s")
        print(f"   Warm scrape: {results.get('warm_scrape_time', 0):.2f}s")
        print(f"   Average time: {results.get('average_iteration_time', 0):.2f}s")
        print(f"   Max time: {results.get('max_iteration_time', 0):.2f}s")
        print(f"   Min time: {results.get('min_iteration_time', 0):.2f}s")
        
        content = results.get('content_analysis', {})
        if content.get('success'):
            print(f"   Content: {content.get('word_count', 0):,} words, {content.get('content_length', 0):,} chars")
            print(f"   Title: {content.get('title', 'N/A')}")
            print(f"   Contains philosophical content: {'✅ YES' if content.get('has_philosophical_content') else '❌ NO'}")
        
        print("\n" + "="*70)
        
        if results.get('target_met'):
            print("🎉 PERFORMANCE TARGET ACHIEVED!")
            print("   Scraper successfully processes classical philosophical texts within target time")
        else:
            print("⚠️  PERFORMANCE TARGET NOT MET")
            print("   Scraping performance needs optimization for large texts")
        
        print("="*70)

async def main():
    """Run Aristotle scraping performance test"""
    
    print("🏛️  Aristotle's Nicomachean Ethics - Scraping Performance Test")
    print("Testing scraping performance on classical philosophy text")
    print("-" * 60)
    
    tester = AristotleScrapingTest()
    
    try:
        # Run scraping performance test
        results = await tester.test_scraping_performance()
        
        # Print results
        tester.print_results(results)
        
        # Return appropriate exit code
        if results.get('target_met'):
            print("\n✅ Test PASSED - Scraping performance target achieved!")
            sys.exit(0)
        else:
            print("\n⚠️  Test completed with performance issues")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Aristotle scraping test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())