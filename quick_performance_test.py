#!/usr/bin/env python3
"""
Quick Performance Test
Fast validation that scraping performance target is met
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

async def quick_test():
    """Quick performance validation"""
    
    print("⚡ Quick Performance Test")
    print("Target: <10 seconds for standard web pages")
    print("-" * 40)
    
    try:
        from agents.scraper.high_performance_scraper import HighPerformanceScraper
        
        test_urls = [
            "https://example.com",
            "https://httpbin.org/html"
        ]
        
        async with HighPerformanceScraper(max_concurrent=2, timeout=8) as scraper:
            print("🧪 Testing 2 URLs...")
            
            start_time = time.time()
            results = await scraper.scrape_urls(test_urls)
            total_time = time.time() - start_time
            
            successful = [r for r in results if r.status == 'success']
            max_time = max(r.processing_time for r in successful) if successful else 0
            
            print(f"✅ Results:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Max individual time: {max_time:.2f}s")
            print(f"   Success rate: {len(successful)}/{len(test_urls)}")
            print(f"   Target met: {'✅ YES' if max_time < 10.0 else '❌ NO'}")
            
            if max_time < 10.0:
                print("\n🎉 PERFORMANCE TARGET ACHIEVED!")
                return True
            else:
                print("\n⚠️  Performance target not met")
                return False
                
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        print("   Run: pip install aiohttp selectolax chardet")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)