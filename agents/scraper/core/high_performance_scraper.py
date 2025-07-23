#!/usr/bin/env python3
"""
High-Performance Scraper Agent
Optimized for <10 seconds performance target for standard web pages
"""

import asyncio
import aiohttp
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin
import hashlib
import json
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor
import gzip
import zlib

# Fast parsing libraries
from selectolax.parser import HTMLParser
import chardet
from bs4 import BeautifulSoup

# Caching and performance
import redis
import pickle
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Optimized scrape result"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    status: str
    word_count: int
    content_hash: str


@dataclass
class PerformanceMetrics:
    """Performance tracking"""
    total_time: float
    network_time: float
    parsing_time: float
    processing_time: float
    cache_hits: int
    cache_misses: int
    concurrent_requests: int


class FastHTMLParser:
    """Ultra-fast HTML parser using selectolax"""
    
    def __init__(self):
        # Pre-compiled regex patterns for common extraction
        self.title_pattern = re.compile(r'<title[^>]*>([^<]+)</title>', re.IGNORECASE | re.DOTALL)
        self.meta_description_pattern = re.compile(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE)
        self.meta_keywords_pattern = re.compile(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE)
        
        # Common noise selectors to remove
        self.noise_selectors = [
            'script', 'style', 'nav', 'footer', 'header', 
            '.advertisement', '.ad', '.sidebar', '.menu',
            '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]'
        ]
    
    def extract_fast(self, html: str) -> Tuple[str, Dict[str, Any]]:
        """Ultra-fast content extraction using selectolax"""
        start_time = time.time()
        
        try:
            # Use selectolax for fast parsing
            tree = HTMLParser(html)
            
            # Extract title (fastest method first)
            title = ""
            title_node = tree.css_first('title')
            if title_node:
                title = title_node.text().strip()
            else:
                # Fallback to regex
                title_match = self.title_pattern.search(html)
                if title_match:
                    title = title_match.group(1).strip()
            
            # Extract metadata quickly
            metadata = {
                'title': title,
                'description': '',
                'keywords': '',
                'extraction_method': 'selectolax_fast',
                'parsing_time': 0
            }
            
            # Fast meta tag extraction
            for meta in tree.css('meta'):
                name = meta.attributes.get('name', '').lower()
                content = meta.attributes.get('content', '')
                
                if name == 'description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
            
            # Remove noise elements
            for selector in self.noise_selectors:
                for node in tree.css(selector):
                    node.decompose()
            
            # Extract main content
            content_candidates = []
            
            # Try semantic HTML5 elements first
            for tag in ['main', 'article', '[role="main"]']:
                nodes = tree.css(tag)
                if nodes:
                    content_candidates.extend([node.text() for node in nodes])
            
            # Fallback to common content containers
            if not content_candidates:
                for selector in ['.content', '.main-content', '.post-content', '#content', '#main']:
                    nodes = tree.css(selector)
                    if nodes:
                        content_candidates.extend([node.text() for node in nodes])
            
            # Final fallback - extract from body
            if not content_candidates:
                body = tree.css_first('body')
                if body:
                    content_candidates.append(body.text())
            
            # Join and clean content
            content = ' '.join(content_candidates).strip()
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            
            metadata['parsing_time'] = time.time() - start_time
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Fast parsing failed: {e}")
            # Fallback to BeautifulSoup
            return self._fallback_extract(html)
    
    def _fallback_extract(self, html: str) -> Tuple[str, Dict[str, Any]]:
        """Fallback to BeautifulSoup for complex pages"""
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract metadata
            metadata = {'title': title, 'extraction_method': 'beautifulsoup_fallback'}
            
            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            # Extract content
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")
            return "", {'title': '', 'extraction_method': 'failed'}


class HighPerformanceCache:
    """High-performance caching system"""
    
    def __init__(self, redis_url: str = None, ttl: int = 3600):
        self.ttl = ttl
        self.memory_cache = {}
        self.memory_cache_size = 1000  # Max items in memory
        
        # Try to connect to Redis for distributed caching
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                logger.info("✅ Redis cache connected")
            else:
                self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}")
            self.redis_client = None
    
    def _generate_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return f"scrape_cache:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
    
    async def get(self, url: str) -> Optional[ScrapeResult]:
        """Get cached result"""
        key = self._generate_key(url)
        
        # Try memory cache first
        if key in self.memory_cache:
            result, timestamp = self.memory_cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return result
            else:
                del self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    # Decompress and deserialize
                    decompressed = gzip.decompress(cached_data)
                    result = pickle.loads(decompressed)
                    
                    # Add to memory cache
                    self.memory_cache[key] = (result, datetime.now())
                    return result
            except Exception as e:
                logger.error(f"Redis cache read error: {e}")
        
        return None
    
    async def set(self, url: str, result: ScrapeResult):
        """Cache result"""
        key = self._generate_key(url)
        
        # Add to memory cache
        if len(self.memory_cache) >= self.memory_cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (result, datetime.now())
        
        # Add to Redis cache
        if self.redis_client:
            try:
                # Serialize and compress
                serialized = pickle.dumps(result)
                compressed = gzip.compress(serialized)
                
                self.redis_client.setex(key, self.ttl, compressed)
            except Exception as e:
                logger.error(f"Redis cache write error: {e}")


class HighPerformanceScraper:
    """Ultra-fast web scraper optimized for <10 second performance"""
    
    def __init__(self, 
                 max_concurrent: int = 20,
                 timeout: int = 8,
                 cache_ttl: int = 3600,
                 redis_url: str = None):
        
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # Initialize components
        self.parser = FastHTMLParser()
        self.cache = HighPerformanceCache(redis_url, cache_ttl)
        self.session = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Optimized headers for speed
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; HighPerformanceScraper/1.0; +http://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start(self):
        """Initialize the scraper"""
        # Optimized connector for high performance
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,  # Total connection pool size
            limit_per_host=self.max_concurrent,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Optimized timeout
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=3,
            sock_read=5
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers,
            auto_decompress=True,
            trust_env=True
        )
        
        logger.info(f"✅ High-performance scraper initialized (max_concurrent={self.max_concurrent})")
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        
        self.thread_pool.shutdown(wait=True)
        logger.info("✅ Scraper resources cleaned up")
    
    async def scrape_url(self, url: str, force_refresh: bool = False) -> ScrapeResult:
        """Scrape a single URL with maximum performance"""
        start_time = time.time()
        
        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_result = await self.cache.get(url)
            if cached_result:
                self.metrics.cache_hits += 1
                logger.info(f"⚡ Cache hit for {url}")
                return cached_result
        
        self.metrics.cache_misses += 1
        
        try:
            # Network request with timing
            network_start = time.time()
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return ScrapeResult(
                        url=url,
                        title="",
                        content="",
                        metadata={'error': f'HTTP {response.status}'},
                        timestamp=datetime.now(),
                        processing_time=time.time() - start_time,
                        status='error',
                        word_count=0,
                        content_hash=""
                    )
                
                # Read content
                content_bytes = await response.read()
                
                # Detect encoding quickly
                encoding = response.charset
                if not encoding:
                    # Fast encoding detection
                    detected = chardet.detect(content_bytes[:1024])
                    encoding = detected.get('encoding', 'utf-8')
                
                html = content_bytes.decode(encoding, errors='ignore')
                
            network_time = time.time() - network_start
            
            # Parse content in thread pool for CPU-bound operations
            parsing_start = time.time()
            
            loop = asyncio.get_event_loop()
            content, metadata = await loop.run_in_executor(
                self.thread_pool, 
                self.parser.extract_fast, 
                html
            )
            
            parsing_time = time.time() - parsing_start
            
            # Create result
            total_time = time.time() - start_time
            
            result = ScrapeResult(
                url=url,
                title=metadata.get('title', ''),
                content=content,
                metadata=metadata,
                timestamp=datetime.now(),
                processing_time=total_time,
                status='success',
                word_count=len(content.split()) if content else 0,
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:16]
            )
            
            # Cache the result
            await self.cache.set(url, result)
            
            # Update metrics
            self.metrics.network_time += network_time
            self.metrics.parsing_time += parsing_time
            self.metrics.total_time += total_time
            
            logger.info(f"✅ Scraped {url} in {total_time:.2f}s (network: {network_time:.2f}s, parsing: {parsing_time:.2f}s)")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout scraping {url}")
            return ScrapeResult(
                url=url, title="", content="", metadata={'error': 'timeout'},
                timestamp=datetime.now(), processing_time=time.time() - start_time,
                status='timeout', word_count=0, content_hash=""
            )
            
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}")
            return ScrapeResult(
                url=url, title="", content="", metadata={'error': str(e)},
                timestamp=datetime.now(), processing_time=time.time() - start_time,
                status='error', word_count=0, content_hash=""
            )
    
    async def scrape_urls(self, urls: List[str], max_concurrent: int = None) -> List[ScrapeResult]:
        """Scrape multiple URLs concurrently"""
        start_time = time.time()
        
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_scrape(url):
            async with semaphore:
                return await self.scrape_url(url)
        
        # Execute all requests concurrently
        tasks = [bounded_scrape(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, ScrapeResult):
                valid_results.append(result)
            else:
                logger.error(f"Task failed: {result}")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(urls) if urls else 0
        
        logger.info(f"🏁 Scraped {len(valid_results)}/{len(urls)} URLs in {total_time:.2f}s (avg: {avg_time:.2f}s per URL)")
        
        return valid_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'total_requests': self.metrics.cache_hits + self.metrics.cache_misses,
            'cache_hit_rate': self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0,
            'average_network_time': self.metrics.network_time / self.metrics.cache_misses if self.metrics.cache_misses > 0 else 0,
            'average_parsing_time': self.metrics.parsing_time / self.metrics.cache_misses if self.metrics.cache_misses > 0 else 0,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'max_concurrent': self.max_concurrent,
            'timeout': self.timeout
        }
    
    async def performance_test(self, test_urls: List[str] = None) -> Dict[str, Any]:
        """Run performance test on common websites"""
        if test_urls is None:
            test_urls = [
                'https://example.com',
                'https://httpbin.org/html',
                'https://wikipedia.org',
                'https://news.ycombinator.com',
                'https://github.com'
            ]
        
        logger.info(f"🧪 Running performance test on {len(test_urls)} URLs...")
        
        start_time = time.time()
        results = await self.scrape_urls(test_urls)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r.status == 'success']
        avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
        max_time = max(r.processing_time for r in successful) if successful else 0
        min_time = min(r.processing_time for r in successful) if successful else 0
        
        performance_report = {
            'test_urls': len(test_urls),
            'successful_scrapes': len(successful),
            'success_rate': len(successful) / len(test_urls) if test_urls else 0,
            'total_time': total_time,
            'average_time_per_url': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'target_met': max_time < 10.0,  # <10 seconds target
            'metrics': self.get_performance_metrics()
        }
        
        logger.info(f"📊 Performance test results:")
        logger.info(f"   Success rate: {performance_report['success_rate']:.1%}")
        logger.info(f"   Average time: {avg_time:.2f}s")
        logger.info(f"   Max time: {max_time:.2f}s")
        logger.info(f"   Target (<10s): {'✅ MET' if performance_report['target_met'] else '❌ NOT MET'}")
        
        return performance_report


# Integration wrapper for existing scraper interface
class OptimizedScraperAgent:
    """Wrapper to integrate with existing scraper interface"""
    
    def __init__(self):
        self.scraper = None
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL with optimized performance - compatible interface"""
        
        if not self.scraper:
            self.scraper = HighPerformanceScraper()
            await self.scraper.start()
        
        try:
            result = await self.scraper.scrape_url(url)
            
            # Return in expected format
            return {
                'success': result.status == 'success',
                'content': result.content,
                'title': result.title,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'word_count': result.word_count,
                'url': result.url
            }
            
        except Exception as e:
            logger.error(f"Optimized scraper error: {e}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'metadata': {'error': str(e)},
                'processing_time': 0,
                'word_count': 0,
                'url': url
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.scraper:
            await self.scraper.close()


# Test and demonstration
async def main():
    """Demo and performance test"""
    
    # Test URLs
    test_urls = [
        'https://example.com',
        'https://httpbin.org/html',
        'https://httpbin.org/status/200'
    ]
    
    async with HighPerformanceScraper() as scraper:
        # Performance test
        performance_report = await scraper.performance_test(test_urls)
        
        print("\n🏆 Performance Test Results:")
        print(f"Success Rate: {performance_report['success_rate']:.1%}")
        print(f"Average Time: {performance_report['average_time_per_url']:.2f}s")
        print(f"Max Time: {performance_report['max_time']:.2f}s")
        print(f"Target Met (<10s): {'✅ YES' if performance_report['target_met'] else '❌ NO'}")
        
        # Individual scrape test
        print("\n🔍 Individual Scrape Test:")
        result = await scraper.scrape_url('https://example.com')
        print(f"URL: {result.url}")
        print(f"Title: {result.title}")
        print(f"Content Length: {len(result.content)} chars")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Status: {result.status}")


if __name__ == "__main__":
    asyncio.run(main())