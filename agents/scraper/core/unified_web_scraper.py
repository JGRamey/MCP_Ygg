#!/usr/bin/env python3
"""
Unified Web Scraper for MCP Yggdrasil
Phase 3: Advanced scraper architecture with multiple extraction methods
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import asyncio

from .anti_detection import AntiDetectionManager, RateLimiter

# Local imports
from .enhanced_content_extractor import EnhancedContentExtractor
from .site_specific_parsers import SiteParserManager

logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Standardized scraping result."""

    url: str
    success: bool
    content: Optional[Dict] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    method_used: str = "unknown"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ScraperConfig:
    """Configuration for the unified scraper."""

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]

        # Scraping behavior
        self.requests_per_second = 1.0
        self.retry_attempts = 3
        self.retry_delay = 2.0
        self.timeout = 30

        # Content extraction
        self.use_trafilatura = True
        self.extract_links = True
        self.extract_images = True

        # Anti-detection
        self.use_selenium_fallback = True
        self.use_proxy_rotation = False
        self.proxies = []

        # Quality control
        self.min_content_length = 100
        self.max_content_length = 1000000
        self.verify_content = True

        # Respect for sites
        self.respect_robots_txt = True
        self.academic_site_delay = 3.0


class UnifiedWebScraper:
    """Unified web scraper with multiple extraction methods and intelligent fallbacks."""

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()

        # Initialize components
        self.content_extractor = EnhancedContentExtractor()
        self.anti_detection = AntiDetectionManager(self._config_to_dict())
        self.rate_limiter = RateLimiter(self.config.requests_per_second)
        self.site_parser_manager = SiteParserManager()

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "method_usage": {"http_requests": 0, "selenium": 0, "cached": 0},
            "average_processing_time": 0.0,
        }

        # Simple cache
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour

        logger.info("✅ Unified Web Scraper initialized")
        logger.info(f"   Trafilatura: {self.config.use_trafilatura}")
        logger.info(f"   Selenium fallback: {self.config.use_selenium_fallback}")
        logger.info(f"   Rate limit: {self.config.requests_per_second} req/s")

    async def scrape_url(self, url: str, **kwargs) -> ScrapingResult:
        """Scrape a single URL with intelligent method selection."""
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Check cache first
            cache_key = self._generate_cache_key(url)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats["method_usage"]["cached"] += 1
                logger.debug(f"Cache hit for {url}")
                return cached_result

            # Respect rate limiting
            self.rate_limiter.wait_if_needed(url)

            # Get scraping strategy
            strategy = self.anti_detection.get_recommended_strategy(url)

            # Try HTTP first (faster)
            html_content = None
            method_used = "http_requests"

            if not strategy["use_selenium"]:
                html_content = await self._scrape_with_http(url)
                self.stats["method_usage"]["http_requests"] += 1

            # Fallback to Selenium if needed
            if not html_content and self.config.use_selenium_fallback:
                html_content = await self._scrape_with_selenium(url)
                method_used = "selenium"
                self.stats["method_usage"]["selenium"] += 1

            if not html_content:
                error_msg = "Failed to fetch content with all methods"
                self.stats["failed_requests"] += 1
                return ScrapingResult(
                    url=url,
                    success=False,
                    error=error_msg,
                    processing_time=time.time() - start_time,
                    method_used=method_used,
                )

            # Try site-specific parser first
            site_parsed_content = self.site_parser_manager.parse_content(
                html_content, url
            )

            if site_parsed_content and site_parsed_content.title:
                # Use site-specific parsed content
                extracted_content = {
                    "title": site_parsed_content.title,
                    "main_text": site_parsed_content.content
                    or site_parsed_content.raw_text
                    or "",
                    "author": site_parsed_content.author,
                    "date": site_parsed_content.date,
                    "abstract": site_parsed_content.abstract,
                    "keywords": site_parsed_content.keywords,
                    "citations": site_parsed_content.citations,
                    "site_specific": True,
                }
                metadata = site_parsed_content.metadata or {}
                logger.info(f"✅ Site-specific parsing successful for {url}")
            else:
                # Fall back to Trafilatura extraction
                extracted_content = self.content_extractor.extract_main_content(
                    html_content, url
                )
                metadata = self.content_extractor.extract_metadata(html_content, url)
                extracted_content["site_specific"] = False
                logger.debug(f"Using Trafilatura extraction for {url}")

            # Extract links if configured
            if self.config.extract_links:
                links_data = self.content_extractor.extract_links_and_references(
                    html_content, url
                )
                extracted_content["links"] = links_data

            # Quality validation
            if self.config.verify_content:
                quality_score = self._assess_content_quality(extracted_content)
                extracted_content["quality_score"] = quality_score

            # Create successful result
            processing_time = time.time() - start_time
            result = ScrapingResult(
                url=url,
                success=True,
                content=extracted_content,
                metadata=metadata,
                processing_time=processing_time,
                method_used=method_used,
            )

            # Cache the result
            self._add_to_cache(cache_key, result)

            # Update stats
            self.stats["successful_requests"] += 1
            self._update_average_time(processing_time)

            logger.info(
                f"✅ Successfully scraped {url} in {processing_time:.2f}s using {method_used}"
            )
            return result

        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"❌ Failed to scrape {url}: {e}")
            return ScrapingResult(
                url=url,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                method_used=method_used,
            )

    async def scrape_multiple(
        self, urls: List[str], max_concurrent: int = 5
    ) -> List[ScrapingResult]:
        """Scrape multiple URLs with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url: str) -> ScrapingResult:
            async with semaphore:
                return await self.scrape_url(url)

        logger.info(
            f"🚀 Starting batch scraping of {len(urls)} URLs (max concurrent: {max_concurrent})"
        )

        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ScrapingResult(
                        url=urls[i],
                        success=False,
                        error=str(result),
                        method_used="exception",
                    )
                )
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"✅ Batch scraping complete: {successful}/{len(urls)} successful")

        return processed_results

    async def _scrape_with_http(self, url: str) -> Optional[str]:
        """Scrape using HTTP requests with anti-detection."""
        headers = self.anti_detection.get_random_headers()
        proxy = (
            self.anti_detection.get_next_proxy()
            if self.config.use_proxy_rotation
            else None
        )

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        for attempt in range(self.config.retry_attempts):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        url,
                        headers=headers,
                        proxy=proxy["http"] if proxy else None,
                        ssl=False,
                    ) as response:
                        if response.status == 200:
                            content = await response.text()
                            logger.debug(
                                f"HTTP success for {url} (attempt {attempt + 1})"
                            )
                            return content
                        elif response.status in [403, 429]:
                            # Likely blocked, don't retry with HTTP
                            logger.warning(
                                f"HTTP blocked ({response.status}) for {url}"
                            )
                            return None
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")

            except Exception as e:
                logger.debug(f"HTTP attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return None

    async def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape using Selenium with stealth mode."""
        driver = self.anti_detection.setup_stealth_webdriver(url)

        if not driver:
            logger.warning("Selenium WebDriver not available")
            return None

        try:
            # Navigate to URL
            driver.get(url)

            # Wait for page load
            await asyncio.sleep(2)

            # Simulate human behavior
            self.anti_detection.simulate_human_behavior(driver)

            # Get page source
            html_content = driver.page_source

            logger.debug(f"Selenium success for {url}")
            return html_content

        except Exception as e:
            logger.error(f"Selenium failed for {url}: {e}")
            return None

        finally:
            try:
                driver.quit()
            except Exception:
                pass

    def _assess_content_quality(self, content: Dict) -> float:
        """Assess content quality (0.0 to 1.0)."""
        score = 0.0

        # Check text length
        main_text = content.get("main_text", "")
        if len(main_text) > self.config.min_content_length:
            score += 0.3

        # Check for title
        if content.get("title"):
            score += 0.2

        # Check for author
        if content.get("author"):
            score += 0.1

        # Check for date
        if content.get("date"):
            score += 0.1

        # Check content metrics
        metrics = content.get("content_metrics", {})
        if metrics.get("word_count", 0) > 50:
            score += 0.2

        # Check language detection
        if content.get("language"):
            score += 0.1

        return min(score, 1.0)

    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[ScrapingResult]:
        """Get result from cache if not expired."""
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["result"]
            else:
                # Remove expired item
                del self.response_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: ScrapingResult):
        """Add result to cache."""
        self.response_cache[cache_key] = {"result": result, "timestamp": time.time()}

    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary for anti-detection manager."""
        return {
            "user_agents": self.config.user_agents,
            "proxies": self.config.proxies,
            "use_proxy_rotation": self.config.use_proxy_rotation,
            "headless_browser": True,
            "selenium_urls": [],
        }

    def _update_average_time(self, processing_time: float):
        """Update average processing time."""
        total_successful = self.stats["successful_requests"]
        current_avg = self.stats["average_processing_time"]

        # Running average calculation
        self.stats["average_processing_time"] = (
            current_avg * (total_successful - 1) + processing_time
        ) / total_successful

    def get_stats(self) -> Dict:
        """Get scraping statistics."""
        total = self.stats["total_requests"]
        success_rate = (
            (self.stats["successful_requests"] / total * 100) if total > 0 else 0
        )

        return {
            "total_requests": total,
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": f"{success_rate:.1f}%",
            "average_processing_time": f"{self.stats['average_processing_time']:.2f}s",
            "method_usage": self.stats["method_usage"],
            "cache_size": len(self.response_cache),
            "cache_hit_rate": (
                f"{(self.stats['method_usage']['cached'] / total * 100):.1f}%"
                if total > 0
                else "0%"
            ),
        }

    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        logger.info("Cache cleared")


# Example usage and testing
async def test_unified_scraper():
    """Test the unified web scraper."""
    print("🕷️ Testing Unified Web Scraper")
    print("=" * 50)

    # Initialize scraper
    config = ScraperConfig()
    config.requests_per_second = 2.0  # Faster for testing
    config.use_selenium_fallback = True

    scraper = UnifiedWebScraper(config)

    # Test URLs
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://httpbin.org/json",
    ]

    print(f"\\n📄 Testing single URL scraping:")
    result = await scraper.scrape_url(test_urls[0])

    print(f"   URL: {result.url}")
    print(f"   Success: {result.success}")
    print(f"   Method: {result.method_used}")
    print(f"   Processing Time: {result.processing_time:.2f}s")

    if result.success and result.content:
        print(f"   Title: {result.content.get('title', 'N/A')}")
        print(f"   Content Length: {len(result.content.get('main_text', ''))}")
        print(f"   Quality Score: {result.content.get('quality_score', 'N/A')}")

    # Test batch scraping
    print(f"\\n📦 Testing batch scraping ({len(test_urls)} URLs):")
    results = await scraper.scrape_multiple(test_urls, max_concurrent=2)

    for i, result in enumerate(results):
        status = "✅" if result.success else "❌"
        print(f"   {i+1}. {status} {result.url} ({result.processing_time:.2f}s)")

    # Show statistics
    print(f"\\n📊 Scraper Statistics:")
    stats = scraper.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")

    print("\\n✅ Unified Web Scraper test complete!")


if __name__ == "__main__":
    asyncio.run(test_unified_scraper())
