#!/usr/bin/env python3
"""
Multi-Source Content Acquisition System for MCP Yggdrasil
Phase 3: Intelligent source selection and content aggregation
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import asyncio

from .site_specific_parsers import ParsedContent
from .unified_web_scraper import ScraperConfig, ScrapingResult, UnifiedWebScraper

logger = logging.getLogger(__name__)


@dataclass
class SourcePriority:
    """Priority configuration for different source types."""

    domain: str
    priority: int = 5  # 1-10, higher is better
    reliability: float = 0.8  # 0.0-1.0
    content_quality: float = 0.7  # 0.0-1.0
    update_frequency: str = "unknown"  # daily, weekly, monthly, yearly, unknown
    source_type: str = "general"  # academic, news, reference, social, technical


@dataclass
class ContentRequest:
    """Request for content on a specific topic."""

    query: str
    domains: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=list)  # academic, reference, etc.
    max_sources: int = 5
    min_quality_score: float = 0.6
    include_citations: bool = True
    language: str = "en"
    date_range: Optional[Tuple[str, str]] = None  # (start_date, end_date)


@dataclass
class AggregatedContent:
    """Aggregated content from multiple sources."""

    query: str
    sources: List[ScrapingResult]
    primary_content: Optional[Dict] = None
    supporting_content: List[Dict] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    confidence_score: float = 0.0
    acquisition_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class SourceDiscovery:
    """Discovers relevant sources for content queries."""

    def __init__(self):
        self.search_engines = {
            "google_scholar": "https://scholar.google.com/scholar?q={}",
            "semantic_scholar": "https://www.semanticscholar.org/search?q={}",
            "arxiv_search": "https://arxiv.org/search/?query={}&searchtype=all",
            "pubmed_search": "https://pubmed.ncbi.nlm.nih.gov/?term={}",
            "wikipedia_search": "https://en.wikipedia.org/w/api.php?action=opensearch&search={}&limit=10&format=json",
        }

        self.source_priorities = {
            # Academic sources (highest priority)
            "arxiv.org": SourcePriority(
                "arxiv.org", 10, 0.95, 0.9, "daily", "academic"
            ),
            "pubmed.ncbi.nlm.nih.gov": SourcePriority(
                "pubmed.ncbi.nlm.nih.gov", 10, 0.95, 0.9, "daily", "academic"
            ),
            "scholar.google.com": SourcePriority(
                "scholar.google.com", 9, 0.9, 0.85, "daily", "academic"
            ),
            "ieee.org": SourcePriority("ieee.org", 9, 0.9, 0.85, "weekly", "academic"),
            "acm.org": SourcePriority("acm.org", 9, 0.9, 0.85, "weekly", "academic"),
            # Reference sources (high priority)
            "wikipedia.org": SourcePriority(
                "wikipedia.org", 8, 0.85, 0.8, "daily", "reference"
            ),
            "britannica.com": SourcePriority(
                "britannica.com", 8, 0.9, 0.85, "monthly", "reference"
            ),
            "plato.stanford.edu": SourcePriority(
                "plato.stanford.edu", 9, 0.95, 0.9, "yearly", "reference"
            ),
            # News and media (medium-high priority)
            "bbc.com": SourcePriority("bbc.com", 7, 0.8, 0.75, "daily", "news"),
            "reuters.com": SourcePriority("reuters.com", 7, 0.8, 0.75, "daily", "news"),
            "nature.com": SourcePriority(
                "nature.com", 9, 0.9, 0.85, "weekly", "academic"
            ),
            "science.org": SourcePriority(
                "science.org", 9, 0.9, 0.85, "weekly", "academic"
            ),
            # Technical sources (medium-high priority)
            "stackoverflow.com": SourcePriority(
                "stackoverflow.com", 7, 0.75, 0.7, "daily", "technical"
            ),
            "github.com": SourcePriority(
                "github.com", 6, 0.7, 0.65, "daily", "technical"
            ),
            "medium.com": SourcePriority("medium.com", 5, 0.6, 0.6, "daily", "blog"),
            # General web (lower priority)
            "default": SourcePriority("default", 4, 0.5, 0.5, "unknown", "general"),
        }

    def get_source_priority(self, url: str) -> SourcePriority:
        """Get priority configuration for a URL."""
        domain = urlparse(url).netloc.lower()

        # Check exact matches first
        if domain in self.source_priorities:
            return self.source_priorities[domain]

        # Check subdomain matches
        for priority_domain, priority in self.source_priorities.items():
            if priority_domain in domain:
                return priority

        return self.source_priorities["default"]

    async def discover_sources(self, request: ContentRequest) -> List[str]:
        """Discover relevant sources for a content request."""
        discovered_urls = []

        # If specific domains requested, prioritize those
        if request.domains:
            for domain in request.domains:
                if domain.startswith("http"):
                    discovered_urls.append(domain)
                else:
                    # Generate search URLs for the domain
                    search_url = f"site:{domain} {request.query}"
                    discovered_urls.append(
                        f"https://www.google.com/search?q={search_url}"
                    )

        # Add source-specific discovery
        if not request.source_types or "academic" in request.source_types:
            discovered_urls.extend(
                [
                    f"https://arxiv.org/search/?query={request.query}&searchtype=all",
                    f"https://pubmed.ncbi.nlm.nih.gov/?term={request.query}",
                    f"https://scholar.google.com/scholar?q={request.query}",
                ]
            )

        if not request.source_types or "reference" in request.source_types:
            discovered_urls.extend(
                [
                    f"https://en.wikipedia.org/w/api.php?action=opensearch&search={request.query}&limit=5&format=json",
                    f"https://www.britannica.com/search?query={request.query}",
                ]
            )

        if not request.source_types or "technical" in request.source_types:
            discovered_urls.extend(
                [
                    f"https://stackoverflow.com/search?q={request.query}",
                    f"https://github.com/search?q={request.query}&type=repositories",
                ]
            )

        # Limit to max_sources
        return discovered_urls[: request.max_sources * 2]  # Get extra for filtering


class ContentAggregator:
    """Aggregates and synthesizes content from multiple sources."""

    def __init__(self, scraper: UnifiedWebScraper):
        self.scraper = scraper
        self.source_discovery = SourceDiscovery()

    def _calculate_content_confidence(self, sources: List[ScrapingResult]) -> float:
        """Calculate confidence score based on source quality and agreement."""
        if not sources:
            return 0.0

        total_weight = 0.0
        weighted_quality = 0.0

        for source in sources:
            if not source.success or not source.content:
                continue

            # Get source priority
            priority = self.source_discovery.get_source_priority(source.url)

            # Calculate content quality
            content_quality = source.content.get("quality_score", 0.5)

            # Combine priority and content quality
            source_weight = (
                (priority.priority / 10.0) * priority.reliability * content_quality
            )

            total_weight += source_weight
            weighted_quality += source_weight * content_quality

        if total_weight == 0:
            return 0.0

        confidence = weighted_quality / total_weight

        # Bonus for multiple sources agreeing
        if len([s for s in sources if s.success]) > 1:
            confidence *= 1.1

        return min(confidence, 1.0)

    def _extract_primary_content(self, sources: List[ScrapingResult]) -> Optional[Dict]:
        """Extract the highest quality content as primary source."""
        if not sources:
            return None

        # Score each source
        scored_sources = []
        for source in sources:
            if not source.success or not source.content:
                continue

            priority = self.source_discovery.get_source_priority(source.url)
            content_quality = source.content.get("quality_score", 0.5)

            # Calculate combined score
            score = (priority.priority / 10.0) * priority.reliability * content_quality
            scored_sources.append((score, source))

        if not scored_sources:
            return None

        # Sort by score and return best
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        best_source = scored_sources[0][1]

        primary_content = best_source.content.copy()
        primary_content["source_url"] = best_source.url
        primary_content["source_priority"] = self.source_discovery.get_source_priority(
            best_source.url
        ).priority

        return primary_content

    def _aggregate_keywords(self, sources: List[ScrapingResult]) -> Set[str]:
        """Aggregate keywords from all sources."""
        all_keywords = set()

        for source in sources:
            if source.success and source.content:
                keywords = source.content.get("keywords", [])
                if keywords:
                    all_keywords.update(keywords)

        return all_keywords

    def _aggregate_citations(self, sources: List[ScrapingResult]) -> List[str]:
        """Aggregate citations from all sources."""
        all_citations = []
        seen_citations = set()

        for source in sources:
            if source.success and source.content:
                citations = source.content.get("citations", [])
                if citations:
                    for citation in citations:
                        if citation and citation not in seen_citations:
                            all_citations.append(citation)
                            seen_citations.add(citation)

        return all_citations

    async def acquire_content(self, request: ContentRequest) -> AggregatedContent:
        """Acquire and aggregate content from multiple sources."""
        start_time = time.time()

        logger.info(
            f"🔍 Starting multi-source content acquisition for: '{request.query}'"
        )

        # Discover potential sources
        discovered_urls = await self.source_discovery.discover_sources(request)
        logger.info(f"   📋 Discovered {len(discovered_urls)} potential sources")

        # Filter and prioritize URLs
        prioritized_urls = []
        for url in discovered_urls:
            priority = self.source_discovery.get_source_priority(url)
            if priority.priority >= 5:  # Only include medium+ priority sources
                prioritized_urls.append((priority.priority, url))

        # Sort by priority and limit
        prioritized_urls.sort(key=lambda x: x[0], reverse=True)
        final_urls = [url for _, url in prioritized_urls[: request.max_sources]]

        logger.info(f"   🎯 Selected {len(final_urls)} high-priority sources")

        # Scrape content from selected sources
        scraping_tasks = []
        for url in final_urls:
            task = self.scraper.scrape_url(url)
            scraping_tasks.append(task)

        # Execute scraping concurrently
        if scraping_tasks:
            sources = await asyncio.gather(*scraping_tasks, return_exceptions=True)

            # Filter out exceptions
            valid_sources = []
            for source in sources:
                if isinstance(source, ScrapingResult):
                    valid_sources.append(source)
                else:
                    logger.warning(f"Scraping task failed: {source}")
        else:
            valid_sources = []

        # Filter by quality threshold
        quality_sources = []
        for source in valid_sources:
            if source.success and source.content:
                quality_score = source.content.get("quality_score", 0.0)
                if quality_score >= request.min_quality_score:
                    quality_sources.append(source)

        logger.info(
            f"   ✅ Successfully acquired {len(quality_sources)} quality sources"
        )

        # Aggregate content
        primary_content = self._extract_primary_content(quality_sources)
        supporting_content = []

        for source in quality_sources:
            if source.content != primary_content:
                support_content = {
                    "title": source.content.get("title"),
                    "summary": source.content.get("main_text", "")[
                        :500
                    ],  # First 500 chars
                    "source_url": source.url,
                    "quality_score": source.content.get("quality_score", 0.0),
                }
                supporting_content.append(support_content)

        # Create aggregated result
        aggregated = AggregatedContent(
            query=request.query,
            sources=quality_sources,
            primary_content=primary_content,
            supporting_content=supporting_content,
            citations=self._aggregate_citations(quality_sources),
            keywords=self._aggregate_keywords(quality_sources),
            confidence_score=self._calculate_content_confidence(quality_sources),
            acquisition_time=time.time() - start_time,
            metadata={
                "total_sources_discovered": len(discovered_urls),
                "sources_scraped": len(valid_sources),
                "quality_sources": len(quality_sources),
                "average_processing_time": (
                    sum(s.processing_time for s in quality_sources)
                    / len(quality_sources)
                    if quality_sources
                    else 0.0
                ),
            },
        )

        logger.info(
            f"🎉 Content acquisition complete in {aggregated.acquisition_time:.2f}s"
        )
        logger.info(f"   📊 Confidence score: {aggregated.confidence_score:.2f}")
        logger.info(f"   🔗 Found {len(aggregated.citations)} citations")
        logger.info(f"   🏷️ Found {len(aggregated.keywords)} keywords")

        return aggregated


class MultiSourceAcquisitionSystem:
    """Main interface for multi-source content acquisition."""

    def __init__(self, scraper_config: Optional[ScraperConfig] = None):
        self.scraper = UnifiedWebScraper(scraper_config)
        self.aggregator = ContentAggregator(self.scraper)

        logger.info("🚀 Multi-Source Acquisition System initialized")

    async def acquire(self, query: str, **kwargs) -> AggregatedContent:
        """Simplified interface for content acquisition."""
        request = ContentRequest(query=query, **kwargs)
        return await self.aggregator.acquire_content(request)

    async def acquire_academic(
        self, query: str, max_sources: int = 3
    ) -> AggregatedContent:
        """Acquire content focused on academic sources."""
        request = ContentRequest(
            query=query,
            source_types=["academic"],
            max_sources=max_sources,
            min_quality_score=0.7,
            include_citations=True,
        )
        return await self.aggregator.acquire_content(request)

    async def acquire_reference(
        self, query: str, max_sources: int = 5
    ) -> AggregatedContent:
        """Acquire content from reference sources (Wikipedia, encyclopedias)."""
        request = ContentRequest(
            query=query,
            source_types=["reference"],
            max_sources=max_sources,
            min_quality_score=0.6,
        )
        return await self.aggregator.acquire_content(request)

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return self.scraper.get_stats()


# Global instance
multi_source_system = MultiSourceAcquisitionSystem()

if __name__ == "__main__":
    # Test the multi-source acquisition system
    async def test_multi_source():
        system = MultiSourceAcquisitionSystem()

        # Test academic content acquisition
        result = await system.acquire_academic(
            "machine learning algorithms", max_sources=2
        )

        print(f"Query: {result.query}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Sources: {len(result.sources)}")
        if result.primary_content:
            print(f"Primary title: {result.primary_content.get('title', 'N/A')}")
        print(f"Keywords: {list(result.keywords)[:5]}")  # First 5 keywords
        print(f"Citations: {len(result.citations)}")
        print(f"Acquisition time: {result.acquisition_time:.2f}s")

    asyncio.run(test_multi_source())
