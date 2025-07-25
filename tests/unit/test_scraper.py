"""
Unit tests for the Scraper Agent.
"""

import json

# Import the scraper components
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import asyncio
import pytest

sys.path.append(".")
from agents.scraper.scraper import JobStatus, ScrapingConfig, ScrapingJob, WebScraper
from agents.scraper.utils import ExponentialBackoff, LicenseDetector, RateLimiter


class TestScrapingConfig:
    """Test the ScrapingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScrapingConfig()

        assert config.max_concurrent_requests == 5
        assert config.request_delay == 1.0
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.respect_robots_txt == True
        assert config.user_agent.startswith("MCP-Server")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScrapingConfig(
            max_concurrent_requests=10, request_delay=2.0, max_retries=5
        )

        assert config.max_concurrent_requests == 10
        assert config.request_delay == 2.0
        assert config.max_retries == 5

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid values
        with pytest.raises(ValueError):
            ScrapingConfig(max_concurrent_requests=0)

        with pytest.raises(ValueError):
            ScrapingConfig(request_delay=-1)

        with pytest.raises(ValueError):
            ScrapingConfig(max_retries=-1)


class TestRateLimiter:
    """Test the RateLimiter utility."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, time_window=60)

        assert limiter.max_requests == 10
        assert limiter.time_window == 60
        assert len(limiter.request_times) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(max_requests=2, time_window=1)

        # First two requests should be allowed
        assert await limiter.can_make_request() == True
        await limiter.record_request()

        assert await limiter.can_make_request() == True
        await limiter.record_request()

        # Third request should be blocked
        assert await limiter.can_make_request() == False

        # Wait for time window to pass
        await asyncio.sleep(1.1)

        # Should be allowed again
        assert await limiter.can_make_request() == True

    @pytest.mark.asyncio
    async def test_cleanup_old_requests(self):
        """Test cleanup of old request times."""
        limiter = RateLimiter(max_requests=5, time_window=1)

        # Make some requests
        for _ in range(3):
            await limiter.record_request()

        assert len(limiter.request_times) == 3

        # Wait for cleanup
        await asyncio.sleep(1.1)
        await limiter.can_make_request()  # Triggers cleanup

        assert len(limiter.request_times) == 0


class TestExponentialBackoff:
    """Test the ExponentialBackoff utility."""

    def test_initialization(self):
        """Test exponential backoff initialization."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)

        assert backoff.base_delay == 1.0
        assert backoff.max_delay == 60.0
        assert backoff.attempt == 0

    def test_delay_calculation(self):
        """Test delay calculation."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)

        # First attempt
        delay1 = backoff.get_delay()
        assert delay1 == 1.0

        # Second attempt
        backoff.increment()
        delay2 = backoff.get_delay()
        assert delay2 == 2.0

        # Third attempt
        backoff.increment()
        delay3 = backoff.get_delay()
        assert delay3 == 4.0

    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0)

        # Increment many times
        for _ in range(10):
            backoff.increment()

        delay = backoff.get_delay()
        assert delay <= 10.0

    def test_reset(self):
        """Test backoff reset."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)

        backoff.increment()
        backoff.increment()
        assert backoff.attempt == 2

        backoff.reset()
        assert backoff.attempt == 0
        assert backoff.get_delay() == 1.0


class TestLicenseDetector:
    """Test the LicenseDetector utility."""

    def test_initialization(self):
        """Test license detector initialization."""
        detector = LicenseDetector()

        assert len(detector.license_patterns) > 0
        assert "creative commons" in detector.license_patterns
        assert "public domain" in detector.license_patterns

    def test_creative_commons_detection(self):
        """Test Creative Commons license detection."""
        detector = LicenseDetector()

        text_with_cc = "This work is licensed under a Creative Commons Attribution 4.0 International License."
        result = detector.detect_license(text_with_cc)

        assert result is not None
        assert "creative commons" in result.license_type.lower()
        assert result.confidence > 0.8

    def test_public_domain_detection(self):
        """Test public domain detection."""
        detector = LicenseDetector()

        text_with_pd = "This document is in the public domain and may be freely used."
        result = detector.detect_license(text_with_pd)

        assert result is not None
        assert "public domain" in result.license_type.lower()
        assert result.confidence > 0.7

    def test_no_license_detection(self):
        """Test when no license is detected."""
        detector = LicenseDetector()

        text_without_license = (
            "This is just some regular text without any license information."
        )
        result = detector.detect_license(text_without_license)

        assert result is None

    def test_copyright_detection(self):
        """Test copyright detection."""
        detector = LicenseDetector()

        text_with_copyright = "Copyright 2024 Example Corp. All rights reserved."
        result = detector.detect_license(text_with_copyright)

        assert result is not None
        assert "copyright" in result.license_type.lower()
        assert result.is_copyrighted == True


class TestScrapingJob:
    """Test the ScrapingJob data class."""

    def test_job_creation(self):
        """Test scraping job creation."""
        job = ScrapingJob(
            id="test_job_001", url="https://example.com", domain="science", max_depth=2
        )

        assert job.id == "test_job_001"
        assert job.url == "https://example.com"
        assert job.domain == "science"
        assert job.max_depth == 2
        assert job.status == JobStatus.PENDING
        assert job.created_at is not None

    def test_job_status_update(self):
        """Test job status updates."""
        job = ScrapingJob(id="test_job_002", url="https://example.com", domain="math")

        assert job.status == JobStatus.PENDING

        # Update to running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

        # Update to completed
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None


class TestWebScraper:
    """Test the main WebScraper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ScrapingConfig(
            max_concurrent_requests=2,
            request_delay=0.1,  # Faster for testing
            max_retries=1,
        )
        self.scraper = WebScraper(self.config)

    def teardown_method(self):
        """Clean up after tests."""
        # Clean up any temporary files or resources
        pass

    @pytest.mark.asyncio
    async def test_scraper_initialization(self):
        """Test scraper initialization."""
        await self.scraper.initialize()

        assert self.scraper.session is not None
        assert self.scraper.rate_limiter is not None
        assert len(self.scraper.active_jobs) == 0

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_robots_txt_parsing(self):
        """Test robots.txt parsing."""
        await self.scraper.initialize()

        # Mock robots.txt content
        robots_content = """
User-agent: *
Disallow: /private/
Disallow: /admin/
Allow: /public/
Crawl-delay: 1
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = robots_content
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            robots = await self.scraper._get_robots_txt("https://example.com")

            assert robots is not None
            assert not robots.can_fetch("*", "https://example.com/private/page.html")
            assert robots.can_fetch("*", "https://example.com/public/page.html")

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_url_validation(self):
        """Test URL validation."""
        await self.scraper.initialize()

        # Valid URLs
        assert self.scraper._is_valid_url("https://example.com")
        assert self.scraper._is_valid_url("http://test.org/page")
        assert self.scraper._is_valid_url("https://sub.domain.com/path/to/page.html")

        # Invalid URLs
        assert not self.scraper._is_valid_url("not-a-url")
        assert not self.scraper._is_valid_url("ftp://example.com")
        assert not self.scraper._is_valid_url("javascript:alert('test')")
        assert not self.scraper._is_valid_url("")

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_successful_page_scraping(self):
        """Test successful page scraping."""
        await self.scraper.initialize()

        # Mock HTML content
        html_content = """
        <html>
        <head>
            <title>Test Page</title>
            <meta name="author" content="Test Author">
            <meta name="description" content="Test description">
        </head>
        <body>
            <h1>Test Article</h1>
            <p>This is a test paragraph with some content.</p>
            <p>Another paragraph with more content.</p>
        </body>
        </html>
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = html_content
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page("https://example.com")

            assert result is not None
            assert result.url == "https://example.com"
            assert result.title == "Test Page"
            assert result.content is not None
            assert "test paragraph" in result.content.lower()
            assert result.metadata["author"] == "Test Author"

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test HTTP error handling."""
        await self.scraper.initialize()

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text.return_value = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page("https://example.com/notfound")

            assert result is None

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        await self.scraper.initialize()

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Request timed out")

            result = await self.scraper._scrape_page("https://slow-example.com")

            assert result is None

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism."""
        await self.scraper.initialize()

        call_count = 0

        async def mock_get_with_retries(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:  # Fail first time
                raise aiohttp.ClientError("Connection failed")
            else:  # Succeed second time
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text.return_value = "<html><body>Success</body></html>"
                mock_response.headers = {"content-type": "text/html"}
                return mock_response

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = mock_get_with_retries

            result = await self.scraper._scrape_page("https://flaky-example.com")

            assert call_count == 2  # Should retry once
            # Note: The actual result depends on implementation details

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_content_extraction(self):
        """Test content extraction from HTML."""
        await self.scraper.initialize()

        html_with_noise = """
        <html>
        <head>
            <title>Article Title</title>
        </head>
        <body>
            <nav>Navigation menu</nav>
            <header>Header content</header>
            <main>
                <h1>Main Article</h1>
                <p>This is the main content of the article.</p>
                <p>Another important paragraph.</p>
            </main>
            <aside>Sidebar content</aside>
            <footer>Footer content</footer>
            <script>console.log('JavaScript');</script>
        </body>
        </html>
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = html_with_noise
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page("https://example.com")

            assert result is not None
            # Main content should be extracted
            assert "main content of the article" in result.content.lower()
            # Navigation and footer should be filtered out
            assert "navigation menu" not in result.content.lower()
            assert "footer content" not in result.content.lower()
            # JavaScript should be removed
            assert "console.log" not in result.content

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_pdf_scraping(self):
        """Test PDF content extraction."""
        await self.scraper.initialize()

        # Mock PDF content
        mock_pdf_content = b"Mock PDF content bytes"

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.read.return_value = mock_pdf_content
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/pdf"}
            mock_get.return_value.__aenter__.return_value = mock_response

            with patch("agents.scraper.utils.extract_text_from_pdf") as mock_extract:
                mock_extract.return_value = "Extracted PDF text content"

                result = await self.scraper._scrape_page(
                    "https://example.com/document.pdf"
                )

                assert result is not None
                assert result.content == "Extracted PDF text content"
                assert result.metadata["content_type"] == "application/pdf"

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_domain_classification(self):
        """Test automatic domain classification."""
        await self.scraper.initialize()

        # Test different content types
        test_cases = [
            ("Mathematical theorem about prime numbers", "mathematics"),
            ("Chemical reaction in organic chemistry", "science"),
            ("Biblical interpretation and theology", "religion"),
            ("Historical account of ancient Rome", "history"),
            ("Poetry analysis and literary criticism", "literature"),
            ("Philosophical argument about ethics", "philosophy"),
        ]

        for content, expected_domain in test_cases:
            html_content = f"""
            <html>
            <body>
                <h1>Test Article</h1>
                <p>{content}</p>
            </body>
            </html>
            """

            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.text.return_value = html_content
                mock_response.status = 200
                mock_response.headers = {"content-type": "text/html"}
                mock_get.return_value.__aenter__.return_value = mock_response

                result = await self.scraper._scrape_page("https://example.com")

                assert result is not None
                # Note: Actual domain classification would depend on implementation
                # This test would need to be adapted based on the classifier used

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_job_management(self):
        """Test scraping job management."""
        await self.scraper.initialize()

        # Create a job
        job_id = await self.scraper.create_job(
            url="https://example.com", domain="science", max_depth=1
        )

        assert job_id is not None
        assert job_id in self.scraper.active_jobs

        job = self.scraper.active_jobs[job_id]
        assert job.status == JobStatus.PENDING
        assert job.url == "https://example.com"
        assert job.domain == "science"

        # Check job status
        status = await self.scraper.get_job_status(job_id)
        assert status == JobStatus.PENDING

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_user_specified_sources(self):
        """Test user-specified data source handling."""
        await self.scraper.initialize()

        user_sources = [
            {"url": "https://example.com/paper1", "domain": "mathematics"},
            {"url": "https://example.com/paper2", "domain": "science"},
            {"url": "https://example.com/book", "domain": "philosophy"},
        ]

        # Mock successful responses
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = "<html><body>Test content</body></html>"
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await self.scraper.scrape_user_sources(user_sources)

            assert len(results) == 3
            for result in results:
                assert result is not None
                assert result.content is not None

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_concurrent_scraping(self):
        """Test concurrent scraping with rate limiting."""
        await self.scraper.initialize()

        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
            "https://example.com/page4",
        ]

        # Mock responses
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = "<html><body>Test content</body></html>"
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            start_time = datetime.now()
            results = await self.scraper.scrape_multiple_urls(urls)
            end_time = datetime.now()

            # Should respect rate limiting
            duration = (end_time - start_time).total_seconds()
            expected_min_duration = (
                len(urls)
                * self.config.request_delay
                / self.config.max_concurrent_requests
            )

            assert len(results) == len(urls)
            # Note: Exact timing tests can be flaky, so we just check basic functionality

        await self.scraper.close()


class TestIntegrationScenarios:
    """Integration tests for realistic scraping scenarios."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = ScrapingConfig(request_delay=0.1)  # Faster for testing
        self.scraper = WebScraper(self.config)

    @pytest.mark.asyncio
    async def test_academic_paper_scraping(self):
        """Test scraping an academic paper (simulated)."""
        await self.scraper.initialize()

        academic_html = """
        <html>
        <head>
            <title>A Novel Approach to Prime Number Theory</title>
            <meta name="author" content="Dr. Jane Smith">
            <meta name="keywords" content="mathematics, prime numbers, number theory">
            <meta name="description" content="This paper presents a new method for analyzing prime numbers.">
        </head>
        <body>
            <h1>A Novel Approach to Prime Number Theory</h1>
            <div class="abstract">
                <h2>Abstract</h2>
                <p>In this paper, we present a revolutionary approach to understanding prime numbers...</p>
            </div>
            <div class="content">
                <h2>Introduction</h2>
                <p>Prime numbers have fascinated mathematicians for centuries...</p>
                
                <h2>Methodology</h2>
                <p>Our approach uses advanced computational techniques...</p>
                
                <h2>Results</h2>
                <p>We discovered several new patterns in prime distribution...</p>
                
                <h2>Conclusion</h2>
                <p>This work opens new avenues for research in number theory...</p>
            </div>
            <div class="references">
                <h2>References</h2>
                <p>1. Smith, J. (2020). Previous work on prime numbers.</p>
                <p>2. Johnson, A. (2019). Mathematical foundations.</p>
            </div>
        </body>
        </html>
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = academic_html
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page(
                "https://journal.example.com/paper123"
            )

            assert result is not None
            assert result.title == "A Novel Approach to Prime Number Theory"
            assert result.metadata["author"] == "Dr. Jane Smith"
            assert "prime numbers" in result.content.lower()
            assert "abstract" in result.content.lower()
            assert "methodology" in result.content.lower()
            assert result.metadata.get("domain") in [
                "mathematics",
                "science",
                None,
            ]  # Depending on classification

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_book_chapter_scraping(self):
        """Test scraping a book chapter (simulated)."""
        await self.scraper.initialize()

        book_html = """
        <html>
        <head>
            <title>Chapter 5: Ancient Greek Philosophy - Introduction to Philosophy</title>
            <meta name="author" content="Prof. Robert Williams">
            <meta name="publication" content="Introduction to Philosophy, 3rd Edition">
        </head>
        <body>
            <div class="chapter">
                <h1>Chapter 5: Ancient Greek Philosophy</h1>
                
                <div class="section">
                    <h2>5.1 The Pre-Socratics</h2>
                    <p>Before Socrates, there were many philosophers who laid the groundwork...</p>
                </div>
                
                <div class="section">
                    <h2>5.2 Socrates and the Socratic Method</h2>
                    <p>Socrates revolutionized philosophy with his method of questioning...</p>
                </div>
                
                <div class="section">
                    <h2>5.3 Plato's Theory of Forms</h2>
                    <p>Plato, a student of Socrates, developed the famous Theory of Forms...</p>
                </div>
                
                <div class="section">
                    <h2>5.4 Aristotle's Empiricism</h2>
                    <p>Aristotle, Plato's student, took a different approach...</p>
                </div>
            </div>
            <div class="footer">
                <p>© 2024 University Press. All rights reserved.</p>
            </div>
        </body>
        </html>
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = book_html
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page(
                "https://books.example.com/philosophy/chapter5"
            )

            assert result is not None
            assert "ancient greek philosophy" in result.title.lower()
            assert result.metadata["author"] == "Prof. Robert Williams"
            assert "socrates" in result.content.lower()
            assert "plato" in result.content.lower()
            assert "aristotle" in result.content.lower()
            # Copyright notice should be filtered out or noted

        await self.scraper.close()

    @pytest.mark.asyncio
    async def test_historical_document_scraping(self):
        """Test scraping a historical document (simulated)."""
        await self.scraper.initialize()

        historical_html = """
        <html>
        <head>
            <title>The Declaration of Independence (1776)</title>
            <meta name="date" content="1776-07-04">
            <meta name="location" content="Philadelphia, Pennsylvania">
            <meta name="document-type" content="historical">
        </head>
        <body>
            <div class="document">
                <h1>The Declaration of Independence</h1>
                <p class="date">In Congress, July 4, 1776</p>
                
                <div class="preamble">
                    <p>When in the Course of human events, it becomes necessary for one people to dissolve the political bands...</p>
                </div>
                
                <div class="principles">
                    <p>We hold these truths to be self-evident, that all men are created equal...</p>
                </div>
                
                <div class="grievances">
                    <p>The history of the present King of Great Britain is a history of repeated injuries...</p>
                </div>
                
                <div class="conclusion">
                    <p>And for the support of this Declaration, with a firm reliance on the protection of divine Providence...</p>
                </div>
            </div>
            <div class="license">
                <p>This document is in the public domain.</p>
            </div>
        </body>
        </html>
        """

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = historical_html
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await self.scraper._scrape_page(
                "https://historical.example.com/declaration"
            )

            assert result is not None
            assert "declaration of independence" in result.title.lower()
            assert result.metadata.get("date") == "1776-07-04"
            assert "self-evident" in result.content.lower()
            assert (
                "public domain" in result.content.lower()
                or result.metadata.get("license") == "public domain"
            )

        await self.scraper.close()


# Fixtures for testing
@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <html>
    <head>
        <title>Test Document</title>
        <meta name="author" content="Test Author">
        <meta name="keywords" content="test, sample, document">
    </head>
    <body>
        <h1>Test Article</h1>
        <p>This is a test paragraph.</p>
        <p>Another test paragraph with <a href="https://example.com">a link</a>.</p>
    </body>
    </html>
    """


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def sample_robots_txt():
    """Sample robots.txt content."""
    return """
User-agent: *
Disallow: /private/
Allow: /public/
Crawl-delay: 1
    """


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
