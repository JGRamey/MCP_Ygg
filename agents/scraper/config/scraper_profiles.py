#!/usr/bin/env python3
"""
Scraper Profiles for MCP Yggdrasil
Configurable scraping profiles for different use cases
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Enhanced scraper configuration with profile support."""

    # User agents and rotation
    user_agents: List[str] = field(default_factory=list)
    use_fake_useragent: bool = True

    # Proxies and rotation
    proxies: List[str] = field(default_factory=list)
    use_proxy_rotation: bool = False

    # Rate limiting and delays
    requests_per_second: float = 1.0
    request_delay_range: tuple = (1, 3)
    respect_robots_txt: bool = True

    # Retry configuration
    retry_attempts: int = 3
    retry_delay: float = 5.0
    backoff_factor: float = 2.0

    # Timeout settings
    connection_timeout: int = 30
    read_timeout: int = 30
    total_timeout: int = 60

    # Selenium configuration
    use_selenium_fallback: bool = True
    headless_browser: bool = True
    browser_binary_path: Optional[str] = None
    use_stealth_mode: bool = True

    # Content extraction preferences
    use_trafilatura: bool = True
    extract_comments: bool = False
    extract_images: bool = True
    extract_links: bool = True
    extract_structured_data: bool = True

    # Language detection
    detect_language: bool = True
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    detect_mixed_languages: bool = False

    # Quality and validation
    min_content_length: int = 100
    verify_content: bool = True
    quality_threshold: float = 0.6

    # Caching
    cache_responses: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Output options
    save_raw_html: bool = False
    save_screenshots: bool = False
    screenshot_path: str = "./screenshots"

    # Site-specific options
    use_site_specific_parsers: bool = True

    def __post_init__(self):
        """Initialize with defaults if not provided."""
        if not self.user_agents:
            self.user_agents = self._get_default_user_agents()

        if not self.proxies:
            self.proxies = self._load_proxies_from_env()

        if not self.target_languages:
            self.target_languages = ["en", "es", "fr", "de", "it", "pt"]

    def _get_default_user_agents(self) -> List[str]:
        """Get default user agents (13 total as specified)."""
        return [
            # Chrome variants (Windows, Mac, Linux)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            # Firefox variants
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/120.0",
            # Safari variants
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
            # Edge variants
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            # Mobile variants
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        ]

    def _load_proxies_from_env(self) -> List[str]:
        """Load proxies from environment variables."""
        proxy_list = []

        # Check for proxy environment variables
        if os.getenv("HTTP_PROXY"):
            proxy_list.append(os.getenv("HTTP_PROXY"))

        if os.getenv("HTTPS_PROXY"):
            proxy_list.append(os.getenv("HTTPS_PROXY"))

        if os.getenv("PROXY_LIST"):
            # Expect comma-separated list
            proxy_list.extend(
                [p.strip() for p in os.getenv("PROXY_LIST").split(",") if p.strip()]
            )

        return proxy_list

    def to_dict(self) -> Dict:
        """Convert config to dictionary for compatibility."""
        return {
            "user_agents": self.user_agents,
            "use_fake_useragent": self.use_fake_useragent,
            "proxies": self.proxies,
            "use_proxy_rotation": self.use_proxy_rotation,
            "requests_per_second": self.requests_per_second,
            "request_delay_range": self.request_delay_range,
            "respect_robots_txt": self.respect_robots_txt,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "connection_timeout": self.connection_timeout,
            "read_timeout": self.read_timeout,
            "use_selenium_fallback": self.use_selenium_fallback,
            "headless_browser": self.headless_browser,
            "use_stealth_mode": self.use_stealth_mode,
            "use_trafilatura": self.use_trafilatura,
            "extract_images": self.extract_images,
            "extract_links": self.extract_links,
            "extract_structured_data": self.extract_structured_data,
            "detect_language": self.detect_language,
            "target_languages": self.target_languages,
            "min_content_length": self.min_content_length,
            "verify_content": self.verify_content,
            "cache_responses": self.cache_responses,
            "cache_ttl": self.cache_ttl,
            "save_raw_html": self.save_raw_html,
            "save_screenshots": self.save_screenshots,
            "use_site_specific_parsers": self.use_site_specific_parsers,
        }


# Pre-configured scraping profiles as specified in the documentation
SCRAPER_PROFILES = {
    "fast": ScraperConfig(
        # Fast profile: Optimized for speed
        requests_per_second=2.0,
        request_delay_range=(0.5, 1.0),
        retry_attempts=1,
        connection_timeout=15,
        read_timeout=15,
        total_timeout=30,
        use_selenium_fallback=False,
        extract_images=False,
        extract_links=False,
        extract_comments=False,
        extract_structured_data=False,
        detect_language=False,
        verify_content=False,
        save_raw_html=False,
        cache_ttl=1800,  # 30 minutes
        min_content_length=50,
    ),
    "comprehensive": ScraperConfig(
        # Comprehensive profile: Maximum data extraction
        requests_per_second=0.5,
        request_delay_range=(2, 4),
        retry_attempts=3,
        connection_timeout=45,
        read_timeout=45,
        total_timeout=120,
        use_selenium_fallback=True,
        use_stealth_mode=True,
        extract_images=True,
        extract_links=True,
        extract_comments=True,
        extract_structured_data=True,
        detect_language=True,
        detect_mixed_languages=True,
        verify_content=True,
        save_raw_html=True,
        save_screenshots=True,
        cache_ttl=7200,  # 2 hours
        quality_threshold=0.7,
        target_languages=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"],
    ),
    "stealth": ScraperConfig(
        # Stealth profile: Maximum anti-detection
        requests_per_second=0.2,
        request_delay_range=(3, 8),
        retry_attempts=5,
        backoff_factor=3.0,
        connection_timeout=60,
        read_timeout=60,
        total_timeout=180,
        use_proxy_rotation=True,
        use_selenium_fallback=True,
        use_stealth_mode=True,
        headless_browser=False,  # More human-like
        extract_images=True,
        extract_links=True,
        extract_structured_data=True,
        detect_language=True,
        verify_content=True,
        save_screenshots=True,
        cache_ttl=3600,  # 1 hour
        respect_robots_txt=True,
        quality_threshold=0.8,
    ),
    "academic": ScraperConfig(
        # Academic profile: Optimized for scholarly content
        requests_per_second=0.5,
        request_delay_range=(1, 3),
        retry_attempts=3,
        connection_timeout=60,
        read_timeout=60,
        total_timeout=150,
        respect_robots_txt=True,
        use_selenium_fallback=True,
        extract_comments=False,  # Focus on main content
        extract_images=False,  # Text-focused
        extract_links=True,  # Important for citations
        extract_structured_data=True,  # Rich metadata
        detect_language=True,
        verify_content=True,
        save_raw_html=True,  # For archival
        cache_ttl=14400,  # 4 hours - academic content is stable
        target_languages=["en"],  # Primarily English academic content
        quality_threshold=0.8,  # High quality threshold
        min_content_length=200,  # Substantial content
        use_site_specific_parsers=True,  # Enhanced parsing for academic sites
    ),
    "news": ScraperConfig(
        # News profile: Optimized for news and media sites
        requests_per_second=1.0,
        request_delay_range=(1, 2),
        retry_attempts=2,
        connection_timeout=30,
        read_timeout=30,
        total_timeout=90,
        use_selenium_fallback=True,  # Many news sites use JS
        extract_images=True,  # News images important
        extract_links=True,  # Related articles
        extract_structured_data=True,  # Rich snippets
        detect_language=True,
        verify_content=True,
        cache_ttl=1800,  # 30 minutes - news changes quickly
        target_languages=["en", "es", "fr", "de"],
        quality_threshold=0.6,
        min_content_length=100,
    ),
    "social": ScraperConfig(
        # Social media profile: Optimized for social platforms
        requests_per_second=0.3,
        request_delay_range=(2, 5),
        retry_attempts=3,
        use_selenium_fallback=True,  # Required for most social platforms
        use_stealth_mode=True,  # High anti-detection
        headless_browser=False,  # More human-like
        extract_images=False,  # Focus on text content
        extract_links=True,
        extract_structured_data=False,  # Limited structured data
        detect_language=True,
        detect_mixed_languages=True,  # Social content often mixed
        cache_ttl=900,  # 15 minutes - very dynamic content
        target_languages=["en", "es", "fr", "de", "pt", "it"],
        quality_threshold=0.4,  # Lower threshold for social content
        min_content_length=20,  # Short posts acceptable
    ),
}


def get_profile(profile_name: str) -> ScraperConfig:
    """Get a scraper profile by name."""
    if profile_name not in SCRAPER_PROFILES:
        logger.warning(f"Profile '{profile_name}' not found, using 'comprehensive'")
        return SCRAPER_PROFILES["comprehensive"]

    return SCRAPER_PROFILES[profile_name]


def list_profiles() -> List[str]:
    """List available scraper profiles."""
    return list(SCRAPER_PROFILES.keys())


def get_profile_info(profile_name: str) -> Dict:
    """Get information about a specific profile."""
    if profile_name not in SCRAPER_PROFILES:
        return {}

    config = SCRAPER_PROFILES[profile_name]

    return {
        "name": profile_name,
        "requests_per_second": config.requests_per_second,
        "use_selenium": config.use_selenium_fallback,
        "stealth_mode": config.use_stealth_mode,
        "extract_structured_data": config.extract_structured_data,
        "quality_threshold": config.quality_threshold,
        "cache_ttl": config.cache_ttl,
        "target_languages": config.target_languages,
        "description": _get_profile_description(profile_name),
    }


def _get_profile_description(profile_name: str) -> str:
    """Get description for a profile."""
    descriptions = {
        "fast": "Optimized for speed with minimal extraction",
        "comprehensive": "Maximum data extraction with all features enabled",
        "stealth": "Maximum anti-detection with slow, careful scraping",
        "academic": "Optimized for scholarly content with high quality standards",
        "news": "Optimized for news and media sites with balanced speed/quality",
        "social": "Optimized for social media platforms with stealth features",
    }
    return descriptions.get(profile_name, "Custom profile")


def create_custom_profile(
    base_profile: str = "comprehensive", **overrides
) -> ScraperConfig:
    """Create a custom profile based on an existing one."""
    if base_profile not in SCRAPER_PROFILES:
        base_profile = "comprehensive"

    base_config = SCRAPER_PROFILES[base_profile]

    # Create new config with overrides
    config_dict = base_config.to_dict()
    config_dict.update(overrides)

    # Convert back to ScraperConfig
    return ScraperConfig(
        **{k: v for k, v in config_dict.items() if hasattr(ScraperConfig, k)}
    )


if __name__ == "__main__":
    # Test the profiles
    print("Available Scraper Profiles:")
    print("=" * 40)

    for profile_name in list_profiles():
        info = get_profile_info(profile_name)
        print(f"\n{profile_name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Speed: {info['requests_per_second']} req/s")
        print(f"  Selenium: {info['use_selenium']}")
        print(f"  Stealth: {info['stealth_mode']}")
        print(f"  Quality threshold: {info['quality_threshold']}")
        print(f"  Languages: {info['target_languages']}")

    # Test custom profile creation
    print("\n" + "=" * 40)
    print("Custom Profile Test:")
    custom = create_custom_profile(
        "academic", requests_per_second=1.5, quality_threshold=0.9
    )
    print(
        f"Custom academic profile: {custom.requests_per_second} req/s, quality: {custom.quality_threshold}"
    )
