#!/usr/bin/env python3
"""
Anti-Detection Manager for MCP Yggdrasil
Phase 3: Advanced anti-blocking and stealth scraping
"""

import random
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse

# Selenium imports with fallbacks
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Selenium stealth with fallback
try:
    from selenium_stealth import stealth
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

# Fake user agent with fallback
try:
    from fake_useragent import UserAgent
    FAKE_UA_AVAILABLE = True
except ImportError:
    FAKE_UA_AVAILABLE = False

logger = logging.getLogger(__name__)

class AntiDetectionManager:
    """Advanced anti-detection manager for web scraping."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize fake user agent if available
        self.ua = UserAgent() if FAKE_UA_AVAILABLE else None
        
        # Comprehensive real browser user agents
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            
            # Chrome on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            
            # Chrome on Linux
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # Firefox variants
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
            
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            
            # Mobile variants
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        ]
        
        # Proxy configuration
        self.proxies = config.get('proxies', [])
        self.current_proxy_index = 0
        
        # Header variations for natural browsing
        self.header_variations = {
            'Accept': [
                'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            ],
            'Accept-Language': [
                'en-US,en;q=0.9',
                'en-US,en;q=0.5',
                'en-GB,en;q=0.5',
                'en-US,en;q=0.9,es;q=0.8',
                'en-US,en;q=0.9,fr;q=0.8'
            ],
            'Accept-Encoding': [
                'gzip, deflate, br',
                'gzip, deflate',
                'gzip, deflate, br, zstd'
            ],
            'Cache-Control': [
                'max-age=0',
                'no-cache',
                'no-cache, no-store, must-revalidate'
            ]
        }
        
        # Sites that require special handling
        self.js_required_domains = [
            'twitter.com', 'x.com', 'instagram.com', 'facebook.com',
            'linkedin.com', 'medium.com', 'reddit.com', 'discord.com',
            'tiktok.com', 'youtube.com', 'spotify.com'
        ]
        
        # Academic sites (handle with more respect)
        self.academic_domains = [
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            'jstor.org', 'springer.com', 'elsevier.com', 'wiley.com',
            'nature.com', 'science.org', 'plato.stanford.edu'
        ]
        
        logger.info(f"✅ Anti-Detection Manager initialized")
        logger.info(f"   Selenium: {'Available' if SELENIUM_AVAILABLE else 'Not available'}")
        logger.info(f"   Stealth: {'Available' if STEALTH_AVAILABLE else 'Not available'}")
        logger.info(f"   Fake UserAgent: {'Available' if FAKE_UA_AVAILABLE else 'Not available'}")
        logger.info(f"   Proxies configured: {len(self.proxies)}")
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent with intelligent selection."""
        if self.ua and random.random() < 0.2:  # 20% use fake-useragent
            try:
                return self.ua.random
            except Exception:
                pass  # Fallback to predefined
        
        # 80% use predefined, more reliable user agents
        return random.choice(self.user_agents)
    
    def get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers that mimic real browser behavior."""
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': random.choice(self.header_variations['Accept']),
            'Accept-Language': random.choice(self.header_variations['Accept-Language']),
            'Accept-Encoding': random.choice(self.header_variations['Accept-Encoding']),
            'Cache-Control': random.choice(self.header_variations['Cache-Control']),
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
        
        # Randomly add optional headers
        if random.random() < 0.5:
            headers['DNT'] = '1'  # Do Not Track
        
        if random.random() < 0.3:
            headers['Connection'] = 'keep-alive'
        
        if random.random() < 0.4:
            headers['Pragma'] = 'no-cache'
        
        # Add referrer occasionally
        if random.random() < 0.2:
            headers['Referer'] = 'https://www.google.com/'
        
        return headers
    
    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy from rotation."""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        
        return {
            'http': proxy,
            'https': proxy
        }
    
    def should_use_selenium(self, url: str) -> bool:
        """Determine if Selenium is needed for a URL."""
        if not SELENIUM_AVAILABLE:
            return False
        
        domain = urlparse(url).netloc.lower()
        
        # Check JavaScript-heavy sites
        for js_domain in self.js_required_domains:
            if js_domain in domain:
                return True
        
        # Check explicit configuration
        selenium_urls = self.config.get('selenium_urls', [])
        if url in selenium_urls:
            return True
        
        # Check domain-specific rules
        selenium_domains = self.config.get('selenium_domains', [])
        for selenium_domain in selenium_domains:
            if selenium_domain in domain:
                return True
        
        return False
    
    def is_academic_site(self, url: str) -> bool:
        """Check if URL is from an academic site."""
        domain = urlparse(url).netloc.lower()
        return any(academic_domain in domain for academic_domain in self.academic_domains)
    
    def get_site_specific_delay(self, url: str) -> tuple:
        """Get site-specific delay range."""
        domain = urlparse(url).netloc.lower()
        
        # Academic sites - be more respectful
        if self.is_academic_site(url):
            return (3.0, 6.0)
        
        # Social media - needs more careful handling
        if any(social in domain for social in ['twitter.com', 'facebook.com', 'instagram.com']):
            return (2.0, 5.0)
        
        # News sites - moderate delays
        if any(news in domain for news in ['cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com']):
            return (1.5, 3.0)
        
        # Default delay
        return (1.0, 2.5)
    
    def setup_stealth_webdriver(self, url: str = None) -> Optional[webdriver.Chrome]:
        """Setup Selenium WebDriver with stealth mode."""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available")
            return None
        
        try:
            options = Options()
            
            # Randomize window size
            window_sizes = [
                (1920, 1080), (1366, 768), (1536, 864),
                (1440, 900), (1280, 720), (1600, 900),
                (1280, 1024), (1024, 768)
            ]
            width, height = random.choice(window_sizes)
            options.add_argument(f'--window-size={width},{height}')
            
            # Essential anti-detection arguments
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Additional stealth options
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-setuid-sandbox')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-webgl')
            options.add_argument('--disable-features=VizDisplayCompositor')
            
            # Disable automation indicators
            options.add_argument('--disable-ipc-flooding-protection')
            options.add_argument('--disable-renderer-backgrounding')
            options.add_argument('--disable-backgrounding-occluded-windows')
            
            # Random user agent
            user_agent = self.get_random_user_agent()
            options.add_argument(f'user-agent={user_agent}')
            
            # Disable images for faster loading (optional)
            if self.config.get('disable_images', False):
                options.add_argument('--blink-settings=imagesEnabled=false')
            
            # Headless mode with random viewport
            if self.config.get('headless_browser', True):
                options.add_argument('--headless')
                options.add_argument('--disable-gpu')
            
            # Create driver
            driver = webdriver.Chrome(options=options)
            
            # Apply selenium-stealth if available
            if STEALTH_AVAILABLE:
                stealth(driver,
                        languages=["en-US", "en"],
                        vendor="Google Inc.",
                        platform="Win32",
                        webgl_vendor="Intel Inc.",
                        renderer="Intel Iris OpenGL Engine",
                        fix_hairline=True,
                        run_on_insecure_origins=True)
            
            # Execute additional anti-detection scripts
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    // Remove webdriver property
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    
                    // Override chrome property
                    Object.defineProperty(window, 'chrome', {
                        get: () => ({
                            runtime: {},
                            loadTimes: function() {},
                            csi: function() {},
                            app: {}
                        })
                    });
                    
                    // Override permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                    
                    // Randomize screen properties
                    Object.defineProperty(screen, 'width', {
                        get: () => ''' + str(width) + '''
                    });
                    Object.defineProperty(screen, 'height', {
                        get: () => ''' + str(height) + '''
                    });
                    
                    // Add realistic plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [
                            {
                                name: 'Chrome PDF Plugin',
                                filename: 'internal-pdf-viewer',
                                description: 'Portable Document Format'
                            }
                        ]
                    });
                '''
            })
            
            logger.info(f"✅ Stealth WebDriver initialized ({width}x{height})")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to setup stealth WebDriver: {e}")
            return None
    
    def random_delay(self, url: str = None, min_seconds: float = None, max_seconds: float = None):
        """Add random delay to mimic human behavior."""
        if min_seconds is None or max_seconds is None:
            if url:
                min_seconds, max_seconds = self.get_site_specific_delay(url)
            else:
                min_seconds, max_seconds = 1.0, 2.5
        
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        logger.debug(f"Random delay: {delay:.2f}s")
    
    def simulate_human_behavior(self, driver: webdriver.Chrome):
        """Simulate human-like behavior in browser."""
        if not driver:
            return
        
        try:
            # Random mouse movements (simulated via JavaScript)
            for _ in range(random.randint(1, 3)):
                x = random.randint(100, 800)
                y = random.randint(100, 600)
                driver.execute_script(f"""
                    var event = new MouseEvent('mousemove', {{
                        view: window,
                        bubbles: true,
                        cancelable: true,
                        clientX: {x},
                        clientY: {y}
                    }});
                    document.dispatchEvent(event);
                """)
                time.sleep(random.uniform(0.1, 0.3))
            
            # Random scrolling
            scroll_positions = [
                random.randint(0, 500),
                random.randint(500, 1000),
                random.randint(200, 800)
            ]
            
            for position in scroll_positions:
                driver.execute_script(f"window.scrollTo(0, {position});")
                time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            logger.debug(f"Human behavior simulation failed: {e}")
    
    def get_detection_risk(self, url: str) -> str:
        """Assess detection risk for a URL."""
        domain = urlparse(url).netloc.lower()
        
        # High-risk sites
        high_risk_indicators = [
            'cloudflare', 'distil', 'imperva', 'bot-protection',
            'anti-bot', 'security'
        ]
        
        if any(indicator in domain for indicator in high_risk_indicators):
            return 'high'
        
        # JavaScript-heavy sites have medium risk
        if any(js_domain in domain for js_domain in self.js_required_domains):
            return 'medium'
        
        # Academic sites are usually low risk
        if self.is_academic_site(url):
            return 'low'
        
        return 'medium'
    
    def get_recommended_strategy(self, url: str) -> Dict[str, any]:
        """Get recommended scraping strategy for URL."""
        risk = self.get_detection_risk(url)
        is_academic = self.is_academic_site(url)
        needs_js = self.should_use_selenium(url)
        
        strategy = {
            'use_selenium': needs_js or risk == 'high',
            'use_proxy': risk in ['high', 'medium'],
            'delay_range': self.get_site_specific_delay(url),
            'retry_attempts': 3 if risk == 'high' else 2,
            'respect_robots': is_academic,
            'risk_level': risk
        }
        
        return strategy

class RateLimiter:
    """Advanced rate limiting for respectful scraping."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 1.0
        self.last_request_time = {}
        self.domain_specific_limits = {}
    
    def set_domain_limit(self, domain: str, requests_per_second: float):
        """Set domain-specific rate limit."""
        self.domain_specific_limits[domain] = requests_per_second
    
    def wait_if_needed(self, url: str):
        """Wait if necessary to respect rate limit."""
        domain = urlparse(url).netloc.lower()
        current_time = time.time()
        
        # Use domain-specific limit if available
        if domain in self.domain_specific_limits:
            min_interval = 1.0 / self.domain_specific_limits[domain]
        else:
            min_interval = self.min_interval
        
        # Check last request time for this domain
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting {domain}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def add_jitter(self, base_delay: float, jitter_percent: float = 0.2) -> float:
        """Add random jitter to delays."""
        jitter = base_delay * jitter_percent
        return base_delay + random.uniform(-jitter, jitter)

# Example usage and testing
async def test_anti_detection():
    """Test the anti-detection system."""
    print("🛡️ Testing Anti-Detection System")
    print("=" * 50)
    
    # Test configuration
    config = {
        'proxies': [],  # No proxies for testing
        'headless_browser': True,
        'disable_images': True
    }
    
    manager = AntiDetectionManager(config)
    
    # Test user agent generation
    print("\\n🎭 Testing User Agent Generation:")
    for i in range(3):
        ua = manager.get_random_user_agent()
        print(f"   {i+1}. {ua}")
    
    # Test header generation
    print("\\n📋 Testing Header Generation:")
    headers = manager.get_random_headers()
    for key, value in list(headers.items())[:5]:  # Show first 5 headers
        print(f"   {key}: {value}")
    
    # Test site analysis
    test_urls = [
        "https://arxiv.org/abs/2401.00001",
        "https://twitter.com/user/status/123",
        "https://example.com/article",
        "https://en.wikipedia.org/wiki/Test"
    ]
    
    print("\\n🔍 Testing Site Analysis:")
    for url in test_urls:
        strategy = manager.get_recommended_strategy(url)
        domain = urlparse(url).netloc
        print(f"   {domain}:")
        print(f"     Risk: {strategy['risk_level']}")
        print(f"     Use Selenium: {strategy['use_selenium']}")
        print(f"     Delay: {strategy['delay_range'][0]:.1f}-{strategy['delay_range'][1]:.1f}s")
    
    # Test rate limiter
    print("\\n⏱️ Testing Rate Limiter:")
    rate_limiter = RateLimiter(requests_per_second=2.0)
    
    start_time = time.time()
    for i in range(3):
        rate_limiter.wait_if_needed("https://example.com")
        elapsed = time.time() - start_time
        print(f"   Request {i+1}: {elapsed:.2f}s elapsed")
    
    # Test Selenium setup (if available)
    if SELENIUM_AVAILABLE:
        print("\\n🌐 Testing Selenium Setup:")
        driver = manager.setup_stealth_webdriver()
        if driver:
            print("   ✅ Stealth WebDriver created successfully")
            driver.quit()
        else:
            print("   ❌ Failed to create WebDriver")
    else:
        print("\\n🌐 Selenium not available - skipping WebDriver test")
    
    print("\\n✅ Anti-Detection System test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_anti_detection())