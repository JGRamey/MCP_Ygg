# Phase 3: Scraper Functionality Enhancement
## 🔄 ROBUST DATA ACQUISITION (Weeks 5-6)

### Overview
This phase focuses on dramatically improving extraction quality and reliability using specialized libraries, making the scraper undetectable and resilient to blocking, and creating a unified architecture with site-specific parsers.

### 🔸 Core Extraction & Data Quality Improvement

#### Task 1: Integrate Trafilatura for Main Content Extraction

##### Current Problem
- CSS selectors are brittle and break when sites update
- BeautifulSoup extracts too much boilerplate content
- Difficult to identify main article content

##### Solution: Trafilatura Integration
**File: `agents/scraper/enhanced_content_extractor.py`**
```python
import trafilatura
from trafilatura.settings import use_config
from typing import Dict, Optional, List
import extruct
import pycld3 as cld3
from urllib.parse import urlparse

class EnhancedContentExtractor:
    def __init__(self):
        # Configure trafilatura for better extraction
        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        
    def extract_main_content(self, html_content: str, url: str) -> Dict:
        """Extract main content using trafilatura."""
        
        # Extract with various options
        main_content = trafilatura.extract(
            html_content,
            url=url,
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_links=True,
            deduplicate=True,
            config=self.config,
            favor_precision=True,
            output_format='json'
        )
        
        if not main_content:
            # Fallback to less strict extraction
            main_content = trafilatura.extract(
                html_content,
                url=url,
                favor_recall=True,
                config=self.config
            )
        
        # Parse JSON output if available
        if isinstance(main_content, str):
            try:
                import json
                content_data = json.loads(main_content)
            except:
                content_data = {'text': main_content}
        else:
            content_data = main_content or {}
        
        return {
            'main_text': content_data.get('text', ''),
            'title': content_data.get('title', ''),
            'author': content_data.get('author', ''),
            'date': content_data.get('date', ''),
            'tags': content_data.get('tags', []),
            'language': content_data.get('language', ''),
            'excerpt': content_data.get('excerpt', ''),
            'source_url': url
        }
    
    def extract_metadata(self, html_content: str, url: str) -> Dict:
        """Extract all available metadata."""
        metadata = {}
        
        # Get trafilatura metadata
        traf_metadata = trafilatura.extract_metadata(html_content, url)
        if traf_metadata:
            metadata.update(traf_metadata)
        
        # Extract additional metadata using extruct
        try:
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            structured = extruct.extract(
                html_content,
                base_url=base_url,
                syntaxes=['json-ld', 'microdata', 'opengraph', 'microformat'],
                uniform=True
            )
            
            # Process JSON-LD (most reliable)
            if structured.get('json-ld'):
                for item in structured['json-ld']:
                    if isinstance(item, dict):
                        metadata['json_ld'] = item
                        
                        # Extract specific fields
                        if item.get('@type') == 'Article':
                            metadata['article_data'] = {
                                'headline': item.get('headline'),
                                'author': item.get('author'),
                                'datePublished': item.get('datePublished'),
                                'description': item.get('description')
                            }
            
            # Process OpenGraph
            if structured.get('opengraph'):
                og_data = structured['opengraph'][0] if structured['opengraph'] else {}
                metadata['opengraph'] = {
                    'title': og_data.get('og:title'),
                    'description': og_data.get('og:description'),
                    'image': og_data.get('og:image'),
                    'type': og_data.get('og:type')
                }
            
        except Exception as e:
            print(f"Extruct extraction failed: {e}")
        
        return metadata
```

#### Task 2: Integrate Extruct for Structured Metadata

**File: `agents/scraper/structured_data_extractor.py`**
```python
import extruct
from typing import Dict, List, Optional
from urllib.parse import urlparse
import json

class StructuredDataExtractor:
    """Extract structured data from web pages."""
    
    def extract_all_structured_data(self, html_content: str, url: str) -> Dict:
        """Extract all types of structured data."""
        try:
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            # Extract all formats
            data = extruct.extract(
                html_content,
                base_url=base_url,
                syntaxes=['json-ld', 'microdata', 'rdfa', 'opengraph', 'microformat'],
                uniform=True,
                errors='ignore'
            )
            
            # Process and clean the data
            processed_data = {
                'json_ld': self._process_json_ld(data.get('json-ld', [])),
                'microdata': self._process_microdata(data.get('microdata', [])),
                'opengraph': self._process_opengraph(data.get('opengraph', [])),
                'rdfa': data.get('rdfa', []),
                'microformat': data.get('microformat', [])
            }
            
            # Extract high-value information
            extracted_info = self._extract_valuable_info(processed_data)
            
            return {
                'raw_structured_data': processed_data,
                'extracted_info': extracted_info,
                'has_structured_data': bool(any(processed_data.values()))
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'raw_structured_data': {},
                'extracted_info': {},
                'has_structured_data': False
            }
    
    def _process_json_ld(self, json_ld_data: List) -> List[Dict]:
        """Process JSON-LD data."""
        processed = []
        
        for item in json_ld_data:
            if isinstance(item, dict):
                # Look for specific types
                item_type = item.get('@type', '')
                
                if 'Article' in str(item_type):
                    processed.append({
                        'type': 'Article',
                        'headline': item.get('headline'),
                        'author': self._extract_author(item.get('author')),
                        'datePublished': item.get('datePublished'),
                        'dateModified': item.get('dateModified'),
                        'description': item.get('description'),
                        'keywords': item.get('keywords'),
                        'publisher': self._extract_publisher(item.get('publisher'))
                    })
                    
                elif 'Person' in str(item_type):
                    processed.append({
                        'type': 'Person',
                        'name': item.get('name'),
                        'url': item.get('url'),
                        'affiliation': item.get('affiliation')
                    })
                    
                elif 'Organization' in str(item_type):
                    processed.append({
                        'type': 'Organization',
                        'name': item.get('name'),
                        'url': item.get('url'),
                        'logo': item.get('logo')
                    })
                    
                elif 'ScholarlyArticle' in str(item_type):
                    processed.append({
                        'type': 'ScholarlyArticle',
                        'headline': item.get('headline'),
                        'author': self._extract_author(item.get('author')),
                        'citation': item.get('citation'),
                        'abstract': item.get('abstract')
                    })
                else:
                    # Keep original for other types
                    processed.append(item)
        
        return processed
    
    def _extract_author(self, author_data) -> Optional[str]:
        """Extract author name from various formats."""
        if isinstance(author_data, str):
            return author_data
        elif isinstance(author_data, dict):
            return author_data.get('name', '')
        elif isinstance(author_data, list) and author_data:
            first_author = author_data[0]
            if isinstance(first_author, dict):
                return first_author.get('name', '')
            return str(first_author)
        return None
    
    def _extract_valuable_info(self, structured_data: Dict) -> Dict:
        """Extract the most valuable information from all structured data."""
        info = {
            'title': None,
            'author': None,
            'date_published': None,
            'description': None,
            'type': None,
            'keywords': [],
            'images': []
        }
        
        # Priority: JSON-LD > OpenGraph > Microdata
        
        # From JSON-LD
        for item in structured_data.get('json_ld', []):
            if isinstance(item, dict):
                if not info['title'] and item.get('headline'):
                    info['title'] = item['headline']
                if not info['author'] and item.get('author'):
                    info['author'] = item['author']
                if not info['date_published'] and item.get('datePublished'):
                    info['date_published'] = item['datePublished']
                if item.get('keywords'):
                    info['keywords'].extend(item['keywords'] if isinstance(item['keywords'], list) else [item['keywords']])
        
        # From OpenGraph
        og_data = structured_data.get('opengraph', [{}])[0] if structured_data.get('opengraph') else {}
        if og_data:
            if not info['title'] and og_data.get('title'):
                info['title'] = og_data['title']
            if not info['description'] and og_data.get('description'):
                info['description'] = og_data['description']
            if og_data.get('image'):
                info['images'].append(og_data['image'])
        
        return info
```

#### Task 3: Upgrade Language Detection

**File: `agents/scraper/language_detector.py`**
```python
import pycld3 as cld3
from typing import Dict, Optional, List
import langdetect
from collections import Counter

class AdvancedLanguageDetector:
    """Advanced language detection with multiple fallbacks."""
    
    def __init__(self):
        # Language name mapping
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'ko': 'Korean'
        }
    
    def detect_language(self, text: str) -> Dict:
        """Detect language with confidence scores."""
        if not text or not text.strip():
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'language_name': 'Unknown',
                'is_reliable': False,
                'alternative_languages': []
            }
        
        # Try CLD3 first (most accurate)
        cld3_result = self._detect_with_cld3(text)
        
        # Try langdetect as fallback
        langdetect_result = self._detect_with_langdetect(text)
        
        # Combine results
        if cld3_result['is_reliable']:
            result = cld3_result
            
            # Add alternative from langdetect if different
            if langdetect_result['language'] != cld3_result['language']:
                result['alternative_languages'].append({
                    'language': langdetect_result['language'],
                    'confidence': langdetect_result['confidence'],
                    'detector': 'langdetect'
                })
        else:
            # Use langdetect if CLD3 is not reliable
            result = langdetect_result
        
        return result
    
    def _detect_with_cld3(self, text: str) -> Dict:
        """Detect using CLD3."""
        try:
            prediction = cld3.get_language(text)
            
            return {
                'language': prediction.language,
                'confidence': prediction.probability,
                'language_name': self.language_names.get(prediction.language, prediction.language),
                'is_reliable': prediction.is_reliable,
                'alternative_languages': [],
                'detector': 'cld3'
            }
        except Exception as e:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'language_name': 'Unknown',
                'is_reliable': False,
                'alternative_languages': [],
                'error': str(e)
            }
    
    def _detect_with_langdetect(self, text: str) -> Dict:
        """Detect using langdetect."""
        try:
            # Get probabilities for all detected languages
            languages = langdetect.detect_langs(text)
            
            if languages:
                top_lang = languages[0]
                
                alternatives = [
                    {
                        'language': lang.lang,
                        'confidence': lang.prob,
                        'detector': 'langdetect'
                    }
                    for lang in languages[1:3]  # Top 3 alternatives
                ]
                
                return {
                    'language': top_lang.lang,
                    'confidence': top_lang.prob,
                    'language_name': self.language_names.get(top_lang.lang, top_lang.lang),
                    'is_reliable': top_lang.prob > 0.95,
                    'alternative_languages': alternatives,
                    'detector': 'langdetect'
                }
            
        except Exception as e:
            pass
        
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'language_name': 'Unknown',
            'is_reliable': False,
            'alternative_languages': []
        }
    
    def detect_mixed_languages(self, text: str, chunk_size: int = 500) -> Dict:
        """Detect if text contains multiple languages."""
        # Split text into chunks
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        # Detect language for each chunk
        detected_languages = []
        for chunk in chunks:
            result = self.detect_language(chunk)
            if result['is_reliable']:
                detected_languages.append(result['language'])
        
        # Count occurrences
        language_counts = Counter(detected_languages)
        
        # Determine if mixed
        is_mixed = len(language_counts) > 1
        
        # Get primary language
        primary_language = language_counts.most_common(1)[0][0] if language_counts else 'unknown'
        
        return {
            'is_mixed': is_mixed,
            'primary_language': primary_language,
            'language_distribution': dict(language_counts),
            'total_chunks': len(chunks),
            'reliable_chunks': len(detected_languages)
        }
```

### 🔸 Robustness & Anti-Blocking Measures

#### Task 4: Implement Proxy and User-Agent Rotation

**File: `agents/scraper/anti_detection.py`**
```python
import random
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth
import time
import requests
from fake_useragent import UserAgent

class AntiDetectionManager:
    """Manage anti-detection measures for web scraping."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ua = UserAgent()
        
        # Real browser user agents
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            
            # Chrome on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # Chrome on Linux
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # Firefox variants
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        ]
        
        # Proxy list (initially empty, can be populated)
        self.proxies = config.get('proxies', [])
        self.current_proxy_index = 0
        
        # Headers to randomize
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
                'en-US,en;q=0.9,es;q=0.8'
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
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        if random.random() < 0.8:  # 80% use predefined
            return random.choice(self.user_agents)
        else:  # 20% use fake-useragent
            return self.ua.random
    
    def get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers."""
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
        
        # Randomly add or remove some headers
        if random.random() < 0.5:
            headers['DNT'] = '1'
        
        if random.random() < 0.3:
            headers['Connection'] = 'keep-alive'
        
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
    
    def setup_stealth_webdriver(self) -> webdriver.Chrome:
        """Setup Selenium with stealth mode."""
        options = Options()
        
        # Randomize window size
        window_sizes = [
            (1920, 1080), (1366, 768), (1536, 864),
            (1440, 900), (1280, 720), (1600, 900)
        ]
        width, height = random.choice(window_sizes)
        options.add_argument(f'--window-size={width},{height}')
        
        # Anti-detection arguments
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Additional options for stealth
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-setuid-sandbox')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-webgl')
        
        # Random user agent
        options.add_argument(f'user-agent={self.get_random_user_agent()}')
        
        # Create driver
        driver = webdriver.Chrome(options=options)
        
        # Apply stealth
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                run_on_insecure_origins=True)
        
        # Execute additional scripts to hide automation
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Override chrome property
                Object.defineProperty(window, 'chrome', {
                    get: () => ({
                        runtime: {}
                    })
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            '''
        })
        
        return driver
    
    def random_delay(self, min_seconds: float = 1, max_seconds: float = 3):
        """Add random delay to mimic human behavior."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def should_use_selenium(self, url: str) -> bool:
        """Determine if Selenium is needed for a URL."""
        # Sites that typically require JavaScript
        js_required_domains = [
            'twitter.com', 'instagram.com', 'facebook.com',
            'linkedin.com', 'medium.com', 'reddit.com'
        ]
        
        # Check if domain requires JS
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        for js_domain in js_required_domains:
            if js_domain in domain:
                return True
        
        # Check if explicitly configured
        if url in self.config.get('selenium_urls', []):
            return True
        
        return False

class RateLimiter:
    """Rate limiting for respectful scraping."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = {}
    
    def wait_if_needed(self, domain: str):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def add_jitter(self, base_delay: float, jitter_percent: float = 0.1):
        """Add random jitter to delays."""
        jitter = base_delay * jitter_percent
        return base_delay + random.uniform(-jitter, jitter)
```

#### Task 5: Enhanced Scraper Configuration

**File: `agents/scraper/enhanced_scraper_config.py`**
```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

@dataclass
class ScraperConfig:
    """Enhanced scraper configuration."""
    
    # User agents
    user_agents: List[str] = None
    
    # Proxies
    proxies: List[str] = None
    use_proxy_rotation: bool = False
    
    # Rate limiting
    requests_per_second: float = 1.0
    request_delay_range: tuple = (1, 3)
    respect_robots_txt: bool = True
    
    # Retries
    retry_attempts: int = 3
    retry_delay: float = 5.0
    backoff_factor: float = 2.0
    
    # Timeouts
    connection_timeout: int = 30
    read_timeout: int = 30
    
    # Selenium options
    use_selenium_fallback: bool = True
    headless_browser: bool = True
    browser_binary_path: Optional[str] = None
    
    # Content extraction
    use_trafilatura: bool = True
    extract_comments: bool = False
    extract_images: bool = True
    extract_links: bool = True
    
    # Language detection
    detect_language: bool = True
    target_languages: List[str] = None
    
    # Caching
    cache_responses: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Output options
    save_raw_html: bool = False
    save_screenshots: bool = False
    screenshot_path: str = "./screenshots"
    
    def __post_init__(self):
        """Initialize with defaults if not provided."""
        if self.user_agents is None:
            self.user_agents = self._get_default_user_agents()
        
        if self.proxies is None:
            self.proxies = self._load_proxies_from_env()
        
        if self.target_languages is None:
            self.target_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
    
    def _get_default_user_agents(self) -> List[str]:
        """Get default user agents."""
        return [
            # Chrome variants
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # Firefox variants
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0',
            
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            
            # Mobile variants
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        ]
    
    def _load_proxies_from_env(self) -> List[str]:
        """Load proxies from environment variables."""
        proxy_list = []
        
        # Check for proxy environment variables
        if os.getenv('HTTP_PROXY'):
            proxy_list.append(os.getenv('HTTP_PROXY'))
        
        if os.getenv('PROXY_LIST'):
            # Expect comma-separated list
            proxy_list.extend(os.getenv('PROXY_LIST').split(','))
        
        return proxy_list
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'user_agents': self.user_agents,
            'proxies': self.proxies,
            'use_proxy_rotation': self.use_proxy_rotation,
            'requests_per_second': self.requests_per_second,
            'request_delay_range': self.request_delay_range,
            'respect_robots_txt': self.respect_robots_txt,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'connection_timeout': self.connection_timeout,
            'use_selenium_fallback': self.use_selenium_fallback,
            'headless_browser': self.headless_browser,
            'use_trafilatura': self.use_trafilatura,
            'detect_language': self.detect_language,
            'target_languages': self.target_languages,
            'cache_responses': self.cache_responses
        }

# Default configurations for different scraping profiles
SCRAPING_PROFILES = {
    'fast': ScraperConfig(
        requests_per_second=2.0,
        retry_attempts=1,
        use_selenium_fallback=False,
        extract_images=False,
        extract_links=False
    ),
    
    'comprehensive': ScraperConfig(
        requests_per_second=0.5,
        retry_attempts=3,
        use_selenium_fallback=True,
        extract_images=True,
        extract_links=True,
        extract_comments=True,
        save_raw_html=True
    ),
    
    'stealth': ScraperConfig(
        requests_per_second=0.2,
        request_delay_range=(2, 5),
        use_proxy_rotation=True,
        use_selenium_fallback=True,
        headless_browser=False
    ),
    
    'academic': ScraperConfig(
        requests_per_second=0.5,
        respect_robots_txt=True,
        extract_comments=False,
        use_trafilatura=True,
        target_languages=['en'],
        save_raw_html=True
    )
}
```

### 🔸 Architectural Improvements

#### Task 6: Unified Scraper Architecture

**File: `agents/scraper/unified_web_scraper.py`**
```python
from typing import Dict, List, Optional, Union
import asyncio
from urllib.parse import urlparse
from datetime import datetime
import hashlib

from .enhanced_content_extractor import EnhancedContentExtractor
from .structured_data_extractor import StructuredDataExtractor
from .language_detector import AdvancedLanguageDetector
from .anti_detection import AntiDetectionManager, RateLimiter
from .enhanced_scraper_config import ScraperConfig, SCRAPING_PROFILES

class UnifiedWebScraper:
    """Unified web scraper with configurable profiles."""
    
    def __init__(self, profile: str = 'comprehensive', custom_config: Optional[ScraperConfig] = None):
        # Load configuration
        if custom_config:
            self.config = custom_config
        else:
            self.config = SCRAPING_PROFILES.get(profile, SCRAPING_PROFILES['comprehensive'])
        
        # Initialize components
        self.content_extractor = EnhancedContentExtractor()
        self.structured_extractor = StructuredDataExtractor()
        self.language_detector = AdvancedLanguageDetector()
        self.anti_detection = AntiDetectionManager(self.config.to_dict())
        self.rate_limiter = RateLimiter(self.config.requests_per_second)
        
        # Site-specific parsers
        self.site_parsers = self._load_site_parsers()
        
        # Session management
        self.session = None
        self.selenium_driver = None
    
    async def scrape_url(self, url: str, **kwargs) -> Dict:
        """Scrape a single URL with all enhancements."""
        start_time = datetime.utcnow()
        
        # Check if site-specific parser exists
        domain = urlparse(url).netloc.lower()
        if domain in self.site_parsers:
            return await self._scrape_with_site_parser(url, domain)
        
        # Standard scraping flow
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed(domain)
            
            # Respect robots.txt if configured
            if self.config.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    return self._create_error_response(url, "Blocked by robots.txt")
            
            # Determine scraping method
            if self.anti_detection.should_use_selenium(url):
                html_content = await self._scrape_with_selenium(url)
            else:
                html_content = await self._scrape_with_requests(url)
            
            if not html_content:
                return self._create_error_response(url, "Failed to fetch content")
            
            # Extract content
            main_content = self.content_extractor.extract_main_content(html_content, url)
            
            # Extract metadata
            metadata = self.content_extractor.extract_metadata(html_content, url)
            
            # Extract structured data
            structured_data = self.structured_extractor.extract_all_structured_data(html_content, url)
            
            # Detect language
            language_info = {}
            if self.config.detect_language and main_content.get('main_text'):
                language_info = self.language_detector.detect_language(main_content['main_text'])
            
            # Generate content hash
            content_hash = self._generate_content_hash(main_content['main_text'])
            
            # Compile results
            result = {
                'url': url,
                'domain': domain,
                'timestamp': datetime.utcnow().isoformat(),
                'scraping_duration': (datetime.utcnow() - start_time).total_seconds(),
                'content': main_content,
                'metadata': metadata,
                'structured_data': structured_data,
                'language': language_info,
                'content_hash': content_hash,
                'scraper_profile': self.config.__class__.__name__,
                'success': True
            }
            
            # Save raw HTML if configured
            if self.config.save_raw_html:
                result['raw_html'] = html_content
            
            return result
            
        except Exception as e:
            return self._create_error_response(url, str(e))
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 5) -> List[Dict]:
        """Scrape multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(self._create_error_response(urls[i], str(result)))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _scrape_with_requests(self, url: str) -> Optional[str]:
        """Scrape using requests with anti-detection."""
        import aiohttp
        
        headers = self.anti_detection.get_random_headers()
        proxy = self.anti_detection.get_next_proxy() if self.config.use_proxy_rotation else None
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.connection_timeout + self.config.read_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.read_timeout
        )
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url,
                    headers=headers,
                    proxy=proxy['http'] if proxy else None,
                    ssl=False  # Consider certificate verification based on requirements
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            print(f"Request error for {url}: {e}")
            
            # Retry logic
            if self.config.retry_attempts > 0:
                await asyncio.sleep(self.config.retry_delay)
                self.config.retry_attempts -= 1
                return await self._scrape_with_requests(url)
            
            return None
    
    async def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape using Selenium with stealth mode."""
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        
        driver = self.anti_detection.setup_stealth_webdriver()
        
        try:
            # Navigate to URL
            driver.get(url)
            
            # Wait for content to load
            wait = WebDriverWait(driver, self.config.connection_timeout)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Random delay to mimic human behavior
            self.anti_detection.random_delay()
            
            # Scroll to load lazy content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            await asyncio.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            await asyncio.sleep(1)
            
            # Get page source
            html_content = driver.page_source
            
            # Take screenshot if configured
            if self.config.save_screenshots:
                screenshot_path = f"{self.config.screenshot_path}/{urlparse(url).netloc}_{datetime.now().timestamp()}.png"
                driver.save_screenshot(screenshot_path)
            
            return html_content
            
        except Exception as e:
            print(f"Selenium error for {url}: {e}")
            return None
            
        finally:
            driver.quit()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for deduplication."""
        if not content:
            return ""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _create_error_response(self, url: str, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            'url': url,
            'success': False,
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _load_site_parsers(self) -> Dict:
        """Load site-specific parsers."""
        # This will be populated by the plugin system
        return {}
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        # Implementation of robots.txt checking
        # For now, return True to allow all
        return True
```

#### Task 7: Plugin Architecture for Site-Specific Parsers

**Directory Structure**:
```
agents/scraper/parsers/
├── __init__.py
├── base_parser.py
├── plato_stanford_edu.py
├── wikipedia_org.py
├── arxiv_org.py
├── pubmed_ncbi_nlm_nih_gov.py
└── medium_com.py
```

**File: `agents/scraper/parsers/base_parser.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
from bs4 import BeautifulSoup

class BaseSiteParser(ABC):
    """Base class for site-specific parsers."""
    
    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain this parser handles."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Parser name."""
        pass
    
    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """Check if this parser can handle the URL."""
        pass
    
    @abstractmethod
    def parse(self, html: str, url: str) -> Dict:
        """Parse HTML and extract structured data."""
        pass
    
    def parse_soup(self, html: str) -> BeautifulSoup:
        """Create BeautifulSoup object."""
        return BeautifulSoup(html, 'html.parser')
    
    def clean_text(self, text: Optional[str]) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        return ' '.join(text.strip().split())
```

**File: `agents/scraper/parsers/plato_stanford_edu.py`**
```python
from .base_parser import BaseSiteParser
from typing import Dict
import re

class StanfordEncyclopediaParser(BaseSiteParser):
    """Parser for Stanford Encyclopedia of Philosophy."""
    
    @property
    def domain(self) -> str:
        return "plato.stanford.edu"
    
    @property
    def name(self) -> str:
        return "Stanford Encyclopedia of Philosophy Parser"
    
    def can_parse(self, url: str) -> bool:
        return "plato.stanford.edu/entries/" in url
    
    def parse(self, html: str, url: str) -> Dict:
        soup = self.parse_soup(html)
        
        # Extract article data
        article_data = {
            'parser': self.name,
            'domain': self.domain,
            'url': url
        }
        
        # Title
        title_elem = soup.find('h1')
        article_data['title'] = self.clean_text(title_elem.text) if title_elem else ""
        
        # Author(s)
        authors = []
        author_elem = soup.find('div', id='aueditable')
        if author_elem:
            for author in author_elem.find_all('a'):
                authors.append(self.clean_text(author.text))
        article_data['authors'] = authors
        
        # Publication info
        pubinfo = soup.find('div', id='pubinfo')
        if pubinfo:
            # Extract first published date
            first_pub = re.search(r'First published \w+ \d+ \w+ \d+', pubinfo.text)
            if first_pub:
                article_data['first_published'] = first_pub.group()
            
            # Extract revision date
            revision = re.search(r'substantive revision \w+ \d+ \w+ \d+', pubinfo.text)
            if revision:
                article_data['last_revised'] = revision.group()
        
        # Table of contents
        toc = []
        toc_elem = soup.find('div', id='toc')
        if toc_elem:
            for li in toc_elem.find_all('li'):
                toc.append(self.clean_text(li.text))
        article_data['table_of_contents'] = toc
        
        # Main content sections
        content_sections = []
        main_text = soup.find('div', id='main-text')
        if main_text:
            for section in main_text.find_all(['h2', 'h3']):
                section_data = {
                    'heading': self.clean_text(section.text),
                    'level': section.name,
                    'content': []
                }
                
                # Get content until next heading
                for sibling in section.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    if sibling.name == 'p':
                        section_data['content'].append(self.clean_text(sibling.text))
                
                content_sections.append(section_data)
        
        article_data['content_sections'] = content_sections
        
        # Bibliography
        bibliography = []
        bib_elem = soup.find('div', id='bibliography')
        if bib_elem:
            for li in bib_elem.find_all('li'):
                bibliography.append(self.clean_text(li.text))
        article_data['bibliography'] = bibliography
        
        # Academic references
        references = []
        for ref in soup.find_all('a', href=re.compile(r'#.+')):
            ref_text = self.clean_text(ref.text)
            if ref_text and len(ref_text) > 2:
                references.append(ref_text)
        article_data['internal_references'] = list(set(references))
        
        # Extract keywords/topics
        keywords = []
        keyword_elem = soup.find('div', id='related-entries')
        if keyword_elem:
            for a in keyword_elem.find_all('a'):
                keywords.append(self.clean_text(a.text))
        article_data['related_topics'] = keywords
        
        return article_data
```

**File: `agents/scraper/parsers/arxiv_org.py`**
```python
from .base_parser import BaseSiteParser
from typing import Dict
import re

class ArxivParser(BaseSiteParser):
    """Parser for arXiv.org papers."""
    
    @property
    def domain(self) -> str:
        return "arxiv.org"
    
    @property
    def name(self) -> str:
        return "arXiv Paper Parser"
    
    def can_parse(self, url: str) -> bool:
        return "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url
    
    def parse(self, html: str, url: str) -> Dict:
        soup = self.parse_soup(html)
        
        paper_data = {
            'parser': self.name,
            'domain': self.domain,
            'url': url,
            'type': 'academic_paper'
        }
        
        # Extract arXiv ID
        arxiv_match = re.search(r'arxiv\.org/[^/]+/(\d+\.\d+)', url)
        if arxiv_match:
            paper_data['arxiv_id'] = arxiv_match.group(1)
        
        # Title
        title_elem = soup.find('h1', class_='title')
        if title_elem:
            paper_data['title'] = self.clean_text(title_elem.text.replace('Title:', ''))
        
        # Authors
        authors = []
        author_elem = soup.find('div', class_='authors')
        if author_elem:
            for a in author_elem.find_all('a'):
                authors.append(self.clean_text(a.text))
        paper_data['authors'] = authors
        
        # Abstract
        abstract_elem = soup.find('blockquote', class_='abstract')
        if abstract_elem:
            paper_data['abstract'] = self.clean_text(
                abstract_elem.text.replace('Abstract:', '')
            )
        
        # Submission date
        dateline = soup.find('div', class_='dateline')
        if dateline:
            paper_data['submission_date'] = self.clean_text(dateline.text)
        
        # Categories/subjects
        subjects = []
        subject_elem = soup.find('td', class_='tablecell subjects')
        if subject_elem:
            subjects = [self.clean_text(s) for s in subject_elem.text.split(';')]
        paper_data['subjects'] = subjects
        
        # Comments (pages, figures, etc.)
        comments_elem = soup.find('td', class_='tablecell comments')
        if comments_elem:
            paper_data['comments'] = self.clean_text(comments_elem.text)
        
        # DOI
        doi_elem = soup.find('td', class_='tablecell doi')
        if doi_elem:
            paper_data['doi'] = self.clean_text(doi_elem.text)
        
        # PDF link
        pdf_link = soup.find('a', href=re.compile(r'/pdf/'))
        if pdf_link:
            paper_data['pdf_url'] = f"https://arxiv.org{pdf_link['href']}"
        
        return paper_data
```

### Implementation Checklist

#### Week 5: Core Extraction Quality
- [ ] Integrate trafilatura for content extraction
- [ ] Implement extruct for structured metadata
- [ ] Upgrade to pycld3 for language detection
- [ ] Add mixed language detection
- [ ] Test extraction quality on 100+ URLs

#### Week 6: Anti-Blocking & Architecture
- [ ] Implement user-agent rotation system
- [ ] Add proxy support and rotation
- [ ] Integrate selenium-stealth
- [ ] Create unified scraper class
- [ ] Implement site-specific parsers for top 10 domains
- [ ] Add rate limiting and respectful delays
- [ ] Test anti-detection on difficult sites

### Success Criteria
- ✅ 95%+ successful extraction rate
- ✅ <5% detection rate on anti-scraping sites
- ✅ Support for 10+ languages
- ✅ Site-specific parsers for academic sources
- ✅ Respectful scraping with robots.txt compliance
- ✅ Configurable profiles for different use cases

### Next Steps
After completing Phase 3, proceed to:
- **Phase 4**: Data Validation Pipeline (`updates/04_data_validation.md`)
- **Phase 5**: UI Workspace Development (`updates/05_ui_workspace.md`)