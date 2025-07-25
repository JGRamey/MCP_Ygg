#!/usr/bin/env python3
"""
MCP Server Scraper Agent
Scrapes academic papers, books, and manuscripts from various sources
"""

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import robots, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiofiles
import aiohttp
import asyncio
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import requests
import scrapy
import yaml
from bs4 import BeautifulSoup
from PIL import Image
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """Data structure for scraped documents"""

    title: str
    author: Optional[str]
    content: str
    date: Optional[str]
    domain: str
    subcategory: Optional[str]
    source: str
    url: str
    language: str
    license: Optional[str]
    file_type: str
    word_count: int
    metadata: Dict
    scraped_at: datetime
    checksum: str


@dataclass
class ScrapingTarget:
    """Configuration for scraping targets"""

    url: str
    domain: str
    subcategory: str
    source_type: str  # 'academic', 'public_domain', 'manuscript', 'tablet'
    auth_required: bool = False
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # seconds between requests
    max_retries: int = 3
    timeout: int = 30


class LicenseDetector:
    """Detects and validates licenses for scraped content"""

    LICENSE_PATTERNS = {
        "public_domain": [
            "public domain",
            "cc0",
            "creative commons zero",
            "no rights reserved",
        ],
        "creative_commons": [
            "creative commons",
            "cc by",
            "cc by-sa",
            "cc by-nc",
            "cc by-nd",
        ],
        "mit": ["mit license"],
        "apache": ["apache license"],
        "gpl": ["gpl", "gnu general public license"],
        "copyright": ["copyright", "©", "all rights reserved"],
    }

    def detect_license(self, text: str, url: str = None) -> Optional[str]:
        """Detect license type from text content"""
        text_lower = text.lower()

        for license_type, patterns in self.LICENSE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return license_type

        # Check URL for additional clues
        if url:
            url_lower = url.lower()
            if "gutenberg" in url_lower or "archive.org" in url_lower:
                return "public_domain"

        return "unknown"

    def is_copyright_free(self, license_type: str) -> bool:
        """Check if license allows free use"""
        free_licenses = {"public_domain", "creative_commons", "mit", "apache", "gpl"}
        return license_type in free_licenses


class RobotChecker:
    """Checks robots.txt compliance"""

    def __init__(self):
        self.robots_cache = {}
        self.cache_expiry = timedelta(hours=1)

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            # Check cache
            if robots_url in self.robots_cache:
                cache_time, rp = self.robots_cache[robots_url]
                if datetime.now() - cache_time < self.cache_expiry:
                    return rp.can_fetch(user_agent, url)

            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            # Cache result
            self.robots_cache[robots_url] = (datetime.now(), rp)

            return rp.can_fetch(user_agent, url)

        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True  # Default to allowing if robots.txt can't be checked


class ExponentialBackoff:
    """Implements exponential backoff for retries"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def retry(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == self.max_retries:
                    break

                delay = min(self.base_delay * (2**attempt), self.max_delay)
                delay += random.uniform(0, delay * 0.1)  # Add jitter

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        raise last_exception


class OCRProcessor:
    """Handles OCR for images and scanned documents"""

    def __init__(self):
        self.tesseract_config = "--oem 3 --psm 6"

    def extract_text_from_image(self, image_path: str, language: str = "eng") -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(
                image, lang=language, config=self.tesseract_config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""

    def extract_text_from_pdf_images(self, pdf_path: str, language: str = "eng") -> str:
        """Extract text from PDF images using OCR"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # Try to extract text directly first
                text = page.get_text()
                if text.strip():
                    text_content.append(text)
                else:
                    # Use OCR for image-based pages
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")

                    # Save temporary image
                    temp_path = f"/tmp/page_{page_num}.png"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)

                    ocr_text = self.extract_text_from_image(temp_path, language)
                    text_content.append(ocr_text)

                    # Clean up
                    Path(temp_path).unlink(missing_ok=True)

            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"PDF OCR failed for {pdf_path}: {e}")
            return ""


class PDFProcessor:
    """Handles PDF document processing"""

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                metadata = {
                    "pages": len(pdf.pages),
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "creation_date": pdf.metadata.get("CreationDate", ""),
                }

                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)

                full_text = "\n\n".join(text_content)

                # If no text extracted, try OCR
                if not full_text.strip():
                    ocr = OCRProcessor()
                    full_text = ocr.extract_text_from_pdf_images(pdf_path)

                return full_text, metadata

        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {e}")
            return "", {}


class WebScraper:
    """Main web scraping class"""

    def __init__(self, config_path: str = "agents/scraper/config.py"):
        self.load_config(config_path)
        self.robot_checker = RobotChecker()
        self.license_detector = LicenseDetector()
        self.backoff = ExponentialBackoff()
        self.pdf_processor = PDFProcessor()
        self.session = None

        # Setup Chrome driver for JavaScript-heavy sites
        self.setup_webdriver()

    def load_config(self, config_path: str) -> None:
        """Load scraper configuration"""
        try:
            with open("agents/scraper/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "user_agent": "MCP-Server/1.0 (Academic Research)",
                "request_delay": 1.0,
                "max_retries": 3,
                "timeout": 30,
                "max_concurrent": 5,
                "output_dir": "data/raw",
                "supported_formats": ["pdf", "html", "txt", "doc", "docx"],
                "max_file_size": 50 * 1024 * 1024,  # 50MB
            }

    def setup_webdriver(self) -> None:
        """Setup Chrome WebDriver for JavaScript rendering"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f'--user-agent={self.config["user_agent"]}')

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.warning(f"Could not setup Chrome driver: {e}")
            self.driver = None

    async def create_session(self) -> aiohttp.ClientSession:
        """Create async HTTP session with proper headers"""
        headers = {
            "User-Agent": self.config["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
        return aiohttp.ClientSession(headers=headers, timeout=timeout)

    def calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum for content"""
        return hashlib.sha256(content.encode()).hexdigest()

    async def download_file(self, url: str, output_path: str) -> bool:
        """Download file from URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content_length = response.headers.get("content-length")
                    if (
                        content_length
                        and int(content_length) > self.config["max_file_size"]
                    ):
                        logger.warning(f"File too large: {url}")
                        return False

                    async with aiofiles.open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    return True
                else:
                    logger.error(f"Failed to download {url}: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return False

    async def scrape_html_content(self, url: str) -> Tuple[str, Dict]:
        """Scrape HTML content and extract text"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract metadata
                metadata = {
                    "title": soup.title.string if soup.title else "",
                    "description": "",
                    "keywords": "",
                    "author": "",
                }

                # Extract meta tags
                for meta in soup.find_all("meta"):
                    name = meta.get("name", "").lower()
                    property_attr = meta.get("property", "").lower()
                    content = meta.get("content", "")

                    if name in ["description", "keywords", "author"]:
                        metadata[name] = content
                    elif property_attr == "og:title":
                        metadata["title"] = content
                    elif property_attr == "og:description":
                        metadata["description"] = content

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Extract main content
                main_content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", class_="content")
                )
                if main_content:
                    text = main_content.get_text()
                else:
                    text = soup.get_text()

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                return text, metadata

        except Exception as e:
            logger.error(f"HTML scraping error for {url}: {e}")
            return "", {}

    async def scrape_javascript_content(self, url: str) -> Tuple[str, Dict]:
        """Scrape JavaScript-rendered content using Selenium"""
        if not self.driver:
            return "", {}

        try:
            self.driver.get(url)

            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Get page source after JavaScript execution
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract metadata and text (similar to scrape_html_content)
            metadata = {
                "title": self.driver.title,
                "url": self.driver.current_url,
            }

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text, metadata

        except Exception as e:
            logger.error(f"JavaScript scraping error for {url}: {e}")
            return "", {}

    def detect_language(self, text: str) -> str:
        """Detect language of text content"""
        # Simple language detection based on character patterns
        # In production, use langdetect or similar library

        if not text:
            return "unknown"

        # Count different character types
        latin_chars = sum(1 for c in text if ord(c) < 256)
        greek_chars = sum(1 for c in text if 0x0370 <= ord(c) <= 0x03FF)
        cyrillic_chars = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)

        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "unknown"

        # Simple heuristics
        if greek_chars / total_chars > 0.1:
            return "greek"
        elif cyrillic_chars / total_chars > 0.1:
            return "russian"
        elif latin_chars / total_chars > 0.8:
            return "english"  # Default to English for Latin script
        else:
            return "unknown"

    async def process_target(self, target: ScrapingTarget) -> Optional[ScrapedDocument]:
        """Process a single scraping target"""
        logger.info(f"Processing: {target.url}")

        # Check robots.txt
        if not self.robot_checker.can_fetch(target.url, self.config["user_agent"]):
            logger.warning(f"Robots.txt disallows scraping: {target.url}")
            return None

        # Rate limiting
        await asyncio.sleep(target.rate_limit)

        try:
            # Determine file type from URL
            parsed_url = urlparse(target.url)
            file_extension = Path(parsed_url.path).suffix.lower()

            if file_extension == ".pdf":
                # Download and process PDF
                output_path = f"/tmp/{hashlib.md5(target.url.encode()).hexdigest()}.pdf"
                if await self.download_file(target.url, output_path):
                    content, metadata = self.pdf_processor.extract_text_from_pdf(
                        output_path
                    )
                    Path(output_path).unlink(missing_ok=True)
                else:
                    return None

            elif file_extension in [".html", ".htm", ""]:
                # Check if JavaScript rendering is needed
                if (
                    "javascript" in target.source_type.lower()
                    or "spa" in target.source_type.lower()
                ):
                    content, metadata = await self.scrape_javascript_content(target.url)
                else:
                    content, metadata = await self.scrape_html_content(target.url)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None

            if not content.strip():
                logger.warning(f"No content extracted from: {target.url}")
                return None

            # Detect license
            license_type = self.license_detector.detect_license(content, target.url)

            # Detect language
            language = self.detect_language(content)

            # Create document
            document = ScrapedDocument(
                title=metadata.get("title", "")
                or f"Document from {urlparse(target.url).netloc}",
                author=metadata.get("author", ""),
                content=content,
                date=metadata.get("creation_date", ""),
                domain=target.domain,
                subcategory=target.subcategory,
                source=target.url,
                url=target.url,
                language=language,
                license=license_type,
                file_type=file_extension or "html",
                word_count=len(content.split()),
                metadata=metadata,
                scraped_at=datetime.now(),
                checksum=self.calculate_checksum(content),
            )

            # Save document
            await self.save_document(document)

            logger.info(f"Successfully scraped: {target.url}")
            return document

        except Exception as e:
            logger.error(f"Error processing {target.url}: {e}")
            return None

    async def save_document(self, document: ScrapedDocument) -> None:
        """Save scraped document to storage"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from checksum
        filename = f"{document.checksum}.json"
        output_path = output_dir / filename

        # Save as JSON
        document_dict = asdict(document)
        document_dict["scraped_at"] = document.scraped_at.isoformat()

        async with aiofiles.open(output_path, "w") as f:
            await f.write(json.dumps(document_dict, indent=2, ensure_ascii=False))

    async def scrape_targets(
        self, targets: List[ScrapingTarget]
    ) -> List[ScrapedDocument]:
        """Scrape multiple targets with concurrency control"""
        self.session = await self.create_session()

        try:
            # Limit concurrency
            semaphore = asyncio.Semaphore(self.config["max_concurrent"])

            async def bounded_process(target):
                async with semaphore:
                    return await self.backoff.retry(self.process_target, target)

            # Process all targets
            tasks = [bounded_process(target) for target in targets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            documents = []
            for result in results:
                if isinstance(result, ScrapedDocument):
                    documents.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")

            return documents

        finally:
            await self.session.close()
            if self.driver:
                self.driver.quit()

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()


async def main():
    """Example usage of the scraper"""
    # Example targets
    targets = [
        ScrapingTarget(
            url="https://www.gutenberg.org/files/1342/1342-h/1342-h.htm",
            domain="literature",
            subcategory="fiction",
            source_type="public_domain",
            rate_limit=1.0,
        ),
        ScrapingTarget(
            url="https://arxiv.org/pdf/2301.00000.pdf",
            domain="science",
            subcategory="computer_science",
            source_type="academic",
            rate_limit=2.0,
        ),
    ]

    scraper = WebScraper()

    try:
        documents = await scraper.scrape_targets(targets)
        print(f"Successfully scraped {len(documents)} documents")

        for doc in documents:
            print(f"- {doc.title} ({doc.word_count} words, {doc.license})")

    finally:
        scraper.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
