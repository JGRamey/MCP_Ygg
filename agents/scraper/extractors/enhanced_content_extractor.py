#!/usr/bin/env python3
"""
Enhanced Content Extractor for MCP Yggdrasil
Phase 3: Advanced content extraction using Trafilatura + fallbacks
"""

import json
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

import extruct
import trafilatura
from trafilatura.settings import use_config

# Language detection with fallbacks
try:
    import pycld3 as cld3

    CLD3_AVAILABLE = True
except ImportError:
    CLD3_AVAILABLE = False
    try:
        import langdetect

        LANGDETECT_AVAILABLE = True
    except ImportError:
        LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedContentExtractor:
    """Advanced content extraction with multiple methods and fallbacks."""

    def __init__(self):
        # Configure trafilatura for optimal extraction
        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        self.config.set("DEFAULT", "MIN_OUTPUT_SIZE", "25")

        # Language detection configuration
        self.language_detector_available = CLD3_AVAILABLE or LANGDETECT_AVAILABLE

        logger.info(f"✅ Enhanced Content Extractor initialized")
        logger.info(f"   Trafilatura: Available")
        logger.info(f"   Extruct: Available")
        logger.info(f"   CLD3: {'Available' if CLD3_AVAILABLE else 'Not available'}")
        logger.info(
            f"   LangDetect: {'Available' if LANGDETECT_AVAILABLE else 'Not available'}"
        )

    def extract_main_content(self, html_content: str, url: str) -> Dict:
        """Extract main content using trafilatura with enhanced settings."""

        # Extract with precision mode first
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
            output_format="json",
        )

        # Fallback to recall mode if precision failed
        if not main_content:
            main_content = trafilatura.extract(
                html_content,
                url=url,
                favor_recall=True,
                config=self.config,
                output_format="json",
            )

        # Fallback to basic extraction if JSON fails
        if not main_content:
            main_content = trafilatura.extract(
                html_content, url=url, config=self.config
            )

        # Parse JSON output if available
        if isinstance(main_content, str):
            try:
                if main_content.startswith("{"):
                    content_data = json.loads(main_content)
                else:
                    content_data = {"text": main_content}
            except json.JSONDecodeError:
                content_data = {"text": main_content}
        else:
            content_data = main_content or {}

        # Ensure we have consistent structure
        extracted_content = {
            "main_text": content_data.get("text", "")
            or content_data.get("raw_text", ""),
            "title": content_data.get("title", ""),
            "author": content_data.get("author", ""),
            "date": content_data.get("date", ""),
            "tags": content_data.get("tags", []) or content_data.get("categories", []),
            "language": content_data.get("language", ""),
            "excerpt": content_data.get("excerpt", "")
            or content_data.get("description", ""),
            "source_url": url,
            "extraction_method": "trafilatura",
        }

        # Detect language if not provided and detector available
        if (
            not extracted_content["language"]
            and extracted_content["main_text"]
            and self.language_detector_available
        ):
            language_info = self._detect_language(extracted_content["main_text"])
            extracted_content["language"] = language_info.get("language", "")
            extracted_content["language_confidence"] = language_info.get(
                "confidence", 0.0
            )

        # Calculate content metrics
        text = extracted_content["main_text"]
        if text:
            extracted_content["content_metrics"] = {
                "character_count": len(text),
                "word_count": len(text.split()),
                "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
                "has_substantial_content": len(text) > 500,
                "readability_estimate": self._estimate_readability(text),
            }

        return extracted_content

    def extract_metadata(self, html_content: str, url: str) -> Dict:
        """Extract comprehensive metadata from HTML."""
        metadata = {}

        # Get trafilatura metadata
        try:
            traf_metadata = trafilatura.extract_metadata(html_content, url)
            if traf_metadata:
                metadata.update(
                    traf_metadata.__dict__
                    if hasattr(traf_metadata, "__dict__")
                    else traf_metadata
                )
        except Exception as e:
            logger.warning(f"Trafilatura metadata extraction failed: {e}")

        # Extract additional metadata using extruct
        try:
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            structured = extruct.extract(
                html_content,
                base_url=base_url,
                syntaxes=["json-ld", "microdata", "opengraph", "microformat"],
                uniform=True,
            )

            # Process JSON-LD (most reliable)
            if structured.get("json-ld"):
                json_ld_data = []
                for item in structured["json-ld"]:
                    if isinstance(item, dict):
                        json_ld_data.append(item)

                        # Extract specific fields for articles
                        if item.get("@type") == "Article":
                            metadata["article_data"] = {
                                "headline": item.get("headline"),
                                "author": self._extract_author_from_structured(
                                    item.get("author")
                                ),
                                "datePublished": item.get("datePublished"),
                                "dateModified": item.get("dateModified"),
                                "description": item.get("description"),
                                "keywords": item.get("keywords"),
                                "publisher": self._extract_publisher_from_structured(
                                    item.get("publisher")
                                ),
                            }

                metadata["json_ld"] = json_ld_data

            # Process OpenGraph
            if structured.get("opengraph"):
                og_data = structured["opengraph"][0] if structured["opengraph"] else {}
                metadata["opengraph"] = {
                    "title": og_data.get("og:title"),
                    "description": og_data.get("og:description"),
                    "image": og_data.get("og:image"),
                    "type": og_data.get("og:type"),
                    "url": og_data.get("og:url"),
                    "site_name": og_data.get("og:site_name"),
                }

            # Process Microdata
            if structured.get("microdata"):
                metadata["microdata"] = structured["microdata"]

        except Exception as e:
            logger.warning(f"Extruct metadata extraction failed: {e}")

        return metadata

    def extract_links_and_references(self, html_content: str, url: str) -> Dict:
        """Extract links and references from content."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        links_data = {
            "internal_links": [],
            "external_links": [],
            "academic_references": [],
            "images": [],
            "total_links": 0,
        }

        # Extract all links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)

            if not href or not text:
                continue

            # Resolve relative URLs
            if href.startswith("/"):
                href = base_url + href
            elif not href.startswith(("http://", "https://")):
                continue

            link_data = {"url": href, "text": text, "title": link.get("title", "")}

            # Categorize links
            if urlparse(href).netloc == urlparse(url).netloc:
                links_data["internal_links"].append(link_data)
            else:
                links_data["external_links"].append(link_data)

                # Check for academic references
                academic_domains = [
                    "doi.org",
                    "arxiv.org",
                    "pubmed.ncbi.nlm.nih.gov",
                    "jstor.org",
                    "springer.com",
                    "elsevier.com",
                    "wiley.com",
                    "nature.com",
                    "science.org",
                ]

                if any(domain in href.lower() for domain in academic_domains):
                    links_data["academic_references"].append(link_data)

        # Extract images
        for img in soup.find_all("img", src=True):
            src = img["src"]
            if src.startswith("/"):
                src = base_url + src

            links_data["images"].append(
                {"src": src, "alt": img.get("alt", ""), "title": img.get("title", "")}
            )

        links_data["total_links"] = len(links_data["internal_links"]) + len(
            links_data["external_links"]
        )

        return links_data

    def _detect_language(self, text: str) -> Dict:
        """Detect language with available detectors."""
        if not text or not text.strip():
            return {"language": "unknown", "confidence": 0.0}

        # Try CLD3 first (most accurate)
        if CLD3_AVAILABLE:
            try:
                prediction = cld3.get_language(text)
                return {
                    "language": prediction.language,
                    "confidence": prediction.probability,
                    "is_reliable": prediction.is_reliable,
                    "detector": "cld3",
                }
            except Exception as e:
                logger.warning(f"CLD3 language detection failed: {e}")

        # Fallback to langdetect
        if LANGDETECT_AVAILABLE:
            try:
                import langdetect

                languages = langdetect.detect_langs(text)
                if languages:
                    top_lang = languages[0]
                    return {
                        "language": top_lang.lang,
                        "confidence": top_lang.prob,
                        "is_reliable": top_lang.prob > 0.9,
                        "detector": "langdetect",
                    }
            except Exception as e:
                logger.warning(f"Langdetect language detection failed: {e}")

        return {"language": "unknown", "confidence": 0.0, "detector": "none"}

    def _extract_author_from_structured(self, author_data) -> Optional[str]:
        """Extract author name from structured data."""
        if isinstance(author_data, str):
            return author_data
        elif isinstance(author_data, dict):
            return author_data.get("name", "")
        elif isinstance(author_data, list) and author_data:
            first_author = author_data[0]
            if isinstance(first_author, dict):
                return first_author.get("name", "")
            return str(first_author)
        return None

    def _extract_publisher_from_structured(self, publisher_data) -> Optional[str]:
        """Extract publisher name from structured data."""
        if isinstance(publisher_data, str):
            return publisher_data
        elif isinstance(publisher_data, dict):
            return publisher_data.get("name", "")
        return None

    def _estimate_readability(self, text: str) -> str:
        """Simple readability estimation."""
        if not text:
            return "unknown"

        words = text.split()
        sentences = text.split(".")

        if len(sentences) == 0:
            return "unknown"

        avg_words_per_sentence = len(words) / len(sentences)

        if avg_words_per_sentence < 10:
            return "easy"
        elif avg_words_per_sentence < 20:
            return "medium"
        else:
            return "difficult"

    def get_extraction_stats(self) -> Dict:
        """Get statistics about extraction capabilities."""
        return {
            "trafilatura_available": True,
            "extruct_available": True,
            "language_detection": {
                "cld3_available": CLD3_AVAILABLE,
                "langdetect_available": LANGDETECT_AVAILABLE,
                "any_available": self.language_detector_available,
            },
            "supported_formats": ["html", "xml"],
            "extraction_methods": [
                "trafilatura_precision",
                "trafilatura_recall",
                "trafilatura_basic",
            ],
            "metadata_sources": [
                "trafilatura",
                "json-ld",
                "opengraph",
                "microdata",
                "microformat",
            ],
        }


# Example usage and testing
async def test_enhanced_extractor():
    """Test the enhanced content extractor."""
    print("🧪 Testing Enhanced Content Extractor")
    print("=" * 50)

    extractor = EnhancedContentExtractor()

    # Show capabilities
    stats = extractor.get_extraction_stats()
    print("📊 Extraction Capabilities:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test with sample HTML
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article</title>
        <meta property="og:title" content="Test Article for Extraction">
        <meta property="og:description" content="This is a test article to demonstrate content extraction.">
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Test Article for Extraction",
            "author": {"@type": "Person", "name": "Test Author"},
            "datePublished": "2024-01-15",
            "description": "Test article description"
        }
        </script>
    </head>
    <body>
        <article>
            <h1>Test Article for Extraction</h1>
            <p>This is the main content of the test article. It contains multiple sentences to test extraction quality.</p>
            <p>This is a second paragraph with more content to analyze. The extractor should capture all main content.</p>
            <a href="https://example.com">External Link</a>
            <a href="/internal">Internal Link</a>
        </article>
    </body>
    </html>
    """

    test_url = "https://test.example.com/article"

    # Extract content
    print(f"\\n📄 Testing content extraction...")
    content = extractor.extract_main_content(sample_html, test_url)

    print(f"   Title: {content.get('title', 'N/A')}")
    print(f"   Author: {content.get('author', 'N/A')}")
    print(f"   Language: {content.get('language', 'N/A')}")
    print(
        f"   Word count: {content.get('content_metrics', {}).get('word_count', 'N/A')}"
    )
    print(f"   Text preview: {content.get('main_text', '')[:100]}...")

    # Extract metadata
    print(f"\\n🏷️  Testing metadata extraction...")
    metadata = extractor.extract_metadata(sample_html, test_url)

    if metadata.get("opengraph"):
        print(f"   OpenGraph title: {metadata['opengraph'].get('title', 'N/A')}")
    if metadata.get("article_data"):
        print(f"   JSON-LD author: {metadata['article_data'].get('author', 'N/A')}")

    # Extract links
    print(f"\\n🔗 Testing link extraction...")
    links = extractor.extract_links_and_references(sample_html, test_url)
    print(f"   Total links: {links['total_links']}")
    print(f"   Internal links: {len(links['internal_links'])}")
    print(f"   External links: {len(links['external_links'])}")

    print("\\n✅ Enhanced Content Extractor test complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_enhanced_extractor())
