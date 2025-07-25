#!/usr/bin/env python3
"""
Structured Data Extractor for MCP Yggdrasil
Advanced extruct integration for JSON-LD, microdata, and structured metadata
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import extruct

logger = logging.getLogger(__name__)


class StructuredDataExtractor:
    """Extract structured data from web pages using extruct."""

    def __init__(self):
        self.supported_syntaxes = [
            "json-ld",
            "microdata",
            "rdfa",
            "opengraph",
            "microformat",
        ]
        logger.info("✅ StructuredDataExtractor initialized")

    def extract_all_structured_data(self, html_content: str, url: str) -> Dict:
        """Extract all types of structured data."""
        try:
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

            # Extract all formats using extruct
            data = extruct.extract(
                html_content,
                base_url=base_url,
                syntaxes=self.supported_syntaxes,
                uniform=True,
                errors="ignore",
            )

            # Process and clean the data
            processed_data = {
                "json_ld": self._process_json_ld(data.get("json-ld", [])),
                "microdata": self._process_microdata(data.get("microdata", [])),
                "opengraph": self._process_opengraph(data.get("opengraph", [])),
                "rdfa": self._process_rdfa(data.get("rdfa", [])),
                "microformat": self._process_microformat(data.get("microformat", [])),
            }

            # Extract high-value information
            extracted_info = self._extract_valuable_info(processed_data)

            # Calculate completeness score
            completeness_score = self._calculate_completeness(processed_data)

            return {
                "raw_structured_data": processed_data,
                "extracted_info": extracted_info,
                "has_structured_data": bool(any(processed_data.values())),
                "completeness_score": completeness_score,
                "extraction_method": "extruct",
                "url": url,
            }

        except Exception as e:
            logger.error(f"Structured data extraction failed for {url}: {e}")
            return {
                "error": str(e),
                "raw_structured_data": {},
                "extracted_info": {},
                "has_structured_data": False,
                "completeness_score": 0.0,
                "url": url,
            }

    def _process_json_ld(self, json_ld_data: List) -> List[Dict]:
        """Process JSON-LD data with enhanced extraction."""
        processed = []

        for item in json_ld_data:
            if isinstance(item, dict):
                item_type = item.get("@type", "")

                # Process different schema.org types
                if "Article" in str(item_type) or "NewsArticle" in str(item_type):
                    processed.append(
                        {
                            "type": "Article",
                            "headline": item.get("headline"),
                            "author": self._extract_author(item.get("author")),
                            "datePublished": item.get("datePublished"),
                            "dateModified": item.get("dateModified"),
                            "description": item.get("description"),
                            "keywords": self._extract_keywords(item.get("keywords")),
                            "publisher": self._extract_publisher(item.get("publisher")),
                            "image": self._extract_image(item.get("image")),
                            "url": item.get("url"),
                            "mainEntityOfPage": item.get("mainEntityOfPage"),
                        }
                    )

                elif "ScholarlyArticle" in str(item_type):
                    processed.append(
                        {
                            "type": "ScholarlyArticle",
                            "headline": item.get("headline"),
                            "author": self._extract_author(item.get("author")),
                            "citation": item.get("citation"),
                            "abstract": item.get("abstract"),
                            "keywords": self._extract_keywords(item.get("keywords")),
                            "doi": item.get("doi"),
                            "issn": item.get("issn"),
                            "datePublished": item.get("datePublished"),
                            "publisher": self._extract_publisher(item.get("publisher")),
                        }
                    )

                elif "Person" in str(item_type):
                    processed.append(
                        {
                            "type": "Person",
                            "name": item.get("name"),
                            "url": item.get("url"),
                            "affiliation": self._extract_affiliation(
                                item.get("affiliation")
                            ),
                            "sameAs": item.get("sameAs"),
                            "jobTitle": item.get("jobTitle"),
                            "email": item.get("email"),
                        }
                    )

                elif "Organization" in str(item_type):
                    processed.append(
                        {
                            "type": "Organization",
                            "name": item.get("name"),
                            "url": item.get("url"),
                            "logo": self._extract_image(item.get("logo")),
                            "sameAs": item.get("sameAs"),
                            "address": item.get("address"),
                            "contactPoint": item.get("contactPoint"),
                        }
                    )

                elif "WebPage" in str(item_type) or "WebSite" in str(item_type):
                    processed.append(
                        {
                            "type": str(item_type),
                            "name": item.get("name"),
                            "url": item.get("url"),
                            "description": item.get("description"),
                            "inLanguage": item.get("inLanguage"),
                            "keywords": self._extract_keywords(item.get("keywords")),
                            "publisher": self._extract_publisher(item.get("publisher")),
                        }
                    )

                elif "BreadcrumbList" in str(item_type):
                    breadcrumbs = []
                    items = item.get("itemListElement", [])
                    for breadcrumb in items:
                        if isinstance(breadcrumb, dict):
                            breadcrumbs.append(
                                {
                                    "position": breadcrumb.get("position"),
                                    "name": breadcrumb.get("name"),
                                    "url": (
                                        breadcrumb.get("item", {}).get("@id")
                                        if isinstance(breadcrumb.get("item"), dict)
                                        else breadcrumb.get("item")
                                    ),
                                }
                            )
                    processed.append(
                        {"type": "BreadcrumbList", "breadcrumbs": breadcrumbs}
                    )

                else:
                    # Keep original for other types
                    processed.append({"type": str(item_type), "original_data": item})

        return processed

    def _process_microdata(self, microdata: List) -> List[Dict]:
        """Process microdata with schema.org awareness."""
        processed = []

        for item in microdata:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                properties = item.get("properties", {})

                processed_item = {"type": item_type, "properties": {}}

                # Extract common properties
                for prop, values in properties.items():
                    if values:
                        if len(values) == 1:
                            processed_item["properties"][prop] = values[0]
                        else:
                            processed_item["properties"][prop] = values

                processed.append(processed_item)

        return processed

    def _process_opengraph(self, opengraph_data: List) -> Dict:
        """Process OpenGraph data."""
        if not opengraph_data:
            return {}

        og_data = opengraph_data[0] if opengraph_data else {}

        return {
            "title": og_data.get("og:title"),
            "description": og_data.get("og:description"),
            "image": og_data.get("og:image"),
            "url": og_data.get("og:url"),
            "type": og_data.get("og:type"),
            "site_name": og_data.get("og:site_name"),
            "locale": og_data.get("og:locale"),
            "article_author": og_data.get("article:author"),
            "article_published_time": og_data.get("article:published_time"),
            "article_modified_time": og_data.get("article:modified_time"),
            "article_section": og_data.get("article:section"),
            "article_tag": og_data.get("article:tag"),
        }

    def _process_rdfa(self, rdfa_data: List) -> List[Dict]:
        """Process RDFa data."""
        return rdfa_data  # Return as-is for now

    def _process_microformat(self, microformat_data: List) -> List[Dict]:
        """Process microformat data."""
        return microformat_data  # Return as-is for now

    def _extract_author(self, author_data) -> Optional[str]:
        """Extract author name from various formats."""
        if isinstance(author_data, str):
            return author_data
        elif isinstance(author_data, dict):
            return author_data.get("name", "")
        elif isinstance(author_data, list) and author_data:
            authors = []
            for author in author_data:
                if isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)
            return ", ".join(authors) if authors else None
        return None

    def _extract_publisher(self, publisher_data) -> Optional[Dict]:
        """Extract publisher information."""
        if isinstance(publisher_data, dict):
            return {
                "name": publisher_data.get("name"),
                "url": publisher_data.get("url"),
                "logo": self._extract_image(publisher_data.get("logo")),
            }
        elif isinstance(publisher_data, str):
            return {"name": publisher_data}
        return None

    def _extract_keywords(self, keywords_data) -> List[str]:
        """Extract keywords from various formats."""
        if isinstance(keywords_data, str):
            # Split by common delimiters
            return [kw.strip() for kw in keywords_data.split(",") if kw.strip()]
        elif isinstance(keywords_data, list):
            keywords = []
            for kw in keywords_data:
                if isinstance(kw, str):
                    keywords.append(kw.strip())
                elif isinstance(kw, dict) and kw.get("name"):
                    keywords.append(kw["name"])
            return keywords
        return []

    def _extract_image(self, image_data) -> Optional[str]:
        """Extract image URL from various formats."""
        if isinstance(image_data, str):
            return image_data
        elif isinstance(image_data, dict):
            return image_data.get("url") or image_data.get("@id")
        elif isinstance(image_data, list) and image_data:
            first_image = image_data[0]
            return self._extract_image(first_image)
        return None

    def _extract_affiliation(self, affiliation_data) -> Optional[str]:
        """Extract affiliation information."""
        if isinstance(affiliation_data, dict):
            return affiliation_data.get("name")
        elif isinstance(affiliation_data, str):
            return affiliation_data
        elif isinstance(affiliation_data, list) and affiliation_data:
            return self._extract_affiliation(affiliation_data[0])
        return None

    def _extract_valuable_info(self, structured_data: Dict) -> Dict:
        """Extract the most valuable information from all structured data."""
        info = {
            "title": None,
            "author": None,
            "date_published": None,
            "date_modified": None,
            "description": None,
            "type": None,
            "keywords": [],
            "images": [],
            "url": None,
            "publisher": None,
            "language": None,
        }

        # Priority: JSON-LD > OpenGraph > Microdata > RDFa > Microformat

        # From JSON-LD (highest priority)
        for item in structured_data.get("json_ld", []):
            if isinstance(item, dict):
                if not info["title"] and item.get("headline"):
                    info["title"] = item["headline"]
                if not info["author"] and item.get("author"):
                    info["author"] = item["author"]
                if not info["date_published"] and item.get("datePublished"):
                    info["date_published"] = item["datePublished"]
                if not info["date_modified"] and item.get("dateModified"):
                    info["date_modified"] = item["dateModified"]
                if not info["description"] and item.get("description"):
                    info["description"] = item["description"]
                if not info["type"] and item.get("type"):
                    info["type"] = item["type"]
                if not info["url"] and item.get("url"):
                    info["url"] = item["url"]
                if not info["publisher"] and item.get("publisher"):
                    info["publisher"] = item["publisher"]

                # Collect keywords
                if item.get("keywords"):
                    info["keywords"].extend(item["keywords"])

                # Collect images
                if item.get("image"):
                    info["images"].append(item["image"])

        # From OpenGraph (second priority)
        og_data = structured_data.get("opengraph", {})
        if og_data:
            if not info["title"] and og_data.get("title"):
                info["title"] = og_data["title"]
            if not info["description"] and og_data.get("description"):
                info["description"] = og_data["description"]
            if not info["url"] and og_data.get("url"):
                info["url"] = og_data["url"]
            if not info["type"] and og_data.get("type"):
                info["type"] = og_data["type"]
            if og_data.get("image"):
                info["images"].append(og_data["image"])
            if not info["author"] and og_data.get("article_author"):
                info["author"] = og_data["article_author"]
            if not info["date_published"] and og_data.get("article_published_time"):
                info["date_published"] = og_data["article_published_time"]

        # From Microdata (third priority)
        for item in structured_data.get("microdata", []):
            if isinstance(item, dict):
                props = item.get("properties", {})
                if not info["title"] and props.get("name"):
                    info["title"] = props["name"]
                if not info["description"] and props.get("description"):
                    info["description"] = props["description"]
                if not info["author"] and props.get("author"):
                    info["author"] = props["author"]

        # Clean up and deduplicate
        info["keywords"] = list(set(info["keywords"]))  # Remove duplicates
        info["images"] = list(set(info["images"]))  # Remove duplicates

        return info

    def _calculate_completeness(self, structured_data: Dict) -> float:
        """Calculate how complete the structured data is."""
        total_score = 0.0
        max_score = 5.0

        # JSON-LD presence (most valuable)
        if structured_data.get("json_ld"):
            total_score += 2.0

        # OpenGraph presence
        if structured_data.get("opengraph"):
            total_score += 1.5

        # Microdata presence
        if structured_data.get("microdata"):
            total_score += 1.0

        # RDFa presence
        if structured_data.get("rdfa"):
            total_score += 0.3

        # Microformat presence
        if structured_data.get("microformat"):
            total_score += 0.2

        return min(total_score / max_score, 1.0)

    def get_schema_types(self, structured_data: Dict) -> List[str]:
        """Extract all schema.org types found in the data."""
        types = []

        # From JSON-LD
        for item in structured_data.get("json_ld", []):
            if isinstance(item, dict) and item.get("type"):
                types.append(item["type"])

        # From Microdata
        for item in structured_data.get("microdata", []):
            if isinstance(item, dict) and item.get("type"):
                types.append(item["type"])

        return list(set(types))

    def extract_citations(self, structured_data: Dict) -> List[str]:
        """Extract academic citations from structured data."""
        citations = []

        # From JSON-LD
        for item in structured_data.get("json_ld", []):
            if isinstance(item, dict):
                if item.get("citation"):
                    if isinstance(item["citation"], list):
                        citations.extend(item["citation"])
                    else:
                        citations.append(item["citation"])

        return citations


if __name__ == "__main__":
    # Test the structured data extractor
    extractor = StructuredDataExtractor()

    # Example HTML with JSON-LD
    test_html = """
    <html>
    <head>
    <script type="application/ld+json">
    {
        "@type": "Article",
        "headline": "Test Article",
        "author": {"@type": "Person", "name": "John Doe"},
        "datePublished": "2024-01-01",
        "keywords": ["test", "example"]
    }
    </script>
    </head>
    <body></body>
    </html>
    """

    result = extractor.extract_all_structured_data(test_html, "https://example.com")
    print(f"Structured data extraction test:")
    print(f"Has structured data: {result['has_structured_data']}")
    print(f"Completeness score: {result['completeness_score']}")
    print(f"Extracted title: {result['extracted_info']['title']}")
