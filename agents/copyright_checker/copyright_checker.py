#!/usr/bin/env python3
"""
MCP Server Copyright Checker Agent
Identifies non-copyrighted materials and detects licenses for compliance
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import aiofiles
import aiohttp
import asyncio
import requests
import yaml
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LicenseInfo:
    """Information about a detected license"""

    license_type: str
    license_name: str
    is_free: bool
    is_copyleft: bool
    commercial_use: bool
    modification_allowed: bool
    distribution_allowed: bool
    attribution_required: bool
    share_alike_required: bool
    confidence: float
    detected_patterns: List[str]
    license_url: Optional[str] = None
    full_text: Optional[str] = None


@dataclass
class CopyrightStatus:
    """Copyright status of a document"""

    doc_id: str
    url: str
    title: str
    copyright_free: bool
    license_info: Optional[LicenseInfo]
    copyright_notices: List[str]
    public_domain_indicators: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    notes: str
    checked_at: datetime
    confidence: float


@dataclass
class PublicDomainWork:
    """Information about a public domain work"""

    title: str
    author: str
    death_year: Optional[int]
    publication_year: Optional[int]
    country: str
    copyright_expired: bool
    reason: str  # 'author_death', 'pre_copyright', 'government_work', etc.
    source: str


class LicenseDetector:
    """Detects various types of licenses in text and metadata"""

    def __init__(self):
        """Initialize license detector"""
        self.license_patterns = self._load_license_patterns()
        self.copyright_patterns = self._load_copyright_patterns()
        self.public_domain_patterns = self._load_public_domain_patterns()

    def _load_license_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load license detection patterns"""
        return {
            "public_domain": {
                "patterns": [
                    r"public\s+domain",
                    r"no\s+rights?\s+reserved",
                    r"copyright\s+expired",
                    r"out\s+of\s+copyright",
                    r"cc0|creative\s+commons\s+zero",
                    r"released\s+to\s+(?:the\s+)?public\s+domain",
                ],
                "indicators": ["public domain", "cc0", "no rights reserved"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": False,
            },
            "cc_by": {
                "patterns": [
                    r"cc\s+by\s+\d\.\d",
                    r"creative\s+commons\s+attribution",
                    r"licensed\s+under\s+cc\s+by",
                    r"creativecommons\.org/licenses/by",
                ],
                "indicators": ["CC BY", "Creative Commons Attribution"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
            },
            "cc_by_sa": {
                "patterns": [
                    r"cc\s+by[_-]sa\s+\d\.\d",
                    r"creative\s+commons\s+attribution[_-]sharealike",
                    r"licensed\s+under\s+cc\s+by[_-]sa",
                    r"creativecommons\.org/licenses/by-sa",
                ],
                "indicators": ["CC BY-SA", "Creative Commons Attribution-ShareAlike"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
                "share_alike_required": True,
            },
            "cc_by_nc": {
                "patterns": [
                    r"cc\s+by[_-]nc\s+\d\.\d",
                    r"creative\s+commons\s+attribution[_-]noncommercial",
                    r"licensed\s+under\s+cc\s+by[_-]nc",
                    r"creativecommons\.org/licenses/by-nc",
                ],
                "indicators": [
                    "CC BY-NC",
                    "Creative Commons Attribution-NonCommercial",
                ],
                "is_free": True,
                "commercial_use": False,
                "modification_allowed": True,
                "attribution_required": True,
            },
            "cc_by_nd": {
                "patterns": [
                    r"cc\s+by[_-]nd\s+\d\.\d",
                    r"creative\s+commons\s+attribution[_-]noderivatives",
                    r"licensed\s+under\s+cc\s+by[_-]nd",
                    r"creativecommons\.org/licenses/by-nd",
                ],
                "indicators": [
                    "CC BY-ND",
                    "Creative Commons Attribution-NoDerivatives",
                ],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": False,
                "attribution_required": True,
            },
            "mit": {
                "patterns": [
                    r"mit\s+licen[sc]e",
                    r"permission\s+is\s+hereby\s+granted",
                    r"without\s+restriction.*use.*copy.*modify.*merge.*publish.*distribute",
                ],
                "indicators": ["MIT License", "MIT"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
            },
            "apache": {
                "patterns": [
                    r"apache\s+licen[sc]e\s+version\s+2\.0",
                    r"licensed\s+under\s+the\s+apache\s+licen[sc]e",
                    r"apache\.org/licenses/LICENSE-2\.0",
                ],
                "indicators": ["Apache License", "Apache 2.0"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
            },
            "gpl": {
                "patterns": [
                    r"gnu\s+general\s+public\s+licen[sc]e",
                    r"gpl\s+v?[23]",
                    r"free\s+software\s+foundation.*gpl",
                    r"gnu\.org/licenses/gpl",
                ],
                "indicators": ["GPL", "GNU General Public License"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
                "share_alike_required": True,
            },
            "bsd": {
                "patterns": [
                    r"bsd\s+licen[sc]e",
                    r"berkeley\s+software\s+distribution",
                    r"redistribution.*use.*source.*binary\s+forms",
                ],
                "indicators": ["BSD License", "BSD"],
                "is_free": True,
                "commercial_use": True,
                "modification_allowed": True,
                "attribution_required": True,
            },
            "fair_use": {
                "patterns": [
                    r"fair\s+use",
                    r"educational\s+use\s+only",
                    r"research\s+purposes?\s+only",
                    r"non[_-]?commercial\s+use\s+only",
                ],
                "indicators": ["fair use", "educational use", "research purposes"],
                "is_free": False,
                "commercial_use": False,
                "modification_allowed": False,
                "attribution_required": True,
            },
            "all_rights_reserved": {
                "patterns": [
                    r"all\s+rights?\s+reserved",
                    r"©.*all\s+rights?\s+reserved",
                    r"copyright.*all\s+rights?\s+reserved",
                    r"proprietary\s+and\s+confidential",
                ],
                "indicators": ["All rights reserved", "© All rights reserved"],
                "is_free": False,
                "commercial_use": False,
                "modification_allowed": False,
                "attribution_required": True,
            },
        }

    def _load_copyright_patterns(self) -> List[str]:
        """Load copyright notice detection patterns"""
        return [
            r"©\s*\d{4}",
            r"copyright\s+©?\s*\d{4}",
            r"copyright\s+©?\s*\(\s*c\s*\)\s*\d{4}",
            r"\(c\)\s*\d{4}",
            r"copr\.\s*\d{4}",
            r"©\s*\d{4}[-–]\d{4}",
            r"copyright\s+©?\s*\d{4}[-–]\d{4}",
        ]

    def _load_public_domain_patterns(self) -> List[str]:
        """Load public domain indicator patterns"""
        return [
            r"public\s+domain",
            r"no\s+known\s+copyright",
            r"copyright\s+expired",
            r"copyright\s+free",
            r"out\s+of\s+copyright",
            r"pre[_-]?copyright",
            r"government\s+work",
            r"federal\s+government",
            r"us\s+government",
            r"released\s+to\s+(?:the\s+)?public\s+domain",
        ]

    def detect_license(
        self, text: str, url: Optional[str] = None
    ) -> Optional[LicenseInfo]:
        """Detect license in text content"""
        text_lower = text.lower()
        detected_licenses = []

        for license_type, license_data in self.license_patterns.items():
            patterns = license_data["patterns"]
            confidence = 0.0
            matched_patterns = []

            # Check patterns
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                match_count = len(list(matches))
                if match_count > 0:
                    confidence += match_count * 0.3
                    matched_patterns.append(pattern)

            # Check indicators
            for indicator in license_data.get("indicators", []):
                if indicator.lower() in text_lower:
                    confidence += 0.4
                    matched_patterns.append(indicator)

            # Normalize confidence
            confidence = min(1.0, confidence)

            if confidence > 0.3:  # Minimum threshold
                license_info = LicenseInfo(
                    license_type=license_type,
                    license_name=license_data.get("indicators", [license_type])[0],
                    is_free=license_data.get("is_free", False),
                    is_copyleft=license_data.get("share_alike_required", False),
                    commercial_use=license_data.get("commercial_use", False),
                    modification_allowed=license_data.get(
                        "modification_allowed", False
                    ),
                    distribution_allowed=license_data.get("is_free", False),
                    attribution_required=license_data.get(
                        "attribution_required", False
                    ),
                    share_alike_required=license_data.get(
                        "share_alike_required", False
                    ),
                    confidence=confidence,
                    detected_patterns=matched_patterns,
                )
                detected_licenses.append((confidence, license_info))

        # Return highest confidence license
        if detected_licenses:
            detected_licenses.sort(key=lambda x: x[0], reverse=True)
            return detected_licenses[0][1]

        return None

    def detect_copyright_notices(self, text: str) -> List[str]:
        """Detect copyright notices in text"""
        notices = []

        for pattern in self.copyright_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                notice = match.group().strip()
                if notice not in notices:
                    notices.append(notice)

        return notices

    def detect_public_domain_indicators(self, text: str) -> List[str]:
        """Detect public domain indicators in text"""
        indicators = []
        text_lower = text.lower()

        for pattern in self.public_domain_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                indicator = match.group().strip()
                if indicator not in indicators:
                    indicators.append(indicator)

        return indicators


class PublicDomainDatabase:
    """Database of known public domain works"""

    def __init__(
        self, db_path: str = "agents/copyright_checker/lists/public_domain.json"
    ):
        """Initialize public domain database"""
        self.db_path = Path(db_path)
        self.works = self._load_database()
        self.author_death_years = self._load_author_death_years()

    def _load_database(self) -> Dict[str, PublicDomainWork]:
        """Load public domain works database"""
        if not self.db_path.exists():
            return {}

        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)

            works = {}
            for work_data in data.get("works", []):
                work = PublicDomainWork(**work_data)
                works[work.title.lower()] = work

            return works

        except Exception as e:
            logger.error(f"Error loading public domain database: {e}")
            return {}

    def _load_author_death_years(self) -> Dict[str, int]:
        """Load author death years for copyright calculation"""
        return {
            "william shakespeare": 1616,
            "jane austen": 1817,
            "charles dickens": 1870,
            "mark twain": 1910,
            "arthur conan doyle": 1930,
            "h.g. wells": 1946,
            "george orwell": 1950,
            "virginia woolf": 1941,
            "james joyce": 1941,
            "f. scott fitzgerald": 1940,
            "ernest hemingway": 1961,
            "agatha christie": 1976,
            # Add more authors as needed
        }

    def is_public_domain(self, title: str, author: str = None) -> Tuple[bool, str]:
        """Check if a work is in the public domain"""
        title_lower = title.lower().strip()

        # Check exact match in database
        if title_lower in self.works:
            work = self.works[title_lower]
            return work.copyright_expired, work.reason

        # Check by author death year (life + 70 years rule)
        if author:
            author_lower = author.lower().strip()
            if author_lower in self.author_death_years:
                death_year = self.author_death_years[author_lower]
                current_year = datetime.now().year
                if current_year - death_year >= 70:  # Standard copyright term
                    return True, f"Author died in {death_year}, copyright expired"

        # Check for pre-1923 works (US public domain)
        year_match = re.search(r"\b(18\d{2}|19[01]\d|192[0-2])\b", title)
        if year_match:
            year = int(year_match.group(1))
            if year < 1923:
                return True, f"Published before 1923 ({year})"

        return False, "Cannot determine public domain status"

    def add_work(self, work: PublicDomainWork) -> None:
        """Add a work to the public domain database"""
        self.works[work.title.lower()] = work
        self._save_database()

    def _save_database(self) -> None:
        """Save public domain database"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "updated_at": datetime.now().isoformat(),
                "works": [asdict(work) for work in self.works.values()],
            }

            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving public domain database: {e}")


class DomainSpecificChecker:
    """Domain-specific copyright checking rules"""

    def __init__(self):
        """Initialize domain-specific checker"""
        self.domain_rules = {
            "religion": {
                "ancient_texts": [
                    "bible",
                    "quran",
                    "torah",
                    "vedas",
                    "upanishads",
                    "tripitaka",
                    "talmud",
                    "book of mormon",
                ],
                "public_domain_threshold": 1900,  # Most religious texts are old
                "risk_factors": ["modern translation", "commentary", "annotation"],
            },
            "philosophy": {
                "ancient_authors": [
                    "plato",
                    "aristotle",
                    "confucius",
                    "lao tzu",
                    "marcus aurelius",
                    "epictetus",
                    "seneca",
                    "cicero",
                ],
                "public_domain_threshold": 1900,
                "risk_factors": ["modern edition", "new translation", "commentary"],
            },
            "literature": {
                "classic_authors": [
                    "homer",
                    "virgil",
                    "dante",
                    "chaucer",
                    "shakespeare",
                    "cervantes",
                    "molière",
                    "goethe",
                ],
                "public_domain_threshold": 1923,
                "risk_factors": [
                    "modern adaptation",
                    "new edition",
                    "annotated version",
                ],
            },
            "history": {
                "ancient_sources": [
                    "herodotus",
                    "thucydides",
                    "tacitus",
                    "suetonius",
                    "plutarch",
                    "josephus",
                    "bede",
                ],
                "public_domain_threshold": 1900,
                "risk_factors": ["modern analysis", "commentary", "interpretation"],
            },
            "science": {
                "historical_works": [
                    "newton",
                    "galileo",
                    "copernicus",
                    "kepler",
                    "darwin",
                ],
                "public_domain_threshold": 1923,
                "risk_factors": [
                    "modern textbook",
                    "recent research",
                    "contemporary analysis",
                ],
            },
            "math": {
                "classical_works": [
                    "euclid",
                    "archimedes",
                    "pythagoras",
                    "fibonacci",
                    "fermat",
                    "pascal",
                    "leibniz",
                    "euler",
                ],
                "public_domain_threshold": 1900,
                "risk_factors": [
                    "modern proof",
                    "contemporary explanation",
                    "new methodology",
                ],
            },
        }

    def check_domain_specific_rules(
        self, domain: str, title: str, author: str, content: str
    ) -> Tuple[bool, str, List[str]]:
        """Apply domain-specific copyright rules"""
        if domain not in self.domain_rules:
            return False, "No domain-specific rules", []

        rules = self.domain_rules[domain]
        title_lower = title.lower()
        author_lower = author.lower() if author else ""
        content_lower = content.lower()

        risk_factors = []
        likely_public_domain = False
        reason = ""

        # Check for ancient texts/authors
        ancient_items = (
            rules.get("ancient_texts", [])
            + rules.get("ancient_authors", [])
            + rules.get("classic_authors", [])
        )
        for item in ancient_items:
            if item in title_lower or item in author_lower:
                likely_public_domain = True
                reason = f"Ancient/classical work: {item}"
                break

        # Check publication year threshold
        threshold = rules.get("public_domain_threshold", 1923)
        year_match = re.search(r"\b(1[89]\d{2}|19[0-5]\d)\b", title + " " + content)
        if year_match:
            year = int(year_match.group(1))
            if year < threshold:
                likely_public_domain = True
                reason = f"Published before {threshold} ({year})"

        # Check for risk factors
        for risk_factor in rules.get("risk_factors", []):
            if risk_factor in title_lower or risk_factor in content_lower:
                risk_factors.append(risk_factor)

        return likely_public_domain, reason, risk_factors


class UrlAnalyzer:
    """Analyzes URLs for copyright indicators"""

    def __init__(self):
        """Initialize URL analyzer"""
        self.trusted_domains = {
            "gutenberg.org": {"type": "public_domain", "confidence": 0.95},
            "archive.org": {"type": "mixed", "confidence": 0.8},
            "wikisource.org": {"type": "mixed", "confidence": 0.8},
            "sacred-texts.com": {"type": "public_domain", "confidence": 0.9},
            "perseus.tufts.edu": {"type": "academic", "confidence": 0.8},
            "ccel.org": {"type": "public_domain", "confidence": 0.9},
            "bartleby.com": {"type": "mixed", "confidence": 0.7},
            "poetryfoundation.org": {"type": "mixed", "confidence": 0.6},
            "loc.gov": {"type": "government", "confidence": 0.9},
            "nist.gov": {"type": "government", "confidence": 0.9},
            "nih.gov": {"type": "government", "confidence": 0.9},
            "arxiv.org": {"type": "academic", "confidence": 0.7},
            "pubmed.ncbi.nlm.nih.gov": {"type": "academic", "confidence": 0.7},
        }

        self.risky_domains = {
            "books.google.com": "Google Books (check copyright)",
            "scribd.com": "Scribd (often copyrighted)",
            "academia.edu": "Academia.edu (mixed copyright)",
            "researchgate.net": "ResearchGate (check individual papers)",
        }

    def analyze_url(self, url: str) -> Tuple[str, float, str]:
        """Analyze URL for copyright implications"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check trusted domains
            if domain in self.trusted_domains:
                info = self.trusted_domains[domain]
                return info["type"], info["confidence"], f"Trusted domain: {domain}"

            # Check risky domains
            if domain in self.risky_domains:
                warning = self.risky_domains[domain]
                return "risky", 0.3, warning

            # Check for government domains
            if domain.endswith(".gov"):
                return "government", 0.9, "Government domain (likely public domain)"

            # Check for educational domains
            if domain.endswith(".edu"):
                return "academic", 0.6, "Educational domain (check individual content)"

            # Check for organization domains
            if domain.endswith(".org"):
                return "mixed", 0.5, "Organization domain (mixed copyright status)"

            return "unknown", 0.2, "Unknown domain - manual review required"

        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return "unknown", 0.0, "URL analysis failed"


class CopyrightChecker:
    """Main copyright checking system"""

    def __init__(self, config_path: str = "agents/copyright_checker/config.yaml"):
        """Initialize copyright checker"""
        self.load_config(config_path)

        # Initialize components
        self.license_detector = LicenseDetector()
        self.public_domain_db = PublicDomainDatabase()
        self.domain_checker = DomainSpecificChecker()
        self.url_analyzer = UrlAnalyzer()

        # Results storage
        self.copyright_results: Dict[str, CopyrightStatus] = {}

        # Load existing results
        self.load_existing_results()

    def load_config(self, config_path: str) -> None:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "risk_thresholds": {
                    "low": 0.8,  # > 80% confidence it's free
                    "medium": 0.5,  # 50-80% confidence
                    "high": 0.5,  # < 50% confidence
                },
                "auto_approve_threshold": 0.9,
                "require_manual_review": True,
                "output_dir": "agents/copyright_checker/lists",
                "check_metadata": True,
                "check_url": True,
                "check_content": True,
            }

    def load_existing_results(self) -> None:
        """Load previously checked results"""
        results_file = Path(self.config["output_dir"]) / "copyright_results.json"

        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                for result_data in data.get("results", []):
                    # Convert datetime strings back
                    result_data["checked_at"] = datetime.fromisoformat(
                        result_data["checked_at"]
                    )

                    # Reconstruct license info if present
                    if result_data.get("license_info"):
                        result_data["license_info"] = LicenseInfo(
                            **result_data["license_info"]
                        )

                    result = CopyrightStatus(**result_data)
                    self.copyright_results[result.doc_id] = result

                logger.info(
                    f"Loaded {len(self.copyright_results)} existing copyright results"
                )

            except Exception as e:
                logger.error(f"Error loading existing results: {e}")

    def save_results(self) -> None:
        """Save copyright results"""
        results_file = Path(self.config["output_dir"])
        results_file.mkdir(parents=True, exist_ok=True)
        results_file = results_file / "copyright_results.json"

        try:
            results_data = []
            for result in self.copyright_results.values():
                result_dict = asdict(result)
                result_dict["checked_at"] = result.checked_at.isoformat()
                results_data.append(result_dict)

            data = {
                "updated_at": datetime.now().isoformat(),
                "total_results": len(results_data),
                "results": results_data,
            }

            with open(results_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    async def check_copyright(self, document_data: Dict[str, Any]) -> CopyrightStatus:
        """Perform comprehensive copyright check"""
        doc_id = document_data.get("doc_id", "")
        url = document_data.get("url", document_data.get("source", ""))
        title = document_data.get("title", "")
        author = document_data.get("author", "")
        content = document_data.get("content", "") or document_data.get(
            "cleaned_content", ""
        )
        domain = document_data.get("domain", "")

        logger.info(f"Checking copyright for document: {doc_id}")

        # Initialize result
        result = CopyrightStatus(
            doc_id=doc_id,
            url=url,
            title=title,
            copyright_free=False,
            license_info=None,
            copyright_notices=[],
            public_domain_indicators=[],
            risk_level="high",
            notes="",
            checked_at=datetime.now(),
            confidence=0.0,
        )

        confidence_factors = []
        notes = []

        try:
            # 1. URL Analysis
            if self.config.get("check_url", True) and url:
                url_type, url_confidence, url_note = self.url_analyzer.analyze_url(url)
                confidence_factors.append(("url", url_confidence))
                notes.append(f"URL: {url_note}")

                if url_type in ["public_domain", "government"]:
                    result.public_domain_indicators.append(f"Trusted domain: {url}")

            # 2. License Detection
            if self.config.get("check_content", True) and content:
                license_info = self.license_detector.detect_license(content, url)
                if license_info:
                    result.license_info = license_info
                    confidence_factors.append(("license", license_info.confidence))
                    notes.append(f"License detected: {license_info.license_name}")

                    if license_info.is_free:
                        result.copyright_free = True

                # Detect copyright notices
                copyright_notices = self.license_detector.detect_copyright_notices(
                    content
                )
                result.copyright_notices = copyright_notices

                if copyright_notices:
                    notes.append(f"Copyright notices found: {len(copyright_notices)}")
                    confidence_factors.append(
                        ("copyright_notices", -0.5)
                    )  # Negative factor

                # Detect public domain indicators
                pd_indicators = self.license_detector.detect_public_domain_indicators(
                    content
                )
                result.public_domain_indicators.extend(pd_indicators)

                if pd_indicators:
                    notes.append(f"Public domain indicators: {len(pd_indicators)}")
                    confidence_factors.append(("public_domain", 0.8))

            # 3. Public Domain Database Check
            is_pd, pd_reason = self.public_domain_db.is_public_domain(title, author)
            if is_pd:
                result.copyright_free = True
                result.public_domain_indicators.append(pd_reason)
                confidence_factors.append(("public_domain_db", 0.9))
                notes.append(f"Public domain: {pd_reason}")

            # 4. Domain-Specific Rules
            if domain:
                likely_pd, reason, risk_factors = (
                    self.domain_checker.check_domain_specific_rules(
                        domain, title, author, content
                    )
                )

                if likely_pd:
                    result.copyright_free = True
                    result.public_domain_indicators.append(reason)
                    confidence_factors.append(("domain_rules", 0.7))
                    notes.append(f"Domain rule: {reason}")

                if risk_factors:
                    notes.append(f"Risk factors: {', '.join(risk_factors)}")
                    confidence_factors.append(("risk_factors", -0.3))

            # 5. Calculate Overall Confidence and Risk
            if confidence_factors:
                # Weighted average of confidence factors
                total_weight = 0
                weighted_sum = 0

                for factor_type, confidence in confidence_factors:
                    weight = self._get_factor_weight(factor_type)
                    weighted_sum += confidence * weight
                    total_weight += weight

                result.confidence = (
                    weighted_sum / total_weight if total_weight > 0 else 0.0
                )
            else:
                result.confidence = 0.0

            # Determine risk level
            result.risk_level = self._calculate_risk_level(
                result.confidence, result.copyright_free
            )

            # Compile notes
            result.notes = "; ".join(notes) if notes else "No specific indicators found"

            # Store result
            self.copyright_results[doc_id] = result
            self.save_results()

            logger.info(
                f"Copyright check completed for {doc_id}: {result.risk_level} risk, {result.confidence:.2f} confidence"
            )

            return result

        except Exception as e:
            logger.error(f"Copyright check failed for {doc_id}: {e}")
            result.notes = f"Error during check: {str(e)}"
            result.risk_level = "high"
            return result

    def _get_factor_weight(self, factor_type: str) -> float:
        """Get weight for different confidence factors"""
        weights = {
            "url": 0.2,
            "license": 0.3,
            "copyright_notices": 0.2,
            "public_domain": 0.3,
            "public_domain_db": 0.4,
            "domain_rules": 0.2,
            "risk_factors": 0.1,
        }
        return weights.get(factor_type, 0.1)

    def _calculate_risk_level(self, confidence: float, copyright_free: bool) -> str:
        """Calculate risk level based on confidence and copyright status"""
        thresholds = self.config["risk_thresholds"]

        if copyright_free and confidence >= thresholds["low"]:
            return "low"
        elif confidence >= thresholds["medium"]:
            return "medium"
        else:
            return "high"

    def get_copyright_status(self, doc_id: str) -> Optional[CopyrightStatus]:
        """Get copyright status for a document"""
        return self.copyright_results.get(doc_id)

    def get_free_documents(self, min_confidence: float = 0.7) -> List[CopyrightStatus]:
        """Get list of copyright-free documents"""
        return [
            result
            for result in self.copyright_results.values()
            if result.copyright_free and result.confidence >= min_confidence
        ]

    def get_risky_documents(self, max_confidence: float = 0.5) -> List[CopyrightStatus]:
        """Get list of potentially copyrighted documents"""
        return [
            result
            for result in self.copyright_results.values()
            if not result.copyright_free or result.confidence <= max_confidence
        ]

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate copyright compliance report"""
        total_docs = len(self.copyright_results)
        free_docs = len(self.get_free_documents())
        risky_docs = len(self.get_risky_documents())

        risk_distribution = {
            "low": len(
                [r for r in self.copyright_results.values() if r.risk_level == "low"]
            ),
            "medium": len(
                [r for r in self.copyright_results.values() if r.risk_level == "medium"]
            ),
            "high": len(
                [r for r in self.copyright_results.values() if r.risk_level == "high"]
            ),
        }

        license_distribution = {}
        for result in self.copyright_results.values():
            if result.license_info:
                license_type = result.license_info.license_type
                license_distribution[license_type] = (
                    license_distribution.get(license_type, 0) + 1
                )

        return {
            "summary": {
                "total_documents": total_docs,
                "copyright_free": free_docs,
                "potentially_copyrighted": risky_docs,
                "compliance_rate": free_docs / total_docs if total_docs > 0 else 0,
            },
            "risk_distribution": risk_distribution,
            "license_distribution": license_distribution,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        risky_docs = self.get_risky_documents()
        if risky_docs:
            recommendations.append(
                f"Review {len(risky_docs)} documents with unclear copyright status"
            )

        high_risk = [
            r for r in self.copyright_results.values() if r.risk_level == "high"
        ]
        if high_risk:
            recommendations.append(
                f"Remove or replace {len(high_risk)} high-risk documents"
            )

        no_license = [r for r in self.copyright_results.values() if not r.license_info]
        if no_license:
            recommendations.append(
                f"Investigate licensing for {len(no_license)} documents without clear licenses"
            )

        return recommendations


async def main():
    """Example usage of copyright checker"""
    # Example documents
    documents = [
        {
            "doc_id": "doc1",
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/files/1342/1342-h/1342-h.htm",
            "domain": "literature",
            "content": "It is a truth universally acknowledged... This work is in the public domain.",
        },
        {
            "doc_id": "doc2",
            "title": "Modern AI Research Paper",
            "author": "Dr. Smith",
            "url": "https://example.com/paper.pdf",
            "domain": "science",
            "content": "Copyright © 2023 Dr. Smith. All rights reserved. This paper presents...",
        },
        {
            "doc_id": "doc3",
            "title": "Open Source Tutorial",
            "author": "John Doe",
            "url": "https://opensource.example.com/tutorial",
            "domain": "science",
            "content": "This tutorial is licensed under Creative Commons Attribution 4.0...",
        },
    ]

    checker = CopyrightChecker()

    try:
        # Check each document
        for doc in documents:
            status = await checker.check_copyright(doc)
            print(f"\nDocument: {status.title}")
            print(f"Copyright Free: {status.copyright_free}")
            print(f"Risk Level: {status.risk_level}")
            print(f"Confidence: {status.confidence:.2f}")
            if status.license_info:
                print(f"License: {status.license_info.license_name}")
            print(f"Notes: {status.notes}")

        # Generate compliance report
        report = checker.generate_compliance_report()
        print(f"\n--- Compliance Report ---")
        print(f"Total Documents: {report['summary']['total_documents']}")
        print(f"Copyright Free: {report['summary']['copyright_free']}")
        print(f"Compliance Rate: {report['summary']['compliance_rate']:.2%}")
        print(f"Risk Distribution: {report['risk_distribution']}")

        if report["recommendations"]:
            print(f"Recommendations:")
            for rec in report["recommendations"]:
                print(f"- {rec}")

    except Exception as e:
        logger.error(f"Copyright checking failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
