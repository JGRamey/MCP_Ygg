#!/usr/bin/env python3
"""Utility functions for Claim Analyzer Agent"""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_claim_id(text: str, source: str) -> str:
    """Generate a unique ID for a claim"""
    return hashlib.md5(f"{text}{source}".encode()).hexdigest()


def sanitize_text(text: str) -> str:
    """Sanitize input text for processing"""
    if not text:
        return ""

    # Remove potentially malicious patterns
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"data:image", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_input(text: str, max_length: int = 100000) -> bool:
    """Validate input text"""
    if not text or not isinstance(text, str):
        return False

    if len(text) > max_length:
        logger.warning(f"Text length {len(text)} exceeds maximum {max_length}")
        return False

    return True


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL"""
    if not url:
        return "unknown"

    pattern = r"https?://(?:www\.)?([^/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else "unknown"


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard index"""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def load_stopwords() -> set:
    """Load common stopwords for text processing"""
    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "the",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "said",
        "each",
        "which",
        "she",
        "do",
        "how",
        "their",
        "if",
        "up",
        "out",
        "many",
        "then",
        "them",
        "these",
        "so",
        "some",
        "her",
        "would",
        "make",
        "like",
        "into",
        "him",
        "time",
        "two",
        "more",
        "go",
        "no",
        "way",
        "could",
        "my",
        "than",
        "first",
        "been",
        "call",
        "who",
        "oil",
        "sit",
        "now",
        "find",
        "down",
        "day",
        "did",
        "get",
        "come",
        "made",
        "may",
        "part",
    }


def clean_claim_text(text: str) -> str:
    """Clean and normalize claim text"""
    if not text:
        return ""

    # Remove quotes and normalize
    text = re.sub(r'^["\']|["\']$', "", text.strip())

    # Normalize punctuation
    text = re.sub(r"[.!?]+$", ".", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    if not url:
        return False

    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return url_pattern.match(url) is not None


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_entities_simple(text: str) -> List[str]:
    """Simple entity extraction using patterns"""
    entities = []

    # Capitalize words (potential proper nouns)
    capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", text)
    entities.extend(capitalized_words)

    # Numbers and dates
    numbers = re.findall(r"\b\d{4}\b|\b\d+(?:\.\d+)?%?\b", text)
    entities.extend(numbers)

    # Simple organization patterns
    orgs = re.findall(r"\b[A-Z]{2,}\b", text)
    entities.extend(orgs)

    return list(set(entities))


def create_evidence_summary(evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary of evidence"""
    if not evidence_list:
        return {
            "total_sources": 0,
            "avg_credibility": 0.0,
            "stance_distribution": {},
            "domains": [],
        }

    total_sources = len(evidence_list)
    avg_credibility = (
        sum(e.get("credibility_score", 0) for e in evidence_list) / total_sources
    )

    stance_counts = {}
    domains = set()

    for evidence in evidence_list:
        stance = evidence.get("stance", "neutral")
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
        domains.add(evidence.get("domain", "unknown"))

    return {
        "total_sources": total_sources,
        "avg_credibility": round(avg_credibility, 2),
        "stance_distribution": stance_counts,
        "domains": list(domains),
    }


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_config = config.get("logging", {})

    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create logs directory if it doesn't exist
    log_file = log_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format=format_str)


class PerformanceTimer:
    """Simple performance timer for measuring execution time"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration.total_seconds():.2f} seconds")

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def batch_process(items: List[Any], batch_size: int = 50):
    """Generator for batch processing"""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def merge_dicts_deep(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value

    return result
