#!/usr/bin/env python3
"""
Scraper Utility Functions
Helper functions for OCR, rate limiting, text processing, and more
"""

import hashlib
import json
import logging
import mimetypes
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiofiles
import aiohttp
import asyncio
import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, max_tokens: int = 10, refill_period: float = 1.0):
        """
        Initialize rate limiter

        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_period: Time in seconds to refill one token
        """
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_period = refill_period
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary"""
        async with self._lock:
            while self.tokens < tokens:
                await self._refill()
                if self.tokens < tokens:
                    wait_time = self.refill_period * (tokens - self.tokens)
                    await asyncio.sleep(wait_time)

            self.tokens -= tokens

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed / self.refill_period

        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now


class CircuitBreaker:
    """Circuit breaker pattern for handling service failures"""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting reset
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.reset_timeout

    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ImageProcessor:
    """Advanced image processing for OCR enhancement"""

    @staticmethod
    def enhance_image_for_ocr(image_path: str, output_path: str = None) -> str:
        """Enhance image quality for better OCR results"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Deskew if needed
            cleaned = ImageProcessor._deskew_image(cleaned)

            # Save enhanced image
            if output_path is None:
                output_path = image_path.replace(".", "_enhanced.")

            cv2.imwrite(output_path, cleaned)
            return output_path

        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image_path  # Return original if enhancement fails

    @staticmethod
    def _deskew_image(image: np.ndarray) -> np.ndarray:
        """Detect and correct skew in image"""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return image

            # Find the largest contour (assumed to be text)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]

            # Correct angle
            if angle < -45:
                angle = 90 + angle

            # Only correct if angle is significant
            if abs(angle) > 0.5:
                height, width = image.shape
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (width, height))

            return image

        except Exception:
            return image  # Return original if deskewing fails

    @staticmethod
    def extract_regions_of_interest(image_path: str) -> List[Tuple[int, int, int, int]]:
        """Extract text regions from image"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
            dilated = cv2.dilate(image, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 20:  # Filter small regions
                    regions.append((x, y, x + w, y + h))

            return regions

        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            return []


class AdvancedOCR:
    """Advanced OCR with multiple engines and post-processing"""

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.confidence_threshold = 30  # Minimum confidence for text

    def extract_text_with_confidence(
        self, image_path: str, language: str = "eng"
    ) -> Dict:
        """Extract text with confidence scores"""
        try:
            # Enhance image first
            enhanced_path = self.image_processor.enhance_image_for_ocr(image_path)

            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {language}"

            # Extract text with detailed output
            data = pytesseract.image_to_data(
                enhanced_path, config=config, output_type=pytesseract.Output.DICT
            )

            # Filter by confidence
            filtered_text = []
            word_confidences = []

            for i, confidence in enumerate(data["conf"]):
                if int(confidence) > self.confidence_threshold:
                    text = data["text"][i].strip()
                    if text:
                        filtered_text.append(text)
                        word_confidences.append(int(confidence))

            result = {
                "text": " ".join(filtered_text),
                "confidence": (
                    sum(word_confidences) / len(word_confidences)
                    if word_confidences
                    else 0
                ),
                "word_count": len(filtered_text),
                "low_confidence_words": len(
                    [c for c in data["conf"] if 0 < int(c) <= self.confidence_threshold]
                ),
            }

            # Clean up enhanced image if it's different from original
            if enhanced_path != image_path:
                Path(enhanced_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0,
                "word_count": 0,
                "low_confidence_words": 0,
            }

    def extract_text_from_regions(
        self, image_path: str, language: str = "eng"
    ) -> List[Dict]:
        """Extract text from specific regions of image"""
        regions = self.image_processor.extract_regions_of_interest(image_path)
        results = []

        try:
            image = cv2.imread(image_path)

            for i, (x1, y1, x2, y2) in enumerate(regions):
                # Crop region
                region = image[y1:y2, x1:x2]
                region_path = f"/tmp/region_{i}_{hash(image_path) % 10000}.png"
                cv2.imwrite(region_path, region)

                # Extract text from region
                result = self.extract_text_with_confidence(region_path, language)
                result["bbox"] = (x1, y1, x2, y2)
                results.append(result)

                # Clean up
                Path(region_path).unlink(missing_ok=True)

            return results

        except Exception as e:
            logger.error(f"Region OCR failed: {e}")
            return []


class TextCleaner:
    """Clean and normalize extracted text"""

    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Clean common OCR errors and artifacts"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common OCR character errors
        replacements = {
            r"\|": "l",  # Vertical bar to lowercase l
            r"0(?=[a-zA-Z])": "O",  # Zero to O when followed by letter
            r"(?<=[a-zA-Z])0": "o",  # Zero to o when preceded by letter
            r"rn": "m",  # Common OCR error
            r"([a-z])1([a-z])": r"\1l\2",  # 1 to l between lowercase letters
            r"([A-Z])1([a-z])": r"\1l\2",  # 1 to l between mixed case
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove isolated single characters (likely OCR noise)
        text = re.sub(r"\b[^aAI]\b", "", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])\s*", r"\1 ", text)

        # Normalize quotes
        text = re.sub(r'[""' "`]", '"', text)

        return text.strip()

    @staticmethod
    def extract_metadata_from_text(text: str) -> Dict:
        """Extract metadata like title, author, date from text"""
        metadata = {
            "title": "",
            "author": "",
            "date": "",
            "chapter": "",
            "page_number": "",
        }

        lines = text.split("\n")[:10]  # Check first 10 lines

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Title detection (usually first non-empty line in caps or title case)
            if not metadata["title"] and (line.isupper() or line.istitle()):
                metadata["title"] = line

            # Author detection
            if re.search(r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", line, re.IGNORECASE):
                match = re.search(
                    r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", line, re.IGNORECASE
                )
                metadata["author"] = match.group(1)

            # Date detection
            date_patterns = [
                r"\b(19|20)\d{2}\b",  # Years
                r"\b\d{1,2}/\d{1,2}/(19|20)\d{2}\b",  # MM/DD/YYYY
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b",
            ]

            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and not metadata["date"]:
                    metadata["date"] = match.group(0)
                    break

            # Chapter detection
            if re.search(r"chapter\s+\d+", line, re.IGNORECASE):
                metadata["chapter"] = line

            # Page number detection
            if re.search(r"page\s+\d+", line, re.IGNORECASE):
                metadata["page_number"] = line

        return metadata

    @staticmethod
    def detect_language_from_text(text: str) -> str:
        """Simple language detection based on character patterns"""
        if not text:
            return "unknown"

        # Character frequency analysis
        char_counts = {"latin": 0, "greek": 0, "cyrillic": 0, "arabic": 0, "chinese": 0}

        for char in text:
            code = ord(char)

            if 0x0020 <= code <= 0x007F or 0x00A0 <= code <= 0x00FF:
                char_counts["latin"] += 1
            elif 0x0370 <= code <= 0x03FF:
                char_counts["greek"] += 1
            elif 0x0400 <= code <= 0x04FF:
                char_counts["cyrillic"] += 1
            elif 0x0600 <= code <= 0x06FF:
                char_counts["arabic"] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                char_counts["chinese"] += 1

        # Find dominant script
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return "unknown"

        dominant_script = max(char_counts, key=char_counts.get)
        dominant_ratio = char_counts[dominant_script] / total_chars

        if dominant_ratio < 0.7:
            return "mixed"

        # Map script to language (simplified)
        script_to_language = {
            "latin": "english",  # Default to English for Latin script
            "greek": "greek",
            "cyrillic": "russian",
            "arabic": "arabic",
            "chinese": "chinese",
        }

        return script_to_language.get(dominant_script, "unknown")


class DuplicateDetector:
    """Detect and handle duplicate content"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.content_cache: Dict[str, str] = {}

    def calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for duplicate detection"""
        # Normalize content for comparison
        normalized = re.sub(r"\s+", " ", content.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check if content is duplicate"""
        content_hash = self.calculate_content_hash(content)

        # Exact duplicate check
        if content_hash in self.seen_hashes:
            return True, content_hash

        # Similarity check against cached content
        for cached_hash, cached_content in self.content_cache.items():
            similarity = self.calculate_similarity(content, cached_content)
            if similarity >= self.similarity_threshold:
                return True, cached_hash

        # Add to cache if not duplicate
        self.seen_hashes.add(content_hash)

        # Limit cache size
        if len(self.content_cache) > 1000:
            # Remove oldest entries
            oldest_hash = next(iter(self.content_cache))
            del self.content_cache[oldest_hash]

        self.content_cache[content_hash] = content[:10000]  # Store first 10k chars

        return False, None


class URLValidator:
    """Validate and categorize URLs"""

    ACADEMIC_DOMAINS = {
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ieee.org",
        "acm.org",
        "jstor.org",
        "springer.com",
        "sciencedirect.com",
        "nature.com",
        "scholar.google.com",
        "researchgate.net",
    }

    PUBLIC_DOMAIN_DOMAINS = {
        "gutenberg.org",
        "archive.org",
        "wikisource.org",
        "sacred-texts.com",
        "perseus.tufts.edu",
        "ccel.org",
        "bartleby.com",
    }

    @classmethod
    def is_valid_url(cls, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @classmethod
    def categorize_source(cls, url: str) -> str:
        """Categorize URL by source type"""
        domain = urlparse(url).netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        if domain in cls.ACADEMIC_DOMAINS:
            return "academic"
        elif domain in cls.PUBLIC_DOMAIN_DOMAINS:
            return "public_domain"
        else:
            return "unknown"

    @classmethod
    def get_file_type(cls, url: str) -> str:
        """Determine file type from URL"""
        path = urlparse(url).path.lower()

        if path.endswith(".pdf"):
            return "pdf"
        elif path.endswith((".html", ".htm")):
            return "html"
        elif path.endswith(".txt"):
            return "txt"
        elif path.endswith((".doc", ".docx")):
            return "doc"
        elif path.endswith(".epub"):
            return "epub"
        else:
            return "unknown"


# Example usage functions
async def process_with_rate_limiting():
    """Example of using rate limiter"""
    limiter = RateLimiter(max_tokens=5, refill_period=1.0)

    for i in range(10):
        await limiter.acquire()
        print(f"Processing request {i}")
        # Simulate work
        await asyncio.sleep(0.1)


def enhance_and_ocr_image(image_path: str, language: str = "eng"):
    """Example of image enhancement and OCR"""
    processor = ImageProcessor()
    ocr = AdvancedOCR()

    # Enhance image
    enhanced_path = processor.enhance_image_for_ocr(image_path)

    # Extract text with confidence
    result = ocr.extract_text_with_confidence(enhanced_path, language)

    # Clean text
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_ocr_text(result["text"])

    return {
        "text": cleaned_text,
        "confidence": result["confidence"],
        "metadata": cleaner.extract_metadata_from_text(cleaned_text),
    }


if __name__ == "__main__":
    # Example usage
    print("Scraper utilities loaded successfully")
