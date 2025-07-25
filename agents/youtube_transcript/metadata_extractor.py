#!/usr/bin/env python3
"""
Metadata Extractor
Extracts and enriches metadata from YouTube videos and transcripts
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    """Data structure for extracted metadata"""

    academic_domains: List[str]
    topics: List[str]
    concepts: List[str]
    references: List[str]
    citations: List[str]
    temporal_markers: List[Dict[str, Any]]
    complexity_score: float
    educational_value: float
    content_categories: List[str]
    language_metrics: Dict[str, Any]


class MetadataExtractor:
    """
    Extracts and enriches metadata from YouTube videos and transcripts
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize metadata extractor"""
        self.config = config

        # Domain classification keywords
        self.domain_keywords = {
            "art": [
                "painting",
                "sculpture",
                "gallery",
                "museum",
                "artist",
                "artwork",
                "canvas",
                "exhibition",
                "aesthetic",
                "visual",
                "design",
                "creative",
            ],
            "science": [
                "research",
                "experiment",
                "hypothesis",
                "theory",
                "data",
                "analysis",
                "biology",
                "chemistry",
                "physics",
                "mathematics",
                "scientific",
                "study",
            ],
            "philosophy": [
                "ethics",
                "metaphysics",
                "logic",
                "philosophy",
                "philosopher",
                "argument",
                "reasoning",
                "moral",
                "virtue",
                "truth",
                "knowledge",
                "wisdom",
            ],
            "technology": [
                "software",
                "hardware",
                "programming",
                "algorithm",
                "computer",
                "digital",
                "artificial intelligence",
                "machine learning",
                "coding",
                "technology",
            ],
            "language": [
                "linguistics",
                "grammar",
                "vocabulary",
                "literature",
                "writing",
                "language",
                "communication",
                "rhetoric",
                "poetry",
                "prose",
                "narrative",
                "discourse",
            ],
            "mathematics": [
                "equation",
                "formula",
                "calculation",
                "theorem",
                "proof",
                "geometry",
                "algebra",
                "calculus",
                "statistics",
                "probability",
                "mathematical",
            ],
        }

        # Academic reference patterns
        self.reference_patterns = [
            r"(?:doi|DOI):\s*([^\s]+)",
            r"(?:isbn|ISBN):\s*([0-9\-X]+)",
            r"(?:arxiv|arXiv):\s*([^\s]+)",
            r"(?:https?://)?(?:www\.)?(?:jstor|pubmed|scholar\.google)",
            r"\b(?:et al\.?|and others)\b",
            r"\(\d{4}\)",  # Publication years
            r"[A-Z][a-z]+,\s+[A-Z]\.",  # Author citations
        ]

        # Temporal markers
        self.temporal_patterns = [
            r"\b(?:in\s+)?(\d{4})\b",  # Years
            r"\b(\d{1,2}th|1st|2nd|3rd)\s+century\b",  # Centuries
            r"\b(ancient|medieval|renaissance|modern|contemporary)\b",  # Periods
            r"\b(BC|AD|BCE|CE)\b",  # Era markers
        ]

        logger.info("Metadata extractor initialized")

    async def extract_metadata(
        self, video_info, transcript_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from video and transcript
        """
        try:
            metadata = {
                "video_metadata": self._extract_video_metadata(video_info),
                "content_analysis": {},
                "academic_classification": {},
                "temporal_analysis": {},
                "quality_metrics": {},
                "extraction_metadata": {
                    "extracted_at": datetime.utcnow().isoformat(),
                    "extractor_version": "1.0.0",
                    "has_transcript": transcript_data is not None,
                },
            }

            # Analyze transcript if available
            if transcript_data:
                text_content = transcript_data.get("cleaned_text", "")

                # Content analysis
                metadata["content_analysis"] = await self._analyze_content(
                    text_content, video_info
                )

                # Academic classification
                metadata["academic_classification"] = self._classify_academic_domains(
                    text_content, video_info
                )

                # Temporal analysis
                metadata["temporal_analysis"] = self._analyze_temporal_content(
                    text_content
                )

                # Quality metrics
                metadata["quality_metrics"] = self._calculate_quality_metrics(
                    transcript_data, video_info
                )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _extract_video_metadata(self, video_info) -> Dict[str, Any]:
        """
        Extract structured metadata from video information
        """
        try:
            return {
                "basic_info": {
                    "video_id": video_info.video_id,
                    "title": video_info.title,
                    "channel_name": video_info.channel_name,
                    "duration_seconds": int(video_info.duration),
                    "published_date": video_info.published_at,
                    "view_count": video_info.view_count,
                    "language": video_info.language,
                },
                "engagement_metrics": {
                    "like_count": video_info.like_count,
                    "comment_count": video_info.comment_count,
                    "views_per_day": self._calculate_views_per_day(
                        video_info.view_count, video_info.published_at
                    ),
                },
                "content_indicators": {
                    "has_chapters": len(video_info.chapters) > 0,
                    "chapter_count": len(video_info.chapters),
                    "has_captions": video_info.captions_available,
                    "available_languages": video_info.transcript_languages,
                    "tag_count": len(video_info.tags),
                    "description_length": len(video_info.description),
                },
                "categorization": {
                    "youtube_category": video_info.category,
                    "tags": video_info.tags[:20],  # Limit tags
                    "channel_type": self._classify_channel_type(
                        video_info.channel_name
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {}

    async def _analyze_content(self, text: str, video_info) -> Dict[str, Any]:
        """
        Analyze content for topics, concepts, and themes
        """
        try:
            # Extract topics using keyword analysis
            topics = self._extract_topics(text)

            # Extract concepts
            concepts = self._extract_concepts(text)

            # Find references and citations
            references = self._find_references(text)

            # Analyze complexity
            complexity_score = self._calculate_complexity_score(text)

            # Educational value assessment
            educational_value = self._assess_educational_value(text, video_info)

            # Content categories
            categories = self._categorize_content(text, video_info)

            return {
                "topics": topics,
                "concepts": concepts,
                "references": references,
                "complexity_score": complexity_score,
                "educational_value": educational_value,
                "content_categories": categories,
                "word_density": self._calculate_word_density(text),
                "readability_score": self._calculate_readability(text),
            }

        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {}

    def _classify_academic_domains(self, text: str, video_info) -> Dict[str, Any]:
        """
        Classify content into academic domains
        """
        domain_scores = {}
        text_lower = text.lower()
        title_lower = video_info.title.lower()
        description_lower = video_info.description.lower()

        # Combine text sources with different weights
        combined_text = f"{title_lower} {title_lower} {description_lower} {text_lower}"

        for domain, keywords in self.domain_keywords.items():
            score = 0
            keyword_matches = []

            for keyword in keywords:
                # Count occurrences with position weighting
                title_matches = title_lower.count(keyword) * 3  # Title weight
                desc_matches = (
                    description_lower.count(keyword) * 2
                )  # Description weight
                text_matches = text_lower.count(keyword)  # Text weight

                total_matches = title_matches + desc_matches + text_matches

                if total_matches > 0:
                    score += total_matches
                    keyword_matches.append(
                        {
                            "keyword": keyword,
                            "matches": total_matches,
                            "in_title": title_matches > 0,
                            "in_description": desc_matches > 0,
                        }
                    )

            if score > 0:
                domain_scores[domain] = {
                    "score": score,
                    "confidence": min(score / 10, 1.0),  # Normalize to 0-1
                    "matched_keywords": keyword_matches,
                }

        # Determine primary domains
        if domain_scores:
            sorted_domains = sorted(
                domain_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )
            primary_domain = sorted_domains[0][0]
            secondary_domains = [d[0] for d in sorted_domains[1:3] if d[1]["score"] > 2]
        else:
            primary_domain = "general"
            secondary_domains = []

        return {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "domain_scores": domain_scores,
            "classification_confidence": domain_scores.get(primary_domain, {}).get(
                "confidence", 0.0
            ),
        }

    def _analyze_temporal_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze temporal references in content
        """
        temporal_markers = []

        for pattern in self.temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_markers.append(
                    {
                        "text": match.group(0),
                        "position": match.start(),
                        "type": self._classify_temporal_marker(match.group(0)),
                    }
                )

        # Analyze historical focus
        historical_periods = self._identify_historical_periods(temporal_markers)

        return {
            "temporal_markers": temporal_markers[:20],  # Limit results
            "historical_periods": historical_periods,
            "temporal_scope": self._determine_temporal_scope(temporal_markers),
            "chronological_focus": self._analyze_chronological_focus(temporal_markers),
        }

    def _calculate_quality_metrics(
        self, transcript_data: Dict[str, Any], video_info
    ) -> Dict[str, Any]:
        """
        Calculate content quality metrics
        """
        try:
            transcript_metadata = transcript_data.get("processing_metadata", {})

            # Basic metrics
            word_count = transcript_data.get("word_count", 0)
            duration = float(video_info.duration)
            speaking_rate = (
                word_count / (duration / 60) if duration > 0 else 0
            )  # words per minute

            # Quality indicators
            auto_generated = transcript_metadata.get("is_auto_generated", True)
            confidence_score = transcript_metadata.get("confidence_avg", 0.5)

            # Content quality
            sentiment_score = transcript_data.get("sentiment_score", 0.0)
            key_phrases_count = len(transcript_data.get("key_phrases", []))

            # Educational quality indicators
            has_references = (
                len(self._find_references(transcript_data.get("cleaned_text", ""))) > 0
            )
            complexity = self._calculate_complexity_score(
                transcript_data.get("cleaned_text", "")
            )

            # Overall quality score
            quality_factors = [
                (not auto_generated) * 0.2,  # Manual transcripts are higher quality
                min(confidence_score, 1.0) * 0.15,
                min(abs(sentiment_score) * 2, 1.0)
                * 0.1,  # Neutral to positive sentiment
                min(key_phrases_count / 10, 1.0) * 0.15,
                has_references * 0.2,
                min(complexity, 1.0) * 0.2,
            ]

            overall_quality = sum(quality_factors)

            return {
                "word_count": word_count,
                "speaking_rate_wpm": round(speaking_rate, 1),
                "transcript_quality": {
                    "is_auto_generated": auto_generated,
                    "confidence_score": confidence_score,
                    "quality_score": round(overall_quality, 2),
                },
                "content_quality": {
                    "sentiment_score": sentiment_score,
                    "key_phrases_count": key_phrases_count,
                    "has_references": has_references,
                    "complexity_score": complexity,
                },
                "educational_indicators": {
                    "structured_content": len(transcript_data.get("chapters", [])) > 1,
                    "concept_density": key_phrases_count / max(word_count / 100, 1),
                    "reference_density": len(
                        self._find_references(transcript_data.get("cleaned_text", ""))
                    )
                    / max(word_count / 1000, 1),
                },
            }

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text
        """
        # Simple topic extraction using keyword frequency
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        word_freq = {}

        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Filter out common words and get top topics
        common_words = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "from",
            "they",
            "been",
            "have",
            "their",
            "said",
            "each",
            "which",
            "make",
            "like",
            "into",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "also",
            "only",
            "know",
            "take",
            "people",
            "could",
            "year",
            "work",
            "well",
            "world",
            "think",
            "still",
            "after",
            "being",
            "now",
            "made",
            "before",
            "through",
            "where",
            "much",
            "good",
            "never",
            "again",
            "right",
            "little",
            "these",
            "going",
            "something",
            "important",
            "different",
            "using",
            "really",
            "things",
            "about",
            "other",
            "between",
            "because",
            "should",
            "looking",
            "need",
            "there",
            "actually",
            "pretty",
            "getting",
            "doing",
            "might",
            "want",
            "what",
            "even",
            "more",
            "most",
            "some",
            "many",
            "than",
            "over",
            "such",
            "both",
            "those",
            "does",
            "would",
            "while",
            "first",
            "second",
            "third",
            "example",
            "part",
            "next",
            "last",
            "video",
            "today",
            "back",
            "going",
            "show",
            "talk",
            "tell",
            "look",
            "see",
        }

        topics = []
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if word not in common_words and freq >= 3 and len(topics) < 20:
                topics.append(word)

        return topics

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text
        """
        # Find noun phrases and technical terms
        concepts = []

        # Pattern for concepts (capitalized terms, technical phrases)
        concept_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized phrases
            r"\b[a-z]+(?:-[a-z]+)+\b",  # Hyphenated terms
            r"\b[a-z]+(?:ology|ism|tion|sion|ment|ness)\b",  # Academic suffixes
        ]

        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3 and match.lower() not in [
                    "this",
                    "that",
                    "with",
                    "from",
                ]:
                    concepts.append(match.lower())

        # Remove duplicates and limit
        unique_concepts = list(set(concepts))
        return unique_concepts[:30]

    def _find_references(self, text: str) -> List[str]:
        """
        Find academic references and citations
        """
        references = []

        for pattern in self.reference_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)

        return references[:20]  # Limit results

    def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate text complexity score
        """
        if not text:
            return 0.0

        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        if not words or not sentences:
            return 0.0

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Technical term density
        technical_words = len([w for w in words if len(w) > 8])
        technical_density = technical_words / len(words)

        # Combine factors
        complexity = (
            (avg_word_length - 4) / 4 * 0.3  # Normalize around 4-letter words
            + (avg_sentence_length - 15)
            / 15
            * 0.4  # Normalize around 15 words/sentence
            + technical_density * 0.3
        )

        return max(0.0, min(1.0, complexity))

    def _assess_educational_value(self, text: str, video_info) -> float:
        """
        Assess educational value of content
        """
        factors = []

        # Length factor (longer content often more educational)
        word_count = len(text.split())
        length_factor = min(word_count / 1000, 1.0)
        factors.append(length_factor * 0.2)

        # Structure factor (chapters, organization)
        has_structure = len(video_info.chapters) > 0
        factors.append(has_structure * 0.3)

        # Reference factor
        references = self._find_references(text)
        ref_factor = min(len(references) / 5, 1.0)
        factors.append(ref_factor * 0.3)

        # Academic language factor
        academic_keywords = [
            "research",
            "study",
            "analysis",
            "theory",
            "hypothesis",
            "evidence",
            "conclusion",
            "methodology",
            "data",
            "experiment",
        ]
        academic_count = sum(
            1 for keyword in academic_keywords if keyword in text.lower()
        )
        academic_factor = min(academic_count / 5, 1.0)
        factors.append(academic_factor * 0.2)

        return sum(factors)

    def _categorize_content(self, text: str, video_info) -> List[str]:
        """
        Categorize content type
        """
        categories = []

        text_lower = text.lower()
        title_lower = video_info.title.lower()

        # Educational categories
        if any(
            word in title_lower for word in ["lecture", "course", "lesson", "tutorial"]
        ):
            categories.append("educational")

        if any(
            word in title_lower for word in ["interview", "discussion", "conversation"]
        ):
            categories.append("interview")

        if any(word in text_lower for word in ["research", "study", "experiment"]):
            categories.append("research")

        if any(word in title_lower for word in ["documentary", "history", "biography"]):
            categories.append("documentary")

        if any(word in title_lower for word in ["review", "analysis", "critique"]):
            categories.append("analysis")

        return categories

    def _calculate_views_per_day(self, view_count: int, published_date: str) -> float:
        """
        Calculate average views per day
        """
        try:
            published = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            days_since_published = (datetime.now() - published).days

            if days_since_published > 0:
                return view_count / days_since_published
            else:
                return view_count

        except Exception:
            return 0.0

    def _classify_channel_type(self, channel_name: str) -> str:
        """
        Classify the type of YouTube channel
        """
        name_lower = channel_name.lower()

        if any(
            word in name_lower for word in ["university", "college", "edu", "academy"]
        ):
            return "academic"
        elif any(
            word in name_lower for word in ["tv", "news", "media", "broadcasting"]
        ):
            return "media"
        elif any(word in name_lower for word in ["tech", "science", "research"]):
            return "technical"
        else:
            return "general"

    def _calculate_word_density(self, text: str) -> Dict[str, float]:
        """
        Calculate word density metrics
        """
        words = text.split()
        if not words:
            return {}

        unique_words = set(words)

        return {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "lexical_diversity": len(unique_words) / len(words),
            "repetition_rate": 1 - (len(unique_words) / len(words)),
        }

    def _calculate_readability(self, text: str) -> float:
        """
        Calculate readability score (simplified Flesch-Kincaid)
        """
        sentences = re.split(r"[.!?]+", text)
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)

        if not sentences or not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch Reading Ease
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        return max(0.0, min(100.0, score)) / 100.0  # Normalize to 0-1

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (approximation)
        """
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Handle silent e
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)  # Every word has at least one syllable

    def _classify_temporal_marker(self, text: str) -> str:
        """
        Classify type of temporal marker
        """
        if re.match(r"\d{4}", text):
            return "year"
        elif "century" in text.lower():
            return "century"
        elif text.upper() in ["BC", "AD", "BCE", "CE"]:
            return "era"
        elif text.lower() in [
            "ancient",
            "medieval",
            "renaissance",
            "modern",
            "contemporary",
        ]:
            return "period"
        else:
            return "other"

    def _identify_historical_periods(
        self, temporal_markers: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify historical periods mentioned
        """
        periods = set()

        for marker in temporal_markers:
            if marker["type"] == "period":
                periods.add(marker["text"].lower())
            elif marker["type"] == "year":
                year = int(marker["text"])
                if year < 500:
                    periods.add("ancient")
                elif year < 1500:
                    periods.add("medieval")
                elif year < 1800:
                    periods.add("early modern")
                elif year < 1950:
                    periods.add("modern")
                else:
                    periods.add("contemporary")

        return list(periods)

    def _determine_temporal_scope(self, temporal_markers: List[Dict[str, Any]]) -> str:
        """
        Determine the temporal scope of the content
        """
        years = []

        for marker in temporal_markers:
            if marker["type"] == "year":
                try:
                    years.append(int(marker["text"]))
                except ValueError:
                    continue

        if not years:
            return "timeless"

        year_range = max(years) - min(years)

        if year_range == 0:
            return "specific"
        elif year_range < 50:
            return "narrow"
        elif year_range < 200:
            return "moderate"
        else:
            return "broad"

    def _analyze_chronological_focus(
        self, temporal_markers: List[Dict[str, Any]]
    ) -> str:
        """
        Analyze the chronological focus of the content
        """
        years = []
        current_year = datetime.now().year

        for marker in temporal_markers:
            if marker["type"] == "year":
                try:
                    years.append(int(marker["text"]))
                except ValueError:
                    continue

        if not years:
            return "present"

        avg_year = sum(years) / len(years)

        if avg_year < 1500:
            return "ancient/medieval"
        elif avg_year < 1900:
            return "historical"
        elif avg_year < current_year - 20:
            return "recent_past"
        else:
            return "contemporary"
