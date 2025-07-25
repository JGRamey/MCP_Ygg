#!/usr/bin/env python3
"""
MCP Server Metadata Analyzer Agent
Generates comprehensive metadata for scraped texts including keywords, word counts,
chapters, subjects, concepts, entities, and language detection
"""

import hashlib
import json
import logging
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import asyncio
import langdetect
import numpy as np
import spacy
import yaml
from langdetect.lang_detect_exception import LangDetectException
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from textstat import (
    automated_readability_index,
    flesch_kincaid_grade,
    flesch_reading_ease,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextMetadata:
    """Comprehensive metadata for a text document"""

    doc_id: str
    title: str
    author: Optional[str]

    # Basic statistics
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int

    # Language information
    language: str
    language_confidence: float

    # Content structure
    chapters: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]

    # Keywords and topics
    keywords: List[Tuple[str, float]]  # (keyword, score)
    key_phrases: List[Tuple[str, float]]
    topics: List[Dict[str, Any]]

    # Named entities
    entities: List[Dict[str, Any]]
    entity_summary: Dict[str, int]

    # Concepts and subjects
    subjects: List[str]
    concepts: List[Dict[str, Any]]

    # Readability metrics
    readability_scores: Dict[str, float]

    # Content quality indicators
    quality_scores: Dict[str, float]

    # Temporal information
    dates_mentioned: List[str]
    time_periods: List[str]

    # References and citations
    citations: List[Dict[str, Any]]
    references: List[str]

    # Generated metadata
    summary: str
    tags: List[str]

    # Processing metadata
    generated_at: datetime
    processing_time: float
    confidence_score: float


@dataclass
class Chapter:
    """Represents a chapter or major section"""

    number: Optional[int]
    title: str
    start_position: int
    end_position: int
    word_count: int
    summary: str


@dataclass
class Entity:
    """Named entity with additional metadata"""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: Optional[str]
    category: str
    context: str


@dataclass
class Concept:
    """Subject concept with relevance"""

    name: str
    category: str
    confidence: float
    frequency: int
    context_examples: List[str]
    related_entities: List[str]


class LanguageDetector:
    """Advanced language detection with domain-specific enhancements"""

    def __init__(self):
        """Initialize language detector"""
        self.ancient_language_patterns = {
            "latin": {
                "patterns": [
                    r"\b(et|in|ad|de|ex|cum|pro|per|ab|sine)\b",
                    r"\w+us\b",
                    r"\w+um\b",
                    r"\w+is\b",
                ],
                "common_words": [
                    "et",
                    "in",
                    "ad",
                    "de",
                    "ex",
                    "cum",
                    "pro",
                    "per",
                    "ab",
                    "sine",
                    "est",
                    "sunt",
                ],
            },
            "greek": {
                "patterns": [r"[α-ωΑ-Ω]", r"\b(καὶ|τοῦ|τῆς|τὸ|τὴν|εἰς|ἐν)\b"],
                "common_words": [
                    "καὶ",
                    "τοῦ",
                    "τῆς",
                    "τὸ",
                    "τὴν",
                    "εἰς",
                    "ἐν",
                    "τῶν",
                    "τὰ",
                    "τῷ",
                ],
            },
            "hebrew": {
                "patterns": [r"[א-ת]", r"\b(את|של|על|אל|כל|זה|הוא|לא|יש)\b"],
                "common_words": ["את", "של", "על", "אל", "כל", "זה", "הוא", "לא", "יש"],
            },
            "sanskrit": {
                "patterns": [r"[अ-ह]", r"\b(च|तत्|एव|तु|अथ|यत्|इति|सः|तस्य)\b"],
                "common_words": ["च", "तत्", "एव", "तु", "अथ", "यत्", "इति", "सः", "तस्य"],
            },
        }

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        if not text.strip():
            return "unknown", 0.0

        # First try modern language detection
        try:
            lang = langdetect.detect(text)
            # Get confidence by detecting multiple times
            detections = []
            for _ in range(5):
                try:
                    detections.append(langdetect.detect(text))
                except LangDetectException:
                    pass

            if detections:
                confidence = detections.count(lang) / len(detections)
                return lang, confidence
        except LangDetectException:
            pass

        # Try ancient language patterns
        for lang, patterns in self.ancient_language_patterns.items():
            score = self._calculate_ancient_language_score(text, patterns)
            if score > 0.3:
                return lang, score

        # Fallback to English with low confidence
        return "en", 0.1

    def _calculate_ancient_language_score(self, text: str, patterns: Dict) -> float:
        """Calculate score for ancient language patterns"""
        total_score = 0.0
        text_lower = text.lower()

        # Pattern matching
        for pattern in patterns["patterns"]:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_score += matches / len(text.split()) if text.split() else 0

        # Common words
        words = text_lower.split()
        common_word_count = sum(1 for word in words if word in patterns["common_words"])
        if words:
            total_score += common_word_count / len(words)

        return min(1.0, total_score)


class StructureAnalyzer:
    """Analyzes document structure (chapters, sections, etc.)"""

    def __init__(self):
        """Initialize structure analyzer"""
        self.chapter_patterns = [
            r"^(Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)[\s\.:]*(.*)$",
            r"^(\d+)[\s\.:]+(.+)$",
            r"^([IVXLCDM]+)[\s\.:]+(.+)$",
            r"^(BOOK|Book)\s+(\d+|[IVXLCDM]+)[\s\.:]*(.*)$",
        ]

        self.section_patterns = [
            r"^(\d+\.\d+)\s+(.+)$",
            r"^(Section|SECTION)\s+(\d+)[\s\.:]*(.*)$",
            r"^([A-Z][^a-z]*?)[\s\.:]*$",  # All caps headers
        ]

    def analyze_structure(
        self, text: str
    ) -> Tuple[List[Chapter], List[Dict[str, Any]]]:
        """Analyze document structure and extract chapters/sections"""
        lines = text.split("\n")
        chapters = []
        sections = []

        current_position = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for chapter patterns
            chapter_match = self._match_chapter_patterns(line)
            if chapter_match:
                chapter = Chapter(
                    number=chapter_match.get("number"),
                    title=chapter_match.get("title", line),
                    start_position=current_position,
                    end_position=current_position + len(line),
                    word_count=0,  # Will be calculated later
                    summary="",  # Will be generated later
                )
                chapters.append(chapter)

            # Check for section patterns
            section_match = self._match_section_patterns(line)
            if section_match and not chapter_match:
                section = {
                    "title": section_match.get("title", line),
                    "number": section_match.get("number"),
                    "position": current_position,
                    "line_number": i + 1,
                }
                sections.append(section)

            current_position += len(line) + 1  # +1 for newline

        # Calculate chapter word counts and summaries
        self._calculate_chapter_metrics(chapters, text)

        return chapters, sections

    def _match_chapter_patterns(self, line: str) -> Optional[Dict[str, Any]]:
        """Match line against chapter patterns"""
        for pattern in self.chapter_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return {
                        "number": (
                            self._parse_chapter_number(groups[1])
                            if len(groups) > 1
                            else None
                        ),
                        "title": (
                            groups[2].strip() if len(groups) > 2 else groups[1].strip()
                        ),
                    }
        return None

    def _match_section_patterns(self, line: str) -> Optional[Dict[str, Any]]:
        """Match line against section patterns"""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    "number": groups[0] if groups else None,
                    "title": groups[1].strip() if len(groups) > 1 else line.strip(),
                }
        return None

    def _parse_chapter_number(self, number_str: str) -> Optional[int]:
        """Parse chapter number from string (handles Roman numerals)"""
        try:
            # Try Arabic numeral first
            return int(number_str)
        except ValueError:
            # Try Roman numeral
            return self._roman_to_int(number_str.upper())

    def _roman_to_int(self, roman: str) -> Optional[int]:
        """Convert Roman numeral to integer"""
        roman_numerals = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        try:
            result = 0
            prev_value = 0
            for char in reversed(roman):
                value = roman_numerals.get(char, 0)
                if value < prev_value:
                    result -= value
                else:
                    result += value
                prev_value = value
            return result if result > 0 else None
        except:
            return None

    def _calculate_chapter_metrics(self, chapters: List[Chapter], text: str) -> None:
        """Calculate word counts and generate summaries for chapters"""
        for i, chapter in enumerate(chapters):
            # Determine chapter text boundaries
            start_pos = chapter.start_position
            if i + 1 < len(chapters):
                end_pos = chapters[i + 1].start_position
            else:
                end_pos = len(text)

            chapter_text = text[start_pos:end_pos]
            chapter.word_count = len(chapter_text.split())
            chapter.end_position = end_pos

            # Generate simple summary (first few sentences)
            sentences = re.split(r"[.!?]+", chapter_text)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            chapter.summary = (
                ". ".join(meaningful_sentences[:2]) + "."
                if meaningful_sentences
                else ""
            )


class KeywordExtractor:
    """Extracts keywords and key phrases from text"""

    def __init__(self):
        """Initialize keyword extractor"""
        self.stop_words = set(STOP_WORDS)
        self.stop_words.update(["said", "say", "says", "telling", "told", "tell"])

        # Domain-specific important terms
        self.domain_terms = {
            "religion": [
                "god",
                "divine",
                "sacred",
                "holy",
                "spiritual",
                "faith",
                "prayer",
                "scripture",
            ],
            "science": [
                "theory",
                "hypothesis",
                "experiment",
                "data",
                "analysis",
                "research",
                "study",
            ],
            "philosophy": [
                "truth",
                "knowledge",
                "existence",
                "consciousness",
                "reality",
                "ethics",
            ],
            "math": [
                "theorem",
                "proof",
                "equation",
                "formula",
                "calculation",
                "mathematics",
            ],
            "literature": [
                "character",
                "plot",
                "theme",
                "narrative",
                "story",
                "poetry",
                "drama",
            ],
            "history": [
                "civilization",
                "empire",
                "war",
                "culture",
                "society",
                "period",
                "era",
            ],
        }

    def extract_keywords(
        self, text: str, domain: Optional[str] = None, max_keywords: int = 20
    ) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF and domain knowledge"""
        if not text.strip():
            return []

        try:
            # TF-IDF extraction
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8,
            )

            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top TF-IDF terms
            tfidf_keywords = []
            for i, score in enumerate(tfidf_scores):
                if score > 0:
                    keyword = feature_names[i]
                    if self._is_valid_keyword(keyword):
                        tfidf_keywords.append((keyword, score))

            # Sort by score
            tfidf_keywords.sort(key=lambda x: x[1], reverse=True)

            # Frequency-based extraction
            freq_keywords = self._extract_frequency_keywords(text)

            # Domain-specific boosting
            if domain and domain in self.domain_terms:
                tfidf_keywords = self._boost_domain_terms(tfidf_keywords, domain)

            # Combine and deduplicate
            combined_keywords = self._combine_keyword_lists(
                tfidf_keywords, freq_keywords
            )

            return combined_keywords[:max_keywords]

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return self._extract_frequency_keywords(text)[:max_keywords]

    def extract_key_phrases(
        self, text: str, max_phrases: int = 15
    ) -> List[Tuple[str, float]]:
        """Extract key phrases (multi-word expressions)"""
        if not text.strip():
            return []

        # Noun phrase extraction patterns
        noun_phrase_patterns = [
            r"\b(?:[A-Z][a-z]+ )+[A-Z][a-z]+\b",  # Proper noun phrases
            r"\b(?:(?:the|a|an) )?(?:[a-z]+ )*[a-z]+(?:ing|ed|tion|sion|ment|ness|ity|ism)\b",  # Complex nouns
            r"\b[a-z]+ (?:of|in|on|at|by|for|with) [a-z]+\b",  # Prepositional phrases
        ]

        phrases = []

        for pattern in noun_phrase_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip().lower()
                if len(phrase.split()) >= 2 and self._is_valid_phrase(phrase):
                    phrases.append(phrase)

        # Count phrase frequencies
        phrase_counts = Counter(phrases)

        # Calculate scores (frequency normalized by phrase length)
        phrase_scores = []
        for phrase, count in phrase_counts.items():
            score = count / len(phrase.split())  # Normalize by length
            phrase_scores.append((phrase, score))

        # Sort by score
        phrase_scores.sort(key=lambda x: x[1], reverse=True)

        return phrase_scores[:max_phrases]

    def _extract_frequency_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords based on frequency"""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words = [word for word in words if word not in self.stop_words]

        word_counts = Counter(words)
        total_words = len(words)

        # Calculate normalized frequency scores
        freq_keywords = []
        for word, count in word_counts.items():
            if count > 1 and self._is_valid_keyword(word):  # Appear more than once
                score = count / total_words
                freq_keywords.append((word, score))

        freq_keywords.sort(key=lambda x: x[1], reverse=True)
        return freq_keywords

    def _is_valid_keyword(self, keyword: str) -> bool:
        """Check if keyword is valid"""
        return (
            len(keyword) >= 3
            and keyword.lower() not in self.stop_words
            and not keyword.isdigit()
            and not all(c in string.punctuation for c in keyword)
        )

    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if phrase is valid"""
        words = phrase.split()
        return (
            len(words) >= 2
            and len(words) <= 5
            and not any(word in self.stop_words for word in words)
            and not phrase.isdigit()
        )

    def _boost_domain_terms(
        self, keywords: List[Tuple[str, float]], domain: str
    ) -> List[Tuple[str, float]]:
        """Boost scores for domain-specific terms"""
        domain_terms = set(self.domain_terms.get(domain, []))
        boosted_keywords = []

        for keyword, score in keywords:
            if any(term in keyword for term in domain_terms):
                score *= 1.5  # Boost domain-relevant terms
            boosted_keywords.append((keyword, score))

        return boosted_keywords

    def _combine_keyword_lists(
        self, list1: List[Tuple[str, float]], list2: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Combine and deduplicate keyword lists"""
        keyword_scores = {}

        # Add first list
        for keyword, score in list1:
            keyword_scores[keyword] = score

        # Add second list, averaging scores for duplicates
        for keyword, score in list2:
            if keyword in keyword_scores:
                keyword_scores[keyword] = (keyword_scores[keyword] + score) / 2
            else:
                keyword_scores[keyword] = score

        # Convert back to list and sort
        combined = list(keyword_scores.items())
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined


class EntityAnalyzer:
    """Analyzes named entities in text"""

    def __init__(self):
        """Initialize entity analyzer"""
        self.nlp = None
        self.load_nlp_model()

        # Entity categories
        self.entity_categories = {
            "PERSON": "People",
            "ORG": "Organizations",
            "GPE": "Places",
            "LOC": "Locations",
            "EVENT": "Events",
            "WORK_OF_ART": "Works of Art",
            "LAW": "Laws",
            "LANGUAGE": "Languages",
            "DATE": "Dates",
            "TIME": "Times",
            "MONEY": "Money",
            "QUANTITY": "Quantities",
            "ORDINAL": "Ordinals",
            "CARDINAL": "Cardinals",
        }

    def load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy model for entity analysis")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded smaller spaCy model for entity analysis")
            except OSError:
                logger.warning("No spaCy model available for entity analysis")
                self.nlp = None

    def analyze_entities(self, text: str) -> Tuple[List[Entity], Dict[str, int]]:
        """Analyze named entities in text"""
        if not self.nlp or not text.strip():
            return [], {}

        try:
            doc = self.nlp(text)
            entities = []
            entity_counts = defaultdict(int)

            for ent in doc.ents:
                # Get context (surrounding words)
                start_context = max(0, ent.start - 10)
                end_context = min(len(doc), ent.end + 10)
                context = doc[start_context:end_context].text

                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores by default
                    description=spacy.explain(ent.label_),
                    category=self.entity_categories.get(ent.label_, "Other"),
                    context=context,
                )
                entities.append(entity)
                entity_counts[ent.label_] += 1

            return entities, dict(entity_counts)

        except Exception as e:
            logger.error(f"Entity analysis failed: {e}")
            return [], {}


class ConceptAnalyzer:
    """Analyzes concepts and subjects in text"""

    def __init__(self):
        """Initialize concept analyzer"""
        self.domain_concepts = {
            "religion": [
                "divine",
                "sacred",
                "holy",
                "spiritual",
                "faith",
                "belief",
                "prayer",
                "worship",
                "scripture",
                "doctrine",
                "theology",
                "salvation",
                "redemption",
                "grace",
                "sin",
                "virtue",
                "soul",
                "eternal",
                "heaven",
                "hell",
                "angel",
                "demon",
            ],
            "science": [
                "theory",
                "hypothesis",
                "experiment",
                "observation",
                "data",
                "analysis",
                "research",
                "method",
                "evidence",
                "discovery",
                "innovation",
                "technology",
                "evolution",
                "genetics",
                "physics",
                "chemistry",
                "biology",
                "astronomy",
            ],
            "philosophy": [
                "truth",
                "knowledge",
                "wisdom",
                "existence",
                "being",
                "consciousness",
                "reality",
                "perception",
                "ethics",
                "morality",
                "justice",
                "beauty",
                "good",
                "evil",
                "mind",
                "reason",
                "logic",
                "argument",
                "premise",
            ],
            "math": [
                "number",
                "equation",
                "formula",
                "theorem",
                "proof",
                "calculation",
                "algebra",
                "geometry",
                "calculus",
                "statistics",
                "probability",
                "infinity",
                "zero",
                "function",
                "variable",
                "constant",
                "set",
            ],
            "literature": [
                "narrative",
                "story",
                "character",
                "plot",
                "theme",
                "symbol",
                "metaphor",
                "poetry",
                "drama",
                "comedy",
                "tragedy",
                "epic",
                "novel",
                "verse",
                "prose",
                "style",
                "voice",
                "perspective",
                "archetype",
                "motif",
            ],
            "history": [
                "civilization",
                "culture",
                "society",
                "empire",
                "kingdom",
                "dynasty",
                "war",
                "battle",
                "revolution",
                "reform",
                "progress",
                "decline",
                "tradition",
                "custom",
                "legacy",
                "influence",
                "change",
                "development",
            ],
        }

    def analyze_concepts(
        self, text: str, domain: Optional[str] = None
    ) -> List[Concept]:
        """Analyze concepts in text"""
        concepts = []
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text_lower)

        # Determine relevant concept lists
        if domain and domain in self.domain_concepts:
            concept_lists = {domain: self.domain_concepts[domain]}
        else:
            concept_lists = self.domain_concepts

        for domain_name, domain_concepts in concept_lists.items():
            for concept_name in domain_concepts:
                if concept_name in text_lower:
                    # Calculate frequency and confidence
                    frequency = text_lower.count(concept_name)
                    confidence = min(1.0, frequency / 10.0)  # Normalize

                    # Find context examples
                    context_examples = self._find_concept_contexts(text, concept_name)

                    # Find related entities (simplified)
                    related_entities = self._find_related_entities(text, concept_name)

                    concept = Concept(
                        name=concept_name,
                        category=domain_name,
                        confidence=confidence,
                        frequency=frequency,
                        context_examples=context_examples[:3],  # Max 3 examples
                        related_entities=related_entities,
                    )
                    concepts.append(concept)

        # Sort by confidence
        concepts.sort(key=lambda x: x.confidence, reverse=True)

        return concepts

    def _find_concept_contexts(self, text: str, concept: str) -> List[str]:
        """Find context examples for a concept"""
        contexts = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if concept.lower() in sentence.lower():
                context = sentence.strip()
                if len(context) > 20:  # Meaningful context
                    contexts.append(context)

        return contexts

    def _find_related_entities(self, text: str, concept: str) -> List[str]:
        """Find entities related to a concept (simplified)"""
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        related = []

        # Find capitalized words near the concept
        pattern = rf"\b{re.escape(concept)}\b.{{0,50}}\b([A-Z][a-z]+)\b"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            entity = match.group(1)
            if entity not in related and len(entity) > 2:
                related.append(entity)

        return related[:5]  # Max 5 related entities


class ReadabilityAnalyzer:
    """Analyzes text readability and quality"""

    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores"""
        if not text.strip():
            return {}

        try:
            scores = {
                "flesch_reading_ease": flesch_reading_ease(text),
                "flesch_kincaid_grade": flesch_kincaid_grade(text),
                "automated_readability_index": automated_readability_index(text),
            }

            # Add custom metrics
            scores.update(self._calculate_custom_metrics(text))

            return scores

        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return self._calculate_custom_metrics(text)

    def _calculate_custom_metrics(self, text: str) -> Dict[str, float]:
        """Calculate custom readability metrics"""
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
            return {}

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Vocabulary diversity (type-token ratio)
        unique_words = set(word.lower() for word in words)
        vocabulary_diversity = len(unique_words) / len(words)

        # Complex word ratio (words with 3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_ratio = complex_words / len(words)

        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "vocabulary_diversity": vocabulary_diversity,
            "complex_word_ratio": complex_word_ratio,
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0

        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)


class MetadataAnalyzer:
    """Main metadata analysis orchestrator"""

    def __init__(self, config_path: str = "agents/metadata_analyzer/config.yaml"):
        """Initialize metadata analyzer"""
        self.load_config(config_path)

        # Initialize components
        self.language_detector = LanguageDetector()
        self.structure_analyzer = StructureAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.entity_analyzer = EntityAnalyzer()
        self.concept_analyzer = ConceptAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()

    def load_config(self, config_path: str) -> None:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "max_keywords": 20,
                "max_key_phrases": 15,
                "max_concepts": 25,
                "summary_sentences": 3,
                "enable_readability": True,
                "enable_structure_analysis": True,
                "output_dir": "data/metadata",
            }

    async def analyze_document(self, document_data: Dict[str, Any]) -> TextMetadata:
        """Analyze document and generate comprehensive metadata"""
        start_time = datetime.now()

        try:
            # Extract basic information
            doc_id = document_data.get("doc_id", "")
            title = document_data.get("title", "")
            author = document_data.get("author")
            content = document_data.get("cleaned_content", "") or document_data.get(
                "content", ""
            )
            domain = document_data.get("domain")

            if not content:
                raise ValueError("No content to analyze")

            # Language detection
            language, lang_confidence = self.language_detector.detect_language(content)

            # Basic statistics
            word_count = len(content.split())
            character_count = len(content)
            sentence_count = len(re.split(r"[.!?]+", content))
            paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

            # Structure analysis
            chapters, sections = [], []
            if self.config.get("enable_structure_analysis", True):
                chapters, sections = self.structure_analyzer.analyze_structure(content)

            # Keyword extraction
            keywords = self.keyword_extractor.extract_keywords(
                content, domain, self.config.get("max_keywords", 20)
            )
            key_phrases = self.keyword_extractor.extract_key_phrases(
                content, self.config.get("max_key_phrases", 15)
            )

            # Entity analysis
            entities_list, entity_summary = self.entity_analyzer.analyze_entities(
                content
            )
            entities = [asdict(entity) for entity in entities_list]

            # Concept analysis
            concepts_list = self.concept_analyzer.analyze_concepts(content, domain)
            concepts = [
                asdict(concept)
                for concept in concepts_list[: self.config.get("max_concepts", 25)]
            ]

            # Readability analysis
            readability_scores = {}
            if self.config.get("enable_readability", True):
                readability_scores = (
                    self.readability_analyzer.calculate_readability_scores(content)
                )

            # Quality scores
            quality_scores = self._calculate_quality_scores(
                content, word_count, sentence_count, lang_confidence
            )

            # Extract dates and time periods
            dates_mentioned = self._extract_dates(content)
            time_periods = self._extract_time_periods(content)

            # Extract citations and references
            citations, references = self._extract_citations_and_references(content)

            # Generate summary
            summary = self._generate_summary(
                content, self.config.get("summary_sentences", 3)
            )

            # Generate tags
            tags = self._generate_tags(keywords, concepts_list, domain)

            # Determine subjects
            subjects = self._determine_subjects(concepts_list, domain)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                lang_confidence, quality_scores, word_count
            )

            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create metadata object
            metadata = TextMetadata(
                doc_id=doc_id,
                title=title,
                author=author,
                word_count=word_count,
                character_count=character_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                language=language,
                language_confidence=lang_confidence,
                chapters=[asdict(chapter) for chapter in chapters],
                sections=sections,
                keywords=keywords,
                key_phrases=key_phrases,
                topics=[],  # Would implement topic modeling
                entities=entities,
                entity_summary=entity_summary,
                subjects=subjects,
                concepts=concepts,
                readability_scores=readability_scores,
                quality_scores=quality_scores,
                dates_mentioned=dates_mentioned,
                time_periods=time_periods,
                citations=citations,
                references=references,
                summary=summary,
                tags=tags,
                generated_at=datetime.now(),
                processing_time=processing_time,
                confidence_score=confidence_score,
            )

            # Save metadata
            await self._save_metadata(metadata)

            logger.info(
                f"Generated metadata for document {doc_id} in {processing_time:.2f}s"
            )
            return metadata

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            raise

    def _calculate_quality_scores(
        self, content: str, word_count: int, sentence_count: int, lang_confidence: float
    ) -> Dict[str, float]:
        """Calculate content quality indicators"""
        scores = {}

        # Content completeness (based on length)
        scores["completeness"] = min(1.0, word_count / 1000)  # Normalize to 1000 words

        # Language confidence
        scores["language_clarity"] = lang_confidence

        # Structure quality (sentence length variation)
        sentences = re.split(r"[.!?]+", content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = statistics.mean(sentence_lengths)
            length_variance = (
                statistics.variance(sentence_lengths)
                if len(sentence_lengths) > 1
                else 0
            )
            scores["structure_quality"] = min(
                1.0, 1.0 - abs(avg_length - 15) / 15
            )  # Optimal ~15 words
        else:
            scores["structure_quality"] = 0.0

        # Information density (unique words per total words)
        words = content.lower().split()
        if words:
            unique_words = len(set(words))
            scores["information_density"] = unique_words / len(words)
        else:
            scores["information_density"] = 0.0

        # Overall quality (weighted average)
        scores["overall_quality"] = (
            scores["completeness"] * 0.3
            + scores["language_clarity"] * 0.3
            + scores["structure_quality"] * 0.2
            + scores["information_density"] * 0.2
        )

        return scores

    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates mentioned in the text"""
        date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # MM-DD-YYYY
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # YYYY-MM-DD
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b\d{4}\s+(?:BCE?|CE?|AD)\b",  # Historical dates
            r"\b(?:19|20)\d{2}\b",  # Years
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            dates.extend([match.group() for match in matches])

        return list(set(dates))  # Remove duplicates

    def _extract_time_periods(self, content: str) -> List[str]:
        """Extract time periods mentioned in the text"""
        period_patterns = [
            r"\b(?:ancient|classical|medieval|renaissance|modern|contemporary)\b",
            r"\b(?:stone age|bronze age|iron age)\b",
            r"\b(?:paleolithic|neolithic|mesolithic)\b",
            r"\b(?:antiquity|middle ages|dark ages)\b",
            r"\b(?:enlightenment|industrial revolution)\b",
            r"\b\d+(?:st|nd|rd|th)\s+century\b",
        ]

        periods = []
        for pattern in period_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            periods.extend([match.group() for match in matches])

        return list(set(periods))

    def _extract_citations_and_references(
        self, content: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extract citations and references"""
        citations = []
        references = []

        # Citation patterns
        citation_patterns = [
            r"\([^)]*\d{4}[^)]*\)",  # (Author, 2020)
            r"\[[^\]]*\d+[^\]]*\]",  # [1], [Author, 2020]
            r"(?:according to|as stated by|cited in)\s+[A-Z][^.!?]*",
        ]

        for pattern in citation_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                citation_text = match.group().strip()
                citations.append(
                    {"text": citation_text, "position": match.start(), "type": "inline"}
                )

        # Reference patterns (simpler extraction)
        reference_patterns = [
            r"^[A-Z][^.]*\.\s*\(\d{4}\)",  # Author. (Year)
            r"^[A-Z][^.]*\d{4}[^.]*\.",  # Author Year.
        ]

        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            for pattern in reference_patterns:
                if re.match(pattern, line):
                    references.append(line)
                    break

        return citations, references

    def _generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return ""

        # Simple heuristic: take first sentence, longest sentence, and last meaningful sentence
        summary_sentences = []

        if sentences:
            summary_sentences.append(sentences[0])  # First sentence

        if len(sentences) > 1:
            # Find longest sentence
            longest = max(sentences[1:], key=len)
            if longest not in summary_sentences:
                summary_sentences.append(longest)

        if len(sentences) > 2 and len(summary_sentences) < max_sentences:
            # Last meaningful sentence
            last_sentence = sentences[-1]
            if last_sentence not in summary_sentences:
                summary_sentences.append(last_sentence)

        return ". ".join(summary_sentences[:max_sentences]) + "."

    def _generate_tags(
        self,
        keywords: List[Tuple[str, float]],
        concepts: List[Concept],
        domain: Optional[str],
    ) -> List[str]:
        """Generate tags from keywords and concepts"""
        tags = []

        # Add top keywords as tags
        for keyword, score in keywords[:10]:
            if score > 0.1:  # Minimum score threshold
                tags.append(keyword)

        # Add concept names as tags
        for concept in concepts[:10]:
            if concept.confidence > 0.3:
                tags.append(concept.name)

        # Add domain tag
        if domain:
            tags.append(domain)

        # Remove duplicates and return
        return list(set(tags))

    def _determine_subjects(
        self, concepts: List[Concept], domain: Optional[str]
    ) -> List[str]:
        """Determine main subjects of the document"""
        subjects = []

        # Add domain as primary subject
        if domain:
            subjects.append(domain)

        # Add high-confidence concepts as subjects
        for concept in concepts:
            if concept.confidence > 0.5:
                subjects.append(concept.category)

        # Remove duplicates
        return list(set(subjects))

    def _calculate_confidence_score(
        self, lang_confidence: float, quality_scores: Dict[str, float], word_count: int
    ) -> float:
        """Calculate overall confidence in the metadata"""
        # Base confidence from language detection
        confidence = lang_confidence * 0.3

        # Quality score contribution
        overall_quality = quality_scores.get("overall_quality", 0.0)
        confidence += overall_quality * 0.4

        # Content length contribution
        length_score = min(1.0, word_count / 500)  # Normalize to 500 words
        confidence += length_score * 0.3

        return min(1.0, confidence)

    async def _save_metadata(self, metadata: TextMetadata) -> None:
        """Save metadata to storage"""
        output_dir = Path(self.config.get("output_dir", "data/metadata"))
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = output_dir / f"{metadata.doc_id}_metadata.json"

        # Convert to dictionary for JSON serialization
        metadata_dict = asdict(metadata)
        metadata_dict["generated_at"] = metadata.generated_at.isoformat()

        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata_dict, indent=2, ensure_ascii=False))


async def main():
    """Example usage of metadata analyzer"""
    # Example document
    document_data = {
        "doc_id": "sample_doc_123",
        "title": "The Nature of Trinity in Christian Theology",
        "author": "Thomas Aquinas",
        "domain": "religion",
        "cleaned_content": """
        The Trinity is a fundamental doctrine in Christian theology that describes 
        God as three distinct persons - the Father, the Son, and the Holy Spirit - 
        who are one in essence and substance. This concept has been central to 
        Christian thought since the early church councils.
        
        Chapter 1: Historical Development
        The doctrine of the Trinity developed over several centuries through 
        theological debates and church councils. The Council of Nicaea in 325 AD 
        was particularly significant in establishing orthodox Christian belief.
        
        The relationship between the three persons of the Trinity is described 
        as perichoresis, meaning mutual indwelling or interpenetration. Each 
        person is fully divine while maintaining their distinct identity.
        """,
    }

    analyzer = MetadataAnalyzer()

    try:
        metadata = await analyzer.analyze_document(document_data)

        print(f"Metadata for: {metadata.title}")
        print(
            f"Language: {metadata.language} (confidence: {metadata.language_confidence:.2f})"
        )
        print(f"Word count: {metadata.word_count}")
        print(f"Chapters: {len(metadata.chapters)}")
        print(f"Keywords: {[kw[0] for kw in metadata.keywords[:5]]}")
        print(f"Entities: {len(metadata.entities)}")
        print(f"Concepts: {len(metadata.concepts)}")
        print(f"Summary: {metadata.summary}")
        print(f"Quality score: {metadata.quality_scores.get('overall_quality', 0):.2f}")
        print(f"Processing time: {metadata.processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Metadata analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
