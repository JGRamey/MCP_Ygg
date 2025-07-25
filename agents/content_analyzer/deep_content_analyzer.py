#!/usr/bin/env python3
"""
Deep Content Analyzer for MCP Yggdrasil
Phase 4: NLP analysis using spaCy and transformers for content understanding
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import asyncio

try:
    import spacy
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Some features will be limited.")

# Import from Phase 4 agents
from ..scraper.intelligent_scraper_agent import ScrapedDocument

logger = logging.getLogger(__name__)


@dataclass
class EntityExtraction:
    """Extracted entity with metadata."""

    text: str
    type: str
    start_pos: int
    end_pos: int
    confidence: float
    linked_concept_id: Optional[str] = None


@dataclass
class ConceptExtraction:
    """High-level concept extraction."""

    name: str
    domain: str
    confidence: float
    context: str
    related_entities: List[str]


@dataclass
class ClaimExtraction:
    """Extracted claim or assertion."""

    claim_text: str
    confidence: float
    supporting_entities: List[str]
    claim_type: str  # factual, opinion, hypothesis
    verifiable: bool


@dataclass
class ContentAnalysis:
    """Complete content analysis results."""

    entities: List[EntityExtraction]
    concepts: List[ConceptExtraction]
    claims: List[ClaimExtraction]
    domain_mapping: Dict[str, float]
    key_topics: List[str]
    sentiment_analysis: Dict
    summary: str


class DeepContentAnalyzer:
    """Deep NLP analysis using spaCy and transformers."""

    def __init__(self):
        # Initialize NLP models with fallback
        try:
            self.nlp = spacy.load("en_core_web_lg")
            self.nlp.add_pipe("sentencizer")
            self.spacy_available = True
        except:
            logger.warning("spaCy model not available. Using basic analysis.")
            self.spacy_available = False
            self.nlp = None

        # Initialize transformers if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
                self.zero_shot = pipeline("zero-shot-classification")
                self.summarizer = pipeline("summarization", max_length=150)
            except:
                logger.warning(
                    "Transformer models not initialized. Using fallback methods."
                )
                self.ner_pipeline = None
                self.zero_shot = None
                self.summarizer = None
        else:
            self.ner_pipeline = None
            self.zero_shot = None
            self.summarizer = None

        # Domain taxonomy for classification
        self.domain_taxonomy = {
            "mathematics": [
                "algebra",
                "geometry",
                "calculus",
                "topology",
                "number theory",
                "statistics",
                "probability",
                "theorem",
                "proof",
                "equation",
            ],
            "science": [
                "physics",
                "chemistry",
                "biology",
                "astronomy",
                "geology",
                "experiment",
                "hypothesis",
                "theory",
                "observation",
                "data",
            ],
            "philosophy": [
                "metaphysics",
                "epistemology",
                "ethics",
                "logic",
                "aesthetics",
                "ontology",
                "phenomenology",
                "existentialism",
                "consciousness",
            ],
            "religion": [
                "theology",
                "mythology",
                "spirituality",
                "doctrine",
                "scripture",
                "faith",
                "belief",
                "ritual",
                "sacred",
                "divine",
            ],
            "art": [
                "painting",
                "sculpture",
                "music",
                "literature",
                "architecture",
                "aesthetic",
                "composition",
                "style",
                "movement",
                "expression",
            ],
            "language": [
                "linguistics",
                "grammar",
                "semantics",
                "phonetics",
                "etymology",
                "syntax",
                "morphology",
                "pragmatics",
                "discourse",
                "dialect",
            ],
        }

        # Claim indicators
        self.claim_indicators = {
            "factual": [
                "is",
                "are",
                "was",
                "were",
                "has been",
                "studies show",
                "research indicates",
                "data suggests",
                "evidence shows",
                "proven",
                "demonstrated",
                "established",
            ],
            "opinion": [
                "believe",
                "think",
                "feel",
                "seems",
                "appears",
                "arguably",
                "in my opinion",
                "suggests that",
                "likely",
                "probably",
            ],
            "hypothesis": [
                "may",
                "might",
                "could",
                "possibly",
                "potentially",
                "hypothesize",
                "theorize",
                "propose",
                "speculate",
            ],
        }

    async def analyze_content(self, scraped_doc: ScrapedDocument) -> ContentAnalysis:
        """Perform deep content analysis."""

        text = scraped_doc.content

        if self.spacy_available and self.nlp:
            doc = self.nlp(text)

            # Extract entities with enhanced NER
            entities = await self.extract_entities_and_concepts(doc, text)

            # Map to domain taxonomy
            domain_mapping = self.map_to_domain_taxonomy(doc, entities)

            # Extract concepts
            concepts = self.extract_concepts_from_text(doc, domain_mapping)

            # Identify claims and assertions
            claims = self.identify_claims_and_assertions(doc)

            # Extract key topics
            key_topics = self.extract_key_topics(doc)
        else:
            # Fallback analysis without spaCy
            entities = self._basic_entity_extraction(text)
            domain_mapping = self._basic_domain_mapping(text)
            concepts = self._basic_concept_extraction(text)
            claims = self._basic_claim_extraction(text)
            key_topics = self._basic_topic_extraction(text)

        # Sentiment analysis
        sentiment = self.analyze_sentiment_and_tone(text)

        # Generate summary
        summary = self.generate_intelligent_summary(text)

        return ContentAnalysis(
            entities=entities,
            concepts=concepts,
            claims=claims,
            domain_mapping=domain_mapping,
            key_topics=key_topics,
            sentiment_analysis=sentiment,
            summary=summary,
        )

    async def extract_entities_and_concepts(
        self, doc, text: str
    ) -> List[EntityExtraction]:
        """Extract named entities and link to concepts."""

        entities = []

        # SpaCy entities
        for ent in doc.ents:
            entities.append(
                EntityExtraction(
                    text=ent.text,
                    type=ent.label_,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8,  # SpaCy doesn't provide confidence
                )
            )

        # Transformer NER for additional entities if available
        if self.ner_pipeline and len(text) < 512:  # Transformer limit
            try:
                transformer_entities = self.ner_pipeline(text[:512])

                for ent in transformer_entities:
                    # Avoid duplicates
                    if not any(e.text == ent["word"] for e in entities):
                        entities.append(
                            EntityExtraction(
                                text=ent["word"],
                                type=ent["entity_group"],
                                start_pos=ent["start"],
                                end_pos=ent["end"],
                                confidence=ent["score"],
                            )
                        )
            except:
                logger.warning("Transformer NER failed, using only spaCy entities")

        return entities

    def map_to_domain_taxonomy(
        self, doc, entities: List[EntityExtraction]
    ) -> Dict[str, float]:
        """Map content to 6-domain taxonomy structure."""

        # Extract all meaningful terms
        terms = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                terms.append(token.lemma_.lower())

        # Add entity texts
        for entity in entities:
            terms.extend(entity.text.lower().split())

        # Score each domain
        domain_scores = {}

        for domain, keywords in self.domain_taxonomy.items():
            score = 0.0

            for term in terms:
                for keyword in keywords:
                    if keyword in term or term in keyword:
                        score += 1

            # Normalize score
            if terms:
                domain_scores[domain] = score / len(terms)
            else:
                domain_scores[domain] = 0.0

        # Use zero-shot classification if available
        if self.zero_shot and doc.text:
            try:
                candidate_labels = list(self.domain_taxonomy.keys())
                result = self.zero_shot(
                    doc.text[:1000],  # First 1000 chars
                    candidate_labels=candidate_labels,
                )

                # Combine with keyword matching
                for i, label in enumerate(result["labels"]):
                    domain_scores[label] = (
                        domain_scores[label] + result["scores"][i]
                    ) / 2
            except:
                pass

        # Normalize to sum to 1
        total = sum(domain_scores.values())
        if total > 0:
            domain_scores = {k: v / total for k, v in domain_scores.items()}

        return domain_scores

    def extract_concepts_from_text(
        self, doc, domain_mapping: Dict[str, float]
    ) -> List[ConceptExtraction]:
        """Extract high-level concepts from text."""

        concepts = []

        # Extract noun phrases as potential concepts
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:  # 2-4 word phrases
                noun_phrases.append(chunk.text)

        # Identify domain-specific concepts
        primary_domain = (
            max(domain_mapping.items(), key=lambda x: x[1])[0]
            if domain_mapping
            else "general"
        )

        for phrase in noun_phrases[:30]:  # Top 30 phrases
            # Skip common phrases
            if phrase.lower() in ["the way", "the fact", "the idea", "the thing"]:
                continue

            # Extract context
            context = self._get_phrase_context(phrase, doc.text)

            # Find related entities
            related = [
                ent.text for ent in doc.ents if phrase in ent.text or ent.text in phrase
            ]

            concepts.append(
                ConceptExtraction(
                    name=phrase,
                    domain=primary_domain,
                    confidence=0.7,  # Base confidence
                    context=context,
                    related_entities=related[:5],  # Limit related entities
                )
            )

        return concepts[:20]  # Top 20 concepts

    def identify_claims_and_assertions(self, doc) -> List[ClaimExtraction]:
        """Extract verifiable claims for fact-checking."""

        claims = []

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Skip short sentences
            if len(sent_text.split()) < 5:
                continue

            # Determine claim type
            claim_type = self._determine_claim_type(sent_text)

            if claim_type:
                # Extract supporting entities
                supporting_entities = [ent.text for ent in sent.ents]

                # Check if verifiable
                verifiable = self._is_verifiable_claim(sent_text, claim_type)

                claims.append(
                    ClaimExtraction(
                        claim_text=sent_text,
                        confidence=0.8,
                        supporting_entities=supporting_entities,
                        claim_type=claim_type,
                        verifiable=verifiable,
                    )
                )

        return claims[:30]  # Limit to top 30 claims

    def _determine_claim_type(self, text: str) -> Optional[str]:
        """Determine the type of claim."""

        text_lower = text.lower()

        # Check for claim indicators
        for claim_type, indicators in self.claim_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return claim_type

        # Pattern matching for factual claims
        factual_patterns = [
            r"\b\d+%\s+of\b",  # Percentages
            r"\b(increased|decreased|rose|fell)\s+by\b",  # Changes
            r"\b(first|last|only)\b.*\b(to|in|of)\b",  # Superlatives
            r"\b(discovered|invented|founded)\b.*\b(in|by)\b",  # Historical
            r"\b\d{4}\b",  # Years (likely historical claims)
        ]

        for pattern in factual_patterns:
            if re.search(pattern, text_lower):
                return "factual"

        return None

    def _is_verifiable_claim(self, text: str, claim_type: str) -> bool:
        """Determine if a claim can be verified."""

        # Factual claims are generally verifiable
        if claim_type == "factual":
            # Check for specific, measurable content
            if any(char.isdigit() for char in text):
                return True
            if any(
                word in text.lower()
                for word in [
                    "first",
                    "last",
                    "only",
                    "never",
                    "always",
                    "discovered",
                    "invented",
                ]
            ):
                return True

        # Opinions are not verifiable
        elif claim_type == "opinion":
            return False

        # Hypotheses may be verifiable through research
        elif claim_type == "hypothesis":
            return "research" in text.lower() or "study" in text.lower()

        return False

    def extract_key_topics(self, doc) -> List[str]:
        """Extract key topics using frequency and position."""

        # Extract candidate terms
        candidates = []
        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN"]
                and not token.is_stop
                and len(token.text) > 3
            ):
                candidates.append(token.lemma_.lower())

        # Count frequencies
        term_freq = Counter(candidates)

        # Get top terms
        key_topics = [term for term, freq in term_freq.most_common(15)]

        return key_topics

    def analyze_sentiment_and_tone(self, text: str) -> Dict:
        """Analyze sentiment and writing tone."""

        sentiment_result = {
            "sentiment": "neutral",
            "confidence": 0.5,
            "tone": "neutral",
        }

        # Use transformers if available
        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers import pipeline

                sentiment_analyzer = pipeline("sentiment-analysis")
                result = sentiment_analyzer(text[:512])
                sentiment_result["sentiment"] = result[0]["label"].lower()
                sentiment_result["confidence"] = result[0]["score"]
            except:
                pass

        # Determine tone based on content
        text_lower = text.lower()
        if any(
            term in text_lower
            for term in ["research", "study", "analysis", "methodology"]
        ):
            sentiment_result["tone"] = "academic"
        elif "!" in text or "?" in text[:100]:
            sentiment_result["tone"] = "conversational"
        elif any(
            term in text_lower
            for term in ["therefore", "hence", "thus", "consequently"]
        ):
            sentiment_result["tone"] = "formal"

        return sentiment_result

    def generate_intelligent_summary(self, text: str) -> str:
        """Generate an intelligent summary."""

        if self.summarizer and len(text.split()) > 100:
            try:
                summary = self.summarizer(
                    text[:1024],  # Limit input
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                )
                return summary[0]["summary_text"]
            except:
                pass

        # Fallback: Extract first and last sentences
        sentences = text.split(". ")
        if len(sentences) > 3:
            return f"{sentences[0]}. ... {sentences[-1]}"
        else:
            return text[:200] + "..." if len(text) > 200 else text

    def _get_phrase_context(self, phrase: str, text: str, window: int = 50) -> str:
        """Get context around a phrase."""

        idx = text.find(phrase)
        if idx == -1:
            return ""

        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)

        return "..." + text[start:end] + "..."

    # Fallback methods for when spaCy is not available
    def _basic_entity_extraction(self, text: str) -> List[EntityExtraction]:
        """Basic entity extraction without NLP models."""
        entities = []

        # Simple pattern matching for common entities
        # Find capitalized words (potential names)
        name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        for match in re.finditer(name_pattern, text):
            entities.append(
                EntityExtraction(
                    text=match.group(),
                    type="PERSON",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.5,
                )
            )

        return entities[:20]

    def _basic_domain_mapping(self, text: str) -> Dict[str, float]:
        """Basic domain mapping without NLP models."""
        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in self.domain_taxonomy.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score

        # Normalize
        total = sum(domain_scores.values()) or 1
        return {k: v / total for k, v in domain_scores.items()}

    def _basic_concept_extraction(self, text: str) -> List[ConceptExtraction]:
        """Basic concept extraction without NLP models."""
        concepts = []

        # Extract phrases between quotes or after "concept of"
        concept_patterns = [
            r'"([^"]+)"',
            r"concept of ([^,.]+)",
            r"theory of ([^,.]+)",
            r"principle of ([^,.]+)",
        ]

        for pattern in concept_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concepts.append(
                    ConceptExtraction(
                        name=match.group(1),
                        domain="general",
                        confidence=0.5,
                        context=match.group(0),
                        related_entities=[],
                    )
                )

        return concepts[:10]

    def _basic_claim_extraction(self, text: str) -> List[ClaimExtraction]:
        """Basic claim extraction without NLP models."""
        claims = []
        sentences = text.split(". ")

        for sent in sentences:
            if len(sent.split()) >= 5:
                claim_type = self._determine_claim_type(sent)
                if claim_type:
                    claims.append(
                        ClaimExtraction(
                            claim_text=sent,
                            confidence=0.5,
                            supporting_entities=[],
                            claim_type=claim_type,
                            verifiable=claim_type == "factual",
                        )
                    )

        return claims[:15]

    def _basic_topic_extraction(self, text: str) -> List[str]:
        """Basic topic extraction without NLP models."""
        # Extract words longer than 5 characters
        words = re.findall(r"\b[a-zA-Z]{6,}\b", text.lower())
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(10)]
