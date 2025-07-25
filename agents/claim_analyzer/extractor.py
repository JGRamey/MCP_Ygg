#!/usr/bin/env python3
"""Claim extraction module for Claim Analyzer Agent"""

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

import spacy
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from .database import DatabaseConnector
from .exceptions import ClaimExtractionError, ModelLoadError, ValidationError
from .models import Claim
from .utils import clean_claim_text, sanitize_text, validate_input

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extracts claims from various text sources using NLP"""

    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self._init_nlp_models()
        self._init_patterns()

    def _init_nlp_models(self):
        """Initialize NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            logger.error(
                "spaCy model not found. Please install: python -m spacy download en_core_web_sm"
            )
            self.nlp = None
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise ModelLoadError(f"spaCy model loading failed: {e}") from e

        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise ModelLoadError(f"Sentence transformer loading failed: {e}") from e

    def _init_patterns(self):
        """Initialize claim detection patterns"""
        self.claim_patterns = [
            r"(?:is|are|was|were|will be|has been|have been)\s+(?:a|an|the)?\s*\w+",
            r"(?:claims?|states?|argues?|believes?|says?)\s+that\s+.*",
            r"(?:according to|research shows|studies indicate|experts say)\s+.*",
            r"(?:it is|this is|that is)\s+(?:true|false|correct|incorrect|a fact)\s+that\s+.*",
        ]

        self.domain_keywords = {
            "science": [
                "experiment",
                "study",
                "research",
                "theory",
                "hypothesis",
                "data",
                "evidence",
            ],
            "math": [
                "theorem",
                "proof",
                "equation",
                "formula",
                "calculate",
                "number",
                "geometry",
            ],
            "religion": [
                "god",
                "faith",
                "belief",
                "scripture",
                "church",
                "prayer",
                "divine",
            ],
            "history": [
                "ancient",
                "war",
                "civilization",
                "empire",
                "century",
                "historical",
                "period",
            ],
            "literature": [
                "novel",
                "poem",
                "author",
                "book",
                "story",
                "character",
                "literary",
            ],
            "philosophy": [
                "ethics",
                "logic",
                "metaphysics",
                "epistemology",
                "moral",
                "existence",
            ],
        }

    async def extract_claims(
        self, text: str, source: str = "unknown", domain: str = "general"
    ) -> List[Claim]:
        """Extract verifiable claims from text"""
        try:
            # Validate and sanitize input
            if not validate_input(text):
                raise ValidationError("Invalid input text")

            text = sanitize_text(text)
            if not text.strip():
                return []

            logger.info(f"Extracting claims from {len(text)} characters of text")

            domain = self._classify_domain(text) if domain == "general" else domain

            claims = (
                await self._extract_with_spacy(text, source, domain)
                if self.nlp
                else await self._extract_with_patterns(text, source, domain)
            )

            # Store claims in databases
            for claim in claims:
                await self._store_claim(claim)

            logger.info(f"Successfully extracted {len(claims)} claims")
            return claims

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            raise ClaimExtractionError(f"Claim extraction failed: {e}") from e

    def _classify_domain(self, text: str) -> str:
        """Classify text into one of the six domains"""
        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score

        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"

    async def _extract_with_spacy(
        self, text: str, source: str, domain: str
    ) -> List[Claim]:
        """Extract claims using spaCy NLP"""
        claims = []
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip()

            if (
                len(sent_text) < 10
                or sent_text.endswith("?")
                or sent_text.startswith("/")
            ):
                continue

            if self._is_claim_sentence(sent_text):
                confidence = self._calculate_claim_confidence(sent_text, sent)

                if confidence > 0.3:
                    entities = [ent.text for ent in sent.ents]

                    claim = Claim(
                        claim_id="",
                        text=sent_text,
                        source=source,
                        domain=domain,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        context=self._get_context(sent, doc),
                        entities=entities,
                    )
                    claims.append(claim)

        return claims

    async def _extract_with_patterns(
        self, text: str, source: str, domain: str
    ) -> List[Claim]:
        """Fallback pattern-based claim extraction"""
        claims = []
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            if self._is_claim_sentence(sentence):
                claim = Claim(
                    claim_id="",
                    text=sentence,
                    source=source,
                    domain=domain,
                    timestamp=datetime.now(),
                    confidence=0.5,
                    context=sentence,
                    entities=[],
                )
                claims.append(claim)

        return claims

    def _is_claim_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a verifiable claim"""
        sentence_lower = sentence.lower()

        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True

        factual_indicators = [
            "fact",
            "truth",
            "evidence",
            "study",
            "research",
            "data",
            "statistics",
            "proven",
            "confirmed",
            "verified",
            "documented",
        ]

        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True

        absolute_terms = [
            "always",
            "never",
            "all",
            "none",
            "every",
            "no one",
            "everyone",
        ]
        if any(term in sentence_lower for term in absolute_terms):
            return True

        return False

    def _calculate_claim_confidence(self, sentence: str, spacy_sent) -> float:
        """Calculate confidence that this sentence contains a verifiable claim"""
        confidence = 0.0

        factual_verbs = {"be", "have", "do", "say", "state", "claim", "prove", "show"}
        for token in spacy_sent:
            if token.lemma_ in factual_verbs and token.pos_ == "VERB":
                confidence += 0.3

        if len([ent for ent in spacy_sent.ents]) > 0:
            confidence += 0.2

        if any(
            token.like_num or token.ent_type_ in ["DATE", "TIME", "PERCENT", "MONEY"]
            for token in spacy_sent
        ):
            confidence += 0.2

        if len(sentence.split()) > 5:
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_context(self, sentence, doc) -> str:
        """Get surrounding context for a claim"""
        sent_start = sentence.start
        sent_end = sentence.end

        context_tokens = []

        prev_sent_tokens = [token for token in doc if token.i < sent_start]
        if prev_sent_tokens:
            context_tokens.extend(prev_sent_tokens[-20:])

        context_tokens.extend([token for token in sentence])

        next_sent_tokens = [token for token in doc if token.i >= sent_end]
        if next_sent_tokens:
            context_tokens.extend(next_sent_tokens[:20])

        return " ".join([token.text for token in context_tokens])

    async def _store_claim(self, claim: Claim):
        """Store claim in Neo4j graph and Qdrant vector database"""
        try:
            # Generate embedding
            embedding = self.sentence_model.encode(claim.text)

            # Store in Neo4j
            await self._store_in_neo4j(claim)

            # Store in Qdrant
            await self._store_in_qdrant(claim, embedding)

            # Cache in Redis
            await self._cache_in_redis(claim)

        except Exception as e:
            logger.error(f"Error storing claim {claim.claim_id}: {e}")

    async def _store_in_neo4j(self, claim: Claim):
        """Store claim in Neo4j"""
        async with self.db_connector.neo4j_driver.session() as session:
            query = """
            CREATE (c:Claim {
                id: $claim_id,
                text: $text,
                source: $source,
                domain: $domain,
                timestamp: $timestamp,
                confidence: $confidence,
                context: $context,
                entities: $entities
            })
            RETURN c.id as node_id
            """

            result = await session.run(
                query,
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "source": claim.source,
                    "domain": claim.domain,
                    "timestamp": claim.timestamp.isoformat(),
                    "confidence": claim.confidence,
                    "context": claim.context,
                    "entities": claim.entities,
                },
            )

            record = await result.single()
            if record:
                logger.info(f"Stored claim in Neo4j: {claim.claim_id}")

    async def _store_in_qdrant(self, claim: Claim, embedding):
        """Store claim in Qdrant"""
        point = PointStruct(
            id=claim.claim_id,
            vector=embedding.tolist(),
            payload={
                "text": claim.text,
                "source": claim.source,
                "domain": claim.domain,
                "timestamp": claim.timestamp.isoformat(),
                "confidence": claim.confidence,
                "entities": claim.entities,
            },
        )

        self.db_connector.qdrant_client.upsert(collection_name="claims", points=[point])
        logger.info(f"Stored claim in Qdrant: {claim.claim_id}")

    async def _cache_in_redis(self, claim: Claim):
        """Cache claim in Redis"""
        cache_key = f"claim:{claim.claim_id}"
        await self.db_connector.redis_client.setex(
            cache_key, 3600, json.dumps(asdict(claim), default=str)
        )
