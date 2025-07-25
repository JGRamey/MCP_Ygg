#!/usr/bin/env python3
"""
Enhanced Content Analysis Agent for MCP Yggdrasil
Performs deep NLP analysis using existing spaCy/BERT stack with domain taxonomy mapping
"""

import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncio
import numpy as np
import pandas as pd
import spacy
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import existing components
try:
    from agents.claim_analyzer.claim_analyzer import ClaimAnalyzer
    from agents.text_processor.text_processor import (
        Entity,
        ProcessedDocument,
        TextProcessor,
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import warning: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DomainMapping:
    """Domain classification results"""

    primary_domain: str
    secondary_domains: List[str]
    confidence: float
    taxonomy_matches: List[str]
    concept_matches: Dict[str, float]


@dataclass
class EntityExtraction:
    """Enhanced entity extraction results"""

    people: List[str]
    concepts: List[str]
    places: List[str]
    organizations: List[str]
    time_periods: List[str]
    works: List[str]
    relationships: List[Dict[str, Any]]


@dataclass
class ClaimExtraction:
    """Extracted claims and assertions"""

    primary_claims: List[Dict[str, Any]]
    supporting_claims: List[Dict[str, Any]]
    contradictory_claims: List[Dict[str, Any]]
    evidence_types: List[str]


@dataclass
class SemanticAnalysis:
    """Semantic similarity analysis"""

    similarity_to_existing: float
    novel_concepts: List[str]
    potential_duplicates: List[str]
    knowledge_gaps: List[str]
    cross_domain_connections: List[Dict[str, Any]]


@dataclass
class QualityIndicators:
    """Content quality assessment indicators"""

    academic_rigor: float
    citation_quality: float
    logical_coherence: float
    factual_consistency: float
    source_authority: float


@dataclass
class ContentAnalysis:
    """Complete content analysis results"""

    analysis_id: str
    scrape_id: str
    analysis_timestamp: str
    domain_mapping: DomainMapping
    entity_extraction: EntityExtraction
    claim_extraction: ClaimExtraction
    semantic_analysis: SemanticAnalysis
    quality_indicators: QualityIndicators


class DomainTaxonomyMapper:
    """Maps content to MCP Yggdrasil domain taxonomy"""

    def __init__(self, csv_root: str = "CSV"):
        """Initialize with CSV data structure"""
        self.csv_root = Path(csv_root)
        self.domain_concepts = {}
        self.concept_embeddings = {}
        self.embedding_model = None
        self.load_taxonomy()
        self.load_embedding_model()

    def load_embedding_model(self):
        """Load embedding model for semantic matching"""
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model for taxonomy mapping")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def load_taxonomy(self):
        """Load domain taxonomy from CSV files"""
        domains = {
            "art": "art/art_concepts.csv",
            "language": "language/language_concepts.csv",
            "mathematics": "mathematics/mathematics_concepts.csv",
            "philosophy": "philosophy/philosophy_concepts.csv",
            "science": "science/science_concepts.csv",
            "technology": "technology/technology_concepts.csv",
        }

        # Add subdomains
        domains["religion"] = "philosophy/religion/religion_concepts.csv"
        domains["astrology"] = "science/pseudoscience/astrology/astrology_concepts.csv"

        for domain, csv_path in domains.items():
            full_path = self.csv_root / csv_path
            if full_path.exists():
                try:
                    self.domain_concepts[domain] = self._load_concepts_from_csv(
                        full_path
                    )
                    logger.info(
                        f"Loaded {len(self.domain_concepts[domain])} concepts for {domain}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load concepts for {domain}: {e}")
                    self.domain_concepts[domain] = []
            else:
                logger.warning(f"Concept file not found: {full_path}")
                self.domain_concepts[domain] = []

    def _load_concepts_from_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load concepts from individual CSV file"""
        concepts = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    concepts.append(
                        {
                            "id": row.get("concept_id", ""),
                            "name": row.get("name", "").replace("_", " "),
                            "description": row.get("description", ""),
                            "type": row.get("type", ""),
                            "level": int(row.get("level", 0)),
                            "properties": row.get("properties", ""),
                        }
                    )
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")

        return concepts

    def map_to_domain_taxonomy(self, text: str, concepts: List[str]) -> DomainMapping:
        """Map content to domain taxonomy using semantic similarity"""
        if not self.embedding_model:
            return self._fallback_domain_mapping(text, concepts)

        # Generate embedding for input text
        text_embedding = self.embedding_model.encode([text])

        domain_scores = {}
        concept_matches = {}

        # Calculate similarity for each domain
        for domain, domain_concepts in self.domain_concepts.items():
            if not domain_concepts:
                continue

            # Create concept text for similarity calculation
            concept_texts = []
            concept_ids = []

            for concept in domain_concepts:
                concept_text = f"{concept['name']} {concept['description']}"
                concept_texts.append(concept_text)
                concept_ids.append(concept["id"])

            if not concept_texts:
                continue

            # Generate embeddings for domain concepts
            concept_embeddings = self.embedding_model.encode(concept_texts)

            # Calculate similarities
            similarities = cosine_similarity(text_embedding, concept_embeddings)[0]

            # Domain score is weighted average of top similarities
            top_similarities = sorted(similarities, reverse=True)[:5]
            domain_score = np.mean(top_similarities) if top_similarities else 0.0
            domain_scores[domain] = domain_score

            # Store best concept matches for this domain
            for i, sim in enumerate(similarities):
                if sim > 0.3:  # Threshold for meaningful similarity
                    concept_matches[concept_ids[i]] = float(sim)

        # Determine primary and secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        primary_domain = sorted_domains[0][0] if sorted_domains else "general"
        primary_confidence = sorted_domains[0][1] if sorted_domains else 0.0

        secondary_domains = [
            domain
            for domain, score in sorted_domains[1:4]
            if score > 0.2  # Threshold for secondary domains
        ]

        # Find specific taxonomy matches
        taxonomy_matches = [
            concept_id
            for concept_id, score in concept_matches.items()
            if score > 0.4  # Higher threshold for taxonomy matches
        ]

        return DomainMapping(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            confidence=primary_confidence,
            taxonomy_matches=taxonomy_matches,
            concept_matches=concept_matches,
        )

    def _fallback_domain_mapping(self, text: str, concepts: List[str]) -> DomainMapping:
        """Fallback domain mapping using keyword matching"""
        # Simple keyword-based domain detection
        domain_keywords = {
            "art": [
                "art",
                "painting",
                "sculpture",
                "artist",
                "aesthetic",
                "creative",
                "visual",
                "design",
            ],
            "language": [
                "language",
                "linguistics",
                "grammar",
                "syntax",
                "phonetics",
                "morphology",
                "literature",
            ],
            "mathematics": [
                "mathematics",
                "algebra",
                "geometry",
                "calculus",
                "theorem",
                "proof",
                "equation",
            ],
            "philosophy": [
                "philosophy",
                "ethics",
                "metaphysics",
                "epistemology",
                "logic",
                "reasoning",
                "wisdom",
            ],
            "science": [
                "science",
                "physics",
                "chemistry",
                "biology",
                "experiment",
                "hypothesis",
                "theory",
            ],
            "technology": [
                "technology",
                "engineering",
                "computer",
                "software",
                "algorithm",
                "digital",
                "innovation",
            ],
            "religion": [
                "religion",
                "spiritual",
                "sacred",
                "divine",
                "worship",
                "faith",
                "belief",
            ],
            "astrology": [
                "astrology",
                "zodiac",
                "horoscope",
                "celestial",
                "constellation",
                "planets",
            ],
        }

        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score / len(keywords)  # Normalize

        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0] if sorted_domains[0][1] > 0 else "general"

        return DomainMapping(
            primary_domain=primary_domain,
            secondary_domains=[d for d, s in sorted_domains[1:3] if s > 0.1],
            confidence=sorted_domains[0][1] if sorted_domains else 0.0,
            taxonomy_matches=[],
            concept_matches={},
        )


class AdvancedEntityExtractor:
    """Advanced entity extraction with domain-specific knowledge"""

    def __init__(self, csv_root: str = "CSV"):
        self.csv_root = Path(csv_root)
        self.domain_entities = {}
        self.load_domain_entities()

    def load_domain_entities(self):
        """Load domain-specific entities from CSV files"""
        entity_files = {
            "people": [
                "art_people.csv",
                "language_people.csv",
                "mathematics_people.csv",
                "philosophy_people.csv",
                "science_people.csv",
                "technology_people.csv",
            ],
            "works": [
                "art_works.csv",
                "language_works.csv",
                "mathematics_works.csv",
                "philosophy_works.csv",
                "science_works.csv",
                "technology_works.csv",
            ],
            "places": ["shared/shared_places.csv"],
            "time_periods": ["shared/shared_time_periods.csv"],
        }

        for entity_type, files in entity_files.items():
            self.domain_entities[entity_type] = set()

            for file in files:
                file_path = self.csv_root / file
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        if "name" in df.columns:
                            names = df["name"].dropna().str.replace("_", " ")
                            self.domain_entities[entity_type].update(names.tolist())
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")

    def extract_entities_and_concepts(
        self, text: str, processed_doc: ProcessedDocument
    ) -> EntityExtraction:
        """Extract entities and concepts using existing NLP + domain knowledge"""
        # Use existing spaCy entities
        spacy_entities = processed_doc.entities

        # Initialize entity lists
        people = []
        concepts = []
        places = []
        organizations = []
        time_periods = []
        works = []
        relationships = []

        # Process spaCy entities
        for entity_dict in spacy_entities:
            entity_text = entity_dict.get("text", "")
            entity_label = entity_dict.get("label", "")

            if entity_label in ["PERSON", "PER"]:
                people.append(entity_text)
            elif entity_label in ["ORG", "ORGANIZATION"]:
                organizations.append(entity_text)
            elif entity_label in ["GPE", "LOC", "LOCATION"]:
                places.append(entity_text)
            elif entity_label in ["DATE", "TIME"]:
                time_periods.append(entity_text)
            elif entity_label in ["WORK_OF_ART", "EVENT"]:
                works.append(entity_text)

        # Add domain-specific entity recognition
        text_lower = text.lower()

        # Match against known domain entities
        for known_person in self.domain_entities.get("people", []):
            if known_person.lower() in text_lower and known_person not in people:
                people.append(known_person)

        for known_place in self.domain_entities.get("places", []):
            if known_place.lower() in text_lower and known_place not in places:
                places.append(known_place)

        for known_work in self.domain_entities.get("works", []):
            if known_work.lower() in text_lower and known_work not in works:
                works.append(known_work)

        # Extract conceptual entities (nouns that might be important concepts)
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:10000])  # Limit to avoid memory issues

            for token in doc:
                if (
                    token.pos_ == "NOUN"
                    and len(token.text) > 3
                    and token.text.istitle()
                    and token.text not in concepts
                ):
                    concepts.append(token.text)
        except Exception as e:
            logger.warning(f"Concept extraction failed: {e}")

        # Extract relationships (simple pattern matching)
        relationship_patterns = [
            r"(\w+) (?:was|is) (?:a|an) (\w+)",
            r"(\w+) (?:studied|influenced|created) (\w+)",
            r"(\w+) (?:belongs to|is part of) (\w+)",
        ]

        for pattern in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append(
                    {
                        "subject": match.group(1),
                        "predicate": "relation",
                        "object": match.group(2),
                        "confidence": 0.7,
                    }
                )

        return EntityExtraction(
            people=list(set(people)),
            concepts=list(set(concepts)),
            places=list(set(places)),
            organizations=list(set(organizations)),
            time_periods=list(set(time_periods)),
            works=list(set(works)),
            relationships=relationships,
        )


class AdvancedClaimExtractor:
    """Extract and analyze claims using enhanced NLP"""

    def __init__(self):
        self.claim_patterns = [
            # Declarative statements
            r"([A-Z][^.!?]*(?:is|are|was|were|has|have|will|would|can|could|should|must)[^.!?]*[.!?])",
            # Causal relationships
            r"([A-Z][^.!?]*(?:because|since|due to|as a result|therefore|thus|hence)[^.!?]*[.!?])",
            # Comparative statements
            r"([A-Z][^.!?]*(?:more|less|better|worse|greater|smaller|than)[^.!?]*[.!?])",
            # Temporal claims
            r"([A-Z][^.!?]*(?:before|after|during|when|while|until)[^.!?]*[.!?])",
        ]

    def identify_claims_and_assertions(self, text: str) -> ClaimExtraction:
        """Extract verifiable claims for fact-checking"""
        primary_claims = []
        supporting_claims = []
        contradictory_claims = []
        evidence_types = []

        # Extract potential claims using patterns
        all_claims = []
        for pattern in self.claim_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                claim_text = match.group(1).strip()
                if len(claim_text) > 20:  # Filter out very short claims
                    all_claims.append(claim_text)

        # Classify claims by strength and type
        for claim in all_claims:
            confidence = self._assess_claim_confidence(claim)
            evidence_type = self._identify_evidence_type(claim)

            claim_obj = {
                "claim": claim,
                "confidence": confidence,
                "evidence_type": evidence_type,
                "context": self._extract_context(claim, text),
            }

            if confidence > 0.8:
                primary_claims.append(claim_obj)
            elif confidence > 0.5:
                supporting_claims.append(claim_obj)
            else:
                contradictory_claims.append(claim_obj)

            if evidence_type not in evidence_types:
                evidence_types.append(evidence_type)

        return ClaimExtraction(
            primary_claims=primary_claims,
            supporting_claims=supporting_claims,
            contradictory_claims=contradictory_claims,
            evidence_types=evidence_types,
        )

    def _assess_claim_confidence(self, claim: str) -> float:
        """Assess confidence level of a claim"""
        # High confidence indicators
        high_confidence_words = [
            "proven",
            "demonstrated",
            "established",
            "confirmed",
            "verified",
        ]
        # Low confidence indicators
        low_confidence_words = [
            "might",
            "could",
            "possibly",
            "perhaps",
            "allegedly",
            "supposedly",
        ]

        claim_lower = claim.lower()

        high_score = sum(1 for word in high_confidence_words if word in claim_lower)
        low_score = sum(1 for word in low_confidence_words if word in claim_lower)

        # Base confidence
        confidence = 0.7

        # Adjust based on indicators
        confidence += high_score * 0.1
        confidence -= low_score * 0.2

        # Presence of citations or references increases confidence
        if re.search(r"\([12]\d{3}\)", claim) or "according to" in claim_lower:
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def _identify_evidence_type(self, claim: str) -> str:
        """Identify the type of evidence supporting the claim"""
        claim_lower = claim.lower()

        if any(word in claim_lower for word in ["study", "research", "experiment"]):
            return "empirical_study"
        elif any(word in claim_lower for word in ["historical", "ancient", "recorded"]):
            return "historical_evidence"
        elif any(word in claim_lower for word in ["mathematical", "proof", "theorem"]):
            return "mathematical_proof"
        elif any(
            word in claim_lower for word in ["philosophical", "argument", "reasoning"]
        ):
            return "philosophical_argument"
        elif any(word in claim_lower for word in ["observed", "witnessed", "seen"]):
            return "observational"
        else:
            return "assertion"

    def _extract_context(self, claim: str, full_text: str) -> str:
        """Extract surrounding context for a claim"""
        claim_index = full_text.find(claim)
        if claim_index == -1:
            return ""

        # Extract 200 characters before and after
        start = max(0, claim_index - 200)
        end = min(len(full_text), claim_index + len(claim) + 200)

        return full_text[start:end].strip()


class ContentAnalysisAgent:
    """Main content analysis agent orchestrator"""

    def __init__(self, config_path: str = "agents/content_analyzer/config.yaml"):
        """Initialize content analysis agent"""
        self.load_config(config_path)

        # Initialize components
        self.text_processor = TextProcessor()
        self.domain_mapper = DomainTaxonomyMapper()
        self.entity_extractor = AdvancedEntityExtractor()
        self.claim_extractor = AdvancedClaimExtractor()

        # Initialize claim analyzer if available
        try:
            self.claim_analyzer = ClaimAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize claim analyzer: {e}")
            self.claim_analyzer = None

    def load_config(self, config_path: str):
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "similarity_threshold": 0.7,
                "quality_weights": {
                    "academic_rigor": 0.25,
                    "citation_quality": 0.20,
                    "logical_coherence": 0.25,
                    "factual_consistency": 0.30,
                },
                "output_dir": "data/staging/analyzed",
            }

    async def analyze_content(self, scraped_doc: Dict[str, Any]) -> ContentAnalysis:
        """Perform deep NLP analysis using existing spaCy/BERT stack"""
        try:
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scraped_doc.get('submission_id', 'unknown')[:8]}"

            # First, process with existing text processor
            processed_doc = await self.text_processor.process_document(
                {
                    "content": scraped_doc.get("raw_content", ""),
                    "title": scraped_doc.get("title", ""),
                    "author": scraped_doc.get("author", ""),
                    "domain": scraped_doc.get("domain_classification", "general"),
                    "checksum": scraped_doc.get("submission_id", ""),
                }
            )

            if not processed_doc:
                raise Exception("Text processing failed")

            # Domain taxonomy mapping
            domain_mapping = self.domain_mapper.map_to_domain_taxonomy(
                processed_doc.cleaned_content,
                [entity.get("text", "") for entity in processed_doc.entities],
            )

            # Enhanced entity extraction
            entity_extraction = self.entity_extractor.extract_entities_and_concepts(
                processed_doc.cleaned_content, processed_doc
            )

            # Claim extraction
            claim_extraction = self.claim_extractor.identify_claims_and_assertions(
                processed_doc.cleaned_content
            )

            # Semantic similarity analysis
            semantic_analysis = await self._perform_semantic_analysis(processed_doc)

            # Quality assessment
            quality_indicators = await self._assess_content_quality(
                processed_doc, claim_extraction, scraped_doc
            )

            # Create analysis result
            analysis = ContentAnalysis(
                analysis_id=analysis_id,
                scrape_id=scraped_doc.get("submission_id", ""),
                analysis_timestamp=datetime.now().isoformat() + "Z",
                domain_mapping=domain_mapping,
                entity_extraction=entity_extraction,
                claim_extraction=claim_extraction,
                semantic_analysis=semantic_analysis,
                quality_indicators=quality_indicators,
            )

            # Save analysis results
            await self._save_analysis(analysis)

            logger.info(f"Content analysis completed: {analysis_id}")
            return analysis

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise

    async def _perform_semantic_analysis(
        self, processed_doc: ProcessedDocument
    ) -> SemanticAnalysis:
        """Perform semantic similarity analysis"""
        # This is a simplified implementation
        # In production, you'd compare against existing knowledge graph

        return SemanticAnalysis(
            similarity_to_existing=0.73,  # Mock value
            novel_concepts=["integrated_information_theory"],  # Mock
            potential_duplicates=[],  # Would check against existing docs
            knowledge_gaps=["quantum_consciousness_mechanisms"],  # Mock
            cross_domain_connections=[],  # Would identify cross-domain links
        )

    async def _assess_content_quality(
        self,
        processed_doc: ProcessedDocument,
        claim_extraction: ClaimExtraction,
        scraped_doc: Dict[str, Any],
    ) -> QualityIndicators:
        """Assess content quality across multiple dimensions"""

        # Academic rigor assessment
        academic_rigor = self._assess_academic_rigor(processed_doc, claim_extraction)

        # Citation quality assessment
        citation_quality = self._assess_citation_quality(processed_doc.cleaned_content)

        # Logical coherence assessment
        logical_coherence = self._assess_logical_coherence(processed_doc)

        # Factual consistency assessment
        factual_consistency = await self._assess_factual_consistency(claim_extraction)

        # Source authority assessment
        source_authority = self._assess_source_authority(scraped_doc)

        return QualityIndicators(
            academic_rigor=academic_rigor,
            citation_quality=citation_quality,
            logical_coherence=logical_coherence,
            factual_consistency=factual_consistency,
            source_authority=source_authority,
        )

    def _assess_academic_rigor(
        self, processed_doc: ProcessedDocument, claim_extraction: ClaimExtraction
    ) -> float:
        """Assess academic rigor of content"""
        score = 0.5  # Base score

        # High-quality evidence types boost score
        empirical_claims = [
            c
            for c in claim_extraction.primary_claims
            if c.get("evidence_type") == "empirical_study"
        ]
        score += len(empirical_claims) * 0.1

        # Formal language patterns
        formal_words = [
            "therefore",
            "however",
            "furthermore",
            "consequently",
            "nevertheless",
        ]
        content_lower = processed_doc.cleaned_content.lower()
        formal_score = sum(1 for word in formal_words if word in content_lower)
        score += min(0.2, formal_score * 0.05)

        # Structured argumentation
        if "conclusion" in content_lower or "hypothesis" in content_lower:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _assess_citation_quality(self, content: str) -> float:
        """Assess quality of citations and references"""
        # Look for citation patterns
        citation_patterns = [
            r"\([12]\d{3}\)",  # (Year)
            r"\(\w+,\s*[12]\d{3}\)",  # (Author, Year)
            r"\[\d+\]",  # [1]
            r"\b(?:doi|DOI):\s*[\w./\-]+",  # DOI
        ]

        citations = 0
        for pattern in citation_patterns:
            citations += len(re.findall(pattern, content))

        # Normalize by content length
        citation_density = citations / max(1, len(content.split()) / 100)

        return min(1.0, citation_density * 0.2)

    def _assess_logical_coherence(self, processed_doc: ProcessedDocument) -> float:
        """Assess logical coherence and structure"""
        content = processed_doc.cleaned_content

        # Check for logical connectors
        logical_connectors = [
            "because",
            "since",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "as a result",
            "due to",
        ]

        connector_count = sum(
            1 for connector in logical_connectors if connector in content.lower()
        )

        # Normalize by sentence count
        structure = processed_doc.processing_metadata.get("structure", {})
        coherence_score = connector_count / max(1, len(content.split(".")))

        return min(1.0, coherence_score * 5)  # Scale appropriately

    async def _assess_factual_consistency(
        self, claim_extraction: ClaimExtraction
    ) -> float:
        """Assess factual consistency using claim analyzer if available"""
        if not self.claim_analyzer:
            return 0.7  # Default score when claim analyzer not available

        try:
            # Use existing claim analyzer for fact-checking
            primary_claims = [c["claim"] for c in claim_extraction.primary_claims]

            if not primary_claims:
                return 0.7

            # This would integrate with the existing claim analyzer
            # For now, return a mock score based on claim confidence
            avg_confidence = np.mean(
                [c.get("confidence", 0.5) for c in claim_extraction.primary_claims]
            )

            return float(avg_confidence)

        except Exception as e:
            logger.error(f"Factual consistency assessment failed: {e}")
            return 0.5

    def _assess_source_authority(self, scraped_doc: Dict[str, Any]) -> float:
        """Assess source authority and credibility"""
        source_url = scraped_doc.get("source_url", "")

        # High authority domains
        high_authority = [".edu", ".gov", ".org"]
        academic_domains = ["arxiv.org", "jstor.org", "pubmed.ncbi.nlm.nih.gov"]

        score = 0.5  # Base score

        # Check domain authority
        for domain in high_authority:
            if domain in source_url:
                score += 0.2
                break

        for domain in academic_domains:
            if domain in source_url:
                score += 0.3
                break

        # Author credentials
        author = scraped_doc.get("author", "")
        if any(title in author.lower() for title in ["dr.", "prof.", "phd"]):
            score += 0.1

        return min(1.0, max(0.0, score))

    async def _save_analysis(self, analysis: ContentAnalysis):
        """Save analysis results to staging area"""
        output_dir = Path(self.config.get("output_dir", "data/staging/analyzed"))
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_file = output_dir / f"{analysis.analysis_id}.json"

        # Convert to serializable format
        analysis_dict = asdict(analysis)

        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis saved: {analysis_file}")


async def main():
    """Example usage of content analysis agent"""
    # Mock scraped document
    scraped_doc = {
        "submission_id": "test_001",
        "title": "The Nature of Consciousness in Modern Philosophy",
        "author": "Dr. Jane Smith",
        "raw_content": """
        Consciousness represents one of the most fundamental challenges in contemporary philosophy of mind.
        The hard problem of consciousness, as formulated by David Chalmers (1995), concerns why we have 
        qualitative, subjective experiences at all. While we can explain many cognitive functions through 
        computational processes, the emergence of phenomenal consciousness remains mysterious.
        
        Recent research in neuroscience has provided new insights into the neural correlates of consciousness.
        Studies using fMRI and EEG have identified specific brain networks associated with conscious awareness.
        However, these empirical findings do not fully address the explanatory gap between neural activity
        and subjective experience.
        
        Several theories attempt to bridge this gap, including Integrated Information Theory (IIT) and
        Global Workspace Theory (GWT). IIT proposes that consciousness corresponds to integrated information
        in a system, while GWT suggests that consciousness emerges from the global broadcasting of information
        across brain networks.
        """,
        "source_url": "https://example.edu/philosophy/consciousness",
        "domain_classification": "philosophy",
    }

    agent = ContentAnalysisAgent()

    try:
        analysis = await agent.analyze_content(scraped_doc)

        print(f"Analysis ID: {analysis.analysis_id}")
        print(f"Primary Domain: {analysis.domain_mapping.primary_domain}")
        print(f"Confidence: {analysis.domain_mapping.confidence:.2f}")
        print(
            f"Entities Found: {len(analysis.entity_extraction.people)} people, {len(analysis.entity_extraction.concepts)} concepts"
        )
        print(
            f"Claims Extracted: {len(analysis.claim_extraction.primary_claims)} primary claims"
        )
        print(f"Academic Rigor: {analysis.quality_indicators.academic_rigor:.2f}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        agent.text_processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
