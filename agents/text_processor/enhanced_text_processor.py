#!/usr/bin/env python3
"""
Enhanced Text Processor Agent with Multilingual Support
Main processor class with comprehensive text analysis capabilities
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import aiofiles
import asyncio
import numpy as np
import torch

from agents.text_processor.entity_linker import EntityLinker

# Import modular components
from agents.text_processor.models import LinkedEntity, ProcessedText
from agents.text_processor.multilingual_processor import MultilingualProcessor

# Import base components from existing text processor
from agents.text_processor.text_processor import (
    EmbeddingGenerator,
    Entity,
    TextChunker,
    TextCleaner,
)
from agents.text_processor.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTextProcessor:
    """Enhanced text processor with multilingual support and transformers"""

    def __init__(self, config_path: str = "agents/text_processor/enhanced_config.yaml"):
        """Initialize enhanced text processor"""
        self.config = load_config(config_path)

        # Initialize components
        self.multilingual_processor = MultilingualProcessor()
        self.entity_linker = EntityLinker()
        self.embedding_generator = EmbeddingGenerator()
        self.text_chunker = TextChunker(
            chunk_size=self.config.get("chunk_size", 1000),
            overlap=self.config.get("chunk_overlap", 200),
        )

        # Cache for processed texts
        self.cache = {}
        self.max_cache_size = self.config.get("max_cache_size", 1000)

    async def process_text(
        self,
        text: str,
        domain: str = "general",
        target_summary_length: int = 150,
        extract_concepts: bool = True,
        link_entities: bool = True,
    ) -> ProcessedText:
        """Comprehensive text processing with multilingual support"""

        # Check cache
        cache_key = hashlib.md5(f"{text[:100]}{domain}".encode()).hexdigest()
        if cache_key in self.cache:
            logger.info("Retrieved from cache")
            return self.cache[cache_key]

        # Detect language
        language, lang_confidence = self.multilingual_processor.detect_language(text)
        logger.info(
            f"Detected language: {language} (confidence: {lang_confidence:.2f})"
        )

        # Clean text
        cleaned_text = TextCleaner.clean_text(text, language)

        # Extract entities using both spaCy and transformers
        entities = await self._extract_entities(cleaned_text, language)

        # Extract concepts and key phrases
        concepts = []
        key_phrases = []
        if extract_concepts:
            concepts, key_phrases = await self._extract_concepts(cleaned_text, language)

        # Generate summary
        summary = await self._generate_summary(
            cleaned_text, language, target_summary_length
        )

        # Analyze sentiment
        sentiment = await self._analyze_sentiment(cleaned_text, language)

        # Link entities to knowledge graph
        linked_entities = []
        if link_entities and self.entity_linker.neo4j_agent:
            entity_objects = [
                Entity(
                    text=e["text"],
                    label=e["label"],
                    start=e.get("start", 0),
                    end=e.get("end", 0),
                    confidence=e.get("confidence", 1.0),
                )
                for e in entities
            ]
            linked = await self.entity_linker.link_entities(entity_objects, domain)
            linked_entities = [asdict(le) for le in linked]

        # Generate embeddings
        embeddings = self.embedding_generator.generate_single_embedding(
            cleaned_text, domain, language
        )

        # Create processed text object
        processed = ProcessedText(
            original_text=text,
            language=language,
            language_confidence=lang_confidence,
            entities=entities,
            concepts=concepts,
            summary=summary,
            sentiment=sentiment,
            key_phrases=key_phrases,
            linked_entities=linked_entities,
            embeddings=embeddings,
            processing_metadata={
                "processor_version": "2.0",
                "models_used": {
                    "language_detection": "langdetect",
                    "nlp": f"spacy_{language}",
                    "summarization": "transformer",
                    "sentiment": "transformer",
                    "embeddings": self.embedding_generator.model_name,
                },
                "processing_time": datetime.now().isoformat(),
                "domain": domain,
            },
        )

        # Cache result
        if self.config.get("enable_caching", True):
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = processed

        return processed

    async def _extract_entities(self, text: str, language: str) -> List[Dict]:
        """Extract entities using multiple models"""
        entities = []
        seen_entities = set()

        # SpaCy NER
        nlp = self.multilingual_processor.get_nlp_model(language)
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                entity_key = f"{ent.text}:{ent.label_}"
                if entity_key not in seen_entities:
                    entities.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "confidence": 0.8,
                            "source": "spacy",
                        }
                    )
                    seen_entities.add(entity_key)

        # Transformer NER
        ner_model = self.multilingual_processor.get_ner_model(language)
        if ner_model:
            try:
                # Process in chunks if text is too long
                max_length = 512
                chunks = [
                    text[i : i + max_length]
                    for i in range(0, len(text), max_length - 50)
                ]

                for chunk_idx, chunk in enumerate(chunks):
                    ner_results = ner_model(chunk)
                    for ent in ner_results:
                        entity_key = f"{ent['word']}:{ent['entity_group']}"
                        if entity_key not in seen_entities:
                            entities.append(
                                {
                                    "text": ent["word"],
                                    "label": ent["entity_group"],
                                    "start": ent["start"]
                                    + (chunk_idx * (max_length - 50)),
                                    "end": ent["end"] + (chunk_idx * (max_length - 50)),
                                    "confidence": ent["score"],
                                    "source": "transformer",
                                }
                            )
                            seen_entities.add(entity_key)
            except Exception as e:
                logger.error(f"Error in transformer NER: {e}")

        return entities

    async def _extract_concepts(
        self, text: str, language: str
    ) -> Tuple[List[Dict], List[str]]:
        """Extract concepts and key phrases"""
        concepts = []
        key_phrases = []

        # Use spaCy for noun phrase extraction
        nlp = self.multilingual_processor.get_nlp_model(language)
        if nlp:
            doc = nlp(text)

            # Extract noun phrases as key phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            key_phrases.extend(noun_phrases[:20])  # Top 20 phrases

            # Extract concepts based on POS patterns
            concept_patterns = []
            for token in doc:
                # Look for specific POS patterns that indicate concepts
                if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in [
                    "nsubj",
                    "dobj",
                    "pobj",
                ]:
                    concept = {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dependency": token.dep_,
                        "frequency": 1,
                    }
                    concept_patterns.append(concept)

            # Count frequencies and filter
            concept_freq = defaultdict(int)
            for concept in concept_patterns:
                concept_freq[concept["lemma"]] += 1

            # Create final concept list
            for lemma, freq in sorted(
                concept_freq.items(), key=lambda x: x[1], reverse=True
            )[:15]:
                concepts.append(
                    {
                        "text": lemma,
                        "type": "concept",
                        "frequency": freq,
                        "confidence": min(
                            freq / 10.0, 1.0
                        ),  # Simple confidence based on frequency
                    }
                )

        return concepts, key_phrases

    async def _generate_summary(
        self, text: str, language: str, target_length: int
    ) -> str:
        """Generate text summary using transformers"""
        summarizer = self.multilingual_processor.get_summarizer(language)
        if not summarizer:
            # Fallback to simple extractive summary
            sentences = text.split(".")[:3]
            return ". ".join(sentences) + "."

        try:
            # Adjust length parameters
            max_length = min(target_length, len(text.split()) // 2)
            min_length = max(30, target_length // 3)

            # Generate summary
            summary_result = summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )

            if summary_result and len(summary_result) > 0:
                return summary_result[0]["summary_text"]

        except Exception as e:
            logger.error(f"Error generating summary: {e}")

        # Fallback
        return text[:target_length] + "..."

    async def _analyze_sentiment(self, text: str, language: str) -> Dict:
        """Analyze text sentiment"""
        analyzer = self.multilingual_processor.get_sentiment_analyzer(language)
        if not analyzer:
            return {"label": "NEUTRAL", "score": 0.5}

        try:
            # Analyze sentiment on first 512 characters (transformer limit)
            result = analyzer(text[:512])
            if result and len(result) > 0:
                return {
                    "label": result[0]["label"],
                    "score": result[0]["score"],
                    "confidence": result[0]["score"],
                }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")

        return {"label": "NEUTRAL", "score": 0.5, "confidence": 0.0}

    async def process_batch(
        self, texts: List[Dict[str, str]], **kwargs
    ) -> List[ProcessedText]:
        """Process multiple texts in batch"""
        tasks = []
        for text_data in texts:
            text = text_data.get("text", "")
            domain = text_data.get("domain", "general")
            task = self.process_text(text, domain, **kwargs)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_texts = []
        for i, result in enumerate(results):
            if isinstance(result, ProcessedText):
                processed_texts.append(result)
            else:
                logger.error(f"Failed to process text {i}: {result}")

        return processed_texts

    def set_neo4j_agent(self, neo4j_agent):
        """Set Neo4j agent for entity linking"""
        self.entity_linker.neo4j_agent = neo4j_agent

    async def save_processed_text(
        self, processed: ProcessedText, output_dir: str = "data/processed_texts"
    ):
        """Save processed text to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate unique ID
        text_id = hashlib.md5(processed.original_text.encode()).hexdigest()[:12]

        # Convert to dict for JSON serialization
        data = asdict(processed)
        data["embeddings"] = processed.embeddings.tolist()
        data["timestamp"] = datetime.now().isoformat()

        # Save to JSON
        output_file = output_path / f"{text_id}_processed.json"
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

        logger.info(f"Saved processed text to {output_file}")


# Example usage
async def main():
    """Example usage of enhanced text processor"""

    # Sample multilingual texts
    texts = [
        {
            "text": "The quick brown fox jumps over the lazy dog. This is a test of the enhanced text processing system with transformers and multilingual support.",
            "domain": "general",
        },
        {
            "text": "La inteligencia artificial está transformando rápidamente muchas industrias. Los modelos de lenguaje como GPT han demostrado capacidades impresionantes.",
            "domain": "technology",
        },
        {
            "text": "Les réseaux de neurones profonds ont révolutionné le traitement du langage naturel. Cette technologie permet des applications innovantes.",
            "domain": "science",
        },
    ]

    processor = EnhancedTextProcessor()

    # Process texts
    results = await processor.process_batch(texts)

    for i, result in enumerate(results):
        print(f"\n--- Text {i+1} ---")
        print(
            f"Language: {result.language} (confidence: {result.language_confidence:.2f})"
        )
        print(f"Summary: {result.summary}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Entities: {len(result.entities)}")
        print(f"Key phrases: {result.key_phrases[:5]}")
        print(f"Concepts: {[c['text'] for c in result.concepts[:5]]}")


if __name__ == "__main__":
    asyncio.run(main())
