#!/usr/bin/env python3
"""
Enhanced Text Processor Agent with Multilingual Support
Provides advanced text processing with transformers, entity linking, and multilingual capabilities
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import langdetect
from langdetect import detect_langs, LangDetectException
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from collections import defaultdict
import aiofiles

# Import base components from existing text processor
from agents.text_processor.text_processor import (
    TextChunker, TextCleaner, Entity, TextChunk,
    ProcessedDocument, EmbeddingGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Enhanced processed text with additional fields"""
    original_text: str
    language: str
    language_confidence: float
    entities: List[Dict]
    concepts: List[Dict]
    summary: str
    sentiment: Dict
    key_phrases: List[str]
    linked_entities: List[Dict]  # Linked to knowledge graph
    embeddings: np.ndarray
    processing_metadata: Dict


@dataclass
class LinkedEntity:
    """Entity linked to knowledge graph"""
    text: str
    label: str
    kb_id: Optional[str]
    kb_type: Optional[str]
    confidence: float
    properties: Dict


class EntityLinker:
    """Link entities to knowledge graph"""
    
    def __init__(self, neo4j_agent=None):
        self.neo4j_agent = neo4j_agent
        self.entity_cache = {}
        
    async def link_entities(self, entities: List[Entity], domain: str = "general") -> List[LinkedEntity]:
        """Link entities to knowledge graph nodes"""
        linked_entities = []
        
        for entity in entities:
            # Check cache first
            cache_key = f"{entity.text}:{entity.label}"
            if cache_key in self.entity_cache:
                linked_entities.append(self.entity_cache[cache_key])
                continue
            
            # Query knowledge graph
            linked = await self._find_in_knowledge_graph(entity, domain)
            if linked:
                self.entity_cache[cache_key] = linked
                linked_entities.append(linked)
            else:
                # Create unlinked entity
                linked = LinkedEntity(
                    text=entity.text,
                    label=entity.label,
                    kb_id=None,
                    kb_type=None,
                    confidence=0.0,
                    properties={}
                )
                linked_entities.append(linked)
        
        return linked_entities
    
    async def _find_in_knowledge_graph(self, entity: Entity, domain: str) -> Optional[LinkedEntity]:
        """Find entity in knowledge graph"""
        if not self.neo4j_agent:
            return None
        
        # Build query based on entity type
        if entity.label in ["PERSON", "PER"]:
            query = """
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($name)
            RETURN p.id as kb_id, 'Person' as kb_type, p as properties
            LIMIT 1
            """
        elif entity.label in ["ORG", "ORGANIZATION"]:
            query = """
            MATCH (o:Organization)
            WHERE toLower(o.name) CONTAINS toLower($name)
            RETURN o.id as kb_id, 'Organization' as kb_type, o as properties
            LIMIT 1
            """
        elif entity.label in ["LOC", "LOCATION", "GPE"]:
            query = """
            MATCH (l:Location)
            WHERE toLower(l.name) CONTAINS toLower($name)
            RETURN l.id as kb_id, 'Location' as kb_type, l as properties
            LIMIT 1
            """
        else:
            # Generic concept search
            query = """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.id as kb_id, 'Concept' as kb_type, c as properties
            LIMIT 1
            """
        
        try:
            result = await self.neo4j_agent.query(query, {"name": entity.text})
            if result and len(result) > 0:
                return LinkedEntity(
                    text=entity.text,
                    label=entity.label,
                    kb_id=result[0]["kb_id"],
                    kb_type=result[0]["kb_type"],
                    confidence=0.8,  # Base confidence
                    properties=dict(result[0]["properties"])
                )
        except Exception as e:
            logger.error(f"Error linking entity {entity.text}: {e}")
        
        return None


class MultilingualProcessor:
    """Enhanced multilingual text processing"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic'
        }
        
        # Load spaCy models for supported languages
        self.nlp_models = {}
        self._load_spacy_models()
        
        # Initialize transformers with GPU support if available
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load transformer pipelines
        self._init_transformers()
        
    def _load_spacy_models(self):
        """Load available spaCy models"""
        model_mapping = {
            'en': 'en_core_web_lg',
            'es': 'es_core_news_lg',
            'fr': 'fr_core_news_lg',
            'de': 'de_core_news_lg',
            'it': 'it_core_news_lg',
            'pt': 'pt_core_news_lg',
            'nl': 'nl_core_news_lg',
            'pl': 'pl_core_news_lg',
            'ru': 'ru_core_news_lg',
            'zh': 'zh_core_web_lg',
            'ja': 'ja_core_news_lg'
        }
        
        for lang_code, model_name in model_mapping.items():
            try:
                self.nlp_models[lang_code] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model for {lang_code}: {model_name}")
            except OSError:
                logger.warning(f"spaCy model {model_name} not available for {lang_code}")
                # Use English as fallback
                if lang_code != 'en' and 'en' in self.nlp_models:
                    self.nlp_models[lang_code] = self.nlp_models['en']
    
    def _init_transformers(self):
        """Initialize transformer pipelines"""
        try:
            # Multilingual summarization
            self.summarizers = {
                'en': pipeline("summarization", model="facebook/bart-large-cnn", device=self.device),
                'multilingual': pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", device=self.device)
            }
            
            # Sentiment analysis
            self.sentiment_analyzers = {
                'en': pipeline("sentiment-analysis", device=self.device),
                'multilingual': pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=self.device)
            }
            
            # Named Entity Recognition with aggregation
            self.ner_models = {
                'en': pipeline("ner", aggregation_strategy="simple", device=self.device),
                'multilingual': pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple", device=self.device)
            }
            
            logger.info("Loaded transformer models successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            # Initialize with None for graceful degradation
            self.summarizers = {}
            self.sentiment_analyzers = {}
            self.ner_models = {}
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        try:
            # Use langdetect for initial detection
            detections = detect_langs(text)
            if detections:
                lang = detections[0]
                lang_code = lang.lang
                confidence = lang.prob
                
                # Map to our supported languages
                if lang_code in self.supported_languages:
                    return lang_code, confidence
                else:
                    # Try to find closest match
                    for supported_code in self.supported_languages:
                        if lang_code.startswith(supported_code) or supported_code.startswith(lang_code):
                            return supported_code, confidence * 0.9  # Slightly lower confidence
                    
                    # Default to English if unsupported
                    return 'en', confidence * 0.5
            
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to English")
        
        return 'en', 0.5
    
    def get_nlp_model(self, language: str) -> spacy.Language:
        """Get spaCy model for language"""
        return self.nlp_models.get(language, self.nlp_models.get('en'))
    
    def get_summarizer(self, language: str):
        """Get appropriate summarizer for language"""
        if language == 'en' and 'en' in self.summarizers:
            return self.summarizers['en']
        elif 'multilingual' in self.summarizers:
            return self.summarizers['multilingual']
        return None
    
    def get_sentiment_analyzer(self, language: str):
        """Get appropriate sentiment analyzer for language"""
        if language == 'en' and 'en' in self.sentiment_analyzers:
            return self.sentiment_analyzers['en']
        elif 'multilingual' in self.sentiment_analyzers:
            return self.sentiment_analyzers['multilingual']
        return None
    
    def get_ner_model(self, language: str):
        """Get appropriate NER model for language"""
        if language == 'en' and 'en' in self.ner_models:
            return self.ner_models['en']
        elif 'multilingual' in self.ner_models:
            return self.ner_models['multilingual']
        return None


class EnhancedTextProcessor:
    """Enhanced text processor with multilingual support and transformers"""
    
    def __init__(self, config_path: str = "agents/text_processor/enhanced_config.yaml"):
        """Initialize enhanced text processor"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.multilingual_processor = MultilingualProcessor()
        self.entity_linker = EntityLinker()
        self.embedding_generator = EmbeddingGenerator()
        self.text_chunker = TextChunker(
            chunk_size=self.config.get('chunk_size', 1000),
            overlap=self.config.get('chunk_overlap', 200)
        )
        
        # Cache for processed texts
        self.cache = {}
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'max_summary_length': 150,
                'min_summary_length': 50,
                'enable_caching': True,
                'max_cache_size': 1000,
                'batch_size': 16,
                'enable_gpu': torch.cuda.is_available()
            }
    
    async def process_text(
        self, 
        text: str, 
        domain: str = "general",
        target_summary_length: int = 150,
        extract_concepts: bool = True,
        link_entities: bool = True
    ) -> ProcessedText:
        """Comprehensive text processing with multilingual support"""
        
        # Check cache
        cache_key = hashlib.md5(f"{text[:100]}{domain}".encode()).hexdigest()
        if cache_key in self.cache:
            logger.info("Retrieved from cache")
            return self.cache[cache_key]
        
        # Detect language
        language, lang_confidence = self.multilingual_processor.detect_language(text)
        logger.info(f"Detected language: {language} (confidence: {lang_confidence:.2f})")
        
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
            cleaned_text, 
            language, 
            target_summary_length
        )
        
        # Analyze sentiment
        sentiment = await self._analyze_sentiment(cleaned_text, language)
        
        # Link entities to knowledge graph
        linked_entities = []
        if link_entities and self.entity_linker.neo4j_agent:
            entity_objects = [
                Entity(
                    text=e['text'],
                    label=e['label'],
                    start=e.get('start', 0),
                    end=e.get('end', 0),
                    confidence=e.get('confidence', 1.0)
                )
                for e in entities
            ]
            linked = await self.entity_linker.link_entities(entity_objects, domain)
            linked_entities = [asdict(le) for le in linked]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_single_embedding(
            cleaned_text, 
            domain, 
            language
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
                'processor_version': '2.0',
                'models_used': {
                    'language_detection': 'langdetect',
                    'nlp': f'spacy_{language}',
                    'summarization': 'transformer',
                    'sentiment': 'transformer',
                    'embeddings': self.embedding_generator.model_name
                },
                'processing_time': datetime.now().isoformat(),
                'domain': domain
            }
        )
        
        # Cache result
        if self.config.get('enable_caching', True):
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
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8,
                        'source': 'spacy'
                    })
                    seen_entities.add(entity_key)
        
        # Transformer NER
        ner_model = self.multilingual_processor.get_ner_model(language)
        if ner_model:
            try:
                # Process in chunks if text is too long
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length-50)]
                
                for chunk_idx, chunk in enumerate(chunks):
                    ner_results = ner_model(chunk)
                    for ent in ner_results:
                        entity_key = f"{ent['word']}:{ent['entity_group']}"
                        if entity_key not in seen_entities:
                            entities.append({
                                'text': ent['word'],
                                'label': ent['entity_group'],
                                'start': ent['start'] + (chunk_idx * (max_length-50)),
                                'end': ent['end'] + (chunk_idx * (max_length-50)),
                                'confidence': ent['score'],
                                'source': 'transformer'
                            })
                            seen_entities.add(entity_key)
            except Exception as e:
                logger.error(f"Error in transformer NER: {e}")
        
        return entities
    
    async def _extract_concepts(self, text: str, language: str) -> Tuple[List[Dict], List[str]]:
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
                if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    concept = {
                        'text': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'dependency': token.dep_,
                        'frequency': 1
                    }
                    concept_patterns.append(concept)
            
            # Count frequencies and filter
            concept_freq = defaultdict(int)
            for concept in concept_patterns:
                concept_freq[concept['lemma']] += 1
            
            # Create final concept list
            for lemma, freq in sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:15]:
                concepts.append({
                    'text': lemma,
                    'type': 'concept',
                    'frequency': freq,
                    'confidence': min(freq / 10.0, 1.0)  # Simple confidence based on frequency
                })
        
        return concepts, key_phrases
    
    async def _generate_summary(self, text: str, language: str, target_length: int) -> str:
        """Generate text summary using transformers"""
        summarizer = self.multilingual_processor.get_summarizer(language)
        if not summarizer:
            # Fallback to simple extractive summary
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
        
        try:
            # Adjust length parameters
            max_length = min(target_length, len(text.split()) // 2)
            min_length = max(30, target_length // 3)
            
            # Generate summary
            summary_result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            if summary_result and len(summary_result) > 0:
                return summary_result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        # Fallback
        return text[:target_length] + "..."
    
    async def _analyze_sentiment(self, text: str, language: str) -> Dict:
        """Analyze text sentiment"""
        analyzer = self.multilingual_processor.get_sentiment_analyzer(language)
        if not analyzer:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Analyze sentiment on first 512 characters (transformer limit)
            result = analyzer(text[:512])
            if result and len(result) > 0:
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score'],
                    'confidence': result[0]['score']
                }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
        
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}
    
    async def process_batch(
        self, 
        texts: List[Dict[str, str]], 
        **kwargs
    ) -> List[ProcessedText]:
        """Process multiple texts in batch"""
        tasks = []
        for text_data in texts:
            text = text_data.get('text', '')
            domain = text_data.get('domain', 'general')
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
    
    async def save_processed_text(self, processed: ProcessedText, output_dir: str = "data/processed_texts"):
        """Save processed text to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique ID
        text_id = hashlib.md5(processed.original_text.encode()).hexdigest()[:12]
        
        # Convert to dict for JSON serialization
        data = asdict(processed)
        data['embeddings'] = processed.embeddings.tolist()
        data['timestamp'] = datetime.now().isoformat()
        
        # Save to JSON
        output_file = output_path / f"{text_id}_processed.json"
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        
        logger.info(f"Saved processed text to {output_file}")


# Example usage
async def main():
    """Example usage of enhanced text processor"""
    
    # Sample multilingual texts
    texts = [
        {
            'text': "The quick brown fox jumps over the lazy dog. This is a test of the enhanced text processing system with transformers and multilingual support.",
            'domain': 'general'
        },
        {
            'text': "La inteligencia artificial está transformando rápidamente muchas industrias. Los modelos de lenguaje como GPT han demostrado capacidades impresionantes.",
            'domain': 'technology'
        },
        {
            'text': "Les réseaux de neurones profonds ont révolutionné le traitement du langage naturel. Cette technologie permet des applications innovantes.",
            'domain': 'science'
        }
    ]
    
    processor = EnhancedTextProcessor()
    
    # Process texts
    results = await processor.process_batch(texts)
    
    for i, result in enumerate(results):
        print(f"\n--- Text {i+1} ---")
        print(f"Language: {result.language} (confidence: {result.language_confidence:.2f})")
        print(f"Summary: {result.summary}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Entities: {len(result.entities)}")
        print(f"Key phrases: {result.key_phrases[:5]}")
        print(f"Concepts: {[c['text'] for c in result.concepts[:5]]}")


if __name__ == "__main__":
    asyncio.run(main())