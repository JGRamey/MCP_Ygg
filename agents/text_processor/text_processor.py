#!/usr/bin/env python3
"""
MCP Server Text Processing Agent
Cleans text, extracts entities, and generates embeddings using spaCy and Sentence-BERT
"""

import asyncio
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import aiofiles
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Data structure for processed documents"""
    doc_id: str
    title: str
    author: Optional[str]
    original_content: str
    cleaned_content: str
    chunks: List[str]
    entities: List[Dict]
    embeddings: np.ndarray
    chunk_embeddings: List[np.ndarray]
    domain: str
    subcategory: str
    language: str
    date: Optional[str]
    word_count: int
    chunk_count: int
    processing_metadata: Dict
    processed_at: datetime


@dataclass
class Entity:
    """Named entity structure"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: Optional[str] = None
    uri: Optional[str] = None  # For knowledge base linking


@dataclass
class TextChunk:
    """Text chunk with metadata"""
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    entities: List[Entity] = None
    metadata: Dict = None


class LanguageDetector:
    """Advanced language detection"""
    
    def __init__(self):
        self.language_patterns = {
            'english': {
                'chars': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'common_words': {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'},
                'char_range': (0x0020, 0x007F)
            },
            'latin': {
                'chars': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'common_words': {'et', 'in', 'ad', 'de', 'ex', 'cum', 'pro', 'per', 'ab', 'sine'},
                'char_range': (0x0020, 0x007F)
            },
            'greek': {
                'chars': set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'),
                'common_words': {'καὶ', 'τοῦ', 'τῆς', 'τὸ', 'τὴν', 'εἰς', 'ἐν', 'τῶν', 'τὰ', 'τῷ'},
                'char_range': (0x0370, 0x03FF)
            },
            'sanskrit': {
                'chars': set(),  # Devanagari script
                'common_words': {'च', 'तत्', 'एव', 'तु', 'अथ', 'यत्', 'इति', 'सः', 'तस्य'},
                'char_range': (0x0900, 0x097F)
            },
            'arabic': {
                'chars': set(),
                'common_words': {'في', 'من', 'إلى', 'على', 'هذا', 'التي', 'التي', 'كان', 'أن'},
                'char_range': (0x0600, 0x06FF)
            },
            'hebrew': {
                'chars': set(),
                'common_words': {'את', 'של', 'על', 'אל', 'כל', 'זה', 'הוא', 'לא', 'יש'},
                'char_range': (0x0590, 0x05FF)
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        if not text.strip():
            return 'unknown', 0.0
        
        scores = {}
        text_words = set(text.lower().split())
        
        for lang, patterns in self.language_patterns.items():
            score = 0.0
            
            # Character range scoring
            char_count = 0
            for char in text:
                char_code = ord(char)
                if patterns['char_range'][0] <= char_code <= patterns['char_range'][1]:
                    char_count += 1
            
            if len(text) > 0:
                char_score = char_count / len(text)
                score += char_score * 0.6
            
            # Common words scoring
            common_word_matches = len(text_words.intersection(patterns['common_words']))
            if len(text_words) > 0:
                word_score = common_word_matches / len(text_words)
                score += word_score * 0.4
            
            scores[lang] = score
        
        if not scores:
            return 'unknown', 0.0
        
        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang]
        
        return best_lang, confidence


class MultilingualNLP:
    """Multilingual NLP processing with spaCy"""
    
    def __init__(self):
        self.models = {}
        self.supported_languages = {
            'english': 'en_core_web_lg',
            'latin': 'la_core_web_sm',  # If available
            'greek': 'grc_proiel_sm',   # Ancient Greek if available
            'german': 'de_core_news_lg',
            'french': 'fr_core_news_lg',
            'spanish': 'es_core_news_lg'
        }
        self.load_models()
    
    def load_models(self) -> None:
        """Load available spaCy models"""
        for lang, model_name in self.supported_languages.items():
            try:
                self.models[lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model for {lang}: {model_name}")
            except OSError:
                logger.warning(f"Model {model_name} not available for {lang}")
                # Fallback to basic English model
                if lang != 'english':
                    try:
                        self.models[lang] = spacy.load('en_core_web_sm')
                        logger.info(f"Using English model as fallback for {lang}")
                    except OSError:
                        logger.error(f"No fallback model available for {lang}")
    
    def get_model(self, language: str) -> spacy.Language:
        """Get spaCy model for language"""
        return self.models.get(language, self.models.get('english'))
    
    def extract_entities(self, text: str, language: str = 'english') -> List[Entity]:
        """Extract named entities from text"""
        nlp = self.get_model(language)
        if not nlp:
            return []
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores by default
                description=spacy.explain(ent.label_)
            )
            entities.append(entity)
        
        return entities
    
    def get_linguistic_features(self, text: str, language: str = 'english') -> Dict:
        """Extract linguistic features from text"""
        nlp = self.get_model(language)
        if not nlp:
            return {}
        
        doc = nlp(text)
        
        features = {
            'sentence_count': len(list(doc.sents)),
            'token_count': len(doc),
            'pos_tags': [(token.text, token.pos_) for token in doc if not token.is_space],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'dependency_relations': [(token.text, token.dep_, token.head.text) for token in doc if not token.is_space],
            'lemmas': [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        }
        
        return features


class EmbeddingGenerator:
    """Generate embeddings using Sentence-BERT and other models"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedding models"""
        self.model_name = model_name
        self.sentence_model = None
        self.multilingual_model = None
        self.domain_models = {}
        self.load_models()
    
    def load_models(self) -> None:
        """Load embedding models"""
        try:
            # General purpose model
            self.sentence_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded Sentence-BERT model: {self.model_name}")
            
            # Multilingual model
            self.multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Loaded multilingual embedding model")
            
            # Domain-specific models
            domain_models = {
                'science': 'allenai/scibert_scivocab_uncased',
                'math': 'microsoft/codebert-base',  # Often good for mathematical text
                'literature': 'sentence-transformers/all-MiniLM-L6-v2'  # General model
            }
            
            for domain, model_name in domain_models.items():
                try:
                    self.domain_models[domain] = SentenceTransformer(model_name)
                    logger.info(f"Loaded domain model for {domain}: {model_name}")
                except Exception as e:
                    logger.warning(f"Could not load domain model for {domain}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading embedding models: {e}")
    
    def get_model_for_domain(self, domain: str, language: str = 'english') -> SentenceTransformer:
        """Get appropriate model for domain and language"""
        # Use multilingual model for non-English languages
        if language != 'english' and self.multilingual_model:
            return self.multilingual_model
        
        # Use domain-specific model if available
        if domain in self.domain_models:
            return self.domain_models[domain]
        
        # Fallback to general model
        return self.sentence_model
    
    def generate_embeddings(self, texts: List[str], domain: str = 'general', language: str = 'english') -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not texts:
            return np.array([])
        
        model = self.get_model_for_domain(domain, language)
        if not model:
            logger.error("No embedding model available")
            return np.array([])
        
        try:
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def generate_single_embedding(self, text: str, domain: str = 'general', language: str = 'english') -> np.ndarray:
        """Generate embedding for single text"""
        embeddings = self.generate_embeddings([text], domain, language)
        return embeddings[0] if len(embeddings) > 0 else np.array([])


class TextChunker:
    """Intelligent text chunking for processing"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size for chunks (in characters)
            overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str, nlp_model: spacy.Language) -> List[TextChunk]:
        """Chunk text by sentences, respecting boundaries"""
        doc = nlp_model(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sent in sentences:
            sent_text = sent.text.strip()
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sent_text) > self.chunk_size and current_chunk:
                chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={'sentence_count': current_chunk.count('.')}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sent_text
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sent_text) - 1
            else:
                if current_chunk:
                    current_chunk += " " + sent_text
                else:
                    current_chunk = sent_text
                    current_start = sent.start_char
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={'sentence_count': current_chunk.count('.')}
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Chunk text by paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={'paragraph_count': current_chunk.count('\n\n') + 1}
                )
                chunks.append(chunk)
                
                current_chunk = para
                current_start = text.find(para, current_start)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = text.find(para)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={'paragraph_count': current_chunk.count('\n\n') + 1}
            )
            chunks.append(chunk)
        
        return chunks


class TextCleaner:
    """Advanced text cleaning and normalization"""
    
    @staticmethod
    def clean_text(text: str, language: str = 'english') -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('\ufeff', '')  # Remove BOM
        text = text.replace('\u00a0', ' ')  # Replace non-breaking space
        
        # Language-specific cleaning
        if language == 'latin':
            # Normalize Latin text
            text = TextCleaner._normalize_latin(text)
        elif language == 'greek':
            # Normalize Greek text
            text = TextCleaner._normalize_greek(text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Normalize punctuation
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'…', '...', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([()[\]{}])\s*', r' \1 ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def _normalize_latin(text: str) -> str:
        """Normalize Latin text"""
        # Common Latin text normalizations
        replacements = {
            'æ': 'ae',
            'œ': 'oe',
            'v': 'u',  # Classical Latin uses 'u' instead of 'v'
            'j': 'i',  # Classical Latin uses 'i' instead of 'j'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def _normalize_greek(text: str) -> str:
        """Normalize Ancient Greek text"""
        # Remove or normalize diacritics if needed
        # This is a simplified version - full Greek normalization is complex
        
        # Normalize breathing marks and accents for consistency
        text = re.sub(r'[᾿῾]', '', text)  # Remove breathing marks if desired
        
        return text
    
    @staticmethod
    def extract_structure(text: str) -> Dict:
        """Extract document structure (headers, sections, etc.)"""
        structure = {
            'headers': [],
            'sections': [],
            'chapters': [],
            'footnotes': [],
            'quotes': []
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect headers (lines in all caps or title case)
            if line and (line.isupper() or (line.istitle() and len(line) < 100)):
                structure['headers'].append({'text': line, 'line': i})
            
            # Detect chapters
            if re.match(r'chapter\s+\d+', line, re.IGNORECASE):
                structure['chapters'].append({'text': line, 'line': i})
            
            # Detect footnotes
            if re.match(r'^\d+\.\s+', line):
                structure['footnotes'].append({'text': line, 'line': i})
            
            # Detect quotes (lines starting with quote marks)
            if line.startswith('"') and line.endswith('"'):
                structure['quotes'].append({'text': line, 'line': i})
        
        return structure


class TextProcessor:
    """Main text processing orchestrator"""
    
    def __init__(self, config_path: str = "agents/text_processor/config.yaml"):
        """Initialize text processor with configuration"""
        self.load_config(config_path)
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.nlp = MultilingualNLP()
        self.embedding_generator = EmbeddingGenerator()
        self.chunker = TextChunker(
            chunk_size=self.config.get('chunk_size', 1000),
            overlap=self.config.get('chunk_overlap', 200)
        )
        
        # Setup parallel processing
        self.max_workers = self.config.get('max_workers', mp.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def load_config(self, config_path: str) -> None:
        """Load processor configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'min_text_length': 100,
                'max_text_length': 1000000,
                'max_workers': mp.cpu_count(),
                'embedding_model': 'all-MiniLM-L6-v2',
                'output_dir': 'data/processed',
                'supported_languages': ['english', 'latin', 'greek', 'german', 'french', 'spanish'],
                'enable_entity_linking': True,
                'enable_chunking': True,
                'parallel_processing': True
            }
    
    async def process_document(self, doc_data: Dict) -> Optional[ProcessedDocument]:
        """Process a single document"""
        try:
            # Extract basic information
            content = doc_data.get('content', '')
            if len(content) < self.config.get('min_text_length', 100):
                logger.warning(f"Document too short: {len(content)} characters")
                return None
            
            if len(content) > self.config.get('max_text_length', 1000000):
                logger.warning(f"Document too long: {len(content)} characters")
                content = content[:self.config.get('max_text_length', 1000000)]
            
            # Detect language
            language, lang_confidence = self.language_detector.detect_language(content)
            logger.info(f"Detected language: {language} (confidence: {lang_confidence:.2f})")
            
            # Clean text
            cleaned_content = TextCleaner.clean_text(content, language)
            
            # Extract structure
            structure = TextCleaner.extract_structure(cleaned_content)
            
            # Extract entities
            entities = await self._extract_entities_async(cleaned_content, language)
            
            # Generate chunks
            chunks = await self._generate_chunks_async(cleaned_content, language)
            
            # Generate embeddings
            doc_embedding = await self._generate_doc_embedding_async(
                cleaned_content, doc_data.get('domain', 'general'), language
            )
            
            chunk_embeddings = await self._generate_chunk_embeddings_async(
                chunks, doc_data.get('domain', 'general'), language
            )
            
            # Create processed document
            doc_id = doc_data.get('checksum', hashlib.md5(content.encode()).hexdigest())
            
            processed_doc = ProcessedDocument(
                doc_id=doc_id,
                title=doc_data.get('title', ''),
                author=doc_data.get('author', ''),
                original_content=content,
                cleaned_content=cleaned_content,
                chunks=[chunk.text for chunk in chunks],
                entities=[asdict(entity) for entity in entities],
                embeddings=doc_embedding,
                chunk_embeddings=chunk_embeddings,
                domain=doc_data.get('domain', ''),
                subcategory=doc_data.get('subcategory', ''),
                language=language,
                date=doc_data.get('date', ''),
                word_count=len(cleaned_content.split()),
                chunk_count=len(chunks),
                processing_metadata={
                    'language_confidence': lang_confidence,
                    'structure': structure,
                    'entity_count': len(entities),
                    'embedding_model': self.embedding_generator.model_name,
                    'chunk_method': 'sentences'
                },
                processed_at=datetime.now()
            )
            
            # Save processed document
            await self._save_processed_document(processed_doc)
            
            logger.info(f"Successfully processed document: {doc_id}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    async def _extract_entities_async(self, text: str, language: str) -> List[Entity]:
        """Extract entities asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.nlp.extract_entities, 
            text, 
            language
        )
    
    async def _generate_chunks_async(self, text: str, language: str) -> List[TextChunk]:
        """Generate chunks asynchronously"""
        nlp_model = self.nlp.get_model(language)
        if not nlp_model:
            # Fallback to paragraph chunking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.chunker.chunk_by_paragraphs,
                text
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.chunker.chunk_by_sentences,
            text,
            nlp_model
        )
    
    async def _generate_doc_embedding_async(self, text: str, domain: str, language: str) -> np.ndarray:
        """Generate document embedding asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.embedding_generator.generate_single_embedding,
            text,
            domain,
            language
        )
    
    async def _generate_chunk_embeddings_async(self, chunks: List[TextChunk], domain: str, language: str) -> List[np.ndarray]:
        """Generate chunk embeddings asynchronously"""
        chunk_texts = [chunk.text for chunk in chunks]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embedding_generator.generate_embeddings,
            chunk_texts,
            domain,
            language
        )
        return [embeddings[i] for i in range(len(embeddings))]
    
    async def _save_processed_document(self, doc: ProcessedDocument) -> None:
        """Save processed document to storage"""
        output_dir = Path(self.config.get('output_dir', 'data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main document data
        doc_file = output_dir / f"{doc.doc_id}.json"
        doc_dict = asdict(doc)
        
        # Convert numpy arrays to lists for JSON serialization
        doc_dict['embeddings'] = doc.embeddings.tolist() if doc.embeddings.size > 0 else []
        doc_dict['chunk_embeddings'] = [emb.tolist() for emb in doc.chunk_embeddings]
        doc_dict['processed_at'] = doc.processed_at.isoformat()
        
        async with aiofiles.open(doc_file, 'w') as f:
            await f.write(json.dumps(doc_dict, indent=2, ensure_ascii=False))
        
        # Save embeddings separately in binary format for efficiency
        embeddings_file = output_dir / f"{doc.doc_id}_embeddings.pkl"
        embeddings_data = {
            'doc_embedding': doc.embeddings,
            'chunk_embeddings': doc.chunk_embeddings,
            'metadata': {
                'model': self.embedding_generator.model_name,
                'dimension': doc.embeddings.shape[0] if doc.embeddings.size > 0 else 0,
                'chunk_count': len(doc.chunk_embeddings)
            }
        }
        
        async with aiofiles.open(embeddings_file, 'wb') as f:
            await f.write(pickle.dumps(embeddings_data))
    
    async def process_batch(self, documents: List[Dict]) -> List[ProcessedDocument]:
        """Process multiple documents in parallel"""
        tasks = [self.process_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_docs = []
        for result in results:
            if isinstance(result, ProcessedDocument):
                processed_docs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Document processing failed: {result}")
        
        return processed_docs
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


async def main():
    """Example usage of text processor"""
    # Example document data
    doc_data = {
        'content': """
        This is a sample document for testing the text processing pipeline.
        It contains multiple sentences and paragraphs to demonstrate chunking.
        
        The document discusses various topics including natural language processing,
        machine learning, and information retrieval. Named entities like OpenAI,
        Google, and New York City should be detected.
        
        This text will be cleaned, analyzed for entities, split into chunks,
        and converted into embeddings for semantic search.
        """,
        'title': 'Sample Document',
        'author': 'Test Author',
        'domain': 'science',
        'subcategory': 'computer_science',
        'checksum': 'sample_doc_123'
    }
    
    processor = TextProcessor()
    
    try:
        processed_doc = await processor.process_document(doc_data)
        if processed_doc:
            print(f"Processed document: {processed_doc.title}")
            print(f"Language: {processed_doc.language}")
            print(f"Word count: {processed_doc.word_count}")
            print(f"Chunks: {processed_doc.chunk_count}")
            print(f"Entities: {len(processed_doc.entities)}")
            print(f"Embedding dimension: {processed_doc.embeddings.shape[0] if processed_doc.embeddings.size > 0 else 0}")
    finally:
        processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
