#!/usr/bin/env python3
"""
Text Processor Utility Functions
Helper functions for text processing, chunking, normalization, and analysis
"""

import re
import string
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime
import statistics
from collections import Counter, defaultdict
import math

import numpy as np
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.matcher import Matcher, PhraseMatcher
from sentence_transformers.util import cos_sim
import torch

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Advanced text normalization utilities"""
    
    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text to canonical form"""
        if not text:
            return ""
        
        try:
            return unicodedata.normalize(form, text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Remove accents from text while preserving base characters"""
        if not text:
            return ""
        
        try:
            # Decompose characters into base + accent
            nfd = unicodedata.normalize('NFD', text)
            # Filter out accent marks (category Mn = Mark, nonspacing)
            without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
            # Recompose
            return unicodedata.normalize('NFC', without_accents)
        except Exception as e:
            logger.warning(f"Accent removal failed: {e}")
            return text
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Normalize various quote characters to standard forms"""
        quote_map = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '`': "'",  # Grave accent used as quote
            '´': "'",  # Acute accent used as quote
            '„': '"',  # Double low-9 quotation mark
            '‚': "'",  # Single low-9 quotation mark
            '«': '"',  # Left-pointing double angle quotation mark
            '»': '"',  # Right-pointing double angle quotation mark
        }
        
        for old, new in quote_map.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def normalize_dashes(text: str) -> str:
        """Normalize various dash characters"""
        dash_map = {
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '―': '-',  # Horizontal bar
            '−': '-',  # Minus sign
        }
        
        for old, new in dash_map.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize various whitespace characters"""
        # Replace various whitespace characters with standard space
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)  # Windows to Unix
        text = re.sub(r'\r', '\n', text)    # Mac to Unix
        
        return text.strip()
    
    @staticmethod
    def clean_control_characters(text: str) -> str:
        """Remove control characters while preserving necessary ones"""
        if not text:
            return ""
        
        # Keep tab, newline, and carriage return
        allowed_controls = {'\t', '\n', '\r'}
        
        cleaned = []
        for char in text:
            if unicodedata.category(char).startswith('C') and char not in allowed_controls:
                continue  # Skip control character
            cleaned.append(char)
        
        return ''.join(cleaned)


class HistoricalTextProcessor:
    """Specialized processing for historical and ancient texts"""
    
    LATIN_ABBREVIATIONS = {
        'etc.': 'et cetera',
        'cf.': 'confer',
        'e.g.': 'exempli gratia',
        'i.e.': 'id est',
        'vs.': 'versus',
        'et al.': 'et alii'
    }
    
    ARCHAIC_SPELLINGS = {
        'ye': 'the',
        'thou': 'you',
        'thee': 'you',
        'thy': 'your',
        'thine': 'your',
        'shalt': 'shall',
        'doth': 'does',
        'hath': 'has'
    }
    
    @classmethod
    def normalize_archaic_english(cls, text: str) -> str:
        """Normalize archaic English spellings"""
        words = text.split()
        normalized = []
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = word.strip(string.punctuation).lower()
            
            if clean_word in cls.ARCHAIC_SPELLINGS:
                # Preserve original capitalization
                if word[0].isupper():
                    replacement = cls.ARCHAIC_SPELLINGS[clean_word].capitalize()
                else:
                    replacement = cls.ARCHAIC_SPELLINGS[clean_word]
                
                # Restore punctuation
                if word != clean_word:
                    punct = ''.join(c for c in word if c in string.punctuation)
                    replacement += punct
                
                normalized.append(replacement)
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    @classmethod
    def handle_biblical_references(cls, text: str) -> List[Dict]:
        """Extract and normalize biblical references"""
        # Pattern for biblical references (e.g., "John 3:16", "1 Corinthians 13:4-7")
        pattern = r'\b(\d*\s*[A-Z][a-z]+)\s+(\d+):(\d+)(?:-(\d+))?(?::(\d+))?\b'
        
        references = []
        for match in re.finditer(pattern, text):
            book = match.group(1).strip()
            chapter = int(match.group(2))
            verse_start = int(match.group(3))
            verse_end = int(match.group(4)) if match.group(4) else verse_start
            
            references.append({
                'book': book,
                'chapter': chapter,
                'verse_start': verse_start,
                'verse_end': verse_end,
                'full_reference': match.group(0),
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        return references
    
    @staticmethod
    def preserve_verse_structure(text: str) -> Dict:
        """Preserve verse structure in religious texts"""
        # Detect verse numbers (1., 2., etc.)
        verse_pattern = r'^(\d+)\.\s*(.+)$'
        verses = []
        
        for line_num, line in enumerate(text.split('\n')):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(verse_pattern, line)
            if match:
                verse_num = int(match.group(1))
                verse_text = match.group(2)
                verses.append({
                    'number': verse_num,
                    'text': verse_text,
                    'line': line_num
                })
        
        return {
            'verses': verses,
            'has_verse_structure': len(verses) > 0,
            'verse_count': len(verses)
        }


class MathematicalTextProcessor:
    """Specialized processing for mathematical texts"""
    
    MATH_SYMBOLS = {
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'θ': 'theta',
        'λ': 'lambda',
        'μ': 'mu',
        'π': 'pi',
        'σ': 'sigma',
        'φ': 'phi',
        '∞': 'infinity',
        '∂': 'partial',
        '∫': 'integral',
        '∑': 'sum',
        '∏': 'product',
        '√': 'sqrt',
        '≤': 'less_equal',
        '≥': 'greater_equal',
        '≠': 'not_equal',
        '≈': 'approximately',
        '∈': 'in',
        '⊆': 'subset',
        '∪': 'union',
        '∩': 'intersection'
    }
    
    @classmethod
    def extract_equations(cls, text: str) -> List[Dict]:
        """Extract mathematical equations from text"""
        equations = []
        
        # LaTeX equations
        latex_patterns = [
            r'\$\$(.+?)\$\$',  # Display math
            r'\$(.+?)\$',      # Inline math
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\begin\{align\}(.+?)\\end\{align\}',
            r'\\\[(.+?)\\]'    # Display math alternative
        ]
        
        for pattern in latex_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                equations.append({
                    'content': match.group(1).strip(),
                    'type': 'latex',
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'full_match': match.group(0)
                })
        
        # Simple mathematical expressions
        math_pattern = r'\b[a-zA-Z]\s*[=<>≤≥≠]\s*[a-zA-Z0-9+\-*/().\s]+\b'
        for match in re.finditer(math_pattern, text):
            equations.append({
                'content': match.group(0).strip(),
                'type': 'simple',
                'start_pos': match.start(),
                'end_pos': match.end(),
                'full_match': match.group(0)
            })
        
        return equations
    
    @classmethod
    def normalize_math_symbols(cls, text: str) -> str:
        """Normalize mathematical symbols to text representations"""
        for symbol, name in cls.MATH_SYMBOLS.items():
            text = text.replace(symbol, f' {name} ')
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_theorems_and_proofs(text: str) -> List[Dict]:
        """Extract mathematical theorems and proofs"""
        results = []
        
        # Theorem pattern
        theorem_pattern = r'(Theorem|Lemma|Corollary|Proposition)\s*(\d+(?:\.\d+)*)?\.?\s*(?:\((.*?)\))?\s*:?\s*(.*?)(?=\n\s*\n|\n\s*(?:Proof|Theorem|Lemma|Corollary|Proposition|$))'
        
        for match in re.finditer(theorem_pattern, text, re.DOTALL | re.IGNORECASE):
            theorem_type = match.group(1)
            theorem_number = match.group(2) or ""
            theorem_name = match.group(3) or ""
            theorem_statement = match.group(4).strip()
            
            results.append({
                'type': 'theorem',
                'theorem_type': theorem_type.lower(),
                'number': theorem_number,
                'name': theorem_name,
                'statement': theorem_statement,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        # Proof pattern
        proof_pattern = r'Proof\.?\s*(.*?)(?=\n\s*\n|\n\s*(?:Theorem|Lemma|Corollary|Proposition|$)|□|∎|QED)'
        
        for match in re.finditer(proof_pattern, text, re.DOTALL | re.IGNORECASE):
            proof_content = match.group(1).strip()
            
            results.append({
                'type': 'proof',
                'content': proof_content,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        return results


class SemanticAnalyzer:
    """Semantic analysis utilities"""
    
    def __init__(self, nlp_model: spacy.Language):
        self.nlp = nlp_model
        self.matcher = Matcher(nlp_model.vocab)
        self.phrase_matcher = PhraseMatcher(nlp_model.vocab)
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup semantic patterns for matching"""
        # Argument structure patterns
        argument_patterns = [
            [{"LOWER": "therefore"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": {"IN": ["NOUN", "PRON"]}}],
            [{"LOWER": "thus"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": {"IN": ["NOUN", "PRON"]}}],
            [{"LOWER": "consequently"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": {"IN": ["NOUN", "PRON"]}}],
            [{"LOWER": "however"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": {"IN": ["NOUN", "PRON"]}}],
        ]
        
        for i, pattern in enumerate(argument_patterns):
            self.matcher.add(f"ARGUMENT_{i}", [pattern])
        
        # Citation patterns
        citation_patterns = [
            [{"LOWER": "according"}, {"LOWER": "to"}, {"POS": "PROPN"}],
            [{"POS": "PROPN"}, {"LOWER": {"IN": ["argues", "claims", "states", "suggests"]}}],
            [{"LOWER": {"IN": ["as", "according to"]}}, {"POS": "PROPN"}, {"LOWER": {"IN": ["noted", "observed", "argued"]}}],
        ]
        
        for i, pattern in enumerate(citation_patterns):
            self.matcher.add(f"CITATION_{i}", [pattern])
    
    def extract_arguments(self, text: str) -> List[Dict]:
        """Extract argumentative structures from text"""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        arguments = []
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label.startswith("ARGUMENT"):
                span = doc[start:end]
                
                # Find the full sentence containing this argument marker
                sent = span.sent
                
                arguments.append({
                    'type': 'argument_marker',
                    'marker': span.text,
                    'sentence': sent.text,
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'pattern': label
                })
        
        return arguments
    
    def extract_citations(self, text: str) -> List[Dict]:
        """Extract citations and references from text"""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        citations = []
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label.startswith("CITATION"):
                span = doc[start:end]
                sent = span.sent
                
                citations.append({
                    'type': 'citation',
                    'text': span.text,
                    'sentence': sent.text,
                    'start_char': span.start_char,
                    'end_char': span.end_char,
                    'pattern': label
                })
        
        return citations
    
    def analyze_text_structure(self, text: str) -> Dict:
        """Analyze the logical structure of text"""
        doc = self.nlp(text)
        
        structure = {
            'sentences': len(list(doc.sents)),
            'paragraphs': len(text.split('\n\n')),
            'arguments': self.extract_arguments(text),
            'citations': self.extract_citations(text),
            'discourse_markers': self._find_discourse_markers(doc),
            'coherence_score': self._calculate_coherence(doc)
        }
        
        return structure
    
    def _find_discourse_markers(self, doc: Doc) -> List[Dict]:
        """Find discourse markers that indicate text structure"""
        markers = {
            'contrast': ['however', 'but', 'nevertheless', 'although', 'despite'],
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'conclusion': ['therefore', 'thus', 'consequently', 'hence', 'in conclusion'],
            'sequence': ['first', 'second', 'next', 'finally', 'then'],
            'causation': ['because', 'since', 'due to', 'as a result', 'leads to']
        }
        
        found_markers = []
        
        for token in doc:
            for marker_type, marker_list in markers.items():
                if token.lower_ in marker_list:
                    found_markers.append({
                        'type': marker_type,
                        'marker': token.text,
                        'position': token.i,
                        'sentence': token.sent.text
                    })
        
        return found_markers
    
    def _calculate_coherence(self, doc: Doc) -> float:
        """Calculate a simple coherence score based on entity overlap"""
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            sent1_entities = {ent.text.lower() for ent in sentences[i].ents}
            sent2_entities = {ent.text.lower() for ent in sentences[i + 1].ents}
            
            if sent1_entities or sent2_entities:
                overlap = len(sent1_entities.intersection(sent2_entities))
                total = len(sent1_entities.union(sent2_entities))
                score = overlap / total if total > 0 else 0.0
                coherence_scores.append(score)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.0


class EmbeddingUtils:
    """Utilities for working with embeddings"""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if a.size == 0 or b.size == 0:
            return 0.0
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def find_similar_chunks(
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Find most similar chunks to query"""
        if not chunk_embeddings or query_embedding.size == 0:
            return []
        
        similarities = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            if chunk_emb.size > 0:
                sim = EmbeddingUtils.cosine_similarity(query_embedding, chunk_emb)
                if sim >= threshold:
                    similarities.append((i, sim))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def cluster_embeddings(
        embeddings: List[np.ndarray],
        n_clusters: int = 5,
        method: str = 'kmeans'
    ) -> List[int]:
        """Cluster embeddings using specified method"""
        if not embeddings or len(embeddings) < n_clusters:
            return list(range(len(embeddings)))
        
        # Convert to matrix
        embedding_matrix = np.vstack([emb for emb in embeddings if emb.size > 0])
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embedding_matrix)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(embedding_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels.tolist()
    
    @staticmethod
    def reduce_dimensionality(
        embeddings: List[np.ndarray],
        n_components: int = 50,
        method: str = 'pca'
    ) -> List[np.ndarray]:
        """Reduce dimensionality of embeddings"""
        if not embeddings:
            return []
        
        embedding_matrix = np.vstack([emb for emb in embeddings if emb.size > 0])
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embedding_matrix)
        elif method == 'umap':
            import umap
            reducer = umap.UMAP(n_components=n_components)
            reduced = reducer.fit_transform(embedding_matrix)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return [reduced[i] for i in range(reduced.shape[0])]


class TextStatistics:
    """Calculate various text statistics"""
    
    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """Calculate readability scores"""
        if not text:
            return {}
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        syllables = sum(TextStatistics._count_syllables(word) for word in words)
        
        if not sentences or not words:
            return {}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            'flesch_reading_ease': flesch_ease,
            'flesch_kincaid_grade': flesch_kincaid,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word,
            'total_sentences': len(sentences),
            'total_words': len(words),
            'total_syllables': syllables
        }
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word"""
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least one syllable
    
    @staticmethod
    def calculate_lexical_diversity(text: str) -> Dict[str, float]:
        """Calculate lexical diversity metrics"""
        if not text:
            return {}
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {}
        
        unique_words = set(words)
        word_freq = Counter(words)
        
        # Type-Token Ratio (TTR)
        ttr = len(unique_words) / len(words)
        
        # Mean Segmental Type-Token Ratio (MSTTR)
        segment_size = 50
        segments = [words[i:i+segment_size] for i in range(0, len(words), segment_size)]
        segment_ttrs = [len(set(segment)) / len(segment) for segment in segments if len(segment) >= 10]
        msttr = statistics.mean(segment_ttrs) if segment_ttrs else 0.0
        
        # Hapax Legomena (words appearing exactly once)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax_count / len(words)
        
        return {
            'type_token_ratio': ttr,
            'mean_segmental_ttr': msttr,
            'hapax_legomena_ratio': hapax_ratio,
            'unique_words': len(unique_words),
            'total_words': len(words),
            'vocabulary_size': len(unique_words)
        }


def load_processed_document(doc_id: str, base_dir: str = "data/processed") -> Optional[Dict]:
    """Load a processed document from storage"""
    doc_file = Path(base_dir) / f"{doc_id}.json"
    embeddings_file = Path(base_dir) / f"{doc_id}_embeddings.pkl"
    
    try:
        # Load main document data
        with open(doc_file, 'r') as f:
            doc_data = json.load(f)
        
        # Load embeddings if available
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
                doc_data['embeddings'] = embeddings_data['doc_embedding']
                doc_data['chunk_embeddings'] = embeddings_data['chunk_embeddings']
        
        return doc_data
        
    except Exception as e:
        logger.error(f"Error loading document {doc_id}: {e}")
        return None


def batch_process_texts(
    texts: List[str],
    processor_func,
    batch_size: int = 10,
    max_workers: int = 4
) -> List[Any]:
    """Process texts in batches with multiprocessing"""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    if len(texts) <= batch_size:
        return [processor_func(text) for text in texts]
    
    # Split into batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    with ProcessPoolExecutor(max_workers=min(max_workers, mp.cpu_count())) as executor:
        batch_results = list(executor.map(
            lambda batch: [processor_func(text) for text in batch],
            batches
        ))
    
    # Flatten results
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results


def calculate_text_hash(text: str, algorithm: str = 'sha256') -> str:
    """Calculate hash of text content for deduplication"""
    if not text:
        return ""
    
    # Normalize text for consistent hashing
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    
    if algorithm == 'md5':
        return hashlib.md5(normalized.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(normalized.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(normalized.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


if __name__ == "__main__":
    print("Text processor utilities loaded successfully")
