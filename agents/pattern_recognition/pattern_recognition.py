#!/usr/bin/env python3
"""
MCP Server Pattern Recognition Agent
Detects cross-domain patterns (e.g., Trinity in religion vs. quantum mechanics) with user validation
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
import re
from collections import defaultdict, Counter
import math
import statistics

import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConceptCluster:
    """Represents a cluster of related concepts"""
    cluster_id: str
    concepts: List[str]
    domains: List[str]
    centroid: np.ndarray
    similarity_scores: List[float]
    representative_concept: str
    cluster_size: int
    intra_cluster_similarity: float


@dataclass
class CrossDomainPattern:
    """Represents a pattern found across multiple domains"""
    pattern_id: str
    name: str
    description: str
    pattern_type: str
    domains: List[str]
    confidence: float
    evidence: List[Dict[str, Any]]
    examples: List[str]
    concept_clusters: List[ConceptCluster]
    statistical_significance: float
    validated: bool = False
    validation_notes: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class PatternEvidence:
    """Evidence supporting a pattern"""
    evidence_id: str
    pattern_id: str
    domain: str
    document_id: str
    concept: str
    context: str
    similarity_score: float
    supporting_text: str
    metadata: Dict[str, Any]


class ConceptExtractor:
    """Extracts and normalizes concepts from text"""
    
    def __init__(self):
        """Initialize concept extractor"""
        self.nlp = None
        self.domain_vocabularies = self._load_domain_vocabularies()
        self.concept_patterns = self._load_concept_patterns()
        self.load_nlp_models()
    
    def load_nlp_models(self):
        """Load NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy model for concept extraction")
        except OSError:
            logger.warning("spaCy model not found, using basic extraction")
            self.nlp = None
    
    def _load_domain_vocabularies(self) -> Dict[str, Set[str]]:
        """Load domain-specific vocabularies"""
        vocabularies = {
            'religion': {
                'trinity', 'triune', 'threefold', 'father', 'son', 'spirit',
                'divine', 'sacred', 'holy', 'salvation', 'resurrection',
                'incarnation', 'redemption', 'grace', 'faith', 'prayer',
                'scripture', 'doctrine', 'theology', 'mysticism', 'ritual'
            },
            'science': {
                'quantum', 'particle', 'wave', 'energy', 'matter', 'force',
                'field', 'dimension', 'space', 'time', 'relativity',
                'uncertainty', 'superposition', 'entanglement', 'duality',
                'conservation', 'symmetry', 'interaction', 'fundamental'
            },
            'philosophy': {
                'existence', 'being', 'consciousness', 'reality', 'truth',
                'knowledge', 'wisdom', 'ethics', 'morality', 'justice',
                'beauty', 'good', 'evil', 'mind', 'soul', 'free will',
                'determinism', 'causation', 'essence', 'substance'
            },
            'math': {
                'number', 'infinity', 'theorem', 'proof', 'axiom', 'set',
                'function', 'limit', 'continuity', 'derivative', 'integral',
                'symmetry', 'group', 'field', 'ring', 'space', 'dimension',
                'topology', 'geometry', 'algebra', 'analysis'
            },
            'literature': {
                'narrative', 'character', 'plot', 'theme', 'symbol',
                'metaphor', 'allegory', 'irony', 'tragedy', 'comedy',
                'epic', 'lyric', 'drama', 'poetry', 'prose', 'style',
                'voice', 'perspective', 'archetype', 'motif'
            },
            'history': {
                'civilization', 'culture', 'empire', 'revolution', 'war',
                'peace', 'ruler', 'dynasty', 'era', 'period', 'evolution',
                'progress', 'decline', 'renaissance', 'enlightenment',
                'tradition', 'innovation', 'legacy', 'influence', 'change'
            }
        }
        return vocabularies
    
    def _load_concept_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate concepts"""
        return {
            'trinity_patterns': [
                r'three\s+(?:part|fold|dimensional?|aspect)',
                r'triad|trinity|triplet|threesome',
                r'father,?\s+son,?\s+(?:and\s+)?(?:holy\s+)?spirit',
                r'mind,?\s+body,?\s+(?:and\s+)?(?:soul|spirit)',
                r'past,?\s+present,?\s+(?:and\s+)?future',
                r'thesis,?\s+antithesis,?\s+(?:and\s+)?synthesis'
            ],
            'duality_patterns': [
                r'two\s+(?:part|fold|dimensional?|aspect)',
                r'dual(?:ity|ism)?|binary|dichotomy',
                r'wave\s+(?:and\s+)?particle',
                r'mind\s+(?:and\s+)?body',
                r'good\s+(?:and\s+)?evil',
                r'light\s+(?:and\s+)?dark(?:ness)?'
            ],
            'unity_patterns': [
                r'one(?:ness)?|unity|unification|integration',
                r'all\s+(?:is\s+)?one|everything\s+(?:is\s+)?connected',
                r'universal|cosmic|absolute',
                r'whole(?:ness)?|totality|completeness'
            ],
            'transformation_patterns': [
                r'transform(?:ation)?|metamorphosis|change',
                r'evolution|development|progress|growth',
                r'become|becoming|emergence|arising',
                r'death\s+(?:and\s+)?(?:re)?birth|resurrection'
            ]
        }
    
    def extract_concepts(self, text: str, domain: str) -> List[Tuple[str, float]]:
        """Extract concepts from text with confidence scores"""
        concepts = []
        text_lower = text.lower()
        
        # Domain vocabulary matching
        domain_vocab = self.domain_vocabularies.get(domain, set())
        for concept in domain_vocab:
            if concept in text_lower:
                # Calculate confidence based on frequency and context
                frequency = text_lower.count(concept)
                confidence = min(1.0, frequency / 10.0)  # Normalize frequency
                concepts.append((concept, confidence))
        
        # Pattern matching
        for pattern_type, patterns in self.concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    concept = match.group().strip()
                    confidence = 0.8  # High confidence for pattern matches
                    concepts.append((concept, confidence))
        
        # NLP-based concept extraction
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'EVENT', 'WORK_OF_ART', 'LAW']:
                    concepts.append((ent.text.lower(), 0.7))
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short phrases
                    concepts.append((chunk.text.lower(), 0.6))
        
        # Remove duplicates and sort by confidence
        concept_dict = {}
        for concept, confidence in concepts:
            if concept in concept_dict:
                concept_dict[concept] = max(concept_dict[concept], confidence)
            else:
                concept_dict[concept] = confidence
        
        return sorted(concept_dict.items(), key=lambda x: x[1], reverse=True)


class SemanticAnalyzer:
    """Analyzes semantic relationships between concepts"""
    
    def __init__(self):
        """Initialize semantic analyzer"""
        self.embedding_model = None
        self.load_embedding_model()
    
    def load_embedding_model(self):
        """Load sentence embedding model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def calculate_semantic_similarity(self, concepts1: List[str], concepts2: List[str]) -> float:
        """Calculate semantic similarity between two sets of concepts"""
        if not self.embedding_model or not concepts1 or not concepts2:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings1 = self.embedding_model.encode(concepts1)
            embeddings2 = self.embedding_model.encode(concepts2)
            
            # Calculate pairwise similarities
            similarities = []
            for emb1 in embeddings1:
                for emb2 in embeddings2:
                    sim = cosine_similarity([emb1], [emb2])[0][0]
                    similarities.append(sim)
            
            # Return average similarity
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def find_concept_clusters(self, concepts: List[str], n_clusters: int = 5) -> List[ConceptCluster]:
        """Cluster concepts based on semantic similarity"""
        if not self.embedding_model or len(concepts) < n_clusters:
            return []
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(concepts)
            
            # Perform clustering
            if len(concepts) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                centroids = kmeans.cluster_centers_
            else:
                # If too few concepts, use DBSCAN
                dbscan = DBSCAN(eps=0.3, min_samples=2)
                cluster_labels = dbscan.fit_predict(embeddings)
                centroids = []
                for label in set(cluster_labels):
                    if label != -1:  # Not noise
                        cluster_embeddings = embeddings[cluster_labels == label]
                        centroid = np.mean(cluster_embeddings, axis=0)
                        centroids.append(centroid)
            
            # Create cluster objects
            clusters = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                
                cluster_concepts = [concepts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_embeddings = embeddings[cluster_labels == cluster_id]
                
                # Calculate intra-cluster similarity
                if len(cluster_embeddings) > 1:
                    similarities = cosine_similarity(cluster_embeddings)
                    intra_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                else:
                    intra_similarity = 1.0
                
                # Find representative concept (closest to centroid)
                if cluster_id < len(centroids):
                    centroid = centroids[cluster_id]
                    distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
                    representative_idx = np.argmin(distances)
                    representative_concept = cluster_concepts[representative_idx]
                else:
                    representative_concept = cluster_concepts[0]
                
                cluster = ConceptCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    concepts=cluster_concepts,
                    domains=[],  # Will be filled by pattern analyzer
                    centroid=centroids[cluster_id] if cluster_id < len(centroids) else cluster_embeddings[0],
                    similarity_scores=[intra_similarity] * len(cluster_concepts),
                    representative_concept=representative_concept,
                    cluster_size=len(cluster_concepts),
                    intra_cluster_similarity=intra_similarity
                )
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering concepts: {e}")
            return []


class StatisticalAnalyzer:
    """Performs statistical analysis on patterns"""
    
    @staticmethod
    def calculate_pattern_significance(
        domain_concepts: Dict[str, List[str]],
        pattern_concepts: List[str],
        min_domains: int = 2
    ) -> Tuple[float, float]:
        """Calculate statistical significance of a pattern"""
        
        # Count pattern occurrences in each domain
        domain_counts = {}
        total_concepts_per_domain = {}
        
        for domain, concepts in domain_concepts.items():
            domain_counts[domain] = sum(1 for concept in concepts if concept in pattern_concepts)
            total_concepts_per_domain[domain] = len(concepts)
        
        # Domains with pattern occurrences
        domains_with_pattern = [d for d, count in domain_counts.items() if count > 0]
        
        if len(domains_with_pattern) < min_domains:
            return 0.0, 1.0  # Not significant
        
        # Calculate chi-square test
        try:
            observed_counts = list(domain_counts.values())
            total_pattern_concepts = sum(observed_counts)
            total_concepts = sum(total_concepts_per_domain.values())
            
            # Expected counts under null hypothesis (random distribution)
            expected_counts = [
                (total_pattern_concepts * total_concepts_per_domain[domain]) / total_concepts
                for domain in domain_concepts.keys()
            ]
            
            # Chi-square test
            chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
            
            # Effect size (Cramér's V)
            n = total_concepts
            k = len(domain_concepts)
            cramers_v = math.sqrt(chi2_stat / (n * (k - 1)))
            
            return cramers_v, p_value
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_mutual_information(concepts1: List[str], concepts2: List[str]) -> float:
        """Calculate mutual information between two sets of concepts"""
        try:
            # Create binary vectors for concept presence
            all_concepts = list(set(concepts1 + concepts2))
            
            vec1 = [1 if concept in concepts1 else 0 for concept in all_concepts]
            vec2 = [1 if concept in concepts2 else 0 for concept in all_concepts]
            
            # Calculate mutual information
            from sklearn.metrics import mutual_info_score
            return mutual_info_score(vec1, vec2)
            
        except Exception as e:
            logger.error(f"Error calculating mutual information: {e}")
            return 0.0


class PatternDetector:
    """Main pattern detection engine"""
    
    def __init__(self, config_path: str = "agents/pattern_recognition/config.yaml"):
        """Initialize pattern detector"""
        self.load_config(config_path)
        self.concept_extractor = ConceptExtractor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Storage for detected patterns
        self.detected_patterns: List[CrossDomainPattern] = []
        self.pattern_cache: Dict[str, CrossDomainPattern] = {}
        
        # Load existing patterns
        self.load_existing_patterns()
    
    def load_config(self, config_path: str) -> None:
        """Load pattern detection configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'similarity_threshold': 0.8,
                'min_domains': 2,
                'min_examples_per_domain': 2,
                'significance_threshold': 0.05,
                'min_confidence': 0.7,
                'clustering': {
                    'n_clusters': 5,
                    'min_cluster_size': 3,
                    'similarity_threshold': 0.7
                },
                'validation': {
                    'require_user_validation': True,
                    'auto_validate_threshold': 0.95
                },
                'output_dir': 'data/patterns'
            }
    
    def load_existing_patterns(self) -> None:
        """Load previously detected patterns"""
        patterns_file = Path(self.config['output_dir']) / "detected_patterns.json"
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_data in patterns_data:
                    # Convert datetime strings back to datetime objects
                    if 'created_at' in pattern_data:
                        pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                    if 'updated_at' in pattern_data:
                        pattern_data['updated_at'] = datetime.fromisoformat(pattern_data['updated_at'])
                    
                    # Reconstruct concept clusters
                    clusters = []
                    for cluster_data in pattern_data.get('concept_clusters', []):
                        cluster = ConceptCluster(**cluster_data)
                        clusters.append(cluster)
                    pattern_data['concept_clusters'] = clusters
                    
                    pattern = CrossDomainPattern(**pattern_data)
                    self.detected_patterns.append(pattern)
                    self.pattern_cache[pattern.pattern_id] = pattern
                
                logger.info(f"Loaded {len(self.detected_patterns)} existing patterns")
                
            except Exception as e:
                logger.error(f"Error loading existing patterns: {e}")
    
    def save_patterns(self) -> None:
        """Save detected patterns to storage"""
        patterns_file = Path(self.config['output_dir'])
        patterns_file.mkdir(parents=True, exist_ok=True)
        patterns_file = patterns_file / "detected_patterns.json"
        
        try:
            patterns_data = []
            for pattern in self.detected_patterns:
                pattern_dict = asdict(pattern)
                # Convert datetime objects to strings
                pattern_dict['created_at'] = pattern.created_at.isoformat()
                pattern_dict['updated_at'] = pattern.updated_at.isoformat()
                patterns_data.append(pattern_dict)
            
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(patterns_data)} patterns")
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    async def detect_patterns(self, documents: List[Dict[str, Any]]) -> List[CrossDomainPattern]:
        """Detect cross-domain patterns in documents"""
        logger.info(f"Starting pattern detection on {len(documents)} documents")
        
        # Step 1: Extract concepts from each document by domain
        domain_concepts = await self._extract_domain_concepts(documents)
        
        # Step 2: Find concept clusters within and across domains
        concept_clusters = await self._find_concept_clusters(domain_concepts)
        
        # Step 3: Detect cross-domain patterns
        patterns = await self._detect_cross_domain_patterns(domain_concepts, concept_clusters)
        
        # Step 4: Validate and score patterns
        validated_patterns = await self._validate_patterns(patterns, documents)
        
        # Step 5: Update detected patterns
        self.detected_patterns.extend(validated_patterns)
        self.save_patterns()
        
        logger.info(f"Detected {len(validated_patterns)} new patterns")
        return validated_patterns
    
    async def _extract_domain_concepts(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract concepts from documents grouped by domain"""
        domain_concepts = defaultdict(list)
        
        for doc in documents:
            domain = doc.get('domain', 'unknown')
            content = doc.get('cleaned_content', '') or doc.get('content', '')
            
            if content:
                concepts = self.concept_extractor.extract_concepts(content, domain)
                # Only keep concepts with confidence above threshold
                filtered_concepts = [
                    concept for concept, confidence in concepts
                    if confidence >= self.config.get('min_confidence', 0.7)
                ]
                domain_concepts[domain].extend(filtered_concepts)
        
        # Remove duplicates while preserving order
        for domain in domain_concepts:
            seen = set()
            unique_concepts = []
            for concept in domain_concepts[domain]:
                if concept not in seen:
                    seen.add(concept)
                    unique_concepts.append(concept)
            domain_concepts[domain] = unique_concepts
        
        logger.info(f"Extracted concepts from {len(domain_concepts)} domains")
        return dict(domain_concepts)
    
    async def _find_concept_clusters(self, domain_concepts: Dict[str, List[str]]) -> List[ConceptCluster]:
        """Find clusters of semantically similar concepts"""
        all_concepts = []
        concept_to_domains = defaultdict(list)
        
        # Collect all concepts and track their domains
        for domain, concepts in domain_concepts.items():
            all_concepts.extend(concepts)
            for concept in concepts:
                concept_to_domains[concept].append(domain)
        
        # Remove duplicates
        unique_concepts = list(set(all_concepts))
        
        if len(unique_concepts) < self.config['clustering']['min_cluster_size']:
            return []
        
        # Cluster concepts
        clusters = self.semantic_analyzer.find_concept_clusters(
            unique_concepts,
            n_clusters=self.config['clustering']['n_clusters']
        )
        
        # Add domain information to clusters
        for cluster in clusters:
            cluster_domains = set()
            for concept in cluster.concepts:
                cluster_domains.update(concept_to_domains[concept])
            cluster.domains = list(cluster_domains)
        
        logger.info(f"Found {len(clusters)} concept clusters")
        return clusters
    
    async def _detect_cross_domain_patterns(
        self,
        domain_concepts: Dict[str, List[str]],
        concept_clusters: List[ConceptCluster]
    ) -> List[CrossDomainPattern]:
        """Detect patterns that span multiple domains"""
        patterns = []
        
        # Look for clusters that span multiple domains
        cross_domain_clusters = [
            cluster for cluster in concept_clusters
            if len(cluster.domains) >= self.config['min_domains']
        ]
        
        for cluster in cross_domain_clusters:
            # Calculate pattern confidence
            confidence = self._calculate_pattern_confidence(cluster, domain_concepts)
            
            if confidence >= self.config['min_confidence']:
                # Calculate statistical significance
                effect_size, p_value = self.statistical_analyzer.calculate_pattern_significance(
                    domain_concepts,
                    cluster.concepts,
                    self.config['min_domains']
                )
                
                # Determine pattern type
                pattern_type = self._classify_pattern_type(cluster.concepts)
                
                # Generate examples
                examples = self._generate_pattern_examples(cluster, domain_concepts)
                
                # Create pattern
                pattern = CrossDomainPattern(
                    pattern_id=f"pattern_{hashlib.md5(cluster.representative_concept.encode()).hexdigest()[:12]}",
                    name=self._generate_pattern_name(cluster.representative_concept, pattern_type),
                    description=self._generate_pattern_description(cluster, pattern_type),
                    pattern_type=pattern_type,
                    domains=cluster.domains,
                    confidence=confidence,
                    evidence=[],  # Will be filled later
                    examples=examples,
                    concept_clusters=[cluster],
                    statistical_significance=effect_size,
                    validated=False
                )
                
                patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} cross-domain patterns")
        return patterns
    
    def _calculate_pattern_confidence(self, cluster: ConceptCluster, domain_concepts: Dict[str, List[str]]) -> float:
        """Calculate confidence score for a pattern"""
        # Base confidence from cluster quality
        base_confidence = cluster.intra_cluster_similarity
        
        # Boost for number of domains
        domain_boost = min(1.0, len(cluster.domains) / 6.0)  # Max 6 domains
        
        # Boost for concept frequency across domains
        concept_frequencies = []
        for domain in cluster.domains:
            domain_concept_count = len([c for c in cluster.concepts if c in domain_concepts.get(domain, [])])
            total_domain_concepts = len(domain_concepts.get(domain, []))
            if total_domain_concepts > 0:
                frequency = domain_concept_count / total_domain_concepts
                concept_frequencies.append(frequency)
        
        frequency_boost = np.mean(concept_frequencies) if concept_frequencies else 0.0
        
        # Combined confidence
        confidence = (base_confidence * 0.4) + (domain_boost * 0.3) + (frequency_boost * 0.3)
        return min(1.0, confidence)
    
    def _classify_pattern_type(self, concepts: List[str]) -> str:
        """Classify the type of pattern based on concepts"""
        concept_text = ' '.join(concepts).lower()
        
        # Check for specific pattern types
        if any(word in concept_text for word in ['three', 'trinity', 'triad', 'triple']):
            return 'trinity'
        elif any(word in concept_text for word in ['dual', 'two', 'binary', 'dichotomy']):
            return 'duality'
        elif any(word in concept_text for word in ['one', 'unity', 'unified', 'universal']):
            return 'unity'
        elif any(word in concept_text for word in ['transform', 'change', 'evolution', 'becoming']):
            return 'transformation'
        elif any(word in concept_text for word in ['cycle', 'circular', 'eternal', 'recurring']):
            return 'cyclical'
        elif any(word in concept_text for word in ['hierarchy', 'level', 'order', 'structure']):
            return 'hierarchical'
        else:
            return 'semantic'
    
    def _generate_pattern_name(self, representative_concept: str, pattern_type: str) -> str:
        """Generate a descriptive name for the pattern"""
        return f"{pattern_type.title()} Pattern: {representative_concept.title()}"
    
    def _generate_pattern_description(self, cluster: ConceptCluster, pattern_type: str) -> str:
        """Generate a description for the pattern"""
        domains_str = ', '.join(cluster.domains)
        concepts_str = ', '.join(cluster.concepts[:5])  # First 5 concepts
        
        return (f"A {pattern_type} pattern found across {domains_str} domains, "
                f"characterized by concepts including: {concepts_str}. "
                f"This pattern suggests common structural or thematic elements "
                f"that transcend domain boundaries.")
    
    def _generate_pattern_examples(self, cluster: ConceptCluster, domain_concepts: Dict[str, List[str]]) -> List[str]:
        """Generate examples of the pattern in different domains"""
        examples = []
        
        for domain in cluster.domains:
            domain_cluster_concepts = [c for c in cluster.concepts if c in domain_concepts.get(domain, [])]
            if domain_cluster_concepts:
                example = f"{domain.title()}: {', '.join(domain_cluster_concepts[:3])}"
                examples.append(example)
        
        return examples
    
    async def _validate_patterns(self, patterns: List[CrossDomainPattern], documents: List[Dict[str, Any]]) -> List[CrossDomainPattern]:
        """Validate detected patterns"""
        validated_patterns = []
        
        for pattern in patterns:
            # Auto-validate if confidence is very high
            if pattern.confidence >= self.config['validation']['auto_validate_threshold']:
                pattern.validated = True
                pattern.validation_notes = "Auto-validated due to high confidence"
                validated_patterns.append(pattern)
            
            # Check if pattern meets minimum requirements
            elif (pattern.confidence >= self.config['min_confidence'] and
                  len(pattern.domains) >= self.config['min_domains'] and
                  pattern.statistical_significance < self.config['significance_threshold']):
                
                # Mark for user validation if required
                if self.config['validation']['require_user_validation']:
                    pattern.validated = False
                    pattern.validation_notes = "Requires user validation"
                else:
                    pattern.validated = True
                    pattern.validation_notes = "Automatically validated"
                
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[CrossDomainPattern]:
        """Get pattern by ID"""
        return self.pattern_cache.get(pattern_id)
    
    def update_pattern_validation(self, pattern_id: str, validated: bool, notes: Optional[str] = None) -> bool:
        """Update pattern validation status"""
        pattern = self.get_pattern_by_id(pattern_id)
        if not pattern:
            return False
        
        pattern.validated = validated
        if notes:
            pattern.validation_notes = notes
        pattern.updated_at = datetime.now()
        
        self.save_patterns()
        return True
    
    def search_patterns(
        self,
        query: Optional[str] = None,
        domains: Optional[List[str]] = None,
        pattern_type: Optional[str] = None,
        validated_only: bool = False,
        min_confidence: Optional[float] = None
    ) -> List[CrossDomainPattern]:
        """Search for patterns with filters"""
        results = self.detected_patterns.copy()
        
        if validated_only:
            results = [p for p in results if p.validated]
        
        if min_confidence:
            results = [p for p in results if p.confidence >= min_confidence]
        
        if pattern_type:
            results = [p for p in results if p.pattern_type == pattern_type]
        
        if domains:
            results = [p for p in results if any(d in p.domains for d in domains)]
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if (query_lower in p.name.lower() or
                    query_lower in p.description.lower() or
                    any(query_lower in concept for concept in sum([cluster.concepts for cluster in p.concept_clusters], [])))
            ]
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results


async def main():
    """Example usage of pattern recognition agent"""
    # Example documents
    documents = [
        {
            'doc_id': 'doc1',
            'domain': 'religion',
            'cleaned_content': 'The Trinity consists of the Father, Son, and Holy Spirit, three persons in one divine essence.',
            'title': 'Christian Doctrine'
        },
        {
            'doc_id': 'doc2', 
            'domain': 'science',
            'cleaned_content': 'Quantum mechanics reveals the three fundamental forces: electromagnetic, weak, and strong nuclear forces.',
            'title': 'Physics Fundamentals'
        },
        {
            'doc_id': 'doc3',
            'domain': 'philosophy',
            'cleaned_content': 'Hegel\'s dialectic involves thesis, antithesis, and synthesis as three stages of development.',
            'title': 'Hegelian Philosophy'
        }
    ]
    
    detector = PatternDetector()
    
    try:
        # Detect patterns
        patterns = await detector.detect_patterns(documents)
        
        print(f"Detected {len(patterns)} patterns:")
        for pattern in patterns:
            print(f"- {pattern.name}")
            print(f"  Type: {pattern.pattern_type}")
            print(f"  Domains: {', '.join(pattern.domains)}")
            print(f"  Confidence: {pattern.confidence:.3f}")
            print(f"  Examples: {'; '.join(pattern.examples)}")
            print(f"  Validated: {pattern.validated}")
            print()
        
        # Search patterns
        trinity_patterns = detector.search_patterns(query="trinity", pattern_type="trinity")
        print(f"Found {len(trinity_patterns)} trinity patterns")
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
