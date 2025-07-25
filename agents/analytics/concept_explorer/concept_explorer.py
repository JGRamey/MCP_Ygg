#!/usr/bin/env python3
"""
Concept Explorer Agent
Advanced relationship discovery and hypothesis generation for MCP Yggdrasil
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncio
import networkx as nx
import numpy as np

# NLP and ML imports
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """Individual concept representation"""

    id: str
    name: str
    domain: str
    description: str
    confidence: float
    source_documents: List[str]
    extraction_method: str


@dataclass
class RelationshipEdge:
    """Relationship between concepts"""

    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    evidence: List[str]
    confidence: float
    bidirectional: bool


@dataclass
class ConceptHypothesis:
    """Generated hypothesis about concept relationships"""

    hypothesis_id: str
    description: str
    supporting_concepts: List[str]
    evidence_strength: float
    novelty_score: float
    testable_predictions: List[str]
    domain_bridge: Optional[Tuple[str, str]]


@dataclass
class ThoughtPath:
    """Traced path of conceptual connections"""

    path_id: str
    start_concept: str
    end_concept: str
    intermediate_steps: List[Dict[str, Any]]
    total_strength: float
    reasoning_chain: List[str]


class ConceptExplorer:
    """
    Advanced concept relationship discovery and hypothesis generation agent
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
    ):
        """Initialize the concept explorer"""

        # Load configuration
        self.config = self._load_config(config_path)

        # Override with provided parameters
        if model_name != "all-MiniLM-L6-v2":
            self.config["models"]["sentence_transformer"] = model_name
        if spacy_model != "en_core_web_sm":
            self.config["models"]["spacy_model"] = spacy_model

        # Load NLP models
        try:
            self.nlp = spacy.load(self.config["models"]["spacy_model"])
            self.sentence_model = SentenceTransformer(
                self.config["models"]["sentence_transformer"]
            )
            logger.info(
                f"Loaded models: {self.config['models']['spacy_model']}, {self.config['models']['sentence_transformer']}"
            )
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

        # Initialize concept graph
        self.concept_graph = nx.Graph()
        self.concept_nodes: Dict[str, ConceptNode] = {}
        self.relationship_edges: Dict[str, RelationshipEdge] = {}

        # Domain mappings from config
        self.domain_keywords = {}
        for domain, domain_config in self.config.get("domains", {}).items():
            self.domain_keywords[domain] = domain_config.get("keywords", [])

        # Relationship types and patterns from config
        self.relationship_patterns = self.config.get("relationships", {}).get(
            "patterns", {}
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml

            if config_path is None:
                # Use default config path
                config_path = Path(__file__).parent / "config.yaml"

            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(
                    f"Config file not found at {config_path}, using defaults"
                )
                return self._get_default_config()

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available"""
        return {
            "models": {
                "spacy_model": "en_core_web_sm",
                "sentence_transformer": "all-MiniLM-L6-v2",
            },
            "domains": {
                "mathematics": {
                    "keywords": [
                        "number",
                        "equation",
                        "theorem",
                        "proof",
                        "geometry",
                        "algebra",
                    ]
                },
                "science": {
                    "keywords": [
                        "physics",
                        "chemistry",
                        "biology",
                        "experiment",
                        "theory",
                        "discovery",
                    ]
                },
                "philosophy": {
                    "keywords": [
                        "ethics",
                        "metaphysics",
                        "logic",
                        "consciousness",
                        "reality",
                        "truth",
                    ]
                },
                "art": {
                    "keywords": [
                        "painting",
                        "sculpture",
                        "music",
                        "literature",
                        "aesthetics",
                        "beauty",
                    ]
                },
                "religion": {
                    "keywords": [
                        "sacred",
                        "divine",
                        "prayer",
                        "ritual",
                        "scripture",
                        "faith",
                    ]
                },
                "technology": {
                    "keywords": [
                        "invention",
                        "engineering",
                        "mechanism",
                        "tool",
                        "innovation",
                    ]
                },
            },
            "relationships": {
                "patterns": {
                    "causal": [
                        "causes",
                        "leads to",
                        "results in",
                        "produces",
                        "generates",
                    ],
                    "temporal": ["before", "after", "during", "precedes", "follows"],
                    "similarity": [
                        "similar to",
                        "like",
                        "resembles",
                        "analogous",
                        "comparable",
                    ],
                    "opposition": [
                        "opposes",
                        "contradicts",
                        "differs from",
                        "opposite",
                    ],
                    "containment": [
                        "includes",
                        "contains",
                        "part of",
                        "subset",
                        "element",
                    ],
                    "influence": [
                        "influences",
                        "affects",
                        "shapes",
                        "impacts",
                        "modifies",
                    ],
                }
            },
        }

    async def extract_concepts(
        self, text: str, source_document: str, domain: Optional[str] = None
    ) -> List[ConceptNode]:
        """
        Extract concepts from text using NLP analysis
        """
        try:
            concepts = []

            # Process text with spaCy
            doc = self.nlp(text)

            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in [
                    "PERSON",
                    "ORG",
                    "EVENT",
                    "WORK_OF_ART",
                    "LAW",
                    "LANGUAGE",
                ]:
                    concept = ConceptNode(
                        id=f"ent_{len(concepts)}_{ent.text.replace(' ', '_').lower()}",
                        name=ent.text,
                        domain=domain or self._classify_domain(ent.text),
                        description=f"{ent.label_}: {ent.text}",
                        confidence=0.8,
                        source_documents=[source_document],
                        extraction_method="named_entity_recognition",
                    )
                    concepts.append(concept)

            # Extract noun phrases as potential concepts
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3 and len(chunk.text) > 3:
                    # Filter out common words
                    if not self._is_common_phrase(chunk.text):
                        concept = ConceptNode(
                            id=f"np_{len(concepts)}_{chunk.text.replace(' ', '_').lower()}",
                            name=chunk.text,
                            domain=domain or self._classify_domain(chunk.text),
                            description=f"Noun phrase: {chunk.text}",
                            confidence=0.6,
                            source_documents=[source_document],
                            extraction_method="noun_phrase_extraction",
                        )
                        concepts.append(concept)

            # Extract domain-specific concepts
            domain_concepts = await self._extract_domain_concepts(
                text, source_document, domain
            )
            concepts.extend(domain_concepts)

            logger.info(f"Extracted {len(concepts)} concepts from {source_document}")
            return concepts

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    async def discover_relationships(
        self, concepts: List[ConceptNode], text: str
    ) -> List[RelationshipEdge]:
        """
        Discover relationships between concepts using multiple methods
        """
        try:
            relationships = []

            # Method 1: Semantic similarity
            semantic_rels = await self._find_semantic_relationships(concepts)
            relationships.extend(semantic_rels)

            # Method 2: Co-occurrence analysis
            cooccurrence_rels = await self._find_cooccurrence_relationships(
                concepts, text
            )
            relationships.extend(cooccurrence_rels)

            # Method 3: Pattern-based extraction
            pattern_rels = await self._find_pattern_relationships(concepts, text)
            relationships.extend(pattern_rels)

            # Method 4: Cross-domain bridging
            bridge_rels = await self._find_cross_domain_relationships(concepts)
            relationships.extend(bridge_rels)

            logger.info(f"Discovered {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Error discovering relationships: {e}")
            return []

    async def generate_hypotheses(
        self, concepts: List[ConceptNode], relationships: List[RelationshipEdge]
    ) -> List[ConceptHypothesis]:
        """
        Generate novel hypotheses about concept relationships
        """
        try:
            hypotheses = []

            # Build temporary graph for analysis
            temp_graph = nx.Graph()
            for concept in concepts:
                temp_graph.add_node(concept.id, **asdict(concept))

            for rel in relationships:
                temp_graph.add_edge(rel.source_id, rel.target_id, **asdict(rel))

            # Generate hypotheses using different strategies

            # Strategy 1: Transitive relationships
            transitive_hyps = await self._generate_transitive_hypotheses(temp_graph)
            hypotheses.extend(transitive_hyps)

            # Strategy 2: Missing links
            missing_link_hyps = await self._generate_missing_link_hypotheses(temp_graph)
            hypotheses.extend(missing_link_hyps)

            # Strategy 3: Cross-domain bridges
            bridge_hyps = await self._generate_bridge_hypotheses(temp_graph)
            hypotheses.extend(bridge_hyps)

            # Strategy 4: Anomaly detection
            anomaly_hyps = await self._generate_anomaly_hypotheses(temp_graph)
            hypotheses.extend(anomaly_hyps)

            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return []

    async def trace_thought_paths(
        self, start_concept: str, end_concept: str, max_depth: int = 5
    ) -> List[ThoughtPath]:
        """
        Trace paths of reasoning between concepts
        """
        try:
            paths = []

            # Use graph algorithms to find paths
            try:
                all_paths = list(
                    nx.all_simple_paths(
                        self.concept_graph, start_concept, end_concept, cutoff=max_depth
                    )
                )

                for i, path in enumerate(all_paths[:10]):  # Limit to top 10 paths
                    # Calculate path strength
                    total_strength = 0.0
                    reasoning_chain = []
                    intermediate_steps = []

                    for j in range(len(path) - 1):
                        current_node = path[j]
                        next_node = path[j + 1]

                        # Get edge data
                        edge_data = self.concept_graph.get_edge_data(
                            current_node, next_node
                        )
                        if edge_data:
                            total_strength += edge_data.get("strength", 0.5)

                            step = {
                                "from": current_node,
                                "to": next_node,
                                "relationship": edge_data.get(
                                    "relationship_type", "unknown"
                                ),
                                "strength": edge_data.get("strength", 0.5),
                                "evidence": edge_data.get("evidence", []),
                            }
                            intermediate_steps.append(step)

                            # Build reasoning chain
                            from_concept = self.concept_nodes.get(current_node, {}).get(
                                "name", current_node
                            )
                            to_concept = self.concept_nodes.get(next_node, {}).get(
                                "name", next_node
                            )
                            relationship = edge_data.get(
                                "relationship_type", "relates to"
                            )

                            reasoning_chain.append(
                                f"{from_concept} {relationship} {to_concept}"
                            )

                    thought_path = ThoughtPath(
                        path_id=f"path_{i}_{start_concept}_{end_concept}",
                        start_concept=start_concept,
                        end_concept=end_concept,
                        intermediate_steps=intermediate_steps,
                        total_strength=total_strength / max(len(path) - 1, 1),
                        reasoning_chain=reasoning_chain,
                    )

                    paths.append(thought_path)

            except nx.NetworkXNoPath:
                logger.info(f"No path found between {start_concept} and {end_concept}")

            logger.info(f"Traced {len(paths)} thought paths")
            return paths

        except Exception as e:
            logger.error(f"Error tracing thought paths: {e}")
            return []

    async def analyze_concept_network(self) -> Dict[str, Any]:
        """
        Analyze the overall concept network structure
        """
        try:
            analysis = {}

            if len(self.concept_graph.nodes) == 0:
                return {"error": "No concepts in graph"}

            # Basic graph metrics
            analysis["node_count"] = len(self.concept_graph.nodes)
            analysis["edge_count"] = len(self.concept_graph.edges)
            analysis["density"] = nx.density(self.concept_graph)

            # Connectivity metrics
            if nx.is_connected(self.concept_graph):
                analysis["diameter"] = nx.diameter(self.concept_graph)
                analysis["average_shortest_path"] = nx.average_shortest_path_length(
                    self.concept_graph
                )
            else:
                analysis["connected_components"] = nx.number_connected_components(
                    self.concept_graph
                )

            # Centrality measures
            degree_centrality = nx.degree_centrality(self.concept_graph)
            betweenness_centrality = nx.betweenness_centrality(self.concept_graph)
            closeness_centrality = nx.closeness_centrality(self.concept_graph)

            # Find most central concepts
            analysis["most_connected"] = sorted(
                degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            analysis["most_influential"] = sorted(
                betweenness_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Domain distribution
            domain_counts = {}
            for node_id, node_data in self.concept_graph.nodes(data=True):
                domain = node_data.get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            analysis["domain_distribution"] = domain_counts

            # Clustering
            analysis["clustering_coefficient"] = nx.average_clustering(
                self.concept_graph
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing concept network: {e}")
            return {"error": str(e)}

    async def _extract_domain_concepts(
        self, text: str, source_document: str, domain: Optional[str]
    ) -> List[ConceptNode]:
        """Extract domain-specific concepts"""
        concepts = []

        # Use domain keywords to identify relevant concepts
        if domain and domain in self.domain_keywords:
            keywords = self.domain_keywords[domain]
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    concept = ConceptNode(
                        id=f"domain_{len(concepts)}_{keyword.replace(' ', '_').lower()}",
                        name=keyword,
                        domain=domain,
                        description=f"Domain keyword: {keyword}",
                        confidence=0.9,
                        source_documents=[source_document],
                        extraction_method="domain_keyword_matching",
                    )
                    concepts.append(concept)

        return concepts

    async def _find_semantic_relationships(
        self, concepts: List[ConceptNode]
    ) -> List[RelationshipEdge]:
        """Find relationships based on semantic similarity"""
        relationships = []

        if len(concepts) < 2:
            return relationships

        # Generate embeddings for concept names
        concept_texts = [concept.name for concept in concepts]
        embeddings = self.sentence_model.encode(concept_texts)

        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = similarities[i][j]

                if similarity > 0.7:  # High similarity threshold
                    relationship = RelationshipEdge(
                        source_id=concepts[i].id,
                        target_id=concepts[j].id,
                        relationship_type="semantic_similarity",
                        strength=float(similarity),
                        evidence=[f"Semantic similarity: {similarity:.3f}"],
                        confidence=float(similarity),
                        bidirectional=True,
                    )
                    relationships.append(relationship)

        return relationships

    async def _find_cooccurrence_relationships(
        self, concepts: List[ConceptNode], text: str
    ) -> List[RelationshipEdge]:
        """Find relationships based on co-occurrence in text"""
        relationships = []

        # Create sliding window to check co-occurrence
        window_size = 50  # words
        words = text.split()

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept1 = concepts[i]
                concept2 = concepts[j]

                # Check if concepts co-occur within window
                cooccurrence_count = 0
                for k in range(len(words) - window_size):
                    window = " ".join(words[k : k + window_size])
                    if (
                        concept1.name.lower() in window.lower()
                        and concept2.name.lower() in window.lower()
                    ):
                        cooccurrence_count += 1

                if cooccurrence_count > 0:
                    strength = min(cooccurrence_count / 10.0, 1.0)  # Normalize

                    relationship = RelationshipEdge(
                        source_id=concept1.id,
                        target_id=concept2.id,
                        relationship_type="cooccurrence",
                        strength=strength,
                        evidence=[f"Co-occurred {cooccurrence_count} times"],
                        confidence=strength,
                        bidirectional=True,
                    )
                    relationships.append(relationship)

        return relationships

    async def _find_pattern_relationships(
        self, concepts: List[ConceptNode], text: str
    ) -> List[RelationshipEdge]:
        """Find relationships using linguistic patterns"""
        relationships = []

        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                # Use regex to find pattern occurrences
                pattern_regex = (
                    rf"(\w+(?:\s+\w+)*)\s+{re.escape(pattern)}\s+(\w+(?:\s+\w+)*)"
                )
                matches = re.finditer(pattern_regex, text, re.IGNORECASE)

                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(2).strip()

                    # Find matching concepts
                    source_concept = None
                    target_concept = None

                    for concept in concepts:
                        if concept.name.lower() in source_text.lower():
                            source_concept = concept
                        if concept.name.lower() in target_text.lower():
                            target_concept = concept

                    if (
                        source_concept
                        and target_concept
                        and source_concept != target_concept
                    ):
                        relationship = RelationshipEdge(
                            source_id=source_concept.id,
                            target_id=target_concept.id,
                            relationship_type=rel_type,
                            strength=0.8,
                            evidence=[f"Pattern: {match.group(0)}"],
                            confidence=0.7,
                            bidirectional=rel_type in ["similarity", "cooccurrence"],
                        )
                        relationships.append(relationship)

        return relationships

    async def _find_cross_domain_relationships(
        self, concepts: List[ConceptNode]
    ) -> List[RelationshipEdge]:
        """Find relationships that bridge different domains"""
        relationships = []

        # Group concepts by domain
        domain_groups = {}
        for concept in concepts:
            domain = concept.domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(concept)

        # Look for cross-domain connections
        domains = list(domain_groups.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]

                for concept1 in domain_groups[domain1]:
                    for concept2 in domain_groups[domain2]:
                        # Check for potential cross-domain relationship
                        # This is a simplified heuristic - in practice, you'd use more sophisticated methods

                        if self._has_cross_domain_potential(concept1, concept2):
                            relationship = RelationshipEdge(
                                source_id=concept1.id,
                                target_id=concept2.id,
                                relationship_type="cross_domain_bridge",
                                strength=0.6,
                                evidence=[
                                    f"Cross-domain connection: {domain1} ↔ {domain2}"
                                ],
                                confidence=0.5,
                                bidirectional=True,
                            )
                            relationships.append(relationship)

        return relationships

    def _classify_domain(self, text: str) -> str:
        """Classify text into a domain"""
        text_lower = text.lower()

        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain

        return "general"

    def _is_common_phrase(self, text: str) -> bool:
        """Check if phrase is too common to be a meaningful concept"""
        common_phrases = {
            "the time",
            "this is",
            "that was",
            "it is",
            "they are",
            "we have",
            "you can",
            "will be",
            "has been",
            "would be",
        }
        return text.lower() in common_phrases or len(text.split()) > 4

    def _has_cross_domain_potential(
        self, concept1: ConceptNode, concept2: ConceptNode
    ) -> bool:
        """Check if two concepts from different domains might be related"""
        # Simple heuristic based on name similarity and common themes
        name1_words = set(concept1.name.lower().split())
        name2_words = set(concept2.name.lower().split())

        # Check for shared words
        shared_words = name1_words.intersection(name2_words)
        if len(shared_words) > 0:
            return True

        # Check for thematic connections (simplified)
        cross_domain_themes = {
            "number",
            "pattern",
            "structure",
            "system",
            "order",
            "harmony",
            "balance",
            "unity",
            "infinity",
            "creation",
        }

        for theme in cross_domain_themes:
            if theme in concept1.name.lower() and theme in concept2.name.lower():
                return True

        return False

    async def _generate_transitive_hypotheses(
        self, graph: nx.Graph
    ) -> List[ConceptHypothesis]:
        """Generate hypotheses based on transitive relationships"""
        hypotheses = []

        # Find triangles in the graph where one edge might be missing
        nodes = list(graph.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    node_a, node_b, node_c = nodes[i], nodes[j], nodes[k]

                    # Check if we have two edges but not the third
                    edges = [
                        graph.has_edge(node_a, node_b),
                        graph.has_edge(node_b, node_c),
                        graph.has_edge(node_a, node_c),
                    ]

                    # If exactly two edges exist, hypothesize the third
                    if sum(edges) == 2:
                        missing_edge = None
                        if not edges[0]:
                            missing_edge = (node_a, node_b)
                        elif not edges[1]:
                            missing_edge = (node_b, node_c)
                        elif not edges[2]:
                            missing_edge = (node_a, node_c)

                        if missing_edge:
                            hypothesis = ConceptHypothesis(
                                hypothesis_id=f"transitive_{len(hypotheses)}",
                                description=f"Transitive relationship between {missing_edge[0]} and {missing_edge[1]}",
                                supporting_concepts=[node_a, node_b, node_c],
                                evidence_strength=0.6,
                                novelty_score=0.7,
                                testable_predictions=[
                                    f"Find evidence of direct connection between {missing_edge[0]} and {missing_edge[1]}"
                                ],
                                domain_bridge=None,
                            )
                            hypotheses.append(hypothesis)

        return hypotheses[:10]  # Limit results

    async def _generate_missing_link_hypotheses(
        self, graph: nx.Graph
    ) -> List[ConceptHypothesis]:
        """Generate hypotheses about missing links in the concept network"""
        hypotheses = []

        try:
            # Use structural similarity to predict missing links
            nodes = list(graph.nodes())

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_a, node_b = nodes[i], nodes[j]

                    # Skip if edge already exists
                    if graph.has_edge(node_a, node_b):
                        continue

                    # Calculate structural similarity
                    neighbors_a = set(graph.neighbors(node_a))
                    neighbors_b = set(graph.neighbors(node_b))

                    if len(neighbors_a) > 0 and len(neighbors_b) > 0:
                        # Jaccard similarity of neighborhoods
                        intersection = neighbors_a.intersection(neighbors_b)
                        union = neighbors_a.union(neighbors_b)
                        jaccard_sim = len(intersection) / len(union) if union else 0

                        # Only generate hypothesis if significant structural similarity
                        if jaccard_sim > 0.3:
                            # Get node data for better description
                            node_a_data = graph.nodes[node_a]
                            node_b_data = graph.nodes[node_b]

                            hypothesis = ConceptHypothesis(
                                hypothesis_id=f"missing_link_{len(hypotheses)}_{node_a}_{node_b}",
                                description=f"Predicted missing link between '{node_a_data.get('name', node_a)}' and '{node_b_data.get('name', node_b)}' based on structural similarity",
                                supporting_concepts=list(intersection),
                                evidence_strength=float(jaccard_sim),
                                novelty_score=0.8,
                                testable_predictions=[
                                    f"Search for co-occurrence patterns between {node_a_data.get('name', node_a)} and {node_b_data.get('name', node_b)}",
                                    f"Analyze semantic similarity in source documents",
                                    f"Check for indirect references or implications",
                                ],
                                domain_bridge=(
                                    (
                                        node_a_data.get("domain"),
                                        node_b_data.get("domain"),
                                    )
                                    if node_a_data.get("domain")
                                    != node_b_data.get("domain")
                                    else None
                                ),
                            )
                            hypotheses.append(hypothesis)

            # Sort by evidence strength and limit results
            hypotheses.sort(key=lambda h: h.evidence_strength, reverse=True)
            return hypotheses[:15]

        except Exception as e:
            logger.error(f"Error generating missing link hypotheses: {e}")
            return []

    async def _generate_bridge_hypotheses(
        self, graph: nx.Graph
    ) -> List[ConceptHypothesis]:
        """Generate hypotheses about cross-domain bridges"""
        hypotheses = []

        try:
            # Group nodes by domain
            domain_nodes = {}
            for node_id, node_data in graph.nodes(data=True):
                domain = node_data.get("domain", "unknown")
                if domain not in domain_nodes:
                    domain_nodes[domain] = []
                domain_nodes[domain].append((node_id, node_data))

            # Define domain bridge potential based on conceptual proximity
            bridge_patterns = {
                ("mathematics", "science"): [
                    "formula",
                    "calculation",
                    "measurement",
                    "ratio",
                    "proportion",
                ],
                ("philosophy", "science"): [
                    "truth",
                    "reality",
                    "existence",
                    "consciousness",
                    "nature",
                ],
                ("art", "mathematics"): [
                    "harmony",
                    "proportion",
                    "symmetry",
                    "pattern",
                    "golden",
                ],
                ("religion", "philosophy"): [
                    "truth",
                    "meaning",
                    "existence",
                    "eternal",
                    "divine",
                ],
                ("technology", "science"): [
                    "invention",
                    "discovery",
                    "innovation",
                    "application",
                    "method",
                ],
                ("art", "philosophy"): [
                    "beauty",
                    "aesthetic",
                    "meaning",
                    "expression",
                    "truth",
                ],
            }

            # Generate cross-domain bridge hypotheses
            domains = list(domain_nodes.keys())
            for i, domain1 in enumerate(domains):
                for j, domain2 in enumerate(domains[i + 1 :], i + 1):
                    if (
                        len(domain_nodes[domain1]) == 0
                        or len(domain_nodes[domain2]) == 0
                    ):
                        continue

                    # Check for potential bridge patterns
                    bridge_key = tuple(sorted([domain1, domain2]))
                    bridge_keywords = bridge_patterns.get(bridge_key, [])

                    # Look for concepts that might bridge these domains
                    for node1_id, node1_data in domain_nodes[domain1]:
                        for node2_id, node2_data in domain_nodes[domain2]:

                            # Check for conceptual bridges
                            potential_bridge = False
                            bridge_evidence = []

                            # Method 1: Shared keywords
                            name1_words = set(
                                node1_data.get("name", "").lower().split()
                            )
                            name2_words = set(
                                node2_data.get("name", "").lower().split()
                            )
                            shared_words = name1_words.intersection(name2_words)

                            if shared_words:
                                potential_bridge = True
                                bridge_evidence.append(
                                    f"Shared concepts: {', '.join(shared_words)}"
                                )

                            # Method 2: Bridge keywords
                            for keyword in bridge_keywords:
                                if (
                                    keyword in node1_data.get("name", "").lower()
                                    or keyword in node2_data.get("name", "").lower()
                                ):
                                    potential_bridge = True
                                    bridge_evidence.append(f"Bridge keyword: {keyword}")

                            # Method 3: Structural analysis - check if they're connected through intermediary nodes
                            try:
                                if nx.has_path(graph, node1_id, node2_id):
                                    path_length = nx.shortest_path_length(
                                        graph, node1_id, node2_id
                                    )
                                    if (
                                        2 <= path_length <= 4
                                    ):  # Indirect but not too distant
                                        potential_bridge = True
                                        bridge_evidence.append(
                                            f"Connected through {path_length-1} intermediary concept(s)"
                                        )
                            except nx.NetworkXNoPath:
                                pass

                            if potential_bridge and len(bridge_evidence) > 0:
                                # Calculate bridge strength based on evidence
                                evidence_strength = min(len(bridge_evidence) * 0.3, 1.0)

                                hypothesis = ConceptHypothesis(
                                    hypothesis_id=f"bridge_{len(hypotheses)}_{domain1}_{domain2}",
                                    description=f"Cross-domain bridge between '{node1_data.get('name', node1_id)}' ({domain1}) and '{node2_data.get('name', node2_id)}' ({domain2})",
                                    supporting_concepts=[node1_id, node2_id],
                                    evidence_strength=evidence_strength,
                                    novelty_score=0.9,  # Cross-domain bridges are highly novel
                                    testable_predictions=[
                                        f"Investigate historical connections between {domain1} and {domain2}",
                                        f"Search for interdisciplinary research connecting these concepts",
                                        f"Analyze influence patterns across domains",
                                        f"Look for common foundational principles",
                                    ],
                                    domain_bridge=(domain1, domain2),
                                )
                                hypotheses.append(hypothesis)

            # Sort by novelty score and evidence strength
            hypotheses.sort(
                key=lambda h: (h.novelty_score * h.evidence_strength), reverse=True
            )
            return hypotheses[:12]

        except Exception as e:
            logger.error(f"Error generating bridge hypotheses: {e}")
            return []

    async def _generate_anomaly_hypotheses(
        self, graph: nx.Graph
    ) -> List[ConceptHypothesis]:
        """Generate hypotheses based on network anomalies"""
        hypotheses = []

        try:
            if len(graph.nodes) < 3:
                return hypotheses

            # Calculate network metrics to identify anomalies
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)

            # Identify nodes with anomalous centrality patterns
            nodes_data = []
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                nodes_data.append(
                    {
                        "id": node_id,
                        "name": node_data.get("name", node_id),
                        "domain": node_data.get("domain", "unknown"),
                        "degree": degree_centrality[node_id],
                        "betweenness": betweenness_centrality[node_id],
                        "closeness": closeness_centrality[node_id],
                    }
                )

            # Calculate mean and std for each centrality measure
            degrees = [n["degree"] for n in nodes_data]
            betweennesses = [n["betweenness"] for n in nodes_data]
            closenesses = [n["closeness"] for n in nodes_data]

            degree_mean, degree_std = np.mean(degrees), np.std(degrees)
            betweenness_mean, betweenness_std = np.mean(betweennesses), np.std(
                betweennesses
            )
            closeness_mean, closeness_std = np.mean(closenesses), np.std(closenesses)

            # Identify anomalies (nodes that are > 2 standard deviations from mean)
            anomalous_nodes = []

            for node in nodes_data:
                anomaly_types = []

                # High degree centrality anomaly (super-connectors)
                if node["degree"] > degree_mean + 2 * degree_std:
                    anomaly_types.append("super_connector")

                # High betweenness centrality (bridges between clusters)
                if node["betweenness"] > betweenness_mean + 2 * betweenness_std:
                    anomaly_types.append("cluster_bridge")

                # Low degree but high closeness (peripheral but important)
                if (
                    node["degree"] < degree_mean - degree_std
                    and node["closeness"] > closeness_mean + degree_std
                ):
                    anomaly_types.append("peripheral_important")

                # Isolated high-importance nodes
                if (
                    node["degree"] < degree_mean - degree_std
                    and node["betweenness"] > betweenness_mean
                ):
                    anomaly_types.append("isolated_bridge")

                if anomaly_types:
                    anomalous_nodes.append((node, anomaly_types))

            # Generate hypotheses for each anomaly
            for node, anomaly_types in anomalous_nodes:
                for anomaly_type in anomaly_types:
                    if anomaly_type == "super_connector":
                        hypothesis = ConceptHypothesis(
                            hypothesis_id=f"anomaly_{len(hypotheses)}_super_connector_{node['id']}",
                            description=f"'{node['name']}' appears to be a super-connector concept with unusually high connectivity (degree: {node['degree']:.3f})",
                            supporting_concepts=[node["id"]]
                            + list(graph.neighbors(node["id"]))[:5],
                            evidence_strength=min(
                                (node["degree"] - degree_mean) / (degree_std + 0.001),
                                1.0,
                            ),
                            novelty_score=0.8,
                            testable_predictions=[
                                f"Verify if '{node['name']}' is a fundamental concept in {node['domain']}",
                                "Check if this concept appears in foundational texts across multiple sources",
                                "Investigate if this concept serves as a definitional anchor for other concepts",
                            ],
                            domain_bridge=None,
                        )
                        hypotheses.append(hypothesis)

                    elif anomaly_type == "cluster_bridge":
                        hypothesis = ConceptHypothesis(
                            hypothesis_id=f"anomaly_{len(hypotheses)}_bridge_{node['id']}",
                            description=f"'{node['name']}' appears to bridge different conceptual clusters (betweenness: {node['betweenness']:.3f})",
                            supporting_concepts=[node["id"]],
                            evidence_strength=min(
                                (node["betweenness"] - betweenness_mean)
                                / (betweenness_std + 0.001),
                                1.0,
                            ),
                            novelty_score=0.9,
                            testable_predictions=[
                                f"Investigate if '{node['name']}' connects different schools of thought",
                                "Look for historical periods where this concept enabled knowledge transfer",
                                "Check if removing this concept would fragment the knowledge network",
                            ],
                            domain_bridge=None,
                        )
                        hypotheses.append(hypothesis)

                    elif anomaly_type == "peripheral_important":
                        hypothesis = ConceptHypothesis(
                            hypothesis_id=f"anomaly_{len(hypotheses)}_peripheral_{node['id']}",
                            description=f"'{node['name']}' has low connectivity but high importance (degree: {node['degree']:.3f}, closeness: {node['closeness']:.3f})",
                            supporting_concepts=[node["id"]],
                            evidence_strength=0.7,
                            novelty_score=0.85,
                            testable_predictions=[
                                f"Research if '{node['name']}' represents an emerging or declining concept",
                                "Check if this concept has specialized but crucial applications",
                                "Investigate temporal patterns of usage and influence",
                            ],
                            domain_bridge=None,
                        )
                        hypotheses.append(hypothesis)

            # Detect structural anomalies
            # Look for unusually dense or sparse regions
            if nx.is_connected(graph):
                try:
                    communities = nx.community.greedy_modularity_communities(graph)
                    if len(communities) > 1:
                        # Analyze community structure anomalies
                        community_sizes = [len(comm) for comm in communities]
                        mean_size = np.mean(community_sizes)

                        for i, community in enumerate(communities):
                            if (
                                len(community) > mean_size * 2
                            ):  # Unusually large community
                                community_nodes = list(community)[:3]  # Sample of nodes
                                hypothesis = ConceptHypothesis(
                                    hypothesis_id=f"anomaly_{len(hypotheses)}_large_cluster_{i}",
                                    description=f"Detected unusually large conceptual cluster with {len(community)} concepts",
                                    supporting_concepts=community_nodes,
                                    evidence_strength=0.6,
                                    novelty_score=0.7,
                                    testable_predictions=[
                                        "Investigate if this cluster represents a well-established domain",
                                        "Check for over-representation of certain concept types",
                                        "Analyze if this clustering reflects historical knowledge organization",
                                    ],
                                    domain_bridge=None,
                                )
                                hypotheses.append(hypothesis)

                except Exception as community_error:
                    logger.warning(f"Community detection failed: {community_error}")

            # Sort hypotheses by combined novelty and evidence strength
            hypotheses.sort(
                key=lambda h: h.novelty_score * h.evidence_strength, reverse=True
            )
            return hypotheses[:10]

        except Exception as e:
            logger.error(f"Error generating anomaly hypotheses: {e}")
            return []

    async def analyze_temporal_evolution(
        self, concepts: List[ConceptNode]
    ) -> Dict[str, Any]:
        """
        Analyze how concepts evolve and influence each other over time
        """
        try:
            evolution_analysis = {
                "temporal_clusters": {},
                "influence_chains": [],
                "evolutionary_patterns": {},
                "anachronisms": [],
            }

            # Group concepts by potential time periods (if available in source documents)
            temporal_groups = {}

            for concept in concepts:
                # Extract temporal indicators from concept names and descriptions
                temporal_indicators = self._extract_temporal_indicators(concept)

                for period in temporal_indicators:
                    if period not in temporal_groups:
                        temporal_groups[period] = []
                    temporal_groups[period].append(concept)

            evolution_analysis["temporal_clusters"] = {
                period: [c.name for c in concepts]
                for period, concepts in temporal_groups.items()
            }

            # Analyze potential influence chains across time periods
            time_periods = sorted(temporal_groups.keys())
            for i in range(len(time_periods) - 1):
                earlier_period = time_periods[i]
                later_period = time_periods[i + 1]

                # Find concepts that might show influence relationships
                earlier_concepts = temporal_groups[earlier_period]
                later_concepts = temporal_groups[later_period]

                for earlier_concept in earlier_concepts:
                    for later_concept in later_concepts:
                        # Check for potential influence using semantic similarity
                        influence_score = await self._calculate_influence_potential(
                            earlier_concept, later_concept
                        )

                        if influence_score > 0.7:
                            evolution_analysis["influence_chains"].append(
                                {
                                    "source": earlier_concept.name,
                                    "target": later_concept.name,
                                    "source_period": earlier_period,
                                    "target_period": later_period,
                                    "influence_score": influence_score,
                                }
                            )

            return evolution_analysis

        except Exception as e:
            logger.error(f"Error analyzing temporal evolution: {e}")
            return {}

    def _extract_temporal_indicators(self, concept: ConceptNode) -> List[str]:
        """Extract temporal period indicators from concept data"""
        temporal_patterns = {
            "ancient": ["ancient", "antiquity", "classical", "prehistoric", "archaic"],
            "medieval": [
                "medieval",
                "middle ages",
                "byzantine",
                "feudal",
                "scholastic",
            ],
            "renaissance": ["renaissance", "humanist", "reformation", "early modern"],
            "enlightenment": [
                "enlightenment",
                "rationalist",
                "empirical",
                "scientific revolution",
            ],
            "modern": [
                "modern",
                "industrial",
                "contemporary",
                "19th century",
                "20th century",
            ],
            "postmodern": ["postmodern", "contemporary", "digital", "information age"],
        }

        text_to_analyze = f"{concept.name} {concept.description}".lower()
        found_periods = []

        for period, keywords in temporal_patterns.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                found_periods.append(period)

        return found_periods or ["unknown"]

    async def _calculate_influence_potential(
        self, earlier_concept: ConceptNode, later_concept: ConceptNode
    ) -> float:
        """Calculate potential influence score between concepts from different time periods"""
        try:
            # Use semantic similarity as a base measure
            earlier_embedding = self.sentence_model.encode([earlier_concept.name])
            later_embedding = self.sentence_model.encode([later_concept.name])

            semantic_similarity = cosine_similarity(earlier_embedding, later_embedding)[
                0
            ][0]

            # Boost score if concepts are in the same domain
            domain_bonus = (
                0.2 if earlier_concept.domain == later_concept.domain else 0.0
            )

            # Check for conceptual evolution keywords
            evolution_indicators = [
                "developed",
                "evolved",
                "influenced",
                "inspired",
                "based on",
                "derived",
                "advanced",
                "refined",
                "transformed",
            ]

            text_to_check = f"{later_concept.description}".lower()
            evolution_bonus = (
                0.1
                if any(indicator in text_to_check for indicator in evolution_indicators)
                else 0.0
            )

            return float(semantic_similarity + domain_bonus + evolution_bonus)

        except Exception as e:
            logger.error(f"Error calculating influence potential: {e}")
            return 0.0
