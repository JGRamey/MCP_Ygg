#!/usr/bin/env python3
"""
Concept Explorer Agent
Advanced relationship discovery and hypothesis generation for MCP Yggdrasil
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

# NLP and ML imports
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

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
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm"):
        """Initialize the concept explorer"""
        
        # Load NLP models
        try:
            self.nlp = spacy.load(spacy_model)
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded models: {spacy_model}, {model_name}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        # Initialize concept graph
        self.concept_graph = nx.Graph()
        self.concept_nodes: Dict[str, ConceptNode] = {}
        self.relationship_edges: Dict[str, RelationshipEdge] = {}
        
        # Domain mappings
        self.domain_keywords = {
            "mathematics": ["number", "equation", "theorem", "proof", "geometry", "algebra"],
            "science": ["physics", "chemistry", "biology", "experiment", "theory", "discovery"],
            "philosophy": ["ethics", "metaphysics", "logic", "consciousness", "reality", "truth"],
            "art": ["painting", "sculpture", "music", "literature", "aesthetics", "beauty"],
            "religion": ["sacred", "divine", "prayer", "ritual", "scripture", "faith"],
            "technology": ["invention", "engineering", "mechanism", "tool", "innovation"]
        }
        
        # Relationship types and patterns
        self.relationship_patterns = {
            "causal": ["causes", "leads to", "results in", "produces", "generates"],
            "temporal": ["before", "after", "during", "precedes", "follows"],
            "similarity": ["similar to", "like", "resembles", "analogous", "comparable"],
            "opposition": ["opposes", "contradicts", "differs from", "opposite"],
            "containment": ["includes", "contains", "part of", "subset", "element"],
            "influence": ["influences", "affects", "shapes", "impacts", "modifies"]
        }
    
    async def extract_concepts(self, 
                             text: str, 
                             source_document: str,
                             domain: Optional[str] = None) -> List[ConceptNode]:
        """
        Extract concepts from text using NLP analysis
        """
        try:
            concepts = []
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                    concept = ConceptNode(
                        id=f"ent_{len(concepts)}_{ent.text.replace(' ', '_').lower()}",
                        name=ent.text,
                        domain=domain or self._classify_domain(ent.text),
                        description=f"{ent.label_}: {ent.text}",
                        confidence=0.8,
                        source_documents=[source_document],
                        extraction_method="named_entity_recognition"
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
                            extraction_method="noun_phrase_extraction"
                        )
                        concepts.append(concept)
            
            # Extract domain-specific concepts
            domain_concepts = await self._extract_domain_concepts(text, source_document, domain)
            concepts.extend(domain_concepts)
            
            logger.info(f"Extracted {len(concepts)} concepts from {source_document}")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []
    
    async def discover_relationships(self, 
                                   concepts: List[ConceptNode],
                                   text: str) -> List[RelationshipEdge]:
        """
        Discover relationships between concepts using multiple methods
        """
        try:
            relationships = []
            
            # Method 1: Semantic similarity
            semantic_rels = await self._find_semantic_relationships(concepts)
            relationships.extend(semantic_rels)
            
            # Method 2: Co-occurrence analysis
            cooccurrence_rels = await self._find_cooccurrence_relationships(concepts, text)
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
    
    async def generate_hypotheses(self, 
                                concepts: List[ConceptNode],
                                relationships: List[RelationshipEdge]) -> List[ConceptHypothesis]:
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
    
    async def trace_thought_paths(self, 
                                start_concept: str,
                                end_concept: str,
                                max_depth: int = 5) -> List[ThoughtPath]:
        """
        Trace paths of reasoning between concepts
        """
        try:
            paths = []
            
            # Use graph algorithms to find paths
            try:
                all_paths = list(nx.all_simple_paths(
                    self.concept_graph, 
                    start_concept, 
                    end_concept, 
                    cutoff=max_depth
                ))
                
                for i, path in enumerate(all_paths[:10]):  # Limit to top 10 paths
                    # Calculate path strength
                    total_strength = 0.0
                    reasoning_chain = []
                    intermediate_steps = []
                    
                    for j in range(len(path) - 1):
                        current_node = path[j]
                        next_node = path[j + 1]
                        
                        # Get edge data
                        edge_data = self.concept_graph.get_edge_data(current_node, next_node)
                        if edge_data:
                            total_strength += edge_data.get('strength', 0.5)
                            
                            step = {
                                "from": current_node,
                                "to": next_node,
                                "relationship": edge_data.get('relationship_type', 'unknown'),
                                "strength": edge_data.get('strength', 0.5),
                                "evidence": edge_data.get('evidence', [])
                            }
                            intermediate_steps.append(step)
                            
                            # Build reasoning chain
                            from_concept = self.concept_nodes.get(current_node, {}).get('name', current_node)
                            to_concept = self.concept_nodes.get(next_node, {}).get('name', next_node)
                            relationship = edge_data.get('relationship_type', 'relates to')
                            
                            reasoning_chain.append(f"{from_concept} {relationship} {to_concept}")
                    
                    thought_path = ThoughtPath(
                        path_id=f"path_{i}_{start_concept}_{end_concept}",
                        start_concept=start_concept,
                        end_concept=end_concept,
                        intermediate_steps=intermediate_steps,
                        total_strength=total_strength / max(len(path) - 1, 1),
                        reasoning_chain=reasoning_chain
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
                analysis["average_shortest_path"] = nx.average_shortest_path_length(self.concept_graph)
            else:
                analysis["connected_components"] = nx.number_connected_components(self.concept_graph)
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(self.concept_graph)
            betweenness_centrality = nx.betweenness_centrality(self.concept_graph)
            closeness_centrality = nx.closeness_centrality(self.concept_graph)
            
            # Find most central concepts
            analysis["most_connected"] = sorted(
                degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            analysis["most_influential"] = sorted(
                betweenness_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Domain distribution
            domain_counts = {}
            for node_id, node_data in self.concept_graph.nodes(data=True):
                domain = node_data.get('domain', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            analysis["domain_distribution"] = domain_counts
            
            # Clustering
            analysis["clustering_coefficient"] = nx.average_clustering(self.concept_graph)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing concept network: {e}")
            return {"error": str(e)}
    
    async def _extract_domain_concepts(self, 
                                     text: str, 
                                     source_document: str,
                                     domain: Optional[str]) -> List[ConceptNode]:
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
                        extraction_method="domain_keyword_matching"
                    )
                    concepts.append(concept)
        
        return concepts
    
    async def _find_semantic_relationships(self, concepts: List[ConceptNode]) -> List[RelationshipEdge]:
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
                        bidirectional=True
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _find_cooccurrence_relationships(self, 
                                             concepts: List[ConceptNode],
                                             text: str) -> List[RelationshipEdge]:
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
                    window = " ".join(words[k:k + window_size])
                    if concept1.name.lower() in window.lower() and concept2.name.lower() in window.lower():
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
                        bidirectional=True
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _find_pattern_relationships(self, 
                                        concepts: List[ConceptNode],
                                        text: str) -> List[RelationshipEdge]:
        """Find relationships using linguistic patterns"""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                # Use regex to find pattern occurrences
                pattern_regex = rf"(\w+(?:\s+\w+)*)\s+{re.escape(pattern)}\s+(\w+(?:\s+\w+)*)"
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
                    
                    if source_concept and target_concept and source_concept != target_concept:
                        relationship = RelationshipEdge(
                            source_id=source_concept.id,
                            target_id=target_concept.id,
                            relationship_type=rel_type,
                            strength=0.8,
                            evidence=[f"Pattern: {match.group(0)}"],
                            confidence=0.7,
                            bidirectional=rel_type in ["similarity", "cooccurrence"]
                        )
                        relationships.append(relationship)
        
        return relationships
    
    async def _find_cross_domain_relationships(self, concepts: List[ConceptNode]) -> List[RelationshipEdge]:
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
                                evidence=[f"Cross-domain connection: {domain1} ↔ {domain2}"],
                                confidence=0.5,
                                bidirectional=True
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
            "the time", "this is", "that was", "it is", "they are",
            "we have", "you can", "will be", "has been", "would be"
        }
        return text.lower() in common_phrases or len(text.split()) > 4
    
    def _has_cross_domain_potential(self, concept1: ConceptNode, concept2: ConceptNode) -> bool:
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
            "number", "pattern", "structure", "system", "order",
            "harmony", "balance", "unity", "infinity", "creation"
        }
        
        for theme in cross_domain_themes:
            if theme in concept1.name.lower() and theme in concept2.name.lower():
                return True
        
        return False
    
    async def _generate_transitive_hypotheses(self, graph: nx.Graph) -> List[ConceptHypothesis]:
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
                        graph.has_edge(node_a, node_c)
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
                                domain_bridge=None
                            )
                            hypotheses.append(hypothesis)
        
        return hypotheses[:10]  # Limit results
    
    async def _generate_missing_link_hypotheses(self, graph: nx.Graph) -> List[ConceptHypothesis]:
        """Generate hypotheses about missing links in the concept network"""
        # This is a placeholder for more sophisticated missing link prediction
        return []
    
    async def _generate_bridge_hypotheses(self, graph: nx.Graph) -> List[ConceptHypothesis]:
        """Generate hypotheses about cross-domain bridges"""
        # This is a placeholder for cross-domain bridge hypothesis generation
        return []
    
    async def _generate_anomaly_hypotheses(self, graph: nx.Graph) -> List[ConceptHypothesis]:
        """Generate hypotheses based on network anomalies"""
        # This is a placeholder for anomaly-based hypothesis generation
        return []