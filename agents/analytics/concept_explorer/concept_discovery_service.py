#!/usr/bin/env python3
"""
Concept Discovery Service
Orchestrates concept discovery and database integration for MCP Yggdrasil
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio
from connection_analyzer import ConnectionAnalyzer
from thought_path_tracer import ThoughtPathTracer

from concept_explorer import (
    ConceptExplorer,
    ConceptHypothesis,
    ConceptNode,
    RelationshipEdge,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of concept discovery analysis"""

    discovery_id: str
    source_document: str
    timestamp: datetime
    concepts: List[ConceptNode]
    relationships: List[RelationshipEdge]
    hypotheses: List[ConceptHypothesis]
    network_analysis: Dict[str, Any]
    thought_paths: List[Dict[str, Any]]
    temporal_evolution: Dict[str, Any]
    confidence_score: float
    processing_time: float


class ConceptDiscoveryService:
    """
    Service that orchestrates concept discovery and database integration
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the concept discovery service"""
        self.concept_explorer = ConceptExplorer(config_path)
        self.connection_analyzer = ConnectionAnalyzer()
        self.thought_path_tracer = ThoughtPathTracer()

        # Track discovered concepts for cross-document analysis
        self.discovered_concepts: Dict[str, ConceptNode] = {}
        self.global_concept_graph = self.concept_explorer.concept_graph

        logger.info("Concept Discovery Service initialized")

    async def discover_concepts_from_content(
        self,
        content: str,
        source_document: str,
        domain: Optional[str] = None,
        include_hypotheses: bool = True,
        include_thought_paths: bool = True,
    ) -> DiscoveryResult:
        """
        Complete concept discovery pipeline for a piece of content
        """
        start_time = datetime.now()
        discovery_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting concept discovery for {source_document}")

            # Step 1: Extract concepts from content
            concepts = await self.concept_explorer.extract_concepts(
                content, source_document, domain
            )

            if not concepts:
                logger.warning(f"No concepts extracted from {source_document}")
                return self._create_empty_result(
                    discovery_id, source_document, start_time
                )

            # Step 2: Discover relationships between concepts
            relationships = await self.concept_explorer.discover_relationships(
                concepts, content
            )

            # Step 3: Add concepts to global graph for cross-document analysis
            await self._add_concepts_to_global_graph(concepts, relationships)

            # Step 4: Generate hypotheses if requested
            hypotheses = []
            if include_hypotheses:
                hypotheses = await self.concept_explorer.generate_hypotheses(
                    concepts, relationships
                )

            # Step 5: Analyze network structure
            network_analysis = await self.concept_explorer.analyze_concept_network()

            # Step 6: Generate thought paths if requested
            thought_paths = []
            if include_thought_paths and len(concepts) > 1:
                thought_paths = await self._generate_thought_paths(concepts)

            # Step 7: Temporal evolution analysis
            temporal_evolution = await self.concept_explorer.analyze_temporal_evolution(
                concepts
            )

            # Step 8: Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(
                concepts, relationships, hypotheses, network_analysis
            )

            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()

            result = DiscoveryResult(
                discovery_id=discovery_id,
                source_document=source_document,
                timestamp=start_time,
                concepts=concepts,
                relationships=relationships,
                hypotheses=hypotheses,
                network_analysis=network_analysis,
                thought_paths=thought_paths,
                temporal_evolution=temporal_evolution,
                confidence_score=confidence_score,
                processing_time=processing_time,
            )

            logger.info(
                f"Concept discovery completed for {source_document}: "
                f"{len(concepts)} concepts, {len(relationships)} relationships, "
                f"{len(hypotheses)} hypotheses in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in concept discovery for {source_document}: {e}")
            return self._create_empty_result(discovery_id, source_document, start_time)

    async def discover_cross_document_patterns(
        self, documents: List[Tuple[str, str]], domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover patterns across multiple documents
        """
        try:
            logger.info(
                f"Starting cross-document pattern analysis for {len(documents)} documents"
            )

            all_concepts = []
            all_relationships = []
            document_results = {}

            # Process each document
            for i, (content, source_doc) in enumerate(documents):
                result = await self.discover_concepts_from_content(
                    content, source_doc, domain, include_hypotheses=False
                )

                all_concepts.extend(result.concepts)
                all_relationships.extend(result.relationships)
                document_results[source_doc] = result

            # Cross-document analysis
            cross_doc_analysis = {
                "total_concepts": len(all_concepts),
                "total_relationships": len(all_relationships),
                "document_count": len(documents),
                "cross_document_connections": [],
                "common_concepts": [],
                "domain_bridges": [],
                "temporal_patterns": {},
            }

            # Find concepts that appear across multiple documents
            concept_frequency = {}
            for concept in all_concepts:
                concept_name = concept.name.lower()
                if concept_name not in concept_frequency:
                    concept_frequency[concept_name] = {
                        "count": 0,
                        "documents": [],
                        "concept_data": concept,
                    }
                concept_frequency[concept_name]["count"] += 1
                concept_frequency[concept_name]["documents"].append(
                    concept.source_documents[0]
                )

            # Identify common concepts (appear in multiple documents)
            for concept_name, data in concept_frequency.items():
                if data["count"] > 1:
                    cross_doc_analysis["common_concepts"].append(
                        {
                            "name": concept_name,
                            "frequency": data["count"],
                            "documents": data["documents"],
                            "concept": data["concept_data"],
                        }
                    )

            # Find cross-document connections
            for i, concept1 in enumerate(all_concepts):
                for j, concept2 in enumerate(all_concepts[i + 1 :], i + 1):
                    if (
                        concept1.source_documents[0] != concept2.source_documents[0]
                        and concept1.domain != concept2.domain
                    ):

                        # Check for potential cross-document bridge
                        bridge_strength = (
                            await self._calculate_cross_document_bridge_strength(
                                concept1, concept2
                            )
                        )

                        if bridge_strength > 0.6:
                            cross_doc_analysis["cross_document_connections"].append(
                                {
                                    "concept1": concept1.name,
                                    "concept2": concept2.name,
                                    "document1": concept1.source_documents[0],
                                    "document2": concept2.source_documents[0],
                                    "bridge_strength": bridge_strength,
                                    "domain_bridge": (concept1.domain, concept2.domain),
                                }
                            )

            # Analyze temporal patterns across documents
            temporal_concepts = {}
            for concept in all_concepts:
                periods = self.concept_explorer._extract_temporal_indicators(concept)
                for period in periods:
                    if period not in temporal_concepts:
                        temporal_concepts[period] = []
                    temporal_concepts[period].append(concept)

            cross_doc_analysis["temporal_patterns"] = {
                period: {
                    "concept_count": len(concepts),
                    "documents": list(set(c.source_documents[0] for c in concepts)),
                    "domains": list(set(c.domain for c in concepts)),
                }
                for period, concepts in temporal_concepts.items()
            }

            logger.info(
                f"Cross-document analysis completed: "
                f"{len(cross_doc_analysis['common_concepts'])} common concepts, "
                f"{len(cross_doc_analysis['cross_document_connections'])} cross-document connections"
            )

            return cross_doc_analysis

        except Exception as e:
            logger.error(f"Error in cross-document pattern discovery: {e}")
            return {}

    async def generate_knowledge_graph_data(
        self, discovery_results: List[DiscoveryResult]
    ) -> Dict[str, Any]:
        """
        Generate data formatted for knowledge graph integration
        """
        try:
            logger.info(
                f"Generating knowledge graph data from {len(discovery_results)} discovery results"
            )

            # Prepare data structures
            nodes = []
            edges = []
            node_ids = set()

            # Process each discovery result
            for result in discovery_results:
                # Add concept nodes
                for concept in result.concepts:
                    if concept.id not in node_ids:
                        node_data = {
                            "id": concept.id,
                            "name": concept.name,
                            "type": "concept",
                            "domain": concept.domain,
                            "description": concept.description,
                            "confidence": concept.confidence,
                            "source_documents": concept.source_documents,
                            "extraction_method": concept.extraction_method,
                            "discovery_id": result.discovery_id,
                        }
                        nodes.append(node_data)
                        node_ids.add(concept.id)

                # Add relationship edges
                for relationship in result.relationships:
                    if (
                        relationship.source_id in node_ids
                        and relationship.target_id in node_ids
                    ):
                        edge_data = {
                            "source": relationship.source_id,
                            "target": relationship.target_id,
                            "type": relationship.relationship_type,
                            "strength": relationship.strength,
                            "confidence": relationship.confidence,
                            "evidence": relationship.evidence,
                            "bidirectional": relationship.bidirectional,
                            "discovery_id": result.discovery_id,
                        }
                        edges.append(edge_data)

                # Add hypothesis nodes
                for hypothesis in result.hypotheses:
                    hypothesis_node = {
                        "id": f"hyp_{hypothesis.hypothesis_id}",
                        "name": f"Hypothesis: {hypothesis.description[:50]}...",
                        "type": "hypothesis",
                        "domain": (
                            "cross_domain" if hypothesis.domain_bridge else "unknown"
                        ),
                        "description": hypothesis.description,
                        "confidence": hypothesis.evidence_strength,
                        "source_documents": [result.source_document],
                        "extraction_method": "hypothesis_generation",
                        "discovery_id": result.discovery_id,
                        "novelty_score": hypothesis.novelty_score,
                        "testable_predictions": hypothesis.testable_predictions,
                    }
                    nodes.append(hypothesis_node)

                    # Connect hypothesis to supporting concepts
                    for concept_id in hypothesis.supporting_concepts:
                        if concept_id in node_ids:
                            edge_data = {
                                "source": concept_id,
                                "target": f"hyp_{hypothesis.hypothesis_id}",
                                "type": "supports_hypothesis",
                                "strength": hypothesis.evidence_strength,
                                "confidence": hypothesis.evidence_strength,
                                "evidence": [f"Supporting concept for hypothesis"],
                                "bidirectional": False,
                                "discovery_id": result.discovery_id,
                            }
                            edges.append(edge_data)

            # Calculate graph statistics
            graph_stats = {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "concept_nodes": len([n for n in nodes if n["type"] == "concept"]),
                "hypothesis_nodes": len(
                    [n for n in nodes if n["type"] == "hypothesis"]
                ),
                "domains": list(set(n["domain"] for n in nodes)),
                "discovery_sessions": len(discovery_results),
            }

            # Prepare final output
            knowledge_graph_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator": "ConceptDiscoveryService",
                    "version": "1.0",
                    "statistics": graph_stats,
                },
                "nodes": nodes,
                "edges": edges,
                "export_format": "neo4j_compatible",
            }

            logger.info(f"Knowledge graph data generated: {graph_stats}")

            return knowledge_graph_data

        except Exception as e:
            logger.error(f"Error generating knowledge graph data: {e}")
            return {}

    async def _add_concepts_to_global_graph(
        self, concepts: List[ConceptNode], relationships: List[RelationshipEdge]
    ):
        """Add concepts and relationships to the global concept graph"""
        try:
            # Add concepts as nodes
            for concept in concepts:
                if concept.id not in self.global_concept_graph:
                    self.global_concept_graph.add_node(concept.id, **asdict(concept))
                    self.discovered_concepts[concept.id] = concept

            # Add relationships as edges
            for relationship in relationships:
                if (
                    relationship.source_id in self.global_concept_graph
                    and relationship.target_id in self.global_concept_graph
                ):

                    self.global_concept_graph.add_edge(
                        relationship.source_id,
                        relationship.target_id,
                        **asdict(relationship),
                    )

        except Exception as e:
            logger.error(f"Error adding concepts to global graph: {e}")

    async def _generate_thought_paths(
        self, concepts: List[ConceptNode]
    ) -> List[Dict[str, Any]]:
        """Generate thought paths between concepts"""
        thought_paths = []

        try:
            # Generate paths between different domain concepts
            domain_concepts = {}
            for concept in concepts:
                if concept.domain not in domain_concepts:
                    domain_concepts[concept.domain] = []
                domain_concepts[concept.domain].append(concept)

            # Generate paths between different domains
            domains = list(domain_concepts.keys())
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    domain1_concepts = domain_concepts[domains[i]]
                    domain2_concepts = domain_concepts[domains[j]]

                    # Try to find paths between concepts from different domains
                    for concept1 in domain1_concepts[
                        :3
                    ]:  # Limit to avoid too many paths
                        for concept2 in domain2_concepts[:3]:
                            paths = await self.concept_explorer.trace_thought_paths(
                                concept1.id, concept2.id, max_depth=4
                            )

                            for path in paths:
                                thought_paths.append(
                                    {
                                        "path_id": path.path_id,
                                        "start_concept": concept1.name,
                                        "end_concept": concept2.name,
                                        "start_domain": concept1.domain,
                                        "end_domain": concept2.domain,
                                        "path_length": len(path.intermediate_steps),
                                        "total_strength": path.total_strength,
                                        "reasoning_chain": path.reasoning_chain,
                                    }
                                )

        except Exception as e:
            logger.error(f"Error generating thought paths: {e}")

        return thought_paths

    async def _calculate_cross_document_bridge_strength(
        self, concept1: ConceptNode, concept2: ConceptNode
    ) -> float:
        """Calculate strength of cross-document bridge between concepts"""
        try:
            # Use semantic similarity as base
            embeddings1 = self.concept_explorer.sentence_model.encode([concept1.name])
            embeddings2 = self.concept_explorer.sentence_model.encode([concept2.name])

            from sklearn.metrics.pairwise import cosine_similarity

            semantic_sim = cosine_similarity(embeddings1, embeddings2)[0][0]

            # Boost for cross-domain bridges
            cross_domain_bonus = 0.2 if concept1.domain != concept2.domain else 0.0

            # Boost for high confidence concepts
            confidence_bonus = (concept1.confidence + concept2.confidence) / 2 * 0.1

            return float(semantic_sim + cross_domain_bonus + confidence_bonus)

        except Exception as e:
            logger.error(f"Error calculating cross-document bridge strength: {e}")
            return 0.0

    def _calculate_confidence_score(
        self,
        concepts: List[ConceptNode],
        relationships: List[RelationshipEdge],
        hypotheses: List[ConceptHypothesis],
        network_analysis: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence score for the discovery result"""
        try:
            if not concepts:
                return 0.0

            # Base confidence from concepts
            concept_confidence = sum(c.confidence for c in concepts) / len(concepts)

            # Relationship confidence
            relationship_confidence = 0.0
            if relationships:
                relationship_confidence = sum(
                    r.confidence for r in relationships
                ) / len(relationships)

            # Network structure quality
            network_quality = 0.0
            if network_analysis.get("node_count", 0) > 0:
                density = network_analysis.get("density", 0.0)
                clustering = network_analysis.get("clustering_coefficient", 0.0)
                network_quality = (density + clustering) / 2

            # Hypothesis quality
            hypothesis_quality = 0.0
            if hypotheses:
                hypothesis_quality = sum(h.evidence_strength for h in hypotheses) / len(
                    hypotheses
                )

            # Weighted average
            weights = [0.4, 0.3, 0.2, 0.1]
            scores = [
                concept_confidence,
                relationship_confidence,
                network_quality,
                hypothesis_quality,
            ]

            return sum(w * s for w, s in zip(weights, scores))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    def _create_empty_result(
        self, discovery_id: str, source_document: str, start_time: datetime
    ) -> DiscoveryResult:
        """Create empty result for failed discovery"""
        return DiscoveryResult(
            discovery_id=discovery_id,
            source_document=source_document,
            timestamp=start_time,
            concepts=[],
            relationships=[],
            hypotheses=[],
            network_analysis={"error": "No concepts discovered"},
            thought_paths=[],
            temporal_evolution={},
            confidence_score=0.0,
            processing_time=(datetime.now() - start_time).total_seconds(),
        )

    async def export_discovery_results(
        self,
        results: List[DiscoveryResult],
        output_path: str,
        format_type: str = "json",
    ) -> bool:
        """Export discovery results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "json":
                # Convert to JSON-serializable format
                export_data = {
                    "metadata": {
                        "export_date": datetime.now().isoformat(),
                        "total_results": len(results),
                        "format": "concept_discovery_results",
                    },
                    "results": [],
                }

                for result in results:
                    result_data = {
                        "discovery_id": result.discovery_id,
                        "source_document": result.source_document,
                        "timestamp": result.timestamp.isoformat(),
                        "concepts": [asdict(c) for c in result.concepts],
                        "relationships": [asdict(r) for r in result.relationships],
                        "hypotheses": [asdict(h) for h in result.hypotheses],
                        "network_analysis": result.network_analysis,
                        "thought_paths": result.thought_paths,
                        "temporal_evolution": result.temporal_evolution,
                        "confidence_score": result.confidence_score,
                        "processing_time": result.processing_time,
                    }
                    export_data["results"].append(result_data)

                with open(output_file, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

                logger.info(f"Discovery results exported to {output_path}")
                return True

            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False

        except Exception as e:
            logger.error(f"Error exporting discovery results: {e}")
            return False


async def main():
    """Test the concept discovery service"""
    service = ConceptDiscoveryService()

    # Test content
    test_content = """
    The golden ratio, also known as the divine proportion, appears frequently in nature and art.
    This mathematical constant, approximately 1.618, has fascinated mathematicians and artists
    for centuries. It represents a perfect balance between order and beauty, connecting
    mathematical precision with aesthetic harmony.
    """

    # Test concept discovery
    result = await service.discover_concepts_from_content(
        test_content, "test_document.txt", domain="mathematics"
    )

    print(f"Discovery completed: {len(result.concepts)} concepts found")
    for concept in result.concepts:
        print(
            f"  - {concept.name} ({concept.domain}, confidence: {concept.confidence:.2f})"
        )

    print(f"\nRelationships: {len(result.relationships)}")
    for rel in result.relationships:
        print(f"  - {rel.source_id} -> {rel.target_id} ({rel.relationship_type})")

    print(f"\nHypotheses: {len(result.hypotheses)}")
    for hyp in result.hypotheses:
        print(f"  - {hyp.description[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
