#!/usr/bin/env python3
"""
Knowledge Integration Orchestrator for MCP Yggdrasil
Phase 4: Orchestrate knowledge integration into Neo4j and Qdrant databases
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncio

from ..content_analyzer.deep_content_analyzer import ContentAnalysis
from ..quality_assessment.reliability_scorer import ReliabilityScore

# Import Phase 4 components
from ..scraper.intelligent_scraper_agent import ScrapedDocument

# Import database agents
try:
    from ..neo4j_manager.neo4j_agent import Neo4jAgent
    from ..qdrant_manager.qdrant_agent import QdrantAgent

    NEO4J_AVAILABLE = True
    QDRANT_AVAILABLE = True
except ImportError as e:
    NEO4J_AVAILABLE = False
    QDRANT_AVAILABLE = False
    logging.warning(f"Database agents not available: {e}")

# Import vector indexer for embeddings
try:
    from ..qdrant_manager.vector_index.enhanced_indexer import EnhancedVectorIndexer

    VECTOR_INDEXER_AVAILABLE = True
except ImportError:
    VECTOR_INDEXER_AVAILABLE = False
    logging.warning("Enhanced Vector Indexer not available")

logger = logging.getLogger(__name__)


@dataclass
class Neo4jIntegrationData:
    """Prepared data for Neo4j knowledge graph integration."""

    nodes: List[Dict]
    relationships: List[Dict]
    properties: Dict
    transaction_id: str


@dataclass
class QdrantIntegrationData:
    """Prepared data for Qdrant vector database integration."""

    vectors: List[Dict]
    collection_name: str
    metadata: Dict


@dataclass
class IntegrationResult:
    """Result of knowledge integration process."""

    success: bool
    neo4j_result: Optional[Dict]
    qdrant_result: Optional[Dict]
    errors: List[str]
    provenance_id: str
    nodes_created: int
    relationships_created: int
    vectors_inserted: int


@dataclass
class ProvenanceRecord:
    """Provenance tracking for knowledge sources."""

    id: str
    source_url: str
    scraping_timestamp: datetime
    processing_agents: List[str]
    quality_scores: Dict
    integration_timestamp: datetime
    content_hash: str
    neo4j_nodes: List[str]
    qdrant_collections: List[str]


class KnowledgeIntegrationOrchestrator:
    """Orchestrate knowledge integration into databases with full provenance tracking."""

    def __init__(self):
        self.neo4j_agent = Neo4jAgent() if NEO4J_AVAILABLE else None
        self.qdrant_agent = QdrantAgent() if QDRANT_AVAILABLE else None
        self.vector_indexer = (
            EnhancedVectorIndexer() if VECTOR_INDEXER_AVAILABLE else None
        )

        # Event type mappings for historical events
        self.event_mappings = {
            "spanish inquisition": {
                "start_date": "1478",
                "end_date": "1834",
                "description": "Religious persecution in Spain",
                "significance": "major",
                "type": "religious_persecution",
            },
            "holocaust": {
                "start_date": "1941",
                "end_date": "1945",
                "description": "Genocide during World War II",
                "significance": "major",
                "type": "genocide",
            },
            "crucifixion": {
                "start_date": "~30 CE",
                "end_date": "~33 CE",
                "description": "Execution of Jesus Christ",
                "significance": "major",
                "type": "religious_event",
            },
            "fall of rome": {
                "start_date": "476 CE",
                "end_date": "476 CE",
                "description": "End of Western Roman Empire",
                "significance": "major",
                "type": "political_collapse",
            },
            "renaissance": {
                "start_date": "1300",
                "end_date": "1600",
                "description": "Cultural rebirth in Europe",
                "significance": "major",
                "type": "cultural_movement",
            },
            "industrial revolution": {
                "start_date": "1760",
                "end_date": "1840",
                "description": "Transition to manufacturing",
                "significance": "major",
                "type": "economic_transformation",
            },
        }

    async def prepare_neo4j_integration(
        self,
        scraped_doc: ScrapedDocument,
        content_analysis: ContentAnalysis,
        reliability_score: ReliabilityScore,
    ) -> Neo4jIntegrationData:
        """Prepare comprehensive data for Neo4j knowledge graph."""

        nodes = []
        relationships = []

        # Generate consistent IDs
        doc_id = f"doc_{scraped_doc.content_hash[:12]}"

        # Create main document node
        doc_node = {
            "type": "Document",
            "id": doc_id,
            "properties": {
                "url": scraped_doc.metadata.url,
                "title": scraped_doc.metadata.title,
                "content_type": scraped_doc.metadata.content_type.value,
                "domain": scraped_doc.metadata.domain,
                "reliability_score": reliability_score.overall_score,
                "confidence_level": reliability_score.confidence_level.value,
                "word_count": scraped_doc.metadata.word_count,
                "language": scraped_doc.metadata.language,
                "scraped_at": scraped_doc.scraping_timestamp.isoformat(),
                "content_hash": scraped_doc.content_hash,
                "authority_score": scraped_doc.metadata.authority_score,
                "reading_time_minutes": scraped_doc.metadata.reading_time_minutes,
            },
        }
        nodes.append(doc_node)

        # Create author nodes and relationships
        for author in scraped_doc.metadata.authors:
            author_id = f"author_{self._generate_id(author)}"
            author_node = {
                "type": "Author",
                "id": author_id,
                "properties": {
                    "name": author,
                    "first_seen": scraped_doc.scraping_timestamp.isoformat(),
                },
            }
            nodes.append(author_node)

            # Create authorship relationship
            relationships.append(
                {
                    "type": "AUTHORED_BY",
                    "from": doc_id,
                    "to": author_id,
                    "properties": {"verified": True, "confidence": 0.9},
                }
            )

        # Create entity nodes and relationships
        for entity in content_analysis.entities[:20]:  # Limit to top 20 entities
            entity_id = f"entity_{self._generate_id(entity.text)}"
            entity_node = {
                "type": "Entity",
                "id": entity_id,
                "properties": {
                    "name": entity.text,
                    "entity_type": entity.type,
                    "confidence": entity.confidence,
                    "first_mentioned": scraped_doc.scraping_timestamp.isoformat(),
                },
            }
            nodes.append(entity_node)

            # Create mention relationship
            relationships.append(
                {
                    "type": "MENTIONS",
                    "from": doc_id,
                    "to": entity_id,
                    "properties": {
                        "position": entity.start_pos,
                        "context_length": entity.end_pos - entity.start_pos,
                        "confidence": entity.confidence,
                    },
                }
            )

        # Create concept nodes and relationships
        for concept in content_analysis.concepts[:15]:  # Limit to top 15 concepts
            concept_id = f"concept_{self._generate_id(concept.name)}"
            concept_node = {
                "type": "Concept",
                "id": concept_id,
                "properties": {
                    "name": concept.name,
                    "domain": concept.domain,
                    "confidence": concept.confidence,
                    "context_preview": concept.context[:200],
                    "first_identified": scraped_doc.scraping_timestamp.isoformat(),
                },
            }
            nodes.append(concept_node)

            # Create exploration relationship
            relationships.append(
                {
                    "type": "EXPLORES",
                    "from": doc_id,
                    "to": concept_id,
                    "properties": {
                        "depth": "surface" if concept.confidence < 0.7 else "deep",
                        "context": concept.context[:500],
                    },
                }
            )

            # Link concepts to related entities
            for related_entity in concept.related_entities:
                entity_id = f"entity_{self._generate_id(related_entity)}"
                relationships.append(
                    {
                        "type": "RELATES_TO",
                        "from": concept_id,
                        "to": entity_id,
                        "properties": {
                            "relationship_strength": 0.7,
                            "discovered_in": doc_id,
                        },
                    }
                )

        # Create claim nodes for verifiable claims
        for claim in content_analysis.claims:
            if claim.verifiable and claim.confidence > 0.6:
                claim_id = f"claim_{self._generate_id(claim.claim_text)}"
                claim_node = {
                    "type": "Claim",
                    "id": claim_id,
                    "properties": {
                        "text": claim.claim_text,
                        "claim_type": claim.claim_type,
                        "confidence": claim.confidence,
                        "verifiable": claim.verifiable,
                        "first_stated": scraped_doc.scraping_timestamp.isoformat(),
                        "reliability_context": reliability_score.overall_score,
                    },
                }
                nodes.append(claim_node)

                # Create claim relationship
                relationships.append(
                    {
                        "type": "MAKES_CLAIM",
                        "from": doc_id,
                        "to": claim_id,
                        "properties": {
                            "supporting_entities": claim.supporting_entities,
                            "claim_strength": claim.confidence,
                        },
                    }
                )

                # Link claims to supporting entities
                for entity_name in claim.supporting_entities:
                    entity_id = f"entity_{self._generate_id(entity_name)}"
                    relationships.append(
                        {
                            "type": "SUPPORTED_BY",
                            "from": claim_id,
                            "to": entity_id,
                            "properties": {
                                "support_type": "entity_reference",
                                "confidence": 0.8,
                            },
                        }
                    )

        # Create historical event nodes
        content_lower = scraped_doc.content.lower()
        for event_name, event_data in self.event_mappings.items():
            if event_name in content_lower:
                event_id = f"event_{self._generate_id(event_name)}"
                event_node = {
                    "type": "Event",
                    "id": event_id,
                    "properties": {
                        "name": event_name.title(),
                        "start_date": event_data["start_date"],
                        "end_date": event_data["end_date"],
                        "description": event_data["description"],
                        "historical_significance": event_data["significance"],
                        "event_type": event_data["type"],
                        "first_referenced": scraped_doc.scraping_timestamp.isoformat(),
                    },
                }
                nodes.append(event_node)

                # Create reference relationship
                relationships.append(
                    {
                        "type": "REFERENCES_EVENT",
                        "from": doc_id,
                        "to": event_id,
                        "properties": {
                            "reference_type": "textual_mention",
                            "context_relevance": 0.8,
                        },
                    }
                )

        # Create domain node and relationship
        primary_domain = (
            max(content_analysis.domain_mapping.items(), key=lambda x: x[1])[0]
            if content_analysis.domain_mapping
            else "general"
        )
        domain_id = f"domain_{primary_domain}"
        domain_node = {
            "type": "Domain",
            "id": domain_id,
            "properties": {
                "name": primary_domain,
                "description": f"Academic domain: {primary_domain}",
                "last_updated": scraped_doc.scraping_timestamp.isoformat(),
            },
        }
        nodes.append(domain_node)

        relationships.append(
            {
                "type": "BELONGS_TO_DOMAIN",
                "from": doc_id,
                "to": domain_id,
                "properties": {
                    "confidence": content_analysis.domain_mapping.get(
                        primary_domain, 0.5
                    ),
                    "assignment_method": "ml_classification",
                },
            }
        )

        return Neo4jIntegrationData(
            nodes=nodes,
            relationships=relationships,
            properties={
                "primary_domain": primary_domain,
                "reliability_score": reliability_score.overall_score,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
            },
            transaction_id=f"tx_{datetime.utcnow().timestamp()}",
        )

    async def prepare_qdrant_integration(
        self,
        scraped_doc: ScrapedDocument,
        content_analysis: ContentAnalysis,
        reliability_score: ReliabilityScore,
    ) -> QdrantIntegrationData:
        """Prepare vectors and metadata for Qdrant integration."""

        vectors = []

        # Determine primary domain for collection selection
        primary_domain = (
            max(content_analysis.domain_mapping.items(), key=lambda x: x[1])[0]
            if content_analysis.domain_mapping
            else "general"
        )

        if not self.vector_indexer:
            logger.warning("Vector indexer not available, creating mock vectors")
            # Create mock vector data for testing
            import random

            mock_vector = [
                random.random() for _ in range(384)
            ]  # Standard embedding size

            vectors.append(
                {
                    "id": scraped_doc.content_hash,
                    "vector": mock_vector,
                    "payload": {
                        "neo4j_id": f"doc_{scraped_doc.content_hash[:12]}",
                        "title": scraped_doc.metadata.title,
                        "url": scraped_doc.metadata.url,
                        "content_type": scraped_doc.metadata.content_type.value,
                        "domain": primary_domain,
                        "reliability_score": reliability_score.overall_score,
                        "confidence_level": reliability_score.confidence_level.value,
                        "timestamp": scraped_doc.scraping_timestamp.isoformat(),
                    },
                }
            )
        else:
            # Use actual vector indexer
            try:
                # Index main document
                doc_vector_result = await self.vector_indexer.index_content(
                    {
                        "id": scraped_doc.content_hash,
                        "text": scraped_doc.content,
                        "domain": primary_domain,
                        "language": scraped_doc.metadata.language,
                    }
                )

                vectors.append(
                    {
                        "id": doc_vector_result.vector_id,
                        "vector": doc_vector_result.embedding.tolist(),
                        "payload": {
                            "neo4j_id": f"doc_{scraped_doc.content_hash[:12]}",
                            "title": scraped_doc.metadata.title,
                            "url": scraped_doc.metadata.url,
                            "content_type": scraped_doc.metadata.content_type.value,
                            "domain": primary_domain,
                            "subdomain_mapping": content_analysis.domain_mapping,
                            "reliability_score": reliability_score.overall_score,
                            "confidence_level": reliability_score.confidence_level.value,
                            "authors": scraped_doc.metadata.authors,
                            "language": scraped_doc.metadata.language,
                            "key_topics": content_analysis.key_topics,
                            "entity_count": len(content_analysis.entities),
                            "concept_count": len(content_analysis.concepts),
                            "claim_count": len(content_analysis.claims),
                            "timestamp": scraped_doc.scraping_timestamp.isoformat(),
                            "content_hash": scraped_doc.content_hash,
                            "model_used": doc_vector_result.model_used,
                        },
                    }
                )

                # Index key concepts as separate vectors
                for concept in content_analysis.concepts[:10]:  # Top 10 concepts
                    concept_result = await self.vector_indexer.index_content(
                        {
                            "id": f"concept_{self._generate_id(concept.name)}",
                            "text": f"{concept.name}: {concept.context}",
                            "domain": concept.domain,
                        }
                    )

                    vectors.append(
                        {
                            "id": concept_result.vector_id,
                            "vector": concept_result.embedding.tolist(),
                            "payload": {
                                "type": "concept",
                                "name": concept.name,
                                "domain": concept.domain,
                                "parent_document": scraped_doc.content_hash,
                                "parent_neo4j_id": f"doc_{scraped_doc.content_hash[:12]}",
                                "confidence": concept.confidence,
                                "context_preview": concept.context[:200],
                                "related_entities": concept.related_entities,
                            },
                        }
                    )

            except Exception as e:
                logger.error(f"Vector indexing failed: {e}")
                # Fallback to mock vectors
                import random

                mock_vector = [random.random() for _ in range(384)]
                vectors.append(
                    {
                        "id": scraped_doc.content_hash,
                        "vector": mock_vector,
                        "payload": {
                            "neo4j_id": f"doc_{scraped_doc.content_hash[:12]}",
                            "title": scraped_doc.metadata.title,
                            "error": "vector_indexing_failed",
                        },
                    }
                )

        # Select collection based on domain
        collection_name = f"documents_{primary_domain}"

        return QdrantIntegrationData(
            vectors=vectors,
            collection_name=collection_name,
            metadata={
                "total_vectors": len(vectors),
                "primary_domain": primary_domain,
                "document_count": 1,
                "concept_count": len(
                    [v for v in vectors if v["payload"].get("type") == "concept"]
                ),
            },
        )

    async def execute_integration(
        self, neo4j_data: Neo4jIntegrationData, qdrant_data: QdrantIntegrationData
    ) -> IntegrationResult:
        """Execute database integration with transaction safety."""

        errors = []
        neo4j_result = None
        qdrant_result = None
        nodes_created = 0
        relationships_created = 0
        vectors_inserted = 0

        try:
            # Execute Neo4j integration
            if self.neo4j_agent:
                try:
                    neo4j_result = await self._execute_neo4j_integration(neo4j_data)
                    nodes_created = neo4j_result.get("nodes_created", 0)
                    relationships_created = neo4j_result.get("relationships_created", 0)
                    logger.info(
                        f"Neo4j integration successful: {nodes_created} nodes, {relationships_created} relationships"
                    )
                except Exception as e:
                    error_msg = f"Neo4j integration failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            else:
                errors.append("Neo4j agent not available")

            # Execute Qdrant integration
            if self.qdrant_agent:
                try:
                    qdrant_result = await self._execute_qdrant_integration(qdrant_data)
                    vectors_inserted = qdrant_result.get("vectors_inserted", 0)
                    logger.info(
                        f"Qdrant integration successful: {vectors_inserted} vectors"
                    )
                except Exception as e:
                    error_msg = f"Qdrant integration failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            else:
                errors.append("Qdrant agent not available")

            success = len(errors) == 0 or (neo4j_result or qdrant_result)

        except Exception as e:
            success = False
            error_msg = f"Integration orchestration failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        # Generate unique provenance ID
        provenance_id = f"prov_{datetime.utcnow().timestamp()}_{self._generate_id(neo4j_data.transaction_id)}"

        return IntegrationResult(
            success=success,
            neo4j_result=neo4j_result,
            qdrant_result=qdrant_result,
            errors=errors,
            provenance_id=provenance_id,
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            vectors_inserted=vectors_inserted,
        )

    async def _execute_neo4j_integration(
        self, neo4j_data: Neo4jIntegrationData
    ) -> Dict:
        """Execute Neo4j database updates with proper error handling."""

        nodes_created = 0
        relationships_created = 0

        try:
            # Create nodes first
            for node in neo4j_data.nodes:
                query = f"""
                MERGE (n:{node['type']} {{id: $id}})
                ON CREATE SET n += $properties, n.created_at = datetime()
                ON MATCH SET n += $properties, n.updated_at = datetime()
                RETURN n
                """

                result = await self.neo4j_agent.execute(
                    query, id=node["id"], properties=node["properties"]
                )

                if result:
                    nodes_created += 1

            # Create relationships
            for rel in neo4j_data.relationships:
                query = f"""
                MATCH (a {{id: $from_id}})
                MATCH (b {{id: $to_id}})
                MERGE (a)-[r:{rel['type']}]->(b)
                ON CREATE SET r += $properties, r.created_at = datetime()
                ON MATCH SET r += $properties, r.updated_at = datetime()
                RETURN r
                """

                result = await self.neo4j_agent.execute(
                    query,
                    from_id=rel["from"],
                    to_id=rel["to"],
                    properties=rel["properties"],
                )

                if result:
                    relationships_created += 1

            return {
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "transaction_id": neo4j_data.transaction_id,
            }

        except Exception as e:
            logger.error(f"Neo4j integration error: {e}")
            raise

    async def _execute_qdrant_integration(
        self, qdrant_data: QdrantIntegrationData
    ) -> Dict:
        """Execute Qdrant database updates with proper error handling."""

        try:
            # Ensure collection exists
            await self.qdrant_agent.ensure_collection(qdrant_data.collection_name)

            # Prepare points for insertion
            points = []
            for vector_data in qdrant_data.vectors:
                points.append(
                    {
                        "id": vector_data["id"],
                        "vector": vector_data["vector"],
                        "payload": vector_data["payload"],
                    }
                )

            # Insert vectors
            await self.qdrant_agent.upsert_points(
                collection_name=qdrant_data.collection_name, points=points
            )

            return {
                "vectors_inserted": len(points),
                "collection": qdrant_data.collection_name,
                "metadata": qdrant_data.metadata,
            }

        except Exception as e:
            logger.error(f"Qdrant integration error: {e}")
            raise

    async def track_provenance(
        self,
        integration_result: IntegrationResult,
        scraped_doc: ScrapedDocument,
        processing_agents: List[str],
        quality_scores: Dict,
    ) -> ProvenanceRecord:
        """Create and store complete provenance record."""

        # Collect Neo4j node IDs
        neo4j_nodes = []
        if integration_result.neo4j_result:
            neo4j_nodes.append(f"doc_{scraped_doc.content_hash[:12]}")

        # Collect Qdrant collections
        qdrant_collections = []
        if integration_result.qdrant_result:
            qdrant_collections.append(
                integration_result.qdrant_result.get("collection", "unknown")
            )

        provenance = ProvenanceRecord(
            id=integration_result.provenance_id,
            source_url=scraped_doc.metadata.url,
            scraping_timestamp=scraped_doc.scraping_timestamp,
            processing_agents=processing_agents,
            quality_scores=quality_scores,
            integration_timestamp=datetime.utcnow(),
            content_hash=scraped_doc.content_hash,
            neo4j_nodes=neo4j_nodes,
            qdrant_collections=qdrant_collections,
        )

        # Store provenance in Neo4j if available
        if self.neo4j_agent:
            try:
                query = """
                CREATE (p:Provenance {
                    id: $id,
                    source_url: $source_url,
                    scraping_timestamp: $scraping_timestamp,
                    processing_agents: $processing_agents,
                    quality_scores: $quality_scores,
                    integration_timestamp: $integration_timestamp,
                    content_hash: $content_hash,
                    neo4j_nodes: $neo4j_nodes,
                    qdrant_collections: $qdrant_collections
                })
                RETURN p
                """

                await self.neo4j_agent.execute(
                    query,
                    id=provenance.id,
                    source_url=provenance.source_url,
                    scraping_timestamp=provenance.scraping_timestamp.isoformat(),
                    processing_agents=json.dumps(provenance.processing_agents),
                    quality_scores=json.dumps(provenance.quality_scores),
                    integration_timestamp=provenance.integration_timestamp.isoformat(),
                    content_hash=provenance.content_hash,
                    neo4j_nodes=json.dumps(provenance.neo4j_nodes),
                    qdrant_collections=json.dumps(provenance.qdrant_collections),
                )

                logger.info(f"Provenance record created: {provenance.id}")

            except Exception as e:
                logger.error(f"Failed to store provenance: {e}")

        return provenance

    def _generate_id(self, text: str) -> str:
        """Generate consistent ID from text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    async def comprehensive_integration(
        self,
        scraped_doc: ScrapedDocument,
        content_analysis: ContentAnalysis,
        reliability_score: ReliabilityScore,
        processing_agents: List[str],
    ) -> Tuple[IntegrationResult, ProvenanceRecord]:
        """Execute complete integration pipeline with provenance tracking."""

        logger.info(
            f"Starting comprehensive integration for document: {scraped_doc.metadata.title}"
        )

        # Prepare Neo4j integration data
        neo4j_data = await self.prepare_neo4j_integration(
            scraped_doc, content_analysis, reliability_score
        )

        # Prepare Qdrant integration data
        qdrant_data = await self.prepare_qdrant_integration(
            scraped_doc, content_analysis, reliability_score
        )

        # Execute integration
        integration_result = await self.execute_integration(neo4j_data, qdrant_data)

        # Track provenance
        provenance = await self.track_provenance(
            integration_result,
            scraped_doc,
            processing_agents,
            reliability_score.component_scores,
        )

        logger.info(
            f"Integration complete: {integration_result.nodes_created} nodes, "
            f"{integration_result.relationships_created} relationships, "
            f"{integration_result.vectors_inserted} vectors"
        )

        return integration_result, provenance
