#!/usr/bin/env python3
"""
Schema Manager for Neo4j Agent
Handles Yggdrasil schema enforcement, validation, and optimization
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from neo4j import AsyncDriver, AsyncSession
from .neo4j_agent import NodeData

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages Neo4j schema for Yggdrasil knowledge structure
    Enforces node types, relationships, and indexes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Define Yggdrasil node types
        self.node_types = {
            "Document": {
                "required_properties": ["id", "title", "domain"],
                "optional_properties": ["timestamp", "content_hash", "author", "language"],
                "indexes": ["id", "domain", "timestamp"]
            },
            "Entity": {
                "required_properties": ["name", "type", "domain"],
                "optional_properties": ["confidence", "description", "aliases"],
                "indexes": ["name", "type", "domain"]
            },
            "Concept": {
                "required_properties": ["id", "name", "domain"],
                "optional_properties": ["description", "type", "level", "earliest_evidence_date"],
                "indexes": ["id", "name", "domain", "type"]
            },
            "Claim": {
                "required_properties": ["id", "text"],
                "optional_properties": ["confidence", "verified", "domain", "source"],
                "indexes": ["id", "domain", "verified"]
            },
            "Author": {
                "required_properties": ["name"],
                "optional_properties": ["period", "domain", "birth_date", "death_date"],
                "indexes": ["name", "period", "domain"]
            },
            "Source": {
                "required_properties": ["id", "type"],
                "optional_properties": ["title", "url", "publication_date", "domain"],
                "indexes": ["id", "type", "domain"]
            }
        }
        
        # Define relationship types
        self.relationship_types = {
            "CONTAINS": {
                "description": "Document contains Entity/Concept",
                "source_types": ["Document"],
                "target_types": ["Entity", "Concept", "Claim"]
            },
            "MENTIONS": {
                "description": "Cross-references between entities",
                "source_types": ["Entity", "Concept", "Document"],
                "target_types": ["Entity", "Concept", "Document"]
            },
            "SIMILAR_TO": {
                "description": "Semantic similarity",
                "source_types": ["Entity", "Concept", "Claim"],
                "target_types": ["Entity", "Concept", "Claim"],
                "properties": ["similarity_score", "algorithm"]
            },
            "TEMPORAL_BEFORE": {
                "description": "Temporal precedence",
                "source_types": ["Document", "Entity", "Concept"],
                "target_types": ["Document", "Entity", "Concept"],
                "properties": ["time_difference", "confidence"]
            },
            "TEMPORAL_AFTER": {
                "description": "Temporal succession",
                "source_types": ["Document", "Entity", "Concept"],
                "target_types": ["Document", "Entity", "Concept"],
                "properties": ["time_difference", "confidence"]
            },
            "INFLUENCES": {
                "description": "Knowledge evolution",
                "source_types": ["Concept", "Entity", "Author"],
                "target_types": ["Concept", "Entity", "Author"],
                "properties": ["influence_strength", "evidence"]
            },
            "VALIDATES": {
                "description": "Supporting evidence",
                "source_types": ["Document", "Source"],
                "target_types": ["Claim", "Concept"],
                "properties": ["validation_strength", "methodology"]
            },
            "CONTRADICTS": {
                "description": "Conflicting evidence",
                "source_types": ["Document", "Source", "Claim"],
                "target_types": ["Claim", "Concept"],
                "properties": ["contradiction_strength", "explanation"]
            },
            "AUTHORED_BY": {
                "description": "Authorship relationship",
                "source_types": ["Document", "Concept", "Claim"],
                "target_types": ["Author"],
                "properties": ["confidence", "attribution_type"]
            },
            "RELATES_TO": {
                "description": "General relationship",
                "source_types": ["Concept"],
                "target_types": ["Concept"],
                "properties": ["relationship_type", "strength", "domain"]
            }
        }
        
        # Domain-specific constraints
        self.domains = [
            "Art", "Language", "Mathematics", "Philosophy", 
            "Science", "Technology", "Religion", "Astrology"
        ]
    
    async def initialize_schema(self, driver: AsyncDriver) -> bool:
        """Initialize the complete Yggdrasil schema"""
        try:
            async with driver.session() as session:
                # Create constraints
                await self._create_constraints(session)
                
                # Create indexes
                await self._create_indexes(session)
                
                # Create domain nodes if they don't exist
                await self._create_domain_nodes(session)
                
                logger.info("Schema initialization completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False
    
    async def _create_constraints(self, session: AsyncSession):
        """Create unique constraints for key properties"""
        constraints = [
            # Unique constraints for IDs
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.id IS UNIQUE",
            "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            
            # Node key constraints for important combinations
            "CREATE CONSTRAINT entity_name_domain IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.domain) IS NODE KEY",
            "CREATE CONSTRAINT author_name_period IF NOT EXISTS FOR (a:Author) REQUIRE (a.name, a.period) IS NODE KEY",
            
            # Property existence constraints
            "CREATE CONSTRAINT concept_required_props IF NOT EXISTS FOR (c:Concept) REQUIRE (c.id, c.name, c.domain) IS NOT NULL",
            "CREATE CONSTRAINT document_required_props IF NOT EXISTS FOR (d:Document) REQUIRE (d.id, d.title, d.domain) IS NOT NULL",
        ]
        
        for constraint in constraints:
            try:
                await session.run(constraint)
                logger.debug(f"Created constraint: {constraint}")
            except Exception as e:
                # Constraint might already exist
                logger.debug(f"Constraint creation skipped: {e}")
    
    async def _create_indexes(self, session: AsyncSession):
        """Create performance indexes"""
        indexes = [
            # Domain-based indexes
            "CREATE INDEX concept_domain_idx IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX document_domain_idx IF NOT EXISTS FOR (d:Document) ON (d.domain)",
            "CREATE INDEX entity_domain_idx IF NOT EXISTS FOR (e:Entity) ON (e.domain)",
            
            # Temporal indexes for Yggdrasil structure
            "CREATE INDEX document_timestamp_idx IF NOT EXISTS FOR (d:Document) ON (d.timestamp)",
            "CREATE INDEX concept_evidence_date_idx IF NOT EXISTS FOR (c:Concept) ON (c.earliest_evidence_date)",
            "CREATE INDEX author_period_idx IF NOT EXISTS FOR (a:Author) ON (a.period)",
            
            # Type-based indexes
            "CREATE INDEX concept_type_idx IF NOT EXISTS FOR (c:Concept) ON (c.type)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX concept_level_idx IF NOT EXISTS FOR (c:Concept) ON (c.level)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX concept_search_idx IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description]",
            "CREATE FULLTEXT INDEX document_search_idx IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]",
            "CREATE FULLTEXT INDEX claim_search_idx IF NOT EXISTS FOR (cl:Claim) ON EACH [cl.text]",
            
            # Verification status indexes
            "CREATE INDEX claim_verified_idx IF NOT EXISTS FOR (cl:Claim) ON (cl.verified)",
            "CREATE INDEX concept_status_idx IF NOT EXISTS FOR (c:Concept) ON (c.research_status)",
        ]
        
        for index in indexes:
            try:
                await session.run(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")
    
    async def _create_domain_nodes(self, session: AsyncSession):
        """Create domain hierarchy nodes"""
        # Create primary domain nodes
        for domain in self.domains:
            query = """
            MERGE (d:Domain {name: $domain})
            SET d.created_at = COALESCE(d.created_at, $timestamp),
                d.type = 'primary_domain',
                d.description = $description
            """
            
            descriptions = {
                "Art": "Visual arts, architecture, performing arts, and creative expression",
                "Language": "Linguistics, communication, and language studies",
                "Mathematics": "Pure and applied mathematics, logic, and quantitative analysis",
                "Philosophy": "Metaphysics, ethics, logic, and philosophical inquiry",
                "Science": "Natural sciences, physics, chemistry, biology, and scientific method",
                "Technology": "Tools, techniques, and technological innovation",
                "Religion": "Religious beliefs, practices, and spiritual traditions",
                "Astrology": "Celestial influence beliefs and astrological traditions"
            }
            
            await session.run(
                query,
                domain=domain,
                timestamp=datetime.now().isoformat(),
                description=descriptions.get(domain, f"{domain} domain")
            )
        
        # Create subdomain relationships
        subdomain_relationships = [
            ("Philosophy", "Religion", "contains_subdomain"),
            ("Science", "Astrology", "contains_pseudoscience")
        ]
        
        for parent, child, rel_type in subdomain_relationships:
            query = """
            MATCH (parent:Domain {name: $parent}), (child:Domain {name: $child})
            MERGE (parent)-[r:CONTAINS_SUBDOMAIN]->(child)
            SET r.relationship_type = $rel_type,
                r.created_at = COALESCE(r.created_at, $timestamp)
            """
            
            await session.run(
                query,
                parent=parent,
                child=child,
                rel_type=rel_type,
                timestamp=datetime.now().isoformat()
            )
    
    async def validate_node(self, node_data: NodeData) -> bool:
        """Validate node data against schema"""
        try:
            node_type = node_data.node_type
            properties = node_data.properties
            
            # Check if node type is defined
            if node_type not in self.node_types:
                logger.warning(f"Unknown node type: {node_type}")
                return False
            
            schema = self.node_types[node_type]
            
            # Check required properties
            required_props = schema["required_properties"]
            missing_props = [prop for prop in required_props if prop not in properties]
            
            if missing_props:
                logger.warning(f"Missing required properties for {node_type}: {missing_props}")
                return False
            
            # Validate domain if specified
            if "domain" in properties:
                domain = properties["domain"]
                if domain not in self.domains:
                    logger.warning(f"Invalid domain: {domain}")
                    return False
            
            # Validate specific node types
            if node_type == "Concept":
                return self._validate_concept(properties)
            elif node_type == "Document":
                return self._validate_document(properties)
            elif node_type == "Claim":
                return self._validate_claim(properties)
            
            return True
            
        except Exception as e:
            logger.error(f"Node validation error: {e}")
            return False
    
    def _validate_concept(self, properties: Dict[str, Any]) -> bool:
        """Validate Concept node specific properties"""
        # Check ID format
        concept_id = properties.get("id", "")
        if not self._validate_concept_id(concept_id):
            logger.warning(f"Invalid concept ID format: {concept_id}")
            return False
        
        # Check level if specified
        level = properties.get("level")
        if level is not None:
            try:
                level_int = int(level)
                if level_int < 1 or level_int > 4:
                    logger.warning(f"Invalid concept level: {level}")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"Concept level must be integer: {level}")
                return False
        
        return True
    
    def _validate_document(self, properties: Dict[str, Any]) -> bool:
        """Validate Document node specific properties"""
        # Check timestamp format if provided
        timestamp = properties.get("timestamp")
        if timestamp:
            try:
                # Try to parse as ISO format
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid timestamp format: {timestamp}")
                return False
        
        return True
    
    def _validate_claim(self, properties: Dict[str, Any]) -> bool:
        """Validate Claim node specific properties"""
        # Check confidence if provided
        confidence = properties.get("confidence")
        if confidence is not None:
            try:
                conf_float = float(confidence)
                if conf_float < 0.0 or conf_float > 1.0:
                    logger.warning(f"Confidence must be between 0.0 and 1.0: {confidence}")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"Confidence must be float: {confidence}")
                return False
        
        # Check verified status
        verified = properties.get("verified")
        if verified is not None and not isinstance(verified, bool):
            logger.warning(f"Verified must be boolean: {verified}")
            return False
        
        return True
    
    def _validate_concept_id(self, concept_id: str) -> bool:
        """Validate concept ID format (DOMAIN####)"""
        if not concept_id or len(concept_id) < 7:
            return False
        
        # Extract domain prefix and numeric part
        domain_prefixes = {
            "ART": "Art",
            "LANG": "Language", 
            "MATH": "Mathematics",
            "PHIL": "Philosophy",
            "SCI": "Science",
            "TECH": "Technology",
            "RELIG": "Religion",
            "ASTRO": "Astrology"
        }
        
        for prefix in domain_prefixes:
            if concept_id.startswith(prefix):
                numeric_part = concept_id[len(prefix):]
                try:
                    int(numeric_part)
                    return len(numeric_part) == 4
                except ValueError:
                    return False
        
        return False
    
    async def validate_relationship(
        self, 
        relationship_type: str, 
        source_type: str, 
        target_type: str
    ) -> bool:
        """Validate relationship between node types"""
        if relationship_type not in self.relationship_types:
            logger.warning(f"Unknown relationship type: {relationship_type}")
            return False
        
        rel_schema = self.relationship_types[relationship_type]
        
        # Check if source and target types are allowed
        allowed_sources = rel_schema["source_types"]
        allowed_targets = rel_schema["target_types"]
        
        if source_type not in allowed_sources:
            logger.warning(
                f"Invalid source type {source_type} for relationship {relationship_type}"
            )
            return False
        
        if target_type not in allowed_targets:
            logger.warning(
                f"Invalid target type {target_type} for relationship {relationship_type}"
            )
            return False
        
        return True
    
    def get_node_schema(self, node_type: str) -> Optional[Dict[str, Any]]:
        """Get schema definition for a node type"""
        return self.node_types.get(node_type)
    
    def get_relationship_schema(self, relationship_type: str) -> Optional[Dict[str, Any]]:
        """Get schema definition for a relationship type"""
        return self.relationship_types.get(relationship_type)
    
    def get_all_domains(self) -> List[str]:
        """Get list of all valid domains"""
        return self.domains.copy()
    
    def get_yggdrasil_structure(self) -> Dict[str, Any]:
        """Get the Yggdrasil tree structure definition"""
        return {
            "levels": {
                1: {
                    "name": "Ancient Knowledge (Trunk)",
                    "description": "Foundational knowledge from ancient sources",
                    "time_range": "Before 1000 BCE",
                    "examples": ["Sumerian tablets", "Egyptian hieroglyphs", "Vedic texts"]
                },
                2: {
                    "name": "Classical Knowledge (Main Branches)",
                    "description": "Classical and early philosophical knowledge",
                    "time_range": "1000 BCE - 500 CE",
                    "examples": ["Greek philosophy", "Roman law", "Early Christianity"]
                },
                3: {
                    "name": "Medieval Knowledge (Branches)",
                    "description": "Medieval and renaissance knowledge",
                    "time_range": "500 - 1500 CE",
                    "examples": ["Scholasticism", "Islamic philosophy", "Renaissance art"]
                },
                4: {
                    "name": "Modern Knowledge (Leaves)",
                    "description": "Modern and contemporary knowledge",
                    "time_range": "1500 CE - Present",
                    "examples": ["Scientific revolution", "Enlightenment", "Modern technology"]
                }
            },
            "domains": {domain: {"primary": True} for domain in self.domains[:6]},
            "subdomains": {
                "Religion": {"parent": "Philosophy", "type": "subdomain"},
                "Astrology": {"parent": "Science", "type": "pseudoscience"}
            }
        }