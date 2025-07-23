#!/usr/bin/env python3
"""
MCP Server Knowledge Graph Builder Agent
Creates and manages the Yggdrasil-shaped knowledge graph in Neo4j
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, date
import json
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

import yaml
import neo4j
from neo4j import GraphDatabase, Transaction, Session, Result
from neo4j.exceptions import ServiceUnavailable, TransientError
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Base class for graph nodes"""
    id: str
    label: str
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            **self.properties
        }


@dataclass
class DocumentNode(GraphNode):
    """Document node in the knowledge graph"""
    title: str
    author: Optional[str]
    content: str
    domain: str
    subcategory: str
    date: Optional[str]
    era: Optional[str]  # Ancient, Classical, Medieval, Renaissance, Modern, Contemporary
    source: str
    language: str
    word_count: int
    yggdrasil_level: int  # 0 = root, higher = closer to leaves
    
    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = self.generate_id()
        self.label = 'Document'
        self.properties = {
            'title': self.title,
            'author': self.author,
            'content': self.content[:1000],  # Store excerpt
            'domain': self.domain,
            'subcategory': self.subcategory,
            'date': self.date,
            'era': self.era,
            'source': self.source,
            'language': self.language,
            'word_count': self.word_count,
            'yggdrasil_level': self.yggdrasil_level
        }
    
    def generate_id(self) -> str:
        """Generate unique ID for document"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
        return f"doc_{self.domain}_{content_hash}"


@dataclass
class PersonNode(GraphNode):
    """Person node in the knowledge graph"""
    name: str
    birth_date: Optional[str]
    death_date: Optional[str]
    era: Optional[str]
    occupation: Optional[str]
    nationality: Optional[str]
    domains: List[str]
    
    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = self.generate_id()
        self.label = 'Person'
        self.properties = {
            'name': self.name,
            'birth_date': self.birth_date,
            'death_date': self.death_date,
            'era': self.era,
            'occupation': self.occupation,
            'nationality': self.nationality,
            'domains': self.domains
        }
    
    def generate_id(self) -> str:
        """Generate unique ID for person"""
        name_hash = hashlib.sha256(self.name.encode()).hexdigest()[:12]
        return f"person_{name_hash}"


@dataclass
class ConceptNode(GraphNode):
    """Concept node in the knowledge graph"""
    name: str
    description: str
    domain: str
    concept_type: str  # theorem, principle, idea, doctrine, etc.
    related_terms: List[str]
    
    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = self.generate_id()
        self.label = 'Concept'
        self.properties = {
            'name': self.name,
            'description': self.description,
            'domain': self.domain,
            'concept_type': self.concept_type,
            'related_terms': self.related_terms
        }
    
    def generate_id(self) -> str:
        """Generate unique ID for concept"""
        concept_hash = hashlib.sha256(f"{self.name}_{self.domain}".encode()).hexdigest()[:12]
        return f"concept_{concept_hash}"


@dataclass
class EventNode(GraphNode):
    """Event node in the knowledge graph"""
    name: str
    description: str
    event_date: Optional[str]
    location: Optional[str]
    event_type: str
    significance: str
    domains: List[str]
    
    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = self.generate_id()
        self.label = 'Event'
        self.properties = {
            'name': self.name,
            'description': self.description,
            'event_date': self.event_date,
            'location': self.location,
            'event_type': self.event_type,
            'significance': self.significance,
            'domains': self.domains
        }
    
    def generate_id(self) -> str:
        """Generate unique ID for event"""
        event_hash = hashlib.sha256(f"{self.name}_{self.event_date}".encode()).hexdigest()[:12]
        return f"event_{event_hash}"


@dataclass
class PatternNode(GraphNode):
    """Pattern node for cross-domain connections"""
    name: str
    description: str
    pattern_type: str
    domains: List[str]
    confidence: float
    examples: List[str]
    validated: bool = False
    
    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = self.generate_id()
        self.label = 'Pattern'
        self.properties = {
            'name': self.name,
            'description': self.description,
            'pattern_type': self.pattern_type,
            'domains': self.domains,
            'confidence': self.confidence,
            'examples': self.examples,
            'validated': self.validated
        }
    
    def generate_id(self) -> str:
        """Generate unique ID for pattern"""
        pattern_hash = hashlib.sha256(f"{self.name}_{self.pattern_type}".encode()).hexdigest()[:12]
        return f"pattern_{pattern_hash}"


@dataclass
class GraphRelationship:
    """Relationship between nodes"""
    from_node: str
    to_node: str
    relationship_type: str
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j"""
        return {
            'created_at': self.created_at.isoformat(),
            **self.properties
        }


class YggdrasilStructure:
    """Manages the Yggdrasil tree structure for temporal organization"""
    
    DOMAINS = ['math', 'science', 'religion', 'history', 'literature', 'philosophy']
    
    ERA_HIERARCHY = {
        'ancient': 0,       # Before 500 CE
        'classical': 1,     # 500-1000 CE
        'medieval': 2,      # 1000-1400 CE
        'renaissance': 3,   # 1400-1600 CE
        'enlightenment': 4, # 1600-1800 CE
        'modern': 5,        # 1800-1950 CE
        'contemporary': 6   # 1950-present
    }
    
    @classmethod
    def determine_era(cls, date_str: Optional[str]) -> str:
        """Determine era from date string"""
        if not date_str:
            return 'unknown'
        
        try:
            # Extract year from various date formats
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = int(year_match.group())
            else:
                # Try other patterns
                year_match = re.search(r'\b\d{1,4}\b', date_str)
                if year_match:
                    year = int(year_match.group())
                else:
                    return 'unknown'
            
            if year < 500:
                return 'ancient'
            elif year < 1000:
                return 'classical'
            elif year < 1400:
                return 'medieval'
            elif year < 1600:
                return 'renaissance'
            elif year < 1800:
                return 'enlightenment'
            elif year < 1950:
                return 'modern'
            else:
                return 'contemporary'
                
        except (ValueError, AttributeError):
            return 'unknown'
    
    @classmethod
    def calculate_yggdrasil_level(cls, era: str, domain: str) -> int:
        """Calculate position in Yggdrasil tree (0 = trunk, higher = leaves)"""
        base_level = cls.ERA_HIERARCHY.get(era, 3)  # Default to renaissance level
        
        # Domain-specific adjustments
        domain_adjustments = {
            'religion': -1,     # Religious texts often older
            'philosophy': -1,   # Philosophy has ancient roots
            'history': 0,       # Neutral
            'math': 1,          # Mathematical concepts build over time
            'science': 2,       # Modern science relatively recent
            'literature': 0     # Literature spans all eras
        }
        
        adjustment = domain_adjustments.get(domain, 0)
        return max(0, base_level + adjustment)


class Neo4jManager:
    """Manages Neo4j database connections and operations"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Dict = None) -> Result:
        """Execute a single query"""
        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {})
    
    def execute_transaction(self, tx_func, *args, **kwargs) -> Any:
        """Execute a transaction"""
        with self.driver.session(database=self.database) as session:
            return session.execute_write(tx_func, *args, **kwargs)
    
    def execute_batch(self, queries: List[Tuple[str, Dict]]) -> List[Result]:
        """Execute multiple queries in a transaction"""
        def batch_tx(tx: Transaction) -> List[Result]:
            results = []
            for query, params in queries:
                results.append(tx.run(query, params))
            return results
        
        return self.execute_transaction(batch_tx)


class GraphBuilder:
    """Main knowledge graph builder"""
    
    def __init__(self, config_path: str = "agents/knowledge_graph/config.yaml"):
        """Initialize graph builder"""
        self.load_config(config_path)
        self.neo4j = Neo4jManager(
            self.config['neo4j']['uri'],
            self.config['neo4j']['user'],
            self.config['neo4j']['password'],
            self.config['neo4j']['database']
        )
        self.yggdrasil = YggdrasilStructure()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Initialize schema
        self.setup_schema()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'user': 'neo4j',
                    'password': 'password',
                    'database': 'neo4j'
                },
                'batch_size': 1000,
                'max_workers': 4,
                'enable_constraints': True,
                'enable_indexes': True
            }
    
    def setup_schema(self) -> None:
        """Set up Neo4j schema with constraints and indexes"""
        logger.info("Setting up Neo4j schema...")
        
        # Constraints for uniqueness
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (pt:Pattern) REQUIRE pt.id IS UNIQUE",
        ]
        
        # Indexes for performance
        indexes = [
            "CREATE INDEX document_domain_date IF NOT EXISTS FOR (d:Document) ON (d.domain, d.date)",
            "CREATE INDEX document_era_level IF NOT EXISTS FOR (d:Document) ON (d.era, d.yggdrasil_level)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.event_date)",
            "CREATE INDEX pattern_domains IF NOT EXISTS FOR (pt:Pattern) ON (pt.domains)",
            "CREATE FULLTEXT INDEX document_content IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]",
            "CREATE FULLTEXT INDEX concept_text IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description]"
        ]
        
        # Execute constraints
        if self.config.get('enable_constraints', True):
            for constraint in constraints:
                try:
                    self.neo4j.execute_query(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation failed (may already exist): {e}")
        
        # Execute indexes
        if self.config.get('enable_indexes', True):
            for index in indexes:
                try:
                    self.neo4j.execute_query(index)
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
        
        logger.info("Schema setup completed")
    
    def create_root_structure(self) -> None:
        """Create the root Yggdrasil structure"""
        logger.info("Creating Yggdrasil root structure...")
        
        # Create World root node
        world_query = """
        MERGE (w:World {id: 'world_root', name: 'World Tree Root'})
        SET w.created_at = datetime()
        RETURN w
        """
        self.neo4j.execute_query(world_query)
        
        # Create domain nodes
        for domain in self.yggdrasil.DOMAINS:
            domain_query = """
            MERGE (d:Domain {id: $domain_id, name: $domain_name, domain: $domain})
            SET d.created_at = datetime()
            
            WITH d
            MATCH (w:World {id: 'world_root'})
            MERGE (w)-[:HAS_DOMAIN]->(d)
            RETURN d
            """
            
            self.neo4j.execute_query(domain_query, {
                'domain_id': f"domain_{domain}",
                'domain_name': domain.capitalize(),
                'domain': domain
            })
        
        logger.info("Root structure created")
    
    def add_document(self, processed_doc: Dict) -> Optional[str]:
        """Add a document to the knowledge graph"""
        try:
            # Determine era and Yggdrasil level
            era = self.yggdrasil.determine_era(processed_doc.get('date'))
            yggdrasil_level = self.yggdrasil.calculate_yggdrasil_level(
                era, processed_doc.get('domain', 'unknown')
            )
            
            # Create document node
            doc_node = DocumentNode(
                id=processed_doc.get('doc_id', ''),
                title=processed_doc.get('title', ''),
                author=processed_doc.get('author'),
                content=processed_doc.get('cleaned_content', ''),
                domain=processed_doc.get('domain', ''),
                subcategory=processed_doc.get('subcategory', ''),
                date=processed_doc.get('date'),
                era=era,
                source=processed_doc.get('source', ''),
                language=processed_doc.get('language', ''),
                word_count=processed_doc.get('word_count', 0),
                yggdrasil_level=yggdrasil_level,
                created_at=datetime.now()
            )
            
            # Insert document
            doc_query = """
            CREATE (d:Document $properties)
            WITH d
            MATCH (domain:Domain {domain: $domain})
            MERGE (domain)-[:CONTAINS]->(d)
            RETURN d.id as doc_id
            """
            
            result = self.neo4j.execute_query(doc_query, {
                'properties': doc_node.to_dict(),
                'domain': doc_node.domain
            })
            
            doc_id = result.single()['doc_id'] if result.single() else None
            
            if doc_id:
                # Add entities and relationships
                self._add_document_entities(doc_id, processed_doc.get('entities', []))
                logger.info(f"Added document: {doc_id}")
                
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return None
    
    def _add_document_entities(self, doc_id: str, entities: List[Dict]) -> None:
        """Add entities from document and create relationships"""
        for entity in entities:
            entity_type = entity.get('label', '')
            entity_text = entity.get('text', '')
            
            if entity_type == 'PERSON':
                self._add_person_entity(doc_id, entity_text)
            elif entity_type in ['ORG', 'WORK_OF_ART', 'LAW']:
                self._add_concept_entity(doc_id, entity_text, entity_type)
            elif entity_type == 'EVENT':
                self._add_event_entity(doc_id, entity_text)
    
    def _add_person_entity(self, doc_id: str, person_name: str) -> None:
        """Add person entity and relationship"""
        person_query = """
        MERGE (p:Person {name: $name})
        ON CREATE SET p.id = randomUUID(), p.created_at = datetime()
        
        WITH p
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:MENTIONS_PERSON]->(p)
        """
        
        self.neo4j.execute_query(person_query, {
            'name': person_name,
            'doc_id': doc_id
        })
    
    def _add_concept_entity(self, doc_id: str, concept_name: str, concept_type: str) -> None:
        """Add concept entity and relationship"""
        concept_query = """
        MERGE (c:Concept {name: $name})
        ON CREATE SET 
            c.id = randomUUID(), 
            c.created_at = datetime(),
            c.concept_type = $concept_type
        
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:CONTAINS_CONCEPT]->(c)
        """
        
        self.neo4j.execute_query(concept_query, {
            'name': concept_name,
            'concept_type': concept_type,
            'doc_id': doc_id
        })
    
    def _add_event_entity(self, doc_id: str, event_name: str) -> None:
        """Add event entity and relationship"""
        event_query = """
        MERGE (e:Event {name: $name})
        ON CREATE SET e.id = randomUUID(), e.created_at = datetime()
        
        WITH e
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:REFERENCES_EVENT]->(e)
        """
        
        self.neo4j.execute_query(event_query, {
            'name': event_name,
            'doc_id': doc_id
        })
    
    def create_temporal_relationships(self) -> None:
        """Create DERIVED_FROM relationships based on temporal order"""
        logger.info("Creating temporal relationships...")
        
        # For each domain, connect documents in temporal order
        for domain in self.yggdrasil.DOMAINS:
            temporal_query = """
            MATCH (d1:Document {domain: $domain})
            MATCH (d2:Document {domain: $domain})
            WHERE d1.yggdrasil_level > d2.yggdrasil_level
            AND NOT (d1)-[:DERIVED_FROM]->(d2)
            
            // Only connect if there's a reasonable temporal gap and similarity
            WITH d1, d2
            WHERE d1.yggdrasil_level - d2.yggdrasil_level <= 2
            
            CREATE (d1)-[:DERIVED_FROM {
                created_at: datetime(),
                temporal_distance: d1.yggdrasil_level - d2.yggdrasil_level
            }]->(d2)
            """
            
            self.neo4j.execute_query(temporal_query, {'domain': domain})
        
        logger.info("Temporal relationships created")
    
    def create_concept_relationships(self) -> None:
        """Create relationships between similar concepts"""
        logger.info("Creating concept relationships...")
        
        similarity_query = """
        MATCH (c1:Concept)
        MATCH (c2:Concept)
        WHERE c1 <> c2
        AND NOT (c1)-[:SIMILAR_CONCEPT]-(c2)
        
        // Use text similarity for now (in production, use embeddings)
        WITH c1, c2, 
             apoc.text.jaroWinkler(c1.name, c2.name) as name_similarity,
             apoc.text.jaroWinkler(c1.description, c2.description) as desc_similarity
        WHERE name_similarity > 0.7 OR desc_similarity > 0.7
        
        CREATE (c1)-[:SIMILAR_CONCEPT {
            created_at: datetime(),
            name_similarity: name_similarity,
            description_similarity: desc_similarity
        }]->(c2)
        """
        
        try:
            self.neo4j.execute_query(similarity_query)
        except Exception as e:
            logger.warning(f"Concept similarity requires APOC plugin: {e}")
            # Fallback to simpler matching
            self._create_concept_relationships_simple()
    
    def _create_concept_relationships_simple(self) -> None:
        """Simple concept relationship creation without APOC"""
        simple_query = """
        MATCH (c1:Concept)
        MATCH (c2:Concept)
        WHERE c1 <> c2
        AND c1.name CONTAINS c2.name OR c2.name CONTAINS c1.name
        AND NOT (c1)-[:SIMILAR_CONCEPT]-(c2)
        
        CREATE (c1)-[:SIMILAR_CONCEPT {
            created_at: datetime(),
            similarity_type: 'name_containment'
        }]->(c2)
        """
        
        self.neo4j.execute_query(simple_query)
    
    def add_cross_domain_pattern(self, pattern: PatternNode) -> Optional[str]:
        """Add a cross-domain pattern to the graph"""
        try:
            pattern_query = """
            CREATE (p:Pattern $properties)
            RETURN p.id as pattern_id
            """
            
            result = self.neo4j.execute_query(pattern_query, {
                'properties': pattern.to_dict()
            })
            
            pattern_id = result.single()['pattern_id'] if result.single() else None
            
            if pattern_id:
                # Connect pattern to relevant domains
                for domain in pattern.domains:
                    domain_query = """
                    MATCH (p:Pattern {id: $pattern_id})
                    MATCH (d:Domain {domain: $domain})
                    MERGE (p)-[:SPANS_DOMAIN]->(d)
                    """
                    self.neo4j.execute_query(domain_query, {
                        'pattern_id': pattern_id,
                        'domain': domain
                    })
                
                logger.info(f"Added cross-domain pattern: {pattern_id}")
                
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error adding pattern: {e}")
            return None
    
    def get_yggdrasil_structure(self, domain: Optional[str] = None) -> Dict:
        """Get the Yggdrasil tree structure for visualization"""
        if domain:
            query = """
            MATCH (d:Domain {domain: $domain})-[:CONTAINS]->(doc:Document)
            OPTIONAL MATCH (doc)-[r:DERIVED_FROM]->(older:Document)
            RETURN doc, r, older
            ORDER BY doc.yggdrasil_level ASC
            """
            params = {'domain': domain}
        else:
            query = """
            MATCH (w:World)-[:HAS_DOMAIN]->(d:Domain)-[:CONTAINS]->(doc:Document)
            OPTIONAL MATCH (doc)-[r:DERIVED_FROM]->(older:Document)
            RETURN d.domain as domain, doc, r, older
            ORDER BY d.domain, doc.yggdrasil_level ASC
            """
            params = {}
        
        result = self.neo4j.execute_query(query, params)
        
        # Process results into tree structure
        tree = {'nodes': [], 'relationships': []}
        
        for record in result:
            doc = record['doc']
            tree['nodes'].append({
                'id': doc['id'],
                'title': doc['title'],
                'domain': doc.get('domain'),
                'era': doc.get('era'),
                'yggdrasil_level': doc.get('yggdrasil_level', 0)
            })
            
            if record.get('r') and record.get('older'):
                tree['relationships'].append({
                    'from': doc['id'],
                    'to': record['older']['id'],
                    'type': 'DERIVED_FROM'
                })
        
        return tree
    
    def search_graph(self, query: str, limit: int = 10) -> List[Dict]:
        """Search the knowledge graph"""
        search_query = """
        CALL db.index.fulltext.queryNodes("document_content", $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j.execute_query(search_query, {
                'query': query,
                'limit': limit
            })
            
            return [
                {
                    'node': dict(record['node']),
                    'score': record['score']
                }
                for record in result
            ]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        stats_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        
        UNION ALL
        
        MATCH ()-[r]->()
        RETURN type(r) as label, count(r) as count
        """
        
        result = self.neo4j.execute_query(stats_query)
        stats = {}
        
        for record in result:
            stats[record['label']] = record['count']
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'neo4j'):
            self.neo4j.close()


async def main():
    """Example usage of graph builder"""
    # Example processed document
    processed_doc = {
        'doc_id': 'sample_doc_123',
        'title': 'Sample Philosophical Text',
        'author': 'Ancient Philosopher',
        'cleaned_content': 'This is a sample philosophical text discussing ethics and metaphysics...',
        'domain': 'philosophy',
        'subcategory': 'ethics',
        'date': '350 BCE',
        'source': 'Ancient Library',
        'language': 'greek',
        'word_count': 1500,
        'entities': [
            {'text': 'Aristotle', 'label': 'PERSON'},
            {'text': 'Ethics', 'label': 'ORG'},
            {'text': 'Metaphysics', 'label': 'WORK_OF_ART'}
        ]
    }
    
    builder = GraphBuilder()
    
    try:
        # Setup
        builder.create_root_structure()
        
        # Add document
        doc_id = builder.add_document(processed_doc)
        print(f"Added document: {doc_id}")
        
        # Create relationships
        builder.create_temporal_relationships()
        builder.create_concept_relationships()
        
        # Get statistics
        stats = builder.get_statistics()
        print(f"Graph statistics: {stats}")
        
    finally:
        builder.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
