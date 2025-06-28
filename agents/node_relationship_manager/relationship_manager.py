#!/usr/bin/env python3
"""
MCP Server Node Relationship Manager Agent
Manages Neo4j node relationships based on user input and web-crawled data with user authorization
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
import hashlib
from urllib.parse import urljoin, urlparse
import time
from collections import defaultdict, Counter

import yaml
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from neo4j import GraphDatabase
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RelationshipProposal:
    """Proposed relationship between nodes"""
    proposal_id: str
    from_node_id: str
    to_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float
    evidence: List[str]
    source_type: str  # 'user_input', 'web_crawl', 'semantic_analysis'
    proposed_at: datetime
    approved: Optional[bool] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    notes: Optional[str] = None


@dataclass
class NodeInfo:
    """Information about a graph node"""
    node_id: str
    labels: List[str]
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]


@dataclass
class RelationshipPattern:
    """Pattern for detecting relationships"""
    pattern_name: str
    from_labels: List[str]
    to_labels: List[str]
    relationship_types: List[str]
    text_patterns: List[str]
    confidence_threshold: float
    requires_approval: bool


class SemanticRelationshipDetector:
    """Detects semantic relationships between nodes"""
    
    def __init__(self):
        """Initialize semantic detector"""
        self.embedding_model = None
        self.load_embedding_model()
        self.relationship_patterns = self._load_relationship_patterns()
    
    def load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model for relationship detection")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def _load_relationship_patterns(self) -> List[RelationshipPattern]:
        """Load relationship detection patterns"""
        return [
            RelationshipPattern(
                pattern_name="author_work",
                from_labels=["Person"],
                to_labels=["Document", "Work"],
                relationship_types=["AUTHORED", "WROTE", "CREATED"],
                text_patterns=[
                    r"written by",
                    r"authored by",
                    r"by ([A-Z][a-z]+ [A-Z][a-z]+)",
                    r"([A-Z][a-z]+ [A-Z][a-z]+) wrote",
                    r"([A-Z][a-z]+ [A-Z][a-z]+)'s work"
                ],
                confidence_threshold=0.7,
                requires_approval=False
            ),
            RelationshipPattern(
                pattern_name="influenced_by",
                from_labels=["Person", "Document", "Concept"],
                to_labels=["Person", "Document", "Concept"],
                relationship_types=["INFLUENCED_BY", "INSPIRED_BY", "DERIVED_FROM"],
                text_patterns=[
                    r"influenced by",
                    r"inspired by",
                    r"based on",
                    r"derived from",
                    r"following ([A-Z][a-z]+ [A-Z][a-z]+)",
                    r"in the tradition of"
                ],
                confidence_threshold=0.6,
                requires_approval=True
            ),
            RelationshipPattern(
                pattern_name="mentions_person",
                from_labels=["Document"],
                to_labels=["Person"],
                relationship_types=["MENTIONS", "DISCUSSES", "REFERENCES"],
                text_patterns=[
                    r"mentions ([A-Z][a-z]+ [A-Z][a-z]+)",
                    r"discusses ([A-Z][a-z]+ [A-Z][a-z]+)",
                    r"according to ([A-Z][a-z]+ [A-Z][a-z]+)",
                    r"([A-Z][a-z]+ [A-Z][a-z]+) argues",
                    r"([A-Z][a-z]+ [A-Z][a-z]+) claims"
                ],
                confidence_threshold=0.8,
                requires_approval=False
            ),
            RelationshipPattern(
                pattern_name="contains_concept",
                from_labels=["Document"],
                to_labels=["Concept"],
                relationship_types=["CONTAINS_CONCEPT", "DISCUSSES_CONCEPT", "EXPLORES"],
                text_patterns=[
                    r"discusses the concept of",
                    r"explores the idea of",
                    r"examines.*concept",
                    r"focuses on.*idea"
                ],
                confidence_threshold=0.7,
                requires_approval=False
            ),
            RelationshipPattern(
                pattern_name="temporal_sequence",
                from_labels=["Document", "Event"],
                to_labels=["Document", "Event"],
                relationship_types=["PRECEDED_BY", "FOLLOWED_BY", "CONTEMPORARY_WITH"],
                text_patterns=[
                    r"before",
                    r"after",
                    r"during the same period",
                    r"contemporary with",
                    r"earlier than",
                    r"later than"
                ],
                confidence_threshold=0.6,
                requires_approval=True
            ),
            RelationshipPattern(
                pattern_name="similar_concept",
                from_labels=["Concept"],
                to_labels=["Concept"],
                relationship_types=["SIMILAR_TO", "RELATED_TO", "ANALOGOUS_TO"],
                text_patterns=[
                    r"similar to",
                    r"analogous to",
                    r"related to",
                    r"comparable to",
                    r"like",
                    r"resembles"
                ],
                confidence_threshold=0.5,
                requires_approval=True
            )
        ]
    
    def detect_semantic_relationships(self, nodes: List[NodeInfo], content: str) -> List[RelationshipProposal]:
        """Detect semantic relationships between nodes based on content"""
        proposals = []
        
        if not self.embedding_model:
            return proposals
        
        try:
            # Create embeddings for node properties
            node_texts = []
            for node in nodes:
                text = f"{node.properties.get('name', '')} {node.properties.get('title', '')} {node.properties.get('description', '')}"
                node_texts.append(text.strip())
            
            if not node_texts:
                return proposals
            
            node_embeddings = self.embedding_model.encode(node_texts)
            content_embedding = self.embedding_model.encode([content])[0]
            
            # Calculate similarities
            similarities = cosine_similarity([content_embedding], node_embeddings)[0]
            
            # Find highly similar node pairs
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    # Calculate semantic similarity between nodes
                    node_similarity = cosine_similarity([node_embeddings[i]], [node_embeddings[j]])[0][0]
                    
                    if node_similarity > 0.7:  # High similarity threshold
                        # Check if nodes have compatible labels for relationships
                        compatible_patterns = self._find_compatible_patterns(node1, node2)
                        
                        for pattern in compatible_patterns:
                            # Check if content supports the relationship
                            content_support = self._check_content_support(content, pattern)
                            
                            if content_support:
                                confidence = (node_similarity + content_support) / 2
                                
                                proposal = RelationshipProposal(
                                    proposal_id=f"semantic_{hashlib.md5(f'{node1.node_id}_{node2.node_id}_{pattern.pattern_name}'.encode()).hexdigest()[:12]}",
                                    from_node_id=node1.node_id,
                                    to_node_id=node2.node_id,
                                    relationship_type=pattern.relationship_types[0],
                                    properties={
                                        'semantic_similarity': node_similarity,
                                        'content_support': content_support,
                                        'detection_method': 'semantic_analysis'
                                    },
                                    confidence=confidence,
                                    evidence=[f"Semantic similarity: {node_similarity:.3f}", "Content analysis"],
                                    source_type='semantic_analysis',
                                    proposed_at=datetime.now()
                                )
                                proposals.append(proposal)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Semantic relationship detection failed: {e}")
            return []
    
    def _find_compatible_patterns(self, node1: NodeInfo, node2: NodeInfo) -> List[RelationshipPattern]:
        """Find relationship patterns compatible with two nodes"""
        compatible = []
        
        for pattern in self.relationship_patterns:
            # Check if node labels match pattern requirements
            node1_matches = any(label in pattern.from_labels for label in node1.labels)
            node2_matches = any(label in pattern.to_labels for label in node2.labels)
            
            if node1_matches and node2_matches:
                compatible.append(pattern)
            
            # Also check reverse direction
            node1_matches_to = any(label in pattern.to_labels for label in node1.labels)
            node2_matches_from = any(label in pattern.from_labels for label in node2.labels)
            
            if node1_matches_to and node2_matches_from:
                compatible.append(pattern)
        
        return compatible
    
    def _check_content_support(self, content: str, pattern: RelationshipPattern) -> float:
        """Check how well content supports a relationship pattern"""
        content_lower = content.lower()
        support_score = 0.0
        
        for text_pattern in pattern.text_patterns:
            matches = len(re.findall(text_pattern, content_lower, re.IGNORECASE))
            if matches > 0:
                support_score += matches * 0.2
        
        return min(1.0, support_score)


class WebCrawlRelationshipDetector:
    """Detects relationships by crawling web pages"""
    
    def __init__(self):
        """Initialize web crawler"""
        self.session = None
        self.relationship_extractors = self._load_relationship_extractors()
    
    def _load_relationship_extractors(self) -> Dict[str, List[str]]:
        """Load web-based relationship extraction patterns"""
        return {
            'wikipedia_infobox': [
                r'influenced\s*=\s*(.+)',
                r'influences\s*=\s*(.+)',
                r'student\s*=\s*(.+)',
                r'teacher\s*=\s*(.+)',
                r'predecessor\s*=\s*(.+)',
                r'successor\s*=\s*(.+)'
            ],
            'academic_citations': [
                r'cited\s+by\s+(.+)',
                r'references\s+(.+)',
                r'builds\s+on\s+(.+)',
                r'extends\s+(.+)'
            ],
            'biographical_relations': [
                r'student\s+of\s+(.+)',
                r'teacher\s+of\s+(.+)',
                r'contemporary\s+of\s+(.+)',
                r'influenced\s+by\s+(.+)',
                r'mentor\s+to\s+(.+)'
            ]
        }
    
    async def crawl_for_relationships(self, nodes: List[NodeInfo]) -> List[RelationshipProposal]:
        """Crawl web pages to find relationships between nodes"""
        proposals = []
        
        if not nodes:
            return proposals
        
        try:
            self.session = aiohttp.ClientSession()
            
            for node in nodes:
                # Generate search queries for the node
                search_queries = self._generate_search_queries(node)
                
                for query in search_queries:
                    try:
                        # Search for web pages about this node
                        urls = await self._search_web(query)
                        
                        # Crawl each URL
                        for url in urls[:3]:  # Limit to first 3 results
                            try:
                                relationships = await self._extract_relationships_from_url(url, node, nodes)
                                proposals.extend(relationships)
                                
                                # Rate limiting
                                await asyncio.sleep(1)
                                
                            except Exception as e:
                                logger.warning(f"Failed to crawl {url}: {e}")
                    
                    except Exception as e:
                        logger.warning(f"Search failed for query '{query}': {e}")
            
            return proposals
            
        except Exception as e:
            logger.error(f"Web crawling for relationships failed: {e}")
            return []
        
        finally:
            if self.session:
                await self.session.close()
    
    def _generate_search_queries(self, node: NodeInfo) -> List[str]:
        """Generate search queries for a node"""
        queries = []
        
        name = node.properties.get('name', '')
        title = node.properties.get('title', '')
        
        if name:
            queries.append(f'"{name}" biography')
            queries.append(f'"{name}" influences')
            queries.append(f'"{name}" students')
            queries.append(f'"{name}" influenced by')
        
        if title:
            queries.append(f'"{title}" author')
            queries.append(f'"{title}" references')
            queries.append(f'"{title}" citations')
        
        return queries[:5]  # Limit queries
    
    async def _search_web(self, query: str) -> List[str]:
        """Search web for URLs (simplified - would use real search API)"""
        # This is a placeholder - in production, use Google Custom Search API, Bing API, etc.
        # For now, return some example URLs
        return [
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            f"https://plato.stanford.edu/entries/{query.lower().replace(' ', '-')}/"
        ]
    
    async def _extract_relationships_from_url(self, url: str, target_node: NodeInfo, all_nodes: List[NodeInfo]) -> List[RelationshipProposal]:
        """Extract relationships from a web page"""
        proposals = []
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    return proposals
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                
                # Extract relationships using patterns
                for extractor_type, patterns in self.relationship_extractors.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        
                        for match in matches:
                            related_entity = match.group(1).strip()
                            
                            # Try to match to existing nodes
                            matching_node = self._find_matching_node(related_entity, all_nodes)
                            
                            if matching_node and matching_node.node_id != target_node.node_id:
                                relationship_type = self._infer_relationship_type(pattern, extractor_type)
                                
                                proposal = RelationshipProposal(
                                    proposal_id=f"web_{hashlib.md5(f'{target_node.node_id}_{matching_node.node_id}_{url}'.encode()).hexdigest()[:12]}",
                                    from_node_id=target_node.node_id,
                                    to_node_id=matching_node.node_id,
                                    relationship_type=relationship_type,
                                    properties={
                                        'source_url': url,
                                        'extraction_pattern': pattern,
                                        'extractor_type': extractor_type
                                    },
                                    confidence=0.6,  # Medium confidence for web-crawled data
                                    evidence=[f"Found on {url}", f"Pattern: {pattern}", f"Text: {match.group(0)}"],
                                    source_type='web_crawl',
                                    proposed_at=datetime.now()
                                )
                                proposals.append(proposal)
            
            return proposals
            
        except Exception as e:
            logger.warning(f"Failed to extract relationships from {url}: {e}")
            return []
    
    def _find_matching_node(self, entity_text: str, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """Find node that matches entity text"""
        entity_lower = entity_text.lower().strip()
        
        for node in nodes:
            name = node.properties.get('name', '').lower()
            title = node.properties.get('title', '').lower()
            
            # Check for exact match
            if entity_lower == name or entity_lower == title:
                return node
            
            # Check for partial match
            if entity_lower in name or name in entity_lower:
                return node
            
            if entity_lower in title or title in entity_lower:
                return node
        
        return None
    
    def _infer_relationship_type(self, pattern: str, extractor_type: str) -> str:
        """Infer relationship type from pattern and extractor"""
        pattern_lower = pattern.lower()
        
        if 'influenced' in pattern_lower:
            return 'INFLUENCED_BY'
        elif 'student' in pattern_lower:
            return 'STUDENT_OF'
        elif 'teacher' in pattern_lower:
            return 'TEACHER_OF'
        elif 'contemporary' in pattern_lower:
            return 'CONTEMPORARY_WITH'
        elif 'predecessor' in pattern_lower:
            return 'PRECEDED_BY'
        elif 'successor' in pattern_lower:
            return 'FOLLOWED_BY'
        elif 'mentor' in pattern_lower:
            return 'MENTORED_BY'
        elif 'cited' in pattern_lower or 'references' in pattern_lower:
            return 'REFERENCES'
        else:
            return 'RELATED_TO'


class UserRelationshipInterface:
    """Interface for user input of relationships"""
    
    def __init__(self):
        """Initialize user interface"""
        self.pending_user_inputs: List[RelationshipProposal] = []
    
    def add_user_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
        evidence: List[str] = None,
        user_id: str = None
    ) -> RelationshipProposal:
        """Add user-proposed relationship"""
        
        proposal = RelationshipProposal(
            proposal_id=f"user_{hashlib.md5(f'{from_node_id}_{to_node_id}_{relationship_type}_{time.time()}'.encode()).hexdigest()[:12]}",
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
            properties=properties or {},
            confidence=1.0,  # High confidence for user input
            evidence=evidence or ["User-provided relationship"],
            source_type='user_input',
            proposed_at=datetime.now(),
            reviewed_by=user_id
        )
        
        self.pending_user_inputs.append(proposal)
        logger.info(f"User relationship proposed: {from_node_id} -> {relationship_type} -> {to_node_id}")
        
        return proposal
    
    def get_pending_user_relationships(self) -> List[RelationshipProposal]:
        """Get pending user relationship proposals"""
        return self.pending_user_inputs.copy()
    
    def approve_user_relationship(self, proposal_id: str, reviewer: str) -> bool:
        """Approve a user relationship proposal"""
        for proposal in self.pending_user_inputs:
            if proposal.proposal_id == proposal_id:
                proposal.approved = True
                proposal.reviewed_by = reviewer
                proposal.reviewed_at = datetime.now()
                return True
        return False


class RelationshipManager:
    """Main relationship management system"""
    
    def __init__(self, config_path: str = "agents/node_relationship_manager/config.yaml"):
        """Initialize relationship manager"""
        self.load_config(config_path)
        
        # Initialize components
        self.semantic_detector = SemanticRelationshipDetector()
        self.web_detector = WebCrawlRelationshipDetector()
        self.user_interface = UserRelationshipInterface()
        
        # Neo4j connection
        self.neo4j_driver = None
        self.connect_neo4j()
        
        # Storage for proposals
        self.relationship_proposals: List[RelationshipProposal] = []
        self.load_existing_proposals()
    
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
                    'password': 'password'
                },
                'approval_requirements': {
                    'user_input': False,      # User input automatically approved
                    'semantic_analysis': True,
                    'web_crawl': True
                },
                'confidence_thresholds': {
                    'auto_approve': 0.9,
                    'manual_review': 0.7,
                    'reject': 0.3
                },
                'max_proposals_per_run': 100,
                'enable_web_crawling': True,
                'enable_semantic_analysis': True,
                'output_dir': 'data/relationships'
            }
    
    def connect_neo4j(self) -> None:
        """Connect to Neo4j database"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config['neo4j']['uri'],
                auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
            )
            
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Connected to Neo4j for relationship management")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def load_existing_proposals(self) -> None:
        """Load existing relationship proposals"""
        proposals_file = Path(self.config['output_dir']) / "relationship_proposals.json"
        
        if proposals_file.exists():
            try:
                with open(proposals_file, 'r') as f:
                    data = json.load(f)
                
                for proposal_data in data.get('proposals', []):
                    # Convert datetime strings back
                    proposal_data['proposed_at'] = datetime.fromisoformat(proposal_data['proposed_at'])
                    if proposal_data.get('reviewed_at'):
                        proposal_data['reviewed_at'] = datetime.fromisoformat(proposal_data['reviewed_at'])
                    
                    proposal = RelationshipProposal(**proposal_data)
                    self.relationship_proposals.append(proposal)
                
                logger.info(f"Loaded {len(self.relationship_proposals)} existing relationship proposals")
                
            except Exception as e:
                logger.error(f"Error loading existing proposals: {e}")
    
    def save_proposals(self) -> None:
        """Save relationship proposals"""
        proposals_file = Path(self.config['output_dir'])
        proposals_file.mkdir(parents=True, exist_ok=True)
        proposals_file = proposals_file / "relationship_proposals.json"
        
        try:
            proposals_data = []
            for proposal in self.relationship_proposals:
                proposal_dict = asdict(proposal)
                proposal_dict['proposed_at'] = proposal.proposed_at.isoformat()
                if proposal.reviewed_at:
                    proposal_dict['reviewed_at'] = proposal.reviewed_at.isoformat()
                proposals_data.append(proposal_dict)
            
            data = {
                'updated_at': datetime.now().isoformat(),
                'total_proposals': len(proposals_data),
                'proposals': proposals_data
            }
            
            with open(proposals_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving proposals: {e}")
    
    async def discover_relationships(self, limit_nodes: int = 100) -> List[RelationshipProposal]:
        """Discover new relationships using all methods"""
        logger.info("Starting relationship discovery")
        
        try:
            # Get nodes from Neo4j
            nodes = self._get_nodes_from_neo4j(limit_nodes)
            
            if not nodes:
                logger.warning("No nodes found in Neo4j")
                return []
            
            all_proposals = []
            
            # 1. Semantic analysis
            if self.config.get('enable_semantic_analysis', True):
                logger.info("Running semantic relationship detection")
                
                # Group nodes by domain for better semantic analysis
                domain_groups = self._group_nodes_by_domain(nodes)
                
                for domain, domain_nodes in domain_groups.items():
                    if len(domain_nodes) > 1:
                        # Get domain content for semantic analysis
                        domain_content = await self._get_domain_content(domain)
                        
                        semantic_proposals = self.semantic_detector.detect_semantic_relationships(
                            domain_nodes, domain_content
                        )
                        all_proposals.extend(semantic_proposals)
            
            # 2. Web crawling
            if self.config.get('enable_web_crawling', True):
                logger.info("Running web crawl relationship detection")
                
                # Limit web crawling to important nodes
                important_nodes = self._select_important_nodes(nodes, 10)
                web_proposals = await self.web_detector.crawl_for_relationships(important_nodes)
                all_proposals.extend(web_proposals)
            
            # 3. Add user proposals
            user_proposals = self.user_interface.get_pending_user_relationships()
            all_proposals.extend(user_proposals)
            
            # Filter and deduplicate proposals
            filtered_proposals = self._filter_and_deduplicate_proposals(all_proposals)
            
            # Add to our collection
            self.relationship_proposals.extend(filtered_proposals)
            self.save_proposals()
            
            logger.info(f"Discovered {len(filtered_proposals)} new relationship proposals")
            return filtered_proposals
            
        except Exception as e:
            logger.error(f"Relationship discovery failed: {e}")
            return []
    
    def _get_nodes_from_neo4j(self, limit: int) -> List[NodeInfo]:
        """Get nodes from Neo4j graph"""
        nodes = []
        
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.id IS NOT NULL
                RETURN n, labels(n) as node_labels
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                
                for record in result:
                    node = record['n']
                    labels = record['node_labels']
                    
                    # Get existing relationships
                    rel_query = """
                    MATCH (n {id: $node_id})-[r]-(connected)
                    RETURN type(r) as rel_type, connected.id as connected_id
                    LIMIT 10
                    """
                    
                    rel_result = session.run(rel_query, node_id=node['id'])
                    relationships = [dict(record) for record in rel_result]
                    
                    node_info = NodeInfo(
                        node_id=node['id'],
                        labels=labels,
                        properties=dict(node),
                        relationships=relationships
                    )
                    nodes.append(node_info)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error getting nodes from Neo4j: {e}")
            return []
    
    def _group_nodes_by_domain(self, nodes: List[NodeInfo]) -> Dict[str, List[NodeInfo]]:
        """Group nodes by domain for semantic analysis"""
        groups = defaultdict(list)
        
        for node in nodes:
            domain = node.properties.get('domain', 'unknown')
            groups[domain].append(node)
        
        return dict(groups)
    
    async def _get_domain_content(self, domain: str) -> str:
        """Get sample content from domain for semantic analysis"""
        # This would retrieve some representative content from the domain
        # For now, return a placeholder
        return f"Sample content for {domain} domain for semantic analysis"
    
    def _select_important_nodes(self, nodes: List[NodeInfo], limit: int) -> List[NodeInfo]:
        """Select most important nodes for web crawling"""
        # Score nodes by importance (number of existing relationships, properties, etc.)
        scored_nodes = []
        
        for node in nodes:
            score = 0
            
            # Score by number of relationships
            score += len(node.relationships) * 2
            
            # Score by completeness of properties
            score += len([v for v in node.properties.values() if v])
            
            # Score by node type
            if 'Person' in node.labels:
                score += 3
            elif 'Document' in node.labels:
                score += 2
            elif 'Concept' in node.labels:
                score += 1
            
            scored_nodes.append((score, node))
        
        # Sort by score and return top nodes
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for score, node in scored_nodes[:limit]]
    
    def _filter_and_deduplicate_proposals(self, proposals: List[RelationshipProposal]) -> List[RelationshipProposal]:
        """Filter and deduplicate relationship proposals"""
        filtered = []
        seen_relationships = set()
        
        for proposal in proposals:
            # Create unique key for relationship
            key = f"{proposal.from_node_id}:{proposal.to_node_id}:{proposal.relationship_type}"
            reverse_key = f"{proposal.to_node_id}:{proposal.from_node_id}:{proposal.relationship_type}"
            
            # Skip if we've seen this relationship
            if key in seen_relationships or reverse_key in seen_relationships:
                continue
            
            # Skip if confidence is too low
            if proposal.confidence < self.config['confidence_thresholds']['reject']:
                continue
            
            # Check if relationship already exists in Neo4j
            if self._relationship_exists(proposal.from_node_id, proposal.to_node_id, proposal.relationship_type):
                continue
            
            seen_relationships.add(key)
            filtered.append(proposal)
        
        return filtered
    
    def _relationship_exists(self, from_node_id: str, to_node_id: str, rel_type: str) -> bool:
        """Check if relationship already exists in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                query = f"""
                MATCH (from {{id: $from_id}})-[r:{rel_type}]->(to {{id: $to_id}})
                RETURN count(r) as count
                """
                
                result = session.run(query, from_id=from_node_id, to_id=to_node_id)
                count = result.single()['count']
                
                return count > 0
                
        except Exception as e:
            logger.error(f"Error checking relationship existence: {e}")
            return False
    
    def review_proposal(self, proposal_id: str, approved: bool, reviewer: str, notes: str = "") -> bool:
        """Review a relationship proposal"""
        for proposal in self.relationship_proposals:
            if proposal.proposal_id == proposal_id:
                proposal.approved = approved
                proposal.reviewed_by = reviewer
                proposal.reviewed_at = datetime.now()
                proposal.notes = notes
                
                self.save_proposals()
                
                if approved:
                    # Create the relationship in Neo4j
                    success = self._create_relationship_in_neo4j(proposal)
                    if success:
                        logger.info(f"Approved and created relationship: {proposal_id}")
                    else:
                        logger.error(f"Failed to create approved relationship: {proposal_id}")
                else:
                    logger.info(f"Rejected relationship proposal: {proposal_id}")
                
                return True
        
        return False
    
    def _create_relationship_in_neo4j(self, proposal: RelationshipProposal) -> bool:
        """Create approved relationship in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                # Dynamic query creation (simplified - in production, use parameterized queries)
                query = f"""
                MATCH (from {{id: $from_id}})
                MATCH (to {{id: $to_id}})
                CREATE (from)-[r:{proposal.relationship_type}]->(to)
                SET r += $properties
                SET r.created_by = $created_by
                SET r.created_at = datetime()
                SET r.source_type = $source_type
                SET r.confidence = $confidence
                RETURN r
                """
                
                result = session.run(query, {
                    'from_id': proposal.from_node_id,
                    'to_id': proposal.to_node_id,
                    'properties': proposal.properties,
                    'created_by': proposal.reviewed_by,
                    'source_type': proposal.source_type,
                    'confidence': proposal.confidence
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error creating relationship in Neo4j: {e}")
            return False
    
    def get_pending_proposals(self, source_type: Optional[str] = None) -> List[RelationshipProposal]:
        """Get pending relationship proposals"""
        pending = [p for p in self.relationship_proposals if p.approved is None]
        
        if source_type:
            pending = [p for p in pending if p.source_type == source_type]
        
        # Sort by confidence (highest first)
        pending.sort(key=lambda x: x.confidence, reverse=True)
        
        return pending
    
    def auto_approve_high_confidence_proposals(self) -> int:
        """Auto-approve proposals with very high confidence"""
        auto_approved = 0
        threshold = self.config['confidence_thresholds']['auto_approve']
        
        for proposal in self.relationship_proposals:
            if (proposal.approved is None and 
                proposal.confidence >= threshold and
                not self.config['approval_requirements'].get(proposal.source_type, True)):
                
                proposal.approved = True
                proposal.reviewed_by = "system_auto_approval"
                proposal.reviewed_at = datetime.now()
                proposal.notes = f"Auto-approved (confidence: {proposal.confidence:.3f})"
                
                # Create in Neo4j
                if self._create_relationship_in_neo4j(proposal):
                    auto_approved += 1
        
        if auto_approved > 0:
            self.save_proposals()
            logger.info(f"Auto-approved {auto_approved} high-confidence proposals")
        
        return auto_approved
    
    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get statistics about relationship proposals"""
        total = len(self.relationship_proposals)
        approved = len([p for p in self.relationship_proposals if p.approved == True])
        rejected = len([p for p in self.relationship_proposals if p.approved == False])
        pending = len([p for p in self.relationship_proposals if p.approved is None])
        
        source_stats = Counter(p.source_type for p in self.relationship_proposals)
        rel_type_stats = Counter(p.relationship_type for p in self.relationship_proposals)
        
        return {
            'total_proposals': total,
            'approved': approved,
            'rejected': rejected,
            'pending': pending,
            'approval_rate': approved / total if total > 0 else 0,
            'source_distribution': dict(source_stats),
            'relationship_type_distribution': dict(rel_type_stats),
            'average_confidence': sum(p.confidence for p in self.relationship_proposals) / total if total > 0 else 0
        }


async def main():
    """Example usage of relationship manager"""
    manager = RelationshipManager()
    
    try:
        # Discover new relationships
        proposals = await manager.discover_relationships(limit_nodes=20)
        print(f"Discovered {len(proposals)} relationship proposals")
        
        # Auto-approve high confidence proposals
        auto_approved = manager.auto_approve_high_confidence_proposals()
        print(f"Auto-approved {auto_approved} proposals")
        
        # Show pending proposals
        pending = manager.get_pending_proposals()
        print(f"\nPending proposals: {len(pending)}")
        
        for proposal in pending[:5]:  # Show first 5
            print(f"- {proposal.from_node_id} -> {proposal.relationship_type} -> {proposal.to_node_id}")
            print(f"  Confidence: {proposal.confidence:.3f}, Source: {proposal.source_type}")
            print(f"  Evidence: {'; '.join(proposal.evidence[:2])}")
        
        # Show statistics
        stats = manager.get_relationship_statistics()
        print(f"\n--- Statistics ---")
        print(f"Total proposals: {stats['total_proposals']}")
        print(f"Approval rate: {stats['approval_rate']:.2%}")
        print(f"Source distribution: {stats['source_distribution']}")
        
        # Example user relationship input
        user_proposal = manager.user_interface.add_user_relationship(
            from_node_id="person_plato",
            to_node_id="person_aristotle",
            relationship_type="TEACHER_OF",
            evidence=["Historical records show Plato taught Aristotle"],
            user_id="admin"
        )
        
        # Auto-approve user input (if configured)
        manager.review_proposal(user_proposal.proposal_id, True, "admin", "User-provided historical fact")
        
        print(f"\nAdded and approved user relationship: Plato -> TEACHER_OF -> Aristotle")
        
    except Exception as e:
        logger.error(f"Relationship management example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
