"""
Base data processor for visualization data extraction.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

from neo4j import AsyncDriver
from ..core.models import VisualizationData, VisualizationNode, VisualizationEdge, NodeType
from ..core.config import VisualizationConfig


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: VisualizationConfig, neo4j_driver: AsyncDriver):
        self.config = config
        self.neo4j_driver = neo4j_driver
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def get_data(self, **kwargs) -> VisualizationData:
        """Get visualization data from Neo4j."""
        pass
    
    def _calculate_temporal_level(self, date_str: Optional[str]) -> int:
        """Calculate level based on temporal distance (recent = higher level)."""
        if not date_str:
            return 5  # Default level for undated items
        
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            current_year = datetime.now().year
            doc_year = date.year
            
            # More recent documents get higher levels (closer to leaves)
            years_ago = current_year - doc_year
            
            if years_ago < 50:
                return 6  # Very recent
            elif years_ago < 200:
                return 5  # Recent
            elif years_ago < 500:
                return 4  # Historical
            elif years_ago < 2000:
                return 3  # Ancient
            else:
                return 2  # Very ancient
                
        except (ValueError, TypeError):
            return 5  # Default level
    
    def _determine_node_type(self, node_labels: List[str]) -> NodeType:
        """Determine node type based on Neo4j labels."""
        node_type = NodeType.DOCUMENT
        
        for label in node_labels:
            if label.lower() == 'person':
                node_type = NodeType.PERSON
                break
            elif label.lower() == 'concept':
                node_type = NodeType.CONCEPT
                break
            elif label.lower() == 'event':
                node_type = NodeType.EVENT
                break
            elif label.lower() == 'pattern':
                node_type = NodeType.PATTERN
                break
            elif label.lower() == 'root':
                node_type = NodeType.ROOT
                break
            elif label.lower() == 'domain':
                node_type = NodeType.DOMAIN
                break
        
        return node_type
    
    def _create_node(
        self, 
        node_data: Dict[str, Any], 
        node_id: str, 
        level: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VisualizationNode:
        """Create a VisualizationNode from Neo4j data."""
        node_type = self._determine_node_type(node_data.get('labels', []))
        
        return VisualizationNode(
            id=node_id,
            label=node_data.get('title', node_data.get('name', f'Node {node_id}')),
            title=node_data.get('title', node_data.get('name', 'Unknown')),
            node_type=node_type,
            domain=node_data.get('domain'),
            date=node_data.get('date'),
            level=level,
            metadata=metadata or {
                'author': node_data.get('author'),
                'source': node_data.get('source'),
                'word_count': node_data.get('word_count')
            }
        )
    
    def _create_edge(
        self, 
        edge_id: str, 
        source_id: str, 
        target_id: str, 
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VisualizationEdge:
        """Create a VisualizationEdge."""
        return VisualizationEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            relationship_type=relationship_type,
            weight=weight,
            metadata=metadata
        )
    
    def _limit_data(
        self, 
        nodes: List[VisualizationNode], 
        edges: List[VisualizationEdge]
    ) -> tuple[List[VisualizationNode], List[VisualizationEdge]]:
        """Limit nodes and edges based on configuration."""
        # Limit nodes
        if len(nodes) > self.config.max_nodes:
            nodes = nodes[:self.config.max_nodes]
            node_ids = {node.id for node in nodes}
            edges = [edge for edge in edges if edge.source in node_ids and edge.target in node_ids]
        
        # Limit edges
        if len(edges) > self.config.max_edges:
            edges = edges[:self.config.max_edges]
        
        return nodes, edges
    
    def _generate_metadata(
        self, 
        nodes: List[VisualizationNode], 
        edges: List[VisualizationEdge],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate metadata for visualization."""
        metadata = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'domains': list(set(node.domain for node in nodes if node.domain)),
            'node_types': list(set(node.node_type.value for node in nodes)),
            'generated_at': datetime.now().isoformat()
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata