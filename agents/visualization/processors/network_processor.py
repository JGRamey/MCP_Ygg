"""
Network data processor for general network graph visualization.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from ..core.models import VisualizationData, VisualizationNode, VisualizationEdge, NodeType
from .data_processor import DataProcessor


class NetworkProcessor(DataProcessor):
    """Processor for general network graph data."""
    
    async def get_data(
        self, 
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> VisualizationData:
        """Get data for network graph visualization."""
        
        async with self.neo4j_driver.session() as session:
            # Build dynamic query based on filters
            node_filter = ""
            if node_types:
                node_labels = " OR ".join([f"n:{nt}" for nt in node_types])
                node_filter = f"WHERE {node_labels}"
            
            rel_filter = ""
            if relationship_types:
                rel_types = " OR ".join([f"type(r) = '{rt}'" for rt in relationship_types])
                rel_filter = f"AND ({rel_types})"
            
            query = f"""
            MATCH (n) {node_filter}
            OPTIONAL MATCH (n)-[r]-(m)
            {rel_filter if rel_filter else ""}
            RETURN 
                collect(DISTINCT n) as nodes,
                collect(DISTINCT {{source: n, target: m, relationship: r}}) as relationships
            LIMIT {self.config.max_nodes}
            """
            
            result = await session.run(query)
            record = await result.single()
            
            nodes = []
            edges = []
            node_ids = set()
            
            # Process nodes
            for node_data in record['nodes'] or []:
                node_id = str(node_data.id)
                if node_id not in node_ids:
                    # Determine node type
                    node_type = self._determine_node_type(node_data.labels)
                    
                    viz_node = VisualizationNode(
                        id=node_id,
                        label=node_data.get('title', f'Node {node_id}'),
                        title=node_data.get('title', 'Unknown'),
                        node_type=node_type,
                        domain=node_data.get('domain'),
                        date=node_data.get('date'),
                        level=0  # Will be calculated in layout
                    )
                    nodes.append(viz_node)
                    node_ids.add(node_id)
            
            # Process relationships
            for rel_data in record['relationships'] or []:
                if (rel_data and rel_data['source'] and rel_data['target'] and 
                    rel_data['relationship']):
                    
                    source_id = str(rel_data['source'].id)
                    target_id = str(rel_data['target'].id)
                    
                    if source_id in node_ids and target_id in node_ids:
                        edge = VisualizationEdge(
                            id=f"{source_id}_to_{target_id}",
                            source=source_id,
                            target=target_id,
                            relationship_type=rel_data['relationship'].type,
                            weight=rel_data['relationship'].get('weight', 1.0)
                        )
                        edges.append(edge)
            
            # Limit nodes and edges if necessary
            nodes, edges = self._limit_data(nodes, edges)
            
            # Generate metadata
            metadata = self._generate_metadata(
                nodes, 
                edges,
                additional_metadata={
                    'node_types_filter': node_types,
                    'relationship_types_filter': relationship_types,
                    'relationship_types': list(set(edge.relationship_type for edge in edges))
                }
            )
            
            return VisualizationData(
                nodes=nodes,
                edges=edges,
                metadata=metadata,
                layout_type="force",
                filters={'node_types': node_types, 'relationship_types': relationship_types}
            )