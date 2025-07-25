"""
Yggdrasil data processor for hierarchical tree visualization.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..core.models import (
    NodeType,
    VisualizationData,
    VisualizationEdge,
    VisualizationNode,
)
from .data_processor import DataProcessor


class YggdrasilProcessor(DataProcessor):
    """Processor for Yggdrasil tree structure data."""

    async def get_data(
        self, domain_filter: Optional[str] = None, max_depth: int = 5
    ) -> VisualizationData:
        """Get data structured for Yggdrasil visualization."""

        async with self.neo4j_driver.session() as session:
            # Build query for Yggdrasil structure
            domain_clause = "WHERE n.domain = $domain" if domain_filter else ""

            query = f"""
            // Get root and domain nodes
            MATCH (root:Root)
            OPTIONAL MATCH (root)-[:HAS_DOMAIN]->(domain:Domain) {domain_clause}
            
            // Get documents and their relationships with depth limit
            OPTIONAL MATCH path = (domain)-[:CONTAINS*1..{max_depth}]->(doc:Document)
            WITH root, domain, doc, path
            
            // Get all relationships
            OPTIONAL MATCH (doc)-[r]-(related)
            
            RETURN 
                root,
                collect(DISTINCT domain) as domains,
                collect(DISTINCT doc) as documents,
                collect(DISTINCT {{source: startNode(r), target: endNode(r), relationship: r}}) as relationships,
                collect(DISTINCT path) as paths
            """

            params = {"domain": domain_filter} if domain_filter else {}
            result = await session.run(query, params)
            record = await result.single()

            if not record:
                return VisualizationData([], [], {}, "hierarchical", {})

            nodes = []
            edges = []
            node_ids = set()

            # Add root node
            if record["root"]:
                root_node = VisualizationNode(
                    id="root",
                    label="World Knowledge",
                    title="Root of all knowledge domains",
                    node_type=NodeType.ROOT,
                    domain=None,
                    date=None,
                    level=0,
                )
                nodes.append(root_node)
                node_ids.add("root")

            # Add domain nodes
            for i, domain in enumerate(record["domains"] or []):
                if domain:
                    domain_id = f"domain_{domain['name']}"
                    domain_node = VisualizationNode(
                        id=domain_id,
                        label=domain["name"],
                        title=domain.get("description", f"Domain: {domain['name']}"),
                        node_type=NodeType.DOMAIN,
                        domain=domain["name"],
                        date=None,
                        level=1,
                    )
                    nodes.append(domain_node)
                    node_ids.add(domain_id)

                    # Add edge from root to domain
                    if "root" in node_ids:
                        edge = VisualizationEdge(
                            id=f"root_to_{domain_id}",
                            source="root",
                            target=domain_id,
                            relationship_type="HAS_DOMAIN",
                        )
                        edges.append(edge)

            # Add document nodes
            for doc in record["documents"] or []:
                if doc:
                    doc_id = str(doc.id)
                    if doc_id not in node_ids:
                        # Determine node type based on labels
                        node_type = self._determine_node_type(doc.labels)

                        # Calculate level based on date (newer = higher level)
                        level = self._calculate_temporal_level(doc.get("date"))

                        doc_node = VisualizationNode(
                            id=doc_id,
                            label=doc.get("title", "Unknown"),
                            title=doc.get("title", "Unknown"),
                            node_type=node_type,
                            domain=doc.get("domain"),
                            date=doc.get("date"),
                            level=level,
                            metadata={
                                "author": doc.get("author"),
                                "source": doc.get("source"),
                                "word_count": doc.get("word_count"),
                            },
                        )
                        nodes.append(doc_node)
                        node_ids.add(doc_id)

            # Add relationship edges
            for rel_data in record["relationships"] or []:
                if rel_data and rel_data["source"] and rel_data["target"]:
                    source_id = str(rel_data["source"].id)
                    target_id = str(rel_data["target"].id)

                    if source_id in node_ids and target_id in node_ids:
                        relationship = rel_data["relationship"]
                        edge = VisualizationEdge(
                            id=f"{source_id}_to_{target_id}",
                            source=source_id,
                            target=target_id,
                            relationship_type=relationship.type,
                            weight=relationship.get("weight", 1.0),
                        )
                        edges.append(edge)

            # Limit nodes and edges if necessary
            nodes, edges = self._limit_data(nodes, edges)

            # Generate metadata
            metadata = self._generate_metadata(
                nodes,
                edges,
                additional_metadata={
                    "domain_filter": domain_filter,
                    "max_depth": max_depth,
                },
            )

            return VisualizationData(
                nodes=nodes,
                edges=edges,
                metadata=metadata,
                layout_type="hierarchical",
                filters={"domain": domain_filter, "max_depth": max_depth},
            )
