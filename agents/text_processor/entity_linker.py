#!/usr/bin/env python3
"""
Entity Linker for Knowledge Graph Integration
Links extracted entities to knowledge graph nodes with caching
"""

import logging
from typing import List, Optional

from agents.text_processor.models import LinkedEntity
from agents.text_processor.text_processor import Entity

logger = logging.getLogger(__name__)


class EntityLinker:
    """Link entities to knowledge graph"""

    def __init__(self, neo4j_agent=None):
        self.neo4j_agent = neo4j_agent
        self.entity_cache = {}

    async def link_entities(
        self, entities: List[Entity], domain: str = "general"
    ) -> List[LinkedEntity]:
        """Link entities to knowledge graph nodes"""
        linked_entities = []

        for entity in entities:
            # Check cache first
            cache_key = f"{entity.text}:{entity.label}"
            if cache_key in self.entity_cache:
                linked_entities.append(self.entity_cache[cache_key])
                continue

            # Query knowledge graph
            linked = await self._find_in_knowledge_graph(entity, domain)
            if linked:
                self.entity_cache[cache_key] = linked
                linked_entities.append(linked)
            else:
                # Create unlinked entity
                linked = LinkedEntity(
                    text=entity.text,
                    label=entity.label,
                    kb_id=None,
                    kb_type=None,
                    confidence=0.0,
                    properties={},
                )
                linked_entities.append(linked)

        return linked_entities

    async def _find_in_knowledge_graph(
        self, entity: Entity, domain: str
    ) -> Optional[LinkedEntity]:
        """Find entity in knowledge graph"""
        if not self.neo4j_agent:
            return None

        # Build query based on entity type
        if entity.label in ["PERSON", "PER"]:
            query = """
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($name)
            RETURN p.id as kb_id, 'Person' as kb_type, p as properties
            LIMIT 1
            """
        elif entity.label in ["ORG", "ORGANIZATION"]:
            query = """
            MATCH (o:Organization)
            WHERE toLower(o.name) CONTAINS toLower($name)
            RETURN o.id as kb_id, 'Organization' as kb_type, o as properties
            LIMIT 1
            """
        elif entity.label in ["LOC", "LOCATION", "GPE"]:
            query = """
            MATCH (l:Location)
            WHERE toLower(l.name) CONTAINS toLower($name)
            RETURN l.id as kb_id, 'Location' as kb_type, l as properties
            LIMIT 1
            """
        else:
            # Generic concept search
            query = """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.id as kb_id, 'Concept' as kb_type, c as properties
            LIMIT 1
            """

        try:
            result = await self.neo4j_agent.query(query, {"name": entity.text})
            if result and len(result) > 0:
                return LinkedEntity(
                    text=entity.text,
                    label=entity.label,
                    kb_id=result[0]["kb_id"],
                    kb_type=result[0]["kb_type"],
                    confidence=0.8,  # Base confidence
                    properties=dict(result[0]["properties"]),
                )
        except Exception as e:
            logger.error(f"Error linking entity {entity.text}: {e}")

        return None
