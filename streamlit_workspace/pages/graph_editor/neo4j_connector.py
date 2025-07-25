"""
Neo4j Connection and Data Management

This module handles Neo4j database connections, data retrieval,
and CSV fallback functionality for the Graph Editor.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from .models import DEMO_CONCEPTS, DataSource, Neo4jStatus


class Neo4jConnector:
    """Handles Neo4j connections and data operations"""

    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")

    def check_connection(self) -> Neo4jStatus:
        """Check if Neo4j is connected and accessible"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            driver.close()
            return Neo4jStatus(connected=True, message="Connected to Neo4j")
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                return Neo4jStatus(
                    connected=False, message="Neo4j is not running (connection refused)"
                )
            elif "ServiceUnavailable" in error_msg:
                return Neo4jStatus(
                    connected=False, message="Neo4j service is unavailable"
                )
            else:
                return Neo4jStatus(
                    connected=False, message=f"Connection error: {error_msg}"
                )

    def get_concepts(
        self, domain: Optional[str] = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get concepts from Neo4j database"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                if domain:
                    query = """
                    MATCH (c:Concept) 
                    WHERE c.domain = $domain 
                    RETURN c.id as id, c.name as name, c.domain as domain, 
                           c.type as type, c.level as level, c.description as description
                    LIMIT $limit
                    """
                    result = session.run(query, domain=domain, limit=limit)
                else:
                    query = """
                    MATCH (c:Concept) 
                    RETURN c.id as id, c.name as name, c.domain as domain,
                           c.type as type, c.level as level, c.description as description
                    LIMIT $limit
                    """
                    result = session.run(query, limit=limit)

                concepts = []
                for record in result:
                    concepts.append(
                        {
                            "id": record["id"],
                            "name": record["name"],
                            "domain": record["domain"],
                            "type": record["type"],
                            "level": record["level"],
                            "description": record["description"],
                        }
                    )

            driver.close()
            return concepts
        except Exception:
            return []

    def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific concept by ID"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                query = """
                MATCH (c:Concept) 
                WHERE c.id = $concept_id
                RETURN c.id as id, c.name as name, c.domain as domain,
                       c.type as type, c.level as level, c.description as description
                """
                result = session.run(query, concept_id=concept_id)
                record = result.single()

                if record:
                    return {
                        "id": record["id"],
                        "name": record["name"],
                        "domain": record["domain"],
                        "type": record["type"],
                        "level": record["level"],
                        "description": record["description"],
                    }
            driver.close()
        except Exception:
            pass
        return None

    def get_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific concept"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                query = """
                MATCH (c:Concept)-[r]->(target:Concept)
                WHERE c.id = $concept_id
                RETURN type(r) as type, target.id as target_id, target.name as target_name,
                       target.domain as target_domain, r.strength as strength, 'outgoing' as direction
                UNION
                MATCH (source:Concept)-[r]->(c:Concept)
                WHERE c.id = $concept_id
                RETURN type(r) as type, source.id as target_id, source.name as target_name,
                       source.domain as target_domain, r.strength as strength, 'incoming' as direction
                """
                result = session.run(query, concept_id=concept_id)

                relationships = []
                for record in result:
                    relationships.append(
                        {
                            "type": record["type"],
                            "target_id": record["target_id"],
                            "target_name": record["target_name"],
                            "target_domain": record["target_domain"],
                            "strength": record["strength"] or 0.5,
                            "direction": record["direction"],
                        }
                    )

            driver.close()
            return relationships
        except Exception:
            return []

    def get_domains(self) -> List[Dict[str, Any]]:
        """Get available domains from Neo4j"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                query = """
                MATCH (c:Concept)
                RETURN c.domain as domain, count(c) as concept_count
                ORDER BY domain
                """
                result = session.run(query)

                domains = []
                for record in result:
                    domains.append(
                        {
                            "domain": record["domain"],
                            "concept_count": record["concept_count"],
                        }
                    )

            driver.close()
            return domains
        except Exception:
            return []

    def get_cross_cultural_concepts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get concepts that appear across multiple domains (cross-cultural connections)"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                # Find concepts with similar names or descriptions across different domains
                query = """
                MATCH (c1:Concept), (c2:Concept)
                WHERE c1.domain <> c2.domain 
                  AND (c1.name = c2.name OR 
                       toLower(c1.name) CONTAINS toLower(c2.name) OR
                       toLower(c2.name) CONTAINS toLower(c1.name))
                RETURN DISTINCT c1.name as concept_name, 
                       collect(DISTINCT c1.domain) + collect(DISTINCT c2.domain) as domains,
                       c1.description as description1,
                       c2.description as description2
                ORDER BY size(collect(DISTINCT c1.domain) + collect(DISTINCT c2.domain)) DESC
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

                cross_cultural = []
                for record in result:
                    domains = list(set(record["domains"]))  # Remove duplicates
                    if len(domains) > 1:  # Only include multi-domain concepts
                        cross_cultural.append(
                            {
                                "concept_name": record["concept_name"],
                                "domains": domains,
                                "domain_count": len(domains),
                                "description1": record["description1"],
                                "description2": record["description2"],
                            }
                        )

            driver.close()
            return cross_cultural
        except Exception:
            return []

    def get_cross_domain_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get relationships that cross domain boundaries"""
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                query = """
                MATCH (c1:Concept)-[r]->(c2:Concept)
                WHERE c1.domain <> c2.domain
                RETURN c1.name as source_concept,
                       c1.domain as source_domain,
                       type(r) as relationship_type,
                       c2.name as target_concept,
                       c2.domain as target_domain,
                       c1.description as source_description,
                       c2.description as target_description
                ORDER BY c1.domain, c2.domain
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

                relationships = []
                for record in result:
                    relationships.append(
                        {
                            "source_concept": record["source_concept"],
                            "source_domain": record["source_domain"],
                            "relationship_type": record["relationship_type"],
                            "target_concept": record["target_concept"],
                            "target_domain": record["target_domain"],
                            "source_description": record["source_description"],
                            "target_description": record["target_description"],
                        }
                    )

            driver.close()
            return relationships
        except Exception:
            return []


class DataSourceManager:
    """Manages different data sources for the Graph Editor"""

    def __init__(self):
        self.neo4j_connector = Neo4jConnector()
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.csv_dir = self.project_root / "CSV"

    def get_concepts_with_fallback(
        self, filters: Dict[str, Any] = None, limit: int = 500
    ) -> tuple[List[Dict[str, Any]], DataSource]:
        """Get concepts with automatic fallback to CSV or demo data"""
        # Try Neo4j first
        concepts = self.neo4j_connector.get_concepts(limit=limit)

        if concepts:
            return self._apply_filters(concepts, filters), DataSource(
                "neo4j", "🗄️ Connected to Neo4j database"
            )

        # Fallback to CSV
        concepts = self._load_concepts_from_csv()
        if concepts:
            return self._apply_filters(concepts, filters), DataSource(
                "csv", "📂 Showing data from CSV files"
            )

        # Final fallback to demo data
        return self._apply_filters(DEMO_CONCEPTS, filters), DataSource(
            "demo", "🎭 Showing demonstration data"
        )

    def _load_concepts_from_csv(self) -> List[Dict[str, Any]]:
        """Load concepts from CSV files"""
        concepts = []

        if not self.csv_dir.exists():
            return []

        domains = [
            "art",
            "language",
            "mathematics",
            "philosophy",
            "science",
            "technology",
        ]

        for domain in domains:
            domain_dir = self.csv_dir / domain
            if domain_dir.exists():
                concepts_file = domain_dir / "concepts.csv"
                if concepts_file.exists():
                    try:
                        df = pd.read_csv(concepts_file)

                        for _, row in df.iterrows():
                            concept = {
                                "id": row.get("id", f"{domain}_{len(concepts)}"),
                                "name": row.get(
                                    "name", row.get("concept_name", "Unknown")
                                ),
                                "domain": domain,
                                "type": row.get(
                                    "type", row.get("concept_type", "leaf")
                                ),
                                "level": int(row.get("level", 1)),
                                "description": row.get(
                                    "description", "No description available"
                                ),
                            }
                            concepts.append(concept)

                            if len(concepts) >= 200:
                                break
                    except Exception:
                        continue

        return concepts

    def _apply_filters(
        self, concepts: List[Dict[str, Any]], filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Apply filters to concept list"""
        if not filters:
            return concepts

        filtered_concepts = concepts

        # Apply domain filter
        if filters.get("domains") and "All Domains" not in filters["domains"]:
            domain_names = [
                d.split(" ", 1)[1].lower() for d in filters["domains"] if " " in d
            ]
            filtered_concepts = [
                c for c in filtered_concepts if c["domain"] in domain_names
            ]

        # Apply type filter
        if filters.get("types") and "All Types" not in filters["types"]:
            filtered_concepts = [
                c for c in filtered_concepts if c["type"] in filters["types"]
            ]

        # Apply level filter
        if filters.get("level_range"):
            min_level, max_level = filters["level_range"]
            filtered_concepts = [
                c
                for c in filtered_concepts
                if min_level <= c.get("level", 1) <= max_level
            ]

        return filtered_concepts

    def get_cross_cultural_data(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], DataSource]:
        """Get cross-cultural concepts and relationships with fallback"""
        # Try Neo4j first
        concepts = self.neo4j_connector.get_cross_cultural_concepts()
        relationships = self.neo4j_connector.get_cross_domain_relationships()

        if concepts or relationships:
            return (
                concepts,
                relationships,
                DataSource("neo4j", "🗄️ Cross-cultural data from Neo4j"),
            )

        # Fallback to mock data if Neo4j unavailable
        mock_concepts = [
            {
                "concept_name": "Trinity",
                "domains": ["philosophy", "religion", "mathematics"],
                "domain_count": 3,
                "description1": "Philosophical concept of threefold unity",
                "description2": "Religious doctrine of divine trinity",
            },
            {
                "concept_name": "Harmony",
                "domains": ["art", "mathematics", "philosophy"],
                "domain_count": 3,
                "description1": "Mathematical ratios in music",
                "description2": "Philosophical balance and order",
            },
            {
                "concept_name": "Infinity",
                "domains": ["mathematics", "philosophy", "religion"],
                "domain_count": 3,
                "description1": "Mathematical concept of limitlessness",
                "description2": "Philosophical notion of the eternal",
            },
        ]

        mock_relationships = [
            {
                "source_concept": "Mathematics",
                "source_domain": "science",
                "relationship_type": "INFLUENCES",
                "target_concept": "Music Theory",
                "target_domain": "art",
                "source_description": "Logical reasoning and numerical relationships",
                "target_description": "Harmonic ratios and musical intervals",
            },
            {
                "source_concept": "Philosophy",
                "source_domain": "philosophy",
                "relationship_type": "CONNECTS_TO",
                "target_concept": "Theology",
                "target_domain": "religion",
                "source_description": "Rational inquiry into existence",
                "target_description": "Study of divine nature",
            },
        ]

        return (
            mock_concepts,
            mock_relationships,
            DataSource("demo", "🎭 Cross-cultural demonstration data"),
        )
