"""
Database Operations Utilities
Core functions for Neo4j, Qdrant, and Redis operations
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st
from neo4j import GraphDatabase

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Database configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
QDRANT_URL = "http://localhost:6333"
REDIS_URL = "redis://localhost:6379"


@st.cache_resource
def get_neo4j_driver():
    """Get Neo4j driver with caching"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None


def test_connections():
    """Test all database connections"""
    connections = {}

    # Test Neo4j
    try:
        driver = get_neo4j_driver()
        if driver:
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                connections["neo4j"] = result.single()["test"] == 1
        else:
            connections["neo4j"] = False
    except Exception:
        connections["neo4j"] = False

    # Test Qdrant
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        connections["qdrant"] = response.status_code == 200
    except Exception:
        connections["qdrant"] = False

    # Test Redis (simplified check)
    try:
        import redis

        r = redis.from_url(REDIS_URL)
        r.ping()
        connections["redis"] = True
    except Exception:
        connections["redis"] = False

    # Test Docker (check if containers are running)
    try:
        import subprocess

        result = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=5
        )
        connections["docker"] = "neo4j" in result.stdout and "qdrant" in result.stdout
    except Exception:
        connections["docker"] = False

    return connections


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_quick_stats():
    """Get quick statistics about the knowledge graph"""
    stats = {}

    try:
        driver = get_neo4j_driver()
        if not driver:
            return {"error": "No Neo4j connection"}

        with driver.session() as session:
            # Get concept count
            result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            stats["concepts"] = result.single()["count"]

            # Get relationship count
            result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count")
            stats["relationships"] = result.single()["count"]

            # Get domain count
            result = session.run(
                "MATCH (c:Concept) RETURN count(DISTINCT c.domain) as count"
            )
            stats["domains"] = result.single()["count"]

    except Exception as e:
        stats["error"] = str(e)

    # Get vector count from Qdrant
    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/mcp_yggdrasil_concepts", timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            stats["vectors"] = data.get("result", {}).get("vectors_count", 0)
        else:
            stats["vectors"] = 0
    except Exception:
        stats["vectors"] = 0

    return stats


def get_all_concepts(limit=None, domain=None):
    """Get all concepts from Neo4j"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return []

        with driver.session() as session:
            query = "MATCH (c:Concept)"
            params = {}

            if domain:
                query += " WHERE c.domain = $domain"
                params["domain"] = domain

            query += " RETURN c.id as id, c.name as name, c.domain as domain, c.type as type, c.level as level, c.description as description"
            query += " ORDER BY c.domain, c.name"

            if limit:
                query += " LIMIT $limit"
                params["limit"] = limit

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Exception as e:
        st.error(f"Error fetching concepts: {e}")
        return []


def get_concept_by_id(concept_id):
    """Get a specific concept by ID"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return None

        with driver.session() as session:
            result = session.run("MATCH (c:Concept {id: $id}) RETURN c", id=concept_id)
            record = result.single()
            if record:
                concept = dict(record["c"])
                return concept
            return None

    except Exception as e:
        st.error(f"Error fetching concept {concept_id}: {e}")
        return None


def get_concept_relationships(concept_id):
    """Get all relationships for a concept"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return []

        with driver.session() as session:
            # Get outgoing relationships
            result = session.run(
                """
                MATCH (c:Concept {id: $concept_id})-[r:RELATES_TO]->(target:Concept)
                RETURN 'outgoing' as direction, r.relationship_type as type, r.strength as strength,
                       target.id as target_id, target.name as target_name, target.domain as target_domain
                UNION ALL
                MATCH (source:Concept)-[r:RELATES_TO]->(c:Concept {id: $concept_id})
                RETURN 'incoming' as direction, r.relationship_type as type, r.strength as strength,
                       source.id as target_id, source.name as target_name, source.domain as target_domain
            """,
                concept_id=concept_id,
            )

            return [dict(record) for record in result]

    except Exception as e:
        st.error(f"Error fetching relationships for {concept_id}: {e}")
        return []


def get_domains():
    """Get all domains from the knowledge graph"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return []

        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)
                RETURN c.domain as domain, count(c) as concept_count
                ORDER BY domain
            """
            )

            return [dict(record) for record in result]

    except Exception as e:
        st.error(f"Error fetching domains: {e}")
        return []


def create_concept(concept_data):
    """Create a new concept in Neo4j"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return False, "No database connection"

        with driver.session() as session:
            # Check if concept ID already exists
            existing = session.run(
                "MATCH (c:Concept {id: $id}) RETURN c", id=concept_data["id"]
            ).single()

            if existing:
                return False, f"Concept with ID {concept_data['id']} already exists"

            # Create the concept
            query = f"""
                CREATE (c:Concept:{concept_data['domain']})
                SET c = $properties
                RETURN c
            """

            result = session.run(query, properties=concept_data)
            if result.single():
                return True, "Concept created successfully"
            else:
                return False, "Failed to create concept"

    except Exception as e:
        return False, f"Error creating concept: {e}"


def update_concept(concept_id, updates):
    """Update an existing concept"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return False, "No database connection"

        with driver.session() as session:
            # Build SET clause for updates
            set_clauses = []
            params = {"id": concept_id}

            for key, value in updates.items():
                if key != "id":  # Don't allow ID updates
                    set_clauses.append(f"c.{key} = ${key}")
                    params[key] = value

            if not set_clauses:
                return False, "No valid updates provided"

            query = f"""
                MATCH (c:Concept {{id: $id}})
                SET {', '.join(set_clauses)}
                RETURN c
            """

            result = session.run(query, params)
            if result.single():
                return True, "Concept updated successfully"
            else:
                return False, "Concept not found"

    except Exception as e:
        return False, f"Error updating concept: {e}"


def delete_concept(concept_id):
    """Delete a concept and its relationships"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return False, "No database connection"

        with driver.session() as session:
            # Delete concept and all its relationships
            result = session.run(
                """
                MATCH (c:Concept {id: $id})
                DETACH DELETE c
                RETURN count(c) as deleted
            """,
                id=concept_id,
            )

            deleted_count = result.single()["deleted"]
            if deleted_count > 0:
                return True, "Concept deleted successfully"
            else:
                return False, "Concept not found"

    except Exception as e:
        return False, f"Error deleting concept: {e}"


def search_concepts(search_term, domain=None, limit=50):
    """Search concepts by name or description"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            return []

        with driver.session() as session:
            query = """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($search_term) 
                   OR toLower(c.description) CONTAINS toLower($search_term)
            """
            params = {"search_term": search_term}

            if domain:
                query += " AND c.domain = $domain"
                params["domain"] = domain

            query += """
                RETURN c.id as id, c.name as name, c.domain as domain, 
                       c.type as type, c.level as level, c.description as description
                ORDER BY c.name
                LIMIT $limit
            """
            params["limit"] = limit

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Exception as e:
        st.error(f"Error searching concepts: {e}")
        return []
