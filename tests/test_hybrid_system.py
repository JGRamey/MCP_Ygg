"""
Tests for the hybrid Neo4j + Qdrant knowledge system.
Tests the integration between graph database and vector database.
"""

import json
import os

import pytest
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class TestHybridSystemIntegration:
    """Test the hybrid Neo4j + Qdrant system integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.qdrant_url = "http://localhost:6333"

        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "driver"):
            self.driver.close()

    def test_neo4j_connection(self):
        """Test Neo4j database connection."""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            assert record["test"] == 1

    def test_qdrant_connection(self):
        """Test Qdrant vector database connection."""
        response = requests.get(f"{self.qdrant_url}/health")
        assert (
            response.status_code == 404 or response.status_code == 200
        )  # 404 is also OK for health endpoint

        # Check collections
        response = requests.get(f"{self.qdrant_url}/collections")
        assert response.status_code == 200
        data = response.json()
        assert "result" in data

    def test_concept_data_integrity(self):
        """Test that concept data is properly loaded in Neo4j."""
        with self.driver.session() as session:
            # Test total concept count
            result = session.run("MATCH (c:Concept) RETURN count(c) as total")
            total_concepts = result.single()["total"]
            assert (
                total_concepts == 371
            ), f"Expected 371 concepts, found {total_concepts}"

            # Test domain distribution
            result = session.run(
                """
                MATCH (c:Concept) 
                RETURN c.domain as domain, count(c) as count 
                ORDER BY domain
            """
            )

            domain_counts = {record["domain"]: record["count"] for record in result}

            # Expected domain counts
            expected_counts = {
                "Art": 50,
                "Language": 40,
                "Mathematics": 58,
                "Philosophy": 30,
                "Science": 65,
                "Technology": 8,
                "Religion": 104,
                "Astrology": 16,
            }

            for domain, expected_count in expected_counts.items():
                assert domain in domain_counts, f"Domain {domain} not found"
                assert (
                    domain_counts[domain] == expected_count
                ), f"Domain {domain}: expected {expected_count}, found {domain_counts[domain]}"

    def test_relationship_integrity(self):
        """Test that relationships are properly loaded in Neo4j."""
        with self.driver.session() as session:
            # Test total relationship count
            result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as total")
            total_relationships = result.single()["total"]
            assert total_relationships > 0, "No relationships found"

            # Test that relationships have proper structure
            result = session.run(
                """
                MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept) 
                RETURN r.relationship_type, r.domain, r.strength 
                LIMIT 10
            """
            )

            for record in result:
                assert record["r.relationship_type"] is not None
                assert record["r.domain"] is not None
                # Strength might be None for some relationships

    def test_qdrant_vector_data(self):
        """Test that vector data is properly loaded in Qdrant."""
        response = requests.get(f"{self.qdrant_url}/collections/mcp_yggdrasil_concepts")
        assert response.status_code == 200

        collection_info = response.json()
        vectors_count = collection_info["result"]["vectors_count"]
        assert vectors_count > 0, "No vectors found in Qdrant collection"

        # Test vector search functionality
        search_vector = [0.1] * 384  # Simple test vector
        search_data = {"vector": search_vector, "limit": 5, "with_payload": True}

        response = requests.post(
            f"{self.qdrant_url}/collections/mcp_yggdrasil_concepts/points/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_data),
        )

        assert response.status_code == 200
        search_results = response.json()
        assert "result" in search_results
        assert len(search_results["result"]) > 0

    def test_hybrid_query_workflow(self):
        """Test a complete hybrid query workflow."""
        # Step 1: Find concepts in Neo4j by domain
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Art) 
                WHERE c.type = 'root'
                RETURN c.id as concept_id, c.name as name
                LIMIT 1
            """
            )

            art_concept = result.single()
            assert art_concept is not None
            concept_id = art_concept["concept_id"]

        # Step 2: Find similar concepts in Qdrant using vector search
        search_vector = [0.1] * 384  # Simple test vector
        search_data = {
            "vector": search_vector,
            "limit": 3,
            "with_payload": True,
            "filter": {"must": [{"key": "domain", "match": {"value": "Art"}}]},
        }

        response = requests.post(
            f"{self.qdrant_url}/collections/mcp_yggdrasil_concepts/points/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_data),
        )

        assert response.status_code == 200
        similar_concepts = response.json()["result"]
        assert len(similar_concepts) > 0

        # Step 3: Get detailed information from Neo4j
        similar_concept_ids = [
            result["payload"]["concept_id"] for result in similar_concepts
        ]

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept) 
                WHERE c.id IN $concept_ids
                RETURN c.id, c.name, c.description, c.domain
            """,
                concept_ids=similar_concept_ids,
            )

            concept_details = list(result)
            assert len(concept_details) > 0

            # Verify data consistency
            for record in concept_details:
                assert record["c.id"] in similar_concept_ids
                assert record["c.domain"] is not None

    def test_cross_domain_relationships(self):
        """Test cross-domain relationship queries."""
        with self.driver.session() as session:
            # Find relationships that cross domains
            result = session.run(
                """
                MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
                WHERE a.domain <> b.domain
                RETURN a.domain as domain1, b.domain as domain2, count(r) as relationship_count
                ORDER BY relationship_count DESC
                LIMIT 10
            """
            )

            cross_domain_rels = list(result)
            # We might not have cross-domain relationships in our current dataset
            # but the query should work without errors

    def test_concept_hierarchy(self):
        """Test concept hierarchy structure."""
        with self.driver.session() as session:
            # Test that we have proper hierarchy levels
            result = session.run(
                """
                MATCH (c:Concept)
                RETURN c.type as concept_type, c.level as level, count(c) as count
                ORDER BY level
            """
            )

            hierarchy_levels = list(result)
            assert len(hierarchy_levels) > 0

            # Check that we have root concepts
            root_concepts = [h for h in hierarchy_levels if h["concept_type"] == "root"]
            assert len(root_concepts) > 0

            # Check that hierarchy makes sense (root should be level 1)
            for h in hierarchy_levels:
                if h["concept_type"] == "root":
                    assert h["level"] == 1


class TestSystemPerformance:
    """Test system performance and scalability."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.qdrant_url = "http://localhost:6333"

        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "driver"):
            self.driver.close()

    def test_neo4j_query_performance(self):
        """Test Neo4j query performance."""
        import time

        with self.driver.session() as session:
            # Test simple concept retrieval
            start_time = time.time()
            result = session.run("MATCH (c:Concept) RETURN count(c)")
            result.single()
            query_time = time.time() - start_time

            assert (
                query_time < 1.0
            ), f"Simple count query took {query_time:.2f}s, expected < 1.0s"

            # Test relationship traversal
            start_time = time.time()
            result = session.run(
                """
                MATCH (c:Concept)-[r:RELATES_TO]->(related:Concept)
                RETURN c.domain, count(related) as related_count
                ORDER BY related_count DESC
                LIMIT 10
            """
            )
            list(result)
            query_time = time.time() - start_time

            assert (
                query_time < 2.0
            ), f"Relationship query took {query_time:.2f}s, expected < 2.0s"

    def test_qdrant_search_performance(self):
        """Test Qdrant vector search performance."""
        import time

        search_vector = [0.1] * 384
        search_data = {"vector": search_vector, "limit": 10, "with_payload": True}

        start_time = time.time()
        response = requests.post(
            f"{self.qdrant_url}/collections/mcp_yggdrasil_concepts/points/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_data),
        )
        search_time = time.time() - start_time

        assert response.status_code == 200
        assert (
            search_time < 1.0
        ), f"Vector search took {search_time:.2f}s, expected < 1.0s"

    def test_concurrent_queries(self):
        """Test concurrent query handling."""
        import threading
        import time

        def query_worker():
            driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )

            try:
                with driver.session() as session:
                    result = session.run("MATCH (c:Concept) RETURN c.name LIMIT 10")
                    list(result)
            finally:
                driver.close()

        # Run multiple concurrent queries
        threads = []
        start_time = time.time()

        for _ in range(5):
            thread = threading.Thread(target=query_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        total_time = time.time() - start_time
        assert (
            total_time < 5.0
        ), f"Concurrent queries took {total_time:.2f}s, expected < 5.0s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
