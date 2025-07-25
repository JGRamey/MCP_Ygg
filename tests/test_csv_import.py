"""
Tests for CSV import functionality.
Tests the import of concept and relationship data into Neo4j.
"""

import csv
import os
import tempfile

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class TestCSVImportFunctionality:
    """Test CSV import functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "driver"):
            self.driver.close()

    def test_concepts_import_structure(self):
        """Test that concepts are imported with correct structure."""
        with self.driver.session() as session:
            # Test concept node structure
            result = session.run(
                """
                MATCH (c:Concept) 
                RETURN c.id, c.name, c.domain, c.type, c.level, c.description
                LIMIT 5
            """
            )

            concepts = list(result)
            assert len(concepts) > 0

            for concept in concepts:
                # Required fields
                assert concept["c.id"] is not None
                assert concept["c.name"] is not None
                assert concept["c.domain"] is not None
                assert concept["c.type"] is not None
                assert concept["c.level"] is not None

                # ID format should be DOMAIN#### (e.g., ART0001)
                concept_id = concept["c.id"]
                assert len(concept_id) >= 6  # Minimum length
                assert concept_id[:3].isalpha()  # First 3 chars should be letters
                assert concept_id[3:].isdigit()  # Rest should be digits

    def test_domain_specific_labels(self):
        """Test that concepts have domain-specific labels."""
        with self.driver.session() as session:
            # Test that concepts have both :Concept and domain-specific labels
            domains = [
                "Art",
                "Science",
                "Mathematics",
                "Philosophy",
                "Language",
                "Technology",
                "Religion",
                "Astrology",
            ]

            for domain in domains:
                result = session.run(
                    f"""
                    MATCH (c:Concept:{domain})
                    RETURN count(c) as count
                """
                )

                count = result.single()["count"]
                if count > 0:  # Only test domains that have data
                    # Verify these also have the base Concept label
                    result = session.run(
                        f"""
                        MATCH (c:{domain})
                        WHERE NOT c:Concept
                        RETURN count(c) as invalid_count
                    """
                    )

                    invalid_count = result.single()["invalid_count"]
                    assert (
                        invalid_count == 0
                    ), f"Found {invalid_count} {domain} nodes without Concept label"

    def test_relationships_import_structure(self):
        """Test that relationships are imported with correct structure."""
        with self.driver.session() as session:
            # Test relationship structure
            result = session.run(
                """
                MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
                RETURN r.relationship_type, r.strength, r.domain, r.description
                LIMIT 10
            """
            )

            relationships = list(result)
            assert len(relationships) > 0

            for rel in relationships:
                # Required fields
                assert rel["r.relationship_type"] is not None
                assert rel["r.domain"] is not None

                # Optional but common fields
                strength = rel["r.strength"]
                if strength is not None:
                    assert isinstance(strength, (int, float))
                    assert 0 <= strength <= 1

    def test_hierarchical_relationships(self):
        """Test that hierarchical relationships are properly imported."""
        with self.driver.session() as session:
            # Test BELONGS_TO relationships (hierarchical)
            result = session.run(
                """
                MATCH (child:Concept)-[r:RELATES_TO {relationship_type: 'BELONGS_TO'}]->(parent:Concept)
                WHERE child.level > parent.level
                RETURN count(r) as hierarchical_count
            """
            )

            hierarchical_count = result.single()["hierarchical_count"]
            assert (
                hierarchical_count > 0
            ), "No hierarchical BELONGS_TO relationships found"

            # Test that root concepts exist
            result = session.run(
                """
                MATCH (root:Concept {type: 'root'})
                RETURN count(root) as root_count
            """
            )

            root_count = result.single()["root_count"]
            assert root_count > 0, "No root concepts found"

            # Test that root concepts are level 1
            result = session.run(
                """
                MATCH (root:Concept {type: 'root'})
                WHERE root.level <> 1
                RETURN count(root) as invalid_root_count
            """
            )

            invalid_root_count = result.single()["invalid_root_count"]
            assert (
                invalid_root_count == 0
            ), f"Found {invalid_root_count} root concepts not at level 1"

    def test_data_integrity_constraints(self):
        """Test data integrity and constraints."""
        with self.driver.session() as session:
            # Test that all concept IDs are unique
            result = session.run(
                """
                MATCH (c:Concept)
                WITH c.id as concept_id, count(c) as count
                WHERE count > 1
                RETURN concept_id, count
            """
            )

            duplicates = list(result)
            assert len(duplicates) == 0, f"Found duplicate concept IDs: {duplicates}"

            # Test that all relationships connect valid concepts
            result = session.run(
                """
                MATCH (a)-[r:RELATES_TO]->(b)
                WHERE NOT a:Concept OR NOT b:Concept
                RETURN count(r) as invalid_rels
            """
            )

            invalid_rels = result.single()["invalid_rels"]
            assert (
                invalid_rels == 0
            ), f"Found {invalid_rels} relationships not connecting Concept nodes"

    def test_domain_completeness(self):
        """Test that all expected domains are imported."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)
                RETURN DISTINCT c.domain as domain
                ORDER BY domain
            """
            )

            imported_domains = [record["domain"] for record in result]

            expected_domains = [
                "Art",
                "Astrology",
                "Language",
                "Mathematics",
                "Philosophy",
                "Religion",
                "Science",
                "Technology",
            ]

            for domain in expected_domains:
                assert (
                    domain in imported_domains
                ), f"Domain {domain} not found in imported data"

    def test_metadata_fields(self):
        """Test that metadata fields are properly imported."""
        with self.driver.session() as session:
            # Test date fields
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.earliest_evidence_date IS NOT NULL
                RETURN c.earliest_evidence_date as date
                LIMIT 5
            """
            )

            dates = [record["date"] for record in result]
            for date in dates:
                assert isinstance(
                    date, int
                ), f"Date should be integer, got {type(date)}"

            # Test location fields
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.location IS NOT NULL AND c.location <> ''
                RETURN count(c) as location_count
            """
            )

            location_count = result.single()["location_count"]
            assert location_count > 0, "No concepts have location data"


class TestCSVDataQuality:
    """Test the quality of imported CSV data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "driver"):
            self.driver.close()

    def test_no_empty_required_fields(self):
        """Test that required fields are not empty."""
        with self.driver.session() as session:
            # Test for empty names
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.name IS NULL OR c.name = ''
                RETURN count(c) as empty_names
            """
            )

            empty_names = result.single()["empty_names"]
            assert empty_names == 0, f"Found {empty_names} concepts with empty names"

            # Test for empty domains
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.domain IS NULL OR c.domain = ''
                RETURN count(c) as empty_domains
            """
            )

            empty_domains = result.single()["empty_domains"]
            assert (
                empty_domains == 0
            ), f"Found {empty_domains} concepts with empty domains"

            # Test for empty types
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.type IS NULL OR c.type = ''
                RETURN count(c) as empty_types
            """
            )

            empty_types = result.single()["empty_types"]
            assert empty_types == 0, f"Found {empty_types} concepts with empty types"

    def test_consistent_naming_conventions(self):
        """Test that naming conventions are consistent."""
        with self.driver.session() as session:
            # Test that concept names don't have excessive underscores
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE c.name CONTAINS '__'
                RETURN c.name as name
            """
            )

            double_underscores = list(result)
            assert (
                len(double_underscores) == 0
            ), f"Found concepts with double underscores: {[r['name'] for r in double_underscores]}"

            # Test that concept names are not excessively long
            result = session.run(
                """
                MATCH (c:Concept)
                WHERE size(c.name) > 100
                RETURN c.name as name, size(c.name) as length
            """
            )

            long_names = list(result)
            assert (
                len(long_names) == 0
            ), f"Found concepts with excessively long names: {long_names}"

    def test_relationship_consistency(self):
        """Test that relationships are consistent."""
        with self.driver.session() as session:
            # Test that BELONGS_TO relationships are properly directional
            result = session.run(
                """
                MATCH (child:Concept)-[r:RELATES_TO {relationship_type: 'BELONGS_TO'}]->(parent:Concept)
                WHERE child.level <= parent.level AND child.level IS NOT NULL AND parent.level IS NOT NULL
                RETURN child.name as child, parent.name as parent, child.level as child_level, parent.level as parent_level
            """
            )

            invalid_hierarchy = list(result)
            assert (
                len(invalid_hierarchy) == 0
            ), f"Found invalid hierarchy relationships: {invalid_hierarchy}"

    def test_domain_specific_concepts(self):
        """Test domain-specific concept validation."""
        with self.driver.session() as session:
            # Test Art domain concepts
            result = session.run(
                """
                MATCH (c:Art)
                WHERE c.name IN ['Art', 'Visual_Arts', 'Paintings', 'Sculpture', 'Architecture']
                RETURN count(c) as art_core_concepts
            """
            )

            art_concepts = result.single()["art_core_concepts"]
            assert art_concepts > 0, "Missing core Art concepts"

            # Test Mathematics domain concepts
            result = session.run(
                """
                MATCH (c:Mathematics)
                WHERE c.name IN ['Mathematics', 'Algebra', 'Geometry', 'Calculus']
                RETURN count(c) as math_core_concepts
            """
            )

            math_concepts = result.single()["math_core_concepts"]
            assert math_concepts > 0, "Missing core Mathematics concepts"

            # Test Science domain concepts
            result = session.run(
                """
                MATCH (c:Science)
                WHERE c.name IN ['Science', 'Physics', 'Chemistry', 'Biology']
                RETURN count(c) as science_core_concepts
            """
            )

            science_concepts = result.single()["science_core_concepts"]
            assert science_concepts > 0, "Missing core Science concepts"


class TestCSVImportPerformance:
    """Test the performance of CSV import operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "driver"):
            self.driver.close()

    def test_query_performance_on_imported_data(self):
        """Test query performance on the imported dataset."""
        import time

        with self.driver.session() as session:
            # Test simple concept lookup by ID
            start_time = time.time()
            result = session.run("MATCH (c:Concept {id: 'ART0001'}) RETURN c")
            result.single()
            lookup_time = time.time() - start_time

            assert (
                lookup_time < 0.1
            ), f"Concept lookup took {lookup_time:.3f}s, expected < 0.1s"

            # Test domain filtering
            start_time = time.time()
            result = session.run("MATCH (c:Art) RETURN count(c)")
            result.single()
            domain_query_time = time.time() - start_time

            assert (
                domain_query_time < 0.5
            ), f"Domain query took {domain_query_time:.3f}s, expected < 0.5s"

            # Test relationship traversal
            start_time = time.time()
            result = session.run(
                """
                MATCH (root:Concept {type: 'root'})-[:RELATES_TO*1..2]->(related:Concept)
                RETURN count(related)
            """
            )
            result.single()
            traversal_time = time.time() - start_time

            assert (
                traversal_time < 1.0
            ), f"Relationship traversal took {traversal_time:.3f}s, expected < 1.0s"

    def test_index_effectiveness(self):
        """Test that indexes are working effectively."""
        with self.driver.session() as session:
            # Test that concept ID index exists and is used
            result = session.run("SHOW INDEXES")
            indexes = list(result)

            # Look for concept ID index
            concept_id_indexes = [
                idx for idx in indexes if "concept_id" in str(idx).lower()
            ]
            assert len(concept_id_indexes) > 0, "No concept ID index found"

            # Test index usage with EXPLAIN
            result = session.run("EXPLAIN MATCH (c:Concept {id: 'ART0001'}) RETURN c")
            plan = result.single()

            # The query plan should use an index scan, not a label scan
            plan_str = str(plan)
            assert (
                "Index" in plan_str or "NodeByLabelScan" not in plan_str
            ), "Query not using index efficiently"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
