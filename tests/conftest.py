#!/usr/bin/env python3
"""
Pytest configuration and fixtures for MCP Yggdrasil test suite
Provides comprehensive test fixtures and utilities
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import asyncio
import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Test configuration
pytest_plugins = []


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.delete.return_value = 1
    mock.scan_iter.return_value = []
    mock.info.return_value = {
        "connected_clients": 1,
        "used_memory": 1024,
        "used_memory_human": "1KB",
        "keyspace_hits": 100,
        "keyspace_misses": 20,
    }
    return mock


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = AsyncMock()
    driver.verify_connectivity.return_value = True

    # Mock session
    session = AsyncMock()
    result = AsyncMock()
    result.single.return_value = {
        "n": {"id": "test_node", "name": "Test Node"},
        "labels": ["Concept"],
    }
    result.data.return_value = [
        {"id": "test_node", "name": "Test Node", "domain": "philosophy"}
    ]
    session.run.return_value = result
    driver.session.return_value.__aenter__.return_value = session

    return driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = AsyncMock()

    # Mock search results
    search_result = {
        "result": [
            {"id": 1, "score": 0.95, "payload": {"domain": "philosophy"}},
            {"id": 2, "score": 0.90, "payload": {"domain": "science"}},
        ]
    }
    client.search.return_value = search_result

    # Mock collection info
    collection_info = {
        "result": {"status": "green", "vectors_count": 1000, "segments_count": 1}
    }
    client.get_collection.return_value = collection_info

    return client


@pytest.fixture
def sample_graph():
    """Create a sample NetworkX graph for testing."""
    G = nx.Graph()

    # Add nodes with attributes
    nodes = [
        (1, {"name": "Philosophy", "domain": "philosophy", "type": "concept"}),
        (2, {"name": "Science", "domain": "science", "type": "concept"}),
        (3, {"name": "Mathematics", "domain": "mathematics", "type": "concept"}),
        (4, {"name": "Aristotle", "domain": "philosophy", "type": "person"}),
        (5, {"name": "Newton", "domain": "science", "type": "person"}),
        (6, {"name": "Euclid", "domain": "mathematics", "type": "person"}),
    ]

    # Add edges
    edges = [
        (1, 4, {"relationship": "studied_by", "weight": 1.0}),
        (2, 5, {"relationship": "studied_by", "weight": 1.0}),
        (3, 6, {"relationship": "studied_by", "weight": 1.0}),
        (1, 2, {"relationship": "influences", "weight": 0.8}),
        (2, 3, {"relationship": "uses", "weight": 0.9}),
        (1, 3, {"relationship": "relates_to", "weight": 0.7}),
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    # Create 365 days of data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    # Generate synthetic data with trend and seasonality
    t = np.arange(len(dates))
    trend = t * 0.01  # Linear trend
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
    noise = np.random.normal(0, 2, len(dates))
    values = 100 + trend + seasonal + noise

    return pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "domain": ["philosophy"] * len(dates),
            "metric": ["node_count"] * len(dates),
        }
    )


@pytest.fixture
def sample_vector_data():
    """Create sample vector data for testing."""
    np.random.seed(42)  # For reproducible results

    vectors = []
    for i in range(100):
        vector = {
            "id": f"vec_{i}",
            "vector": np.random.randn(384).tolist(),  # 384-dimensional vector
            "payload": {
                "domain": np.random.choice(["philosophy", "science", "mathematics"]),
                "type": np.random.choice(["concept", "person", "work"]),
                "name": f"Entity_{i}",
                "created_at": (
                    datetime.now() - timedelta(days=np.random.randint(0, 365))
                ).isoformat(),
            },
        }
        vectors.append(vector)

    return vectors


@pytest.fixture
def analysis_config():
    """Create analysis configuration for testing."""
    from agents.analytics.graph_analysis.base import AnalysisConfig

    return AnalysisConfig(
        graph_type="knowledge_graph",
        analysis_level="comprehensive",
        include_visualization=True,
        cache_results=False,
        parallel_processing=False,
        max_nodes=10000,
        max_edges=50000,
        timeout=30,
    )


@pytest.fixture
def mock_analysis_session():
    """Create mock analysis session."""
    session = AsyncMock()

    # Mock query results
    mock_result = AsyncMock()
    mock_result.data.return_value = [
        {"node_id": 1, "label": "Concept", "domain": "philosophy"},
        {"node_id": 2, "label": "Person", "domain": "science"},
    ]
    session.run.return_value = mock_result

    return session


@pytest.fixture
def mock_cache_manager():
    """Create mock cache manager for testing."""
    cache = AsyncMock()

    # Mock cache operations
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = 1
    cache.delete_pattern.return_value = 5
    cache.health_check.return_value = {
        "status": "healthy",
        "redis_connected": True,
        "read_write_test": True,
    }
    cache.get_stats.return_value = {
        "hit_rate": 85.0,
        "total_keys": 1000,
        "used_memory": 50 * 1024 * 1024,
        "used_memory_human": "50MB",
    }

    return cache


@pytest.fixture
def sample_concepts():
    """Create sample concept data for testing."""
    concepts = [
        {
            "id": "concept_1",
            "name": "Metaphysics",
            "domain": "philosophy",
            "type": "concept",
            "description": "The branch of philosophy that studies the nature of reality",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        },
        {
            "id": "concept_2",
            "name": "Quantum Mechanics",
            "domain": "science",
            "type": "concept",
            "description": "The branch of physics that studies matter and energy at the smallest scales",
            "created_at": "2023-01-02T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
        },
        {
            "id": "concept_3",
            "name": "Calculus",
            "domain": "mathematics",
            "type": "concept",
            "description": "The mathematical study of continuous change",
            "created_at": "2023-01-03T00:00:00Z",
            "updated_at": "2023-01-03T00:00:00Z",
        },
    ]

    return concepts


@pytest.fixture
def sample_relationships():
    """Create sample relationship data for testing."""
    relationships = [
        {
            "id": "rel_1",
            "source_id": "concept_1",
            "target_id": "person_1",
            "relationship_type": "studied_by",
            "weight": 1.0,
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "id": "rel_2",
            "source_id": "concept_2",
            "target_id": "person_2",
            "relationship_type": "discovered_by",
            "weight": 1.0,
            "created_at": "2023-01-02T00:00:00Z",
        },
        {
            "id": "rel_3",
            "source_id": "concept_1",
            "target_id": "concept_2",
            "relationship_type": "influences",
            "weight": 0.8,
            "created_at": "2023-01-03T00:00:00Z",
        },
    ]

    return relationships


@pytest.fixture
def mock_streamlit_session():
    """Create mock Streamlit session state for testing."""
    session_state = {
        "user_id": "test_user",
        "session_id": "test_session",
        "active_domain": "philosophy",
        "search_history": [],
        "cached_results": {},
        "ui_state": {"selected_concept": None, "active_tab": "concepts", "filters": {}},
    }

    return session_state


@pytest.fixture
def performance_test_data():
    """Create performance test data for large-scale testing."""
    # Create larger dataset for performance testing
    nodes = []
    edges = []

    # Generate 1000 nodes
    for i in range(1000):
        node = {
            "id": f"node_{i}",
            "name": f"Entity_{i}",
            "domain": np.random.choice(
                [
                    "philosophy",
                    "science",
                    "mathematics",
                    "art",
                    "language",
                    "technology",
                ]
            ),
            "type": np.random.choice(["concept", "person", "work", "place"]),
            "created_at": (
                datetime.now() - timedelta(days=np.random.randint(0, 365))
            ).isoformat(),
        }
        nodes.append(node)

    # Generate 5000 edges
    for i in range(5000):
        source = np.random.randint(0, 1000)
        target = np.random.randint(0, 1000)
        if source != target:  # Avoid self-loops
            edge = {
                "id": f"edge_{i}",
                "source_id": f"node_{source}",
                "target_id": f"node_{target}",
                "relationship_type": np.random.choice(
                    ["influences", "studies", "creates", "belongs_to"]
                ),
                "weight": np.random.uniform(0.1, 1.0),
                "created_at": (
                    datetime.now() - timedelta(days=np.random.randint(0, 365))
                ).isoformat(),
            }
            edges.append(edge)

    return {"nodes": nodes, "edges": edges}


# Pytest configuration functions
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line(
        "markers", "database: marks tests that require database connections"
    )
    config.addinivalue_line(
        "markers", "external_api: marks tests that require external API access"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU resources")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add markers based on test names
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)

        if "database" in item.name or "neo4j" in item.name or "qdrant" in item.name:
            item.add_marker(pytest.mark.database)

        if "external" in item.name or "api" in item.name:
            item.add_marker(pytest.mark.external_api)


# Custom assertions for testing
def assert_analysis_result(result, expected_type: str = None):
    """Assert that an analysis result has the expected structure."""
    assert hasattr(result, "success")
    assert hasattr(result, "data")
    assert hasattr(result, "execution_time")
    assert hasattr(result, "timestamp")

    if expected_type:
        assert hasattr(result, "analysis_type")
        assert result.analysis_type == expected_type

    if result.success:
        assert result.data is not None
        assert result.execution_time >= 0
    else:
        assert hasattr(result, "error_message")
        assert result.error_message is not None


def assert_cache_performance(cache_stats: Dict[str, Any], min_hit_rate: float = 70.0):
    """Assert that cache performance meets minimum requirements."""
    assert "hit_rate" in cache_stats
    assert cache_stats["hit_rate"] >= min_hit_rate

    assert "total_keys" in cache_stats
    assert cache_stats["total_keys"] >= 0

    assert "used_memory" in cache_stats
    assert cache_stats["used_memory"] >= 0


def assert_vector_result(result, expected_count: int = None):
    """Assert that a vector operation result has the expected structure."""
    assert hasattr(result, "success")
    assert hasattr(result, "data")
    assert hasattr(result, "execution_time")

    if result.success:
        assert result.data is not None
        assert result.execution_time >= 0

        if expected_count:
            assert len(result.data.get("results", [])) == expected_count


def assert_graph_structure(graph, min_nodes: int = 1, min_edges: int = 0):
    """Assert that a graph has the expected structure."""
    assert graph.number_of_nodes() >= min_nodes
    assert graph.number_of_edges() >= min_edges

    # Check that nodes have required attributes
    for node in graph.nodes():
        assert "name" in graph.nodes[node] or "id" in graph.nodes[node]

    # Check that edges have weights (if any edges exist)
    if graph.number_of_edges() > 0:
        for edge in graph.edges():
            # Edges should have some attributes
            assert len(graph.edges[edge]) >= 0


# Helper functions for test data generation
def generate_test_vectors(count: int, dimension: int = 384) -> List[Dict]:
    """Generate test vectors with specified count and dimension."""
    np.random.seed(42)  # For reproducible results

    vectors = []
    for i in range(count):
        vector = {
            "id": f"test_vector_{i}",
            "vector": np.random.randn(dimension).tolist(),
            "payload": {
                "domain": np.random.choice(["philosophy", "science", "mathematics"]),
                "type": "concept",
                "name": f"Test Concept {i}",
            },
        }
        vectors.append(vector)

    return vectors


def generate_test_time_series(
    days: int = 365, domains: List[str] = None
) -> pd.DataFrame:
    """Generate test time series data."""
    if domains is None:
        domains = ["philosophy", "science", "mathematics"]

    data = []
    start_date = datetime.now() - timedelta(days=days)

    for domain in domains:
        for i in range(days):
            date = start_date + timedelta(days=i)
            value = 100 + i * 0.1 + np.random.randn() * 5

            data.append(
                {"date": date, "value": value, "domain": domain, "metric": "node_count"}
            )

    return pd.DataFrame(data)


# Async test utilities
async def async_test_timeout(coro, timeout: float = 10.0):
    """Run an async test with timeout."""
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        pytest.fail(f"Test timed out after {timeout} seconds")


def mock_async_context_manager(mock_obj):
    """Create an async context manager from a mock object."""
    mock_obj.__aenter__ = AsyncMock(return_value=mock_obj)
    mock_obj.__aexit__ = AsyncMock(return_value=None)
    return mock_obj
