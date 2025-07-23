# MCP Yggdrasil - Comprehensive Testing Guide for Claude Code

## 📋 Overview

This testing guide provides a complete roadmap for implementing comprehensive tests for the MCP Yggdrasil project after all phases are complete. The testing strategy covers unit tests, integration tests, performance tests, and end-to-end validation.

## 🔧 Pre-Testing Setup

### 1. Linting Setup (First Priority)

```bash
# Install linting tools
make setup-lint

# Run comprehensive linting on all code
make lint

# Auto-fix linting issues
make lint-fix

# Verify linting configuration
python tests/lint/lint_project.py --tools flake8 black mypy
```

### 2. Testing Dependencies Installation

```bash
# Install all testing dependencies
pip install -r requirements-dev.txt

# Verify pytest installation
pytest --version

# Check coverage tools
coverage --version
```

### 3. Environment Configuration

```bash
# Create test environment file
cp .env.example .env.test

# Set test database connections
export NEO4J_TEST_URI="bolt://localhost:7688"
export QDRANT_TEST_HOST="localhost"
export QDRANT_TEST_PORT="6334"
export REDIS_TEST_URL="redis://localhost:6380"
```

## 🧪 Testing Categories

### 1. Unit Tests

#### Agent Testing
```python
# tests/unit/test_agents/test_scraper.py
import pytest
from agents.scraper.unified_web_scraper import UnifiedWebScraper, ScraperConfig

class TestUnifiedWebScraper:
    @pytest.fixture
    def scraper_config(self):
        return ScraperConfig(
            profile="test",
            timeout=5,
            rate_limit=10
        )
    
    @pytest.mark.asyncio
    async def test_scraper_initialization(self, scraper_config):
        scraper = UnifiedWebScraper(scraper_config)
        assert scraper.config.profile == "test"
        assert scraper.is_initialized
    
    @pytest.mark.asyncio
    async def test_content_extraction(self, scraper_config):
        scraper = UnifiedWebScraper(scraper_config)
        result = await scraper.scrape("https://example.com")
        assert result.status == "success"
        assert result.content is not None
```

#### Database Manager Testing
```python
# tests/unit/test_database/test_neo4j_manager.py
import pytest
from agents.neo4j_manager.neo4j_agent import Neo4jAgent

class TestNeo4jManager:
    @pytest.mark.asyncio
    async def test_connection(self, mock_neo4j):
        agent = Neo4jAgent(driver=mock_neo4j)
        assert await agent.verify_connection()
    
    @pytest.mark.asyncio
    async def test_node_creation(self, mock_neo4j):
        agent = Neo4jAgent(driver=mock_neo4j)
        node_id = await agent.create_node(
            label="Concept",
            properties={"name": "Test", "domain": "science"}
        )
        assert node_id is not None
```

### 2. Integration Tests

#### Multi-Agent Workflow Testing
```python
# tests/integration/test_validation_pipeline.py
import pytest
from agents.fact_verifier.enhanced_verification_agent import FactVerifier
from agents.anomaly_detector.anomaly_detector import AnomalyDetector

class TestValidationPipeline:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        # Test the full validation pipeline
        content = "The Earth orbits the Sun in 365.25 days."
        
        # Step 1: Fact verification
        verifier = FactVerifier()
        fact_result = await verifier.verify_claim(content)
        assert fact_result.confidence > 0.8
        
        # Step 2: Anomaly detection
        detector = AnomalyDetector()
        anomaly_result = await detector.analyze(content)
        assert anomaly_result.is_normal
```

#### Database Synchronization Testing
```python
# tests/integration/test_db_sync.py
class TestDatabaseSync:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_neo4j_qdrant_sync(self, neo4j_agent, qdrant_agent):
        # Create node in Neo4j
        node_data = {
            "id": "test_123",
            "content": "Test concept",
            "domain": "science"
        }
        await neo4j_agent.create_node(**node_data)
        
        # Verify sync to Qdrant
        await asyncio.sleep(2)  # Wait for sync
        vector_result = await qdrant_agent.search(
            query="Test concept",
            collection="concepts"
        )
        assert len(vector_result) > 0
        assert vector_result[0].id == "test_123"
```

### 3. Performance Tests

#### Load Testing
```python
# tests/performance/test_load.py
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestLoadPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_scraping(self):
        urls = ["https://example.com"] * 100
        scraper = UnifiedWebScraper()
        
        start_time = time.time()
        results = await asyncio.gather(*[
            scraper.scrape(url) for url in urls
        ])
        duration = time.time() - start_time
        
        assert len(results) == 100
        assert duration < 60  # Should complete in under 60 seconds
        
    @pytest.mark.performance
    def test_graph_query_performance(self, large_graph):
        agent = Neo4jAgent()
        
        # Test complex graph traversal
        start_time = time.time()
        result = agent.find_shortest_path(
            start_node="philosophy_001",
            end_node="science_999",
            max_depth=10
        )
        duration = time.time() - start_time
        
        assert duration < 0.2  # 200ms target
```

### 4. End-to-End Tests

#### Complete Workflow Testing
```python
# tests/e2e/test_full_workflow.py
class TestEndToEnd:
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_content_to_knowledge_pipeline(self):
        # 1. Scrape content
        scraper = UnifiedWebScraper()
        content = await scraper.scrape("https://en.wikipedia.org/wiki/Quantum_mechanics")
        
        # 2. Process text
        processor = TextProcessor()
        processed = await processor.process(content.text)
        
        # 3. Extract concepts
        extractor = ConceptExtractor()
        concepts = await extractor.extract(processed)
        
        # 4. Validate facts
        verifier = FactVerifier()
        validated_concepts = []
        for concept in concepts:
            if await verifier.verify(concept):
                validated_concepts.append(concept)
        
        # 5. Store in databases
        neo4j = Neo4jAgent()
        qdrant = QdrantAgent()
        
        for concept in validated_concepts:
            node_id = await neo4j.create_concept(concept)
            await qdrant.index_concept(concept, node_id)
        
        assert len(validated_concepts) > 0
```

### 5. UI Testing (Streamlit)

#### Page Testing
```python
# tests/ui/test_streamlit_pages.py
from streamlit.testing import AppTest

class TestStreamlitUI:
    def test_main_dashboard(self):
        app = AppTest("streamlit_workspace/main_dashboard.py")
        app.run()
        
        # Test navigation
        assert "MCP Yggdrasil" in app.title
        assert len(app.sidebar.selectbox) > 0
        
    def test_graph_editor(self):
        app = AppTest("streamlit_workspace/pages/02_📊_Graph_Editor.py")
        app.run()
        
        # Test graph visualization
        assert app.get("graph_container") is not None
```

## 📊 Testing Metrics & Coverage

### Coverage Requirements

```yaml
# .coveragerc
[run]
source = .
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    */archive/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

### Coverage Targets by Component

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Core Agents | 90% | Critical |
| Database Managers | 85% | Critical |
| API Endpoints | 80% | High |
| Scrapers | 75% | High |
| UI Components | 70% | Medium |
| Utilities | 60% | Low |

## 🚀 Test Execution Commands

### Running All Tests

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m performance    # Performance tests
pytest -m e2e           # End-to-end tests

# Run tests for specific components
pytest tests/unit/test_agents/
pytest tests/integration/test_validation_pipeline.py
pytest tests/performance/ -k "load"

# Run with parallel execution
pytest -n auto          # Auto-detect CPU cores
pytest -n 4            # Use 4 parallel workers
```

### Continuous Testing

```bash
# Watch mode for development
pytest-watch --clear --onpass "make lint"

# Run tests on file changes
watchmedo shell-command \
    --patterns="*.py" \
    --recursive \
    --command='pytest ${watch_src_path}'
```

## 📝 Test Data Management

### Using Existing Test Fixtures

The project already has comprehensive fixtures in `tests/conftest.py`:

```python
# Available fixtures (from conftest.py)
- mock_redis
- mock_neo4j_driver  
- mock_qdrant_client
- sample_graph_data
- sample_concepts
- performance_monitor
```

### Creating Test Data

```python
# tests/fixtures/test_data.py
import json
from pathlib import Path

class TestDataManager:
    @staticmethod
    def load_sample_concepts():
        path = Path("tests/fixtures/data/concepts.json")
        with open(path) as f:
            return json.load(f)
    
    @staticmethod
    def generate_large_graph(nodes=1000, edges=5000):
        # Generate test graph for performance testing
        pass
```

## 🔄 Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      neo4j:
        image: neo4j:5.15
        env:
          NEO4J_AUTH: neo4j/password
        ports:
          - 7688:7687
      
      redis:
        image: redis:7
        ports:
          - 6380:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: make lint
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🎯 Testing Best Practices

### 1. Test Organization
- One test file per module
- Group related tests in classes
- Use descriptive test names

### 2. Mock External Dependencies
- Always mock external API calls
- Use fixtures for database connections
- Simulate network failures

### 3. Performance Testing
- Set clear performance targets
- Test with realistic data volumes
- Monitor memory usage

### 4. Data Validation
- Test edge cases
- Validate data transformations
- Check error handling

## 📚 Additional Testing Resources

### Existing Test Files to Update/Extend

1. **Unit Tests** (already exist):
   - `tests/unit/test_scraper.py`
   - `tests/unit/test_anomaly_detector.py`
   - `tests/unit/test_graph_metrics.py`

2. **Integration Tests** (need expansion):
   - `tests/integration/test_integration.py`

3. **Performance Tests** (need creation):
   - Create `tests/performance/` directory
   - Add load testing scenarios

### Testing Documentation

- Comprehensive pytest configuration in `pyproject.toml`
- Test fixtures in `tests/conftest.py` (532 lines)
- Linting setup in `tests/lint/README.md`

## ✅ Testing Checklist

Before considering testing complete:

- [ ] All code passes linting (flake8, black, mypy)
- [ ] Unit test coverage > 80% for core components
- [ ] Integration tests cover all major workflows
- [ ] Performance tests validate all targets
- [ ] E2E tests pass for complete pipelines
- [ ] UI tests cover all Streamlit pages
- [ ] Documentation updated with test examples
- [ ] CI/CD pipeline configured and passing
- [ ] Test data fixtures properly maintained
- [ ] Security tests for API endpoints

---

**Note**: This testing guide should be executed after Phase 6 completion. Some test examples reference components that may need adjustment based on final implementation details.