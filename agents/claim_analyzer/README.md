# Claim Analyzer Agent - Refactored

A modular, well-organized claim analysis and fact-checking agent for the MCP Server project.

## 📁 Structure

```
claim_analyzer/
├── __init__.py              # Package initialization with exports
├── README.md                # This documentation
├── config.yaml              # YAML configuration file
├── models.py                # Data models (Claim, Evidence, FactCheckResult)
├── database.py              # Database connection management
├── extractor.py             # Claim extraction logic
├── checker.py               # Fact-checking logic
├── claim_analyzer.py        # Main agent class
├── utils.py                 # Utility functions
├── exceptions.py            # Custom exception classes
└── logs/                    # Log directory (auto-created)
```

## 🚀 Key Improvements

### 1. **Modular Architecture**
- **Before**: 1099 lines in single file
- **After**: Separated into focused modules (50-400 lines each)
- **Benefits**: Easier maintenance, testing, and development

### 2. **Better Error Handling**
- Custom exception hierarchy
- Graceful degradation for non-critical failures
- Comprehensive logging with performance metrics

### 3. **Enhanced Configuration**
- YAML-based configuration (was hardcoded)
- Environment-specific settings
- Runtime reconfiguration support

### 4. **Improved Type Safety**
- Comprehensive type hints throughout
- Better IDE support and error detection
- Cleaner interfaces

### 5. **Professional Documentation**
- Detailed docstrings with parameter descriptions
- Usage examples and architectural overview
- Clear separation of concerns

## 🔧 Usage

### Basic Usage

```python
from agents.analytics.claim_analyzer import ClaimAnalyzerAgent

# Initialize with custom config
agent = ClaimAnalyzerAgent("path/to/config.yaml")
await agent.initialize()

# Process text for claims
results = await agent.process_text(
    text="The Earth is flat and vaccines cause autism.",
    source="social_media",
    domain="science"
)

# Fact-check single claim
result = await agent.fact_check_single_claim(
    "Climate change is a hoax",
    domain="science"
)

# Find similar claims
similar = await agent.get_similar_claims("Earth is not round")
```

### Advanced Usage

```python
from agents.analytics.claim_analyzer import (
    ClaimExtractor, FactChecker, DatabaseConnector
)

# Use individual components
db_connector = DatabaseConnector(config)
await db_connector.initialize()

extractor = ClaimExtractor(db_connector)
claims = await extractor.extract_claims(text, source, domain)

checker = FactChecker(db_connector, config)
for claim in claims:
    result = await checker.fact_check_claim(claim)
```

## 🎯 Components

### Core Classes

#### `ClaimAnalyzerAgent`
Main orchestrator class that coordinates all components.

**Key Methods:**
- `initialize()`: Set up database connections and models
- `process_text()`: Extract and fact-check claims from text
- `fact_check_single_claim()`: Verify individual claims
- `get_similar_claims()`: Find related claims in database

#### `ClaimExtractor`
Extracts verifiable claims from text using NLP.

**Features:**
- spaCy-based sentence analysis
- Pattern-based fallback extraction
- Domain classification
- Entity recognition

#### `FactChecker`
Performs comprehensive fact-checking using multiple strategies.

**Strategies:**
- Vector similarity search (Qdrant)
- Graph relationship analysis (Neo4j)
- External API integration
- Cross-domain pattern analysis

#### `DatabaseConnector`
Manages connections to Neo4j, Qdrant, and Redis.

**Features:**
- Connection pooling
- Health monitoring
- Error recovery
- Graceful shutdown

### Data Models

#### `Claim`
```python
@dataclass
class Claim:
    claim_id: str
    text: str
    source: str
    domain: str
    timestamp: datetime
    confidence: float = 0.0
    context: str = ""
    entities: List[str] = field(default_factory=list)
```

#### `Evidence`
```python
@dataclass
class Evidence:
    evidence_id: str
    text: str
    source_url: str
    credibility_score: float
    stance: str  # "supports", "refutes", "neutral"
    domain: str
    timestamp: datetime
    vector_embedding: Optional[np.ndarray] = None
```

#### `FactCheckResult`
```python
@dataclass
class FactCheckResult:
    claim: Claim
    verdict: str  # "True", "False", "Partially True", "Unverified", "Opinion"
    confidence: float
    evidence_list: List[Evidence]
    reasoning: str
    sources: List[str]
    cross_domain_patterns: List[str]
    timestamp: datetime
    graph_node_id: Optional[str] = None
```

## ⚙️ Configuration

### config.yaml Structure

```yaml
# Database connections
database:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
    max_pool_size: 20
  qdrant:
    host: "localhost"
    port: 6333
    timeout: 30
  redis:
    url: "redis://localhost:6379"
    max_connections: 50

# Agent settings
agent:
  max_results: 10
  confidence_threshold: 0.5
  batch_size: 50
  processing_interval: 300

# NLP models
models:
  spacy_model: "en_core_web_sm"
  sentence_transformer: "all-MiniLM-L6-v2"
  embedding_dimensions: 384

# Source credibility
source_credibility:
  "snopes.com": 0.95
  "factcheck.org": 0.95
  "nasa.gov": 0.95
  "wikipedia.org": 0.80

# Domain keywords for classification
domain_keywords:
  science: ["experiment", "study", "research"]
  math: ["theorem", "proof", "equation"]
  religion: ["god", "faith", "belief"]
  # ... etc
```

## 🔍 Error Handling

### Exception Hierarchy

```python
ClaimAnalyzerError              # Base exception
├── DatabaseConnectionError     # Database issues
├── ConfigurationError          # Config problems
├── ClaimExtractionError       # Extraction failures
├── FactCheckingError          # Fact-checking issues
├── ModelLoadError             # Model loading problems
├── ValidationError            # Input validation
├── RateLimitError            # API rate limits
└── EvidenceSearchError       # Evidence search issues
```

### Example Error Handling

```python
try:
    agent = ClaimAnalyzerAgent()
    await agent.initialize()
    results = await agent.process_text(text)
except DatabaseConnectionError:
    logger.error("Database unavailable, using offline mode")
    # Fallback to pattern-based extraction
except ModelLoadError:
    logger.error("NLP models unavailable, using simple extraction")
    # Use regex-based extraction
except ValidationError as e:
    logger.warning(f"Input validation failed: {e}")
    # Return error to user
```

## 📊 Performance Features

### Performance Monitoring
```python
# Automatic timing for operations
with PerformanceTimer("fact_check_claim"):
    result = await fact_checker.fact_check_claim(claim)
```

### Batch Processing
```python
# Process claims in batches
for batch in batch_process(claims, batch_size=50):
    results.extend(await process_batch(batch))
```

### Caching Strategy
- Redis caching for frequent queries
- Vector similarity caching
- Evidence result caching
- Configurable TTL settings

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
pytest tests/claim_analyzer/test_extractor.py -v
pytest tests/claim_analyzer/test_checker.py -v
pytest tests/claim_analyzer/test_database.py -v
```

### Integration Tests
```bash
# Test full workflow
pytest tests/claim_analyzer/test_integration.py -v
```

### Performance Tests
```bash
# Benchmark performance
pytest tests/claim_analyzer/test_performance.py -v
```

## 🔧 Development

### Adding New Features

1. **New Extraction Strategy**:
   ```python
   # In extractor.py
   async def _extract_with_custom_method(self, text, source, domain):
       # Implementation
       pass
   ```

2. **New Evidence Source**:
   ```python
   # In checker.py
   async def _search_custom_api(self, claim):
       # Implementation
       pass
   ```

3. **New Domain**:
   ```yaml
   # In config.yaml
   domain_keywords:
     new_domain:
       - "keyword1"
       - "keyword2"
   ```

### Code Quality

- **Type checking**: `mypy agents/analytics/claim_analyzer/`
- **Linting**: `flake8 agents/analytics/claim_analyzer/`
- **Formatting**: `black agents/analytics/claim_analyzer/`
- **Testing**: `pytest tests/claim_analyzer/ --cov`

## 📈 Monitoring

### Metrics Collected
- Claims processed per minute
- Fact-checks performed
- Average confidence scores
- Database query performance
- Model inference time
- Error rates by type

### Health Checks
```python
# Check system health
health = await agent.get_agent_stats()
print(f"Database status: {health['database_status']}")
print(f"Claims processed: {health['processed_claims']}")
```

## 🚀 Production Deployment

### Resource Requirements
- **CPU**: 4+ cores (for NLP processing)
- **Memory**: 8GB+ RAM (for models and caching)
- **Storage**: 50GB+ (for logs and cache)
- **Network**: High bandwidth for external API calls

### Scaling Considerations
- Horizontal scaling via multiple agent instances
- Load balancing across database connections
- Rate limiting for external APIs
- Caching strategies for performance

### Monitoring in Production
- Prometheus metrics on port 9091
- Health check endpoint
- Structured logging with correlation IDs
- Alert thresholds for key metrics

## 🔄 Migration from Original

### Breaking Changes
- Configuration moved from Python to YAML
- Import paths changed (use package imports)
- Some method signatures updated for type safety

### Migration Script
```python
# Convert old config to new format
from claim_analyzer.utils import migrate_config
migrate_config("old_config.py", "new_config.yaml")
```

### Compatibility Layer
```python
# For gradual migration
from claim_analyzer.compat import LegacyClaimAnalyzer
legacy_agent = LegacyClaimAnalyzer()  # Uses old interface
```

## 📚 Further Reading

- [MCP Server Architecture](../../../docs/architecture.md)
- [Database Schema](../../../docs/database.md)
- [API Documentation](../../../api/docs/claim_analyzer.md)
- [Deployment Guide](../../../docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**Ready to use the refactored, production-ready Claim Analyzer Agent!**