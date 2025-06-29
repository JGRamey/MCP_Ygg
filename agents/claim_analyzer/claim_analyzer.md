# Claim Analyzer Agent for MCP Server

An advanced AI agent that extracts verifiable claims from text and performs automated fact-checking using the existing MCP server's hybrid Neo4j/Qdrant database infrastructure.

## 🎯 Overview

The Claim Analyzer Agent integrates seamlessly with your existing MCP server architecture to provide:

- **Intelligent Claim Extraction**: Uses spaCy NLP to identify verifiable claims in text
- **Multi-Source Fact-Checking**: Cross-references claims against your knowledge graph and external APIs
- **Cross-Domain Analysis**: Leverages your six-domain structure (Math, Science, Religion, History, Literature, Philosophy)
- **Vector Similarity Search**: Uses Qdrant for semantic claim matching
- **Graph-Based Evidence**: Explores relationships in your Neo4j knowledge graph
- **Real-Time Processing**: Async processing with Redis caching

## 🏗️ Architecture Integration

```
┌─────────────────────────┐    ┌─────────────────────────┐
│   Existing MCP Server   │    │  Claim Analyzer Agent   │
│                         │    │                         │
│ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
│ │    Neo4j Graph      │◄┼────┼►│  Claim Extractor    │ │
│ │  (Knowledge Base)   │ │    │ │   (spaCy + NLP)     │ │
│ └─────────────────────┘ │    │ └─────────────────────┘ │
│                         │    │                         │
│ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
│ │   Qdrant Vectors    │◄┼────┼►│   Fact Checker      │ │
│ │  (Semantic Search)  │ │    │ │ (Evidence Analysis) │ │
│ └─────────────────────┘ │    │ └─────────────────────┘ │
│                         │    │                         │
│ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
│ │   Redis Cache       │◄┼────┼►│  Results Storage    │ │
│ │  (Performance)      │ │    │ │   (Caching)         │ │
│ └─────────────────────┘ │    │ └─────────────────────┘ │
└─────────────────────────┘    └─────────────────────────┘
```

## 🚀 Quick Installation

### 1. Install the Agent

```bash
# Navigate to your MCP server directory
cd /path/to/your/mcp-server

# Run the installation script
python scripts/install_claim_analyzer.py

# Or install manually
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm
```

### 2. Copy Agent Files

Place these files in your MCP server structure:

```
mcp-server/
├── agents/
│   └── claim_analyzer/
│       ├── __init__.py
│       ├── claim_analyzer_agent.py
│       ├── config.yaml
│       └── logs/
├── api/routes/
│   └── claim_analyzer.py
├── dashboard/pages/
│   └── claim_analyzer.py
└── scripts/
    └── install_claim_analyzer.py
```

### 3. Update Main Configuration

Add to your main `config/server.yaml`:

```yaml
agents:
  claim_analyzer:
    enabled: true
    config_path: "agents/claim_analyzer/config.yaml"
    auto_start: true
```

### 4. Update API Router

Add to your main API application (`api/main.py`):

```python
from api.routes import claim_analyzer

app.include_router(claim_analyzer.router, prefix="/api/v1")
```

### 5. Update Dashboard

Add to your dashboard navigation (`dashboard/app.py`):

```python
import streamlit as st
from dashboard.pages import claim_analyzer

# Add to your page selection
if page == "Claim Analyzer":
    claim_analyzer.main()
```

## ⚙️ Configuration

### Database Integration

The agent automatically connects to your existing databases:

```yaml
# agents/claim_analyzer/config.yaml
database:
  neo4j:
    uri: "bolt://localhost:7687"  # Your Neo4j instance
    user: "neo4j"
    password: "password"
    
  qdrant:
    host: "localhost"             # Your Qdrant instance
    port: 6333
    
  redis:
    url: "redis://localhost:6379" # Your Redis instance
```

### Domain Classification

Leverages your existing six-domain structure:

```yaml
domain_keywords:
  science: ["experiment", "study", "research", "theory"]
  math: ["theorem", "proof", "equation", "formula"]
  religion: ["god", "faith", "belief", "scripture"]
  history: ["ancient", "war", "civilization", "empire"]
  literature: ["novel", "poem", "author", "book"]
  philosophy: ["ethics", "logic", "metaphysics", "moral"]
```

### Source Credibility

Configure trusted sources for fact-checking:

```yaml
source_credibility:
  "snopes.com": 0.95
  "factcheck.org": 0.95
  "nasa.gov": 0.95
  "cdc.gov": 0.95
  "wikipedia.org": 0.80
```

## 📋 Usage Examples

### 1. API Integration

```python
import requests

# Analyze text for claims
response = requests.post("http://localhost:8000/api/v1/claim-analyzer/analyze-text", 
    json={
        "text": "The Earth is flat and climate change is a hoax.",
        "source": "social_media",
        "domain": "science"
    }
)

results = response.json()
print(f"Found {results['total_claims']} claims")
```

### 2. Direct Agent Usage

```python
from agents.claim_analyzer import ClaimAnalyzerAgent

agent = ClaimAnalyzerAgent()
await agent.initialize()

# Process text
results = await agent.process_text(
    text="Vaccines are completely safe and effective.",
    source="medical_article",
    domain="science"
)

# Fact-check single claim
result = await agent.fact_check_single_claim(
    claim_text="The moon landing was faked",
    domain="history"
)
```

### 3. Integration with Other Agents

```python
# agents/scraper/scraper_agent.py
from agents.claim_analyzer import ClaimAnalyzerAgent

class WebScraperAgent:
    def __init__(self):
        self.claim_analyzer = ClaimAnalyzerAgent()
    
    async def process_scraped_content(self, content, url, domain):
        # Scrape content as usual
        # ...
        
        # Automatically analyze claims
        claim_results = await self.claim_analyzer.process_text(
            text=content,
            source=url,
            domain=domain
        )
        
        return claim_results
```

## 🔧 Advanced Features

### Cross-Domain Pattern Analysis

The agent automatically detects patterns across your six domains:

```python
# Example: Trinity concept across religion and philosophy
patterns = await agent._analyze_cross_domain_patterns(claim)
# Returns: ["Trinity appears across religion and philosophy domains"]
```

### Vector Similarity Search

Leverages your existing Qdrant setup:

```python
similar_claims = await agent.get_similar_claims(
    claim_text="Earth's shape is spherical",
    limit=5
)
```

### Graph-Based Evidence

Explores relationships in your Neo4j knowledge graph:

```cypher
MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)
MATCH (e)<-[:MENTIONS]-(d:Document {domain: $domain})
RETURN d.content as evidence
```

## 📊 Dashboard Integration

The agent includes a comprehensive Streamlit dashboard:

### Pages Available:
- **Text Analysis**: Upload/paste text for claim extraction
- **Single Fact-Check**: Verify individual claims
- **Similar Claims**: Find related claims in database
- **Agent Stats**: Performance monitoring
- **Health Monitor**: System status

### Access Dashboard:
```bash
streamlit run dashboard/app.py
# Navigate to "Claim Analyzer" in the sidebar
```

## 🧪 Testing

### Unit Tests

```bash
# Run claim analyzer tests
pytest tests/agents/test_claim_analyzer.py -v

# Test database integration
pytest tests/agents/test_claim_analyzer_integration.py -v
```

### Manual Testing

```bash
# Test the agent directly
python agents/claim_analyzer/claim_analyzer_agent.py

# Test API endpoints
curl -X POST http://localhost:8000/api/v1/claim-analyzer/health
```

## 📈 Performance Optimization

### Batch Processing

```yaml
# config.yaml
agent:
  batch_size: 50
  max_concurrent_checks: 5
  processing_interval: 300
```

### Caching Strategy

```yaml
performance:
  caching:
    enable_claim_cache: true
    enable_evidence_cache: true
    cache_size_limit: "1GB"
```

### Database Optimization

```yaml
database_optimization:
  connection_pooling: true
  query_timeout: 30
  batch_operations: true
```

## 🔍 Monitoring & Metrics

### Prometheus Metrics

The agent exposes metrics on port 9091:

- `claims_processed_total`
- `fact_checks_performed_total`
- `verdict_distribution`
- `average_confidence_score`
- `cross_domain_patterns_found`

### Health Checks

```bash
# Check agent health
curl http://localhost:8000/api/v1/claim-analyzer/health

# Response:
{
  "status": "healthy",
  "database_connections": {
    "neo4j": true,
    "qdrant": true,
    "redis": true
  },
  "agent_running": true
}
```

## 🔒 Security & Privacy

### Input Validation

```yaml
security:
  input_validation:
    max_text_length: 100000
    sanitize_html: true
    filter_malicious_patterns: true
```

### Rate Limiting

```yaml
security:
  rate_limiting:
    max_requests_per_minute: 60
    max_claims_per_request: 10
```

### Data Privacy

```yaml
security:
  data_privacy:
    anonymize_sources: false
    retention_period_days: 365
    encrypt_sensitive_data: false
```

## 🐛 Troubleshooting

### Common Issues

#### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

#### "Database connection failed"
```bash
# Check your existing MCP server databases
docker-compose ps
kubectl get pods -n mcp-server
```

#### "Qdrant collection not found"
```bash
# Collections are auto-created on first run
# Check Qdrant UI: http://localhost:6333/dashboard
```

#### "Low fact-checking accuracy"
```yaml
# Adjust confidence thresholds in config.yaml
agent:
  confidence_threshold: 0.3  # Lower = more permissive
  similarity_threshold: 0.6  # Lower = more matches
```

### Debug Mode

```yaml
# config.yaml
logging:
  level: "DEBUG"
  components:
    claim_extractor: "DEBUG"
    fact_checker: "DEBUG"
```

## 🚦 Integration Checklist

- [ ] Install dependencies (`spacy`, `sentence-transformers`)
- [ ] Download NLP models (`en_core_web_sm`)
- [ ] Copy agent files to correct directories
- [ ] Update main server configuration
- [ ] Add API routes to main application
- [ ] Include dashboard pages
- [ ] Configure database connections
- [ ] Set up monitoring/metrics
- [ ] Run health checks
- [ ] Test with sample data

## 📚 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/claim-analyzer/analyze-text` | Analyze text for claims |
| POST | `/api/v1/claim-analyzer/fact-check` | Fact-check single claim |
| POST | `/api/v1/claim-analyzer/similar-claims` | Find similar claims |
| GET | `/api/v1/claim-analyzer/stats` | Get agent statistics |
| GET | `/api/v1/claim-analyzer/health` | Health check |

### Request/Response Examples

#### Analyze Text
```json
// Request
{
  "text": "The Earth is flat and vaccines cause autism.",
  "source": "social_media",
  "domain": "science"
}

// Response
{
  "total_claims": 2,
  "fact_check_results": [
    {
      "claim": {
        "text": "The Earth is flat",
        "domain": "science",
        "confidence": 0.8
      },
      "verdict": "False",
      "confidence": 0.95,
      "reasoning": "Scientific evidence confirms Earth is spherical...",
      "evidence_list": [...]
    }
  ]
}
```

## 🤝 Contributing

### Adding New Features

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-claim-type`
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

### Extending Fact-Checking

```python
# agents/claim_analyzer/extensions/custom_checker.py
class CustomFactChecker:
    async def check_domain_specific_claim(self, claim, domain):
        # Add domain-specific fact-checking logic
        pass
```

## 📄 License

This agent inherits the same license as the main MCP server project.

## 🙏 Acknowledgments

- Built on the existing MCP server infrastructure
- Integrates with Neo4j knowledge graph
- Leverages Qdrant vector search capabilities
- Uses spaCy for natural language processing

---

**Ready to enhance your MCP server with intelligent claim analysis and fact-checking capabilities!**