# MCP Server - Hybrid Neo4j and Qdrant Knowledge Base

A high-performance, modular MCP (Main Control Platform) server that creates a hybrid database combining Neo4j (knowledge graph) and Qdrant (vector database). The system scrapes and processes diverse data across six domains—math, science, religion, history, literature, and philosophy—organizing it in a time-sensitive knowledge graph shaped like a "Yggdrasil" tree with advanced analytics and pattern recognition.

![MCP Server Architecture](docs/images/architecture-overview.png)

## 🌟 Features

### Core Capabilities
- **Hybrid Database Architecture**: Combines Neo4j knowledge graphs with Qdrant vector search
- **Six-Domain Knowledge Organization**: Mathematics, Science, Religion, History, Literature, Philosophy
- **Yggdrasil Tree Structure**: Timeline-based graph with recent documents at leaves, ancient at trunk
- **Advanced Pattern Recognition**: Cross-domain pattern detection with user validation
- **Intelligent Recommendations**: Multi-strategy recommendation engine
- **Real-time Anomaly Detection**: Automated outlier and inconsistency detection
- **Interactive Dashboard**: Streamlit-based web interface for data management
- **Comprehensive Analytics**: Trend analysis, network analysis, and performance metrics

### Technical Features
- **Scalable Architecture**: Kubernetes-ready with auto-scaling
- **High Performance**: Redis caching, query optimization, batch processing
- **Security**: OAuth2/JWT authentication, role-based access control, audit logging
- **Monitoring**: Prometheus/Grafana integration with custom dashboards
- **Backup & Recovery**: Automated backups with integrity checks
- **API-First Design**: RESTful API with comprehensive endpoint coverage

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │    Storage      │
│                 │    │                 │    │                 │
│ • Web Scraping  │───▶│ • Text Proc.    │───▶│ • Neo4j Graph   │
│ • User Upload   │    │ • NLP/Entities  │    │ • Qdrant Vectors│
│ • APIs         │    │ • Embeddings    │    │ • Redis Cache   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Analytics     │    │      API        │    │   Interface     │
│                 │    │                 │    │                 │
│ • Pattern Rec.  │◀───│ • REST API      │───▶│ • Dashboard     │
│ • Trend Analysis│    │ • WebSockets    │    │ • Visualization │
│ • Anomaly Det.  │    │ • Authentication│    │ • Admin Panel   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Kubernetes cluster** (for production deployment)
- **Git**

### Local Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/mcp-server.git
cd mcp-server
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start services with Docker:**
```bash
docker-compose up -d
```

6. **Initialize the system:**
```bash
python scripts/initialize_system.py
```

7. **Start the API server:**
```bash
python -m api.main
```

8. **Launch the dashboard:**
```bash
streamlit run dashboard/app.py
```

The system will be available at:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474
- **Qdrant UI**: http://localhost:6333/dashboard

## 📦 Production Deployment

### Kubernetes Deployment

1. **Prepare the cluster:**
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply secrets and config
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmaps.yaml
```

2. **Deploy databases:**
```bash
# Deploy Neo4j, Qdrant, Redis
kubectl apply -f k8s/databases/
```

3. **Deploy application:**
```bash
# Deploy API, workers, dashboard
kubectl apply -f k8s/application/
```

4. **Set up monitoring:**
```bash
# Deploy Prometheus, Grafana, AlertManager
kubectl apply -f k8s/monitoring/
```

5. **Configure ingress:**
```bash
# Set up external access
kubectl apply -f k8s/ingress.yaml
```

### Docker Compose (Simplified)

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api=3 --scale workers=2
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_HOST=localhost
QDRANT_PORT=6333

REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=your_jwt_secret

# Performance
CACHE_TYPE=hybrid
OPTIMIZATION_LEVEL=aggressive
MAX_CONCURRENT_REQUESTS=10

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### Configuration Files

- **`config/server.yaml`**: Main server configuration
- **`config/database.yaml`**: Database connection settings
- **`config/security.yaml`**: Authentication and authorization
- **`config/visualization.yaml`**: Chart colors and styling

## 📊 Usage Examples

### 1. Data Ingestion

#### Upload Documents via API
```python
import requests

# Upload a document
response = requests.post(
    "http://localhost:8000/api/v1/documents",
    json={
        "title": "Introduction to Quantum Physics",
        "content": "Quantum physics is the study of matter and energy...",
        "author": "Dr. Jane Smith",
        "domain": "science",
        "source": "academic_paper"
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

#### Web Scraping
```python
# Start a scraping job
scraping_job = {
    "urls": ["https://example.com/papers"],
    "domain": "mathematics",
    "max_depth": 2,
    "respect_robots": True
}

response = requests.post(
    "http://localhost:8000/api/v1/scraping/jobs",
    json=scraping_job,
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### 2. Querying Data

#### Search Documents
```python
# Semantic search
search_results = requests.get(
    "http://localhost:8000/api/v1/search",
    params={
        "query": "prime numbers theorem",
        "type": "semantic",
        "limit": 10
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

#### Graph Queries
```python
# Cypher query
graph_query = {
    "query": """
        MATCH (d:Document)-[:CONTAINS_CONCEPT]->(c:Concept)
        WHERE d.domain = 'mathematics'
        RETURN d.title, c.name
        LIMIT 20
    """,
    "parameters": {}
}

response = requests.post(
    "http://localhost:8000/api/v1/graph/query",
    json=graph_query,
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### 3. Analytics and Patterns

#### Pattern Detection
```python
# Detect cross-domain patterns
pattern_request = {
    "domains": ["religion", "science", "philosophy"],
    "similarity_threshold": 0.7,
    "pattern_types": ["structural", "conceptual"]
}

patterns = requests.post(
    "http://localhost:8000/api/v1/patterns/detect",
    json=pattern_request,
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

#### Trend Analysis
```python
# Analyze document growth trends
trend_analysis = requests.get(
    "http://localhost:8000/api/v1/analytics/trends",
    params={
        "trend_type": "document_growth",
        "domain": "science",
        "time_period": "last_year"
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### 4. Recommendations

```python
# Get recommendations for a document
recommendations = requests.get(
    f"http://localhost:8000/api/v1/recommendations/{document_id}",
    params={"limit": 5, "types": "similar_content,related_concepts"},
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

## 🔍 API Reference

### Authentication

All API endpoints require authentication using JWT tokens:

```bash
# Get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Core Endpoints

#### Documents
- `GET /api/v1/documents` - List documents
- `POST /api/v1/documents` - Create document
- `GET /api/v1/documents/{id}` - Get document
- `PUT /api/v1/documents/{id}` - Update document
- `DELETE /api/v1/documents/{id}` - Delete document

#### Search
- `GET /api/v1/search` - Search documents
- `POST /api/v1/search/semantic` - Semantic search
- `POST /api/v1/search/graph` - Graph-based search

#### Analytics
- `GET /api/v1/analytics/trends` - Trend analysis
- `GET /api/v1/analytics/network` - Network analysis
- `POST /api/v1/patterns/detect` - Pattern detection
- `GET /api/v1/anomalies` - List anomalies

#### Recommendations
- `GET /api/v1/recommendations/{node_id}` - Get recommendations
- `POST /api/v1/recommendations/feedback` - Provide feedback

#### Administration
- `GET /api/v1/admin/health` - System health
- `GET /api/v1/admin/metrics` - Performance metrics
- `POST /api/v1/admin/maintenance` - Maintenance actions

## 🎨 Dashboard Guide

### Overview Page
- System metrics and KPIs
- Recent activity feed
- Growth trends visualization
- Domain distribution charts

### Data Input
- **File Upload**: Drag-and-drop interface for documents
- **Web Scraping**: Configure scraping jobs with URL lists
- **Manual Entry**: Form-based document input
- **Batch Import**: CSV/JSON bulk import

### Query & Search
- **Text Search**: Full-text search with filters
- **Semantic Search**: Vector similarity search
- **Graph Queries**: Cypher query interface
- **Relationship Explorer**: Interactive graph navigation

### Visualizations
- **Yggdrasil Tree**: Interactive timeline-based graph
- **Network Graph**: Force-directed layout
- **Timeline View**: Temporal document distribution
- **Custom Charts**: Authority maps, concept clusters

### Analytics
- **Trend Analysis**: Time-series analysis of various metrics
- **Network Analysis**: Centrality measures, community detection
- **Pattern Analysis**: Cross-domain pattern discovery
- **Reports**: Automated report generation

### Maintenance
- **System Health**: Component status monitoring
- **Pending Actions**: Approval workflow for changes
- **Backup Management**: Backup creation and restoration
- **Performance Tuning**: Cache and optimization settings

## 🧪 Testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v

# All tests with coverage
pytest --cov=agents --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component interactions
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and authorization
- **API Tests**: Endpoint validation

## 📈 Performance Optimization

### Caching Strategy

The system implements a three-tier caching strategy:

1. **L1 Cache (Memory)**: Fast access for frequently used data
2. **L2 Cache (Redis)**: Shared cache across instances
3. **L3 Cache (Query Results)**: Database query result caching

### Query Optimization

- **Index Strategy**: Composite indexes on frequently queried fields
- **Query Hints**: Automatic query optimization with performance monitoring
- **Batch Processing**: Configurable batch sizes for different operations
- **Connection Pooling**: Optimized database connection management

### Monitoring & Metrics

Key performance indicators tracked:
- API response times (95th percentile < 2s)
- Cache hit rates (target > 80%)
- Database query performance
- Memory and CPU usage
- Error rates and availability

## 🔒 Security

### Authentication & Authorization

- **JWT-based authentication** with configurable expiration
- **Role-based access control** (Admin, Developer, Read-only)
- **API rate limiting** to prevent abuse
- **Input validation** and sanitization

### Data Security

- **Encryption at rest** using AES-256
- **Encryption in transit** using TLS 1.3
- **Audit logging** for all system changes
- **Backup encryption** with key rotation

### Network Security

- **Network policies** in Kubernetes
- **Firewall rules** for database access
- **VPN requirements** for admin access
- **Regular security updates**

## 🛠 Maintenance

### Daily Operations

1. **Health Checks**: Monitor system dashboards
2. **Log Review**: Check for errors and warnings
3. **Performance Metrics**: Review response times and throughput
4. **Backup Verification**: Ensure backups completed successfully

### Weekly Tasks

1. **Anomaly Review**: Investigate detected anomalies
2. **Pattern Analysis**: Review new cross-domain patterns
3. **Performance Tuning**: Optimize slow queries
4. **Security Updates**: Apply security patches

### Monthly Tasks

1. **Capacity Planning**: Review resource usage trends
2. **Backup Testing**: Test restore procedures
3. **Performance Analysis**: Deep dive into system metrics
4. **Documentation Updates**: Update procedures and runbooks

### Troubleshooting

#### Common Issues

**Database Connection Failures**
```bash
# Check database status
kubectl get pods -n mcp-server | grep -E "(neo4j|qdrant|redis)"

# View logs
kubectl logs -n mcp-server deployment/neo4j
```

**High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n mcp-server

# Adjust resource limits
kubectl patch deployment mcp-server-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"mcp-server-api","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

**Cache Performance Issues**
```bash
# Check cache hit rates
curl http://localhost:8000/api/v1/admin/metrics | jq '.cache_stats'

# Clear cache if needed
curl -X POST http://localhost:8000/api/v1/admin/cache/clear
```

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting guidelines

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all functions
- Include docstrings for all public methods
- Maintain test coverage above 80%

## 📚 Additional Resources

### Documentation
- [API Documentation](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Performance Tuning](docs/performance.md)

### Examples
- [Tutorial Notebooks](examples/notebooks/)
- [Sample Configurations](examples/configs/)
- [Integration Examples](examples/integrations/)

### Community
- [GitHub Discussions](https://github.com/your-org/mcp-server/discussions)
- [Slack Channel](https://your-slack.slack.com/channels/mcp-server)
- [Blog Posts](https://blog.your-org.com/tag/mcp-server)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Neo4j** for the excellent graph database platform
- **Qdrant** for high-performance vector search capabilities
- **Anthropic** for Claude's invaluable development assistance
- **Open Source Community** for the foundational tools and libraries

---

## 📞 Support

For support and questions:

- **Documentation**: Check this README and [docs/](docs/) folder
- **Issues**: Create a [GitHub issue](https://github.com/your-org/mcp-server/issues)
- **Email**: support@your-org.com
- **Community**: Join our [Slack channel](https://your-slack.slack.com/channels/mcp-server)

---

**Built with ❤️ by the MCP Server team**

*Last updated: January 2025*
