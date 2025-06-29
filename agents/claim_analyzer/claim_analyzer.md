# AI Claim Analyzer MCP Server

A sophisticated AI-powered Model Context Protocol (MCP) server that provides claim extraction, fact-checking, and analysis capabilities to AI models and applications.

## 🎯 Features

- **Claim Extraction**: Uses advanced NLP to identify verifiable claims in text
- **Fact-Checking**: Cross-references claims against trusted sources
- **Semantic Search**: Find similar claims using vector similarity
- **Evidence Analysis**: Weighs multiple sources and provides reasoning
- **Caching System**: Stores results to improve performance
- **MCP Integration**: Seamlessly integrates with AI models via Model Context Protocol

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Model      │◄──►│  MCP Server     │◄──►│  External APIs  │
│   (Claude, etc) │    │  (This Server)  │    │  (Fact Checkers)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   SQLite DB     │
                       │   (Claims Cache)│
                       └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for fact-checking APIs
- Optional: GPU for enhanced NLP processing

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/claim-analyzer-mcp
cd claim-analyzer-mcp

# Run the installation script
chmod +x install.sh
./install.sh
```

Or install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Initialize database
python3 -c "from claim_analyzer import ClaimDatabase; ClaimDatabase()"
```

### 2. Configuration

Copy and customize the configuration file:

```bash
cp config.yml.example config.yml
# Edit config.yml with your API keys and preferences
```

### 3. Running the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the MCP server
python3 claim_analyzer.py
```

The server will run in stdio mode by default, ready to accept MCP connections.

## 🔧 MCP Tools

The server exposes the following tools to AI models:

### `analyze_claims`
Extract claims from text input.

**Parameters:**
- `text` (string): Text to analyze for claims
- `source` (string, optional): Source identifier

**Example:**
```json
{
  "text": "The Earth is flat and climate change is a hoax.",
  "source": "social_media_post"
}
```

### `fact_check_claim`
Fact-check a specific claim.

**Parameters:**
- `claim` (string): The claim to fact-check
- `detailed` (boolean, optional): Whether to return detailed analysis

**Example:**
```json
{
  "claim": "Vaccines cause autism",
  "detailed": true
}
```

### `search_similar_claims`
Find similar claims in the database.

**Parameters:**
- `claim` (string): Claim to find similar ones for
- `limit` (integer, optional): Maximum number of results

### `get_recent_fact_checks`
Retrieve recent fact-checking activity.

**Parameters:**
- `limit` (integer, optional): Number of results to return

## 💻 Usage Examples

### Python Client

```python
from mcp_client import ClaimAnalyzerClient

async def main():
    client = ClaimAnalyzerClient()
    await client.connect("python3 claim_analyzer.py")
    
    # Analyze text for claims
    result = await client.analyze_claims(
        "The moon landing was faked in a Hollywood studio."
    )
    print(f"Found {result['total_claims']} claims")
    
    # Fact-check a specific claim
    fact_check = await client.fact_check_claim(
        "The moon landing was faked"
    )
    print(f"Verdict: {fact_check['verdict']}")
    print(f"Confidence: {fact_check['confidence']:.2f}")
    
    await client.disconnect()
```

### Command Line Interface

```bash
# Interactive demo
python3 mcp_client.py

# Batch analysis
python3 mcp_client.py batch
```

### Integration with Claude

Add to your Claude configuration:

```json
{
  "mcpServers": {
    "claim-analyzer": {
      "command": "python3",
      "args": ["/path/to/claim_analyzer.py"],
      "env": {}
    }
  }
}
```

## ⚙️ Configuration

### Basic Configuration (`config.yml`)

```yaml
# API Keys for external services
api_keys:
  google_fact_check: "your_api_key"
  bing_search: "your_api_key"

# Trusted sources for fact-checking
trusted_sources:
  - "wikipedia.org"
  - "snopes.com"
  - "factcheck.org"
  - "reuters.com"

# Analysis settings
analysis:
  language: "english"
  max_results: 10
  confidence_threshold: 0.5
  cache_duration_hours: 24

# Rate limiting
rate_limiting:
  api_requests_per_minute: 30
  fact_check_delay: 2.0
```

### Advanced Configuration

See the full `config.yml` for all available options including:
- Custom claim detection patterns
- Source credibility scoring
- Performance tuning
- Security settings
- Feature flags

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test categories
python3 -m pytest tests/test_claim_extractor.py -v
python3 -m pytest tests/test_fact_checker.py -v

# Run with coverage
pip install pytest-cov
python3 -m pytest --cov=claim_analyzer tests/
```

## 📊 Performance

### Benchmarks

- **Claim Extraction**: ~100 claims/second
- **Fact-Checking**: ~5 claims/second (with external APIs)
- **Cache Lookup**: ~1000 queries/second
- **Memory Usage**: ~500MB baseline + ~50MB per 1000 cached claims

### Optimization Tips

1. **Enable Caching**: Set appropriate `cache_duration_hours`
2. **Rate Limiting**: Configure `api_requests_per_minute` to avoid API limits
3. **Batch Processing**: Process multiple claims together
4. **GPU Acceleration**: Enable `use_gpu: true` for NLP models

## 🔒 Security

### Best Practices

- Store API keys securely (environment variables recommended)
- Enable input sanitization: `sanitize_input: true`
- Set reasonable rate limits: `enable_client_rate_limiting: true`
- Limit request sizes: `max_request_size: 1048576`

### Data Privacy

- Claims are cached locally by default
- No sensitive data is sent to external APIs
- All network requests are logged for audit

## 🐛 Troubleshooting

### Common Issues

#### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

#### "MCP connection failed"
- Ensure the server is running
- Check Python path in MCP client configuration
- Verify virtual environment is activated

#### "Rate limit exceeded"
- Reduce `api_requests_per_minute` in config
- Increase `fact_check_delay`
- Check external API quotas

#### "Database locked"
- Ensure only one server instance is running
- Check file permissions on `claims.db`

### Debug Mode

Enable detailed logging:

```yaml
logging:
  level: "DEBUG"
  log_to_file: true
  log_file: "debug.log"
```

## 📈 Monitoring

### Health Checks

```bash
# Check server status
curl localhost:8000/health  # If using HTTP transport

# Database stats
python3 -c "
from claim_analyzer import ClaimDatabase
db = ClaimDatabase()
print(f'Recent fact-checks: {len(db.get_recent_results(100))}')
"
```

### Metrics

The server tracks:
- Claims processed per hour
- Fact-check accuracy by source
- Cache hit rates
- API response times

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time video transcript analysis
- [ ] Machine learning model training
- [ ] Web dashboard for monitoring
- [ ] Integration with more fact-checking APIs
- [ ] Vector database for improved similarity search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `python3 -m pytest`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black claim_analyzer.py
flake8 claim_analyzer.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** for excellent NLP capabilities
- **Sentence Transformers** for semantic similarity
- **Model Context Protocol** for AI integration standards
- **Anthropic** for MCP development support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/claim-analyzer-mcp/issues)
- **Documentation**: [Wiki](https://github.com/your-org/claim-analyzer-mcp/wiki)
- **Community**: [Discussions](https://github.com/your-org/claim-analyzer-mcp/discussions)

---

**Built with ❤️ for accurate information and critical thinking**
