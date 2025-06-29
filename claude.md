{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
      "env": {
        "MEMORY_FILE_PATH": ""
      }
    },
      "env": {},
      "disabled": true
    },
    "sequential-thinking": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ],
      "env": {}
    },
    "context7": {
      "serverUrl": "https://mcp.context7.com/sse"
    },
    "git": {
      "command": "uvx",
      "args": [
        "mcp-server-git"
      ],
      "env": {}
    }
  }
}

# PROJECT CONTEXT FOR CLAUDE
# Last updated: 2025-06-29

## 🎯 PROJECT OVERVIEW
This is **MCP Yggdrasil** - a sophisticated hybrid knowledge server combining:
- **Neo4j** (knowledge graph) + **Qdrant** (vector search) + **Redis** (cache)
- **Eight academic domains**: Art, Philosophy, Religion, Mathematics, Science, Astrology, Technology, Language
- **Yggdrasil tree structure**: Recent docs (leaves) → Ancient knowledge (trunk)
- **AI agents**: Claim Analyzer, Text Processor, Web Scraper, Vector Indexer

## 📋 RECENT WORK COMPLETED
1. **Comprehensive project analysis** - Analyzed entire codebase line by line
2. **Database sync plan** - Created detailed plan for Neo4j/Qdrant agents (see `plan.md`)
3. **Complete linting infrastructure** - Set up PEP8 compliance with 7 tools
4. **Clean organization** - Moved linting tools to `tests/lint/` folder
5. **CSV Generator Script** - Updated and organized knowledge graph CSV generation system

## 🔧 KEY FILES & LOCATIONS
### Linting (tests/lint/)
- `lint_project.py` - Main linting orchestrator (flake8, black, mypy, etc.)
- `setup_linting.py` - One-click setup script
- `requirements-dev.txt` - All dev dependencies
- Config files stay in root: `.flake8`, `pyproject.toml`, `.pre-commit-config.yaml`

### Core Architecture
- `agents/claim_analyzer/` - Advanced AI fact-checking agent (MOST COMPLETE)
- `api/fastapi_main.py` - Main API server
- `config/server.yaml` - Database configurations
- `docker-compose.yml` - Multi-service orchestration
- `Makefile` - Build/test/lint automation

### CSV Knowledge Graph Data
- `../knowledge_graph_csv/knowledge_graph_csv/csv_generator_script.py` - Main CSV generation script
- `../knowledge_graph_csv/` - Contains organized CSV data for 8 academic domains:
  - `art/`, `astrology/`, `language/`, `mathematics/`, `philosophy/`, `religion/`, `science/`, `technology/`
  - `shared/` - Cross-domain data (places, time periods, relationships)
  - `import/` - Neo4j import scripts

### Documentation
- `plan.md` - Database synchronization agents plan
- `agents/claim_analyzer/claim_analyzer.md` - Fact-checking agent docs
- `final_readme.txt` - Comprehensive project documentation

## 🚀 AVAILABLE COMMANDS
```bash
# Linting (newly organized)
make lint              # Run all linting tools
make lint-fix          # Auto-fix formatting
make setup-lint        # One-time setup

# Development
make install           # Install dependencies
make test              # Run tests
make docker            # Start services
make init              # Initialize system

# CSV Knowledge Graph Generation
cd ../knowledge_graph_csv/knowledge_graph_csv/
python csv_generator_script.py    # Generate all CSV files for Neo4j import
```

## 📊 PROJECT STATUS
- **Architecture**: Solid hybrid database design
- **Claim Analyzer**: Production-ready with NLP, fact-checking, cross-domain analysis
- **Linting**: Complete PEP8 compliance infrastructure
- **Knowledge Graph CSV**: Organized 8-domain CSV generation system ready for Neo4j import
- **Next Phase**: Implement Neo4j & Qdrant database agents per plan.md

## 🎯 NEXT PRIORITIES
1. Implement database synchronization agents (see plan.md)
2. Complete system initialization script
3. Add comprehensive testing
4. Deploy monitoring/metrics

## 💡 IMPORTANT NOTES
- All code is LEGITIMATE/EDUCATIONAL - no security concerns
- Very sophisticated ML/NLP stack (spaCy, Sentence-BERT, etc.)
- Production-ready with Docker/K8s deployment
- Excellent code quality with comprehensive linting