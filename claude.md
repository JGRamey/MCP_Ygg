{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
      "env": {
        "MEMORY_FILE_PATH": "/Users/grant/Documents/GitHub/MCP_Ygg/chat_logs/memory.json"
      }
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

### Use the following Context7 libraries for coding: Python, Rust (Qdrant), Cypher (Neo4j), and any other libraries if necessary but those are the essential libraries. ###

# PROJECT CONTEXT FOR CLAUDE
# Last updated: 2025-07-01 (✅ COMPLETE IDE-LIKE STREAMLIT WORKSPACE DEPLOYED)

## 🎯 PROJECT OVERVIEW
This is **MCP Yggdrasil** - a sophisticated hybrid knowledge server combining:
- **Neo4j** (knowledge graph) + **Qdrant** (vector search) + **Redis** (cache)
- **Six primary academic domains**: Art, Language, Mathematics, Philosophy (includes Religion), Science (includes Astrology), Technology
- **Yggdrasil tree structure**: Recent docs (leaves) → Ancient knowledge (trunk)
- **AI agents**: Claim Analyzer, Text Processor, Web Scraper, Vector Indexer
- **Database management interface**: Streamlit-based interface for database operations and content scraping

## 📋 RECENT WORK COMPLETED
1. **✅ STREAMLIT DATABASE INTERFACE** - Database management and content scraping interface
2. **✅ DATABASE MANAGEMENT INTERFACE** - Full CRUD operations with visual concept cards and relationship management
3. **✅ INTERACTIVE GRAPH VISUALIZATION** - NetworkX + Plotly integration with multiple layout algorithms
4. **✅ CSV DATA MANAGEMENT** - CSV editor for database content, data validation
5. **✅ DATABASE OPERATIONS CONSOLE** - Cypher query editor, database monitoring
6. **✅ CONTENT SCRAPING INTERFACE** - Multi-source content acquisition and processing
7. **✅ DATA VALIDATION TOOLS** - Quality assessment and validation for database content
8. **✅ STREAMLIT UI FRAMEWORK** - Clean interface for database and scraping operations
9. **✅ HYBRID DATABASE DEPLOYMENT** - Complete Neo4j + Qdrant + Redis system deployment
10. **✅ MODULAR PLAN ARCHITECTURE** - Comprehensive plan.md restructure with modular design patterns
11. **✅ PROGRESS TRACKING SYSTEM** - New workflow with p_completed.md for completed tasks management
12. **✅ DATA INTEGRATION & CLEANUP** - 371 concepts standardized across 6 primary domains

## 🔧 KEY FILES & LOCATIONS
### Linting (tests/lint/)
- `lint_project.py` - Main linting orchestrator (flake8, black, mypy, etc.)
- `setup_linting.py` - One-click setup script
- `requirements-dev.txt` - All dev dependencies
- Config files stay in root: `.flake8`, `pyproject.toml`, `.pre-commit-config.yaml`

### Core Architecture
- `streamlit_workspace/` - **COMPLETE IDE-LIKE WORKSPACE** - Professional Streamlit application
  - `main_dashboard.py` - Main navigation and system status
  - `pages/01_🗄️_Database_Manager.py` - Full CRUD operations with visual interfaces
  - `pages/02_📊_Graph_Editor.py` - Interactive network visualization with multiple layouts
  - `pages/03_📁_File_Manager.py` - Project file browser, CSV editor, Git integration
  - `pages/04_⚡_Operations_Console.py` - Cypher queries, monitoring, service control
  - `pages/05_🎯_Knowledge_Tools.py` - Advanced knowledge engineering and quality tools
  - `pages/06_📈_Analytics.py` - Comprehensive analytics and reporting dashboard
  - `utils/` - Database operations and session management utilities
- `agents/claim_analyzer/` - Advanced AI fact-checking agent
- `api/fastapi_main.py` - FastAPI backend server
- `config/server.yaml` - Database configurations
- `docker-compose.yml` - Multi-service orchestration (Neo4j + Qdrant + Redis + RabbitMQ)
- `Makefile` - Build/test/lint automation

### CSV Knowledge Graph Data (CLEANED & PRODUCTION-READY)
- `CSV/` - **Main cleaned CSV structure** ready for Neo4j import:
  - Standard domains: `art/`, `language/`, `mathematics/`, `philosophy/`, `science/`, `technology/`
  - Subdomains: `philosophy/religion/`, `science/pseudoscience/astrology/`
  - `shared/` - Cross-domain data (places, time periods, relationships)
  - `import/` - Neo4j import scripts
  - `sources/` - Document metadata for Qdrant integration
  - `vectors/` - Neo4j↔Qdrant sync metadata
- `scripts/csv_cleanup_script.py` - **Production-ready cleanup tool**
- `CSV_CLEANUP_SUMMARY.md` - **Detailed cleanup results and metrics**

### Documentation
- `plan.md` - **UPDATED**: Comprehensive development plan with modular structure (active tasks only)
- `p_completed.md` - **NEW**: Completed tasks and implementations (moved from plan.md)
- `data_validation_pipeline_plan.md` - **NEW**: Multi-agent intelligent data validation plan
- `agents/claim_analyzer/claim_analyzer.md` - Fact-checking agent docs
- `final_readme.txt` - Comprehensive project documentation
- `CSV_CLEANUP_SUMMARY.md` - Detailed data integration and cleanup results

### Chat Logs & Session Management
- `chat_logs/` - **NEW**: Organized chat logs by date/time with action summaries
- `scripts/chat_logger.py` - **NEW**: Automated chat logging system with timestamps

### Project Planning & Progress
- `plan.md` - **ACTIVE**: Current and pending implementation tasks with modular structure
- `p_completed.md` - **NEW**: Archive of completed tasks and implementations
- `prompt.md` - **NEW**: Modular coding guidelines and best practices for implementation

## 🚀 AVAILABLE COMMANDS
```bash
# Database Management Interface (PRIMARY INTERFACE)
streamlit run main_dashboard.py --server.port 8502    # Launch database management interface
# Access at: http://localhost:8502

# Linting (newly organized)
make lint              # Run all linting tools
make lint-fix          # Auto-fix formatting
make setup-lint        # One-time setup

# Development
make install           # Install dependencies
make test              # Run tests
make docker            # Start services
make init              # Initialize system

# CSV Management & Cleanup
python scripts/csv_cleanup_script.py    # Clean and standardize all CSV files
python scripts/enhanced_yggdrasil_integrator.py    # Integration script (already completed)

# Chat Logging & Session Management
python scripts/chat_logger.py    # Test chat logging functionality
# Note: Chat logs automatically created in chat_logs/ directory by date/time
```
