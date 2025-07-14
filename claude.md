### NEEDS TO BE UPDATED WITH THE CURRENT CLAUDE MEMORY FILE ###


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
- **Yggdrasil tree structure**: Root = Main Categories/Subjects -> Branch = Subjects/fields within the main Categories/Fields -> Limb = Groups/Cultures/Civilizations, people,  place, ideas, etc. within particular branches -> Leaf = Individual works, texts, accomplishments,   doctrines, etc. -> 
Properties = Attributes, Qualities, Information, etc. or any important details about each individual node (Nodes are all things just mentioned, Groups, People, Ideas, Texts, Manuscripts, Categories, Fields, Subjects, etc. Everything in the Neo4j graph is considered a node unless it's a relationship) -> 
Relationships = Related nodes to some capacity eg. Father of, Mother of, Brother of, Sister of, etc. and Student of, Teacher of, Creator/Founder of, Inspired/Influenced by, etc. and for places here are some examples we can expand on, Birthplace of (person, religion, doctrine, idea, art style, language, etc), battle of, etc.
Please ask if you need clarity or ideas when it comes to creating and labeling nodes.
- **AI agents**: Text Processor, Web Scraper & Text Scraper,  Vector Indexer, Graph Manager, Data/Information analysis agents
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
12. **✅ AGENT REORGANIZATION** - Functional categorization: Scraping → Analysis → Database Management
13. **✅ ANOMALY DETECTOR REFACTORING** - Modular structure (768→242 lines) with comprehensive tests
14. **✅ DATA INTEGRATION & CLEANUP** - 371 concepts standardized across 6 primary domains

## 🔧 KEY FILES & LOCATIONS
### Linting (tests/lint/)
- `lint_project.py` - Main linting orchestrator (flake8, black, mypy, etc.)
- `setup_linting.py` - One-click setup script
- `requirements-dev.txt` - All dev dependencies
- Config files stay in root: `.flake8`, `pyproject.toml`, `.pre-commit-config.yaml`
- Updated plan.md file for updates and improvements: C:\Users\zochr\Desktop\GitHub\Yggdrasil\MCP_Ygg\plan.md

### Core Architecture
- `streamlit_workspace/` - **DATABASE MANAGEMENT INTERFACE** - Professional Streamlit application
  - `main_dashboard.py` - Main navigation and system status
  - `pages/01_🗄️_Database_Manager.py` - Full CRUD operations with visual interfaces
  - `pages/02_📊_Graph_Editor.py` - Interactive network visualization with multiple layouts
  - `pages/03_📁_File_Manager.py` - CSV data editor, database content management
  - `pages/04_⚡_Operations_Console.py` - Cypher queries, monitoring, service control
  - `pages/05_🎯_Knowledge_Tools.py` - Advanced knowledge engineering and quality tools
  - `pages/06_📈_Analytics.py` - Comprehensive analytics and reporting dashboard
  - `utils/` - Database operations and session management utilities
- `agents/` - **FUNCTIONALLY ORGANIZED AGENTS** - Three-tier architecture:
  - **Scraping Process** (`scraper/`, `youtube_transcript/`, `copyright_checker/`, `text_processor/`)
  - **Data Analysis** (`analytics/`, `fact_verifier/`, `metadata_analyzer/`, `pattern_recognition/`, `recommendation/`)
  - **Database Management** (`neo4j_manager/`, `qdrant_manager/`, `sync_manager/`, `knowledge_graph/`, `backup/`)
- `api/fastapi_main.py` - FastAPI backend server
- `config/server.yaml` - Database configurations
- `docker-compose.yml` - Multi-service orchestration (Neo4j + Qdrant + Redis + RabbitMQ)
- `Makefile` - Build/test/lint automation

### Agent Import Patterns & Dependencies
**⚠️ IMPORTANT**: Agent organization follows functional workflow:
```python
# Scraping Process Agents
from agents.scraper.scraper_agent import WebScraper
from agents.youtube_transcript.youtube_agent import YouTubeAgent
from agents.text_processor.text_processor import TextProcessor

# Data Analysis Agents  
from agents.analytics.anomaly_detector.anomaly_detector import AnomalyDetector
from agents.analytics.claim_analyzer.claim_analyzer import ClaimAnalyzer
from agents.analytics.concept_explorer.concept_explorer import ConceptExplorer
from agents.fact_verifier.enhanced_verification_agent import FactVerifier

# Database Management Agents
from agents.neo4j_manager.neo4j_agent import Neo4jAgent
from agents.qdrant_manager.qdrant_agent import QdrantAgent
from agents.sync_manager.sync_manager import SyncManager
from agents.knowledge_graph.knowledge_graph_builder import KnowledgeGraphBuilder
```

**Agent Workflow Dependencies**:
1. **Scraping → Analysis → Database**: Linear data flow
2. **Cross-functional**: Analytics agents can call database agents
3. **Modular**: Each agent group can operate independently

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

