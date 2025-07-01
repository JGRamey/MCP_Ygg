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

### Use the following Context7 libraries for coding: Python, Rust (Qdrant), Cypher (Neo4j), and any other libraries if necessary but those are the essential libraries. ###

# PROJECT CONTEXT FOR CLAUDE
# Last updated: 2025-07-01 (Updated with multi-agent validation pipeline plan)

## 🎯 PROJECT OVERVIEW
This is **MCP Yggdrasil** - a sophisticated hybrid knowledge server combining:
- **Neo4j** (knowledge graph) + **Qdrant** (vector search) + **Redis** (cache)
- **Eight academic domains**: Art, Philosophy, Religion, Mathematics, Science, Astrology, Technology, Language
- **Yggdrasil tree structure**: Recent docs (leaves) → Ancient knowledge (trunk)
- **AI agents**: Claim Analyzer, Text Processor, Web Scraper, Vector Indexer
- **Next-gen pipeline**: Multi-agent validation system for academic-grade content verification

## 📋 RECENT WORK COMPLETED
1. **Comprehensive project analysis** - Analyzed entire codebase line by line
2. **Database sync plan** - Created detailed plan for Neo4j/Qdrant agents (see `plan.md`)
3. **Complete linting infrastructure** - Set up PEP8 compliance with 7 tools
4. **Clean organization** - Moved linting tools to `tests/lint/` folder
5. **CSV Generator Script** - Updated and organized knowledge graph CSV generation system
6. **✅ DESKTOP YGGDRASIL DATA INTEGRATION** - Successfully imported Excel taxonomies into CSV structure
7. **✅ CSV CLEANUP & STANDARDIZATION** - Removed 237 duplicates, standardized IDs, fixed malformed entries
8. **✅ MULTI-AGENT VALIDATION PIPELINE PLAN** - Comprehensive 12-week plan for intelligent data validation

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
- `plan.md` - Database synchronization agents plan (Neo4j ↔ Qdrant sync)
- `data_validation_pipeline_plan.md` - **NEW**: Multi-agent intelligent data validation plan
- `agents/claim_analyzer/claim_analyzer.md` - Fact-checking agent docs
- `final_readme.txt` - Comprehensive project documentation
- `CSV_CLEANUP_SUMMARY.md` - Detailed data integration and cleanup results

### Chat Logs & Session Management
- `chat_logs/` - **NEW**: Organized chat logs by date/time with action summaries
- `scripts/chat_logger.py` - **NEW**: Automated chat logging system with timestamps

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

# CSV Management & Cleanup
python scripts/csv_cleanup_script.py    # Clean and standardize all CSV files
python scripts/enhanced_yggdrasil_integrator.py    # Integration script (already completed)

# Chat Logging & Session Management
python scripts/chat_logger.py    # Test chat logging functionality
# Note: Chat logs automatically created in chat_logs/ directory by date/time
```

## 📊 PROJECT STATUS
- **Architecture**: Solid hybrid database design
- **Claim Analyzer**: Production-ready with NLP, fact-checking, cross-domain analysis
- **Linting**: Complete PEP8 compliance infrastructure
- **✅ Data Integration**: Desktop Yggdrasil Excel taxonomies successfully imported
- **✅ CSV Structure**: Clean, standardized, 371 concepts across 8 domains ready for Neo4j
- **✅ Data Quality**: 100% duplicate removal, consistent DOMAIN#### IDs, malformed entries fixed
- **Next Phase**: Neo4j import and Qdrant vector database setup

## 🎯 NEXT PRIORITIES (Two Parallel Tracks)

### **Track A: Database Infrastructure (plan.md)**
1. **Neo4j Database Import** - Load cleaned CSV data using enhanced Cypher scripts
2. **Qdrant Vector Database** - Set up document metadata and vector indexing  
3. **Database Synchronization Agents** - Implement Neo4j↔Qdrant sync agents

### **Track B: Intelligent Data Pipeline (data_validation_pipeline_plan.md)**
1. **Enhanced Web Scraper** - Add intelligence layer with JSON staging
2. **Multi-Agent Validation** - Content analysis, fact verification, quality assessment
3. **Academic Cross-Referencing** - Authoritative source validation system

### **Track C: Integration & Production**
1. **System Testing** - Validate hybrid architecture end-to-end
2. **Pipeline Integration** - Connect validation pipeline to database infrastructure
3. **Production Deployment** - Complete system with monitoring and alerts

## 💡 IMPORTANT NOTES
- All code is LEGITIMATE/EDUCATIONAL - no security concerns
- Very sophisticated ML/NLP stack (spaCy, Sentence-BERT, etc.)
- Production-ready with Docker/K8s deployment
- Excellent code quality with comprehensive linting

## 📈 DATA INTEGRATION COMPLETED (2025-07-01)

### Desktop Yggdrasil Integration Results:
- **Source**: Excel schema files from `/Users/grant/Desktop/Yggdrasil/`
- **Target**: Enhanced CSV structure in `/Users/grant/Documents/GitHub/MCP_Ygg/CSV/`
- **Method**: Custom integration script with domain-specific mapping
- **Result**: 371 unique concepts successfully integrated across all 8 domains

### CSV Cleanup Results:
- **Before**: 608 total concepts (with duplicates and malformed entries)
- **After**: 371 unique concepts (237 duplicates removed = 58% reduction)
- **ID Format**: Standardized to DOMAIN#### (e.g., ART0001, PHIL0001, SCI0001)
- **Quality**: 100% data integrity, all relationships updated, malformed entries fixed
- **Structure**: Enhanced for hybrid Neo4j+Qdrant architecture

### Domain Breakdown (Final):
- **Art**: 50 concepts (Visual Arts, Architecture, Performing Arts, etc.)
- **Language**: 40 concepts (Linguistic categories and subcategories)
- **Mathematics**: 58 concepts (Pure, Applied, Interdisciplinary)
- **Philosophy**: 30 concepts (Metaphysics, Ethics, Logic, etc.)
- **Science**: 65 concepts (Physics, Chemistry, Biology, etc.)
- **Technology**: 8 concepts (Ancient technologies)
- **Religion**: 104 concepts (Monotheistic, Polytheistic, Non-theistic traditions)
- **Astrology**: 16 concepts (Pseudoscience subcategory)

### Key Scripts Created:
- `scripts/enhanced_yggdrasil_integrator.py` - Excel to CSV integration tool
- `scripts/csv_cleanup_script.py` - Production-ready cleanup and standardization
- `CSV_CLEANUP_SUMMARY.md` - Detailed metrics and validation results

**STATUS**: ✅ Data integration phase complete. Ready for Neo4j import and Qdrant setup.

## 🚀 MULTI-AGENT VALIDATION PIPELINE PLAN (2025-07-01)

### Strategic Decision: Academic-Grade Data Validation
Created comprehensive plan for transforming MCP Yggdrasil from basic knowledge storage into an **intelligent academic validation system** with multi-agent pipeline ensuring only verified, cross-referenced content enters the knowledge graph.

### Pipeline Architecture (5 Agents):
1. **Enhanced Web Scraper** - Intelligence layer with source authority scoring
2. **Content Analysis Agent** - NLP processing using existing spaCy/BERT stack
3. **Fact Verification Agent** - Cross-reference engine with authoritative academic sources
4. **Quality Assessment Agent** - Reliability scoring with confidence levels (High/Medium/Low)
5. **Knowledge Integration Agent** - Transaction-safe database updates with provenance

### Key Innovation: JSON Staging Area
- **Staging workflow**: pending → processing → verified/flagged/rejected
- **Full audit trail** for all content processing decisions
- **Manual review queue** for edge cases requiring human oversight
- **Quality gates** preventing low-confidence content from entering database

### Academic Rigor Features:
- **Multi-source cross-referencing** against authoritative databases
- **Citation validation** for academic reference verification
- **Expert consensus checking** within each domain
- **Contradiction detection** against existing knowledge graph
- **Provenance tracking** for complete audit capability

### Integration with Existing Infrastructure:
- **✅ Leverages Claim Analyzer**: Enhanced for cross-referencing capabilities
- **✅ Uses NLP Stack**: spaCy, Sentence-BERT for content analysis
- **✅ Domain Taxonomy**: 8-domain classification system for content mapping
- **✅ Neo4j + Qdrant**: Transaction-safe integration with hybrid database

### Implementation Timeline:
- **12-week phased approach** with 5 implementation phases
- **Week 1-2**: Enhanced Web Scraper with JSON staging
- **Week 3-4**: Content Analysis Agent development
- **Week 5-6**: Enhanced Fact Verification system
- **Week 7-8**: Quality Assessment Agent
- **Week 9-12**: Integration, testing, and production deployment

### Success Metrics:
- **80%+ content** achieving >0.8 reliability score
- **<5% false positive rate** for approved content
- **>95% citation accuracy** for academic references
- **<15% manual review rate** for efficient processing

### Quality Assurance:
- **Confidence-based integration**: High (auto-approve), Medium (review), Low (reject)
- **Multi-layer validation**: Source authority + Cross-reference + Citations + Expert consensus
- **Academic standards**: Peer-review quality validation process

**PLAN STATUS**: ✅ Complete comprehensive plan ready for implementation alongside database infrastructure.

## 📝 CHAT LOG SYSTEM IMPLEMENTATION (2025-07-01)

### Automated Session Tracking:
Created comprehensive chat logging system to maintain conversation history and track development progress across sessions.

### Chat Log Features:
- **Organized Storage**: `chat_logs/` directory with files named by date/time (YYYY-MM-DD_HH-MM.md)
- **Structured Format**: Markdown format with timestamps, participants, and action summaries
- **Action Tracking**: Code changes and file modifications summarized between dialogue
- **Session Continuity**: Each session includes summary and full conversation history
- **Automated Creation**: `scripts/chat_logger.py` provides programmatic logging functionality

### Log Structure:
```markdown
# MCP Yggdrasil Chat Log
**Session Date:** YYYY-MM-DD
**Session Time:** HH:MM
**Participants:** JGR (User), Claude (Assistant)

## Session Summary
[High-level summary of session achievements]

## Chat Log
### [HH:MM:SS] JGR:
[User message]

### [HH:MM:SS] Claude:
*ACTION: [Summary of code/file changes]*
[Assistant response]
```

### Integration Benefits:
- **Project Continuity**: Complete conversation history preserved across sessions
- **Development Tracking**: Action summaries provide quick overview of changes made
- **Knowledge Transfer**: New team members can review decision-making process
- **Audit Trail**: Full documentation of project evolution and reasoning

**CHAT LOG STATUS**: ✅ System implemented and ready for continuous session tracking.

## 🔄 MANDATORY SESSION PROTOCOL

### **CRITICAL REMINDER FOR ALL FUTURE SESSIONS:**
When Claude starts any new session, the **FIRST ACTION** must be to create a new chat log file using the format:
- **File**: `chat_logs/YYYY-MM-DD_HH-MM.md`
- **Initialize**: Session header with date, time, participants
- **Log**: Every message exchange with timestamps and action summaries
- **Update**: Session summary as work progresses

### **Session Logging Requirements:**
1. **Immediate Setup**: Create log file before any other work
2. **Real-time Updates**: Log each JGR/Claude exchange with actions
3. **Action Documentation**: Summarize all code changes and file modifications
4. **Session Continuity**: Complete audit trail for project development

**This protocol ensures NO LOSS of conversation history and maintains complete project continuity across all sessions.**