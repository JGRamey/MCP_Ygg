# 📚 MCP YGGDRASIL - PROJECT CONTEXT FOR CLAUDE
**Last Updated**: 2025-07-15 | **Phase**: 2.0 Performance Optimized | **Status**: Production Ready

## 🏗️ MCP SERVER CONFIGURATION

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "/Users/grant/Documents/GitHub/MCP_Ygg/chat_logs/memory.json"
      }
    },
    "sequential-thinking": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
      "env": {}
    },
    "context7": {
      "serverUrl": "https://mcp.context7.com/sse"
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"],
      "env": {}
    }
  }
}
```

### Use the following Context7 libraries for coding: Python, Rust (Qdrant), Cypher (Neo4j), and any other libraries if necessary but those are the essential libraries.

---

## 🎯 PROJECT OVERVIEW

**MCP Yggdrasil** is a sophisticated hybrid knowledge server combining multiple cutting-edge technologies:

### **Core Technology Stack**
- **Neo4j** - Knowledge graph database for complex relationships
- **Qdrant** - Vector search engine for semantic queries  
- **Redis** - High-performance caching layer
- **FastAPI** - Modern async API framework
- **Streamlit** - Interactive database management interface

### **Knowledge Domains**
**Six Primary Academic Domains**:
- **Art** - Visual arts, literature, music, cultural expressions
- **Language** - Linguistics, etymology, translation, communication
- **Mathematics** - Pure & applied mathematics, logic, computation
- **Philosophy** - Ethics, metaphysics, epistemology, religion
- **Science** - Natural sciences, research, astronomy, pseudoscience
- **Technology** - Engineering, computer science, innovations

### **Yggdrasil Tree Structure**
```
Root (Categories) → Branch (Subjects) → Limb (Groups/People/Places) → Leaf (Individual Works/Ideas)
```
- **Root**: Main Categories/Subjects
- **Branch**: Subjects/fields within main categories
- **Limb**: Groups, cultures, civilizations, people, places, ideas
- **Leaf**: Individual works, texts, accomplishments, doctrines
- **Properties**: Attributes, qualities, information about each node
- **Relationships**: Connections between nodes (Father of, Student of, Inspired by, etc.)

### **AI Agent Architecture**
**Three-Tier Functional Organization**:
1. **Scraping Process**: Web scraper, YouTube transcript, text processor
2. **Data Analysis**: Fact verifier, anomaly detector, pattern recognition
3. **Database Management**: Neo4j manager, Qdrant manager, sync manager


---

## 🏆 MAJOR ACHIEVEMENTS - PHASE 1 COMPLETE

### **✅ PHASE 1: FOUNDATION & REFACTORING** (100% Complete)

**Massive Code Refactoring Achievement**:
- **7 major files refactored**: 10,000+ lines → 31 modular components
- **Repository size reduced**: ~70MB cleanup
- **All critical infrastructure**: Modularized and production-ready

#### **Completed Refactoring Projects**:

1. **✅ Network Analysis** (1,712 lines → 11 modules)
   - Core analyzer, centrality analysis, community detection
   - Influence analysis, bridge analysis, flow analysis
   - Structural analysis, clustering, path analysis, visualization

2. **✅ Trend Analysis** (1,010 lines → 7 modules)  
   - Advanced statistical analysis, seasonality detection
   - Prediction engine, data collectors, visualization

3. **✅ Streamlit Dashboard** (1,617 lines → 6 components + shared library)
   - Config management, UI components, page renderers
   - Data operations, search operations, ~1,200 lines shared library

4. **✅ Content Scraper** (1,508 lines → 4 modules, 94.6% reduction)
   - Main interface, scraping engine, content processors
   - Submission manager with comprehensive workflow

5. **✅ Knowledge Tools** (1,385 lines → 5 modules, 89% reduction)
   - Concept builder, quality assurance, knowledge analytics
   - AI recommendations, relationship manager

6. **✅ Visualization Agent** (1,026 lines → 13 modules, 92.6% reduction)
   - Core models, data processors, layout engines
   - Template management, export handlers

7. **✅ Anomaly Detector** (768 lines → modular structure)

#### **Infrastructure Achievements**:
- **✅ Dependency Management**: Modular pip-tools setup
- **✅ Repository Cleanup**: 70MB reduction, clean .gitignore
- **✅ Backup System**: All original files preserved in archive/
- **✅ Chat Logging**: Automated session documentation
- **✅ Progress Tracking**: p_completed.md system
- **✅ Data Integration**: 371 concepts across 6 domains

---

## 🚀 CURRENT STATUS - PHASE 2 IMPLEMENTATION

### **✅ PHASE 2: PERFORMANCE OPTIMIZATION** (Recently Completed)

**Version 2.0.0 - "Phase 2 Performance Optimized"**

#### **Core Enhancements Implemented**:

1. **✅ Enhanced FastAPI Application**
   - **Performance Middleware**: Custom timing headers (x-process-time)
   - **Security Integration**: OAuth2/JWT with audit logging
   - **Cache Integration**: Redis with health checks and warming
   - **Middleware Stack**: Security → Performance → CORS → Compression

2. **✅ System Integration Excellence**
   - **Graceful Dependency Handling**: Works with missing dependencies
   - **Enhanced Health Check**: Comprehensive system status reporting
   - **Performance Monitoring**: Integrated benchmarking routes
   - **Error Resilience**: Fallbacks for all optional components

3. **✅ Workflow Innovation**
   - **Duplicate Prevention Protocol**: Mandatory scanning before implementation
   - **6-Step Standard Workflow**: Systematic development process
   - **"Enhance vs. Create"**: Integration over duplication principle

---

## 📁 PROJECT STRUCTURE & KEY FILES

### **Core Architecture**
```
MCP_Ygg/
├── streamlit_workspace/          # Database Management Interface
│   ├── main_dashboard.py         # Main navigation (187 lines)
│   ├── shared/                   # Shared component library (~1,200 lines)
│   ├── pages/                    # Streamlit pages (all refactored)
│   └── utils/                    # Database utilities
├── agents/                       # AI Agents (3-tier architecture)
│   ├── scraper/                  # Web scraping agents
│   ├── analytics/                # Data analysis agents  
│   └── *_manager/                # Database management agents
├── api/                          # FastAPI Backend (v2.0.0)
│   ├── fastapi_main.py           # Enhanced main application
│   ├── middleware/               # Security & performance middleware
│   └── routes/                   # API endpoints
├── cache/                        # Redis caching system
├── CSV/                          # Knowledge graph data (production-ready)
├── chat_logs/                    # Session documentation
└── updates/                      # Implementation plans
```

### **Agent Import Patterns**
```python
# Scraping Process Agents
from agents.scraper.scraper_agent import WebScraper
from agents.text_processor.text_processor import TextProcessor

# Data Analysis Agents  
from agents.analytics.anomaly_detector.anomaly_detector import AnomalyDetector
from agents.fact_verifier.enhanced_verification_agent import FactVerifier

# Database Management Agents
from agents.neo4j_manager.neo4j_agent import Neo4jAgent
from agents.qdrant_manager.qdrant_agent import QdrantAgent
```

### **Documentation & Planning**
- **`plan.md`** - Master development plan (active tasks)
- **`p_completed.md`** - Completed implementations archive
- **`updates/`** - Detailed phase implementation plans
- **`chat_logs/`** - Comprehensive session documentation
- **`archive/`** - Original file backups (11 files preserved)

---

## 🚀 AVAILABLE COMMANDS

### **Primary Interfaces**
```bash
# Database Management Interface (Primary)
streamlit run main_dashboard.py --server.port 8502
# Access at: http://localhost:8502

# FastAPI Backend (v2.0.0 Performance Optimized)
python api/fastapi_main.py
# Access at: http://localhost:8000
```

### **Development Commands**
```bash
# Linting & Quality
make lint              # Run all linting tools
make lint-fix          # Auto-fix formatting  
make setup-lint        # One-time setup

# Development Workflow
make install           # Install dependencies
make test              # Run tests
make docker            # Start services (Neo4j + Qdrant + Redis)
make init              # Initialize system

# Data Management
python scripts/csv_cleanup_script.py    # Clean CSV data
```

---

## 🔄 STANDARD SESSION WORKFLOW

**Every session MUST follow this 6-step workflow for consistency and efficiency:**

### **1. PROJECT STATUS ANALYSIS** 📊
```bash
# Read key project files to understand current state
1. Read CLAUDE.md - Project context and recent work
2. Read plan.md - Master development plan overview  
3. Read updates/ - Current phase tasks and status
4. Assess current project maturity and next priorities
```

### **2. TODO LIST CREATION** ✅
```python
# Create structured todo list with priorities
TodoWrite([
    {"id": "task_description", "status": "pending", "priority": "high"},
    {"id": "next_task", "status": "pending", "priority": "medium"}
])
```

### **3. CHAT LOG CREATION** 📝
```bash
# Create timestamped chat log
Location: /Users/grant/Documents/GitHub/MCP_Ygg/chat_logs/
Format: YYYY-MM-DD_HH-MM_session-description.md
```

### **4. DUPLICATE PREVENTION SCAN** 🔍
**⚠️ MANDATORY: Before creating ANY new files or implementing features**

```bash
# 1. Search for existing similar functionality
Grep pattern="[feature_name|middleware|connection|cache|auth]" glob="**/*.py"

# 2. Check specific directories
LS path="/Users/grant/Documents/GitHub/MCP_Ygg/api"
LS path="/Users/grant/Documents/GitHub/MCP_Ygg/agents" 
LS path="/Users/grant/Documents/GitHub/MCP_Ygg/cache"

# 3. Search for existing API routes and main files  
Glob pattern="**/main*.py"
Glob pattern="**/route*.py"

# 4. Check for existing database connections
Grep pattern="AsyncGraphDatabase|ConnectionPool|QdrantClient|redis.*pool" glob="**/*.py"
```

**Before Implementation Checklist:**
- ✅ Scanned for existing files/functions with similar purpose
- ✅ Checked relevant directories (api/, agents/, cache/, middleware/)
- ✅ Verified no duplicate functionality exists
- ✅ **If duplicates found**: Enhance existing files instead of creating new ones
- ✅ Document what exists vs. what needs to be added/modified

### **5. IMPLEMENTATION EXECUTION** 🔨
1. **Complete Duplicate Prevention Scan** - MANDATORY first step
2. **Mark Current Task In Progress** - Update TodoWrite status
3. **Execute Implementation** - Complete work (enhance existing or create new)
4. **Mark Task Complete** - Update TodoWrite status  
5. **Update Chat Log** - Document progress
6. **Move to Next Task** - Repeat cycle

### **6. SESSION CONTINUATION** 🔄
- Update chat log with progress
- Keep TodoWrite current with status changes
- Document issues or blockers
- Set up next session priorities

---

## 📋 AGENT OPERATIONAL GUIDELINES

### **Core Principles**
1. **🔍 MANDATORY Duplicate Prevention**: Always complete Step 4 scan before implementation
2. **🔄 Enhance vs. Create**: Integrate with existing files instead of creating duplicates  
3. **📝 Document Integration**: Record what exists and how new functionality integrates
4. **✅ Follow Standard Workflow**: Use the 6-step process for all sessions
5. **📊 Track Progress**: Update plan.md and p_completed.md appropriately

### **Session Management**
- **New Session Protocol**: Follow 6-step workflow
- **Task Completion**: Mark tasks complete in plan.md and update files
- **Progress Tracking**: Use TodoWrite tool throughout session
- **Documentation**: Create chat logs following established format

### **Quality Assurance**
- **Pre-implementation Check**: Verify current project state
- **Import Verification**: Test all modified files work correctly
- **Graceful Error Handling**: Handle missing dependencies properly
- **Backwards Compatibility**: Preserve existing functionality

---

## 🎯 CURRENT PRIORITIES & NEXT STEPS

### **Immediate Phase 2 Continuation**
1. **Performance Baseline Testing** - Verify <500ms API response targets
2. **Dependency Optimization** - Install missing optional dependencies
3. **Load Testing** - Validate system under production conditions
4. **Enhanced Monitoring** - Prometheus + Grafana setup

### **Future Phase Planning**
- **Phase 3**: Scraper Enhancement & Anti-blocking
- **Phase 4**: Data Validation Pipeline & Multi-agent validation
- **Phase 5**: UI Workspace Development & Advanced features
- **Phase 6**: Production Deployment & Enterprise security

---

**🚀 MCP Yggdrasil Status**: Production-ready Phase 2 performance-optimized system with comprehensive workflow protocols and modular architecture.