# 📚 MCP YGGDRASIL - PROJECT CONTEXT FOR CLAUDE
**Last Updated**: 2025-07-16 | **Development Week**: 1 of 12 | **Overall Progress**: 15%

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

### **Primary Tech Stack**: Python, Rust (Qdrant), Cypher (Neo4j), TypeScript (Streamlit)

---

## 🎯 PROJECT OVERVIEW

**MCP Yggdrasil** is an enterprise-grade knowledge management system that combines graph databases, vector search, and AI agents to create a sophisticated academic knowledge network.

### **Core Architecture**
- **Neo4j**: Knowledge graph for complex relationships (371+ concepts)
- **Qdrant**: Vector search for semantic queries (7 collections)
- **Redis**: High-performance caching and session management
- **FastAPI**: Async REST API with performance optimization
- **Streamlit**: Interactive database management UI (8 pages)

### **Knowledge Structure - The Yggdrasil Tree**
```
🌳 Root (6 Domains) → Branch (Subjects) → Limb (Entities) → Leaf (Works/Ideas)
```
- **Domains**: Art, Language, Mathematics, Philosophy, Science, Technology
- **Concepts**: Ideas and theoretical constructs (371+ indexed)
- **Relationships**: Cross-domain connections and influences
- **Events**: Historical occurrences that shaped knowledge

### **AI Agent Ecosystem (20+ Agents)**
1. **Data Acquisition**: Scraper, YouTube transcript, copyright checker
2. **Analysis & Validation**: Text processor, fact verifier, anomaly detector
3. **Database Management**: Neo4j/Qdrant managers, sync coordinator

---

## 📊 PROJECT STATUS SUMMARY

### **Actual Development Progress**
| Phase | Status | Actual % | Target Week | Key Missing Components |
|-------|--------|----------|-------------|------------------------|
| **Phase 1: Foundation** | 🔄 Active | 75% | Week 2 | pip-tools, 8 large files, performance metrics |
| **Phase 2: Performance** | ⏳ Partial | 30% | Week 4 | Enhanced AI agents, Celery, full monitoring |
| **Phase 3: Scraper** | ✅ Mostly Complete | 85% | Week 6 | Plugin architecture, selenium-stealth |
| **Phase 4: Validation** | ⏳ Pending | 0% | Week 8 | All components |
| **Phase 5: UI** | ⏳ Pending | 0% | Week 10 | psutil already installed, graph editor |
| **Phase 6: Advanced** | ⏳ Pending | 0% | Week 12 | All components |

**Overall Completion: 30% of 12-week roadmap**

### **Critical Issues Requiring Immediate Action**
1. **Dependencies**: No pip-tools implementation (requirements.in files missing)
2. **Large Files**: 8+ files over 1000 lines need refactoring
3. **Duplicate Files**: Multiple "2.py" files in scraper folder need cleanup
4. **Phase Gaps**: Missing enhanced AI agents from Phase 2
5. **No Performance Metrics**: Baseline metrics not established

---

## 🗂️ NEW MODULAR PLAN STRUCTURE

### **How to Navigate the Development Plan**
```
📁 Project Root/
├── 📄 plan.md                    # Quick overview & navigation
├── 📁 updates/                   # Detailed implementation guides
│   ├── 01_foundation_fixes.md    # Week 1-2: Technical debt
│   ├── 02_performance_optimization.md # Week 3-4: Performance
│   ├── 03_scraper_enhancement.md # Week 5-6: Scraping
│   ├── 04_data_validation.md     # Week 7-8: Validation
│   ├── 05_ui_workspace.md        # Week 9-10: UI fixes
│   ├── 06_technical_specs.md     # Architecture reference
│   ├── 07_metrics_timeline.md    # KPIs & timeline
│   ├── 08_repository_structure.md # File system map
│   └── 09_implementation_status.md # Progress tracking
├── 📄 p_completed.md             # Completed work archive
└── 📁 chat_logs/                 # Session documentation
```

### **Efficient Workflow with Modular Files**
1. **Start**: Read `plan.md` for current priorities
2. **Deep Dive**: Open specific phase file for implementation details
3. **Reference**: Check `08_repository_structure.md` before creating files
4. **Track**: Update `09_implementation_status.md` after completing tasks
5. **Archive**: Move completed items to `p_completed.md`

---

## 🔄 MANDATORY SESSION WORKFLOW

### **Step 1: Project Analysis & Verification** 📊
```bash
# Read in this exact order:
1. claude.md          # Current context (this file)
2. plan.md           # Overview of all phases
3. updates/09_implementation_status.md # Current progress
4. updates/[current_phase].md # Full implementation details

# Verify implementation status:
- Check actual files exist for claimed completions
- Test imports and functionality
- Update percentages based on reality
```

### **Step 2: Task Planning** ✅
```python
# Create TODO list for session
TodoWrite([
    {"id": "fix_psutil_import", "status": "pending", "priority": "critical"},
    {"id": "implement_pip_tools", "status": "pending", "priority": "high"},
    {"id": "refactor_network_analyzer", "status": "pending", "priority": "medium"}
])
```

### **Step 3: Documentation** 📝
```bash
# Create session log
Path: /Users/grant/Documents/GitHub/MCP_Ygg/chat_logs/
Format: YYYY-MM-DD_HH-MM_phase-X-task-description.md
```

### **Step 4: Duplicate Prevention** 🔍
```bash
# MANDATORY before creating ANY file
Grep pattern="[feature_name]" glob="**/*.py"
LS path="[target_directory]"

# If similar functionality exists:
- Enhance existing file
- Do NOT create duplicate
```

### **Step 5: Implementation** 🔨
- Follow phase-specific checklist from `updates/` files
- Update TODO status as you progress
- Test each component before marking complete
- Do not skip anything in the phase files from `updates/` - Don't be lazy, read the entire file

### **Step 6: Progress Update** 📈
- Update `09_implementation_status.md`
- Move completed items to `p_completed.md`
- Document any blockers or issues
- Update this file (claude.md-> "Next Steps" & "Project Status") once tasks are complete and once phases are complete so we can pick
up from where we left off in another session just in case

---

## 🚀 IMMEDIATE PRIORITIES (Week 1)

### **Critical Fixes Required**
1. **Install psutil**: Fix Operations Console crash
   ```bash
   pip install psutil>=5.9.0
   # Add to requirements.txt with version
   ```

2. **Implement pip-tools**: Manage dependencies properly
   ```bash
   # See updates/01_foundation_fixes.md for full implementation
   pip install pip-tools
   pip-compile requirements.in
   ```

3. **Start Refactoring**: Break down large files
   - `analytics/network_analyzer.py` (1,711 lines)
   - `streamlit_workspace/existing_dashboard.py` (1,617 lines)

### **Today's Focus**
- [ ] Read `updates/01_foundation_fixes.md` completely
- [ ] Fix psutil installation
- [ ] Create `requirements.in` with pip-tools
- [ ] Begin refactoring one large file
- [ ] Update implementation status

---

## 📁 KEY PROJECT PATHS

### **Essential Directories**
```
/Users/grant/Documents/GitHub/MCP_Ygg/
├── agents/           # 20+ AI agents (check before creating new)
├── api/             # FastAPI backend (enhanced in Phase 2)
├── cache/           # Redis caching (needs implementation)
├── CSV/             # 371+ concepts across 6 domains
├── streamlit_workspace/ # UI with 8 pages
├── tests/           # Test suite (needs expansion)
└── updates/         # NEW: Modular implementation plans
```

### **Command Reference**
```bash
# Start UI
streamlit run streamlit_workspace/main_dashboard.py

# Start API
python api/fastapi_main.py

# Run tests (when implemented)
pytest tests/

# Clean CSV data
python scripts/csv_cleanup_script.py
```

---

## 📈 SUCCESS METRICS

### **Week 1-2 Targets (Foundation)**
- [ ] Dependencies managed with pip-tools
- [ ] Repository size reduced by 70MB
- [ ] All files under 500 lines
- [ ] 50% test coverage achieved
- [ ] Redis caching operational

### **Performance Goals (Week 3-4)**
- [ ] API response <500ms (currently 2-3s)
- [ ] Cache hit rate >85% (currently <50%)
- [ ] Memory usage <1GB (currently 2-3GB)

### **Knowledge Base Growth**
- Current: 371 concepts, 1,200 relationships
- 3-Month Target: 1,000 concepts, 5,000 relationships
- Required Growth: +5 concepts/day, +42 relationships/day

---

## ⚠️ CRITICAL WARNINGS

### **Before Creating ANY File**
1. Check `updates/08_repository_structure.md`
2. Search for existing similar functionality
3. Enhance existing files rather than creating new ones

### **Known Duplicate Directories**
- `analytics/` exists in both root and `agents/`
- Use `agents/analytics/` for new analytics agents

### **Do NOT Skip**
- Reading complete phase specifications before implementation
- Verifying claimed completions against actual code
- Testing imports and functionality
- Updating progress tracking

---

## 🎯 NEXT STEPS
### **Session Completed** (2025-07-16): Status verification complete
- ✅ Verified actual vs claimed progress for all phases
- ✅ Updated CLAUDE.md with accurate percentages
- ✅ Identified critical gaps and missing components
- ✅ Found psutil already in requirements.txt
- ✅ Found Redis caching already implemented
- ✅ Found testing framework with comprehensive conftest.py

### **Priority Actions for Next Session**
1. **Phase 1 Completion** (25% remaining):
   - Implement pip-tools with requirements.in files
   - Refactor 8 large files (1000+ lines each)
   - Establish performance baseline metrics
   - Clean up duplicate "2.py" files

2. **Phase 2 Enhancement** (70% remaining):
   - Create enhanced AI agents (claim analyzer, text processor)
   - Complete Prometheus monitoring setup
   - Implement Celery async task queue
   - Add structured JSON logging

3. **Phase 3 Finalization** (15% remaining):
   - Complete plugin architecture
   - Finalize selenium-stealth integration

---

**🚀 Project Status**: Week 1 of 12-week roadmap. Critical foundation work needed before proceeding to advanced features. Focus on technical debt elimination and establishing solid testing/monitoring infrastructure.