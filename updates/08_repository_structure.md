# Current Repository Structure Documentation
## 📁 COMPLETE MCP YGGDRASIL FILE SYSTEM

### Overview
This document provides a comprehensive view of the current repository structure, explaining the purpose of each directory and highlighting key files. Use this as a reference to avoid creating duplicate files or directories.

### ⚠️ CRITICAL NOTE
**Before creating ANY new file or directory, check this document to ensure it doesn't already exist.**

### 📊 Repository Statistics
- **Total Directories**: 50+
- **Total Python Files**: 100+
- **Total CSV Files**: 40+
- **Knowledge Concepts**: 371+
- **Specialized Agents**: 20+
- **Streamlit Pages**: 8
- **Test Files**: 15+

### 🏗️ Complete Directory Structure

```
MCP_Ygg/
│
├── 📁 Root Configuration Files
│   ├── .gitattributes                    # Git attributes configuration
│   ├── .gitignore                        # Git ignore rules (comprehensive)
│   ├── .pre-commit-config.yaml           # Pre-commit hooks configuration
│   ├── Dockerfile                        # Main application Docker image
│   ├── docker-compose.yml                # Docker services orchestration
│   ├── docker-compose.override.yml       # Local development overrides
│   ├── docker-start.sh                   # Docker startup script
│   ├── Makefile                          # Build and deployment automation
│   ├── pyproject.toml                    # Python project configuration
│   ├── requirements.txt                  # Python dependencies (71+ packages)
│   ├── requirements-dev.txt              # Development dependencies
│   ├── start_app.sh                      # Application startup script
│   └── app_main.py                       # Main application entry point
│
├── 📁 Documentation Files
│   ├── CLAUDE_SESSION.md                 # Claude interaction guidelines
│   ├── CONCEPT_PHILOSOPHY.md             # Project philosophy & concepts
│   ├── SESSION_OPTIMIZATION_GUIDE.md     # Optimization strategies
│   ├── claude.md                         # Claude-specific documentation
│   ├── final_readme.txt                  # Project overview
│   ├── plan.md                          # Master development plan
│   └── prompt.md                        # AI interaction prompts
│
├── 📁 .claude/
│   └── settings.local.json              # Claude local settings
│
├── 📁 CSV/ (Knowledge Graph Data - 371+ concepts)
│   ├── 📁 art/                          # Art domain data
│   │   ├── art_concepts.csv             # Art concepts and ideas
│   │   ├── art_people.csv               # Artists and art figures
│   │   ├── art_relationships.csv        # Relationships in art
│   │   └── art_works.csv                # Artworks and creations
│   │
│   ├── 📁 language/                     # Language domain data
│   │   ├── language_concepts.csv        # Linguistic concepts
│   │   ├── language_people.csv          # Linguists and writers
│   │   ├── language_relationships.csv   # Language relationships
│   │   └── language_works.csv           # Literary works
│   │
│   ├── 📁 mathematics/                  # Mathematics domain data
│   │   ├── mathematics_concepts.csv     # Mathematical concepts
│   │   ├── mathematics_people.csv       # Mathematicians
│   │   ├── mathematics_relationships.csv # Mathematical relationships
│   │   └── mathematics_works.csv        # Mathematical works
│   │
│   ├── 📁 philosophy/                   # Philosophy domain data
│   │   ├── philosophy_concepts.csv      # Philosophical concepts
│   │   ├── philosophy_people.csv        # Philosophers
│   │   ├── philosophy_relationships.csv # Philosophical relationships
│   │   ├── philosophy_works.csv         # Philosophical works
│   │   └── 📁 religion/                 # Religion subdomain
│   │       ├── religion_concepts.csv    # Religious concepts
│   │       ├── religion_people.csv      # Religious figures
│   │       ├── religion_relationships.csv # Religious relationships
│   │       └── religion_works.csv       # Religious texts
│   │
│   ├── 📁 science/                      # Science domain data
│   │   ├── science_concepts.csv         # Scientific concepts
│   │   ├── science_people.csv           # Scientists
│   │   ├── science_relationships.csv    # Scientific relationships
│   │   ├── science_works.csv            # Scientific works
│   │   └── 📁 pseudoscience/            # Pseudoscience subdomain
│   │       └── 📁 astrology/            # Astrology data
│   │           ├── astrology_concepts.csv
│   │           ├── astrology_people.csv
│   │           ├── astrology_relationships.csv
│   │           └── astrology_works.csv
│   │
│   ├── 📁 technology/                   # Technology domain data
│   │   ├── technology_concepts.csv      # Tech concepts
│   │   ├── technology_people.csv        # Tech innovators
│   │   ├── technology_relationships.csv # Tech relationships
│   │   └── technology_works.csv         # Tech innovations
│   │
│   ├── 📁 shared/                       # Cross-domain shared data
│   │   ├── cross_domain_relationships.csv # Inter-domain connections
│   │   ├── shared_places.csv            # Geographic locations
│   │   └── shared_time_periods.csv      # Historical periods
│   │
│   ├── 📁 sources/                      # Source material metadata
│   │   ├── 📁 books/
│   │   │   └── book_metadata.csv        # Book sources
│   │   ├── 📁 manuscripts/
│   │   │   └── manuscript_metadata.csv   # Manuscript sources
│   │   ├── 📁 modern_sources/
│   │   │   └── scholarly_articles.csv    # Modern academic sources
│   │   └── 📁 tablets/
│   │       └── cuneiform_tablets.csv     # Ancient tablet sources
│   │
│   ├── 📁 vectors/                      # Vector synchronization
│   │   └── sync_metadata.csv            # Vector sync tracking
│   │
│   └── 📁 import/                       # Import staging area (empty)
│
├── 📁 agents/ (AI Agent Modules - 20+ specialized agents)
│   ├── __init__.py
│   │
│   ├── 🕷️ SCRAPING PROCESS AGENTS
│   ├── 📁 scraper/                      # Web content acquisition
│   │   ├── __init__.py
│   │   ├── IMPORTANT.md                 # Scraper documentation
│   │   ├── high_performance_scraper.py  # Performance-optimized scraper
│   │   ├── scraper_agent.py             # Main scraper agent
│   │   ├── scraper_config.py            # Scraper configuration
│   │   ├── scraper_utils.py             # Utility functions
│   │   └── testscrape.md                # Scraper testing docs
│   │
│   ├── 📁 youtube_transcript/           # YouTube content extraction
│   │   ├── __init__.py
│   │   ├── config.yaml                  # YouTube API config
│   │   ├── metadata_extractor.py        # Video metadata extraction
│   │   ├── transcript_processor.py      # Transcript processing
│   │   ├── youtube_agent.py             # Main YouTube agent
│   │   ├── youtube_agent_efficient.py   # Optimized version
│   │   └── youtube_agent_simple.py      # Simplified version
│   │
│   ├── 📁 copyright_checker/            # Content validation
│   │   ├── __init__.py
│   │   ├── copyright_checker.py         # Copyright validation
│   │   └── 📁 lists/                    # Copyright lists
│   │       └── .gitkeep
│   │
│   ├── 📁 text_processor/               # Text analysis
│   │   ├── __init__.py
│   │   ├── text_processor.py            # Main text processor
│   │   ├── text_processor_config.py     # Configuration
│   │   └── text_processor_utils.py      # Utilities
│   │
│   ├── 🔍 DATA ANALYSIS AGENTS
│   ├── 📁 analytics/                    # Analytics suite
│   │   ├── __init__.py
│   │   ├── base.py                      # Base analytics classes
│   │   ├── community_analysis.py        # Community detection
│   │   ├── complete_trend_analyzer.py   # Trend analysis
│   │   ├── graph_metrics.py             # Graph measurements
│   │   ├── network_analyzer.py          # Network analysis (1,711 lines)
│   │   ├── pattern_detection.py         # Pattern finding
│   │   ├── 📁 plots/                    # Visualization output
│   │   │   └── .gitkeep
│   │   ├── 📁 anomaly_detector/         # Anomaly detection
│   │   │   ├── __init__.py
│   │   │   ├── anomaly_detector.py      # Main detector (768 lines)
│   │   │   └── 📁 models/               # ML models
│   │   │       └── .gitkeep
│   │   ├── 📁 claim_analyzer/           # Fact verification
│   │   │   ├── claim_analyzer.md        # Documentation
│   │   │   ├── claim_analyzer.py        # Claim analysis
│   │   │   └── claim_analyzer_config.py # Configuration
│   │   ├── 📁 concept_explorer/         # Concept relationships
│   │   │   ├── __init__.py
│   │   │   ├── concept_discovery_service.py # Discovery service
│   │   │   ├── concept_explorer.py      # Main explorer
│   │   │   ├── config.yaml              # Configuration
│   │   │   ├── connection_analyzer.py   # Connection analysis
│   │   │   └── thought_path_tracer.py   # Path tracing
│   │   └── 📁 content_analyzer/         # Content analysis
│   │       ├── __init__.py
│   │       ├── config.yaml              # Configuration
│   │       └── content_analysis_agent.py # Analysis agent
│   │
│   ├── 📁 fact_verifier/                # Enhanced fact checking
│   │   ├── __init__.py
│   │   └── enhanced_verification_agent.py # Verification agent
│   │
│   ├── 📁 metadata_analyzer/            # Metadata analysis
│   │   ├── __init__.py
│   │   └── metadata_analyzer.py         # Metadata processing
│   │
│   ├── 📁 pattern_recognition/          # Pattern detection
│   │   ├── __init__.py
│   │   └── pattern_recognition.py       # Pattern algorithms
│   │
│   ├── 📁 recommendation/               # Recommendations
│   │   ├── __init__.py
│   │   └── recommendation_agent.py      # Recommendation engine
│   │
│   ├── 🗄️ DATABASE MANAGEMENT AGENTS
│   ├── 📁 neo4j_manager/                # Neo4j operations
│   │   ├── __init__.py
│   │   ├── config.yaml                  # Neo4j config
│   │   ├── neo4j_agent.py               # Neo4j agent
│   │   ├── query_optimizer.py           # Query optimization
│   │   └── schema_manager.py            # Schema management
│   │
│   ├── 📁 qdrant_manager/               # Qdrant operations
│   │   ├── __init__.py
│   │   ├── collection_manager.py        # Collection management
│   │   ├── config.yaml                  # Qdrant config
│   │   └── qdrant_agent.py              # Qdrant agent
│   │
│   ├── 📁 vector_index/                 # Vector operations
│   │   ├── __init__.py
│   │   ├── vector_index_config.py       # Vector config
│   │   └── vector_indexer.py            # Indexing operations
│   │
│   ├── 📁 sync_manager/                 # DB synchronization
│   │   ├── __init__.py
│   │   ├── config.yaml                  # Sync config
│   │   ├── conflict_resolver.py         # Conflict resolution
│   │   ├── event_dispatcher.py          # Event handling
│   │   └── sync_manager.py              # Sync orchestration
│   │
│   ├── 📁 knowledge_graph/              # Graph construction
│   │   ├── __init__.py
│   │   └── knowledge_graph_builder.py   # Graph builder
│   │
│   ├── 📁 node_relationship_manager/    # Relationship management
│   │   ├── __init__.py
│   │   └── relationship_manager.py      # Relationship ops
│   │
│   ├── 📁 backup/                       # Backup operations
│   │   ├── __init__.py
│   │   └── backup_agent.py              # Backup management
│   │
│   ├── 📁 maintenance/                  # System maintenance
│   │   ├── __init__.py
│   │   └── maintenance_agent.py         # Maintenance tasks
│   │
│   └── 🌍 TRANSLATION AGENTS (Documentation only)
│       ├── ENG-Handwritting2text_agent.md  # English OCR
│       ├── greektranslater.md              # Greek translation
│       ├── hebrewtranslator.md             # Hebrew translation
│       └── latintranslator.md              # Latin translation
│
├── 📁 analytics/                        # Analytics module (duplicate?)
│   ├── __init__.py
│   ├── base.py
│   ├── community_analysis.py
│   ├── complete_trend_analyzer.py
│   ├── graph_metrics.py
│   ├── network_analyzer.py
│   ├── pattern_detection.py
│   └── 📁 plots/
│       └── .gitkeep
│
├── 📁 api/                              # FastAPI application
│   ├── __init__.py
│   ├── fastapi_main.py                  # Main FastAPI app
│   ├── simple_main.py                   # Simplified API
│   ├── 📁 middleware/                   # API middleware
│   │   ├── __init__.py
│   │   └── security_middleware.py       # Security layer
│   └── 📁 routes/                       # API endpoints
│       ├── __init__.py
│       ├── analysis_pipeline.py         # Analysis endpoints
│       ├── api_routes.py                # Core API routes
│       ├── concept_discovery.py         # Concept endpoints
│       ├── content_scraping.py          # Scraping endpoints
│       └── performance_monitoring.py    # Monitoring endpoints
│
├── 📁 cache/                            # Caching system
│   ├── __init__.py
│   ├── cache_manager.py                 # Cache management
│   └── config.py                        # Cache configuration
│
├── 📁 chat_logs/                        # Session logs
│   ├── memory.json                      # Session memory
│   ├── prompt.md                        # Interaction prompts
│   └── [Multiple dated session logs]    # Historical sessions
│
├── 📁 config/                           # Configuration files
│   ├── analysis_pipeline.yaml           # Analysis config
│   ├── content_scraping.yaml            # Scraping config
│   ├── database_agents.yaml             # Database config
│   ├── server.yaml                      # Server config
│   └── visualization.yaml               # Visualization config
│
├── 📁 data/                             # Data management
│   ├── staging_manager.py               # Staging operations
│   ├── 📁 backups/                      # Backup storage
│   │   └── .gitkeep
│   ├── 📁 metadata/                     # Metadata storage
│   │   └── .gitkeep
│   ├── 📁 processed/                    # Processed data
│   │   └── .gitkeep
│   ├── 📁 raw/                          # Raw data
│   │   └── .gitkeep
│   └── 📁 staging/                      # Staging workflow
│       ├── README.md                    # Staging docs
│       ├── 📁 analyzed/                 # Analyzed content
│       │   └── example-analyzed-content.json
│       ├── 📁 approved/                 # Approved content
│       │   └── example-approved-content.json
│       ├── 📁 pending/                  # Pending content
│       │   └── example-youtube-submission.json
│       ├── 📁 processing/               # Processing queue
│       └── 📁 rejected/                 # Rejected content
│
├── 📁 k8s/                              # Kubernetes configs
│   ├── k8s-deployment.yaml.txt         # K8s deployment
│   └── 📁 monitoring/                   # Monitoring setup
│       └── prometheus-grafana.yaml.txt  # Prometheus config
│
├── 📁 opus_update/                      # Update documentation
│   ├── UIplan.md                       # UI development plan
│   ├── analysis.md                     # System analysis
│   ├── critical_implementation.md      # Critical features
│   ├── data_validation_pipeline_plan.md # Validation plan
│   ├── refactoring.md                  # Refactoring guide
│   └── scraper_update.md               # Scraper updates
│
├── 📁 scripts/                          # Utility scripts
│   ├── chat_logger.py                   # Chat logging
│   ├── csv_cleanup_script.py           # CSV maintenance
│   ├── dependency_validation.py.txt     # Dependency check
│   ├── enhanced_yggdrasil_integrator.py # Enhanced integration
│   ├── initialize_system.py             # System initialization
│   ├── run_tests.py                     # Test runner
│   └── yggdrasil_integrator.py         # Basic integration
│
├── 📁 streamlit_workspace/              # Streamlit UI
│   ├── __init__.py
│   ├── existing_dashboard.py            # Legacy dashboard (1,617 lines)
│   ├── main_dashboard.py                # Main UI entry
│   ├── 📁 data/                         # UI data
│   │   └── 📁 staging/                  # Staging mirror
│   ├── 📁 pages/                        # Streamlit pages
│   │   ├── 01_🗄️_Database_Manager.py   # Database operations
│   │   ├── 02_📊_Graph_Editor.py        # Graph visualization
│   │   ├── 03_📁_File_Manager.py        # File management
│   │   ├── 03_📁_File_Manager_Old.py    # Legacy file manager
│   │   ├── 04_⚡_Operations_Console.py  # System operations
│   │   ├── 05_🎯_Knowledge_Tools.py     # Knowledge tools
│   │   ├── 06_📈_Analytics.py           # Analytics dashboard
│   │   ├── 07_📥_Content_Scraper.py     # Content scraping
│   │   └── 08_🔄_Processing_Queue.py    # Queue management
│   └── 📁 utils/                        # UI utilities
│       ├── database_operations.py       # DB operations
│       └── session_management.py        # Session handling
│
├── 📁 summaries/                        # Project summaries
│   ├── CSV_CLEANUP_SUMMARY.md          # CSV cleanup docs
│   ├── FULL_STACK_DEPLOYMENT_SUMMARY.md # Deployment guide
│   └── HYBRID_ARCHITECTURE_SUMMARY.md   # Architecture docs
│
├── 📁 tests/                            # Test suite
│   ├── __init__.py
│   ├── test_csv_import.py               # CSV import tests
│   ├── test_hybrid_system.py            # System tests
│   ├── 📁 integration/                  # Integration tests
│   │   └── test_integration.py
│   ├── 📁 lint/                         # Linting setup
│   │   ├── __init__.py
│   │   ├── ORGANIZATION.md              # Lint organization
│   │   ├── README.md                    # Lint readme
│   │   ├── lint_project.py              # Project linter
│   │   └── setup_linting.py             # Lint setup
│   ├── 📁 performance/                  # Performance tests
│   │   └── performance_optimization.py
│   └── 📁 unit/                         # Unit tests
│       └── test_scraper.py
│
├── 📁 agents/visualization/             # Visualization module (moved to agents/)
│   ├── __init__.py
│   ├── visualization_agent.py           # Viz agent (76 lines - refactored)
│   ├── 📁 output/                       # Output files
│   │   └── .gitkeep
│   └── 📁 templates/                    # Viz templates
│       └── .gitkeep
│
└── 📁 Test Files (Root Level)
    ├── test_*.py                        # Multiple test files
    └── dashboard_backup_*.tar.gz        # Backup archives (remove)
```

### 📌 Key Files to Remember

#### Large Files Needing Refactoring
1. **analytics/network_analyzer.py** - 1,711 lines
2. **streamlit_workspace/existing_dashboard.py** - 1,617 lines
3. **visualization/visualization_agent.py** - 1,026 lines
4. **agents/anomaly_detector/anomaly_detector.py** - 768 lines

#### Critical Configuration Files
1. **requirements.txt** - 71+ dependencies (needs cleanup)
2. **docker-compose.yml** - Service orchestration
3. **pyproject.toml** - Project configuration
4. **.gitignore** - Comprehensive ignore rules

#### Entry Points
1. **app_main.py** - Main application
2. **api/fastapi_main.py** - API server
3. **streamlit_workspace/main_dashboard.py** - UI entry

### 🚨 Common Mistakes to Avoid

#### DO NOT Create These (They Already Exist):
- ❌ `analytics/` directory in agents/ (duplicate exists at root)
- ❌ New test directories (use existing `tests/` structure)
- ❌ Additional config files (use existing `config/` directory)
- ❌ New agent directories without checking `agents/` first
- ❌ Cache directories (use existing `cache/` module)

#### Files That Should Be Removed:
- 🗑️ `venv/` directory (42.6 MB)
- 🗑️ `__pycache__/` directories
- 🗑️ `dashboard_backup_*.tar.gz` files
- 🗑️ `.pyc` files

### 📋 Directory Purposes

#### Core Functionality
- **agents/**: All AI agents organized by function
- **api/**: FastAPI REST endpoints
- **streamlit_workspace/**: Interactive UI

#### Data Storage
- **CSV/**: Knowledge graph source data
- **data/**: Processed and staged content
- **cache/**: Redis caching implementation

#### Configuration
- **config/**: YAML configuration files
- **.claude/**: Claude-specific settings
- **k8s/**: Kubernetes deployment configs

#### Development
- **tests/**: Comprehensive test suite
- **scripts/**: Utility and maintenance scripts
- **opus_update/**: Enhancement documentation

#### Analysis & Visualization
- **analytics/**: Data analysis algorithms
- **agents/visualization/**: Graph and data visualization

### 🔍 Quick Reference

#### To Find...
- **Scraping code**: `agents/scraper/`
- **Database operations**: `agents/neo4j_manager/` and `agents/qdrant_manager/`
- **API endpoints**: `api/routes/`
- **UI pages**: `streamlit_workspace/pages/`
- **Configuration**: `config/` directory
- **Tests**: `tests/` directory
- **Documentation**: Root `.md` files and `opus_update/`

#### Before Adding...
- **New agent**: Check `agents/` subdirectories
- **New test**: Use appropriate `tests/` subdirectory
- **New config**: Add to existing `config/` files
- **New UI page**: Add to `streamlit_workspace/pages/`
- **New API endpoint**: Add to `api/routes/`

---

*This structure documentation is critical for maintaining project organization and preventing duplicate work.*