# Current Repository Structure Documentation
## 📁 COMPLETE MCP YGGDRASIL FILE SYSTEM - UPDATED 2025-07-23

### Overview
This document provides a comprehensive view of the current repository structure, explaining the purpose of each directory and highlighting key files. Use this as a reference to avoid creating duplicate files or directories.

### ⚠️ CRITICAL NOTE
**Before creating ANY new file or directory, check this document to ensure it doesn't already exist.**

### 📊 Repository Statistics (Updated)
- **Total Directories**: 85+ (including nested)
- **Total Python Files**: 120+
- **Total CSV Files**: 40+
- **Knowledge Concepts**: 371+
- **Specialized Agents**: 25+ (enhanced with Phase 2 improvements)
- **Streamlit Pages**: 8
- **Test Files**: 20+
- **Archive Files**: 15+ (cleaned old implementations)

### 🏗️ Complete Directory Structure

```
MCP_Ygg/
│
├── 📁 Root Configuration Files
│   ├── .gitattributes                    # Git attributes configuration
│   ├── .gitignore                        # Git ignore rules (comprehensive)
│   ├── Dockerfile                        # Main application Docker image
│   ├── docker-compose.yml                # Docker services orchestration
│   ├── docker-compose.override.yml       # Local development overrides
│   ├── docker-start.sh                   # Docker startup script
│   ├── Makefile                          # Build and deployment automation
│   ├── pyproject.toml                    # Python project configuration
│   ├── requirements.txt                  # Python dependencies (managed by pip-tools)
│   ├── requirements.in                   # Production dependencies source
│   ├── requirements-dev.txt              # Development dependencies
│   ├── requirements-dev.in               # Dev dependencies source
│   ├── start_app.sh                      # Application startup script
│   ├── app_main.py                       # Main application entry point
│   ├── run_tests.py                      # Test runner script
│   └── print_repo_structure.py           # Repository structure generator
│
├── 📁 Documentation Files
│   ├── CONCEPT_PHILOSOPHY.md             # Project philosophy & concepts
│   ├── SESSION_OPTIMIZATION_GUIDE.md     # Optimization strategies
│   ├── claude.md                         # Claude context & workflow (CRITICAL)
│   ├── final_readme.txt                  # Project overview
│   ├── plan.md                           # Master development plan
│   └── p_completed.md                    # Completed work archive
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
│   └── 📁 import/                       # Import commands
│       ├── neo4j_import_commands.cypher  # Basic import commands
│       └── enhanced_neo4j_import_commands.cypher # Enhanced import
│
├── 📁 agents/ (AI Agent Modules - 25+ specialized agents) ✅ REORGANIZED
│   ├── __init__.py
│   │
│   ├── 🕷️ SCRAPING PROCESS AGENTS
│   ├── 📁 scraper/                      # Web content acquisition
│   │   ├── __init__.py
│   │   ├── IMPORTANT.md                 # Scraper documentation
│   │   ├── testscrape.md                # Scraper testing docs
│   │   ├── scraper_agent.py             # Main scraper agent
│   │   ├── scraper_config.py            # Scraper configuration
│   │   ├── scraper_utils.py             # Utility functions
│   │   ├── high_performance_scraper.py  # Performance-optimized scraper
│   │   ├── unified_web_scraper.py       # Unified scraping interface
│   │   ├── enhanced_content_extractor.py # Trafilatura integration
│   │   ├── anti_detection.py            # Anti-blocking measures
│   │   ├── scraper_profiles.py          # Configurable profiles
│   │   ├── site_specific_parsers.py     # Site-specific parsers
│   │   ├── structured_data_extractor.py # Structured data extraction
│   │   ├── multi_source_acquisition.py # Multi-source content
│   │   └── advanced_language_detector.py # Language detection
│   │
│   ├── 📁 youtube_transcript/           # YouTube content extraction
│   │   ├── __init__.py
│   │   ├── config.yaml                  # YouTube API config
│   │   ├── youtube_agent.py             # Main YouTube agent
│   │   ├── youtube_agent_efficient.py   # Optimized version
│   │   ├── youtube_agent_simple.py      # Simplified version
│   │   ├── metadata_extractor.py        # Video metadata extraction
│   │   └── transcript_processor.py      # Transcript processing
│   │
│   ├── 📁 copyright_checker/            # Content validation
│   │   ├── __init__.py
│   │   ├── copyright_checker.py         # Copyright validation
│   │   └── 📁 lists/                    # Copyright lists
│   │
│   ├── 📁 text_processor/               # Text analysis ✅ ENHANCED
│   │   ├── __init__.py
│   │   ├── enhanced_text_processor.py   # Enhanced multilingual processor
│   │   ├── enhanced_config.yaml         # Enhanced configuration
│   │   └── text_processor_utils.py      # Utility functions
│   │
│   ├── 🔍 DATA ANALYSIS AGENTS
│   ├── 📁 analytics/                    # Analytics suite ✅ REFACTORED
│   │   ├── __init__.py
│   │   ├── base.py                      # Base analytics classes
│   │   ├── 📁 plots/                    # Visualization output
│   │   ├── 📁 concept_explorer/         # Concept relationships
│   │   │   ├── __init__.py
│   │   │   ├── concept_explorer.py      # Main explorer
│   │   │   ├── concept_discovery_service.py # Discovery service
│   │   │   ├── connection_analyzer.py   # Connection analysis
│   │   │   ├── thought_path_tracer.py   # Path tracing
│   │   │   └── config.yaml              # Configuration
│   │   ├── 📁 content_analyzer/         # Content analysis
│   │   │   ├── __init__.py
│   │   │   ├── content_analysis_agent.py # Analysis agent
│   │   │   └── config.yaml              # Configuration
│   │   ├── 📁 graph_analysis/           # Network analysis ✅ MODULARIZED
│   │   │   ├── __init__.py
│   │   │   ├── README_update.md         # Update documentation
│   │   │   ├── analysis.py              # Main analysis
│   │   │   ├── community_analysis.py    # Community detection
│   │   │   ├── config.py                # Configuration
│   │   │   ├── graph_metrics.py         # Graph measurements
│   │   │   ├── graph_utils.py           # Utility functions
│   │   │   ├── models.py                # Data models
│   │   │   ├── pattern_detection.py     # Pattern finding
│   │   │   ├── improve_community_an.md  # Improvement notes
│   │   │   ├── improve_trend_an.md      # Trend improvement notes
│   │   │   ├── 📁 network_analysis/     # Network analysis modules
│   │   │   │   ├── __init__.py
│   │   │   │   ├── core_analyzer.py     # Core analysis
│   │   │   │   ├── bridge_analysis.py   # Bridge detection
│   │   │   │   ├── centrality_analysis.py # Centrality measures
│   │   │   │   ├── clustering_analysis.py # Clustering algorithms
│   │   │   │   ├── community_detection.py # Community algorithms
│   │   │   │   ├── flow_analysis.py     # Flow analysis
│   │   │   │   ├── influence_analysis.py # Influence measures
│   │   │   │   ├── network_visualization.py # Network visualizations
│   │   │   │   ├── path_analysis.py     # Path algorithms
│   │   │   │   └── structural_analysis.py # Structural analysis
│   │   │   └── 📁 trend_analysis/       # Trend analysis modules
│   │   │       ├── __init__.py
│   │   │       ├── core_analyzer.py     # Core trend analysis
│   │   │       ├── data_collectors.py   # Data collection
│   │   │       ├── predictor.py         # Prediction algorithms
│   │   │       ├── seasonality_detector.py # Seasonality detection
│   │   │       ├── statistics_engine.py # Statistical analysis
│   │   │       ├── trend_detector.py    # Trend detection
│   │   │       └── trend_visualization.py # Trend visualizations
│   │
│   ├── 📁 claim_analyzer/               # Enhanced fact checking ✅ ENHANCED
│   │   ├── __init__.py
│   │   ├── README.md                    # Documentation
│   │   ├── claim_analyzer.py            # Main analyzer
│   │   ├── checker.py                   # Enhanced checker with multi-source
│   │   ├── config.yaml                  # Configuration
│   │   ├── database.py                  # Database operations
│   │   ├── exceptions.py                # Custom exceptions
│   │   ├── extractor.py                 # Claim extraction
│   │   ├── models.py                    # Data models
│   │   └── utils.py                     # Utility functions
│   │
│   ├── 📁 enhanced_verification/        # Multi-source verification
│   │   ├── __init__.py
│   │   └── multi_source_verifier.py     # Verification agent
│   │
│   ├── 📁 enhanced_reasoning/           # Advanced reasoning
│   │   └── langchain_integration.py     # LangChain integration
│   │
│   ├── 📁 metadata_analyzer/            # Metadata analysis
│   │   ├── __init__.py
│   │   └── metadata_analyzer.py         # Metadata processing
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
│   │   ├── cached_neo4j_agent.py        # Cached operations
│   │   ├── query_optimizer.py           # Query optimization
│   │   ├── schema_manager.py            # Schema management
│   │   ├── updates.md                   # Update documentation
│   │   ├── 📁 knowledge_graph/          # Knowledge graph ✅ MOVED HERE
│   │   │   ├── __init__.py
│   │   │   └── knowledge_graph_builder.py # Graph builder
│   │   └── 📁 node_relationship_manager/ # Relationship management ✅ MOVED HERE
│   │       ├── __init__.py
│   │       └── relationship_manager.py  # Relationship ops
│   │
│   ├── 📁 qdrant_manager/               # Qdrant operations
│   │   ├── __init__.py
│   │   ├── config.yaml                  # Qdrant config
│   │   ├── qdrant_agent.py              # Qdrant agent
│   │   ├── cached_qdrant_agent.py       # Cached operations
│   │   ├── collection_manager.py        # Collection management
│   │   └── 📁 vector_index/             # Vector operations ✅ MOVED HERE
│   │       ├── __init__.py
│   │       ├── vector_index_config.py   # Vector config
│   │       └── vector_indexer.py        # Indexing operations
│   │
│   ├── 📁 sync_manager/                 # DB synchronization
│   │   ├── __init__.py
│   │   ├── config.yaml                  # Sync config
│   │   ├── sync_manager.py              # Sync orchestration
│   │   ├── conflict_resolver.py         # Conflict resolution
│   │   └── event_dispatcher.py          # Event handling
│   │
│   ├── 📁 backup/                       # Backup operations
│   │   ├── __init__.py
│   │   └── backup_agent.py              # Backup management
│   │
│   ├── 📁 maintenance/                  # System maintenance
│   │   ├── __init__.py
│   │   ├── maintenance_agent.py         # Maintenance tasks
│   │   └── 📁 logs/                     # Maintenance logs
│   │
│   ├── 📁 visualization/                # Visualization ✅ REFACTORED
│   │   ├── __init__.py
│   │   ├── visualization_agent.py       # Main visualization agent
│   │   ├── 📁 core/                     # Core visualization
│   │   │   ├── __init__.py
│   │   │   ├── chart_generator.py       # Chart generation
│   │   │   ├── config.py                # Configuration
│   │   │   └── models.py                # Data models
│   │   ├── 📁 exporters/                # Export functionality
│   │   │   ├── __init__.py
│   │   │   └── html_exporter.py         # HTML export
│   │   ├── 📁 layouts/                  # Layout algorithms
│   │   │   ├── __init__.py
│   │   │   ├── force_layout.py          # Force-directed layout
│   │   │   └── yggdrasil_layout.py      # Yggdrasil tree layout
│   │   ├── 📁 processors/               # Data processors
│   │   │   ├── __init__.py
│   │   │   ├── data_processor.py        # Data processing
│   │   │   ├── network_processor.py     # Network processing
│   │   │   └── yggdrasil_processor.py   # Yggdrasil processing
│   │   ├── 📁 templates/                # Visualization templates
│   │   │   ├── __init__.py
│   │   │   ├── template_manager.py      # Template management
│   │   │   └── visjs_template.html      # Vis.js template
│   │   └── 📁 output/                   # Output files
│   │
│   ├── 📁 anomaly_detector/             # Anomaly detection
│   │   └── 📁 logs/                     # Anomaly logs
│   │
│   ├── 📁 concept_explorer/             # Concept exploration (empty - functionality moved)
│   │
│   └── 🌍 TRANSLATION AGENTS (Documentation only)
│       ├── ENG-Handwritting2text_agent.md  # English OCR
│       ├── greektranslater.md              # Greek translation
│       ├── hebrewtranslator.md             # Hebrew translation
│       └── latintranslator.md              # Latin translation
│
├── 📁 api/                              # FastAPI application ✅ ENHANCED
│   ├── __init__.py
│   ├── fastapi_main.py                  # Main FastAPI app (v2.0.0)
│   ├── simple_main.py                   # Simplified API
│   ├── 📁 middleware/                   # API middleware
│   │   ├── __init__.py
│   │   └── security_middleware.py       # Security layer
│   └── 📁 routes/                       # API endpoints
│       ├── __init__.py
│       ├── api_routes.py                # Core API routes
│       ├── analysis_pipeline.py         # Analysis endpoints
│       ├── concept_discovery.py         # Concept endpoints
│       ├── content_scraping.py          # Scraping endpoints
│       └── performance_monitoring.py    # Monitoring endpoints
│
├── 📁 archive/                          # Archived/deprecated files ✅ ORGANIZED
│   ├── 01_database_manager_original.py.bak
│   ├── 02_graph_editor_original.py.bak
│   ├── 03_file_manager_original.py.bak
│   ├── 04_operations_console_original.py.bak
│   ├── 05_knowledge_tools_original.py.bak
│   ├── 06_analytics_original.py.bak
│   ├── 08_processing_queue_original.py.bak
│   ├── main_dashboard_original_backup.py.bak
│   ├── main_dashboard_current.py.bak
│   ├── network_analyzer.py.bak
│   ├── trend_analyzer_original.py.bak
│   ├── visualization_agent_original.py.bak
│   ├── text_processor_original.py.bak   # Archived enhanced text processor
│   ├── text_processor_config.py         # Original config file
│   ├── claude.md.bak                    # Claude backup
│   ├── claude2.md.bak                   # Claude backup 2
│   └── 📁 claim_analyzer_old/           # Old claim analyzer files
│       ├── MIGRATION_SUMMARY.md
│       ├── claim_analyzer.md
│       ├── enhanced_checker.py
│       ├── migrate_config.py
│       └── test_refactor.py
│
├── 📁 cache/                            # Caching system ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── cache_manager.py                 # Redis cache management
│   ├── config.py                        # Cache configuration
│   └── integration_manager.py           # Cache integration
│
├── 📁 chat_logs/                        # Session logs ✅ COMPREHENSIVE
│   ├── memory.json                      # Session memory ✅ CRITICAL WORKFLOW
│   ├── prompt.md                        # Interaction prompts
│   ├── 2025-07-01_09-57.md
│   ├── 2025-07-01_15-59.md
│   ├── 2025-07-01_20-30.md
│   ├── 2025-07-03_current.md
│   ├── 2025-07-03_enhanced-content-pipeline.md
│   ├── 2025-07-04_11-00.md
│   ├── 2025-07-04_12-00.md
│   ├── 2025-07-06_11-33_implementation-completion.md
│   ├── 2025-07-06_12-00_project-updates-continuation.md
│   ├── 2025-07-06_concept-discovery-implementation.md
│   ├── 2025-07-06_scraping-performance-optimization.md
│   ├── 2025-07-07_session-continuation_implementation-status.md
│   ├── 2025-07-08_16-00_plan_md_update_session.md
│   ├── 2025-07-08_16-25.md
│   ├── 2025-07-08_22-15_agent-refactoring-continuation.md
│   ├── 2025-07-08_22-54_claim-analyzer-refactoring-completion.md
│   ├── 2025-07-14_12-00_graph-analysis-refactoring.md
│   ├── 2025-07-14_14-30_streamlit-dashboard-refactoring.md
│   ├── 2025-07-14_18-15_streamlit-backup-and-continuation.md
│   ├── 2025-07-14_20-30_knowledge-tools-refactoring-completion.md
│   ├── 2025-07-15_05-39_visualization-agent-refactoring-completion.md
│   ├── 2025-07-15_06-06_phase1-completion-and-phase2-planning.md
│   ├── 2025-07-15_19-00_phase2-performance-optimization-implementation.md
│   ├── 2025-07-15_20-00_project-status-analysis-and-documentation-update.md
│   ├── 2025-07-16_21-30_context7-mcp-configuration-fix.md
│   ├── 2025-07-16_21-40session-phase2-continuation.md
│   ├── 2025-07-16_phase3-scraper-enhancement.md
│   ├── 2025-07-21_implementation-status-verification-update.md
│   ├── 2025-07-22_08-45_phase1-completion-session.md
│   ├── 2025-07-22_13-49_phase-2-claim-analyzer-enhancement.md
│   └── 2025-07-23_10-30_phase2-enhanced-ai-agents.md
│
├── 📁 config/                           # Configuration files
│   ├── analysis_pipeline.yaml           # Analysis config
│   ├── content_scraping.yaml            # Scraping config
│   ├── database_agents.yaml             # Database config
│   ├── server.yaml                      # Server config
│   └── visualization.yaml               # Visualization config
│
├── 📁 data/                             # Data management
│   ├── staging_manager.py               # Staging operations ✅ WELL-STRUCTURED
│   ├── admin_password.txt               # Admin credentials
│   ├── users.json                       # User data
│   ├── 📁 backups/                      # Backup storage
│   ├── 📁 cache/                        # Data cache
│   ├── 📁 metadata/                     # Metadata storage
│   ├── 📁 processed/                    # Processed data
│   ├── 📁 raw/                          # Raw data
│   ├── 📁 uploads/                      # Upload staging
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
├── 📁 dependencies/                     # Dependency management ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── cli.py                           # Command line interface
│   ├── config.py                        # Configuration
│   ├── requirements_manager.py          # Requirements management
│   └── validators.py                    # Dependency validation
│
├── 📁 docs/                             # Documentation (empty)
│
├── 📁 k8s/                              # Kubernetes configs
│   ├── k8s-deployment.yaml.txt         # K8s deployment
│   ├── 📁 application/                  # Application configs
│   ├── 📁 databases/                    # Database configs
│   └── 📁 monitoring/                   # Monitoring setup
│       └── prometheus-grafana.yaml.txt  # Prometheus config
│
├── 📁 monitoring/                       # Monitoring infrastructure ✅ PARTIAL
│   ├── grafana_dashboard.json           # Grafana dashboard
│   ├── prometheus_config.yml            # Prometheus configuration
│   └── setup_monitoring.py             # Monitoring setup script
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
├── 📁 streamlit_workspace/              # Streamlit UI ✅ REFACTORED
│   ├── __init__.py
│   ├── main_dashboard.py                # Main UI entry (orchestrator)
│   ├── 📁 assets/                       # Static assets
│   │   └── 📁 components/               # Component assets
│   ├── 📁 components/                   # Shared UI components ✅ MODULAR
│   │   ├── __init__.py
│   │   ├── config_management.py         # Configuration components
│   │   ├── data_operations.py           # Data operation components
│   │   ├── page_renderers.py            # Page rendering
│   │   ├── search_operations.py         # Search components
│   │   └── ui_components.py             # UI elements
│   ├── 📁 data/                         # UI data
│   │   └── 📁 staging/                  # Staging mirror
│   │       ├── 📁 analyzed/
│   │       ├── 📁 approved/
│   │       ├── 📁 pending/
│   │       ├── 📁 processing/
│   │       └── 📁 rejected/
│   ├── 📁 pages/                        # Streamlit pages ✅ MODULARIZED
│   │   ├── 01_🗄️_Database_Manager.py   # Database operations
│   │   ├── 02_📊_Graph_Editor.py        # Graph visualization
│   │   ├── 03_📁_File_Manager.py        # File management
│   │   ├── 04_⚡_Operations_Console.py  # System operations
│   │   ├── 05_🎯_Knowledge_Tools.py     # Knowledge tools
│   │   ├── 06_📈_Analytics.py           # Analytics dashboard
│   │   ├── 07_📥_Content_Scraper.py     # Content scraping
│   │   ├── 08_🔄_Processing_Queue.py    # Queue management
│   │   ├── 📁 content_scraper/          # Content scraper modules
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   └── 📁 knowledge_tools/          # Knowledge tools modules
│   │       ├── __init__.py
│   │       ├── ai_recommendations.py    # AI recommendations
│   │       ├── concept_builder.py       # Concept building
│   │       ├── knowledge_analytics.py   # Knowledge analytics
│   │       ├── quality_assurance.py     # Quality assurance
│   │       ├── relationship_manager.py  # Relationship management
│   │       └── shared_utils.py          # Shared utilities
│   ├── 📁 shared/                       # Shared components ✅ MODULAR
│   │   ├── __init__.py
│   │   ├── 📁 config/                   # Configuration components
│   │   ├── 📁 data/                     # Data components
│   │   │   ├── __init__.py
│   │   │   └── processors.py            # Data processors
│   │   ├── 📁 search/                   # Search components
│   │   │   ├── __init__.py
│   │   │   └── text_search.py           # Text search
│   │   └── 📁 ui/                       # UI components
│   │       ├── __init__.py
│   │       ├── cards.py                 # Card components
│   │       ├── forms.py                 # Form components
│   │       ├── headers.py               # Header components
│   │       ├── sidebars.py              # Sidebar components
│   │       └── styling.py               # Styling utilities
│   ├── 📁 static/                       # Static files
│   ├── 📁 templates/                    # Templates
│   └── 📁 utils/                        # UI utilities
│       ├── database_operations.py       # DB operations
│       └── session_management.py        # Session handling
│
├── 📁 tests/                            # Test suite ✅ COMPREHENSIVE
│   ├── __init__.py
│   ├── conftest.py                      # Pytest configuration (532 lines)
│   ├── test_cache_system.py             # Cache system tests
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
│   ├── 📁 performance/                  # Performance tests ✅ IMPLEMENTED
│   │   ├── baseline_metrics_simple.py   # Simple baseline
│   │   ├── establish_baseline_metrics.py # Baseline establishment
│   │   ├── performance_optimization.py  # Performance optimization
│   │   └── 📁 baseline_results/         # Baseline results
│   │       ├── baseline_metrics_20250722_084753.json
│   │       └── baseline_report_20250722_084753.md
│   └── 📁 unit/                         # Unit tests
│       ├── test_anomaly_detector.py     # Anomaly detector tests
│       ├── test_cached_agents.py        # Cached agents tests
│       ├── test_network_analysis.py     # Network analysis tests
│       ├── test_scraper.py              # Scraper tests
│       └── test_trend_analysis.py       # Trend analysis tests
│
├── 📁 updates/                          # Implementation guides ✅ MODULAR
│   ├── 01_foundation_fixes.md           # Week 1-2: Technical debt
│   ├── 02_performance_optimization.md   # Week 3-4: Performance
│   ├── 03_scraper_enhancement.md        # Week 5-6: Scraping
│   ├── 04_data_validation.md            # Week 7-8: Validation
│   ├── 05_ui_workspace.md               # Week 9-10: UI fixes
│   ├── 06_technical_specs.md            # Architecture reference
│   ├── 07_metrics_timeline.md           # KPIs & timeline
│   ├── 08_repository_structure.md       # File system map ✅ THIS FILE
│   ├── 09_implementation_status.md      # Progress tracking
│   └── 📁 refactoring/                  # Refactoring documentation
│       ├── prompt.md                    # Refactoring prompts
│       ├── refactoring.md               # Refactoring guide
│       ├── streamlit_backup_summary.md  # Streamlit backup info
│       └── streamlit_refactoring_summary.md # Streamlit refactoring
│
└── 📁 Test Files (Root Level)
    ├── test_aristotle_performance.py     # Aristotle performance test
    ├── test_aristotle_scraping_only.py   # Aristotle scraping test
    ├── test_concept_discovery.py         # Concept discovery test
    ├── test_scraper_performance.py       # Scraper performance test
    └── test_youtube_4hour.py             # YouTube 4-hour test
```

### 📌 Key Changes & Updates (2025-07-23)

#### ✅ Major Reorganizations
1. **Agent Structure Reorganized**:
   - `knowledge_graph/` moved into `neo4j_manager/`
   - `node_relationship_manager/` moved into `neo4j_manager/`
   - `vector_index/` moved into `qdrant_manager/`
   - This provides better logical grouping by database type

2. **Enhanced AI Agents** (Phase 2 Complete):
   - `text_processor/` now has enhanced multilingual processor
   - `claim_analyzer/` enhanced with multi-source verification
   - Original files archived properly

3. **Archive Directory** properly organized:
   - All deprecated files moved to `/archive/`
   - Proper `.bak` extensions maintained
   - Migration summaries included

#### 📊 Updated Statistics
- **Enhanced Agents**: 25+ (up from 20+)
- **Directories**: 85+ (including nested)
- **Archive Files**: 15+ organized files

#### 🔄 Current Status
- **Phase 1**: 95% Complete (foundation solid)
- **Phase 2**: 80% Complete (2 of 3 enhanced agents done)
- **Phase 3**: 85% Complete (scraper enhancements done)

### 🚨 Critical Files to Remember

#### Files That Control Everything
1. **`claude.md`** - Project context & mandatory workflow
2. **`chat_logs/memory.json`** - Session workflow memory
3. **`updates/08_repository_structure.md`** - This file (check before creating files)
4. **`updates/09_implementation_status.md`** - Progress tracking

#### Entry Points
1. **`app_main.py`** - Main application
2. **`api/fastapi_main.py`** - API server (v2.0.0)
3. **`streamlit_workspace/main_dashboard.py`** - UI entry

#### Configuration
1. **`requirements.in`** / **`requirements-dev.in`** - Dependency sources
2. **`pyproject.toml`** - Project configuration
3. **Config files** in `/config/` directory

### 🚫 Common Mistakes to Avoid

#### DO NOT Create These (They Already Exist):
- ❌ New agent directories without checking `/agents/` structure
- ❌ New cache directories (use existing `/cache/` module)
- ❌ Duplicate configuration files (use existing `/config/` directory)
- ❌ New test directories (use existing `/tests/` structure)

#### Files That Should Be Removed:
- 🗑️ `__pycache__/` directories (ignored by git)
- 🗑️ `.pyc` files (ignored by git)

### 📋 Directory Purposes

#### Core Functionality
- **`agents/`**: All AI agents organized by function (25+ agents)
- **`api/`**: FastAPI REST endpoints (v2.0.0 performance framework)
- **`streamlit_workspace/`**: Interactive UI (8 pages, modularized)

#### Data & Knowledge
- **`CSV/`**: Knowledge graph source data (371+ concepts, 6 domains)
- **`data/`**: Processed and staged content with workflow
- **`cache/`**: Redis caching implementation (full integration)

#### Development & Testing
- **`tests/`**: Comprehensive test suite (performance baselines established)
- **`scripts/`**: Utility and maintenance scripts
- **`dependencies/`**: pip-tools dependency management

#### Documentation & Tracking
- **`updates/`**: Modular implementation plans (9 files)
- **`chat_logs/`**: Session documentation with workflow memory
- **`archive/`**: Archived/deprecated files (clean structure)

### 🔍 Quick Reference

#### To Find...
- **Current workflow**: `chat_logs/memory.json` (read first!)
- **Project status**: `claude.md` and `updates/09_implementation_status.md`
- **Scraping code**: `agents/scraper/` (11 specialized files)
- **Database operations**: `agents/neo4j_manager/` and `agents/qdrant_manager/`
- **API endpoints**: `api/routes/` (6 route files)
- **UI pages**: `streamlit_workspace/pages/` (8 pages + modules)
- **Configuration**: `config/` directory (5 YAML files)
- **Tests**: `tests/` directory (comprehensive framework)

#### Before Adding...
- **Step 1**: Read `chat_logs/memory.json` for workflow
- **Step 2**: Check this file (`08_repository_structure.md`) for existing files
- **Step 3**: Search with `Grep` for similar functionality
- **Step 4**: Use appropriate subdirectory (don't create duplicates)
- **Step 5**: Archive old files if replacing

---

*This structure documentation reflects the actual state as of 2025-07-23 after Phase 2 enhancements. All agent reorganizations, archive cleanups, and enhanced implementations are accurately documented.*