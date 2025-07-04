# MCP Yggdrasil IDE Workspace - UI Development Plan
**Version:** 1.0  
**Date:** 2025-07-01  
**Status:** Planning Phase  

## 🎯 PROJECT VISION
Transform the current basic HTML dashboard into a comprehensive **IDE-like workspace** for complete MCP Yggdrasil project management. This workspace will serve as the primary interface for database administration, knowledge graph editing, project file management, and system operations.

## 📋 REQUIREMENTS SUMMARY

## User notes ##
- I dont want an IDE like interface, only file management of the stored data and new data being imported/scraped into the databases. The files displayed in the UI should only be database material (CSV files and such, not project files which is being shown currently)
- Scraper page is blank. Should have options for the type of source that is going to be scraped (Youtube video/transcript, Book, PDF, Picture/Image, Webpage, Web article, Manuscript, Scroll, etc)
- Operations console is showing error: ModuleNotFoundError: No module named 'psutil'
Traceback:
File "/Users/grant/Documents/GitHub/MCP_Ygg/streamlit_workspace/pages/04_⚡_Operations_Console.py", line 12, in <module>
    import psutil
- Graph editor is blank and says "No concepts match the current filters" - Which is completely wrong and should just show the Neo4j knowledge graph. It should basically be an exact replica of the Neo4j dashboard with drag and drop options and editor options to change the graph and update files, relationships, and concepts. 
- IMPORTANT!!! - Concepts are found within scraped data, for instance I'm using the word "Concept" as a form of "Idea" or a specific area of thought,example: Metaphysics is a type of "Concept" even though it's a subject/branch of Philosophy. Another example for a concept is the "trinity" which is found in Christianity and many other forms of belief or even numerolgy. This is the whole point of the "concept" connector which is to connect like minded concepts across multiple cultures and data/information found in the world. Is this incorrect? Should I go about this differently? If so, stop and talk to me about it so we can correct this.

### **Primary Goal**
Create a professional workspace where users can:
- **Directly manipulate** Neo4j and Qdrant databases
- **Visually edit** knowledge graphs and relationships
- **Manage project files** and configurations
- **Monitor system operations** in real-time
- **Perform advanced analytics** and data quality operations

### **Target User Experience**
- **Real-time database editing** with immediate visual feedback
- **Professional workflow** for knowledge engineering
- **Complete project control** through web interface

## 🏗️ TECHNICAL ARCHITECTURE

### **Framework Decision: Streamlit**
**Rationale:**
- ✅ **Python native** - Direct integration with existing Neo4j/Qdrant code
- ✅ **Rapid development** - Fast iteration and prototyping
- ✅ **Built-in components** - Charts, graphs, data tables, file handling
- ✅ **Minimal setup** - Single command deployment
- ✅ **Perfect for internal tools** - Ideal for database administration

### **Application Structure**
```
📁 streamlit_workspace/
├── 🏠 main_dashboard.py               # Main entry point & navigation
├── 📄 pages/
│   ├── 01_🗄️_Database_Manager.py     # Core CRUD operations
│   ├── 02_📊_Graph_Editor.py          # Visual graph editing
│   ├── 03_📁_File_Manager.py          # Project file management
│   ├── 04_⚡_Operations_Console.py    # Real-time operations
│   ├── 05_🎯_Knowledge_Tools.py       # Advanced knowledge engineering
│   └── 06_📈_Analytics.py             # System analytics & monitoring
├── 🔧 utils/
│   ├── database_operations.py         # Database CRUD functions
│   ├── graph_visualization.py         # Graph rendering & interaction
│   ├── file_operations.py             # File management utilities
│   ├── validation.py                  # Data validation & integrity
│   └── session_management.py          # Workspace state management
└── 🎨 assets/
    ├── styles.css                     # Custom CSS styling
    └── components/                    # Reusable UI components
```

## 📋 DETAILED MODULE SPECIFICATIONS

### **Module 1: 🗄️ Database Manager**
**Purpose:** Complete CRUD operations for concepts, relationships, and domains

#### **Core Features:**
- **Concept Management**
  - Create new concepts with different data files (texts, articles, manuscripts, etc)
  - Edit existing concepts with real-time validation
  - Duplicate concept detection and merging
  - Bulk import/export operations

- **Relationship Management**
  - Create relationships between concepts
  - Edit relationship properties (type, strength, description)
  - Delete relationships with impact analysis
  - Relationship type management (BELONGS_TO, RELATES_TO, FATHER_OF, SON_OF, DAUGHTER_OF, DISCIPLE_OF, FOLLOWER_OF, etc.)
  - Visual relationship browser

- **Domain & Category Management**
  - Create new domains and sub-domains
  - Modify domain hierarchies and structures
  - Category type management (root, sub_root, branch, limb, leaf)
  - Domain-specific validation rules
  - Cross-domain relationship management

#### **UI Components:**
- **Concept Editor**: Form-based editing with auto-completion
- **Relationship Builder**: Visual drag-and-drop interface
- **Domain Tree**: Hierarchical domain structure editor
- **Bulk Operations Panel**: CSV import/export with validation
- **Search & Filter**: Advanced filtering by domain, type, level

#### **Technical Requirements:**
- Real-time Neo4j transaction management
- Data validation with immediate feedback
- Undo/redo functionality for operations
- Session state management for unsaved changes
- Integration with existing FastAPI backend

### **Module 2: 📊 Graph Editor**
**Purpose:** Visual knowledge graph editing and exploration

#### **Core Features:**
- **Interactive Network Visualization**
  - Drag-and-drop node positioning
  - Real-time graph layout algorithms (force-directed, hierarchical)
  - Zoom, pan, and selection controls
  - Node and edge styling customization

- **Live Graph Editing**
  - Add nodes directly in visualization
  - Create relationships by connecting nodes
  - Edit properties through context menus
  - Delete nodes/relationships with confirmation

- **Graph Analysis Tools**
  - Shortest path finding between concepts
  - Community detection and clustering
  - Centrality analysis (betweenness, closeness, degree)
  - Subgraph extraction and filtering

#### **UI Components:**
- **Graph Canvas**: Interactive network visualization (using Cytoscape.js or Plotly)
- **Node Properties Panel**: Detailed editing for selected nodes
- **Graph Controls**: Layout options, filters, styling controls
- **Minimap**: Overview navigation for large graphs
- **Analysis Panel**: Graph metrics and community detection results

#### **Technical Requirements:**
- Integration with Cytoscape.js or similar graph library
- Real-time synchronization with Neo4j database
- Performance optimization for large graphs (>1000 nodes)
- Export capabilities (PNG, SVG, GraphML, GEXF)
- Layout persistence and customization

### **Module 3: 📁 File Manager**
**Purpose:** Complete project file and configuration management

#### **Core Features:**
- **CSV File Editor**
  - Direct editing of concept and relationship CSV files
  - Real-time validation and error highlighting
  - Auto-save with change tracking
  - Import/export with format validation

- **Configuration Management**
  - Edit .env files with environment variable validation
  - docker-compose.yml editor with syntax highlighting
  - Agent configuration files (JSON/YAML editing)
  - Database connection settings management

- **Project Structure Browser**
  - File tree navigation with search
  - File content preview and editing
  - Git integration for version control
  - Backup and restore capabilities

#### **UI Components:**
- **File Tree**: Expandable project structure browser
- **Code Editor**: Syntax-highlighted editor for various file types
- **Configuration Forms**: Guided editing for complex configurations
- **Backup Manager**: Create and restore project snapshots

#### **Technical Requirements:**
- File system access and modification
- Syntax highlighting for multiple file formats
- Git integration for change tracking
- Backup scheduling and management
- File validation and integrity checking

### **Module 4: ⚡ Operations Console**
**Purpose:** Real-time system operations and monitoring

#### **Core Features:**
- **Live Cypher Editor**
  - Syntax highlighting and auto-completion
  - Query execution with result visualization
  - Query history and favorites
  - Performance analysis and optimization suggestions

- **System Monitoring**
  - Docker container status and logs
  - Database connection health
  - Query performance metrics
  - Memory and resource usage

- **Transaction Management**
  - Active transaction monitoring
  - Rollback capabilities
  - Transaction log analysis
  - Deadlock detection and resolution

#### **UI Components:**
- **Query Editor**: Advanced code editor with Cypher syntax support
- **Results Viewer**: Tabular and graph result visualization
- **System Dashboard**: Real-time metrics and status indicators
- **Log Viewer**: Searchable, filterable log display
- **Performance Monitor**: Charts for query times and system resources

#### **Technical Requirements:**
- Real-time system monitoring integration
- Advanced query editor with IntelliSense
- Log aggregation and analysis
- Performance metrics collection
- Alert system for critical issues

### **Module 5: 🎯 Knowledge Tools**
**Purpose:** Advanced knowledge engineering and quality assurance

#### **Core Features:**
- **Concept Builder Wizard**
  - Guided concept creation with templates
  - Domain-specific validation rules
  - Auto-suggestion based on existing concepts
  - Bulk concept generation from text

- **Data Quality Assurance**
  - Duplicate detection and merging
  - Inconsistency identification
  - Missing relationship detection
  - Data validation reports

- **Knowledge Graph Analytics**
  - Concept coverage analysis by domain
  - Relationship pattern detection
  - Orphaned node identification
  - Graph density and connectivity metrics

#### **UI Components:**
- **Wizard Interface**: Step-by-step concept creation
- **Quality Dashboard**: Data quality metrics and reports
- **Analytics Panels**: Interactive charts and statistics
- **Recommendation Engine**: Suggested improvements and connections
- **Export Tools**: Report generation and data export

#### **Technical Requirements:**
- Advanced analytics algorithms
- Machine learning integration for recommendations
- Report generation capabilities
- Data validation rule engine
- Integration with existing agent systems

### **Module 6: 📈 Analytics Dashboard**
**Purpose:** System analytics, monitoring, and insights

#### **Core Features:**
- **System Statistics**
  - Real-time concept and relationship counts
  - Domain distribution analysis
  - Growth trends and patterns
  - Usage analytics and user behavior

- **Performance Analytics**
  - Query performance trends
  - Database optimization recommendations
  - Resource utilization analysis
  - Bottleneck identification

- **Knowledge Graph Insights**
  - Most connected concepts
  - Community structure analysis
  - Knowledge gaps identification
  - Cross-domain relationship patterns

#### **UI Components:**
- **Metrics Dashboard**: Real-time system statistics
- **Interactive Charts**: Plotly-based visualizations
- **Performance Graphs**: Time-series performance data
- **Insight Panels**: AI-generated insights and recommendations
- **Export Interface**: Report generation and sharing

#### **Technical Requirements:**
- Real-time data aggregation
- Advanced visualization libraries (Plotly, Bokeh)
- Statistical analysis capabilities
- Report generation and scheduling
- Integration with monitoring systems

## 🎨 USER INTERFACE DESIGN

### **Design Principles**
- **Professional Aesthetic**: Clean, modern interface similar to popular IDEs
- **Intuitive Navigation**: Clear module separation with consistent navigation
- **Responsive Design**: Adaptable to different screen sizes and resolutions
- **Performance Focused**: Fast loading and responsive interactions
- **Accessibility**: Keyboard shortcuts and screen reader support

### **Color Scheme & Branding**
- **Primary Colors**: Forest green (#2E8B57) to match Yggdrasil tree theme
- **Secondary Colors**: Complementary blues and grays for balance
- **Accent Colors**: Bright colors for alerts, success, and warning states
- **Typography**: Modern, readable fonts (Inter, Roboto, or system fonts)

### **Layout Structure**
- **Sidebar Navigation**: Persistent module navigation with icons
- **Main Content Area**: Context-specific content for each module
- **Properties Panel**: Collapsible panel for detailed editing
- **Status Bar**: System status and current operation indicators
- **Modal Dialogs**: For complex operations and confirmations

## 📅 DEVELOPMENT TIMELINE

### **Phase 1: Foundation (Week 1-2)**
- **✅ Planning & Documentation** (UIplan.md creation)
- **📦 Environment Setup** (Streamlit installation and configuration)
- **🏠 Main Dashboard** (Navigation structure and basic layout)
- **🗄️ Database Manager** (Basic CRUD operations)

### **Phase 2: Core Functionality (Week 3-4)**
- **📊 Graph Editor** (Basic visualization and interaction)
- **📁 File Manager** (Project file browsing and editing)
- **⚡ Operations Console** (Cypher editor and system monitoring)

### **Phase 3: Advanced Features (Week 5-6)**
- **🎯 Knowledge Tools** (Quality assurance and analytics)
- **📈 Analytics Dashboard** (Comprehensive system insights)
- **🎨 UI Polish** (Styling, responsiveness, and user experience)

### **Phase 4: Integration & Testing (Week 7-8)**
- **🔧 System Integration** (Full workflow testing)
- **🐛 Bug Fixes** (Issue resolution and optimization)
- **📚 Documentation** (User guides and API documentation)
- **🚀 Production Deployment** (Final deployment and monitoring)

## 🛠️ TECHNICAL IMPLEMENTATION DETAILS

### **Required Dependencies**
```python
# Core Streamlit and web framework
streamlit>=1.28.0
streamlit-extras>=0.3.0

# Database connectivity
neo4j>=5.0.0
qdrant-client>=1.6.0
redis>=4.5.0

# Visualization and graphics
plotly>=5.17.0
networkx>=3.2.0
cytoscape-widget>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Data processing
pandas>=2.1.0
numpy>=1.24.0

# File handling and utilities
pyyaml>=6.0.0
python-dotenv>=1.0.0
watchdog>=3.0.0

# Code editing and syntax highlighting
streamlit-ace>=0.1.1
pygments>=2.16.0
```

### **Integration Points**
- **Existing FastAPI Backend**: Leverage current API endpoints
- **Neo4j Database**: Direct driver integration for real-time operations
- **Qdrant Vector Database**: Client integration for vector operations
- **Docker Services**: Container management and monitoring
- **File System**: Project file access and modification

### **Performance Considerations**
- **Caching**: Streamlit caching for expensive operations
- **Lazy Loading**: Load data on-demand for large datasets
- **Pagination**: Handle large result sets efficiently
- **Background Processing**: Async operations for long-running tasks
- **Memory Management**: Efficient data structures and cleanup

## 🔒 SECURITY & SAFETY

### **Data Protection**
- **Transaction Safety**: Rollback capabilities for all operations
- **Backup Integration**: Automatic backups before major changes
- **Validation**: Comprehensive data validation before database updates
- **Access Control**: Session management and user authentication (future)

### **Error Handling**
- **Graceful Degradation**: Fallback options when services unavailable
- **User Feedback**: Clear error messages and recovery suggestions
- **Logging**: Comprehensive operation logging for debugging
- **Recovery**: Automated recovery from common error states

## ✅ SUCCESS CRITERIA

### **Functional Requirements**
- ✅ **Complete CRUD Operations**: All database entities manageable through UI
- ✅ **Visual Graph Editing**: Interactive knowledge graph manipulation
- ✅ **File Management**: Complete project file editing capabilities
- ✅ **Real-time Operations**: Live system monitoring and control
- ✅ **Professional Interface**: IDE-quality user experience

### **Performance Requirements**
- ✅ **Response Times**: < 2 seconds for most operations
- ✅ **Large Dataset Handling**: Support for 10,000+ concepts
- ✅ **Real-time Updates**: Live synchronization between views
- ✅ **Stability**: 99%+ uptime during development sessions

### **User Experience Requirements**
- ✅ **Intuitive Navigation**: Users can find features without training
- ✅ **Workflow Efficiency**: Common tasks completable in minimal steps
- ✅ **Error Prevention**: Validation prevents common mistakes
- ✅ **Professional Feel**: Interface comparable to commercial tools

## 🔄 FUTURE ENHANCEMENTS

### **Advanced Features (Post-MVP)**
- **Multi-user Collaboration**: Real-time collaborative editing
- **Advanced Analytics**: Machine learning insights and recommendations
- **Plugin System**: Extensible architecture for custom tools
- **API Integration**: External data source integration
- **Mobile Support**: Responsive design for tablet/mobile access

### **Integration Opportunities**
- **Agent Integration**: Direct control of MCP agents through UI
- **External APIs**: Integration with academic databases and sources
- **Export Capabilities**: Integration with external tools and formats
- **Automation**: Scheduled operations and automated workflows

---

## 📝 NOTES & DECISIONS

### **Key Design Decisions**
1. **Streamlit Choice**: Prioritizes rapid development and Python integration
2. **Module Structure**: Clear separation of concerns for maintainability
3. **Real-time Focus**: Immediate feedback for all operations
4. **Professional Target**: IDE-quality interface for power users

### **Risk Mitigation**
- **Performance**: Caching and pagination for large datasets
- **Complexity**: Phased development approach with MVP focus
- **User Adoption**: Familiar IDE-like interface patterns
- **Maintenance**: Clean architecture and comprehensive documentation

---

**Status:** ✅ Planning Complete - Ready for Implementation  
**Next Step:** Begin Phase 1 development with main dashboard and database manager