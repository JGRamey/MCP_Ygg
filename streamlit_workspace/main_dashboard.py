"""
MCP Yggdrasil IDE Workspace - Main Dashboard
Entry point for the comprehensive database management workspace
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import utilities
from utils.session_management import initialize_session_state
from utils.database_operations import test_connections

def main():
    """Main dashboard entry point"""
    
    # Configure page
    st.set_page_config(
        page_title="MCP Yggdrasil IDE Workspace",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/mcp-yggdrasil',
            'Report a bug': 'https://github.com/your-repo/mcp-yggdrasil/issues',
            'About': "MCP Yggdrasil - Hybrid Knowledge Management System"
        }
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for IDE-like interface
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E8B57;
        --secondary-color: #3CB371;
        --accent-color: #20B2AA;
        --background-color: #F8F9FA;
        --text-color: #2F3349;
        --border-color: #E1E5E9;
    }
    
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Status indicators */
    .status-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .status-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .status-connected { color: #28a745; }
    .status-disconnected { color: #dc3545; }
    .status-warning { color: #ffc107; }
    
    /* Module cards */
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .module-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    
    .module-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .module-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .module-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .module-description {
        color: #6c757d;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Quick stats */
    .quick-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
    }
    
    .stat-value {
        font-weight: 700;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌳 MCP Yggdrasil IDE Workspace</h1>
        <p>Comprehensive Knowledge Management & Database Administration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    show_system_status()
    
    # Welcome message and navigation
    if 'current_page' not in st.session_state:
        show_welcome_page()
    
    # Sidebar navigation
    show_sidebar_navigation()

def show_system_status():
    """Display system connection status"""
    
    st.subheader("🔧 System Status")
    
    # Test database connections
    connections = test_connections()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        neo4j_status = "✅ Connected" if connections.get('neo4j', False) else "❌ Disconnected"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-indicator {'status-connected' if connections.get('neo4j', False) else 'status-disconnected'}">
                Neo4j
            </div>
            <div>{neo4j_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        qdrant_status = "✅ Connected" if connections.get('qdrant', False) else "❌ Disconnected"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-indicator {'status-connected' if connections.get('qdrant', False) else 'status-disconnected'}">
                Qdrant
            </div>
            <div>{qdrant_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        redis_status = "✅ Connected" if connections.get('redis', False) else "❌ Disconnected"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-indicator {'status-connected' if connections.get('redis', False) else 'status-disconnected'}">
                Redis
            </div>
            <div>{redis_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        docker_status = "✅ Running" if connections.get('docker', False) else "❌ Stopped"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-indicator {'status-connected' if connections.get('docker', False) else 'status-disconnected'}">
                Docker
            </div>
            <div>{docker_status}</div>
        </div>
        """, unsafe_allow_html=True)

def show_welcome_page():
    """Show welcome page with module overview"""
    
    st.markdown("---")
    st.subheader("🚀 Available Modules")
    
    # Module cards
    modules = [
        {
            "icon": "🗄️",
            "title": "Database Manager", 
            "description": "Complete CRUD operations for concepts, relationships, and domains. Create, edit, and delete knowledge graph entities with real-time validation.",
            "page": "01_🗄️_Database_Manager"
        },
        {
            "icon": "📊", 
            "title": "Graph Editor",
            "description": "Visual knowledge graph editing with interactive network visualization. Drag-and-drop editing, layout controls, and real-time graph analysis.",
            "page": "02_📊_Graph_Editor"
        },
        {
            "icon": "📁",
            "title": "File Manager", 
            "description": "Project file and configuration management. Edit CSV files, configuration files, and manage project structure with version control.",
            "page": "03_📁_File_Manager"
        },
        {
            "icon": "⚡",
            "title": "Operations Console",
            "description": "Real-time system operations and monitoring. Cypher query editor, transaction management, and system performance monitoring.",
            "page": "04_⚡_Operations_Console"
        },
        {
            "icon": "🎯",
            "title": "Knowledge Tools",
            "description": "Advanced knowledge engineering tools. Concept builder wizard, data quality assurance, and analytics for knowledge graph optimization.",
            "page": "05_🎯_Knowledge_Tools"
        },
        {
            "icon": "📈",
            "title": "Analytics Dashboard", 
            "description": "System analytics and insights. Performance monitoring, usage statistics, and AI-powered recommendations for graph improvements.",
            "page": "06_📈_Analytics"
        }
    ]
    
    # Create module grid
    cols = st.columns(2)
    for i, module in enumerate(modules):
        with cols[i % 2]:
            if st.button(f"{module['icon']} {module['title']}", key=f"module_{i}", use_container_width=True):
                st.switch_page(f"pages/{module['page']}.py")
            st.caption(module['description'])
            st.markdown("<br>", unsafe_allow_html=True)

def show_sidebar_navigation():
    """Show sidebar with navigation and quick stats"""
    
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Quick navigation buttons
        if st.button("🗄️ Database Manager", use_container_width=True):
            st.switch_page("pages/01_🗄️_Database_Manager.py")
        
        if st.button("📊 Graph Editor", use_container_width=True):
            st.switch_page("pages/02_📊_Graph_Editor.py")
        
        if st.button("📁 File Manager", use_container_width=True):
            st.switch_page("pages/03_📁_File_Manager.py")
        
        if st.button("⚡ Operations Console", use_container_width=True):
            st.switch_page("pages/04_⚡_Operations_Console.py")
        
        if st.button("🎯 Knowledge Tools", use_container_width=True):
            st.switch_page("pages/05_🎯_Knowledge_Tools.py")
        
        if st.button("📈 Analytics", use_container_width=True):
            st.switch_page("pages/06_📈_Analytics.py")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        try:
            from utils.database_operations import get_quick_stats
            stats = get_quick_stats()
            
            st.markdown(f"""
            <div class="quick-stats">
                <div class="stat-item">
                    <span>Concepts:</span>
                    <span class="stat-value">{stats.get('concepts', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span>Relationships:</span>
                    <span class="stat-value">{stats.get('relationships', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span>Domains:</span>
                    <span class="stat-value">{stats.get('domains', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span>Vectors:</span>
                    <span class="stat-value">{stats.get('vectors', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load stats: {str(e)}")
        
        st.markdown("---")
        
        # System actions
        st.markdown("### ⚙️ System Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📱 Status", use_container_width=True):
                st.info("System operational")

if __name__ == "__main__":
    main()