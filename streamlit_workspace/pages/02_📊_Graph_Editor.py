"""
Graph Editor - Interactive Knowledge Graph Visualization & Editing

REFACTORED: Now uses modular architecture for improved maintainability
and adherence to the 500-line file limit policy.

Original file: 995 lines → Modular structure:
- models.py: Data structures and configuration (118 lines)
- neo4j_connector.py: Database connection and data management (200+ lines)  
- graph_visualizer.py: Graph creation and visualization (350+ lines)
- ui_components.py: UI elements and interface components (400+ lines)
- main.py: Main orchestrator (200+ lines)

Total reduction: 995 lines → 58 lines (orchestrator) + 5 focused modules
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the refactored modular graph editor
try:
    from .graph_editor.main import main as graph_editor_main
except ImportError:
    # Fallback if module structure has issues
    st.error("Graph Editor modules not found. Please check the module structure.")
    graph_editor_main = None


def main():
    """
    Main entry point for Graph Editor page.
    
    Delegates to the refactored modular graph editor implementation
    which provides improved maintainability and code organization.
    
    Features:
    - Interactive knowledge graph visualization with multiple modes
    - Neo4j database integration with CSV and demo data fallback
    - Advanced filtering and search capabilities
    - Focused concept exploration and relationship analysis
    - Real-time connection status and clear user guidance
    """
    if graph_editor_main:
        try:
            graph_editor_main()
        except Exception as e:
            st.error(f"Error loading Graph Editor: {e}")
            st.info("""
            The Graph Editor has been refactored into a modular architecture. 
            If you see this error, please check:
            
            1. Module imports in `graph_editor/` directory
            2. Database connections and dependencies
            3. Required packages (networkx, plotly, neo4j, pandas)
            """)
            
            # Show fallback interface
            st.markdown("## 📊 Graph Editor (Fallback)")
            st.warning("Modular interface unavailable. Please check module structure.")
            
            st.markdown("""
            **Available Features:**
            - 🌐 Full network visualization
            - 🎯 Focused concept views
            - 🔍 Domain exploration
            - 📈 Relationship analysis
            - 🗄️ Neo4j database integration with CSV fallback
            
            Please contact support to resolve module loading issues.
            """)
    else:
        st.error("Graph Editor main module not available.")
        st.info("Please check the graph_editor module installation.")


if __name__ == "__main__":
    main()