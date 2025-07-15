"""
Knowledge Tools - Advanced Knowledge Engineering and Quality Assurance
Lightweight orchestrator for modular knowledge tools

This is a delegator that routes to specialized modules:
- Concept Builder: Guided wizards and templates
- Quality Assurance: Data validation and cleanup
- Knowledge Analytics: Growth trends and network analysis
- AI Recommendations: Intelligent suggestions
- Relationship Tools: Connection management

Refactored from 1,385-line monolith into modular architecture
following Content Scraper pattern (94.6% size reduction).
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent))

from utils.database_operations import get_all_concepts, get_domains
from utils.session_management import add_to_history

# Import shared styling
try:
    from shared.ui.styling import get_knowledge_tools_css
except ImportError:
    def get_knowledge_tools_css():
        return ""

# Import modular components
try:
    from knowledge_tools import (
        show_concept_builder,
        show_quality_assurance,
        show_knowledge_analytics,
        show_ai_recommendations,
        show_relationship_tools,
        analyze_data_quality,
        generate_knowledge_report
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Knowledge tools modules not available: {e}")
    MODULES_AVAILABLE = False

def main():
    """Main Knowledge Tools interface - Lightweight orchestrator"""
    
    st.set_page_config(
        page_title="Knowledge Tools - MCP Yggdrasil",
        page_icon="🎯",
        layout="wide"
    )
    
    # Apply shared styling
    css = get_knowledge_tools_css()
    if css:
        st.markdown(css, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🎯 Knowledge Tools")
    st.markdown("**Advanced knowledge engineering and quality assurance**")
    
    # Check module availability
    if not MODULES_AVAILABLE:
        st.error("⚠️ Knowledge tools modules are not available. Please check the installation.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛠️ Tool Categories")
        
        tool_category = st.selectbox(
            "Select Tool Category",
            ["🏗️ Concept Builder", "🔍 Quality Assurance", "📊 Knowledge Analytics", "🤖 AI Recommendations", "🔗 Relationship Tools"]
        )
        
        st.markdown("---")
        
        # Quick stats
        show_knowledge_stats()
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔍 Run Full QA Scan", use_container_width=True):
            st.session_state.run_qa_scan = True
            st.rerun()
        
        if st.button("📊 Generate Report", use_container_width=True):
            generate_knowledge_report()
        
        if st.button("🧹 Data Cleanup", use_container_width=True):
            st.session_state.show_cleanup_tools = True
            st.rerun()
    
    # Route to appropriate module
    try:
        if tool_category == "🏗️ Concept Builder":
            show_concept_builder()
        elif tool_category == "🔍 Quality Assurance":
            show_quality_assurance()
        elif tool_category == "📊 Knowledge Analytics":
            show_knowledge_analytics()
        elif tool_category == "🤖 AI Recommendations":
            show_ai_recommendations()
        elif tool_category == "🔗 Relationship Tools":
            show_relationship_tools()
    except Exception as e:
        st.error(f"Error loading {tool_category}: {str(e)}")
        st.info("Please check that all knowledge tools modules are properly installed.")

def show_knowledge_stats():
    """Show quick knowledge graph statistics"""
    st.markdown("### 📊 Knowledge Stats")
    
    try:
        concepts = get_all_concepts(limit=1000)
        domains = get_domains()
        
        st.metric("Total Concepts", len(concepts))
        st.metric("Domains", len(domains))
        
        # Quality score (simplified calculation)
        if MODULES_AVAILABLE:
            quality_issues = analyze_data_quality(concepts[:100])  # Sample for performance
            total_checks = len(concepts) * 5  # Assume 5 quality checks per concept
            passed_checks = total_checks - sum(len(issues) for issues in quality_issues.values())
            quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            st.metric("Quality Score", f"{quality_score:.1f}%")
        else:
            st.metric("Quality Score", "N/A")
        
    except Exception as e:
        st.warning(f"Could not load stats: {e}")

if __name__ == "__main__":
    main()
