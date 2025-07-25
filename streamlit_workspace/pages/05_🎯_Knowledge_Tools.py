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

import sys
from pathlib import Path

import streamlit as st

# Add utils to path for API operations
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import APIClient, run_async
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
        analyze_data_quality,
        generate_knowledge_report,
        show_ai_recommendations,
        show_concept_builder,
        show_knowledge_analytics,
        show_quality_assurance,
        show_relationship_tools,
    )

    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Knowledge tools modules not available: {e}")
    MODULES_AVAILABLE = False


def main():
    """Main Knowledge Tools interface - Lightweight orchestrator"""

    st.set_page_config(
        page_title="Knowledge Tools - MCP Yggdrasil", page_icon="🎯", layout="wide"
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
        st.error(
            "⚠️ Knowledge tools modules are not available. Please check the installation."
        )
        return

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛠️ Tool Categories")

        tool_category = st.selectbox(
            "Select Tool Category",
            [
                "🏗️ Concept Builder",
                "🔍 Quality Assurance",
                "📊 Knowledge Analytics",
                "🤖 AI Recommendations",
                "🔗 Relationship Tools",
            ],
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


@run_async
async def show_knowledge_stats():
    """Show quick knowledge graph statistics via API"""
    st.markdown("### 📊 Knowledge Stats")

    try:
        client = APIClient()
        
        # Get concepts and domains via API
        concepts_data = await client.manage_database("list", {})
        concepts = concepts_data.get("concepts", [])
        
        # Get analytics for domain information
        analytics_data = await client.get_analytics("overview")
        domain_count = analytics_data.get("domain_count", 0)

        st.metric("Total Concepts", len(concepts))
        st.metric("Domains", domain_count)

        # Quality score from analytics
        quality_score = analytics_data.get("quality_score", 0)
        st.metric("Quality Score", f"{quality_score:.1f}%")

    except Exception as e:
        st.warning(f"Could not load stats: {e}")
        # Fallback metrics
        st.metric("Total Concepts", "N/A")
        st.metric("Domains", "N/A") 
        st.metric("Quality Score", "N/A")


if __name__ == "__main__":
    main()
