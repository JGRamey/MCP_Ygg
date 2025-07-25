"""
Content Scraper - API-First Implementation
Clean UI layer that communicates exclusively with FastAPI backend
Replaces agent imports with API calls for true separation of concerns
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncio
import pandas as pd
import streamlit as st

# Import our API client
try:
    from streamlit_workspace.utils.api_client import APIProgress, api_client, run_async

    API_CLIENT_AVAILABLE = True
except ImportError as e:
    st.error(f"API client not available: {e}")
    API_CLIENT_AVAILABLE = False


class APIContentScraperUI:
    """
    API-first content scraper interface.

    This implementation delegates ALL operations to the FastAPI backend,
    maintaining clean separation between UI and business logic.
    """

    def __init__(self):
        """Initialize the API-based scraper UI."""
        self.domains = [
            "mathematics",
            "science",
            "philosophy",
            "religion",
            "art",
            "language",
            "general",
        ]
        self.priorities = ["low", "medium", "high", "urgent"]

        # Initialize session state
        if "submission_history" not in st.session_state:
            st.session_state.submission_history = []
        if "active_submissions" not in st.session_state:
            st.session_state.active_submissions = {}

    def render_page(self):
        """Render the main content scraper page."""
        st.set_page_config(page_title="Content Scraper", page_icon="📥", layout="wide")

        st.title("📥 Content Scraper")
        st.markdown("Multi-source content acquisition powered by FastAPI backend")

        if not API_CLIENT_AVAILABLE:
            st.error("API client is not available. Please check the system setup.")
            return

        # Show API health status
        self._show_api_status()

        # Main interface tabs
        tabs = st.tabs(
            [
                "🌐 Web Scraping",
                "📺 YouTube",
                "✍️ Text Input",
                "📊 Status & History",
                "⚙️ Queue Management",
            ]
        )

        with tabs[0]:
            self._render_web_scraping_tab()

        with tabs[1]:
            self._render_youtube_tab()

        with tabs[2]:
            self._render_text_input_tab()

        with tabs[3]:
            self._render_status_history_tab()

        with tabs[4]:
            self._render_queue_management_tab()

    @run_async
    async def _show_api_status(self):
        """Show API connectivity status."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            is_healthy = await api_client.health_check()
            if is_healthy:
                st.success("🟢 API Connected")
            else:
                st.error("🔴 API Disconnected")

        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()

        with col3:
            queue_stats = await api_client.get_queue_stats()
            if queue_stats:
                pending = queue_stats.get("pending_count", 0)
                st.metric("Queue", pending)

    def _render_web_scraping_tab(self):
        """Render web scraping interface."""
        st.subheader("🌐 Web Content Scraping")

        # URL input form
        with st.form("web_scraping_form"):
            col1, col2 = st.columns([3, 1])

            with col1:
                url = st.text_input(
                    "Enter URL",
                    placeholder="https://example.com/article",
                    help="URL to scrape content from",
                )

            with col2:
                priority = st.selectbox("Priority", self.priorities, index=1)

            col1, col2 = st.columns(2)

            with col1:
                domain = st.selectbox(
                    "Content Domain",
                    self.domains,
                    help="Primary content domain for classification",
                )

            with col2:
                use_advanced = st.checkbox(
                    "Advanced Pipeline", help="Use enhanced analysis agents"
                )

            # Advanced options
            if use_advanced:
                st.subheader("🔧 Advanced Options")
                col1, col2 = st.columns(2)

                with col1:
                    extract_citations = st.checkbox("Extract Citations", value=True)
                    extract_entities = st.checkbox("Extract Entities", value=True)

                with col2:
                    concept_analysis = st.checkbox("Concept Analysis", value=True)
                    quality_check = st.checkbox("Quality Assessment", value=True)

            # Submit button
            submitted = st.form_submit_button("🚀 Scrape URL", type="primary")

            if submitted and url:
                # Prepare advanced pipeline if selected
                agent_pipeline = None
                if use_advanced:
                    agent_pipeline = {
                        "extract_citations": extract_citations,
                        "extract_entities": extract_entities,
                        "concept_analysis": concept_analysis,
                        "quality_check": quality_check,
                    }

                self._submit_url_scraping(url, domain, priority, agent_pipeline)

    def _render_youtube_tab(self):
        """Render YouTube processing interface."""
        st.subheader("📺 YouTube Transcript Processing")

        with st.form("youtube_form"):
            col1, col2 = st.columns([3, 1])

            with col1:
                youtube_url = st.text_input(
                    "YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="YouTube video URL for transcript extraction",
                )

            with col2:
                priority = st.selectbox(
                    "Priority", self.priorities, index=1, key="yt_priority"
                )

            col1, col2 = st.columns(2)

            with col1:
                domain = st.selectbox("Content Domain", self.domains, key="yt_domain")

            with col2:
                extract_metadata = st.checkbox(
                    "Extract Metadata",
                    value=True,
                    help="Include video metadata (title, description, etc.)",
                )

            submitted = st.form_submit_button("🎬 Process Video", type="primary")

            if submitted and youtube_url:
                self._submit_youtube_processing(
                    youtube_url, domain, priority, extract_metadata
                )

    def _render_text_input_tab(self):
        """Render text input interface."""
        st.subheader("✍️ Direct Text Submission")

        with st.form("text_input_form"):
            col1, col2 = st.columns([2, 1])

            with col1:
                title = st.text_input("Title", placeholder="Enter content title")

            with col2:
                priority = st.selectbox(
                    "Priority", self.priorities, index=1, key="text_priority"
                )

            text_content = st.text_area(
                "Text Content",
                height=200,
                placeholder="Paste or type your content here...",
                help="The main text content to be analyzed",
            )

            col1, col2 = st.columns(2)

            with col1:
                domain = st.selectbox("Content Domain", self.domains, key="text_domain")
                author = st.text_input(
                    "Author (optional)", placeholder="Content author"
                )

            with col2:
                st.write("**Processing Options**")
                full_analysis = st.checkbox("Full NLP Analysis", value=True)
                concept_extraction = st.checkbox("Concept Extraction", value=True)
                relationship_mapping = st.checkbox("Relationship Mapping", value=False)

            submitted = st.form_submit_button("📝 Submit Text", type="primary")

            if submitted and text_content and title:
                agent_pipeline = (
                    {
                        "full_analysis": full_analysis,
                        "concept_extraction": concept_extraction,
                        "relationship_mapping": relationship_mapping,
                    }
                    if any([full_analysis, concept_extraction, relationship_mapping])
                    else None
                )

                self._submit_text_content(
                    text_content, title, author, domain, priority, agent_pipeline
                )

    def _render_status_history_tab(self):
        """Render status tracking and history interface."""
        st.subheader("📊 Submission Status & History")

        # Auto-refresh toggle
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=False)

        with col2:
            if st.button("🔄 Refresh Now"):
                self._refresh_submission_status()

        with col3:
            if st.button("🧹 Clear History"):
                st.session_state.submission_history = []
                st.session_state.active_submissions = {}
                st.rerun()

        # Auto-refresh implementation
        if auto_refresh:
            placeholder = st.empty()
            for i in range(10):  # Refresh 10 times
                with placeholder.container():
                    self._display_active_submissions()
                    self._display_submission_history()
                time.sleep(5)  # Wait 5 seconds between refreshes
        else:
            self._display_active_submissions()
            self._display_submission_history()

    def _render_queue_management_tab(self):
        """Render queue management interface."""
        st.subheader("⚙️ Processing Queue Management")

        # Queue statistics
        self._display_queue_statistics()

        # Recent queue activity
        st.subheader("📋 Recent Activity")
        if st.session_state.submission_history:
            recent_df = pd.DataFrame(st.session_state.submission_history[-10:])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent activity to display.")

    @run_async
    async def _submit_url_scraping(
        self, url: str, domain: str, priority: str, agent_pipeline: Optional[Dict]
    ):
        """Submit URL for scraping via API."""
        with APIProgress("🌐 Submitting URL for scraping..."):
            result = await api_client.scrape_url(
                url=url, domain=domain, priority=priority, agent_pipeline=agent_pipeline
            )

            if result and result.get("submission_id"):
                st.success(
                    f"✅ URL submitted successfully! ID: {result['submission_id']}"
                )
                self._add_to_history(result, "url_scraping", url)
                self._track_submission(result["submission_id"], "url_scraping")
            else:
                st.error("❌ Failed to submit URL for scraping")

    @run_async
    async def _submit_youtube_processing(
        self, youtube_url: str, domain: str, priority: str, extract_metadata: bool
    ):
        """Submit YouTube URL for processing via API."""
        with APIProgress("📺 Submitting YouTube URL for processing..."):
            result = await api_client.scrape_youtube(
                youtube_url=youtube_url,
                domain=domain,
                priority=priority,
                extract_metadata=extract_metadata,
            )

            if result and result.get("submission_id"):
                st.success(
                    f"✅ YouTube URL submitted successfully! ID: {result['submission_id']}"
                )
                self._add_to_history(result, "youtube", youtube_url)
                self._track_submission(result["submission_id"], "youtube")
            else:
                st.error("❌ Failed to submit YouTube URL for processing")

    @run_async
    async def _submit_text_content(
        self,
        text_content: str,
        title: str,
        author: Optional[str],
        domain: str,
        priority: str,
        agent_pipeline: Optional[Dict],
    ):
        """Submit text content via API."""
        with APIProgress("✍️ Submitting text content for analysis..."):
            result = await api_client.submit_text(
                text_content=text_content,
                title=title,
                author=author,
                domain=domain,
                priority=priority,
                agent_pipeline=agent_pipeline,
            )

            if result and result.get("submission_id"):
                st.success(
                    f"✅ Text submitted successfully! ID: {result['submission_id']}"
                )
                self._add_to_history(result, "text_input", title)
                self._track_submission(result["submission_id"], "text_input")
            else:
                st.error("❌ Failed to submit text content")

    @run_async
    async def _refresh_submission_status(self):
        """Refresh status of active submissions."""
        if not st.session_state.active_submissions:
            return

        for submission_id in list(st.session_state.active_submissions.keys()):
            status = await api_client.get_scrape_status(submission_id)
            if status:
                st.session_state.active_submissions[submission_id]["status"] = (
                    status.get("status", "unknown")
                )

                # Remove completed submissions from active tracking
                if status.get("status") in ["completed", "failed"]:
                    del st.session_state.active_submissions[submission_id]

    @run_async
    async def _display_queue_statistics(self):
        """Display current queue statistics."""
        queue_stats = await api_client.get_queue_stats()

        if queue_stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Pending", queue_stats.get("pending_count", 0))

            with col2:
                st.metric("Processing", queue_stats.get("processing_count", 0))

            with col3:
                st.metric("Completed", queue_stats.get("completed_count", 0))

            with col4:
                st.metric("Failed", queue_stats.get("failed_count", 0))

            # Queue details
            if queue_stats.get("queue_details"):
                st.subheader("📋 Queue Details")
                queue_df = pd.DataFrame(queue_stats["queue_details"])
                st.dataframe(queue_df, use_container_width=True)
        else:
            st.warning("Unable to retrieve queue statistics")

    def _display_active_submissions(self):
        """Display currently active submissions."""
        if st.session_state.active_submissions:
            st.subheader("🔄 Active Submissions")

            active_data = []
            for submission_id, info in st.session_state.active_submissions.items():
                active_data.append(
                    {
                        "Submission ID": submission_id,
                        "Type": info["type"],
                        "Content": (
                            info["content"][:50] + "..."
                            if len(info["content"]) > 50
                            else info["content"]
                        ),
                        "Status": info.get("status", "pending"),
                        "Submitted": info["timestamp"],
                    }
                )

            active_df = pd.DataFrame(active_data)
            st.dataframe(active_df, use_container_width=True)
        else:
            st.info("No active submissions")

    def _display_submission_history(self):
        """Display submission history."""
        if st.session_state.submission_history:
            st.subheader("📚 Submission History")

            # Filter controls
            col1, col2, col3 = st.columns(3)

            with col1:
                type_filter = st.selectbox(
                    "Filter by Type", ["All", "url_scraping", "youtube", "text_input"]
                )

            with col2:
                status_filter = st.selectbox(
                    "Filter by Status",
                    ["All", "pending", "processing", "completed", "failed"],
                )

            with col3:
                limit = st.number_input(
                    "Show last N items", value=20, min_value=5, max_value=100
                )

            # Filter and display history
            history_df = pd.DataFrame(st.session_state.submission_history[-limit:])

            if type_filter != "All":
                history_df = history_df[history_df["type"] == type_filter]
            if status_filter != "All":
                history_df = history_df[history_df["status"] == status_filter]

            st.dataframe(history_df, use_container_width=True)

            # Download option
            if not history_df.empty:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    "📥 Download History CSV",
                    csv,
                    f"content_scraper_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                )
        else:
            st.info("No submission history available")

    def _add_to_history(self, result: Dict, submission_type: str, content: str):
        """Add submission result to history."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "submission_id": result.get("submission_id", "unknown"),
            "type": submission_type,
            "content": content,
            "status": result.get("status", "pending"),
            "message": result.get("message", "Submitted for processing"),
        }

        st.session_state.submission_history.append(history_entry)

        # Keep only last 100 entries
        if len(st.session_state.submission_history) > 100:
            st.session_state.submission_history = st.session_state.submission_history[
                -100:
            ]

    def _track_submission(self, submission_id: str, submission_type: str):
        """Track active submission for status monitoring."""
        st.session_state.active_submissions[submission_id] = {
            "type": submission_type,
            "content": submission_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }


def main():
    """Main function to render the API-based content scraper interface."""
    scraper_ui = APIContentScraperUI()
    scraper_ui.render_page()


if __name__ == "__main__":
    main()
