"""
Content Scraper - Thin UI layer for scraping operations
Delegates all processing to FastAPI backend
"""

import json
from typing import Dict, List

import streamlit as st
from streamlit_workspace.utils.api_client import api_client, run_async

st.set_page_config(page_title="Content Scraper", page_icon="📥", layout="wide")


class ContentScraperUI:
    """UI-only content scraper interface"""

    def __init__(self):
        self.source_configs = self._load_source_configs()

    def _load_source_configs(self) -> Dict:
        """Load source type configurations"""
        return {
            "webpage": {
                "icon": "🌐",
                "fields": ["url", "extract_images", "extract_links"],
            },
            "academic_paper": {
                "icon": "📚",
                "fields": ["url", "doi", "extract_citations", "extract_figures"],
            },
            "ancient_text": {
                "icon": "📜",
                "fields": ["url", "language", "include_annotations", "translations"],
            },
            "youtube": {
                "icon": "📺",
                "fields": ["url", "include_transcript", "include_metadata"],
            },
        }

    def render(self):
        """Main render method"""
        st.title("📥 Content Scraper")
        st.markdown("Multi-source content acquisition interface")

        # Source type selection
        col1, col2 = st.columns([2, 1])

        with col1:
            source_type = st.selectbox(
                "Select Content Source",
                options=list(self.source_configs.keys()),
                format_func=lambda x: f"{self.source_configs[x]['icon']} {x.replace('_', ' ').title()}",
            )

        with col2:
            priority = st.select_slider(
                "Priority",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="Higher priority items are processed first",
            )

        # Dynamic form based on source type
        self._render_source_form(source_type, priority)

        # Batch upload section
        with st.expander("📦 Batch Processing"):
            self._render_batch_upload()

        # Recent jobs status
        self._render_job_status()

    def _render_source_form(self, source_type: str, priority: int):
        """Render form based on source type"""
        config = self.source_configs[source_type]

        with st.form(f"{source_type}_form"):
            st.subheader(
                f"{config['icon']} {source_type.replace('_', ' ').title()} Scraping"
            )

            # Common fields
            url = st.text_input("URL", key=f"{source_type}_url")
            domain = st.selectbox(
                "Domain",
                [
                    "mathematics",
                    "science",
                    "philosophy",
                    "religion",
                    "history",
                    "literature",
                ],
            )

            # Source-specific fields
            options = {}
            if "doi" in config["fields"]:
                options["doi"] = st.text_input("DOI (optional)")

            if "extract_citations" in config["fields"]:
                options["extract_citations"] = st.checkbox(
                    "Extract Citations", value=True
                )

            if "language" in config["fields"]:
                options["language"] = st.selectbox(
                    "Original Language",
                    ["Latin", "Greek", "Hebrew", "Sanskrit", "Arabic", "Other"],
                )

            if "include_transcript" in config["fields"]:
                options["include_transcript"] = st.checkbox(
                    "Include Transcript", value=True
                )

            # Submit button
            submitted = st.form_submit_button("🚀 Start Scraping")

            if submitted and url:
                self._process_scraping(url, domain, source_type, priority, options)

    @run_async
    async def _process_scraping(
        self, url: str, domain: str, source_type: str, priority: int, options: Dict
    ):
        """Process scraping request via API"""
        with st.spinner("🔄 Processing..."):
            result = await api_client.scrape_content(
                urls=[url],
                domain=domain,
                source_type=source_type,
                options={**options, "priority": priority},
            )

            if result and result.get("job_id"):
                st.success(f"✅ Scraping job started: {result['job_id']}")
                st.info("Check the job status below for progress updates")
            else:
                st.error("❌ Failed to start scraping job")

    def _render_batch_upload(self):
        """Render batch upload interface"""
        uploaded_file = st.file_uploader(
            "Upload CSV/JSON file with URLs", type=["csv", "json"]
        )

        if uploaded_file:
            # Parse and display preview
            st.info(f"📄 File: {uploaded_file.name}")
            # Implementation for file parsing and batch submission

    def _render_job_status(self):
        """Display recent job status"""
        st.subheader("📊 Recent Jobs")

        # This would fetch from API
        # For now, showing placeholder
        status_placeholder = st.empty()
        status_placeholder.info("Job status will appear here...")


# Initialize and render
scraper_ui = ContentScraperUI()
scraper_ui.render()
