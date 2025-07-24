"""
Content Scraper UI Integration
Thin UI layer that integrates with existing scraper agents at /agents/scraper
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import existing scraper agents
try:
    from agents.scraper.core.unified_web_scraper import UnifiedWebScraper, ScraperConfig
    from agents.scraper.intelligent_scraper_agent import IntelligentScraperAgent, ContentType, AuthorityLevel
    from agents.scraper.config.scraper_profiles import ScraperProfileManager
    SCRAPER_AVAILABLE = True
except ImportError as e:
    st.error(f"Scraper agents not available: {e}")
    SCRAPER_AVAILABLE = False

class ContentScraperUI:
    """
    Thin UI layer that integrates with existing scraper agent system.
    
    This replaces the previous 783-line duplicate implementation with a clean
    interface that delegates actual scraping to the existing agent ecosystem.
    """
    
    def __init__(self):
        """Initialize the scraper UI interface."""
        if not SCRAPER_AVAILABLE:
            return
            
        # Initialize scraper components
        self.config = ScraperConfig()
        self.unified_scraper = UnifiedWebScraper(self.config)
        self.intelligent_scraper = IntelligentScraperAgent()
        self.profile_manager = ScraperProfileManager()
        
        # Initialize session state
        if 'scraper_history' not in st.session_state:
            st.session_state.scraper_history = []
        if 'active_jobs' not in st.session_state:
            st.session_state.active_jobs = {}

    def render_page(self):
        """Render the main content scraper page."""
        st.set_page_config(
            page_title="Content Scraper",
            page_icon="📥",
            layout="wide"
        )
        
        st.title("📥 Content Scraper")
        st.markdown("Multi-source content acquisition powered by existing scraper agents")
        
        if not SCRAPER_AVAILABLE:
            st.error("Scraper agents are not available. Please check the system setup.")
            return
        
        # Main interface tabs
        tabs = st.tabs(["📋 Single URL", "📚 Batch URLs", "📊 History", "⚙️ Settings"])
        
        with tabs[0]:
            self._render_single_url_tab()
        
        with tabs[1]:
            self._render_batch_urls_tab()
        
        with tabs[2]:
            self._render_history_tab()
        
        with tabs[3]:
            self._render_settings_tab()

    def _render_single_url_tab(self):
        """Render single URL scraping interface."""
        st.subheader("Single URL Scraping")
        
        # URL input
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="URL to scrape content from"
        )
        
        # Scraper profile selection
        col1, col2 = st.columns(2)
        
        with col1:
            profile = st.selectbox(
                "Scraper Profile",
                ["fast", "comprehensive", "stealth", "academic", "news", "social"],
                help="Pre-configured scraping profile"
            )
        
        with col2:
            domain = st.selectbox(
                "Content Domain",
                ["mathematics", "science", "philosophy", "religion", "art", "language"],
                help="Primary content domain for classification"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_selenium = st.checkbox("Use JavaScript rendering", value=False)
                extract_images = st.checkbox("Extract images", value=True)
                follow_redirects = st.checkbox("Follow redirects", value=True)
            
            with col2:
                min_content_length = st.number_input("Min content length", value=100, min_value=10)
                timeout = st.number_input("Timeout (seconds)", value=30, min_value=5, max_value=120)
                retry_attempts = st.number_input("Retry attempts", value=3, min_value=1, max_value=10)
        
        # Scrape button
        if st.button("🚀 Scrape URL", type="primary", disabled=not url):
            if url:
                self._process_single_url(url, profile, domain, {
                    'use_selenium': use_selenium,
                    'extract_images': extract_images,
                    'follow_redirects': follow_redirects,
                    'min_content_length': min_content_length,
                    'timeout': timeout,
                    'retry_attempts': retry_attempts
                })

    def _render_batch_urls_tab(self):
        """Render batch URL processing interface."""
        st.subheader("Batch URL Processing")
        
        # URL input
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_profile = st.selectbox(
                "Batch Profile",
                ["fast", "comprehensive", "stealth", "academic"],
                key="batch_profile"
            )
            batch_domain = st.selectbox(
                "Default Domain",
                ["Auto-detect"] + ["mathematics", "science", "philosophy", "religion", "art", "language"],
                key="batch_domain"
            )
        
        with col2:
            max_concurrent = st.number_input("Max concurrent", value=3, min_value=1, max_value=10)
            delay_between = st.number_input("Delay between requests (s)", value=1.0, min_value=0.1, max_value=10.0)
        
        if st.button("🚀 Process Batch", disabled=not urls_text):
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            if urls:
                self._process_batch_urls(urls, batch_profile, batch_domain, max_concurrent, delay_between)

    def _render_history_tab(self):
        """Render scraping history interface."""
        st.subheader("Scraping History")
        
        if st.session_state.scraper_history:
            # Create DataFrame from history
            history_df = pd.DataFrame(st.session_state.scraper_history)
            
            # Add filtering
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox("Filter by Status", ["All", "Success", "Failed", "Processing"])
            
            with col2:
                domain_filter = st.selectbox("Filter by Domain", ["All"] + list(history_df['domain'].unique()) if 'domain' in history_df.columns else ["All"])
            
            with col3:
                if st.button("Clear History"):
                    st.session_state.scraper_history = []
                    st.rerun()
            
            # Apply filters
            filtered_df = history_df.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter.lower()]
            if domain_filter != "All":
                filtered_df = filtered_df[filtered_df['domain'] == domain_filter]
            
            # Display results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download option
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download History CSV",
                    csv,
                    f"scraper_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        else:
            st.info("No scraping history available yet.")

    def _render_settings_tab(self):
        """Render scraper settings interface."""
        st.subheader("Scraper Configuration")
        
        # Display current scraper stats
        if hasattr(self, 'unified_scraper'):
            stats = self.unified_scraper.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Requests", stats.get('total_requests', 0))
            with col2:
                st.metric("Successful", stats.get('successful_requests', 0))
            with col3:
                st.metric("Failed", stats.get('failed_requests', 0))
            with col4:
                st.metric("Avg Time (s)", f"{stats.get('average_processing_time', 0):.2f}")
            
            # Method usage breakdown
            st.subheader("Method Usage")
            method_stats = stats.get('method_usage', {})
            if method_stats:
                method_df = pd.DataFrame(list(method_stats.items()), columns=['Method', 'Count'])
                st.bar_chart(method_df.set_index('Method'))
        
        # Configuration options
        st.subheader("Global Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rate Limiting**")
            new_rate = st.number_input("Requests per second", value=self.config.requests_per_second, min_value=0.1, max_value=10.0)
            new_timeout = st.number_input("Request timeout", value=self.config.timeout, min_value=5, max_value=180)
            
        with col2:
            st.write("**Content Extraction**")
            use_trafilatura = st.checkbox("Use Trafilatura", value=self.config.use_trafilatura)
            respect_robots = st.checkbox("Respect robots.txt", value=self.config.respect_robots_txt)
        
        if st.button("Update Settings"):
            self.config.requests_per_second = new_rate
            self.config.timeout = new_timeout
            self.config.use_trafilatura = use_trafilatura
            self.config.respect_robots_txt = respect_robots
            st.success("Settings updated!")
        
        # Clear cache option
        if st.button("Clear Scraper Cache"):
            if hasattr(self, 'unified_scraper'):
                self.unified_scraper.clear_cache()
                st.success("Cache cleared!")

    def _process_single_url(self, url: str, profile: str, domain: str, options: Dict):
        """Process a single URL using the existing scraper agents."""
        # Create progress indicators
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize
                status_text.text("Initializing scraper...")
                progress_bar.progress(0.2)
                
                # Configure scraper based on profile
                profile_config = self.profile_manager.get_profile(profile)
                if profile_config:
                    # Apply profile settings to config
                    for key, value in profile_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                
                # Step 2: Scrape content
                status_text.text("Scraping content...")
                progress_bar.progress(0.5)
                
                # Use asyncio to run the async scraper
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.unified_scraper.scrape_url(url))
                finally:
                    loop.close()
                
                if not result.success:
                    st.error(f"Scraping failed: {result.error}")
                    return
                
                # Step 3: Intelligent analysis
                status_text.text("Analyzing content...")
                progress_bar.progress(0.8)
                
                # Use intelligent scraper for classification
                if hasattr(self.intelligent_scraper, 'analyze_content'):
                    analysis = self.intelligent_scraper.analyze_content(result.content, url)
                else:
                    analysis = {'content_type': 'unknown', 'authority_score': 0.5}
                
                # Step 4: Complete
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)
                
                # Display results
                self._display_scraping_results(url, result, analysis, domain)
                
                # Add to history
                self._add_to_history(url, result, analysis, domain, "success")
                
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")
                self._add_to_history(url, None, None, domain, "failed", str(e))

    def _process_batch_urls(self, urls: List[str], profile: str, domain: str, max_concurrent: int, delay: float):
        """Process multiple URLs using the existing scraper agents."""
        total_urls = len(urls)
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            # Configure scraper
            self.config.requests_per_second = 1.0 / delay if delay > 0 else 1.0
            
            # Process in batches
            status_text.text(f"Processing {total_urls} URLs...")
            
            # Use asyncio for batch processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    self.unified_scraper.scrape_multiple(urls, max_concurrent=max_concurrent)
                )
            finally:
                loop.close()
            
            # Process results
            successful = 0
            failed = 0
            
            for i, (url, result) in enumerate(zip(urls, results)):
                progress_bar.progress((i + 1) / total_urls)
                
                if result.success:
                    successful += 1
                    self._add_to_history(url, result, {}, domain, "success")
                else:
                    failed += 1
                    self._add_to_history(url, result, {}, domain, "failed", result.error)
            
            # Show summary
            with results_container:
                st.success(f"Batch processing complete! ✅ {successful} successful, ❌ {failed} failed")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Successful", successful)
                with col2:
                    st.metric("Failed", failed)
        
        except Exception as e:
            st.error(f"Batch processing error: {str(e)}")

    def _display_scraping_results(self, url: str, result, analysis: Dict, domain: str):
        """Display results from scraping operation."""
        st.subheader("Scraping Results")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
        
        with col2:
            st.metric("Method Used", result.method_used.title())
        
        with col3:
            content_length = len(result.content.get('text', '')) if result.content else 0
            st.metric("Content Length", f"{content_length:,} chars")
        
        # Content preview
        if result.content and 'text' in result.content:
            st.subheader("Content Preview")
            preview_text = result.content['text'][:500] + "..." if len(result.content['text']) > 500 else result.content['text']
            st.text_area("Content", preview_text, height=150, disabled=True)
        
        # Metadata
        if result.metadata:
            st.subheader("Metadata")
            st.json(result.metadata)
        
        # Analysis results
        if analysis:
            st.subheader("Content Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Content Type**: {analysis.get('content_type', 'Unknown')}")
                st.write(f"**Authority Score**: {analysis.get('authority_score', 0):.2f}")
            
            with col2:
                st.write(f"**Domain**: {domain}")
                st.write(f"**Language**: {analysis.get('language', 'Unknown')}")

    def _add_to_history(self, url: str, result, analysis: Dict, domain: str, status: str, error: str = None):
        """Add scraping result to history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'domain': domain,
            'status': status,
            'processing_time': result.processing_time if result else 0,
            'method_used': result.method_used if result else 'unknown',
            'content_length': len(result.content.get('text', '')) if result and result.content else 0,
            'content_type': analysis.get('content_type', 'unknown') if analysis else 'unknown',
            'authority_score': analysis.get('authority_score', 0) if analysis else 0,
            'error': error
        }
        
        st.session_state.scraper_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(st.session_state.scraper_history) > 100:
            st.session_state.scraper_history = st.session_state.scraper_history[-100:]


def main():
    """Main function to render the content scraper interface."""
    scraper_ui = ContentScraperUI()
    scraper_ui.render_page()


if __name__ == "__main__":
    main()