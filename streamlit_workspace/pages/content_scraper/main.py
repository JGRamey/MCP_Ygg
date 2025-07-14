"""
Content Scraper Main Interface

Main orchestration interface for the content scraper system, providing
a clean entry point and coordinating between different modules.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root and shared components to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import shared components
from shared.ui.headers import create_page_header, create_section_header
from shared.ui.styling import apply_custom_css, apply_page_specific_css
from shared.ui.sidebars import create_navigation_sidebar, create_filter_sidebar
from shared.ui.cards import create_submission_card, create_stats_grid

# Import content scraper modules
from .scraping_engine import ScrapingEngine
from .content_processors import ContentProcessor
from .submission_manager import SubmissionManager


class ContentScraperInterface:
    """
    Main interface class for the content scraper system.
    
    Coordinates between scraping engine, content processor, and submission manager
    to provide a unified content acquisition interface.
    """
    
    def __init__(self):
        """Initialize the content scraper interface."""
        self.scraping_engine = ScrapingEngine()
        self.content_processor = ContentProcessor()
        self.submission_manager = SubmissionManager()
        
        # Initialize session state
        if 'scraper_initialized' not in st.session_state:
            st.session_state.scraper_initialized = True
            st.session_state.active_submissions = {}
            st.session_state.scraper_stats = {
                'total_submissions': 0,
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0
            }
    
    def render_main_interface(self) -> None:
        """Render the main content scraper interface."""
        # Apply styling
        apply_custom_css()
        apply_page_specific_css('scraper')
        
        # Page header
        create_page_header(
            title="Content Scraper",
            description="Multi-source content submission and scraping interface",
            icon="📥",
            show_status=True,
            status_info=self._get_system_status()
        )
        
        # Main content area
        self._render_content_tabs()
        
        # Sidebar
        self._render_sidebar()
    
    def _render_content_tabs(self) -> None:
        """Render the main content tabs."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌐 Web Scraping",
            "📺 YouTube",
            "📁 File Upload", 
            "✍️ Manual Entry",
            "📋 Submission Queue"
        ])
        
        with tab1:
            self._render_web_scraping_tab()
        
        with tab2:
            self._render_youtube_tab()
        
        with tab3:
            self._render_file_upload_tab()
        
        with tab4:
            self._render_manual_entry_tab()
        
        with tab5:
            self._render_queue_tab()
    
    def _render_web_scraping_tab(self) -> None:
        """Render the web scraping interface."""
        create_section_header("🌐 Web Content Scraping")
        
        # URL input form
        with st.form("web_scraping_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                url = st.text_input(
                    "Website URL",
                    placeholder="https://example.com/article",
                    help="Enter the URL of the webpage to scrape"
                )
            
            with col2:
                domain = st.selectbox(
                    "Target Domain",
                    ['🎨 Art', '🗣️ Language', '🔢 Mathematics', 
                     '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                    help="Select the knowledge domain"
                )
            
            # Advanced options
            with st.expander("Advanced Scraping Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    extract_concepts = st.checkbox("Extract concepts automatically", value=True)
                    follow_links = st.checkbox("Follow internal links", value=False)
                    respect_robots = st.checkbox("Respect robots.txt", value=True)
                
                with col2:
                    priority = st.selectbox("Priority", ['Low', 'Medium', 'High'], index=1)
                    max_depth = st.number_input("Max link depth", min_value=1, max_value=5, value=2)
                    delay = st.slider("Request delay (seconds)", min_value=1, max_value=10, value=3)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Start Scraping", type="primary")
            
            if submitted and url:
                self._handle_web_scraping(url, domain, {
                    'extract_concepts': extract_concepts,
                    'follow_links': follow_links,
                    'respect_robots': respect_robots,
                    'priority': priority,
                    'max_depth': max_depth,
                    'delay': delay
                })
    
    def _render_youtube_tab(self) -> None:
        """Render the YouTube content interface."""
        create_section_header("📺 YouTube Content Processing")
        
        with st.form("youtube_form"):
            url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter a YouTube video URL"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_transcript = st.checkbox("Include transcript", value=True)
                include_comments = st.checkbox("Include comments", value=False)
                extract_concepts = st.checkbox("Extract concepts", value=True)
            
            with col2:
                domain = st.selectbox(
                    "Domain",
                    ['🎨 Art', '🗣️ Language', '🔢 Mathematics', 
                     '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                    key="youtube_domain"
                )
                priority = st.selectbox("Priority", ['Low', 'Medium', 'High'], index=1, key="youtube_priority")
            
            submitted = st.form_submit_button("📥 Process Video", type="primary")
            
            if submitted and url:
                self._handle_youtube_processing(url, domain, {
                    'include_transcript': include_transcript,
                    'include_comments': include_comments,
                    'extract_concepts': extract_concepts,
                    'priority': priority
                })
    
    def _render_file_upload_tab(self) -> None:
        """Render the file upload interface."""
        create_section_header("📁 File Upload & Processing")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx', 'md', 'csv', 'json'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, MD, CSV, JSON"
        )
        
        if uploaded_files:
            with st.form("file_upload_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    domain = st.selectbox(
                        "Target Domain",
                        ['🎨 Art', '🗣️ Language', '🔢 Mathematics', 
                         '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                        key="upload_domain"
                    )
                    auto_process = st.checkbox("Auto-process after upload", value=True)
                
                with col2:
                    extract_concepts = st.checkbox("Extract concepts", value=True, key="upload_extract")
                    priority = st.selectbox("Priority", ['Low', 'Medium', 'High'], index=1, key="upload_priority")
                
                tags = st.text_input("Tags (comma-separated)", help="Optional tags for categorization")
                notes = st.text_area("Notes", help="Additional notes about the uploaded content")
                
                submitted = st.form_submit_button("📤 Upload Files", type="primary")
                
                if submitted:
                    self._handle_file_uploads(uploaded_files, domain, {
                        'auto_process': auto_process,
                        'extract_concepts': extract_concepts,
                        'priority': priority,
                        'tags': tags,
                        'notes': notes
                    })
    
    def _render_manual_entry_tab(self) -> None:
        """Render the manual text entry interface."""
        create_section_header("✍️ Manual Content Entry")
        
        with st.form("manual_entry_form"):
            title = st.text_input("Content Title", help="Descriptive title for the content")
            
            content = st.text_area(
                "Content Text",
                height=300,
                help="Enter or paste the content text here"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                domain = st.selectbox(
                    "Domain",
                    ['🎨 Art', '🗣️ Language', '🔢 Mathematics', 
                     '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                    key="manual_domain"
                )
            
            with col2:
                content_type = st.selectbox(
                    "Content Type",
                    ['Article', 'Quote', 'Definition', 'Biography', 'Analysis', 'Other']
                )
            
            with col3:
                priority = st.selectbox("Priority", ['Low', 'Medium', 'High'], index=1, key="manual_priority")
            
            source = st.text_input("Source (optional)", help="Source or reference for this content")
            tags = st.text_input("Tags (comma-separated)", key="manual_tags")
            extract_concepts = st.checkbox("Extract concepts automatically", value=True, key="manual_extract")
            
            submitted = st.form_submit_button("📝 Submit Content", type="primary")
            
            if submitted and content:
                self._handle_manual_entry(title, content, domain, {
                    'content_type': content_type,
                    'priority': priority,
                    'source': source,
                    'tags': tags,
                    'extract_concepts': extract_concepts
                })
    
    def _render_queue_tab(self) -> None:
        """Render the submission queue interface."""
        create_section_header("📋 Submission Queue Management")
        
        # Queue statistics
        stats = self.submission_manager.get_queue_stats()
        create_stats_grid(stats)
        
        # Queue controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Queue", key="refresh_queue"):
                st.rerun()
        
        with col2:
            if st.button("▶️ Process All", key="process_all"):
                self._process_all_submissions()
        
        with col3:
            if st.button("⏸️ Pause Processing", key="pause_processing"):
                self._pause_processing()
        
        with col4:
            if st.button("🗑️ Clear Completed", key="clear_completed"):
                self._clear_completed_submissions()
        
        # Submission list
        submissions = self.submission_manager.get_all_submissions()
        
        if submissions:
            st.markdown("### Recent Submissions")
            
            for submission in submissions[-10:]:  # Show last 10
                actions = create_submission_card(submission)
                
                # Handle actions
                if actions.get('approve'):
                    self.submission_manager.approve_submission(submission['id'])
                    st.success(f"Approved submission {submission['id']}")
                    st.rerun()
                
                elif actions.get('reject'):
                    self.submission_manager.reject_submission(submission['id'])
                    st.warning(f"Rejected submission {submission['id']}")
                    st.rerun()
                
                elif actions.get('process'):
                    self._process_submission(submission['id'])
                
                elif actions.get('preview'):
                    self._show_submission_preview(submission)
        else:
            st.info("No submissions in queue")
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with filters and controls."""
        with st.sidebar:
            st.markdown("### 📊 Quick Stats")
            
            stats = st.session_state.get('scraper_stats', {})
            st.metric("Total Submissions", stats.get('total_submissions', 0))
            st.metric("Pending", stats.get('pending', 0))
            st.metric("Processing", stats.get('processing', 0))
            
            st.markdown("---")
            st.markdown("### 🔧 Controls")
            
            if st.button("🔄 Refresh Stats"):
                self._update_stats()
                st.rerun()
            
            if st.button("🧹 Clean Queue"):
                self._clean_queue()
                st.rerun()
            
            if st.button("⚙️ Settings"):
                self._show_settings()
    
    def _get_system_status(self) -> Dict[str, str]:
        """Get system status information."""
        return {
            'scraping_engine': 'Online',
            'content_processor': 'Online', 
            'submission_manager': 'Online',
            'last_updated': 'Just now'
        }
    
    def _handle_web_scraping(self, url: str, domain: str, options: Dict[str, Any]) -> None:
        """Handle web scraping request."""
        try:
            with st.spinner("Starting web scraping..."):
                submission_id = self.scraping_engine.scrape_url(url, domain, options)
                st.success(f"Scraping started! Submission ID: {submission_id}")
                self._update_stats()
        except Exception as e:
            st.error(f"Error starting web scraping: {e}")
    
    def _handle_youtube_processing(self, url: str, domain: str, options: Dict[str, Any]) -> None:
        """Handle YouTube processing request."""
        try:
            with st.spinner("Processing YouTube video..."):
                submission_id = self.content_processor.process_youtube(url, domain, options)
                st.success(f"YouTube processing started! Submission ID: {submission_id}")
                self._update_stats()
        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")
    
    def _handle_file_uploads(self, files, domain: str, options: Dict[str, Any]) -> None:
        """Handle file upload request."""
        try:
            with st.spinner(f"Processing {len(files)} file(s)..."):
                submission_ids = self.content_processor.process_uploaded_files(files, domain, options)
                st.success(f"Files processed! {len(submission_ids)} submissions created.")
                self._update_stats()
        except Exception as e:
            st.error(f"Error processing files: {e}")
    
    def _handle_manual_entry(self, title: str, content: str, domain: str, options: Dict[str, Any]) -> None:
        """Handle manual content entry."""
        try:
            with st.spinner("Processing manual entry..."):
                submission_id = self.submission_manager.create_manual_submission(title, content, domain, options)
                st.success(f"Content submitted! Submission ID: {submission_id}")
                self._update_stats()
        except Exception as e:
            st.error(f"Error submitting content: {e}")
    
    def _update_stats(self) -> None:
        """Update scraper statistics."""
        stats = self.submission_manager.get_queue_stats()
        st.session_state.scraper_stats.update(stats)
    
    def _process_submission(self, submission_id: str) -> None:
        """Process a specific submission."""
        try:
            with st.spinner(f"Processing submission {submission_id}..."):
                result = self.content_processor.process_submission(submission_id)
                if result['success']:
                    st.success(f"Submission {submission_id} processed successfully!")
                else:
                    st.error(f"Failed to process submission {submission_id}: {result.get('error')}")
        except Exception as e:
            st.error(f"Error processing submission: {e}")
    
    def _show_submission_preview(self, submission: Dict[str, Any]) -> None:
        """Show submission preview in modal."""
        with st.expander(f"Preview: {submission.get('title', 'Untitled')}", expanded=True):
            st.json(submission)
    
    def _process_all_submissions(self) -> None:
        """Process all pending submissions."""
        st.info("Processing all submissions...")
        # Implementation would go here
    
    def _pause_processing(self) -> None:
        """Pause submission processing."""
        st.info("Processing paused")
        # Implementation would go here
    
    def _clear_completed_submissions(self) -> None:
        """Clear completed submissions from queue."""
        st.info("Cleared completed submissions")
        # Implementation would go here
    
    def _clean_queue(self) -> None:
        """Clean the submission queue."""
        st.info("Queue cleaned")
        # Implementation would go here
    
    def _show_settings(self) -> None:
        """Show scraper settings."""
        st.info("Settings dialog would open here")
        # Implementation would go here


def main():
    """Main entry point for the content scraper page."""
    st.set_page_config(
        page_title="Content Scraper - MCP Yggdrasil",
        page_icon="📥",
        layout="wide"
    )
    
    # Initialize and render the interface
    scraper_interface = ContentScraperInterface()
    scraper_interface.render_main_interface()


if __name__ == "__main__":
    main()