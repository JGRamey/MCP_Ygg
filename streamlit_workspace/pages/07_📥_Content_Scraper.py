"""
Content Scraper - MCP Yggdrasil IDE Workspace
Multi-source content submission and scraping interface
"""

import streamlit as st
import asyncio
import json
import sys
import os
import uuid
import tempfile
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import validators
import requests
from urllib.parse import urlparse
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from data.staging_manager import StagingManager, ContentMetadata, SourceType, Priority, AgentPipeline
    from agents.youtube_transcript.youtube_agent import YouTubeAgent
    from agents.scraper.scraper_agent import ScraperAgent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    """Main content scraper interface"""
    
    st.set_page_config(
        page_title="Content Scraper",
        page_icon="📥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 10px 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-box:hover {
        border-color: #007bff;
        background-color: #e3f2fd;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,123,255,0.2);
    }
    
    .drag-active {
        border-color: #28a745 !important;
        background-color: #d4edda !important;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .upload-icon {
        font-size: 48px;
        color: #6c757d;
        margin-bottom: 10px;
    }
    
    .upload-text {
        font-size: 18px;
        font-weight: 500;
        color: #495057;
        margin-bottom: 8px;
    }
    
    .upload-hint {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 15px;
    }
    
    .status-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
        background-color: white;
    }
    
    .metrics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📥 Content Scraper")
    st.markdown("**Multi-source content submission and analysis pipeline**")
    
    # Initialize staging manager
    if 'staging_manager' not in st.session_state:
        st.session_state.staging_manager = StagingManager()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Submit Content", 
        "🔧 Configure Analysis", 
        "📊 Queue Status", 
        "📋 Recent Submissions"
    ])
    
    with tab1:
        render_content_submission()
    
    with tab2:
        render_analysis_configuration()
    
    with tab3:
        render_queue_status()
    
    with tab4:
        render_recent_submissions()

def render_content_submission():
    """Render content submission interface"""
    
    st.header("Content Submission")
    
    # Source type selection
    source_type = st.selectbox(
        "Select Content Source",
        ["URL (Website/Article)", "YouTube Video", "File Upload", "Direct Text Input"],
        help="Choose the type of content you want to scrape and analyze"
    )
    
    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    if source_type == "URL (Website/Article)":
        render_url_submission()
    elif source_type == "YouTube Video":
        render_youtube_submission()
    elif source_type == "File Upload":
        render_file_upload()
    elif source_type == "Direct Text Input":
        render_text_input()

def render_url_submission():
    """Render URL submission form"""
    
    st.subheader("🌐 Website/Article Scraping")
    
    with st.form("url_submission_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            url = st.text_input(
                "Enter URL",
                placeholder="https://example.com/article",
                help="Enter the full URL of the website or article to scrape"
            )
        
        with col2:
            validate_btn = st.form_submit_button("🔍 Validate URL", type="secondary")
        
        # URL validation
        if validate_btn and url:
            if validators.url(url):
                st.success("✅ Valid URL format")
                
                # Try to fetch basic info
                try:
                    response = requests.head(url, timeout=10)
                    st.info(f"📄 Content-Type: {response.headers.get('content-type', 'Unknown')}")
                except:
                    st.warning("⚠️ Could not fetch URL headers")
            else:
                st.error("❌ Invalid URL format")
        
        # Metadata inputs
        st.subheader("Content Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            title = st.text_input("Title", help="Content title (will be auto-detected if left empty)")
            domain = st.selectbox("Academic Domain", 
                                ["art", "science", "philosophy", "mathematics", "language", "technology"],
                                help="Primary academic domain for this content")
        
        with col2:
            author = st.text_input("Author", help="Content author (optional)")
            priority = st.selectbox("Processing Priority", ["high", "medium", "low"])
        
        with col3:
            language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"])
            date = st.date_input("Publication Date", help="Leave as today if unknown")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit for Scraping", type="primary")
        
        if submitted and url:
            if not validators.url(url):
                st.error("Please enter a valid URL")
            else:
                # Create submission
                asyncio.run(submit_url_content(url, title, author, domain, language, priority, str(date)))

def render_youtube_submission():
    """Render YouTube submission form"""
    
    st.subheader("📺 YouTube Video Processing")
    
    with st.form("youtube_submission_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter YouTube video URL (supports various formats)"
            )
        
        with col2:
            extract_info_btn = st.form_submit_button("📺 Get Video Info", type="secondary")
        
        # Video info extraction
        if extract_info_btn and youtube_url:
            video_info = extract_youtube_info(youtube_url)
            if video_info:
                st.session_state.youtube_info = video_info
                st.success("✅ Video information extracted")
        
        # Display video info if available
        if 'youtube_info' in st.session_state:
            info = st.session_state.youtube_info
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"📺 **Title:** {info.get('title', 'Unknown')}")
                st.info(f"👤 **Channel:** {info.get('channel', 'Unknown')}")
            
            with col2:
                st.info(f"⏱️ **Duration:** {info.get('duration', 'Unknown')}")
                st.info(f"🗣️ **Language:** {info.get('language', 'Unknown')}")
        
        # YouTube-specific options
        st.subheader("Transcript Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transcript_lang = st.selectbox("Preferred Language", 
                                         ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                                         help="Language for transcript extraction")
            include_timestamps = st.checkbox("Include Timestamps", value=True)
        
        with col2:
            auto_translate = st.checkbox("Auto-translate if needed", value=True)
            extract_chapters = st.checkbox("Extract Chapters", value=True)
        
        # Metadata
        st.subheader("Content Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            domain = st.selectbox("Academic Domain", 
                                ["art", "science", "philosophy", "mathematics", "language", "technology"],
                                help="Primary academic domain for this content",
                                key="yt_domain")
        
        with col2:
            priority = st.selectbox("Processing Priority", ["high", "medium", "low"], key="yt_priority")
        
        with col3:
            language = st.selectbox("Content Language", 
                                  ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                                  key="yt_language")
        
        # Submit button
        submitted = st.form_submit_button("🎬 Process YouTube Video", type="primary")
        
        if submitted and youtube_url:
            # Create YouTube submission
            asyncio.run(submit_youtube_content(
                youtube_url, domain, language, priority, 
                transcript_lang, include_timestamps, auto_translate, extract_chapters
            ))

def render_file_upload():
    """Render file upload form"""
    
    st.subheader("📄 File Upload Processing")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'md', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, DOCX, TXT, MD, JPG, PNG"
    )
    
    if uploaded_file:
        # Display file info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📁 **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📏 **Size:** {uploaded_file.size:,} bytes")
        with col3:
            st.info(f"🗂️ **Type:** {uploaded_file.type}")
        
        # Processing options based on file type
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png']:
            st.subheader("🖼️ Image Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                ocr_language = st.selectbox("OCR Language", ["eng", "spa", "fra", "deu"])
                enhance_image = st.checkbox("Enhance image quality", value=True)
            
            with col2:
                extract_tables = st.checkbox("Extract tables", value=False)
                preserve_layout = st.checkbox("Preserve layout", value=True)
            
            # Store OCR language in session state for processing
            st.session_state['ocr_language'] = ocr_language
        
        # Metadata form
        with st.form("file_upload_form"):
            st.subheader("Content Metadata")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                title = st.text_input("Title", value=uploaded_file.name)
                domain = st.selectbox("Academic Domain", 
                                    ["art", "science", "philosophy", "mathematics", "language", "technology"],
                                    key="file_domain")
            
            with col2:
                author = st.text_input("Author")
                priority = st.selectbox("Processing Priority", ["high", "medium", "low"], key="file_priority")
            
            with col3:
                language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                                      key="file_language")
                date = st.date_input("Date", key="file_date")
            
            # Submit button
            submitted = st.form_submit_button("📤 Upload and Process", type="primary")
            
            if submitted:
                # Save uploaded file and create submission
                asyncio.run(submit_file_content(uploaded_file, title, author, domain, language, priority, str(date)))

def render_text_input():
    """Render direct text input form"""
    
    st.subheader("✏️ Direct Text Input")
    
    with st.form("text_input_form"):
        # Text area
        text_content = st.text_area(
            "Enter or paste your content",
            height=300,
            placeholder="Paste your text content here...",
            help="Enter the text content you want to analyze"
        )
        
        # Character count
        if text_content:
            char_count = len(text_content)
            word_count = len(text_content.split())
            st.caption(f"📊 Characters: {char_count:,} | Words: {word_count:,}")
        
        # Metadata
        st.subheader("Content Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            title = st.text_input("Title", help="Give your content a descriptive title")
            domain = st.selectbox("Academic Domain", 
                                ["art", "science", "philosophy", "mathematics", "language", "technology"],
                                key="text_domain")
        
        with col2:
            author = st.text_input("Author")
            priority = st.selectbox("Processing Priority", ["high", "medium", "low"], key="text_priority")
        
        with col3:
            language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                                  key="text_language")
            date = st.date_input("Date", key="text_date")
        
        # Submit button
        submitted = st.form_submit_button("📝 Submit Text Content", type="primary")
        
        if submitted and text_content.strip():
            if not title.strip():
                st.error("Please provide a title for your content")
            else:
                # Create text submission
                asyncio.run(submit_text_content(text_content, title, author, domain, language, priority, str(date)))

def render_analysis_configuration():
    """Render analysis configuration interface"""
    
    st.header("🔧 Analysis Configuration")
    st.markdown("Configure which agents will analyze your content and in what order.")
    
    # Available agents
    available_agents = {
        "text_processor": {
            "name": "Text Processor",
            "description": "Extract entities, perform NLP analysis, and identify key concepts",
            "parameters": {
                "extract_entities": True,
                "extract_math": False,
                "language": "en"
            }
        },
        "claim_analyzer": {
            "name": "Claim Analyzer", 
            "description": "Fact-check claims and verify against existing knowledge",
            "parameters": {
                "confidence_threshold": 0.8,
                "verify_citations": True
            }
        },
        "concept_explorer": {
            "name": "Concept Explorer",
            "description": "Discover relationships and connections between concepts",
            "parameters": {
                "depth": 2,
                "cross_domain": True,
                "mathematical_focus": False
            }
        },
        "vector_indexer": {
            "name": "Vector Indexer",
            "description": "Create semantic embeddings for similarity search",
            "parameters": {
                "embedding_model": "general",
                "chunk_size": 512
            }
        }
    }
    
    # Agent selection
    st.subheader("Select Analysis Agents")
    
    selected_agents = []
    agent_parameters = {}
    
    for agent_id, agent_info in available_agents.items():
        with st.expander(f"🤖 {agent_info['name']}", expanded=False):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(agent_info['description'])
            
            with col2:
                is_selected = st.checkbox("Enable", key=f"agent_{agent_id}")
            
            if is_selected:
                selected_agents.append(agent_id)
                
                # Agent-specific parameters
                st.markdown("**Parameters:**")
                params = {}
                
                for param_name, default_value in agent_info['parameters'].items():
                    if isinstance(default_value, bool):
                        params[param_name] = st.checkbox(
                            param_name.replace('_', ' ').title(),
                            value=default_value,
                            key=f"param_{agent_id}_{param_name}"
                        )
                    elif isinstance(default_value, (int, float)):
                        params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=default_value,
                            key=f"param_{agent_id}_{param_name}"
                        )
                    else:
                        params[param_name] = st.text_input(
                            param_name.replace('_', ' ').title(),
                            value=str(default_value),
                            key=f"param_{agent_id}_{param_name}"
                        )
                
                agent_parameters[agent_id] = params
    
    # Processing order
    if selected_agents:
        st.subheader("Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            processing_order = st.radio(
                "Processing Order",
                ["sequential", "parallel"],
                help="Sequential: agents run one after another. Parallel: agents run simultaneously."
            )
        
        with col2:
            if processing_order == "sequential":
                st.markdown("**Agent Order:**")
                for i, agent_id in enumerate(selected_agents, 1):
                    st.write(f"{i}. {available_agents[agent_id]['name']}")
        
        # Save configuration
        if st.button("💾 Save Agent Configuration", type="primary"):
            st.session_state.agent_pipeline = AgentPipeline(
                selected_agents=selected_agents,
                processing_order=processing_order,
                agent_parameters=agent_parameters,
                completion_status={}
            )
            st.success("✅ Agent configuration saved!")
    
    else:
        st.info("👆 Select at least one agent to configure the analysis pipeline.")

def render_queue_status():
    """Render queue status and metrics"""
    
    st.header("📊 Processing Queue Status")
    
    # Get queue statistics
    try:
        stats = asyncio.run(st.session_state.staging_manager.get_queue_stats())
        
        if stats:
            # Status cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metrics-card">
                    <h3>{stats['queue_counts'].get('pending', 0)}</h3>
                    <p>Pending</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metrics-card">
                    <h3>{stats['queue_counts'].get('processing', 0)}</h3>
                    <p>Processing</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metrics-card">
                    <h3>{stats['queue_counts'].get('analyzed', 0)}</h3>
                    <p>Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metrics-card">
                    <h3>{stats['queue_counts'].get('approved', 0)}</h3>
                    <p>Approved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metrics-card">
                    <h3>{stats['queue_counts'].get('rejected', 0)}</h3>
                    <p>Rejected</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics
            st.subheader("📈 Processing Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Items", stats['total_items'])
                if stats.get('oldest_pending'):
                    st.metric("Oldest Pending", stats['oldest_pending'][:8] + "...")
            
            with col2:
                avg_time = stats.get('average_processing_time_minutes', 0)
                st.metric("Avg Processing Time", f"{avg_time:.1f} min")
                st.metric("Last Updated", stats.get('last_updated', 'Unknown')[:16])
        
        else:
            st.info("No queue statistics available.")
    
    except Exception as e:
        st.error(f"Error fetching queue status: {e}")
    
    # Manual queue management
    st.subheader("🛠️ Queue Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Stats"):
            st.experimental_rerun()
    
    with col2:
        if st.button("🧹 Cleanup Old Items"):
            cleaned = asyncio.run(st.session_state.staging_manager.cleanup_old_items(30))
            st.success(f"Cleaned up {cleaned} old items")
    
    with col3:
        if st.button("📊 Export Stats"):
            st.download_button(
                "Download Statistics",
                data=json.dumps(stats, indent=2),
                file_name=f"queue_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_recent_submissions():
    """Render recent submissions list"""
    
    st.header("📋 Recent Submissions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "pending", "processing", "analyzed", "approved", "rejected"]
        )
    
    with col2:
        domain_filter = st.selectbox(
            "Filter by Domain", 
            ["All", "art", "science", "philosophy", "mathematics", "language", "technology"]
        )
    
    with col3:
        limit = st.number_input("Items to show", min_value=5, max_value=100, value=20)
    
    # Get submissions
    try:
        status = None if status_filter == "All" else status_filter
        domain = None if domain_filter == "All" else domain_filter
        
        submissions = asyncio.run(st.session_state.staging_manager.list_content(
            status=status,
            domain=domain, 
            limit=limit
        ))
        
        if submissions:
            for submission in submissions:
                with st.expander(f"📄 {submission.metadata.title} ({submission.submission_id[:8]}...)", expanded=False):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Status:** {submission.processing_status.value}")
                        st.write(f"**Domain:** {submission.metadata.domain}")
                        st.write(f"**Source:** {submission.source_type.value}")
                        if submission.metadata.author:
                            st.write(f"**Author:** {submission.metadata.author}")
                    
                    with col2:
                        st.write(f"**Submitted:** {submission.timestamps.get('submitted', 'Unknown')[:16]}")
                        st.write(f"**Priority:** {submission.metadata.priority.value}")
                        st.write(f"**Language:** {submission.metadata.language}")
                        if submission.source_url:
                            st.write(f"**URL:** {submission.source_url[:50]}...")
                    
                    # Analysis results if available
                    if submission.analysis_results:
                        st.subheader("Analysis Results")
                        results = submission.analysis_results
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Quality Score:** {results.quality_score}")
                            st.write(f"**Confidence:** {results.confidence_level}")
                        
                        with col2:
                            st.write(f"**Concepts Found:** {len(results.concepts_extracted)}")
                            st.write(f"**Claims Identified:** {len(results.claims_identified)}")
                        
                        if results.concepts_extracted:
                            st.write("**Key Concepts:**")
                            st.write(", ".join(results.concepts_extracted[:10]))
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"📄 View Content", key=f"view_{submission.submission_id}"):
                            st.session_state.selected_submission = submission.submission_id
                    
                    with col2:
                        if submission.processing_status.value == "analyzed":
                            if st.button(f"✅ Approve", key=f"approve_{submission.submission_id}"):
                                asyncio.run(st.session_state.staging_manager.approve_content(
                                    submission.submission_id, "streamlit_user", "Approved via interface"
                                ))
                                st.success("Content approved!")
                                st.experimental_rerun()
                    
                    with col3:
                        if submission.processing_status.value == "analyzed":
                            if st.button(f"❌ Reject", key=f"reject_{submission.submission_id}"):
                                asyncio.run(st.session_state.staging_manager.reject_content(
                                    submission.submission_id, "streamlit_user", "Rejected via interface"
                                ))
                                st.warning("Content rejected!")
                                st.experimental_rerun()
        
        else:
            st.info("No submissions found matching the current filters.")
    
    except Exception as e:
        st.error(f"Error loading submissions: {e}")

# Helper functions for content submission

async def submit_url_content(url: str, title: str, author: str, domain: str, language: str, priority: str, date: str):
    """Submit URL content for processing"""
    try:
        # Create metadata
        metadata = ContentMetadata(
            title=title or "Web Content",
            author=author or None,
            date=date,
            domain=domain,
            language=language,
            priority=Priority(priority),
            submitted_by="streamlit_user",
            file_size=None,
            content_type="website"
        )
        
        # Get agent pipeline if configured
        agent_pipeline = st.session_state.get('agent_pipeline')
        
        # Submit to staging (we'll use placeholder content for now)
        submission_id = await st.session_state.staging_manager.submit_content(
            source_type=SourceType.WEBSITE,
            raw_content=f"URL content from: {url}",  # Would be replaced with actual scraping
            metadata=metadata,
            source_url=url,
            agent_pipeline=agent_pipeline
        )
        
        st.success(f"✅ Content submitted successfully! Submission ID: {submission_id[:8]}...")
        
    except Exception as e:
        st.error(f"Error submitting content: {e}")

async def submit_youtube_content(url: str, domain: str, language: str, priority: str, 
                                transcript_lang: str, include_timestamps: bool, 
                                auto_translate: bool, extract_chapters: bool):
    """Submit YouTube content for processing"""
    try:
        # Extract video info
        video_info = extract_youtube_info(url)
        
        # Create metadata
        metadata = ContentMetadata(
            title=video_info.get('title', 'YouTube Video'),
            author=video_info.get('channel', None),
            date=video_info.get('published_date', str(datetime.now().date())),
            domain=domain,
            language=language,
            priority=Priority(priority),
            submitted_by="streamlit_user",
            file_size=None,
            content_type="video"
        )
        
        # Get agent pipeline if configured
        agent_pipeline = st.session_state.get('agent_pipeline')
        
        # Submit to staging (placeholder content)
        submission_id = await st.session_state.staging_manager.submit_content(
            source_type=SourceType.YOUTUBE,
            raw_content=f"YouTube video transcript from: {url}",  # Would be replaced with actual transcript
            metadata=metadata,
            source_url=url,
            agent_pipeline=agent_pipeline
        )
        
        st.success(f"✅ YouTube video submitted successfully! Submission ID: {submission_id[:8]}...")
        
    except Exception as e:
        st.error(f"Error submitting YouTube content: {e}")

async def submit_file_content(uploaded_file, title: str, author: str, domain: str, language: str, priority: str, date: str):
    """Submit uploaded file for processing"""
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Process file content based on type
        raw_content = ""
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Use OCR for image files
                from agents.scraper.scraper_agent import OCRProcessor
                ocr_processor = OCRProcessor()
                ocr_language = st.session_state.get('ocr_language', 'eng')
                raw_content = ocr_processor.extract_text_from_image(str(file_path), ocr_language)
                st.info(f"🔍 OCR Processing: Extracted {len(raw_content)} characters from image")
                
            elif file_ext == '.pdf':
                # Use PDF processor with OCR fallback
                from agents.scraper.scraper_agent import PDFProcessor
                pdf_processor = PDFProcessor()
                raw_content, _ = pdf_processor.extract_text_from_pdf(str(file_path))
                st.info(f"📄 PDF Processing: Extracted {len(raw_content)} characters")
                
            elif file_ext in ['.txt', '.md']:
                # Read text files directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                st.info(f"📝 Text Processing: Loaded {len(raw_content)} characters")
                
            else:
                # Default fallback for other file types
                raw_content = f"File content from: {uploaded_file.name} (Content extraction not implemented for this file type)"
                st.warning(f"⚠️ Limited support for .{file_ext} files")
                
        except Exception as processing_error:
            st.warning(f"⚠️ Content extraction failed: {processing_error}")
            raw_content = f"File content from: {uploaded_file.name} (Content extraction failed)"
        
        # Create metadata
        metadata = ContentMetadata(
            title=title,
            author=author or None,
            date=date,
            domain=domain,
            language=language,
            priority=Priority(priority),
            submitted_by="streamlit_user",
            file_size=uploaded_file.size,
            content_type="document"
        )
        
        # Get agent pipeline if configured
        agent_pipeline = st.session_state.get('agent_pipeline')
        
        # Submit to staging with extracted content
        submission_id = await st.session_state.staging_manager.submit_content(
            source_type=SourceType.UPLOAD,
            raw_content=raw_content,
            metadata=metadata,
            source_url=f"upload://{file_path.name}",
            agent_pipeline=agent_pipeline
        )
        
        st.success(f"✅ File uploaded and submitted! Submission ID: {submission_id[:8]}...")
        
        # Show content preview if available
        if raw_content and len(raw_content) > 50:
            with st.expander("📋 Content Preview"):
                st.text_area("Extracted Content", raw_content[:500] + "..." if len(raw_content) > 500 else raw_content, height=200)
        
    except Exception as e:
        st.error(f"Error submitting file: {e}")

async def submit_text_content(text: str, title: str, author: str, domain: str, language: str, priority: str, date: str):
    """Submit text content for processing"""
    try:
        # Create metadata
        metadata = ContentMetadata(
            title=title,
            author=author or None,
            date=date,
            domain=domain,
            language=language,
            priority=Priority(priority),
            submitted_by="streamlit_user",
            file_size=len(text.encode('utf-8')),
            content_type="text"
        )
        
        # Get agent pipeline if configured
        agent_pipeline = st.session_state.get('agent_pipeline')
        
        # Submit to staging
        submission_id = await st.session_state.staging_manager.submit_content(
            source_type=SourceType.TEXT,
            raw_content=text,
            metadata=metadata,
            source_url=None,
            agent_pipeline=agent_pipeline
        )
        
        st.success(f"✅ Text content submitted successfully! Submission ID: {submission_id[:8]}...")
        
    except Exception as e:
        st.error(f"Error submitting text content: {e}")

def extract_youtube_info(url: str) -> Dict[str, Any]:
    """Extract basic YouTube video information"""
    try:
        # This is a placeholder - would use actual YouTube API
        video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)', url)
        
        if video_id_match:
            return {
                'title': 'Example YouTube Video',
                'channel': 'Example Channel',
                'duration': '10:30',
                'language': 'en',
                'published_date': '2024-01-15'
            }
        else:
            return {}
    
    except Exception:
        return {}

if __name__ == "__main__":
    main()