"""
Form components for user input

Provides reusable form components extracted from existing pages to ensure
consistent form layouts and validation across the workspace.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, date


def create_concept_form(concept_data: Optional[Dict[str, Any]] = None, 
                       mode: str = "create") -> Dict[str, Any]:
    """
    Create a form for concept creation or editing.
    
    Args:
        concept_data: Existing concept data (for edit mode)
        mode: Form mode ('create' or 'edit')
    
    Returns:
        Dictionary containing form data and submission status
    """
    form_data = {}
    
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown(f'<div class="form-header">{"Edit" if mode == "edit" else "Create"} Concept</div>', 
                unsafe_allow_html=True)
    
    with st.form(f"concept_form_{mode}"):
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            form_data['title'] = st.text_input(
                "Title *",
                value=concept_data.get('title', '') if concept_data else '',
                help="Enter a descriptive title for the concept"
            )
            
            form_data['domain'] = st.selectbox(
                "Domain *",
                ['🎨 Art', '🗣️ Language', '🔢 Mathematics', '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                index=0 if not concept_data else 
                ['🎨 Art', '🗣️ Language', '🔢 Mathematics', '🤔 Philosophy', '🔬 Science', '💻 Technology'].index(
                    concept_data.get('domain', '🎨 Art')
                ),
                help="Select the primary domain for this concept"
            )
        
        with col2:
            form_data['type'] = st.selectbox(
                "Type *",
                ['Person', 'Place', 'Event', 'Idea', 'Work', 'Movement', 'Institution'],
                index=0 if not concept_data else 
                ['Person', 'Place', 'Event', 'Idea', 'Work', 'Movement', 'Institution'].index(
                    concept_data.get('type', 'Person')
                ),
                help="Select the type of concept"
            )
            
            form_data['priority'] = st.selectbox(
                "Priority",
                ['Low', 'Medium', 'High', 'Critical'],
                index=1 if not concept_data else 
                ['Low', 'Medium', 'High', 'Critical'].index(
                    concept_data.get('priority', 'Medium')
                ),
                help="Set the priority level for processing"
            )
        
        # Description
        form_data['description'] = st.text_area(
            "Description *",
            value=concept_data.get('description', '') if concept_data else '',
            height=100,
            help="Provide a detailed description of the concept"
        )
        
        # Tags and metadata
        col1, col2 = st.columns(2)
        
        with col1:
            form_data['tags'] = st.text_input(
                "Tags",
                value=', '.join(concept_data.get('tags', [])) if concept_data else '',
                help="Enter comma-separated tags"
            )
        
        with col2:
            form_data['source'] = st.text_input(
                "Source",
                value=concept_data.get('source', '') if concept_data else '',
                help="Source of the concept information"
            )
        
        # Additional properties
        with st.expander("Additional Properties", expanded=False):
            form_data['birth_date'] = st.date_input(
                "Birth/Start Date",
                value=concept_data.get('birth_date') if concept_data else None,
                help="Birth date for persons or start date for events"
            )
            
            form_data['death_date'] = st.date_input(
                "Death/End Date", 
                value=concept_data.get('death_date') if concept_data else None,
                help="Death date for persons or end date for events"
            )
            
            form_data['location'] = st.text_input(
                "Location",
                value=concept_data.get('location', '') if concept_data else '',
                help="Geographic location associated with the concept"
            )
            
            form_data['external_links'] = st.text_area(
                "External Links",
                value=concept_data.get('external_links', '') if concept_data else '',
                help="One URL per line"
            )
        
        # Form submission
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            form_data['submitted'] = st.form_submit_button(
                f"{'Update' if mode == 'edit' else 'Create'} Concept",
                type="primary"
            )
        
        with col2:
            form_data['save_draft'] = st.form_submit_button("Save Draft")
        
        with col3:
            form_data['cancel'] = st.form_submit_button("Cancel")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation
    if form_data.get('submitted') or form_data.get('save_draft'):
        form_data['is_valid'] = bool(
            form_data.get('title') and 
            form_data.get('domain') and 
            form_data.get('description')
        )
        
        if not form_data['is_valid']:
            st.error("Please fill in all required fields (marked with *)")
    
    return form_data


def create_upload_form(accepted_types: List[str] = None, 
                      max_file_size: int = 200) -> Dict[str, Any]:
    """
    Create a file upload form with validation.
    
    Args:
        accepted_types: List of accepted file extensions
        max_file_size: Maximum file size in MB
    
    Returns:
        Dictionary containing upload data and files
    """
    if accepted_types is None:
        accepted_types = ['txt', 'pdf', 'docx', 'md', 'csv', 'json']
    
    upload_data = {}
    
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-header">📁 File Upload</div>', unsafe_allow_html=True)
    
    with st.form("file_upload_form"):
        # File uploader
        upload_data['files'] = st.file_uploader(
            "Choose files",
            type=accepted_types,
            accept_multiple_files=True,
            help=f"Accepted types: {', '.join(accepted_types)}. Max size: {max_file_size}MB per file"
        )
        
        # Upload options
        col1, col2 = st.columns(2)
        
        with col1:
            upload_data['domain'] = st.selectbox(
                "Target Domain",
                ['🎨 Art', '🗣️ Language', '🔢 Mathematics', '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                help="Select the domain for uploaded content"
            )
            
            upload_data['auto_process'] = st.checkbox(
                "Auto-process after upload",
                value=True,
                help="Automatically process files after upload"
            )
        
        with col2:
            upload_data['priority'] = st.selectbox(
                "Processing Priority",
                ['Low', 'Medium', 'High'],
                index=1,
                help="Set processing priority"
            )
            
            upload_data['extract_concepts'] = st.checkbox(
                "Extract concepts automatically",
                value=True,
                help="Automatically extract concepts from uploaded content"
            )
        
        # Additional options
        upload_data['tags'] = st.text_input(
            "Tags (optional)",
            help="Comma-separated tags to apply to uploaded content"
        )
        
        upload_data['notes'] = st.text_area(
            "Notes (optional)",
            height=60,
            help="Additional notes about the uploaded content"
        )
        
        # Submit button
        upload_data['submitted'] = st.form_submit_button(
            "📤 Upload Files",
            type="primary"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File validation
    if upload_data.get('files') and upload_data.get('submitted'):
        upload_data['valid_files'] = []
        upload_data['errors'] = []
        
        for file in upload_data['files']:
            # Check file size
            if file.size > max_file_size * 1024 * 1024:
                upload_data['errors'].append(f"File {file.name} exceeds {max_file_size}MB limit")
                continue
            
            # Check file type
            file_ext = file.name.split('.')[-1].lower()
            if file_ext not in accepted_types:
                upload_data['errors'].append(f"File {file.name} has unsupported type: {file_ext}")
                continue
            
            upload_data['valid_files'].append(file)
        
        upload_data['is_valid'] = len(upload_data['valid_files']) > 0
        
        if upload_data['errors']:
            for error in upload_data['errors']:
                st.error(error)
    
    return upload_data


def create_search_form(search_types: List[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive search form.
    
    Args:
        search_types: List of available search types
    
    Returns:
        Dictionary containing search parameters
    """
    if search_types is None:
        search_types = ['Text Search', 'Semantic Search', 'Graph Query']
    
    search_data = {}
    
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-header">🔍 Advanced Search</div>', unsafe_allow_html=True)
    
    with st.form("search_form"):
        # Search query
        search_data['query'] = st.text_input(
            "Search Query *",
            help="Enter your search terms or query"
        )
        
        # Search configuration
        col1, col2 = st.columns(2)
        
        with col1:
            search_data['search_type'] = st.selectbox(
                "Search Type",
                search_types,
                help="Choose the type of search to perform"
            )
            
            search_data['domains'] = st.multiselect(
                "Domains",
                ['🎨 Art', '🗣️ Language', '🔢 Mathematics', '🤔 Philosophy', '🔬 Science', '💻 Technology'],
                default=[],
                help="Filter by specific domains"
            )
        
        with col2:
            search_data['max_results'] = st.number_input(
                "Max Results",
                min_value=1,
                max_value=100,
                value=10,
                help="Maximum number of results to return"
            )
            
            search_data['sort_by'] = st.selectbox(
                "Sort By",
                ['Relevance', 'Date Created', 'Date Modified', 'Title', 'Domain'],
                help="Sort order for results"
            )
        
        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            if search_data.get('search_type') == 'Text Search':
                search_data['case_sensitive'] = st.checkbox("Case Sensitive")
                search_data['whole_words'] = st.checkbox("Whole Words Only")
                search_data['use_regex'] = st.checkbox("Use Regular Expressions")
            
            elif search_data.get('search_type') == 'Semantic Search':
                search_data['similarity_threshold'] = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05
                )
                search_data['include_related'] = st.checkbox("Include Related Concepts")
            
            elif search_data.get('search_type') == 'Graph Query':
                search_data['max_depth'] = st.number_input(
                    "Max Relationship Depth",
                    min_value=1,
                    max_value=10,
                    value=3
                )
                search_data['relationship_types'] = st.multiselect(
                    "Relationship Types",
                    ['Related_To', 'Part_Of', 'Created_By', 'Influenced_By', 'Located_In'],
                    default=[]
                )
        
        # Date range filter
        date_filter = st.checkbox("Filter by Date Range")
        if date_filter:
            col1, col2 = st.columns(2)
            with col1:
                search_data['start_date'] = st.date_input("Start Date")
            with col2:
                search_data['end_date'] = st.date_input("End Date")
        
        # Submit button
        search_data['submitted'] = st.form_submit_button(
            "🔍 Search",
            type="primary"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation
    if search_data.get('submitted'):
        search_data['is_valid'] = bool(search_data.get('query', '').strip())
        
        if not search_data['is_valid']:
            st.error("Please enter a search query")
    
    return search_data


def create_settings_form(current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a settings configuration form.
    
    Args:
        current_settings: Current application settings
    
    Returns:
        Dictionary containing updated settings and submission status
    """
    settings_data = {}
    
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-header">⚙️ Application Settings</div>', unsafe_allow_html=True)
    
    with st.form("settings_form"):
        # Database settings
        st.markdown("**Database Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            settings_data['neo4j_uri'] = st.text_input(
                "Neo4j URI",
                value=current_settings.get('neo4j_uri', 'bolt://localhost:7687'),
                help="Neo4j database connection URI"
            )
            
            settings_data['qdrant_host'] = st.text_input(
                "Qdrant Host",
                value=current_settings.get('qdrant_host', 'localhost'),
                help="Qdrant vector database host"
            )
        
        with col2:
            settings_data['redis_host'] = st.text_input(
                "Redis Host",
                value=current_settings.get('redis_host', 'localhost'),
                help="Redis cache host"
            )
            
            settings_data['cache_ttl'] = st.number_input(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=3600,
                value=current_settings.get('cache_ttl', 300),
                help="Default cache time-to-live"
            )
        
        # Performance settings
        st.markdown("**Performance Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            settings_data['max_workers'] = st.number_input(
                "Max Workers",
                min_value=1,
                max_value=16,
                value=current_settings.get('max_workers', 4),
                help="Maximum number of worker threads"
            )
            
            settings_data['query_timeout'] = st.number_input(
                "Query Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=current_settings.get('query_timeout', 30),
                help="Database query timeout"
            )
        
        with col2:
            settings_data['batch_size'] = st.number_input(
                "Batch Size",
                min_value=10,
                max_value=1000,
                value=current_settings.get('batch_size', 100),
                help="Processing batch size"
            )
            
            settings_data['enable_caching'] = st.checkbox(
                "Enable Caching",
                value=current_settings.get('enable_caching', True),
                help="Enable result caching"
            )
        
        # UI settings
        st.markdown("**UI Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            settings_data['theme'] = st.selectbox(
                "Theme",
                ['Light', 'Dark', 'Auto'],
                index=['Light', 'Dark', 'Auto'].index(current_settings.get('theme', 'Light')),
                help="Application theme"
            )
            
            settings_data['items_per_page'] = st.number_input(
                "Items per Page",
                min_value=5,
                max_value=100,
                value=current_settings.get('items_per_page', 20),
                help="Number of items to display per page"
            )
        
        with col2:
            settings_data['auto_refresh'] = st.checkbox(
                "Auto Refresh",
                value=current_settings.get('auto_refresh', False),
                help="Enable automatic page refresh"
            )
            
            if settings_data['auto_refresh']:
                settings_data['refresh_interval'] = st.number_input(
                    "Refresh Interval (seconds)",
                    min_value=30,
                    max_value=300,
                    value=current_settings.get('refresh_interval', 60)
                )
        
        # Security settings
        with st.expander("Security Settings", expanded=False):
            settings_data['enable_auth'] = st.checkbox(
                "Enable Authentication",
                value=current_settings.get('enable_auth', False),
                help="Enable user authentication"
            )
            
            settings_data['session_timeout'] = st.number_input(
                "Session Timeout (minutes)",
                min_value=5,
                max_value=480,
                value=current_settings.get('session_timeout', 60),
                help="User session timeout"
            )
            
            settings_data['api_rate_limit'] = st.number_input(
                "API Rate Limit (requests/minute)",
                min_value=10,
                max_value=1000,
                value=current_settings.get('api_rate_limit', 100),
                help="API request rate limit"
            )
        
        # Form submission
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            settings_data['submitted'] = st.form_submit_button(
                "💾 Save Settings",
                type="primary"
            )
        
        with col2:
            settings_data['reset'] = st.form_submit_button("🔄 Reset")
        
        with col3:
            settings_data['test'] = st.form_submit_button("🧪 Test")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return settings_data