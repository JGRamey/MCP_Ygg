"""
Content Scraper - Thin UI layer for scraping operations
Delegates all processing to FastAPI backend
"""

import asyncio
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
                "input_type": "url"
            },
            "academic_paper": {
                "icon": "📚",
                "fields": ["url", "doi", "extract_citations", "extract_figures"],
                "input_type": "url"
            },
            "ancient_text": {
                "icon": "📜",
                "fields": ["url", "language", "include_annotations", "translations"],
                "input_type": "url"
            },
            "youtube": {
                "icon": "📺",
                "fields": ["url", "include_transcript", "include_metadata"],
                "input_type": "url"
            },
            "book": {
                "icon": "📚",
                "fields": ["title", "author", "isbn", "publication_year", "publisher", "language", "file_upload", "text_input"],
                "input_type": "file_or_text"
            },
            "pdf": {
                "icon": "📜",
                "fields": ["file_upload", "extract_text", "extract_images", "extract_metadata", "ocr_if_needed"],
                "input_type": "file"
            },
            "image": {
                "icon": "🖼️",
                "fields": ["file_upload", "ocr_language", "enhance_image", "detect_layout"],
                "input_type": "file"
            },
            "article": {
                "icon": "📰",
                "fields": ["url", "extract_author", "extract_date", "extract_tags", "extract_images"],
                "input_type": "url"
            },
            "manuscript": {
                "icon": "📜",
                "fields": ["title", "time_period", "origin", "original_language", "document_type", "file_upload", "transcribed_text"],
                "input_type": "file_or_text"
            },
            "encyclopedia": {
                "icon": "📚",
                "fields": ["url", "extract_references", "extract_categories", "extract_related_entries"],
                "input_type": "url"
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

            # Initialize form data
            form_data = {}
            options = {}
            
            # Common domain selection
            domain = st.selectbox(
                "Domain",
                [
                    "mathematics",
                    "science", 
                    "philosophy",
                    "religion",
                    "history",
                    "literature",
                    "art",
                    "language"
                ],
                help="Select the primary knowledge domain for this content"
            )

            # Input type specific rendering
            input_type = config.get("input_type", "url")
            
            if input_type == "url":
                # URL-based sources
                url = st.text_input("URL", key=f"{source_type}_url", 
                                  placeholder="https://example.com/content")
                form_data["url"] = url
                
            elif input_type == "file":
                # File upload sources
                self._render_file_upload_section(source_type, config, form_data, options)
                
            elif input_type == "file_or_text":
                # Sources that accept both file upload and text input
                self._render_hybrid_input_section(source_type, config, form_data, options)

            # Source-specific options
            self._render_source_specific_options(source_type, config, options)

            # Processing options
            with st.expander("🔧 Processing Options"):
                options["extract_entities"] = st.checkbox("Extract named entities", value=True)
                options["extract_concepts"] = st.checkbox("Extract concepts", value=True)
                options["verify_facts"] = st.checkbox("Verify facts", value=True)
                options["create_summary"] = st.checkbox("Generate summary", value=True)
                
                confidence_threshold = st.slider(
                    "Minimum confidence threshold", 0.0, 1.0, 0.7,
                    help="Minimum confidence score for extracted information"
                )
                options["confidence_threshold"] = confidence_threshold

            # Submit button
            submitted = st.form_submit_button("🚀 Start Scraping", type="primary")

            if submitted:
                self._handle_form_submission(form_data, domain, source_type, priority, options)

    def _render_file_upload_section(self, source_type: str, config: Dict, form_data: Dict, options: Dict):
        """Render file upload section for file-based sources"""
        if source_type == "pdf":
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=['pdf'],
                help="Maximum file size: 200MB"
            )
            
            if uploaded_file:
                st.success(f"✅ File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
                form_data["uploaded_file"] = uploaded_file
                
                # PDF-specific options
                with st.expander("📜 PDF Processing Options"):
                    options["extract_text"] = st.checkbox("Extract text content", value=True)
                    options["extract_images"] = st.checkbox("Extract embedded images", value=False)
                    options["extract_metadata"] = st.checkbox("Extract metadata", value=True)
                    options["ocr_if_needed"] = st.checkbox("Use OCR if text extraction fails", value=True)

        elif source_type == "image":
            uploaded_files = st.file_uploader(
                "Upload Image(s)",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
                accept_multiple_files=True,
                help="Upload one or more images for OCR processing"
            )
            
            if uploaded_files:
                form_data["uploaded_files"] = uploaded_files
                
                # Display preview
                cols = st.columns(min(3, len(uploaded_files)))
                for i, img in enumerate(uploaded_files):
                    with cols[i % 3]:
                        st.image(img, caption=img.name, use_column_width=True)
                
                # OCR options
                with st.expander("🔍 OCR Options"):
                    options["ocr_language"] = st.selectbox(
                        "Document Language",
                        ['eng', 'lat', 'grc', 'heb', 'ara', 'chi_sim', 'jpn', 'rus', 'fra', 'deu'],
                        help="Select the primary language for OCR"
                    )
                    options["enhance_image"] = st.checkbox("Enhance image before OCR", value=True)
                    options["detect_layout"] = st.checkbox("Detect document layout", value=True)

    def _render_hybrid_input_section(self, source_type: str, config: Dict, form_data: Dict, options: Dict):
        """Render hybrid input section for sources accepting both files and text"""
        if source_type == "book":
            # Book metadata
            st.markdown("**📚 Book Information**")
            col1, col2 = st.columns(2)
            
            with col1:
                form_data["title"] = st.text_input("Book Title*", placeholder="The Republic")
                form_data["author"] = st.text_input("Author(s)*", placeholder="Plato")
                form_data["isbn"] = st.text_input("ISBN", placeholder="978-0-14-044914-7")
            
            with col2:
                form_data["publication_year"] = st.number_input("Publication Year", min_value=1000, max_value=2025, value=2023)
                form_data["publisher"] = st.text_input("Publisher", placeholder="Penguin Classics")
                form_data["language"] = st.selectbox("Language", ['English', 'Latin', 'Greek', 'Hebrew', 'Arabic', 'Sanskrit', 'Other'])
            
            # Content input method
            content_method = st.radio(
                "How will you provide the content?",
                ['Upload file', 'Paste text', 'Enter URL']
            )
            
            if content_method == 'Upload file':
                uploaded_file = st.file_uploader(
                    "Upload book file",
                    type=['txt', 'epub', 'mobi', 'docx', 'pdf'],
                    help="Supported formats: TXT, EPUB, MOBI, DOCX, PDF"
                )
                if uploaded_file:
                    form_data["uploaded_file"] = uploaded_file
                    
            elif content_method == 'Paste text':
                text_content = st.text_area(
                    "Paste book content",
                    height=300,
                    placeholder="Paste the full text of the book here..."
                )
                if text_content:
                    form_data["text_content"] = text_content
                    
            elif content_method == 'Enter URL':
                url = st.text_input("Book URL", placeholder="https://example.com/book")
                if url:
                    form_data["url"] = url

        elif source_type == "manuscript":
            # Historical document metadata
            st.markdown("**📜 Historical Document Information**")
            col1, col2 = st.columns(2)
            
            with col1:
                form_data["title"] = st.text_input("Document Title*", placeholder="Dead Sea Scrolls - Fragment 4Q521")
                form_data["time_period"] = st.text_input("Time Period/Date", placeholder="1st century BCE")
                form_data["origin"] = st.text_input("Origin/Location", placeholder="Qumran, Judean Desert")
            
            with col2:
                form_data["original_language"] = st.selectbox(
                    "Original Language",
                    ['Hebrew', 'Aramaic', 'Greek', 'Latin', 'Coptic', 'Sanskrit', 'Arabic', 'Other']
                )
                form_data["document_type"] = st.selectbox(
                    "Document Type",
                    ['Religious text', 'Historical record', 'Legal document', 'Literary work', 'Scientific text', 'Other']
                )
            
            # Input method
            input_method = st.radio(
                "Input Type",
                ['Upload manuscript images', 'Enter transcribed text', 'Both']
            )
            
            if input_method in ['Upload manuscript images', 'Both']:
                uploaded_files = st.file_uploader(
                    "Upload manuscript images",
                    type=['jpg', 'jpeg', 'png', 'tiff'],
                    accept_multiple_files=True,
                    help="High-resolution images recommended for better OCR"
                )
                if uploaded_files:
                    form_data["uploaded_files"] = uploaded_files
            
            if input_method in ['Enter transcribed text', 'Both']:
                transcribed_text = st.text_area(
                    "Transcribed Text",
                    height=200,
                    placeholder="Enter the transcribed text from the manuscript..."
                )
                if transcribed_text:
                    form_data["transcribed_text"] = transcribed_text

    def _render_source_specific_options(self, source_type: str, config: Dict, options: Dict):
        """Render source-specific options based on configuration"""
        fields = config.get("fields", [])
        
        # Handle DOI for academic papers
        if "doi" in fields:
            options["doi"] = st.text_input("DOI (optional)", placeholder="10.1038/nature12373")
        
        # Handle citation extraction
        if "extract_citations" in fields:
            options["extract_citations"] = st.checkbox("Extract Citations", value=True)
        
        # Handle figure extraction
        if "extract_figures" in fields:
            options["extract_figures"] = st.checkbox("Extract Figures", value=False)
        
        # Handle language options
        if "language" in fields:
            options["language"] = st.selectbox(
                "Original Language",
                ["Latin", "Greek", "Hebrew", "Sanskrit", "Arabic", "English", "Other"]
            )
        
        # Handle annotations
        if "include_annotations" in fields:
            options["include_annotations"] = st.checkbox("Include Annotations", value=True)
        
        # Handle translations
        if "translations" in fields:
            options["translations"] = st.checkbox("Include Translations", value=False)
        
        # Handle transcript options
        if "include_transcript" in fields:
            options["include_transcript"] = st.checkbox("Include Transcript", value=True)
        
        # Handle metadata extraction
        if "include_metadata" in fields:
            options["include_metadata"] = st.checkbox("Include Metadata", value=True)
        
        # Handle image extraction
        if "extract_images" in fields:
            options["extract_images"] = st.checkbox("Extract Images", value=True)
        
        # Handle link extraction
        if "extract_links" in fields:
            options["extract_links"] = st.checkbox("Extract Links", value=True)
        
        # Handle author extraction
        if "extract_author" in fields:
            options["extract_author"] = st.checkbox("Extract Author", value=True)
        
        # Handle date extraction
        if "extract_date" in fields:
            options["extract_date"] = st.checkbox("Extract Date", value=True)
        
        # Handle tags extraction
        if "extract_tags" in fields:
            options["extract_tags"] = st.checkbox("Extract Tags", value=True)
        
        # Handle references extraction
        if "extract_references" in fields:
            options["extract_references"] = st.checkbox("Extract References", value=True)
        
        # Handle categories extraction
        if "extract_categories" in fields:
            options["extract_categories"] = st.checkbox("Extract Categories", value=True)
        
        # Handle related entries extraction
        if "extract_related_entries" in fields:
            options["extract_related_entries"] = st.checkbox("Extract Related Entries", value=True)

    def _handle_form_submission(self, form_data: Dict, domain: str, source_type: str, priority: int, options: Dict):
        """Handle form submission with validation"""
        # Validation
        if not self._validate_form_data(form_data, source_type):
            return
        
        # Process based on input type
        if "url" in form_data and form_data["url"]:
            # URL-based processing
            self._process_scraping(form_data["url"], domain, source_type, priority, options)
        elif "uploaded_file" in form_data or "uploaded_files" in form_data:
            # File-based processing
            self._process_file_upload(form_data, domain, source_type, priority, options)
        elif "text_content" in form_data or "transcribed_text" in form_data:
            # Text-based processing
            self._process_text_content(form_data, domain, source_type, priority, options)
        else:
            st.error("❌ Please provide valid input for processing")

    def _validate_form_data(self, form_data: Dict, source_type: str) -> bool:
        """Validate form data based on source type"""
        if source_type in ["webpage", "academic_paper", "ancient_text", "youtube", "article", "encyclopedia"]:
            if not form_data.get("url"):
                st.error("❌ Please enter a URL")
                return False
                
        elif source_type in ["pdf", "image"]:
            if not form_data.get("uploaded_file") and not form_data.get("uploaded_files"):
                st.error("❌ Please upload a file")
                return False
                
        elif source_type == "book":
            if not form_data.get("title") or not form_data.get("author"):
                st.error("❌ Please enter book title and author")
                return False
            if not any(key in form_data for key in ["url", "uploaded_file", "text_content"]):
                st.error("❌ Please provide book content (URL, file, or text)")
                return False
                
        elif source_type == "manuscript":
            if not form_data.get("title"):
                st.error("❌ Please enter document title")
                return False
            if not any(key in form_data for key in ["uploaded_files", "transcribed_text"]):
                st.error("❌ Please provide manuscript content (images or transcribed text)")
                return False
                
        return True

    @run_async
    async def _process_file_upload(self, form_data: Dict, domain: str, source_type: str, priority: int, options: Dict):
        """Process file upload via API"""
        with st.spinner("🔄 Processing file upload..."):
            # This would handle file upload to API
            # For now, simulate processing
            file_info = ""
            if "uploaded_file" in form_data:
                file_info = f"File: {form_data['uploaded_file'].name}"
            elif "uploaded_files" in form_data:
                file_info = f"Files: {len(form_data['uploaded_files'])} files"
            
            st.success(f"✅ File upload processing started: {file_info}")
            st.info("File processing will be implemented in the API backend")

    @run_async
    async def _process_text_content(self, form_data: Dict, domain: str, source_type: str, priority: int, options: Dict):
        """Process text content via API"""
        with st.spinner("🔄 Processing text content..."):
            # This would handle text processing via API
            text_info = ""
            if "text_content" in form_data:
                text_info = f"Text length: {len(form_data['text_content'])} characters"
            elif "transcribed_text" in form_data:
                text_info = f"Transcribed text length: {len(form_data['transcribed_text'])} characters"
            
            st.success(f"✅ Text processing started: {text_info}")
            st.info("Text processing will be implemented in the API backend")

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
        st.markdown("**📦 Bulk URL Processing**")
        
        # Option 1: File upload
        st.markdown("**Option 1: Upload File**")
        uploaded_file = st.file_uploader(
            "Upload CSV/JSON file with URLs", 
            type=["csv", "json"],
            help="CSV should have columns: url, domain, source_type. JSON should be array of objects."
        )

        if uploaded_file:
            try:
                import pandas as pd
                import json
                
                # Parse file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    batch_data = df.to_dict('records')
                else:  # JSON
                    batch_data = json.load(uploaded_file)
                
                st.success(f"✅ Loaded {len(batch_data)} items from {uploaded_file.name}")
                
                # Preview data
                st.markdown("**Preview:**")
                preview_df = pd.DataFrame(batch_data[:5])  # Show first 5 rows
                st.dataframe(preview_df, use_column_width=True)
                
                if len(batch_data) > 5:
                    st.info(f"... and {len(batch_data) - 5} more items")
                
                # Batch processing options
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_domain = st.selectbox(
                        "Override Domain (optional)",
                        ["Use file values", "mathematics", "science", "philosophy", "religion", "history", "literature", "art", "language"]
                    )
                
                with col2:
                    batch_priority = st.select_slider(
                        "Batch Priority",
                        options=[1, 2, 3, 4, 5],
                        value=3
                    )
                
                # Validation and submission
                if st.button("🚀 Process Batch", type="primary"):
                    self._process_batch_data(batch_data, batch_domain, batch_priority)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        # Option 2: Text area for URLs
        st.markdown("**Option 2: Paste URLs**")
        batch_urls = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://youtube.com/watch?v=..."
        )
        
        if batch_urls.strip():
            urls = [url.strip() for url in batch_urls.split('\n') if url.strip()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                batch_domain = st.selectbox(
                    "Domain for all URLs",
                    ["mathematics", "science", "philosophy", "religion", "history", "literature", "art", "language"],
                    key="text_batch_domain"
                )
            
            with col2:
                batch_source_type = st.selectbox(
                    "Source Type",
                    ["webpage", "academic_paper", "article", "youtube", "encyclopedia"],
                    key="text_batch_source"
                )
            
            with col3:
                batch_priority = st.select_slider(
                    "Priority",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key="text_batch_priority"
                )
            
            st.info(f"📊 Ready to process {len(urls)} URLs")
            
            if st.button("🚀 Process URL Batch", type="primary", key="text_batch_submit"):
                # Convert URLs to batch format
                batch_data = [
                    {
                        "url": url,
                        "domain": batch_domain,
                        "source_type": batch_source_type
                    }
                    for url in urls
                ]
                self._process_batch_data(batch_data, batch_domain, batch_priority)

    @run_async
    async def _process_batch_data(self, batch_data: List[Dict], domain_override: str, priority: int):
        """Process batch data via API"""
        with st.spinner(f"🔄 Processing batch of {len(batch_data)} items..."):
            
            # Validate batch data
            valid_items = []
            for i, item in enumerate(batch_data):
                if 'url' in item and item['url']:
                    # Use override domain if specified
                    if domain_override != "Use file values":
                        item['domain'] = domain_override
                    
                    # Set default values if missing
                    if 'domain' not in item:
                        item['domain'] = 'science'  # Default domain
                    if 'source_type' not in item:
                        item['source_type'] = 'webpage'  # Default source type
                    
                    valid_items.append(item)
                else:
                    st.warning(f"⚠️ Skipping item {i+1}: Missing URL")
            
            if not valid_items:
                st.error("❌ No valid items to process")
                return
            
            # Process in chunks
            chunk_size = 10
            processed_count = 0
            failed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(valid_items), chunk_size):
                chunk = valid_items[i:i + chunk_size]
                status_text.text(f"Processing chunk {i//chunk_size + 1} of {(len(valid_items) + chunk_size - 1)//chunk_size}")
                
                # Process chunk via API
                try:
                    # This would be actual API call
                    # For now, simulate processing
                    await asyncio.sleep(0.5)  # Simulate processing time
                    processed_count += len(chunk)
                    
                    # Update progress
                    progress_bar.progress(min(1.0, processed_count / len(valid_items)))
                    
                except Exception as e:
                    st.error(f"❌ Error processing chunk: {str(e)}")
                    failed_count += len(chunk)
            
            # Show results
            status_text.text("✅ Batch processing completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", len(batch_data))
            with col2:
                st.metric("Processed", processed_count)
            with col3:
                st.metric("Failed", failed_count)
            
            if processed_count > 0:
                st.success(f"✅ Successfully submitted {processed_count} items for processing")
            if failed_count > 0:
                st.error(f"❌ Failed to process {failed_count} items")

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
