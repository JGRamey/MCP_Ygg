"""
Data processing utilities

Provides file processing, content extraction, and validation utilities
extracted from existing components and pages.
"""

import streamlit as st
import pandas as pd
import json
import yaml
import tempfile
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def process_uploaded_file(uploaded_file, domain: str = None, 
                         extract_concepts: bool = True) -> Dict[str, Any]:
    """
    Process an uploaded file and extract relevant information.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        domain: Target domain for the content
        extract_concepts: Whether to extract concepts automatically
    
    Returns:
        Dictionary containing processed file information
    """
    try:
        if uploaded_file is None:
            return {'success': False, 'error': 'No file provided'}
        
        # Basic file information
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'upload_time': datetime.now().isoformat()
        }
        
        # Determine file type and processing method
        file_ext = Path(uploaded_file.name).suffix.lower()
        content = None
        metadata = {}
        
        if file_ext in ['.txt', '.md']:
            content = process_text_file(uploaded_file)
        elif file_ext == '.pdf':
            content = process_pdf_file(uploaded_file)
        elif file_ext in ['.doc', '.docx']:
            content = process_word_file(uploaded_file)
        elif file_ext == '.csv':
            content = process_csv_file(uploaded_file)
        elif file_ext == '.json':
            content = process_json_file(uploaded_file)
        elif file_ext in ['.yaml', '.yml']:
            content = process_yaml_file(uploaded_file)
        else:
            # Try to read as text
            try:
                content = uploaded_file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'file_info': file_info
                }
        
        # Extract metadata
        metadata = extract_metadata(content, uploaded_file.name, domain)
        
        # Extract concepts if requested
        concepts = []
        if extract_concepts and isinstance(content, str):
            concepts = extract_concepts_from_text(content, domain)
        
        return {
            'success': True,
            'file_info': file_info,
            'content': content,
            'metadata': metadata,
            'concepts': concepts,
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return {
            'success': False,
            'error': str(e),
            'file_info': file_info if 'file_info' in locals() else {}
        }


def process_text_file(uploaded_file) -> str:
    """Process a text file and return its content."""
    try:
        return uploaded_file.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        # Try with different encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                uploaded_file.seek(0)
                return uploaded_file.getvalue().decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("Unable to decode file with any supported encoding")


def process_pdf_file(uploaded_file) -> str:
    """Process a PDF file and extract text content."""
    try:
        # This would require PyPDF2 or similar library
        # For now, return placeholder
        return f"PDF content extraction not implemented. File: {uploaded_file.name}"
    except Exception as e:
        logger.error(f"Error processing PDF file: {e}")
        return f"Error extracting PDF content: {e}"


def process_word_file(uploaded_file) -> str:
    """Process a Word document and extract text content."""
    try:
        # This would require python-docx library
        # For now, return placeholder
        return f"Word document content extraction not implemented. File: {uploaded_file.name}"
    except Exception as e:
        logger.error(f"Error processing Word file: {e}")
        return f"Error extracting Word content: {e}"


def process_csv_file(uploaded_file) -> pd.DataFrame:
    """Process a CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise


def process_json_file(uploaded_file) -> Dict[str, Any]:
    """Process a JSON file and return parsed data."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error processing JSON file: {e}")
        raise


def process_yaml_file(uploaded_file) -> Dict[str, Any]:
    """Process a YAML file and return parsed data."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        return yaml.safe_load(content)
    except Exception as e:
        logger.error(f"Error processing YAML file: {e}")
        raise


def process_web_content(url: str, content_type: str = "article", 
                       domain: str = None) -> Dict[str, Any]:
    """
    Process web content from a URL.
    
    Args:
        url: URL to process
        content_type: Type of content ('article', 'video', 'document')
        domain: Target domain for the content
    
    Returns:
        Dictionary containing processed web content
    """
    try:
        # This would integrate with scraping agents
        # For now, return placeholder structure
        
        result = {
            'success': False,
            'url': url,
            'content_type': content_type,
            'domain': domain,
            'processed_time': datetime.now().isoformat(),
            'error': 'Web content processing not implemented'
        }
        
        # Placeholder for actual implementation
        if url.startswith(('http://', 'https://')):
            result.update({
                'title': 'Extracted Title',
                'content': 'Extracted content would go here',
                'metadata': {
                    'author': 'Unknown',
                    'publish_date': None,
                    'word_count': 0,
                    'language': 'en'
                },
                'concepts': [],
                'success': True,
                'error': None
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing web content: {e}")
        return {
            'success': False,
            'url': url,
            'error': str(e),
            'processed_time': datetime.now().isoformat()
        }


def extract_metadata(content: Any, filename: str, domain: str = None) -> Dict[str, Any]:
    """
    Extract metadata from content.
    
    Args:
        content: Content to analyze
        filename: Original filename
        domain: Target domain
    
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        'filename': filename,
        'domain': domain,
        'extracted_time': datetime.now().isoformat(),
        'content_type': determine_content_type(content),
        'language': 'unknown',
        'encoding': 'utf-8'
    }
    
    if isinstance(content, str):
        metadata.update({
            'character_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.splitlines()),
            'language': detect_language(content)
        })
    elif isinstance(content, pd.DataFrame):
        metadata.update({
            'row_count': len(content),
            'column_count': len(content.columns),
            'columns': list(content.columns)
        })
    elif isinstance(content, (dict, list)):
        metadata.update({
            'structure_type': type(content).__name__,
            'size': len(content) if hasattr(content, '__len__') else 0
        })
    
    return metadata


def determine_content_type(content: Any) -> str:
    """Determine the type of content."""
    if isinstance(content, str):
        return 'text'
    elif isinstance(content, pd.DataFrame):
        return 'tabular'
    elif isinstance(content, dict):
        return 'structured'
    elif isinstance(content, list):
        return 'list'
    else:
        return 'unknown'


def detect_language(text: str) -> str:
    """
    Detect the language of text content.
    
    Args:
        text: Text to analyze
    
    Returns:
        Detected language code
    """
    # This would require a language detection library like langdetect
    # For now, return default
    return 'en'


def extract_concepts_from_text(text: str, domain: str = None) -> List[Dict[str, Any]]:
    """
    Extract concepts from text content using NLP.
    
    Args:
        text: Text to analyze
        domain: Target domain for context
    
    Returns:
        List of extracted concepts
    """
    try:
        # This would integrate with NLP agents
        # For now, return placeholder
        
        concepts = []
        
        # Simple keyword extraction placeholder
        words = text.split()
        unique_words = set(word.strip('.,!?;:').lower() for word in words if len(word) > 3)
        
        for i, word in enumerate(list(unique_words)[:10]):  # Limit to 10 concepts
            concepts.append({
                'title': word.title(),
                'type': 'keyword',
                'domain': domain or 'Unknown',
                'confidence': 0.5,
                'context': f"Extracted from position {i} in text",
                'source': 'text_analysis'
            })
        
        return concepts
        
    except Exception as e:
        logger.error(f"Error extracting concepts: {e}")
        return []


def validate_content(content: Any, content_type: str = None, 
                    rules: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate content against specified rules.
    
    Args:
        content: Content to validate
        content_type: Type of content for validation
        rules: Validation rules dictionary
    
    Returns:
        Validation result dictionary
    """
    if rules is None:
        rules = {}
    
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': [],
        'validated_time': datetime.now().isoformat()
    }
    
    try:
        # Basic content validation
        if content is None:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Content is None")
            return validation_result
        
        # Type-specific validation
        if content_type == 'text':
            if isinstance(content, str):
                if len(content.strip()) == 0:
                    validation_result['warnings'].append("Content is empty")
                elif len(content) < rules.get('min_length', 0):
                    validation_result['warnings'].append(
                        f"Content length {len(content)} below minimum {rules.get('min_length')}"
                    )
                elif len(content) > rules.get('max_length', float('inf')):
                    validation_result['warnings'].append(
                        f"Content length {len(content)} above maximum {rules.get('max_length')}"
                    )
            else:
                validation_result['errors'].append("Expected text content, got " + type(content).__name__)
        
        elif content_type == 'tabular':
            if isinstance(content, pd.DataFrame):
                if content.empty:
                    validation_result['warnings'].append("DataFrame is empty")
                elif len(content) < rules.get('min_rows', 0):
                    validation_result['warnings'].append(
                        f"Row count {len(content)} below minimum {rules.get('min_rows')}"
                    )
            else:
                validation_result['errors'].append("Expected DataFrame, got " + type(content).__name__)
        
        # Check for required fields if specified
        required_fields = rules.get('required_fields', [])
        if isinstance(content, dict) and required_fields:
            missing_fields = [field for field in required_fields if field not in content]
            if missing_fields:
                validation_result['errors'].extend([f"Missing required field: {field}" for field in missing_fields])
        
        # Update overall validity
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating content: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'info': [],
            'validated_time': datetime.now().isoformat()
        }


def batch_process_files(files: List, domain: str = None, 
                       extract_concepts: bool = True) -> List[Dict[str, Any]]:
    """
    Process multiple files in batch.
    
    Args:
        files: List of uploaded file objects
        domain: Target domain for the content
        extract_concepts: Whether to extract concepts automatically
    
    Returns:
        List of processing results
    """
    results = []
    
    for i, file in enumerate(files):
        try:
            # Show progress
            if len(files) > 1:
                progress = (i + 1) / len(files)
                st.progress(progress, text=f"Processing file {i+1} of {len(files)}: {file.name}")
            
            result = process_uploaded_file(file, domain, extract_concepts)
            result['batch_index'] = i
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'file_info': {'name': file.name},
                'batch_index': i
            })
    
    return results


def create_content_summary(content: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a summary of content for display.
    
    Args:
        content: Content to summarize
        metadata: Optional metadata dictionary
    
    Returns:
        Summary dictionary
    """
    summary = {
        'type': determine_content_type(content),
        'created_time': datetime.now().isoformat()
    }
    
    if isinstance(content, str):
        summary.update({
            'preview': content[:200] + '...' if len(content) > 200 else content,
            'character_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.splitlines())
        })
    elif isinstance(content, pd.DataFrame):
        summary.update({
            'shape': content.shape,
            'columns': list(content.columns),
            'head': content.head(3).to_dict('records') if not content.empty else []
        })
    elif isinstance(content, dict):
        summary.update({
            'keys': list(content.keys())[:10],  # First 10 keys
            'key_count': len(content),
            'preview': {k: v for i, (k, v) in enumerate(content.items()) if i < 3}
        })
    elif isinstance(content, list):
        summary.update({
            'length': len(content),
            'preview': content[:3] if content else [],
            'item_types': list(set(type(item).__name__ for item in content[:10]))
        })
    
    if metadata:
        summary['metadata'] = metadata
    
    return summary