"""
Text search utilities extracted from existing components
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def perform_text_search(query: str, domains: List[str] = None, 
                       case_sensitive: bool = False, 
                       whole_words: bool = False) -> List[Dict[str, Any]]:
    """
    Perform text search across the knowledge base.
    
    Args:
        query: Search query string
        domains: List of domains to search in
        case_sensitive: Whether search should be case sensitive
        whole_words: Whether to match whole words only
    
    Returns:
        List of search results
    """
    try:
        # Placeholder implementation - would integrate with database agents
        results = [
            {
                'id': f'concept_{i}',
                'title': f'Sample Concept {i}',
                'domain': domains[0] if domains else 'Philosophy',
                'type': 'Concept',
                'description': f'Sample description containing "{query}"',
                'relevance_score': 0.8 - (i * 0.1),
                'created_date': '2024-01-01'
            }
            for i in range(min(5, 10))
        ]
        
        return results
        
    except Exception as e:
        logger.error(f"Error performing text search: {e}")
        return []


def search_concepts_by_domain(domain: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search concepts within a specific domain.
    
    Args:
        domain: Domain to search in
        limit: Maximum number of results
    
    Returns:
        List of concepts in the domain
    """
    try:
        # Placeholder implementation
        return [
            {
                'id': f'concept_{domain}_{i}',
                'title': f'{domain} Concept {i}',
                'domain': domain,
                'type': 'Concept',
                'description': f'A concept related to {domain}',
                'created_date': '2024-01-01'
            }
            for i in range(min(limit, 20))
        ]
        
    except Exception as e:
        logger.error(f"Error searching concepts by domain: {e}")
        return []