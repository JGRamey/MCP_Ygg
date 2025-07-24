"""
Data models for File Manager
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


@dataclass
class DatabaseFilters:
    """Database filtering configuration."""
    domains: List[str]
    content_types: List[str]


@dataclass 
class CSVFileInfo:
    """Information about a CSV file."""
    path: Path
    name: str
    domain: str
    file_type: str  # concepts, relationships, people, etc.
    size_bytes: int
    row_count: int
    column_count: int
    last_modified: datetime
    
    @property
    def size_kb(self) -> float:
        """File size in KB."""
        return self.size_bytes / 1024
    
    @property
    def file_type_display(self) -> str:
        """Display-friendly file type."""
        type_map = {
            'concepts': '📄 Concepts',
            'relationships': '🔗 Relationships', 
            'people': '👤 People',
            'works': '📚 Works',
            'places': '📍 Places'
        }
        return type_map.get(self.file_type, f'📋 {self.file_type.title()}')


@dataclass
class Neo4jNodeInfo:
    """Information about a Neo4j node."""
    node_id: int
    labels: List[str]
    properties: Dict[str, Any]
    
    @property
    def primary_label(self) -> str:
        """Primary node label."""
        return self.labels[0] if self.labels else 'Unknown'
    
    @property
    def display_name(self) -> str:
        """Display name for the node."""
        return (
            self.properties.get('name') or 
            self.properties.get('title') or 
            f"{self.primary_label} {self.node_id}"
        )


@dataclass 
class QdrantCollectionInfo:
    """Information about a Qdrant collection."""
    name: str
    vectors_count: int
    vector_size: int
    distance_metric: str
    status: str
    indexed: bool


@dataclass
class ScrapedContentInfo:
    """Information about scraped content."""
    content_id: str
    title: str
    url: str
    domain: str
    content_type: str
    scraped_at: datetime
    status: str  # pending, approved, rejected
    content_length: int
    
    @property
    def status_display(self) -> str:
        """Display-friendly status."""
        status_map = {
            'pending': '⏳ Pending',
            'approved': '✅ Approved',
            'rejected': '❌ Rejected',
            'processing': '🔄 Processing'
        }
        return status_map.get(self.status, f'❓ {self.status.title()}')