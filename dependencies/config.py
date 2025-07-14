"""Dependency configuration management."""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DependencyConfig:
    """Configuration for dependency management."""
    
    # Core server and API
    CORE_DEPS = {
        'fastapi': '>=0.104.0,<0.105.0',
        'uvicorn[standard]': '>=0.24.0,<0.25.0',
        'pydantic': '>=2.5.0,<3.0.0',
        'psutil': '>=5.9.0,<6.0.0',  # Fix for operations console
    }
    
    # Database connections
    DATABASE_DEPS = {
        'neo4j': '>=5.15.0,<6.0.0',
        'qdrant-client': '>=1.7.0,<2.0.0',
        'redis[hiredis]': '>=5.0.0,<6.0.0',
    }
    
    # NLP and ML
    ML_DEPS = {
        'spacy': '>=3.7.0,<4.0.0',
        'sentence-transformers': '>=2.2.0,<3.0.0',
        'scikit-learn': '>=1.3.0,<2.0.0',
    }
    
    # Web scraping
    SCRAPING_DEPS = {
        'beautifulsoup4': '>=4.12.0,<5.0.0',
        'scrapy': '>=2.11.0,<3.0.0',
        'selenium': '>=4.16.0,<5.0.0',
        'trafilatura': '>=1.6.0,<2.0.0',  # New for better extraction
        'selenium-stealth': '>=1.0.6',     # Anti-detection
    }
    
    # YouTube processing
    YOUTUBE_DEPS = {
        'yt-dlp': '>=2023.12.0',
        'youtube-transcript-api': '>=0.6.0,<1.0.0',
    }
    
    # UI
    UI_DEPS = {
        'streamlit': '>=1.28.0,<2.0.0',
        'plotly': '>=5.18.0,<6.0.0',
    }
    
    # Development only
    DEV_DEPS = {
        'pytest': '>=7.4.0,<8.0.0',
        'pytest-cov': '>=4.1.0,<5.0.0',
        'black': '>=23.12.0,<24.0.0',
        'flake8': '>=6.1.0,<7.0.0',
        'mypy': '>=1.8.0,<2.0.0',
        'pip-tools': '>=7.3.0,<8.0.0',
    }
