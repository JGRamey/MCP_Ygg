"""
Content Scraper - Modular Multi-source Content Submission Interface

Refactored from monolithic 1,508-line file into focused modules following
the established modular architecture patterns.

Modules:
- main.py: Main interface and orchestration (300 lines)
- scraping_engine.py: Core scraping logic and URL processing (400 lines)
- content_processors.py: Content processing and analysis (400 lines)
- submission_manager.py: Submission handling and queue management (400 lines)

Features:
- Multi-source content acquisition (web, YouTube, file upload, manual text)
- Content processing pipeline with staging and approval workflow
- Advanced scraping with anti-blocking measures
- Intelligent content analysis and concept extraction
- Real-time submission queue management and monitoring
"""

from .main import ContentScraperInterface
from .scraping_engine import ScrapingEngine, URLProcessor
from .content_processors import ContentProcessor, ConceptExtractor
from .submission_manager import SubmissionManager, QueueManager

__all__ = [
    'ContentScraperInterface',
    'ScrapingEngine', 'URLProcessor',
    'ContentProcessor', 'ConceptExtractor',
    'SubmissionManager', 'QueueManager'
]

__version__ = "2.0.0"
__author__ = "MCP Yggdrasil Team"
__description__ = "Modular content scraping and submission system"