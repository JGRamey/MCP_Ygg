"""Claim Analyzer Agent Package"""

from .claim_analyzer import ClaimAnalyzerAgent
from .models import Claim, Evidence, FactCheckResult
from .database import DatabaseConnector
from .extractor import ClaimExtractor
from .checker import FactChecker

__version__ = "1.0.0"
__author__ = "MCP Server Team"

__all__ = [
    'ClaimAnalyzerAgent',
    'Claim',
    'Evidence', 
    'FactCheckResult',
    'DatabaseConnector',
    'ClaimExtractor',
    'FactChecker'
]