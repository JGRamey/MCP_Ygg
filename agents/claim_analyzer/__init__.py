"""Claim Analyzer Agent Package - Enhanced Phase 2 Version"""

from .claim_analyzer import ClaimAnalyzerAgent
from .models import Claim, Evidence, FactCheckResult
from .database import DatabaseConnector
from .extractor import ClaimExtractor
from .checker import FactChecker

__version__ = "2.0.0"
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