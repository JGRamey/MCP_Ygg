"""Claim Analyzer Agent Package - Enhanced Phase 2 Version"""

from .checker import FactChecker
from .claim_analyzer import ClaimAnalyzerAgent
from .database import DatabaseConnector
from .extractor import ClaimExtractor
from .models import Claim, Evidence, FactCheckResult

__version__ = "2.0.0"
__author__ = "MCP Server Team"

__all__ = [
    "ClaimAnalyzerAgent",
    "Claim",
    "Evidence",
    "FactCheckResult",
    "DatabaseConnector",
    "ClaimExtractor",
    "FactChecker",
]
