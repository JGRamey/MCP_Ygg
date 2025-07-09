#!/usr/bin/env python3
"""Custom exceptions for Claim Analyzer Agent"""


class ClaimAnalyzerError(Exception):
    """Base exception for Claim Analyzer Agent"""
    pass


class DatabaseConnectionError(ClaimAnalyzerError):
    """Database connection related errors"""
    pass


class ConfigurationError(ClaimAnalyzerError):
    """Configuration loading or validation errors"""
    pass


class ClaimExtractionError(ClaimAnalyzerError):
    """Errors during claim extraction"""
    pass


class FactCheckingError(ClaimAnalyzerError):
    """Errors during fact-checking process"""
    pass


class ModelLoadError(ClaimAnalyzerError):
    """Errors loading NLP models"""
    pass


class ValidationError(ClaimAnalyzerError):
    """Input validation errors"""
    pass


class RateLimitError(ClaimAnalyzerError):
    """Rate limiting errors"""
    pass


class EvidenceSearchError(ClaimAnalyzerError):
    """Errors during evidence search"""
    pass