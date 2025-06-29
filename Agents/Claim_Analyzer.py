"""
AI Claim Analyzer MCP Server
Model Context Protocol server providing claim extraction, fact-checking, and analysis tools
"""

import asyncio
import logging
import json
import sqlite3
import re
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import yaml
import traceback

import aiohttp
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# MCP protocol imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    PromptMessage, GetPromptResult
)
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Claim:
    """Represents a claim to be fact-checked"""
    text: str
    source: str
    timestamp: datetime
    claim_id: str
    confidence: float = 0.0
    context: str = ""
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.md5(self.text.encode()).hexdigest()

@dataclass
class FactCheckResult:
    """Represents the result of a fact-check"""
    claim: Claim
    verdict: str  # True, False, Partially True, Unverified, Opinion
    confidence: float
    evidence: List[Dict[str, Any]]
    reasoning: str
    sources: List[str]
    timestamp: datetime

@dataclass
class Source:
    """Represents a credible source"""
    url: str
    title: str
    credibility_score: float
    domain: str
    content: str = ""

class ClaimExtractor:
    """Extracts claims from various text sources using NLP"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            # Use a simple fallback extractor
            self.nlp = None
        
        # Patterns that typically indicate claims
        self.claim_patterns = [
            r"(?:is|are|was|were|will be|has been|have been)\s+(?:a|an|the)?\s*\w+",
            r"(?:claims?|states?|argues?|believes?|says?)\s+that\s+.*",
            r"(?:according to|research shows|studies indicate|experts say)\s+.*",
            r"(?:it is|this is|that is)\s+(?:true|false|correct|incorrect|a fact)\s+that\s+.*"
        ]
    
    def extract_claims(self, text: str, source: str = "unknown") -> List[Claim]:
        """Extract verifiable claims from text"""
        claims = []
        
        if self.nlp:
            # Use spaCy for better claim extraction
            claims = self._extract_with_spacy(text, source)
        else:
            # Fallback to pattern-based extraction
            claims = self._extract_with_patterns(text, source)
        
        return claims
    
    def _extract_with_spacy(self, text: str, source: str) -> List[Claim]:
        """Extract claims using spaCy NLP"""
        claims = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Skip questions, commands, and very short sentences
            if len(sent_text) < 10 or sent_text.endswith('?') or sent_text.startswith('/'):
                continue
            
            # Check if sentence contains claim indicators
            if self._is_claim_sentence(sent_text):
                confidence = self._calculate_claim_confidence(sent_text, sent)
                
                if confidence > 0.3:  # Threshold for claim detection
                    claim = Claim(
                        text=sent_text,
                        source=source,
                        timestamp=datetime.now(),
                        claim_id="",
                        confidence=confidence,
                        context=self._get_context(sent, doc)
                    )
                    claims.append(claim)
        
        return claims
    
    def _extract_with_patterns(self, text: str, source: str) -> List[Claim]:
        """Fallback pattern-based claim extraction"""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            if self._is_claim_sentence(sentence):
                claim = Claim(
                    text=sentence,
                    source=source,
                    timestamp=datetime.now(),
                    claim_id="",
                    confidence=0.5,  # Default confidence for pattern-based
                    context=sentence
                )
                claims.append(claim)
        
        return claims
    
    def _is_claim_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a verifiable claim"""
        sentence_lower = sentence.lower()
        
        # Check for claim patterns
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # Check for factual statements
        factual_indicators = [
            'fact', 'truth', 'evidence', 'study', 'research', 'data',
            'statistics', 'proven', 'confirmed', 'verified', 'documented'
        ]
        
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
        
        # Check for absolute statements
        absolute_terms = ['always', 'never', 'all', 'none', 'every', 'no one', 'everyone']
        if any(term in sentence_lower for term in absolute_terms):
            return True
        
        return False
    
    def _calculate_claim_confidence(self, sentence: str, spacy_sent) -> float:
        """Calculate confidence that this sentence contains a verifiable claim"""
        confidence = 0.0
        
        # Check for verbs indicating factual statements
        factual_verbs = {'be', 'have', 'do', 'say', 'state', 'claim', 'prove', 'show'}
        for token in spacy_sent:
            if token.lemma_ in factual_verbs and token.pos_ == 'VERB':
                confidence += 0.3
        
        # Check for named entities (more likely to be factual)
        if len([ent for ent in spacy_sent.ents]) > 0:
            confidence += 0.2
        
        # Check for numbers/dates (often factual)
        if any(token.like_num or token.ent_type_ in ['DATE', 'TIME', 'PERCENT', 'MONEY'] 
               for token in spacy_sent):
            confidence += 0.2
        
        # Check sentence structure
        if len(sentence.split()) > 5:  # Substantial sentences more likely to contain claims
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_context(self, sentence, doc) -> str:
        """Get surrounding context for a claim"""
        sent_start = sentence.start
        sent_end = sentence.end
        
        # Get previous and next sentence if available
        context_tokens = []
        
        # Previous sentence
        prev_sent_tokens = [token for token in doc if token.i < sent_start]
        if prev_sent_tokens:
            context_tokens.extend(prev_sent_tokens[-20:])  # Last 20 tokens
        
        # Current sentence
        context_tokens.extend([token for token in sentence])
        
        # Next sentence
        next_sent_tokens = [token for token in doc if token.i >= sent_end]
        if next_sent_tokens:
            context_tokens.extend(next_sent_tokens[:20])  # First 20 tokens
        
        return ' '.join([token.text for token in context_tokens])

class FactChecker:
    """Fact-checks claims using multiple sources and reasoning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentence_model = None
        
        # Initialize semantic similarity model
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
        
        # Initialize trusted sources
        self.trusted_domains = config.get('trusted_sources', [
            'wikipedia.org', 'snopes.com', 'factcheck.org', 'politifact.com',
            'reuters.com', 'bbc.com', 'npr.org', 'ap.org', 'cdc.gov', 'nasa.gov'
        ])
        
        # Source credibility scores
        self.source_credibility = {
            'wikipedia.org': 0.8,
            'snopes.com': 0.9,
            'factcheck.org': 0.9,
            'politifact.com': 0.9,
            'reuters.com': 0.9,
            'bbc.com': 0.8,
            'npr.org': 0.8,
            'ap.org': 0.9,
            'cdc.gov': 0.95,
            'nasa.gov': 0.95,
        }
    
    async def fact_check_claim(self, claim: Claim) -> FactCheckResult:
        """Perform comprehensive fact-checking on a claim"""
        logger.info(f"Fact-checking claim: {claim.text[:100]}...")
        
        try:
            # Search for similar claims and evidence
            evidence = await self._search_evidence(claim)
            
            # Analyze the evidence
            verdict, confidence, reasoning = self._analyze_evidence(claim, evidence)
            
            # Extract source URLs
            sources = [item.get('url', '') for item in evidence if item.get('url')]
            
            result = FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                reasoning=reasoning,
                sources=sources,
                timestamp=datetime.now()
            )
            
            logger.info(f"Fact-check complete: {verdict} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error fact-checking claim: {str(e)}")
            return FactCheckResult(
                claim=claim,
                verdict="Error",
                confidence=0.0,
                evidence=[],
                reasoning=f"Error during fact-checking: {str(e)}",
                sources=[],
                timestamp=datetime.now()
            )
    
    async def _search_evidence(self, claim: Claim) -> List[Dict[str, Any]]:
        """Search for evidence related to the claim"""
        evidence = []
        
        # Use multiple search strategies
        strategies = [
            self._search_web,
            self._search_fact_check_apis,
            self._search_cached_claims
        ]
        
        for strategy in strategies:
            try:
                results = await strategy(claim)
                evidence.extend(results)
            except Exception as e:
                logger.warning(f"Search strategy failed: {str(e)}")
        
        # Remove duplicates and limit results
        unique_evidence = []
        seen_urls = set()
        
        for item in evidence:
            url = item.get('url', '')
            if url not in seen_urls:
                unique_evidence.append(item)
                seen_urls.add(url)
        
        return unique_evidence[:self.config.get('max_results', 10)]
    
    async def _search_web(self, claim: Claim) -> List[Dict[str, Any]]:
        """Search the web for evidence using search APIs"""
        evidence = []
        
        # Extract search terms from claim
        search_terms = self._extract_search_terms(claim.text)
        search_query = ' '.join(search_terms[:5])  # Limit to 5 terms
        
        # Example evidence - in production, this would use real search APIs
        mock_evidence = await self._get_mock_evidence(claim)
        evidence.extend(mock_evidence)
        
        return evidence
    
    async def _search_fact_check_apis(self, claim: Claim) -> List[Dict[str, Any]]:
        """Search fact-checking APIs"""
        evidence = []
        
        # Google Fact Check Tools API integration would go here
        # For demonstration, return relevant mock data based on common claims
        
        claim_lower = claim.text.lower()
        
        if 'moon landing' in claim_lower and ('fake' in claim_lower or 'hoax' in claim_lower):
            evidence.append({
                'title': 'Moon Landing Fact Check - NASA',
                'url': 'https://www.nasa.gov/mission_pages/apollo/apollo11.html',
                'snippet': 'NASA provides extensive documentation of the Apollo moon landings with evidence including lunar rocks, photos, and independent verification.',
                'credibility': 0.95,
                'source_type': 'official',
                'verdict': 'False'
            })
        
        elif 'earth' in claim_lower and 'flat' in claim_lower:
            evidence.append({
                'title': 'Scientific Consensus on Earth\'s Shape',
                'url': 'https://example.com/earth-shape',
                'snippet': 'Scientific evidence from multiple sources confirms Earth is an oblate spheroid, including satellite imagery and physics.',
                'credibility': 0.9,
                'source_type': 'scientific',
                'verdict': 'False'
            })
        
        elif 'vaccine' in claim_lower and 'autism' in claim_lower:
            evidence.append({
                'title': 'CDC Vaccine Safety Research',
                'url': 'https://www.cdc.gov/vaccines/safety/',
                'snippet': 'Multiple large-scale studies have found no link between vaccines and autism.',
                'credibility': 0.95,
                'source_type': 'medical',
                'verdict': 'False'
            })
        
        return evidence
    
    async def _search_cached_claims(self, claim: Claim) -> List[Dict[str, Any]]:
        """Search previously cached fact-check results"""
        # This would search the local database for similar claims
        # For now, return empty list
        return []
    
    async def _get_mock_evidence(self, claim: Claim) -> List[Dict[str, Any]]:
        """Generate appropriate mock evidence based on claim content"""
        evidence = []
        claim_lower = claim.text.lower()
        
        # Generate contextually relevant mock evidence
        if any(term in claim_lower for term in ['climate', 'global warming', 'temperature']):
            evidence.append({
                'title': 'Climate Research Findings',
                'url': 'https://climate.nasa.gov/',
                'snippet': 'NASA climate data shows consistent warming trends supported by multiple measurement methods.',
                'credibility': 0.9,
                'source_type': 'scientific'
            })
        
        elif any(term in claim_lower for term in ['study', 'research', 'scientist']):
            evidence.append({
                'title': 'Peer-Reviewed Research',
                'url': 'https://example.com/research',
                'snippet': 'Scientific studies provide evidence regarding the claim with statistical analysis.',
                'credibility': 0.85,
                'source_type': 'academic'
            })
        
        else:
            evidence.append({
                'title': 'General Information Source',
                'url': 'https://example.com/info',
                'snippet': 'Information available from credible sources regarding this topic.',
                'credibility': 0.7,
                'source_type': 'general'
            })
        
        return evidence
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract key search terms from claim text"""
        # Simple keyword extraction - could be improved with NLP
        words = re.findall(r'\b[A-Za-z]{3,}\b', text)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'that', 'this', 'with', 'for'}
        keywords = [word for word in words if word.lower() not in stop_words]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _analyze_evidence(self, claim: Claim, evidence: List[Dict[str, Any]]) -> Tuple[str, float, str]:
        """Analyze collected evidence to determine verdict"""
        if not evidence:
            return "Unverified", 0.0, "No evidence found to verify this claim."
        
        verdicts = []
        reasoning_parts = []
        total_credibility = 0
        
        for item in evidence:
            snippet = item.get('snippet', '')
            credibility = item.get('credibility', 0.5)
            total_credibility += credibility
            
            # Determine stance of evidence
            if any(term in snippet.lower() for term in ['false', 'debunked', 'incorrect', 'myth']):
                verdicts.append(('False', credibility))
            elif any(term in snippet.lower() for term in ['true', 'confirmed', 'verified', 'accurate']):
                verdicts.append(('True', credibility))
            elif any(term in snippet.lower() for term in ['partially', 'mixed', 'some truth']):
                verdicts.append(('Partially True', credibility))
            else:
                verdicts.append(('Unverified', credibility * 0.5))
            
            reasoning_parts.append(f"Source: {item.get('title', 'Unknown')} - {snippet[:100]}...")
        
        # Determine overall verdict
        if not verdicts:
            return "Unverified", 0.0, "Evidence found but stance unclear."
        
        # Weight verdicts by credibility scores
        verdict_scores = {}
        for verdict, score in verdicts:
            verdict_scores[verdict] = verdict_scores.get(verdict, 0) + score
        
        # Get the verdict with highest weighted score
        final_verdict = max(verdict_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate overall confidence
        max_score = max(verdict_scores.values())
        total_scores = sum(verdict_scores.values())
        confidence = max_score / total_scores if total_scores > 0 else 0.0
        
        # Adjust confidence based on number of sources and their credibility
        avg_credibility = total_credibility / len(evidence) if evidence else 0.0
        confidence = min(confidence * avg_credibility * min(len(evidence) / 3, 1.0), 1.0)
        
        reasoning = f"Based on {len(evidence)} sources. " + " ".join(reasoning_parts[:3])
        
        return final_verdict, confidence, reasoning

class ClaimDatabase:
    """Manages local database for caching claims and results"""
    
    def __init__(self, db_path: str = "claims.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                timestamp TEXT,
                confidence REAL,
                context TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fact_check_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT,
                verdict TEXT,
                confidence REAL,
                reasoning TEXT,
                evidence TEXT,
                sources TEXT,
                timestamp TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_claim(self, claim: Claim):
        """Store a claim in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO claims 
            (id, text, source, timestamp, confidence, context)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            claim.claim_id,
            claim.text,
            claim.source,
            claim.timestamp.isoformat(),
            claim.confidence,
            claim.context
        ))
        
        conn.commit()
        conn.close()
    
    def store_result(self, result: FactCheckResult):
        """Store a fact-check result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First store the claim
        self.store_claim(result.claim)
        
        # Then store the result
        cursor.execute('''
            INSERT INTO fact_check_results 
            (claim_id, verdict, confidence, reasoning, evidence, sources, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.claim.claim_id,
            result.verdict,
            result.confidence,
            result.reasoning,
            json.dumps(result.evidence),
            json.dumps(result.sources),
            result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """Get recent fact-check results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.text, r.verdict, r.confidence, r.reasoning, r.timestamp
            FROM fact_check_results r
            JOIN claims c ON r.claim_id = c.id
            ORDER BY r.timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'claim_text': row[0],
                'verdict': row[1],
                'confidence': row[2],
                'reasoning': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return results

# MCP Server Implementation
class ClaimAnalyzerMCPServer:
    """MCP Server for Claim Analysis tools"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config = self._load_config(config_path)
        self.claim_extractor = ClaimExtractor()
        self.fact_checker = FactChecker(self.config)
        self.database = ClaimDatabase()
        
        # Initialize MCP server
        self.server = Server("claim-analyzer")
        
        # Register tools
        self._register_tools()
        self._register_resources()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'api_keys': {},
            'trusted_sources': [
                'wikipedia.org', 'snopes.com', 'factcheck.org',
                'politifact.com', 'reuters.com', 'bbc.com'
            ],
            'max_results': 10,
            'language': 'english',
            'confidence_threshold': 0.5,
            'cache_duration_hours': 24
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="analyze_claims",
                    description="Extract and analyze claims from text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze for claims"
                            },
                            "source": {
                                "type": "string",
                                "description": "Source identifier for the text",
                                "default": "unknown"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="fact_check_claim",
                    description="Fact-check a specific claim",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "The claim to fact-check"
                            },
                            "detailed": {
                                "type": "boolean",
                                "description": "Whether to return detailed analysis",
                                "default": True
                            }
                        },
                        "required": ["claim"]
                    }
                ),
                Tool(
                    name="search_similar_claims",
                    description="Search for similar claims in the database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "Claim to find similar ones for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5
                            }
                        },
                        "required": ["claim"]
                    }
                ),
                Tool(
                    name="get_recent_fact_checks",
                    description="Get recent fact-check results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 10
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            try:
                if name == "analyze_claims":
                    return await self._handle_analyze_claims(arguments)
                elif name == "fact_check_claim":
                    return await self._handle_fact_check_claim(arguments)
                elif name == "search_similar_claims":
                    return await self._handle_search_similar_claims(arguments)
                elif name == "get_recent_fact_checks":
                    return await self._handle_get_recent_fact_checks(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
    
    async def _handle_analyze_claims(self, arguments: dict) -> list[types.TextContent]:
        """Handle claim analysis requests"""
        text = arguments.get("text", "")
        source = arguments.get("source", "unknown")
        
        if not text.strip():
            return [types.TextContent(
                type="text",
                text="Error: No text provided for analysis"
            )]
        
        # Extract claims
        claims = self.claim_extractor.extract_claims(text, source)
        
        if not claims:
            return [types.TextContent(
                type="text",
                text="No verifiable claims found in the provided text."
            )]
        
        # Format results
        result = {
            "total_claims": len(claims),
            "claims": []
        }
        
        for claim in claims:
            result["claims"].append({
                "text": claim.text,
                "confidence": claim.confidence,
                "context": claim.context[:200] + "..." if len(claim.context) > 200 else claim.context,
                "claim_id": claim.claim_id
            })
        
        return [types.TextContent(
            type="text", 
            text=json.dumps(result, indent=2, default=str)
        )]
    
    async def _handle_fact_check_claim(self, arguments: dict) -> list[types.TextContent]:
        """Handle fact-checking requests"""
        claim_text = arguments.get("claim", "")
        detailed = arguments.get("detailed", True)
        
        if not claim_text.strip():
            return [types.TextContent(
                type="text",
                text="Error: No claim provided for fact-checking"
            )]
        
        # Create claim object
        claim = Claim(
            text=claim_text,
            source="mcp_request",
            timestamp=datetime.now(),
            claim_id=""
        )
        
        # Perform fact-check
        result = await self.fact_checker.fact_check_claim(claim)
        
        # Store result
        self.database.store_result(result)
        
        # Format response
        if detailed:
            response = {
                "claim": result.claim.text,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "sources": result.sources,
                "evidence_count": len(result.evidence),
                "timestamp": result.timestamp.isoformat()
            }
        else:
            response = {
                "claim": result.claim.text,
                "verdict": result.verdict,
                "confidence": result.confidence
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2, default=str)
        )]
    
    async def _handle_search_similar_claims(self, arguments: dict) -> list[types.TextContent]:
        """Handle similar claim search requests"""
        claim = arguments.get("claim", "")
        limit = arguments.get("limit", 5)
        
        if not claim.strip():
            return [types.TextContent(
                type="text",
                text="Error: No claim provided for similarity search"
            )]
        
        # This is a simplified search - in production, you'd use vector similarity
        similar_claims = self.database.get_recent_results(limit * 2)  # Get more to filter
        
        # Simple text matching for now
        matching_claims = []
        claim_words = set(claim.lower().split())
        
        for stored_claim in similar_claims:
            stored_words = set(stored_claim['claim_text'].lower().split())
            similarity = len(claim_words.intersection(stored_words)) / len(claim_words.union(stored_words))
            
            if similarity > 0.3:  # Similarity threshold
                matching_claims.append({
                    "claim": stored_claim['claim_text'],
                    "verdict": stored_claim['verdict'],
                    "confidence": stored_claim['confidence'],
                    "similarity": similarity,
                    "timestamp": stored_claim['timestamp']
                })
        
        # Sort by similarity and limit results
        matching_claims.sort(key=lambda x: x['similarity'], reverse=True)
        matching_claims = matching_claims[:limit]
        
        result = {
            "query_claim": claim,
            "similar_claims": matching_claims,
            "total_found": len(matching_claims)
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
    
    async def _handle_get_recent_fact_checks(self, arguments: dict) -> list[types.TextContent]:
        """Handle recent fact-checks requests"""
        limit = arguments.get("limit", 10)
        
        recent_results = self.database.get_recent_results(limit)
        
        result = {
            "recent_fact_checks": recent_results,
            "total_count": len(recent_results)
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
    
    def _register_resources(self):
        """Register MCP resources"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="claim://config",
                    name="Claim Analyzer Configuration",
                    description="Current configuration settings",
                    mimeType="application/json"
                ),
                Resource(
                    uri="claim://stats",
                    name="Fact-Check Statistics",
                    description="Statistics about fact-checking activity",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "claim://config":
                return json.dumps(self.config, indent=2)
            elif uri == "claim://stats":
                recent = self.database.get_recent_results(100)
                verdicts = {}
                for result in recent:
                    verdict = result['verdict']
                    verdicts[verdict] = verdicts.get(verdict, 0) + 1
                
                stats = {
                    "total_fact_checks": len(recent),
                    "verdict_distribution": verdicts,
                    "average_confidence": sum(r['confidence'] for r in recent) / len(recent) if recent else 0
                }
                return json.dumps(stats, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def run(self, transport_type: str = "stdio"):
        """Run the MCP server"""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="claim-analyzer",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

# Main execution
async def main():
    """Run the Claim Analyzer MCP Server"""
    server = ClaimAnalyzerMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
