#!/usr/bin/env python3
"""
Intelligent Scraper Agent for MCP Yggdrasil
Phase 4: Enhanced web scraper with intelligence layer for content classification and authority scoring
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import re
import logging
from urllib.parse import urlparse

# Import existing scraper components
from .core.unified_web_scraper import UnifiedWebScraper

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type classification."""
    ACADEMIC_PAPER = "academic_paper"
    ENCYCLOPEDIA = "encyclopedia"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    FORUM_DISCUSSION = "forum_discussion"
    BOOK_EXCERPT = "book_excerpt"
    MANUSCRIPT = "manuscript"
    TECHNICAL_DOCS = "technical_documentation"
    RELIGIOUS_TEXT = "religious_text"
    HISTORICAL_RECORD = "historical_record"

class AuthorityLevel(Enum):
    """Source authority levels."""
    ACADEMIC = 5  # .edu, peer-reviewed
    GOVERNMENT = 4  # .gov
    ESTABLISHED_MEDIA = 3  # Major news outlets
    COMMUNITY_VERIFIED = 2  # Wikipedia, etc.
    PERSONAL = 1  # Blogs, personal sites

@dataclass
class DocumentMetadata:
    """Enhanced document metadata."""
    url: str
    title: str
    authors: List[str]
    publication_date: Optional[datetime]
    last_modified: Optional[datetime]
    domain: str
    content_type: ContentType
    authority_score: float
    citations: List[str]
    keywords: List[str]
    language: str
    word_count: int
    reading_time_minutes: int

@dataclass
class ScrapedDocument:
    """Complete scraped document with metadata."""
    content: str
    metadata: DocumentMetadata
    raw_html: Optional[str]
    extracted_data: Dict
    scraping_timestamp: datetime
    content_hash: str

class ContentClassifier:
    """Classify content type based on multiple signals."""
    
    def __init__(self):
        self.academic_indicators = [
            'abstract', 'methodology', 'results', 'conclusion',
            'references', 'keywords', 'doi:', 'issn:', 'isbn:'
        ]
        self.news_indicators = [
            'breaking news', 'reporter', 'correspondent', 'update:',
            'developing story', 'press release', 'statement'
        ]
        self.blog_indicators = [
            'posted by', 'comments', 'tags:', 'share this',
            'subscribe', 'follow me', 'personal opinion'
        ]
        
    def classify(self, content: str, scraped_data: Dict) -> ContentType:
        """Classify content type using multiple signals."""
        content_lower = content.lower()
        
        # Check structured data first
        structured = scraped_data.get('structured_data', {})
        if structured.get('extracted_info', {}).get('type'):
            schema_type = structured['extracted_info']['type']
            if schema_type == 'ScholarlyArticle':
                return ContentType.ACADEMIC_PAPER
            elif schema_type == 'NewsArticle':
                return ContentType.NEWS_ARTICLE
            elif schema_type == 'BlogPosting':
                return ContentType.BLOG_POST
        
        # Academic paper detection
        academic_score = sum(1 for indicator in self.academic_indicators 
                           if indicator in content_lower)
        if academic_score >= 3:
            return ContentType.ACADEMIC_PAPER
        
        # News article detection
        news_score = sum(1 for indicator in self.news_indicators 
                        if indicator in content_lower)
        if news_score >= 2:
            return ContentType.NEWS_ARTICLE
        
        # Blog post detection
        blog_score = sum(1 for indicator in self.blog_indicators 
                        if indicator in content_lower)
        if blog_score >= 2:
            return ContentType.BLOG_POST
        
        # Encyclopedia detection
        if any(term in content_lower for term in ['encyclopedia', 'reference work', 'wiki']):
            return ContentType.ENCYCLOPEDIA
        
        # Technical documentation
        if any(term in content_lower for term in ['documentation', 'api reference', 'user guide']):
            return ContentType.TECHNICAL_DOCS
        
        # Default to blog post for unclassified content
        return ContentType.BLOG_POST

class AuthorityScorer:
    """Score source authority and reliability."""
    
    def __init__(self):
        self.academic_domains = [
            '.edu', 'arxiv.org', 'pubmed', 'jstor.org', 'ieee.org',
            'springer.com', 'elsevier.com', 'nature.com', 'science.org'
        ]
        self.government_domains = ['.gov', '.mil']
        self.encyclopedia_domains = [
            'britannica.com', 'stanford.edu', 'iep.utm.edu',
            'plato.stanford.edu', 'wikipedia.org'
        ]
        self.news_domains = [
            'nytimes.com', 'washingtonpost.com', 'bbc.com',
            'reuters.com', 'apnews.com', 'cnn.com', 'npr.org'
        ]
        
    def score(self, domain: str) -> Tuple[float, AuthorityLevel]:
        """Score domain authority."""
        domain_lower = domain.lower()
        
        # Academic domains
        if any(d in domain_lower for d in self.academic_domains):
            return 0.9, AuthorityLevel.ACADEMIC
        
        # Government domains
        if any(d in domain_lower for d in self.government_domains):
            return 0.85, AuthorityLevel.GOVERNMENT
        
        # Established encyclopedias
        if any(d in domain_lower for d in self.encyclopedia_domains):
            return 0.8, AuthorityLevel.ESTABLISHED_MEDIA
        
        # Major news outlets
        if any(d in domain_lower for d in self.news_domains):
            return 0.7, AuthorityLevel.ESTABLISHED_MEDIA
        
        # Community verified (Wikipedia)
        if 'wikipedia.org' in domain_lower:
            return 0.6, AuthorityLevel.COMMUNITY_VERIFIED
        
        # Default/Personal
        return 0.3, AuthorityLevel.PERSONAL

class IntelligentScraperAgent:
    """Enhanced web scraper with intelligence layer."""
    
    def __init__(self, profile: str = 'academic'):
        self.unified_scraper = UnifiedWebScraper(profile=profile)
        self.authority_scorer = AuthorityScorer()
        self.content_classifier = ContentClassifier()
        
    async def scrape_with_intelligence(self, url: str) -> ScrapedDocument:
        """Enhanced scraping with metadata extraction and classification."""
        
        # Scrape content using existing infrastructure
        raw_result = await self.unified_scraper.scrape_url(url)
        
        if not raw_result['success']:
            raise Exception(f"Scraping failed: {raw_result.get('error')}")
        
        # Extract content and metadata
        content = raw_result['content']['main_text']
        
        # Classify content type
        content_type = self.detect_content_type(content, url, raw_result)
        
        # Extract comprehensive metadata
        metadata = self.extract_metadata(raw_result, content_type)
        
        # Assess source authority
        authority_score, authority_level = self.assess_source_authority(
            url, metadata, content_type
        )
        metadata.authority_score = authority_score
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return ScrapedDocument(
            content=content,
            metadata=metadata,
            raw_html=raw_result.get('raw_html'),
            extracted_data=raw_result,
            scraping_timestamp=datetime.utcnow(),
            content_hash=content_hash
        )
    
    def detect_content_type(self, content: str, url: str, scraped_data: Dict) -> ContentType:
        """Classify content type using multiple signals."""
        
        # URL patterns
        url_lower = url.lower()
        if any(domain in url_lower for domain in ['.edu', 'arxiv.org', 'pubmed', 'jstor']):
            if 'abstract' in content.lower()[:500]:
                return ContentType.ACADEMIC_PAPER
        
        if any(domain in url_lower for domain in ['wikipedia.org', 'britannica.com', 'stanford.edu/entries']):
            return ContentType.ENCYCLOPEDIA
        
        # Use content classifier
        return self.content_classifier.classify(content, scraped_data)
    
    def extract_metadata(self, scraped_data: Dict, content_type: ContentType) -> DocumentMetadata:
        """Extract comprehensive metadata from scraped content."""
        
        content_data = scraped_data.get('content', {})
        metadata_data = scraped_data.get('metadata', {})
        structured = scraped_data.get('structured_data', {}).get('extracted_info', {})
        
        # Extract authors
        authors = []
        if content_data.get('author'):
            authors.append(content_data['author'])
        elif structured.get('author'):
            if isinstance(structured['author'], dict):
                authors.append(structured['author'].get('name', ''))
            else:
                authors.append(str(structured['author']))
        elif metadata_data.get('article_data', {}).get('author'):
            authors.append(metadata_data['article_data']['author'])
        
        # Extract dates
        pub_date = None
        date_str = content_data.get('date') or structured.get('date_published')
        if date_str:
            try:
                pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                pass
        
        # Extract keywords/tags
        keywords = content_data.get('tags', [])
        if structured.get('keywords'):
            if isinstance(structured['keywords'], str):
                keywords.extend(structured['keywords'].split(','))
            else:
                keywords.extend(structured['keywords'])
        
        # Calculate reading metrics
        word_count = len(content_data.get('main_text', '').split())
        reading_time = max(1, word_count // 200)  # 200 words per minute
        
        # Extract domain
        parsed_url = urlparse(scraped_data['url'])
        domain = parsed_url.netloc
        
        return DocumentMetadata(
            url=scraped_data['url'],
            title=content_data.get('title', 'Untitled'),
            authors=[a for a in authors if a],  # Filter empty authors
            publication_date=pub_date,
            last_modified=None,
            domain=domain,
            content_type=content_type,
            authority_score=0.0,  # Set later
            citations=self._extract_citations(scraped_data),
            keywords=list(set(keywords)),
            language=scraped_data.get('language', {}).get('language', 'en'),
            word_count=word_count,
            reading_time_minutes=reading_time
        )
    
    def assess_source_authority(self, url: str, metadata: DocumentMetadata, 
                               content_type: ContentType) -> Tuple[float, AuthorityLevel]:
        """Score source reliability and authority."""
        
        # Get base domain score
        base_score, authority_level = self.authority_scorer.score(metadata.domain)
        
        # Adjust for content type
        content_multipliers = {
            ContentType.ACADEMIC_PAPER: 1.1,
            ContentType.ENCYCLOPEDIA: 1.05,
            ContentType.TECHNICAL_DOCS: 1.0,
            ContentType.NEWS_ARTICLE: 0.9,
            ContentType.BLOG_POST: 0.7,
            ContentType.FORUM_DISCUSSION: 0.6
        }
        
        multiplier = content_multipliers.get(content_type, 0.8)
        score = base_score * multiplier
        
        # Author credibility bonus
        if metadata.authors and len(metadata.authors) > 0:
            score += 0.05
        
        # Citation bonus
        if len(metadata.citations) > 5:
            score += 0.1
        elif len(metadata.citations) > 0:
            score += 0.05
        
        return min(1.0, score), authority_level
    
    def _extract_citations(self, scraped_data: Dict) -> List[str]:
        """Extract citations and references."""
        citations = []
        
        # Look in various places for citations
        content = scraped_data.get('content', {}).get('main_text', '')
        
        # Academic citation patterns
        patterns = [
            r'\([A-Z][a-z]+ et al\., \d{4}\)',  # (Smith et al., 2023)
            r'\([A-Z][a-z]+ \d{4}\)',           # (Smith 2023)
            r'\[[0-9]+\]',                       # [1], [2], etc.
            r'doi:\s*[^\s]+',                    # DOI references
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        # Also check structured data
        structured = scraped_data.get('structured_data', {}).get('extracted_info', {})
        if structured.get('citation'):
            citations.extend(structured['citation'])
        
        return list(set(citations))[:20]  # Limit to 20 unique citations