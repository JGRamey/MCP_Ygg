#!/usr/bin/env python3
"""
Site-Specific Parser Plugins for MCP Yggdrasil
Phase 3: Specialized parsers for major academic and content sites
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ParsedContent:
    """Standardized parsed content structure."""
    title: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None
    citations: List[str] = None
    metadata: Dict[str, Any] = None
    raw_text: Optional[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.citations is None:
            self.citations = []
        if self.metadata is None:
            self.metadata = {}

class BaseSiteParser(ABC):
    """Base class for site-specific parsers."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.domains = self.get_supported_domains()
        
    @abstractmethod
    def get_supported_domains(self) -> List[str]:
        """Return list of supported domains."""
        pass
    
    @abstractmethod
    def can_parse(self, url: str) -> bool:
        """Check if this parser can handle the given URL."""
        pass
    
    @abstractmethod
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse HTML content and extract structured data."""
        pass
    
    def extract_json_ld(self, html_content: str) -> Dict[str, Any]:
        """Extract JSON-LD structured data."""
        try:
            json_ld_pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
            matches = re.findall(json_ld_pattern, html_content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    return data
                except json.JSONDecodeError:
                    continue
            return {}
        except Exception as e:
            logger.warning(f"JSON-LD extraction failed: {e}")
            return {}

class WikipediaParser(BaseSiteParser):
    """Parser for Wikipedia articles."""
    
    def get_supported_domains(self) -> List[str]:
        return ['wikipedia.org', 'en.wikipedia.org', 'simple.wikipedia.org']
    
    def can_parse(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.domains)
    
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse Wikipedia article."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = None
            title_elem = soup.find('h1', {'class': 'firstHeading'})
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            content = ""
            if content_div:
                # Remove infoboxes, navigation, and reference sections
                for unwanted in content_div.find_all(['table', 'div'], class_=['infobox', 'navbox', 'reflist']):
                    unwanted.decompose()
                
                # Extract paragraphs
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Extract categories (as keywords)
            keywords = []
            cat_links = soup.find_all('a', href=re.compile(r'/wiki/Category:'))
            keywords = [link.get_text() for link in cat_links[:10]]  # Limit to 10
            
            # Extract external links as citations
            citations = []
            external_links = soup.find('div', {'id': 'mw-external-links-box'})
            if external_links:
                links = external_links.find_all('a', {'class': 'external'})
                citations = [link.get('href', '') for link in links[:5]]  # Limit to 5
            
            return ParsedContent(
                title=title,
                content=content,
                keywords=keywords,
                citations=citations,
                metadata={'source': 'Wikipedia', 'url': url}
            )
            
        except Exception as e:
            logger.error(f"Wikipedia parsing failed: {e}")
            return ParsedContent(metadata={'error': str(e), 'url': url})

class ArXivParser(BaseSiteParser):
    """Parser for arXiv academic papers."""
    
    def get_supported_domains(self) -> List[str]:
        return ['arxiv.org']
    
    def can_parse(self, url: str) -> bool:
        parsed = urlparse(url)
        return 'arxiv.org' in parsed.netloc
    
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse arXiv paper page."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = None
            title_elem = soup.find('h1', {'class': 'title'})
            if title_elem:
                title = title_elem.get_text().replace('Title:', '').strip()
            
            # Extract authors
            author = None
            authors_elem = soup.find('div', {'class': 'authors'})
            if authors_elem:
                author_links = authors_elem.find_all('a')
                authors = [link.get_text().strip() for link in author_links]
                author = ', '.join(authors)
            
            # Extract abstract
            abstract = None
            abstract_elem = soup.find('blockquote', {'class': 'abstract'})
            if abstract_elem:
                abstract = abstract_elem.get_text().replace('Abstract:', '').strip()
            
            # Extract submission date
            date = None
            date_elem = soup.find('div', {'class': 'dateline'})
            if date_elem:
                date_text = date_elem.get_text()
                date_match = re.search(r'\d{1,2} \w+ \d{4}', date_text)
                if date_match:
                    date = date_match.group()
            
            # Extract keywords from subject classes
            keywords = []
            subj_elem = soup.find('td', {'class': 'subj-class'})
            if subj_elem:
                subjects = subj_elem.get_text().split(';')
                keywords = [subj.strip() for subj in subjects]
            
            return ParsedContent(
                title=title,
                content=abstract,  # For arXiv, abstract is the main content
                author=author,
                date=date,
                abstract=abstract,
                keywords=keywords,
                metadata={'source': 'arXiv', 'url': url, 'type': 'academic_paper'}
            )
            
        except Exception as e:
            logger.error(f"arXiv parsing failed: {e}")
            return ParsedContent(metadata={'error': str(e), 'url': url})

class PubMedParser(BaseSiteParser):
    """Parser for PubMed research papers."""
    
    def get_supported_domains(self) -> List[str]:
        return ['pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov']
    
    def can_parse(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.domains) and 'pubmed' in url
    
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse PubMed article page."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = None
            title_elem = soup.find('h1', {'class': 'heading-title'})
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract authors
            author = None
            authors_elem = soup.find('div', {'class': 'authors-list'})
            if authors_elem:
                author_buttons = authors_elem.find_all('button', {'class': 'full-name'})
                authors = [btn.get_text().strip() for btn in author_buttons]
                author = ', '.join(authors[:5])  # Limit to first 5 authors
            
            # Extract abstract
            abstract = None
            abstract_elem = soup.find('div', {'class': 'abstract-content'})
            if abstract_elem:
                abstract = abstract_elem.get_text().strip()
            
            # Extract publication date
            date = None
            date_elem = soup.find('span', {'class': 'cit'})
            if date_elem:
                date_text = date_elem.get_text()
                date_match = re.search(r'\d{4}', date_text)
                if date_match:
                    date = date_match.group()
            
            # Extract MeSH terms as keywords
            keywords = []
            mesh_terms = soup.find_all('button', {'class': 'keyword-actions-trigger'})
            keywords = [term.get_text().strip() for term in mesh_terms[:10]]
            
            # Extract DOI and PMID
            metadata = {'source': 'PubMed', 'url': url, 'type': 'medical_research'}
            
            doi_elem = soup.find('span', {'class': 'identifier'})
            if doi_elem and 'doi:' in doi_elem.get_text():
                metadata['doi'] = doi_elem.get_text().replace('doi:', '').strip()
            
            pmid_match = re.search(r'pubmed/(\d+)', url)
            if pmid_match:
                metadata['pmid'] = pmid_match.group(1)
            
            return ParsedContent(
                title=title,
                content=abstract,
                author=author,
                date=date,
                abstract=abstract,
                keywords=keywords,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"PubMed parsing failed: {e}")
            return ParsedContent(metadata={'error': str(e), 'url': url})

class StackOverflowParser(BaseSiteParser):
    """Parser for Stack Overflow questions and answers."""
    
    def get_supported_domains(self) -> List[str]:
        return ['stackoverflow.com', 'stackexchange.com']
    
    def can_parse(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.domains)
    
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse Stack Overflow question page."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract question title
            title = None
            title_elem = soup.find('h1', {'itemprop': 'name'})
            if not title_elem:
                title_elem = soup.find('a', {'class': 'question-hyperlink'})
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract question content
            question_elem = soup.find('div', {'class': 'js-post-body'})
            content = ""
            if question_elem:
                content = question_elem.get_text().strip()
            
            # Extract tags as keywords
            keywords = []
            tag_elems = soup.find_all('a', {'class': 'post-tag'})
            keywords = [tag.get_text().strip() for tag in tag_elems]
            
            # Extract author
            author = None
            author_elem = soup.find('div', {'class': 'user-details'})
            if author_elem:
                author_link = author_elem.find('a')
                if author_link:
                    author = author_link.get_text().strip()
            
            # Extract accepted answer if available
            accepted_answer = ""
            accepted_elem = soup.find('div', {'class': 'accepted-answer'})
            if accepted_elem:
                answer_body = accepted_elem.find('div', {'class': 'js-post-body'})
                if answer_body:
                    accepted_answer = answer_body.get_text().strip()[:500]  # First 500 chars
            
            if accepted_answer:
                content += f"\n\nAccepted Answer:\n{accepted_answer}"
            
            return ParsedContent(
                title=title,
                content=content,
                author=author,
                keywords=keywords,
                metadata={'source': 'Stack Overflow', 'url': url, 'type': 'technical_qa'}
            )
            
        except Exception as e:
            logger.error(f"Stack Overflow parsing failed: {e}")
            return ParsedContent(metadata={'error': str(e), 'url': url})

class GitHubParser(BaseSiteParser):
    """Parser for GitHub repositories and documentation."""
    
    def get_supported_domains(self) -> List[str]:
        return ['github.com']
    
    def can_parse(self, url: str) -> bool:
        parsed = urlparse(url)
        return 'github.com' in parsed.netloc
    
    def parse(self, html_content: str, url: str) -> ParsedContent:
        """Parse GitHub repository or file page."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Determine if it's a repo main page or file
            is_file = '/blob/' in url or '/tree/' in url
            
            if is_file:
                return self._parse_file_page(soup, url)
            else:
                return self._parse_repo_page(soup, url)
                
        except Exception as e:
            logger.error(f"GitHub parsing failed: {e}")
            return ParsedContent(metadata={'error': str(e), 'url': url})
    
    def _parse_repo_page(self, soup, url: str) -> ParsedContent:
        """Parse GitHub repository main page."""
        # Extract repository name
        title = None
        title_elem = soup.find('strong', {'itemprop': 'name'})
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Extract description
        desc_elem = soup.find('p', {'class': 'f4'})
        content = ""
        if desc_elem:
            content = desc_elem.get_text().strip()
        
        # Extract README content
        readme_elem = soup.find('div', {'data-target': 'readme-toc.content'})
        if readme_elem:
            readme_text = readme_elem.get_text().strip()
            content += f"\n\nREADME:\n{readme_text[:1000]}"  # First 1000 chars
        
        # Extract topics as keywords
        keywords = []
        topic_elems = soup.find_all('a', {'class': 'topic-tag'})
        keywords = [topic.get_text().strip() for topic in topic_elems]
        
        # Extract programming languages
        lang_elems = soup.find_all('span', {'class': 'color-fg-default'})
        languages = [lang.get_text().strip() for lang in lang_elems if lang.get_text().strip()]
        keywords.extend(languages[:5])  # Add up to 5 languages
        
        return ParsedContent(
            title=title,
            content=content,
            keywords=keywords,
            metadata={'source': 'GitHub', 'url': url, 'type': 'repository'}
        )
    
    def _parse_file_page(self, soup, url: str) -> ParsedContent:
        """Parse GitHub file page."""
        # Extract file name from URL
        title = url.split('/')[-1] if '/' in url else "GitHub File"
        
        # Extract file content
        content = ""
        code_elem = soup.find('table', {'class': 'js-file-line-container'})
        if code_elem:
            content = code_elem.get_text().strip()
        
        # Extract file type as keyword
        keywords = []
        if '.' in title:
            file_ext = title.split('.')[-1]
            keywords.append(file_ext)
        
        return ParsedContent(
            title=title,
            content=content[:2000],  # Limit content
            keywords=keywords,
            metadata={'source': 'GitHub', 'url': url, 'type': 'source_file'}
        )

class SiteParserManager:
    """Manager for site-specific parsers."""
    
    def __init__(self):
        self.parsers = [
            WikipediaParser(),
            ArXivParser(),
            PubMedParser(),
            StackOverflowParser(),
            GitHubParser()
        ]
        
        # Build domain mapping for fast lookup
        self.domain_map = {}
        for parser in self.parsers:
            for domain in parser.domains:
                if domain not in self.domain_map:
                    self.domain_map[domain] = []
                self.domain_map[domain].append(parser)
        
        logger.info(f"Initialized {len(self.parsers)} site-specific parsers")
        logger.info(f"Supported domains: {list(self.domain_map.keys())}")
    
    def get_parser(self, url: str) -> Optional[BaseSiteParser]:
        """Get the appropriate parser for a URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check exact domain match first
        if domain in self.domain_map:
            for parser in self.domain_map[domain]:
                if parser.can_parse(url):
                    return parser
        
        # Check subdomain matches
        for registered_domain, parsers in self.domain_map.items():
            if registered_domain in domain:
                for parser in parsers:
                    if parser.can_parse(url):
                        return parser
        
        return None
    
    def parse_content(self, html_content: str, url: str) -> Optional[ParsedContent]:
        """Parse content using the appropriate site-specific parser."""
        parser = self.get_parser(url)
        if parser:
            logger.info(f"Using {parser.name} for {url}")
            return parser.parse(html_content, url)
        
        logger.debug(f"No specific parser found for {url}")
        return None
    
    def get_supported_sites(self) -> Dict[str, List[str]]:
        """Get mapping of parser names to supported domains."""
        return {parser.name: parser.domains for parser in self.parsers}

# Global instance
site_parser_manager = SiteParserManager()

if __name__ == "__main__":
    # Test the parser manager
    manager = SiteParserManager()
    
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://arxiv.org/abs/2301.00001",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "https://stackoverflow.com/questions/12345/python-question",
        "https://github.com/user/repo"
    ]
    
    for url in test_urls:
        parser = manager.get_parser(url)
        if parser:
            print(f"✅ {url} -> {parser.name}")
        else:
            print(f"❌ {url} -> No parser found")