#!/usr/bin/env python3
"""
Scraper Configuration and User Sources Management
"""

import yaml
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# agents/scraper/config.yaml
SCRAPER_CONFIG = {
    'general': {
        'user_agent': 'MCP-Server/1.0 (Academic Research; +https://github.com/mcp-project)',
        'request_delay': 1.0,
        'max_retries': 3,
        'timeout': 30,
        'max_concurrent': 5,
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'supported_formats': ['pdf', 'html', 'txt', 'doc', 'docx', 'epub'],
        'output_dir': 'data/raw',
        'respect_robots_txt': True
    },
    
    'domains': {
        'math': {
            'sources': [
                'arxiv.org/list/math',
                'mathworld.wolfram.com',
                'oeis.org',
                'projecteuclid.org'
            ],
            'keywords': ['theorem', 'proof', 'equation', 'mathematics', 'algebra', 'geometry'],
            'rate_limit': 2.0
        },
        'science': {
            'sources': [
                'arxiv.org/list/physics',
                'pubmed.ncbi.nlm.nih.gov',
                'nature.com',
                'sciencedirect.com'
            ],
            'keywords': ['research', 'experiment', 'hypothesis', 'data', 'analysis'],
            'rate_limit': 2.0
        },
        'religion': {
            'sources': [
                'sacred-texts.com',
                'ccel.org',
                'earlychristianwritings.com',
                'perseus.tufts.edu'
            ],
            'keywords': ['scripture', 'theology', 'doctrine', 'faith', 'spiritual'],
            'rate_limit': 1.0
        },
        'history': {
            'sources': [
                'archive.org',
                'loc.gov',
                'europeana.eu',
                'worldhistory.org'
            ],
            'keywords': ['historical', 'ancient', 'medieval', 'chronology', 'civilization'],
            'rate_limit': 1.5
        },
        'literature': {
            'sources': [
                'gutenberg.org',
                'archive.org/details/texts',
                'poetryfoundation.org',
                'bartleby.com'
            ],
            'keywords': ['poetry', 'novel', 'drama', 'literature', 'author', 'text'],
            'rate_limit': 1.0
        },
        'philosophy': {
            'sources': [
                'plato.stanford.edu',
                'iep.utm.edu',
                'archive.org/details/philosophy',
                'perseus.tufts.edu'
            ],
            'keywords': ['philosophy', 'ethics', 'metaphysics', 'logic', 'epistemology'],
            'rate_limit': 1.5
        }
    },
    
    'source_types': {
        'academic': {
            'api_keys': {
                'crossref': None,
                'pubmed': None,
                'arxiv': None,
                'ieee': None
            },
            'rate_limits': {
                'crossref': 1.0,
                'pubmed': 0.5,
                'arxiv': 3.0,
                'ieee': 2.0
            },
            'authentication': True
        },
        'public_domain': {
            'rate_limits': {
                'gutenberg': 1.0,
                'archive_org': 2.0,
                'wikisource': 1.0
            },
            'authentication': False
        },
        'manuscript': {
            'ocr_required': True,
            'languages': ['en', 'la', 'grc', 'de', 'fr'],
            'preprocessing': True
        },
        'tablet': {
            'ocr_required': True,
            'languages': ['akk', 'sux', 'egy'],  # Akkadian, Sumerian, Egyptian
            'special_processing': True
        }
    },
    
    'licensing': {
        'allowed_licenses': [
            'public_domain',
            'creative_commons',
            'mit',
            'apache',
            'gpl'
        ],
        'check_copyright': True,
        'require_license': True,
        'fallback_to_fair_use': False
    },
    
    'quality_control': {
        'min_word_count': 100,
        'max_word_count': 1000000,
        'language_detection': True,
        'duplicate_detection': True,
        'content_validation': True
    }
}


@dataclass
class UserSource:
    """User-specified data source"""
    url: str
    domain: str
    subcategory: str
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    priority: int = 1  # 1-5, higher is more important
    added_by: Optional[str] = None
    added_at: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class UserSourcesManager:
    """Manages user-specified data sources"""
    
    def __init__(self, sources_file: str = "agents/scraper/user_sources.json"):
        self.sources_file = Path(sources_file)
        self.sources: List[UserSource] = []
        self.load_sources()
    
    def load_sources(self) -> None:
        """Load user sources from file"""
        if self.sources_file.exists():
            try:
                with open(self.sources_file, 'r') as f:
                    data = json.load(f)
                    self.sources = [UserSource(**item) for item in data.get('sources', [])]
                logger.info(f"Loaded {len(self.sources)} user sources")
            except Exception as e:
                logger.error(f"Error loading user sources: {e}")
                self.sources = []
        else:
            self.sources = []
    
    def save_sources(self) -> None:
        """Save user sources to file"""
        try:
            self.sources_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'sources': [asdict(source) for source in self.sources],
                'total_count': len(self.sources),
                'last_updated': None
            }
            
            with open(self.sources_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(self.sources)} user sources")
            
        except Exception as e:
            logger.error(f"Error saving user sources: {e}")
    
    def add_source(self, source: UserSource) -> bool:
        """Add a new user source"""
        # Check for duplicates
        for existing in self.sources:
            if existing.url == source.url:
                logger.warning(f"Source already exists: {source.url}")
                return False
        
        # Validate URL
        if not self.validate_url(source.url):
            logger.error(f"Invalid URL: {source.url}")
            return False
        
        # Validate domain
        if not self.validate_domain(source.domain):
            logger.error(f"Invalid domain: {source.domain}")
            return False
        
        self.sources.append(source)
        self.save_sources()
        logger.info(f"Added user source: {source.url}")
        return True
    
    def remove_source(self, url: str) -> bool:
        """Remove a user source by URL"""
        for i, source in enumerate(self.sources):
            if source.url == url:
                del self.sources[i]
                self.save_sources()
                logger.info(f"Removed user source: {url}")
                return True
        
        logger.warning(f"Source not found: {url}")
        return False
    
    def get_sources_by_domain(self, domain: str) -> List[UserSource]:
        """Get sources for a specific domain"""
        return [source for source in self.sources if source.domain == domain]
    
    def get_sources_by_priority(self, min_priority: int = 1) -> List[UserSource]:
        """Get sources with minimum priority"""
        return [source for source in self.sources if source.priority >= min_priority]
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def validate_domain(self, domain: str) -> bool:
        """Validate domain name"""
        valid_domains = ['math', 'science', 'religion', 'history', 'literature', 'philosophy']
        return domain.lower() in valid_domains
    
    def search_sources(self, query: str) -> List[UserSource]:
        """Search sources by title, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for source in self.sources:
            if (query_lower in (source.title or '').lower() or
                query_lower in (source.description or '').lower() or
                any(query_lower in tag.lower() for tag in source.tags)):
                results.append(source)
        
        return results
    
    def bulk_add_from_file(self, file_path: str) -> int:
        """Add sources from CSV or JSON file"""
        file_path = Path(file_path)
        added_count = 0
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        source = UserSource(**item)
                        if self.add_source(source):
                            added_count += 1
            
            elif file_path.suffix.lower() == '.csv':
                import csv
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert tags from string to list
                        if 'tags' in row and isinstance(row['tags'], str):
                            row['tags'] = [tag.strip() for tag in row['tags'].split(',')]
                        
                        source = UserSource(**row)
                        if self.add_source(source):
                            added_count += 1
            
            logger.info(f"Bulk added {added_count} sources from {file_path}")
            return added_count
            
        except Exception as e:
            logger.error(f"Error bulk adding sources: {e}")
            return 0


class SourceValidator:
    """Validates and categorizes data sources"""
    
    ACADEMIC_PATTERNS = [
        r'arxiv\.org',
        r'pubmed\.ncbi\.nlm\.nih\.gov',
        r'ieee\.org',
        r'acm\.org',
        r'jstor\.org',
        r'springer\.com',
        r'sciencedirect\.com',
        r'nature\.com'
    ]
    
    PUBLIC_DOMAIN_PATTERNS = [
        r'gutenberg\.org',
        r'archive\.org',
        r'wikisource\.org',
        r'sacred-texts\.com',
        r'perseus\.tufts\.edu'
    ]
    
    def __init__(self):
        self.academic_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.ACADEMIC_PATTERNS]
        self.public_domain_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.PUBLIC_DOMAIN_PATTERNS]
    
    def classify_source(self, url: str) -> str:
        """Classify source type based on URL"""
        for pattern in self.academic_regex:
            if pattern.search(url):
                return 'academic'
        
        for pattern in self.public_domain_regex:
            if pattern.search(url):
                return 'public_domain'
        
        return 'unknown'
    
    def validate_academic_source(self, url: str) -> Dict[str, bool]:
        """Validate academic source requirements"""
        return {
            'has_doi': 'doi.org' in url or 'dx.doi.org' in url,
            'peer_reviewed': self.classify_source(url) == 'academic',
            'accessible': True,  # Would check actual accessibility
            'recent': True  # Would check publication date
        }
    
    def suggest_domain(self, url: str, title: str = '', content: str = '') -> str:
        """Suggest domain based on URL and content"""
        text = f"{url} {title} {content}".lower()
        
        domain_keywords = {
            'math': ['math', 'theorem', 'proof', 'equation', 'algebra', 'geometry', 'calculus'],
            'science': ['research', 'experiment', 'hypothesis', 'data', 'physics', 'chemistry', 'biology'],
            'religion': ['scripture', 'theology', 'doctrine', 'faith', 'spiritual', 'religious'],
            'history': ['historical', 'ancient', 'medieval', 'chronology', 'civilization'],
            'literature': ['poetry', 'novel', 'drama', 'literature', 'author', 'text'],
            'philosophy': ['philosophy', 'ethics', 'metaphysics', 'logic', 'epistemology']
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[domain] = score
        
        return max(scores, key=scores.get) if scores else 'unknown'


def create_scraper_config_file():
    """Create the scraper configuration file"""
    config_path = Path("agents/scraper/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(SCRAPER_CONFIG, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created scraper config: {config_path}")


def create_example_user_sources():
    """Create example user sources file"""
    manager = UserSourcesManager()
    
    # Add some example sources
    example_sources = [
        UserSource(
            url="https://www.gutenberg.org/files/1342/1342-h/1342-h.htm",
            domain="literature",
            subcategory="fiction",
            title="Pride and Prejudice",
            author="Jane Austen",
            description="Classic English novel",
            priority=3,
            tags=["classic", "novel", "19th-century"]
        ),
        UserSource(
            url="https://plato.stanford.edu/entries/aristotle/",
            domain="philosophy",
            subcategory="ancient",
            title="Aristotle",
            author="Stanford Encyclopedia of Philosophy",
            description="Comprehensive article on Aristotle",
            priority=4,
            tags=["aristotle", "ancient", "ethics"]
        ),
        UserSource(
            url="https://sacred-texts.com/bib/kjv/",
            domain="religion",
            subcategory="christianity",
            title="King James Bible",
            author="Various",
            description="King James Version of the Bible",
            priority=5,
            tags=["bible", "christianity", "scripture"]
        )
    ]
    
    for source in example_sources:
        manager.add_source(source)
    
    print(f"✅ Created example user sources with {len(example_sources)} entries")


if __name__ == "__main__":
    create_scraper_config_file()
    create_example_user_sources()
