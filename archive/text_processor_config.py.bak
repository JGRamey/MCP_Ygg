#!/usr/bin/env python3
"""
Text Processor Configuration and Settings
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import multiprocessing as mp


# Default configuration for text processor
TEXT_PROCESSOR_CONFIG = {
    'processing': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'min_text_length': 100,
        'max_text_length': 1000000,
        'max_workers': min(mp.cpu_count(), 8),  # Limit to 8 cores max
        'batch_size': 50,
        'enable_parallel_processing': True,
        'timeout_seconds': 300  # 5 minutes timeout per document
    },
    
    'embedding_models': {
        'default': 'all-MiniLM-L6-v2',
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
        'domain_specific': {
            'science': 'allenai/scibert_scivocab_uncased',
            'math': 'sentence-transformers/all-MiniLM-L6-v2',
            'literature': 'sentence-transformers/all-MiniLM-L6-v2',
            'philosophy': 'sentence-transformers/all-MiniLM-L6-v2',
            'history': 'sentence-transformers/all-MiniLM-L6-v2',
            'religion': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'cache_embeddings': True,
        'embedding_dimension': 384,  # Default for MiniLM models
        'normalize_embeddings': True
    },
    
    'nlp_models': {
        'spacy_models': {
            'english': 'en_core_web_lg',
            'latin': 'la_core_web_sm',
            'greek': 'grc_proiel_sm',
            'german': 'de_core_news_lg',
            'french': 'fr_core_news_lg',
            'spanish': 'es_core_news_lg',
            'italian': 'it_core_news_lg',
            'portuguese': 'pt_core_news_lg'
        },
        'fallback_model': 'en_core_web_sm',
        'disable_components': ['parser'],  # Disable for speed if not needed
        'entity_types': [
            'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART',
            'LAW', 'LANGUAGE', 'DATE', 'TIME', 'MONEY', 'QUANTITY',
            'ORDINAL', 'CARDINAL'
        ]
    },
    
    'languages': {
        'supported': [
            'english', 'latin', 'greek', 'german', 'french', 'spanish',
            'italian', 'portuguese', 'russian', 'arabic', 'hebrew',
            'sanskrit', 'chinese', 'japanese'
        ],
        'detection': {
            'confidence_threshold': 0.7,
            'fallback_language': 'english',
            'use_statistical_detection': True
        },
        'preprocessing': {
            'normalize_unicode': True,
            'remove_accents': False,  # Keep accents for historical texts
            'lowercase': False,  # Preserve case for proper nouns
            'remove_punctuation': False
        }
    },
    
    'chunking': {
        'methods': ['sentences', 'paragraphs', 'semantic'],
        'default_method': 'sentences',
        'sentence_chunking': {
            'min_sentences_per_chunk': 3,
            'max_sentences_per_chunk': 20,
            'respect_paragraph_boundaries': True
        },
        'paragraph_chunking': {
            'min_paragraphs_per_chunk': 1,
            'max_paragraphs_per_chunk': 5
        },
        'semantic_chunking': {
            'similarity_threshold': 0.8,
            'min_chunk_size': 500,
            'max_chunk_size': 2000
        }
    },
    
    'entity_extraction': {
        'enable_custom_entities': True,
        'custom_patterns': {
            'biblical_books': [
                'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                'Matthew', 'Mark', 'Luke', 'John', 'Romans', 'Corinthians'
            ],
            'mathematical_concepts': [
                'theorem', 'lemma', 'corollary', 'proof', 'axiom',
                'algebra', 'geometry', 'calculus', 'topology'
            ],
            'philosophical_concepts': [
                'metaphysics', 'epistemology', 'ethics', 'aesthetics',
                'logic', 'ontology', 'phenomenology'
            ]
        },
        'confidence_threshold': 0.5,
        'enable_entity_linking': True,
        'knowledge_bases': {
            'wikidata': True,
            'dbpedia': True,
            'custom': True
        }
    },
    
    'text_cleaning': {
        'remove_headers_footers': True,
        'remove_page_numbers': True,
        'fix_ocr_errors': True,
        'normalize_whitespace': True,
        'preserve_structure': True,
        'handle_footnotes': 'preserve',  # 'preserve', 'remove', or 'inline'
        'encoding_normalization': {
            'target_encoding': 'utf-8',
            'handle_unknown_chars': 'replace',
            'normalize_unicode': 'NFC'
        }
    },
    
    'quality_control': {
        'min_word_count': 50,
        'max_word_count': 100000,
        'min_sentence_count': 3,
        'language_detection_confidence': 0.7,
        'duplicate_detection': {
            'enabled': True,
            'similarity_threshold': 0.95,
            'hash_algorithm': 'sha256'
        },
        'content_validation': {
            'check_encoding': True,
            'check_language_consistency': True,
            'detect_corrupted_text': True
        }
    },
    
    'output': {
        'base_dir': 'data/processed',
        'subdirs': {
            'documents': 'documents',
            'embeddings': 'embeddings',
            'entities': 'entities',
            'chunks': 'chunks',
            'metadata': 'metadata'
        },
        'formats': {
            'json': True,
            'pickle': True,
            'parquet': False,  # For large-scale processing
            'hdf5': False     # For numerical data
        },
        'compression': {
            'enabled': True,
            'algorithm': 'gzip',  # 'gzip', 'bz2', 'lzma'
            'level': 6
        }
    },
    
    'performance': {
        'memory_limit_mb': 4096,  # 4GB memory limit
        'gc_threshold': 100,      # Run garbage collection every N documents
        'profile_performance': False,
        'cache_models': True,
        'lazy_loading': True,
        'optimization': {
            'use_fast_tokenizers': True,
            'enable_mixed_precision': False,  # For GPU processing
            'batch_inference': True
        }
    },
    
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/text_processor.log',
        'max_file_size_mb': 100,
        'backup_count': 5,
        'log_processing_stats': True
    },
    
    'domain_specific': {
        'math': {
            'enable_latex_parsing': True,
            'preserve_equations': True,
            'extract_theorems': True,
            'symbol_normalization': True
        },
        'religion': {
            'enable_scripture_parsing': True,
            'preserve_verse_structure': True,
            'extract_references': True,
            'handle_archaic_language': True
        },
        'history': {
            'enable_date_extraction': True,
            'preserve_chronology': True,
            'extract_locations': True,
            'handle_historical_names': True
        },
        'literature': {
            'preserve_formatting': True,
            'extract_dialogue': True,
            'identify_narrative_structure': True,
            'handle_poetry': True
        },
        'philosophy': {
            'extract_arguments': True,
            'identify_logical_structure': True,
            'preserve_citations': True,
            'handle_technical_terms': True
        },
        'science': {
            'extract_methodology': True,
            'preserve_data_sections': True,
            'identify_hypotheses': True,
            'handle_technical_notation': True
        }
    }
}


class ConfigManager:
    """Manages text processor configuration"""
    
    def __init__(self, config_file: str = "agents/text_processor/config.yaml"):
        """Initialize configuration manager"""
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge with defaults
                config = self._deep_merge(TEXT_PROCESSOR_CONFIG.copy(), file_config)
                return config
                
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                return TEXT_PROCESSOR_CONFIG.copy()
        else:
            return TEXT_PROCESSOR_CONFIG.copy()
    
    def save_config(self, config: Optional[Dict] = None) -> None:
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        env_mappings = {
            'TEXT_PROCESSOR_MAX_WORKERS': 'processing.max_workers',
            'TEXT_PROCESSOR_CHUNK_SIZE': 'processing.chunk_size',
            'TEXT_PROCESSOR_BATCH_SIZE': 'processing.batch_size',
            'TEXT_PROCESSOR_OUTPUT_DIR': 'output.base_dir',
            'TEXT_PROCESSOR_LOG_LEVEL': 'logging.level',
            'TEXT_PROCESSOR_MEMORY_LIMIT': 'performance.memory_limit_mb',
            'EMBEDDING_MODEL_DEFAULT': 'embedding_models.default',
            'SPACY_MODEL_ENGLISH': 'nlp_models.spacy_models.english'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_key.endswith(('_workers', '_size', '_limit', '_count')):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_key.endswith(('_threshold', '_confidence')):
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                
                self.set(config_key, value)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required directories
        output_dir = Path(self.get('output.base_dir', 'data/processed'))
        if not output_dir.parent.exists():
            issues.append(f"Output directory parent does not exist: {output_dir.parent}")
        
        # Check memory limits
        memory_limit = self.get('performance.memory_limit_mb', 4096)
        if memory_limit < 512:
            issues.append("Memory limit too low (minimum 512MB recommended)")
        
        # Check worker count
        max_workers = self.get('processing.max_workers', 4)
        if max_workers > mp.cpu_count() * 2:
            issues.append(f"Too many workers ({max_workers}), maximum recommended: {mp.cpu_count() * 2}")
        
        # Check chunk size
        chunk_size = self.get('processing.chunk_size', 1000)
        chunk_overlap = self.get('processing.chunk_overlap', 200)
        if chunk_overlap >= chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
        
        return issues
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def get_domain_config(self, domain: str) -> Dict:
        """Get domain-specific configuration"""
        base_config = self.config.copy()
        domain_config = self.get(f'domain_specific.{domain}', {})
        
        if domain_config:
            base_config.update(domain_config)
        
        return base_config
    
    def get_language_config(self, language: str) -> Dict:
        """Get language-specific configuration"""
        config = {
            'spacy_model': self.get(f'nlp_models.spacy_models.{language}'),
            'embedding_model': self.get('embedding_models.multilingual') if language != 'english' else self.get('embedding_models.default'),
            'preprocessing': self.get('languages.preprocessing', {}),
            'fallback_model': self.get('nlp_models.fallback_model')
        }
        
        return config


def create_config_file():
    """Create the configuration file with defaults"""
    config_path = Path("agents/text_processor/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(TEXT_PROCESSOR_CONFIG, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created text processor config: {config_path}")


def validate_spacy_models():
    """Validate that required spaCy models are available"""
    import spacy
    
    required_models = TEXT_PROCESSOR_CONFIG['nlp_models']['spacy_models']
    available_models = []
    missing_models = []
    
    for language, model_name in required_models.items():
        try:
            spacy.load(model_name)
            available_models.append((language, model_name))
        except OSError:
            missing_models.append((language, model_name))
    
    print(f"Available spaCy models: {len(available_models)}")
    for lang, model in available_models:
        print(f"  ✅ {lang}: {model}")
    
    if missing_models:
        print(f"\nMissing spaCy models: {len(missing_models)}")
        for lang, model in missing_models:
            print(f"  ❌ {lang}: {model}")
        
        print("\nTo install missing models:")
        for lang, model in missing_models:
            print(f"  python -m spacy download {model}")


if __name__ == "__main__":
    create_config_file()
    validate_spacy_models()
