#!/usr/bin/env python3
"""
Vector Index Configuration and Utility Functions
"""

import yaml
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import json
import hashlib
from datetime import datetime
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


# agents/vector_index/config.yaml
VECTOR_INDEX_CONFIG = {
    'qdrant': {
        'host': 'localhost',
        'port': 6333,
        'api_key': None,
        'timeout': 60,
        'prefer_grpc': True,
        'https': False,
        'prefix': 'mcp_',
        'connection_pool_size': 10
    },
    
    'redis': {
        'url': 'redis://localhost:6379',
        'prefix': 'mcp_vector:',
        'max_connections': 20,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
        'retry_on_timeout': True,
        'decode_responses': False
    },
    
    'collections': {
        'vector_size': 384,  # Default for sentence-transformers
        'distance_metric': 'Cosine',  # Cosine, Dot, Euclid
        'hnsw_config': {
            'm': 16,                    # Number of bi-directional links for each node
            'ef_construct': 100,        # Size of the dynamic candidate list
            'full_scan_threshold': 10000,  # Threshold for full scan vs HNSW
            'max_indexing_threads': 0,  # 0 = auto-detect
            'on_disk': False,           # Store HNSW index on disk
            'payload_m': None           # Number of links for payload index
        },
        'quantization': {
            'enabled': True,
            'type': 'scalar',           # scalar, product
            'scalar_config': {
                'type': 'int8',         # int8, uint8
                'quantile': 0.99,       # Quantile for clipping
                'always_ram': True      # Keep quantized vectors in RAM
            }
        },
        'optimizer': {
            'deleted_threshold': 0.2,   # Fraction of deleted vectors to trigger optimization
            'vacuum_min_vector_number': 1000,  # Minimum vectors for vacuum
            'default_segment_number': 0,       # 0 = auto
            'max_segment_size': None,           # Maximum segment size in KB
            'memmap_threshold': None,           # Threshold for memory mapping
            'indexing_threshold': 20000,        # Threshold for indexing
            'flush_interval_sec': 5,            # Flush interval
            'max_optimization_threads': 1      # Max optimization threads
        },
        'replication': {
            'replication_factor': 1,    # Number of replicas
            'write_consistency_factor': 1,  # Write consistency
            'read_fan_out_factor': None     # Read fan-out factor
        }
    },
    
    'indexing': {
        'batch_size': 100,
        'max_workers': min(mp.cpu_count(), 8),
        'chunk_size': 1000,
        'enable_parallel_processing': True,
        'timeout_per_batch': 300,   # 5 minutes
        'retry_failed_batches': True,
        'max_retries': 3,
        'retry_delay': 5.0,
        'memory_limit_mb': 2048,    # 2GB per worker
        'progress_reporting': True
    },
    
    'search': {
        'default_limit': 10,
        'max_limit': 1000,
        'score_threshold': 0.7,
        'enable_caching': True,
        'cache_ttl': 3600,          # 1 hour
        'exact_search_threshold': 1000,  # Use exact search below this size
        'search_timeout': 30,       # Search timeout in seconds
        'enable_rescoring': False,  # Enable result rescoring
        'rescore_query': None       # Custom rescoring query
    },
    
    'domains': {
        'math': {
            'collection_suffix': 'math',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 200,  # Higher for mathematical precision
                'quantization_enabled': True
            }
        },
        'science': {
            'collection_suffix': 'science',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 150,
                'quantization_enabled': True
            }
        },
        'religion': {
            'collection_suffix': 'religion',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 100,
                'quantization_enabled': True
            }
        },
        'history': {
            'collection_suffix': 'history',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 100,
                'quantization_enabled': True
            }
        },
        'literature': {
            'collection_suffix': 'literature',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 120,
                'quantization_enabled': True
            }
        },
        'philosophy': {
            'collection_suffix': 'philosophy',
            'vector_size': 384,
            'special_config': {
                'hnsw_ef_construct': 150,
                'quantization_enabled': True
            }
        }
    },
    
    'backup': {
        'enabled': True,
        'schedule': 'daily',        # daily, weekly, monthly
        'retention_days': 30,
        'backup_location': 'data/backups/vector_index',
        'compression': True,
        'incremental': True,
        'verify_backups': True
    },
    
    'monitoring': {
        'enabled': True,
        'metrics_collection': True,
        'performance_logging': True,
        'alert_thresholds': {
            'memory_usage_percent': 85,
            'disk_usage_percent': 90,
            'search_latency_ms': 1000,
            'indexing_failure_rate': 0.05
        },
        'health_check_interval': 60  # seconds
    },
    
    'optimization': {
        'auto_optimize': True,
        'optimization_schedule': 'nightly',  # nightly, weekly, manual
        'vacuum_threshold': 0.1,     # Fraction of deleted vectors
        'merge_threshold': 10,       # Number of segments to trigger merge
        'background_optimization': True,
        'max_optimization_time': 3600  # 1 hour
    },
    
    'security': {
        'enable_api_key': False,
        'enable_tls': False,
        'tls_config': {
            'cert_file': None,
            'key_file': None,
            'ca_file': None
        },
        'rate_limiting': {
            'enabled': False,
            'requests_per_minute': 1000,
            'burst_size': 100
        }
    },
    
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/vector_indexer.log',
        'max_file_size_mb': 100,
        'backup_count': 5,
        'log_search_queries': True,
        'log_performance_metrics': True
    }
}


class VectorUtils:
    """Utility functions for vector operations"""
    
    @staticmethod
    def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """Normalize vector to unit length"""
        if isinstance(vector, list):
            vector = np.array(vector)
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    @staticmethod
    def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
        """Calculate cosine similarity between two vectors"""
        if isinstance(v1, list):
            v1 = np.array(v1)
        if isinstance(v2, list):
            v2 = np.array(v2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def euclidean_distance(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if isinstance(v1, list):
            v1 = np.array(v1)
        if isinstance(v2, list):
            v2 = np.array(v2)
        
        return np.linalg.norm(v1 - v2)
    
    @staticmethod
    def dot_product(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
        """Calculate dot product between two vectors"""
        if isinstance(v1, list):
            v1 = np.array(v1)
        if isinstance(v2, list):
            v2 = np.array(v2)
        
        return np.dot(v1, v2)
    
    @staticmethod
    def batch_normalize_vectors(vectors: List[Union[List[float], np.ndarray]]) -> List[np.ndarray]:
        """Normalize a batch of vectors"""
        return [VectorUtils.normalize_vector(v) for v in vectors]
    
    @staticmethod
    def calculate_centroid(vectors: List[Union[List[float], np.ndarray]]) -> np.ndarray:
        """Calculate centroid of a set of vectors"""
        if not vectors:
            return np.array([])
        
        arrays = [np.array(v) if isinstance(v, list) else v for v in vectors]
        return np.mean(arrays, axis=0)
    
    @staticmethod
    def find_outliers(vectors: List[Union[List[float], np.ndarray]], threshold: float = 2.0) -> List[int]:
        """Find outlier vectors using standard deviation"""
        if len(vectors) < 3:
            return []
        
        arrays = [np.array(v) if isinstance(v, list) else v for v in vectors]
        centroid = np.mean(arrays, axis=0)
        
        distances = [np.linalg.norm(v - centroid) for v in arrays]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        outliers = []
        for i, distance in enumerate(distances):
            if abs(distance - mean_distance) > threshold * std_distance:
                outliers.append(i)
        
        return outliers
    
    @staticmethod
    def reduce_dimensionality(vectors: List[Union[List[float], np.ndarray]], target_dim: int) -> List[np.ndarray]:
        """Reduce dimensionality using PCA"""
        try:
            from sklearn.decomposition import PCA
            
            arrays = [np.array(v) if isinstance(v, list) else v for v in vectors]
            if not arrays:
                return []
            
            matrix = np.vstack(arrays)
            
            pca = PCA(n_components=min(target_dim, matrix.shape[1]))
            reduced = pca.fit_transform(matrix)
            
            return [reduced[i] for i in range(reduced.shape[0])]
            
        except ImportError:
            logger.warning("scikit-learn not available for PCA")
            return vectors


class MetadataExtractor:
    """Extract and format metadata for vector storage"""
    
    @staticmethod
    def extract_document_metadata(processed_doc: Dict) -> Dict[str, Any]:
        """Extract metadata from processed document"""
        metadata = {
            'doc_id': processed_doc.get('doc_id', ''),
            'title': processed_doc.get('title', '')[:200],  # Truncate for storage
            'author': processed_doc.get('author', '')[:100],
            'domain': processed_doc.get('domain', ''),
            'subcategory': processed_doc.get('subcategory', ''),
            'language': processed_doc.get('language', ''),
            'date': processed_doc.get('date', ''),
            'era': processed_doc.get('processing_metadata', {}).get('era', ''),
            'source': processed_doc.get('source', '')[:200],
            'word_count': processed_doc.get('word_count', 0),
            'chunk_count': processed_doc.get('chunk_count', 0),
            'entity_count': len(processed_doc.get('entities', [])),
            'content_hash': MetadataExtractor._calculate_content_hash(processed_doc.get('cleaned_content', '')),
            'indexed_at': datetime.now().isoformat(),
            'embedding_model': processed_doc.get('processing_metadata', {}).get('embedding_model', ''),
            'processing_version': '1.0'
        }
        
        # Add quality scores
        quality_scores = MetadataExtractor._calculate_quality_scores(processed_doc)
        metadata.update(quality_scores)
        
        return metadata
    
    @staticmethod
    def extract_chunk_metadata(processed_doc: Dict, chunk_index: int, chunk_text: str) -> Dict[str, Any]:
        """Extract metadata for a document chunk"""
        base_metadata = MetadataExtractor.extract_document_metadata(processed_doc)
        
        chunk_metadata = {
            **base_metadata,
            'type': 'chunk',
            'chunk_index': chunk_index,
            'chunk_text': chunk_text[:500],  # Store excerpt
            'chunk_word_count': len(chunk_text.split()),
            'chunk_char_count': len(chunk_text),
            'parent_doc_id': processed_doc.get('doc_id', ''),
            'chunk_hash': MetadataExtractor._calculate_content_hash(chunk_text)
        }
        
        return chunk_metadata
    
    @staticmethod
    def _calculate_content_hash(content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @staticmethod
    def _calculate_quality_scores(processed_doc: Dict) -> Dict[str, float]:
        """Calculate quality scores for the document"""
        content = processed_doc.get('cleaned_content', '')
        word_count = processed_doc.get('word_count', 0)
        entities = processed_doc.get('entities', [])
        
        scores = {
            'content_quality_score': 0.0,
            'entity_density_score': 0.0,
            'language_confidence_score': processed_doc.get('processing_metadata', {}).get('language_confidence', 0.0)
        }
        
        # Content quality based on length and structure
        if word_count > 0:
            sentence_count = content.count('.') + content.count('!') + content.count('?')
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Score based on reasonable sentence length (10-30 words)
            if 10 <= avg_sentence_length <= 30:
                scores['content_quality_score'] = min(1.0, word_count / 1000)
            else:
                scores['content_quality_score'] = max(0.3, min(1.0, word_count / 1000)) * 0.8
        
        # Entity density score
        if word_count > 0:
            entity_density = len(entities) / word_count
            scores['entity_density_score'] = min(1.0, entity_density * 100)  # Normalize
        
        return scores


class FilterBuilder:
    """Build search filters for Qdrant queries"""
    
    @staticmethod
    def build_domain_filter(domain: str) -> Dict:
        """Build filter for specific domain"""
        return {
            "must": [
                {"key": "domain", "match": {"value": domain}}
            ]
        }
    
    @staticmethod
    def build_date_range_filter(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """Build filter for date range"""
        conditions = []
        
        if start_date:
            conditions.append({
                "key": "date",
                "range": {"gte": start_date}
            })
        
        if end_date:
            conditions.append({
                "key": "date",
                "range": {"lte": end_date}
            })
        
        return {"must": conditions} if conditions else {}
    
    @staticmethod
    def build_language_filter(language: str) -> Dict:
        """Build filter for specific language"""
        return {
            "must": [
                {"key": "language", "match": {"value": language}}
            ]
        }
    
    @staticmethod
    def build_author_filter(author: str) -> Dict:
        """Build filter for specific author"""
        return {
            "must": [
                {"key": "author", "match": {"value": author}}
            ]
        }
    
    @staticmethod
    def build_quality_filter(min_quality_score: float = 0.5) -> Dict:
        """Build filter for minimum quality score"""
        return {
            "must": [
                {"key": "content_quality_score", "range": {"gte": min_quality_score}}
            ]
        }
    
    @staticmethod
    def build_type_filter(content_type: str = "document") -> Dict:
        """Build filter for content type (document or chunk)"""
        return {
            "must": [
                {"key": "type", "match": {"value": content_type}}
            ]
        }
    
    @staticmethod
    def build_complex_filter(
        domain: Optional[str] = None,
        language: Optional[str] = None,
        author: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_quality: Optional[float] = None,
        content_type: Optional[str] = None
    ) -> Dict:
        """Build complex filter combining multiple conditions"""
        conditions = []
        
        if domain:
            conditions.append({"key": "domain", "match": {"value": domain}})
        
        if language:
            conditions.append({"key": "language", "match": {"value": language}})
        
        if author:
            conditions.append({"key": "author", "match": {"value": author}})
        
        if start_date:
            conditions.append({"key": "date", "range": {"gte": start_date}})
        
        if end_date:
            conditions.append({"key": "date", "range": {"lte": end_date}})
        
        if min_quality:
            conditions.append({"key": "content_quality_score", "range": {"gte": min_quality}})
        
        if content_type:
            conditions.append({"key": "type", "match": {"value": content_type}})
        
        return {"must": conditions} if conditions else {}


class PerformanceMonitor:
    """Monitor vector indexing and search performance"""
    
    def __init__(self):
        self.metrics = {
            'indexing_times': [],
            'search_times': [],
            'memory_usage': [],
            'error_count': 0,
            'success_count': 0
        }
    
    def record_indexing_time(self, duration: float) -> None:
        """Record indexing operation duration"""
        self.metrics['indexing_times'].append(duration)
    
    def record_search_time(self, duration: float) -> None:
        """Record search operation duration"""
        self.metrics['search_times'].append(duration)
    
    def record_success(self) -> None:
        """Record successful operation"""
        self.metrics['success_count'] += 1
    
    def record_error(self) -> None:
        """Record failed operation"""
        self.metrics['error_count'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_operations': self.metrics['success_count'] + self.metrics['error_count'],
            'success_rate': self.metrics['success_count'] / max(1, self.metrics['success_count'] + self.metrics['error_count']),
            'error_rate': self.metrics['error_count'] / max(1, self.metrics['success_count'] + self.metrics['error_count'])
        }
        
        if self.metrics['indexing_times']:
            stats['avg_indexing_time'] = np.mean(self.metrics['indexing_times'])
            stats['max_indexing_time'] = np.max(self.metrics['indexing_times'])
            stats['min_indexing_time'] = np.min(self.metrics['indexing_times'])
        
        if self.metrics['search_times']:
            stats['avg_search_time'] = np.mean(self.metrics['search_times'])
            stats['max_search_time'] = np.max(self.metrics['search_times'])
            stats['min_search_time'] = np.min(self.metrics['search_times'])
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics = {
            'indexing_times': [],
            'search_times': [],
            'memory_usage': [],
            'error_count': 0,
            'success_count': 0
        }


def create_vector_config_file():
    """Create the vector indexer configuration file"""
    config_path = Path("agents/vector_index/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(VECTOR_INDEX_CONFIG, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created vector indexer config: {config_path}")


def validate_qdrant_connection(host: str = "localhost", port: int = 6333) -> bool:
    """Validate Qdrant connection"""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=host, port=port, timeout=10)
        collections = client.get_collections()
        print(f"✅ Qdrant connection successful. Collections: {len(collections.collections)}")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False


def validate_redis_connection(redis_url: str = "redis://localhost:6379") -> bool:
    """Validate Redis connection"""
    try:
        import redis
        
        client = redis.from_url(redis_url, socket_timeout=10)
        client.ping()
        print("✅ Redis connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


if __name__ == "__main__":
    create_vector_config_file()
    validate_qdrant_connection()
    validate_redis_connection()
