#!/usr/bin/env python3
"""
Enhanced Vector Indexer - Main Implementation
Core enhanced indexing functionality with dynamic model selection and quality checking
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import yaml
import numpy as np

from .models import VectorIndexResult, EmbeddingQuality
from .model_manager import ModelManager
from .quality_checker import EmbeddingQualityChecker
from .utils import (
    BASE_INDEXER_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    PerformanceTimer,
    create_mock_embedding,
    validate_embedding
)

# Import base functionality if available
if BASE_INDEXER_AVAILABLE:
    from .vector_indexer import VectorIndexer, VectorDocument, SearchResult, QdrantManager, RedisCache

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedVectorIndexer:
    """Enhanced vector indexer with dynamic model selection and quality checking"""
    
    def __init__(self, config_path: str = "agents/qdrant_manager/vector_index/enhanced_config.yaml"):
        """Initialize enhanced vector indexer"""
        self.config = self._load_config(config_path)
        
        # Initialize enhanced components
        self.model_manager = ModelManager()
        self.quality_checker = EmbeddingQualityChecker()
        
        # Initialize base indexer if available
        if BASE_INDEXER_AVAILABLE:
            try:
                self.base_indexer = VectorIndexer(config_path)
                self.qdrant = self.base_indexer.qdrant
                self.cache = self.base_indexer.cache
                logger.info("Base vector indexer integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize base indexer: {e}")
                self._setup_fallback_components()
        else:
            self._setup_fallback_components()
        
        # Initialize thread executor
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('performance', {}).get('max_workers', 4)
        )
        
        logger.info("Enhanced Vector Indexer initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enhanced configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enhanced_features': {
                'dynamic_model_selection': True,
                'quality_checking': True,
                'performance_monitoring': True
            },
            'quality_thresholds': {
                'minimum_quality_score': 0.6,
                'reindex_threshold': 0.4
            },
            'performance': {
                'batch_size': 50,
                'max_workers': 4,
                'cache_ttl': 7200
            }
        }
    
    def _setup_fallback_components(self) -> None:
        """Setup fallback components when base indexer unavailable"""
        logger.warning("Setting up fallback components - limited functionality")
        self.base_indexer = None
        self.qdrant = None
        self.cache = None
    
    async def index_content_enhanced(
        self, 
        content: Dict[str, Any], 
        force_model: Optional[str] = None
    ) -> VectorIndexResult:
        """Enhanced content indexing with dynamic model selection and quality checking"""
        with PerformanceTimer("Enhanced indexing") as timer:
            try:
                # Validate input
                if not content.get('text'):
                    raise ValueError("No text content provided for indexing")
                
                # Select optimal model
                selected_model = self._select_model(content, force_model)
                logger.debug(f"Selected model '{selected_model}' for content: {content.get('title', 'untitled')}")
                
                # Generate embedding
                embedding = await self._generate_embedding(content, selected_model)
                if embedding is None:
                    raise ValueError("Failed to generate embedding")
                
                # Quality assessment
                quality_assessment = self.quality_checker.assess_quality(
                    embedding, content, selected_model
                )
                
                # Check if quality meets threshold and try alternative if needed
                if self._should_try_alternative_model(quality_assessment):
                    alternative_model = self.model_manager.get_alternative_model(selected_model)
                    if alternative_model != selected_model:
                        logger.info(f"Trying alternative model '{alternative_model}' for quality improvement")
                        alt_embedding = await self._generate_embedding(content, alternative_model)
                        if alt_embedding is not None:
                            alt_quality = self.quality_checker.assess_quality(
                                alt_embedding, content, alternative_model
                            )
                            # Use alternative if significantly better
                            if alt_quality.quality_score > quality_assessment.quality_score + 0.1:
                                embedding = alt_embedding
                                quality_assessment = alt_quality
                                selected_model = alternative_model
                                logger.info(f"Used alternative model with improved quality: {alt_quality.quality_score:.3f}")
                
                # Index using base indexer if available
                if self.base_indexer:
                    indexing_success = await self._index_with_base_indexer(content, embedding)
                else:
                    indexing_success = True  # Mock success for fallback
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(quality_assessment, selected_model)
                
                # Create result
                result = VectorIndexResult(
                    vector_id=content.get('doc_id', str(uuid.uuid4())),
                    embedding=embedding,
                    model_used=selected_model,
                    quality_assessment=quality_assessment,
                    metadata={
                        **content.get('metadata', {}),
                        'model_used': selected_model,
                        'quality_score': quality_assessment.quality_score,
                        'indexed_at': datetime.now().isoformat(),
                        'enhanced_indexing': True,
                        'indexing_success': indexing_success
                    },
                    processing_time=timer.get_duration(),
                    confidence_score=confidence_score
                )
                
                logger.info(
                    f"Enhanced indexing completed: quality={quality_assessment.quality_score:.3f}, "
                    f"confidence={confidence_score:.3f}, time={timer.get_duration():.2f}s"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Enhanced indexing failed: {e}")
                # Return failed result instead of raising
                return VectorIndexResult(
                    vector_id=content.get('doc_id', 'failed'),
                    embedding=np.array([]),
                    model_used='none',
                    quality_assessment=EmbeddingQuality(
                        doc_id=content.get('doc_id', 'failed'),
                        model_used='none',
                        quality_score=0.0,
                        consistency_score=0.0,
                        semantic_density=0.0,
                        outlier_score=0.0,
                        recommendations=[f"Indexing failed: {str(e)}"]
                    ),
                    metadata={'error': str(e)},
                    processing_time=timer.get_duration(),
                    confidence_score=0.0
                )
    
    def _select_model(self, content: Dict[str, Any], force_model: Optional[str]) -> str:
        """Select appropriate model for content"""
        if force_model and force_model in self.model_manager.models:
            return force_model
        
        if self.config.get('enhanced_features', {}).get('dynamic_model_selection', True):
            return self.model_manager.select_optimal_model(content)
        
        # Fallback to first available model
        available_models = list(self.model_manager.models.keys())
        return available_models[0] if available_models else 'general'
    
    async def _generate_embedding(self, content: Dict[str, Any], model_key: str) -> Optional[np.ndarray]:
        """Generate embedding using specified model"""
        text = content.get('text', '')
        if not text:
            logger.warning("No text content provided for embedding")
            return None
        
        model = self.model_manager.get_model(model_key)
        
        # Handle mock models
        if model == 'mock_general' or not TRANSFORMERS_AVAILABLE:
            return create_mock_embedding()
        
        try:
            # Run embedding generation in thread pool for async
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._generate_embedding_sync,
                model, text
            )
            
            if embedding is not None and validate_embedding(embedding):
                return embedding
            else:
                logger.warning(f"Generated invalid embedding with model {model_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding with model {model_key}: {e}")
            return None
    
    def _generate_embedding_sync(self, model, text: str) -> Optional[np.ndarray]:
        """Synchronous embedding generation"""
        try:
            if hasattr(model, 'encode'):
                embedding = model.encode([text], show_progress_bar=False)[0]
                return np.array(embedding)
            else:
                return create_mock_embedding()
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            return None
    
    def _should_try_alternative_model(self, quality_assessment: EmbeddingQuality) -> bool:
        """Determine if alternative model should be tried"""
        min_threshold = self.config.get('quality_thresholds', {}).get('minimum_quality_score', 0.6)
        return quality_assessment.quality_score < min_threshold
    
    async def _index_with_base_indexer(self, content: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Index using base indexer if available"""
        try:
            # Prepare content for base indexer
            indexer_content = {
                **content,
                'embeddings': embedding.tolist()
            }
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self.base_indexer.index_document,
                indexer_content
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Base indexer integration failed: {e}")
            return False
    
    def _calculate_confidence_score(self, quality_assessment: EmbeddingQuality, model_used: str) -> float:
        """Calculate overall confidence score"""
        model_performance = self.model_manager.model_performances.get(model_used)
        
        if not model_performance:
            return quality_assessment.quality_score * 0.8
        
        # Weighted combination of factors
        confidence = (
            quality_assessment.quality_score * 0.4 +
            quality_assessment.consistency_score * 0.2 +
            quality_assessment.semantic_density * 0.2 +
            model_performance.accuracy * 0.2
        )
        
        # Penalty for outliers
        if quality_assessment.outlier_score > 2.0:
            confidence *= 0.9
        
        return min(1.0, max(0.0, confidence))
    
    async def search_enhanced(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
        quality_threshold: Optional[float] = None,
        model_preference: Optional[str] = None
    ) -> List:
        """Enhanced search with quality filtering"""
        try:
            # Select model for query embedding
            query_content = {
                'text': query, 
                'domain': domain, 
                'type': 'query'
            }
            
            query_model = (
                model_preference if model_preference in self.model_manager.models
                else self.model_manager.select_optimal_model(query_content)
            )
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query_content, query_model)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform search using base indexer
            if self.base_indexer and hasattr(self.base_indexer, 'search'):
                results = self.base_indexer.search(
                    query=query,
                    query_embedding=query_embedding.tolist(),
                    domain=domain,
                    limit=limit * 2  # Get more results to filter by quality
                )
                
                # Filter by quality if threshold specified
                if quality_threshold is not None:
                    filtered_results = []
                    for result in results:
                        result_quality = result.metadata.get('quality_score', 0.5)
                        if result_quality >= quality_threshold:
                            filtered_results.append(result)
                    results = filtered_results[:limit]
                
                return results
            else:
                logger.warning("Base indexer search not available")
                return []
                
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    async def batch_process_enhanced(
        self, 
        content_list: List[Dict[str, Any]], 
        progress_callback=None
    ) -> List[VectorIndexResult]:
        """Enhanced batch processing with progress tracking"""
        results = []
        total = len(content_list)
        
        logger.info(f"Starting batch processing of {total} items")
        
        # Process with concurrency limit
        batch_size = self.config.get('performance', {}).get('batch_size', 10)
        
        for i in range(0, total, batch_size):
            batch = content_list[i:i + batch_size]
            batch_tasks = [
                self.index_content_enhanced(content) 
                for content in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch item {i+j} failed: {result}")
                        # Create failed result
                        failed_result = VectorIndexResult(
                            vector_id=f"failed_{i+j}",
                            embedding=np.array([]),
                            model_used='none',
                            quality_assessment=EmbeddingQuality(
                                doc_id=f"failed_{i+j}",
                                model_used='none',
                                quality_score=0.0,
                                consistency_score=0.0,
                                semantic_density=0.0,
                                outlier_score=0.0,
                                recommendations=[f"Processing failed: {str(result)}"]
                            ),
                            metadata={'error': str(result)},
                            processing_time=0.0,
                            confidence_score=0.0
                        )
                        results.append(failed_result)
                    else:
                        results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        min(i + batch_size, total), 
                        total, 
                        f"Processed batch {(i // batch_size) + 1}"
                    )
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                continue
        
        successful = sum(1 for r in results if r.confidence_score > 0.5)
        logger.info(f"Batch processing completed: {successful}/{total} successful")
        
        return results
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        return self.model_manager.get_model_stats()
    
    def optimize_model_selection(self, feedback_data: List[Dict[str, Any]]) -> None:
        """Optimize model selection based on user feedback"""
        logger.info("Optimizing model selection based on feedback...")
        
        for feedback in feedback_data:
            model_used = feedback.get('model_used')
            user_rating = feedback.get('rating', 0.5)
            
            if model_used:
                self.model_manager.update_model_performance(model_used, user_rating)
        
        logger.info("Model optimization completed")
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics across domains"""
        stats = {}
        
        from .utils import SUPPORTED_DOMAINS
        for domain in SUPPORTED_DOMAINS:
            domain_stats = self.quality_checker.get_domain_statistics(domain)
            if 'error' not in domain_stats:
                stats[domain] = domain_stats
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if hasattr(self, 'base_indexer') and self.base_indexer:
            self.base_indexer.cleanup()
        
        # Clear quality checker references to free memory
        self.quality_checker.clear_references()
        
        logger.info("Enhanced Vector Indexer cleanup completed")


# Example usage and testing
async def main():
    """Example usage of enhanced vector indexer"""
    # Test content
    test_content = {
        'doc_id': 'enhanced_test_456',
        'title': 'Advanced Mathematical Concepts',
        'text': 'This document explores complex mathematical theories and their applications in modern computational systems.',
        'domain': 'mathematics',
        'language': 'en',
        'type': 'academic',
        'metadata': {
            'author': 'Test Author',
            'date': '2024-01-01',
            'source': 'Test Collection'
        }
    }
    
    # Initialize enhanced indexer
    indexer = EnhancedVectorIndexer()
    
    try:
        logger.info("Testing enhanced vector indexing...")
        
        # Test enhanced indexing
        result = await indexer.index_content_enhanced(test_content)
        
        print(f"\nEnhanced Indexing Result:")
        print(f"- Vector ID: {result.vector_id}")
        print(f"- Model used: {result.model_used}")
        print(f"- Quality score: {result.quality_assessment.quality_score:.3f}")
        print(f"- Confidence: {result.confidence_score:.3f}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        print(f"- Success: {result.is_successful()}")
        
        if result.quality_assessment.recommendations:
            print("- Recommendations:")
            for rec in result.quality_assessment.recommendations:
                print(f"  * {rec}")
        
        # Test batch processing
        print(f"\nTesting batch processing...")
        batch_content = [test_content.copy() for _ in range(3)]
        for i, content in enumerate(batch_content):
            content['doc_id'] = f'batch_test_{i}'
            content['title'] = f'Batch Test Document {i}'
        
        batch_results = await indexer.batch_process_enhanced(batch_content)
        print(f"Batch processing: {len(batch_results)} results")
        
        # Show model performance stats
        stats = indexer.get_model_performance_stats()
        print(f"\nModel Performance Statistics:")
        for model, perf in stats.items():
            print(f"- {model}: quality={perf['quality_score']:.2f}, speed={perf['speed']:.1f} docs/sec")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        indexer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())