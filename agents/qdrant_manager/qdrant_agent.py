#!/usr/bin/env python3
"""
Qdrant Manager Agent
Centralized Qdrant vector database operations with collection management and optimization
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import yaml
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, SearchRequest, UpdateStatus
)

from .collection_manager import CollectionManager
from .vector_optimizer import VectorOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VectorOperationResult:
    """Result of a vector operation"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    points_affected: int = 0
    operation_id: Optional[str] = None

@dataclass
class VectorPoint:
    """Standardized vector point structure"""
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any]

@dataclass
class SearchResult:
    """Vector search result"""
    id: Union[str, int]
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

class QdrantAgent:
    """
    Centralized Qdrant vector database operations agent for MCP Yggdrasil
    Provides vector CRUD operations, semantic search, and collection management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Qdrant Agent with configuration"""
        self.config = self._load_config(config_path)
        self.client: Optional[QdrantClient] = None
        self.collection_manager = CollectionManager(self.config)
        self.vector_optimizer = VectorOptimizer(self.config)
        self._metrics = {
            "operations_total": 0,
            "operations_success": 0,
            "operations_failed": 0,
            "points_total": 0,
            "searches_total": 0,
            "avg_search_time": 0.0,
            "collections_managed": 0
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('qdrant_agent', {})
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "connection": {
                "host": "localhost",
                "port": 6333,
                "timeout": 30
            },
            "collections": {
                "auto_create": True,
                "default_vector_size": 384,
                "default_distance": "Cosine"
            },
            "performance": {
                "batch_size": 100,
                "max_retries": 3
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the Qdrant connection and setup collections"""
        try:
            # Create client
            self.client = QdrantClient(
                host=self.config["connection"]["host"],
                port=self.config["connection"]["port"],
                timeout=self.config["connection"]["timeout"],
                prefer_grpc=self.config["connection"].get("prefer_grpc", False),
                api_key=self.config["connection"].get("api_key")
            )
            
            # Verify connection
            collections = self.client.get_collections()
            logger.info(f"Qdrant connection established. Found {len(collections.collections)} collections")
            
            # Initialize collections
            if self.config["collections"]["auto_create"]:
                await self.collection_manager.initialize_collections(self.client)
            
            # Update metrics
            self._metrics["collections_managed"] = len(collections.collections)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant Agent: {e}")
            return False
    
    def close(self):
        """Close the Qdrant connection"""
        if self.client:
            self.client.close()
            logger.info("Qdrant connection closed")
    
    async def upsert_point(self, collection_name: str, point: VectorPoint) -> VectorOperationResult:
        """Insert or update a vector point"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            # Validate vector
            if not self._validate_vector(point.vector, collection_name):
                return VectorOperationResult(
                    success=False,
                    error="Vector validation failed",
                    execution_time=time.time() - start_time,
                    operation_id=operation_id
                )
            
            # Optimize vector if enabled
            if self.config["optimization"]["enable_compression"]:
                point.vector = await self.vector_optimizer.optimize_vector(point.vector)
            
            # Add metadata to payload
            enhanced_payload = point.payload.copy()
            enhanced_payload.update({
                "created_at": datetime.now().isoformat(),
                "agent_created": "qdrant_agent",
                "operation_id": operation_id,
                "vector_size": len(point.vector)
            })
            
            # Create point
            qdrant_point = PointStruct(
                id=point.id,
                vector=point.vector,
                payload=enhanced_payload
            )
            
            # Upsert to collection
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=[qdrant_point],
                wait=True
            )
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(True, execution_time, points_affected=1)
            
            return VectorOperationResult(
                success=True,
                data={
                    "point_id": point.id,
                    "operation_info": operation_info.dict() if hasattr(operation_info, 'dict') else str(operation_info)
                },
                execution_time=execution_time,
                points_affected=1,
                operation_id=operation_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to upsert point: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_id=operation_id
            )
    
    async def get_point(self, collection_name: str, point_id: Union[str, int]) -> VectorOperationResult:
        """Retrieve a vector point by ID"""
        start_time = time.time()
        
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            execution_time = time.time() - start_time
            
            if points:
                point = points[0]
                self._update_metrics(True, execution_time)
                return VectorOperationResult(
                    success=True,
                    data={
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload
                    },
                    execution_time=execution_time,
                    points_affected=1
                )
            else:
                return VectorOperationResult(
                    success=False,
                    error="Point not found",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to get point: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def delete_point(self, collection_name: str, point_id: Union[str, int]) -> VectorOperationResult:
        """Delete a vector point"""
        start_time = time.time()
        
        try:
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                ),
                wait=True
            )
            
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time, points_affected=1)
            
            return VectorOperationResult(
                success=True,
                data={"operation_info": operation_info.dict() if hasattr(operation_info, 'dict') else str(operation_info)},
                execution_time=execution_time,
                points_affected=1
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to delete point: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def search_vectors(
        self, 
        collection_name: str, 
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> VectorOperationResult:
        """Perform semantic vector search"""
        start_time = time.time()
        
        try:
            # Validate query vector
            if not self._validate_vector(query_vector, collection_name):
                return VectorOperationResult(
                    success=False,
                    error="Query vector validation failed",
                    execution_time=time.time() - start_time
                )
            
            # Build filter
            search_filter = None
            if filter_conditions:
                search_filter = self._build_filter(filter_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            execution_time = time.time() - start_time
            
            # Convert to standardized format
            results = []
            for result in search_results:
                results.append(SearchResult(
                    id=result.id,
                    score=result.score,
                    payload=result.payload or {}
                ))
            
            # Update metrics
            self._update_metrics(True, execution_time, searches=1)
            
            return VectorOperationResult(
                success=True,
                data={
                    "results": results,
                    "total_results": len(results),
                    "search_params": {
                        "limit": limit,
                        "score_threshold": score_threshold,
                        "filter_applied": filter_conditions is not None
                    }
                },
                execution_time=execution_time,
                points_affected=len(results)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Vector search failed: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def batch_upsert(self, collection_name: str, points: List[VectorPoint]) -> VectorOperationResult:
        """Batch insert/update multiple vector points"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            batch_size = self.config["performance"]["batch_size"]
            total_points = len(points)
            processed_points = 0
            
            # Process in batches
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                
                # Prepare Qdrant points
                qdrant_points = []
                for point in batch:
                    # Validate vector
                    if not self._validate_vector(point.vector, collection_name):
                        logger.warning(f"Skipping invalid vector for point {point.id}")
                        continue
                    
                    # Enhance payload
                    enhanced_payload = point.payload.copy()
                    enhanced_payload.update({
                        "created_at": datetime.now().isoformat(),
                        "agent_created": "qdrant_agent",
                        "batch_operation_id": operation_id
                    })
                    
                    qdrant_points.append(PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=enhanced_payload
                    ))
                
                # Upsert batch
                if qdrant_points:
                    operation_info = self.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points,
                        wait=True
                    )
                    processed_points += len(qdrant_points)
            
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time, points_affected=processed_points)
            
            return VectorOperationResult(
                success=True,
                data={
                    "processed_points": processed_points,
                    "total_points": total_points,
                    "batches_processed": (total_points + batch_size - 1) // batch_size
                },
                execution_time=execution_time,
                points_affected=processed_points,
                operation_id=operation_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Batch upsert failed: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                operation_id=operation_id
            )
    
    async def similarity_search_by_id(
        self, 
        collection_name: str, 
        point_id: Union[str, int],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> VectorOperationResult:
        """Find similar vectors to a given point ID"""
        start_time = time.time()
        
        try:
            # First get the point to extract its vector
            point_result = await self.get_point(collection_name, point_id)
            if not point_result.success:
                return VectorOperationResult(
                    success=False,
                    error=f"Source point not found: {point_id}",
                    execution_time=time.time() - start_time
                )
            
            query_vector = point_result.data["vector"]
            
            # Perform similarity search
            search_result = await self.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit + 1,  # +1 to exclude the source point
                filter_conditions=filter_conditions
            )
            
            if search_result.success:
                # Filter out the source point from results
                results = [
                    result for result in search_result.data["results"] 
                    if result.id != point_id
                ][:limit]
                
                search_result.data["results"] = results
                search_result.data["total_results"] = len(results)
                search_result.data["source_point_id"] = point_id
            
            return search_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Similarity search by ID failed: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def get_collection_info(self, collection_name: str) -> VectorOperationResult:
        """Get information about a collection"""
        start_time = time.time()
        
        try:
            collection_info = self.client.get_collection(collection_name)
            
            # Get additional statistics
            collection_stats = {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": collection_info.config.dict() if hasattr(collection_info.config, 'dict') else str(collection_info.config),
                "status": collection_info.status
            }
            
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time)
            
            return VectorOperationResult(
                success=True,
                data=collection_stats,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to get collection info: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _validate_vector(self, vector: List[float], collection_name: str) -> bool:
        """Validate vector format and dimensions"""
        try:
            # Check if vector is list/array of numbers
            if not isinstance(vector, (list, np.ndarray)):
                return False
            
            # Convert to list if numpy array
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            # Check if all elements are numbers
            if not all(isinstance(x, (int, float)) for x in vector):
                return False
            
            # Check vector dimensions
            expected_size = self._get_collection_vector_size(collection_name)
            if expected_size and len(vector) != expected_size:
                logger.warning(f"Vector size mismatch: expected {expected_size}, got {len(vector)}")
                return False
            
            # Check for NaN or infinity values
            if any(np.isnan(x) or np.isinf(x) for x in vector):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Vector validation error: {e}")
            return False
    
    def _get_collection_vector_size(self, collection_name: str) -> Optional[int]:
        """Get expected vector size for a collection"""
        domain_collections = self.config.get("collections", {}).get("domain_collections", {})
        
        if collection_name in domain_collections:
            return domain_collections[collection_name]["vector_size"]
        
        return self.config.get("collections", {}).get("default_vector_size", 384)
    
    def _build_filter(self, filter_conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions"""
        conditions = []
        
        for field, value in filter_conditions.items():
            if isinstance(value, dict):
                # Handle complex conditions
                if "match" in value:
                    conditions.append(
                        FieldCondition(key=field, match=models.MatchValue(value=value["match"]))
                    )
                elif "range" in value:
                    range_val = value["range"]
                    conditions.append(
                        FieldCondition(
                            key=field, 
                            range=models.Range(
                                gte=range_val.get("gte"),
                                lte=range_val.get("lte"),
                                gt=range_val.get("gt"),
                                lt=range_val.get("lt")
                            )
                        )
                    )
            else:
                # Simple equality match
                conditions.append(
                    FieldCondition(key=field, match=models.MatchValue(value=value))
                )
        
        return Filter(must=conditions)
    
    def _update_metrics(self, success: bool, execution_time: float, points_affected: int = 0, searches: int = 0):
        """Update performance metrics"""
        self._metrics["operations_total"] += 1
        
        if success:
            self._metrics["operations_success"] += 1
        else:
            self._metrics["operations_failed"] += 1
        
        if points_affected:
            self._metrics["points_total"] += points_affected
        
        if searches:
            self._metrics["searches_total"] += searches
            # Update average search time
            total_searches = self._metrics["searches_total"]
            current_avg = self._metrics["avg_search_time"]
            self._metrics["avg_search_time"] = (
                (current_avg * (total_searches - 1) + execution_time) / total_searches
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "No client connection"}
            
            # Test basic connectivity
            collections = self.client.get_collections()
            
            # Get cluster info if available
            try:
                cluster_info = self.client.cluster_info()
                cluster_status = cluster_info.status if hasattr(cluster_info, 'status') else 'unknown'
            except:
                cluster_status = 'standalone'
            
            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "cluster_status": cluster_status,
                "metrics": self.get_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_collection(self, collection_name: str) -> VectorOperationResult:
        """Optimize a collection for better performance"""
        start_time = time.time()
        
        try:
            # Get collection info
            info_result = await self.get_collection_info(collection_name)
            if not info_result.success:
                return info_result
            
            collection_info = info_result.data
            
            # Check if optimization is needed
            if collection_info["points_count"] < self.config["optimization"]["auto_optimize_threshold"]:
                return VectorOperationResult(
                    success=True,
                    data={"message": "Collection too small for optimization"},
                    execution_time=time.time() - start_time
                )
            
            # Perform optimization using vector optimizer
            optimization_result = await self.vector_optimizer.optimize_collection(
                self.client, collection_name, collection_info
            )
            
            execution_time = time.time() - start_time
            
            if optimization_result:
                self._update_metrics(True, execution_time)
                return VectorOperationResult(
                    success=True,
                    data=optimization_result,
                    execution_time=execution_time
                )
            else:
                return VectorOperationResult(
                    success=False,
                    error="Optimization failed",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Collection optimization failed: {e}")
            return VectorOperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )