#!/usr/bin/env python3
"""
Collection Manager for Qdrant Agent
Handles domain-specific collection creation and management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, 
    HnswConfigDiff, OptimizersConfigDiff, CollectionInfo
)

logger = logging.getLogger(__name__)

class CollectionManager:
    """
    Manages Qdrant collections for domain-specific vector storage
    Handles creation, configuration, and optimization of collections
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_collections = self._get_collection_configs()
        
    def _get_collection_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get collection configurations from config"""
        return self.config.get("collections", {}).get("domain_collections", {})
    
    async def initialize_collections(self, client: QdrantClient) -> bool:
        """Initialize all domain-specific collections"""
        try:
            existing_collections = client.get_collections()
            existing_names = {col.name for col in existing_collections.collections}
            
            created_count = 0
            for collection_name, collection_config in self.domain_collections.items():
                if collection_name not in existing_names:
                    success = await self._create_collection(client, collection_name, collection_config)
                    if success:
                        created_count += 1
                        logger.info(f"Created collection: {collection_name}")
                    else:
                        logger.error(f"Failed to create collection: {collection_name}")
                else:
                    logger.debug(f"Collection already exists: {collection_name}")
            
            logger.info(f"Collection initialization completed. Created {created_count} new collections")
            return True
            
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            return False
    
    async def _create_collection(
        self, 
        client: QdrantClient, 
        collection_name: str, 
        config: Dict[str, Any]
    ) -> bool:
        """Create a single collection with specified configuration"""
        try:
            # Extract configuration
            vector_size = config.get("vector_size", 384)
            distance = config.get("distance", "Cosine")
            hnsw_config = config.get("hnsw_config", {})
            
            # Map distance string to enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            distance_enum = distance_map.get(distance, Distance.COSINE)
            
            # Create HNSW configuration
            hnsw_config_obj = HnswConfigDiff(
                m=hnsw_config.get("m", 16),
                ef_construct=hnsw_config.get("ef_construct", 200),
                full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000),
                max_indexing_threads=hnsw_config.get("max_indexing_threads", 0)
            )
            
            # Create optimizers configuration
            optimizers_config = OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                max_segment_size=None,
                memmap_threshold=None,
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=1
            )
            
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_enum,
                    hnsw_config=hnsw_config_obj
                ),
                optimizers_config=optimizers_config,
                replication_factor=1,
                write_consistency_factor=1,
                timeout=60
            )
            
            # Create indexes for common payload fields
            await self._create_payload_indexes(client, collection_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def _create_payload_indexes(self, client: QdrantClient, collection_name: str):
        """Create indexes for common payload fields"""
        try:
            # Common indexes for all collections
            indexes = [
                ("domain", models.PayloadSchemaType.KEYWORD),
                ("created_at", models.PayloadSchemaType.DATETIME),
                ("agent_created", models.PayloadSchemaType.KEYWORD),
            ]
            
            # Domain-specific indexes
            if collection_name.startswith("documents_"):
                indexes.extend([
                    ("title", models.PayloadSchemaType.TEXT),
                    ("author", models.PayloadSchemaType.KEYWORD),
                    ("timestamp", models.PayloadSchemaType.DATETIME),
                    ("content_hash", models.PayloadSchemaType.KEYWORD),
                    ("language", models.PayloadSchemaType.KEYWORD)
                ])
            elif collection_name == "entities":
                indexes.extend([
                    ("name", models.PayloadSchemaType.TEXT),
                    ("type", models.PayloadSchemaType.KEYWORD),
                    ("confidence", models.PayloadSchemaType.FLOAT)
                ])
            elif collection_name == "concepts":
                indexes.extend([
                    ("concept_id", models.PayloadSchemaType.KEYWORD),
                    ("name", models.PayloadSchemaType.TEXT),
                    ("type", models.PayloadSchemaType.KEYWORD),
                    ("level", models.PayloadSchemaType.INTEGER)
                ])
            elif collection_name == "claims":
                indexes.extend([
                    ("text", models.PayloadSchemaType.TEXT),
                    ("verified", models.PayloadSchemaType.BOOL),
                    ("confidence", models.PayloadSchemaType.FLOAT)
                ])
            
            # Create each index
            for field_name, field_type in indexes:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                        wait=True
                    )
                    logger.debug(f"Created index for {collection_name}.{field_name}")
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation skipped for {collection_name}.{field_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to create payload indexes for {collection_name}: {e}")
    
    def get_collection_for_domain(self, domain: str) -> str:
        """Get the appropriate collection name for a domain"""
        domain_lower = domain.lower()
        
        # Map domains to collections
        domain_mapping = {
            "art": "documents_art",
            "language": "documents_language", 
            "mathematics": "documents_mathematics",
            "philosophy": "documents_philosophy",
            "science": "documents_science",
            "technology": "documents_technology",
            "religion": "documents_religion",
            "astrology": "documents_astrology"
        }
        
        return domain_mapping.get(domain_lower, "documents_general")
    
    def get_collection_for_content_type(self, content_type: str) -> str:
        """Get the appropriate collection name for a content type"""
        content_type_lower = content_type.lower()
        
        type_mapping = {
            "document": "documents_general",
            "entity": "entities",
            "concept": "concepts",
            "claim": "claims"
        }
        
        return type_mapping.get(content_type_lower, "documents_general")
    
    async def ensure_collection_exists(self, client: QdrantClient, collection_name: str) -> bool:
        """Ensure a collection exists, create if it doesn't"""
        try:
            # Check if collection exists
            collections = client.get_collections()
            existing_names = {col.name for col in collections.collections}
            
            if collection_name in existing_names:
                return True
            
            # Collection doesn't exist, create it
            if collection_name in self.domain_collections:
                # Use predefined configuration
                config = self.domain_collections[collection_name]
                return await self._create_collection(client, collection_name, config)
            else:
                # Use default configuration
                default_config = {
                    "vector_size": self.config.get("collections", {}).get("default_vector_size", 384),
                    "distance": self.config.get("collections", {}).get("default_distance", "Cosine"),
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 200
                    }
                }
                return await self._create_collection(client, collection_name, default_config)
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists {collection_name}: {e}")
            return False
    
    async def delete_collection(self, client: QdrantClient, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            client.delete_collection(collection_name=collection_name, timeout=60)
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def recreate_collection(self, client: QdrantClient, collection_name: str) -> bool:
        """Recreate a collection with fresh configuration"""
        try:
            # Delete existing collection
            await self.delete_collection(client, collection_name)
            
            # Recreate with current configuration
            if collection_name in self.domain_collections:
                config = self.domain_collections[collection_name]
                return await self._create_collection(client, collection_name, config)
            else:
                logger.error(f"No configuration found for collection: {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to recreate collection {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, client: QdrantClient) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            collections = client.get_collections()
            stats = {
                "total_collections": len(collections.collections),
                "collections": {}
            }
            
            for collection in collections.collections:
                try:
                    info = client.get_collection(collection.name)
                    stats["collections"][collection.name] = {
                        "vectors_count": info.vectors_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                        "points_count": info.points_count,
                        "segments_count": info.segments_count,
                        "status": info.status,
                        "optimizer_status": info.optimizer_status.dict() if hasattr(info, 'optimizer_status') else None
                    }
                except Exception as e:
                    stats["collections"][collection.name] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def validate_collection_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate collection configuration"""
        errors = []
        
        # Check required fields
        required_fields = ["vector_size", "distance"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate vector size
        vector_size = config.get("vector_size")
        if vector_size and (not isinstance(vector_size, int) or vector_size <= 0):
            errors.append("vector_size must be a positive integer")
        
        # Validate distance metric
        distance = config.get("distance")
        valid_distances = ["Cosine", "Euclidean", "Dot"]
        if distance and distance not in valid_distances:
            errors.append(f"distance must be one of: {valid_distances}")
        
        # Validate HNSW config
        hnsw_config = config.get("hnsw_config", {})
        if hnsw_config:
            if "m" in hnsw_config:
                m = hnsw_config["m"]
                if not isinstance(m, int) or m < 4 or m > 64:
                    errors.append("hnsw_config.m must be an integer between 4 and 64")
            
            if "ef_construct" in hnsw_config:
                ef_construct = hnsw_config["ef_construct"]
                if not isinstance(ef_construct, int) or ef_construct < 10:
                    errors.append("hnsw_config.ef_construct must be an integer >= 10")
        
        return errors
    
    def get_recommended_config(self, collection_purpose: str, expected_size: int) -> Dict[str, Any]:
        """Get recommended configuration for a collection"""
        base_config = {
            "vector_size": 384,
            "distance": "Cosine",
            "hnsw_config": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000
            }
        }
        
        # Adjust based on collection purpose
        if collection_purpose == "high_precision":
            base_config["hnsw_config"]["m"] = 24
            base_config["hnsw_config"]["ef_construct"] = 300
        elif collection_purpose == "high_speed":
            base_config["hnsw_config"]["m"] = 12
            base_config["hnsw_config"]["ef_construct"] = 150
        elif collection_purpose == "balanced":
            # Use default values
            pass
        
        # Adjust based on expected size
        if expected_size > 100000:
            # Large collection optimizations
            base_config["hnsw_config"]["full_scan_threshold"] = 50000
            base_config["hnsw_config"]["max_indexing_threads"] = 4
        elif expected_size < 1000:
            # Small collection optimizations
            base_config["hnsw_config"]["full_scan_threshold"] = 100
            base_config["hnsw_config"]["ef_construct"] = 100
        
        return base_config
    
    def get_all_collection_names(self) -> List[str]:
        """Get list of all configured collection names"""
        return list(self.domain_collections.keys())
    
    def get_domain_collections(self) -> List[str]:
        """Get list of domain-specific document collections"""
        return [name for name in self.domain_collections.keys() if name.startswith("documents_")]
    
    def get_entity_collections(self) -> List[str]:
        """Get list of entity-type collections"""
        return [name for name in self.domain_collections.keys() if name in ["entities", "concepts", "claims"]]