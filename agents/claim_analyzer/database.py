#!/usr/bin/env python3
"""Database connector for Claim Analyzer Agent"""

import logging
from typing import Dict, Any

import redis.asyncio as redis
import qdrant_client
from neo4j import AsyncGraphDatabase
from qdrant_client.http.models import Distance, VectorParams

from .exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Manages connections to Neo4j, Qdrant, and Redis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neo4j_driver = None
        self.qdrant_client = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            await self._init_neo4j()
            await self._init_qdrant()
            await self._init_redis()
            await self._initialize_qdrant_collections()
            logger.info("Database connections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}") from e
    
    async def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            neo4j_config = self.config.get('database', {}).get('neo4j', {})
            self.neo4j_driver = AsyncGraphDatabase.driver(
                neo4j_config.get('uri', 'bolt://localhost:7687'),
                auth=(
                    neo4j_config.get('user', 'neo4j'),
                    neo4j_config.get('password', 'password')
                ),
                max_connection_pool_size=neo4j_config.get('max_pool_size', 20)
            )
            logger.info("Neo4j connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise DatabaseConnectionError(f"Neo4j initialization failed: {e}") from e
        
    async def _init_qdrant(self):
        """Initialize Qdrant connection"""
        try:
            qdrant_config = self.config.get('database', {}).get('qdrant', {})
            self.qdrant_client = qdrant_client.QdrantClient(
                host=qdrant_config.get('host', 'localhost'),
                port=qdrant_config.get('port', 6333),
                timeout=qdrant_config.get('timeout', 30)
            )
            logger.info("Qdrant connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise DatabaseConnectionError(f"Qdrant initialization failed: {e}") from e
        
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_config = self.config.get('database', {}).get('redis', {})
            self.redis_client = redis.from_url(
                redis_config.get('url', 'redis://localhost:6379'),
                max_connections=redis_config.get('max_connections', 50)
            )
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise DatabaseConnectionError(f"Redis initialization failed: {e}") from e
        
    async def _initialize_qdrant_collections(self):
        """Initialize Qdrant collections for claims and evidence"""
        collections = [
            {
                'name': 'claims',
                'vector_size': 384,  # all-MiniLM-L6-v2 embedding size
                'distance': Distance.COSINE
            },
            {
                'name': 'evidence',
                'vector_size': 384,
                'distance': Distance.COSINE
            }
        ]
        
        for collection in collections:
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection['name'],
                    vectors_config=VectorParams(
                        size=collection['vector_size'],
                        distance=collection['distance']
                    )
                )
                logger.info(f"Created Qdrant collection: {collection['name']}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Qdrant collection already exists: {collection['name']}")
                else:
                    logger.error(f"Error creating collection {collection['name']}: {e}")
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
            
    async def health_check(self) -> Dict[str, bool]:
        """Check status of all database connections"""
        status = {}
        
        try:
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            status['neo4j'] = True
        except Exception:
            status['neo4j'] = False
        
        try:
            collections = self.qdrant_client.get_collections()
            status['qdrant'] = True
        except Exception:
            status['qdrant'] = False
        
        try:
            await self.redis_client.ping()
            status['redis'] = True
        except Exception:
            status['redis'] = False
        
        return status