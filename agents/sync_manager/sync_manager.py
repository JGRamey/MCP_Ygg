#!/usr/bin/env python3
"""
Database Synchronization Manager
Central coordinator for Neo4j ↔ Qdrant database synchronization
"""

import asyncio
import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Database imports
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyncEvent:
    """Synchronization event structure"""
    event_id: str
    event_type: str  # node_created, node_updated, node_deleted, etc.
    source_database: str  # neo4j or qdrant
    target_database: str
    entity_id: str
    entity_type: str
    timestamp: str
    data: Dict[str, Any]
    status: str  # pending, processing, completed, failed
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class SyncTransaction:
    """Cross-database transaction wrapper"""
    transaction_id: str
    operations: List[Dict[str, Any]]
    status: str  # pending, committed, rolled_back
    created_at: str
    completed_at: Optional[str] = None


@dataclass
class ConsistencyCheck:
    """Database consistency check result"""
    check_id: str
    check_type: str
    timestamp: str
    neo4j_count: int
    qdrant_count: int
    inconsistencies: List[Dict[str, Any]]
    status: str  # passed, failed, warning


class DatabaseSyncManager:
    """
    Central coordinator for database synchronization between Neo4j and Qdrant
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        """Initialize the sync manager"""
        
        # Database connections
        self.neo4j_driver = None
        self.qdrant_client = None
        self.redis_client = None
        
        # Connection parameters
        self.neo4j_config = {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_password
        }
        self.qdrant_config = {
            "host": qdrant_host,
            "port": qdrant_port
        }
        self.redis_config = {
            "host": redis_host,
            "port": redis_port
        }
        
        # Sync state
        self.sync_events: Dict[str, SyncEvent] = {}
        self.sync_transactions: Dict[str, SyncTransaction] = {}
        self.is_running = False
        
        # Configuration
        self.config = {
            "sync_interval": 30,  # seconds
            "batch_size": 100,
            "max_retries": 3,
            "consistency_check_interval": 300,  # seconds
            "event_retention_days": 7
        }
        
        logger.info("Database Sync Manager initialized")
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"])
            )
            
            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()
            
            # Initialize Qdrant connection
            self.qdrant_client = QdrantClient(
                host=self.qdrant_config["host"],
                port=self.qdrant_config["port"]
            )
            
            # Test Qdrant connection
            collections = self.qdrant_client.get_collections()
            
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            logger.info("All database connections initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing connections: {e}")
            return False
    
    async def start_sync_service(self):
        """Start the synchronization service"""
        if not await self.initialize():
            raise Exception("Failed to initialize database connections")
        
        self.is_running = True
        logger.info("Starting database synchronization service")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._consistency_checker()),
            asyncio.create_task(self._cleanup_service())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in sync service: {e}")
        finally:
            self.is_running = False
            await self.shutdown()
    
    async def stop_sync_service(self):
        """Stop the synchronization service"""
        self.is_running = False
        logger.info("Stopping database synchronization service")
    
    async def shutdown(self):
        """Clean shutdown of all connections"""
        try:
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            
            if self.qdrant_client:
                self.qdrant_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("All connections closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def submit_sync_event(self, 
                              event_type: str,
                              source_database: str,
                              entity_id: str,
                              entity_type: str,
                              data: Dict[str, Any]) -> str:
        """Submit a synchronization event"""
        try:
            event_id = str(uuid.uuid4())
            target_database = "qdrant" if source_database == "neo4j" else "neo4j"
            
            sync_event = SyncEvent(
                event_id=event_id,
                event_type=event_type,
                source_database=source_database,
                target_database=target_database,
                entity_id=entity_id,
                entity_type=entity_type,
                timestamp=datetime.utcnow().isoformat() + "Z",
                data=data,
                status="pending"
            )
            
            # Store event
            self.sync_events[event_id] = sync_event
            
            # Add to Redis queue
            await self.redis_client.lpush("sync_events", json.dumps(asdict(sync_event)))
            
            logger.info(f"Submitted sync event: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error submitting sync event: {e}")
            raise
    
    async def execute_sync_transaction(self, operations: List[Dict[str, Any]]) -> str:
        """Execute a cross-database transaction"""
        try:
            transaction_id = str(uuid.uuid4())
            
            transaction = SyncTransaction(
                transaction_id=transaction_id,
                operations=operations,
                status="pending",
                created_at=datetime.utcnow().isoformat() + "Z"
            )
            
            self.sync_transactions[transaction_id] = transaction
            
            # Execute operations
            neo4j_ops = [op for op in operations if op.get("target") == "neo4j"]
            qdrant_ops = [op for op in operations if op.get("target") == "qdrant"]
            
            success = True
            error_message = None
            
            try:
                # Execute Neo4j operations
                if neo4j_ops:
                    await self._execute_neo4j_operations(neo4j_ops)
                
                # Execute Qdrant operations
                if qdrant_ops:
                    await self._execute_qdrant_operations(qdrant_ops)
                
                # Commit transaction
                transaction.status = "committed"
                transaction.completed_at = datetime.utcnow().isoformat() + "Z"
                
            except Exception as e:
                success = False
                error_message = str(e)
                
                # Rollback operations
                await self._rollback_transaction(transaction_id, neo4j_ops, qdrant_ops)
                transaction.status = "rolled_back"
                transaction.completed_at = datetime.utcnow().isoformat() + "Z"
            
            logger.info(f"Transaction {transaction_id} {'committed' if success else 'rolled back'}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error executing sync transaction: {e}")
            raise
    
    async def check_consistency(self, check_type: str = "full") -> ConsistencyCheck:
        """Perform database consistency check"""
        try:
            check_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            inconsistencies = []
            
            # Count entities in both databases
            neo4j_count = await self._count_neo4j_entities()
            qdrant_count = await self._count_qdrant_points()
            
            # Detailed consistency checks
            if check_type == "full":
                inconsistencies = await self._detailed_consistency_check()
            
            # Determine status
            status = "passed"
            if abs(neo4j_count - qdrant_count) > 0:
                status = "warning"
            if inconsistencies:
                status = "failed"
            
            consistency_check = ConsistencyCheck(
                check_id=check_id,
                check_type=check_type,
                timestamp=timestamp,
                neo4j_count=neo4j_count,
                qdrant_count=qdrant_count,
                inconsistencies=inconsistencies,
                status=status
            )
            
            logger.info(f"Consistency check {check_id}: {status}")
            return consistency_check
            
        except Exception as e:
            logger.error(f"Error performing consistency check: {e}")
            raise
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        try:
            # Event statistics
            event_stats = {
                "pending": len([e for e in self.sync_events.values() if e.status == "pending"]),
                "processing": len([e for e in self.sync_events.values() if e.status == "processing"]),
                "completed": len([e for e in self.sync_events.values() if e.status == "completed"]),
                "failed": len([e for e in self.sync_events.values() if e.status == "failed"])
            }
            
            # Transaction statistics
            transaction_stats = {
                "pending": len([t for t in self.sync_transactions.values() if t.status == "pending"]),
                "committed": len([t for t in self.sync_transactions.values() if t.status == "committed"]),
                "rolled_back": len([t for t in self.sync_transactions.values() if t.status == "rolled_back"])
            }
            
            # Connection status
            connection_status = {
                "neo4j": await self._check_neo4j_connection(),
                "qdrant": await self._check_qdrant_connection(),
                "redis": await self._check_redis_connection()
            }
            
            return {
                "service_running": self.is_running,
                "event_statistics": event_stats,
                "transaction_statistics": transaction_stats,
                "connection_status": connection_status,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {"error": str(e)}
    
    async def _event_processor(self):
        """Background task to process sync events"""
        while self.is_running:
            try:
                # Process events from Redis queue
                event_data = await self.redis_client.brpop("sync_events", timeout=1)
                
                if event_data:
                    event_json = event_data[1]
                    event_dict = json.loads(event_json)
                    
                    # Convert back to SyncEvent
                    sync_event = SyncEvent(**event_dict)
                    
                    # Process the event
                    await self._process_sync_event(sync_event)
                
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_sync_event(self, event: SyncEvent):
        """Process a single sync event"""
        try:
            logger.info(f"Processing sync event: {event.event_id}")
            
            # Update status
            event.status = "processing"
            self.sync_events[event.event_id] = event
            
            # Route to appropriate handler
            if event.target_database == "qdrant":
                success = await self._sync_to_qdrant(event)
            else:
                success = await self._sync_to_neo4j(event)
            
            # Update final status
            if success:
                event.status = "completed"
            else:
                event.status = "failed"
                event.retry_count += 1
                
                # Retry if under limit
                if event.retry_count < self.config["max_retries"]:
                    event.status = "pending"
                    await self.redis_client.lpush("sync_events", json.dumps(asdict(event)))
            
            self.sync_events[event.event_id] = event
            
        except Exception as e:
            logger.error(f"Error processing sync event {event.event_id}: {e}")
            event.status = "failed"
            event.error_message = str(e)
            self.sync_events[event.event_id] = event
    
    async def _sync_to_qdrant(self, event: SyncEvent) -> bool:
        """Sync data from Neo4j to Qdrant"""
        try:
            if event.event_type == "node_created":
                return await self._create_qdrant_point(event)
            elif event.event_type == "node_updated":
                return await self._update_qdrant_point(event)
            elif event.event_type == "node_deleted":
                return await self._delete_qdrant_point(event)
            else:
                logger.warning(f"Unknown event type for Qdrant: {event.event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing to Qdrant: {e}")
            return False
    
    async def _sync_to_neo4j(self, event: SyncEvent) -> bool:
        """Sync data from Qdrant to Neo4j"""
        try:
            if event.event_type == "point_created":
                return await self._create_neo4j_node(event)
            elif event.event_type == "point_updated":
                return await self._update_neo4j_node(event)
            elif event.event_type == "point_deleted":
                return await self._delete_neo4j_node(event)
            else:
                logger.warning(f"Unknown event type for Neo4j: {event.event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing to Neo4j: {e}")
            return False
    
    async def _consistency_checker(self):
        """Background task for consistency checking"""
        while self.is_running:
            try:
                # Perform consistency check
                consistency_check = await self.check_consistency("basic")
                
                # Log results
                if consistency_check.status != "passed":
                    logger.warning(f"Consistency check failed: {consistency_check.check_id}")
                
                # Wait for next check
                await asyncio.sleep(self.config["consistency_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in consistency checker: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_service(self):
        """Background task for cleanup operations"""
        while self.is_running:
            try:
                # Clean up old events
                cutoff_date = datetime.utcnow() - timedelta(days=self.config["event_retention_days"])
                
                events_to_remove = []
                for event_id, event in self.sync_events.items():
                    event_date = datetime.fromisoformat(event.timestamp.replace("Z", ""))
                    if event_date < cutoff_date and event.status in ["completed", "failed"]:
                        events_to_remove.append(event_id)
                
                # Remove old events
                for event_id in events_to_remove:
                    del self.sync_events[event_id]
                
                if events_to_remove:
                    logger.info(f"Cleaned up {len(events_to_remove)} old sync events")
                
                # Wait for next cleanup
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup service: {e}")
                await asyncio.sleep(3600)
    
    # Database operation methods (simplified implementations)
    async def _create_qdrant_point(self, event: SyncEvent) -> bool:
        """Create a point in Qdrant"""
        # Implementation would create vector point with metadata
        return True
    
    async def _update_qdrant_point(self, event: SyncEvent) -> bool:
        """Update a point in Qdrant"""
        # Implementation would update vector point
        return True
    
    async def _delete_qdrant_point(self, event: SyncEvent) -> bool:
        """Delete a point from Qdrant"""
        # Implementation would delete vector point
        return True
    
    async def _create_neo4j_node(self, event: SyncEvent) -> bool:
        """Create a node in Neo4j"""
        # Implementation would create graph node
        return True
    
    async def _update_neo4j_node(self, event: SyncEvent) -> bool:
        """Update a node in Neo4j"""
        # Implementation would update graph node
        return True
    
    async def _delete_neo4j_node(self, event: SyncEvent) -> bool:
        """Delete a node from Neo4j"""
        # Implementation would delete graph node
        return True
    
    async def _execute_neo4j_operations(self, operations: List[Dict[str, Any]]):
        """Execute Neo4j operations in transaction"""
        async with self.neo4j_driver.session() as session:
            async with session.begin_transaction() as tx:
                for op in operations:
                    await tx.run(op["query"], op.get("parameters", {}))
    
    async def _execute_qdrant_operations(self, operations: List[Dict[str, Any]]):
        """Execute Qdrant operations"""
        for op in operations:
            if op["operation"] == "upsert":
                self.qdrant_client.upsert(
                    collection_name=op["collection"],
                    points=op["points"]
                )
            elif op["operation"] == "delete":
                self.qdrant_client.delete(
                    collection_name=op["collection"],
                    points_selector=op["selector"]
                )
    
    async def _rollback_transaction(self, 
                                  transaction_id: str,
                                  neo4j_ops: List[Dict[str, Any]],
                                  qdrant_ops: List[Dict[str, Any]]):
        """Rollback transaction operations"""
        logger.warning(f"Rolling back transaction: {transaction_id}")
        # Implementation would reverse the operations
    
    async def _count_neo4j_entities(self) -> int:
        """Count entities in Neo4j"""
        async with self.neo4j_driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            return record["count"] if record else 0
    
    async def _count_qdrant_points(self) -> int:
        """Count points in Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            total_count = 0
            for collection in collections.collections:
                info = self.qdrant_client.get_collection(collection.name)
                total_count += info.points_count
            return total_count
        except Exception:
            return 0
    
    async def _detailed_consistency_check(self) -> List[Dict[str, Any]]:
        """Perform detailed consistency check"""
        inconsistencies = []
        # Implementation would compare entities between databases
        return inconsistencies
    
    async def _check_neo4j_connection(self) -> bool:
        """Check Neo4j connection status"""
        try:
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    async def _check_qdrant_connection(self) -> bool:
        """Check Qdrant connection status"""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception:
            return False
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connection status"""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False