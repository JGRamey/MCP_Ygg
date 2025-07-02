#!/usr/bin/env python3
"""
Neo4j Manager Agent
Centralized Neo4j database operations with schema enforcement and event publishing
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
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import (
    Neo4jError, TransientError, DatabaseError, 
    ClientError, CypherSyntaxError
)

from .schema_manager import SchemaManager
from .query_optimizer import QueryOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OperationResult:
    """Result of a Neo4j operation"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    records_affected: int = 0
    event_id: Optional[str] = None

@dataclass
class NodeData:
    """Standardized node data structure"""
    node_type: str
    properties: Dict[str, Any]
    labels: Optional[List[str]] = None
    
@dataclass
class RelationshipData:
    """Standardized relationship data structure"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None

class Neo4jAgent:
    """
    Centralized Neo4j database operations agent for MCP Yggdrasil
    Provides CRUD operations, schema enforcement, and event publishing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Neo4j Agent with configuration"""
        self.config = self._load_config(config_path)
        self.driver: Optional[AsyncDriver] = None
        self.schema_manager = SchemaManager(self.config)
        self.query_optimizer = QueryOptimizer(self.config)
        self.event_queue: List[Dict] = []
        self._metrics = {
            "operations_total": 0,
            "operations_success": 0,
            "operations_failed": 0,
            "avg_execution_time": 0.0,
            "slow_queries": 0
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('neo4j_agent', {})
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "connection": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password",
                "max_pool_size": 50,
                "connection_timeout": 30
            },
            "performance": {
                "batch_size": 100,
                "query_timeout": 60,
                "enable_query_cache": True
            },
            "schema": {
                "enforce_yggdrasil_structure": True,
                "auto_create_indexes": True
            },
            "events": {
                "enable_publishing": True
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the Neo4j connection and setup schema"""
        try:
            # Create driver
            self.driver = AsyncGraphDatabase.driver(
                self.config["connection"]["uri"],
                auth=(
                    self.config["connection"]["user"], 
                    self.config["connection"]["password"]
                ),
                max_connection_pool_size=self.config["connection"]["max_pool_size"],
                connection_timeout=self.config["connection"]["connection_timeout"]
            )
            
            # Verify connection
            await self.driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")
            
            # Initialize schema
            if self.config["schema"]["enforce_yggdrasil_structure"]:
                await self.schema_manager.initialize_schema(self.driver)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j Agent: {e}")
            return False
    
    async def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def create_node(self, node_data: NodeData) -> OperationResult:
        """Create a new node in the graph"""
        start_time = time.time()
        
        try:
            # Validate node data
            if not await self.schema_manager.validate_node(node_data):
                return OperationResult(
                    success=False,
                    error="Node validation failed",
                    execution_time=time.time() - start_time
                )
            
            # Prepare Cypher query
            labels = node_data.labels or [node_data.node_type]
            label_str = ":".join(labels)
            
            # Add Yggdrasil metadata
            properties = node_data.properties.copy()
            properties.update({
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "agent_created": "neo4j_agent",
                "yggdrasil_level": self._calculate_yggdrasil_level(properties)
            })
            
            query = f"""
            CREATE (n:{label_str})
            SET n = $properties
            RETURN n, elementId(n) as node_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, properties=properties)
                record = await result.single()
                
                if record:
                    node_id = record["node_id"]
                    execution_time = time.time() - start_time
                    
                    # Publish event
                    event_id = await self._publish_event(
                        "NODE_CREATED", 
                        {"node_id": node_id, "node_type": node_data.node_type}
                    )
                    
                    # Update metrics
                    self._update_metrics(True, execution_time)
                    
                    return OperationResult(
                        success=True,
                        data={"node_id": node_id, "node": dict(record["n"])},
                        execution_time=execution_time,
                        records_affected=1,
                        event_id=event_id
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to create node: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def get_node(self, node_id: str) -> OperationResult:
        """Retrieve a node by ID"""
        start_time = time.time()
        
        try:
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            RETURN n, labels(n) as labels
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                
                execution_time = time.time() - start_time
                
                if record:
                    self._update_metrics(True, execution_time)
                    return OperationResult(
                        success=True,
                        data={
                            "node": dict(record["n"]),
                            "labels": record["labels"]
                        },
                        execution_time=execution_time,
                        records_affected=1
                    )
                else:
                    return OperationResult(
                        success=False,
                        error="Node not found",
                        execution_time=execution_time
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to get node: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> OperationResult:
        """Update a node's properties"""
        start_time = time.time()
        
        try:
            # Add update metadata
            update_properties = properties.copy()
            update_properties["updated_at"] = datetime.now().isoformat()
            update_properties["agent_updated"] = "neo4j_agent"
            
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            SET n += $properties
            RETURN n
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query, 
                    node_id=node_id, 
                    properties=update_properties
                )
                record = await result.single()
                
                execution_time = time.time() - start_time
                
                if record:
                    # Publish event
                    event_id = await self._publish_event(
                        "NODE_UPDATED", 
                        {"node_id": node_id, "updated_properties": list(properties.keys())}
                    )
                    
                    self._update_metrics(True, execution_time)
                    return OperationResult(
                        success=True,
                        data={"node": dict(record["n"])},
                        execution_time=execution_time,
                        records_affected=1,
                        event_id=event_id
                    )
                else:
                    return OperationResult(
                        success=False,
                        error="Node not found",
                        execution_time=execution_time
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to update node: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def delete_node(self, node_id: str) -> OperationResult:
        """Delete a node and its relationships"""
        start_time = time.time()
        
        try:
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                
                execution_time = time.time() - start_time
                deleted_count = record["deleted_count"] if record else 0
                
                if deleted_count > 0:
                    # Publish event
                    event_id = await self._publish_event(
                        "NODE_DELETED", 
                        {"node_id": node_id}
                    )
                    
                    self._update_metrics(True, execution_time)
                    return OperationResult(
                        success=True,
                        data={"deleted_count": deleted_count},
                        execution_time=execution_time,
                        records_affected=deleted_count,
                        event_id=event_id
                    )
                else:
                    return OperationResult(
                        success=False,
                        error="Node not found",
                        execution_time=execution_time
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to delete node: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def create_relationship(self, rel_data: RelationshipData) -> OperationResult:
        """Create a relationship between two nodes"""
        start_time = time.time()
        
        try:
            # Prepare relationship properties
            properties = rel_data.properties or {}
            properties.update({
                "created_at": datetime.now().isoformat(),
                "agent_created": "neo4j_agent"
            })
            
            query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $source_id AND elementId(b) = $target_id
            CREATE (a)-[r:{rel_data.relationship_type}]->(b)
            SET r = $properties
            RETURN r, elementId(r) as rel_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    source_id=rel_data.source_id,
                    target_id=rel_data.target_id,
                    properties=properties
                )
                record = await result.single()
                
                execution_time = time.time() - start_time
                
                if record:
                    rel_id = record["rel_id"]
                    
                    # Publish event
                    event_id = await self._publish_event(
                        "RELATIONSHIP_CREATED", 
                        {
                            "rel_id": rel_id,
                            "source_id": rel_data.source_id,
                            "target_id": rel_data.target_id,
                            "type": rel_data.relationship_type
                        }
                    )
                    
                    self._update_metrics(True, execution_time)
                    return OperationResult(
                        success=True,
                        data={"rel_id": rel_id, "relationship": dict(record["r"])},
                        execution_time=execution_time,
                        records_affected=1,
                        event_id=event_id
                    )
                else:
                    return OperationResult(
                        success=False,
                        error="Failed to create relationship - nodes not found",
                        execution_time=execution_time
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to create relationship: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> OperationResult:
        """Execute a custom Cypher query"""
        start_time = time.time()
        
        try:
            # Optimize query if enabled
            if self.config["performance"]["enable_query_cache"]:
                query = await self.query_optimizer.optimize_query(query)
            
            async with self.driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                
                execution_time = time.time() - start_time
                
                # Log slow queries
                if (execution_time > self.config.get("monitoring", {}).get("slow_query_threshold", 5.0)):
                    logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
                    self._metrics["slow_queries"] += 1
                
                self._update_metrics(True, execution_time)
                return OperationResult(
                    success=True,
                    data=records,
                    execution_time=execution_time,
                    records_affected=len(records)
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Failed to execute Cypher query: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def batch_operations(self, operations: List[Dict]) -> OperationResult:
        """Execute multiple operations in a batch transaction"""
        start_time = time.time()
        results = []
        
        try:
            async with self.driver.session() as session:
                async with session.begin_transaction() as tx:
                    for operation in operations:
                        op_type = operation.get("type")
                        op_data = operation.get("data")
                        
                        if op_type == "create_node":
                            result = await self._batch_create_node(tx, NodeData(**op_data))
                        elif op_type == "create_relationship":
                            result = await self._batch_create_relationship(tx, RelationshipData(**op_data))
                        elif op_type == "cypher":
                            result = await self._batch_execute_cypher(
                                tx, op_data["query"], op_data.get("parameters")
                            )
                        else:
                            result = {"error": f"Unknown operation type: {op_type}"}
                        
                        results.append(result)
                    
                    await tx.commit()
            
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time)
            
            return OperationResult(
                success=True,
                data=results,
                execution_time=execution_time,
                records_affected=len(operations)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"Batch operation failed: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _calculate_yggdrasil_level(self, properties: Dict[str, Any]) -> int:
        """Calculate Yggdrasil tree level based on temporal data"""
        # Extract date information
        date_str = properties.get("earliest_evidence_date", "0")
        try:
            date_value = int(date_str)
            
            # Ancient knowledge (trunk) - BCE dates
            if date_value < self.config.get("yggdrasil", {}).get("ancient_knowledge_threshold", -1000):
                return 1
            # Classical knowledge (main branches) - BCE to 500 CE
            elif date_value < 500:
                return 2
            # Medieval knowledge (branches) - 500-1500 CE
            elif date_value < self.config.get("yggdrasil", {}).get("modern_knowledge_threshold", 1500):
                return 3
            # Modern knowledge (leaves) - 1500+ CE
            else:
                return 4
                
        except (ValueError, TypeError):
            return 3  # Default to middle level
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Publish event for sync manager"""
        if not self.config["events"]["enable_publishing"]:
            return ""
        
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "agent": "neo4j_agent",
            "data": data
        }
        
        self.event_queue.append(event)
        
        # Process events in batches
        if len(self.event_queue) >= self.config["events"].get("event_batch_size", 50):
            await self._flush_events()
        
        return event_id
    
    async def _flush_events(self):
        """Flush events to the event system"""
        if not self.event_queue:
            return
        
        # Here we would send events to Redis/RabbitMQ
        # For now, just log them
        logger.info(f"Publishing {len(self.event_queue)} events")
        self.event_queue.clear()
    
    def _update_metrics(self, success: bool, execution_time: float):
        """Update performance metrics"""
        self._metrics["operations_total"] += 1
        if success:
            self._metrics["operations_success"] += 1
        else:
            self._metrics["operations_failed"] += 1
        
        # Update average execution time
        total_ops = self._metrics["operations_total"]
        current_avg = self._metrics["avg_execution_time"]
        self._metrics["avg_execution_time"] = (
            (current_avg * (total_ops - 1) + execution_time) / total_ops
        )
    
    async def _batch_create_node(self, tx, node_data: NodeData) -> Dict:
        """Create node within a transaction"""
        try:
            labels = node_data.labels or [node_data.node_type]
            label_str = ":".join(labels)
            
            properties = node_data.properties.copy()
            properties.update({
                "created_at": datetime.now().isoformat(),
                "agent_created": "neo4j_agent"
            })
            
            query = f"CREATE (n:{label_str}) SET n = $properties RETURN elementId(n) as node_id"
            result = await tx.run(query, properties=properties)
            record = await result.single()
            
            return {"success": True, "node_id": record["node_id"]}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _batch_create_relationship(self, tx, rel_data: RelationshipData) -> Dict:
        """Create relationship within a transaction"""
        try:
            properties = rel_data.properties or {}
            properties["created_at"] = datetime.now().isoformat()
            
            query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $source_id AND elementId(b) = $target_id
            CREATE (a)-[r:{rel_data.relationship_type}]->(b)
            SET r = $properties
            RETURN elementId(r) as rel_id
            """
            
            result = await tx.run(
                query,
                source_id=rel_data.source_id,
                target_id=rel_data.target_id,
                properties=properties
            )
            record = await result.single()
            
            return {"success": True, "rel_id": record["rel_id"]}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _batch_execute_cypher(self, tx, query: str, parameters: Optional[Dict]) -> Dict:
        """Execute Cypher query within a transaction"""
        try:
            result = await tx.run(query, parameters or {})
            records = await result.data()
            return {"success": True, "records": records}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self.driver:
                return {"status": "unhealthy", "error": "No driver connection"}
            
            # Test basic connectivity
            await self.driver.verify_connectivity()
            
            # Test simple query
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            return {
                "status": "healthy",
                "metrics": self.get_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }