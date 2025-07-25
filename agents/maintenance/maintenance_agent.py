"""
Maintenance Agent for MCP Server
Handles database updates with user authorization, logging, and validation.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import asyncio
import yaml
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


class ActionType(Enum):
    """Types of maintenance actions."""

    ADD_NODE = "add_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    ADD_RELATIONSHIP = "add_relationship"
    DELETE_RELATIONSHIP = "delete_relationship"
    REINDEX_VECTOR = "reindex_vector"
    OPTIMIZE_GRAPH = "optimize_graph"
    CLEANUP_ORPHANS = "cleanup_orphans"


class ActionStatus(Enum):
    """Status of maintenance actions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class MaintenanceAction:
    """Represents a maintenance action requiring approval."""

    id: str
    action_type: ActionType
    description: str
    details: Dict[str, Any]
    created_at: datetime
    created_by: str
    status: ActionStatus = ActionStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    rollback_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class MaintenanceConfig(BaseModel):
    """Configuration for maintenance operations."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    qdrant_host: str
    qdrant_port: int
    log_dir: str
    auto_approve_safe_actions: bool = False
    max_pending_actions: int = 100
    backup_before_critical: bool = True
    require_justification: bool = True


class DatabaseMaintainer:
    """Main maintenance agent for Neo4j and Qdrant databases."""

    def __init__(self, config_path: str = "config/maintenance.yaml"):
        """Initialize the maintenance agent."""
        self.config = self._load_config(config_path)
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.pending_actions: Dict[str, MaintenanceAction] = {}
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> MaintenanceConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return MaintenanceConfig(**config_data)
        except Exception as e:
            # Fallback configuration
            return MaintenanceConfig(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                qdrant_host="localhost",
                qdrant_port=6333,
                log_dir="agents/maintenance/logs",
            )

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("maintenance_agent")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.log_dir / f"maintenance_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    async def initialize(self) -> None:
        """Initialize database connections."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()

            # Initialize Qdrant client
            self.qdrant_client = AsyncQdrantClient(
                host=self.config.qdrant_host, port=self.config.qdrant_port
            )

            # Test Qdrant connection
            await self.qdrant_client.get_collections()

            self.logger.info("Database connections initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database connections: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            await self.qdrant_client.close()
        self.logger.info("Database connections closed")

    async def propose_action(
        self,
        action_type: ActionType,
        description: str,
        details: Dict[str, Any],
        created_by: str,
        justification: Optional[str] = None,
    ) -> str:
        """Propose a maintenance action for approval."""

        if self.config.require_justification and not justification:
            raise ValueError("Justification is required for maintenance actions")

        if len(self.pending_actions) >= self.config.max_pending_actions:
            raise ValueError(
                "Too many pending actions. Please review existing actions first."
            )

        action_id = str(uuid.uuid4())
        action = MaintenanceAction(
            id=action_id,
            action_type=action_type,
            description=description,
            details=details,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
        )

        if justification:
            action.details["justification"] = justification

        self.pending_actions[action_id] = action

        # Auto-approve safe actions if configured
        if self.config.auto_approve_safe_actions and self._is_safe_action(action):
            await self.approve_action(action_id, "system_auto_approval")

        await self._log_action(action, "Action proposed")
        self.logger.info(f"Action proposed: {action_id} - {description}")

        return action_id

    def _is_safe_action(self, action: MaintenanceAction) -> bool:
        """Determine if an action is safe for auto-approval."""
        safe_actions = {
            ActionType.OPTIMIZE_GRAPH,
            ActionType.CLEANUP_ORPHANS,
            ActionType.REINDEX_VECTOR,
        }
        return action.action_type in safe_actions

    async def approve_action(self, action_id: str, approved_by: str) -> bool:
        """Approve a pending maintenance action."""
        if action_id not in self.pending_actions:
            raise ValueError(f"Action {action_id} not found")

        action = self.pending_actions[action_id]
        if action.status != ActionStatus.PENDING:
            raise ValueError(f"Action {action_id} is not pending approval")

        action.status = ActionStatus.APPROVED
        action.approved_by = approved_by
        action.approved_at = datetime.now(timezone.utc)

        await self._log_action(action, f"Action approved by {approved_by}")
        self.logger.info(f"Action approved: {action_id} by {approved_by}")

        return True

    async def reject_action(
        self, action_id: str, rejected_by: str, reason: str
    ) -> bool:
        """Reject a pending maintenance action."""
        if action_id not in self.pending_actions:
            raise ValueError(f"Action {action_id} not found")

        action = self.pending_actions[action_id]
        if action.status != ActionStatus.PENDING:
            raise ValueError(f"Action {action_id} is not pending approval")

        action.status = ActionStatus.REJECTED
        action.details["rejection_reason"] = reason
        action.details["rejected_by"] = rejected_by

        await self._log_action(action, f"Action rejected by {rejected_by}: {reason}")
        self.logger.info(f"Action rejected: {action_id} by {rejected_by} - {reason}")

        return True

    async def execute_approved_actions(self) -> List[str]:
        """Execute all approved maintenance actions."""
        executed_actions = []

        for action_id, action in list(self.pending_actions.items()):
            if action.status == ActionStatus.APPROVED:
                try:
                    success = await self._execute_action(action)
                    if success:
                        action.status = ActionStatus.COMPLETED
                        action.executed_at = datetime.now(timezone.utc)
                        executed_actions.append(action_id)
                        await self._log_action(action, "Action executed successfully")
                        self.logger.info(f"Action executed successfully: {action_id}")
                    else:
                        action.status = ActionStatus.FAILED
                        await self._log_action(action, "Action execution failed")
                        self.logger.error(f"Action execution failed: {action_id}")

                except Exception as e:
                    action.status = ActionStatus.FAILED
                    action.error_message = str(e)
                    await self._log_action(action, f"Action execution error: {e}")
                    self.logger.error(f"Action execution error {action_id}: {e}")

        # Clean up completed/failed actions
        self._cleanup_actions()

        return executed_actions

    async def _execute_action(self, action: MaintenanceAction) -> bool:
        """Execute a specific maintenance action."""
        try:
            if action.action_type == ActionType.ADD_NODE:
                return await self._add_node(action.details)
            elif action.action_type == ActionType.UPDATE_NODE:
                return await self._update_node(action.details)
            elif action.action_type == ActionType.DELETE_NODE:
                return await self._delete_node(action.details)
            elif action.action_type == ActionType.ADD_RELATIONSHIP:
                return await self._add_relationship(action.details)
            elif action.action_type == ActionType.DELETE_RELATIONSHIP:
                return await self._delete_relationship(action.details)
            elif action.action_type == ActionType.REINDEX_VECTOR:
                return await self._reindex_vector(action.details)
            elif action.action_type == ActionType.OPTIMIZE_GRAPH:
                return await self._optimize_graph(action.details)
            elif action.action_type == ActionType.CLEANUP_ORPHANS:
                return await self._cleanup_orphans(action.details)
            else:
                self.logger.error(f"Unknown action type: {action.action_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing action {action.id}: {e}")
            return False

    async def _add_node(self, details: Dict[str, Any]) -> bool:
        """Add a new node to Neo4j."""
        async with self.neo4j_driver.session() as session:
            query = f"""
            CREATE (n:{details['label']} $properties)
            RETURN id(n) as node_id
            """
            result = await session.run(query, properties=details["properties"])
            record = await result.single()
            return record is not None

    async def _update_node(self, details: Dict[str, Any]) -> bool:
        """Update an existing node in Neo4j."""
        async with self.neo4j_driver.session() as session:
            query = f"""
            MATCH (n:{details['label']})
            WHERE id(n) = $node_id
            SET n += $properties
            RETURN n
            """
            result = await session.run(
                query, node_id=details["node_id"], properties=details["properties"]
            )
            record = await result.single()
            return record is not None

    async def _delete_node(self, details: Dict[str, Any]) -> bool:
        """Delete a node from Neo4j."""
        async with self.neo4j_driver.session() as session:
            # Store rollback data
            rollback_query = f"""
            MATCH (n:{details['label']})
            WHERE id(n) = $node_id
            RETURN n, [(n)-[r]-(m) | {{rel: r, node: m}}] as relationships
            """
            rollback_result = await session.run(
                rollback_query, node_id=details["node_id"]
            )
            rollback_data = await rollback_result.single()

            if rollback_data:
                # Delete node and relationships
                delete_query = f"""
                MATCH (n:{details['label']})
                WHERE id(n) = $node_id
                DETACH DELETE n
                """
                await session.run(delete_query, node_id=details["node_id"])
                return True
            return False

    async def _add_relationship(self, details: Dict[str, Any]) -> bool:
        """Add a relationship between nodes."""
        async with self.neo4j_driver.session() as session:
            query = f"""
            MATCH (a), (b)
            WHERE id(a) = $start_node_id AND id(b) = $end_node_id
            CREATE (a)-[r:{details['relationship_type']} $properties]->(b)
            RETURN r
            """
            result = await session.run(
                query,
                start_node_id=details["start_node_id"],
                end_node_id=details["end_node_id"],
                properties=details.get("properties", {}),
            )
            record = await result.single()
            return record is not None

    async def _delete_relationship(self, details: Dict[str, Any]) -> bool:
        """Delete a relationship between nodes."""
        async with self.neo4j_driver.session() as session:
            query = f"""
            MATCH (a)-[r:{details['relationship_type']}]->(b)
            WHERE id(a) = $start_node_id AND id(b) = $end_node_id
            DELETE r
            """
            result = await session.run(
                query,
                start_node_id=details["start_node_id"],
                end_node_id=details["end_node_id"],
            )
            return True

    async def _reindex_vector(self, details: Dict[str, Any]) -> bool:
        """Reindex vectors in Qdrant."""
        collection_name = details["collection_name"]

        # Get collection info
        collection_info = await self.qdrant_client.get_collection(collection_name)

        # Recreate collection with optimized settings
        await self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=collection_info.config.params.vectors.size,
                distance=collection_info.config.params.vectors.distance,
            ),
            hnsw_config=models.HnswConfigDiff(
                ef_construct=details.get("ef_construct", 100), m=details.get("m", 16)
            ),
        )
        return True

    async def _optimize_graph(self, details: Dict[str, Any]) -> bool:
        """Optimize Neo4j graph performance."""
        async with self.neo4j_driver.session() as session:
            # Update statistics
            await session.run("CALL db.stats.retrieve('GRAPH COUNTS')")

            # Rebuild indexes if specified
            if details.get("rebuild_indexes", False):
                indexes_result = await session.run("SHOW INDEXES")
                async for record in indexes_result:
                    index_name = record["name"]
                    await session.run(f"DROP INDEX {index_name} IF EXISTS")

                # Recreate essential indexes
                essential_indexes = [
                    "CREATE INDEX document_date_domain IF NOT EXISTS FOR (d:Document) ON (d.date, d.domain)",
                    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
                    "CREATE INDEX concept_name_domain IF NOT EXISTS FOR (c:Concept) ON (c.name, c.domain)",
                ]

                for index_query in essential_indexes:
                    await session.run(index_query)

            return True

    async def _cleanup_orphans(self, details: Dict[str, Any]) -> bool:
        """Clean up orphaned nodes and relationships."""
        async with self.neo4j_driver.session() as session:
            # Find and delete orphaned nodes (nodes without relationships)
            if details.get("delete_orphan_nodes", True):
                query = """
                MATCH (n)
                WHERE NOT (n)--()
                AND NOT n:Root  // Keep root nodes
                DELETE n
                """
                result = await session.run(query)

            # Clean up duplicate relationships
            if details.get("clean_duplicate_rels", True):
                query = """
                MATCH (a)-[r1]->(b)
                WITH a, b, type(r1) as rel_type, collect(r1) as rels
                WHERE size(rels) > 1
                UNWIND rels[1..] as duplicate_rel
                DELETE duplicate_rel
                """
                await session.run(query)

            return True

    async def _log_action(self, action: MaintenanceAction, message: str) -> None:
        """Log maintenance action to file."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_id": action.id,
            "action_type": action.action_type.value,
            "status": action.status.value,
            "message": message,
            "details": action.details,
        }

        log_file = self.log_dir / f"actions_{datetime.now().strftime('%Y%m%d')}.json"

        async with aiofiles.open(log_file, "a") as f:
            await f.write(json.dumps(log_entry) + "\n")

    def _cleanup_actions(self) -> None:
        """Remove completed and failed actions from memory."""
        completed_statuses = {
            ActionStatus.COMPLETED,
            ActionStatus.FAILED,
            ActionStatus.REJECTED,
        }
        self.pending_actions = {
            action_id: action
            for action_id, action in self.pending_actions.items()
            if action.status not in completed_statuses
        }

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get list of pending actions for review."""
        return [
            {
                "id": action.id,
                "action_type": action.action_type.value,
                "description": action.description,
                "created_at": action.created_at.isoformat(),
                "created_by": action.created_by,
                "status": action.status.value,
                "details": action.details,
            }
            for action in self.pending_actions.values()
            if action.status == ActionStatus.PENDING
        ]

    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        health_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "neo4j_status": "unknown",
            "qdrant_status": "unknown",
            "pending_actions": len(
                [
                    a
                    for a in self.pending_actions.values()
                    if a.status == ActionStatus.PENDING
                ]
            ),
            "total_actions": len(self.pending_actions),
        }

        try:
            # Check Neo4j health
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    "CALL dbms.components() YIELD name, versions, edition"
                )
                await result.single()
                health_info["neo4j_status"] = "healthy"
        except Exception as e:
            health_info["neo4j_status"] = f"error: {str(e)}"

        try:
            # Check Qdrant health
            collections = await self.qdrant_client.get_collections()
            health_info["qdrant_status"] = "healthy"
            health_info["qdrant_collections"] = len(collections.collections)
        except Exception as e:
            health_info["qdrant_status"] = f"error: {str(e)}"

        return health_info


# CLI Interface for maintenance operations
async def main():
    """Main CLI interface for maintenance operations."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Maintenance Agent")
    parser.add_argument(
        "--config", default="config/maintenance.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--action",
        choices=["list", "approve", "reject", "execute", "health", "propose"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--action-id", help="Action ID for approve/reject operations")
    parser.add_argument("--user", default="cli_user", help="User performing the action")
    parser.add_argument(
        "--reason", help="Reason for rejection or justification for proposal"
    )

    args = parser.parse_args()

    maintainer = DatabaseMaintainer(args.config)
    await maintainer.initialize()

    try:
        if args.action == "list":
            actions = maintainer.get_pending_actions()
            print(f"Pending actions: {len(actions)}")
            for action in actions:
                print(
                    f"  {action['id']}: {action['description']} (by {action['created_by']})"
                )

        elif args.action == "approve":
            if not args.action_id:
                print("Error: --action-id required for approve")
                return
            await maintainer.approve_action(args.action_id, args.user)
            print(f"Action {args.action_id} approved")

        elif args.action == "reject":
            if not args.action_id or not args.reason:
                print("Error: --action-id and --reason required for reject")
                return
            await maintainer.reject_action(args.action_id, args.user, args.reason)
            print(f"Action {args.action_id} rejected")

        elif args.action == "execute":
            executed = await maintainer.execute_approved_actions()
            print(f"Executed {len(executed)} actions: {executed}")

        elif args.action == "health":
            health = await maintainer.get_system_health()
            print(json.dumps(health, indent=2))

        elif args.action == "propose":
            # Example proposal (would typically come from other agents)
            action_id = await maintainer.propose_action(
                ActionType.CLEANUP_ORPHANS,
                "Clean up orphaned nodes",
                {"delete_orphan_nodes": True, "clean_duplicate_rels": True},
                args.user,
                args.reason or "Routine maintenance",
            )
            print(f"Action proposed: {action_id}")

    finally:
        await maintainer.close()


if __name__ == "__main__":
    asyncio.run(main())
