#!/usr/bin/env python3
"""
Event Dispatcher
Handles event-driven synchronization between Neo4j and Qdrant
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import asyncio
import redis.asyncio as redis
from redis.asyncio import ConnectionError as RedisConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Database synchronization event types"""

    NODE_CREATED = "node.created"
    NODE_UPDATED = "node.updated"
    NODE_DELETED = "node.deleted"
    RELATIONSHIP_CREATED = "relationship.created"
    RELATIONSHIP_UPDATED = "relationship.updated"
    RELATIONSHIP_DELETED = "relationship.deleted"
    VECTOR_CREATED = "vector.created"
    VECTOR_UPDATED = "vector.updated"
    VECTOR_DELETED = "vector.deleted"
    COLLECTION_CREATED = "collection.created"
    COLLECTION_UPDATED = "collection.updated"
    COLLECTION_DELETED = "collection.deleted"


class EventPriority(Enum):
    """Event processing priority levels"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SyncEvent:
    """Synchronization event structure"""

    event_id: str
    event_type: EventType
    source_database: str  # neo4j, qdrant
    target_database: str
    entity_id: str
    entity_type: str
    timestamp: str
    priority: EventPriority
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[str] = None
    processed_at: Optional[str] = None


class EventDispatcher:
    """
    Handles event-driven synchronization between databases
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        """Initialize event dispatcher"""

        self.redis_config = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "decode_responses": True,
        }

        self.redis_client = None
        self.is_running = False
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.processed_events = 0
        self.failed_events = 0

        # Queue names
        self.event_queue = "mcp:sync:events"
        self.priority_queues = {
            EventPriority.HIGH: "mcp:sync:events:high",
            EventPriority.MEDIUM: "mcp:sync:events:medium",
            EventPriority.LOW: "mcp:sync:events:low",
        }
        self.dlq = "mcp:sync:events:dlq"  # Dead letter queue

        logger.info("Event Dispatcher initialized")

    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def start(self):
        """Start the event dispatcher"""
        if not await self.initialize():
            raise Exception("Failed to initialize Redis connection")

        self.is_running = True
        logger.info("Event Dispatcher started")

        # Start event processing tasks
        tasks = [
            asyncio.create_task(self._process_priority_queue(EventPriority.HIGH)),
            asyncio.create_task(self._process_priority_queue(EventPriority.MEDIUM)),
            asyncio.create_task(self._process_priority_queue(EventPriority.LOW)),
            asyncio.create_task(self._monitor_queues()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in event dispatcher: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the event dispatcher"""
        self.is_running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Event Dispatcher stopped")

    async def dispatch_event(self, event: SyncEvent) -> str:
        """Dispatch a synchronization event"""
        try:
            if not event.event_id:
                event.event_id = str(uuid.uuid4())

            if not event.created_at:
                event.created_at = datetime.utcnow().isoformat() + "Z"

            # Serialize event
            event_data = json.dumps(asdict(event), default=str)

            # Add to appropriate priority queue
            queue_name = self.priority_queues[event.priority]
            await self.redis_client.lpush(queue_name, event_data)

            logger.debug(f"Dispatched event {event.event_id} to {queue_name}")
            return event.event_id

        except Exception as e:
            logger.error(f"Error dispatching event: {e}")
            raise

    async def register_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    async def unregister_handler(self, event_type: EventType, handler: Callable):
        """Unregister an event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"Unregistered handler for {event_type.value}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type.value}")

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {}

            for priority, queue_name in self.priority_queues.items():
                length = await self.redis_client.llen(queue_name)
                stats[priority.value] = length

            dlq_length = await self.redis_client.llen(self.dlq)
            stats["dead_letter_queue"] = dlq_length
            stats["processed_events"] = self.processed_events
            stats["failed_events"] = self.failed_events

            return stats

        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}

    async def _process_priority_queue(self, priority: EventPriority):
        """Process events from a priority queue"""
        queue_name = self.priority_queues[priority]

        while self.is_running:
            try:
                # Block for up to 1 second waiting for events
                result = await self.redis_client.brpop(queue_name, timeout=1)

                if result:
                    _, event_data = result
                    await self._process_event(event_data)

            except RedisConnectionError:
                logger.error(f"Redis connection lost, reconnecting...")
                await asyncio.sleep(5)
                try:
                    await self.initialize()
                except Exception as e:
                    logger.error(f"Failed to reconnect to Redis: {e}")

            except Exception as e:
                logger.error(f"Error processing {priority.value} queue: {e}")
                await asyncio.sleep(1)

    async def _process_event(self, event_data: str):
        """Process a single event"""
        try:
            # Deserialize event
            event_dict = json.loads(event_data)
            event_dict["event_type"] = EventType(event_dict["event_type"])
            event_dict["priority"] = EventPriority(event_dict["priority"])

            event = SyncEvent(**event_dict)

            # Process event
            success = await self._handle_event(event)

            if success:
                self.processed_events += 1
                event.processed_at = datetime.utcnow().isoformat() + "Z"
                logger.debug(f"Successfully processed event {event.event_id}")
            else:
                await self._handle_failed_event(event, event_data)

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self.failed_events += 1

    async def _handle_event(self, event: SyncEvent) -> bool:
        """Handle an event by calling registered handlers"""
        try:
            handlers = self.event_handlers.get(event.event_type, [])

            if not handlers:
                logger.warning(f"No handlers registered for {event.event_type.value}")
                return False

            # Call all handlers
            results = []
            for handler in handlers:
                try:
                    result = await handler(event)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Handler failed for {event.event_type.value}: {e}")
                    results.append(False)

            # Consider successful if at least one handler succeeded
            return any(results)

        except Exception as e:
            logger.error(f"Error handling event {event.event_id}: {e}")
            return False

    async def _handle_failed_event(self, event: SyncEvent, event_data: str):
        """Handle a failed event"""
        event.retry_count += 1

        if event.retry_count <= event.max_retries:
            # Retry with exponential backoff
            delay = 2**event.retry_count
            await asyncio.sleep(delay)

            # Re-queue the event
            queue_name = self.priority_queues[event.priority]
            updated_event_data = json.dumps(asdict(event), default=str)
            await self.redis_client.lpush(queue_name, updated_event_data)

            logger.info(
                f"Retrying event {event.event_id} (attempt {event.retry_count})"
            )
        else:
            # Move to dead letter queue
            await self.redis_client.lpush(self.dlq, event_data)
            self.failed_events += 1
            logger.error(
                f"Event {event.event_id} moved to dead letter queue after {event.retry_count} attempts"
            )

    async def _monitor_queues(self):
        """Monitor queue health and performance"""
        while self.is_running:
            try:
                stats = await self.get_queue_stats()

                # Log queue status periodically
                total_pending = sum(stats.get(p.value, 0) for p in EventPriority)
                if total_pending > 0:
                    logger.info(
                        f"Queue status - Pending: {total_pending}, "
                        f"Processed: {self.processed_events}, "
                        f"Failed: {self.failed_events}"
                    )

                # Check for queue backup
                if total_pending > 1000:
                    logger.warning(
                        f"High queue backlog detected: {total_pending} events"
                    )

                # Check dead letter queue
                dlq_size = stats.get("dead_letter_queue", 0)
                if dlq_size > 100:
                    logger.warning(f"High dead letter queue size: {dlq_size} events")

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error in queue monitoring: {e}")
                await asyncio.sleep(60)

    async def create_node_event(
        self,
        source_db: str,
        node_id: str,
        node_type: str,
        node_data: Dict[str, Any],
        priority: EventPriority = EventPriority.MEDIUM,
    ) -> str:
        """Create a node creation event"""
        event = SyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.NODE_CREATED,
            source_database=source_db,
            target_database="qdrant" if source_db == "neo4j" else "neo4j",
            entity_id=node_id,
            entity_type=node_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            priority=priority,
            data=node_data,
            metadata={"operation": "create", "source": source_db},
        )

        return await self.dispatch_event(event)

    async def create_vector_event(
        self,
        source_db: str,
        vector_id: str,
        collection: str,
        vector_data: Dict[str, Any],
        priority: EventPriority = EventPriority.MEDIUM,
    ) -> str:
        """Create a vector operation event"""
        event = SyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VECTOR_CREATED,
            source_database=source_db,
            target_database="neo4j" if source_db == "qdrant" else "qdrant",
            entity_id=vector_id,
            entity_type=collection,
            timestamp=datetime.utcnow().isoformat() + "Z",
            priority=priority,
            data=vector_data,
            metadata={"operation": "create", "collection": collection},
        )

        return await self.dispatch_event(event)

    async def clear_dead_letter_queue(self) -> int:
        """Clear the dead letter queue and return count of cleared events"""
        try:
            count = await self.redis_client.llen(self.dlq)
            await self.redis_client.delete(self.dlq)
            logger.info(f"Cleared {count} events from dead letter queue")
            return count
        except Exception as e:
            logger.error(f"Error clearing dead letter queue: {e}")
            return 0

    async def reprocess_dead_letter_queue(self) -> int:
        """Reprocess events from dead letter queue"""
        try:
            count = 0
            while True:
                event_data = await self.redis_client.rpop(self.dlq)
                if not event_data:
                    break

                # Reset retry count and requeue
                event_dict = json.loads(event_data)
                event_dict["retry_count"] = 0

                event = SyncEvent(
                    **{
                        k: v
                        for k, v in event_dict.items()
                        if k in SyncEvent.__dataclass_fields__
                    }
                )

                await self.dispatch_event(event)
                count += 1

            logger.info(f"Requeued {count} events from dead letter queue")
            return count

        except Exception as e:
            logger.error(f"Error reprocessing dead letter queue: {e}")
            return 0
