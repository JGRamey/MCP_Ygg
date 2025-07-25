#!/usr/bin/env python3
"""
Conflict Resolver
Handles conflicts in database synchronization between Neo4j and Qdrant
"""

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of synchronization conflicts"""

    CONTENT_MISMATCH = "content_mismatch"
    TIMESTAMP_CONFLICT = "timestamp_conflict"
    DUPLICATE_ENTITY = "duplicate_entity"
    MISSING_DEPENDENCY = "missing_dependency"
    SCHEMA_MISMATCH = "schema_mismatch"
    VERSION_CONFLICT = "version_conflict"
    CIRCULAR_REFERENCE = "circular_reference"


class ResolutionStrategy(Enum):
    """Conflict resolution strategies"""

    TIMESTAMP = "timestamp"  # Use most recent timestamp
    NEO4J_PRIORITY = "neo4j_priority"  # Neo4j takes precedence
    QDRANT_PRIORITY = "qdrant_priority"  # Qdrant takes precedence
    CONTENT_HASH = "content_hash"  # Use content hash comparison
    MANUAL_REVIEW = "manual_review"  # Require manual intervention
    MERGE = "merge"  # Attempt to merge both versions
    ROLLBACK = "rollback"  # Rollback to previous state


class ConflictSeverity(Enum):
    """Conflict severity levels"""

    LOW = "low"  # Auto-resolvable
    MEDIUM = "medium"  # Requires attention but not critical
    HIGH = "high"  # Critical, may affect data integrity
    CRITICAL = "critical"  # Immediate attention required


@dataclass
class ConflictData:
    """Represents conflicting data from different sources"""

    source: str  # neo4j or qdrant
    entity_id: str
    entity_type: str
    data: Dict[str, Any]
    timestamp: str
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class Conflict:
    """Represents a synchronization conflict"""

    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    entity_id: str
    entity_type: str
    neo4j_data: Optional[ConflictData]
    qdrant_data: Optional[ConflictData]
    detected_at: str
    resolved_at: Optional[str]
    resolution_strategy: Optional[ResolutionStrategy]
    resolution_data: Optional[Dict[str, Any]]
    manual_review_required: bool
    description: str
    impact_assessment: str


class ConflictResolver:
    """
    Handles conflict resolution in database synchronization
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.TIMESTAMP,
        manual_review_threshold: float = 0.5,
    ):
        """Initialize conflict resolver"""

        self.default_strategy = default_strategy
        self.manual_review_threshold = manual_review_threshold

        # Active conflicts tracking
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolved_conflicts: List[Conflict] = []

        # Resolution statistics
        self.resolution_stats = {
            "total_conflicts": 0,
            "auto_resolved": 0,
            "manual_resolved": 0,
            "failed_resolutions": 0,
            "by_strategy": {strategy.value: 0 for strategy in ResolutionStrategy},
        }

        logger.info(
            f"Conflict Resolver initialized with strategy: {default_strategy.value}"
        )

    async def detect_conflict(
        self,
        entity_id: str,
        entity_type: str,
        neo4j_data: Optional[Dict[str, Any]] = None,
        qdrant_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conflict]:
        """Detect if there's a conflict between Neo4j and Qdrant data"""

        try:
            # Basic validation
            if not neo4j_data and not qdrant_data:
                return None

            # Detect conflict type
            conflict_type = await self._analyze_conflict_type(neo4j_data, qdrant_data)
            if not conflict_type:
                return None

            # Create conflict data objects
            neo4j_conflict_data = None
            if neo4j_data:
                neo4j_conflict_data = ConflictData(
                    source="neo4j",
                    entity_id=entity_id,
                    entity_type=entity_type,
                    data=neo4j_data,
                    timestamp=neo4j_data.get(
                        "last_modified", datetime.utcnow().isoformat() + "Z"
                    ),
                    content_hash=self._calculate_content_hash(neo4j_data),
                    metadata=neo4j_data.get("metadata", {}),
                )

            qdrant_conflict_data = None
            if qdrant_data:
                qdrant_conflict_data = ConflictData(
                    source="qdrant",
                    entity_id=entity_id,
                    entity_type=entity_type,
                    data=qdrant_data,
                    timestamp=qdrant_data.get(
                        "last_modified", datetime.utcnow().isoformat() + "Z"
                    ),
                    content_hash=self._calculate_content_hash(qdrant_data),
                    metadata=qdrant_data.get("payload", {}),
                )

            # Determine severity
            severity = await self._assess_conflict_severity(
                conflict_type, neo4j_conflict_data, qdrant_conflict_data
            )

            # Create conflict object
            conflict = Conflict(
                conflict_id=str(uuid.uuid4()),
                conflict_type=conflict_type,
                severity=severity,
                entity_id=entity_id,
                entity_type=entity_type,
                neo4j_data=neo4j_conflict_data,
                qdrant_data=qdrant_conflict_data,
                detected_at=datetime.utcnow().isoformat() + "Z",
                resolved_at=None,
                resolution_strategy=None,
                resolution_data=None,
                manual_review_required=severity
                in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL],
                description=await self._generate_conflict_description(
                    conflict_type, entity_id, entity_type
                ),
                impact_assessment=await self._assess_impact(conflict_type, entity_type),
            )

            # Track conflict
            self.active_conflicts[conflict.conflict_id] = conflict
            self.resolution_stats["total_conflicts"] += 1

            logger.warning(
                f"Conflict detected: {conflict.conflict_id} - {conflict_type.value}"
            )
            return conflict

        except Exception as e:
            logger.error(f"Error detecting conflict for {entity_id}: {e}")
            return None

    async def resolve_conflict(
        self, conflict: Conflict, strategy: Optional[ResolutionStrategy] = None
    ) -> bool:
        """Resolve a synchronization conflict"""

        try:
            # Use provided strategy or default
            resolution_strategy = strategy or self.default_strategy

            # Check if manual review is required
            if (
                conflict.manual_review_required
                and resolution_strategy != ResolutionStrategy.MANUAL_REVIEW
            ):
                logger.warning(
                    f"Conflict {conflict.conflict_id} requires manual review"
                )
                return False

            # Apply resolution strategy
            success = await self._apply_resolution_strategy(
                conflict, resolution_strategy
            )

            if success:
                # Mark as resolved
                conflict.resolved_at = datetime.utcnow().isoformat() + "Z"
                conflict.resolution_strategy = resolution_strategy

                # Update statistics
                self.resolution_stats["auto_resolved"] += 1
                self.resolution_stats["by_strategy"][resolution_strategy.value] += 1

                # Move to resolved conflicts
                if conflict.conflict_id in self.active_conflicts:
                    del self.active_conflicts[conflict.conflict_id]
                self.resolved_conflicts.append(conflict)

                logger.info(
                    f"Conflict {conflict.conflict_id} resolved using {resolution_strategy.value}"
                )
                return True
            else:
                self.resolution_stats["failed_resolutions"] += 1
                logger.error(f"Failed to resolve conflict {conflict.conflict_id}")
                return False

        except Exception as e:
            logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
            self.resolution_stats["failed_resolutions"] += 1
            return False

    async def get_active_conflicts(self) -> List[Conflict]:
        """Get all active conflicts"""
        return list(self.active_conflicts.values())

    async def get_conflict_by_id(self, conflict_id: str) -> Optional[Conflict]:
        """Get a specific conflict by ID"""
        return self.active_conflicts.get(conflict_id)

    async def get_conflicts_by_entity(self, entity_id: str) -> List[Conflict]:
        """Get all conflicts for a specific entity"""
        return [c for c in self.active_conflicts.values() if c.entity_id == entity_id]

    async def get_resolution_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics"""
        active_count = len(self.active_conflicts)
        resolved_count = len(self.resolved_conflicts)

        return {
            **self.resolution_stats,
            "active_conflicts": active_count,
            "resolved_conflicts": resolved_count,
            "resolution_rate": resolved_count
            / max(1, self.resolution_stats["total_conflicts"]),
            "auto_resolution_rate": self.resolution_stats["auto_resolved"]
            / max(1, resolved_count),
        }

    async def _analyze_conflict_type(
        self,
        neo4j_data: Optional[Dict[str, Any]],
        qdrant_data: Optional[Dict[str, Any]],
    ) -> Optional[ConflictType]:
        """Analyze and determine the type of conflict"""

        if not neo4j_data and qdrant_data:
            return ConflictType.MISSING_DEPENDENCY

        if neo4j_data and not qdrant_data:
            return ConflictType.MISSING_DEPENDENCY

        if not neo4j_data or not qdrant_data:
            return None

        # Compare content hashes
        neo4j_hash = self._calculate_content_hash(neo4j_data)
        qdrant_hash = self._calculate_content_hash(qdrant_data)

        if neo4j_hash != qdrant_hash:
            # Check timestamps
            neo4j_timestamp = neo4j_data.get("last_modified", "")
            qdrant_timestamp = qdrant_data.get("last_modified", "")

            if neo4j_timestamp and qdrant_timestamp:
                if (
                    abs(
                        datetime.fromisoformat(
                            neo4j_timestamp.replace("Z", "")
                        ).timestamp()
                        - datetime.fromisoformat(
                            qdrant_timestamp.replace("Z", "")
                        ).timestamp()
                    )
                    > 60
                ):
                    return ConflictType.TIMESTAMP_CONFLICT

            return ConflictType.CONTENT_MISMATCH

        return None

    async def _assess_conflict_severity(
        self,
        conflict_type: ConflictType,
        neo4j_data: Optional[ConflictData],
        qdrant_data: Optional[ConflictData],
    ) -> ConflictSeverity:
        """Assess the severity of a conflict"""

        # Critical severity conditions
        if conflict_type == ConflictType.CIRCULAR_REFERENCE:
            return ConflictSeverity.CRITICAL

        if conflict_type == ConflictType.SCHEMA_MISMATCH:
            return ConflictSeverity.HIGH

        # High severity conditions
        if conflict_type == ConflictType.MISSING_DEPENDENCY:
            return ConflictSeverity.HIGH

        # Medium severity conditions
        if conflict_type == ConflictType.TIMESTAMP_CONFLICT:
            # Check time difference
            if neo4j_data and qdrant_data:
                neo4j_time = datetime.fromisoformat(
                    neo4j_data.timestamp.replace("Z", "")
                )
                qdrant_time = datetime.fromisoformat(
                    qdrant_data.timestamp.replace("Z", "")
                )
                time_diff = abs((neo4j_time - qdrant_time).total_seconds())

                if time_diff > 3600:  # More than 1 hour difference
                    return ConflictSeverity.MEDIUM

        # Default to low severity
        return ConflictSeverity.LOW

    async def _apply_resolution_strategy(
        self, conflict: Conflict, strategy: ResolutionStrategy
    ) -> bool:
        """Apply a specific resolution strategy"""

        try:
            if strategy == ResolutionStrategy.TIMESTAMP:
                return await self._resolve_by_timestamp(conflict)

            elif strategy == ResolutionStrategy.NEO4J_PRIORITY:
                return await self._resolve_neo4j_priority(conflict)

            elif strategy == ResolutionStrategy.QDRANT_PRIORITY:
                return await self._resolve_qdrant_priority(conflict)

            elif strategy == ResolutionStrategy.CONTENT_HASH:
                return await self._resolve_by_content_hash(conflict)

            elif strategy == ResolutionStrategy.MERGE:
                return await self._resolve_by_merge(conflict)

            elif strategy == ResolutionStrategy.MANUAL_REVIEW:
                return await self._mark_for_manual_review(conflict)

            elif strategy == ResolutionStrategy.ROLLBACK:
                return await self._resolve_by_rollback(conflict)

            else:
                logger.error(f"Unknown resolution strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Error applying resolution strategy {strategy.value}: {e}")
            return False

    async def _resolve_by_timestamp(self, conflict: Conflict) -> bool:
        """Resolve conflict using most recent timestamp"""

        if not conflict.neo4j_data or not conflict.qdrant_data:
            return False

        neo4j_time = datetime.fromisoformat(
            conflict.neo4j_data.timestamp.replace("Z", "")
        )
        qdrant_time = datetime.fromisoformat(
            conflict.qdrant_data.timestamp.replace("Z", "")
        )

        if neo4j_time > qdrant_time:
            # Neo4j is more recent
            conflict.resolution_data = {
                "chosen_source": "neo4j",
                "reason": "More recent timestamp",
                "winning_data": asdict(conflict.neo4j_data),
            }
        else:
            # Qdrant is more recent (or equal)
            conflict.resolution_data = {
                "chosen_source": "qdrant",
                "reason": "More recent timestamp",
                "winning_data": asdict(conflict.qdrant_data),
            }

        return True

    async def _resolve_neo4j_priority(self, conflict: Conflict) -> bool:
        """Resolve conflict by giving Neo4j priority"""

        if not conflict.neo4j_data:
            return False

        conflict.resolution_data = {
            "chosen_source": "neo4j",
            "reason": "Neo4j priority strategy",
            "winning_data": asdict(conflict.neo4j_data),
        }

        return True

    async def _resolve_qdrant_priority(self, conflict: Conflict) -> bool:
        """Resolve conflict by giving Qdrant priority"""

        if not conflict.qdrant_data:
            return False

        conflict.resolution_data = {
            "chosen_source": "qdrant",
            "reason": "Qdrant priority strategy",
            "winning_data": asdict(conflict.qdrant_data),
        }

        return True

    async def _resolve_by_content_hash(self, conflict: Conflict) -> bool:
        """Resolve conflict using content hash comparison"""

        if not conflict.neo4j_data or not conflict.qdrant_data:
            return False

        # Use lexicographically larger hash as winner (arbitrary but consistent)
        if conflict.neo4j_data.content_hash > conflict.qdrant_data.content_hash:
            conflict.resolution_data = {
                "chosen_source": "neo4j",
                "reason": "Content hash comparison",
                "winning_data": asdict(conflict.neo4j_data),
            }
        else:
            conflict.resolution_data = {
                "chosen_source": "qdrant",
                "reason": "Content hash comparison",
                "winning_data": asdict(conflict.qdrant_data),
            }

        return True

    async def _resolve_by_merge(self, conflict: Conflict) -> bool:
        """Resolve conflict by merging data from both sources"""

        if not conflict.neo4j_data or not conflict.qdrant_data:
            return False

        # Simple merge strategy - combine non-conflicting fields
        merged_data = conflict.neo4j_data.data.copy()

        for key, value in conflict.qdrant_data.data.items():
            if key not in merged_data:
                merged_data[key] = value
            elif merged_data[key] != value:
                # Handle conflicts in individual fields
                merged_data[key] = {
                    "neo4j": merged_data[key],
                    "qdrant": value,
                    "conflict": True,
                }

        conflict.resolution_data = {
            "chosen_source": "merged",
            "reason": "Data merge strategy",
            "winning_data": merged_data,
            "merge_conflicts": [
                k
                for k, v in merged_data.items()
                if isinstance(v, dict) and v.get("conflict")
            ],
        }

        return True

    async def _mark_for_manual_review(self, conflict: Conflict) -> bool:
        """Mark conflict for manual review"""

        conflict.manual_review_required = True
        conflict.resolution_data = {
            "chosen_source": "manual_review",
            "reason": "Requires manual intervention",
            "review_queue": True,
        }

        logger.info(f"Conflict {conflict.conflict_id} marked for manual review")
        return True

    async def _resolve_by_rollback(self, conflict: Conflict) -> bool:
        """Resolve conflict by rolling back to previous state"""

        # This would require access to previous versions
        # For now, just mark as requiring manual intervention
        return await self._mark_for_manual_review(conflict)

    def _calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """Calculate a hash of the content for comparison"""

        # Remove metadata fields that shouldn't affect content hash
        filtered_data = {
            k: v
            for k, v in data.items()
            if k not in ["last_modified", "metadata", "timestamps"]
        }

        # Sort keys for consistent hashing
        content_str = json.dumps(filtered_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    async def _generate_conflict_description(
        self, conflict_type: ConflictType, entity_id: str, entity_type: str
    ) -> str:
        """Generate a human-readable conflict description"""

        descriptions = {
            ConflictType.CONTENT_MISMATCH: f"Content mismatch detected for {entity_type} {entity_id}",
            ConflictType.TIMESTAMP_CONFLICT: f"Timestamp conflict for {entity_type} {entity_id}",
            ConflictType.DUPLICATE_ENTITY: f"Duplicate entity detected: {entity_type} {entity_id}",
            ConflictType.MISSING_DEPENDENCY: f"Missing dependency for {entity_type} {entity_id}",
            ConflictType.SCHEMA_MISMATCH: f"Schema mismatch for {entity_type} {entity_id}",
            ConflictType.VERSION_CONFLICT: f"Version conflict for {entity_type} {entity_id}",
            ConflictType.CIRCULAR_REFERENCE: f"Circular reference detected involving {entity_type} {entity_id}",
        }

        return descriptions.get(
            conflict_type, f"Unknown conflict type for {entity_type} {entity_id}"
        )

    async def _assess_impact(
        self, conflict_type: ConflictType, entity_type: str
    ) -> str:
        """Assess the potential impact of the conflict"""

        impact_levels = {
            ConflictType.CONTENT_MISMATCH: "May cause data inconsistency between databases",
            ConflictType.TIMESTAMP_CONFLICT: "May affect temporal queries and version tracking",
            ConflictType.DUPLICATE_ENTITY: "May cause duplicate data and query inconsistencies",
            ConflictType.MISSING_DEPENDENCY: "May break referential integrity",
            ConflictType.SCHEMA_MISMATCH: "May cause application errors and data corruption",
            ConflictType.VERSION_CONFLICT: "May cause data loss or overwrites",
            ConflictType.CIRCULAR_REFERENCE: "May cause infinite loops and system instability",
        }

        base_impact = impact_levels.get(conflict_type, "Unknown impact")
        return f"{base_impact} for {entity_type} entities"
