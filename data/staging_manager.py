#!/usr/bin/env python3
"""
JSON Staging System Manager
Manages the content processing workflow through JSON staging areas
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Content processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    APPROVED = "approved"
    REJECTED = "rejected"


class SourceType(Enum):
    """Content source type enumeration"""
    YOUTUBE = "youtube"
    WEBSITE = "website"
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    UPLOAD = "upload"


class Priority(Enum):
    """Processing priority enumeration"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContentMetadata:
    """Content metadata structure"""
    title: str
    author: Optional[str]
    date: Optional[str]
    domain: str
    language: str
    priority: Priority
    submitted_by: str
    file_size: Optional[int]
    content_type: str


@dataclass
class AnalysisResults:
    """Analysis results structure"""
    concepts_extracted: List[str]
    claims_identified: List[str]
    connections_discovered: List[Dict[str, Any]]
    agent_recommendations: Dict[str, Any]
    quality_score: float
    confidence_level: str


@dataclass
class AgentPipeline:
    """Agent pipeline configuration"""
    selected_agents: List[str]
    processing_order: str  # "sequential" or "parallel"
    agent_parameters: Dict[str, Any]
    completion_status: Dict[str, str]


@dataclass
class ReviewData:
    """Review and approval data"""
    reviewer: Optional[str]
    review_notes: str
    approval_reason: str
    rejection_reason: str


@dataclass
class StagedContent:
    """Complete staged content structure"""
    submission_id: str
    source_type: SourceType
    source_url: Optional[str]
    metadata: ContentMetadata
    raw_content: str
    processing_status: ProcessingStatus
    analysis_results: Optional[AnalysisResults]
    agent_pipeline: Optional[AgentPipeline]
    timestamps: Dict[str, Optional[str]]
    review_data: Optional[ReviewData]


class StagingManager:
    """
    Manages the JSON staging system for content processing
    """
    
    def __init__(self, staging_root: str = "data/staging"):
        """Initialize staging manager"""
        self.staging_root = Path(staging_root)
        self.directories = {
            ProcessingStatus.PENDING: self.staging_root / "pending",
            ProcessingStatus.PROCESSING: self.staging_root / "processing",
            ProcessingStatus.ANALYZED: self.staging_root / "analyzed",
            ProcessingStatus.APPROVED: self.staging_root / "approved",
            ProcessingStatus.REJECTED: self.staging_root / "rejected"
        }
        
        # Ensure directories exist
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Staging manager initialized at {self.staging_root}")
    
    async def submit_content(self, 
                           source_type: SourceType,
                           raw_content: str,
                           metadata: ContentMetadata,
                           source_url: Optional[str] = None,
                           agent_pipeline: Optional[AgentPipeline] = None) -> str:
        """
        Submit new content to the staging system
        """
        try:
            submission_id = str(uuid.uuid4())
            
            # Create staged content object
            staged_content = StagedContent(
                submission_id=submission_id,
                source_type=source_type,
                source_url=source_url,
                metadata=metadata,
                raw_content=raw_content,
                processing_status=ProcessingStatus.PENDING,
                analysis_results=None,
                agent_pipeline=agent_pipeline,
                timestamps={
                    "submitted": datetime.utcnow().isoformat() + "Z",
                    "analysis_started": None,
                    "analysis_completed": None,
                    "reviewed": None,
                    "approved_rejected": None
                },
                review_data=None
            )
            
            # Save to pending directory
            await self._save_staged_content(staged_content)
            
            logger.info(f"Content submitted with ID: {submission_id}")
            return submission_id
            
        except Exception as e:
            logger.error(f"Error submitting content: {e}")
            raise
    
    async def get_content(self, submission_id: str) -> Optional[StagedContent]:
        """
        Retrieve staged content by ID
        """
        try:
            for status, directory in self.directories.items():
                file_path = directory / f"{submission_id}.json"
                if file_path.exists():
                    return await self._load_staged_content(file_path)
            
            logger.warning(f"Content not found: {submission_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving content {submission_id}: {e}")
            return None
    
    async def list_content(self, 
                         status: Optional[ProcessingStatus] = None,
                         domain: Optional[str] = None,
                         priority: Optional[Priority] = None,
                         limit: int = 100) -> List[StagedContent]:
        """
        List staged content with optional filters
        """
        try:
            content_list = []
            
            # Determine which directories to search
            search_dirs = [self.directories[status]] if status else self.directories.values()
            
            for directory in search_dirs:
                if not directory.exists():
                    continue
                
                for file_path in directory.glob("*.json"):
                    if len(content_list) >= limit:
                        break
                    
                    try:
                        content = await self._load_staged_content(file_path)
                        if content:
                            # Apply filters
                            if domain and content.metadata.domain != domain:
                                continue
                            if priority and content.metadata.priority != priority:
                                continue
                            
                            content_list.append(content)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                        continue
            
            # Sort by submission time (newest first)
            content_list.sort(
                key=lambda x: x.timestamps.get("submitted", ""), 
                reverse=True
            )
            
            return content_list[:limit]
            
        except Exception as e:
            logger.error(f"Error listing content: {e}")
            return []
    
    async def update_status(self, 
                          submission_id: str, 
                          new_status: ProcessingStatus,
                          analysis_results: Optional[AnalysisResults] = None,
                          review_data: Optional[ReviewData] = None) -> bool:
        """
        Update content status and move to appropriate directory
        """
        try:
            # Find current content
            content = await self.get_content(submission_id)
            if not content:
                logger.error(f"Content not found for status update: {submission_id}")
                return False
            
            # Update status and timestamps
            old_status = content.processing_status
            content.processing_status = new_status
            
            # Update timestamps based on status
            now = datetime.utcnow().isoformat() + "Z"
            if new_status == ProcessingStatus.PROCESSING:
                content.timestamps["analysis_started"] = now
            elif new_status == ProcessingStatus.ANALYZED:
                content.timestamps["analysis_completed"] = now
            elif new_status in [ProcessingStatus.APPROVED, ProcessingStatus.REJECTED]:
                content.timestamps["approved_rejected"] = now
                content.timestamps["reviewed"] = now
            
            # Update analysis results if provided
            if analysis_results:
                content.analysis_results = analysis_results
            
            # Update review data if provided
            if review_data:
                content.review_data = review_data
            
            # Remove from old directory
            old_file = self.directories[old_status] / f"{submission_id}.json"
            if old_file.exists():
                old_file.unlink()
            
            # Save to new directory
            await self._save_staged_content(content)
            
            logger.info(f"Content {submission_id} moved from {old_status.value} to {new_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating status for {submission_id}: {e}")
            return False
    
    async def start_processing(self, submission_id: str) -> bool:
        """
        Mark content as processing
        """
        return await self.update_status(submission_id, ProcessingStatus.PROCESSING)
    
    async def complete_analysis(self, 
                              submission_id: str, 
                              analysis_results: AnalysisResults) -> bool:
        """
        Mark analysis as complete with results
        """
        return await self.update_status(
            submission_id, 
            ProcessingStatus.ANALYZED, 
            analysis_results=analysis_results
        )
    
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get staging queue statistics
        """
        try:
            stats = {}
            
            for status in ProcessingStatus:
                directory = self.directories[status]
                if directory.exists():
                    count = len(list(directory.glob("*.json")))
                    stats[status.value] = count
                else:
                    stats[status.value] = 0
            
            # Calculate processing metrics
            total_items = sum(stats.values())
            
            # Get oldest pending item
            oldest_pending = await self._get_oldest_item(ProcessingStatus.PENDING)
            
            # Get average processing time
            avg_processing_time = await self._calculate_avg_processing_time()
            
            return {
                "queue_counts": stats,
                "total_items": total_items,
                "oldest_pending": oldest_pending,
                "average_processing_time_minutes": avg_processing_time,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
    
    async def cleanup_old_items(self, days_old: int = 30) -> int:
        """
        Clean up old approved/rejected items
        """
        try:
            cleaned_count = 0
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            for status in [ProcessingStatus.APPROVED, ProcessingStatus.REJECTED]:
                directory = self.directories[status]
                if not directory.exists():
                    continue
                
                for file_path in directory.glob("*.json"):
                    try:
                        content = await self._load_staged_content(file_path)
                        if content and content.timestamps.get("approved_rejected"):
                            item_date = datetime.fromisoformat(
                                content.timestamps["approved_rejected"].replace("Z", "")
                            )
                            
                            if item_date < cutoff_date:
                                file_path.unlink()
                                cleaned_count += 1
                                
                    except Exception as e:
                        logger.error(f"Error processing {file_path} during cleanup: {e}")
                        continue
            
            logger.info(f"Cleaned up {cleaned_count} old items")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    async def export_content(self, 
                           submission_id: str, 
                           format: str = "json") -> Optional[str]:
        """
        Export staged content in specified format
        """
        try:
            content = await self.get_content(submission_id)
            if not content:
                return None
            
            if format.lower() == "json":
                return json.dumps(asdict(content), indent=2, ensure_ascii=False)
            elif format.lower() == "text":
                return content.raw_content
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting content {submission_id}: {e}")
            return None
    
    async def batch_process(self, 
                          status: ProcessingStatus,
                          processor_func,
                          batch_size: int = 10) -> Dict[str, Any]:
        """
        Process items in batches
        """
        try:
            items = await self.list_content(status=status, limit=batch_size)
            results = {
                "processed": 0,
                "failed": 0,
                "errors": []
            }
            
            for item in items:
                try:
                    await processor_func(item)
                    results["processed"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{item.submission_id}: {str(e)}")
                    logger.error(f"Error processing {item.submission_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {"processed": 0, "failed": 0, "errors": [str(e)]}
    
    async def _save_staged_content(self, content: StagedContent):
        """
        Save staged content to appropriate directory
        """
        directory = self.directories[content.processing_status]
        file_path = directory / f"{content.submission_id}.json"
        
        # Convert to dict and handle enums
        content_dict = asdict(content)
        content_dict["source_type"] = content.source_type.value
        content_dict["processing_status"] = content.processing_status.value
        content_dict["metadata"]["priority"] = content.metadata.priority.value
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content_dict, f, indent=2, ensure_ascii=False)
    
    async def _load_staged_content(self, file_path: Path) -> Optional[StagedContent]:
        """
        Load staged content from file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert enums back
            data["source_type"] = SourceType(data["source_type"])
            data["processing_status"] = ProcessingStatus(data["processing_status"])
            data["metadata"]["priority"] = Priority(data["metadata"]["priority"])
            
            # Reconstruct objects
            metadata = ContentMetadata(**data["metadata"])
            
            analysis_results = None
            if data["analysis_results"]:
                analysis_results = AnalysisResults(**data["analysis_results"])
            
            agent_pipeline = None
            if data["agent_pipeline"]:
                agent_pipeline = AgentPipeline(**data["agent_pipeline"])
            
            review_data = None
            if data["review_data"]:
                review_data = ReviewData(**data["review_data"])
            
            return StagedContent(
                submission_id=data["submission_id"],
                source_type=data["source_type"],
                source_url=data["source_url"],
                metadata=metadata,
                raw_content=data["raw_content"],
                processing_status=data["processing_status"],
                analysis_results=analysis_results,
                agent_pipeline=agent_pipeline,
                timestamps=data["timestamps"],
                review_data=review_data
            )
            
        except Exception as e:
            logger.error(f"Error loading staged content from {file_path}: {e}")
            return None
    
    async def _get_oldest_item(self, status: ProcessingStatus) -> Optional[str]:
        """
        Get the oldest item in a specific status
        """
        try:
            directory = self.directories[status]
            if not directory.exists():
                return None
            
            oldest_time = None
            oldest_id = None
            
            for file_path in directory.glob("*.json"):
                content = await self._load_staged_content(file_path)
                if content and content.timestamps.get("submitted"):
                    submit_time = content.timestamps["submitted"]
                    if oldest_time is None or submit_time < oldest_time:
                        oldest_time = submit_time
                        oldest_id = content.submission_id
            
            return oldest_id
            
        except Exception as e:
            logger.error(f"Error finding oldest item: {e}")
            return None
    
    async def _calculate_avg_processing_time(self) -> float:
        """
        Calculate average processing time in minutes
        """
        try:
            processing_times = []
            
            for status in [ProcessingStatus.ANALYZED, ProcessingStatus.APPROVED, ProcessingStatus.REJECTED]:
                directory = self.directories[status]
                if not directory.exists():
                    continue
                
                for file_path in directory.glob("*.json"):
                    content = await self._load_staged_content(file_path)
                    if (content and 
                        content.timestamps.get("submitted") and 
                        content.timestamps.get("analysis_completed")):
                        
                        start_time = datetime.fromisoformat(
                            content.timestamps["submitted"].replace("Z", "")
                        )
                        end_time = datetime.fromisoformat(
                            content.timestamps["analysis_completed"].replace("Z", "")
                        )
                        
                        duration = (end_time - start_time).total_seconds() / 60
                        processing_times.append(duration)
            
            if processing_times:
                return sum(processing_times) / len(processing_times)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating average processing time: {e}")
            return 0.0
    
    async def get_all_content(self) -> List[StagedContent]:
        """
        Get all staged content across all statuses
        """
        return await self.list_content(status=None, limit=1000)
    
    async def approve_content(self, submission_id: str, reviewer_notes: Optional[str] = None) -> bool:
        """
        Approve content for database integration (updated signature)
        """
        review_data = ReviewData(
            reviewer="api_user",
            review_notes=reviewer_notes or "",
            approval_reason="API approval",
            rejection_reason=""
        )
        
        return await self.update_status(
            submission_id, 
            ProcessingStatus.APPROVED, 
            review_data=review_data
        )
    
    async def reject_content(self, submission_id: str, rejection_reason: str, reviewer_notes: Optional[str] = None) -> bool:
        """
        Reject content with reason (updated signature)
        """
        review_data = ReviewData(
            reviewer="api_user",
            review_notes=reviewer_notes or "",
            approval_reason="",
            rejection_reason=rejection_reason
        )
        
        return await self.update_status(
            submission_id, 
            ProcessingStatus.REJECTED, 
            review_data=review_data
        )
    
    async def batch_process_by_priority(self, priority: Priority, processor_func, batch_size: int = 10) -> Dict[str, Any]:
        """
        Process items by priority level in batches
        """
        try:
            # Get pending items with specific priority
            items = await self.list_content(
                status=ProcessingStatus.PENDING, 
                priority=priority, 
                limit=batch_size
            )
            
            if not items:
                return {
                    "processed": 0,
                    "failed": 0,
                    "errors": [],
                    "message": f"No pending items with {priority.value} priority"
                }
            
            results = {
                "processed": 0,
                "failed": 0,
                "errors": [],
                "priority": priority.value,
                "batch_id": str(uuid.uuid4()),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "item_results": []
            }
            
            for item in items:
                try:
                    # Mark as processing
                    await self.start_processing(item.submission_id)
                    
                    # Process item
                    result = await processor_func(item)
                    
                    results["processed"] += 1
                    results["item_results"].append({
                        "submission_id": item.submission_id,
                        "status": "completed",
                        "result": result
                    })
                    
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{item.submission_id}: {str(e)}")
                    results["item_results"].append({
                        "submission_id": item.submission_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error(f"Error processing {item.submission_id}: {e}")
            
            results["completed_at"] = datetime.utcnow().isoformat() + "Z"
            return results
            
        except Exception as e:
            logger.error(f"Error in priority batch processing: {e}")
            return {"processed": 0, "failed": 0, "errors": [str(e)]}
    
    async def get_priority_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics broken down by priority
        """
        try:
            stats = {
                "by_priority": {},
                "by_status": {},
                "total_items": 0,
                "processing_metrics": {}
            }
            
            # Count by priority and status
            for priority in Priority:
                stats["by_priority"][priority.value] = {}
                for status in ProcessingStatus:
                    count = len(await self.list_content(status=status, priority=priority, limit=1000))
                    stats["by_priority"][priority.value][status.value] = count
                    
                    # Also update status totals
                    if status.value not in stats["by_status"]:
                        stats["by_status"][status.value] = 0
                    stats["by_status"][status.value] += count
            
            # Calculate total
            stats["total_items"] = sum(stats["by_status"].values())
            
            # Calculate processing metrics
            if stats["total_items"] > 0:
                pending_total = stats["by_status"].get("pending", 0)
                processing_total = stats["by_status"].get("processing", 0)
                completed_total = stats["by_status"].get("analyzed", 0) + stats["by_status"].get("approved", 0)
                
                stats["processing_metrics"] = {
                    "completion_rate": completed_total / stats["total_items"] if stats["total_items"] > 0 else 0,
                    "active_processing_rate": processing_total / stats["total_items"] if stats["total_items"] > 0 else 0,
                    "queue_backlog": pending_total,
                    "high_priority_pending": stats["by_priority"].get("high", {}).get("pending", 0),
                    "estimated_completion_time_hours": pending_total * 2 / 60  # Rough estimate: 2 min per item
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting priority queue stats: {e}")
            return {}
    
    async def aggregate_results_by_domain(self, status: ProcessingStatus = ProcessingStatus.ANALYZED) -> Dict[str, Any]:
        """
        Aggregate analysis results by domain
        """
        try:
            items = await self.list_content(status=status, limit=1000)
            
            domain_aggregates = {}
            total_items = 0
            
            for item in items:
                domain = item.metadata.domain
                if domain not in domain_aggregates:
                    domain_aggregates[domain] = {
                        "count": 0,
                        "total_concepts": 0,
                        "total_claims": 0,
                        "avg_quality_score": 0,
                        "confidence_levels": {"high": 0, "medium": 0, "low": 0},
                        "processing_times": [],
                        "word_counts": []
                    }
                
                domain_data = domain_aggregates[domain]
                domain_data["count"] += 1
                total_items += 1
                
                # Aggregate analysis results
                if item.analysis_results:
                    domain_data["total_concepts"] += len(item.analysis_results.concepts_extracted or [])
                    domain_data["total_claims"] += len(item.analysis_results.claims_identified or [])
                    domain_data["avg_quality_score"] += item.analysis_results.quality_score or 0
                    
                    confidence = item.analysis_results.confidence_level
                    if confidence in domain_data["confidence_levels"]:
                        domain_data["confidence_levels"][confidence] += 1
                
                # Calculate processing time if available
                if (item.timestamps.get("submitted") and 
                    item.timestamps.get("analysis_completed")):
                    try:
                        start_time = datetime.fromisoformat(item.timestamps["submitted"].replace("Z", ""))
                        end_time = datetime.fromisoformat(item.timestamps["analysis_completed"].replace("Z", ""))
                        processing_time = (end_time - start_time).total_seconds() / 60  # minutes
                        domain_data["processing_times"].append(processing_time)
                    except:
                        pass
                
                # Word count from content
                if item.raw_content:
                    word_count = len(item.raw_content.split())
                    domain_data["word_counts"].append(word_count)
            
            # Calculate averages and final metrics
            for domain, data in domain_aggregates.items():
                if data["count"] > 0:
                    data["avg_quality_score"] = data["avg_quality_score"] / data["count"]
                    data["avg_concepts_per_item"] = data["total_concepts"] / data["count"]
                    data["avg_claims_per_item"] = data["total_claims"] / data["count"]
                    
                    if data["processing_times"]:
                        data["avg_processing_time_minutes"] = sum(data["processing_times"]) / len(data["processing_times"])
                    
                    if data["word_counts"]:
                        data["avg_word_count"] = sum(data["word_counts"]) / len(data["word_counts"])
                        data["total_words_processed"] = sum(data["word_counts"])
            
            return {
                "domain_aggregates": domain_aggregates,
                "total_items_analyzed": total_items,
                "domains_covered": len(domain_aggregates),
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Error aggregating results by domain: {e}")
            return {}
    
    async def process_priority_queue(self, max_items: int = 50) -> Dict[str, Any]:
        """
        Process queue items by priority (high -> medium -> low)
        """
        try:
            results = {
                "batch_id": str(uuid.uuid4()),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "priority_results": {},
                "total_processed": 0,
                "total_failed": 0
            }
            
            remaining_capacity = max_items
            
            # Process in priority order
            for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                if remaining_capacity <= 0:
                    break
                
                # Get items for this priority level
                items = await self.list_content(
                    status=ProcessingStatus.PENDING,
                    priority=priority,
                    limit=min(remaining_capacity, 20)  # Max 20 per priority level
                )
                
                if not items:
                    results["priority_results"][priority.value] = {
                        "processed": 0,
                        "message": "No pending items"
                    }
                    continue
                
                # Mock processing function for demonstration
                async def mock_processor(item):
                    await asyncio.sleep(0.1)  # Simulate processing
                    return {"status": "mock_processed"}
                
                # Process items for this priority
                priority_result = await self.batch_process_by_priority(
                    priority, mock_processor, len(items)
                )
                
                results["priority_results"][priority.value] = priority_result
                results["total_processed"] += priority_result["processed"]
                results["total_failed"] += priority_result["failed"]
                
                remaining_capacity -= len(items)
            
            results["completed_at"] = datetime.utcnow().isoformat() + "Z"
            results["success"] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing priority queue: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_processed": 0,
                "total_failed": 0
            }