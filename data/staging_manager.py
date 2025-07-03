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
    
    async def approve_content(self, 
                            submission_id: str, 
                            reviewer: str, 
                            reason: str = "") -> bool:
        """
        Approve content for database integration
        """
        review_data = ReviewData(
            reviewer=reviewer,
            review_notes="",
            approval_reason=reason,
            rejection_reason=""
        )
        
        return await self.update_status(
            submission_id, 
            ProcessingStatus.APPROVED, 
            review_data=review_data
        )
    
    async def reject_content(self, 
                           submission_id: str, 
                           reviewer: str, 
                           reason: str) -> bool:
        """
        Reject content with reason
        """
        review_data = ReviewData(
            reviewer=reviewer,
            review_notes="",
            approval_reason="",
            rejection_reason=reason
        )
        
        return await self.update_status(
            submission_id, 
            ProcessingStatus.REJECTED, 
            review_data=review_data
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