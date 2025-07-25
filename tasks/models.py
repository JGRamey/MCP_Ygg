"""
Task Queue Data Models
Pydantic models for task management and progress tracking
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration"""

    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


class TaskPriority(str, Enum):
    """Task priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskResult(BaseModel):
    """Task result model"""

    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentProcessingTask(BaseModel):
    """Document processing task configuration"""

    documents: List[Dict[str, Any]]
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback_url: Optional[str] = None
    user_id: Optional[str] = None


class AnalysisTask(BaseModel):
    """Content analysis task configuration"""

    content_id: str
    analysis_types: List[str] = Field(default=["text_processor", "claim_analyzer"])
    options: Dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL


class ScrapingTask(BaseModel):
    """Web scraping task configuration"""

    urls: List[str]
    scraping_profile: str = "comprehensive"
    domain: str = "general"
    options: Dict[str, Any] = Field(default_factory=dict)
    rate_limit: int = 10  # requests per minute
