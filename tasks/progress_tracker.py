"""
Task Progress Tracking
Redis-based progress tracking for async tasks
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .models import TaskStatus

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available for progress tracking")


class TaskProgressTracker:
    """Redis-based task progress tracker with fallback"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.key = f"task_progress:{task_id}"
        self.ttl = 3600  # 1 hour
        
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis.ping()
                self._redis_available = True
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis_available = False
        else:
            self._redis_available = False
        
        # Fallback in-memory storage
        if not self._redis_available:
            if not hasattr(TaskProgressTracker, '_memory_store'):
                TaskProgressTracker._memory_store = {}
    
    def update(
        self,
        current: int,
        total: int,
        status: TaskStatus = TaskStatus.PROGRESS,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update task progress"""
        progress_data = {
            'task_id': self.task_id,
            'current': current,
            'total': total,
            'percentage': (current / total * 100) if total > 0 else 0,
            'status': status.value,
            'message': message,
            'metadata': metadata or {},
            'updated_at': datetime.utcnow().isoformat()
        }
        
        self._store_progress(progress_data)
    
    def complete(
        self,
        status: TaskStatus = TaskStatus.SUCCESS,
        result: Optional[Dict[str, Any]] = None,
        message: str = "Task completed"
    ):
        """Mark task as complete"""
        progress_data = {
            'task_id': self.task_id,
            'current': 1,
            'total': 1,
            'percentage': 100,
            'status': status.value,
            'message': message,
            'result': result,
            'completed_at': datetime.utcnow().isoformat()
        }
        
        self._store_progress(progress_data)
    
    def error(self, error_message: str, traceback: Optional[str] = None):
        """Mark task as failed"""
        progress_data = {
            'task_id': self.task_id,
            'status': TaskStatus.FAILURE.value,
            'error': error_message,
            'traceback': traceback,
            'failed_at': datetime.utcnow().isoformat()
        }
        
        self._store_progress(progress_data)
    
    def get(self) -> Optional[Dict[str, Any]]:
        """Get current progress"""
        if self._redis_available:
            try:
                data = self.redis.get(self.key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get failed: {e}")
        
        # Fallback to memory store
        return TaskProgressTracker._memory_store.get(self.task_id)
    
    def _store_progress(self, progress_data: Dict[str, Any]):
        """Store progress data with Redis/memory fallback"""
        if self._redis_available:
            try:
                self.redis.setex(
                    self.key,
                    self.ttl,
                    json.dumps(progress_data)
                )
                return
            except Exception as e:
                logger.error(f"Redis store failed: {e}")
        
        # Fallback to memory
        TaskProgressTracker._memory_store[self.task_id] = progress_data


def get_task_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task progress by ID"""
    tracker = TaskProgressTracker(task_id)
    return tracker.get()


def cleanup_old_progress(max_age_hours: int = 24):
    """Clean up old progress entries"""
    if not REDIS_AVAILABLE:
        return
    
    try:
        redis_client = redis.Redis(decode_responses=True)
        pattern = "task_progress:*"
        
        # Get all progress keys
        keys = redis_client.keys(pattern)
        
        current_time = datetime.utcnow()
        cleaned = 0
        
        for key in keys:
            try:
                data = redis_client.get(key)
                if data:
                    progress = json.loads(data)
                    updated_at = datetime.fromisoformat(
                        progress.get('updated_at', current_time.isoformat())
                    )
                    
                    # Remove if older than max_age_hours
                    age = current_time - updated_at
                    if age.total_seconds() > (max_age_hours * 3600):
                        redis_client.delete(key)
                        cleaned += 1
            except Exception as e:
                logger.error(f"Error cleaning progress key {key}: {e}")
        
        logger.info(f"Cleaned {cleaned} old progress entries")
        
    except Exception as e:
        logger.error(f"Progress cleanup failed: {e}")