"""
Task Queue Utilities
Helper functions for task management and monitoring
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .celery_config import celery_app, CELERY_AVAILABLE
from .models import TaskStatus, TaskResult
from .progress_tracker import get_task_progress

logger = logging.getLogger(__name__)


def get_task_status(task_id: str) -> Optional[TaskResult]:
    """Get comprehensive task status including Celery and progress data"""
    try:
        result_data = {
            'task_id': task_id,
            'status': TaskStatus.PENDING,
            'created_at': datetime.utcnow(),
            'progress': {},
            'metadata': {}
        }
        
        # Get Celery task result if available
        if CELERY_AVAILABLE and hasattr(celery_app, 'AsyncResult'):
            try:
                celery_result = celery_app.AsyncResult(task_id)
                result_data['status'] = TaskStatus(celery_result.status.lower())
                
                if celery_result.ready():
                    if celery_result.successful():
                        result_data['result'] = celery_result.result
                        result_data['completed_at'] = datetime.utcnow()
                    else:
                        result_data['error'] = str(celery_result.info)
                        result_data['traceback'] = getattr(celery_result, 'traceback', None)
                        
            except Exception as e:
                logger.error(f"Error getting Celery task status: {e}")
        
        # Get progress data
        progress_data = get_task_progress(task_id)
        if progress_data:
            result_data.update(progress_data)
            # Override status if progress has more recent info
            if 'status' in progress_data:
                result_data['status'] = TaskStatus(progress_data['status'])
        
        return TaskResult(**result_data)
        
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return None


def cancel_task(task_id: str) -> bool:
    """Cancel a running task"""
    try:
        if CELERY_AVAILABLE and hasattr(celery_app, 'control'):
            celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Task {task_id} cancelled")
            return True
        else:
            logger.warning("Celery not available, cannot cancel task")
            return False
            
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return False


def get_active_tasks() -> List[Dict[str, Any]]:
    """Get list of currently active tasks"""
    try:
        if not CELERY_AVAILABLE:
            return []
        
        # Get active tasks from Celery
        active_tasks = []
        
        if hasattr(celery_app, 'control'):
            inspect = celery_app.control.inspect()
            
            # Get active tasks from all workers
            active = inspect.active()
            if active:
                for worker, tasks in active.items():
                    for task in tasks:
                        active_tasks.append({
                            'task_id': task['id'],
                            'name': task['name'],
                            'worker': worker,
                            'args': task.get('args', []),
                            'kwargs': task.get('kwargs', {}),
                            'time_start': task.get('time_start')
                        })
        
        return active_tasks
        
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        return []


def get_task_stats() -> Dict[str, Any]:
    """Get overall task queue statistics"""
    try:
        stats = {
            'celery_available': CELERY_AVAILABLE,
            'active_tasks': 0,
            'queues': {},
            'workers': {}
        }
        
        if CELERY_AVAILABLE and hasattr(celery_app, 'control'):
            inspect = celery_app.control.inspect()
            
            # Get active tasks count
            active = inspect.active()
            if active:
                stats['active_tasks'] = sum(len(tasks) for tasks in active.values())
                stats['workers'] = {
                    worker: len(tasks) for worker, tasks in active.items()
                }
            
            # Get queue lengths (if Redis is available)
            try:
                import redis
                redis_client = redis.Redis(decode_responses=True)
                
                queue_names = ['documents', 'analysis', 'scraping', 'sync']
                for queue_name in queue_names:
                    length = redis_client.llen(queue_name)
                    stats['queues'][queue_name] = length
                    
            except Exception as e:
                logger.warning(f"Could not get queue stats: {e}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting task stats: {e}")
        return {'error': str(e)}


def health_check() -> Dict[str, Any]:
    """Perform health check on task queue system"""  
    health = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {}
    }
    
    # Check Celery availability
    health['components']['celery'] = {
        'available': CELERY_AVAILABLE,
        'status': 'healthy' if CELERY_AVAILABLE else 'unavailable'
    }
    
    # Check Redis connection
    try:
        import redis
        redis_client = redis.Redis(socket_connect_timeout=2)
        redis_client.ping()
        health['components']['redis'] = {
            'available': True,
            'status': 'healthy'
        }
    except Exception as e:
        health['components']['redis'] = {
            'available': False,
            'status': 'error',
            'error': str(e)
        }
        health['status'] = 'degraded'
    
    # Check worker availability
    if CELERY_AVAILABLE:
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            worker_count = len(stats) if stats else 0
            
            health['components']['workers'] = {
                'count': worker_count,
                'status': 'healthy' if worker_count > 0 else 'no_workers'
            }
            
            if worker_count == 0:
                health['status'] = 'degraded'
                
        except Exception as e:
            health['components']['workers'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
    
    return health