"""
Celery Configuration for MCP Yggdrasil
Redis-based task queue with optimized settings
"""

import logging
import os

from celery import Celery
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

# Celery configuration with graceful degradation
try:
    # Redis URL from environment or default
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Create Celery app
    celery_app = Celery(
        "mcp_yggdrasil_tasks",
        broker=f"{REDIS_URL}/0",
        backend=f"{REDIS_URL}/1",
        include=[
            "tasks.document_tasks",
            "tasks.analysis_tasks",
            "tasks.scraping_tasks",
            "tasks.sync_tasks",
        ],
    )

    # Task configuration
    celery_app.conf.update(
        # Serialization
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # Timezone settings
        timezone="UTC",
        enable_utc=True,
        # Task execution settings
        task_track_started=True,
        task_time_limit=3600,  # 1 hour max
        task_soft_time_limit=3300,  # 55 minutes warning
        # Worker settings
        worker_prefetch_multiplier=4,
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        # Result backend settings
        result_expires=3600,  # Results expire after 1 hour
        result_backend_transport_options={
            "master_name": "mymaster",
            "visibility_timeout": 3600,
        },
        # Task routing
        task_routes={
            "tasks.document_tasks.*": {"queue": "documents"},
            "tasks.analysis_tasks.*": {"queue": "analysis"},
            "tasks.scraping_tasks.*": {"queue": "scraping"},
            "tasks.sync_tasks.*": {"queue": "sync"},
        },
        # Rate limiting
        task_annotations={
            "tasks.scraping_tasks.scrape_url_task": {"rate_limit": "10/m"},
            "tasks.analysis_tasks.analyze_content_task": {"rate_limit": "50/m"},
        },
    )

    # Define queues with priorities
    celery_app.conf.task_queues = (
        Queue(
            "documents",
            Exchange("documents"),
            routing_key="documents",
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            "analysis",
            Exchange("analysis"),
            routing_key="analysis",
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            "scraping",
            Exchange("scraping"),
            routing_key="scraping",
            queue_arguments={"x-max-priority": 5},
        ),
        Queue(
            "sync",
            Exchange("sync"),
            routing_key="sync",
            queue_arguments={"x-max-priority": 8},
        ),
    )

    # Task prioritization
    celery_app.conf.task_default_priority = 5
    celery_app.conf.worker_hijack_root_logger = False

    logger.info("✅ Celery app configured successfully")
    CELERY_AVAILABLE = True

except ImportError as e:
    logger.warning(f"⚠️ Celery not available: {e}")

    # Create mock celery app for graceful degradation
    class MockCelery:
        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = lambda *args, **kwargs: None
                func.apply_async = lambda *args, **kwargs: None
                return func

            return decorator

        @property
        def conf(self):
            return {}

    celery_app = MockCelery()
    CELERY_AVAILABLE = False

except Exception as e:
    logger.error(f"❌ Celery configuration failed: {e}")
    celery_app = None
    CELERY_AVAILABLE = False
