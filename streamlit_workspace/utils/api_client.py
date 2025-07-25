"""
Unified API client for Streamlit workspace
Handles all communication with FastAPI backend
"""

import json
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

import asyncio
import httpx
import streamlit as st


class APIClient:
    """Centralized API client with error handling and caching"""

    def __init__(self):
        self.base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        self.timeout = httpx.Timeout(30.0, connect=5.0)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of httpx client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"X-Client": "streamlit-workspace"},
            )
        return self._client

    async def scrape_content(
        self,
        urls: List[str],
        domain: str,
        source_type: str = "webpage",
        options: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit content for scraping via API"""
        payload = {
            "urls": urls,
            "domain": domain,
            "source_type": source_type,
            "priority": options.get("priority", 3) if options else 3,
            "subcategory": options.get("subcategory", "") if options else "",
        }

        return await self._post("/api/scrape", payload)

    async def scrape_url(
        self,
        url: str,
        domain: Optional[str] = None,
        priority: str = "medium",
        agent_pipeline: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit URL for scraping via API"""
        payload = {
            "url": url,
            "domain": domain,
            "priority": priority,
            "agent_pipeline": agent_pipeline,
        }

        return await self._post("/api/content/scrape/url", payload)

    async def scrape_youtube(
        self,
        youtube_url: str,
        domain: str = "general",
        priority: str = "medium",
        extract_metadata: bool = True,
        agent_pipeline: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit YouTube URL for transcript extraction"""
        payload = {
            "youtube_url": youtube_url,
            "domain": domain,
            "priority": priority,
            "extract_metadata": extract_metadata,
            "agent_pipeline": agent_pipeline,
        }

        return await self._post("/api/content/scrape/youtube", payload)

    async def submit_text(
        self,
        text_content: str,
        title: str,
        author: Optional[str] = None,
        domain: str = "general",
        priority: str = "medium",
        agent_pipeline: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Submit text content for analysis"""
        payload = {
            "text_content": text_content,
            "title": title,
            "author": author,
            "domain": domain,
            "priority": priority,
            "agent_pipeline": agent_pipeline,
        }

        return await self._post("/api/content/submit/text", payload)

    async def get_scrape_status(self, submission_id: str) -> Dict[str, Any]:
        """Check processing status of submitted content"""
        return await self._get(f"/api/content/status/{submission_id}")

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        return await self._get("/api/content/queue/stats")

    async def search_concepts(
        self, query: str, domain: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Search knowledge graph via API"""
        params = {"query": query, "limit": limit}
        if domain:
            params["domain"] = domain

        return await self._get("/api/query/search", params)

    async def get_graph_data(
        self, concept_id: Optional[str] = None, depth: int = 2
    ) -> Dict[str, Any]:
        """Fetch graph visualization data"""
        endpoint = (
            f"/api/graph/concepts/{concept_id}" if concept_id else "/api/graph/overview"
        )
        return await self._get(endpoint, {"depth": depth})

    async def manage_database(
        self, operation: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Database CRUD operations"""
        endpoints = {
            "create": "/api/concepts",
            "update": f"/api/concepts/{data.get('id')}",
            "delete": f"/api/concepts/{data.get('id')}",
            "list": "/api/concepts",
        }

        if operation == "list":
            return await self._get(endpoints[operation], data)
        elif operation == "delete":
            return await self._delete(endpoints[operation])
        else:
            return await self._post(endpoints[operation], data)

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a processing job"""
        return await self._get(f"/api/jobs/{job_id}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system health and metrics"""
        return await self._get("/api/system/status")

    async def get_analytics(
        self, metric_type: str, domain: Optional[str] = None, time_range: str = "7d"
    ) -> Dict[str, Any]:
        """Fetch analytics data"""
        params = {"metric_type": metric_type, "time_range": time_range}
        if domain:
            params["domain"] = domain

        return await self._get("/api/analytics", params)

    async def manage_files(
        self, operation: str, file_type: str = "csv", data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """File management operations"""
        if operation == "list":
            return await self._get(f"/api/files/{file_type}", data or {})
        elif operation == "upload":
            return await self._post(f"/api/files/{file_type}/upload", data or {})
        elif operation == "delete":
            return await self._delete(f"/api/files/{file_type}/{data.get('filename')}")
        else:
            return await self._get(f"/api/files/{file_type}/{data.get('filename')}")

    async def process_queue_operation(
        self, operation: str, task_id: Optional[str] = None, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Processing queue operations"""
        if operation == "list":
            return await self._get("/api/queue/tasks", data or {})
        elif operation == "cancel":
            return await self._post(f"/api/queue/tasks/{task_id}/cancel", {})
        elif operation == "retry":
            return await self._post(f"/api/queue/tasks/{task_id}/retry", {})
        else:
            return await self._get(f"/api/queue/tasks/{task_id}")

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Generic GET request with error handling"""
        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            st.error("⏱️ Request timed out. Please try again.")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                st.warning("🔍 Resource not found")
            elif e.response.status_code >= 500:
                st.error("🚨 Server error. Please try again later.")
            else:
                st.error(f"❌ API Error: {e.response.status_code}")
            return None
        except httpx.ConnectError:
            st.error("🔌 Cannot connect to API. Please check if the server is running.")
            return None
        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")
            return None

    async def _post(self, endpoint: str, data: Dict) -> Any:
        """Generic POST request with error handling"""
        try:
            response = await self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            st.error("⏱️ Request timed out. Please try again.")
            return None
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = error_body.get("detail", "")
            except:
                pass

            if e.response.status_code == 422:
                st.error(f"❌ Invalid request data: {error_detail}")
            elif e.response.status_code >= 500:
                st.error("🚨 Server error. Please try again later.")
            else:
                st.error(f"❌ API Error: {e.response.status_code} {error_detail}")
            return None
        except httpx.ConnectError:
            st.error("🔌 Cannot connect to API. Please check if the server is running.")
            return None
        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")
            return None

    async def _delete(self, endpoint: str) -> Any:
        """Generic DELETE request with error handling"""
        try:
            response = await self.client.delete(endpoint)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                st.warning("🔍 Resource not found")
            else:
                st.error(f"❌ API Error: {e.response.status_code}")
            return None
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
            return None

    async def health_check(self) -> bool:
        """Check API availability"""
        try:
            response = await self._get("/health")
            return response and response.get("status") == "healthy"
        except:
            return False

    def __del__(self):
        """Cleanup client on deletion"""
        if self._client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._client.aclose())
                else:
                    loop.run_until_complete(self._client.aclose())
            except:
                pass


# Singleton instance
api_client = APIClient()


# Async runner for Streamlit
def run_async(async_func):
    """Decorator to run async functions in Streamlit"""

    @wraps(async_func)
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    return wrapper


# Caching decorator with TTL
def cached_api_call(ttl: int = 300):
    """Decorator for caching API responses with TTL"""

    def decorator(func):
        @wraps(func)
        @st.cache_data(ttl=ttl)
        def wrapper(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))

        return wrapper

    return decorator


# Progress tracking context manager
class APIProgress:
    """Context manager for API operations with progress tracking"""

    def __init__(self, message: str = "Processing..."):
        self.message = message
        self.spinner = None
        self.progress_bar = None
        self.status_text = None

    def __enter__(self):
        self.spinner = st.spinner(self.message)
        self.spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spinner:
            self.spinner.__exit__(exc_type, exc_val, exc_tb)
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()

    def update_progress(self, current: int, total: int, status: str = ""):
        """Update progress bar and status"""
        if not self.progress_bar:
            self.progress_bar = st.progress(0)
        if not self.status_text:
            self.status_text = st.empty()

        progress = current / total if total > 0 else 0
        self.progress_bar.progress(progress)
        if status:
            self.status_text.text(status)


# Batch operation helper
async def batch_api_operation(
    items: List[Any],
    operation: callable,
    batch_size: int = 10,
    progress_message: str = "Processing batch...",
) -> List[Any]:
    """Execute API operations in batches with progress tracking"""
    results = []
    total = len(items)

    with APIProgress(progress_message) as progress:
        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[operation(item) for item in batch], return_exceptions=True
            )

            # Filter out exceptions
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)

            progress.update_progress(
                min(i + batch_size, total),
                total,
                f"Processed {min(i + batch_size, total)} of {total} items",
            )

    return results
