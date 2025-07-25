"""
Web Scraping Tasks
Rate-limited web scraping with enhanced scrapers
"""

import logging
from typing import Any, Dict, List

from .celery_config import CELERY_AVAILABLE, celery_app
from .models import TaskStatus

logger = logging.getLogger(__name__)


@celery_app.task(rate_limit="10/m")
def scrape_url_task(url: str, options: Dict = None) -> Dict:
    """Rate-limited web scraping task"""
    if options is None:
        options = {}

    try:
        from agents.scraper.unified_web_scraper import UnifiedWebScraper

        # Configure scraper
        scraper_config = {
            "profile": options.get("profile", "comprehensive"),
            "domain": options.get("domain", "general"),
            "respect_robots": options.get("respect_robots", True),
        }

        scraper = UnifiedWebScraper(scraper_config)
        result = scraper.scrape(url)

        # Queue for analysis if requested
        if options.get("auto_analyze", True):
            from .analysis_tasks import analyze_content_task

            analyze_content_task.delay(
                result.get("id"), ["text_processor", "claim_analyzer"]
            )

        return {
            "url": url,
            "success": True,
            "content_id": result.get("id"),
            "title": result.get("title"),
            "word_count": result.get("word_count", 0),
            "language": result.get("language"),
            "scraped_at": result.get("scraped_at"),
        }

    except Exception as e:
        logger.error(f"Scraping task failed for {url}: {e}")
        return {"url": url, "success": False, "error": str(e)}


@celery_app.task
def scrape_multiple_urls_task(urls: List[str], options: Dict = None) -> Dict:
    """Scrape multiple URLs with progress tracking"""
    if options is None:
        options = {}

    results = []
    errors = []

    for i, url in enumerate(urls):
        try:
            result = scrape_url_task(url, options)
            results.append(result)
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    return {
        "total_urls": len(urls),
        "successful": len([r for r in results if r.get("success")]),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
