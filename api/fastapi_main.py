#!/usr/bin/env python3
"""
MCP Server FastAPI Application
Main API server for agent coordination and user interaction
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import json
import uuid
import traceback

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn
import yaml

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing systems for integration
try:
    from api.middleware.security_middleware import SecurityMiddleware, audit_logger
    logger.info("✅ Security middleware imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Security middleware not available: {e}")
    SecurityMiddleware = None
    audit_logger = None
except Exception as e:
    logger.warning(f"⚠️ Security middleware initialization failed: {e}")
    SecurityMiddleware = None
    audit_logger = None

try:
    from cache.cache_manager import cache
    logger.info("✅ Cache manager imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Cache manager not available: {e}")
    cache = None

try:
    from api.routes.performance_monitoring import router as performance_router
    logger.info("✅ Performance monitoring imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Performance monitoring not available: {e}")
    performance_router = None

try:
    from api.middleware.metrics_middleware import MetricsMiddleware
    from monitoring.metrics import metrics_collector
    logger.info("✅ Metrics middleware imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Metrics middleware not available: {e}")
    MetricsMiddleware = None
    metrics_collector = None

# Import our agents with error handling
try:
    from agents.scraper.scraper_agent import ScraperAgent as WebScraper
except ImportError:
    WebScraper = None

try:
    from agents.text_processor.text_processor import TextProcessor
    logger.info("✅ Text processor imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Text processor not available: {e}")
    TextProcessor = None
except Exception as e:
    logger.warning(f"⚠️ Text processor initialization failed: {e}")
    TextProcessor = None

try:
    from agents.knowledge_graph.knowledge_graph_builder import GraphBuilder
    logger.info("✅ Graph builder imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Graph builder not available: {e}")
    GraphBuilder = None
except Exception as e:
    logger.warning(f"⚠️ Graph builder initialization failed: {e}")
    GraphBuilder = None

try:
    from agents.vector_index.vector_indexer import VectorIndexer
    logger.info("✅ Vector indexer imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Vector indexer not available: {e}")
    VectorIndexer = None
except Exception as e:
    logger.warning(f"⚠️ Vector indexer initialization failed: {e}")
    VectorIndexer = None

# Logging already configured above


# Pydantic models for API requests/responses
class ScrapingRequest(BaseModel):
    """Request model for scraping operations"""
    urls: List[str] = Field(..., description="List of URLs to scrape")
    domain: str = Field(..., description="Domain category (math, science, religion, history, literature, philosophy)")
    subcategory: Optional[str] = Field(None, description="Subcategory within domain")
    priority: int = Field(1, description="Priority level (1-5)")
    user_id: Optional[str] = Field(None, description="User ID for attribution")


class QueryRequest(BaseModel):
    """Request model for knowledge graph queries"""
    query: str = Field(..., description="Search query")
    domain: Optional[str] = Field(None, description="Specific domain to search")
    limit: int = Field(10, description="Maximum number of results")
    include_chunks: bool = Field(True, description="Include text chunks in results")
    similarity_threshold: float = Field(0.7, description="Minimum similarity score")


class PatternRequest(BaseModel):
    """Request model for pattern detection"""
    domains: List[str] = Field(..., description="Domains to analyze for patterns")
    pattern_type: str = Field("semantic", description="Type of pattern to detect")
    confidence_threshold: float = Field(0.8, description="Minimum confidence for patterns")
    max_patterns: int = Field(10, description="Maximum number of patterns to return")


class DocumentResponse(BaseModel):
    """Response model for document information"""
    doc_id: str
    title: str
    author: Optional[str]
    domain: str
    subcategory: str
    date: Optional[str]
    word_count: int
    source: str
    language: str
    status: str


class SearchResult(BaseModel):
    """Response model for search results"""
    doc_id: str
    title: str
    score: float
    excerpt: str
    metadata: Dict[str, Any]


class PatternResult(BaseModel):
    """Response model for detected patterns"""
    pattern_id: str
    name: str
    description: str
    domains: List[str]
    confidence: float
    examples: List[str]
    validated: bool


class SystemStatus(BaseModel):
    """Response model for system status"""
    status: str
    timestamp: datetime
    agents: Dict[str, str]
    databases: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Any]


# Global agent instances
scraper_agent: Optional[WebScraper] = None
text_processor: Optional[TextProcessor] = None
graph_builder: Optional[GraphBuilder] = None
vector_indexer: Optional[VectorIndexer] = None


class PerformanceMiddleware:
    """Performance middleware for request timing and monitoring"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                message["headers"].append([
                    b"x-process-time", 
                    f"{process_time:.4f}".encode()
                ])
                message["headers"].append([
                    b"x-server", 
                    b"MCP-Yggdrasil/2.0"
                ])
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


def create_app(config_path: str = "config/server.yaml") -> FastAPI:
    """Create and configure FastAPI application with integrated systems"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Create FastAPI app
    app = FastAPI(
        title=config['api']['title'],
        description=config['api']['description'] + " - Phase 2 Performance Optimized",
        version="2.0.0",  # Phase 2 version
        docs_url=config['api']['docs_url'],
        redoc_url=config['api']['redoc_url']
    )
    
    # Add middleware in order (outer to inner)
    
    # 1. Security middleware (if available)
    if SecurityMiddleware and audit_logger:
        app.add_middleware(SecurityMiddleware, audit_logger=audit_logger)
        logger.info("✅ Security middleware integrated")
    else:
        logger.warning("⚠️ Security middleware not available")
    
    # 2. Metrics middleware (if available)
    if MetricsMiddleware:
        app.add_middleware(MetricsMiddleware)
        logger.info("✅ Metrics middleware integrated")
    else:
        logger.warning("⚠️ Metrics middleware not available")
    
    # 3. Performance middleware
    app.add_middleware(PerformanceMiddleware)
    logger.info("✅ Performance middleware added")
    
    # 4. CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config['cors']['allow_origins'],
        allow_credentials=config['cors']['allow_credentials'],
        allow_methods=config['cors']['allow_methods'],
        allow_headers=config['cors']['allow_headers'],
    )
    
    # 5. Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Initialize cache on startup
    if cache:
        logger.info("✅ Cache system available")
    else:
        logger.warning("⚠️ Cache system not available")
    
    return app


def load_config(config_path: str) -> Dict[str, Any]:
    """Load server configuration"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'server': {'host': '0.0.0.0', 'port': 8000, 'workers': 1},
            'api': {
                'title': 'MCP Knowledge Graph Server',
                'description': 'Hybrid Neo4j and Qdrant database for knowledge management',
                'version': '1.0.0',
                'docs_url': '/docs',
                'redoc_url': '/redoc'
            },
            'cors': {
                'allow_origins': ['*'],
                'allow_credentials': True,
                'allow_methods': ['*'],
                'allow_headers': ['*']
            }
        }


async def initialize_agents():
    """Initialize all agent instances and integrated systems"""
    global scraper_agent, text_processor, graph_builder, vector_indexer
    
    try:
        logger.info("🚀 Initializing MCP Yggdrasil Phase 2 systems...")
        
        # Initialize cache system
        if cache:
            try:
                health = await cache.health_check()
                if health['status'] == 'healthy':
                    logger.info("✅ Cache system initialized and healthy")
                    # Warm cache with common queries
                    from cache.cache_manager import warm_system_cache
                    await warm_system_cache()
                    logger.info("✅ Cache warmed with common data")
                else:
                    logger.warning("⚠️ Cache system unhealthy")
            except Exception as e:
                logger.error(f"❌ Cache initialization failed: {e}")
        
        # Initialize agents
        if WebScraper:
            scraper_agent = WebScraper()
        if TextProcessor:
            text_processor = TextProcessor()
        if GraphBuilder:
            graph_builder = GraphBuilder()
            # Setup initial structures
            graph_builder.create_root_structure()
        if VectorIndexer:
            vector_indexer = VectorIndexer()
        
        logger.info("✅ All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        raise


async def cleanup_agents():
    """Cleanup agent resources and integrated systems"""
    global scraper_agent, text_processor, graph_builder, vector_indexer
    
    logger.info("🔒 Cleaning up MCP Yggdrasil systems...")
    
    # Cleanup cache
    if cache:
        try:
            await cache.close()
            logger.info("✅ Cache system closed")
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")
    
    # Cleanup agents
    if scraper_agent:
        scraper_agent.cleanup()
    if text_processor:
        text_processor.cleanup()
    if graph_builder:
        graph_builder.cleanup()
    if vector_indexer:
        vector_indexer.cleanup()
    
    logger.info("✅ Cleanup completed")


# Create FastAPI app
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Handle application startup"""
    await initialize_agents()


@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown"""
    await cleanup_agents()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_id": str(uuid.uuid4())}
    )


# Enhanced health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with integrated systems status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "phase": "Phase 2 - Performance Optimized",
        "systems": {}
    }
    
    # Check cache system
    if cache:
        try:
            cache_health = await cache.health_check()
            health_status["systems"]["cache"] = cache_health
        except Exception as e:
            health_status["systems"]["cache"] = {"status": "error", "error": str(e)}
            health_status["status"] = "degraded"
    else:
        health_status["systems"]["cache"] = {"status": "unavailable"}
    
    # Check security system
    health_status["systems"]["security"] = {
        "middleware": SecurityMiddleware is not None,
        "audit_logging": audit_logger is not None
    }
    
    # Check performance monitoring
    health_status["systems"]["performance"] = {
        "monitoring_routes": performance_router is not None,
        "middleware": True  # Always available since we created it
    }
    
    # Check agents
    health_status["systems"]["agents"] = {
        "scraper": scraper_agent is not None,
        "text_processor": text_processor is not None,
        "graph_builder": graph_builder is not None,
        "vector_indexer": vector_indexer is not None
    }
    
    return health_status


# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Get comprehensive system status"""
    try:
        # Check agent status
        agents_status = {
            "scraper": "running" if scraper_agent else "not_initialized",
            "text_processor": "running" if text_processor else "not_initialized",
            "graph_builder": "running" if graph_builder else "not_initialized",
            "vector_indexer": "running" if vector_indexer else "not_initialized"
        }
        
        # Check database status
        databases_status = {}
        
        if graph_builder:
            try:
                stats = graph_builder.get_statistics()
                databases_status["neo4j"] = {"status": "connected", "stats": stats}
            except Exception as e:
                databases_status["neo4j"] = {"status": "error", "error": str(e)}
        
        if vector_indexer:
            try:
                stats = vector_indexer.get_all_statistics()
                databases_status["qdrant"] = {"status": "connected", "stats": stats}
            except Exception as e:
                databases_status["qdrant"] = {"status": "error", "error": str(e)}
        
        # Compile statistics
        statistics = {
            "uptime": "unknown",  # Would implement uptime tracking
            "total_requests": "unknown",  # Would implement request counting
            "active_connections": "unknown"  # Would implement connection tracking
        }
        
        return SystemStatus(
            status="operational",
            timestamp=datetime.now(),
            agents=agents_status,
            databases=databases_status,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if metrics_collector:
        return await metrics_collector.get_metrics()
    else:
        return JSONResponse(
            status_code=503,
            content={"detail": "Metrics collector not available"}
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MCP Knowledge Graph Server",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "/status",
        "health": "/health",
        "metrics": "/metrics"
    }


# Include route modules with error handling
try:
    from api.routes.scraper_routes import router as scraper_router
    app.include_router(scraper_router, prefix="/api/scraper", tags=["Scraping"])
except ImportError:
    logger.warning("⚠️ Scraper routes not available")

try:
    from api.routes.query_routes import router as query_router
    app.include_router(query_router, prefix="/api/query", tags=["Querying"])
except ImportError:
    logger.warning("⚠️ Query routes not available")

try:
    from api.routes.admin_routes import router as admin_router
    app.include_router(admin_router, prefix="/api/admin", tags=["Administration"])
except ImportError:
    logger.warning("⚠️ Admin routes not available")

try:
    from api.routes.relationship_routes import router as relationship_router
    app.include_router(relationship_router, prefix="/api/relationships", tags=["Relationships"])
except ImportError:
    logger.warning("⚠️ Relationship routes not available")

# Add performance monitoring routes
if performance_router:
    app.include_router(performance_router, tags=["Performance"])
    logger.info("✅ Performance monitoring routes integrated")
else:
    logger.warning("⚠️ Performance monitoring routes not available")


def main():
    """Main function to run the server"""
    config = load_config("config/server.yaml")
    
    server_config = config.get('server', {})
    
    uvicorn.run(
        "api.main:app",
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
        workers=server_config.get('workers', 1),
        reload=server_config.get('reload', False),
        log_level=server_config.get('log_level', 'info')
    )


if __name__ == "__main__":
    main()
