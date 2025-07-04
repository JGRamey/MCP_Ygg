#!/usr/bin/env python3
"""
MCP Server FastAPI Application
Main API server for agent coordination and user interaction
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import uuid
import traceback

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn
import yaml

# Import our agents with error handling
try:
    from agents.scraper.scraper_agent import ScraperAgent as WebScraper
except ImportError:
    WebScraper = None

try:
    from agents.text_processor.text_processor import TextProcessor
except ImportError:
    TextProcessor = None

try:
    from agents.knowledge_graph.knowledge_graph_builder import GraphBuilder
except ImportError:
    GraphBuilder = None

try:
    from agents.vector_index.vector_indexer import VectorIndexer
except ImportError:
    VectorIndexer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def create_app(config_path: str = "config/server.yaml") -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Create FastAPI app
    app = FastAPI(
        title=config['api']['title'],
        description=config['api']['description'],
        version=config['api']['version'],
        docs_url=config['api']['docs_url'],
        redoc_url=config['api']['redoc_url']
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config['cors']['allow_origins'],
        allow_credentials=config['cors']['allow_credentials'],
        allow_methods=config['cors']['allow_methods'],
        allow_headers=config['cors']['allow_headers'],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
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
    """Initialize all agent instances"""
    global scraper_agent, text_processor, graph_builder, vector_indexer
    
    try:
        logger.info("Initializing agents...")
        
        # Initialize agents
        scraper_agent = WebScraper()
        text_processor = TextProcessor()
        graph_builder = GraphBuilder()
        vector_indexer = VectorIndexer()
        
        # Setup initial structures
        graph_builder.create_root_structure()
        
        logger.info("All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise


async def cleanup_agents():
    """Cleanup agent resources"""
    global scraper_agent, text_processor, graph_builder, vector_indexer
    
    logger.info("Cleaning up agents...")
    
    if scraper_agent:
        scraper_agent.cleanup()
    if text_processor:
        text_processor.cleanup()
    if graph_builder:
        graph_builder.cleanup()
    if vector_indexer:
        vector_indexer.cleanup()


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


# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


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


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MCP Knowledge Graph Server",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status",
        "health": "/health"
    }


# Include route modules
from api.routes.scraper_routes import router as scraper_router
from api.routes.query_routes import router as query_router
from api.routes.admin_routes import router as admin_router
from api.routes.relationship_routes import router as relationship_router

app.include_router(scraper_router, prefix="/api/scraper", tags=["Scraping"])
app.include_router(query_router, prefix="/api/query", tags=["Querying"])
app.include_router(admin_router, prefix="/api/admin", tags=["Administration"])
app.include_router(relationship_router, prefix="/api/relationships", tags=["Relationships"])


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
