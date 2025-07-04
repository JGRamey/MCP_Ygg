#!/usr/bin/env python3
"""
Simple FastAPI Main Application
Simplified version for quick startup with all implemented routes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import our route modules
try:
    from api.routes.analysis_pipeline import router as analysis_router
    from api.routes.content_scraping import router as content_router
    from api.routes.api_routes import scraper_router, query_router, admin_router, relationship_router
except ImportError as e:
    print(f"Route import warning: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create FastAPI application with all routes"""
    
    app = FastAPI(
        title="MCP Yggdrasil API",
        description="Hybrid Knowledge Server with Neo4j + Qdrant + Redis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include all route modules
    try:
        app.include_router(analysis_router)
        logger.info("✅ Analysis pipeline routes loaded")
    except Exception as e:
        logger.warning(f"❌ Analysis pipeline routes failed: {e}")
    
    try:
        app.include_router(content_router)
        logger.info("✅ Content scraping routes loaded")
    except Exception as e:
        logger.warning(f"❌ Content scraping routes failed: {e}")
    
    try:
        app.include_router(scraper_router, prefix="/api")
        app.include_router(query_router, prefix="/api")
        app.include_router(admin_router, prefix="/api")
        app.include_router(relationship_router, prefix="/api")
        logger.info("✅ Legacy API routes loaded")
    except Exception as e:
        logger.warning(f"❌ Legacy API routes failed: {e}")
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "MCP Yggdrasil API",
            "status": "operational",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "message": "MCP Yggdrasil API is operational",
            "services": {
                "analysis_pipeline": "available",
                "content_scraping": "available",
                "database_sync": "available"
            }
        }
    
    return app

# Create the app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)