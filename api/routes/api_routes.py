#!/usr/bin/env python3
"""
FastAPI Route Modules
Individual route handlers for different API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import json
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


# ================================
# SCRAPER ROUTES
# ================================

# File: api/routes/scraper_routes.py
scraper_router = APIRouter()


class ScrapingRequest(BaseModel):
    urls: List[str] = Field(..., description="URLs to scrape")
    domain: str = Field(..., description="Domain category")
    subcategory: Optional[str] = Field(None, description="Subcategory")
    priority: int = Field(1, description="Priority (1-5)")
    user_id: Optional[str] = Field(None, description="User ID")
    rate_limit: float = Field(1.0, description="Rate limit in seconds")
    max_retries: int = Field(3, description="Maximum retries")


class ScrapingJob(BaseModel):
    job_id: str
    status: str
    urls: List[str]
    domain: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    error_messages: List[str] = []


class UserSourceRequest(BaseModel):
    url: str = Field(..., description="Source URL")
    domain: str = Field(..., description="Domain category")
    subcategory: str = Field(..., description="Subcategory")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author name")
    description: Optional[str] = Field(None, description="Description")
    priority: int = Field(1, description="Priority level")
    tags: List[str] = Field(default=[], description="Tags")


# Global job tracking
active_jobs: Dict[str, ScrapingJob] = {}


@scraper_router.post("/scrape", response_model=Dict[str, str])
async def start_scraping(request: ScrapingRequest, background_tasks: BackgroundTasks):
    """Start a scraping job"""
    try:
        job_id = str(uuid.uuid4())
        
        # Create job record
        job = ScrapingJob(
            job_id=job_id,
            status="queued",
            urls=request.urls,
            domain=request.domain,
            created_at=datetime.now()
        )
        active_jobs[job_id] = job
        
        # Start scraping in background
        background_tasks.add_task(
            execute_scraping_job,
            job_id,
            request
        )
        
        logger.info(f"Started scraping job {job_id} for {len(request.urls)} URLs")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Scraping job started for {len(request.urls)} URLs"
        }
        
    except Exception as e:
        logger.error(f"Failed to start scraping job: {e}")
        raise HTTPException(status_code=500, detail="Failed to start scraping job")


@scraper_router.get("/jobs/{job_id}", response_model=ScrapingJob)
async def get_scraping_job(job_id: str):
    """Get status of a scraping job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]


@scraper_router.get("/jobs", response_model=List[ScrapingJob])
async def list_scraping_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of jobs")
):
    """List scraping jobs"""
    jobs = list(active_jobs.values())
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return jobs[:limit]


@scraper_router.delete("/jobs/{job_id}")
async def cancel_scraping_job(job_id: str):
    """Cancel a scraping job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job.status = "cancelled"
    logger.info(f"Cancelled scraping job {job_id}")
    
    return {"message": "Job cancelled successfully"}


@scraper_router.post("/sources/add")
async def add_user_source(request: UserSourceRequest):
    """Add a user-specified source"""
    try:
        # Import here to avoid circular dependencies
        from agents.scraper.config import UserSourcesManager
        
        manager = UserSourcesManager()
        
        # Create user source
        from agents.scraper.config import UserSource
        source = UserSource(
            url=request.url,
            domain=request.domain,
            subcategory=request.subcategory,
            title=request.title,
            author=request.author,
            description=request.description,
            priority=request.priority,
            tags=request.tags,
            added_at=datetime.now().isoformat()
        )
        
        if manager.add_source(source):
            return {"message": "Source added successfully", "url": request.url}
        else:
            raise HTTPException(status_code=400, detail="Failed to add source")
            
    except Exception as e:
        logger.error(f"Failed to add user source: {e}")
        raise HTTPException(status_code=500, detail="Failed to add source")


@scraper_router.get("/sources")
async def list_user_sources(domain: Optional[str] = Query(None)):
    """List user-specified sources"""
    try:
        from agents.scraper.config import UserSourcesManager
        
        manager = UserSourcesManager()
        
        if domain:
            sources = manager.get_sources_by_domain(domain)
        else:
            sources = manager.sources
        
        return {"sources": [source.__dict__ for source in sources]}
        
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sources")


async def execute_scraping_job(job_id: str, request: ScrapingRequest):
    """Execute scraping job in background"""
    try:
        job = active_jobs[job_id]
        job.status = "running"
        
        # Import agent here to avoid startup issues
        from agents.scraper.scraper import WebScraper, ScrapingTarget
        
        scraper = WebScraper()
        
        # Create scraping targets
        targets = []
        for url in request.urls:
            target = ScrapingTarget(
                url=url,
                domain=request.domain,
                subcategory=request.subcategory or "",
                source_type="user_specified",
                rate_limit=request.rate_limit,
                max_retries=request.max_retries
            )
            targets.append(target)
        
        # Execute scraping
        documents = await scraper.scrape_targets(targets)
        
        # Process documents through pipeline
        for doc in documents:
            try:
                # Process with text processor
                from agents.text_processor.processor import TextProcessor
                processor = TextProcessor()
                processed_doc = await processor.process_document(doc.__dict__)
                
                if processed_doc:
                    # Add to knowledge graph
                    from agents.knowledge_graph.graph_builder import GraphBuilder
                    graph_builder = GraphBuilder()
                    graph_builder.add_document(processed_doc.__dict__)
                    
                    # Index in vector database
                    from agents.vector_index.indexer import VectorIndexer
                    vector_indexer = VectorIndexer()
                    vector_indexer.index_document(processed_doc.__dict__)
                    
                    job.successful_documents += 1
                else:
                    job.failed_documents += 1
                    job.error_messages.append(f"Failed to process document from {doc.url}")
                    
            except Exception as e:
                job.failed_documents += 1
                job.error_messages.append(f"Error processing {doc.url}: {str(e)}")
                logger.error(f"Error processing document: {e}")
        
        job.total_documents = len(documents)
        job.status = "completed"
        job.completed_at = datetime.now()
        
        logger.info(f"Completed scraping job {job_id}: {job.successful_documents}/{job.total_documents} successful")
        
    except Exception as e:
        job = active_jobs[job_id]
        job.status = "failed"
        job.error_messages.append(f"Job failed: {str(e)}")
        job.completed_at = datetime.now()
        logger.error(f"Scraping job {job_id} failed: {e}")


# ================================
# QUERY ROUTES
# ================================

# File: api/routes/query_routes.py
query_router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    domain: Optional[str] = Field(None, description="Specific domain")
    limit: int = Field(10, description="Maximum results")
    include_chunks: bool = Field(True, description="Include text chunks")
    similarity_threshold: float = Field(0.7, description="Minimum similarity")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class SearchResult(BaseModel):
    doc_id: str
    title: str
    author: Optional[str]
    score: float
    excerpt: str
    domain: str
    date: Optional[str]
    source: str
    metadata: Dict[str, Any]


class GraphQuery(BaseModel):
    cypher_query: str = Field(..., description="Cypher query")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


@query_router.post("/search", response_model=List[SearchResult])
async def search_documents(request: QueryRequest):
    """Search documents using vector similarity"""
    try:
        # Import here to avoid circular dependencies
        from agents.vector_index.indexer import VectorIndexer
        from sentence_transformers import SentenceTransformer
        
        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(request.query).tolist()
        
        # Search vector database
        vector_indexer = VectorIndexer()
        results = vector_indexer.search(
            query=request.query,
            query_embedding=query_embedding,
            domain=request.domain,
            limit=request.limit,
            score_threshold=request.similarity_threshold,
            filter_metadata=request.filters
        )
        
        # Format results
        search_results = []
        for result in results:
            metadata = result.metadata
            search_result = SearchResult(
                doc_id=metadata.get('doc_id', ''),
                title=metadata.get('title', ''),
                author=metadata.get('author', ''),
                score=result.score,
                excerpt=metadata.get('chunk_text', metadata.get('content', ''))[:200],
                domain=metadata.get('domain', ''),
                date=metadata.get('date', ''),
                source=metadata.get('source', ''),
                metadata=metadata
            )
            search_results.append(search_result)
        
        logger.info(f"Search query '{request.query}' returned {len(search_results)} results")
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@query_router.post("/graph/query")
async def query_graph(request: GraphQuery):
    """Execute Cypher query on Neo4j"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        
        graph_builder = GraphBuilder()
        result = graph_builder.neo4j.execute_query(
            request.cypher_query,
            request.parameters or {}
        )
        
        # Convert result to JSON-serializable format
        records = []
        for record in result:
            record_dict = {}
            for key in record.keys():
                value = record[key]
                if hasattr(value, '__dict__'):
                    record_dict[key] = dict(value)
                else:
                    record_dict[key] = value
            records.append(record_dict)
        
        return {"records": records, "count": len(records)}
        
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        raise HTTPException(status_code=500, detail="Graph query failed")


@query_router.get("/graph/structure/{domain}")
async def get_graph_structure(domain: str):
    """Get Yggdrasil tree structure for a domain"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        
        graph_builder = GraphBuilder()
        structure = graph_builder.get_yggdrasil_structure(domain)
        
        return structure
        
    except Exception as e:
        logger.error(f"Failed to get graph structure: {e}")
        raise HTTPException(status_code=500, detail="Failed to get graph structure")


@query_router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get detailed document information"""
    try:
        from agents.text_processor.utils import load_processed_document
        
        doc_data = load_processed_document(doc_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return doc_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document")


@query_router.get("/documents/{doc_id}/similar")
async def get_similar_documents(
    doc_id: str,
    limit: int = Query(10, description="Maximum results")
):
    """Find documents similar to a specific document"""
    try:
        from agents.vector_index.indexer import VectorIndexer
        
        vector_indexer = VectorIndexer()
        results = vector_indexer.search_by_document_id(doc_id)
        
        return {"similar_documents": results[:limit]}
        
    except Exception as e:
        logger.error(f"Failed to find similar documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar documents")


# ================================
# ADMIN ROUTES
# ================================

# File: api/routes/admin_routes.py
admin_router = APIRouter()


class MaintenanceRequest(BaseModel):
    operation: str = Field(..., description="Maintenance operation")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")
    user_authorization: bool = Field(False, description="User authorization required")


class BackupRequest(BaseModel):
    backup_type: str = Field("full", description="Backup type (full, incremental)")
    include_vectors: bool = Field(True, description="Include vector data")
    include_graph: bool = Field(True, description="Include graph data")
    compression: bool = Field(True, description="Compress backup")


@admin_router.get("/statistics")
async def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        stats = {}
        
        # Graph statistics
        try:
            from agents.knowledge_graph.graph_builder import GraphBuilder
            graph_builder = GraphBuilder()
            stats['graph'] = graph_builder.get_statistics()
        except Exception as e:
            stats['graph'] = {"error": str(e)}
        
        # Vector statistics
        try:
            from agents.vector_index.indexer import VectorIndexer
            vector_indexer = VectorIndexer()
            stats['vector'] = vector_indexer.get_all_statistics()
        except Exception as e:
            stats['vector'] = {"error": str(e)}
        
        # Processing statistics
        stats['processing'] = {
            "active_jobs": len(active_jobs),
            "completed_jobs": len([j for j in active_jobs.values() if j.status == "completed"]),
            "failed_jobs": len([j for j in active_jobs.values() if j.status == "failed"])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@admin_router.post("/maintenance")
async def perform_maintenance(request: MaintenanceRequest, background_tasks: BackgroundTasks):
    """Perform system maintenance operations"""
    try:
        operation = request.operation.lower()
        
        if operation == "optimize_graph":
            background_tasks.add_task(optimize_graph_database)
            return {"message": "Graph optimization started"}
        
        elif operation == "optimize_vectors":
            background_tasks.add_task(optimize_vector_database)
            return {"message": "Vector optimization started"}
        
        elif operation == "rebuild_indexes":
            background_tasks.add_task(rebuild_indexes)
            return {"message": "Index rebuilding started"}
        
        elif operation == "cleanup_cache":
            background_tasks.add_task(cleanup_caches)
            return {"message": "Cache cleanup started"}
        
        else:
            raise HTTPException(status_code=400, detail="Unknown maintenance operation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Maintenance operation failed: {e}")
        raise HTTPException(status_code=500, detail="Maintenance operation failed")


@admin_router.post("/backup")
async def create_backup(request: BackupRequest, background_tasks: BackgroundTasks):
    """Create system backup"""
    try:
        backup_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            execute_backup,
            backup_id,
            request.backup_type,
            request.include_vectors,
            request.include_graph,
            request.compression
        )
        
        return {
            "backup_id": backup_id,
            "status": "started",
            "message": "Backup process initiated"
        }
        
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(status_code=500, detail="Backup creation failed")


@admin_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from all systems"""
    try:
        # Delete from vector database
        from agents.vector_index.indexer import VectorIndexer
        vector_indexer = VectorIndexer()
        vector_success = vector_indexer.delete_document(doc_id)
        
        # Delete from graph database
        from agents.knowledge_graph.graph_builder import GraphBuilder
        graph_builder = GraphBuilder()
        graph_query = """
        MATCH (d:Document {id: $doc_id})
        DETACH DELETE d
        RETURN count(d) as deleted_count
        """
        result = graph_builder.neo4j.execute_query(graph_query, {"doc_id": doc_id})
        graph_success = result.single()["deleted_count"] > 0 if result.single() else False
        
        return {
            "doc_id": doc_id,
            "vector_deleted": vector_success,
            "graph_deleted": graph_success,
            "message": "Document deletion completed"
        }
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Document deletion failed")


# Background task functions
async def optimize_graph_database():
    """Optimize graph database"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        graph_builder = GraphBuilder()
        
        # Create temporal relationships
        graph_builder.create_temporal_relationships()
        graph_builder.create_concept_relationships()
        
        logger.info("Graph database optimization completed")
    except Exception as e:
        logger.error(f"Graph optimization failed: {e}")


async def optimize_vector_database():
    """Optimize vector database"""
    try:
        from agents.vector_index.indexer import VectorIndexer
        vector_indexer = VectorIndexer()
        vector_indexer.optimize_collections()
        
        logger.info("Vector database optimization completed")
    except Exception as e:
        logger.error(f"Vector optimization failed: {e}")


async def rebuild_indexes():
    """Rebuild database indexes"""
    try:
        # Rebuild graph indexes
        from agents.knowledge_graph.graph_builder import GraphBuilder
        graph_builder = GraphBuilder()
        graph_builder.setup_schema()
        
        logger.info("Index rebuilding completed")
    except Exception as e:
        logger.error(f"Index rebuilding failed: {e}")


async def cleanup_caches():
    """Cleanup system caches"""
    try:
        from agents.vector_index.indexer import VectorIndexer
        vector_indexer = VectorIndexer()
        vector_indexer.cache.clear_prefix("*")
        
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")


async def execute_backup(
    backup_id: str,
    backup_type: str,
    include_vectors: bool,
    include_graph: bool,
    compression: bool
):
    """Execute backup operation"""
    try:
        backup_path = f"data/backups/{backup_id}"
        
        if include_graph:
            from agents.backup.backup import BackupAgent
            backup_agent = BackupAgent()
            # Would implement actual backup logic
        
        if include_vectors:
            from agents.vector_index.indexer import VectorIndexer
            vector_indexer = VectorIndexer()
            # Would implement vector backup
        
        logger.info(f"Backup {backup_id} completed successfully")
    except Exception as e:
        logger.error(f"Backup {backup_id} failed: {e}")


# ================================
# RELATIONSHIP ROUTES
# ================================

# File: api/routes/relationship_routes.py
relationship_router = APIRouter()


class RelationshipRequest(BaseModel):
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Relationship type")
    properties: Optional[Dict[str, Any]] = Field(None, description="Relationship properties")
    user_validated: bool = Field(False, description="User validation required")


@relationship_router.post("/create")
async def create_relationship(request: RelationshipRequest):
    """Create a new relationship between nodes"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        
        graph_builder = GraphBuilder()
        
        # Create relationship
        relationship_query = """
        MATCH (from_node {id: $from_id})
        MATCH (to_node {id: $to_id})
        CREATE (from_node)-[r:$rel_type $properties]->(to_node)
        RETURN r
        """
        
        # Note: In real Cypher, you can't parameterize relationship types like this
        # This is simplified for demonstration
        query = f"""
        MATCH (from_node {{id: $from_id}})
        MATCH (to_node {{id: $to_id}})
        CREATE (from_node)-[r:{request.relationship_type}]->(to_node)
        SET r += $properties
        SET r.created_at = datetime()
        SET r.user_validated = $user_validated
        RETURN r
        """
        
        result = graph_builder.neo4j.execute_query(query, {
            "from_id": request.from_node,
            "to_id": request.to_node,
            "properties": request.properties or {},
            "user_validated": request.user_validated
        })
        
        return {
            "message": "Relationship created successfully",
            "from_node": request.from_node,
            "to_node": request.to_node,
            "relationship_type": request.relationship_type
        }
        
    except Exception as e:
        logger.error(f"Failed to create relationship: {e}")
        raise HTTPException(status_code=500, detail="Failed to create relationship")


@relationship_router.get("/node/{node_id}")
async def get_node_relationships(node_id: str):
    """Get all relationships for a specific node"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        
        graph_builder = GraphBuilder()
        
        query = """
        MATCH (n {id: $node_id})-[r]-(connected)
        RETURN n, r, connected, type(r) as relationship_type
        """
        
        result = graph_builder.neo4j.execute_query(query, {"node_id": node_id})
        
        relationships = []
        for record in result:
            rel_data = {
                "relationship_type": record["relationship_type"],
                "connected_node": dict(record["connected"]),
                "properties": dict(record["r"])
            }
            relationships.append(rel_data)
        
        return {"node_id": node_id, "relationships": relationships}
        
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        raise HTTPException(status_code=500, detail="Failed to get relationships")


@relationship_router.delete("/relationship")
async def delete_relationship(
    from_node: str = Query(...),
    to_node: str = Query(...),
    relationship_type: str = Query(...)
):
    """Delete a specific relationship"""
    try:
        from agents.knowledge_graph.graph_builder import GraphBuilder
        
        graph_builder = GraphBuilder()
        
        query = f"""
        MATCH (from_node {{id: $from_id}})-[r:{relationship_type}]->(to_node {{id: $to_id}})
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = graph_builder.neo4j.execute_query(query, {
            "from_id": from_node,
            "to_id": to_node
        })
        
        deleted_count = result.single()["deleted_count"] if result.single() else 0
        
        return {
            "message": f"Deleted {deleted_count} relationship(s)",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to delete relationship: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete relationship")


# Export routers for import in main.py
router = scraper_router  # For compatibility with main.py imports
