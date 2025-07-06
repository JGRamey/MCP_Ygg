"""
Content Scraping API Routes
FastAPI endpoints for content submission and scraping operations
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, HttpUrl
import asyncio
import json
import uuid
import tempfile
import os
from pathlib import Path

# Import agents and managers
try:
    from data.staging_manager import (
        StagingManager, ContentMetadata, SourceType, Priority, AgentPipeline
    )
    from agents.concept_explorer.concept_discovery_service import ConceptDiscoveryService
except ImportError as e:
    print(f"Staging manager import error: {e}")

# Initialize concept discovery service
try:
    concept_discovery_service = ConceptDiscoveryService()
    print("✅ Concept discovery service initialized")
except Exception as e:
    print(f"Concept discovery service initialization warning: {e}")
    concept_discovery_service = None

# Import efficient agent implementations
try:
    from agents.youtube_transcript.youtube_agent_efficient import EfficientYouTubeAgent
    from agents.scraper.high_performance_scraper import OptimizedScraperAgent
    # Create agent instances
    YouTubeAgent = EfficientYouTubeAgent
    ScraperAgent = OptimizedScraperAgent
    print("✅ High-performance scraper loaded successfully")
    print("✅ Efficient YouTube agent loaded successfully")
except ImportError as e:
    print(f"Agent import warning: {e}")
    # Create minimal fallback implementations
    class YouTubeAgent:
        async def extract_transcript(self, url, extract_metadata=True):
            return {"transcript": "YouTube agent not available", "success": False}
    
    class ScraperAgent:
        async def scrape_url(self, url):
            return {"content": "Scraper agent not available", "success": False}

router = APIRouter(prefix="/api/content", tags=["content_scraping"])

# Initialize managers
try:
    staging_manager = StagingManager()
except Exception as e:
    print(f"Error initializing staging manager: {e}")
    staging_manager = None


# Request models
class URLScrapeRequest(BaseModel):
    url: HttpUrl
    domain: Optional[str] = None
    priority: str = "medium"
    agent_pipeline: Optional[Dict[str, Any]] = None


class YouTubeScrapeRequest(BaseModel):
    youtube_url: HttpUrl
    domain: Optional[str] = "general"
    priority: str = "medium"
    extract_metadata: bool = True
    agent_pipeline: Optional[Dict[str, Any]] = None


class TextSubmissionRequest(BaseModel):
    text_content: str
    title: str
    author: Optional[str] = None
    domain: str = "general"
    priority: str = "medium"
    agent_pipeline: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    submission_id: str
    selected_agents: List[str]
    processing_order: str = "sequential"
    agent_parameters: Optional[Dict[str, Any]] = None


# Response models
class ScrapeResponse(BaseModel):
    submission_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    submission_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None


@router.post("/scrape/url", response_model=ScrapeResponse)
async def scrape_url(request: URLScrapeRequest, background_tasks: BackgroundTasks):
    """
    Submit URL for scraping and staging
    """
    try:
        # Validate URL
        url_str = str(request.url)
        
        # Create metadata
        metadata = ContentMetadata(
            title=f"Web content from {url_str}",
            author=None,
            date=None,
            domain=request.domain or "general",
            language="en",
            priority=Priority(request.priority),
            submitted_by="api_user",
            file_size=None,
            content_type="text/html"
        )
        
        # Create agent pipeline if provided
        agent_pipeline = None
        if request.agent_pipeline:
            agent_pipeline = AgentPipeline(
                selected_agents=request.agent_pipeline.get("agents", []),
                processing_order=request.agent_pipeline.get("order", "sequential"),
                agent_parameters=request.agent_pipeline.get("parameters", {}),
                completion_status={}
            )
        
        # Start background scraping task
        background_tasks.add_task(
            scrape_url_background,
            url_str,
            metadata,
            agent_pipeline
        )
        
        # Return immediate response
        submission_id = str(uuid.uuid4())
        
        return ScrapeResponse(
            submission_id=submission_id,
            status="submitted",
            message=f"URL scraping initiated for {url_str}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error submitting URL: {str(e)}")


@router.post("/scrape/youtube", response_model=ScrapeResponse)
async def scrape_youtube(request: YouTubeScrapeRequest, background_tasks: BackgroundTasks):
    """
    Submit YouTube URL for transcript extraction
    """
    try:
        # Validate YouTube URL
        youtube_url = str(request.youtube_url)
        if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Create metadata
        metadata = ContentMetadata(
            title=f"YouTube transcript from {youtube_url}",
            author=None,
            date=None,
            domain=request.domain,
            language="en",
            priority=Priority(request.priority),
            submitted_by="api_user",
            file_size=None,
            content_type="video/transcript"
        )
        
        # Create agent pipeline if provided
        agent_pipeline = None
        if request.agent_pipeline:
            agent_pipeline = AgentPipeline(
                selected_agents=request.agent_pipeline.get("agents", []),
                processing_order=request.agent_pipeline.get("order", "sequential"),
                agent_parameters=request.agent_pipeline.get("parameters", {}),
                completion_status={}
            )
        
        # Start background YouTube processing task
        background_tasks.add_task(
            scrape_youtube_background,
            youtube_url,
            metadata,
            agent_pipeline,
            request.extract_metadata
        )
        
        # Return immediate response
        submission_id = str(uuid.uuid4())
        
        return ScrapeResponse(
            submission_id=submission_id,
            status="submitted",
            message=f"YouTube transcript extraction initiated for {youtube_url}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error submitting YouTube URL: {str(e)}")


@router.post("/scrape/file", response_model=ScrapeResponse)
async def scrape_file(
    file: UploadFile = File(...),
    domain: str = Form("general"),
    priority: str = Form("medium"),
    agent_pipeline: Optional[str] = Form(None)
):
    """
    Upload file for processing
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 100MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        # Create metadata
        metadata = ContentMetadata(
            title=file.filename,
            author=None,
            date=None,
            domain=domain,
            language="en",
            priority=Priority(priority),
            submitted_by="api_user",
            file_size=file_size,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Parse agent pipeline if provided
        pipeline = None
        if agent_pipeline:
            try:
                pipeline_data = json.loads(agent_pipeline)
                pipeline = AgentPipeline(
                    selected_agents=pipeline_data.get("agents", []),
                    processing_order=pipeline_data.get("order", "sequential"),
                    agent_parameters=pipeline_data.get("parameters", {}),
                    completion_status={}
                )
            except json.JSONDecodeError:
                pass
        
        # Save file temporarily and process
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process file content based on type
        raw_content = ""
        file_ext = Path(file.filename).suffix.lower()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Use OCR for image files
                from agents.scraper.scraper_agent import OCRProcessor
                ocr_processor = OCRProcessor()
                raw_content = ocr_processor.extract_text_from_image(tmp_file_path, ocr_language or 'eng')
                
            elif file_ext == '.pdf':
                # Use PDF processor with OCR fallback
                from agents.scraper.scraper_agent import PDFProcessor
                pdf_processor = PDFProcessor()
                raw_content, _ = pdf_processor.extract_text_from_pdf(tmp_file_path)
                
            elif file_ext in ['.txt', '.md']:
                # Read text files directly
                raw_content = content.decode('utf-8', errors='ignore')
                
            else:
                # Default fallback for other file types
                raw_content = f"File content from: {file.filename} (Content extraction not implemented for this file type)"
                
        except Exception as processing_error:
            # Fallback to basic content decoding
            raw_content = content.decode('utf-8', errors='ignore') if file.content_type and 'text' in file.content_type else f"File content from: {file.filename} (Content extraction failed: {processing_error})"
        
        # Submit to staging
        submission_id = await staging_manager.submit_content(
            source_type=SourceType.UPLOAD,
            raw_content=raw_content,
            metadata=metadata,
            source_url=f"file://{file.filename}",
            agent_pipeline=pipeline
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return ScrapeResponse(
            submission_id=submission_id,
            status="submitted",
            message=f"File {file.filename} uploaded and staged successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading file: {str(e)}")


@router.post("/submit/text", response_model=ScrapeResponse)
async def submit_text(request: TextSubmissionRequest):
    """
    Submit text content for analysis
    """
    try:
        # Create metadata
        metadata = ContentMetadata(
            title=request.title,
            author=request.author,
            date=None,
            domain=request.domain,
            language="en",
            priority=Priority(request.priority),
            submitted_by="api_user",
            file_size=len(request.text_content.encode('utf-8')),
            content_type="text/plain"
        )
        
        # Create agent pipeline if provided
        agent_pipeline = None
        if request.agent_pipeline:
            agent_pipeline = AgentPipeline(
                selected_agents=request.agent_pipeline.get("agents", []),
                processing_order=request.agent_pipeline.get("order", "sequential"),
                agent_parameters=request.agent_pipeline.get("parameters", {}),
                completion_status={}
            )
        
        # Submit to staging
        submission_id = await staging_manager.submit_content(
            source_type=SourceType.TEXT,
            raw_content=request.text_content,
            metadata=metadata,
            agent_pipeline=agent_pipeline
        )
        
        return ScrapeResponse(
            submission_id=submission_id,
            status="submitted",
            message="Text content submitted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error submitting text: {str(e)}")


@router.get("/status/{submission_id}", response_model=StatusResponse)
async def get_scrape_status(submission_id: str):
    """
    Check processing status of submitted content
    """
    try:
        # Get content from staging manager
        content = await staging_manager.get_content(submission_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Prepare progress information
        progress = {
            "submitted": content.timestamps.get("submitted"),
            "analysis_started": content.timestamps.get("analysis_started"),
            "analysis_completed": content.timestamps.get("analysis_completed"),
            "reviewed": content.timestamps.get("reviewed"),
        }
        
        # Prepare results if available
        results = None
        if content.analysis_results:
            results = {
                "concepts_extracted": content.analysis_results.concepts_extracted,
                "claims_identified": content.analysis_results.claims_identified,
                "quality_score": content.analysis_results.quality_score,
                "confidence_level": content.analysis_results.confidence_level
            }
        
        return StatusResponse(
            submission_id=submission_id,
            status=content.processing_status.value,
            progress=progress,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.post("/analyze/run")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Trigger selected agents on staged content
    """
    try:
        # Get content
        content = await staging_manager.get_content(request.submission_id)
        if not content:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Update agent pipeline
        agent_pipeline = AgentPipeline(
            selected_agents=request.selected_agents,
            processing_order=request.processing_order,
            agent_parameters=request.agent_parameters or {},
            completion_status={}
        )
        
        # Start processing
        await staging_manager.start_processing(request.submission_id)
        
        # Start background analysis task
        background_tasks.add_task(
            run_analysis_background,
            request.submission_id,
            agent_pipeline
        )
        
        return JSONResponse(
            content={
                "submission_id": request.submission_id,
                "status": "analysis_started",
                "agents": request.selected_agents,
                "message": "Analysis pipeline started"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")


@router.get("/queue/stats")
async def get_queue_stats():
    """
    Get current queue statistics
    """
    try:
        stats = await staging_manager.get_queue_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting queue stats: {str(e)}")


# Background task functions
async def scrape_url_background(url: str, metadata: ContentMetadata, agent_pipeline: Optional[AgentPipeline]):
    """Background task for URL scraping"""
    try:
        # Initialize scraper agent
        scraper = ScraperAgent()
        
        # Scrape content
        scraped_content = await scraper.scrape_url(url)
        raw_content = scraped_content.get("content", "")
        
        # Perform concept discovery if service is available
        concept_discovery_result = None
        if concept_discovery_service and raw_content.strip():
            try:
                concept_discovery_result = await concept_discovery_service.discover_concepts_from_content(
                    content=raw_content,
                    source_document=url,
                    domain=metadata.domain,
                    include_hypotheses=True,
                    include_thought_paths=True
                )
                print(f"✅ Concept discovery completed: {len(concept_discovery_result.concepts)} concepts found")
            except Exception as cd_error:
                print(f"⚠️ Concept discovery failed: {cd_error}")
        
        # Submit to staging
        submission_id = await staging_manager.submit_content(
            source_type=SourceType.WEBSITE,
            raw_content=raw_content,
            metadata=metadata,
            source_url=url,
            agent_pipeline=agent_pipeline
        )
        
        # Store concept discovery results if available
        if concept_discovery_result:
            # This would integrate with the database sync system
            print(f"📊 Concepts for {submission_id}: {[c.name for c in concept_discovery_result.concepts[:5]]}")
        
        print(f"URL scraping completed: {submission_id}")
        
    except Exception as e:
        print(f"Error in URL scraping background task: {e}")


async def scrape_youtube_background(
    youtube_url: str, 
    metadata: ContentMetadata, 
    agent_pipeline: Optional[AgentPipeline],
    extract_metadata: bool
):
    """Background task for YouTube transcript extraction"""
    try:
        # Initialize YouTube agent
        youtube_agent = YouTubeAgent()
        
        # Extract transcript
        transcript_data = await youtube_agent.extract_transcript(youtube_url, extract_metadata)
        raw_transcript = transcript_data.get("transcript", "")
        
        # Update metadata with extracted info
        if extract_metadata and transcript_data.get("metadata"):
            video_metadata = transcript_data["metadata"]
            metadata.title = video_metadata.get("title", metadata.title)
            metadata.author = video_metadata.get("channel", metadata.author)
            metadata.date = video_metadata.get("publish_date", metadata.date)
        
        # Perform concept discovery on transcript if service is available
        concept_discovery_result = None
        if concept_discovery_service and raw_transcript.strip():
            try:
                concept_discovery_result = await concept_discovery_service.discover_concepts_from_content(
                    content=raw_transcript,
                    source_document=youtube_url,
                    domain=metadata.domain,
                    include_hypotheses=True,
                    include_thought_paths=True
                )
                print(f"✅ YouTube concept discovery completed: {len(concept_discovery_result.concepts)} concepts found")
            except Exception as cd_error:
                print(f"⚠️ YouTube concept discovery failed: {cd_error}")
        
        # Submit to staging
        submission_id = await staging_manager.submit_content(
            source_type=SourceType.YOUTUBE,
            raw_content=raw_transcript,
            metadata=metadata,
            source_url=youtube_url,
            agent_pipeline=agent_pipeline
        )
        
        # Store concept discovery results if available
        if concept_discovery_result:
            # This would integrate with the database sync system
            print(f"📺 YouTube concepts for {submission_id}: {[c.name for c in concept_discovery_result.concepts[:5]]}")
        
        print(f"YouTube transcript extraction completed: {submission_id}")
        
    except Exception as e:
        print(f"Error in YouTube background task: {e}")


async def run_analysis_background(submission_id: str, agent_pipeline: AgentPipeline):
    """Background task for running analysis pipeline"""
    try:
        # Simulate analysis (replace with actual agent calls)
        await asyncio.sleep(2)  # Simulate processing time
        
        # Create mock analysis results
        from data.staging_manager import AnalysisResults
        
        analysis_results = AnalysisResults(
            concepts_extracted=["concept1", "concept2", "concept3"],
            claims_identified=["claim1", "claim2"],
            connections_discovered=[{"source": "concept1", "target": "concept2", "type": "similarity"}],
            agent_recommendations={"recommendation": "High quality content"},
            quality_score=0.85,
            confidence_level="high"
        )
        
        # Complete analysis
        await staging_manager.complete_analysis(submission_id, analysis_results)
        
        print(f"Analysis completed for: {submission_id}")
        
    except Exception as e:
        print(f"Error in analysis background task: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for content scraping services"""
    try:
        stats = await staging_manager.get_queue_stats()
        
        return JSONResponse(
            content={
                "status": "healthy",
                "staging_manager": "operational",
                "queue_stats": stats,
                "timestamp": "2025-07-04T00:00:00Z"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-07-04T00:00:00Z"
            }
        )