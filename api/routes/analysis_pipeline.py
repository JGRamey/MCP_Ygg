"""
Analysis Pipeline API Routes
FastAPI endpoints for managing analysis pipelines and workflow execution
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import managers and agents
try:
    from data.staging_manager import StagingManager, ProcessingStatus, Priority
    from agents.sync_manager.event_dispatcher import EventDispatcher
    from agents.text_processor.text_processor import TextProcessor
    from agents.claim_analyzer.claim_analyzer import ClaimAnalyzer
    from agents.vector_index.vector_indexer import VectorIndexer
except ImportError as e:
    logger.warning(f"Import warning: {e}")
    # Continue without agents for now
    # Create fallback enums
    from enum import Enum
    class ProcessingStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        ANALYZED = "analyzed"
        APPROVED = "approved"
        REJECTED = "rejected"
    
    class Priority(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

router = APIRouter(prefix="/api/analysis", tags=["analysis_pipeline"])

# Initialize managers
staging_manager = StagingManager()
event_dispatcher = EventDispatcher()

# Enums
class AnalysisMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY = "priority"

class AgentType(str, Enum):
    TEXT_PROCESSOR = "text_processor"
    CLAIM_ANALYZER = "claim_analyzer"
    VECTOR_INDEXER = "vector_indexer"
    CONCEPT_EXPLORER = "concept_explorer"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Request models
class AnalysisRequest(BaseModel):
    submission_id: str = Field(..., description="ID of staged content")
    selected_agents: List[AgentType] = Field(..., description="Agents to run")
    processing_mode: AnalysisMode = Field(AnalysisMode.SEQUENTIAL, description="Processing mode")
    agent_parameters: Optional[Dict[str, Any]] = Field(None, description="Agent-specific parameters")
    priority: int = Field(1, description="Processing priority (1-5)")
    timeout: int = Field(300, description="Timeout in seconds")

class WorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    pipeline_config: Dict[str, Any] = Field(..., description="Pipeline configuration")
    default_parameters: Optional[Dict[str, Any]] = Field(None, description="Default parameters")

class BatchAnalysisRequest(BaseModel):
    submission_ids: List[str] = Field(..., description="List of submission IDs")
    workflow_name: str = Field(..., description="Workflow to execute")
    batch_size: int = Field(10, description="Batch size for processing")
    max_concurrent: int = Field(3, description="Max concurrent jobs")

# Response models
class AnalysisResponse(BaseModel):
    job_id: str
    status: WorkflowStatus
    message: str
    estimated_duration: Optional[int] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    status: str
    created_at: datetime

class JobStatus(BaseModel):
    job_id: str
    status: WorkflowStatus
    progress: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agents_completed: List[str] = []
    agents_failed: List[str] = []

class BatchStatus(BaseModel):
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    running_jobs: int
    overall_status: WorkflowStatus
    job_statuses: List[JobStatus]

# Global tracking
active_jobs: Dict[str, JobStatus] = {}
workflows: Dict[str, Dict[str, Any]] = {}
batch_jobs: Dict[str, BatchStatus] = {}

# Core analysis endpoints
@router.post("/run", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Trigger selected agents on staged content
    """
    try:
        # Validate submission exists
        content = await staging_manager.get_content(request.submission_id)
        if not content:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Check if already processing
        if content.processing_status in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise HTTPException(
                status_code=400, 
                detail=f"Content already in status: {content.processing_status.value}"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        job_status = JobStatus(
            job_id=job_id,
            status=WorkflowStatus.PENDING,
            progress={"stage": "initializing", "percentage": 0},
            started_at=datetime.now(),
            agents_completed=[],
            agents_failed=[]
        )
        
        active_jobs[job_id] = job_status
        
        # Estimate duration based on agents
        agent_times = {
            AgentType.TEXT_PROCESSOR: 30,
            AgentType.CLAIM_ANALYZER: 60,
            AgentType.VECTOR_INDEXER: 45,
            AgentType.CONCEPT_EXPLORER: 90
        }
        
        estimated_duration = sum(agent_times.get(agent, 30) for agent in request.selected_agents)
        if request.processing_mode == AnalysisMode.PARALLEL:
            estimated_duration = max(agent_times.get(agent, 30) for agent in request.selected_agents)
        
        # Start background processing
        background_tasks.add_task(
            execute_analysis_pipeline,
            job_id,
            request.submission_id,
            request.selected_agents,
            request.processing_mode,
            request.agent_parameters or {},
            request.timeout
        )
        
        # Update staging manager
        await staging_manager.start_processing(request.submission_id)
        
        logger.info(f"Started analysis job {job_id} for submission {request.submission_id}")
        
        return AnalysisResponse(
            job_id=job_id,
            status=WorkflowStatus.PENDING,
            message=f"Analysis started with {len(request.selected_agents)} agents",
            estimated_duration=estimated_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis")

@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get detailed status of an analysis job
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of jobs")
):
    """
    List analysis jobs with optional filtering
    """
    jobs = list(active_jobs.values())
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by start time (newest first)
    jobs.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
    
    return jobs[:limit]

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running analysis job
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job.status = WorkflowStatus.CANCELLED
    job.completed_at = datetime.now()
    
    # Send cancellation event
    await event_dispatcher.dispatch_event({
        "type": "analysis.job.cancelled",
        "job_id": job_id,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"Cancelled analysis job {job_id}")
    
    return {"message": "Job cancelled successfully"}

# Workflow management endpoints
@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowRequest):
    """
    Create a new analysis workflow template
    """
    try:
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            "id": workflow_id,
            "name": request.name,
            "description": request.description,
            "pipeline_config": request.pipeline_config,
            "default_parameters": request.default_parameters or {},
            "created_at": datetime.now(),
            "status": "active"
        }
        
        workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id}: {request.name}")
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            name=request.name,
            status="active",
            created_at=workflow["created_at"]
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows():
    """
    List all available analysis workflows
    """
    workflow_list = []
    for workflow in workflows.values():
        workflow_list.append(WorkflowResponse(
            workflow_id=workflow["id"],
            name=workflow["name"],
            status=workflow["status"],
            created_at=workflow["created_at"]
        ))
    
    return workflow_list

@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    Get detailed workflow configuration
    """
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflows[workflow_id]

@router.put("/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, request: WorkflowRequest):
    """
    Update an existing workflow
    """
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflows[workflow_id]
    workflow.update({
        "name": request.name,
        "description": request.description,
        "pipeline_config": request.pipeline_config,
        "default_parameters": request.default_parameters or {},
        "updated_at": datetime.now()
    })
    
    logger.info(f"Updated workflow {workflow_id}: {request.name}")
    
    return {"message": "Workflow updated successfully"}

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow
    """
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del workflows[workflow_id]
    
    logger.info(f"Deleted workflow {workflow_id}")
    
    return {"message": "Workflow deleted successfully"}

# Batch processing endpoints
@router.post("/batch", response_model=BatchStatus)
async def run_batch_analysis(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Run analysis on multiple submissions in batches
    """
    try:
        # Validate workflow exists
        workflow = None
        for wf in workflows.values():
            if wf["name"] == request.workflow_name:
                workflow = wf
                break
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Validate all submissions exist
        invalid_submissions = []
        for submission_id in request.submission_ids:
            content = await staging_manager.get_content(submission_id)
            if not content:
                invalid_submissions.append(submission_id)
        
        if invalid_submissions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid submissions: {invalid_submissions}"
            )
        
        # Create batch job
        batch_id = str(uuid.uuid4())
        batch_status = BatchStatus(
            batch_id=batch_id,
            total_jobs=len(request.submission_ids),
            completed_jobs=0,
            failed_jobs=0,
            running_jobs=0,
            overall_status=WorkflowStatus.PENDING,
            job_statuses=[]
        )
        
        batch_jobs[batch_id] = batch_status
        
        # Start background batch processing
        background_tasks.add_task(
            execute_batch_analysis,
            batch_id,
            request.submission_ids,
            workflow,
            request.batch_size,
            request.max_concurrent
        )
        
        logger.info(f"Started batch analysis {batch_id} for {len(request.submission_ids)} submissions")
        
        return batch_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to start batch analysis")

@router.get("/batch/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """
    Get status of a batch analysis job
    """
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    return batch_jobs[batch_id]

@router.get("/batch", response_model=List[BatchStatus])
async def list_batch_jobs(limit: int = Query(20, description="Maximum number of batches")):
    """
    List batch analysis jobs
    """
    batches = list(batch_jobs.values())
    return batches[:limit]

# Enhanced processing queue endpoints
@router.post("/queue/process-priority")
async def process_priority_queue(max_items: int = 50):
    """
    Process queue items by priority (high -> medium -> low)
    """
    try:
        results = await staging_manager.process_priority_queue(max_items)
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Failed to process priority queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to process priority queue")

@router.get("/queue/priority-stats")
async def get_priority_queue_stats():
    """
    Get detailed queue statistics broken down by priority
    """
    try:
        stats = await staging_manager.get_priority_queue_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Failed to get priority stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get priority stats")

@router.get("/results/aggregate-by-domain")
async def aggregate_results_by_domain(status: str = "analyzed"):
    """
    Aggregate analysis results by domain
    """
    try:
        # Convert string to enum
        processing_status = ProcessingStatus(status)
        aggregates = await staging_manager.aggregate_results_by_domain(processing_status)
        return JSONResponse(content=aggregates)
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        raise HTTPException(status_code=500, detail="Failed to aggregate results")

@router.post("/batch/priority")
async def batch_process_by_priority(
    priority: str,
    background_tasks: BackgroundTasks,
    batch_size: int = 10
):
    """
    Process a batch of items by specific priority level
    """
    try:
        # Validate priority
        try:
            priority_enum = Priority(priority)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Mock processing function
        async def mock_batch_processor(item):
            await asyncio.sleep(0.1)  # Simulate processing
            return {"item_id": item.submission_id, "status": "processed"}
        
        # Start background processing
        background_tasks.add_task(
            process_batch_background,
            priority_enum,
            mock_batch_processor,
            batch_size
        )
        
        return JSONResponse(content={
            "message": f"Started batch processing for {priority} priority items",
            "priority": priority,
            "batch_size": batch_size,
            "status": "started"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start batch processing")

# Staging content management endpoints
@router.get("/staging/list")
async def list_staged_content(
    status: Optional[str] = Query(None, description="Filter by processing status"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    limit: int = Query(50, description="Maximum results")
):
    """
    List all staged content with filters
    """
    try:
        # Get all staged content
        content_list = await staging_manager.get_all_content()
        
        # Apply filters
        if status:
            content_list = [c for c in content_list if c.processing_status.value == status]
        
        if domain:
            content_list = [c for c in content_list if c.metadata.domain == domain]
        
        # Sort by submission time (newest first)
        content_list.sort(key=lambda x: x.timestamps.get("submitted", ""), reverse=True)
        
        # Format response
        results = []
        for content in content_list[:limit]:
            results.append({
                "submission_id": content.submission_id,
                "status": content.processing_status.value,
                "title": content.metadata.title,
                "domain": content.metadata.domain,
                "source_type": content.source_type.value,
                "submitted_at": content.timestamps.get("submitted"),
                "content_preview": content.raw_content[:200] + "..." if len(content.raw_content) > 200 else content.raw_content
            })
        
        return JSONResponse(content={"staged_content": results, "total": len(results)})
        
    except Exception as e:
        logger.error(f"Failed to list staged content: {e}")
        raise HTTPException(status_code=500, detail="Failed to list staged content")

@router.post("/staging/approve")
async def approve_content(
    submission_id: str,
    reviewer_notes: Optional[str] = None
):
    """
    Approve content for database integration
    """
    try:
        # Get content
        content = await staging_manager.get_content(submission_id)
        if not content:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Approve content
        await staging_manager.approve_content(submission_id, reviewer_notes)
        
        # Send approval event
        await event_dispatcher.dispatch_event({
            "type": "content.approved",
            "submission_id": submission_id,
            "reviewer_notes": reviewer_notes,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Approved content: {submission_id}")
        
        return JSONResponse(content={
            "submission_id": submission_id,
            "status": "approved",
            "message": "Content approved for database integration"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve content: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve content")

@router.post("/staging/reject")
async def reject_content(
    submission_id: str,
    rejection_reason: str,
    reviewer_notes: Optional[str] = None
):
    """
    Reject content with reason
    """
    try:
        # Get content
        content = await staging_manager.get_content(submission_id)
        if not content:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Reject content
        await staging_manager.reject_content(submission_id, rejection_reason, reviewer_notes)
        
        # Send rejection event
        await event_dispatcher.dispatch_event({
            "type": "content.rejected",
            "submission_id": submission_id,
            "rejection_reason": rejection_reason,
            "reviewer_notes": reviewer_notes,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Rejected content: {submission_id} - {rejection_reason}")
        
        return JSONResponse(content={
            "submission_id": submission_id,
            "status": "rejected",
            "reason": rejection_reason,
            "message": "Content rejected"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject content: {e}")
        raise HTTPException(status_code=500, detail="Failed to reject content")

# Monitoring and metrics endpoints
@router.get("/metrics")
async def get_pipeline_metrics():
    """
    Get analysis pipeline performance metrics
    """
    try:
        # Calculate metrics
        total_jobs = len(active_jobs)
        completed_jobs = len([j for j in active_jobs.values() if j.status == WorkflowStatus.COMPLETED])
        failed_jobs = len([j for j in active_jobs.values() if j.status == WorkflowStatus.FAILED])
        running_jobs = len([j for j in active_jobs.values() if j.status == WorkflowStatus.RUNNING])
        
        # Agent usage statistics
        agent_usage = {}
        for job in active_jobs.values():
            for agent in job.agents_completed:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        # Calculate average processing time
        completed_with_times = [
            j for j in active_jobs.values() 
            if j.status == WorkflowStatus.COMPLETED and j.started_at and j.completed_at
        ]
        
        avg_processing_time = None
        if completed_with_times:
            total_time = sum(
                (j.completed_at - j.started_at).total_seconds() 
                for j in completed_with_times
            )
            avg_processing_time = total_time / len(completed_with_times)
        
        # Get staging statistics
        staging_stats = await staging_manager.get_queue_stats()
        
        return JSONResponse(content={
            "analysis_pipeline": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
                "avg_processing_time": avg_processing_time
            },
            "agent_usage": agent_usage,
            "staging_stats": staging_stats,
            "workflows": {
                "total_workflows": len(workflows),
                "active_workflows": len([w for w in workflows.values() if w["status"] == "active"])
            },
            "batch_processing": {
                "total_batches": len(batch_jobs),
                "active_batches": len([b for b in batch_jobs.values() if b.overall_status == WorkflowStatus.RUNNING])
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Background task functions
async def execute_analysis_pipeline(
    job_id: str,
    submission_id: str,
    selected_agents: List[AgentType],
    processing_mode: AnalysisMode,
    agent_parameters: Dict[str, Any],
    timeout: int
):
    """Execute analysis pipeline in background"""
    job = active_jobs[job_id]
    
    try:
        job.status = WorkflowStatus.RUNNING
        job.progress = {"stage": "loading_content", "percentage": 10}
        
        # Get content
        content = await staging_manager.get_content(submission_id)
        if not content:
            raise Exception("Content not found")
        
        # Execute agents based on processing mode
        if processing_mode == AnalysisMode.PARALLEL:
            await execute_agents_parallel(job, content, selected_agents, agent_parameters)
        else:
            await execute_agents_sequential(job, content, selected_agents, agent_parameters)
        
        # Complete job
        job.status = WorkflowStatus.COMPLETED
        job.completed_at = datetime.now()
        job.progress = {"stage": "completed", "percentage": 100}
        
        # Complete analysis in staging manager
        await staging_manager.complete_analysis(submission_id, job.results)
        
        logger.info(f"Analysis job {job_id} completed successfully")
        
    except Exception as e:
        job.status = WorkflowStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        logger.error(f"Analysis job {job_id} failed: {e}")

async def execute_agents_sequential(job: JobStatus, content, selected_agents: List[AgentType], agent_parameters: Dict[str, Any]):
    """Execute agents in sequential order"""
    total_agents = len(selected_agents)
    
    for i, agent_type in enumerate(selected_agents):
        try:
            job.progress = {
                "stage": f"processing_{agent_type.value}",
                "percentage": 20 + (i * 60 // total_agents)
            }
            
            # Execute agent (mock implementation)
            await asyncio.sleep(2)  # Simulate processing time
            
            job.agents_completed.append(agent_type.value)
            
        except Exception as e:
            job.agents_failed.append(agent_type.value)
            logger.error(f"Agent {agent_type.value} failed: {e}")

async def execute_agents_parallel(job: JobStatus, content, selected_agents: List[AgentType], agent_parameters: Dict[str, Any]):
    """Execute agents in parallel"""
    job.progress = {"stage": "processing_parallel", "percentage": 20}
    
    # Create tasks for parallel execution
    tasks = []
    for agent_type in selected_agents:
        task = asyncio.create_task(execute_single_agent(agent_type, content, agent_parameters))
        tasks.append((agent_type, task))
    
    # Wait for all tasks to complete
    for agent_type, task in tasks:
        try:
            await task
            job.agents_completed.append(agent_type.value)
        except Exception as e:
            job.agents_failed.append(agent_type.value)
            logger.error(f"Agent {agent_type.value} failed: {e}")

async def execute_single_agent(agent_type: AgentType, content, agent_parameters: Dict[str, Any]):
    """Execute a single agent"""
    # Mock implementation - replace with actual agent calls
    await asyncio.sleep(2)  # Simulate processing time
    return {"agent": agent_type.value, "status": "completed"}

async def execute_batch_analysis(
    batch_id: str,
    submission_ids: List[str],
    workflow: Dict[str, Any],
    batch_size: int,
    max_concurrent: int
):
    """Execute batch analysis in background"""
    batch = batch_jobs[batch_id]
    
    try:
        batch.overall_status = WorkflowStatus.RUNNING
        
        # Process submissions in batches
        for i in range(0, len(submission_ids), batch_size):
            batch_submissions = submission_ids[i:i + batch_size]
            
            # Create semaphore for concurrent processing
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Process batch
            tasks = []
            for submission_id in batch_submissions:
                task = asyncio.create_task(
                    process_single_submission(semaphore, submission_id, workflow, batch)
                )
                tasks.append(task)
            
            # Wait for batch to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update final status
        if batch.failed_jobs > 0:
            batch.overall_status = WorkflowStatus.FAILED
        else:
            batch.overall_status = WorkflowStatus.COMPLETED
        
        logger.info(f"Batch analysis {batch_id} completed: {batch.completed_jobs}/{batch.total_jobs} successful")
        
    except Exception as e:
        batch.overall_status = WorkflowStatus.FAILED
        logger.error(f"Batch analysis {batch_id} failed: {e}")

async def process_single_submission(semaphore: asyncio.Semaphore, submission_id: str, workflow: Dict[str, Any], batch: BatchStatus):
    """Process a single submission in batch"""
    async with semaphore:
        try:
            batch.running_jobs += 1
            
            # Mock processing
            await asyncio.sleep(1)
            
            batch.running_jobs -= 1
            batch.completed_jobs += 1
            
        except Exception as e:
            batch.running_jobs -= 1
            batch.failed_jobs += 1
            logger.error(f"Failed to process submission {submission_id}: {e}")

async def process_batch_background(priority: Priority, processor_func, batch_size: int):
    """Process batch of items by priority in background"""
    try:
        results = await staging_manager.batch_process_by_priority(
            priority, processor_func, batch_size
        )
        logger.info(f"Batch processing completed for {priority.value}: {results['processed']} processed, {results['failed']} failed")
        
    except Exception as e:
        logger.error(f"Batch processing failed for {priority.value}: {e}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for analysis pipeline services"""
    try:
        return JSONResponse(content={
            "status": "healthy",
            "analysis_pipeline": "operational",
            "active_jobs": len(active_jobs),
            "active_workflows": len(workflows),
            "staging_manager": "operational",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )