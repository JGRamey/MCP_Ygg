#!/usr/bin/env python3
"""
Concept Discovery API Routes
Enhanced concept discovery and knowledge graph integration endpoints
"""

import json
import logging

# Import concept discovery components
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import asyncio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from data.staging_manager import StagingManager

    from agents.concept_explorer.concept_discovery_service import (
        ConceptDiscoveryService,
        DiscoveryResult,
    )
    from agents.concept_explorer.concept_explorer import (
        ConceptHypothesis,
        ConceptNode,
        RelationshipEdge,
    )
except ImportError as e:
    logging.error(f"Import error in concept discovery routes: {e}")

router = APIRouter(prefix="/api/concept-discovery", tags=["concept-discovery"])
logger = logging.getLogger(__name__)

# Global service instance
concept_service = None


def get_concept_service():
    """Get or create concept discovery service instance"""
    global concept_service
    if concept_service is None:
        concept_service = ConceptDiscoveryService()
    return concept_service


# Request/Response Models
class ConceptDiscoveryRequest(BaseModel):
    content: str = Field(..., description="Text content to analyze")
    source_document: str = Field(..., description="Source document identifier")
    domain: Optional[str] = Field(None, description="Domain classification hint")
    include_hypotheses: bool = Field(True, description="Whether to generate hypotheses")
    include_thought_paths: bool = Field(
        True, description="Whether to generate thought paths"
    )
    analysis_depth: str = Field(
        "standard", description="Analysis depth: basic, standard, deep"
    )


class BatchDiscoveryRequest(BaseModel):
    documents: List[Dict[str, str]] = Field(
        ..., description="List of {content, source_document} pairs"
    )
    domain: Optional[str] = Field(None, description="Domain classification hint")
    include_cross_document_analysis: bool = Field(
        True, description="Include cross-document pattern analysis"
    )


class ConceptFilterRequest(BaseModel):
    domain: Optional[str] = None
    confidence_threshold: Optional[float] = 0.5
    limit: Optional[int] = 100
    include_relationships: bool = True


class HypothesisTestRequest(BaseModel):
    hypothesis_id: str
    test_criteria: List[str]
    evidence_sources: List[str]


class KnowledgeGraphExportRequest(BaseModel):
    discovery_ids: List[str]
    format_type: str = Field("neo4j", description="Export format: neo4j, gephi, json")
    include_hypotheses: bool = True
    include_metadata: bool = True


# API Endpoints


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_content(request: ConceptDiscoveryRequest):
    """
    Analyze content for concept discovery
    """
    try:
        service = get_concept_service()

        # Validate request
        if not request.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        # Perform concept discovery
        result = await service.discover_concepts_from_content(
            content=request.content,
            source_document=request.source_document,
            domain=request.domain,
            include_hypotheses=request.include_hypotheses,
            include_thought_paths=request.include_thought_paths,
        )

        # Convert result to JSON-serializable format
        response_data = {
            "discovery_id": result.discovery_id,
            "source_document": result.source_document,
            "timestamp": result.timestamp.isoformat(),
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "statistics": {
                "concepts_count": len(result.concepts),
                "relationships_count": len(result.relationships),
                "hypotheses_count": len(result.hypotheses),
                "thought_paths_count": len(result.thought_paths),
            },
            "concepts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "domain": c.domain,
                    "description": c.description,
                    "confidence": c.confidence,
                    "extraction_method": c.extraction_method,
                }
                for c in result.concepts
            ],
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.relationship_type,
                    "strength": r.strength,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                }
                for r in result.relationships
            ],
            "network_analysis": result.network_analysis,
            "temporal_evolution": result.temporal_evolution,
        }

        # Add hypotheses if requested
        if request.include_hypotheses:
            response_data["hypotheses"] = [
                {
                    "id": h.hypothesis_id,
                    "description": h.description,
                    "evidence_strength": h.evidence_strength,
                    "novelty_score": h.novelty_score,
                    "supporting_concepts": h.supporting_concepts,
                    "testable_predictions": h.testable_predictions,
                    "domain_bridge": h.domain_bridge,
                }
                for h in result.hypotheses
            ]

        # Add thought paths if requested
        if request.include_thought_paths:
            response_data["thought_paths"] = result.thought_paths

        return response_data

    except Exception as e:
        logger.error(f"Error in concept discovery analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-batch", response_model=Dict[str, Any])
async def analyze_batch(request: BatchDiscoveryRequest):
    """
    Analyze multiple documents for cross-document patterns
    """
    try:
        service = get_concept_service()

        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")

        # Prepare documents list
        documents = [
            (doc["content"], doc["source_document"]) for doc in request.documents
        ]

        # Perform cross-document analysis
        cross_doc_analysis = await service.discover_cross_document_patterns(
            documents=documents, domain=request.domain
        )

        # Individual document results
        individual_results = []
        for doc in request.documents:
            result = await service.discover_concepts_from_content(
                content=doc["content"],
                source_document=doc["source_document"],
                domain=request.domain,
                include_hypotheses=False,
                include_thought_paths=False,
            )

            individual_results.append(
                {
                    "discovery_id": result.discovery_id,
                    "source_document": result.source_document,
                    "concepts_count": len(result.concepts),
                    "relationships_count": len(result.relationships),
                    "confidence_score": result.confidence_score,
                }
            )

        response_data = {
            "batch_id": str(uuid.uuid4()),
            "processed_documents": len(request.documents),
            "individual_results": individual_results,
            "cross_document_analysis": cross_doc_analysis,
            "timestamp": datetime.now().isoformat(),
        }

        return response_data

    except Exception as e:
        logger.error(f"Error in batch concept discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.post("/upload-file")
async def upload_file_for_analysis(
    file: UploadFile = File(...),
    domain: Optional[str] = None,
    include_hypotheses: bool = True,
):
    """
    Upload a file for concept discovery analysis
    """
    try:
        # Validate file type
        allowed_types = [
            "text/plain",
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file.content_type}"
            )

        # Read file content
        content = await file.read()

        # Extract text based on file type
        if file.content_type == "text/plain":
            text_content = content.decode("utf-8")
        elif file.content_type == "application/pdf":
            # Would need PDF extraction library
            raise HTTPException(
                status_code=501, detail="PDF processing not implemented yet"
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Perform analysis
        service = get_concept_service()
        result = await service.discover_concepts_from_content(
            content=text_content,
            source_document=file.filename,
            domain=domain,
            include_hypotheses=include_hypotheses,
            include_thought_paths=True,
        )

        return {
            "discovery_id": result.discovery_id,
            "filename": file.filename,
            "file_size": len(content),
            "concepts_found": len(result.concepts),
            "relationships_found": len(result.relationships),
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time,
        }

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@router.get("/concepts", response_model=Dict[str, Any])
async def get_discovered_concepts(
    domain: Optional[str] = None, confidence_threshold: float = 0.5, limit: int = 100
):
    """
    Retrieve discovered concepts with filtering options
    """
    try:
        service = get_concept_service()

        # Get all discovered concepts
        all_concepts = list(service.discovered_concepts.values())

        # Apply filters
        filtered_concepts = []
        for concept in all_concepts:
            if domain and concept.domain != domain:
                continue
            if concept.confidence < confidence_threshold:
                continue
            filtered_concepts.append(concept)

        # Limit results
        filtered_concepts = filtered_concepts[:limit]

        # Format response
        concepts_data = [
            {
                "id": c.id,
                "name": c.name,
                "domain": c.domain,
                "description": c.description,
                "confidence": c.confidence,
                "source_documents": c.source_documents,
                "extraction_method": c.extraction_method,
            }
            for c in filtered_concepts
        ]

        return {
            "total_concepts": len(concepts_data),
            "filters_applied": {
                "domain": domain,
                "confidence_threshold": confidence_threshold,
                "limit": limit,
            },
            "concepts": concepts_data,
        }

    except Exception as e:
        logger.error(f"Error retrieving concepts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve concepts: {str(e)}"
        )


@router.get("/network-analysis", response_model=Dict[str, Any])
async def get_network_analysis():
    """
    Get comprehensive network analysis of the concept graph
    """
    try:
        service = get_concept_service()

        # Get network analysis
        network_analysis = await service.concept_explorer.analyze_concept_network()

        # Get thought path tracer statistics
        thought_tracer_stats = service.thought_path_tracer.get_reasoning_statistics()

        # Get reasoning patterns
        reasoning_patterns = (
            await service.thought_path_tracer.discover_reasoning_patterns(
                service.global_concept_graph
            )
        )

        return {
            "network_analysis": network_analysis,
            "reasoning_statistics": thought_tracer_stats,
            "reasoning_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "frequency": p.frequency,
                    "strength": p.pattern_strength,
                }
                for p in reasoning_patterns
            ],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in network analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Network analysis failed: {str(e)}"
        )


@router.post("/export-knowledge-graph")
async def export_knowledge_graph(request: KnowledgeGraphExportRequest):
    """
    Export concept discovery results as knowledge graph data
    """
    try:
        service = get_concept_service()

        # This would need to be implemented with actual discovery result storage
        # For now, return a mock response
        return {
            "export_id": str(uuid.uuid4()),
            "format": request.format_type,
            "discovery_ids": request.discovery_ids,
            "export_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "message": "Knowledge graph export feature to be implemented with discovery result storage",
        }

    except Exception as e:
        logger.error(f"Error exporting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/thought-paths/{start_concept}/{end_concept}")
async def trace_thought_paths(
    start_concept: str, end_concept: str, max_paths: int = 5, max_depth: int = 6
):
    """
    Trace thought paths between two concepts
    """
    try:
        service = get_concept_service()

        # Trace paths using the concept explorer
        paths = await service.concept_explorer.trace_thought_paths(
            start_concept=start_concept, end_concept=end_concept, max_depth=max_depth
        )

        # Format response
        paths_data = []
        for path in paths[:max_paths]:
            paths_data.append(
                {
                    "path_id": path.path_id,
                    "start_concept": path.start_concept,
                    "end_concept": path.end_concept,
                    "total_strength": path.total_strength,
                    "reasoning_chain": path.reasoning_chain,
                    "intermediate_steps": path.intermediate_steps,
                }
            )

        return {
            "start_concept": start_concept,
            "end_concept": end_concept,
            "paths_found": len(paths_data),
            "paths": paths_data,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error tracing thought paths: {e}")
        raise HTTPException(status_code=500, detail=f"Path tracing failed: {str(e)}")


@router.get("/hypotheses")
async def get_generated_hypotheses(
    domain: Optional[str] = None,
    min_evidence_strength: float = 0.4,
    min_novelty_score: float = 0.5,
    limit: int = 50,
):
    """
    Retrieve generated hypotheses with filtering
    """
    try:
        # This would need to be implemented with hypothesis storage
        # For now, return a mock response indicating the feature
        return {
            "total_hypotheses": 0,
            "filters_applied": {
                "domain": domain,
                "min_evidence_strength": min_evidence_strength,
                "min_novelty_score": min_novelty_score,
                "limit": limit,
            },
            "hypotheses": [],
            "message": "Hypothesis storage and retrieval to be implemented",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrieving hypotheses: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve hypotheses: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for concept discovery service
    """
    try:
        service = get_concept_service()

        # Basic health checks
        concept_count = len(service.discovered_concepts)
        graph_nodes = len(service.global_concept_graph.nodes)
        graph_edges = len(service.global_concept_graph.edges)

        return {
            "status": "healthy",
            "service": "concept_discovery",
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "discovered_concepts": concept_count,
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
            },
            "version": "1.0.0",
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# Additional utility endpoints


@router.delete("/reset")
async def reset_concept_graph():
    """
    Reset the concept graph (for development/testing)
    """
    try:
        global concept_service
        concept_service = ConceptDiscoveryService()  # Create fresh instance

        return {
            "status": "success",
            "message": "Concept graph reset successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error resetting concept graph: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@router.get("/statistics")
async def get_discovery_statistics():
    """
    Get comprehensive statistics about concept discovery
    """
    try:
        service = get_concept_service()

        # Gather statistics
        stats = {
            "total_concepts": len(service.discovered_concepts),
            "graph_statistics": {
                "nodes": len(service.global_concept_graph.nodes),
                "edges": len(service.global_concept_graph.edges),
                "density": 0.0,
                "clustering_coefficient": 0.0,
            },
            "domain_distribution": {},
            "extraction_methods": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate graph metrics if graph is not empty
        if len(service.global_concept_graph.nodes) > 0:
            import networkx as nx

            try:
                stats["graph_statistics"]["density"] = nx.density(
                    service.global_concept_graph
                )
                stats["graph_statistics"]["clustering_coefficient"] = (
                    nx.average_clustering(service.global_concept_graph)
                )
            except:
                pass  # Handle edge cases

        # Domain distribution
        for concept in service.discovered_concepts.values():
            domain = concept.domain
            stats["domain_distribution"][domain] = (
                stats["domain_distribution"].get(domain, 0) + 1
            )

            method = concept.extraction_method
            stats["extraction_methods"][method] = (
                stats["extraction_methods"].get(method, 0) + 1
            )

        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Statistics retrieval failed: {str(e)}"
        )
