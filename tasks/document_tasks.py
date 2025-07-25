"""
Document Processing Tasks
Async document processing using enhanced AI agents
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import asyncio

from .celery_config import CELERY_AVAILABLE, celery_app
from .models import DocumentProcessingTask, TaskStatus
from .progress_tracker import TaskProgressTracker

logger = logging.getLogger(__name__)

if CELERY_AVAILABLE:
    from celery import Task

    class CallbackTask(Task):
        """Task with progress callbacks"""

        def on_success(self, retval, task_id, args, kwargs):
            """Success callback"""
            logger.info(f"Task {task_id} completed successfully")
            # Could add webhook notifications here

        def on_failure(self, exc, task_id, args, kwargs, einfo):
            """Failure callback"""
            logger.error(f"Task {task_id} failed: {exc}")
            # Could add error reporting here


@celery_app.task(bind=True, base=CallbackTask if CELERY_AVAILABLE else None)
def process_documents_task(self, documents: List[Dict], options: Dict = None) -> Dict:
    """Process multiple documents asynchronously"""
    if not CELERY_AVAILABLE:
        logger.warning("Celery not available, running synchronously")
        return _process_documents_sync(documents, options or {})

    task_id = self.request.id
    progress = TaskProgressTracker(task_id)

    total_docs = len(documents)
    processed = 0
    errors = []
    results = []

    try:
        # Update initial progress
        progress.update(
            current=0,
            total=total_docs,
            status=TaskStatus.STARTED,
            message="Starting document processing",
        )

        for i, doc in enumerate(documents):
            try:
                # Update progress
                progress.update(
                    current=i,
                    total=total_docs,
                    status=TaskStatus.PROGRESS,
                    message=f"Processing: {doc.get('title', f'Document {i+1}')}",
                )

                # Process single document
                result = _process_single_document(doc, options or {})
                results.append(result)
                processed += 1

            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                errors.append(
                    {
                        "document_index": i,
                        "document_id": doc.get("id", f"doc_{i}"),
                        "error": str(e),
                    }
                )

        # Final update
        final_result = {
            "total": total_docs,
            "processed": processed,
            "errors": len(errors),
            "error_details": errors,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }

        progress.complete(
            status=TaskStatus.SUCCESS,
            result=final_result,
            message=f"Processed {processed}/{total_docs} documents",
        )

        return final_result

    except Exception as e:
        logger.error(f"Document processing task failed: {e}")
        progress.error(str(e))
        raise


@celery_app.task(rate_limit="20/m")
def process_single_document_task(document: Dict, options: Dict = None) -> Dict:
    """Process a single document"""
    try:
        return _process_single_document(document, options or {})
    except Exception as e:
        logger.error(f"Single document processing failed: {e}")
        raise


def _process_single_document(document: Dict, options: Dict) -> Dict:
    """Internal document processing logic"""
    try:
        # Import enhanced agents
        from agents.claim_analyzer.claim_analyzer import ClaimAnalyzerAgent
        from agents.qdrant_manager.vector_index.enhanced_indexer import (
            EnhancedVectorIndexer,
        )
        from agents.text_processor.enhanced_text_processor import EnhancedTextProcessor

        # Initialize processors
        text_processor = EnhancedTextProcessor()
        claim_analyzer = ClaimAnalyzerAgent()
        vector_indexer = EnhancedVectorIndexer()

        result = {
            "document_id": document.get("id", "unknown"),
            "title": document.get("title", "Untitled"),
            "processed_at": datetime.utcnow().isoformat(),
            "processing_steps": [],
        }

        text_content = document.get("content", document.get("text", ""))

        # Step 1: Enhanced text processing
        if options.get("enable_text_processing", True):
            try:
                processed_text = asyncio.run(text_processor.process_text(text_content))
                result["text_analysis"] = {
                    "language": processed_text.language,
                    "entities": len(processed_text.entities),
                    "concepts": len(processed_text.concepts),
                    "summary": processed_text.summary,
                    "sentiment": processed_text.sentiment,
                }
                result["processing_steps"].append("text_processing")
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Text processing: {str(e)}")

        # Step 2: Claim analysis
        if options.get("enable_claim_analysis", True):
            try:
                # Extract and analyze claims
                claims = asyncio.run(
                    claim_analyzer.extract_and_analyze_claims(text_content)
                )
                result["claim_analysis"] = {
                    "claims_found": len(claims),
                    "claims": [claim.dict() for claim in claims[:5]],  # Limit results
                }
                result["processing_steps"].append("claim_analysis")
            except Exception as e:
                logger.error(f"Claim analysis failed: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Claim analysis: {str(e)}")

        # Step 3: Vector indexing
        if options.get("enable_vector_indexing", True):
            try:
                vector_result = asyncio.run(
                    vector_indexer.index_content(
                        {
                            "id": document.get("id"),
                            "text": text_content,
                            "domain": document.get("domain", "general"),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )
                result["vector_indexing"] = {
                    "vector_id": vector_result.vector_id,
                    "model_used": vector_result.model_used,
                    "quality_score": vector_result.quality_score,
                }
                result["processing_steps"].append("vector_indexing")
            except Exception as e:
                logger.error(f"Vector indexing failed: {e}")
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Vector indexing: {str(e)}")

        return result

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return {
            "document_id": document.get("id", "unknown"),
            "error": str(e),
            "processed_at": datetime.utcnow().isoformat(),
        }


def _process_documents_sync(documents: List[Dict], options: Dict) -> Dict:
    """Synchronous fallback for document processing"""
    processed = 0
    errors = []
    results = []

    for i, doc in enumerate(documents):
        try:
            result = _process_single_document(doc, options)
            results.append(result)
            processed += 1
        except Exception as e:
            errors.append({"document_index": i, "error": str(e)})

    return {
        "total": len(documents),
        "processed": processed,
        "errors": len(errors),
        "error_details": errors,
        "results": results,
        "mode": "synchronous_fallback",
    }
