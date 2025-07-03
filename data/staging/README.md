# JSON Staging System

This directory contains the structured workflow for content processing in MCP Yggdrasil.

## Directory Structure

- **pending/**: New content submissions awaiting processing
- **processing/**: Content currently being analyzed by agents
- **analyzed/**: Completed analysis results ready for review
- **approved/**: Content approved for database integration
- **rejected/**: Content rejected with reasons

## JSON Schema

Each staged content item follows this structure:

```json
{
  "submission_id": "uuid4-string",
  "source_type": "youtube|website|image|pdf|text",
  "source_url": "original_source_url",
  "metadata": {
    "title": "Content Title",
    "author": "Author Name",
    "date": "2024-01-15",
    "domain": "science|art|philosophy|mathematics|language|technology",
    "language": "en",
    "priority": "high|medium|low",
    "submitted_by": "user_id",
    "file_size": 1024,
    "content_type": "video|article|document|image"
  },
  "raw_content": "extracted_text_content",
  "processing_status": "pending|processing|analyzed|approved|rejected",
  "analysis_results": {
    "concepts_extracted": [],
    "claims_identified": [],
    "connections_discovered": [],
    "agent_recommendations": {},
    "quality_score": 0.85,
    "confidence_level": "high|medium|low"
  },
  "agent_pipeline": {
    "selected_agents": ["text_processor", "claim_analyzer"],
    "processing_order": "sequential|parallel",
    "agent_parameters": {},
    "completion_status": {}
  },
  "timestamps": {
    "submitted": "2024-01-15T10:30:00Z",
    "analysis_started": "2024-01-15T10:31:00Z",
    "analysis_completed": "2024-01-15T10:35:00Z",
    "reviewed": null,
    "approved_rejected": null
  },
  "review_data": {
    "reviewer": null,
    "review_notes": "",
    "approval_reason": "",
    "rejection_reason": ""
  }
}
```

## Workflow States

1. **Submission** → `pending/`
2. **Analysis** → `processing/`
3. **Review** → `analyzed/`
4. **Decision** → `approved/` or `rejected/`
5. **Integration** → Database sync

## File Naming Convention

Files are named using the submission ID: `{submission_id}.json`

Example: `a1b2c3d4-e5f6-7890-abcd-ef1234567890.json`