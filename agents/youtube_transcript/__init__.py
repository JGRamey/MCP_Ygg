"""
YouTube Transcript Agent Package
YouTube video processing and transcript extraction for MCP Yggdrasil
"""

# Import efficient agent by default, with fallback to simple agent
try:
    from .youtube_agent_efficient import EfficientYouTubeAgent as YouTubeAgent
except ImportError:
    try:
        from .youtube_agent_simple import YouTubeAgent
    except ImportError:
        # Fallback class if no agents are available
        class YouTubeAgent:
            async def extract_transcript(self, url, extract_metadata=True):
                return {"transcript": "YouTube agent not available", "success": False}


# Optional imports for advanced functionality
try:
    from .transcript_processor import TranscriptProcessor
except ImportError:
    TranscriptProcessor = None

try:
    from .metadata_extractor import MetadataExtractor
except ImportError:
    MetadataExtractor = None

__all__ = ["YouTubeAgent", "TranscriptProcessor", "MetadataExtractor"]
