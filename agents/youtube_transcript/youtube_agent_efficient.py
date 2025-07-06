#!/usr/bin/env python3
"""
Efficient YouTube Agent
Combines yt-dlp with YouTube Transcript API for optimal transcript extraction
"""

import asyncio
import logging
import re
import os
import yaml
from typing import Dict, Optional, Any, List
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import json

# YouTube processing imports
try:
    import yt_dlp
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        from youtube_transcript_api._errors import TranscriptsRetrievalError
    except ImportError:
        # Fallback for different versions of youtube-transcript-api
        class TranscriptsRetrievalError(Exception):
            pass
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import isodate
    YT_AVAILABLE = True
except ImportError as e:
    print(f"YouTube dependencies not available: {e}")
    YT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EfficientYouTubeAgent:
    """
    Efficient YouTube Agent using yt-dlp + YouTube Transcript API
    Optimized for transcript extraction with minimal API calls
    """
    
    def __init__(self, config_path: str = "config/content_scraping.yaml"):
        """Initialize YouTube agent with configuration"""
        self.load_config(config_path)
        self.youtube_service = None
        
        # Initialize YouTube API if available
        api_key = self.config.get('scraping', {}).get('youtube', {}).get('api_key')
        if api_key and api_key != "${YOUTUBE_API_KEY}":
            try:
                self.youtube_service = build('youtube', 'v3', developerKey=api_key)
                logger.info("YouTube Data API initialized")
            except Exception as e:
                logger.warning(f"Could not initialize YouTube API: {e}")
        
        # Configure yt-dlp
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': False,  # We'll use transcript API instead
            'writeautomaticsub': False,
            'skip_download': True,  # Only extract metadata
        }
        
        logger.info("Efficient YouTube Agent initialized")
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'scraping': {
                    'youtube': {
                        'max_transcript_length': 50000,
                        'supported_languages': ['en', 'es', 'fr', 'de'],
                        'timeout': 30
                    }
                }
            }
            logger.warning(f"Config file {config_path} not found, using defaults")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata using yt-dlp (faster than API for basic info)"""
        if not YT_AVAILABLE:
            return {"error": "YouTube dependencies not available"}
        
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract metadata without downloading
                info = ydl.extract_info(video_url, download=False)
                
                metadata = {
                    'video_id': video_id,
                    'title': info.get('title', ''),
                    'channel': info.get('uploader', '') or info.get('channel', ''),
                    'channel_id': info.get('channel_id', ''),
                    'duration': info.get('duration', 0),  # seconds
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'description': info.get('description', ''),
                    'tags': info.get('tags', []),
                    'categories': info.get('categories', []),
                    'language': info.get('language', 'en'),
                    'availability': info.get('availability', 'public')
                }
                
                # Format upload date
                if metadata['upload_date']:
                    try:
                        date_str = metadata['upload_date']
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        metadata['publish_date'] = formatted_date
                    except:
                        metadata['publish_date'] = metadata['upload_date']
                
                # Format duration
                if metadata['duration']:
                    duration_str = f"{metadata['duration'] // 3600:02d}:{(metadata['duration'] % 3600) // 60:02d}:{metadata['duration'] % 60:02d}"
                    metadata['duration_formatted'] = duration_str
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata for {video_id}: {e}")
            return {"error": str(e)}
    
    async def get_transcript(self, video_id: str, language_preference: List[str] = None) -> Dict[str, Any]:
        """Get transcript using YouTube Transcript API (most reliable method)"""
        if not YT_AVAILABLE:
            return {"error": "YouTube dependencies not available"}
        
        try:
            # Get language preferences from config
            if not language_preference:
                language_preference = self.config.get('scraping', {}).get('youtube', {}).get('supported_languages', ['en'])
            
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find the best available transcript
            selected_transcript = None
            transcript_info = {
                'language': None,
                'is_generated': False,
                'is_translatable': False
            }
            
            # Priority 1: Manual transcript in preferred language
            for lang in language_preference:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    selected_transcript = transcript
                    transcript_info['language'] = lang
                    transcript_info['is_generated'] = False
                    break
                except:
                    continue
            
            # Priority 2: Auto-generated transcript in preferred language
            if not selected_transcript:
                for lang in language_preference:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        selected_transcript = transcript
                        transcript_info['language'] = lang
                        transcript_info['is_generated'] = True
                        break
                    except:
                        continue
            
            # Priority 3: Any available transcript (translatable)
            if not selected_transcript:
                try:
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0]
                        # Try to translate to preferred language
                        if hasattr(transcript, 'translate') and language_preference:
                            try:
                                selected_transcript = transcript.translate(language_preference[0])
                                transcript_info['language'] = language_preference[0]
                                transcript_info['is_translatable'] = True
                            except:
                                selected_transcript = transcript
                                transcript_info['language'] = transcript.language_code
                        else:
                            selected_transcript = transcript
                            transcript_info['language'] = transcript.language_code
                except Exception as e:
                    logger.error(f"Error finding any transcript: {e}")
            
            if not selected_transcript:
                return {
                    "error": "No transcript available",
                    "available_languages": [t.language_code for t in transcript_list] if 'transcript_list' in locals() else []
                }
            
            # Fetch transcript data
            transcript_data = selected_transcript.fetch()
            
            # Process transcript
            full_text = ""
            segments = []
            
            for segment in transcript_data:
                text = segment.get('text', '').strip()
                start_time = segment.get('start', 0)
                duration = segment.get('duration', 0)
                
                if text:
                    full_text += text + " "
                    segments.append({
                        'text': text,
                        'start': start_time,
                        'duration': duration,
                        'end': start_time + duration
                    })
            
            # Check length limits
            max_length = self.config.get('scraping', {}).get('youtube', {}).get('max_transcript_length', 50000)
            if len(full_text) > max_length:
                logger.warning(f"Transcript truncated from {len(full_text)} to {max_length} characters")
                full_text = full_text[:max_length] + "..."
            
            return {
                'transcript': full_text.strip(),
                'segments': segments,
                'language': transcript_info['language'],
                'is_generated': transcript_info['is_generated'],
                'is_translatable': transcript_info['is_translatable'],
                'word_count': len(full_text.split()),
                'duration_seconds': segments[-1]['end'] if segments else 0,
                'segment_count': len(segments)
            }
            
        except TranscriptsRetrievalError as e:
            logger.error(f"Transcript retrieval error for {video_id}: {e}")
            return {"error": f"Transcript not available: {str(e)}"}
        except Exception as e:
            logger.error(f"Error getting transcript for {video_id}: {e}")
            return {"error": str(e)}
    
    async def extract_transcript(self, youtube_url: str, extract_metadata: bool = True) -> Dict[str, Any]:
        """
        Main method: Extract transcript and metadata from YouTube URL
        Optimized for efficiency and reliability
        """
        try:
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                return {
                    "error": "Invalid YouTube URL",
                    "success": False
                }
            
            result = {
                "video_id": video_id,
                "url": youtube_url,
                "success": False
            }
            
            # Get transcript (primary goal)
            transcript_result = await self.get_transcript(video_id)
            
            if "error" not in transcript_result:
                result["transcript"] = transcript_result["transcript"]
                result["transcript_info"] = {
                    "language": transcript_result["language"],
                    "is_generated": transcript_result["is_generated"],
                    "word_count": transcript_result["word_count"],
                    "segment_count": transcript_result["segment_count"]
                }
                result["success"] = True
            else:
                result["transcript"] = ""
                result["transcript_error"] = transcript_result["error"]
            
            # Get metadata if requested
            if extract_metadata:
                metadata_result = await self.get_video_metadata(video_id)
                
                if "error" not in metadata_result:
                    result["metadata"] = {
                        "title": metadata_result["title"],
                        "channel": metadata_result["channel"],
                        "publish_date": metadata_result.get("publish_date", ""),
                        "duration": metadata_result.get("duration_formatted", ""),
                        "view_count": metadata_result["view_count"],
                        "description": metadata_result["description"][:500] + "..." if len(metadata_result["description"]) > 500 else metadata_result["description"]
                    }
                else:
                    result["metadata"] = {
                        "title": f"Video {video_id}",
                        "channel": "Unknown",
                        "publish_date": "",
                        "duration": ""
                    }
                    result["metadata_error"] = metadata_result["error"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL {youtube_url}: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def batch_extract_transcripts(self, urls: List[str], extract_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Extract transcripts from multiple YouTube URLs efficiently
        """
        results = []
        
        for url in urls:
            try:
                result = await self.extract_transcript(url, extract_metadata)
                results.append(result)
                
                # Rate limiting - be respectful
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append({
                    "url": url,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported transcript languages"""
        return self.config.get('scraping', {}).get('youtube', {}).get('supported_languages', ['en'])
    
    def is_available(self) -> bool:
        """Check if YouTube processing is available"""
        return YT_AVAILABLE


# Convenience function for API integration
async def extract_youtube_transcript(url: str, extract_metadata: bool = True) -> Dict[str, Any]:
    """
    Standalone function for extracting YouTube transcripts
    """
    agent = EfficientYouTubeAgent()
    return await agent.extract_transcript(url, extract_metadata)


# Example usage
async def main():
    """Example usage of the efficient YouTube agent"""
    agent = EfficientYouTubeAgent()
    
    if not agent.is_available():
        print("❌ YouTube processing dependencies not available")
        return
    
    # Test with a sample video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
    
    print(f"🎥 Testing YouTube transcript extraction...")
    print(f"📋 Supported languages: {agent.get_supported_languages()}")
    
    result = await agent.extract_transcript(test_url, extract_metadata=True)
    
    if result.get("success"):
        print(f"✅ Success! Extracted {result['transcript_info']['word_count']} words")
        print(f"📝 Language: {result['transcript_info']['language']}")
        print(f"🎬 Title: {result['metadata']['title']}")
        print(f"👤 Channel: {result['metadata']['channel']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())