#!/usr/bin/env python3
"""
YouTube Agent
Core YouTube API integration for video processing and transcript extraction
"""

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import aiohttp
import asyncio
import isodate
import youtube_dl
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import TranscriptsRetrievalError, YouTubeTranscriptApi

from .metadata_extractor import MetadataExtractor
from .transcript_processor import TranscriptProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class YouTubeVideo:
    """Data structure for YouTube video information"""

    video_id: str
    title: str
    channel_name: str
    channel_id: str
    duration: str
    published_at: str
    description: str
    view_count: int
    like_count: Optional[int]
    comment_count: Optional[int]
    language: Optional[str]
    captions_available: bool
    transcript_languages: List[str]
    chapters: List[Dict[str, Any]]
    tags: List[str]
    category: str


@dataclass
class ProcessingResult:
    """Result structure for video processing"""

    video_info: YouTubeVideo
    transcript: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str]


class YouTubeAgent:
    """
    YouTube Agent for video processing and transcript extraction
    """

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """Initialize YouTube Agent"""
        self.config = config
        self.api_key = api_key or config.get("api", {}).get("key")

        # Initialize YouTube API service
        if self.api_key:
            self.youtube_service = build("youtube", "v3", developerKey=self.api_key)
        else:
            self.youtube_service = None
            logger.warning("No YouTube API key provided - limited functionality")

        # Initialize processors
        self.transcript_processor = TranscriptProcessor(config)
        self.metadata_extractor = MetadataExtractor(config)

        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit = config.get("rate_limiting", {})

        # Cache
        self.cache = {}
        self.cache_config = config.get("cache", {})

        logger.info("YouTube Agent initialized")

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats
        """
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
            r"youtube\.com\/v\/([^&\n?#]+)",
            r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def extract_playlist_id(self, url: str) -> Optional[str]:
        """
        Extract playlist ID from YouTube URL
        """
        pattern = r"[?&]list=([^&\n?#]+)"
        match = re.search(pattern, url)
        return match.group(1) if match else None

    async def get_video_info(self, video_id: str) -> Optional[YouTubeVideo]:
        """
        Get comprehensive video information from YouTube API
        """
        if not self.youtube_service:
            logger.error("YouTube API service not available")
            return None

        try:
            # Rate limiting
            await self._enforce_rate_limit()

            # Check cache first
            cache_key = f"video_info_{video_id}"
            if self._is_cached(cache_key):
                return self._get_from_cache(cache_key)

            # Get video details
            video_response = (
                self.youtube_service.videos()
                .list(part="snippet,statistics,contentDetails,status", id=video_id)
                .execute()
            )

            if not video_response.get("items"):
                logger.error(f"Video not found: {video_id}")
                return None

            video_data = video_response["items"][0]
            snippet = video_data["snippet"]
            statistics = video_data.get("statistics", {})
            content_details = video_data["contentDetails"]

            # Parse duration
            duration_iso = content_details.get("duration", "PT0S")
            duration_seconds = isodate.parse_duration(duration_iso).total_seconds()

            # Check duration limits
            max_duration = self.config.get("video", {}).get("max_duration", 14400)
            min_duration = self.config.get("video", {}).get("min_duration", 30)

            if duration_seconds > max_duration:
                logger.warning(f"Video too long: {duration_seconds}s > {max_duration}s")
                return None

            if duration_seconds < min_duration:
                logger.warning(
                    f"Video too short: {duration_seconds}s < {min_duration}s"
                )
                return None

            # Get available transcript languages
            transcript_languages = await self._get_transcript_languages(video_id)

            # Extract chapters if available
            chapters = await self._extract_chapters(
                video_id, snippet.get("description", "")
            )

            # Create video object
            video_info = YouTubeVideo(
                video_id=video_id,
                title=snippet.get("title", ""),
                channel_name=snippet.get("channelTitle", ""),
                channel_id=snippet.get("channelId", ""),
                duration=str(int(duration_seconds)),
                published_at=snippet.get("publishedAt", ""),
                description=snippet.get("description", ""),
                view_count=int(statistics.get("viewCount", 0)),
                like_count=(
                    int(statistics.get("likeCount", 0))
                    if statistics.get("likeCount")
                    else None
                ),
                comment_count=(
                    int(statistics.get("commentCount", 0))
                    if statistics.get("commentCount")
                    else None
                ),
                language=snippet.get("defaultLanguage")
                or snippet.get("defaultAudioLanguage"),
                captions_available=len(transcript_languages) > 0,
                transcript_languages=transcript_languages,
                chapters=chapters,
                tags=snippet.get("tags", []),
                category=snippet.get("categoryId", "Unknown"),
            )

            # Cache result
            self._cache_result(cache_key, video_info)

            return video_info

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    async def _get_transcript_languages(self, video_id: str) -> List[str]:
        """
        Get available transcript languages for a video
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = []

            for transcript in transcript_list:
                languages.append(transcript.language_code)

            return languages

        except TranscriptsRetrievalError:
            return []
        except Exception as e:
            logger.error(f"Error getting transcript languages: {e}")
            return []

    async def _extract_chapters(
        self, video_id: str, description: str
    ) -> List[Dict[str, Any]]:
        """
        Extract chapter information from video description
        """
        chapters = []

        # Common chapter patterns
        patterns = [
            r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(.+?)(?=\n|\d{1,2}:\d{2}|$)",
            r"(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)(?=\n|\d{1,2}:\d{2}|$)",
            r"Chapter \d+:?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]?\s*(.+?)(?=\n|Chapter|$)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, description, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                timestamp = match.group(1)
                title = match.group(2).strip()

                # Convert timestamp to seconds
                time_parts = timestamp.split(":")
                if len(time_parts) == 3:
                    seconds = (
                        int(time_parts[0]) * 3600
                        + int(time_parts[1]) * 60
                        + int(time_parts[2])
                    )
                else:
                    seconds = int(time_parts[0]) * 60 + int(time_parts[1])

                chapters.append(
                    {"timestamp": timestamp, "start_seconds": seconds, "title": title}
                )

            if chapters:
                break  # Use first successful pattern

        # Sort chapters by start time
        chapters.sort(key=lambda x: x["start_seconds"])

        # Add end times
        for i, chapter in enumerate(chapters):
            if i < len(chapters) - 1:
                chapter["end_seconds"] = chapters[i + 1]["start_seconds"]
            else:
                chapter["end_seconds"] = None  # Will be set to video duration

        return chapters

    async def get_transcript(
        self, video_id: str, language: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get transcript for a video in specified language
        """
        try:
            # Check cache first
            cache_key = f"transcript_{video_id}_{language or 'auto'}"
            if self._is_cached(cache_key):
                return self._get_from_cache(cache_key)

            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Determine which transcript to use
            target_language = language or self.config.get("processing", {}).get(
                "default_language", "en"
            )
            selected_transcript = None

            # Try to find exact language match
            try:
                selected_transcript = transcript_list.find_transcript([target_language])
            except:
                pass

            # If no exact match, try auto-generated
            if not selected_transcript:
                try:
                    selected_transcript = transcript_list.find_generated_transcript(
                        [target_language]
                    )
                except:
                    pass

            # If still no match, get any available transcript
            if not selected_transcript:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    selected_transcript = available_transcripts[0]

                    # Try to translate if auto_translate is enabled
                    if self.config.get("processing", {}).get("auto_translate", True):
                        try:
                            selected_transcript = selected_transcript.translate(
                                target_language
                            )
                        except:
                            pass

            if not selected_transcript:
                logger.warning(f"No transcript available for video: {video_id}")
                return None

            # Fetch transcript
            transcript_data = selected_transcript.fetch()

            # Process transcript
            processed_transcript = await self.transcript_processor.process_transcript(
                transcript_data, video_id
            )

            # Cache result
            self._cache_result(cache_key, processed_transcript)

            return processed_transcript

        except TranscriptsRetrievalError as e:
            logger.error(f"Transcript retrieval error for {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting transcript for {video_id}: {e}")
            return None

    async def process_video(
        self, url: str, language: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single YouTube video
        """
        start_time = time.time()

        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                return ProcessingResult(
                    video_info=None,
                    transcript=None,
                    metadata={},
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid YouTube URL",
                )

            # Get video information
            video_info = await self.get_video_info(video_id)
            if not video_info:
                return ProcessingResult(
                    video_info=None,
                    transcript=None,
                    metadata={},
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Could not retrieve video information",
                )

            # Get transcript
            transcript = await self.get_transcript(video_id, language)

            # Extract additional metadata
            metadata = await self.metadata_extractor.extract_metadata(
                video_info, transcript
            )

            return ProcessingResult(
                video_info=video_info,
                transcript=transcript,
                metadata=metadata,
                processing_time=time.time() - start_time,
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Error processing video {url}: {e}")
            return ProcessingResult(
                video_info=None,
                transcript=None,
                metadata={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    async def process_playlist(
        self, url: str, language: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Process all videos in a YouTube playlist
        """
        playlist_id = self.extract_playlist_id(url)
        if not playlist_id:
            logger.error("Invalid playlist URL")
            return []

        try:
            # Get playlist videos
            video_ids = await self._get_playlist_videos(playlist_id)

            # Process videos in batches
            batch_size = self.config.get("playlist", {}).get("batch_size", 10)
            delay = self.config.get("playlist", {}).get("delay_between_requests", 1)

            results = []
            for i in range(0, len(video_ids), batch_size):
                batch = video_ids[i : i + batch_size]

                # Process batch
                batch_tasks = []
                for video_id in batch:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    batch_tasks.append(self.process_video(video_url, language))

                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Filter out exceptions
                for result in batch_results:
                    if isinstance(result, ProcessingResult):
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {result}")

                # Delay between batches
                if i + batch_size < len(video_ids):
                    await asyncio.sleep(delay)

            return results

        except Exception as e:
            logger.error(f"Error processing playlist: {e}")
            return []

    async def _get_playlist_videos(self, playlist_id: str) -> List[str]:
        """
        Get all video IDs from a playlist
        """
        if not self.youtube_service:
            logger.error("YouTube API service not available")
            return []

        video_ids = []
        next_page_token = None
        max_videos = self.config.get("playlist", {}).get("max_videos", 100)

        try:
            while len(video_ids) < max_videos:
                await self._enforce_rate_limit()

                request = self.youtube_service.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_videos - len(video_ids)),
                    pageToken=next_page_token,
                )

                response = request.execute()

                for item in response.get("items", []):
                    video_id = item["contentDetails"]["videoId"]
                    video_ids.append(video_id)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            return video_ids

        except HttpError as e:
            logger.error(f"Error getting playlist videos: {e}")
            return []

    async def _enforce_rate_limit(self):
        """
        Enforce rate limiting for API requests
        """
        now = time.time()
        requests_per_minute = self.rate_limit.get("requests_per_minute", 100)

        # Reset counter if minute has passed
        if now - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = now

        # Check if we've exceeded rate limit
        if self.request_count >= requests_per_minute:
            sleep_time = 60 - (now - self.last_request_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()

        self.request_count += 1

    def _is_cached(self, key: str) -> bool:
        """
        Check if result is cached and still valid
        """
        if not self.cache_config.get("enabled", True):
            return False

        if key not in self.cache:
            return False

        cache_entry = self.cache[key]
        ttl = self.cache_config.get("ttl", 86400)

        return time.time() - cache_entry["timestamp"] < ttl

    def _get_from_cache(self, key: str) -> Any:
        """
        Get result from cache
        """
        return self.cache[key]["data"]

    def _cache_result(self, key: str, data: Any):
        """
        Cache result with timestamp
        """
        if not self.cache_config.get("enabled", True):
            return

        max_size = self.cache_config.get("max_size", 1000)

        # Clean cache if too large
        if len(self.cache) >= max_size:
            # Remove oldest entries
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            for i in range(len(sorted_cache) // 2):
                del self.cache[sorted_cache[i][0]]

        self.cache[key] = {"data": data, "timestamp": time.time()}

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics
        """
        return {
            "cache_size": len(self.cache),
            "request_count": self.request_count,
            "last_request_time": self.last_request_time,
            "api_available": self.youtube_service is not None,
        }
