#!/usr/bin/env python3
"""
Transcript Processor
Processes YouTube transcripts with cleaning, segmentation, and analysis
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Data structure for transcript segments"""

    start_time: float
    end_time: float
    text: str
    speaker: Optional[str]
    confidence: Optional[float]
    word_count: int
    is_auto_generated: bool


@dataclass
class ProcessedTranscript:
    """Data structure for processed transcript"""

    video_id: str
    language: str
    total_duration: float
    word_count: int
    segments: List[TranscriptSegment]
    full_text: str
    cleaned_text: str
    chapters: List[Dict[str, Any]]
    speakers: List[str]
    key_phrases: List[str]
    sentiment_score: Optional[float]
    processing_metadata: Dict[str, Any]


class TranscriptProcessor:
    """
    Processes YouTube transcripts with cleaning, segmentation, and analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize transcript processor"""
        self.config = config

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy English model not found. Some features will be limited."
            )
            self.nlp = None

        # Processing settings
        self.segment_duration = config.get("output", {}).get("segment_duration", 60)
        self.preserve_timestamps = config.get("processing", {}).get(
            "preserve_timestamps", True
        )

        logger.info("Transcript processor initialized")

    async def process_transcript(
        self, raw_transcript: List[Dict[str, Any]], video_id: str
    ) -> ProcessedTranscript:
        """
        Process raw YouTube transcript data
        """
        try:
            # Determine transcript language and type
            language = self._detect_language(raw_transcript)
            is_auto_generated = self._is_auto_generated(raw_transcript)

            # Clean and process transcript segments
            segments = self._process_segments(raw_transcript, is_auto_generated)

            # Calculate total duration
            total_duration = segments[-1].end_time if segments else 0

            # Generate full text
            full_text = " ".join([segment.text for segment in segments])

            # Clean text
            cleaned_text = self._clean_text(full_text)

            # Create time-based chapters
            chapters = self._create_chapters(segments, self.segment_duration)

            # Identify speakers (if available)
            speakers = self._identify_speakers(segments)

            # Extract key phrases
            key_phrases = await self._extract_key_phrases(cleaned_text)

            # Calculate sentiment (if NLP available)
            sentiment_score = self._calculate_sentiment(cleaned_text)

            # Calculate total word count
            word_count = sum(segment.word_count for segment in segments)

            # Processing metadata
            processing_metadata = {
                "processed_at": datetime.utcnow().isoformat(),
                "segment_count": len(segments),
                "is_auto_generated": is_auto_generated,
                "language_detected": language,
                "has_speakers": len(speakers) > 1,
                "confidence_avg": self._calculate_average_confidence(segments),
                "processing_method": "youtube_transcript_api",
            }

            return ProcessedTranscript(
                video_id=video_id,
                language=language,
                total_duration=total_duration,
                word_count=word_count,
                segments=segments,
                full_text=full_text,
                cleaned_text=cleaned_text,
                chapters=chapters,
                speakers=speakers,
                key_phrases=key_phrases,
                sentiment_score=sentiment_score,
                processing_metadata=processing_metadata,
            )

        except Exception as e:
            logger.error(f"Error processing transcript for {video_id}: {e}")
            raise

    def _process_segments(
        self, raw_transcript: List[Dict[str, Any]], is_auto_generated: bool
    ) -> List[TranscriptSegment]:
        """
        Process raw transcript segments
        """
        segments = []

        for entry in raw_transcript:
            start_time = entry.get("start", 0)
            duration = entry.get("duration", 0)
            end_time = start_time + duration
            text = entry.get("text", "").strip()

            if not text:
                continue

            # Clean text
            cleaned_text = self._clean_segment_text(text)

            # Count words
            word_count = len(cleaned_text.split())

            # Extract speaker info (if available)
            speaker = self._extract_speaker(text)

            # Confidence (usually not available from YouTube API)
            confidence = entry.get("confidence")

            segment = TranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                text=cleaned_text,
                speaker=speaker,
                confidence=confidence,
                word_count=word_count,
                is_auto_generated=is_auto_generated,
            )

            segments.append(segment)

        return segments

    def _clean_segment_text(self, text: str) -> str:
        """
        Clean individual segment text
        """
        # Remove common auto-caption artifacts
        text = re.sub(r"\[.*?\]", "", text)  # Remove [Music], [Applause], etc.
        text = re.sub(r"\(.*?\)", "", text)  # Remove (music), (applause), etc.

        # Fix common auto-caption issues
        text = re.sub(r"\s+", " ", text)  # Multiple spaces
        text = re.sub(r"([.!?])\s*([a-z])", r"\1 \2", text)  # Sentence spacing

        # Remove speaker patterns from auto-generated captions
        text = re.sub(r"^[A-Z][a-z]+:\s*", "", text)  # "Speaker: text"
        text = re.sub(r"^\>\>\s*", "", text)  # ">> text"

        return text.strip()

    def _clean_text(self, text: str) -> str:
        """
        Clean full transcript text
        """
        # Remove multiple spaces and newlines
        text = re.sub(r"\s+", " ", text)

        # Fix punctuation spacing
        text = re.sub(r"\s+([.!?,:;])", r"\1", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        # Remove repeated phrases (common in auto-generated captions)
        text = self._remove_repetitions(text)

        return text.strip()

    def _remove_repetitions(self, text: str) -> str:
        """
        Remove repetitive phrases common in auto-generated captions
        """
        words = text.split()
        if len(words) < 6:
            return text

        # Find and remove repetitive patterns
        cleaned_words = []
        i = 0

        while i < len(words):
            word = words[i]

            # Check for repetition patterns
            repetition_found = False
            for pattern_length in range(2, min(6, len(words) - i)):
                if i + pattern_length * 2 <= len(words):
                    pattern = words[i : i + pattern_length]
                    next_pattern = words[i + pattern_length : i + pattern_length * 2]

                    if pattern == next_pattern:
                        # Skip the repeated pattern
                        i += pattern_length
                        repetition_found = True
                        break

            if not repetition_found:
                cleaned_words.append(word)
                i += 1

        return " ".join(cleaned_words)

    def _extract_speaker(self, text: str) -> Optional[str]:
        """
        Extract speaker information from text
        """
        # Pattern matching for speaker indicators
        speaker_patterns = [
            r"^([A-Z][a-z]+):\s*",  # "Speaker: text"
            r"^\>\>\s*([A-Z][a-z]+)\s*:",  # ">> Speaker:"
            r"^\[([A-Z][a-z\s]+)\]",  # "[Speaker Name]"
        ]

        for pattern in speaker_patterns:
            match = re.match(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _detect_language(self, raw_transcript: List[Dict[str, Any]]) -> str:
        """
        Detect transcript language
        """
        # This would ideally use language detection, but for now return default
        # Could use langdetect library here
        return self.config.get("processing", {}).get("default_language", "en")

    def _is_auto_generated(self, raw_transcript: List[Dict[str, Any]]) -> bool:
        """
        Determine if transcript is auto-generated
        """
        # Check for common auto-generated indicators
        if not raw_transcript:
            return True

        sample_text = " ".join([entry.get("text", "") for entry in raw_transcript[:5]])

        # Auto-generated captions often have these characteristics
        auto_indicators = [
            r"\[.*?\]",  # [Music], [Applause]
            r"\(.*?\)",  # (music), (applause)
            r"^\>\>",  # >> at start
            r"[a-z][A-Z]",  # Missing spaces between sentences
        ]

        indicator_count = sum(
            1 for pattern in auto_indicators if re.search(pattern, sample_text)
        )

        return indicator_count >= 2

    def _create_chapters(
        self, segments: List[TranscriptSegment], duration: int
    ) -> List[Dict[str, Any]]:
        """
        Create time-based chapters from segments
        """
        chapters = []

        if not segments:
            return chapters

        current_start = 0
        chapter_num = 1

        while current_start < segments[-1].end_time:
            chapter_end = min(current_start + duration, segments[-1].end_time)

            # Find segments in this chapter
            chapter_segments = [
                s
                for s in segments
                if s.start_time >= current_start and s.start_time < chapter_end
            ]

            if chapter_segments:
                # Generate chapter title from first few words
                chapter_text = " ".join([s.text for s in chapter_segments[:3]])
                words = chapter_text.split()[:5]
                title = " ".join(words) + ("..." if len(words) == 5 else "")

                chapter = {
                    "chapter_number": chapter_num,
                    "start_time": current_start,
                    "end_time": chapter_end,
                    "duration": chapter_end - current_start,
                    "title": title,
                    "word_count": sum(s.word_count for s in chapter_segments),
                }

                chapters.append(chapter)
                chapter_num += 1

            current_start = chapter_end

        return chapters

    def _identify_speakers(self, segments: List[TranscriptSegment]) -> List[str]:
        """
        Identify unique speakers from segments
        """
        speakers = set()

        for segment in segments:
            if segment.speaker:
                speakers.add(segment.speaker)

        # If no explicit speakers found, try to infer from patterns
        if not speakers:
            # Could add more sophisticated speaker identification here
            speakers.add("Unknown Speaker")

        return sorted(list(speakers))

    async def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from transcript text
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text[:1000000])  # Limit text size for performance

            # Extract noun phrases
            noun_phrases = [
                chunk.text.lower()
                for chunk in doc.noun_chunks
                if len(chunk.text.split()) <= 3 and len(chunk.text) > 3
            ]

            # Extract named entities
            entities = [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]
            ]

            # Combine and count frequency
            all_phrases = noun_phrases + entities
            phrase_counts = Counter(all_phrases)

            # Return top phrases
            return [
                phrase for phrase, count in phrase_counts.most_common(20) if count >= 2
            ]

        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []

    def _calculate_sentiment(self, text: str) -> Optional[float]:
        """
        Calculate sentiment score for transcript
        """
        if not self.nlp:
            return None

        try:
            # Simple sentiment based on positive/negative words
            # In production, would use more sophisticated sentiment analysis
            doc = self.nlp(text[:100000])  # Limit for performance

            positive_words = {
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "awesome",
                "brilliant",
                "perfect",
                "love",
            }
            negative_words = {
                "bad",
                "terrible",
                "awful",
                "horrible",
                "hate",
                "disgusting",
                "worst",
                "stupid",
                "annoying",
                "boring",
            }

            tokens = [token.text.lower() for token in doc if not token.is_stop]

            positive_count = sum(1 for token in tokens if token in positive_words)
            negative_count = sum(1 for token in tokens if token in negative_words)

            total_sentiment_words = positive_count + negative_count

            if total_sentiment_words == 0:
                return 0.0  # Neutral

            return (positive_count - negative_count) / total_sentiment_words

        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return None

    def _calculate_average_confidence(
        self, segments: List[TranscriptSegment]
    ) -> Optional[float]:
        """
        Calculate average confidence score across segments
        """
        confidence_scores = [s.confidence for s in segments if s.confidence is not None]

        if not confidence_scores:
            return None

        return sum(confidence_scores) / len(confidence_scores)

    def format_transcript_for_display(
        self,
        transcript: ProcessedTranscript,
        include_timestamps: bool = True,
        include_speakers: bool = True,
    ) -> str:
        """
        Format transcript for display
        """
        lines = []

        for segment in transcript.segments:
            line_parts = []

            # Add timestamp
            if include_timestamps:
                start_time = self._format_timestamp(segment.start_time)
                line_parts.append(f"[{start_time}]")

            # Add speaker
            if include_speakers and segment.speaker:
                line_parts.append(f"{segment.speaker}:")

            # Add text
            line_parts.append(segment.text)

            lines.append(" ".join(line_parts))

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp as MM:SS or HH:MM:SS
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
