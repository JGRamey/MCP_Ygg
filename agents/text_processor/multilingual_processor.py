#!/usr/bin/env python3
"""
Multilingual Text Processing Capabilities
Advanced multilingual support with transformers and spaCy models
"""

import logging
from typing import Tuple

import spacy
import torch
from langdetect import LangDetectException, detect_langs
from transformers import pipeline

logger = logging.getLogger(__name__)


class MultilingualProcessor:
    """Enhanced multilingual text processing"""

    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "pl": "Polish",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ar": "Arabic",
        }

        # Load spaCy models for supported languages
        self.nlp_models = {}
        self._load_spacy_models()

        # Initialize transformers with GPU support if available
        self.device = 0 if torch.cuda.is_available() else -1

        # Load transformer pipelines
        self._init_transformers()

    def _load_spacy_models(self):
        """Load available spaCy models"""
        model_mapping = {
            "en": "en_core_web_lg",
            "es": "es_core_news_lg",
            "fr": "fr_core_news_lg",
            "de": "de_core_news_lg",
            "it": "it_core_news_lg",
            "pt": "pt_core_news_lg",
            "nl": "nl_core_news_lg",
            "pl": "pl_core_news_lg",
            "ru": "ru_core_news_lg",
            "zh": "zh_core_web_lg",
            "ja": "ja_core_news_lg",
        }

        for lang_code, model_name in model_mapping.items():
            try:
                self.nlp_models[lang_code] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model for {lang_code}: {model_name}")
            except OSError:
                logger.warning(
                    f"spaCy model {model_name} not available for {lang_code}"
                )
                # Use English as fallback
                if lang_code != "en" and "en" in self.nlp_models:
                    self.nlp_models[lang_code] = self.nlp_models["en"]

    def _init_transformers(self):
        """Initialize transformer pipelines"""
        try:
            # Multilingual summarization
            self.summarizers = {
                "en": pipeline(
                    "summarization", model="facebook/bart-large-cnn", device=self.device
                ),
                "multilingual": pipeline(
                    "summarization",
                    model="csebuetnlp/mT5_multilingual_XLSum",
                    device=self.device,
                ),
            }

            # Sentiment analysis
            self.sentiment_analyzers = {
                "en": pipeline("sentiment-analysis", device=self.device),
                "multilingual": pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=self.device,
                ),
            }

            # Named Entity Recognition with aggregation
            self.ner_models = {
                "en": pipeline(
                    "ner", aggregation_strategy="simple", device=self.device
                ),
                "multilingual": pipeline(
                    "ner",
                    model="Davlan/bert-base-multilingual-cased-ner-hrl",
                    aggregation_strategy="simple",
                    device=self.device,
                ),
            }

            logger.info("Loaded transformer models successfully")

        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            # Initialize with None for graceful degradation
            self.summarizers = {}
            self.sentiment_analyzers = {}
            self.ner_models = {}

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        try:
            # Use langdetect for initial detection
            detections = detect_langs(text)
            if detections:
                lang = detections[0]
                lang_code = lang.lang
                confidence = lang.prob

                # Map to our supported languages
                if lang_code in self.supported_languages:
                    return lang_code, confidence
                else:
                    # Try to find closest match
                    for supported_code in self.supported_languages:
                        if lang_code.startswith(
                            supported_code
                        ) or supported_code.startswith(lang_code):
                            return (
                                supported_code,
                                confidence * 0.9,
                            )  # Slightly lower confidence

                    # Default to English if unsupported
                    return "en", confidence * 0.5

        except LangDetectException:
            logger.warning("Language detection failed, defaulting to English")

        return "en", 0.5

    def get_nlp_model(self, language: str) -> spacy.Language:
        """Get spaCy model for language"""
        return self.nlp_models.get(language, self.nlp_models.get("en"))

    def get_summarizer(self, language: str):
        """Get appropriate summarizer for language"""
        if language == "en" and "en" in self.summarizers:
            return self.summarizers["en"]
        elif "multilingual" in self.summarizers:
            return self.summarizers["multilingual"]
        return None

    def get_sentiment_analyzer(self, language: str):
        """Get appropriate sentiment analyzer for language"""
        if language == "en" and "en" in self.sentiment_analyzers:
            return self.sentiment_analyzers["en"]
        elif "multilingual" in self.sentiment_analyzers:
            return self.sentiment_analyzers["multilingual"]
        return None

    def get_ner_model(self, language: str):
        """Get appropriate NER model for language"""
        if language == "en" and "en" in self.ner_models:
            return self.ner_models["en"]
        elif "multilingual" in self.ner_models:
            return self.ner_models["multilingual"]
        return None
