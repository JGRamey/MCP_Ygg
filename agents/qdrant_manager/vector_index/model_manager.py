#!/usr/bin/env python3
"""
Model Manager for Enhanced Vector Indexer
Handles dynamic model selection and performance tracking
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ModelConfig, ModelPerformance
from .utils import (
    TRANSFORMERS_AVAILABLE,
    PerformanceTimer,
    calculate_processing_speed,
    get_model_configs,
    get_selection_criteria_weights,
    normalize_language_code,
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages multiple embedding models and selects optimal ones"""

    def __init__(self):
        self.models = {}
        self.model_performances = {}
        self.model_configs = {}
        self.selection_weights = get_selection_criteria_weights()
        self.load_models()

    def load_models(self) -> None:
        """Load and initialize embedding models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Advanced models not available - using mock models")
            self._setup_mock_models()
            return

        # Load model configurations
        model_configs = get_model_configs()

        # Load available models
        for model_key, config in model_configs.items():
            try:
                logger.info(f"Loading model: {config['name']}")

                # Import here to avoid issues if not available
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(config["name"])

                # Store model and config
                self.models[model_key] = {
                    "model": model,
                    "config": config,
                    "loaded_at": datetime.now(),
                }

                self.model_configs[model_key] = ModelConfig(
                    name=config["name"],
                    model_path=config["name"],
                    strengths=config["strengths"],
                    languages=config["languages"],
                    vector_size=config["vector_size"],
                    memory_requirement=config["memory_requirement"],
                    preferred_domains=config["preferred_domains"],
                )

                # Initialize performance metrics
                self.model_performances[model_key] = ModelPerformance(
                    model_name=config["name"],
                    accuracy=0.85,  # Will be updated with actual benchmarks
                    speed=100.0,  # Will be measured
                    quality_score=0.8,
                    memory_usage=200.0,
                    vector_size=config["vector_size"],
                    supported_languages=config["languages"],
                    last_evaluated=datetime.now(),
                )

                logger.info(f"Successfully loaded model: {model_key}")

            except Exception as e:
                logger.warning(f"Failed to load model {model_key}: {e}")
                continue

        if not self.models:
            logger.error("No models loaded successfully - falling back to mock models")
            self._setup_mock_models()

    def _setup_mock_models(self) -> None:
        """Setup mock models for testing when transformers unavailable"""
        self.models = {
            "general": {
                "model": "mock_general",
                "config": {
                    "name": "mock_general_model",
                    "strengths": ["general_purpose"],
                    "languages": ["en"],
                    "vector_size": 384,
                },
                "loaded_at": datetime.now(),
            }
        }

        self.model_configs = {
            "general": ModelConfig(
                name="mock_general_model",
                model_path="mock_general_model",
                strengths=["general_purpose"],
                languages=["en"],
                vector_size=384,
                memory_requirement="low",
                preferred_domains=["general"],
            )
        }

        self.model_performances = {
            "general": ModelPerformance(
                model_name="mock_general_model",
                accuracy=0.75,
                speed=1000.0,
                quality_score=0.7,
                memory_usage=50.0,
                vector_size=384,
                supported_languages=["en"],
                last_evaluated=datetime.now(),
            )
        }

    def select_optimal_model(self, content: Dict[str, Any]) -> str:
        """Dynamically select the best model for given content"""
        if not self.models:
            return "general"  # Fallback

        # Extract content characteristics
        language = normalize_language_code(content.get("language", "en"))
        domain = content.get("domain", "general").lower()
        text_length = len(content.get("text", ""))
        content_type = content.get("type", "general")

        # Scoring function for model selection
        best_model = None
        best_score = -1

        for model_key, model_info in self.models.items():
            score = self._calculate_model_score(
                model_key, language, domain, text_length, content_type
            )

            # Update best model if this scores higher
            if score > best_score:
                best_score = score
                best_model = model_key

        selected_model = best_model or "general"
        logger.debug(
            f"Selected model '{selected_model}' with score {best_score:.2f} for content: {content.get('title', 'unknown')}"
        )
        return selected_model

    def _calculate_model_score(
        self,
        model_key: str,
        language: str,
        domain: str,
        text_length: int,
        content_type: str,
    ) -> float:
        """Calculate score for model selection"""
        config = self.model_configs[model_key]
        performance = self.model_performances[model_key]

        score = 0.0

        # Language support scoring
        if config.supports_language(language):
            score += self.selection_weights["language_support"]
        elif "multilingual" in config.strengths:
            score += self.selection_weights["language_support"] * 0.7
        else:
            score += self.selection_weights["language_support"] * 0.2

        # Domain-specific scoring
        if domain in ["science", "technology", "mathematics"]:
            if "academic" in config.strengths or "scientific" in config.strengths:
                score += self.selection_weights["domain_expertise"]
            elif "semantic_quality" in config.strengths:
                score += self.selection_weights["domain_expertise"] * 0.75

        if domain in ["philosophy", "literature", "art"]:
            if "semantic_quality" in config.strengths:
                score += self.selection_weights["domain_expertise"]
            elif "general_purpose" in config.strengths:
                score += self.selection_weights["domain_expertise"] * 0.5

        # Performance-based scoring
        score += performance.quality_score * self.selection_weights["quality_score"]
        score += (performance.accuracy * self.selection_weights["quality_score"]) * 0.5

        # Text length considerations
        if text_length > 5000:  # Long documents
            if "speed" in config.strengths:
                score += self.selection_weights["processing_speed"]
        elif text_length < 500:  # Short documents
            if "semantic_quality" in config.strengths:
                score += self.selection_weights["processing_speed"] * 0.5

        # Memory efficiency consideration
        if config.memory_requirement == "low":
            score += self.selection_weights["memory_efficiency"]
        elif config.memory_requirement == "medium":
            score += self.selection_weights["memory_efficiency"] * 0.7
        # High memory requirement gets no bonus

        return score

    def get_model(self, model_key: str):
        """Get model instance"""
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not found, using general")
            model_key = list(self.models.keys())[0]  # Use first available model

        return self.models[model_key]["model"]

    def get_alternative_model(self, current_model: str) -> str:
        """Get alternative model for quality improvement"""
        # Model preference order for alternatives
        alternatives = {
            "general": "semantic",
            "semantic": "multilingual",
            "multilingual": "general",
            "fast": "general",
            "academic": "semantic",
        }

        alternative = alternatives.get(current_model, "general")

        # Check if alternative is available
        if alternative in self.models:
            return alternative

        # Return first available model that's different from current
        available_models = [m for m in self.models.keys() if m != current_model]
        return available_models[0] if available_models else current_model

    def benchmark_model(
        self, model_key: str, test_texts: List[str]
    ) -> ModelPerformance:
        """Benchmark model performance"""
        if model_key not in self.models or not TRANSFORMERS_AVAILABLE:
            return self.model_performances.get(
                model_key, self.model_performances["general"]
            )

        model = self.models[model_key]["model"]

        try:
            with PerformanceTimer(f"Benchmarking {model_key}") as timer:
                # Generate embeddings for benchmark
                if hasattr(model, "encode"):
                    embeddings = model.encode(test_texts, show_progress_bar=False)
                else:
                    # Mock benchmark
                    embeddings = np.random.rand(len(test_texts), 384)

            # Calculate metrics
            processing_time = timer.get_duration()
            speed = len(test_texts) / processing_time if processing_time > 0 else 0

            # Quality assessment (simplified)
            if TRANSFORMERS_AVAILABLE and len(embeddings) > 0:
                embedding_array = np.array(embeddings)
                avg_norm = np.mean(np.linalg.norm(embedding_array, axis=1))
                quality_score = min(avg_norm / 10.0, 1.0)  # Normalized quality
            else:
                quality_score = 0.7  # Default for mock

            # Update performance record
            performance = ModelPerformance(
                model_name=self.model_configs[model_key].name,
                accuracy=0.85,  # Would need labeled data for real accuracy
                speed=speed,
                quality_score=quality_score,
                memory_usage=200.0,  # Would measure actual memory
                vector_size=len(embeddings[0]) if len(embeddings) > 0 else 384,
                supported_languages=self.model_configs[model_key].languages,
                last_evaluated=datetime.now(),
            )

            self.model_performances[model_key] = performance
            logger.info(
                f"Benchmarked {model_key}: speed={speed:.1f} docs/sec, quality={quality_score:.2f}"
            )
            return performance

        except Exception as e:
            logger.error(f"Error benchmarking model {model_key}: {e}")
            return self.model_performances.get(
                model_key, self.model_performances["general"]
            )

    def update_model_performance(self, model_key: str, feedback_score: float) -> None:
        """Update model performance based on user feedback"""
        if model_key in self.model_performances:
            performance = self.model_performances[model_key]
            # Simple learning: adjust accuracy based on feedback
            performance.accuracy = performance.accuracy * 0.9 + feedback_score * 0.1
            performance.last_evaluated = datetime.now()
            logger.debug(
                f"Updated {model_key} performance based on feedback: {feedback_score}"
            )

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models"""
        stats = {}

        for model_key, performance in self.model_performances.items():
            stats[model_key] = performance.to_dict()

        return stats

    def get_usage_distribution(self) -> Dict[str, float]:
        """Get distribution of model usage (simplified)"""
        # This would track actual usage in a real implementation
        total_models = len(self.models)
        if total_models == 0:
            return {}

        # Mock distribution based on model characteristics
        distribution = {}
        for model_key in self.models.keys():
            if "general" in model_key:
                distribution[model_key] = 0.4
            elif "multilingual" in model_key:
                distribution[model_key] = 0.3
            elif "semantic" in model_key:
                distribution[model_key] = 0.2
            else:
                distribution[model_key] = 0.1

        # Normalize to sum to 1.0
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        return distribution
