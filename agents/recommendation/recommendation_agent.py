"""
Recommendation Agent for MCP Server
Suggests related documents, concepts, and nodes based on user queries and graph analysis.
"""

import json
import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import asyncio
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationType(Enum):
    """Types of recommendations."""

    SIMILAR_CONTENT = "similar_content"
    RELATED_CONCEPTS = "related_concepts"
    TEMPORAL_RELATED = "temporal_related"
    AUTHORITY_BASED = "authority_based"
    COLLABORATIVE = "collaborative"
    CROSS_DOMAIN = "cross_domain"
    PATHWAY = "pathway"


class RecommendationReason(Enum):
    """Reasons for recommendations."""

    CONTENT_SIMILARITY = "content_similarity"
    GRAPH_PROXIMITY = "graph_proximity"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    DOMAIN_EXPERTISE = "domain_expertise"
    USER_BEHAVIOR = "user_behavior"
    CITATION_NETWORK = "citation_network"
    CONCEPT_OVERLAP = "concept_overlap"


@dataclass
class Recommendation:
    """Represents a single recommendation."""

    id: str
    target_node_id: str
    target_type: str
    title: str
    description: str
    recommendation_type: RecommendationType
    reason: RecommendationReason
    confidence_score: float
    relevance_score: float
    details: Dict[str, Any]
    generated_at: datetime
    features: Dict[str, float]


@dataclass
class RecommendationQuery:
    """Represents a recommendation query."""

    node_id: Optional[str] = None
    content: Optional[str] = None
    domain: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10
    include_types: Optional[List[RecommendationType]] = None
    exclude_types: Optional[List[RecommendationType]] = None
    min_confidence: float = 0.1


class RecommendationConfig:
    """Configuration for recommendation system."""

    def __init__(self, config_path: str = "agents/recommendation/config.py"):
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333

        # Algorithm parameters
        self.max_features_tfidf = 10000
        self.svd_components = 100
        self.kmeans_clusters = 50
        self.pagerank_alpha = 0.85
        self.pagerank_max_iter = 100
        self.similarity_threshold = 0.1
        self.temporal_weight = 0.2
        self.authority_weight = 0.3
        self.content_weight = 0.5

        # Graph traversal parameters
        self.max_graph_depth = 3
        self.max_recommendations_per_type = 20
        self.diversity_threshold = 0.8

        # Caching
        self.cache_embeddings = True
        self.cache_graph_metrics = True
        self.cache_duration_hours = 24

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from file if it exists."""
        try:
            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")


class RecommendationEngine:
    """Main recommendation engine."""

    def __init__(self, config: Optional[RecommendationConfig] = None):
        """Initialize the recommendation engine."""
        self.config = config or RecommendationConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None

        # ML models and data structures
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features_tfidf,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.svd = TruncatedSVD(n_components=self.config.svd_components)
        self.kmeans = KMeans(n_clusters=self.config.kmeans_clusters, random_state=42)

        # Cached data
        self.graph_cache: Optional[nx.Graph] = None
        self.content_matrix: Optional[np.ndarray] = None
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.pagerank_scores: Dict[str, float] = {}
        self.user_interactions: Dict[str, List[str]] = defaultdict(list)

        # Storage
        self.recommendations_cache: Dict[str, List[Recommendation]] = {}

        # Set up logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("recommendation_engine")
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    async def initialize(self) -> None:
        """Initialize database connections and load cached data."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()

            # Initialize Qdrant client
            self.qdrant_client = AsyncQdrantClient(
                host=self.config.qdrant_host, port=self.config.qdrant_port
            )

            # Test Qdrant connection
            await self.qdrant_client.get_collections()

            # Load cached data
            await self._load_graph_cache()
            await self._load_embeddings_cache()
            await self._compute_graph_metrics()

            self.logger.info("Recommendation engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize recommendation engine: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            await self.qdrant_client.close()
        self.logger.info("Recommendation engine closed")

    async def get_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on query parameters."""
        recommendations = []

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query)

            # Check cache first
            if cache_key in self.recommendations_cache:
                self.logger.info(f"Returning cached recommendations for query")
                return self.recommendations_cache[cache_key]

            # Generate recommendations using different strategies
            if (
                not query.include_types
                or RecommendationType.SIMILAR_CONTENT in query.include_types
            ):
                content_recs = await self._get_content_based_recommendations(query)
                recommendations.extend(content_recs)

            if (
                not query.include_types
                or RecommendationType.AUTHORITY_BASED in query.include_types
            ):
                authority_recs = await self._get_authority_based_recommendations(query)
                recommendations.extend(authority_recs)

            if (
                not query.include_types
                or RecommendationType.RELATED_CONCEPTS in query.include_types
            ):
                concept_recs = await self._get_concept_based_recommendations(query)
                recommendations.extend(concept_recs)

            if (
                not query.include_types
                or RecommendationType.TEMPORAL_RELATED in query.include_types
            ):
                temporal_recs = await self._get_temporal_recommendations(query)
                recommendations.extend(temporal_recs)

            if (
                not query.include_types
                or RecommendationType.COLLABORATIVE in query.include_types
            ):
                collaborative_recs = await self._get_collaborative_recommendations(
                    query
                )
                recommendations.extend(collaborative_recs)

            if (
                not query.include_types
                or RecommendationType.CROSS_DOMAIN in query.include_types
            ):
                cross_domain_recs = await self._get_cross_domain_recommendations(query)
                recommendations.extend(cross_domain_recs)

            if (
                not query.include_types
                or RecommendationType.PATHWAY in query.include_types
            ):
                pathway_recs = await self._get_pathway_recommendations(query)
                recommendations.extend(pathway_recs)

            # Filter and sort recommendations
            filtered_recs = self._filter_and_rank_recommendations(
                recommendations, query
            )

            # Cache results
            self.recommendations_cache[cache_key] = filtered_recs

            self.logger.info(f"Generated {len(filtered_recs)} recommendations")
            return filtered_recs

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

    def _generate_cache_key(self, query: RecommendationQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            f"node_{query.node_id}",
            f"domain_{query.domain}",
            f"user_{query.user_id}",
            f"limit_{query.limit}",
            f"types_{sorted([t.value for t in query.include_types]) if query.include_types else 'all'}",
        ]
        return "_".join(key_parts)

    async def _load_graph_cache(self) -> None:
        """Load graph data into NetworkX for analysis."""
        async with self.neo4j_driver.session() as session:
            # Get nodes
            nodes_query = """
            MATCH (n)
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                n.title as title,
                n.domain as domain,
                n.date as date,
                n.word_count as word_count
            """
            nodes_result = await session.run(nodes_query)

            # Get relationships
            rels_query = """
            MATCH (a)-[r]->(b)
            RETURN 
                id(a) as source,
                id(b) as target,
                type(r) as rel_type,
                r.weight as weight
            """
            rels_result = await session.run(rels_query)

            # Build NetworkX graph
            self.graph_cache = nx.DiGraph()

            # Add nodes
            async for record in nodes_result:
                node_data = dict(record)
                self.graph_cache.add_node(
                    node_data["node_id"],
                    **{k: v for k, v in node_data.items() if k != "node_id"},
                )

            # Add edges
            async for record in rels_result:
                rel_data = dict(record)
                weight = rel_data.get("weight", 1.0)
                self.graph_cache.add_edge(
                    rel_data["source"],
                    rel_data["target"],
                    weight=weight,
                    rel_type=rel_data["rel_type"],
                )

        self.logger.info(
            f"Loaded graph with {len(self.graph_cache.nodes)} nodes and {len(self.graph_cache.edges)} edges"
        )

    async def _load_embeddings_cache(self) -> None:
        """Load embeddings from Qdrant."""
        try:
            collections = await self.qdrant_client.get_collections()

            for collection in collections.collections:
                collection_name = collection.name

                # Get all points
                points, _ = await self.qdrant_client.scroll(
                    collection_name=collection_name,
                    with_payload=True,
                    with_vectors=True,
                    limit=10000,  # Adjust based on your data size
                )

                for point in points:
                    if point.payload and "node_id" in point.payload:
                        node_id = str(point.payload["node_id"])
                        self.node_embeddings[node_id] = np.array(point.vector)

            self.logger.info(f"Loaded {len(self.node_embeddings)} node embeddings")

        except Exception as e:
            self.logger.warning(f"Could not load embeddings: {e}")

    async def _compute_graph_metrics(self) -> None:
        """Compute graph metrics like PageRank."""
        if not self.graph_cache:
            return

        try:
            # Compute PageRank
            pagerank = nx.pagerank(
                self.graph_cache,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
            )
            self.pagerank_scores = pagerank

            self.logger.info("Computed graph metrics")

        except Exception as e:
            self.logger.warning(f"Could not compute graph metrics: {e}")

    async def _get_content_based_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on content similarity."""
        recommendations = []

        if not query.node_id or query.node_id not in self.node_embeddings:
            return recommendations

        try:
            query_embedding = self.node_embeddings[query.node_id]
            similarities = []

            for node_id, embedding in self.node_embeddings.items():
                if node_id != query.node_id:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1), embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((node_id, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Create recommendations
            for node_id, similarity in similarities[
                : self.config.max_recommendations_per_type
            ]:
                if similarity > self.config.similarity_threshold:
                    node_data = self.graph_cache.nodes.get(node_id, {})

                    recommendation = Recommendation(
                        id=f"content_{query.node_id}_{node_id}",
                        target_node_id=node_id,
                        target_type=node_data.get("labels", ["Unknown"])[0],
                        title=node_data.get("title", "Unknown"),
                        description=f"Similar content based on embeddings",
                        recommendation_type=RecommendationType.SIMILAR_CONTENT,
                        reason=RecommendationReason.CONTENT_SIMILARITY,
                        confidence_score=similarity,
                        relevance_score=similarity * self.config.content_weight,
                        details={"similarity_score": similarity},
                        generated_at=datetime.now(timezone.utc),
                        features={"content_similarity": similarity},
                    )
                    recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {e}")

        return recommendations

    async def _get_authority_based_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on node authority (PageRank)."""
        recommendations = []

        if not self.pagerank_scores:
            return recommendations

        try:
            # Filter nodes by domain if specified
            domain_filter = query.domain
            filtered_nodes = []

            for node_id, score in self.pagerank_scores.items():
                node_data = self.graph_cache.nodes.get(node_id, {})
                if not domain_filter or node_data.get("domain") == domain_filter:
                    filtered_nodes.append((node_id, score))

            # Sort by PageRank score
            filtered_nodes.sort(key=lambda x: x[1], reverse=True)

            # Create recommendations
            for node_id, score in filtered_nodes[
                : self.config.max_recommendations_per_type
            ]:
                if query.node_id and node_id == query.node_id:
                    continue

                node_data = self.graph_cache.nodes.get(node_id, {})

                recommendation = Recommendation(
                    id=f"authority_{node_id}",
                    target_node_id=node_id,
                    target_type=node_data.get("labels", ["Unknown"])[0],
                    title=node_data.get("title", "Unknown"),
                    description=f"Authoritative source (PageRank: {score:.3f})",
                    recommendation_type=RecommendationType.AUTHORITY_BASED,
                    reason=RecommendationReason.DOMAIN_EXPERTISE,
                    confidence_score=score,
                    relevance_score=score * self.config.authority_weight,
                    details={"pagerank_score": score},
                    generated_at=datetime.now(timezone.utc),
                    features={"authority_score": score},
                )
                recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in authority-based recommendations: {e}")

        return recommendations

    async def _get_concept_based_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on shared concepts."""
        recommendations = []

        if not query.node_id or not self.graph_cache:
            return recommendations

        try:
            # Find concepts connected to the query node
            query_concepts = set()
            if query.node_id in self.graph_cache:
                for neighbor in self.graph_cache.neighbors(query.node_id):
                    neighbor_data = self.graph_cache.nodes.get(neighbor, {})
                    if "Concept" in neighbor_data.get("labels", []):
                        query_concepts.add(neighbor)

            if not query_concepts:
                return recommendations

            # Find other nodes connected to these concepts
            concept_connections = defaultdict(set)
            for concept in query_concepts:
                for neighbor in self.graph_cache.neighbors(concept):
                    if neighbor != query.node_id:
                        concept_connections[neighbor].add(concept)

            # Score by number of shared concepts
            for node_id, shared_concepts in concept_connections.items():
                if len(shared_concepts) > 0:
                    node_data = self.graph_cache.nodes.get(node_id, {})
                    overlap_score = len(shared_concepts) / len(query_concepts)

                    recommendation = Recommendation(
                        id=f"concept_{query.node_id}_{node_id}",
                        target_node_id=node_id,
                        target_type=node_data.get("labels", ["Unknown"])[0],
                        title=node_data.get("title", "Unknown"),
                        description=f"Shares {len(shared_concepts)} concepts",
                        recommendation_type=RecommendationType.RELATED_CONCEPTS,
                        reason=RecommendationReason.CONCEPT_OVERLAP,
                        confidence_score=overlap_score,
                        relevance_score=overlap_score,
                        details={
                            "shared_concepts_count": len(shared_concepts),
                            "total_query_concepts": len(query_concepts),
                        },
                        generated_at=datetime.now(timezone.utc),
                        features={"concept_overlap": overlap_score},
                    )
                    recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in concept-based recommendations: {e}")

        return recommendations

    async def _get_temporal_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on temporal proximity."""
        recommendations = []

        if not query.node_id or not self.graph_cache:
            return recommendations

        try:
            query_node_data = self.graph_cache.nodes.get(query.node_id, {})
            query_date = query_node_data.get("date")

            if not query_date:
                return recommendations

            # Find nodes with similar dates
            temporal_candidates = []
            for node_id, node_data in self.graph_cache.nodes(data=True):
                if node_id == query.node_id:
                    continue

                node_date = node_data.get("date")
                if node_date:
                    try:
                        # Calculate temporal proximity (assuming dates are comparable)
                        time_diff = (
                            abs((query_date - node_date).days)
                            if hasattr(query_date - node_date, "days")
                            else 0
                        )
                        proximity_score = 1.0 / (
                            1.0 + time_diff / 365.25
                        )  # Decay over years

                        if proximity_score > 0.1:  # Minimum threshold
                            temporal_candidates.append(
                                (node_id, proximity_score, node_data)
                            )

                    except (TypeError, AttributeError):
                        continue

            # Sort by temporal proximity
            temporal_candidates.sort(key=lambda x: x[1], reverse=True)

            # Create recommendations
            for node_id, proximity_score, node_data in temporal_candidates[
                : self.config.max_recommendations_per_type
            ]:
                recommendation = Recommendation(
                    id=f"temporal_{query.node_id}_{node_id}",
                    target_node_id=node_id,
                    target_type=node_data.get("labels", ["Unknown"])[0],
                    title=node_data.get("title", "Unknown"),
                    description=f"Temporally related (proximity: {proximity_score:.3f})",
                    recommendation_type=RecommendationType.TEMPORAL_RELATED,
                    reason=RecommendationReason.TEMPORAL_PROXIMITY,
                    confidence_score=proximity_score,
                    relevance_score=proximity_score * self.config.temporal_weight,
                    details={"temporal_proximity": proximity_score},
                    generated_at=datetime.now(timezone.utc),
                    features={"temporal_proximity": proximity_score},
                )
                recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in temporal recommendations: {e}")

        return recommendations

    async def _get_collaborative_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations based on user behavior patterns."""
        recommendations = []

        if not query.user_id or not self.user_interactions:
            return recommendations

        try:
            user_history = self.user_interactions.get(query.user_id, [])
            if not user_history:
                return recommendations

            # Find users with similar interaction patterns
            similar_users = []
            for other_user, other_history in self.user_interactions.items():
                if other_user == query.user_id:
                    continue

                # Calculate Jaccard similarity
                intersection = set(user_history) & set(other_history)
                union = set(user_history) | set(other_history)
                similarity = len(intersection) / len(union) if union else 0

                if similarity > 0.1:
                    similar_users.append((other_user, similarity, other_history))

            # Get recommendations from similar users
            recommendation_scores = defaultdict(float)
            for other_user, similarity, other_history in similar_users:
                for node_id in other_history:
                    if node_id not in user_history:
                        recommendation_scores[node_id] += similarity

            # Sort and create recommendations
            sorted_recs = sorted(
                recommendation_scores.items(), key=lambda x: x[1], reverse=True
            )

            for node_id, score in sorted_recs[
                : self.config.max_recommendations_per_type
            ]:
                node_data = self.graph_cache.nodes.get(node_id, {})

                recommendation = Recommendation(
                    id=f"collaborative_{query.user_id}_{node_id}",
                    target_node_id=node_id,
                    target_type=node_data.get("labels", ["Unknown"])[0],
                    title=node_data.get("title", "Unknown"),
                    description=f"Users with similar interests also viewed this",
                    recommendation_type=RecommendationType.COLLABORATIVE,
                    reason=RecommendationReason.USER_BEHAVIOR,
                    confidence_score=score,
                    relevance_score=score,
                    details={"collaborative_score": score},
                    generated_at=datetime.now(timezone.utc),
                    features={"collaborative_score": score},
                )
                recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {e}")

        return recommendations

    async def _get_cross_domain_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations from other domains."""
        recommendations = []

        if not query.node_id or not self.graph_cache:
            return recommendations

        try:
            query_node_data = self.graph_cache.nodes.get(query.node_id, {})
            query_domain = query_node_data.get("domain")

            if not query_domain:
                return recommendations

            # Find patterns that span multiple domains
            cross_domain_patterns = []

            # Look for pattern nodes that connect to multiple domains
            for node_id, node_data in self.graph_cache.nodes(data=True):
                if "Pattern" in node_data.get("labels", []):
                    connected_domains = set()
                    for neighbor in self.graph_cache.neighbors(node_id):
                        neighbor_data = self.graph_cache.nodes.get(neighbor, {})
                        neighbor_domain = neighbor_data.get("domain")
                        if neighbor_domain:
                            connected_domains.add(neighbor_domain)

                    if len(connected_domains) > 1 and query_domain in connected_domains:
                        cross_domain_patterns.append((node_id, connected_domains))

            # Find recommendations through these patterns
            for pattern_id, domains in cross_domain_patterns:
                for neighbor in self.graph_cache.neighbors(pattern_id):
                    neighbor_data = self.graph_cache.nodes.get(neighbor, {})
                    neighbor_domain = neighbor_data.get("domain")

                    if neighbor_domain and neighbor_domain != query_domain:
                        cross_domain_score = 1.0 / len(
                            domains
                        )  # Inverse of domain count

                        recommendation = Recommendation(
                            id=f"cross_domain_{query.node_id}_{neighbor}",
                            target_node_id=neighbor,
                            target_type=neighbor_data.get("labels", ["Unknown"])[0],
                            title=neighbor_data.get("title", "Unknown"),
                            description=f"Cross-domain connection via pattern",
                            recommendation_type=RecommendationType.CROSS_DOMAIN,
                            reason=RecommendationReason.CONCEPT_OVERLAP,
                            confidence_score=cross_domain_score,
                            relevance_score=cross_domain_score,
                            details={
                                "pattern_id": pattern_id,
                                "target_domain": neighbor_domain,
                                "connected_domains": list(domains),
                            },
                            generated_at=datetime.now(timezone.utc),
                            features={"cross_domain_score": cross_domain_score},
                        )
                        recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"Error in cross-domain recommendations: {e}")

        return recommendations

    async def _get_pathway_recommendations(
        self, query: RecommendationQuery
    ) -> List[Recommendation]:
        """Get recommendations that form learning pathways."""
        recommendations = []

        if not query.node_id or not self.graph_cache:
            return recommendations

        try:
            # Find shortest paths to authoritative nodes
            high_authority_nodes = [
                node_id
                for node_id, score in self.pagerank_scores.items()
                if score > np.percentile(list(self.pagerank_scores.values()), 90)
            ]

            for target_node in high_authority_nodes:
                if target_node == query.node_id:
                    continue

                try:
                    if nx.has_path(self.graph_cache, query.node_id, target_node):
                        path = nx.shortest_path(
                            self.graph_cache, query.node_id, target_node
                        )

                        if len(path) > 2:  # Only recommend if there's a meaningful path
                            next_node = path[1]  # First step in the path
                            node_data = self.graph_cache.nodes.get(next_node, {})

                            pathway_score = 1.0 / len(path)  # Shorter paths are better

                            recommendation = Recommendation(
                                id=f"pathway_{query.node_id}_{next_node}",
                                target_node_id=next_node,
                                target_type=node_data.get("labels", ["Unknown"])[0],
                                title=node_data.get("title", "Unknown"),
                                description=f"Step in learning pathway (path length: {len(path)})",
                                recommendation_type=RecommendationType.PATHWAY,
                                reason=RecommendationReason.GRAPH_PROXIMITY,
                                confidence_score=pathway_score,
                                relevance_score=pathway_score,
                                details={
                                    "path_length": len(path),
                                    "target_authority": target_node,
                                    "full_path": path,
                                },
                                generated_at=datetime.now(timezone.utc),
                                features={"pathway_score": pathway_score},
                            )
                            recommendations.append(recommendation)

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        except Exception as e:
            self.logger.error(f"Error in pathway recommendations: {e}")

        return recommendations

    def _filter_and_rank_recommendations(
        self, recommendations: List[Recommendation], query: RecommendationQuery
    ) -> List[Recommendation]:
        """Filter and rank recommendations based on query parameters."""

        # Filter by confidence threshold
        filtered = [
            rec
            for rec in recommendations
            if rec.confidence_score >= query.min_confidence
        ]

        # Remove duplicates (same target node)
        seen_targets = set()
        unique_recs = []
        for rec in filtered:
            if rec.target_node_id not in seen_targets:
                seen_targets.add(rec.target_node_id)
                unique_recs.append(rec)

        # Sort by relevance score
        unique_recs.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply diversity filtering
        final_recs = self._apply_diversity_filter(unique_recs)

        # Limit results
        return final_recs[: query.limit]

    def _apply_diversity_filter(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Apply diversity filtering to avoid too similar recommendations."""
        if not recommendations:
            return []

        diverse_recs = [recommendations[0]]  # Always include the top recommendation

        for rec in recommendations[1:]:
            # Check diversity against already selected recommendations
            is_diverse = True
            for selected_rec in diverse_recs:
                # Simple diversity check based on recommendation type
                if (
                    rec.recommendation_type == selected_rec.recommendation_type
                    and rec.confidence_score / selected_rec.confidence_score
                    > self.config.diversity_threshold
                ):
                    is_diverse = False
                    break

            if is_diverse:
                diverse_recs.append(rec)

        return diverse_recs

    async def record_user_interaction(self, user_id: str, node_id: str) -> None:
        """Record user interaction for collaborative filtering."""
        self.user_interactions[user_id].append(node_id)

        # Keep only recent interactions (last 1000)
        if len(self.user_interactions[user_id]) > 1000:
            self.user_interactions[user_id] = self.user_interactions[user_id][-1000:]

    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get statistics about recommendations."""
        stats = {
            "cache_size": len(self.recommendations_cache),
            "graph_nodes": len(self.graph_cache.nodes) if self.graph_cache else 0,
            "graph_edges": len(self.graph_cache.edges) if self.graph_cache else 0,
            "embeddings_count": len(self.node_embeddings),
            "user_interactions": len(self.user_interactions),
            "pagerank_computed": len(self.pagerank_scores) > 0,
        }
        return stats


# CLI Interface
async def main():
    """Main CLI interface for recommendations."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Recommendation Engine")
    parser.add_argument(
        "--action",
        choices=["recommend", "stats", "record"],
        default="recommend",
        help="Action to perform",
    )
    parser.add_argument("--node-id", help="Node ID for recommendations")
    parser.add_argument("--user-id", help="User ID for collaborative filtering")
    parser.add_argument("--domain", help="Domain filter")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of recommendations"
    )
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()

    config = (
        RecommendationConfig(args.config) if args.config else RecommendationConfig()
    )
    engine = RecommendationEngine(config)

    await engine.initialize()

    try:
        if args.action == "recommend":
            query = RecommendationQuery(
                node_id=args.node_id,
                user_id=args.user_id,
                domain=args.domain,
                limit=args.limit,
            )

            recommendations = await engine.get_recommendations(query)

            print(f"Found {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec.title} ({rec.recommendation_type.value})")
                print(
                    f"   Confidence: {rec.confidence_score:.3f}, Relevance: {rec.relevance_score:.3f}"
                )
                print(f"   Reason: {rec.reason.value}")
                print()

        elif args.action == "stats":
            stats = engine.get_recommendation_stats()
            print(json.dumps(stats, indent=2))

        elif args.action == "record":
            if not args.user_id or not args.node_id:
                print("Error: --user-id and --node-id required for record action")
                return

            await engine.record_user_interaction(args.user_id, args.node_id)
            print(f"Recorded interaction: user {args.user_id} -> node {args.node_id}")

    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
