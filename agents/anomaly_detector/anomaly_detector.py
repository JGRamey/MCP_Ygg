"""
Anomaly Detector Agent for MCP Server
Detects outliers and anomalies in scraped data using various ML techniques.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as stats

from neo4j import AsyncGraphDatabase, AsyncDriver
from qdrant_client import AsyncQdrantClient
import aiofiles


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TEMPORAL = "temporal"  # Date/time inconsistencies
    METADATA = "metadata"  # Unusual metadata patterns
    CONTENT = "content"  # Content anomalies
    DOMAIN = "domain"  # Domain classification issues
    RELATIONSHIP = "relationship"  # Unusual graph relationships
    VECTOR = "vector"  # Vector embedding anomalies
    STATISTICAL = "statistical"  # Statistical outliers


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    data_source: str
    node_id: Optional[str]
    anomaly_score: float
    details: Dict[str, Any]
    detected_at: datetime
    features: Dict[str, float]
    suggestions: List[str]


class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    def __init__(self, config_path: str = "agents/anomaly_detector/config.py"):
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.log_dir = "agents/anomaly_detector/logs"
        self.model_dir = "agents/anomaly_detector/models"
        
        # Detection thresholds
        self.isolation_forest_contamination = 0.1
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5
        self.lof_n_neighbors = 20
        self.temporal_threshold_days = 365 * 100  # 100 years
        self.content_similarity_threshold = 0.95
        self.min_word_count = 10
        self.max_word_count = 1000000
        
        # Processing settings
        self.batch_size = 1000
        self.max_features_tfidf = 10000
        self.pca_components = 50
        self.enable_models = {
            "isolation_forest": True,
            "dbscan": True,
            "lof": True,
            "statistical": True,
            "temporal": True,
            "content": True
        }
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file if it exists."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")


class AnomalyDetector:
    """Main anomaly detection agent."""
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """Initialize the anomaly detector."""
        self.config = config or AnomalyConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        
        # ML models
        self.isolation_forest = None
        self.dbscan = None
        self.lof = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features_tfidf,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.pca = PCA(n_components=self.config.pca_components)
        
        # Storage
        self.detected_anomalies: List[Anomaly] = []
        self.log_dir = Path(self.config.log_dir)
        self.model_dir = Path(self.config.model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("anomaly_detector")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"anomalies_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize database connections and ML models."""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            # Initialize Qdrant client
            self.qdrant_client = AsyncQdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port
            )
            
            # Test Qdrant connection
            await self.qdrant_client.get_collections()
            
            # Initialize ML models
            await self._initialize_models()
            
            self.logger.info("Anomaly detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detector: {e}")
            raise
    
    async def _initialize_models(self) -> None:
        """Initialize or load ML models."""
        try:
            # Try to load existing models
            await self._load_models()
        except Exception as e:
            self.logger.info(f"Could not load existing models, initializing new ones: {e}")
            # Initialize new models
            if self.config.enable_models["isolation_forest"]:
                self.isolation_forest = IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=42,
                    n_jobs=-1
                )
            
            if self.config.enable_models["dbscan"]:
                self.dbscan = DBSCAN(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples,
                    n_jobs=-1
                )
            
            if self.config.enable_models["lof"]:
                self.lof = LocalOutlierFactor(
                    n_neighbors=self.config.lof_n_neighbors,
                    contamination=self.config.isolation_forest_contamination,
                    n_jobs=-1
                )
    
    async def _load_models(self) -> None:
        """Load trained models from disk."""
        model_files = {
            'isolation_forest': self.model_dir / 'isolation_forest.pkl',
            'scaler': self.model_dir / 'scaler.pkl',
            'tfidf_vectorizer': self.model_dir / 'tfidf_vectorizer.pkl',
            'pca': self.model_dir / 'pca.pkl',
            'label_encoders': self.model_dir / 'label_encoders.pkl'
        }
        
        for model_name, file_path in model_files.items():
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                    setattr(self, model_name, model)
                self.logger.info(f"Loaded {model_name} from {file_path}")
    
    async def _save_models(self) -> None:
        """Save trained models to disk."""
        models_to_save = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'pca': self.pca,
            'label_encoders': self.label_encoders
        }
        
        for model_name, model in models_to_save.items():
            if model is not None:
                file_path = self.model_dir / f'{model_name}.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Saved {model_name} to {file_path}")
    
    async def close(self) -> None:
        """Close database connections and save models."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            await self.qdrant_client.close()
        
        await self._save_models()
        self.logger.info("Anomaly detector closed")
    
    async def detect_anomalies(self, data_source: str = "all") -> List[Anomaly]:
        """Main method to detect anomalies in the system."""
        self.logger.info(f"Starting anomaly detection for: {data_source}")
        detected_anomalies = []
        
        try:
            # Get data from databases
            graph_data = await self._get_graph_data()
            vector_data = await self._get_vector_data()
            
            # Combine data for analysis
            combined_data = await self._combine_data(graph_data, vector_data)
            
            if len(combined_data) == 0:
                self.logger.warning("No data found for anomaly detection")
                return detected_anomalies
            
            # Run different anomaly detection methods
            if self.config.enable_models["temporal"]:
                temporal_anomalies = await self._detect_temporal_anomalies(combined_data)
                detected_anomalies.extend(temporal_anomalies)
            
            if self.config.enable_models["statistical"]:
                statistical_anomalies = await self._detect_statistical_anomalies(combined_data)
                detected_anomalies.extend(statistical_anomalies)
            
            if self.config.enable_models["content"]:
                content_anomalies = await self._detect_content_anomalies(combined_data)
                detected_anomalies.extend(content_anomalies)
            
            if self.config.enable_models["isolation_forest"]:
                isolation_anomalies = await self._detect_isolation_forest_anomalies(combined_data)
                detected_anomalies.extend(isolation_anomalies)
            
            # Log and store results
            self.detected_anomalies.extend(detected_anomalies)
            await self._log_anomalies(detected_anomalies)
            
            self.logger.info(f"Detected {len(detected_anomalies)} anomalies")
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")
            raise
        
        return detected_anomalies
    
    async def _get_graph_data(self) -> pd.DataFrame:
        """Get data from Neo4j graph."""
        async with self.neo4j_driver.session() as session:
            query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                n.title as title,
                n.author as author,
                n.date as date,
                n.domain as domain,
                n.source as source,
                n.language as language,
                n.word_count as word_count,
                count(r) as relationship_count,
                collect(distinct type(r)) as relationship_types
            """
            result = await session.run(query)
            
            records = []
            async for record in result:
                records.append(dict(record))
            
            return pd.DataFrame(records)
    
    async def _get_vector_data(self) -> Dict[str, Any]:
        """Get data from Qdrant vector database."""
        vector_data = {}
        
        try:
            collections = await self.qdrant_client.get_collections()
            
            for collection in collections.collections:
                collection_name = collection.name
                
                # Get collection info
                info = await self.qdrant_client.get_collection(collection_name)
                vector_data[collection_name] = {
                    'vectors_count': info.vectors_count,
                    'points_count': info.points_count,
                    'segments_count': info.segments_count
                }
                
                # Sample some points for analysis
                points = await self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=min(1000, info.points_count or 0),
                    with_payload=True,
                    with_vectors=True
                )
                
                vector_data[collection_name]['sample_points'] = points[0] if points else []
        
        except Exception as e:
            self.logger.warning(f"Could not get vector data: {e}")
        
        return vector_data
    
    async def _combine_data(self, graph_data: pd.DataFrame, vector_data: Dict[str, Any]) -> pd.DataFrame:
        """Combine graph and vector data for analysis."""
        if graph_data.empty:
            return pd.DataFrame()
        
        # Clean and preprocess graph data
        df = graph_data.copy()
        
        # Handle missing values
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce')
        df['relationship_count'] = pd.to_numeric(df['relationship_count'], errors='coerce')
        
        # Fill missing values
        df['word_count'].fillna(0, inplace=True)
        df['relationship_count'].fillna(0, inplace=True)
        df['domain'].fillna('unknown', inplace=True)
        df['language'].fillna('unknown', inplace=True)
        
        # Add derived features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['has_author'] = df['author'].notna().astype(int)
        df['has_title'] = df['title'].notna().astype(int)
        df['label_count'] = df['labels'].apply(lambda x: len(x) if x else 0)
        
        # Add vector-based features if available
        for collection_name, coll_data in vector_data.items():
            df[f'{collection_name}_available'] = 0  # Default to not available
            
            if 'sample_points' in coll_data:
                points = coll_data['sample_points']
                if points:
                    # Mark nodes that have vectors
                    vector_node_ids = set()
                    for point in points:
                        if point.payload and 'node_id' in point.payload:
                            vector_node_ids.add(point.payload['node_id'])
                    
                    df[f'{collection_name}_available'] = df['node_id'].isin(vector_node_ids).astype(int)
        
        return df
    
    async def _detect_temporal_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect temporal anomalies in dates."""
        anomalies = []
        
        # Check for future dates
        future_dates = data[data['date'] > datetime.now()]
        for _, row in future_dates.iterrows():
            anomaly = Anomaly(
                id=f"temporal_{row['node_id']}_future",
                anomaly_type=AnomalyType.TEMPORAL,
                severity=AnomalySeverity.HIGH,
                description="Document has future date",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=1.0,
                details={'date': str(row['date']), 'current_date': str(datetime.now())},
                detected_at=datetime.now(timezone.utc),
                features={'date_diff_days': (row['date'] - datetime.now()).days},
                suggestions=["Verify document date", "Check data source accuracy"]
            )
            anomalies.append(anomaly)
        
        # Check for extremely old dates (before 3000 BCE)
        very_old_dates = data[data['year'] < -3000]
        for _, row in very_old_dates.iterrows():
            if pd.notna(row['year']):
                anomaly = Anomaly(
                    id=f"temporal_{row['node_id']}_ancient",
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=AnomalySeverity.MEDIUM,
                    description="Document has extremely ancient date",
                    data_source="neo4j",
                    node_id=str(row['node_id']),
                    anomaly_score=0.8,
                    details={'year': row['year'], 'date': str(row['date'])},
                    detected_at=datetime.now(timezone.utc),
                    features={'year': row['year']},
                    suggestions=["Verify historical accuracy", "Check BCE/CE designation"]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_statistical_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect statistical outliers using z-scores."""
        anomalies = []
        
        numerical_columns = ['word_count', 'relationship_count', 'label_count']
        
        for column in numerical_columns:
            if column in data.columns and not data[column].empty:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(data[column].dropna()))
                outliers = data[z_scores > 3]  # 3 standard deviations
                
                for _, row in outliers.iterrows():
                    severity = AnomalySeverity.LOW
                    if z_scores[row.name] > 5:
                        severity = AnomalySeverity.HIGH
                    elif z_scores[row.name] > 4:
                        severity = AnomalySeverity.MEDIUM
                    
                    anomaly = Anomaly(
                        id=f"statistical_{row['node_id']}_{column}",
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=severity,
                        description=f"Statistical outlier in {column}",
                        data_source="neo4j",
                        node_id=str(row['node_id']),
                        anomaly_score=min(z_scores[row.name] / 5.0, 1.0),
                        details={
                            column: row[column],
                            'z_score': z_scores[row.name],
                            'mean': data[column].mean(),
                            'std': data[column].std()
                        },
                        detected_at=datetime.now(timezone.utc),
                        features={f'{column}_zscore': z_scores[row.name]},
                        suggestions=[f"Review {column} value", "Verify data accuracy"]
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_content_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect content-based anomalies."""
        anomalies = []
        
        # Check for extremely short content
        short_content = data[
            (data['word_count'] < self.config.min_word_count) & 
            (data['word_count'] > 0)
        ]
        
        for _, row in short_content.iterrows():
            anomaly = Anomaly(
                id=f"content_{row['node_id']}_short",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Document has unusually short content",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=1.0 - (row['word_count'] / self.config.min_word_count),
                details={'word_count': row['word_count'], 'threshold': self.config.min_word_count},
                detected_at=datetime.now(timezone.utc),
                features={'word_count': row['word_count']},
                suggestions=["Verify content extraction", "Check for partial documents"]
            )
            anomalies.append(anomaly)
        
        # Check for extremely long content
        long_content = data[data['word_count'] > self.config.max_word_count]
        
        for _, row in long_content.iterrows():
            anomaly = Anomaly(
                id=f"content_{row['node_id']}_long",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Document has unusually long content",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=min(row['word_count'] / self.config.max_word_count - 1.0, 1.0),
                details={'word_count': row['word_count'], 'threshold': self.config.max_word_count},
                detected_at=datetime.now(timezone.utc),
                features={'word_count': row['word_count']},
                suggestions=["Verify document boundaries", "Check for concatenated documents"]
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_isolation_forest_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest."""
        anomalies = []
        
        if not self.config.enable_models["isolation_forest"] or data.empty:
            return anomalies
        
        try:
            # Prepare features for isolation forest
            feature_columns = [
                'word_count', 'relationship_count', 'label_count',
                'has_author', 'has_title', 'year', 'month', 'day_of_year'
            ]
            
            # Filter and prepare data
            feature_data = data[feature_columns].copy()
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:  # Need minimum samples
                return anomalies
            
            # Encode categorical features
            categorical_columns = ['domain', 'language']
            for col in categorical_columns:
                if col in data.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        encoded = self.label_encoders[col].fit_transform(data[col].fillna('unknown'))
                    else:
                        # Handle new categories
                        known_categories = set(self.label_encoders[col].classes_)
                        data_categories = set(data[col].fillna('unknown'))
                        new_categories = data_categories - known_categories
                        
                        if new_categories:
                            # Add new categories to encoder
                            all_categories = list(known_categories) + list(new_categories)
                            self.label_encoders[col].classes_ = np.array(all_categories)
                        
                        encoded = self.label_encoders[col].transform(data[col].fillna('unknown'))
                    
                    feature_data[f'{col}_encoded'] = encoded
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Train and predict with Isolation Forest
            if self.isolation_forest is None:
                self.isolation_forest = IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=42,
                    n_jobs=-1
                )
            
            outlier_predictions = self.isolation_forest.fit_predict(scaled_features)
            outlier_scores = self.isolation_forest.decision_function(scaled_features)
            
            # Create anomalies for outliers
            outlier_indices = np.where(outlier_predictions == -1)[0]
            
            for idx in outlier_indices:
                row = data.iloc[idx]
                score = abs(outlier_scores[idx])
                
                # Determine severity based on score
                if score > 0.5:
                    severity = AnomalySeverity.HIGH
                elif score > 0.3:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                anomaly = Anomaly(
                    id=f"isolation_{row['node_id']}",
                    anomaly_type=AnomalyType.METADATA,
                    severity=severity,
                    description="Isolation Forest detected anomalous pattern",
                    data_source="neo4j",
                    node_id=str(row['node_id']),
                    anomaly_score=score,
                    details={
                        'isolation_score': outlier_scores[idx],
                        'features': {col: row[col] for col in feature_columns if col in row}
                    },
                    detected_at=datetime.now(timezone.utc),
                    features={f'feature_{i}': scaled_features[idx][i] for i in range(len(scaled_features[idx]))},
                    suggestions=["Review metadata values", "Verify data source", "Check for data entry errors"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in isolation forest detection: {e}")
        
        return anomalies
    
    async def _log_anomalies(self, anomalies: List[Anomaly]) -> None:
        """Log detected anomalies to file."""
        if not anomalies:
            return
        
        log_file = self.log_dir / f"detected_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        anomaly_data = [asdict(anomaly) for anomaly in anomalies]
        
        # Convert datetime objects to strings
        for anomaly_dict in anomaly_data:
            if 'detected_at' in anomaly_dict:
                anomaly_dict['detected_at'] = anomaly_dict['detected_at'].isoformat()
        
        async with aiofiles.open(log_file, 'w') as f:
            await f.write(json.dumps(anomaly_data, indent=2, default=str))
        
        self.logger.info(f"Logged {len(anomalies)} anomalies to {log_file}")
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        if not self.detected_anomalies:
            return {"total": 0}
        
        summary = {
            "total": len(self.detected_anomalies),
            "by_type": {},
            "by_severity": {},
            "recent_count": 0
        }
        
        recent_threshold = datetime.now(timezone.utc).timestamp() - 86400  # 24 hours ago
        
        for anomaly in self.detected_anomalies:
            # Count by type
            type_name = anomaly.anomaly_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Count by severity
            severity_name = anomaly.severity.value
            summary["by_severity"][severity_name] = summary["by_severity"].get(severity_name, 0) + 1
            
            # Count recent anomalies
            if anomaly.detected_at.timestamp() > recent_threshold:
                summary["recent_count"] += 1
        
        return summary
    
    async def get_anomalies_by_node(self, node_id: str) -> List[Anomaly]:
        """Get all anomalies for a specific node."""
        return [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.node_id == node_id
        ]
    
    async def resolve_anomaly(self, anomaly_id: str, resolution_notes: str) -> bool:
        """Mark an anomaly as resolved."""
        for anomaly in self.detected_anomalies:
            if anomaly.id == anomaly_id:
                anomaly.details["resolved"] = True
                anomaly.details["resolution_notes"] = resolution_notes
                anomaly.details["resolved_at"] = datetime.now(timezone.utc).isoformat()
                return True
        return False


# CLI Interface
async def main():
    """Main CLI interface for anomaly detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Anomaly Detector")
    parser.add_argument("--action", choices=["detect", "summary", "resolve"], 
                       default="detect", help="Action to perform")
    parser.add_argument("--data-source", default="all", help="Data source to analyze")
    parser.add_argument("--anomaly-id", help="Anomaly ID for resolution")
    parser.add_argument("--resolution", help="Resolution notes for anomaly")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    config = AnomalyConfig(args.config) if args.config else AnomalyConfig()
    detector = AnomalyDetector(config)
    
    await detector.initialize()
    
    try:
        if args.action == "detect":
            anomalies = await detector.detect_anomalies(args.data_source)
            print(f"Detected {len(anomalies)} anomalies")
            
            for anomaly in anomalies[:10]:  # Show first 10
                print(f"  {anomaly.id}: {anomaly.description} (Severity: {anomaly.severity.value})")
                
        elif args.action == "summary":
            summary = detector.get_anomaly_summary()
            print(json.dumps(summary, indent=2))
            
        elif args.action == "resolve":
            if not args.anomaly_id or not args.resolution:
                print("Error: --anomaly-id and --resolution required for resolve action")
                return
            
            success = await detector.resolve_anomaly(args.anomaly_id, args.resolution)
            print(f"Anomaly resolution: {'Success' if success else 'Failed'}")
    
    finally:
        await detector.close()


if __name__ == "__main__":
    asyncio.run(main())
