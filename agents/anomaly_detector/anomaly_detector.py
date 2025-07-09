"""
Anomaly Detector Agent for MCP Server - Refactored with modular structure.
Main orchestrator that coordinates detection using specialized components.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from neo4j import AsyncGraphDatabase, AsyncDriver
from qdrant_client import AsyncQdrantClient

from .config import AnomalyConfig
from .models import Anomaly
from .data_fetcher import DataFetcher
from .detectors import (
    TemporalAnomalyDetector,
    StatisticalAnomalyDetector,
    ContentAnomalyDetector,
    IsolationForestDetector
)
from .utils import ModelManager, AnomalyLogger, AnomalySummaryGenerator, setup_logging


class AnomalyDetector:
    """Main anomaly detection agent - orchestrates modular detection components."""
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """Initialize the anomaly detector with modular components."""
        self.config = config or AnomalyConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        
        # Storage
        self.detected_anomalies: List[Anomaly] = []
        self.log_dir = Path(self.config.log_dir)
        self.model_dir = Path(self.config.model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(self.log_dir)
        
        # Initialize utility components
        self.model_manager = ModelManager(self.model_dir)
        self.anomaly_logger = AnomalyLogger(self.log_dir)
        
        # Initialize detection components
        self.temporal_detector = TemporalAnomalyDetector()
        self.statistical_detector = StatisticalAnomalyDetector()
        self.content_detector = ContentAnomalyDetector(
            self.config.min_word_count, 
            self.config.max_word_count
        )
        self.isolation_detector = IsolationForestDetector(
            self.config.isolation_forest_contamination
        )
        
        # Data fetcher (initialized after database connections)
        self.data_fetcher: Optional[DataFetcher] = None
    
    async def initialize(self) -> None:
        """Initialize database connections and load models."""
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
            
            # Initialize data fetcher
            self.data_fetcher = DataFetcher(self.neo4j_driver, self.qdrant_client)
            
            # Load existing models
            await self._load_models()
            
            self.logger.info("Anomaly detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detector: {e}")
            raise
    
    async def _load_models(self) -> None:
        """Load trained models from disk using model manager."""
        try:
            model_names = ['isolation_forest', 'scaler', 'tfidf_vectorizer', 'pca', 'label_encoders']
            loaded_models = await self.model_manager.load_models(model_names)
            
            # Apply loaded models to detectors
            if loaded_models.get('isolation_forest'):
                self.isolation_detector.isolation_forest = loaded_models['isolation_forest']
            if loaded_models.get('scaler'):
                self.isolation_detector.scaler = loaded_models['scaler']
            if loaded_models.get('label_encoders'):
                self.isolation_detector.label_encoders = loaded_models['label_encoders']
                
        except Exception as e:
            self.logger.info(f"Could not load existing models: {e}")
    
    async def _save_models(self) -> None:
        """Save trained models to disk using model manager."""
        models_to_save = {
            'isolation_forest': getattr(self.isolation_detector, 'isolation_forest', None),
            'scaler': getattr(self.isolation_detector, 'scaler', None),
            'label_encoders': getattr(self.isolation_detector, 'label_encoders', None)
        }
        
        await self.model_manager.save_models(models_to_save)
    
    async def close(self) -> None:
        """Close database connections and save models."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            await self.qdrant_client.close()
        
        await self._save_models()
        self.logger.info("Anomaly detector closed")
    
    async def detect_anomalies(self, data_source: str = "all") -> List[Anomaly]:
        """Main method to detect anomalies using modular detectors."""
        self.logger.info(f"Starting anomaly detection for: {data_source}")
        detected_anomalies = []
        
        try:
            # Get data from databases using data fetcher
            graph_data = await self.data_fetcher.get_graph_data()
            vector_data = await self.data_fetcher.get_vector_data()
            
            # Combine data for analysis
            combined_data = self.data_fetcher.combine_data(graph_data, vector_data)
            
            if len(combined_data) == 0:
                self.logger.warning("No data found for anomaly detection")
                return detected_anomalies
            
            # Run different anomaly detection methods using specialized detectors
            if self.config.enable_models["temporal"]:
                temporal_anomalies = self.temporal_detector.detect(combined_data)
                detected_anomalies.extend(temporal_anomalies)
            
            if self.config.enable_models["statistical"]:
                statistical_anomalies = self.statistical_detector.detect(combined_data)
                detected_anomalies.extend(statistical_anomalies)
            
            if self.config.enable_models["content"]:
                content_anomalies = self.content_detector.detect(combined_data)
                detected_anomalies.extend(content_anomalies)
            
            if self.config.enable_models["isolation_forest"]:
                isolation_anomalies = self.isolation_detector.detect(combined_data)
                detected_anomalies.extend(isolation_anomalies)
            
            # Log and store results using anomaly logger
            self.detected_anomalies.extend(detected_anomalies)
            await self.anomaly_logger.log_anomalies(detected_anomalies)
            
            self.logger.info(f"Detected {len(detected_anomalies)} anomalies")
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")
            raise
        
        return detected_anomalies
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies using summary generator."""
        return AnomalySummaryGenerator.generate_summary(self.detected_anomalies)
    
    async def get_anomalies_by_node(self, node_id: str) -> List[Anomaly]:
        """Get all anomalies for a specific node."""
        return AnomalySummaryGenerator.get_anomalies_by_node(self.detected_anomalies, node_id)
    
    async def resolve_anomaly(self, anomaly_id: str, resolution_notes: str) -> bool:
        """Mark an anomaly as resolved."""
        return AnomalySummaryGenerator.resolve_anomaly(
            self.detected_anomalies, anomaly_id, resolution_notes
        )


# CLI Interface
async def main():
    """Main CLI interface for anomaly detection."""
    import argparse
    import json
    
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