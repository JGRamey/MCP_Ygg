"""Data fetching utilities for anomaly detection."""
import logging
import pandas as pd
from typing import Dict, Any, Optional
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient


class DataFetcher:
    """Handles data retrieval from various sources."""
    
    def __init__(self, neo4j_driver: AsyncDriver, qdrant_client: AsyncQdrantClient):
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger("anomaly_detector.data_fetcher")
    
    async def get_graph_data(self) -> pd.DataFrame:
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
    
    async def get_vector_data(self) -> Dict[str, Any]:
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
    
    def combine_data(self, graph_data: pd.DataFrame, vector_data: Dict[str, Any]) -> pd.DataFrame:
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