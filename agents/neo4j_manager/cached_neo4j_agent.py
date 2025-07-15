#!/usr/bin/env python3
"""
Cached Neo4j Manager Agent
Extends the base Neo4j agent with comprehensive caching capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .neo4j_agent import Neo4jAgent, OperationResult, NodeData, RelationshipData
from cache.integration_manager import cache_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedNeo4jAgent(Neo4jAgent):
    """Neo4j Agent with integrated caching capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.cache_manager = cache_integration
        
    async def initialize(self) -> bool:
        """Initialize the Neo4j connection and cache system."""
        # Initialize base Neo4j agent
        neo4j_init = await super().initialize()
        
        # Initialize cache system
        try:
            await self.cache_manager.initialize()
            logger.info("Cache system initialized successfully")
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")
            # Continue without cache
        
        return neo4j_init
    
    # Cached read operations
    @cache_integration.cache_neo4j_query(ttl=300)  # 5 minutes
    async def get_node_cached(self, node_id: str) -> OperationResult:
        """Get node with caching."""
        return await super().get_node(node_id)
    
    @cache_integration.cache_neo4j_query(ttl=600)  # 10 minutes
    async def get_concepts_by_domain(self, domain: str, limit: int = 100) -> OperationResult:
        """Get concepts by domain with caching."""
        query = """
        MATCH (c:Concept)-[:BELONGS_TO]->(d:Domain {name: $domain})
        RETURN c, elementId(c) as id
        ORDER BY c.name
        LIMIT $limit
        """
        
        return await self.execute_cypher(query, {"domain": domain, "limit": limit})
    
    @cache_integration.cache_neo4j_query(ttl=600)  # 10 minutes
    async def get_relationships_by_type(self, relationship_type: str, limit: int = 100) -> OperationResult:
        """Get relationships by type with caching."""
        query = f"""
        MATCH (a)-[r:{relationship_type}]->(b)
        RETURN a, r, b, elementId(a) as source_id, elementId(b) as target_id
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        
        return await self.execute_cypher(query, {"limit": limit})
    
    @cache_integration.cache_neo4j_query(ttl=1800)  # 30 minutes
    async def get_node_degree_stats(self, node_id: str) -> OperationResult:
        """Get node degree statistics with caching."""
        query = """
        MATCH (n)
        WHERE elementId(n) = $node_id
        OPTIONAL MATCH (n)-[r]-()
        RETURN n, 
               count(r) as total_degree,
               count(DISTINCT type(r)) as relationship_types,
               collect(DISTINCT type(r)) as relationship_list
        """
        
        return await self.execute_cypher(query, {"node_id": node_id})
    
    @cache_integration.cache_neo4j_query(ttl=3600)  # 1 hour
    async def get_domain_statistics(self, domain: str) -> OperationResult:
        """Get domain statistics with caching."""
        query = """
        MATCH (d:Domain {name: $domain})
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(c:Concept)
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(p:Person)
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(w:Work)
        RETURN d,
               count(DISTINCT c) as concept_count,
               count(DISTINCT p) as person_count,
               count(DISTINCT w) as work_count
        """
        
        return await self.execute_cypher(query, {"domain": domain})
    
    @cache_integration.cache_neo4j_query(ttl=3600)  # 1 hour
    async def get_graph_overview_stats(self) -> OperationResult:
        """Get graph overview statistics with caching."""
        query = """
        MATCH (n)
        OPTIONAL MATCH ()-[r]-()
        RETURN 
            count(DISTINCT n) as total_nodes,
            count(DISTINCT r) as total_relationships,
            count(DISTINCT labels(n)) as node_types,
            collect(DISTINCT labels(n)) as all_labels
        """
        
        return await self.execute_cypher(query)
    
    @cache_integration.cache_neo4j_query(ttl=1800)  # 30 minutes
    async def search_nodes_by_name(self, search_term: str, limit: int = 20) -> OperationResult:
        """Search nodes by name with caching."""
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $search_term
        RETURN n, labels(n) as labels, elementId(n) as id
        ORDER BY n.name
        LIMIT $limit
        """
        
        return await self.execute_cypher(query, {"search_term": search_term, "limit": limit})
    
    @cache_integration.cache_neo4j_query(ttl=600)  # 10 minutes
    async def get_node_connections(self, node_id: str, depth: int = 1) -> OperationResult:
        """Get node connections with caching."""
        query = f"""
        MATCH (start)
        WHERE elementId(start) = $node_id
        MATCH path = (start)-[*1..{depth}]-(connected)
        RETURN start, connected, relationships(path) as path_relationships,
               elementId(start) as start_id, elementId(connected) as connected_id
        LIMIT 50
        """
        
        return await self.execute_cypher(query, {"node_id": node_id})
    
    # Cache invalidation methods
    async def create_node_with_cache_invalidation(self, node_data: NodeData) -> OperationResult:
        """Create node and invalidate relevant cache entries."""
        result = await super().create_node(node_data)
        
        if result.success:
            # Invalidate caches that might be affected
            await self.cache_manager.invalidate_domain_cache(node_data.properties.get('domain', ''))
            await self.cache_manager.clear_analytics_cache()
        
        return result
    
    async def update_node_with_cache_invalidation(self, node_id: str, properties: Dict[str, Any]) -> OperationResult:
        """Update node and invalidate relevant cache entries."""
        result = await super().update_node(node_id, properties)
        
        if result.success:
            # Invalidate specific node cache
            await self.cache_manager.invalidate_concept_cache(node_id)
            
            # Invalidate domain cache if domain changed
            if 'domain' in properties:
                await self.cache_manager.invalidate_domain_cache(properties['domain'])
            
            # Clear analytics cache
            await self.cache_manager.clear_analytics_cache()
        
        return result
    
    async def delete_node_with_cache_invalidation(self, node_id: str) -> OperationResult:
        """Delete node and invalidate relevant cache entries."""
        result = await super().delete_node(node_id)
        
        if result.success:
            # Invalidate all related caches
            await self.cache_manager.invalidate_concept_cache(node_id)
            await self.cache_manager.clear_analytics_cache()
            await self.cache_manager.clear_search_cache()
        
        return result
    
    async def create_relationship_with_cache_invalidation(self, relationship_data: RelationshipData) -> OperationResult:
        """Create relationship and invalidate relevant cache entries."""
        result = await super().create_relationship(relationship_data)
        
        if result.success:
            # Invalidate connection caches for both nodes
            await self.cache_manager.invalidate_concept_cache(relationship_data.source_id)
            await self.cache_manager.invalidate_concept_cache(relationship_data.target_id)
            
            # Clear analytics cache
            await self.cache_manager.clear_analytics_cache()
        
        return result
    
    # Cache management methods
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and performance metrics."""
        return await self.cache_manager.get_integration_status()
    
    async def get_cache_performance_report(self) -> Dict[str, Any]:
        """Get detailed cache performance report."""
        return await self.cache_manager.get_cache_performance_report()
    
    async def clear_all_cache(self):
        """Clear all Neo4j-related cache entries."""
        await self.cache_manager.clear_search_cache()
        await self.cache_manager.clear_analytics_cache()
    
    async def warm_cache_for_domain(self, domain: str):
        """Warm cache for a specific domain."""
        try:
            # Pre-load commonly accessed data
            await self.get_concepts_by_domain(domain, 50)
            await self.get_domain_statistics(domain)
            
            logger.info(f"Cache warmed for domain: {domain}")
        except Exception as e:
            logger.warning(f"Cache warming failed for domain {domain}: {e}")
    
    async def close(self):
        """Close Neo4j agent and cache system."""
        await super().close()
        await self.cache_manager.close()
    
    # Override execute_cypher to add cache metrics
    async def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> OperationResult:
        """Execute Cypher query with cache integration."""
        # For write operations, clear relevant caches
        if any(keyword in query.upper() for keyword in ['CREATE', 'UPDATE', 'DELETE', 'MERGE', 'SET']):
            # This is a write operation - invalidate caches after execution
            result = await super().execute_cypher(query, parameters)
            if result.success:
                # Invalidate relevant caches based on query pattern
                await self._invalidate_cache_by_query_pattern(query)
            return result
        else:
            # This is a read operation - use regular execution
            return await super().execute_cypher(query, parameters)
    
    async def _invalidate_cache_by_query_pattern(self, query: str):
        """Invalidate cache based on query patterns."""
        query_upper = query.upper()
        
        # If query affects nodes, clear analytics cache
        if any(pattern in query_upper for pattern in ['CREATE', 'DELETE', 'MERGE']):
            await self.cache_manager.clear_analytics_cache()
        
        # If query affects relationships, clear search cache
        if 'RELATIONSHIP' in query_upper or '-[' in query:
            await self.cache_manager.clear_search_cache()


# Global cached Neo4j agent instance
cached_neo4j_agent = CachedNeo4jAgent()


# Convenience functions for direct use
async def init_cached_neo4j_agent(config_path: Optional[str] = None) -> bool:
    """Initialize the cached Neo4j agent."""
    if config_path:
        global cached_neo4j_agent
        cached_neo4j_agent = CachedNeo4jAgent(config_path)
    
    return await cached_neo4j_agent.initialize()


async def get_cached_neo4j_agent() -> CachedNeo4jAgent:
    """Get the global cached Neo4j agent instance."""
    return cached_neo4j_agent