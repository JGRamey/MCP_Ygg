#!/usr/bin/env python3
"""
Query Optimizer for Neo4j Agent
Handles Cypher query optimization, caching, and performance analysis
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class QueryStats:
    """Query execution statistics"""
    query_hash: str
    query: str
    execution_count: int
    total_execution_time: float
    avg_execution_time: float
    last_executed: float
    optimizations_applied: List[str]

@dataclass
class OptimizationResult:
    """Result of query optimization"""
    original_query: str
    optimized_query: str
    optimizations_applied: List[str]
    estimated_improvement: float

class QueryOptimizer:
    """
    Optimizes Cypher queries for better performance
    Provides query caching, pattern recognition, and optimization suggestions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_cache: Dict[str, str] = {}
        self.query_stats: Dict[str, QueryStats] = {}
        self.performance_cache_size = config.get("performance", {}).get("cache_size", 1000)
        
        # Query optimization patterns
        self.optimization_patterns = [
            {
                "name": "index_hint_addition",
                "pattern": r"MATCH \((\w+):(\w+)\) WHERE \1\.(\w+) = \$(\w+)",
                "optimization": self._add_index_hints,
                "description": "Add index hints for property lookups"
            },
            {
                "name": "limit_early_application",
                "pattern": r"MATCH.*RETURN.*LIMIT (\d+)",
                "optimization": self._apply_early_limit,
                "description": "Apply LIMIT earlier in query execution"
            },
            {
                "name": "unnecessary_optional_removal",
                "pattern": r"OPTIONAL MATCH.*WHERE.*IS NOT NULL",
                "optimization": self._remove_unnecessary_optional,
                "description": "Remove unnecessary OPTIONAL MATCH clauses"
            },
            {
                "name": "relationship_direction_optimization",
                "pattern": r"MATCH \(.*\)-\[\*\]-\(.*\)",
                "optimization": self._optimize_relationship_direction,
                "description": "Optimize bidirectional relationship traversals"
            },
            {
                "name": "redundant_where_removal",
                "pattern": r"WHERE.*AND.*\1",
                "optimization": self._remove_redundant_conditions,
                "description": "Remove redundant WHERE conditions"
            }
        ]
        
        # Common query templates for caching
        self.query_templates = {
            "get_concept_by_id": """
                MATCH (c:Concept {id: $concept_id})
                RETURN c, labels(c) as labels
            """,
            "get_relationships": """
                MATCH (source)-[r]->(target)
                WHERE elementId(source) = $source_id
                RETURN r, target, type(r) as rel_type
            """,
            "concept_search": """
                CALL db.index.fulltext.queryNodes('concept_search_idx', $search_term)
                YIELD node, score
                WHERE node.domain = $domain OR $domain IS NULL
                RETURN node, score
                ORDER BY score DESC
                LIMIT $limit
            """,
            "domain_stats": """
                MATCH (c:Concept)
                RETURN c.domain as domain, count(c) as count
                ORDER BY count DESC
            """,
            "temporal_concepts": """
                MATCH (c:Concept)
                WHERE c.earliest_evidence_date >= $start_date 
                  AND c.earliest_evidence_date <= $end_date
                RETURN c
                ORDER BY c.earliest_evidence_date
            """
        }
    
    async def optimize_query(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Optimize a Cypher query for better performance"""
        try:
            query_hash = self._hash_query(query, parameters)
            
            # Check cache first
            if query_hash in self.query_cache:
                logger.debug(f"Using cached optimized query: {query_hash[:8]}")
                return self.query_cache[query_hash]
            
            # Perform optimization
            optimization_result = await self._apply_optimizations(query)
            optimized_query = optimization_result.optimized_query
            
            # Cache the result
            if len(self.query_cache) < self.performance_cache_size:
                self.query_cache[query_hash] = optimized_query
            
            # Update statistics
            self._update_query_stats(query_hash, query, optimization_result)
            
            logger.debug(f"Query optimized with {len(optimization_result.optimizations_applied)} improvements")
            return optimized_query
            
        except Exception as e:
            logger.warning(f"Query optimization failed, using original: {e}")
            return query
    
    async def _apply_optimizations(self, query: str) -> OptimizationResult:
        """Apply all available optimizations to a query"""
        optimized_query = query.strip()
        optimizations_applied = []
        estimated_improvement = 0.0
        
        # Apply each optimization pattern
        for pattern_info in self.optimization_patterns:
            try:
                result = pattern_info["optimization"](optimized_query, pattern_info)
                if result["modified"]:
                    optimized_query = result["query"]
                    optimizations_applied.append(pattern_info["name"])
                    estimated_improvement += result.get("improvement_estimate", 0.1)
                    
                    logger.debug(f"Applied optimization: {pattern_info['name']}")
                    
            except Exception as e:
                logger.debug(f"Optimization {pattern_info['name']} failed: {e}")
        
        # Apply template-based optimizations
        template_result = self._apply_template_optimization(optimized_query)
        if template_result["modified"]:
            optimized_query = template_result["query"]
            optimizations_applied.append("template_optimization")
            estimated_improvement += 0.2
        
        # Apply general performance optimizations
        general_result = self._apply_general_optimizations(optimized_query)
        if general_result["modified"]:
            optimized_query = general_result["query"]
            optimizations_applied.extend(general_result["optimizations"])
            estimated_improvement += general_result.get("improvement_estimate", 0.1)
        
        return OptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            optimizations_applied=optimizations_applied,
            estimated_improvement=estimated_improvement
        )
    
    def _add_index_hints(self, query: str, pattern_info: Dict) -> Dict:
        """Add index hints for property lookups"""
        pattern = pattern_info["pattern"]
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        modified = False
        optimized_query = query
        
        for match in matches:
            node_var, label, property_name, param_name = match.groups()
            
            # Check if this property likely has an index
            if self._has_likely_index(label, property_name):
                # Add USING INDEX hint
                index_hint = f"USING INDEX {node_var}:{label}({property_name})"
                
                # Insert hint after the MATCH clause
                match_end = match.end()
                optimized_query = (
                    optimized_query[:match_end] + 
                    f"\n{index_hint}" + 
                    optimized_query[match_end:]
                )
                modified = True
        
        return {
            "modified": modified,
            "query": optimized_query,
            "improvement_estimate": 0.3 if modified else 0.0
        }
    
    def _apply_early_limit(self, query: str, pattern_info: Dict) -> Dict:
        """Apply LIMIT earlier in query execution"""
        # Look for queries where LIMIT could be applied earlier
        pattern = r"(MATCH.*?)(RETURN.*?LIMIT \d+)"
        match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        
        if match and "ORDER BY" not in query.upper():
            # Can apply LIMIT earlier if no ORDER BY
            match_part = match.group(1)
            return_limit = match.group(2)
            
            # Add WITH clause with LIMIT before RETURN
            with_clause = f"\nWITH * LIMIT {re.search(r'LIMIT (\d+)', return_limit).group(1)}"
            optimized_query = match_part + with_clause + "\n" + return_limit
            
            return {
                "modified": True,
                "query": optimized_query,
                "improvement_estimate": 0.4
            }
        
        return {"modified": False, "query": query}
    
    def _remove_unnecessary_optional(self, query: str, pattern_info: Dict) -> Dict:
        """Remove unnecessary OPTIONAL MATCH clauses"""
        # Look for OPTIONAL MATCH followed by WHERE ... IS NOT NULL
        pattern = r"OPTIONAL MATCH (.*?) WHERE (.*?) IS NOT NULL"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        modified = False
        optimized_query = query
        
        for match in matches:
            optional_match = match.group(0)
            regular_match = optional_match.replace("OPTIONAL MATCH", "MATCH").replace(" IS NOT NULL", "")
            optimized_query = optimized_query.replace(optional_match, regular_match)
            modified = True
        
        return {
            "modified": modified,
            "query": optimized_query,
            "improvement_estimate": 0.2 if modified else 0.0
        }
    
    def _optimize_relationship_direction(self, query: str, pattern_info: Dict) -> Dict:
        """Optimize bidirectional relationship traversals"""
        # Look for undirected relationship patterns that could be optimized
        pattern = r"MATCH \((\w+)\)-\[(\w*)\*?(\d*)\]-\((\w+)\)"
        matches = re.finditer(pattern, query, re.IGNORECASE)
        
        modified = False
        optimized_query = query
        
        for match in matches:
            start_node, rel_var, depth, end_node = match.groups()
            
            # If we can determine a likely direction based on node types, optimize
            if self._can_optimize_direction(start_node, end_node, query):
                original_pattern = match.group(0)
                # Make relationship directional
                rel_part = f"[{rel_var}{'*' + depth if depth else ''}]" if rel_var else ""
                optimized_pattern = f"MATCH ({start_node})-{rel_part}->({end_node})"
                optimized_query = optimized_query.replace(original_pattern, optimized_pattern)
                modified = True
        
        return {
            "modified": modified,
            "query": optimized_query,
            "improvement_estimate": 0.3 if modified else 0.0
        }
    
    def _remove_redundant_conditions(self, query: str, pattern_info: Dict) -> Dict:
        """Remove redundant WHERE conditions"""
        # Look for duplicate conditions in WHERE clauses
        where_pattern = r"WHERE\s+(.*?)(?=\s+(?:RETURN|WITH|MATCH|$))"
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if not where_match:
            return {"modified": False, "query": query}
        
        where_clause = where_match.group(1)
        conditions = [cond.strip() for cond in re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)]
        
        # Remove duplicates while preserving order
        unique_conditions = []
        seen = set()
        
        for condition in conditions:
            if condition.lower() not in seen:
                unique_conditions.append(condition)
                seen.add(condition.lower())
        
        if len(unique_conditions) < len(conditions):
            new_where = "WHERE " + " AND ".join(unique_conditions)
            optimized_query = re.sub(
                where_pattern, 
                new_where.replace("WHERE ", ""), 
                query, 
                flags=re.IGNORECASE | Re.DOTALL
            )
            
            return {
                "modified": True,
                "query": optimized_query,
                "improvement_estimate": 0.1
            }
        
        return {"modified": False, "query": query}
    
    def _apply_template_optimization(self, query: str) -> Dict:
        """Apply optimizations based on common query templates"""
        query_lower = query.lower().strip()
        
        # Check if query matches a known template
        for template_name, template_query in self.query_templates.items():
            template_lower = template_query.lower().strip()
            
            # Simple similarity check
            if self._query_similarity(query_lower, template_lower) > 0.8:
                return {
                    "modified": True,
                    "query": template_query,
                    "template_used": template_name
                }
        
        return {"modified": False, "query": query}
    
    def _apply_general_optimizations(self, query: str) -> Dict:
        """Apply general performance optimizations"""
        optimized_query = query
        optimizations = []
        modified = False
        
        # Add PROFILE hint for performance analysis in development
        if self.config.get("monitoring", {}).get("enable_profiling", False):
            if not query.upper().startswith("PROFILE"):
                optimized_query = "PROFILE " + optimized_query
                optimizations.append("profile_added")
                modified = True
        
        # Optimize DISTINCT usage
        if "DISTINCT" in query.upper() and "ORDER BY" in query.upper():
            # DISTINCT with ORDER BY can be expensive, suggest alternatives
            logger.debug("Query uses DISTINCT with ORDER BY - consider optimization")
        
        # Check for potentially expensive operations
        expensive_ops = ["OPTIONAL MATCH", "FOREACH", "UNWIND", "REDUCE"]
        for op in expensive_ops:
            if op in query.upper():
                logger.debug(f"Query contains potentially expensive operation: {op}")
        
        return {
            "modified": modified,
            "query": optimized_query,
            "optimizations": optimizations,
            "improvement_estimate": 0.1 if modified else 0.0
        }
    
    def _has_likely_index(self, label: str, property_name: str) -> bool:
        """Check if a property likely has an index"""
        # Based on schema definition, these properties should have indexes
        indexed_properties = {
            "Concept": ["id", "name", "domain", "type"],
            "Document": ["id", "domain", "timestamp"],
            "Entity": ["name", "domain", "type"],
            "Claim": ["id", "domain", "verified"],
            "Author": ["name", "period", "domain"]
        }
        
        return property_name in indexed_properties.get(label, [])
    
    def _can_optimize_direction(self, start_node: str, end_node: str, query: str) -> bool:
        """Determine if relationship direction can be optimized"""
        # Simple heuristic based on common patterns
        # In practice, this would analyze query context more thoroughly
        
        # Look for patterns that suggest direction
        direction_hints = [
            ("Document", "contains", ["Entity", "Concept"]),
            ("Author", "authored", ["Document", "Concept"]),
            ("Concept", "influences", ["Concept"]),
        ]
        
        query_lower = query.lower()
        for source_type, relationship, target_types in direction_hints:
            if (source_type.lower() in query_lower and 
                any(target.lower() in query_lower for target in target_types)):
                return True
        
        return False
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple token-based similarity
        tokens1 = set(re.findall(r'\w+', query1.lower()))
        tokens2 = set(re.findall(r'\w+', query2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _hash_query(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Create a hash for query caching"""
        query_normalized = re.sub(r'\s+', ' ', query.strip())
        hash_input = query_normalized
        
        if parameters:
            # Include parameter keys (not values) in hash
            param_keys = sorted(parameters.keys())
            hash_input += str(param_keys)
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _update_query_stats(self, query_hash: str, query: str, optimization_result: OptimizationResult):
        """Update query execution statistics"""
        current_time = time.time()
        
        if query_hash in self.query_stats:
            stats = self.query_stats[query_hash]
            stats.execution_count += 1
            stats.last_executed = current_time
        else:
            stats = QueryStats(
                query_hash=query_hash,
                query=query[:200] + "..." if len(query) > 200 else query,
                execution_count=1,
                total_execution_time=0.0,
                avg_execution_time=0.0,
                last_executed=current_time,
                optimizations_applied=optimization_result.optimizations_applied
            )
            self.query_stats[query_hash] = stats
    
    def record_execution_time(self, query_hash: str, execution_time: float):
        """Record actual execution time for a query"""
        if query_hash in self.query_stats:
            stats = self.query_stats[query_hash]
            stats.total_execution_time += execution_time
            stats.avg_execution_time = stats.total_execution_time / stats.execution_count
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        stats_summary = {
            "total_queries": len(self.query_stats),
            "cache_size": len(self.query_cache),
            "most_frequent_queries": [],
            "slowest_queries": [],
            "optimization_success_rate": 0.0
        }
        
        # Sort by execution count
        frequent_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )[:10]
        
        stats_summary["most_frequent_queries"] = [
            {
                "query": stats.query,
                "execution_count": stats.execution_count,
                "avg_execution_time": stats.avg_execution_time
            }
            for stats in frequent_queries
        ]
        
        # Sort by average execution time
        slow_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.avg_execution_time,
            reverse=True
        )[:10]
        
        stats_summary["slowest_queries"] = [
            {
                "query": stats.query,
                "avg_execution_time": stats.avg_execution_time,
                "execution_count": stats.execution_count
            }
            for stats in slow_queries
        ]
        
        # Calculate optimization success rate
        optimized_queries = sum(
            1 for stats in self.query_stats.values() 
            if stats.optimizations_applied
        )
        
        if self.query_stats:
            stats_summary["optimization_success_rate"] = optimized_queries / len(self.query_stats)
        
        return stats_summary
    
    def suggest_optimizations(self, query: str) -> List[Dict[str, Any]]:
        """Suggest manual optimizations for a query"""
        suggestions = []
        
        # Check for common anti-patterns
        if "MATCH ()" in query.upper():
            suggestions.append({
                "type": "anti_pattern",
                "description": "Avoid unbound MATCH patterns - use specific labels",
                "impact": "high"
            })
        
        if re.search(r"WHERE.*=.*OR.*=", query, re.IGNORECASE):
            suggestions.append({
                "type": "index_usage",
                "description": "Consider using IN clause instead of multiple OR conditions",
                "impact": "medium"
            })
        
        if "OPTIONAL MATCH" in query.upper() and "WHERE" in query.upper():
            suggestions.append({
                "type": "optional_match",
                "description": "Review if OPTIONAL MATCH is necessary with WHERE conditions",
                "impact": "medium"
            })
        
        if not re.search(r"LIMIT \d+", query, re.IGNORECASE):
            suggestions.append({
                "type": "limit_missing",
                "description": "Consider adding LIMIT to prevent large result sets",
                "impact": "low"
            })
        
        return suggestions
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def clear_stats(self):
        """Clear query statistics"""
        self.query_stats.clear()
        logger.info("Query statistics cleared")