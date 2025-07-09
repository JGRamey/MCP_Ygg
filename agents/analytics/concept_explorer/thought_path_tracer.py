#!/usr/bin/env python3
"""
Thought Path Tracer
Traces paths of reasoning between concepts for advanced knowledge discovery
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import networkx as nx
from itertools import combinations

logger = logging.getLogger(__name__)

@dataclass
class ThoughtStep:
    """A single step in a thought path"""
    from_concept: str
    to_concept: str
    reasoning: str
    confidence: float
    relationship_type: str
    evidence: List[str]

@dataclass 
class ThoughtPath:
    """A complete path of reasoning"""
    path_id: str
    start_concept: str
    end_concept: str
    steps: List[ThoughtStep]
    total_confidence: float
    path_length: int
    reasoning_chain: List[str]
    domains_traversed: List[str]
    novelty_score: float

@dataclass
class ReasoningPattern:
    """A pattern of reasoning discovered in the graph"""
    pattern_id: str
    description: str
    frequency: int
    concepts_involved: List[str]
    pattern_strength: float

class ThoughtPathTracer:
    """Advanced tracer for paths of reasoning between concepts"""
    
    def __init__(self, max_path_length: int = 6):
        self.reasoning_graph = nx.DiGraph()
        self.concept_domains = {}
        self.reasoning_patterns = []
        self.max_path_length = max_path_length
        
        # Define reasoning type weights
        self.reasoning_weights = {
            "causal": 0.9,
            "temporal": 0.8,
            "similarity": 0.7,
            "containment": 0.8,
            "influence": 0.85,
            "opposition": 0.6,
            "cross_domain_bridge": 0.95,
            "semantic_similarity": 0.6,
            "cooccurrence": 0.5
        }
        
    async def trace_path(self, 
                        start_concept: str, 
                        end_concept: str,
                        concept_graph: nx.Graph,
                        max_paths: int = 5) -> List[ThoughtPath]:
        """Trace sophisticated reasoning paths between concepts"""
        try:
            paths = []
            
            if start_concept == end_concept:
                return paths
            
            # Check if concepts exist in the graph
            if start_concept not in concept_graph or end_concept not in concept_graph:
                logger.warning(f"Concepts {start_concept} or {end_concept} not found in graph")
                return paths
            
            # Find all simple paths up to max length
            try:
                all_simple_paths = list(nx.all_simple_paths(
                    concept_graph, 
                    start_concept, 
                    end_concept, 
                    cutoff=self.max_path_length
                ))
                
                # Score and rank paths
                scored_paths = []
                for i, path_nodes in enumerate(all_simple_paths[:20]):  # Limit to avoid explosion
                    thought_path = await self._create_thought_path(
                        f"path_{i}_{start_concept}_{end_concept}",
                        path_nodes,
                        concept_graph
                    )
                    if thought_path:
                        scored_paths.append(thought_path)
                
                # Sort by total confidence and novelty
                scored_paths.sort(
                    key=lambda p: (p.total_confidence * p.novelty_score), 
                    reverse=True
                )
                
                paths = scored_paths[:max_paths]
                
            except nx.NetworkXNoPath:
                logger.info(f"No direct path found between {start_concept} and {end_concept}")
                
                # Try to find indirect paths through hubs
                hub_paths = await self._find_hub_mediated_paths(
                    start_concept, end_concept, concept_graph
                )
                paths.extend(hub_paths[:max_paths])
            
            return paths
            
        except Exception as e:
            logger.error(f"Error tracing thought path: {e}")
            return []
    
    async def _create_thought_path(self, 
                                 path_id: str,
                                 path_nodes: List[str],
                                 concept_graph: nx.Graph) -> Optional[ThoughtPath]:
        """Create a thought path from a sequence of nodes"""
        try:
            if len(path_nodes) < 2:
                return None
            
            steps = []
            reasoning_chain = []
            domains_traversed = []
            total_confidence = 0.0
            
            for i in range(len(path_nodes) - 1):
                current_node = path_nodes[i]
                next_node = path_nodes[i + 1]
                
                # Get edge data
                edge_data = concept_graph.get_edge_data(current_node, next_node, {})
                
                # Get node data for context
                current_data = concept_graph.nodes.get(current_node, {})
                next_data = concept_graph.nodes.get(next_node, {})
                
                current_name = current_data.get('name', current_node)
                next_name = next_data.get('name', next_node)
                current_domain = current_data.get('domain', 'unknown')
                next_domain = next_data.get('domain', 'unknown')
                
                # Track domains
                if current_domain not in domains_traversed:
                    domains_traversed.append(current_domain)
                if next_domain not in domains_traversed:
                    domains_traversed.append(next_domain)
                
                # Determine reasoning type and confidence
                relationship_type = edge_data.get('relationship_type', 'unknown')
                base_confidence = edge_data.get('confidence', 0.5)
                
                # Adjust confidence based on reasoning type
                type_weight = self.reasoning_weights.get(relationship_type, 0.5)
                step_confidence = base_confidence * type_weight
                
                # Generate reasoning explanation
                reasoning = self._generate_reasoning_explanation(
                    current_name, next_name, relationship_type, current_domain, next_domain
                )
                
                # Create step
                step = ThoughtStep(
                    from_concept=current_node,
                    to_concept=next_node,
                    reasoning=reasoning,
                    confidence=step_confidence,
                    relationship_type=relationship_type,
                    evidence=edge_data.get('evidence', [])
                )
                
                steps.append(step)
                reasoning_chain.append(reasoning)
                total_confidence += step_confidence
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(steps) if steps else 0.0
            
            # Calculate novelty score
            novelty_score = self._calculate_path_novelty(
                path_nodes, domains_traversed, len(steps)
            )
            
            return ThoughtPath(
                path_id=path_id,
                start_concept=path_nodes[0],
                end_concept=path_nodes[-1],
                steps=steps,
                total_confidence=avg_confidence,
                path_length=len(steps),
                reasoning_chain=reasoning_chain,
                domains_traversed=domains_traversed,
                novelty_score=novelty_score
            )
            
        except Exception as e:
            logger.error(f"Error creating thought path: {e}")
            return None
    
    def _generate_reasoning_explanation(self, 
                                      from_concept: str,
                                      to_concept: str,
                                      relationship_type: str,
                                      from_domain: str,
                                      to_domain: str) -> str:
        """Generate human-readable reasoning explanation"""
        
        explanations = {
            "causal": f"'{from_concept}' causes or leads to '{to_concept}'",
            "temporal": f"'{from_concept}' precedes '{to_concept}' in time",
            "similarity": f"'{from_concept}' is similar to '{to_concept}'",
            "containment": f"'{from_concept}' contains or includes '{to_concept}'",
            "influence": f"'{from_concept}' influences '{to_concept}'",
            "opposition": f"'{from_concept}' opposes or contrasts with '{to_concept}'",
            "cross_domain_bridge": f"'{from_concept}' ({from_domain}) bridges to '{to_concept}' ({to_domain})",
            "semantic_similarity": f"'{from_concept}' shares semantic meaning with '{to_concept}'",
            "cooccurrence": f"'{from_concept}' frequently appears together with '{to_concept}'"
        }
        
        base_explanation = explanations.get(
            relationship_type, 
            f"'{from_concept}' relates to '{to_concept}'"
        )
        
        # Add domain context if crossing domains
        if from_domain != to_domain and from_domain != 'unknown' and to_domain != 'unknown':
            base_explanation += f" (bridging {from_domain} and {to_domain} domains)"
        
        return base_explanation
    
    def _calculate_path_novelty(self, 
                              path_nodes: List[str],
                              domains_traversed: List[str],
                              path_length: int) -> float:
        """Calculate novelty score for a reasoning path"""
        novelty_score = 0.5  # Base score
        
        # Cross-domain paths are more novel
        if len(set(domains_traversed)) > 1:
            novelty_score += 0.3
        
        # Longer paths that are still coherent are more novel
        if 3 <= path_length <= 5:
            novelty_score += 0.2
        elif path_length > 5:
            novelty_score += 0.1  # Very long paths lose some coherence
        
        # Paths involving rare concepts are more novel
        # (This would require frequency analysis in practice)
        
        return min(novelty_score, 1.0)
    
    async def _find_hub_mediated_paths(self, 
                                     start_concept: str,
                                     end_concept: str,
                                     concept_graph: nx.Graph) -> List[ThoughtPath]:
        """Find paths mediated through high-centrality hub concepts"""
        try:
            paths = []
            
            # Calculate centrality measures to find hubs
            degree_centrality = nx.degree_centrality(concept_graph)
            betweenness_centrality = nx.betweenness_centrality(concept_graph)
            
            # Identify top hubs
            hubs = []
            for node, degree in degree_centrality.items():
                betweenness = betweenness_centrality.get(node, 0)
                hub_score = (degree + betweenness) / 2
                
                if hub_score > 0.1 and node not in [start_concept, end_concept]:
                    hubs.append((node, hub_score))
            
            # Sort hubs by score
            hubs.sort(key=lambda x: x[1], reverse=True)
            
            # Try to find paths through top hubs
            for hub_node, hub_score in hubs[:5]:  # Try top 5 hubs
                try:
                    # Path from start to hub
                    start_to_hub = list(nx.shortest_path(concept_graph, start_concept, hub_node))
                    
                    # Path from hub to end
                    hub_to_end = list(nx.shortest_path(concept_graph, hub_node, end_concept))
                    
                    # Combine paths (remove duplicate hub node)
                    full_path = start_to_hub + hub_to_end[1:]
                    
                    if len(full_path) <= self.max_path_length:
                        thought_path = await self._create_thought_path(
                            f"hub_path_{start_concept}_{hub_node}_{end_concept}",
                            full_path,
                            concept_graph
                        )
                        
                        if thought_path:
                            # Boost confidence for hub-mediated paths
                            thought_path.total_confidence *= (1 + hub_score * 0.2)
                            paths.append(thought_path)
                
                except nx.NetworkXNoPath:
                    continue
            
            return paths
            
        except Exception as e:
            logger.error(f"Error finding hub-mediated paths: {e}")
            return []
    
    async def discover_reasoning_patterns(self, concept_graph: nx.Graph) -> List[ReasoningPattern]:
        """Discover common patterns of reasoning in the concept graph"""
        try:
            patterns = []
            
            # Analyze common relationship sequences
            relationship_sequences = {}
            
            # Sample paths to analyze patterns
            nodes = list(concept_graph.nodes())
            sample_pairs = list(combinations(nodes[:50], 2))  # Sample to avoid explosion
            
            for start, end in sample_pairs:
                try:
                    paths = list(nx.all_simple_paths(concept_graph, start, end, cutoff=4))
                    
                    for path in paths[:3]:  # Sample paths
                        if len(path) >= 3:  # Need at least 2 relationships
                            sequence = []
                            for i in range(len(path) - 1):
                                edge_data = concept_graph.get_edge_data(path[i], path[i+1], {})
                                rel_type = edge_data.get('relationship_type', 'unknown')
                                sequence.append(rel_type)
                            
                            sequence_key = " -> ".join(sequence)
                            if sequence_key not in relationship_sequences:
                                relationship_sequences[sequence_key] = {
                                    "count": 0,
                                    "examples": []
                                }
                            
                            relationship_sequences[sequence_key]["count"] += 1
                            relationship_sequences[sequence_key]["examples"].append(path)
                
                except nx.NetworkXNoPath:
                    continue
            
            # Create patterns from frequent sequences
            for sequence, data in relationship_sequences.items():
                if data["count"] >= 3:  # Pattern appears at least 3 times
                    pattern = ReasoningPattern(
                        pattern_id=f"pattern_{len(patterns)}",
                        description=f"Reasoning pattern: {sequence}",
                        frequency=data["count"],
                        concepts_involved=[],  # Could be extracted from examples
                        pattern_strength=min(data["count"] / 10.0, 1.0)
                    )
                    patterns.append(pattern)
            
            # Sort by frequency
            patterns.sort(key=lambda p: p.frequency, reverse=True)
            
            logger.info(f"Discovered {len(patterns)} reasoning patterns")
            return patterns[:20]  # Return top 20 patterns
            
        except Exception as e:
            logger.error(f"Error discovering reasoning patterns: {e}")
            return []
    
    def add_reasoning_connection(self, 
                               from_concept: str, 
                               to_concept: str, 
                               reasoning: str, 
                               confidence: float,
                               relationship_type: str = "manual"):
        """Add a reasoning connection to the graph"""
        self.reasoning_graph.add_edge(
            from_concept, 
            to_concept, 
            reasoning=reasoning, 
            confidence=confidence,
            relationship_type=relationship_type
        )
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning graph"""
        try:
            stats = {
                "total_concepts": len(self.reasoning_graph.nodes),
                "total_connections": len(self.reasoning_graph.edges),
                "average_connections_per_concept": 0,
                "most_connected_concepts": [],
                "reasoning_patterns": len(self.reasoning_patterns)
            }
            
            if stats["total_concepts"] > 0:
                stats["average_connections_per_concept"] = (
                    stats["total_connections"] / stats["total_concepts"]
                )
            
            # Find most connected concepts
            degree_dict = dict(self.reasoning_graph.degree())
            if degree_dict:
                sorted_by_degree = sorted(
                    degree_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                stats["most_connected_concepts"] = sorted_by_degree[:10]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting reasoning statistics: {e}")
            return {}