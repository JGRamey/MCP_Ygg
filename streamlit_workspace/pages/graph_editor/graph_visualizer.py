"""
Graph Visualization Engine

This module handles the creation and rendering of interactive network graphs
using NetworkX and Plotly for the Graph Editor interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from .models import DOMAIN_COLORS, GraphSettings
from .neo4j_connector import Neo4jConnector


class GraphVisualizer:
    """Creates and manages graph visualizations"""

    def __init__(self):
        self.neo4j_connector = Neo4jConnector()

    def create_network_graph(
        self,
        concepts: List[Dict[str, Any]],
        settings: GraphSettings,
        relationship_filters: Dict[str, Any] = None,
    ) -> Optional[go.Figure]:
        """
        Main network graph creation function

        Purpose: Converts concept data into interactive Plotly network visualization
        Use: Called when displaying full network or domain-specific graphs
        """
        try:
            if not concepts:
                return None

            # Create NetworkX graph structure
            G = self._build_networkx_graph(concepts, relationship_filters)

            if len(G.nodes()) == 0:
                return None

            # Generate node positions using specified layout
            pos = self._generate_layout(G, settings.layout_type)

            # Create Plotly figure
            fig = self._create_plotly_figure(G, pos, settings)

            return fig

        except Exception as e:
            return None

    def create_focused_graph(
        self, concept_id: str, depth: int, settings: GraphSettings
    ) -> Optional[go.Figure]:
        """
        Focused view graph creation

        Purpose: Creates a graph centered on a specific concept with limited depth
        Use: Called when user wants to focus on a particular concept and its connections
        """
        try:
            # Get central concept
            central_concept = self.neo4j_connector.get_concept_by_id(concept_id)
            if not central_concept:
                return None

            # Build focused network
            G = self._build_focused_network(concept_id, central_concept, depth)

            if len(G.nodes()) <= 1:
                return None

            # Generate layout and create visualization
            pos = self._generate_layout(G, settings.layout_type)
            fig = self._create_focused_plotly_figure(G, pos, settings, concept_id)

            return fig

        except Exception:
            return None

    def _build_networkx_graph(
        self,
        concepts: List[Dict[str, Any]],
        relationship_filters: Dict[str, Any] = None,
    ) -> nx.Graph:
        """
        NetworkX graph construction from concepts

        Purpose: Converts concept data into NetworkX graph structure with nodes and edges
        Use: Internal helper for preparing graph data structure
        """
        G = nx.Graph()

        # Add nodes from concepts
        for concept in concepts:
            G.add_node(
                concept["id"],
                name=concept["name"],
                domain=concept["domain"],
                type=concept["type"],
                level=concept.get("level", 1),
            )

        # Add edges from relationships (limited for performance)
        relationship_filters = relationship_filters or {
            "show": True,
            "min_strength": 0.0,
        }

        if relationship_filters.get("show", True):
            for concept in concepts[:50]:  # Limit for performance
                relationships = self.neo4j_connector.get_concept_relationships(
                    concept["id"]
                )
                for rel in relationships:
                    if rel["target_id"] in G.nodes():
                        strength = rel.get("strength", 0.5)
                        if strength >= relationship_filters.get("min_strength", 0.0):
                            G.add_edge(
                                concept["id"],
                                rel["target_id"],
                                relationship=rel["type"],
                                strength=strength,
                            )

        return G

    def _build_focused_network(
        self, concept_id: str, central_concept: Dict[str, Any], depth: int
    ) -> nx.Graph:
        """
        Focused network construction

        Purpose: Builds a network centered on one concept up to specified depth
        Use: Internal helper for focused view visualization
        """
        G = nx.Graph()

        # Add central node
        G.add_node(
            concept_id,
            name=central_concept["name"],
            domain=central_concept["domain"],
            type=central_concept["type"],
            level=central_concept.get("level", 1),
            distance=0,
        )

        # BFS exploration up to specified depth
        to_explore = [(concept_id, 0)]
        explored = {concept_id}

        while to_explore:
            current_id, current_depth = to_explore.pop(0)

            if current_depth < depth:
                relationships = self.neo4j_connector.get_concept_relationships(
                    current_id
                )

                for rel in relationships:
                    target_id = rel["target_id"]

                    if target_id not in explored:
                        target_concept = self.neo4j_connector.get_concept_by_id(
                            target_id
                        )
                        if target_concept:
                            G.add_node(
                                target_id,
                                name=target_concept["name"],
                                domain=target_concept["domain"],
                                type=target_concept["type"],
                                level=target_concept.get("level", 1),
                                distance=current_depth + 1,
                            )
                            explored.add(target_id)
                            to_explore.append((target_id, current_depth + 1))

                    if target_id in G.nodes():
                        G.add_edge(
                            current_id,
                            target_id,
                            relationship=rel["type"],
                            strength=rel.get("strength", 0.5),
                        )

        return G

    def _generate_layout(
        self, G: nx.Graph, layout_type: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Graph layout generation

        Purpose: Calculates node positions based on selected layout algorithm
        Use: Internal helper for positioning nodes in visualization
        """
        layout_functions = {
            "spring": lambda g: nx.spring_layout(g, k=1, iterations=50),
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
        }

        layout_func = layout_functions.get(layout_type, layout_functions["spring"])
        return layout_func(G)

    def _create_plotly_figure(
        self, G: nx.Graph, pos: Dict[str, Tuple[float, float]], settings: GraphSettings
    ) -> go.Figure:
        """
        Plotly figure creation for network graph

        Purpose: Converts NetworkX graph to interactive Plotly visualization
        Use: Internal helper for rendering the final graph visualization
        """
        fig = go.Figure()

        # Add edges to the plot
        self._add_edges_to_figure(fig, G, pos, settings.edge_width)

        # Add nodes to the plot
        self._add_nodes_to_figure(fig, G, pos, settings.node_size)

        # Configure layout
        fig.update_layout(
            title=f"Knowledge Graph Network ({len(G.nodes())} concepts, {len(G.edges())} relationships)",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"Layout: {settings.layout_type} | Nodes: {len(G.nodes())} | Edges: {len(G.edges())}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def _create_focused_plotly_figure(
        self,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        settings: GraphSettings,
        focus_node: str,
    ) -> go.Figure:
        """
        Focused view Plotly figure creation

        Purpose: Creates specialized visualization for focused view with distance-based coloring
        Use: Internal helper for focused view rendering
        """
        fig = go.Figure()

        # Add edges
        self._add_edges_to_figure(fig, G, pos, settings.edge_width)

        # Add nodes with distance-based coloring
        self._add_focused_nodes_to_figure(fig, G, pos, settings.node_size, focus_node)

        fig.update_layout(
            title=f"Focused View: {focus_node} ({len(G.nodes())} concepts)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig

    def _add_edges_to_figure(
        self,
        fig: go.Figure,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        edge_width: int,
    ):
        """
        Edge rendering helper

        Purpose: Adds relationship lines to the Plotly figure
        Use: Internal helper for drawing connections between concepts
        """
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=edge_width, color="rgba(125,125,125,0.5)"),
                hoverinfo="none",
                mode="lines",
            )
        )

    def _add_nodes_to_figure(
        self,
        fig: go.Figure,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        node_size: int,
    ):
        """
        Node rendering helper

        Purpose: Adds concept nodes to the Plotly figure with domain-based coloring
        Use: Internal helper for drawing concept nodes in the visualization
        """
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size_list = []
        hover_text = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            # Node properties
            name = node[1]["name"]
            domain = node[1]["domain"]
            node_text.append(f"{node[0]}")
            node_color.append(DOMAIN_COLORS.get(domain, "#95A5A6"))

            # Size based on connections
            connections = G.degree(node[0])
            size = max(node_size, min(node_size * 2, node_size + connections * 2))
            node_size_list.append(size)

            # Hover information
            hover_text.append(
                f"{name}<br>Domain: {domain}<br>Type: {node[1]['type']}<br>Connections: {connections}"
            )

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_size_list,
                    color=node_color,
                    line=dict(width=2, color="white"),
                ),
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    def _add_focused_nodes_to_figure(
        self,
        fig: go.Figure,
        G: nx.Graph,
        pos: Dict[str, Tuple[float, float]],
        node_size: int,
        focus_node: str,
    ):
        """
        Focused view node rendering

        Purpose: Adds nodes with distance-based coloring for focused view
        Use: Internal helper for focused view node visualization
        """
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        hover_text = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])

            # Color based on distance from focus node
            if node[0] == focus_node:
                node_color.append("red")  # Central node
            else:
                distance = node[1].get("distance", 1)
                alpha = 1.0 - distance * 0.3
                node_color.append(f"rgba(70, 130, 180, {alpha})")

            hover_text.append(
                f"{node[1]['name']}<br>Domain: {node[1]['domain']}<br>Distance: {node[1].get('distance', 0)}"
            )

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=node_size, color=node_color),
                text=node_text,
                textposition="middle center",
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
