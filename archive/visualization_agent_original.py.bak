"""
Visualization Agent for MCP Server
Generates interactive Yggdrasil tree charts and other visualizations for Neo4j graphs.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import base64
from jinja2 import Template

from neo4j import AsyncGraphDatabase, AsyncDriver
import networkx as nx
import numpy as np


class VisualizationType(Enum):
    """Types of visualizations that can be generated."""
    YGGDRASIL_TREE = "yggdrasil_tree"
    NETWORK_GRAPH = "network_graph"
    TIMELINE = "timeline"
    DOMAIN_CLUSTER = "domain_cluster"
    AUTHORITY_MAP = "authority_map"
    CONCEPT_MAP = "concept_map"
    RELATIONSHIP_FLOW = "relationship_flow"


class NodeType(Enum):
    """Node types with corresponding colors."""
    DOCUMENT = "document"
    CONCEPT = "concept"
    PERSON = "person"
    EVENT = "event"
    PATTERN = "pattern"
    ROOT = "root"
    DOMAIN = "domain"


@dataclass
class VisualizationNode:
    """Represents a node in the visualization."""
    id: str
    label: str
    title: str
    node_type: NodeType
    domain: Optional[str]
    date: Optional[str]
    level: int
    x: Optional[float] = None
    y: Optional[float] = None
    size: Optional[float] = None
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationEdge:
    """Represents an edge in the visualization."""
    id: str
    source: str
    target: str
    relationship_type: str
    weight: float = 1.0
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationData:
    """Complete visualization data structure."""
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any]
    layout_type: str
    filters: Dict[str, Any]


class VisualizationConfig:
    """Configuration for visualization generation."""
    
    def __init__(self, config_path: str = "config/visualization.yaml"):
        # Database connection
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        
        # Color mappings (as specified in the project plan)
        self.node_colors = {
            NodeType.DOCUMENT: "#3498db",      # Blue
            NodeType.CONCEPT: "#e74c3c",       # Red
            NodeType.PERSON: "#2ecc71",        # Green
            NodeType.EVENT: "#f1c40f",         # Yellow
            NodeType.PATTERN: "#9b59b6",       # Purple
            NodeType.ROOT: "#34495e",          # Dark gray
            NodeType.DOMAIN: "#e67e22"         # Orange
        }
        
        # Edge colors
        self.edge_colors = {
            "DERIVED_FROM": "#95a5a6",
            "REFERENCES": "#3498db",
            "INFLUENCED_BY": "#e74c3c",
            "CONTAINS_CONCEPT": "#2ecc71",
            "SIMILAR_CONCEPT": "#9b59b6",
            "default": "#bdc3c7"
        }
        
        # Layout parameters
        self.tree_spacing = {
            "node_distance": 100,
            "level_distance": 200,
            "root_x": 0,
            "root_y": 0
        }
        
        # Size parameters
        self.node_sizes = {
            "min_size": 10,
            "max_size": 50,
            "size_factor": 1.5
        }
        
        # Visualization parameters
        self.max_nodes = 1000
        self.max_edges = 2000
        self.enable_physics = True
        self.show_labels = True
        self.enable_zoom = True
        self.enable_pan = True
        
        # Template settings
        self.template_dir = "visualization/templates"
        self.output_dir = "visualization/output"
        
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


class ChartGenerator:
    """Main chart generation class."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the chart generator."""
        self.config = config or VisualizationConfig()
        self.neo4j_driver: Optional[AsyncDriver] = None
        
        # Template paths
        self.template_dir = Path(self.config.template_dir)
        self.output_dir = Path(self.config.output_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize templates
        self._initialize_templates()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("chart_generator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize database connections."""
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
            
            self.logger.info("Chart generator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chart generator: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        self.logger.info("Chart generator closed")
    
    def _initialize_templates(self) -> None:
        """Initialize HTML templates."""
        # Vis.js template for interactive graphs
        self.visjs_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        #visualization { width: 100%; height: 800px; border: 1px solid #ccc; }
        .controls { margin-bottom: 20px; }
        .filter-group { display: inline-block; margin-right: 20px; }
        .legend { margin-top: 20px; }
        .legend-item { display: inline-block; margin-right: 15px; }
        .legend-color { width: 20px; height: 20px; display: inline-block; margin-right: 5px; }
        .info-panel { position: absolute; top: 20px; right: 20px; width: 300px; background: #f9f9f9; padding: 15px; border-radius: 5px; display: none; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="controls">
        <div class="filter-group">
            <label>Domain:</label>
            <select id="domainFilter">
                <option value="">All Domains</option>
                {% for domain in domains %}
                <option value="{{ domain }}">{{ domain }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="filter-group">
            <label>Node Type:</label>
            <select id="nodeTypeFilter">
                <option value="">All Types</option>
                {% for node_type in node_types %}
                <option value="{{ node_type }}">{{ node_type }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="filter-group">
            <label>Date Range:</label>
            <input type="number" id="startYear" placeholder="Start Year" />
            <input type="number" id="endYear" placeholder="End Year" />
            <button onclick="applyDateFilter()">Apply</button>
        </div>
        
        <div class="filter-group">
            <button onclick="resetFilters()">Reset Filters</button>
            <button onclick="exportSVG()">Export SVG</button>
            <button onclick="exportPNG()">Export PNG</button>
        </div>
    </div>
    
    <div id="visualization"></div>
    
    <div class="info-panel" id="infoPanel">
        <h3 id="infoTitle">Node Information</h3>
        <div id="infoContent"></div>
        <button onclick="closeInfoPanel()">Close</button>
    </div>
    
    <div class="legend">
        <h3>Legend</h3>
        {% for node_type, color in node_colors.items() %}
        <div class="legend-item">
            <span class="legend-color" style="background-color: {{ color }};"></span>
            <span>{{ node_type }}</span>
        </div>
        {% endfor %}
    </div>

    <script>
        // Data
        const nodes = new vis.DataSet({{ nodes_json | safe }});
        const edges = new vis.DataSet({{ edges_json | safe }});
        const allNodes = nodes.get();
        const allEdges = edges.get();
        
        // Configuration
        const options = {
            {% if layout_type == "hierarchical" %}
            layout: {
                hierarchical: {
                    direction: "UD",
                    sortMethod: "directed",
                    nodeSpacing: {{ node_spacing }},
                    levelSeparation: {{ level_separation }},
                    treeSpacing: {{ tree_spacing }}
                }
            },
            {% endif %}
            physics: {
                enabled: {{ physics_enabled | lower }},
                stabilization: {iterations: 100}
            },
            interaction: {
                zoomView: {{ enable_zoom | lower }},
                dragView: {{ enable_pan | lower }},
                selectConnectedEdges: false
            },
            nodes: {
                shape: "dot",
                scaling: {
                    min: {{ min_node_size }},
                    max: {{ max_node_size }}
                },
                font: {
                    size: 14,
                    color: "#333"
                },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 2,
                color: {inherit: false},
                smooth: {
                    type: "dynamic"
                },
                arrows: {
                    to: {enabled: true, scaleFactor: 1}
                }
            }
        };
        
        // Initialize network
        const container = document.getElementById('visualization');
        const data = {nodes: nodes, edges: edges};
        const network = new vis.Network(container, data, options);
        
        // Event handlers
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                showNodeInfo(node);
            }
        });
        
        function showNodeInfo(node) {
            const infoPanel = document.getElementById('infoPanel');
            const infoTitle = document.getElementById('infoTitle');
            const infoContent = document.getElementById('infoContent');
            
            infoTitle.textContent = node.label;
            infoContent.innerHTML = `
                <p><strong>Type:</strong> ${node.node_type}</p>
                <p><strong>Domain:</strong> ${node.domain || 'N/A'}</p>
                <p><strong>Date:</strong> ${node.date || 'N/A'}</p>
                <p><strong>Description:</strong> ${node.title}</p>
                ${node.metadata ? '<p><strong>Metadata:</strong> ' + JSON.stringify(node.metadata, null, 2) + '</p>' : ''}
            `;
            infoPanel.style.display = 'block';
        }
        
        function closeInfoPanel() {
            document.getElementById('infoPanel').style.display = 'none';
        }
        
        // Filter functions
        function applyFilters() {
            const domainFilter = document.getElementById('domainFilter').value;
            const nodeTypeFilter = document.getElementById('nodeTypeFilter').value;
            
            let filteredNodes = allNodes.filter(node => {
                if (domainFilter && node.domain !== domainFilter) return false;
                if (nodeTypeFilter && node.node_type !== nodeTypeFilter) return false;
                return true;
            });
            
            const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredEdges = allEdges.filter(edge => 
                filteredNodeIds.has(edge.from) && filteredNodeIds.has(edge.to)
            );
            
            nodes.clear();
            edges.clear();
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
        }
        
        function applyDateFilter() {
            const startYear = parseInt(document.getElementById('startYear').value);
            const endYear = parseInt(document.getElementById('endYear').value);
            
            let filteredNodes = allNodes.filter(node => {
                if (!node.date) return true;
                const nodeYear = new Date(node.date).getFullYear();
                if (startYear && nodeYear < startYear) return false;
                if (endYear && nodeYear > endYear) return false;
                return true;
            });
            
            const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredEdges = allEdges.filter(edge => 
                filteredNodeIds.has(edge.from) && filteredNodeIds.has(edge.to)
            );
            
            nodes.clear();
            edges.clear();
            nodes.add(filteredNodes);
            edges.add(filteredEdges);
        }
        
        function resetFilters() {
            document.getElementById('domainFilter').value = '';
            document.getElementById('nodeTypeFilter').value = '';
            document.getElementById('startYear').value = '';
            document.getElementById('endYear').value = '';
            
            nodes.clear();
            edges.clear();
            nodes.add(allNodes);
            edges.add(allEdges);
        }
        
        function exportSVG() {
            // This would require additional libraries for SVG export
            alert('SVG export functionality would be implemented with additional libraries');
        }
        
        function exportPNG() {
            // This would require additional libraries for PNG export
            alert('PNG export functionality would be implemented with additional libraries');
        }
        
        // Add filter event listeners
        document.getElementById('domainFilter').addEventListener('change', applyFilters);
        document.getElementById('nodeTypeFilter').addEventListener('change', applyFilters);
    </script>
</body>
</html>
        """)
        
        # Save template to file
        template_file = self.template_dir / "visjs_template.html"
        with open(template_file, 'w') as f:
            f.write(self.visjs_template.template.source)
    
    async def generate_yggdrasil_chart(
        self,
        title: str = "Yggdrasil Knowledge Tree",
        domain_filter: Optional[str] = None,
        max_depth: int = 5,
        export_format: str = "html"
    ) -> str:
        """Generate the main Yggdrasil tree visualization."""
        
        try:
            # Get data from Neo4j
            viz_data = await self._get_yggdrasil_data(domain_filter, max_depth)
            
            # Apply Yggdrasil layout
            self._apply_yggdrasil_layout(viz_data)
            
            # Generate visualization
            output_file = await self._generate_interactive_chart(
                viz_data,
                title,
                VisualizationType.YGGDRASIL_TREE,
                export_format
            )
            
            self.logger.info(f"Generated Yggdrasil chart: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating Yggdrasil chart: {e}")
            raise
    
    async def _get_yggdrasil_data(self, domain_filter: Optional[str], max_depth: int) -> VisualizationData:
        """Get data structured for Yggdrasil visualization."""
        
        async with self.neo4j_driver.session() as session:
            # Build query for Yggdrasil structure
            domain_clause = "WHERE n.domain = $domain" if domain_filter else ""
            
            query = f"""
            // Get root and domain nodes
            MATCH (root:Root)
            OPTIONAL MATCH (root)-[:HAS_DOMAIN]->(domain:Domain) {domain_clause}
            
            // Get documents and their relationships with depth limit
            OPTIONAL MATCH path = (domain)-[:CONTAINS*1..{max_depth}]->(doc:Document)
            WITH root, domain, doc, path
            
            // Get all relationships
            OPTIONAL MATCH (doc)-[r]-(related)
            
            RETURN 
                root,
                collect(DISTINCT domain) as domains,
                collect(DISTINCT doc) as documents,
                collect(DISTINCT {{source: startNode(r), target: endNode(r), relationship: r}}) as relationships,
                collect(DISTINCT path) as paths
            """
            
            params = {"domain": domain_filter} if domain_filter else {}
            result = await session.run(query, params)
            record = await result.single()
            
            if not record:
                return VisualizationData([], [], {}, "hierarchical", {})
            
            nodes = []
            edges = []
            node_ids = set()
            
            # Add root node
            if record['root']:
                root_node = VisualizationNode(
                    id="root",
                    label="World Knowledge",
                    title="Root of all knowledge domains",
                    node_type=NodeType.ROOT,
                    domain=None,
                    date=None,
                    level=0
                )
                nodes.append(root_node)
                node_ids.add("root")
            
            # Add domain nodes
            for i, domain in enumerate(record['domains'] or []):
                if domain:
                    domain_id = f"domain_{domain['name']}"
                    domain_node = VisualizationNode(
                        id=domain_id,
                        label=domain['name'],
                        title=domain.get('description', f"Domain: {domain['name']}"),
                        node_type=NodeType.DOMAIN,
                        domain=domain['name'],
                        date=None,
                        level=1
                    )
                    nodes.append(domain_node)
                    node_ids.add(domain_id)
                    
                    # Add edge from root to domain
                    if "root" in node_ids:
                        edge = VisualizationEdge(
                            id=f"root_to_{domain_id}",
                            source="root",
                            target=domain_id,
                            relationship_type="HAS_DOMAIN"
                        )
                        edges.append(edge)
            
            # Add document nodes
            for doc in record['documents'] or []:
                if doc:
                    doc_id = str(doc.id)
                    if doc_id not in node_ids:
                        # Determine node type based on labels
                        node_type = NodeType.DOCUMENT
                        if 'Person' in doc.labels:
                            node_type = NodeType.PERSON
                        elif 'Concept' in doc.labels:
                            node_type = NodeType.CONCEPT
                        elif 'Event' in doc.labels:
                            node_type = NodeType.EVENT
                        elif 'Pattern' in doc.labels:
                            node_type = NodeType.PATTERN
                        
                        # Calculate level based on date (newer = higher level)
                        level = self._calculate_temporal_level(doc.get('date'))
                        
                        doc_node = VisualizationNode(
                            id=doc_id,
                            label=doc.get('title', 'Unknown'),
                            title=doc.get('title', 'Unknown'),
                            node_type=node_type,
                            domain=doc.get('domain'),
                            date=doc.get('date'),
                            level=level,
                            metadata={
                                'author': doc.get('author'),
                                'source': doc.get('source'),
                                'word_count': doc.get('word_count')
                            }
                        )
                        nodes.append(doc_node)
                        node_ids.add(doc_id)
            
            # Add relationship edges
            for rel_data in record['relationships'] or []:
                if rel_data and rel_data['source'] and rel_data['target']:
                    source_id = str(rel_data['source'].id)
                    target_id = str(rel_data['target'].id)
                    
                    if source_id in node_ids and target_id in node_ids:
                        relationship = rel_data['relationship']
                        edge = VisualizationEdge(
                            id=f"{source_id}_to_{target_id}",
                            source=source_id,
                            target=target_id,
                            relationship_type=relationship.type,
                            weight=relationship.get('weight', 1.0)
                        )
                        edges.append(edge)
            
            # Limit nodes and edges if necessary
            if len(nodes) > self.config.max_nodes:
                nodes = nodes[:self.config.max_nodes]
                node_ids = {node.id for node in nodes}
                edges = [edge for edge in edges if edge.source in node_ids and edge.target in node_ids]
            
            if len(edges) > self.config.max_edges:
                edges = edges[:self.config.max_edges]
            
            metadata = {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'domains': list(set(node.domain for node in nodes if node.domain)),
                'node_types': list(set(node.node_type.value for node in nodes)),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            return VisualizationData(
                nodes=nodes,
                edges=edges,
                metadata=metadata,
                layout_type="hierarchical",
                filters={'domain': domain_filter, 'max_depth': max_depth}
            )
    
    def _calculate_temporal_level(self, date_str: Optional[str]) -> int:
        """Calculate level based on temporal distance (recent = higher level)."""
        if not date_str:
            return 5  # Default level for undated items
        
        try:
            from datetime import datetime
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            current_year = datetime.now().year
            doc_year = date.year
            
            # More recent documents get higher levels (closer to leaves)
            years_ago = current_year - doc_year
            
            if years_ago < 50:
                return 6  # Very recent
            elif years_ago < 200:
                return 5  # Recent
            elif years_ago < 500:
                return 4  # Historical
            elif years_ago < 2000:
                return 3  # Ancient
            else:
                return 2  # Very ancient
                
        except (ValueError, TypeError):
            return 5  # Default level
    
    def _apply_yggdrasil_layout(self, viz_data: VisualizationData) -> None:
        """Apply Yggdrasil tree layout to nodes."""
        
        # Group nodes by level and domain
        level_groups = {}
        for node in viz_data.nodes:
            level = node.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes in tree structure
        base_y = 0
        level_height = self.config.tree_spacing["level_distance"]
        
        for level in sorted(level_groups.keys()):
            nodes_at_level = level_groups[level]
            
            if level == 0:  # Root node
                for node in nodes_at_level:
                    node.x = self.config.tree_spacing["root_x"]
                    node.y = self.config.tree_spacing["root_y"]
            else:
                # Distribute nodes horizontally
                if len(nodes_at_level) == 1:
                    nodes_at_level[0].x = 0
                else:
                    total_width = (len(nodes_at_level) - 1) * self.config.tree_spacing["node_distance"]
                    start_x = -total_width / 2
                    
                    for i, node in enumerate(nodes_at_level):
                        node.x = start_x + i * self.config.tree_spacing["node_distance"]
                
                # Set Y position based on level
                for node in nodes_at_level:
                    node.y = base_y - level * level_height
        
        # Apply colors and sizes
        for node in viz_data.nodes:
            node.color = self.config.node_colors.get(node.node_type, "#999999")
            
            # Size based on importance (could be PageRank, degree, etc.)
            base_size = self.config.node_sizes["min_size"]
            if node.metadata and node.metadata.get('word_count'):
                # Size based on word count
                word_count = node.metadata['word_count']
                size_multiplier = min(word_count / 1000, 5)  # Cap at 5x
                node.size = base_size + size_multiplier * self.config.node_sizes["size_factor"]
            else:
                node.size = base_size
            
            # Ensure size is within bounds
            node.size = max(
                self.config.node_sizes["min_size"],
                min(node.size or base_size, self.config.node_sizes["max_size"])
            )
        
        # Apply edge colors
        for edge in viz_data.edges:
            edge.color = self.config.edge_colors.get(
                edge.relationship_type,
                self.config.edge_colors["default"]
            )
    
    async def _generate_interactive_chart(
        self,
        viz_data: VisualizationData,
        title: str,
        chart_type: VisualizationType,
        export_format: str
    ) -> str:
        """Generate interactive chart using vis.js."""
        
        # Convert nodes to vis.js format
        vis_nodes = []
        for node in viz_data.nodes:
            vis_node = {
                'id': node.id,
                'label': node.label,
                'title': node.title,
                'node_type': node.node_type.value,
                'domain': node.domain,
                'date': node.date,
                'color': node.color,
                'size': node.size,
                'x': node.x,
                'y': node.y,
                'metadata': node.metadata
            }
            vis_nodes.append(vis_node)
        
        # Convert edges to vis.js format
        vis_edges = []
        for edge in viz_data.edges:
            vis_edge = {
                'id': edge.id,
                'from': edge.source,
                'to': edge.target,
                'label': edge.relationship_type,
                'color': edge.color,
                'width': edge.weight,
                'arrows': 'to'
            }
            vis_edges.append(vis_edge)
        
        # Prepare template data
        template_data = {
            'title': title,
            'nodes_json': json.dumps(vis_nodes),
            'edges_json': json.dumps(vis_edges),
            'domains': viz_data.metadata.get('domains', []),
            'node_types': viz_data.metadata.get('node_types', []),
            'node_colors': {nt.value: color for nt, color in self.config.node_colors.items()},
            'layout_type': viz_data.layout_type,
            'physics_enabled': self.config.enable_physics,
            'enable_zoom': self.config.enable_zoom,
            'enable_pan': self.config.enable_pan,
            'node_spacing': self.config.tree_spacing["node_distance"],
            'level_separation': self.config.tree_spacing["level_distance"],
            'tree_spacing': self.config.tree_spacing["node_distance"],
            'min_node_size': self.config.node_sizes["min_size"],
            'max_node_size': self.config.node_sizes["max_size"]
        }
        
        # Render template
        html_content = self.visjs_template.render(**template_data)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{chart_type.value}_{timestamp}.{export_format}"
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    async def generate_network_graph(
        self,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        title: str = "Knowledge Network"
    ) -> str:
        """Generate a general network graph visualization."""
        
        # Get network data
        viz_data = await self._get_network_data(node_types, relationship_types)
        
        # Apply force-directed layout
        self._apply_force_layout(viz_data)
        
        # Generate visualization
        output_file = await self._generate_interactive_chart(
            viz_data,
            title,
            VisualizationType.NETWORK_GRAPH,
            "html"
        )
        
        self.logger.info(f"Generated network graph: {output_file}")
        return output_file
    
    async def _get_network_data(
        self,
        node_types: Optional[List[str]],
        relationship_types: Optional[List[str]]
    ) -> VisualizationData:
        """Get data for network graph visualization."""
        
        async with self.neo4j_driver.session() as session:
            # Build dynamic query based on filters
            node_filter = ""
            if node_types:
                node_labels = " OR ".join([f"n:{nt}" for nt in node_types])
                node_filter = f"WHERE {node_labels}"
            
            rel_filter = ""
            if relationship_types:
                rel_types = " OR ".join([f"type(r) = '{rt}'" for rt in relationship_types])
                rel_filter = f"AND ({rel_types})"
            
            query = f"""
            MATCH (n) {node_filter}
            OPTIONAL MATCH (n)-[r]-(m)
            {rel_filter if rel_filter else ""}
            RETURN 
                collect(DISTINCT n) as nodes,
                collect(DISTINCT {{source: n, target: m, relationship: r}}) as relationships
            LIMIT {self.config.max_nodes}
            """
            
            result = await session.run(query)
            record = await result.single()
            
            nodes = []
            edges = []
            node_ids = set()
            
            # Process nodes
            for node_data in record['nodes'] or []:
                node_id = str(node_data.id)
                if node_id not in node_ids:
                    # Determine node type
                    node_type = NodeType.DOCUMENT
                    for label in node_data.labels:
                        if label.lower() in [nt.value for nt in NodeType]:
                            node_type = NodeType(label.lower())
                            break
                    
                    viz_node = VisualizationNode(
                        id=node_id,
                        label=node_data.get('title', f'Node {node_id}'),
                        title=node_data.get('title', 'Unknown'),
                        node_type=node_type,
                        domain=node_data.get('domain'),
                        date=node_data.get('date'),
                        level=0  # Will be calculated in layout
                    )
                    nodes.append(viz_node)
                    node_ids.add(node_id)
            
            # Process relationships
            for rel_data in record['relationships'] or []:
                if (rel_data and rel_data['source'] and rel_data['target'] and 
                    rel_data['relationship']):
                    
                    source_id = str(rel_data['source'].id)
                    target_id = str(rel_data['target'].id)
                    
                    if source_id in node_ids and target_id in node_ids:
                        edge = VisualizationEdge(
                            id=f"{source_id}_to_{target_id}",
                            source=source_id,
                            target=target_id,
                            relationship_type=rel_data['relationship'].type,
                            weight=rel_data['relationship'].get('weight', 1.0)
                        )
                        edges.append(edge)
            
            metadata = {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': list(set(node.node_type.value for node in nodes)),
                'relationship_types': list(set(edge.relationship_type for edge in edges)),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            return VisualizationData(
                nodes=nodes,
                edges=edges,
                metadata=metadata,
                layout_type="force",
                filters={'node_types': node_types, 'relationship_types': relationship_types}
            )
    
    def _apply_force_layout(self, viz_data: VisualizationData) -> None:
        """Apply force-directed layout using NetworkX."""
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node in viz_data.nodes:
                G.add_node(node.id)
            
            # Add edges
            for edge in viz_data.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Calculate layout
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, k=200, iterations=50)
                
                # Apply positions
                for node in viz_data.nodes:
                    if node.id in pos:
                        node.x = pos[node.id][0] * 1000  # Scale up
                        node.y = pos[node.id][1] * 1000
            
            # Apply colors and sizes
            for node in viz_data.nodes:
                node.color = self.config.node_colors.get(node.node_type, "#999999")
                node.size = self.config.node_sizes["min_size"]
            
            for edge in viz_data.edges:
                edge.color = self.config.edge_colors.get(
                    edge.relationship_type,
                    self.config.edge_colors["default"]
                )
        
        except Exception as e:
            self.logger.warning(f"Error in force layout: {e}")
            # Fallback to simple layout
            for i, node in enumerate(viz_data.nodes):
                node.x = (i % 10) * 100
                node.y = (i // 10) * 100
                node.color = self.config.node_colors.get(node.node_type, "#999999")
                node.size = self.config.node_sizes["min_size"]


# CLI Interface
async def main():
    """Main CLI interface for chart generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Chart Generator")
    parser.add_argument("--chart-type", choices=[
        "yggdrasil", "network", "timeline", "domain"
    ], default="yggdrasil", help="Type of chart to generate")
    parser.add_argument("--title", help="Chart title")
    parser.add_argument("--domain", help="Domain filter")
    parser.add_argument("--format", choices=["html", "svg", "png"], 
                       default="html", help="Export format")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    config = VisualizationConfig(args.config) if args.config else VisualizationConfig()
    generator = ChartGenerator(config)
    
    await generator.initialize()
    
    try:
        if args.chart_type == "yggdrasil":
            title = args.title or "Yggdrasil Knowledge Tree"
            output_file = await generator.generate_yggdrasil_chart(
                title=title,
                domain_filter=args.domain,
                export_format=args.format
            )
            print(f"Generated Yggdrasil chart: {output_file}")
        
        elif args.chart_type == "network":
            title = args.title or "Knowledge Network"
            output_file = await generator.generate_network_graph(title=title)
            print(f"Generated network graph: {output_file}")
        
        else:
            print(f"Chart type '{args.chart_type}' not yet implemented")
    
    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main())
