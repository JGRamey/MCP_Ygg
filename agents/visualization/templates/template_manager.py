"""
Template management for visualization HTML generation.
"""

import json
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

from ..core.config import VisualizationConfig
from ..core.models import VisualizationData


class TemplateManager:
    """Manages HTML templates for visualization generation."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.template_dir = Path(config.template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Initialize templates
        self.visjs_template = self._create_visjs_template()
        self._save_template_to_file()

    def _create_visjs_template(self) -> Template:
        """Create the vis.js template for interactive graphs."""
        self.template_content = """<!DOCTYPE html>
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
            alert('SVG export functionality would be implemented with additional libraries');
        }
        
        function exportPNG() {
            alert('PNG export functionality would be implemented with additional libraries');
        }
        
        // Add filter event listeners
        document.getElementById('domainFilter').addEventListener('change', applyFilters);
        document.getElementById('nodeTypeFilter').addEventListener('change', applyFilters);
    </script>
</body>
</html>"""

        return Template(self.template_content)

    def _save_template_to_file(self) -> None:
        """Save template to file for external use."""
        template_file = self.template_dir / "visjs_template.html"
        with open(template_file, "w") as f:
            f.write(self.template_content)

    def render_template(
        self, viz_data: VisualizationData, title: str, layout_params: Dict[str, Any]
    ) -> str:
        """Render the visualization template with data."""

        # Convert nodes to vis.js format
        vis_nodes = []
        for node in viz_data.nodes:
            vis_node = {
                "id": node.id,
                "label": node.label,
                "title": node.title,
                "node_type": node.node_type.value,
                "domain": node.domain,
                "date": node.date,
                "color": node.color,
                "size": node.size,
                "x": node.x,
                "y": node.y,
                "metadata": node.metadata,
            }
            vis_nodes.append(vis_node)

        # Convert edges to vis.js format
        vis_edges = []
        for edge in viz_data.edges:
            vis_edge = {
                "id": edge.id,
                "from": edge.source,
                "to": edge.target,
                "label": edge.relationship_type,
                "color": edge.color,
                "width": edge.weight,
                "arrows": "to",
            }
            vis_edges.append(vis_edge)

        # Prepare template data
        template_data = {
            "title": title,
            "nodes_json": json.dumps(vis_nodes),
            "edges_json": json.dumps(vis_edges),
            "domains": viz_data.metadata.get("domains", []),
            "node_types": viz_data.metadata.get("node_types", []),
            "node_colors": {
                nt.value: color for nt, color in self.config.node_colors.items()
            },
            "layout_type": viz_data.layout_type,
            "physics_enabled": self.config.enable_physics,
            "enable_zoom": self.config.enable_zoom,
            "enable_pan": self.config.enable_pan,
            "min_node_size": self.config.node_sizes["min_size"],
            "max_node_size": self.config.node_sizes["max_size"],
        }

        # Add layout-specific parameters
        template_data.update(layout_params)

        return self.visjs_template.render(**template_data)
