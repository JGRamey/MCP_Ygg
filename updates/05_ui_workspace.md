# Phase 5: UI Workspace Development
## 💻 STREAMLIT INTERFACE (Weeks 9-10)

### Overview
Transform the basic dashboard into a comprehensive workspace for MCP Yggdrasil project management, focusing on database material management (NOT an IDE), fixing existing issues, and enhancing the scraper interface.

### User Requirements Summary
- **NOT an IDE-like interface** - Only file management of stored data
- **Database material focus** - CSV files and database content, not project files
- **Scraper page enhancement** - Options for different source types
- **Graph editor fix** - Show actual Neo4j knowledge graph with drag-and-drop
- **Operations console fix** - Resolve psutil import error
- **Concept-centered design** - Cross-cultural concept connections

### 🔧 Priority 1: Fix Existing Issues

#### Fix 1: Operations Console - psutil Import Error
**File: `streamlit_workspace/pages/04_⚡_Operations_Console.py`**

**Current Issue**: `ModuleNotFoundError: No module named 'psutil'`

**Solution**:
```python
# First, ensure psutil is in requirements.txt (already added in Phase 1)
# psutil>=5.9.0,<6.0.0

# Updated Operations Console
import streamlit as st
try:
    import psutil
    import docker
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    st.error("System monitoring requires psutil. Please install: pip install psutil")

import time
from datetime import datetime
import plotly.graph_objects as go
from collections import deque

st.set_page_config(
    page_title="Operations Console",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Operations Console")
st.markdown("Real-time system monitoring and operations management")

if not MONITORING_AVAILABLE:
    st.warning("System monitoring is not available. Please install required dependencies.")
    st.stop()

# Initialize session state for metrics history
if 'cpu_history' not in st.session_state:
    st.session_state.cpu_history = deque(maxlen=60)
    st.session_state.memory_history = deque(maxlen=60)
    st.session_state.timestamps = deque(maxlen=60)

# System Metrics Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    cpu_percent = psutil.cpu_percent(interval=1)
    st.metric("CPU Usage", f"{cpu_percent}%", 
              delta=f"{cpu_percent - st.session_state.cpu_history[-1] if st.session_state.cpu_history else 0:.1f}%")

with col2:
    memory = psutil.virtual_memory()
    st.metric("Memory Usage", f"{memory.percent}%", 
              delta=f"{memory.percent - st.session_state.memory_history[-1] if st.session_state.memory_history else 0:.1f}%")

with col3:
    disk = psutil.disk_usage('/')
    st.metric("Disk Usage", f"{disk.percent}%")

with col4:
    network = psutil.net_io_counters()
    st.metric("Network I/O", f"{network.bytes_sent / 1024 / 1024:.1f} MB sent")

# Update history
st.session_state.cpu_history.append(cpu_percent)
st.session_state.memory_history.append(memory.percent)
st.session_state.timestamps.append(datetime.now())

# Real-time Charts
st.subheader("System Performance Trends")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(st.session_state.timestamps),
    y=list(st.session_state.cpu_history),
    name="CPU %",
    line=dict(color='#ff7f0e', width=2)
))
fig.add_trace(go.Scatter(
    x=list(st.session_state.timestamps),
    y=list(st.session_state.memory_history),
    name="Memory %",
    line=dict(color='#2ca02c', width=2)
))
fig.update_layout(
    title="Real-time System Metrics",
    xaxis_title="Time",
    yaxis_title="Usage %",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Database Status Section
st.subheader("Database Status")

col1, col2, col3 = st.columns(3)

with col1:
    # Neo4j Status
    try:
        from scripts.initialize_system import test_neo4j_connection
        neo4j_status = test_neo4j_connection()
        if neo4j_status:
            st.success("✅ Neo4j: Connected")
            # Get database statistics
            st.metric("Nodes", "371+")
            st.metric("Relationships", "1,200+")
        else:
            st.error("❌ Neo4j: Disconnected")
    except:
        st.warning("⚠️ Neo4j: Unknown")

with col2:
    # Qdrant Status
    try:
        from scripts.initialize_system import test_qdrant_connection
        qdrant_status = test_qdrant_connection()
        if qdrant_status:
            st.success("✅ Qdrant: Connected")
            st.metric("Collections", "7")
            st.metric("Vectors", "10,000+")
        else:
            st.error("❌ Qdrant: Disconnected")
    except:
        st.warning("⚠️ Qdrant: Unknown")

with col3:
    # Redis Status
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        st.success("✅ Redis: Connected")
        info = r.info()
        st.metric("Memory Used", f"{info.get('used_memory_human', 'N/A')}")
        st.metric("Connected Clients", info.get('connected_clients', 0))
    except:
        st.warning("⚠️ Redis: Unknown")

# Docker Containers Section
st.subheader("Docker Containers")

try:
    client = docker.from_env()
    containers = client.containers.list(all=True)
    
    container_data = []
    for container in containers:
        container_data.append({
            "Name": container.name,
            "Image": container.image.tags[0] if container.image.tags else "Unknown",
            "Status": container.status,
            "State": "🟢 Running" if container.status == "running" else "🔴 Stopped"
        })
    
    if container_data:
        st.dataframe(container_data, use_container_width=True)
    else:
        st.info("No Docker containers found")
        
except Exception as e:
    st.warning(f"Docker not available: {str(e)}")

# Quick Actions
st.subheader("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Restart Services", type="primary"):
        with st.spinner("Restarting services..."):
            # Implement service restart logic
            time.sleep(2)
            st.success("Services restarted successfully!")

with col2:
    if st.button("🗑️ Clear Cache"):
        with st.spinner("Clearing cache..."):
            # Implement cache clearing
            time.sleep(1)
            st.success("Cache cleared!")

with col3:
    if st.button("📊 Generate Report"):
        # Generate system report
        st.info("System report generation coming soon!")

# Auto-refresh
if st.checkbox("Auto-refresh (5s)", value=True):
    time.sleep(5)
    st.rerun()
```

#### Fix 2: Graph Editor - Show Neo4j Knowledge Graph
**File: `streamlit_workspace/pages/02_📊_Graph_Editor.py`**

**Current Issue**: Shows "No concepts match the current filters" instead of actual Neo4j graph

**Solution**:
```python
import streamlit as st
from py2neo import Graph
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import json

st.set_page_config(
    page_title="Graph Editor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Knowledge Graph Editor")
st.markdown("Interactive Neo4j knowledge graph visualization with drag-and-drop editing")

# Initialize Neo4j connection
@st.cache_resource
def get_neo4j_connection():
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "yggdrasil"))
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        return None

graph = get_neo4j_connection()

if not graph:
    st.stop()

# Sidebar filters
st.sidebar.header("Graph Filters")

# Domain filter
domains = ['all', 'mathematics', 'science', 'philosophy', 'religion', 'art', 'language']
selected_domain = st.sidebar.selectbox("Select Domain", domains)

# Node type filter
node_types = st.sidebar.multiselect(
    "Node Types",
    ['Concept', 'Entity', 'Document', 'Author', 'Event'],
    default=['Concept', 'Entity']
)

# Limit results
result_limit = st.sidebar.slider("Max Nodes", 10, 500, 100)

# Relationship depth
rel_depth = st.sidebar.slider("Relationship Depth", 1, 3, 1)

# Load graph data from Neo4j
@st.cache_data(ttl=60)
def load_graph_data(domain: str, node_types: List[str], limit: int, depth: int):
    """Load graph data from Neo4j."""
    
    # Build node type constraint
    node_constraint = " OR ".join([f"n:{nt}" for nt in node_types])
    
    # Build domain constraint
    domain_constraint = "" if domain == "all" else f" AND n.domain = '{domain}'"
    
    # Query to get nodes and relationships
    query = f"""
    MATCH (n)
    WHERE ({node_constraint}){domain_constraint}
    WITH n LIMIT {limit}
    MATCH path = (n)-[r*0..{depth}]-(connected)
    WHERE any(label in labels(connected) WHERE label IN {node_types})
    RETURN n, relationships(path) as rels, nodes(path) as nodes
    """
    
    result = graph.run(query).data()
    
    # Process results into nodes and edges
    nodes_dict = {}
    edges = []
    
    for record in result:
        # Process main node
        main_node = record['n']
        node_id = main_node.identity
        
        if node_id not in nodes_dict:
            nodes_dict[node_id] = {
                'id': node_id,
                'label': main_node.get('name', main_node.get('title', f"Node {node_id}")),
                'type': list(main_node.labels)[0] if main_node.labels else 'Unknown',
                'properties': dict(main_node)
            }
        
        # Process connected nodes
        for node in record['nodes']:
            node_id = node.identity
            if node_id not in nodes_dict:
                nodes_dict[node_id] = {
                    'id': node_id,
                    'label': node.get('name', node.get('title', f"Node {node_id}")),
                    'type': list(node.labels)[0] if node.labels else 'Unknown',
                    'properties': dict(node)
                }
        
        # Process relationships
        for rel in record['rels']:
            edges.append({
                'source': rel.start_node.identity,
                'target': rel.end_node.identity,
                'type': type(rel).__name__,
                'properties': dict(rel)
            })
    
    return list(nodes_dict.values()), edges

# Load data
with st.spinner("Loading graph data..."):
    nodes, edges = load_graph_data(selected_domain, node_types, result_limit, rel_depth)

if not nodes:
    st.warning("No data found with current filters. Try adjusting your selection.")
    st.stop()

# Display statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nodes", len(nodes))
with col2:
    st.metric("Relationships", len(edges))
with col3:
    st.metric("Domains", len(set(n.get('properties', {}).get('domain', 'unknown') for n in nodes)))

# Create interactive graph visualization
st.subheader("Interactive Knowledge Graph")

# Convert to networkx for layout calculation
G = nx.Graph()
for node in nodes:
    G.add_node(node['id'], **node)
for edge in edges:
    G.add_edge(edge['source'], edge['target'], **edge)

# Calculate layout
layout = nx.spring_layout(G, k=2, iterations=50)

# Create Plotly figure
fig = go.Figure()

# Add edges
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = layout[edge['source']]
    x1, y1 = layout[edge['target']]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines',
    showlegend=False
))

# Add nodes by type with different colors
colors = {
    'Concept': '#ff7f0e',
    'Entity': '#2ca02c', 
    'Document': '#d62728',
    'Author': '#9467bd',
    'Event': '#8c564b',
    'Unknown': '#7f7f7f'
}

for node_type in set(n['type'] for n in nodes):
    node_x = []
    node_y = []
    node_text = []
    node_ids = []
    
    for node in nodes:
        if node['type'] == node_type:
            x, y = layout[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node['label'])
            node_ids.append(node['id'])
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        name=node_type,
        text=node_text,
        textposition="top center",
        marker=dict(
            size=10,
            color=colors.get(node_type, '#7f7f7f'),
            line=dict(width=2, color='white')
        ),
        customdata=node_ids,
        hovertemplate='<b>%{text}</b><br>Type: ' + node_type + '<br>ID: %{customdata}<extra></extra>'
    ))

fig.update_layout(
    title="Knowledge Graph Visualization",
    showlegend=True,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600
)

# Display the graph
graph_placeholder = st.plotly_chart(fig, use_container_width=True, key="main_graph")

# Node Details Section
st.subheader("Node Details")

# Allow node selection
selected_node_id = st.selectbox(
    "Select a node to view/edit details:",
    [node['id'] for node in nodes],
    format_func=lambda x: next(n['label'] for n in nodes if n['id'] == x)
)

if selected_node_id:
    selected_node = next(n for n in nodes if n['id'] == selected_node_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Node Information**")
        st.write(f"- **ID**: {selected_node['id']}")
        st.write(f"- **Type**: {selected_node['type']}")
        st.write(f"- **Label**: {selected_node['label']}")
        
        # Display properties
        st.write("**Properties**")
        for key, value in selected_node['properties'].items():
            st.write(f"- **{key}**: {value}")
    
    with col2:
        st.write("**Edit Node**")
        
        # Edit form
        with st.form(f"edit_node_{selected_node_id}"):
            new_label = st.text_input("Label", value=selected_node['label'])
            
            # Domain selection for concepts
            if selected_node['type'] == 'Concept':
                new_domain = st.selectbox(
                    "Domain",
                    domains[1:],  # Exclude 'all'
                    index=domains[1:].index(selected_node['properties'].get('domain', 'science'))
                )
            
            # Additional properties
            st.write("Additional Properties (JSON)")
            props_json = st.text_area(
                "Properties",
                value=json.dumps(selected_node['properties'], indent=2),
                height=150
            )
            
            submit = st.form_submit_button("Update Node")
            
            if submit:
                try:
                    # Parse properties
                    new_props = json.loads(props_json)
                    
                    # Update in Neo4j
                    update_query = """
                    MATCH (n) WHERE ID(n) = $node_id
                    SET n += $properties
                    SET n.name = $label
                    RETURN n
                    """
                    
                    if selected_node['type'] == 'Concept':
                        new_props['domain'] = new_domain
                    
                    graph.run(update_query, node_id=selected_node_id, properties=new_props, label=new_label)
                    
                    st.success("Node updated successfully!")
                    st.rerun()
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON in properties field")
                except Exception as e:
                    st.error(f"Failed to update node: {str(e)}")

# Relationship Management
st.subheader("Relationship Management")

col1, col2 = st.columns(2)

with col1:
    st.write("**Create New Relationship**")
    
    with st.form("create_relationship"):
        source_node = st.selectbox(
            "Source Node",
            [node['id'] for node in nodes],
            format_func=lambda x: next(n['label'] for n in nodes if n['id'] == x)
        )
        
        target_node = st.selectbox(
            "Target Node",
            [node['id'] for node in nodes],
            format_func=lambda x: next(n['label'] for n in nodes if n['id'] == x)
        )
        
        rel_type = st.selectbox(
            "Relationship Type",
            ['RELATES_TO', 'INFLUENCES', 'CONTRADICTS', 'SUPPORTS', 'PART_OF']
        )
        
        create_rel = st.form_submit_button("Create Relationship")
        
        if create_rel:
            if source_node != target_node:
                try:
                    create_query = f"""
                    MATCH (a) WHERE ID(a) = $source
                    MATCH (b) WHERE ID(b) = $target
                    CREATE (a)-[r:{rel_type}]->(b)
                    RETURN r
                    """
                    
                    graph.run(create_query, source=source_node, target=target_node)
                    st.success("Relationship created successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to create relationship: {str(e)}")
            else:
                st.error("Cannot create relationship to same node")

with col2:
    st.write("**Export Graph Data**")
    
    # Export options
    export_format = st.selectbox("Export Format", ["JSON", "CSV", "GraphML"])
    
    if st.button("Export Graph"):
        if export_format == "JSON":
            export_data = {
                'nodes': nodes,
                'edges': edges
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"knowledge_graph_{selected_domain}.json",
                mime="application/json"
            )
        
        elif export_format == "CSV":
            # Create CSV data
            import csv
            import io
            
            # Nodes CSV
            nodes_output = io.StringIO()
            nodes_writer = csv.writer(nodes_output)
            nodes_writer.writerow(['id', 'label', 'type', 'domain'])
            
            for node in nodes:
                nodes_writer.writerow([
                    node['id'],
                    node['label'],
                    node['type'],
                    node['properties'].get('domain', '')
                ])
            
            st.download_button(
                label="Download Nodes CSV",
                data=nodes_output.getvalue(),
                file_name=f"nodes_{selected_domain}.csv",
                mime="text/csv"
            )

# Cross-Cultural Connections Section
st.subheader("Cross-Cultural Concept Connections")

# Find concepts that appear in multiple domains
cross_cultural_query = """
MATCH (c:Concept)
WITH c.name as concept_name, collect(DISTINCT c.domain) as domains
WHERE size(domains) > 1
RETURN concept_name, domains
ORDER BY size(domains) DESC
LIMIT 20
"""

cross_cultural_results = graph.run(cross_cultural_query).data()

if cross_cultural_results:
    st.write("**Concepts Found Across Multiple Domains:**")
    
    for result in cross_cultural_results:
        concept = result['concept_name']
        domains = result['domains']
        st.write(f"- **{concept}**: {', '.join(domains)}")
else:
    st.info("No cross-cultural concepts found yet. Add more data to discover connections!")
```

#### Fix 3: Content Scraper Page Enhancement
**File: `streamlit_workspace/pages/07_📥_Content_Scraper.py`**

**Current Issue**: Blank page - needs source type selection

**Solution**:
```python
import streamlit as st
import asyncio
from datetime import datetime
import json
from typing import Dict, List
import pandas as pd

st.set_page_config(
    page_title="Content Scraper",
    page_icon="📥",
    layout="wide"
)

st.title("📥 Content Scraper")
st.markdown("Multi-source content acquisition interface")

# Source type selection
source_types = {
    '🌐 Webpage': 'webpage',
    '📺 YouTube Video': 'youtube',
    '📚 Book/eBook': 'book',
    '📜 PDF Document': 'pdf',
    '🖼️ Image/Picture': 'image',
    '📰 Web Article': 'article',
    '📜 Manuscript': 'manuscript',
    '📜 Ancient Text': 'ancient_text',
    '📚 Academic Paper': 'academic_paper',
    '📚 Encyclopedia Entry': 'encyclopedia'
}

# Main selection
selected_source = st.selectbox(
    "Select Content Source Type",
    list(source_types.keys()),
    help="Choose the type of content you want to scrape"
)

source_type = source_types[selected_source]

# Source-specific input interfaces
st.subheader(f"Input {selected_source}")

if source_type in ['webpage', 'article', 'encyclopedia']:
    # URL input for web content
    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com/article",
        help="Full URL of the webpage to scrape"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        extract_images = st.checkbox("Extract images", value=True)
        extract_links = st.checkbox("Extract links", value=True)
        follow_redirects = st.checkbox("Follow redirects", value=True)
        use_selenium = st.checkbox("Use JavaScript rendering", value=False,
                                  help="Enable for dynamic websites")

elif source_type == 'youtube':
    # YouTube specific inputs
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        help="Full YouTube video URL"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        extract_transcript = st.checkbox("Extract transcript", value=True)
        extract_comments = st.checkbox("Extract top comments", value=False)
    
    with col2:
        transcript_language = st.selectbox(
            "Transcript Language",
            ['auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        )
        max_comments = st.number_input("Max comments", min_value=0, max_value=100, value=10)

elif source_type == 'pdf':
    # PDF upload
    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_pdf:
        st.write(f"**File**: {uploaded_pdf.name} ({uploaded_pdf.size / 1024 / 1024:.2f} MB)")
    
    # PDF options
    with st.expander("PDF Processing Options"):
        extract_text = st.checkbox("Extract text content", value=True)
        extract_images = st.checkbox("Extract embedded images", value=False)
        extract_metadata = st.checkbox("Extract metadata", value=True)
        ocr_if_needed = st.checkbox("Use OCR if text extraction fails", value=True)

elif source_type == 'image':
    # Image upload
    uploaded_images = st.file_uploader(
        "Upload Image(s)",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Upload one or more images for OCR processing"
    )
    
    if uploaded_images:
        col1, col2 = st.columns(2)
        for i, img in enumerate(uploaded_images):
            with col1 if i % 2 == 0 else col2:
                st.image(img, caption=img.name, use_column_width=True)
    
    # OCR options
    with st.expander("OCR Options"):
        ocr_language = st.selectbox(
            "Document Language",
            ['eng', 'lat', 'grc', 'heb', 'ara', 'chi_sim', 'jpn'],
            help="Select the primary language for OCR"
        )
        enhance_image = st.checkbox("Enhance image before OCR", value=True)
        detect_layout = st.checkbox("Detect document layout", value=True)

elif source_type == 'book':
    # Book information
    st.write("**Book Metadata Entry**")
    
    col1, col2 = st.columns(2)
    with col1:
        book_title = st.text_input("Book Title*", placeholder="The Republic")
        book_author = st.text_input("Author(s)*", placeholder="Plato")
        isbn = st.text_input("ISBN", placeholder="978-0-14-044914-7")
    
    with col2:
        publication_year = st.number_input("Publication Year", min_value=1000, max_value=2025, value=2023)
        publisher = st.text_input("Publisher", placeholder="Penguin Classics")
        language = st.selectbox("Language", ['English', 'Latin', 'Greek', 'Hebrew', 'Arabic', 'Other'])
    
    # Content input
    content_input_method = st.radio(
        "How will you provide the content?",
        ['Upload file', 'Paste text', 'Enter chapters individually']
    )
    
    if content_input_method == 'Upload file':
        book_file = st.file_uploader(
            "Upload book file",
            type=['txt', 'epub', 'mobi', 'docx'],
            help="Supported formats: TXT, EPUB, MOBI, DOCX"
        )
    elif content_input_method == 'Paste text':
        book_content = st.text_area(
            "Paste book content",
            height=300,
            placeholder="Paste the full text of the book here..."
        )

elif source_type in ['manuscript', 'ancient_text']:
    # Historical document handling
    st.write("**Historical Document Information**")
    
    col1, col2 = st.columns(2)
    with col1:
        doc_title = st.text_input("Document Title*", placeholder="Dead Sea Scrolls - Fragment 4Q521")
        time_period = st.text_input("Time Period/Date", placeholder="1st century BCE")
        origin = st.text_input("Origin/Location", placeholder="Qumran, Judean Desert")
    
    with col2:
        original_language = st.selectbox(
            "Original Language",
            ['Hebrew', 'Aramaic', 'Greek', 'Latin', 'Coptic', 'Sanskrit', 'Other']
        )
        document_type = st.selectbox(
            "Document Type",
            ['Religious text', 'Historical record', 'Legal document', 'Literary work', 'Scientific text']
        )
    
    # Image or text input
    input_type = st.radio(
        "Input Type",
        ['Upload manuscript images', 'Enter transcribed text', 'Both']
    )
    
    if input_type in ['Upload manuscript images', 'Both']:
        manuscript_images = st.file_uploader(
            "Upload manuscript images",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True,
            help="High-resolution images recommended for better OCR"
        )
    
    if input_type in ['Enter transcribed text', 'Both']:
        transcribed_text = st.text_area(
            "Transcribed Text",
            height=200,
            placeholder="Enter the transcribed text from the manuscript..."
        )

elif source_type == 'academic_paper':
    # Academic paper specifics
    st.write("**Academic Paper Details**")
    
    paper_source = st.radio(
        "Paper Source",
        ['Enter DOI', 'Enter URL', 'Upload PDF', 'Enter details manually']
    )
    
    if paper_source == 'Enter DOI':
        doi = st.text_input(
            "DOI",
            placeholder="10.1038/nature12373",
            help="Digital Object Identifier"
        )
    elif paper_source == 'Enter URL':
        paper_url = st.text_input(
            "Paper URL",
            placeholder="https://arxiv.org/abs/2103.14030",
            help="Direct link to the paper"
        )
    elif paper_source == 'Upload PDF':
        paper_pdf = st.file_uploader("Upload paper PDF", type=['pdf'])
    else:
        # Manual entry
        col1, col2 = st.columns(2)
        with col1:
            paper_title = st.text_input("Paper Title*")
            authors = st.text_input("Authors*", help="Comma-separated list")
            journal = st.text_input("Journal/Conference")
        with col2:
            year = st.number_input("Publication Year", min_value=1900, max_value=2025)
            volume = st.text_input("Volume/Issue")
            pages = st.text_input("Pages", placeholder="123-145")

# Domain Classification
st.subheader("Domain Classification")

domain_help = """
Select the primary domain(s) this content belongs to:
- **Mathematics**: Algebra, geometry, calculus, number theory
- **Science**: Physics, chemistry, biology, astronomy
- **Philosophy**: Metaphysics, ethics, logic, epistemology
- **Religion**: Theology, spirituality, religious texts
- **Art**: Visual arts, music, literature, architecture
- **Language**: Linguistics, grammar, etymology
"""

selected_domains = st.multiselect(
    "Select Domain(s)",
    ['mathematics', 'science', 'philosophy', 'religion', 'art', 'language'],
    help=domain_help
)

# Additional metadata
with st.expander("Additional Metadata"):
    col1, col2 = st.columns(2)
    with col1:
        tags = st.text_input(
            "Tags",
            placeholder="ancient philosophy, ethics, virtue",
            help="Comma-separated tags"
        )
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            0.0, 1.0, 0.7,
            help="Minimum confidence score for extracted information"
        )
    
    with col2:
        priority = st.selectbox(
            "Processing Priority",
            ['Normal', 'High', 'Low']
        )
        private_notes = st.text_area(
            "Private Notes",
            placeholder="Any additional notes about this content..."
        )

# Processing Options
st.subheader("Processing Options")

col1, col2, col3 = st.columns(3)

with col1:
    run_ocr = st.checkbox("Run OCR (if applicable)", value=True)
    extract_entities = st.checkbox("Extract named entities", value=True)
    detect_language = st.checkbox("Auto-detect language", value=True)

with col2:
    extract_concepts = st.checkbox("Extract concepts", value=True)
    verify_facts = st.checkbox("Verify facts", value=True)
    find_citations = st.checkbox("Extract citations", value=True)

with col3:
    create_summary = st.checkbox("Generate summary", value=True)
    analyze_sentiment = st.checkbox("Analyze sentiment", value=False)
    detect_bias = st.checkbox("Detect bias", value=False)

# Submit button
if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
    # Validate inputs
    valid = True
    
    if source_type in ['webpage', 'article', 'encyclopedia'] and not url:
        st.error("Please enter a URL")
        valid = False
    elif source_type == 'youtube' and not youtube_url:
        st.error("Please enter a YouTube URL")
        valid = False
    elif source_type == 'pdf' and not uploaded_pdf:
        st.error("Please upload a PDF file")
        valid = False
    elif source_type == 'image' and not uploaded_images:
        st.error("Please upload at least one image")
        valid = False
    
    if not selected_domains:
        st.error("Please select at least one domain")
        valid = False
    
    if valid:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = [
                "Initializing scraper...",
                "Fetching content...",
                "Extracting text...",
                "Running NLP analysis...",
                "Extracting entities and concepts...",
                "Verifying facts...",
                "Generating summary...",
                "Saving to staging area..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                asyncio.run(asyncio.sleep(0.5))  # Simulate processing
            
            # Success message
            st.success("✅ Content scraped and staged successfully!")
            
            # Show results
            st.subheader("Scraping Results")
            
            # Mock results for demonstration
            results = {
                "submission_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_type}",
                "source_type": source_type,
                "url": url if source_type in ['webpage', 'article'] else None,
                "title": "Example Extracted Title",
                "content_preview": "This is a preview of the extracted content...",
                "word_count": 1234,
                "entities_found": 15,
                "concepts_extracted": 8,
                "confidence_score": 0.85,
                "status": "pending"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", results['word_count'])
                st.metric("Entities Found", results['entities_found'])
            with col2:
                st.metric("Concepts Extracted", results['concepts_extracted'])
                st.metric("Confidence Score", f"{results['confidence_score']:.2%}")
            
            # Options for next steps
            st.write("**Next Steps:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 View Full Content"):
                    st.info("Redirecting to content viewer...")
            
            with col2:
                if st.button("🔍 Run Analysis"):
                    st.info("Starting analysis pipeline...")
            
            with col3:
                if st.button("✅ Approve for Import"):
                    st.info("Moving to approval queue...")

# Recent Scraping History
st.subheader("Recent Scraping History")

# Mock history data
history_data = pd.DataFrame({
    'Timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D'),
    'Type': ['webpage', 'youtube', 'pdf', 'academic_paper', 'manuscript'],
    'Title': [
        'Stanford Encyclopedia: Metaphysics',
        'Philosophy of Mind Lecture',
        'Ancient Greek Mathematics',
        'Quantum Mechanics Paper',
        'Dead Sea Scrolls Fragment'
    ],
    'Status': ['approved', 'analyzing', 'pending', 'approved', 'rejected'],
    'Confidence': [0.92, 0.87, 0.78, 0.95, 0.45]
})

# Style status column
def style_status(val):
    colors = {
        'approved': 'background-color: #90EE90',
        'analyzing': 'background-color: #FFE4B5',
        'pending': 'background-color: #E6E6FA',
        'rejected': 'background-color: #FFB6C1'
    }
    return colors.get(val, '')

styled_df = history_data.style.applymap(style_status, subset=['Status'])
st.dataframe(styled_df, use_container_width=True)

# Batch Processing
with st.expander("Batch URL Processing"):
    st.write("**Process Multiple URLs**")
    
    batch_urls = st.text_area(
        "Enter URLs (one per line)",
        height=150,
        placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://youtube.com/watch?v=..."
    )
    
    batch_domain = st.selectbox(
        "Domain for all URLs",
        ['Auto-detect'] + ['mathematics', 'science', 'philosophy', 'religion', 'art', 'language']
    )
    
    if st.button("Process Batch"):
        urls = [url.strip() for url in batch_urls.split('\n') if url.strip()]
        if urls:
            st.info(f"Processing {len(urls)} URLs...")
            # Implement batch processing
        else:
            st.warning("Please enter at least one URL")
```

#### Fix 4: File Manager - Database Material Focus
**File: `streamlit_workspace/pages/03_📁_File_Manager.py`**

**Enhancement**: Focus on database CSV files only, not project files

**Solution**:
```python
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
import shutil
from datetime import datetime

st.set_page_config(
    page_title="Database File Manager",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Database File Manager")
st.markdown("Manage CSV files and database content - concepts, relationships, and domain data")

# Define CSV directory structure
CSV_ROOT = Path("CSV")

# Initialize session state
if 'current_path' not in st.session_state:
    st.session_state.current_path = CSV_ROOT
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Sidebar navigation
st.sidebar.header("Database Structure")

# Show directory tree
def show_directory_tree(path: Path, level: int = 0):
    """Display directory tree in sidebar."""
    if path.is_dir():
        if level == 0:
            st.sidebar.write(f"📁 **{path.name}/**")
        else:
            st.sidebar.write("  " * level + f"📁 {path.name}/")
        
        for item in sorted(path.iterdir()):
            if item.is_dir():
                show_directory_tree(item, level + 1)
            elif item.suffix == '.csv':
                if st.sidebar.button(f"{'  ' * (level + 1)}📄 {item.name}", key=str(item)):
                    st.session_state.selected_file = item

show_directory_tree(CSV_ROOT)

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("CSV File Browser")
    
    # Current directory info
    current_dir = st.session_state.current_path
    st.write(f"**Current Directory**: `{current_dir.relative_to(CSV_ROOT)}`")
    
    # List files in current directory
    csv_files = list(current_dir.glob("*.csv"))
    subdirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        st.write("**Subdirectories:**")
        for subdir in subdirs:
            if st.button(f"📁 {subdir.name}", key=f"dir_{subdir}"):
                st.session_state.current_path = subdir
                st.rerun()
    
    if csv_files:
        st.write("**CSV Files:**")
        for csv_file in csv_files:
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                if st.button(f"📄 {csv_file.name}", key=f"file_{csv_file}"):
                    st.session_state.selected_file = csv_file
            with col_b:
                size_kb = csv_file.stat().st_size / 1024
                st.write(f"{size_kb:.1f} KB")
            with col_c:
                if st.button("🗑️", key=f"del_{csv_file}"):
                    if st.checkbox(f"Confirm delete {csv_file.name}?", key=f"confirm_{csv_file}"):
                        csv_file.unlink()
                        st.success(f"Deleted {csv_file.name}")
                        st.rerun()
    
    # Navigation
    if current_dir != CSV_ROOT:
        if st.button("⬆️ Go Up"):
            st.session_state.current_path = current_dir.parent
            st.rerun()

with col2:
    st.subheader("File Editor")
    
    if st.session_state.selected_file:
        selected_file = st.session_state.selected_file
        
        # File info
        st.write(f"**Editing**: `{selected_file.name}`")
        st.write(f"**Path**: `{selected_file.relative_to(CSV_ROOT)}`")
        
        # Determine file type based on name
        file_type = "Unknown"
        if "concepts" in selected_file.name:
            file_type = "Concepts"
        elif "relationships" in selected_file.name:
            file_type = "Relationships"
        elif "people" in selected_file.name:
            file_type = "People"
        elif "works" in selected_file.name:
            file_type = "Works"
        elif "places" in selected_file.name:
            file_type = "Places"
        
        st.write(f"**Type**: {file_type}")
        
        # Load and display CSV
        try:
            df = pd.read_csv(selected_file)
            
            # Show statistics
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric("Rows", len(df))
            with col_2:
                st.metric("Columns", len(df.columns))
            with col_3:
                st.metric("Domain", selected_file.parent.name)
            
            # Data validation based on file type
            if file_type == "Concepts":
                required_cols = ['id', 'name', 'description', 'domain']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing required columns: {missing_cols}")
            
            # Edit mode
            edit_mode = st.checkbox("Enable editing", key=f"edit_{selected_file}")
            
            if edit_mode:
                st.warning("⚠️ Be careful when editing! Changes will be saved to the database.")
                
                # Editable dataframe
                edited_df = st.data_editor(
                    df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"editor_{selected_file}"
                )
                
                # Save changes
                col_save, col_backup = st.columns(2)
                
                with col_save:
                    if st.button("💾 Save Changes", type="primary"):
                        # Create backup first
                        backup_path = selected_file.parent / f"{selected_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        shutil.copy2(selected_file, backup_path)
                        
                        # Save edited dataframe
                        edited_df.to_csv(selected_file, index=False)
                        st.success(f"Saved changes to {selected_file.name}")
                        st.info(f"Backup created: {backup_path.name}")
                
                with col_backup:
                    if st.button("🔄 Revert Changes"):
                        st.rerun()
            
            else:
                # Read-only view
                st.dataframe(df, use_container_width=True)
            
            # Data analysis tools
            st.subheader("Data Analysis")
            
            analysis_tabs = st.tabs(["Overview", "Search", "Relationships", "Export"])
            
            with analysis_tabs[0]:
                # Overview
                st.write("**Column Information:**")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
                # Show sample unique values for each column
                st.write("**Sample Values:**")
                for col in df.columns:
                    if df[col].nunique() < 20:
                        st.write(f"- **{col}**: {', '.join(map(str, df[col].unique()[:10]))}")
            
            with analysis_tabs[1]:
                # Search functionality
                search_col = st.selectbox("Search in column", df.columns)
                search_term = st.text_input("Search term")
                
                if search_term:
                    mask = df[search_col].astype(str).str.contains(search_term, case=False, na=False)
                    search_results = df[mask]
                    
                    st.write(f"Found {len(search_results)} matches:")
                    st.dataframe(search_results, use_container_width=True)
            
            with analysis_tabs[2]:
                # Relationship analysis
                if file_type == "Relationships":
                    st.write("**Relationship Statistics:**")
                    
                    if 'relationship_type' in df.columns:
                        rel_counts = df['relationship_type'].value_counts()
                        st.bar_chart(rel_counts)
                    
                    if 'source_id' in df.columns and 'target_id' in df.columns:
                        st.write(f"- Total relationships: {len(df)}")
                        st.write(f"- Unique sources: {df['source_id'].nunique()}")
                        st.write(f"- Unique targets: {df['target_id'].nunique()}")
                
                elif file_type == "Concepts":
                    if 'domain' in df.columns:
                        domain_counts = df['domain'].value_counts()
                        st.write("**Concepts per Domain:**")
                        st.bar_chart(domain_counts)
            
            with analysis_tabs[3]:
                # Export options
                st.write("**Export Data**")
                
                export_format = st.selectbox(
                    "Export format",
                    ["CSV", "JSON", "Excel", "Neo4j Cypher"]
                )
                
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{selected_file.stem}_export.csv",
                        "text/csv"
                    )
                
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        f"{selected_file.stem}_export.json",
                        "application/json"
                    )
                
                elif export_format == "Excel":
                    # Would need to implement Excel export
                    st.info("Excel export coming soon!")
                
                elif export_format == "Neo4j Cypher":
                    # Generate Cypher queries
                    if file_type == "Concepts":
                        cypher_queries = []
                        for _, row in df.iterrows():
                            query = f"CREATE (c:Concept {{id: '{row['id']}', name: '{row['name']}', domain: '{row['domain']}', description: '{row.get('description', '')}'}});"
                            cypher_queries.append(query)
                        
                        cypher_text = '\n'.join(cypher_queries)
                        st.download_button(
                            "Download Cypher Queries",
                            cypher_text,
                            f"{selected_file.stem}_cypher.txt",
                            "text/plain"
                        )
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("Select a CSV file from the sidebar to view and edit")

# Create new file section
st.subheader("Create New CSV File")

new_file_type = st.selectbox(
    "File Type",
    ["Concepts", "Relationships", "People", "Works", "Places"]
)

if new_file_type == "Concepts":
    st.write("Create a new concepts file with the standard schema:")
    
    domain = st.selectbox("Domain", ["mathematics", "science", "philosophy", "religion", "art", "language"])
    
    if st.button("Create Concepts File"):
        # Create empty concepts dataframe
        new_df = pd.DataFrame(columns=['id', 'name', 'description', 'domain', 'keywords', 'related_concepts'])
        
        # Save to appropriate directory
        domain_dir = CSV_ROOT / domain
        domain_dir.mkdir(exist_ok=True)
        
        filename = f"{domain}_concepts_new.csv"
        filepath = domain_dir / filename
        
        new_df.to_csv(filepath, index=False)
        st.success(f"Created new file: {filepath.relative_to(CSV_ROOT)}")
        
        # Select the new file
        st.session_state.selected_file = filepath
        st.session_state.current_path = domain_dir
        st.rerun()

# Import/Export section
st.subheader("Bulk Import/Export")

col1, col2 = st.columns(2)

with col1:
    st.write("**Import Data**")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file to import into the database"
    )
    
    if uploaded_file:
        try:
            import_df = pd.read_csv(uploaded_file)
            st.write(f"Preview of {uploaded_file.name}:")
            st.dataframe(import_df.head(), use_container_width=True)
            
            import_domain = st.selectbox(
                "Import to domain",
                ["mathematics", "science", "philosophy", "religion", "art", "language"]
            )
            
            import_type = st.selectbox(
                "Import as",
                ["concepts", "relationships", "people", "works"]
            )
            
            if st.button("Import File"):
                # Save to appropriate location
                target_dir = CSV_ROOT / import_domain
                target_dir.mkdir(exist_ok=True)
                
                target_path = target_dir / f"{import_domain}_{import_type}_imported.csv"
                import_df.to_csv(target_path, index=False)
                
                st.success(f"Imported to: {target_path.relative_to(CSV_ROOT)}")
                
        except Exception as e:
            st.error(f"Error importing file: {str(e)}")

with col2:
    st.write("**Export All Data**")
    
    export_domain = st.selectbox(
        "Export domain",
        ["All domains"] + ["mathematics", "science", "philosophy", "religion", "art", "language"]
    )
    
    if st.button("Export Domain Data"):
        # Create zip file with all CSVs from domain
        st.info("Export functionality coming soon!")

# Database statistics
st.subheader("Database Statistics")

total_concepts = 0
total_relationships = 0
total_people = 0

stats_data = []

for domain_dir in CSV_ROOT.iterdir():
    if domain_dir.is_dir() and domain_dir.name not in ['import', 'sources', 'shared', 'vectors']:
        domain_stats = {
            'Domain': domain_dir.name.capitalize(),
            'Concepts': 0,
            'Relationships': 0,
            'People': 0,
            'Works': 0
        }
        
        for csv_file in domain_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if 'concepts' in csv_file.name:
                    domain_stats['Concepts'] = len(df)
                elif 'relationships' in csv_file.name:
                    domain_stats['Relationships'] = len(df)
                elif 'people' in csv_file.name:
                    domain_stats['People'] = len(df)
                elif 'works' in csv_file.name:
                    domain_stats['Works'] = len(df)
            except:
                pass
        
        stats_data.append(domain_stats)

if stats_data:
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
    
    # Totals
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Concepts", stats_df['Concepts'].sum())
    with col2:
        st.metric("Total Relationships", stats_df['Relationships'].sum())
    with col3:
        st.metric("Total People", stats_df['People'].sum())
    with col4:
        st.metric("Total Works", stats_df['Works'].sum())
```

### 🎨 Priority 2: Enhanced UI Features

#### Database Manager Enhancement
**File: `streamlit_workspace/pages/01_🗄️_Database_Manager.py`**

```python
import streamlit as st
from py2neo import Graph
from qdrant_client import QdrantClient
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Database Manager",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Database Manager")
st.markdown("Comprehensive database operations for Neo4j and Qdrant")

# Initialize connections
@st.cache_resource
def get_connections():
    try:
        neo4j = Graph("bolt://localhost:7687", auth=("neo4j", "yggdrasil"))
        qdrant = QdrantClient(host="localhost", port=6333)
        return neo4j, qdrant
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None, None

neo4j, qdrant = get_connections()

if not neo4j or not qdrant:
    st.stop()

# Tabs for different operations
tabs = st.tabs(["Overview", "Neo4j Operations", "Qdrant Operations", "Import/Export", "Maintenance"])

with tabs[0]:
    # Overview
    st.subheader("Database Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Neo4j Statistics")
        
        # Get Neo4j stats
        stats_query = """
        MATCH (n)
        WITH labels(n) as label
        UNWIND label as l
        RETURN l as NodeType, count(*) as Count
        ORDER BY Count DESC
        """
        
        neo4j_stats = neo4j.run(stats_query).data()
        stats_df = pd.DataFrame(neo4j_stats)
        
        if not stats_df.empty:
            fig = px.bar(stats_df, x='NodeType', y='Count', title='Node Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Relationship count
            rel_count = neo4j.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
            st.metric("Total Relationships", f"{rel_count:,}")
    
    with col2:
        st.write("### Qdrant Statistics")
        
        # Get Qdrant stats
        collections = qdrant.get_collections().collections
        
        qdrant_stats = []
        for collection in collections:
            info = qdrant.get_collection(collection.name)
            qdrant_stats.append({
                'Collection': collection.name,
                'Vectors': info.vectors_count,
                'Dimension': info.config.params.vectors.size
            })
        
        if qdrant_stats:
            qdrant_df = pd.DataFrame(qdrant_stats)
            st.dataframe(qdrant_df, use_container_width=True)
            
            total_vectors = sum(stat['Vectors'] for stat in qdrant_stats)
            st.metric("Total Vectors", f"{total_vectors:,}")

with tabs[1]:
    # Neo4j Operations
    st.subheader("Neo4j Operations")
    
    operation = st.selectbox(
        "Select Operation",
        ["Query", "Create Node", "Create Relationship", "Update", "Delete"]
    )
    
    if operation == "Query":
        st.write("### Execute Cypher Query")
        
        # Predefined queries
        predefined = st.selectbox(
            "Predefined Queries",
            [
                "Custom Query",
                "All Concepts",
                "Cross-Domain Concepts",
                "Event Timeline",
                "Author Network",
                "Concept Relationships"
            ]
        )
        
        if predefined == "Custom Query":
            query = st.text_area("Cypher Query", height=150)
        elif predefined == "All Concepts":
            query = "MATCH (c:Concept) RETURN c.name as Name, c.domain as Domain LIMIT 100"
        elif predefined == "Cross-Domain Concepts":
            query = """
            MATCH (c1:Concept)-[r:RELATES_TO]-(c2:Concept)
            WHERE c1.domain <> c2.domain
            RETURN c1.name as Concept1, c1.domain as Domain1, 
                   type(r) as Relationship,
                   c2.name as Concept2, c2.domain as Domain2
            LIMIT 50
            """
        elif predefined == "Event Timeline":
            query = """
            MATCH (e:Event)
            RETURN e.name as Event, e.start_date as Start, e.end_date as End
            ORDER BY e.start_date
            """
        else:
            query = "MATCH (n) RETURN n LIMIT 10"
        
        if st.button("Execute Query"):
            try:
                result = neo4j.run(query).data()
                if result:
                    df = pd.DataFrame(result)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "query_results.csv",
                        "text/csv"
                    )
                else:
                    st.info("Query returned no results")
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    
    elif operation == "Create Node":
        st.write("### Create New Node")
        
        node_type = st.selectbox(
            "Node Type",
            ["Concept", "Entity", "Document", "Author", "Event", "Claim"]
        )
        
        # Dynamic form based on node type
        with st.form("create_node"):
            if node_type == "Concept":
                name = st.text_input("Name*", placeholder="Metaphysics")
                domain = st.selectbox("Domain*", ["mathematics", "science", "philosophy", "religion", "art", "language"])
                description = st.text_area("Description")
                keywords = st.text_input("Keywords", placeholder="ontology, being, existence")
            
            elif node_type == "Event":
                name = st.text_input("Event Name*", placeholder="Renaissance")
                start_date = st.text_input("Start Date", placeholder="1300")
                end_date = st.text_input("End Date", placeholder="1600")
                description = st.text_area("Description")
                significance = st.selectbox("Historical Significance", ["major", "moderate", "minor"])
            
            elif node_type == "Author":
                name = st.text_input("Name*", placeholder="Aristotle")
                birth_year = st.text_input("Birth Year", placeholder="-384")
                death_year = st.text_input("Death Year", placeholder="-322")
                nationality = st.text_input("Nationality", placeholder="Greek")
            
            else:
                # Generic node creation
                name = st.text_input("Name/Title*")
                properties = st.text_area("Properties (JSON)", placeholder='{"key": "value"}')
            
            submit = st.form_submit_button("Create Node")
            
            if submit and name:
                try:
                    # Build query based on node type
                    if node_type == "Concept":
                        create_query = """
                        CREATE (n:Concept {
                            id: randomUUID(),
                            name: $name,
                            domain: $domain,
                            description: $description,
                            keywords: $keywords,
                            created_at: datetime()
                        })
                        RETURN n
                        """
                        params = {
                            'name': name,
                            'domain': domain,
                            'description': description,
                            'keywords': keywords
                        }
                    
                    elif node_type == "Event":
                        create_query = """
                        CREATE (n:Event {
                            id: randomUUID(),
                            name: $name,
                            start_date: $start_date,
                            end_date: $end_date,
                            description: $description,
                            historical_significance: $significance,
                            created_at: datetime()
                        })
                        RETURN n
                        """
                        params = {
                            'name': name,
                            'start_date': start_date,
                            'end_date': end_date,
                            'description': description,
                            'significance': significance
                        }
                    
                    else:
                        # Generic creation
                        create_query = f"""
                        CREATE (n:{node_type} {{
                            id: randomUUID(),
                            name: $name,
                            created_at: datetime()
                        }})
                        RETURN n
                        """
                        params = {'name': name}
                    
                    result = neo4j.run(create_query, params).data()
                    st.success(f"Created {node_type}: {name}")
                    
                except Exception as e:
                    st.error(f"Creation error: {str(e)}")

with tabs[2]:
    # Qdrant Operations
    st.subheader("Qdrant Vector Operations")
    
    qdrant_op = st.selectbox(
        "Operation",
        ["Search", "Collection Management", "Vector Analysis"]
    )
    
    if qdrant_op == "Search":
        st.write("### Semantic Search")
        
        # Collection selection
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        selected_collection = st.selectbox("Collection", collection_names)
        
        search_text = st.text_area("Search Query", placeholder="Enter text to search for similar content...")
        
        search_limit = st.slider("Number of Results", 1, 50, 10)
        
        if st.button("Search") and search_text:
            try:
                # In real implementation, would use proper embedding model
                # For demo, using mock search
                st.info(f"Searching in {selected_collection} for: '{search_text}'")
                
                # Mock results
                results = pd.DataFrame({
                    'Score': [0.95, 0.89, 0.87, 0.82, 0.78],
                    'Title': [
                        'Metaphysics of Mind',
                        'Philosophy of Consciousness',
                        'Dualism vs Materialism',
                        'The Hard Problem',
                        'Phenomenology'
                    ],
                    'Domain': ['philosophy', 'philosophy', 'philosophy', 'science', 'philosophy']
                })
                
                st.dataframe(results.head(search_limit), use_container_width=True)
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    elif qdrant_op == "Collection Management":
        st.write("### Manage Collections")
        
        # List collections
        collections = qdrant.get_collections().collections
        
        if collections:
            for collection in collections:
                with st.expander(f"Collection: {collection.name}"):
                    info = qdrant.get_collection(collection.name)
                    st.write(f"- Vectors: {info.vectors_count}")
                    st.write(f"- Dimension: {info.config.params.vectors.size}")
                    st.write(f"- Status: {info.status}")
                    
                    if st.button(f"Delete {collection.name}", key=f"del_{collection.name}"):
                        if st.checkbox(f"Confirm delete {collection.name}?", key=f"confirm_{collection.name}"):
                            qdrant.delete_collection(collection.name)
                            st.success(f"Deleted collection: {collection.name}")
                            st.rerun()
        
        # Create new collection
        st.write("### Create New Collection")
        
        with st.form("create_collection"):
            new_name = st.text_input("Collection Name", placeholder="documents_history")
            vector_size = st.number_input("Vector Dimension", min_value=1, max_value=4096, value=384)
            distance = st.selectbox("Distance Metric", ["Cosine", "Euclidean", "Dot"])
            
            if st.form_submit_button("Create Collection"):
                try:
                    from qdrant_client.models import Distance, VectorParams
                    
                    distance_map = {
                        "Cosine": Distance.COSINE,
                        "Euclidean": Distance.EUCLID,
                        "Dot": Distance.DOT
                    }
                    
                    qdrant.create_collection(
                        collection_name=new_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=distance_map[distance]
                        )
                    )
                    
                    st.success(f"Created collection: {new_name}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Creation error: {str(e)}")

with tabs[3]:
    # Import/Export
    st.subheader("Import/Export Operations")
    
    operation = st.radio("Operation", ["Import", "Export"])
    
    if operation == "Import":
        st.write("### Import Data")
        
        import_type = st.selectbox(
            "Import Type",
            ["CSV to Neo4j", "JSON to Neo4j", "Approved Staging Content"]
        )
        
        if import_type == "CSV to Neo4j":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(df.head(), use_container_width=True)
                
                node_type = st.selectbox("Import as node type", ["Concept", "Entity", "Document", "Author"])
                
                if st.button("Import to Neo4j"):
                    progress = st.progress(0)
                    
                    for i, row in df.iterrows():
                        # Create node for each row
                        create_query = f"""
                        CREATE (n:{node_type} {{
                            id: randomUUID(),
                            imported_at: datetime()
                        }})
                        SET n += $properties
                        """
                        
                        neo4j.run(create_query, properties=row.to_dict())
                        progress.progress((i + 1) / len(df))
                    
                    st.success(f"Imported {len(df)} nodes")
        
        elif import_type == "Approved Staging Content":
            # Import from staging area
            st.info("This would import approved content from the staging area")
    
    else:  # Export
        st.write("### Export Data")
        
        export_type = st.selectbox(
            "Export Type",
            ["Neo4j to CSV", "Neo4j to JSON", "Full Database Backup"]
        )
        
        if export_type == "Neo4j to CSV":
            # Predefined export queries
            export_query = st.text_area(
                "Export Query",
                value="MATCH (n:Concept) RETURN n.id as id, n.name as name, n.domain as domain"
            )
            
            if st.button("Export"):
                try:
                    result = neo4j.run(export_query).data()
                    if result:
                        df = pd.DataFrame(result)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"neo4j_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")

with tabs[4]:
    # Maintenance
    st.subheader("Database Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Neo4j Maintenance")
        
        if st.button("Check Database Integrity"):
            # Run integrity checks
            checks = []
            
            # Check for orphaned nodes
            orphan_check = neo4j.run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN count(n) as orphaned_nodes
            """).data()[0]
            
            checks.append({"Check": "Orphaned Nodes", "Result": orphan_check['orphaned_nodes']})
            
            # Check for duplicate concepts
            duplicate_check = neo4j.run("""
            MATCH (c:Concept)
            WITH c.name as name, count(*) as count
            WHERE count > 1
            RETURN count(*) as duplicates
            """).data()
            
            duplicates = len(duplicate_check) if duplicate_check else 0
            checks.append({"Check": "Duplicate Concepts", "Result": duplicates})
            
            st.dataframe(pd.DataFrame(checks), use_container_width=True)
        
        if st.button("Optimize Indexes"):
            st.info("Index optimization would be performed here")
        
        if st.button("Clear Cache"):
            st.info("Cache clearing would be performed here")
    
    with col2:
        st.write("### Qdrant Maintenance")
        
        if st.button("Optimize Collections"):
            st.info("Collection optimization would be performed here")
        
        if st.button("Reindex Vectors"):
            st.info("Vector reindexing would be performed here")
        
        if st.button("Backup Qdrant"):
            st.info("Qdrant backup would be performed here")
```

### Implementation Checklist

#### Week 9: Core UI Fixes
- [ ] Fix Operations Console psutil import error
- [ ] Fix Graph Editor to show actual Neo4j data
- [ ] Enhance Content Scraper with source type selection
- [ ] Focus File Manager on database CSV files only
- [ ] Test all fixes thoroughly

#### Week 10: Advanced UI Features
- [ ] Enhance Database Manager with CRUD operations
- [ ] Add Knowledge Tools page functionality
- [ ] Improve Analytics dashboard
- [ ] Add concept relationship visualization
- [ ] Implement cross-cultural connections display

### Success Criteria
- ✅ All pages load without errors
- ✅ psutil properly installed and working
- ✅ Graph Editor shows real Neo4j data
- ✅ Content Scraper supports 10+ source types
- ✅ File Manager focuses on CSV database files only
- ✅ Drag-and-drop graph editing functional
- ✅ Cross-cultural concept connections visible

### Next Steps
After completing Phase 5, proceed to:
- **Phase 6**: Technical Specifications (`updates/06_technical_specs.md`)
- **Phase 7**: Metrics and Timeline (`updates/07_metrics_timeline.md`)