"""
MCP Yggdrasil Web Application
Simple FastAPI app for database management and querying
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import requests
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MCP Yggdrasil Knowledge System",
    description="Hybrid Neo4j + Qdrant knowledge management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connections
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
QDRANT_URL = 'http://localhost:6333'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class ConceptSearch(BaseModel):
    text: str
    domain: Optional[str] = None
    limit: Optional[int] = 5

class ConceptUpdate(BaseModel):
    concept_id: str
    field: str
    value: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Yggdrasil Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2E8B57; margin-bottom: 30px; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .button { background: #2E8B57; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .button:hover { background: #236B47; }
            input, textarea { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 3px; }
            .result { background: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 10px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .stat-card { background: #e8f5e8; padding: 15px; border-radius: 5px; text-align: center; }
            .stat-number { font-size: 2em; font-weight: bold; color: #2E8B57; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌳 MCP Yggdrasil Knowledge System</h1>
                <p>Hybrid Neo4j + Qdrant Database Management</p>
            </div>
            
            <div id="stats" class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="concept-count">Loading...</div>
                    <div>Total Concepts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="relationship-count">Loading...</div>
                    <div>Total Relationships</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="domain-count">Loading...</div>
                    <div>Domains</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="vector-count">Loading...</div>
                    <div>Vector Embeddings</div>
                </div>
            </div>

            <div class="section">
                <h3>🔍 Search Concepts</h3>
                <input type="text" id="search-input" placeholder="Search concepts by name or description...">
                <select id="domain-filter">
                    <option value="">All Domains</option>
                    <option value="Art">Art</option>
                    <option value="Science">Science</option>
                    <option value="Mathematics">Mathematics</option>
                    <option value="Philosophy">Philosophy</option>
                    <option value="Language">Language</option>
                    <option value="Technology">Technology</option>
                    <option value="Religion">Religion</option>
                    <option value="Astrology">Astrology</option>
                </select>
                <button class="button" onclick="searchConcepts()">Search</button>
                <div id="search-results" class="result" style="display:none;"></div>
            </div>

            <div class="section">
                <h3>📊 Domain Overview</h3>
                <button class="button" onclick="loadDomainStats()">Load Domain Statistics</button>
                <div id="domain-stats" class="result" style="display:none;"></div>
            </div>

            <div class="section">
                <h3>🔗 Relationship Explorer</h3>
                <input type="text" id="concept-id" placeholder="Enter Concept ID (e.g., ART0001)">
                <button class="button" onclick="exploreRelationships()">Explore Relationships</button>
                <div id="relationship-results" class="result" style="display:none;"></div>
            </div>

            <div class="section">
                <h3>⚡ Custom Cypher Query</h3>
                <textarea id="cypher-query" rows="4" placeholder="Enter Cypher query (e.g., MATCH (c:Concept) RETURN c.name LIMIT 10)"></textarea>
                <button class="button" onclick="runQuery()">Execute Query</button>
                <div id="query-results" class="result" style="display:none;"></div>
            </div>

            <div class="section">
                <h3>🎯 Vector Similarity Search</h3>
                <input type="text" id="vector-search" placeholder="Enter text to find similar concepts...">
                <button class="button" onclick="vectorSearch()">Search Similar</button>
                <div id="vector-results" class="result" style="display:none;"></div>
            </div>
        </div>

        <script>
            // Load initial stats
            window.onload = function() {
                loadStats();
            };

            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    document.getElementById('concept-count').textContent = stats.concepts;
                    document.getElementById('relationship-count').textContent = stats.relationships;
                    document.getElementById('domain-count').textContent = stats.domains;
                    document.getElementById('vector-count').textContent = stats.vectors;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }

            async function searchConcepts() {
                const query = document.getElementById('search-input').value;
                const domain = document.getElementById('domain-filter').value;
                
                try {
                    const response = await fetch('/api/search/concepts', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: query, domain: domain || null, limit: 10})
                    });
                    const results = await response.json();
                    
                    let html = '<h4>Search Results:</h4>';
                    results.forEach(concept => {
                        html += `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 3px;">
                                <strong>${concept.id}: ${concept.name}</strong> (${concept.domain})<br>
                                <em>${concept.description}</em><br>
                                <small>Type: ${concept.type}, Level: ${concept.level}</small>
                            </div>
                        `;
                    });
                    
                    document.getElementById('search-results').innerHTML = html;
                    document.getElementById('search-results').style.display = 'block';
                } catch (error) {
                    console.error('Error searching concepts:', error);
                }
            }

            async function loadDomainStats() {
                try {
                    const response = await fetch('/api/domains');
                    const domains = await response.json();
                    
                    let html = '<h4>Domain Statistics:</h4>';
                    Object.entries(domains).forEach(([domain, count]) => {
                        html += `<div><strong>${domain}:</strong> ${count} concepts</div>`;
                    });
                    
                    document.getElementById('domain-stats').innerHTML = html;
                    document.getElementById('domain-stats').style.display = 'block';
                } catch (error) {
                    console.error('Error loading domain stats:', error);
                }
            }

            async function exploreRelationships() {
                const conceptId = document.getElementById('concept-id').value;
                
                try {
                    const response = await fetch(`/api/concepts/${conceptId}/relationships`);
                    const relationships = await response.json();
                    
                    let html = '<h4>Relationships:</h4>';
                    relationships.forEach(rel => {
                        html += `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 3px;">
                                <strong>${rel.relationship_type}:</strong> ${rel.target_name} (${rel.target_id})<br>
                                <small>Domain: ${rel.domain}</small>
                            </div>
                        `;
                    });
                    
                    document.getElementById('relationship-results').innerHTML = html;
                    document.getElementById('relationship-results').style.display = 'block';
                } catch (error) {
                    console.error('Error exploring relationships:', error);
                }
            }

            async function runQuery() {
                const query = document.getElementById('cypher-query').value;
                
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query, limit: 20})
                    });
                    const results = await response.json();
                    
                    let html = '<h4>Query Results:</h4><pre>' + JSON.stringify(results, null, 2) + '</pre>';
                    document.getElementById('query-results').innerHTML = html;
                    document.getElementById('query-results').style.display = 'block';
                } catch (error) {
                    console.error('Error running query:', error);
                }
            }

            async function vectorSearch() {
                const text = document.getElementById('vector-search').value;
                
                try {
                    const response = await fetch('/api/search/vector', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, limit: 5})
                    });
                    const results = await response.json();
                    
                    let html = '<h4>Similar Concepts:</h4>';
                    results.forEach(result => {
                        html += `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 3px;">
                                <strong>${result.concept_id}: ${result.name}</strong> (Score: ${result.score.toFixed(3)})<br>
                                <em>${result.description}</em><br>
                                <small>Domain: ${result.domain}</small>
                            </div>
                        `;
                    });
                    
                    document.getElementById('vector-results').innerHTML = html;
                    document.getElementById('vector-results').style.display = 'block';
                } catch (error) {
                    console.error('Error in vector search:', error);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    with driver.session() as session:
        # Get concept count
        result = session.run("MATCH (c:Concept) RETURN count(c) as count")
        concepts = result.single()["count"]
        
        # Get relationship count
        result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count")
        relationships = result.single()["count"]
        
        # Get domain count
        result = session.run("MATCH (c:Concept) RETURN count(DISTINCT c.domain) as count")
        domains = result.single()["count"]
    
    # Get vector count from Qdrant
    try:
        response = requests.get(f"{QDRANT_URL}/collections/mcp_yggdrasil_concepts")
        if response.status_code == 200:
            vectors = response.json()["result"]["vectors_count"]
        else:
            vectors = 0
    except:
        vectors = 0
    
    return {
        "concepts": concepts,
        "relationships": relationships,
        "domains": domains,
        "vectors": vectors
    }

@app.get("/api/domains")
async def get_domains():
    """Get concept count by domain"""
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            RETURN c.domain as domain, count(c) as count
            ORDER BY count DESC
        """)
        
        domains = {record["domain"]: record["count"] for record in result}
        return domains

@app.post("/api/search/concepts")
async def search_concepts(request: ConceptSearch):
    """Search concepts by text"""
    with driver.session() as session:
        if request.domain:
            query = """
                MATCH (c:Concept)
                WHERE c.domain = $domain AND 
                      (toLower(c.name) CONTAINS toLower($text) OR 
                       toLower(c.description) CONTAINS toLower($text))
                RETURN c.id as id, c.name as name, c.description as description, 
                       c.domain as domain, c.type as type, c.level as level
                LIMIT $limit
            """
            result = session.run(query, text=request.text, domain=request.domain, limit=request.limit)
        else:
            query = """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($text) OR 
                      toLower(c.description) CONTAINS toLower($text)
                RETURN c.id as id, c.name as name, c.description as description, 
                       c.domain as domain, c.type as type, c.level as level
                LIMIT $limit
            """
            result = session.run(query, text=request.text, limit=request.limit)
        
        concepts = [dict(record) for record in result]
        return concepts

@app.get("/api/concepts/{concept_id}/relationships")
async def get_concept_relationships(concept_id: str):
    """Get relationships for a specific concept"""
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Concept {id: $concept_id})-[r:RELATES_TO]->(target:Concept)
            RETURN r.relationship_type as relationship_type, r.domain as domain,
                   target.id as target_id, target.name as target_name
            LIMIT 20
        """, concept_id=concept_id)
        
        relationships = [dict(record) for record in result]
        return relationships

@app.post("/api/query")
async def run_cypher_query(request: QueryRequest):
    """Execute a custom Cypher query"""
    try:
        with driver.session() as session:
            result = session.run(request.query)
            records = [dict(record) for record in result.data()]
            return records[:request.limit]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")

@app.post("/api/search/vector")
async def vector_search(request: ConceptSearch):
    """Perform vector similarity search"""
    # Simple hash-based embedding for demo (same as used in sync)
    import hashlib
    
    def simple_embedding(text, size=384):
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        vector = []
        for i in range(size):
            char_val = ord(hash_hex[i % len(hash_hex)]) / 255.0
            vector.append(char_val - 0.5)
        magnitude = sum(x*x for x in vector) ** 0.5
        return [x/magnitude for x in vector]
    
    search_vector = simple_embedding(request.text)
    
    try:
        search_data = {
            "vector": search_vector,
            "limit": request.limit,
            "with_payload": True
        }
        
        if request.domain:
            search_data["filter"] = {
                "must": [{"key": "domain", "match": {"value": request.domain}}]
            }
        
        response = requests.post(
            f"{QDRANT_URL}/collections/mcp_yggdrasil_concepts/points/search",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(search_data)
        )
        
        if response.status_code == 200:
            results = response.json()["result"]
            return [
                {
                    "concept_id": result["payload"]["concept_id"],
                    "name": result["payload"]["name"],
                    "description": result["payload"]["description"],
                    "domain": result["payload"]["domain"],
                    "score": result["score"]
                }
                for result in results
            ]
        else:
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    driver.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_main:app", host="0.0.0.0", port=8000, reload=True)