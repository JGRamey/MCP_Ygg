#!/bin/bash

echo "🐳 Starting Docker services for MCP Yggdrasil..."

# Start Docker services
echo "Starting Neo4j..."
docker run -d --name mcp-neo4j --rm -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.15-community

echo "Starting Qdrant..."
docker run -d --name mcp-qdrant --rm -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:v1.7.3

echo "Starting Redis..."
docker run -d --name mcp-redis --rm -p 6379:6379 \
  redis:7.2-alpine

echo "✅ Docker services started!"
echo "🔗 Access points:"
echo "   - Neo4j Browser: http://localhost:7474"
echo "   - Qdrant API: http://localhost:6333"
echo "   - Redis: localhost:6379"
echo "   - Streamlit App: http://localhost:8502"