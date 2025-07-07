#!/bin/bash

echo "🚀 Starting MCP Yggdrasil Workspace..."

# Set working directory
cd "$(dirname "$0")/streamlit_workspace"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️ Streamlit not found. Installing..."
    pip3 install streamlit
fi

# Install required dependencies
echo "📦 Installing dependencies..."
pip3 install plotly networkx pandas python-dotenv neo4j requests validators

# Create .env file if it doesn't exist
if [ ! -f "../.env" ]; then
    echo "🔧 Creating .env file..."
    cat > ../.env << EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
EOF
fi

echo "🌐 Starting Streamlit application..."
echo "📍 Access the app at: http://localhost:8502"
echo "   - Graph Editor will show demo data if Neo4j isn't connected"
echo "   - File Manager shows CSV database files"
echo "   - Scraper has all requested source types"

# Start the application
streamlit run main_dashboard.py --server.port 8502 --server.headless true