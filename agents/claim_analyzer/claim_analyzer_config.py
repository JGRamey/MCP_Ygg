# agents/claim_analyzer/config.yaml
# Claim Analyzer Agent Configuration for MCP Server

# Database configuration (inherits from main MCP server config)
database:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
    max_pool_size: 20
    connection_timeout: 30
    
  qdrant:
    host: "localhost"
    port: 6333
    timeout: 30
    collection_prefix: "claim_analyzer_"
    
  redis:
    url: "redis://localhost:6379"
    max_connections: 50
    cache_ttl: 3600  # 1 hour

# Agent-specific settings
agent:
  name: "claim_analyzer"
  version: "1.0.0"
  
  # Processing settings
  max_results: 10
  confidence_threshold: 0.5
  batch_size: 50
  processing_interval: 300  # 5 minutes
  
  # Claim extraction settings
  min_claim_length: 10
  max_claim_length: 1000
  claim_confidence_threshold: 0.3
  
  # Fact-checking settings
  similarity_threshold: 0.7
  evidence_limit: 15
  cross_domain_analysis: true
  
  # Performance settings
  max_concurrent_checks: 5
  rate_limit_delay: 2.0
  enable_caching: true

# NLP and ML models
models:
  spacy_model: "en_core_web_sm"
  sentence_transformer: "all-MiniLM-L6-v2"
  
  # Model-specific settings
  embedding_dimensions: 384
  batch_size: 32
  device: "cpu"  # or "cuda" for GPU

# Source credibility scoring
source_credibility:
  # Fact-checking sites
  "snopes.com": 0.95
  "factcheck.org": 0.95
  "politifact.com": 0.90
  "fullfact.org": 0.90
  
  # News organizations
  "reuters.com": 0.90
  "bbc.com": 0.85
  "npr.org": 0.85
  "ap.org": 0.90
  "reuters.org": 0.90
  
  # Government/Official sources
  "nasa.gov": 0.95
  "cdc.gov": 0.95
  "who.int": 0.90
  "nih.gov": 0.90
  "noaa.gov": 0.90
  
  # Academic/Research
  "wikipedia.org": 0.80
  "nature.com": 0.95
  "science.org": 0.95
  "arxiv.org": 0.80
  "pubmed.ncbi.nlm.nih.gov": 0.90
  
  # Default scores by domain
  default_credibility:
    government: 0.85
    academic: 0.80
    news: 0.70
    blog: 0.40
    social_media: 0.20
    unknown: 0.50

# Domain classification keywords
domain_keywords:
  science:
    - "experiment"
    - "study" 
    - "research"
    - "theory"
    - "hypothesis"
    - "data"
    - "evidence"
    - "scientific"
    - "empirical"
    - "methodology"
    
  math:
    - "theorem"
    - "proof"
    - "equation"
    - "formula"
    - "calculate"
    - "number"
    - "geometry"
    - "algebra"
    - "statistics"
    - "mathematical"
    
  religion:
    - "god"
    - "faith"
    - "belief"
    - "scripture"
    - "church"
    - "prayer"
    - "divine"
    - "spiritual"
    - "holy"
    - "sacred"
    
  history:
    - "ancient"
    - "war"
    - "civilization"
    - "empire"
    - "century"
    - "historical"
    - "period"
    - "dynasty"
    - "archaeological"
    - "medieval"
    
  literature:
    - "novel"
    - "poem"
    - "author"
    - "book"
    - "story"
    - "character"
    - "literary"
    - "narrative"
    - "fiction"
    - "poetry"
    
  philosophy:
    - "ethics"
    - "logic"
    - "metaphysics"
    - "epistemology"
    - "moral"
    - "existence"
    - "philosophical"
    - "ontology"
    - "phenomenology"
    - "existential"

# Claim detection patterns (regex)
claim_patterns:
  factual_statements:
    - "(?:is|are|was|were|will be|has been|have been)\\s+(?:a|an|the)?\\s*\\w+"
    - "(?:claims?|states?|argues?|believes?|says?)\\s+that\\s+.*"
    - "(?:according to|research shows|studies indicate|experts say)\\s+.*"
    - "(?:it is|this is|that is)\\s+(?:true|false|correct|incorrect|a fact)\\s+that\\s+.*"
    
  absolute_statements:
    - "\\b(?:always|never|all|none|every|no one|everyone)\\b"
    - "\\b(?:completely|totally|absolutely|entirely|wholly)\\b"
    
  statistical_claims:
    - "\\d+\\s*(?:percent|%|out of|in \\d+)"
    - "\\b(?:majority|minority|most|few|many|several)\\b"
    
  temporal_claims:
    - "\\b(?:since|until|before|after|during|while)\\b.*\\b(?:year|century|decade|era)\\b"
    - "\\b(?:first|last|earliest|latest|recent|ancient)\\b"

# External API configuration
external_apis:
  # Google Fact Check Tools API
  google_fact_check:
    enabled: false
    api_key: ""
    base_url: "https://factchecktools.googleapis.com/v1alpha1/claims"
    rate_limit: 100  # requests per day
    
  # Bing Search API
  bing_search:
    enabled: false
    api_key: ""
    base_url: "https://api.bing.microsoft.com/v7.0/search"
    rate_limit: 1000  # requests per month
    
  # Custom fact-checking APIs
  custom_apis:
    enabled: false
    endpoints: []

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "agents/claim_analyzer/logs/claim_analyzer.log"
  max_size: "100MB"
  backup_count: 5
  
  # Component-specific logging
  components:
    claim_extractor: "INFO"
    fact_checker: "INFO"
    database: "WARNING"
    models: "WARNING"

# Monitoring and metrics
monitoring:
  enabled: true
  metrics_port: 9091
  
  # Custom metrics
  metrics:
    - claims_processed_total
    - fact_checks_performed_total
    - verdict_distribution
    - average_confidence_score
    - evidence_sources_count
    - cross_domain_patterns_found
    - database_query_duration
    - model_inference_duration
    
  # Alerts
  alerts:
    low_confidence_threshold: 0.3
    high_error_rate_threshold: 0.1
    slow_response_threshold: 30  # seconds

# Integration with other MCP agents
integration:
  # Text Processor Agent
  text_processor:
    enabled: true
    preprocess_text: true
    extract_entities: true
    
  # Scraper Agent
  scraper:
    enabled: true
    auto_process_scraped_content: true
    domains_filter: ["science", "history", "philosophy"]
    
  # Pattern Recognition Agent
  pattern_recognition:
    enabled: true
    share_cross_domain_findings: true
    pattern_confidence_threshold: 0.7
    
  # Backup Agent
  backup:
    enabled: true
    backup_claims_data: true
    backup_interval: "daily"

# Security settings
security:
  input_validation:
    max_text_length: 100000
    sanitize_html: true
    filter_malicious_patterns: true
    
  rate_limiting:
    max_requests_per_minute: 60
    max_claims_per_request: 10
    burst_limit: 5
    
  data_privacy:
    anonymize_sources: false
    retention_period_days: 365
    encrypt_sensitive_data: false

# Performance optimization
performance:
  # Caching strategies
  caching:
    enable_claim_cache: true
    enable_evidence_cache: true
    enable_similarity_cache: true
    cache_size_limit: "1GB"
    
  # Parallel processing
  parallel_processing:
    max_workers: 4
    chunk_size: 25
    enable_async: true
    
  # Database optimization
  database_optimization:
    connection_pooling: true
    query_timeout: 30
    batch_operations: true
    index_optimization: true

# Feature flags
features:
  # Core features
  claim_extraction: true
  fact_checking: true
  similarity_search: true
  cross_domain_analysis: true
  
  # Advanced features
  real_time_processing: true
  automated_verification: true
  sentiment_analysis: false
  bias_detection: false
  multimodal_analysis: false
  
  # Experimental features
  llm_integration: false
  automated_learning: false
  community_validation: false
  blockchain_verification: false

---
# agents/claim_analyzer/__init__.py
"""
Claim Analyzer Agent Package
"""

from .claim_analyzer_agent import ClaimAnalyzerAgent, Claim, Evidence, FactCheckResult

__version__ = "1.0.0"
__author__ = "MCP Server Team"

__all__ = [
    'ClaimAnalyzerAgent',
    'Claim', 
    'Evidence',
    'FactCheckResult'
]

---
# scripts/install_claim_analyzer.py
#!/usr/bin/env python3
"""
Installation script for Claim Analyzer Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "spacy>=3.4.0",
        "sentence-transformers>=2.2.0", 
        "neo4j>=5.0.0",
        "qdrant-client>=1.6.0",
        "redis[hiredis]>=4.5.0",
        "aiohttp>=3.8.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "PyYAML>=6.0"
    ]
    
    print("📦 Installing Claim Analyzer dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def download_models():
    """Download required NLP models"""
    print("🧠 Downloading NLP models...")
    
    # Download spaCy model
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ Downloaded spaCy model")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False
    
    # Sentence transformer will download automatically on first use
    print("✅ Sentence transformer will download on first use")
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "agents/claim_analyzer",
        "agents/claim_analyzer/logs",
        "agents/claim_analyzer/cache",
        "agents/claim_analyzer/models",
        "data/claims",
        "data/evidence"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_config_file():
    """Create configuration file if it doesn't exist"""
    config_path = Path("agents/claim_analyzer/config.yaml")
    
    if not config_path.exists():
        print("⚙️ Creating default configuration file...")
        # The config content would be copied from the artifact above
        # For brevity, just creating a placeholder
        config_path.write_text("# Claim Analyzer Configuration\n# See full config in artifacts\n")
        print("✅ Created config file")
    else:
        print("⚙️ Configuration file already exists")
    
    return True

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import spacy
        import sentence_transformers
        import neo4j
        import qdrant_client
        import redis
        
        print("✅ All imports successful")
        
        # Test spaCy model
        nlp = spacy.load("en_core_web_sm")
        test_doc = nlp("This is a test sentence.")
        print("✅ spaCy model working")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🔍 Installing Claim Analyzer Agent for MCP Server")
    print("=" * 60)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Downloading models", download_models), 
        ("Setting up directories", setup_directories),
        ("Creating config file", create_config_file),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Installation failed at step: {step_name}")
            sys.exit(1)
    
    print("\n🎉 Claim Analyzer Agent installation completed successfully!")
    print("\n📖 Next steps:")
    print("1. Configure agents/claim_analyzer/config.yaml")
    print("2. Start the MCP server: make run-api")
    print("3. Access the claim analyzer via the API or dashboard")
    print("\n📚 Documentation: docs/agents/claim_analyzer.md")

if __name__ == "__main__":
    main()

---
# api/routes/claim_analyzer.py
"""
API routes for Claim Analyzer Agent
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio

from agents.claim_analyzer.claim_analyzer_agent import ClaimAnalyzerAgent

router = APIRouter(prefix="/claim-analyzer", tags=["claim-analyzer"])

# Global agent instance
claim_analyzer: Optional[ClaimAnalyzerAgent] = None

class ClaimAnalysisRequest(BaseModel):
    text: str
    source: str = "api"
    domain: str = "general"

class SingleClaimRequest(BaseModel):
    claim: str
    source: str = "api"
    domain: str = "general"

class SimilarClaimsRequest(BaseModel):
    claim: str
    limit: int = 5

async def get_claim_analyzer():
    """Dependency to get claim analyzer instance"""
    global claim_analyzer
    if claim_analyzer is None:
        claim_analyzer = ClaimAnalyzerAgent()
        await claim_analyzer.initialize()
    return claim_analyzer

@router.post("/analyze-text")
async def analyze_text(
    request: ClaimAnalysisRequest,
    analyzer: ClaimAnalyzerAgent = Depends(get_claim_analyzer)
):
    """Analyze text to extract and fact-check claims"""
    try:
        result = await analyzer.process_text(
            text=request.text,
            source=request.source,
            domain=request.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fact-check")
async def fact_check_claim(
    request: SingleClaimRequest,
    analyzer: ClaimAnalyzerAgent = Depends(get_claim_analyzer)
):
    """Fact-check a single claim"""
    try:
        result = await analyzer.fact_check_single_claim(
            claim_text=request.claim,
            source=request.source,
            domain=request.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar-claims")
async def get_similar_claims(
    request: SimilarClaimsRequest,
    analyzer: ClaimAnalyzerAgent = Depends(get_claim_analyzer)
):
    """Find similar claims in the database"""
    try:
        result = await analyzer.get_similar_claims(
            claim_text=request.claim,
            limit=request.limit
        )
        return {"similar_claims": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_agent_stats(
    analyzer: ClaimAnalyzerAgent = Depends(get_claim_analyzer)
):
    """Get agent statistics"""
    try:
        stats = await analyzer.get_agent_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    analyzer: ClaimAnalyzerAgent = Depends(get_claim_analyzer)
):
    """Health check endpoint"""
    try:
        database_status = await analyzer._check_database_status()
        
        overall_health = all(database_status.values())
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "database_connections": database_status,
            "agent_running": analyzer.is_running
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

---
# dashboard/pages/claim_analyzer.py
"""
Streamlit dashboard page for Claim Analyzer Agent
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Claim Analyzer",
    page_icon="🔍",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

def main():
    st.title("🔍 Claim Analyzer Agent")
    st.markdown("Analyze text for claims and perform automated fact-checking")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Text Analysis", "Single Fact-Check", "Similar Claims", "Agent Stats", "Health Monitor"]
        )
    
    if page == "Text Analysis":
        text_analysis_page()
    elif page == "Single Fact-Check":
        single_fact_check_page()
    elif page == "Similar Claims":
        similar_claims_page()
    elif page == "Agent Stats":
        agent_stats_page()
    elif page == "Health Monitor":
        health_monitor_page()

def text_analysis_page():
    st.header("📝 Text Analysis")
    st.markdown("Upload or paste text to extract and fact-check claims")
    
    # Input options
    input_method = st.radio("Input Method", ["Paste Text", "Upload File"])
    
    text_content = ""
    source = "dashboard"
    domain = "general"
    
    if input_method == "Paste Text":
        text_content = st.text_area(
            "Enter text to analyze",
            height=200,
            placeholder="Paste your text here..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source", value="dashboard")
        with col2:
            domain = st.selectbox(
                "Domain",
                ["general", "science", "math", "religion", "history", "literature", "philosophy"]
            )
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Upload text file",
            type=['txt', 'md', 'csv']
        )
        
        if uploaded_file:
            text_content = uploaded_file.read().decode('utf-8')
            source = uploaded_file.name
            domain = st.selectbox(
                "Domain",
                ["general", "science", "math", "religion", "history", "literature", "philosophy"]
            )
    
    if st.button("Analyze Text", type="primary") and text_content.strip():
        with st.spinner("Analyzing text and fact-checking claims..."):
            try:
                response = requests.post(
                    f"{API_BASE}/claim-analyzer/analyze-text",
                    json={
                        "text": text_content,
                        "source": source,
                        "domain": domain
                    }
                )
                
                if response.status_code == 200:
                    results = response.json()
                    display_analysis_results(results)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

def display_analysis_results(results):
    """Display text analysis results"""
    st.success(f"Found {results['total_claims']} claims")
    
    if results['fact_check_results']:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        verdicts = [r['verdict'] for r in results['fact_check_results']]
        confidences = [r['confidence'] for r in results['fact_check_results']]
        
        with col1:
            st.metric("Total Claims", len(verdicts))
        with col2:
            st.metric("Avg Confidence", f"{sum(confidences)/len(confidences):.2f}")
        with col3:
            true_claims = sum(1 for v in verdicts if v == "True")
            st.metric("True Claims", true_claims)
        with col4:
            false_claims = sum(1 for v in verdicts if v == "False")
            st.metric("False Claims", false_claims)
        
        # Verdict distribution chart
        verdict_counts = pd.Series(verdicts).value_counts()
        fig = px.pie(
            values=verdict_counts.values,
            names=verdict_counts.index,
            title="Fact-Check Verdict Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.header("Detailed Results")
        
        for i, result in enumerate(results['fact_check_results'], 1):
            with st.expander(f"Claim {i}: {result['claim']['text'][:80]}..."):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Full Claim:** {result['claim']['text']}")
                    st.markdown(f"**Domain:** {result['claim']['domain']}")
                    st.markdown(f"**Source:** {result['claim']['source']}")
                
                with col2:
                    # Color-coded verdict
                    verdict_color = {
                        "True": "🟢",
                        "False": "🔴", 
                        "Partially True": "🟡",
                        "Unverified": "⚪",
                        "Opinion": "🔵"
                    }
                    
                    st.markdown(f"**Verdict:** {verdict_color.get(result['verdict'], '⚪')} {result['verdict']}")
                    st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                
                st.markdown(f"**Reasoning:** {result['reasoning']}")
                
                if result['evidence_list']:
                    st.markdown("**Evidence Sources:**")
                    for evidence in result['evidence_list'][:5]:  # Show max 5 sources
                        st.markdown(f"- [{evidence['source_url']}]({evidence['source_url']}) (credibility: {evidence['credibility_score']:.2f})")
                
                if result['cross_domain_patterns']:
                    st.markdown("**Cross-Domain Patterns:**")
                    for pattern in result['cross_domain_patterns']:
                        st.markdown(f"- {pattern}")

def single_fact_check_page():
    st.header("🔎 Single Fact-Check")
    st.markdown("Fact-check a specific claim")
    
    claim_text = st.text_area(
        "Enter claim to fact-check",
        height=100,
        placeholder="Enter a specific claim to verify..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        source = st.text_input("Source", value="manual_check")
    with col2:
        domain = st.selectbox(
            "Domain",
            ["general", "science", "math", "religion", "history", "literature", "philosophy"]
        )
    
    if st.button("Fact-Check Claim", type="primary") and claim_text.strip():
        with st.spinner("Fact-checking claim..."):
            try:
                response = requests.post(
                    f"{API_BASE}/claim-analyzer/fact-check",
                    json={
                        "claim": claim_text,
                        "source": source,
                        "domain": domain
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    display_single_result(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

def display_single_result(result):
    """Display single fact-check result"""
    # Main result display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Claim:** {result['claim']['text']}")
    
    with col2:
        verdict_color = {
            "True": "🟢",
            "False": "🔴",
            "Partially True": "🟡", 
            "Unverified": "⚪",
            "Opinion": "🔵"
        }
        
        st.markdown(f"### {verdict_color.get(result['verdict'], '⚪')} {result['verdict']}")
        st.markdown(f"**Confidence:** {result['confidence']:.2f}")
    
    # Confidence meter
    confidence_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result['confidence'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    confidence_fig.update_layout(height=300)
    st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Detailed information
    st.markdown("**Reasoning:**")
    st.markdown(result['reasoning'])
    
    if result['evidence_list']:
        st.markdown("**Evidence Sources:**")
        evidence_df = pd.DataFrame([{
            'Source': ev['source_url'],
            'Credibility': ev['credibility_score'],
            'Stance': ev['stance'],
            'Domain': ev['domain']
        } for ev in result['evidence_list']])
        
        st.dataframe(evidence_df, use_container_width=True)
    
    if result['cross_domain_patterns']:
        st.markdown("**Cross-Domain Patterns:**")
        for pattern in result['cross_domain_patterns']:
            st.markdown(f"- {pattern}")

def similar_claims_page():
    st.header("🔍 Similar Claims Search")
    st.markdown("Find similar claims in the database")
    
    search_claim = st.text_area(
        "Enter claim to find similar ones",
        height=100,
        placeholder="Enter a claim to search for similar ones..."
    )
    
    limit = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if st.button("Search Similar Claims", type="primary") and search_claim.strip():
        with st.spinner("Searching for similar claims..."):
            try:
                response = requests.post(
                    f"{API_BASE}/claim-analyzer/similar-claims",
                    json={
                        "claim": search_claim,
                        "limit": limit
                    }
                )
                
                if response.status_code == 200:
                    results = response.json()
                    display_similar_claims(results['similar_claims'])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

def display_similar_claims(similar_claims):
    """Display similar claims results"""
    if not similar_claims:
        st.info("No similar claims found")
        return
    
    st.success(f"Found {len(similar_claims)} similar claims")
    
    # Create DataFrame for better display
    df = pd.DataFrame(similar_claims)
    
    # Similarity score distribution
    fig = px.histogram(
        df,
        x='similarity_score',
        nbins=10,
        title="Similarity Score Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display claims
    for i, claim in enumerate(similar_claims, 1):
        with st.expander(f"Similar Claim {i} (similarity: {claim['similarity_score']:.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Text:** {claim['text']}")
                st.markdown(f"**Source:** {claim['source']}")
            
            with col2:
                st.markdown(f"**Domain:** {claim['domain']}")
                st.markdown(f"**Similarity:** {claim['similarity_score']:.3f}")
                
            if claim.get('timestamp'):
                st.markdown(f"**Created:** {claim['timestamp']}")

def agent_stats_page():
    st.header("📊 Agent Statistics")
    st.markdown("Monitor agent performance and activity")
    
    if st.button("Refresh Stats"):
        try:
            response = requests.get(f"{API_BASE}/claim-analyzer/stats")
            
            if response.status_code == 200:
                stats = response.json()
                display_agent_stats(stats)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

def display_agent_stats(stats):
    """Display agent statistics"""
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Claims Processed", stats['processed_claims'])
    with col2:
        st.metric("Fact-Checks Performed", stats['fact_checks_performed'])
    with col3:
        status_icon = "🟢" if stats['is_running'] else "🔴"
        st.metric("Agent Status", f"{status_icon} {'Running' if stats['is_running'] else 'Stopped'}")
    with col4:
        db_status = stats['database_status']
        healthy_dbs = sum(db_status.values())
        st.metric("Database Health", f"{healthy_dbs}/{len(db_status)} healthy")
    
    # Database status details
    st.subheader("Database Connections")
    db_df = pd.DataFrame([
        {"Database": db, "Status": "🟢 Connected" if status else "🔴 Disconnected"}
        for db, status in stats['database_status'].items()
    ])
    st.dataframe(db_df, use_container_width=True)

def health_monitor_page():
    st.header("🏥 Health Monitor")
    st.markdown("Real-time health monitoring")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)")
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.experimental_rerun()
    
    if st.button("Check Health Now"):
        try:
            response = requests.get(f"{API_BASE}/claim-analyzer/health")
            
            if response.status_code == 200:
                health = response.json()
                display_health_status(health)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

def display_health_status(health):
    """Display health status"""
    # Overall status
    status_color = {
        "healthy": "🟢",
        "unhealthy": "🔴", 
        "error": "⚠️"
    }
    
    st.markdown(f"## {status_color.get(health['status'], '⚠️')} Overall Status: {health['status'].upper()}")
    
    # Database connections
    st.subheader("Database Connections")
    
    db_cols = st.columns(len(health['database_connections']))
    for i, (db, status) in enumerate(health['database_connections'].items()):
        with db_cols[i]:
            icon = "🟢" if status else "🔴"
            st.metric(db.title(), f"{icon} {'Connected' if status else 'Disconnected'}")
    
    # Agent status
    st.subheader("Agent Status")
    agent_icon = "🟢" if health['agent_running'] else "🔴"
    st.markdown(f"**Agent Running:** {agent_icon} {'Yes' if health['agent_running'] else 'No'}")
    
    if health.get('error'):
        st.error(f"Error: {health['error']}")

if __name__ == "__main__":
    main()