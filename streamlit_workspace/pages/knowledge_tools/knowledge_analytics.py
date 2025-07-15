"""
Knowledge Analytics Module
Advanced analytics and reporting for knowledge graphs

Provides comprehensive analytics functionality:
- Growth trends analysis
- Network analysis metrics
- Relationship pattern analysis
- Domain analysis and statistics

Extracted from knowledge_tools.py as part of modular refactoring.
Functions: show_knowledge_analytics, show_growth_trends, show_network_analysis,
show_relationship_patterns, show_domain_analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database_operations import get_domains

def show_knowledge_analytics():
    """Show knowledge analytics dashboard"""
    st.markdown("## 📊 Knowledge Analytics")
    
    analytics_type = st.selectbox(
        "Analytics Type",
        ["📈 Growth Trends", "🌐 Network Analysis", "🔗 Relationship Patterns", "📊 Domain Analysis"]
    )
    
    if analytics_type == "📈 Growth Trends":
        show_growth_trends()
    elif analytics_type == "🌐 Network Analysis":
        show_network_analysis()
    elif analytics_type == "🔗 Relationship Patterns":
        show_relationship_patterns()
    elif analytics_type == "📊 Domain Analysis":
        show_domain_analysis()

def show_growth_trends():
    """Show knowledge graph growth trends"""
    st.markdown("### 📈 Growth Trends")
    
    # Mock growth data
    dates = pd.date_range(start='2025-01-01', end='2025-07-01', freq='W')
    concepts_added = [5, 12, 8, 15, 20, 18, 25, 30, 22, 28, 35, 40, 45, 38, 42, 50, 55, 48, 52, 60, 58, 65, 70, 68, 72, 75]
    
    df_growth = pd.DataFrame({
        'Date': dates[:len(concepts_added)],
        'Concepts Added': concepts_added,
        'Cumulative': pd.Series(concepts_added).cumsum()
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_growth['Date'],
        y=df_growth['Concepts Added'],
        name='Weekly Additions',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_growth['Date'],
        y=df_growth['Cumulative'],
        mode='lines+markers',
        name='Cumulative Total',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Knowledge Graph Growth Over Time',
        xaxis_title='Date',
        yaxis=dict(title='Weekly Additions', side='left'),
        yaxis2=dict(title='Cumulative Total', side='right', overlaying='y'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth statistics
    st.markdown("#### Growth Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_added = sum(concepts_added)
        st.metric("Total Added", total_added)
    
    with col2:
        avg_weekly = total_added / len(concepts_added)
        st.metric("Avg Weekly", f"{avg_weekly:.1f}")
    
    with col3:
        peak_week = max(concepts_added)
        st.metric("Peak Week", peak_week)
    
    with col4:
        growth_rate = ((concepts_added[-1] - concepts_added[0]) / concepts_added[0] * 100) if concepts_added[0] > 0 else 0
        st.metric("Growth Rate", f"{growth_rate:.1f}%")

def show_network_analysis():
    """Show network analysis metrics"""
    st.markdown("### 🌐 Network Analysis")
    
    # Mock network metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Network Density", "0.045", delta="0.005")
    
    with col2:
        st.metric("Avg Path Length", "3.2", delta="-0.1")
    
    with col3:
        st.metric("Clustering Coefficient", "0.67", delta="0.02")
    
    with col4:
        st.metric("Connected Components", "1", delta="0")
    
    # Additional network metrics
    st.markdown("#### Detailed Network Metrics")
    
    # Create sample network metrics data
    metrics_data = {
        'Metric': [
            'Total Nodes', 'Total Edges', 'Average Degree', 'Graph Diameter',
            'Transitivity', 'Assortativity', 'Small World Coefficient'
        ],
        'Value': [145, 408, 5.6, 8, 0.67, -0.15, 1.23],
        'Interpretation': [
            'Number of concepts in the graph',
            'Number of relationships between concepts',
            'Average connections per concept',
            'Longest shortest path in the graph',
            'Probability that neighbors are connected',
            'Tendency for similar nodes to connect',
            'Graph exhibits small-world properties'
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)
    
    # Network topology visualization placeholder
    st.markdown("#### Network Topology")
    st.info("🔄 Interactive network visualization would be displayed here using NetworkX and Plotly")
    
    # Centrality analysis
    st.markdown("#### Centrality Analysis")
    
    # Mock centrality data
    centrality_data = {
        'Concept ID': ['SCI0001', 'PHIL0003', 'MATH0001', 'ART0015', 'TECH0002'],
        'Concept Name': ['Quantum Mechanics', 'Epistemology', 'Calculus', 'Renaissance Art', 'Computer Science'],
        'Betweenness': [0.45, 0.38, 0.31, 0.28, 0.25],
        'Closeness': [0.82, 0.75, 0.71, 0.68, 0.65],
        'PageRank': [0.12, 0.09, 0.08, 0.07, 0.06]
    }
    
    df_centrality = pd.DataFrame(centrality_data)
    st.dataframe(df_centrality, use_container_width=True)

def show_relationship_patterns():
    """Show relationship pattern analysis"""
    st.markdown("### 🔗 Relationship Patterns")
    
    # Mock relationship data
    relationship_types = ['BELONGS_TO', 'RELATES_TO', 'DERIVED_FROM', 'INFLUENCES', 'CONTAINS']
    counts = [45, 38, 22, 15, 12]
    
    # Relationship type distribution
    fig1 = px.bar(
        x=relationship_types,
        y=counts,
        title="Relationship Type Distribution",
        color=counts,
        color_continuous_scale='Viridis',
        labels={'x': 'Relationship Type', 'y': 'Count'}
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Relationship strength analysis
    st.markdown("#### Relationship Strength Analysis")
    
    # Mock strength distribution data
    strength_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    strength_counts = [8, 15, 32, 41, 36]
    
    fig2 = px.pie(
        values=strength_counts,
        names=strength_ranges,
        title="Relationship Strength Distribution"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Cross-domain relationships
    st.markdown("#### Cross-Domain Relationships")
    
    # Mock cross-domain data
    cross_domain_data = {
        'Source Domain': ['Science', 'Philosophy', 'Mathematics', 'Art', 'Technology'],
        'Target Domain': ['Philosophy', 'Science', 'Science', 'Philosophy', 'Science'],
        'Relationship Count': [12, 8, 15, 5, 9],
        'Avg Strength': [0.75, 0.68, 0.82, 0.55, 0.71]
    }
    
    df_cross_domain = pd.DataFrame(cross_domain_data)
    st.dataframe(df_cross_domain, use_container_width=True)
    
    # Relationship patterns over time
    st.markdown("#### Relationship Creation Trends")
    
    # Mock temporal data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
    relationships_created = [25, 32, 28, 35, 42, 38, 45]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=months,
        y=relationships_created,
        mode='lines+markers',
        name='Relationships Created',
        line=dict(color='#2E8B57', width=3)
    ))
    
    fig3.update_layout(
        title='Monthly Relationship Creation',
        xaxis_title='Month',
        yaxis_title='Relationships Created'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

def show_domain_analysis():
    """Show detailed domain analysis"""
    st.markdown("### 📊 Domain Analysis")
    
    domains = get_domains()
    
    if domains:
        df_domains = pd.DataFrame(domains)
        
        # Domain comparison chart
        fig1 = px.bar(
            df_domains, 
            x='domain', 
            y='concept_count',
            title="Concepts per Domain",
            color='concept_count',
            color_continuous_scale='Blues',
            labels={'domain': 'Domain', 'concept_count': 'Number of Concepts'}
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Domain statistics table
        st.markdown("#### Domain Statistics")
        
        # Calculate additional statistics
        total_concepts = df_domains['concept_count'].sum()
        df_domains['Percentage'] = (df_domains['concept_count'] / total_concepts * 100).round(2)
        df_domains['Coverage Rating'] = df_domains['concept_count'].apply(
            lambda x: 'High' if x > total_concepts * 0.15 
                     else 'Medium' if x > total_concepts * 0.08 
                     else 'Low'
        )
        
        st.dataframe(df_domains, use_container_width=True)
        
        # Domain maturity analysis
        st.markdown("#### Domain Maturity Analysis")
        
        # Mock maturity data
        maturity_data = {
            'Domain': [d['domain'] for d in domains],
            'Concepts': [d['concept_count'] for d in domains],
            'Avg Depth': [3.2, 2.8, 4.1, 3.5, 2.9, 3.7, 2.4, 2.6],  # Mock data
            'Interconnectedness': [0.65, 0.72, 0.58, 0.68, 0.61, 0.75, 0.52, 0.48],  # Mock data
            'Maturity Score': [7.5, 8.2, 7.8, 8.0, 7.1, 8.5, 6.8, 6.5]  # Mock data
        }
        
        df_maturity = pd.DataFrame(maturity_data)
        
        # Maturity score visualization
        fig2 = px.scatter(
            df_maturity,
            x='Concepts',
            y='Interconnectedness',
            size='Maturity Score',
            color='Avg Depth',
            hover_name='Domain',
            title='Domain Maturity Analysis',
            labels={
                'Concepts': 'Number of Concepts',
                'Interconnectedness': 'Interconnectedness Score',
                'Avg Depth': 'Average Depth'
            }
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Domain health indicators
        st.markdown("#### Domain Health Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_concepts = df_domains['concept_count'].mean()
            st.metric("Avg Concepts/Domain", f"{avg_concepts:.1f}")
        
        with col2:
            balanced_domains = len(df_domains[df_domains['Coverage Rating'] == 'Medium'])
            st.metric("Balanced Domains", f"{balanced_domains}/{len(domains)}")
        
        with col3:
            total_coverage = len(df_domains[df_domains['concept_count'] > 0])
            st.metric("Active Domains", f"{total_coverage}/{len(domains)}")
        
        # Recommendations
        st.markdown("#### Recommendations")
        
        underrepresented = df_domains[df_domains['Coverage Rating'] == 'Low']['domain'].tolist()
        if underrepresented:
            st.warning(f"⚠️ Consider expanding these domains: {', '.join(underrepresented)}")
        
        overrepresented = df_domains[df_domains['Coverage Rating'] == 'High']['domain'].tolist()
        if overrepresented:
            st.info(f"ℹ️ These domains are well-developed: {', '.join(overrepresented)}")
        
    else:
        st.error("Could not load domain data for analysis")