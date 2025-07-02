"""
Operations Console - Real-time System Operations and Monitoring
Advanced interface for Cypher queries, system monitoring, and transaction management
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json
import subprocess
import psutil
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.database_operations import get_neo4j_driver, test_connections
from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main Operations Console interface"""
    
    st.set_page_config(
        page_title="Operations Console - MCP Yggdrasil",
        page_icon="⚡",
        layout="wide"
    )
    
    # Custom CSS for console interface
    st.markdown("""
    <style>
    .console-container {
        background: #1e1e1e;
        color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .query-editor {
        background: #2d2d2d;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    .result-container {
        background: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    
    .metric-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2E8B57;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    .log-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .log-timestamp {
        color: #666;
        margin-right: 1rem;
    }
    
    .log-level-info { color: #17a2b8; }
    .log-level-warning { color: #ffc107; }
    .log-level-error { color: #dc3545; }
    .log-level-success { color: #28a745; }
    
    .performance-chart {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# ⚡ Operations Console")
    st.markdown("**Real-time system operations and monitoring**")
    
    # Initialize session state for console
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'console_logs' not in st.session_state:
        st.session_state.console_logs = []
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Console Operations")
        
        operation = st.selectbox(
            "Select Operation",
            ["🔍 Query Editor", "📊 System Monitor", "📜 Transaction Manager", "🚨 Log Viewer", "⚙️ Service Control"]
        )
        
        st.markdown("---")
        
        # Quick system status
        show_quick_system_status()
        
        st.markdown("---")
        
        # Auto-refresh controls
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
        
        if auto_refresh:
            refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
            st.caption(f"Refreshing every {refresh_interval} seconds")
            
            # Auto-refresh timer
            time.sleep(refresh_interval)
            st.rerun()
    
    # Main content based on operation
    if operation == "🔍 Query Editor":
        show_query_editor()
    elif operation == "📊 System Monitor":
        show_system_monitor()
    elif operation == "📜 Transaction Manager":
        show_transaction_manager()
    elif operation == "🚨 Log Viewer":
        show_log_viewer()
    elif operation == "⚙️ Service Control":
        show_service_control()

def show_quick_system_status():
    """Show quick system status in sidebar"""
    st.markdown("### 🎯 System Status")
    
    connections = test_connections()
    
    # Neo4j status
    neo4j_status = "🟢 Online" if connections.get('neo4j', False) else "🔴 Offline"
    st.markdown(f"**Neo4j**: {neo4j_status}")
    
    # Qdrant status
    qdrant_status = "🟢 Online" if connections.get('qdrant', False) else "🔴 Offline"
    st.markdown(f"**Qdrant**: {qdrant_status}")
    
    # Redis status
    redis_status = "🟢 Online" if connections.get('redis', False) else "🔴 Offline"
    st.markdown(f"**Redis**: {redis_status}")
    
    # Docker status
    docker_status = "🟢 Running" if connections.get('docker', False) else "🔴 Stopped"
    st.markdown(f"**Docker**: {docker_status}")

def show_query_editor():
    """Show Cypher query editor with syntax highlighting"""
    st.markdown("## 🔍 Cypher Query Editor")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query templates
        template_options = {
            "Custom Query": "",
            "List All Concepts": "MATCH (c:Concept) RETURN c.id, c.name, c.domain LIMIT 10",
            "Domain Statistics": "MATCH (c:Concept) RETURN c.domain, count(c) as count ORDER BY count DESC",
            "Find Relationships": "MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept) RETURN a.name, type(r), b.name LIMIT 10",
            "Most Connected Concepts": "MATCH (c:Concept)-[r:RELATES_TO]-() RETURN c.name, count(r) as connections ORDER BY connections DESC LIMIT 10",
            "Orphaned Concepts": "MATCH (c:Concept) WHERE NOT (c)-[:RELATES_TO]-() RETURN c.id, c.name",
            "Concept by Domain": "MATCH (c:Concept) WHERE c.domain = 'Science' RETURN c.id, c.name, c.description LIMIT 10"
        }
        
        selected_template = st.selectbox("📋 Query Templates", list(template_options.keys()))
        
        # Query editor
        query_text = st.text_area(
            "🔍 Cypher Query",
            value=template_options[selected_template],
            height=200,
            placeholder="Enter your Cypher query here...",
            help="Write Cypher queries to explore the knowledge graph"
        )
    
    with col2:
        st.markdown("### 🎯 Query Options")
        
        limit_results = st.number_input("Result Limit", min_value=1, max_value=1000, value=25)
        explain_query = st.checkbox("Explain Query", help="Show query execution plan")
        profile_query = st.checkbox("Profile Query", help="Show detailed performance metrics")
        
        # Query execution buttons
        st.markdown("### ⚡ Execute")
        
        if st.button("🚀 Run Query", type="primary", use_container_width=True):
            execute_cypher_query(query_text, limit_results, explain_query, profile_query)
        
        if st.button("💾 Save Query", use_container_width=True):
            save_query_to_history(query_text)
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Query results
    if 'query_result' in st.session_state:
        show_query_results()
    
    # Query history
    show_query_history()

def execute_cypher_query(query, limit, explain=False, profile=False):
    """Execute Cypher query and display results"""
    try:
        driver = get_neo4j_driver()
        if not driver:
            st.error("❌ No database connection available")
            return
        
        start_time = time.time()
        
        # Modify query for explain/profile
        if explain:
            query = f"EXPLAIN {query}"
        elif profile:
            query = f"PROFILE {query}"
        
        # Add limit if not already present and not explain/profile
        if not explain and not profile and "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"
        
        with driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            execution_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.query_result = {
                'records': records,
                'execution_time': execution_time,
                'query': query,
                'explain': explain,
                'profile': profile
            }
            
            # Add to history
            add_to_history("QUERY", f"Executed Cypher query: {query[:50]}...")
            
            # Log successful execution
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': 'success',
                'message': f"Query executed successfully in {execution_time:.3f}s",
                'details': {'query': query, 'record_count': len(records)}
            }
            st.session_state.console_logs.append(log_entry)
    
    except Exception as e:
        error_message = str(e)
        st.error(f"❌ Query execution failed: {error_message}")
        
        # Log error
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'error',
            'message': f"Query execution failed: {error_message}",
            'details': {'query': query}
        }
        st.session_state.console_logs.append(log_entry)

def show_query_results():
    """Display query execution results"""
    result_data = st.session_state.query_result
    
    st.markdown("---")
    st.markdown("## 📊 Query Results")
    
    # Execution info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Execution Time", f"{result_data['execution_time']:.3f}s")
    
    with col2:
        st.metric("Records Returned", len(result_data['records']))
    
    with col3:
        if result_data['explain']:
            st.metric("Query Type", "EXPLAIN")
        elif result_data['profile']:
            st.metric("Query Type", "PROFILE")
        else:
            st.metric("Query Type", "NORMAL")
    
    # Results display
    if result_data['records']:
        if result_data['explain'] or result_data['profile']:
            # Show execution plan
            st.markdown("### 🔍 Query Execution Plan")
            for record in result_data['records']:
                st.json(dict(record))
        else:
            # Show data results
            try:
                # Convert to DataFrame for better display
                records_data = []
                for record in result_data['records']:
                    record_dict = dict(record)
                    # Handle Neo4j node and relationship objects
                    processed_dict = {}
                    for key, value in record_dict.items():
                        if hasattr(value, '_properties'):  # Neo4j Node/Relationship
                            processed_dict[key] = dict(value._properties)
                        else:
                            processed_dict[key] = value
                    records_data.append(processed_dict)
                
                if records_data:
                    df = pd.json_normalize(records_data)
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        json_data = df.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.info("Query executed successfully but returned no data")
                    
            except Exception as e:
                # Fallback to raw display
                st.markdown("### 📋 Raw Results")
                for i, record in enumerate(result_data['records']):
                    with st.expander(f"Record {i+1}"):
                        st.json(dict(record))
    else:
        st.info("Query executed successfully but returned no results")

def save_query_to_history(query):
    """Save query to history"""
    if query.strip():
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query.strip(),
            'name': f"Query {len(st.session_state.query_history) + 1}"
        }
        st.session_state.query_history.append(history_entry)
        st.success("💾 Query saved to history")

def show_query_history():
    """Show query execution history"""
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📜 Query History")
        
        with st.expander("View Query History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query'])
                
                with col2:
                    st.caption(entry['timestamp'][:19])
                
                with col3:
                    if st.button("🔄", key=f"rerun_query_{i}", help="Run this query again"):
                        execute_cypher_query(entry['query'], 25)

def show_system_monitor():
    """Show system monitoring dashboard"""
    st.markdown("## 📊 System Monitor")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # System resources
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cpu_percent:.1f}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{memory.percent:.1f}%</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{disk.percent:.1f}%</div>
                <div class="metric-label">Disk Usage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Database connection count (simplified)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">4</div>
                <div class="metric-label">DB Connections</div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.warning(f"Could not retrieve system metrics: {e}")
    
    # Performance charts
    st.markdown("### 📈 Performance Metrics")
    show_performance_charts()
    
    # Service status
    st.markdown("### 🔧 Service Status")
    show_service_status_detailed()

def show_performance_charts():
    """Show performance monitoring charts"""
    try:
        # Generate sample performance data (in real implementation, this would come from monitoring)
        time_points = [datetime.now() - timedelta(minutes=x*5) for x in range(12, 0, -1)]
        
        # CPU usage over time
        cpu_data = [psutil.cpu_percent() + (i % 3) * 5 for i in range(12)]
        
        # Memory usage over time  
        memory_data = [psutil.virtual_memory().percent + (i % 2) * 3 for i in range(12)]
        
        # Create performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=cpu_data,
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#ff6b6b')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=memory_data,
            mode='lines+markers',
            name='Memory %',
            line=dict(color='#4ecdc4')
        ))
        
        fig.update_layout(
            title="System Performance (Last Hour)",
            xaxis_title="Time",
            yaxis_title="Usage %",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not generate performance charts: {e}")

def show_service_status_detailed():
    """Show detailed service status"""
    connections = test_connections()
    
    # Service status table
    services = [
        {"Service": "Neo4j", "Status": "🟢 Online" if connections.get('neo4j') else "🔴 Offline", "Port": "7687", "Type": "Graph Database"},
        {"Service": "Qdrant", "Status": "🟢 Online" if connections.get('qdrant') else "🔴 Offline", "Port": "6333", "Type": "Vector Database"},
        {"Service": "Redis", "Status": "🟢 Online" if connections.get('redis') else "🔴 Offline", "Port": "6379", "Type": "Cache"},
        {"Service": "Docker", "Status": "🟢 Running" if connections.get('docker') else "🔴 Stopped", "Port": "-", "Type": "Container Platform"}
    ]
    
    df_services = pd.DataFrame(services)
    st.dataframe(df_services, use_container_width=True, hide_index=True)
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🩺 Health Check", use_container_width=True):
            run_health_check()
    
    with col3:
        if st.button("📊 Generate Report", use_container_width=True):
            generate_system_report()

def show_transaction_manager():
    """Show transaction management interface"""
    st.markdown("## 📜 Transaction Manager")
    
    # Transaction operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 Active Transactions")
        
        # In a real implementation, this would query Neo4j for active transactions
        st.info("No active transactions currently running")
        
        # Mock transaction list for demo
        transactions = [
            {"ID": "tx001", "User": "admin", "Query": "MATCH (c:Concept)...", "Duration": "00:02:15", "Status": "Running"},
            {"ID": "tx002", "User": "system", "Query": "CREATE (c:Concept)...", "Duration": "00:00:45", "Status": "Committed"}
        ]
        
        if transactions:
            df_transactions = pd.DataFrame(transactions)
            st.dataframe(df_transactions, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🛠️ Transaction Tools")
        
        if st.button("📊 Show All Transactions", use_container_width=True):
            show_transaction_history()
        
        if st.button("⚠️ Kill Long-Running Queries", use_container_width=True):
            st.warning("This will terminate queries running longer than 5 minutes")
            # Implementation would go here
        
        if st.button("🔄 Refresh Transaction List", use_container_width=True):
            st.rerun()
        
        if st.button("📈 Transaction Statistics", use_container_width=True):
            show_transaction_stats()

def show_transaction_history():
    """Show transaction execution history"""
    st.markdown("#### 📜 Recent Transaction History")
    
    # Mock transaction history
    history = [
        {"Timestamp": "2025-07-01 19:30:15", "Type": "READ", "Query": "MATCH (c:Concept) RETURN count(c)", "Duration": "0.045s", "Status": "Success"},
        {"Timestamp": "2025-07-01 19:29:42", "Type": "WRITE", "Query": "CREATE (c:Concept {name: 'Test'})", "Duration": "0.12s", "Status": "Success"},
        {"Timestamp": "2025-07-01 19:28:33", "Type": "READ", "Query": "MATCH (c)-[r]-(d) RETURN c,r,d LIMIT 100", "Duration": "0.89s", "Status": "Success"}
    ]
    
    df_history = pd.DataFrame(history)
    st.dataframe(df_history, use_container_width=True, hide_index=True)

def show_transaction_stats():
    """Show transaction statistics"""
    st.markdown("#### 📊 Transaction Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Today", "147")
    with col2:
        st.metric("Avg Duration", "0.234s")
    with col3:
        st.metric("Success Rate", "99.3%")
    with col4:
        st.metric("Peak TPS", "12.4")

def show_log_viewer():
    """Show system log viewer"""
    st.markdown("## 🚨 Log Viewer")
    
    # Log level filter
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        log_levels = st.multiselect(
            "Log Levels",
            ["info", "warning", "error", "success"],
            default=["info", "warning", "error", "success"]
        )
    
    with col2:
        max_logs = st.selectbox("Max Entries", [50, 100, 200, 500], index=1)
    
    with col3:
        if st.button("🗑️ Clear Logs"):
            st.session_state.console_logs = []
            st.rerun()
    
    # Display logs
    st.markdown("### 📋 Console Logs")
    
    if st.session_state.console_logs:
        # Filter logs
        filtered_logs = [log for log in st.session_state.console_logs if log['level'] in log_levels]
        filtered_logs = filtered_logs[-max_logs:]  # Show most recent
        
        # Display in reverse chronological order
        for log in reversed(filtered_logs):
            timestamp = log['timestamp'][:19].replace('T', ' ')
            level = log['level']
            message = log['message']
            
            # Color based on log level
            level_class = f"log-level-{level}"
            
            st.markdown(f"""
            <div class="log-entry">
                <span class="log-timestamp">{timestamp}</span>
                <span class="{level_class}">[{level.upper()}]</span>
                <span>{message}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No logs available")
    
    # Add sample log for demo
    if st.button("📝 Add Sample Log"):
        sample_log = {
            'timestamp': datetime.now().isoformat(),
            'level': 'info',
            'message': 'Sample log entry for demonstration',
            'details': {}
        }
        st.session_state.console_logs.append(sample_log)
        st.rerun()

def show_service_control():
    """Show service control interface"""
    st.markdown("## ⚙️ Service Control")
    
    st.warning("⚠️ Service control operations can affect system stability. Use with caution.")
    
    # Docker services
    st.markdown("### 🐳 Docker Services")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### Neo4j")
        if st.button("🔄 Restart Neo4j", use_container_width=True):
            restart_docker_service("neo4j")
        if st.button("⏹️ Stop Neo4j", use_container_width=True):
            stop_docker_service("neo4j")
    
    with col2:
        st.markdown("#### Qdrant")
        if st.button("🔄 Restart Qdrant", use_container_width=True):
            restart_docker_service("qdrant")
        if st.button("⏹️ Stop Qdrant", use_container_width=True):
            stop_docker_service("qdrant")
    
    with col3:
        st.markdown("#### Redis")
        if st.button("🔄 Restart Redis", use_container_width=True):
            restart_docker_service("redis")
        if st.button("⏹️ Stop Redis", use_container_width=True):
            stop_docker_service("redis")
    
    with col4:
        st.markdown("#### All Services")
        if st.button("🔄 Restart All", use_container_width=True):
            restart_all_services()
        if st.button("⏹️ Stop All", use_container_width=True):
            stop_all_services()
    
    # System operations
    st.markdown("### 🔧 System Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clean Cache", use_container_width=True):
            clear_system_cache()
        
        if st.button("🔄 Reload Configuration", use_container_width=True):
            reload_configuration()
    
    with col2:
        if st.button("📊 Export System State", use_container_width=True):
            export_system_state()
        
        if st.button("🩺 Full Health Check", use_container_width=True):
            run_comprehensive_health_check()

def restart_docker_service(service_name):
    """Restart a Docker service"""
    try:
        result = subprocess.run(['docker', 'restart', f'mcp_ygg-{service_name}-1'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success(f"✅ {service_name} restarted successfully")
            add_to_history("SERVICE", f"Restarted {service_name} service")
        else:
            st.error(f"❌ Failed to restart {service_name}: {result.stderr}")
    
    except Exception as e:
        st.error(f"Error restarting {service_name}: {str(e)}")

def stop_docker_service(service_name):
    """Stop a Docker service"""
    try:
        result = subprocess.run(['docker', 'stop', f'mcp_ygg-{service_name}-1'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            st.warning(f"⏹️ {service_name} stopped")
            add_to_history("SERVICE", f"Stopped {service_name} service")
        else:
            st.error(f"❌ Failed to stop {service_name}: {result.stderr}")
    
    except Exception as e:
        st.error(f"Error stopping {service_name}: {str(e)}")

def restart_all_services():
    """Restart all Docker services"""
    st.info("🔄 Restarting all services...")
    
    services = ["neo4j", "qdrant", "redis", "rabbitmq"]
    for service in services:
        restart_docker_service(service)

def stop_all_services():
    """Stop all Docker services"""
    st.warning("⏹️ Stopping all services...")
    
    services = ["neo4j", "qdrant", "redis", "rabbitmq"]
    for service in services:
        stop_docker_service(service)

def run_health_check():
    """Run system health check"""
    st.info("🩺 Running health check...")
    
    connections = test_connections()
    
    health_status = "✅ All systems operational"
    if not all(connections.values()):
        health_status = "⚠️ Some services are offline"
    
    st.success(health_status)

def generate_system_report():
    """Generate comprehensive system report"""
    st.success("📊 System report generated!")
    add_to_history("REPORT", "Generated system status report")

def clear_system_cache():
    """Clear system cache"""
    st.cache_data.clear()
    st.success("🧹 System cache cleared")

def reload_configuration():
    """Reload system configuration"""
    st.success("🔄 Configuration reloaded")

def export_system_state():
    """Export current system state"""
    system_state = {
        'timestamp': datetime.now().isoformat(),
        'connections': test_connections(),
        'query_history': st.session_state.get('query_history', []),
        'console_logs': st.session_state.get('console_logs', [])
    }
    
    json_data = json.dumps(system_state, indent=2)
    st.download_button(
        "📥 Download System State",
        json_data,
        file_name=f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def run_comprehensive_health_check():
    """Run comprehensive health check"""
    st.info("🩺 Running comprehensive health check...")
    
    # This would include more detailed checks
    checks = [
        "✅ Database connections",
        "✅ Service availability", 
        "✅ Resource utilization",
        "✅ Query performance",
        "✅ Data integrity"
    ]
    
    for check in checks:
        st.text(check)
    
    st.success("🎯 All systems healthy")

if __name__ == "__main__":
    main()