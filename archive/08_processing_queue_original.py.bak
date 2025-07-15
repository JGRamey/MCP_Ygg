"""
Processing Queue - MCP Yggdrasil IDE Workspace
Analysis queue management and monitoring dashboard
"""

import streamlit as st
import asyncio
import json
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from data.staging_manager import (
        StagingManager, ProcessingStatus, SourceType, Priority,
        StagedContent, AnalysisResults, ReviewData
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

class ProcessingQueueManager:
    """Manages the processing queue interface"""
    
    def __init__(self):
        self.staging_manager = StagingManager()
    
    async def get_queue_data(self):
        """Get current queue data"""
        return await self.staging_manager.get_queue_stats()
    
    async def get_items_by_status(self, status: ProcessingStatus, limit: int = 50):
        """Get items by processing status"""
        return await self.staging_manager.list_content(status=status, limit=limit)
    
    async def approve_item(self, submission_id: str, reviewer: str, reason: str):
        """Approve an item"""
        return await self.staging_manager.approve_content(submission_id, reviewer, reason)
    
    async def reject_item(self, submission_id: str, reviewer: str, reason: str):
        """Reject an item"""
        return await self.staging_manager.reject_content(submission_id, reviewer, reason)
    
    async def export_item(self, submission_id: str, format: str = "json"):
        """Export an item"""
        return await self.staging_manager.export_content(submission_id, format)

def display_queue_overview():
    """Display queue overview with real-time statistics"""
    st.header("🔄 Processing Queue Overview")
    
    # Initialize session state
    if 'queue_manager' not in st.session_state:
        st.session_state.queue_manager = ProcessingQueueManager()
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Real-time Queue Statistics")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Get queue statistics
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue_stats = loop.run_until_complete(
            st.session_state.queue_manager.get_queue_data()
        )
        loop.close()
        
        if queue_stats:
            # Status metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "📥 Pending", 
                    queue_stats['queue_counts'].get('pending', 0),
                    help="Items waiting for processing"
                )
            
            with col2:
                st.metric(
                    "⚙️ Processing", 
                    queue_stats['queue_counts'].get('processing', 0),
                    help="Items currently being analyzed"
                )
            
            with col3:
                st.metric(
                    "🔍 Analyzed", 
                    queue_stats['queue_counts'].get('analyzed', 0),
                    help="Items completed analysis, awaiting review"
                )
            
            with col4:
                st.metric(
                    "✅ Approved", 
                    queue_stats['queue_counts'].get('approved', 0),
                    help="Items approved for database integration"
                )
            
            with col5:
                st.metric(
                    "❌ Rejected", 
                    queue_stats['queue_counts'].get('rejected', 0),
                    help="Items rejected during review"
                )
            
            # Queue distribution chart
            if queue_stats['total_items'] > 0:
                st.subheader("📊 Queue Distribution")
                
                # Prepare data for pie chart
                labels = []
                values = []
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffa726']
                
                for i, (status, count) in enumerate(queue_stats['queue_counts'].items()):
                    if count > 0:
                        labels.append(status.title())
                        values.append(count)
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values,
                    hole=0.3,
                    marker_colors=colors[:len(labels)]
                )])
                
                fig.update_layout(
                    title="Processing Queue Distribution",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Total Items", 
                    queue_stats['total_items'],
                    help="Total items in all queues"
                )
            
            with col2:
                if queue_stats.get('average_processing_time_minutes'):
                    avg_time = queue_stats['average_processing_time_minutes']
                    st.metric(
                        "⏱️ Avg Processing Time", 
                        f"{avg_time:.1f} min",
                        help="Average time from submission to analysis completion"
                    )
                else:
                    st.metric("⏱️ Avg Processing Time", "N/A")
            
            with col3:
                if queue_stats.get('oldest_pending'):
                    st.metric(
                        "⏳ Oldest Pending", 
                        queue_stats['oldest_pending'][:8] + "...",
                        help="ID of oldest pending item"
                    )
                else:
                    st.metric("⏳ Oldest Pending", "None")
        
        else:
            st.warning("Unable to retrieve queue statistics")
    
    except Exception as e:
        st.error(f"Error loading queue overview: {e}")

def display_queue_management():
    """Display queue management interface"""
    st.header("📋 Queue Management")
    
    # Status filter tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Pending", "⚙️ Processing", "🔍 Analyzed", "✅ Approved", "❌ Rejected"
    ])
    
    status_tabs = {
        tab1: ProcessingStatus.PENDING,
        tab2: ProcessingStatus.PROCESSING,
        tab3: ProcessingStatus.ANALYZED,
        tab4: ProcessingStatus.APPROVED,
        tab5: ProcessingStatus.REJECTED
    }
    
    for tab, status in status_tabs.items():
        with tab:
            display_items_by_status(status)

def display_items_by_status(status: ProcessingStatus):
    """Display items for a specific status"""
    try:
        # Get items
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        items = loop.run_until_complete(
            st.session_state.queue_manager.get_items_by_status(status, limit=100)
        )
        loop.close()
        
        if not items:
            st.info(f"No items in {status.value} status")
            return
        
        # Display controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader(f"{status.value.title()} Items ({len(items)})")
        with col2:
            limit = st.selectbox("Show items:", [10, 25, 50, 100], index=1, key=f"limit_{status.value}")
        with col3:
            if st.button("🔄 Refresh", key=f"refresh_{status.value}"):
                st.rerun()
        
        # Display items
        for i, item in enumerate(items[:limit]):
            display_item_card(item, status)
    
    except Exception as e:
        st.error(f"Error loading {status.value} items: {e}")

def display_item_card(item: StagedContent, status: ProcessingStatus):
    """Display individual item card"""
    with st.expander(f"📄 {item.metadata.title[:60]}... - {item.submission_id[:8]}"):
        
        # Item details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Source Type:** {item.source_type.value}")
            st.write(f"**Domain:** {item.metadata.domain}")
            st.write(f"**Priority:** {item.metadata.priority.value}")
            if item.metadata.author:
                st.write(f"**Author:** {item.metadata.author}")
            
            # Timestamps
            st.write("**Timeline:**")
            for key, timestamp in item.timestamps.items():
                if timestamp:
                    formatted_time = datetime.fromisoformat(timestamp.replace("Z", "")).strftime("%Y-%m-%d %H:%M")
                    st.write(f"- {key.replace('_', ' ').title()}: {formatted_time}")
        
        with col2:
            # Action buttons based on status
            if status == ProcessingStatus.ANALYZED:
                st.subheader("📋 Review Actions")
                
                reviewer = st.text_input("Reviewer:", key=f"reviewer_{item.submission_id}")
                
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    approval_reason = st.text_area(
                        "Approval reason:", 
                        height=60,
                        key=f"approve_reason_{item.submission_id}"
                    )
                    if st.button("✅ Approve", key=f"approve_{item.submission_id}"):
                        if reviewer:
                            approve_item(item.submission_id, reviewer, approval_reason)
                        else:
                            st.error("Please enter reviewer name")
                
                with col_reject:
                    rejection_reason = st.text_area(
                        "Rejection reason:", 
                        height=60,
                        key=f"reject_reason_{item.submission_id}"
                    )
                    if st.button("❌ Reject", key=f"reject_{item.submission_id}"):
                        if reviewer and rejection_reason:
                            reject_item(item.submission_id, reviewer, rejection_reason)
                        else:
                            st.error("Please enter reviewer name and rejection reason")
            
            # Export options
            st.subheader("📤 Export")
            export_format = st.selectbox(
                "Format:", 
                ["json", "text"], 
                key=f"export_format_{item.submission_id}"
            )
            if st.button("📥 Export", key=f"export_{item.submission_id}"):
                export_item(item.submission_id, export_format)
        
        # Content preview
        if st.checkbox("👁️ Preview Content", key=f"preview_{item.submission_id}"):
            content_preview = item.raw_content[:1000]
            if len(item.raw_content) > 1000:
                content_preview += "..."
            st.text_area("Content Preview:", content_preview, height=200, disabled=True)
        
        # Analysis results
        if item.analysis_results and st.checkbox("🔍 Analysis Results", key=f"analysis_{item.submission_id}"):
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if item.analysis_results.concepts_extracted:
                    st.write("**Concepts Extracted:**")
                    for concept in item.analysis_results.concepts_extracted:
                        st.write(f"- {concept}")
                
                if item.analysis_results.claims_identified:
                    st.write("**Claims Identified:**")
                    for claim in item.analysis_results.claims_identified:
                        st.write(f"- {claim}")
            
            with col2:
                st.metric("Quality Score", f"{item.analysis_results.quality_score:.2f}")
                st.metric("Confidence Level", item.analysis_results.confidence_level)
                
                if item.analysis_results.connections_discovered:
                    st.write("**Connections Discovered:**")
                    for connection in item.analysis_results.connections_discovered:
                        st.json(connection)

def approve_item(submission_id: str, reviewer: str, reason: str):
    """Approve an item"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(
            st.session_state.queue_manager.approve_item(submission_id, reviewer, reason)
        )
        loop.close()
        
        if success:
            st.success(f"Item {submission_id[:8]} approved successfully!")
            st.rerun()
        else:
            st.error("Failed to approve item")
    
    except Exception as e:
        st.error(f"Error approving item: {e}")

def reject_item(submission_id: str, reviewer: str, reason: str):
    """Reject an item"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(
            st.session_state.queue_manager.reject_item(submission_id, reviewer, reason)
        )
        loop.close()
        
        if success:
            st.success(f"Item {submission_id[:8]} rejected successfully!")
            st.rerun()
        else:
            st.error("Failed to reject item")
    
    except Exception as e:
        st.error(f"Error rejecting item: {e}")

def export_item(submission_id: str, format: str):
    """Export an item"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exported_data = loop.run_until_complete(
            st.session_state.queue_manager.export_item(submission_id, format)
        )
        loop.close()
        
        if exported_data:
            # Offer download
            filename = f"{submission_id}.{format}"
            st.download_button(
                label=f"💾 Download {format.upper()}",
                data=exported_data,
                file_name=filename,
                mime="application/json" if format == "json" else "text/plain"
            )
        else:
            st.error("Failed to export item")
    
    except Exception as e:
        st.error(f"Error exporting item: {e}")

def display_analytics():
    """Display processing analytics"""
    st.header("📈 Processing Analytics")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get data for all statuses
        analytics_data = {}
        for status in ProcessingStatus:
            items = loop.run_until_complete(
                st.session_state.queue_manager.get_items_by_status(status, limit=1000)
            )
            analytics_data[status.value] = items
        
        loop.close()
        
        # Processing timeline
        st.subheader("⏱️ Processing Timeline")
        
        timeline_data = []
        for status_name, items in analytics_data.items():
            for item in items:
                if item.timestamps.get("submitted"):
                    timeline_data.append({
                        "date": datetime.fromisoformat(item.timestamps["submitted"].replace("Z", "")),
                        "status": status_name,
                        "domain": item.metadata.domain,
                        "source_type": item.source_type.value
                    })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Daily submissions chart
            daily_counts = df.groupby([df['date'].dt.date, 'status']).size().reset_index(name='count')
            
            fig = px.bar(
                daily_counts, 
                x='date', 
                y='count', 
                color='status',
                title="Daily Processing Activity",
                labels={'date': 'Date', 'count': 'Number of Items'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Domain distribution
            col1, col2 = st.columns(2)
            
            with col1:
                domain_counts = df['domain'].value_counts()
                fig_domain = px.pie(
                    values=domain_counts.values,
                    names=domain_counts.index,
                    title="Content by Domain"
                )
                st.plotly_chart(fig_domain, use_container_width=True)
            
            with col2:
                source_counts = df['source_type'].value_counts()
                fig_source = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    title="Content by Source Type"
                )
                st.plotly_chart(fig_source, use_container_width=True)
        
        else:
            st.info("No timeline data available")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def display_batch_operations():
    """Display batch operations interface"""
    st.header("🔄 Batch Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧹 Cleanup Operations")
        
        cleanup_days = st.number_input(
            "Delete approved/rejected items older than (days):",
            min_value=1,
            max_value=365,
            value=30
        )
        
        if st.button("🗑️ Cleanup Old Items"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cleaned_count = loop.run_until_complete(
                    st.session_state.queue_manager.staging_manager.cleanup_old_items(cleanup_days)
                )
                loop.close()
                
                st.success(f"Cleaned up {cleaned_count} old items")
            
            except Exception as e:
                st.error(f"Error during cleanup: {e}")
    
    with col2:
        st.subheader("📊 Export Operations")
        
        export_status = st.selectbox(
            "Export items by status:",
            ["pending", "processing", "analyzed", "approved", "rejected"]
        )
        
        export_format_batch = st.selectbox(
            "Export format:",
            ["json", "csv", "text"]
        )
        
        if st.button("📤 Export All"):
            st.info("Batch export functionality coming soon...")

def main():
    """Main processing queue interface"""
    
    st.set_page_config(
        page_title="Processing Queue",
        page_icon="🔄",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .status-pending { border-left-color: #ff6b6b; }
    .status-processing { border-left-color: #4ecdc4; }
    .status-analyzed { border-left-color: #45b7d1; }
    .status-approved { border-left-color: #96ceb4; }
    .status-rejected { border-left-color: #ffa726; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔄 Processing Queue Management")
    st.markdown("Monitor and manage content processing workflows")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📋 Queue Management", "📈 Analytics", "🔄 Batch Operations"
    ])
    
    with tab1:
        display_queue_overview()
    
    with tab2:
        display_queue_management()
    
    with tab3:
        display_analytics()
    
    with tab4:
        display_batch_operations()

if __name__ == "__main__":
    main()