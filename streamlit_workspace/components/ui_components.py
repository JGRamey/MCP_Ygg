"""
UI Components for Streamlit Dashboard

This module provides reusable UI components for the MCP Yggdrasil dashboard,
including header, sidebar, forms, and common UI elements with consistent styling.

Key Features:
- Dashboard header with system status indicators
- Navigation sidebar with quick actions and metrics
- Reusable metric cards and data display components
- Custom CSS styling and theming
- Modal dialogs and interactive elements

Author: MCP Yggdrasil Analytics Team
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

from .config_management import get_dashboard_state

logger = logging.getLogger(__name__)


class UIComponents:
    """
    Reusable UI components for the Streamlit dashboard.
    
    Provides consistent styling, layout components, and interactive elements
    that can be used across different dashboard pages.
    """
    
    def __init__(self):
        """Initialize UI components."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dashboard_state = get_dashboard_state()
        
        # Apply custom styling
        self._apply_custom_styling()
    
    def _apply_custom_styling(self):
        """Apply custom CSS styling to the dashboard."""
        custom_css = """
        <style>
            .main-header {
                font-size: 3rem;
                color: #2E8B57;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 0.5rem 0;
            }
            .status-good { 
                color: #28a745; 
                font-weight: bold;
            }
            .status-warning { 
                color: #ffc107; 
                font-weight: bold;
            }
            .status-error { 
                color: #dc3545; 
                font-weight: bold;
            }
            .sidebar-section {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 4px solid #2E8B57;
            }
            .data-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .quick-action-button {
                width: 100%;
                margin: 0.25rem 0;
                border-radius: 8px;
                border: none;
                padding: 0.5rem;
                background: linear-gradient(45deg, #2E8B57, #3CB371);
                color: white;
                font-weight: bold;
            }
            .system-metric {
                background-color: #f0f8f0;
                padding: 0.75rem;
                border-radius: 6px;
                margin: 0.5rem 0;
                border-left: 3px solid #2E8B57;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy { background-color: #28a745; }
            .status-unhealthy { background-color: #dc3545; }
            .status-unknown { background-color: #6c757d; }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main dashboard header with system status."""
        try:
            # Main header
            st.markdown('<h1 class="main-header">🌳 MCP Server Dashboard</h1>', unsafe_allow_html=True)
            
            # System status bar
            col1, col2, col3, col4 = st.columns(4)
            
            # Get system status from session state or dashboard state
            system_status = getattr(st.session_state, 'system_status', {})
            databases = system_status.get('databases', {})
            
            with col1:
                neo4j_status = databases.get('neo4j', 'unknown')
                self._render_status_indicator("Neo4j", neo4j_status)
            
            with col2:
                qdrant_status = databases.get('qdrant', 'unknown')
                self._render_status_indicator("Qdrant", qdrant_status)
            
            with col3:
                redis_status = databases.get('redis', 'unknown')
                self._render_status_indicator("Redis", redis_status)
            
            with col4:
                # Refresh status button
                if st.button("🔄 Refresh Status", key="refresh_status"):
                    self._refresh_system_status()
                    st.rerun()
            
            # Last refresh info
            if hasattr(self.dashboard_state, 'last_refresh') and self.dashboard_state.last_refresh:
                time_diff = datetime.now() - self.dashboard_state.last_refresh
                st.caption(f"Last refresh: {time_diff.seconds}s ago")
            
        except Exception as e:
            self.logger.error(f"Error rendering header: {e}")
            st.error("Error rendering dashboard header")
    
    def _render_status_indicator(self, service_name: str, status: str):
        """Render a status indicator for a service."""
        status_colors = {
            'healthy': 'status-healthy',
            'active': 'status-healthy',
            'unhealthy': 'status-unhealthy',
            'failed': 'status-unhealthy',
            'unknown': 'status-unknown'
        }
        
        color_class = status_colors.get(status.lower(), 'status-unknown')
        status_text = status.title()
        
        st.markdown(
            f'<div><span class="status-indicator {color_class}"></span>'
            f'<span class="status-good" style="color: {"#28a745" if status.lower() in ["healthy", "active"] else "#dc3545" if status.lower() in ["unhealthy", "failed"] else "#6c757d"}">'
            f'{service_name}: {status_text}</span></div>',
            unsafe_allow_html=True
        )
    
    def _refresh_system_status(self):
        """Refresh system status information."""
        try:
            # Mock status check - would be real health checks in production
            st.session_state.system_status = {
                'databases': {
                    'neo4j': 'healthy',
                    'qdrant': 'healthy', 
                    'redis': 'healthy'
                },
                'agents': {
                    'scraper': 'active',
                    'processor': 'active',
                    'analyzer': 'active'
                },
                'last_check': datetime.now()
            }
            
            if hasattr(self.dashboard_state, 'last_refresh'):
                self.dashboard_state.last_refresh = datetime.now()
            
            st.success("System status refreshed!")
            
        except Exception as e:
            self.logger.error(f"Error refreshing system status: {e}")
            st.error("Failed to refresh system status")
    
    def render_sidebar(self) -> str:
        """
        Render the navigation sidebar with quick actions and metrics.
        
        Returns:
            Selected page name
        """
        try:
            st.sidebar.markdown("## 🧭 Navigation")
            
            # Main navigation
            pages = [
                "🏠 Overview",
                "📥 Data Input", 
                "🔍 Query & Search",
                "📊 Visualizations",
                "🔧 Maintenance",
                "📈 Analytics",
                "⚠️ Anomalies",
                "💡 Recommendations"
            ]
            
            selected_page = st.sidebar.radio("Select Page", pages, key="main_navigation")
            
            st.sidebar.markdown("---")
            
            # Quick actions section
            self._render_quick_actions()
            
            st.sidebar.markdown("---")
            
            # System metrics section
            self._render_system_metrics()
            
            # Return page name without emoji
            return selected_page.split(" ", 1)[1] if " " in selected_page else selected_page
            
        except Exception as e:
            self.logger.error(f"Error rendering sidebar: {e}")
            return "Overview"
    
    def _render_quick_actions(self):
        """Render quick action buttons in sidebar."""
        st.sidebar.markdown("## ⚡ Quick Actions")
        
        # Quick action buttons
        if st.sidebar.button("🚀 Run Full Pipeline", key="quick_pipeline"):
            st.session_state.trigger_pipeline = True
            
        if st.sidebar.button("📊 Generate Chart", key="quick_chart"):
            st.session_state.show_chart_modal = True
            
        if st.sidebar.button("🔍 Quick Search", key="quick_search"):
            st.session_state.show_search_modal = True
            
        if st.sidebar.button("💾 Export Data", key="quick_export"):
            st.session_state.show_export_modal = True
    
    def _render_system_metrics(self):
        """Render system metrics in sidebar."""
        st.sidebar.markdown("## 📈 System Metrics")
        
        try:
            # Get metrics from dashboard state or use mock data
            metrics = getattr(st.session_state, 'system_metrics', {
                "Total Nodes": 15420,
                "Total Relationships": 38750,
                "Vector Embeddings": 15420,
                "Active Patterns": 127,
                "Pending Actions": 3,
                "Cache Hit Rate": "87%",
                "Avg Response Time": "245ms"
            })
            
            # Display metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                st.sidebar.metric(metric_name, formatted_value)
                
        except Exception as e:
            self.logger.error(f"Error rendering system metrics: {e}")
    
    def create_metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal") -> None:
        """
        Create a styled metric card.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta/change value
            delta_color: Color for delta (normal, inverse, off)
        """
        try:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3 style="margin: 0; font-size: 1.2rem;">{title}</h3>'
                    f'<h2 style="margin: 0.5rem 0; font-size: 2rem;">{value}</h2>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            if delta:
                with col2:
                    delta_class = "status-good" if delta_color == "normal" else "status-error"
                    st.markdown(
                        f'<div class="{delta_class}" style="text-align: center; padding-top: 1rem;">'
                        f'{delta}</div>',
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            self.logger.error(f"Error creating metric card: {e}")
    
    def create_data_card(self, title: str, content: str, actions: Optional[List[str]] = None) -> None:
        """
        Create a styled data display card.
        
        Args:
            title: Card title
            content: Card content
            actions: Optional list of action button labels
        """
        try:
            st.markdown(
                f'<div class="data-card">'
                f'<h4 style="margin-top: 0; color: #2E8B57;">{title}</h4>'
                f'<div>{content}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            if actions:
                cols = st.columns(len(actions))
                for i, action in enumerate(actions):
                    with cols[i]:
                        if st.button(action, key=f"action_{title}_{i}"):
                            st.session_state[f"action_{action.lower().replace(' ', '_')}"] = True
                            
        except Exception as e:
            self.logger.error(f"Error creating data card: {e}")
    
    def render_loading_spinner(self, message: str = "Loading..."):
        """Render a loading spinner with message."""
        with st.spinner(message):
            st.empty()
    
    def render_progress_bar(self, progress: float, message: str = ""):
        """
        Render a progress bar.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
        if message:
            st.caption(message)
        st.progress(progress)
    
    def render_status_badge(self, status: str, label: str = "") -> None:
        """
        Render a status badge.
        
        Args:
            status: Status type (success, warning, error, info)
            label: Optional label text
        """
        badge_colors = {
            'success': '#28a745',
            'warning': '#ffc107', 
            'error': '#dc3545',
            'info': '#17a2b8'
        }
        
        color = badge_colors.get(status.lower(), '#6c757d')
        
        st.markdown(
            f'<span style="background-color: {color}; color: white; padding: 0.25rem 0.5rem; '
            f'border-radius: 0.25rem; font-size: 0.875rem; font-weight: bold;">'
            f'{label or status.title()}</span>',
            unsafe_allow_html=True
        )


# Standalone functions for backward compatibility
def render_header():
    """Render the dashboard header."""
    ui_components = UIComponents()
    ui_components.render_header()


def render_sidebar() -> str:
    """Render the sidebar and return selected page."""
    ui_components = UIComponents()
    return ui_components.render_sidebar()


def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      delta_color: str = "normal") -> None:
    """Create a metric card."""
    ui_components = UIComponents()
    ui_components.create_metric_card(title, value, delta, delta_color)


def create_data_card(title: str, content: str, actions: Optional[List[str]] = None) -> None:
    """Create a data card."""
    ui_components = UIComponents()
    ui_components.create_data_card(title, content, actions)


def apply_custom_styling():
    """Apply custom CSS styling."""
    ui_components = UIComponents()
    ui_components._apply_custom_styling()


# Factory function for easy instantiation
def create_ui_components() -> UIComponents:
    """
    Create and configure a UIComponents instance.
    
    Returns:
        Configured UIComponents instance
    """
    return UIComponents()


# Export main classes and functions
__all__ = [
    'UIComponents',
    'render_header',
    'render_sidebar',
    'create_metric_card',
    'create_data_card',
    'apply_custom_styling',
    'create_ui_components'
]