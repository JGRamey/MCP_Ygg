"""
Styling and CSS utilities for Streamlit workspace

Provides consistent theming, custom CSS, and styling utilities extracted 
from existing pages to ensure visual consistency across the application.
"""

import streamlit as st
from typing import Dict, Any, Optional


def get_theme_colors() -> Dict[str, str]:
    """
    Get the standard color theme for the MCP Yggdrasil application.
    
    Returns:
        Dictionary containing color codes for consistent theming
    """
    return {
        'primary': '#2E8B57',      # Sea Green
        'secondary': '#667eea',    # Blue gradient start
        'accent': '#764ba2',       # Purple gradient end
        'success': '#90EE90',      # Light Green
        'warning': '#FFD700',      # Gold
        'error': '#FFB6C1',        # Light Pink
        'neutral': '#DDD',         # Light Gray
        'text_dark': '#333',       # Dark text
        'text_light': '#666',      # Light text
        'background': '#f8f9fa',   # Light background
        'border': '#ddd'           # Border color
    }


def apply_custom_css() -> None:
    """
    Apply comprehensive custom CSS styling to the Streamlit application.
    
    Includes styling for cards, headers, metrics, forms, and interactive elements
    extracted and unified from existing pages.
    """
    colors = get_theme_colors()
    
    custom_css = f"""
    <style>
    /* Main Application Styling */
    .main-header {{
        font-size: 3rem;
        color: {colors['primary']};
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .section-header {{
        font-size: 1.5rem;
        color: {colors['primary']};
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid {colors['primary']};
        padding-bottom: 0.5rem;
    }}
    
    /* Card Components */
    .metric-card {{
        background: linear-gradient(135deg, {colors['secondary']} 0%, {colors['accent']} 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 0.5rem;
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    
    .metric-change {{
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    
    .data-card {{
        background: white;
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s ease;
    }}
    
    .data-card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }}
    
    .concept-card {{
        background: white;
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {colors['primary']};
    }}
    
    .concept-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {colors['primary']};
        margin-bottom: 0.5rem;
    }}
    
    .concept-meta {{
        color: {colors['text_light']};
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    
    /* Form Styling */
    .form-section {{
        background: {colors['background']};
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid {colors['primary']};
    }}
    
    .form-header {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {colors['primary']};
        margin-bottom: 1rem;
    }}
    
    /* Button Styling */
    .action-button {{
        margin: 0.2rem;
        border-radius: 6px;
        font-weight: 500;
    }}
    
    .primary-button {{
        background-color: {colors['primary']};
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }}
    
    .primary-button:hover {{
        background-color: #1F5F3F;
    }}
    
    /* Status Indicators */
    .status-online {{
        color: {colors['success']};
        font-weight: 600;
    }}
    
    .status-offline {{
        color: {colors['error']};
        font-weight: 600;
    }}
    
    .status-warning {{
        color: {colors['warning']};
        font-weight: 600;
    }}
    
    /* Metric Indicators */
    .metric-positive {{ color: {colors['success']}; }}
    .metric-negative {{ color: {colors['error']}; }}
    .metric-neutral {{ color: {colors['neutral']}; }}
    
    /* Dashboard Grid */
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    /* Analytics Specific */
    .analytics-container {{
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .insight-card {{
        background: {colors['background']};
        border-left: 4px solid {colors['primary']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }}
    
    .insight-header {{
        font-weight: 600;
        color: {colors['primary']};
        margin-bottom: 0.5rem;
    }}
    
    .trend-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.9rem;
    }}
    
    /* File Browser */
    .file-browser {{
        background: white;
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }}
    
    .file-item {{
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: background-color 0.2s ease;
    }}
    
    .file-item:hover {{
        background-color: {colors['background']};
    }}
    
    /* Navigation */
    .nav-section {{
        margin: 1rem 0;
        padding: 1rem 0;
        border-bottom: 1px solid {colors['border']};
    }}
    
    .nav-header {{
        font-weight: 600;
        color: {colors['text_dark']};
        margin-bottom: 0.5rem;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .metrics-grid {{
            grid-template-columns: 1fr;
        }}
        
        .stats-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
        
        .main-header {{
            font-size: 2rem;
        }}
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {colors['background']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {colors['border']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {colors['text_light']};
    }}
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)


def apply_page_specific_css(page_type: str) -> None:
    """
    Apply page-specific CSS styling.
    
    Args:
        page_type: Type of page ('analytics', 'database', 'scraper', etc.)
    """
    colors = get_theme_colors()
    
    page_styles = {
        'analytics': f"""
        <style>
        .metric-dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .chart-container {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        </style>
        """,
        
        'database': f"""
        <style>
        .crud-section {{
            background: {colors['background']};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid {colors['primary']};
        }}
        
        .concept-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        </style>
        """,
        
        'scraper': f"""
        <style>
        .scraper-section {{
            background: white;
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        
        .submission-card {{
            background: {colors['background']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        </style>
        """
    }
    
    if page_type in page_styles:
        st.markdown(page_styles[page_type], unsafe_allow_html=True)


def create_custom_metric(label: str, value: str, delta: Optional[str] = None, 
                        delta_color: str = "normal") -> None:
    """
    Create a custom metric display with enhanced styling.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator (optional)
        delta_color: Color for delta ('positive', 'negative', 'normal')
    """
    delta_class = {
        'positive': 'metric-positive',
        'negative': 'metric-negative', 
        'normal': 'metric-neutral'
    }.get(delta_color, 'metric-neutral')
    
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ''
    
    metric_html = f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(metric_html, unsafe_allow_html=True)