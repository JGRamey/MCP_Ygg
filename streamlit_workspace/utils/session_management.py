"""
Session Management Utilities
Handles Streamlit session state and workspace configuration
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    
    # Workspace configuration
    if 'workspace_config' not in st.session_state:
        st.session_state.workspace_config = {
            'theme': 'default',
            'auto_save': True,
            'show_notifications': True,
            'debug_mode': False
        }
    
    # Database connections
    if 'db_connections' not in st.session_state:
        st.session_state.db_connections = {
            'neo4j': None,
            'qdrant': None,
            'redis': None
        }
    
    # Current selections
    if 'current_concept' not in st.session_state:
        st.session_state.current_concept = None
    
    if 'current_domain' not in st.session_state:
        st.session_state.current_domain = None
    
    if 'selected_relationships' not in st.session_state:
        st.session_state.selected_relationships = []
    
    # Operation history
    if 'operation_history' not in st.session_state:
        st.session_state.operation_history = []
    
    # Unsaved changes
    if 'unsaved_changes' not in st.session_state:
        st.session_state.unsaved_changes = False
    
    # Last activity timestamp
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()

def update_last_activity():
    """Update the last activity timestamp"""
    st.session_state.last_activity = datetime.now()

def add_to_history(operation_type, description, details=None):
    """Add an operation to the history"""
    
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation_type': operation_type,
        'description': description,
        'details': details or {}
    }
    
    # Add to history (keep last 100 operations)
    if len(st.session_state.operation_history) >= 100:
        st.session_state.operation_history.pop(0)
    
    st.session_state.operation_history.append(history_entry)
    update_last_activity()

def mark_unsaved_changes(has_changes=True):
    """Mark whether there are unsaved changes"""
    st.session_state.unsaved_changes = has_changes
    if has_changes:
        update_last_activity()

def get_workspace_config():
    """Get current workspace configuration"""
    return st.session_state.workspace_config.copy()

def update_workspace_config(key, value):
    """Update workspace configuration"""
    st.session_state.workspace_config[key] = value
    save_workspace_config()

def save_workspace_config():
    """Save workspace configuration to file"""
    try:
        config_dir = Path(__file__).parent.parent / 'assets'
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / 'workspace_config.json'
        with open(config_file, 'w') as f:
            json.dump(st.session_state.workspace_config, f, indent=2)
    except Exception as e:
        st.error(f"Could not save workspace config: {e}")

def load_workspace_config():
    """Load workspace configuration from file"""
    try:
        config_file = Path(__file__).parent.parent / 'assets' / 'workspace_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                st.session_state.workspace_config.update(config)
    except Exception as e:
        st.warning(f"Could not load workspace config: {e}")

def clear_session_state():
    """Clear all session state (use with caution)"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def get_session_info():
    """Get information about current session"""
    return {
        'last_activity': st.session_state.last_activity,
        'unsaved_changes': st.session_state.unsaved_changes,
        'operation_count': len(st.session_state.operation_history),
        'current_concept': st.session_state.current_concept,
        'current_domain': st.session_state.current_domain
    }