"""
Concept Builder Module
Advanced concept creation tools and wizards

Provides comprehensive tools for creating knowledge graph concepts:
- Guided step-by-step wizard
- Template-based builder
- Bulk import capabilities
- Concept cloning functionality

Extracted from knowledge_tools.py as part of modular refactoring.
Functions: show_concept_builder, show_guided_wizard, show_template_builder,
show_bulk_import, show_concept_cloner, and related wizard steps.
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
import sys

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database_operations import (
    create_concept, search_concepts, get_concept_by_id
)
from utils.session_management import add_to_history, mark_unsaved_changes

# Import shared utilities
from .shared_utils import generate_concept_id, validate_concept_data, clone_concept, import_concepts_from_df, import_concepts_from_text

def show_concept_builder():
    """Show concept builder wizard"""
    st.markdown("## 🏗️ Concept Builder Wizard")
    
    # Builder mode selection
    builder_mode = st.radio(
        "Builder Mode",
        ["🧙 Guided Wizard", "📝 Template Builder", "📚 Bulk Import", "🔄 Concept Cloner"],
        horizontal=True
    )
    
    if builder_mode == "🧙 Guided Wizard":
        show_guided_wizard()
    elif builder_mode == "📝 Template Builder":
        show_template_builder()
    elif builder_mode == "📚 Bulk Import":
        show_bulk_import()
    elif builder_mode == "🔄 Concept Cloner":
        show_concept_cloner()

def show_guided_wizard():
    """Show step-by-step concept creation wizard"""
    st.markdown("### 🧙 Guided Concept Creation Wizard")
    
    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    
    # Progress indicator
    progress = st.session_state.wizard_step / 6
    st.progress(progress, text=f"Step {st.session_state.wizard_step} of 6")
    
    if st.session_state.wizard_step == 1:
        show_wizard_step_1()
    elif st.session_state.wizard_step == 2:
        show_wizard_step_2()
    elif st.session_state.wizard_step == 3:
        show_wizard_step_3()
    elif st.session_state.wizard_step == 4:
        show_wizard_step_4()
    elif st.session_state.wizard_step == 5:
        show_wizard_step_5()
    elif st.session_state.wizard_step == 6:
        show_wizard_step_6()

def show_wizard_step_1():
    """Wizard Step 1: Basic Information"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 1: Basic Concept Information")
    
    concept_name = st.text_input("Concept Name", placeholder="e.g., Quantum_Mechanics")
    domain = st.selectbox("Domain", ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"])
    
    # Auto-generate ID suggestion
    if concept_name:
        suggested_id = generate_concept_id(concept_name, domain)
        concept_id = st.text_input("Concept ID", value=suggested_id)
    else:
        concept_id = st.text_input("Concept ID", placeholder="e.g., SCI0066")
    
    description = st.text_area("Description", placeholder="Detailed description of the concept...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col2:
        if st.button("Next →", type="primary", disabled=not all([concept_name, domain, concept_id])):
            st.session_state.wizard_data = {
                'name': concept_name,
                'domain': domain,
                'id': concept_id,
                'description': description
            }
            st.session_state.wizard_step = 2
            st.rerun()

def show_wizard_step_2():
    """Wizard Step 2: Classification"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 2: Concept Classification")
    
    concept_type = st.selectbox("Concept Type", ["root", "sub_root", "branch", "leaf"])
    level = st.number_input("Hierarchical Level", min_value=1, max_value=10, value=2)
    
    # Show hierarchy suggestion
    domain = st.session_state.wizard_data['domain']
    st.info(f"💡 For {domain} domain: Level 1 = {domain}, Level 2 = Major categories, Level 3+ = Specific concepts")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.wizard_step = 1
            st.rerun()
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.wizard_data.update({
                'type': concept_type,
                'level': level
            })
            st.session_state.wizard_step = 3
            st.rerun()

def show_wizard_step_3():
    """Wizard Step 3: Metadata"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 3: Additional Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Geographic Location", placeholder="Optional")
        cultural_context = st.text_input("Cultural Context", placeholder="e.g., Ancient Greek, Medieval European")
    
    with col2:
        earliest_date = st.number_input("Earliest Evidence Date", value=None, placeholder="Year")
        latest_date = st.number_input("Latest Evidence Date", value=None, placeholder="Year")
    
    certainty_level = st.selectbox("Certainty Level", ["High", "Medium", "Low", "Unknown"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.wizard_step = 2
            st.rerun()
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.wizard_data.update({
                'location': location,
                'cultural_context': cultural_context,
                'earliest_evidence_date': earliest_date,
                'latest_evidence_date': latest_date,
                'certainty_level': certainty_level
            })
            st.session_state.wizard_step = 4
            st.rerun()

def show_wizard_step_4():
    """Wizard Step 4: Relationships"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 4: Define Relationships")
    
    st.markdown("##### Parent Concept")
    parent_search = st.text_input("Search for parent concept", placeholder="Start typing to search...")
    
    if parent_search:
        parent_candidates = search_concepts(parent_search, limit=5)
        if parent_candidates:
            parent_options = [f"{c['id']}: {c['name']}" for c in parent_candidates]
            selected_parent = st.selectbox("Select Parent", parent_options)
        else:
            st.info("No matching concepts found")
    
    st.markdown("##### Related Concepts")
    related_search = st.text_input("Search for related concepts", placeholder="Start typing to search...")
    
    if related_search:
        related_candidates = search_concepts(related_search, limit=10)
        if related_candidates:
            related_options = [f"{c['id']}: {c['name']}" for c in related_candidates]
            selected_related = st.multiselect("Select Related Concepts", related_options)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.wizard_step = 3
            st.rerun()
    with col2:
        if st.button("Next →", type="primary"):
            st.session_state.wizard_data.update({
                'parent_concept': locals().get('selected_parent'),
                'related_concepts': locals().get('selected_related', [])
            })
            st.session_state.wizard_step = 5
            st.rerun()

def show_wizard_step_5():
    """Wizard Step 5: Validation"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 5: Validation & Review")
    
    # Show concept preview
    wizard_data = st.session_state.wizard_data
    
    st.markdown("##### Concept Preview")
    st.json(wizard_data)
    
    # Run validation checks
    st.markdown("##### Validation Results")
    validation_results = validate_concept_data(wizard_data)
    
    for check, result in validation_results.items():
        if result['status'] == 'pass':
            st.markdown(f'<div class="validation-result validation-pass">✅ {check}: {result["message"]}</div>', unsafe_allow_html=True)
        elif result['status'] == 'warn':
            st.markdown(f'<div class="validation-result validation-warn">⚠️ {check}: {result["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="validation-result validation-fail">❌ {check}: {result["message"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.wizard_step = 4
            st.rerun()
    with col2:
        all_passed = all(r['status'] in ['pass', 'warn'] for r in validation_results.values())
        if st.button("Create Concept →", type="primary", disabled=not all_passed):
            st.session_state.wizard_step = 6
            st.rerun()

def show_wizard_step_6():
    """Wizard Step 6: Creation"""
    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    st.markdown("#### Step 6: Concept Creation")
    
    wizard_data = st.session_state.wizard_data
    
    try:
        # Create the concept
        success, message = create_concept(wizard_data)
        
        if success:
            st.success(f"✅ {message}")
            st.balloons()
            
            add_to_history("CREATE", f"Created concept via wizard: {wizard_data['id']}")
            mark_unsaved_changes(False)
            
            # Show next steps
            st.markdown("##### What's Next?")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 View Concept", use_container_width=True):
                    st.session_state.selected_concept = wizard_data['id']
                    st.switch_page("pages/01_🗄️_Database_Manager.py")
            
            with col2:
                if st.button("🏗️ Create Another", use_container_width=True):
                    # Reset wizard
                    del st.session_state.wizard_step
                    del st.session_state.wizard_data
                    st.rerun()
            
            with col3:
                if st.button("📊 View Graph", use_container_width=True):
                    st.switch_page("pages/02_📊_Graph_Editor.py")
        
        else:
            st.error(f"❌ {message}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Review"):
                    st.session_state.wizard_step = 5
                    st.rerun()
            with col2:
                if st.button("🔄 Try Again"):
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error creating concept: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_template_builder():
    """Show template-based concept builder"""
    st.markdown("### 📝 Template-Based Builder")
    
    # Template selection
    templates = {
        "Scientific Concept": {
            "type": "branch",
            "level": 3,
            "domain": "Science",
            "certainty_level": "High"
        },
        "Historical Figure": {
            "type": "leaf",
            "level": 4,
            "domain": "Art",
            "certainty_level": "High"
        },
        "Philosophical Concept": {
            "type": "branch",
            "level": 3,
            "domain": "Philosophy",
            "certainty_level": "Medium"
        },
        "Mathematical Theorem": {
            "type": "leaf",
            "level": 4,
            "domain": "Mathematics",
            "certainty_level": "High"
        }
    }
    
    selected_template = st.selectbox("Select Template", list(templates.keys()))
    template_data = templates[selected_template]
    
    st.markdown("#### Template Preview")
    st.json(template_data)
    
    # Customize template
    st.markdown("#### Customize Template")
    
    with st.form("template_form"):
        concept_name = st.text_input("Concept Name")
        concept_id = st.text_input("Concept ID")
        description = st.text_area("Description")
        
        # Override template defaults if needed
        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Domain", ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"], 
                                 index=["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"].index(template_data["domain"]))
        with col2:
            level = st.number_input("Level", value=template_data["level"])
        
        if st.form_submit_button("Create from Template", type="primary"):
            concept_data = {
                **template_data,
                'name': concept_name,
                'id': concept_id,
                'description': description,
                'domain': domain,
                'level': level
            }
            
            success, message = create_concept(concept_data)
            if success:
                st.success(f"✅ {message}")
                add_to_history("CREATE", f"Created concept from template: {concept_id}")
            else:
                st.error(f"❌ {message}")

def show_bulk_import():
    """Show bulk import interface"""
    st.markdown("### 📚 Bulk Import Tool")
    
    import_method = st.radio("Import Method", ["📋 CSV Upload", "📝 Text List", "🔗 API Import"])
    
    if import_method == "📋 CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("#### Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column mapping
                st.markdown("#### Column Mapping")
                required_columns = ['name', 'domain', 'type', 'level']
                
                mapping = {}
                for req_col in required_columns:
                    mapping[req_col] = st.selectbox(f"Map to '{req_col}'", df.columns.tolist())
                
                if st.button("Import Concepts", type="primary"):
                    import_concepts_from_df(df, mapping)
            
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    elif import_method == "📝 Text List":
        st.markdown("#### Text List Import")
        text_input = st.text_area("Enter concept names (one per line)", height=200)
        
        domain = st.selectbox("Default Domain", ["Art", "Science", "Mathematics", "Philosophy", "Language", "Technology", "Religion", "Astrology"])
        concept_type = st.selectbox("Default Type", ["root", "sub_root", "branch", "leaf"])
        level = st.number_input("Default Level", value=3)
        
        if st.button("Import from Text", type="primary") and text_input:
            import_concepts_from_text(text_input, domain, concept_type, level)

def show_concept_cloner():
    """Show concept cloning interface"""
    st.markdown("### 🔄 Concept Cloner")
    
    # Source concept selection
    source_search = st.text_input("Search for concept to clone", placeholder="Enter concept name or ID...")
    
    if source_search:
        candidates = search_concepts(source_search, limit=10)
        
        if candidates:
            candidate_options = [f"{c['id']}: {c['name']}" for c in candidates]
            selected_candidate = st.selectbox("Select Concept to Clone", candidate_options)
            
            if selected_candidate:
                source_id = selected_candidate.split(":")[0]
                source_concept = get_concept_by_id(source_id)
                
                if source_concept:
                    st.markdown("#### Source Concept")
                    st.json(source_concept)
                    
                    # Clone configuration
                    st.markdown("#### Clone Configuration")
                    
                    with st.form("clone_form"):
                        new_name = st.text_input("New Concept Name", value=f"{source_concept['name']}_Copy")
                        new_id = st.text_input("New Concept ID", value=generate_concept_id(new_name, source_concept['domain']))
                        
                        # What to copy
                        copy_relationships = st.checkbox("Copy Relationships", value=False)
                        copy_metadata = st.checkbox("Copy Metadata", value=True)
                        
                        if st.form_submit_button("Clone Concept", type="primary"):
                            clone_concept(source_concept, new_name, new_id, copy_relationships, copy_metadata)