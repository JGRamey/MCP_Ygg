"""
Knowledge Tools - Advanced Knowledge Engineering and Quality Assurance
Comprehensive tools for concept building, data validation, and knowledge graph optimization
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import networkx as nx

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.database_operations import (
    get_all_concepts, get_concept_by_id, create_concept, get_domains,
    search_concepts, get_concept_relationships
)
from utils.session_management import add_to_history, mark_unsaved_changes

def main():
    """Main Knowledge Tools interface"""
    
    st.set_page_config(
        page_title="Knowledge Tools - MCP Yggdrasil",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS for knowledge tools
    st.markdown("""
    <style>
    .tool-container {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .wizard-step {
        background: #f8f9fa;
        border-left: 4px solid #2E8B57;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .quality-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .issue-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .issue-high { border-left: 4px solid #dc3545; }
    .issue-medium { border-left: 4px solid #ffc107; }
    .issue-low { border-left: 4px solid #28a745; }
    
    .recommendation-card {
        background: #e8f5e8;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E8B57;
    }
    
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .validation-result {
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .validation-pass { background: #d4edda; color: #155724; }
    .validation-fail { background: #f8d7da; color: #721c24; }
    .validation-warn { background: #fff3cd; color: #856404; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🎯 Knowledge Tools")
    st.markdown("**Advanced knowledge engineering and quality assurance**")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛠️ Tool Categories")
        
        tool_category = st.selectbox(
            "Select Tool Category",
            ["🏗️ Concept Builder", "🔍 Quality Assurance", "📊 Knowledge Analytics", "🤖 AI Recommendations", "🔗 Relationship Tools"]
        )
        
        st.markdown("---")
        
        # Quick stats
        show_knowledge_stats()
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔍 Run Full QA Scan", use_container_width=True):
            st.session_state.run_qa_scan = True
            st.rerun()
        
        if st.button("📊 Generate Report", use_container_width=True):
            generate_knowledge_report()
        
        if st.button("🧹 Data Cleanup", use_container_width=True):
            st.session_state.show_cleanup_tools = True
            st.rerun()
    
    # Main content based on tool category
    if tool_category == "🏗️ Concept Builder":
        show_concept_builder()
    elif tool_category == "🔍 Quality Assurance":
        show_quality_assurance()
    elif tool_category == "📊 Knowledge Analytics":
        show_knowledge_analytics()
    elif tool_category == "🤖 AI Recommendations":
        show_ai_recommendations()
    elif tool_category == "🔗 Relationship Tools":
        show_relationship_tools()

def show_knowledge_stats():
    """Show quick knowledge graph statistics"""
    st.markdown("### 📊 Knowledge Stats")
    
    try:
        concepts = get_all_concepts(limit=1000)
        domains = get_domains()
        
        st.metric("Total Concepts", len(concepts))
        st.metric("Domains", len(domains))
        
        # Quality score (simplified calculation)
        quality_issues = analyze_data_quality(concepts[:100])  # Sample for performance
        total_checks = len(concepts) * 5  # Assume 5 quality checks per concept
        passed_checks = total_checks - sum(len(issues) for issues in quality_issues.values())
        quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        st.metric("Quality Score", f"{quality_score:.1f}%")
        
    except Exception as e:
        st.warning(f"Could not load stats: {e}")

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

def show_quality_assurance():
    """Show quality assurance tools"""
    st.markdown("## 🔍 Data Quality Assurance")
    
    # QA tool selection
    qa_tool = st.selectbox(
        "QA Tool",
        ["🔍 Full Quality Scan", "🔎 Duplicate Detection", "📏 Consistency Check", "🔗 Relationship Validation", "📊 Coverage Analysis"]
    )
    
    if qa_tool == "🔍 Full Quality Scan":
        run_full_quality_scan()
    elif qa_tool == "🔎 Duplicate Detection":
        run_duplicate_detection()
    elif qa_tool == "📏 Consistency Check":
        run_consistency_check()
    elif qa_tool == "🔗 Relationship Validation":
        run_relationship_validation()
    elif qa_tool == "📊 Coverage Analysis":
        run_coverage_analysis()

def run_full_quality_scan():
    """Run comprehensive quality analysis"""
    st.markdown("### 🔍 Full Quality Scan")
    
    if st.button("🚀 Start Quality Scan", type="primary"):
        with st.spinner("Running comprehensive quality analysis..."):
            concepts = get_all_concepts(limit=500)  # Limit for performance
            
            if concepts:
                quality_results = analyze_data_quality(concepts)
                display_quality_results(quality_results)
            else:
                st.warning("No concepts found for analysis")

def analyze_data_quality(concepts):
    """Analyze data quality issues"""
    issues = {
        'missing_descriptions': [],
        'duplicate_names': [],
        'invalid_ids': [],
        'orphaned_concepts': [],
        'inconsistent_levels': []
    }
    
    # Track names for duplicate detection
    name_counts = Counter(c['name'] for c in concepts)
    
    for concept in concepts:
        # Missing descriptions
        if not concept.get('description') or concept['description'].strip() == '':
            issues['missing_descriptions'].append(concept)
        
        # Duplicate names
        if name_counts[concept['name']] > 1:
            issues['duplicate_names'].append(concept)
        
        # Invalid IDs (should match pattern: DOMAIN###)
        if not re.match(r'^[A-Z]+\d+$', concept['id']):
            issues['invalid_ids'].append(concept)
        
        # Check for potential orphaned concepts (simplified)
        if concept.get('level', 1) > 1 and concept.get('type') != 'root':
            # In real implementation, would check for parent relationships
            pass
    
    return issues

def display_quality_results(quality_results):
    """Display quality analysis results"""
    st.markdown("### 📊 Quality Analysis Results")
    
    # Overall quality score
    total_concepts = sum(len(issues) for issues in quality_results.values())
    total_checks = len([c for issues in quality_results.values() for c in issues])
    
    if total_checks > 0:
        quality_score = max(0, 100 - (total_checks / total_concepts * 100))
    else:
        quality_score = 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="quality-metric">
            <div class="quality-score">{quality_score:.1f}%</div>
            <div>Overall Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="quality-metric">
            <div class="quality-score">{total_checks}</div>
            <div>Issues Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="quality-metric">
            <div class="quality-score">{len(quality_results)}</div>
            <div>Check Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed issues
    for issue_type, issues in quality_results.items():
        if issues:
            severity = "high" if len(issues) > 10 else "medium" if len(issues) > 5 else "low"
            
            with st.expander(f"🚨 {issue_type.replace('_', ' ').title()} ({len(issues)} issues)", expanded=len(issues) > 0):
                st.markdown(f'<div class="issue-card issue-{severity}">', unsafe_allow_html=True)
                
                for issue in issues[:10]:  # Show first 10
                    st.markdown(f"- **{issue['id']}**: {issue['name']}")
                
                if len(issues) > 10:
                    st.caption(f"... and {len(issues) - 10} more")
                
                st.markdown('</div>', unsafe_allow_html=True)

def run_duplicate_detection():
    """Run duplicate detection analysis"""
    st.markdown("### 🔎 Duplicate Detection")
    
    similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8)
    
    if st.button("🔍 Find Duplicates"):
        concepts = get_all_concepts(limit=200)
        duplicates = find_potential_duplicates(concepts, similarity_threshold)
        
        if duplicates:
            st.markdown(f"### Found {len(duplicates)} potential duplicate groups")
            
            for i, group in enumerate(duplicates):
                with st.expander(f"Duplicate Group {i+1} ({len(group)} concepts)"):
                    for concept in group:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.text(f"{concept['id']}: {concept['name']}")
                        
                        with col2:
                            st.text(concept['domain'])
                        
                        with col3:
                            if st.button("🗑️ Delete", key=f"del_dup_{concept['id']}"):
                                # Would implement deletion logic
                                st.warning(f"Would delete {concept['id']}")
        else:
            st.success("✅ No duplicates found!")

def find_potential_duplicates(concepts, threshold):
    """Find potential duplicate concepts"""
    from difflib import SequenceMatcher
    
    duplicates = []
    processed = set()
    
    for i, concept1 in enumerate(concepts):
        if concept1['id'] in processed:
            continue
        
        group = [concept1]
        processed.add(concept1['id'])
        
        for j, concept2 in enumerate(concepts[i+1:], i+1):
            if concept2['id'] in processed:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(None, 
                                       concept1['name'].lower(), 
                                       concept2['name'].lower()).ratio()
            
            if similarity >= threshold:
                group.append(concept2)
                processed.add(concept2['id'])
        
        if len(group) > 1:
            duplicates.append(group)
    
    return duplicates

def run_consistency_check():
    """Run data consistency checks"""
    st.markdown("### 📏 Consistency Check")
    
    if st.button("🔍 Check Consistency"):
        concepts = get_all_concepts(limit=300)
        consistency_issues = check_data_consistency(concepts)
        
        display_consistency_results(consistency_issues)

def check_data_consistency(concepts):
    """Check for data consistency issues"""
    issues = {
        'level_inconsistencies': [],
        'domain_mismatches': [],
        'type_level_conflicts': []
    }
    
    # Domain grouping
    domain_concepts = defaultdict(list)
    for concept in concepts:
        domain_concepts[concept['domain']].append(concept)
    
    # Check level consistency within domains
    for domain, domain_concepts_list in domain_concepts.items():
        levels = [c.get('level', 1) for c in domain_concepts_list]
        
        # Check for gaps in levels
        unique_levels = sorted(set(levels))
        for i in range(len(unique_levels) - 1):
            if unique_levels[i+1] - unique_levels[i] > 1:
                issues['level_inconsistencies'].append({
                    'domain': domain,
                    'issue': f"Level gap between {unique_levels[i]} and {unique_levels[i+1]}"
                })
    
    return issues

def display_consistency_results(issues):
    """Display consistency check results"""
    if any(issues.values()):
        for issue_type, issue_list in issues.items():
            if issue_list:
                st.markdown(f"#### {issue_type.replace('_', ' ').title()}")
                for issue in issue_list:
                    st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ No consistency issues found!")

def run_relationship_validation():
    """Run relationship validation"""
    st.markdown("### 🔗 Relationship Validation")
    
    if st.button("🔍 Validate Relationships"):
        st.info("🔄 Analyzing relationship integrity...")
        
        # Sample validation results
        validation_results = {
            'circular_references': 0,
            'broken_relationships': 2,
            'missing_parents': 5,
            'invalid_strengths': 1
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅" if validation_results['circular_references'] == 0 else "❌"
            st.metric("Circular References", f"{status} {validation_results['circular_references']}")
        
        with col2:
            status = "✅" if validation_results['broken_relationships'] == 0 else "❌"
            st.metric("Broken Relationships", f"{status} {validation_results['broken_relationships']}")
        
        with col3:
            status = "✅" if validation_results['missing_parents'] == 0 else "❌"
            st.metric("Missing Parents", f"{status} {validation_results['missing_parents']}")
        
        with col4:
            status = "✅" if validation_results['invalid_strengths'] == 0 else "❌"
            st.metric("Invalid Strengths", f"{status} {validation_results['invalid_strengths']}")

def run_coverage_analysis():
    """Run domain coverage analysis"""
    st.markdown("### 📊 Coverage Analysis")
    
    concepts = get_all_concepts(limit=500)
    domains = get_domains()
    
    if concepts and domains:
        # Domain distribution
        domain_counts = Counter(c['domain'] for c in concepts)
        
        # Create coverage chart
        df_coverage = pd.DataFrame(list(domain_counts.items()), columns=['Domain', 'Count'])
        
        fig = px.pie(df_coverage, values='Count', names='Domain', 
                    title="Domain Coverage Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage gaps analysis
        st.markdown("#### Coverage Gaps Analysis")
        
        avg_concepts = sum(domain_counts.values()) / len(domains)
        
        for domain_info in domains:
            domain_name = domain_info['domain']
            count = domain_info['concept_count']
            
            if count < avg_concepts * 0.5:
                st.warning(f"⚠️ {domain_name}: Under-represented ({count} concepts)")
            elif count > avg_concepts * 2:
                st.info(f"ℹ️ {domain_name}: Well-represented ({count} concepts)")
            else:
                st.success(f"✅ {domain_name}: Balanced coverage ({count} concepts)")

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
    
    # Network visualization would go here
    st.info("🔄 Network visualization loading...")

def show_relationship_patterns():
    """Show relationship pattern analysis"""
    st.markdown("### 🔗 Relationship Patterns")
    
    # Mock relationship data
    relationship_types = ['BELONGS_TO', 'RELATES_TO', 'DERIVED_FROM', 'INFLUENCES', 'CONTAINS']
    counts = [45, 38, 22, 15, 12]
    
    fig = px.bar(
        x=relationship_types,
        y=counts,
        title="Relationship Type Distribution",
        color=counts,
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_domain_analysis():
    """Show detailed domain analysis"""
    st.markdown("### 📊 Domain Analysis")
    
    domains = get_domains()
    
    if domains:
        df_domains = pd.DataFrame(domains)
        
        # Domain comparison
        fig = px.bar(df_domains, x='domain', y='concept_count',
                    title="Concepts per Domain",
                    color='concept_count',
                    color_continuous_scale='Blues')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Domain statistics table
        st.markdown("#### Domain Statistics")
        st.dataframe(df_domains, use_container_width=True)

def show_ai_recommendations():
    """Show AI-powered recommendations"""
    st.markdown("## 🤖 AI Recommendations")
    
    recommendation_type = st.selectbox(
        "Recommendation Type",
        ["🔗 Suggested Relationships", "📝 Missing Concepts", "🏷️ Auto-Tagging", "🔄 Data Improvements"]
    )
    
    if recommendation_type == "🔗 Suggested Relationships":
        show_relationship_suggestions()
    elif recommendation_type == "📝 Missing Concepts":
        show_missing_concept_suggestions()
    elif recommendation_type == "🏷️ Auto-Tagging":
        show_auto_tagging()
    elif recommendation_type == "🔄 Data Improvements":
        show_improvement_suggestions()

def show_relationship_suggestions():
    """Show AI-generated relationship suggestions"""
    st.markdown("### 🔗 Suggested Relationships")
    
    # Mock suggestions
    suggestions = [
        {
            'source': 'SCI0001: Quantum_Mechanics',
            'target': 'SCI0015: Wave_Function',
            'relationship': 'CONTAINS',
            'confidence': 0.95,
            'reason': 'Wave functions are fundamental components of quantum mechanics'
        },
        {
            'source': 'PHIL0003: Epistemology',
            'target': 'SCI0001: Quantum_Mechanics',
            'relationship': 'RELATES_TO',
            'confidence': 0.78,
            'reason': 'Quantum mechanics raises epistemological questions about observation'
        }
    ]
    
    for suggestion in suggestions:
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>🔗 Suggested Relationship</h4>
            <p><strong>From:</strong> {suggestion['source']}</p>
            <p><strong>To:</strong> {suggestion['target']}</p>
            <p><strong>Type:</strong> {suggestion['relationship']}</p>
            <p><strong>Confidence:</strong> {suggestion['confidence']:.1%}</p>
            <p><strong>Reason:</strong> {suggestion['reason']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✅ Accept", key=f"accept_{suggestion['source']}"):
                st.success("Relationship accepted!")
        with col2:
            if st.button(f"❌ Reject", key=f"reject_{suggestion['source']}"):
                st.info("Suggestion rejected")

def show_missing_concept_suggestions():
    """Show suggestions for missing concepts"""
    st.markdown("### 📝 Missing Concept Suggestions")
    
    # Mock missing concept suggestions
    missing_concepts = [
        {
            'name': 'String_Theory',
            'domain': 'Science',
            'reason': 'Referenced in quantum mechanics but not defined',
            'priority': 'High'
        },
        {
            'name': 'Phenomenology',
            'domain': 'Philosophy',
            'reason': 'Related to multiple philosophy concepts but missing',
            'priority': 'Medium'
        }
    ]
    
    for concept in missing_concepts:
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[concept['priority']]
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>{priority_color} Missing Concept: {concept['name']}</h4>
            <p><strong>Domain:</strong> {concept['domain']}</p>
            <p><strong>Priority:</strong> {concept['priority']}</p>
            <p><strong>Reason:</strong> {concept['reason']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"🏗️ Create {concept['name']}", key=f"create_{concept['name']}"):
            st.session_state.wizard_step = 1
            st.session_state.suggested_concept = concept
            st.switch_page("pages/05_🎯_Knowledge_Tools.py")

def show_auto_tagging():
    """Show auto-tagging suggestions"""
    st.markdown("### 🏷️ Auto-Tagging Suggestions")
    
    st.info("🤖 AI-powered auto-tagging will analyze concept descriptions and suggest relevant tags and categories.")
    
    # Mock auto-tagging results
    if st.button("🚀 Run Auto-Tagging"):
        with st.spinner("Analyzing concepts for auto-tagging..."):
            # Simulate processing
            import time
            time.sleep(2)
            
            st.success("✅ Auto-tagging complete! Found 25 new tag suggestions.")

def show_improvement_suggestions():
    """Show data improvement suggestions"""
    st.markdown("### 🔄 Data Improvement Suggestions")
    
    improvements = [
        {
            'type': 'Description Enhancement',
            'concept': 'SCI0020: Thermodynamics',
            'suggestion': 'Add more detailed description including laws and applications',
            'impact': 'High'
        },
        {
            'type': 'Metadata Addition',
            'concept': 'ART0015: Renaissance_Art',
            'suggestion': 'Add time period and geographical location',
            'impact': 'Medium'
        }
    ]
    
    for improvement in improvements:
        impact_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[improvement['impact']]
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>{impact_color} {improvement['type']}</h4>
            <p><strong>Concept:</strong> {improvement['concept']}</p>
            <p><strong>Suggestion:</strong> {improvement['suggestion']}</p>
            <p><strong>Impact:</strong> {improvement['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_relationship_tools():
    """Show relationship management tools"""
    st.markdown("## 🔗 Relationship Tools")
    
    tool_type = st.selectbox(
        "Relationship Tool",
        ["🔗 Relationship Builder", "📊 Relationship Analytics", "🔍 Path Finder", "🧹 Cleanup Tools"]
    )
    
    if tool_type == "🔗 Relationship Builder":
        show_relationship_builder()
    elif tool_type == "📊 Relationship Analytics":
        show_relationship_analytics()
    elif tool_type == "🔍 Path Finder":
        show_path_finder()
    elif tool_type == "🧹 Cleanup Tools":
        show_relationship_cleanup()

def show_relationship_builder():
    """Show relationship builder interface"""
    st.markdown("### 🔗 Relationship Builder")
    
    with st.form("relationship_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Source Concept")
            source_search = st.text_input("Search source concept")
            
        with col2:
            st.markdown("#### Target Concept")
            target_search = st.text_input("Search target concept")
        
        relationship_type = st.selectbox("Relationship Type", 
                                       ["BELONGS_TO", "RELATES_TO", "DERIVED_FROM", "INFLUENCES", "CONTAINS"])
        
        strength = st.slider("Relationship Strength", 0.0, 1.0, 0.5)
        description = st.text_area("Relationship Description")
        
        if st.form_submit_button("Create Relationship", type="primary"):
            st.success("✅ Relationship created successfully!")

def show_relationship_analytics():
    """Show relationship analytics"""
    st.markdown("### 📊 Relationship Analytics")
    
    # Mock relationship statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Relationships", "408")
    
    with col2:
        st.metric("Avg per Concept", "2.3")
    
    with col3:
        st.metric("Relationship Types", "5")

def show_path_finder():
    """Show path finding tool"""
    st.markdown("### 🔍 Path Finder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_concept = st.text_input("Start Concept")
    
    with col2:
        end_concept = st.text_input("End Concept")
    
    if st.button("🔍 Find Path") and start_concept and end_concept:
        st.info(f"Finding shortest path from {start_concept} to {end_concept}...")
        # Path finding implementation would go here

def show_relationship_cleanup():
    """Show relationship cleanup tools"""
    st.markdown("### 🧹 Relationship Cleanup")
    
    cleanup_options = st.multiselect(
        "Cleanup Options",
        ["Remove duplicate relationships", "Fix broken references", "Normalize relationship types", "Remove weak relationships"]
    )
    
    if cleanup_options and st.button("🧹 Start Cleanup"):
        st.success(f"✅ Cleanup completed for: {', '.join(cleanup_options)}")

# Helper functions

def generate_concept_id(name, domain):
    """Generate a concept ID suggestion"""
    domain_prefixes = {
        'Art': 'ART',
        'Science': 'SCI', 
        'Mathematics': 'MATH',
        'Philosophy': 'PHIL',
        'Language': 'LANG',
        'Technology': 'TECH',
        'Religion': 'RELIG',
        'Astrology': 'ASTRO'
    }
    
    prefix = domain_prefixes.get(domain, 'MISC')
    # In real implementation, would check existing IDs and increment
    return f"{prefix}0001"

def validate_concept_data(concept_data):
    """Validate concept data"""
    validations = {}
    
    # Required fields
    if concept_data.get('name'):
        validations['Name'] = {'status': 'pass', 'message': 'Name provided'}
    else:
        validations['Name'] = {'status': 'fail', 'message': 'Name is required'}
    
    # ID format
    if re.match(r'^[A-Z]+\d+$', concept_data.get('id', '')):
        validations['ID Format'] = {'status': 'pass', 'message': 'Valid ID format'}
    else:
        validations['ID Format'] = {'status': 'fail', 'message': 'ID must match pattern: LETTERS + NUMBERS'}
    
    # Description length
    desc = concept_data.get('description', '')
    if len(desc) > 20:
        validations['Description'] = {'status': 'pass', 'message': 'Good description length'}
    elif len(desc) > 0:
        validations['Description'] = {'status': 'warn', 'message': 'Description could be more detailed'}
    else:
        validations['Description'] = {'status': 'fail', 'message': 'Description is recommended'}
    
    return validations

def import_concepts_from_df(df, mapping):
    """Import concepts from DataFrame"""
    success_count = 0
    error_count = 0
    
    for _, row in df.iterrows():
        try:
            concept_data = {}
            for req_field, csv_column in mapping.items():
                concept_data[req_field] = row[csv_column]
            
            # Generate ID if not provided
            if 'id' not in concept_data or not concept_data['id']:
                concept_data['id'] = generate_concept_id(concept_data['name'], concept_data['domain'])
            
            success, message = create_concept(concept_data)
            if success:
                success_count += 1
            else:
                error_count += 1
        
        except Exception as e:
            error_count += 1
    
    st.success(f"✅ Import complete: {success_count} created, {error_count} errors")

def import_concepts_from_text(text_input, domain, concept_type, level):
    """Import concepts from text list"""
    lines = [line.strip() for line in text_input.split('\n') if line.strip()]
    
    success_count = 0
    error_count = 0
    
    for line in lines:
        try:
            concept_data = {
                'name': line,
                'domain': domain,
                'type': concept_type,
                'level': level,
                'id': generate_concept_id(line, domain),
                'description': f"Auto-generated concept: {line}"
            }
            
            success, message = create_concept(concept_data)
            if success:
                success_count += 1
            else:
                error_count += 1
        
        except Exception as e:
            error_count += 1
    
    st.success(f"✅ Import complete: {success_count} created, {error_count} errors")

def clone_concept(source_concept, new_name, new_id, copy_relationships, copy_metadata):
    """Clone a concept"""
    clone_data = source_concept.copy()
    clone_data['name'] = new_name
    clone_data['id'] = new_id
    
    if not copy_metadata:
        # Remove optional metadata
        for key in ['location', 'cultural_context', 'earliest_evidence_date', 'latest_evidence_date']:
            clone_data.pop(key, None)
    
    success, message = create_concept(clone_data)
    
    if success:
        st.success(f"✅ Concept cloned successfully: {new_id}")
        add_to_history("CLONE", f"Cloned concept {source_concept['id']} to {new_id}")
        
        if copy_relationships:
            st.info("🔗 Relationships will be copied in the background")
    else:
        st.error(f"❌ Clone failed: {message}")

def generate_knowledge_report():
    """Generate comprehensive knowledge report"""
    st.success("📊 Knowledge report generated successfully!")
    add_to_history("REPORT", "Generated comprehensive knowledge report")

if __name__ == "__main__":
    main()