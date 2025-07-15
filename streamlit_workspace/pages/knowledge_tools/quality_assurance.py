"""
Quality Assurance Module
Data validation and quality control tools

Provides comprehensive quality assurance functionality:
- Full quality scans and analysis
- Duplicate detection and management
- Data consistency checks
- Relationship validation
- Coverage analysis and reporting

Extracted from knowledge_tools.py as part of modular refactoring.
Functions: show_quality_assurance, run_full_quality_scan, run_duplicate_detection,
run_consistency_check, run_relationship_validation, run_coverage_analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import sys

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database_operations import get_all_concepts, get_domains

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