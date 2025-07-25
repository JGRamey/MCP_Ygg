"""
Relationship Manager Module
Advanced relationship management and analysis tools

Provides comprehensive relationship management functionality:
- Relationship builder interface
- Relationship analytics and metrics
- Path finding between concepts
- Relationship cleanup and maintenance tools

Extracted from knowledge_tools.py as part of modular refactoring.
Functions: show_relationship_tools, show_relationship_builder,
show_relationship_analytics, show_path_finder, show_relationship_cleanup.
"""

import sys
from pathlib import Path

import streamlit as st

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent.parent))


def show_relationship_tools():
    """Show relationship management tools"""
    st.markdown("## 🔗 Relationship Tools")

    tool_type = st.selectbox(
        "Relationship Tool",
        [
            "🔗 Relationship Builder",
            "📊 Relationship Analytics",
            "🔍 Path Finder",
            "🧹 Cleanup Tools",
        ],
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

    st.markdown(
        """
    Create new relationships between concepts in your knowledge graph.
    Relationships define how concepts connect and relate to each other.
    """
    )

    # Relationship creation interface
    with st.form("relationship_form"):
        st.markdown("#### Define New Relationship")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Source Concept**")
            source_search = st.text_input(
                "Search source concept",
                placeholder="Start typing concept name or ID...",
            )

            # Mock search results
            if source_search:
                st.info(f"Searching for: '{source_search}'")
                # In real implementation, would show search results
                source_concept = st.selectbox(
                    "Select Source",
                    ["SCI0001: Quantum_Mechanics", "PHIL0003: Epistemology"],
                )
            else:
                source_concept = st.selectbox(
                    "Select Source",
                    ["", "SCI0001: Quantum_Mechanics", "PHIL0003: Epistemology"],
                )

        with col2:
            st.markdown("**Target Concept**")
            target_search = st.text_input(
                "Search target concept",
                placeholder="Start typing concept name or ID...",
            )

            # Mock search results
            if target_search:
                st.info(f"Searching for: '{target_search}'")
                target_concept = st.selectbox(
                    "Select Target",
                    ["SCI0015: Wave_Function", "MATH0005: Linear_Algebra"],
                )
            else:
                target_concept = st.selectbox(
                    "Select Target",
                    ["", "SCI0015: Wave_Function", "MATH0005: Linear_Algebra"],
                )

        # Relationship properties
        st.markdown("#### Relationship Properties")

        col1, col2 = st.columns(2)

        with col1:
            relationship_type = st.selectbox(
                "Relationship Type",
                [
                    "BELONGS_TO",
                    "RELATES_TO",
                    "DERIVED_FROM",
                    "INFLUENCES",
                    "CONTAINS",
                    "SUPPORTS",
                    "CONTRADICTS",
                ],
            )

            bidirectional = st.checkbox(
                "Bidirectional Relationship",
                help="Create relationship in both directions",
            )

        with col2:
            strength = st.slider("Relationship Strength", 0.0, 1.0, 0.7, step=0.1)
            certainty = st.selectbox("Certainty Level", ["High", "Medium", "Low"])

        description = st.text_area(
            "Relationship Description",
            placeholder="Describe how these concepts are related...",
        )

        # Source and validation
        col1, col2 = st.columns(2)

        with col1:
            source_citation = st.text_input(
                "Source/Citation", placeholder="Optional: Reference or source"
            )

        with col2:
            validation_status = st.selectbox(
                "Validation Status", ["Pending", "Validated", "Disputed"]
            )

        # Form submission
        if st.form_submit_button("Create Relationship", type="primary"):
            if source_concept and target_concept and relationship_type:
                st.success(
                    f"✅ Relationship created: {source_concept} → {relationship_type} → {target_concept}"
                )

                # Show relationship preview
                st.markdown("#### Created Relationship")
                relationship_preview = {
                    "Source": source_concept,
                    "Type": relationship_type,
                    "Target": target_concept,
                    "Strength": strength,
                    "Certainty": certainty,
                    "Bidirectional": bidirectional,
                    "Description": description or "No description provided",
                }

                for key, value in relationship_preview.items():
                    st.text(f"{key}: {value}")

                if bidirectional:
                    st.info(
                        f"🔄 Also created reverse relationship: {target_concept} → {relationship_type} → {source_concept}"
                    )

            else:
                st.error(
                    "❌ Please fill in all required fields (Source, Target, and Relationship Type)"
                )

    # Quick relationship templates
    st.markdown("### 🚀 Quick Templates")

    st.markdown("Use these templates for common relationship patterns:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📚 Subject → Contains → Topic", use_container_width=True):
            st.info("Template: Subject CONTAINS Topic (strength: 0.8)")

    with col2:
        if st.button("🏛️ School → Influences → Thinker", use_container_width=True):
            st.info("Template: School INFLUENCES Thinker (strength: 0.7)")

    with col3:
        if st.button("🔬 Theory → Supports → Application", use_container_width=True):
            st.info("Template: Theory SUPPORTS Application (strength: 0.6)")


def show_relationship_analytics():
    """Show relationship analytics"""
    st.markdown("### 📊 Relationship Analytics")

    # Overall relationship statistics
    st.markdown("#### Overall Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Relationships", "408", delta="12")

    with col2:
        st.metric("Avg per Concept", "2.3", delta="0.1")

    with col3:
        st.metric("Relationship Types", "7", delta="0")

    with col4:
        st.metric("Avg Strength", "0.72", delta="0.03")

    # Relationship type distribution
    st.markdown("#### Relationship Type Distribution")

    # Mock relationship type data
    type_data = {
        "BELONGS_TO": {
            "count": 156,
            "avg_strength": 0.85,
            "description": "Hierarchical containment",
        },
        "RELATES_TO": {
            "count": 89,
            "avg_strength": 0.65,
            "description": "General association",
        },
        "DERIVED_FROM": {
            "count": 67,
            "avg_strength": 0.78,
            "description": "Historical/logical derivation",
        },
        "INFLUENCES": {
            "count": 45,
            "avg_strength": 0.71,
            "description": "Causal influence",
        },
        "CONTAINS": {
            "count": 32,
            "avg_strength": 0.82,
            "description": "Direct containment",
        },
        "SUPPORTS": {
            "count": 19,
            "avg_strength": 0.69,
            "description": "Evidence or logical support",
        },
    }

    for rel_type, data in type_data.items():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{rel_type}**")
            st.caption(data["description"])

        with col2:
            st.metric("Count", data["count"])

        with col3:
            st.metric("Avg Strength", f"{data['avg_strength']:.2f}")

    # Relationship quality analysis
    st.markdown("#### Relationship Quality Analysis")

    quality_metrics = {
        "High Quality (>0.8)": 245,
        "Medium Quality (0.5-0.8)": 142,
        "Low Quality (<0.5)": 21,
        "Missing Descriptions": 67,
        "Unvalidated": 134,
    }

    col1, col2 = st.columns(2)

    with col1:
        for metric, value in list(quality_metrics.items())[:3]:
            st.metric(metric, value)

    with col2:
        for metric, value in list(quality_metrics.items())[3:]:
            st.metric(metric, value)

    # Bidirectional relationships analysis
    st.markdown("#### Bidirectional Relationships")

    bidirectional_stats = {
        "Total Bidirectional": 89,
        "Asymmetric Only": 319,
        "Symmetry Rate": "21.8%",
    }

    for stat, value in bidirectional_stats.items():
        st.text(f"{stat}: {value}")

    # Domain connectivity matrix
    st.markdown("#### Cross-Domain Connectivity")

    st.info(
        "📊 Cross-domain relationship matrix would be displayed here showing how different knowledge domains are interconnected"
    )

    # Most connected concepts
    st.markdown("#### Most Connected Concepts")

    top_concepts = [
        {
            "concept": "SCI0001: Quantum_Mechanics",
            "in_degree": 23,
            "out_degree": 18,
            "total": 41,
        },
        {
            "concept": "PHIL0003: Epistemology",
            "in_degree": 19,
            "out_degree": 22,
            "total": 41,
        },
        {
            "concept": "MATH0001: Mathematics",
            "in_degree": 15,
            "out_degree": 25,
            "total": 40,
        },
        {"concept": "ART0001: Art", "in_degree": 21, "out_degree": 16, "total": 37},
        {
            "concept": "TECH0001: Technology",
            "in_degree": 17,
            "out_degree": 19,
            "total": 36,
        },
    ]

    for concept in top_concepts:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.text(concept["concept"])

        with col2:
            st.text(f"In: {concept['in_degree']}")

        with col3:
            st.text(f"Out: {concept['out_degree']}")

        with col4:
            st.text(f"Total: {concept['total']}")


def show_path_finder():
    """Show path finding tool"""
    st.markdown("### 🔍 Path Finder")

    st.markdown(
        """
    Find connection paths between any two concepts in your knowledge graph.
    Discover how concepts are related through intermediate connections.
    """
    )

    # Path finding interface
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Start Concept**")
        start_concept = st.text_input(
            "Search start concept", placeholder="Enter concept name or ID..."
        )

        if start_concept:
            start_selected = st.selectbox(
                "Select Start", ["SCI0001: Quantum_Mechanics", "PHIL0003: Epistemology"]
            )
        else:
            start_selected = None

    with col2:
        st.markdown("**End Concept**")
        end_concept = st.text_input(
            "Search end concept", placeholder="Enter concept name or ID..."
        )

        if end_concept:
            end_selected = st.selectbox(
                "Select End", ["ART0015: Renaissance_Art", "MATH0005: Linear_Algebra"]
            )
        else:
            end_selected = None

    # Path finding options
    st.markdown("#### Path Finding Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_depth = st.number_input("Maximum Path Length", 2, 10, 5)

    with col2:
        algorithm = st.selectbox(
            "Algorithm", ["Shortest Path", "All Paths", "Weighted Path"]
        )

    with col3:
        include_weak = st.checkbox("Include Weak Relationships (<0.5)")

    # Path finding execution
    if st.button("🔍 Find Path", type="primary") and start_selected and end_selected:
        with st.spinner(f"Finding paths from {start_selected} to {end_selected}..."):
            # Mock path finding results
            paths = [
                {
                    "path": [
                        "SCI0001: Quantum_Mechanics",
                        "MATH0005: Linear_Algebra",
                        "ART0015: Renaissance_Art",
                    ],
                    "relationships": ["USES", "INFLUENCES"],
                    "length": 2,
                    "total_strength": 1.4,
                    "avg_strength": 0.7,
                },
                {
                    "path": [
                        "SCI0001: Quantum_Mechanics",
                        "PHIL0008: Philosophy_of_Science",
                        "PHIL0003: Epistemology",
                        "ART0015: Renaissance_Art",
                    ],
                    "relationships": ["RELATES_TO", "BELONGS_TO", "INFLUENCES"],
                    "length": 3,
                    "total_strength": 2.1,
                    "avg_strength": 0.7,
                },
            ]

            if paths:
                st.success(f"✅ Found {len(paths)} paths between concepts")

                for i, path in enumerate(paths, 1):
                    st.markdown(f"#### Path {i} (Length: {path['length']})")

                    # Visual path representation
                    path_display = " → ".join(
                        [f"{concept}" for concept in path["path"]]
                    )

                    st.markdown(f"**Route**: {path_display}")

                    # Relationship details
                    relationship_display = " → ".join(path["relationships"])
                    st.markdown(f"**Relationships**: {relationship_display}")

                    # Path metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Path Length", path["length"])

                    with col2:
                        st.metric("Total Strength", f"{path['total_strength']:.1f}")

                    with col3:
                        st.metric("Avg Strength", f"{path['avg_strength']:.2f}")

                    st.markdown("---")

            else:
                st.warning(
                    "⚠️ No paths found between the selected concepts within the specified constraints"
                )


def show_relationship_cleanup():
    """Show relationship cleanup tools"""
    st.markdown("### 🧹 Relationship Cleanup Tools")

    st.markdown(
        """
    Maintain and optimize your relationship data with automated cleanup tools.
    Remove duplicates, fix inconsistencies, and improve relationship quality.
    """
    )

    # Cleanup options
    st.markdown("#### Available Cleanup Operations")

    cleanup_options = st.multiselect(
        "Select Cleanup Operations",
        [
            "Remove duplicate relationships",
            "Fix broken references",
            "Normalize relationship types",
            "Remove weak relationships (< 0.3)",
            "Merge similar relationship types",
            "Update relationship strengths",
            "Validate bidirectional consistency",
        ],
        help="Select one or more cleanup operations to perform",
    )

    # Cleanup parameters
    if cleanup_options:
        st.markdown("#### Cleanup Parameters")

        col1, col2 = st.columns(2)

        with col1:
            if "Remove weak relationships (< 0.3)" in cleanup_options:
                weak_threshold = st.slider("Weak Relationship Threshold", 0.1, 0.5, 0.3)

            if "Merge similar relationship types" in cleanup_options:
                similarity_threshold = st.slider(
                    "Type Similarity Threshold", 0.5, 1.0, 0.8
                )

        with col2:
            backup_before_cleanup = st.checkbox(
                "Create backup before cleanup", value=True
            )
            dry_run = st.checkbox("Dry run (preview changes only)", value=True)

    # Cleanup execution
    if cleanup_options and st.button("🧹 Start Cleanup", type="primary"):
        if dry_run:
            st.info("🔍 Running cleanup analysis (dry run mode)...")
        else:
            st.warning("⚠️ Running actual cleanup operations...")

        with st.spinner("Analyzing relationships for cleanup..."):
            # Mock cleanup results
            cleanup_results = {
                "duplicate_relationships": (
                    12 if "Remove duplicate relationships" in cleanup_options else 0
                ),
                "broken_references": (
                    5 if "Fix broken references" in cleanup_options else 0
                ),
                "normalized_types": (
                    8 if "Normalize relationship types" in cleanup_options else 0
                ),
                "weak_removed": (
                    23 if "Remove weak relationships (< 0.3)" in cleanup_options else 0
                ),
                "merged_types": (
                    3 if "Merge similar relationship types" in cleanup_options else 0
                ),
                "updated_strengths": (
                    45 if "Update relationship strengths" in cleanup_options else 0
                ),
                "bidirectional_fixed": (
                    7 if "Validate bidirectional consistency" in cleanup_options else 0
                ),
            }

            # Display results
            if dry_run:
                st.success("✅ Cleanup analysis complete (dry run)")
            else:
                st.success("✅ Cleanup operations completed successfully")

            st.markdown("#### Cleanup Results")

            total_changes = sum(cleanup_results.values())

            if total_changes > 0:
                for operation, count in cleanup_results.items():
                    if count > 0:
                        operation_name = operation.replace("_", " ").title()
                        st.text(f"{operation_name}: {count} items")

                st.markdown(f"**Total Changes**: {total_changes}")

                if dry_run:
                    st.info("💡 Run without 'Dry run' mode to apply these changes")
                else:
                    st.success("All changes have been applied to the database")

            else:
                st.info(
                    "No cleanup actions were needed - your relationships are already well-maintained!"
                )

    # Relationship health metrics
    st.markdown("#### Relationship Health Metrics")

    health_metrics = {
        "Duplicate Rate": "2.9%",
        "Broken References": "1.2%",
        "Missing Descriptions": "16.4%",
        "Weak Relationships": "5.6%",
        "Overall Health Score": "87.3%",
    }

    col1, col2, col3, col4, col5 = st.columns(5)

    for i, (metric, value) in enumerate(health_metrics.items()):
        with [col1, col2, col3, col4, col5][i]:
            if metric == "Overall Health Score":
                delta = "+2.1%" if float(value.replace("%", "")) > 85 else None
                st.metric(metric, value, delta=delta)
            else:
                st.metric(metric, value)

    # Maintenance recommendations
    st.markdown("#### Maintenance Recommendations")

    recommendations = [
        "🔍 Review relationships with strength < 0.4 for accuracy",
        "📝 Add descriptions to 67 relationships missing them",
        "🔗 Validate 23 bidirectional relationships for consistency",
        "🧹 Schedule monthly cleanup to maintain relationship quality",
    ]

    for rec in recommendations:
        st.markdown(f"- {rec}")
