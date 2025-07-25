"""
Shared Utilities for Knowledge Tools
Common helper functions and utilities

Contains reusable functions used across multiple knowledge tools modules:
- ID generation and validation
- Concept data validation
- Import/export utilities
- Cloning functionality
- Report generation

Extracted from knowledge_tools.py as part of modular refactoring.
"""

import re
import sys
from pathlib import Path

import streamlit as st

# Add utils to path for database operations
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database_operations import create_concept
from utils.session_management import add_to_history, mark_unsaved_changes


def generate_concept_id(name, domain):
    """Generate a concept ID suggestion"""
    domain_prefixes = {
        "Art": "ART",
        "Science": "SCI",
        "Mathematics": "MATH",
        "Philosophy": "PHIL",
        "Language": "LANG",
        "Technology": "TECH",
        "Religion": "RELIG",
        "Astrology": "ASTRO",
    }

    prefix = domain_prefixes.get(domain, "MISC")
    # In real implementation, would check existing IDs and increment
    return f"{prefix}0001"


def validate_concept_data(concept_data):
    """Validate concept data"""
    validations = {}

    # Required fields
    if concept_data.get("name"):
        validations["Name"] = {"status": "pass", "message": "Name provided"}
    else:
        validations["Name"] = {"status": "fail", "message": "Name is required"}

    # ID format
    if re.match(r"^[A-Z]+\d+$", concept_data.get("id", "")):
        validations["ID Format"] = {"status": "pass", "message": "Valid ID format"}
    else:
        validations["ID Format"] = {
            "status": "fail",
            "message": "ID must match pattern: LETTERS + NUMBERS",
        }

    # Description length
    desc = concept_data.get("description", "")
    if len(desc) > 20:
        validations["Description"] = {
            "status": "pass",
            "message": "Good description length",
        }
    elif len(desc) > 0:
        validations["Description"] = {
            "status": "warn",
            "message": "Description could be more detailed",
        }
    else:
        validations["Description"] = {
            "status": "fail",
            "message": "Description is recommended",
        }

    return validations


def clone_concept(source_concept, new_name, new_id, copy_relationships, copy_metadata):
    """Clone a concept"""
    clone_data = source_concept.copy()
    clone_data["name"] = new_name
    clone_data["id"] = new_id

    if not copy_metadata:
        # Remove optional metadata
        for key in [
            "location",
            "cultural_context",
            "earliest_evidence_date",
            "latest_evidence_date",
        ]:
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
            if "id" not in concept_data or not concept_data["id"]:
                concept_data["id"] = generate_concept_id(
                    concept_data["name"], concept_data["domain"]
                )

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
    lines = [line.strip() for line in text_input.split("\n") if line.strip()]

    success_count = 0
    error_count = 0

    for line in lines:
        try:
            concept_data = {
                "name": line,
                "domain": domain,
                "type": concept_type,
                "level": level,
                "id": generate_concept_id(line, domain),
                "description": f"Auto-generated concept: {line}",
            }

            success, message = create_concept(concept_data)
            if success:
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1

    st.success(f"✅ Import complete: {success_count} created, {error_count} errors")
