"""
Knowledge Tools Module
Advanced knowledge engineering and quality assurance components

This module provides modular components for:
- Concept building (guided wizard, templates, bulk import)
- Quality assurance (data validation, duplicate detection)
- Knowledge analytics (growth trends, network analysis)
- AI recommendations (relationship suggestions, auto-tagging)
- Relationship management (builder, analytics, path finder)

Refactored from monolithic 1,385-line file into focused components
following established patterns from Content Scraper refactoring.
"""

from .ai_recommendations import (
    show_ai_recommendations,
    show_auto_tagging,
    show_improvement_suggestions,
    show_missing_concept_suggestions,
    show_relationship_suggestions,
)

# Import all main components for easy access
from .concept_builder import (
    show_bulk_import,
    show_concept_builder,
    show_concept_cloner,
    show_guided_wizard,
    show_template_builder,
)
from .knowledge_analytics import (
    show_domain_analysis,
    show_growth_trends,
    show_knowledge_analytics,
    show_network_analysis,
    show_relationship_patterns,
)
from .quality_assurance import (
    analyze_data_quality,
    run_consistency_check,
    run_coverage_analysis,
    run_duplicate_detection,
    run_full_quality_scan,
    run_relationship_validation,
    show_quality_assurance,
)
from .relationship_manager import (
    show_path_finder,
    show_relationship_analytics,
    show_relationship_builder,
    show_relationship_cleanup,
    show_relationship_tools,
)

# Export helper functions
from .shared_utils import (
    clone_concept,
    generate_concept_id,
    generate_knowledge_report,
    import_concepts_from_df,
    import_concepts_from_text,
    validate_concept_data,
)

__all__ = [
    # Concept Builder
    "show_concept_builder",
    "show_guided_wizard",
    "show_template_builder",
    "show_bulk_import",
    "show_concept_cloner",
    # Quality Assurance
    "show_quality_assurance",
    "run_full_quality_scan",
    "run_duplicate_detection",
    "run_consistency_check",
    "run_relationship_validation",
    "run_coverage_analysis",
    "analyze_data_quality",
    # Knowledge Analytics
    "show_knowledge_analytics",
    "show_growth_trends",
    "show_network_analysis",
    "show_relationship_patterns",
    "show_domain_analysis",
    # AI Recommendations
    "show_ai_recommendations",
    "show_relationship_suggestions",
    "show_missing_concept_suggestions",
    "show_auto_tagging",
    "show_improvement_suggestions",
    # Relationship Management
    "show_relationship_tools",
    "show_relationship_builder",
    "show_relationship_analytics",
    "show_path_finder",
    "show_relationship_cleanup",
    # Shared Utilities
    "generate_concept_id",
    "validate_concept_data",
    "import_concepts_from_df",
    "import_concepts_from_text",
    "clone_concept",
    "generate_knowledge_report",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "MCP Yggdrasil Knowledge Tools"
__description__ = "Modular knowledge engineering and quality assurance tools"
