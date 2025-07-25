"""
Shared UI Components

Reusable UI elements extracted from existing pages and components to provide
consistent styling and functionality across the Streamlit workspace.

Components:
- headers.py: Page headers and section headers
- sidebars.py: Navigation and filter sidebars
- cards.py: Metric cards, data cards, concept cards
- forms.py: Common form components
- styling.py: CSS styling and theming utilities
"""

from .cards import create_concept_card, create_data_card, create_metric_card
from .forms import create_concept_form, create_search_form, create_upload_form
from .headers import create_page_header, create_section_header
from .sidebars import create_filter_sidebar, create_navigation_sidebar
from .styling import apply_custom_css, get_theme_colors

__all__ = [
    "create_page_header",
    "create_section_header",
    "create_navigation_sidebar",
    "create_filter_sidebar",
    "create_metric_card",
    "create_data_card",
    "create_concept_card",
    "create_search_form",
    "create_upload_form",
    "create_concept_form",
    "apply_custom_css",
    "get_theme_colors",
]
