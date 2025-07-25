#!/usr/bin/env python3
"""
Enhanced Yggdrasil Integration Script
Merges Desktop Yggdrasil Excel data into enhanced CSV hybrid Neo4j+Qdrant structure
Maintains existing hierarchy, enhances ID system, and prepares for document integration
"""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedYggdrasilIntegrator:
    def __init__(self, excel_base_path: str, csv_base_path: str):
        """
        Initialize the enhanced integrator with paths to Excel and CSV directories

        Args:
            excel_base_path: Path to Desktop/Yggdrasil folder
            csv_base_path: Path to MCP_Ygg/CSV folder
        """
        self.excel_path = Path(excel_base_path)
        self.csv_path = Path(csv_base_path)

        # Enhanced domain mappings with hierarchical support
        self.domain_mappings = {
            "Art": "art",
            "Language": "language",
            "Mathematics": "mathematics",
            "Philsophy": "philosophy",  # Note: keeping original spelling
            "Science": "science",
            "Technology": "technology",
        }

        # Subdomain structure (maintaining existing nested organization)
        self.subdomain_mappings = {
            "Philosophy/Religion-Mythology": "philosophy/religion",
            "Science/PsuedoScience/Astrology": "science/pseudoscience/astrology",
        }

        # Enhanced ID system with 4-digit support
        self.id_prefixes = {
            "art": "ART",
            "language": "LANG",
            "mathematics": "MATH",
            "philosophy": "PHIL",
            "science": "SCI",
            "technology": "TECH",
            "religion": "RELIG",
            "astrology": "ASTRO",
        }

        # Track existing IDs to continue sequences
        self.existing_ids = {}
        self.load_existing_ids()

    def load_existing_ids(self):
        """Load existing concept IDs from CSV files to continue sequences with 4-digit support"""
        # Load main domain IDs
        for domain in self.domain_mappings.values():
            self._load_domain_ids(domain)

        # Load subdomain IDs
        for subdomain_path in self.subdomain_mappings.values():
            parts = subdomain_path.split("/")
            subdomain_key = parts[-1]  # e.g., 'religion', 'astrology'
            self._load_domain_ids(subdomain_key, subdomain_path)

        logger.info(f"Loaded existing IDs: {self.existing_ids}")

    def _load_domain_ids(self, domain_key: str, csv_path_suffix: str = None):
        """Helper to load IDs for a domain or subdomain with enhanced 4-digit support"""
        if csv_path_suffix:
            concepts_file = (
                self.csv_path / csv_path_suffix / f"{domain_key}_concepts.csv"
            )
        else:
            concepts_file = self.csv_path / domain_key / f"{domain_key}_concepts.csv"

        if concepts_file.exists():
            df = pd.read_csv(concepts_file)
            if not df.empty:
                prefix = self.id_prefixes.get(domain_key, domain_key.upper()[:4])

                existing_nums = []
                for concept_id in df["concept_id"]:
                    if concept_id.startswith(prefix):
                        # Enhanced to handle both 3-digit (001) and 4-digit (0001) formats
                        num_part = re.findall(r"\d+", concept_id)
                        if num_part:
                            existing_nums.append(int(num_part[0]))

                self.existing_ids[domain_key] = (
                    max(existing_nums) if existing_nums else 0
                )
            else:
                self.existing_ids[domain_key] = 0
        else:
            self.existing_ids[domain_key] = 0

    def generate_concept_id(self, domain: str) -> str:
        """Generate next sequential concept ID with enhanced 4-digit format"""
        self.existing_ids[domain] += 1
        prefix = self.id_prefixes[domain]
        # Use 4-digit format for scalability (supports up to 9999 entries)
        return f"{prefix}{self.existing_ids[domain]:04d}"

    def read_excel_schema(self, domain: str, subdomain: str = None) -> pd.DataFrame:
        """Read Excel schema file for a domain or subdomain"""
        if subdomain:
            excel_file = self.excel_path / domain / subdomain / "Schema.xlsx"
        elif domain == "Technology":
            # Special case: Technology has "Ancient Tech.xlsx" instead of "Schema.xlsx"
            excel_file = self.excel_path / domain / "Ancient Tech.xlsx"
        else:
            excel_file = self.excel_path / domain / "Schema.xlsx"

        if not excel_file.exists():
            logger.warning(f"Excel file not found: {excel_file}")
            return pd.DataFrame()

        try:
            df = pd.read_excel(excel_file)
            logger.info(f"Read {len(df)} rows from {excel_file}")
            return df
        except Exception as e:
            logger.error(f"Error reading {excel_file}: {e}")
            return pd.DataFrame()

    def map_art_schema(self, df: pd.DataFrame) -> List[Dict]:
        """Map Art Excel schema to enhanced CSV concept format"""
        concepts = []
        relationships = []

        # Group by Major Art Form
        current_major_form = None
        major_form_id = None

        for _, row in df.iterrows():
            major_form = row.get("Major Art Form", "")
            subcategory = row.get("Subcategory", "")
            description = row.get("Description", "")

            # Skip header rows
            if major_form in ["Major Art Form", ""] and subcategory in [
                "Subcategory",
                "",
            ]:
                continue

            # New major form (level 2)
            if pd.notna(major_form) and major_form.strip():
                current_major_form = major_form.strip()
                major_form_id = self.generate_concept_id("art")

                concepts.append(
                    {
                        "concept_id": major_form_id,
                        "name": current_major_form.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major category of {current_major_form.lower()}",
                        "earliest_evidence_date": -40000,  # Default to cave art
                        "earliest_evidence_type": "Archaeological",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Major art form category",
                        "research_status": "confirmed",
                    }
                )

                # Create relationship to root Art concept
                relationships.append(
                    {
                        "source_id": major_form_id,
                        "target_id": "ART0001",  # Assumes root Art concept exists
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_major_form} belongs to Art",
                        "time_period": -40000,
                    }
                )

            # Subcategory (level 3)
            if pd.notna(subcategory) and subcategory.strip() and major_form_id:
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id("art")

                concepts.append(
                    {
                        "concept_id": subcategory_id,
                        "name": subcategory_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": (
                            description
                            if pd.notna(description)
                            else f"{subcategory_clean} within {current_major_form}"
                        ),
                        "earliest_evidence_date": self.estimate_art_date(
                            subcategory_clean
                        ),
                        "earliest_evidence_type": "Historical",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Subcategory of {current_major_form}",
                        "research_status": "confirmed",
                    }
                )

                # Create relationship to major form
                relationships.append(
                    {
                        "source_id": subcategory_id,
                        "target_id": major_form_id,
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{subcategory_clean} belongs to {current_major_form}",
                        "time_period": self.estimate_art_date(subcategory_clean),
                    }
                )

        return concepts, relationships

    def estimate_art_date(self, art_form: str) -> int:
        """Estimate earliest evidence date for art forms"""
        art_dates = {
            "Paintings": -40000,
            "Cave": -40000,
            "Sculpture": -35000,
            "Drawing": -40000,
            "Music": -40000,
            "Dance": -30000,
            "Theatre": -2500,
            "Photography": 1826,
            "Digital": 1960,
            "Video": 1960,
            "Film": 1890,
            "Architecture": -10000,
            "Printmaking": 200,
            "Installation": 1950,
            "Performance": 1950,
        }

        for key, date in art_dates.items():
            if key.lower() in art_form.lower():
                return date
        return -5000  # Default ancient date

    def integrate_domain(self, domain: str, subdomain: str = None) -> bool:
        """Integrate Excel data for a specific domain or subdomain"""
        domain_name = f"{domain}/{subdomain}" if subdomain else domain
        logger.info(f"Integrating domain: {domain_name}")

        # Read Excel schema
        df = self.read_excel_schema(domain, subdomain)
        if df.empty:
            return False

        # Map to concepts and relationships based on domain
        if domain == "Art":
            new_concepts, new_relationships = self.map_art_schema(df)
            target_domain = "art"
        elif domain == "Language":
            new_concepts, new_relationships = self.map_language_schema(df)
            target_domain = "language"
        elif domain == "Mathematics":
            new_concepts, new_relationships = self.map_mathematics_schema(df)
            target_domain = "mathematics"
        elif domain == "Philsophy" and not subdomain:
            new_concepts, new_relationships = self.map_philosophy_schema(df)
            target_domain = "philosophy"
        elif domain == "Philsophy" and subdomain == "Religion-Mythology":
            new_concepts, new_relationships = self.map_religion_schema(df)
            target_domain = "religion"
        elif domain == "Science" and not subdomain:
            new_concepts, new_relationships = self.map_science_schema(df)
            target_domain = "science"
        elif domain == "Science" and subdomain == "PsuedoScience":
            new_concepts, new_relationships = self.map_astrology_schema(df)
            target_domain = "astrology"
        elif domain == "Technology":
            new_concepts, new_relationships = self.map_technology_schema(df)
            target_domain = "technology"
        else:
            logger.warning(f"No mapping defined for domain: {domain_name}")
            return False

        if not new_concepts:
            logger.warning(f"No concepts extracted for domain: {domain_name}")
            return False

        # Determine CSV path based on subdomain structure
        if subdomain and domain == "Philsophy" and subdomain == "Religion-Mythology":
            csv_path = "philosophy/religion"
            file_prefix = "religion"
        elif subdomain and domain == "Science" and subdomain == "PsuedoScience":
            csv_path = "science/pseudoscience/astrology"
            file_prefix = "astrology"
        else:
            csv_path = target_domain
            file_prefix = target_domain

        # Append to existing CSVs
        success = True
        success &= self.append_to_csv(csv_path, f"{file_prefix}_concepts", new_concepts)
        success &= self.append_to_csv(
            csv_path, f"{file_prefix}_relationships", new_relationships
        )

        return success

    def append_to_csv(self, csv_path: str, filename: str, data: List[Dict]) -> bool:
        """Append new data to existing CSV file with enhanced path handling"""
        csv_file = self.csv_path / csv_path / f"{filename}.csv"

        try:
            # Ensure directory exists
            csv_file.parent.mkdir(parents=True, exist_ok=True)

            # Read existing data
            if csv_file.exists():
                existing_df = pd.read_csv(csv_file)
            else:
                existing_df = pd.DataFrame()

            # Create new DataFrame
            new_df = pd.DataFrame(data)

            # Combine and save
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_file, index=False)

            logger.info(f"Added {len(data)} entries to {csv_file}")
            return True

        except Exception as e:
            logger.error(f"Error appending to {csv_file}: {e}")
            return False

    def run_integration(self) -> bool:
        """Run full integration process for enhanced hybrid architecture"""
        logger.info("Starting Enhanced Yggdrasil integration process...")

        success_count = 0
        total_operations = 0

        # Integrate main domains
        for excel_domain in self.domain_mappings.keys():
            total_operations += 1
            if self.integrate_domain(excel_domain):
                success_count += 1
            else:
                logger.error(f"Failed to integrate domain: {excel_domain}")

        # Integrate subdomains
        subdomain_operations = [
            ("Philsophy", "Religion-Mythology"),
            ("Science", "PsuedoScience"),
        ]

        for domain, subdomain in subdomain_operations:
            total_operations += 1
            if self.integrate_domain(domain, subdomain):
                success_count += 1
            else:
                logger.error(f"Failed to integrate subdomain: {domain}/{subdomain}")

        logger.info(
            f"Integration complete: {success_count}/{total_operations} operations successful"
        )
        return success_count == total_operations

    def map_language_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Language Excel schema to CSV concept format"""
        concepts = []
        relationships = []

        current_group = None
        group_id = None

        for _, row in df.iterrows():
            category_group = row.get("Category Group", "")
            subcategory = row.get("Subcategory", "")
            description = row.get("Description", "")

            # Skip headers
            if category_group == "Category Group":
                continue

            # New category group (level 2)
            if pd.notna(category_group) and category_group.strip():
                current_group = category_group.strip()
                group_id = self.generate_concept_id("language")

                concepts.append(
                    {
                        "concept_id": group_id,
                        "name": current_group.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major linguistic category: {current_group}",
                        "earliest_evidence_date": -100000,
                        "earliest_evidence_type": "Inferred",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Language category group",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": group_id,
                        "target_id": "LANG0001",
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_group} belongs to Language",
                        "time_period": -100000,
                    }
                )

            # Subcategory (level 3)
            if pd.notna(subcategory) and subcategory.strip() and group_id:
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id("language")

                concepts.append(
                    {
                        "concept_id": subcategory_id,
                        "name": subcategory_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": (
                            description
                            if pd.notna(description)
                            else f"{subcategory_clean} within {current_group}"
                        ),
                        "earliest_evidence_date": self.estimate_language_date(
                            subcategory_clean
                        ),
                        "earliest_evidence_type": "Historical/Linguistic",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Subcategory of {current_group}",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": subcategory_id,
                        "target_id": group_id,
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{subcategory_clean} belongs to {current_group}",
                        "time_period": self.estimate_language_date(subcategory_clean),
                    }
                )

        return concepts, relationships

    def estimate_language_date(self, language_area: str) -> int:
        """Estimate earliest evidence date for language areas"""
        lang_dates = {
            "Grammar": -50000,
            "Writing": -3100,
            "Phonetics": -50000,
            "Etymology": -3000,
            "Translation": -2000,
            "Computation": 1950,
            "Applied": 1940,
            "Pragmatics": -400,
            "Semantics": -400,
        }

        for key, date in lang_dates.items():
            if key.lower() in language_area.lower():
                return date
        return -10000

    def map_mathematics_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Mathematics Excel schema to CSV concept format"""
        concepts = []
        relationships = []

        current_category = None
        category_id = None

        for _, row in df.iterrows():
            main_category = row.get("Main Category", "")
            subcategories = row.get("Subcategories", "")

            # Skip headers
            if main_category == "Main Category":
                continue

            # New main category (level 2)
            if pd.notna(main_category) and main_category.strip():
                current_category = main_category.strip()
                category_id = self.generate_concept_id("mathematics")

                concepts.append(
                    {
                        "concept_id": category_id,
                        "name": current_category.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major mathematical category: {current_category}",
                        "earliest_evidence_date": self.estimate_math_date(
                            current_category
                        ),
                        "earliest_evidence_type": "Mathematical",
                        "location": "Global",
                        "language": "Mathematical",
                        "properties": f"Mathematical category",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": category_id,
                        "target_id": "MATH0001",
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_category} belongs to Mathematics",
                        "time_period": self.estimate_math_date(current_category),
                    }
                )

            # Parse subcategories
            if pd.notna(subcategories) and subcategories.strip() and category_id:
                subcat_items = self.parse_subcategories(subcategories)

                for subcat in subcat_items:
                    subcat_id = self.generate_concept_id("mathematics")

                    concepts.append(
                        {
                            "concept_id": subcat_id,
                            "name": subcat["name"].replace(" ", "_"),
                            "type": "branch",
                            "level": 3,
                            "description": subcat["description"],
                            "earliest_evidence_date": self.estimate_math_date(
                                subcat["name"]
                            ),
                            "earliest_evidence_type": "Mathematical",
                            "location": "Global",
                            "language": "Mathematical",
                            "properties": f"Subcategory of {current_category}",
                            "research_status": "confirmed",
                        }
                    )

                    relationships.append(
                        {
                            "source_id": subcat_id,
                            "target_id": category_id,
                            "relationship_type": "BELONGS_TO",
                            "strength": 1.0,
                            "description": f"{subcat['name']} belongs to {current_category}",
                            "time_period": self.estimate_math_date(subcat["name"]),
                        }
                    )

        return concepts, relationships

    def parse_subcategories(self, text: str) -> List[Dict]:
        """Parse formatted subcategory text like '- Algebra • Group Theory: Description'"""
        items = []
        if not text or pd.isna(text):
            return items

        # Split by bullet points or dashes
        parts = re.split(r"[•-]\s*", str(text))

        for part in parts:
            if ":" in part:
                name_desc = part.split(":", 1)
                name = name_desc[0].strip()
                description = name_desc[1].strip() if len(name_desc) > 1 else name

                if name and len(name) > 2:  # Filter out short artifacts
                    items.append({"name": name, "description": description})

        return items

    def estimate_math_date(self, math_area: str) -> int:
        """Estimate earliest evidence date for mathematical areas"""
        math_dates = {
            "Arithmetic": -20000,
            "Geometry": -3000,
            "Algebra": -1800,
            "Calculus": 1665,
            "Statistics": 1663,
            "Logic": -350,
            "Analysis": 1821,
            "Topology": 1736,
            "Number Theory": -500,
            "Combinatorics": -300,
            "Game Theory": 1944,
            "Cryptography": -400,
            "Pure": -3000,
            "Applied": -1000,
            "Interdisciplinary": -300,
        }

        for key, date in math_dates.items():
            if key.lower() in math_area.lower():
                return date
        return -1000

    def map_philosophy_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Philosophy Excel schema to CSV concept format"""
        concepts = []
        relationships = []

        current_category = None
        category_id = None

        for _, row in df.iterrows():
            main_category = row.get("Main Category", "")
            subcategory = row.get("Subcategory", "")
            description = row.get("Description", "")

            # Skip headers
            if main_category == "Main Category":
                continue

            # New main category (level 2)
            if pd.notna(main_category) and main_category.strip():
                current_category = main_category.strip()
                category_id = self.generate_concept_id("philosophy")

                concepts.append(
                    {
                        "concept_id": category_id,
                        "name": current_category.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major philosophical category: {current_category}",
                        "earliest_evidence_date": self.estimate_philosophy_date(
                            current_category
                        ),
                        "earliest_evidence_type": "Philosophical",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Philosophy category",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": category_id,
                        "target_id": "PHIL0001",
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_category} belongs to Philosophy",
                        "time_period": self.estimate_philosophy_date(current_category),
                    }
                )

            # Subcategory (level 3)
            if pd.notna(subcategory) and subcategory.strip() and category_id:
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id("philosophy")

                concepts.append(
                    {
                        "concept_id": subcategory_id,
                        "name": subcategory_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": (
                            description
                            if pd.notna(description)
                            else f"{subcategory_clean} within {current_category}"
                        ),
                        "earliest_evidence_date": self.estimate_philosophy_date(
                            subcategory_clean
                        ),
                        "earliest_evidence_type": "Philosophical",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Subcategory of {current_category}",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": subcategory_id,
                        "target_id": category_id,
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{subcategory_clean} belongs to {current_category}",
                        "time_period": self.estimate_philosophy_date(subcategory_clean),
                    }
                )

        return concepts, relationships

    def estimate_philosophy_date(self, phil_area: str) -> int:
        """Estimate earliest evidence date for philosophical areas"""
        phil_dates = {
            "Ethics": -2400,
            "Wisdom": -2600,
            "Classical": -600,
            "Metaphysics": -350,
            "Logic": -350,
            "Epistemology": -400,
            "Existentialism": 1841,
            "Political": -400,
            "Religion": -2500,
            "Eastern": -1500,
            "Modern": 1600,
            "Contemporary": 1900,
        }

        for key, date in phil_dates.items():
            if key.lower() in phil_area.lower():
                return date
        return -600

    def map_religion_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Religion-Mythology schema to CSV concept format"""
        concepts = []
        relationships = []

        current_category = None
        category_id = None

        for _, row in df.iterrows():
            main_category = row.get("Main Category", "")
            subcategories = row.get("Subcategories", "")

            # Skip headers
            if main_category == "Main Category":
                continue

            # New main category (level 2)
            if pd.notna(main_category) and main_category.strip():
                current_category = main_category.strip()
                category_id = self.generate_concept_id("religion")

                concepts.append(
                    {
                        "concept_id": category_id,
                        "name": current_category.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major religious category: {current_category}",
                        "earliest_evidence_date": self.estimate_religion_date(
                            current_category
                        ),
                        "earliest_evidence_type": "Religious",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Religion category",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": category_id,
                        "target_id": "RELIG0001",  # Assumes root Religion concept exists
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_category} belongs to Religion",
                        "time_period": self.estimate_religion_date(current_category),
                    }
                )

            # Parse subcategories
            if pd.notna(subcategories) and subcategories.strip() and category_id:
                subcat_items = self.parse_religion_subcategories(subcategories)

                for subcat in subcat_items:
                    subcat_id = self.generate_concept_id("religion")

                    concepts.append(
                        {
                            "concept_id": subcat_id,
                            "name": subcat["name"].replace(" ", "_"),
                            "type": "branch",
                            "level": 3,
                            "description": subcat["description"],
                            "earliest_evidence_date": self.estimate_religion_date(
                                subcat["name"]
                            ),
                            "earliest_evidence_type": "Religious",
                            "location": "Global",
                            "language": "Various",
                            "properties": f"Subcategory of {current_category}",
                            "research_status": "confirmed",
                        }
                    )

                    relationships.append(
                        {
                            "source_id": subcat_id,
                            "target_id": category_id,
                            "relationship_type": "BELONGS_TO",
                            "strength": 1.0,
                            "description": f"{subcat['name']} belongs to {current_category}",
                            "time_period": self.estimate_religion_date(subcat["name"]),
                        }
                    )

        return concepts, relationships

    def parse_religion_subcategories(self, text: str) -> List[Dict]:
        """Parse religion subcategory text like '- Christianity (Catholicism, Protestantism)'"""
        items = []
        if not text or pd.isna(text):
            return items

        # Split by dashes or newlines
        parts = re.split(r"[-\n]\s*", str(text))

        for part in parts:
            part = part.strip()
            if part and len(part) > 2:
                # Extract main religion name before parentheses
                if "(" in part:
                    main_name = part.split("(")[0].strip()
                    description = part
                else:
                    main_name = part
                    description = f"Religious tradition: {part}"

                if main_name:
                    items.append({"name": main_name, "description": description})

        return items

    def estimate_religion_date(self, religion_area: str) -> int:
        """Estimate earliest evidence date for religious areas"""
        religion_dates = {
            "Christianity": 30,
            "Islam": 610,
            "Judaism": -1300,
            "Hinduism": -1500,
            "Buddhism": -500,
            "Zoroastrianism": -600,
            "Greek": -800,
            "Roman": -700,
            "Norse": -200,
            "Egyptian": -3100,
            "Mesopotamian": -3500,
            "Celtic": -500,
            "Monotheistic": -1300,
            "Polytheistic": -3000,
        }

        for key, date in religion_dates.items():
            if key.lower() in religion_area.lower():
                return date
        return -2000

    def map_science_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Science Excel schema to CSV concept format"""
        concepts = []
        relationships = []

        current_discipline = None
        discipline_id = None

        for _, row in df.iterrows():
            discipline = row.get("Column 1", "") or row.get("Main Discipline", "")
            subcategory = row.get("Column 2", "") or row.get("Subcategory", "")
            description = row.get("Column 3", "") or row.get("Description", "")

            # Skip headers and empty rows
            if "Comprehensive" in str(discipline) or discipline == "Main Discipline":
                continue

            # New discipline (level 2)
            if (
                pd.notna(discipline)
                and discipline.strip()
                and discipline not in ["NaN", ""]
            ):
                current_discipline = discipline.strip()
                discipline_id = self.generate_concept_id("science")

                concepts.append(
                    {
                        "concept_id": discipline_id,
                        "name": current_discipline.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Major scientific discipline: {current_discipline}",
                        "earliest_evidence_date": self.estimate_science_date(
                            current_discipline
                        ),
                        "earliest_evidence_type": "Scientific",
                        "location": "Global",
                        "language": "Scientific",
                        "properties": f"Science discipline",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": discipline_id,
                        "target_id": "SCI0001",
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_discipline} belongs to Science",
                        "time_period": self.estimate_science_date(current_discipline),
                    }
                )

            # Subcategory (level 3)
            if (
                pd.notna(subcategory)
                and subcategory.strip()
                and discipline_id
                and subcategory not in ["NaN", ""]
            ):
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id("science")

                concepts.append(
                    {
                        "concept_id": subcategory_id,
                        "name": subcategory_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": (
                            description
                            if pd.notna(description)
                            else f"{subcategory_clean} within {current_discipline}"
                        ),
                        "earliest_evidence_date": self.estimate_science_date(
                            subcategory_clean
                        ),
                        "earliest_evidence_type": "Scientific",
                        "location": "Global",
                        "language": "Scientific",
                        "properties": f"Subcategory of {current_discipline}",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": subcategory_id,
                        "target_id": discipline_id,
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{subcategory_clean} belongs to {current_discipline}",
                        "time_period": self.estimate_science_date(subcategory_clean),
                    }
                )

        return concepts, relationships

    def estimate_science_date(self, science_area: str) -> int:
        """Estimate earliest evidence date for science areas"""
        science_dates = {
            "Astronomy": -1800,
            "Physics": -600,
            "Chemistry": -1500,
            "Biology": -400,
            "Medicine": -2600,
            "Mathematics": -20000,
            "Mechanics": -300,
            "Quantum": 1900,
            "Relativity": 1905,
            "Nuclear": 1896,
            "Organic": 1828,
            "Genetics": 1866,
            "Evolution": 1859,
            "Thermodynamics": 1824,
        }

        for key, date in science_dates.items():
            if key.lower() in science_area.lower():
                return date
        return -1000

    def map_astrology_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Astrology schema to CSV concept format"""
        concepts = []
        relationships = []

        current_category = None
        category_id = None

        for _, row in df.iterrows():
            category = row.get("Column 1", "") or row.get("Pseudoscience Category", "")
            subcategory = row.get("Column 2", "") or row.get(
                "Subcategory (if applicable)", ""
            )
            description = row.get("Column 3", "") or row.get("Description", "")

            # Skip headers
            if "Pseudoscience Category" in str(category) or category == "Column 1":
                continue

            # New category (level 2)
            if pd.notna(category) and category.strip() and category not in ["NaN", ""]:
                current_category = category.strip()
                category_id = self.generate_concept_id("astrology")

                concepts.append(
                    {
                        "concept_id": category_id,
                        "name": current_category.replace(" ", "_"),
                        "type": "sub_root",
                        "level": 2,
                        "description": f"Pseudoscience category: {current_category}",
                        "earliest_evidence_date": self.estimate_astrology_date(
                            current_category
                        ),
                        "earliest_evidence_type": "Historical",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Pseudoscience category",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": category_id,
                        "target_id": "ASTRO0001",  # Assumes root Astrology concept exists
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{current_category} belongs to Astrology",
                        "time_period": self.estimate_astrology_date(current_category),
                    }
                )

            # Subcategory (level 3)
            if (
                pd.notna(subcategory)
                and subcategory.strip()
                and category_id
                and subcategory not in ["NaN", "", "-"]
            ):
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id("astrology")

                concepts.append(
                    {
                        "concept_id": subcategory_id,
                        "name": subcategory_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": (
                            description
                            if pd.notna(description)
                            else f"{subcategory_clean} within {current_category}"
                        ),
                        "earliest_evidence_date": self.estimate_astrology_date(
                            subcategory_clean
                        ),
                        "earliest_evidence_type": "Historical",
                        "location": "Global",
                        "language": "Various",
                        "properties": f"Subcategory of {current_category}",
                        "research_status": "confirmed",
                    }
                )

                relationships.append(
                    {
                        "source_id": subcategory_id,
                        "target_id": category_id,
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{subcategory_clean} belongs to {current_category}",
                        "time_period": self.estimate_astrology_date(subcategory_clean),
                    }
                )

        return concepts, relationships

    def estimate_astrology_date(self, astro_area: str) -> int:
        """Estimate earliest evidence date for astrology areas"""
        astro_dates = {
            "Astrology": -2000,
            "Natal": -600,
            "Horary": -400,
            "Electional": -400,
            "Mundane": -600,
            "Medical": -400,
            "Homeopathy": 1796,
            "Phrenology": 1796,
            "Numerology": -500,
            "Graphology": 1622,
        }

        for key, date in astro_dates.items():
            if key.lower() in astro_area.lower():
                return date
        return -1000

    def map_technology_schema(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Map Technology (Ancient Tech) schema to CSV concept format"""
        concepts = []
        relationships = []

        for _, row in df.iterrows():
            example = row.get("Column 1", "") or row.get("Example", "")
            description = row.get("Column 2", "") or row.get("Description", "")
            relevance = row.get("Column 3", "") or row.get("Relevance to Theory", "")

            # Skip headers
            if example == "Example" or example == "Column 1":
                continue

            # Create technology concept (level 3)
            if pd.notna(example) and example.strip():
                example_clean = example.strip()
                tech_id = self.generate_concept_id("technology")

                full_description = (
                    description
                    if pd.notna(description)
                    else f"Ancient technology: {example_clean}"
                )
                if pd.notna(relevance):
                    full_description += f" {relevance}"

                concepts.append(
                    {
                        "concept_id": tech_id,
                        "name": example_clean.replace(" ", "_"),
                        "type": "branch",
                        "level": 3,
                        "description": full_description,
                        "earliest_evidence_date": self.estimate_technology_date(
                            example_clean
                        ),
                        "earliest_evidence_type": "Archaeological",
                        "location": self.extract_location(example_clean),
                        "language": "Various",
                        "properties": f"Ancient technology example",
                        "research_status": (
                            "speculative"
                            if "hypothesis" in example_clean.lower()
                            else "confirmed"
                        ),
                    }
                )

                relationships.append(
                    {
                        "source_id": tech_id,
                        "target_id": "TECH0001",  # Assumes root Technology concept exists
                        "relationship_type": "BELONGS_TO",
                        "strength": 1.0,
                        "description": f"{example_clean} belongs to Technology",
                        "time_period": self.estimate_technology_date(example_clean),
                    }
                )

        return concepts, relationships

    def estimate_technology_date(self, tech_item: str) -> int:
        """Estimate earliest evidence date for technology items"""
        tech_dates = {
            "Silurian": -65000000,  # Theoretical ancient civilization
            "Abusir": -2500,
            "Puma Punku": 600,
            "Göbekli Tepe": -9600,
            "Antikythera": -100,
            "Roman": -753,
            "Egyptian": -3100,
            "Mesopotamian": -3500,
            "Stone tools": -2600000,
        }

        for key, date in tech_dates.items():
            if key.lower() in tech_item.lower():
                return date
        return -10000

    def extract_location(self, tech_item: str) -> str:
        """Extract location from technology item name"""
        locations = {
            "Egypt": "Egypt",
            "Bolivia": "Bolivia",
            "Turkey": "Turkey",
            "Greek": "Greece",
            "Roman": "Rome",
            "Mesopotamian": "Mesopotamia",
        }

        for key, location in locations.items():
            if key.lower() in tech_item.lower():
                return location
        return "Unknown"


def main():
    """Main execution function"""
    # Define paths
    excel_path = "/Users/grant/Desktop/Yggdrasil"
    csv_path = "/Users/grant/Documents/GitHub/MCP_Ygg/CSV"

    # Initialize and run enhanced integrator
    integrator = EnhancedYggdrasilIntegrator(excel_path, csv_path)

    if integrator.run_integration():
        logger.info("✅ Enhanced Yggdrasil integration completed successfully!")
        logger.info("🔗 Hybrid Neo4j+Qdrant structure ready for document integration")
    else:
        logger.error("❌ Enhanced Yggdrasil integration failed")


if __name__ == "__main__":
    main()
