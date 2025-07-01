#!/usr/bin/env python3
"""
Yggdrasil Integration Script
Merges Desktop Yggdrasil Excel data into existing CSV knowledge graph structure
Maintains existing hierarchy, IDs, and Neo4j compatibility
"""

import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YggdrasilIntegrator:
    def __init__(self, excel_base_path: str, csv_base_path: str):
        """
        Initialize the integrator with paths to Excel and CSV directories
        
        Args:
            excel_base_path: Path to Desktop/Yggdrasil folder
            csv_base_path: Path to MCP_Ygg/CSV folder
        """
        self.excel_path = Path(excel_base_path)
        self.csv_path = Path(csv_base_path)
        self.domain_mappings = {
            'Art': 'art',
            'Language': 'language', 
            'Mathematics': 'mathematics',
            'Philsophy': 'philosophy',  # Note: keeping original spelling
            'Science': 'science',
            'Technology': 'technology'
        }
        
        # Track subdomain structure
        self.subdomain_mappings = {
            'Philosophy/Religion-Mythology': 'philosophy/religion',
            'Science/PsuedoScience/Astrology': 'science/pseudoscience/astrology'
        }
        
        # Track existing IDs to continue sequences
        self.existing_ids = {}
        self.load_existing_ids()
        
    def load_existing_ids(self):
        """Load existing concept IDs from CSV files to continue sequences"""
        for domain in self.domain_mappings.values():
            concepts_file = self.csv_path / domain / f"{domain}_concepts.csv"
            if concepts_file.exists():
                df = pd.read_csv(concepts_file)
                if not df.empty:
                    # Extract numeric part from IDs like ART001, PHIL002
                    prefix = domain.upper()[:3] if domain != 'mathematics' else 'MATH'
                    if domain == 'philosophy':
                        prefix = 'PHIL'
                    elif domain == 'science':
                        prefix = 'SCI'
                    elif domain == 'technology':
                        prefix = 'TECH'
                    elif domain == 'language':
                        prefix = 'LANG'
                    
                    existing_nums = []
                    for concept_id in df['concept_id']:
                        if concept_id.startswith(prefix):
                            num_part = re.findall(r'\d+', concept_id)
                            if num_part:
                                existing_nums.append(int(num_part[0]))
                    
                    self.existing_ids[domain] = max(existing_nums) if existing_nums else 0
                else:
                    self.existing_ids[domain] = 0
            else:
                self.existing_ids[domain] = 0
                
        logger.info(f"Loaded existing IDs: {self.existing_ids}")

    def generate_concept_id(self, domain: str) -> str:
        """Generate next sequential concept ID for domain"""
        self.existing_ids[domain] += 1
        
        prefix_map = {
            'art': 'ART',
            'language': 'LANG', 
            'mathematics': 'MATH',
            'philosophy': 'PHIL',
            'science': 'SCI',
            'technology': 'TECH'
        }
        
        prefix = prefix_map[domain]
        return f"{prefix}{self.existing_ids[domain]:03d}"

    def read_excel_schema(self, domain: str) -> pd.DataFrame:
        """Read Excel schema file for a domain"""
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
        """Map Art Excel schema to CSV concept format"""
        concepts = []
        
        # Group by Major Art Form
        current_major_form = None
        major_form_id = None
        
        for _, row in df.iterrows():
            major_form = row.get('Major Art Form', '')
            subcategory = row.get('Subcategory', '')
            description = row.get('Description', '')
            
            # Skip header rows
            if major_form in ['Major Art Form', ''] and subcategory in ['Subcategory', '']:
                continue
                
            # New major form (level 2)
            if pd.notna(major_form) and major_form.strip():
                current_major_form = major_form.strip()
                major_form_id = self.generate_concept_id('art')
                
                concepts.append({
                    'concept_id': major_form_id,
                    'name': current_major_form.replace(' ', '_'),
                    'type': 'sub_root',
                    'level': 2,
                    'description': f"Major category of {current_major_form.lower()}",
                    'earliest_evidence_date': -40000,  # Default to cave art
                    'earliest_evidence_type': 'Archaeological',
                    'location': 'Global',
                    'language': 'Various',
                    'properties': f"Major art form category",
                    'research_status': 'confirmed'
                })
                
            # Subcategory (level 3)
            if pd.notna(subcategory) and subcategory.strip() and major_form_id:
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id('art')
                
                concepts.append({
                    'concept_id': subcategory_id,
                    'name': subcategory_clean.replace(' ', '_'),
                    'type': 'branch',
                    'level': 3,
                    'description': description if pd.notna(description) else f"{subcategory_clean} within {current_major_form}",
                    'earliest_evidence_date': self.estimate_art_date(subcategory_clean),
                    'earliest_evidence_type': 'Historical',
                    'location': 'Global',
                    'language': 'Various',
                    'properties': f"Subcategory of {current_major_form}",
                    'research_status': 'confirmed'
                })
                
        return concepts

    def estimate_art_date(self, art_form: str) -> int:
        """Estimate earliest evidence date for art forms"""
        art_dates = {
            'Paintings': -40000,
            'Cave': -40000, 
            'Sculpture': -35000,
            'Drawing': -40000,
            'Music': -40000,
            'Dance': -30000,
            'Theatre': -2500,
            'Photography': 1826,
            'Digital': 1960,
            'Video': 1960,
            'Film': 1890,
            'Architecture': -10000
        }
        
        for key, date in art_dates.items():
            if key.lower() in art_form.lower():
                return date
        return -5000  # Default ancient date

    def map_language_schema(self, df: pd.DataFrame) -> List[Dict]:
        """Map Language Excel schema to CSV concept format"""
        concepts = []
        
        current_group = None
        group_id = None
        
        for _, row in df.iterrows():
            category_group = row.get('Category Group', '')
            subcategory = row.get('Subcategory', '')
            description = row.get('Description', '')
            
            # Skip headers
            if category_group == 'Category Group':
                continue
                
            # New category group (level 2)
            if pd.notna(category_group) and category_group.strip():
                current_group = category_group.strip()
                group_id = self.generate_concept_id('language')
                
                concepts.append({
                    'concept_id': group_id,
                    'name': current_group.replace(' ', '_'),
                    'type': 'sub_root',
                    'level': 2,
                    'description': f"Major linguistic category: {current_group}",
                    'earliest_evidence_date': -100000,
                    'earliest_evidence_type': 'Inferred',
                    'location': 'Global',
                    'language': 'Various',
                    'properties': f"Language category group",
                    'research_status': 'confirmed'
                })
                
            # Subcategory (level 3)
            if pd.notna(subcategory) and subcategory.strip() and group_id:
                subcategory_clean = subcategory.strip()
                subcategory_id = self.generate_concept_id('language')
                
                concepts.append({
                    'concept_id': subcategory_id,
                    'name': subcategory_clean.replace(' ', '_'),
                    'type': 'branch',
                    'level': 3,
                    'description': description if pd.notna(description) else f"{subcategory_clean} within {current_group}",
                    'earliest_evidence_date': self.estimate_language_date(subcategory_clean),
                    'earliest_evidence_type': 'Historical/Linguistic',
                    'location': 'Global',
                    'language': 'Various',
                    'properties': f"Subcategory of {current_group}",
                    'research_status': 'confirmed'
                })
                
        return concepts

    def estimate_language_date(self, language_area: str) -> int:
        """Estimate earliest evidence date for language areas"""
        lang_dates = {
            'Grammar': -50000,
            'Writing': -3100,
            'Phonetics': -50000,
            'Etymology': -3000,
            'Translation': -2000,
            'Computation': 1950,
            'Applied': 1940
        }
        
        for key, date in lang_dates.items():
            if key.lower() in language_area.lower():
                return date
        return -10000

    def map_mathematics_schema(self, df: pd.DataFrame) -> List[Dict]:
        """Map Mathematics Excel schema to CSV concept format"""
        concepts = []
        
        current_category = None
        category_id = None
        
        for _, row in df.iterrows():
            main_category = row.get('Main Category', '')
            subcategories = row.get('Subcategories', '')
            
            # Skip headers
            if main_category == 'Main Category':
                continue
                
            # New main category (level 2)
            if pd.notna(main_category) and main_category.strip():
                current_category = main_category.strip()
                category_id = self.generate_concept_id('mathematics')
                
                concepts.append({
                    'concept_id': category_id,
                    'name': current_category.replace(' ', '_'),
                    'type': 'sub_root',
                    'level': 2,
                    'description': f"Major mathematical category: {current_category}",
                    'earliest_evidence_date': self.estimate_math_date(current_category),
                    'earliest_evidence_type': 'Mathematical',
                    'location': 'Global',
                    'language': 'Mathematical',
                    'properties': f"Mathematical category",
                    'research_status': 'confirmed'
                })
                
            # Parse subcategories
            if pd.notna(subcategories) and subcategories.strip() and category_id:
                # Extract individual subcategories from formatted text
                subcat_items = self.parse_subcategories(subcategories)
                
                for subcat in subcat_items:
                    subcat_id = self.generate_concept_id('mathematics')
                    
                    concepts.append({
                        'concept_id': subcat_id,
                        'name': subcat['name'].replace(' ', '_'),
                        'type': 'branch',
                        'level': 3,
                        'description': subcat['description'],
                        'earliest_evidence_date': self.estimate_math_date(subcat['name']),
                        'earliest_evidence_type': 'Mathematical',
                        'location': 'Global',
                        'language': 'Mathematical',
                        'properties': f"Subcategory of {current_category}",
                        'research_status': 'confirmed'
                    })
                    
        return concepts

    def parse_subcategories(self, text: str) -> List[Dict]:
        """Parse formatted subcategory text like '- Algebra • Group Theory: Description'"""
        items = []
        if not text or pd.isna(text):
            return items
            
        # Split by bullet points or dashes
        parts = re.split(r'[•-]\s*', str(text))
        
        for part in parts:
            if ':' in part:
                name_desc = part.split(':', 1)
                name = name_desc[0].strip()
                description = name_desc[1].strip() if len(name_desc) > 1 else name
                
                if name and len(name) > 2:  # Filter out short artifacts
                    items.append({
                        'name': name,
                        'description': description
                    })
                    
        return items

    def estimate_math_date(self, math_area: str) -> int:
        """Estimate earliest evidence date for mathematical areas"""
        math_dates = {
            'Arithmetic': -20000,
            'Geometry': -3000,
            'Algebra': -1800,
            'Calculus': 1665,
            'Statistics': 1663,
            'Logic': -350,
            'Analysis': 1821,
            'Topology': 1736,
            'Number Theory': -500,
            'Combinatorics': -300,
            'Game Theory': 1944,
            'Cryptography': -400
        }
        
        for key, date in math_dates.items():
            if key.lower() in math_area.lower():
                return date
        return -1000

    def integrate_domain(self, domain: str) -> bool:
        """Integrate Excel data for a specific domain"""
        logger.info(f"Integrating domain: {domain}")
        
        # Read Excel schema
        df = self.read_excel_schema(domain)
        if df.empty:
            return False
            
        # Map to concepts based on domain
        if domain == 'Art':
            new_concepts = self.map_art_schema(df)
        elif domain == 'Language':
            new_concepts = self.map_language_schema(df)
        elif domain == 'Mathematics':
            new_concepts = self.map_mathematics_schema(df)
        elif domain == 'Philsophy':  # Note spelling
            new_concepts = self.map_philosophy_schema(df)
        elif domain == 'Science':
            new_concepts = self.map_science_schema(df)
        else:
            logger.warning(f"No mapping defined for domain: {domain}")
            return False
            
        if not new_concepts:
            logger.warning(f"No concepts extracted for domain: {domain}")
            return False
            
        # Append to existing CSV
        csv_domain = self.domain_mappings[domain]
        return self.append_to_csv(csv_domain, 'concepts', new_concepts)

    def append_to_csv(self, domain: str, file_type: str, data: List[Dict]) -> bool:
        """Append new data to existing CSV file"""
        csv_file = self.csv_path / domain / f"{domain}_{file_type}.csv"
        
        try:
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
        """Run full integration process"""
        logger.info("Starting Yggdrasil integration process...")
        
        success_count = 0
        total_domains = len(self.domain_mappings)
        
        for excel_domain in self.domain_mappings.keys():
            if self.integrate_domain(excel_domain):
                success_count += 1
            else:
                logger.error(f"Failed to integrate domain: {excel_domain}")
                
        logger.info(f"Integration complete: {success_count}/{total_domains} domains successful")
        return success_count == total_domains

def main():
    """Main execution function"""
    # Define paths
    excel_path = "/Users/grant/Desktop/Yggdrasil"
    csv_path = "/Users/grant/Documents/GitHub/MCP_Ygg/CSV"
    
    # Initialize and run integrator
    integrator = YggdrasilIntegrator(excel_path, csv_path)
    
    if integrator.run_integration():
        logger.info("✅ Yggdrasil integration completed successfully!")
    else:
        logger.error("❌ Yggdrasil integration failed")

if __name__ == "__main__":
    main()