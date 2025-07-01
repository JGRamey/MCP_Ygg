#!/usr/bin/env python3
"""
CSV Cleanup Script for MCP Yggdrasil
Removes duplicates, standardizes IDs, and fixes malformed entries
"""

import pandas as pd
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVCleanupManager:
    def __init__(self, csv_base_path: str = "/Users/grant/Documents/GitHub/MCP_Ygg/CSV"):
        self.csv_base_path = Path(csv_base_path)
        self.domains = ['art', 'language', 'mathematics', 'philosophy', 'science', 'technology']
        self.special_paths = {
            'religion': 'philosophy/religion',
            'astrology': 'science/pseudoscience/astrology'
        }
        self.id_prefixes = {
            'art': 'ART',
            'language': 'LANG', 
            'mathematics': 'MATH',
            'philosophy': 'PHIL',
            'science': 'SCI',
            'technology': 'TECH',
            'religion': 'RELIG',
            'astrology': 'ASTRO'
        }
        
    def standardize_concept_id(self, concept_id: str, domain: str) -> str:
        """Standardize concept ID to DOMAIN#### format"""
        prefix = self.id_prefixes[domain]
        
        # Extract numeric part
        numeric_match = re.search(r'(\d+)', concept_id)
        if numeric_match:
            number = int(numeric_match.group(1))
            return f"{prefix}{number:04d}"
        
        logger.warning(f"Could not parse ID: {concept_id}")
        return concept_id
    
    def remove_duplicates_by_name(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Remove duplicate concepts based on name, keeping the first occurrence"""
        original_count = len(df)
        
        # Remove exact duplicates first
        df = df.drop_duplicates()
        
        # Remove duplicates by name, keeping first occurrence
        df = df.drop_duplicates(subset=['name'], keep='first')
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"{domain}: Removed {removed_count} duplicate entries")
            
        return df
    
    def fix_philosophy_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix malformed philosophy entries with proper concept names"""
        fixed_count = 0
        
        for idx, row in df.iterrows():
            name = str(row['name'])
            
            # Check for malformed entries (very long names with underscores)
            if len(name) > 50 and '_' in name:
                # Extract meaningful concept name from the beginning
                if name.startswith('Metaphysics_includes'):
                    df.loc[idx, 'name'] = 'Ontology_Note'
                    df.loc[idx, 'type'] = 'note'
                    df.loc[idx, 'level'] = 4
                elif name.startswith('Ethics_encompasses'):
                    df.loc[idx, 'name'] = 'Philosophy_of_Law_Note' 
                    df.loc[idx, 'type'] = 'note'
                    df.loc[idx, 'level'] = 4
                elif name.startswith('Philosophy_of_Religion'):
                    df.loc[idx, 'name'] = 'Religion_Mythology_Note'
                    df.loc[idx, 'type'] = 'note'
                    df.loc[idx, 'level'] = 4
                elif name.startswith('Existentialism'):
                    df.loc[idx, 'name'] = 'Modern_Philosophy_Note'
                    df.loc[idx, 'type'] = 'note'
                    df.loc[idx, 'level'] = 4
                elif name.startswith('This_table'):
                    df.loc[idx, 'name'] = 'Database_Expansion_Note'
                    df.loc[idx, 'type'] = 'note'
                    df.loc[idx, 'level'] = 4
                else:
                    # Generic cleanup for other long names
                    clean_name = name.split('_')[0:3]  # Take first 3 parts
                    df.loc[idx, 'name'] = '_'.join(clean_name)
                
                fixed_count += 1
        
        if fixed_count > 0:
            logger.info(f"Philosophy: Fixed {fixed_count} malformed entries")
            
        return df
    
    def renumber_concepts(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Renumber concept IDs sequentially starting from 0001"""
        prefix = self.id_prefixes[domain]
        
        # Sort by original ID to maintain some order
        df = df.sort_values('concept_id')
        
        # Create new sequential IDs
        for idx, (row_idx, row) in enumerate(df.iterrows(), 1):
            new_id = f"{prefix}{idx:04d}"
            df.loc[row_idx, 'concept_id'] = new_id
            
        logger.info(f"{domain}: Renumbered {len(df)} concepts")
        return df
    
    def update_relationships(self, concepts_df: pd.DataFrame, relationships_df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Update relationship IDs to match cleaned concept IDs"""
        # Create mapping from old names to new IDs
        name_to_id = dict(zip(concepts_df['name'], concepts_df['concept_id']))
        
        updated_count = 0
        
        # Update source and target IDs based on name matching
        for idx, row in relationships_df.iterrows():
            source_desc = row['description']
            target_desc = row['description']
            
            # Extract names from description
            source_match = re.search(r'^(\w+(?:_\w+)*)', source_desc)
            target_match = re.search(r'belongs to (\w+(?:_\w+)*)', source_desc)
            
            if source_match and target_match:
                source_name = source_match.group(1)
                target_name = target_match.group(1).replace(' ', '_')
                
                if source_name in name_to_id and target_name in name_to_id:
                    relationships_df.loc[idx, 'source_id'] = name_to_id[source_name]
                    relationships_df.loc[idx, 'target_id'] = name_to_id[target_name]
                    updated_count += 1
        
        logger.info(f"{domain}: Updated {updated_count} relationship IDs")
        return relationships_df
    
    def clean_domain(self, domain: str) -> bool:
        """Clean a single domain's CSV files"""
        try:
            # Handle special domain paths
            if domain in self.special_paths:
                domain_path = self.csv_base_path / self.special_paths[domain]
            else:
                domain_path = self.csv_base_path / domain
            
            concepts_file = domain_path / f"{domain}_concepts.csv"
            relationships_file = domain_path / f"{domain}_relationships.csv"
            
            if not concepts_file.exists():
                logger.warning(f"Concepts file not found: {concepts_file}")
                return False
                
            # Read concepts
            concepts_df = pd.read_csv(concepts_file)
            logger.info(f"{domain}: Loaded {len(concepts_df)} concepts")
            
            # Clean concepts
            concepts_df = self.remove_duplicates_by_name(concepts_df, domain)
            
            if domain == 'philosophy':
                concepts_df = self.fix_philosophy_entries(concepts_df)
            
            concepts_df = self.renumber_concepts(concepts_df, domain)
            
            # Save cleaned concepts
            concepts_df.to_csv(concepts_file, index=False)
            logger.info(f"{domain}: Saved {len(concepts_df)} cleaned concepts")
            
            # Clean relationships if file exists
            if relationships_file.exists():
                relationships_df = pd.read_csv(relationships_file)
                relationships_df = self.update_relationships(concepts_df, relationships_df, domain)
                relationships_df.to_csv(relationships_file, index=False)
                logger.info(f"{domain}: Updated relationships file")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {domain}: {e}")
            return False
    
    def clean_all_domains(self) -> Dict[str, bool]:
        """Clean all domain CSV files"""
        results = {}
        all_domains = self.domains + list(self.special_paths.keys())
        
        logger.info("Starting CSV cleanup for all domains...")
        
        for domain in all_domains:
            logger.info(f"\n--- Cleaning {domain.upper()} ---")
            results[domain] = self.clean_domain(domain)
        
        success_count = sum(results.values())
        logger.info(f"\n✅ Cleanup complete: {success_count}/{len(all_domains)} domains successful")
        
        return results

def main():
    """Main execution function"""
    cleanup_manager = CSVCleanupManager()
    results = cleanup_manager.clean_all_domains()
    
    # Print summary
    print("\n" + "="*50)
    print("CSV CLEANUP SUMMARY")
    print("="*50)
    
    for domain, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{domain.upper():12} | {status}")
    
    success_count = sum(results.values())
    print(f"\nTotal: {success_count}/{len(results)} domains cleaned successfully")
    
    if success_count == len(results):
        print("🎉 All domains cleaned successfully!")
    else:
        print("⚠️  Some domains had issues - check logs above")

if __name__ == "__main__":
    main()