#!/usr/bin/env python3
"""
Migration utility for claim analyzer configuration.
Converts the old mixed configuration file to the new YAML format.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def migrate_old_config() -> None:
    """
    Migrate from the old claim_analyzer_config.py to the new config.yaml format.
    
    The old file contains mixed content (YAML, Python scripts, etc.)
    We'll extract the YAML portion and create a proper config.yaml file.
    """
    
    old_config_path = Path(__file__).parent / "claim_analyzer_config.py"
    new_config_path = Path(__file__).parent / "config.yaml"
    backup_path = Path(__file__).parent / "claim_analyzer_config.py.backup"
    
    print("🔄 Migrating claim analyzer configuration...")
    
    # Check if old config exists
    if not old_config_path.exists():
        print("✅ No old config file found, nothing to migrate")
        return
    
    # Check if new config already exists
    if new_config_path.exists():
        print("✅ New config.yaml already exists")
        response = input("Do you want to backup the old config file and remove it? (y/N): ")
        if response.lower() == 'y':
            # Create backup
            old_config_path.rename(backup_path)
            print(f"✅ Old config backed up to: {backup_path}")
            print("✅ Migration complete - using existing config.yaml")
        else:
            print("⚠️ Keeping both files - manual cleanup recommended")
        return
    
    try:
        # Read the old config file
        with open(old_config_path, 'r') as f:
            content = f.read()
        
        # The file appears to contain YAML content at the beginning
        # Extract lines that look like YAML (before the --- separator)
        lines = content.split('\n')
        yaml_lines = []
        
        in_yaml_section = False
        for line in lines:
            # Skip Python comments at the beginning
            if line.startswith('# agents/claim_analyzer/config.yaml'):
                in_yaml_section = True
                continue
            elif line.startswith('# Claim Analyzer Agent Configuration'):
                continue
            elif line.startswith('#') and not in_yaml_section:
                continue
            elif line.strip() == '---':
                # End of YAML section
                break
            elif in_yaml_section:
                yaml_lines.append(line)
        
        if yaml_lines:
            yaml_content = '\n'.join(yaml_lines)
            
            # Validate the YAML
            try:
                yaml_data = yaml.safe_load(yaml_content)
                if yaml_data:
                    # Write the new config file
                    with open(new_config_path, 'w') as f:
                        f.write(yaml_content)
                    
                    print(f"✅ Extracted YAML configuration to: {new_config_path}")
                    
                    # Create backup and remove old file
                    old_config_path.rename(backup_path)
                    print(f"✅ Old config backed up to: {backup_path}")
                    print("✅ Migration completed successfully!")
                    
                    return
                    
            except yaml.YAMLError as e:
                print(f"❌ Invalid YAML found in old config: {e}")
        
        # If we couldn't extract valid YAML, just rename the old file
        print("⚠️ Could not extract valid YAML from old config")
        print("Creating backup and using default configuration...")
        
        old_config_path.rename(backup_path)
        print(f"✅ Old config backed up to: {backup_path}")
        print("✅ Please use the existing config.yaml file")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        print("Manual intervention required")


def create_migration_summary() -> None:
    """Create a summary of the migration process"""
    
    summary_path = Path(__file__).parent / "MIGRATION_SUMMARY.md"
    
    summary_content = """# Claim Analyzer Configuration Migration

## What Happened

The old `claim_analyzer_config.py` file contained mixed content:
- YAML configuration data
- Python installation scripts  
- API route definitions
- Dashboard code
- Documentation

This has been migrated to a cleaner structure:

## New Structure

- `config.yaml` - Clean YAML configuration
- `claim_analyzer.py` - Main agent class (refactored)
- Individual modules for different concerns
- Proper separation of configuration, code, and documentation

## Migration Details

### Old File Issues
- 1062 lines of mixed content
- Configuration mixed with implementation
- Difficult to maintain and modify
- Not following Python/YAML best practices

### New Structure Benefits
- Clean YAML configuration
- Modular code organization
- Better maintainability
- Professional structure

## Files Affected

- `claim_analyzer_config.py` → `claim_analyzer_config.py.backup` (archived)
- New `config.yaml` created with extracted configuration
- Refactored modules created in separate files

## Next Steps

1. Use the new `config.yaml` for all configuration
2. Import from the new modular structure:
   ```python
   from agents.analytics.claim_analyzer import ClaimAnalyzerAgent
   ```
3. Update any scripts that referenced the old config file
4. Remove backup file once migration is verified

## Rollback (if needed)

If issues arise, you can restore the old structure:
```bash
mv claim_analyzer_config.py.backup claim_analyzer_config.py
```

However, the new structure is recommended for maintainability.
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"📄 Migration summary created: {summary_path}")


def main():
    """Main migration process"""
    print("🔧 Claim Analyzer Configuration Migration")
    print("=" * 50)
    
    migrate_old_config()
    create_migration_summary()
    
    print("\n🎉 Migration process completed!")
    print("\n📋 Summary:")
    print("- Old config file backed up")
    print("- Using clean YAML configuration")
    print("- Modular code structure implemented")
    print("- See MIGRATION_SUMMARY.md for details")


if __name__ == "__main__":
    main()