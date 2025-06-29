#!/usr/bin/env python3
"""
Setup script for linting infrastructure in MCP Yggdrasil Project
Installs dependencies, configures tools, and validates setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

def run_command(command: List[str], description: str, check: bool = True) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True, result.stdout
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False, str(e)
    except FileNotFoundError:
        print(f"   ❌ Command not found: {command[0]}")
        return False, f"Command not found: {command[0]}"

def check_python_version() -> bool:
    """Check if Python version meets requirements"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.10+ required")
        return False

def install_linting_tools() -> bool:
    """Install all linting and development tools"""
    print("\n📦 Installing linting and development tools...")
    
    # Core linting tools
    core_tools = [
        "black>=23.11.0",
        "isort>=5.12.0", 
        "flake8>=6.1.0",
        "mypy>=1.7.1",
        "pylint>=3.0.0",
        "bandit>=1.7.5",
        "ruff>=0.1.6",
    ]
    
    # Flake8 plugins
    flake8_plugins = [
        "flake8-docstrings>=1.7.0",
        "flake8-import-order>=0.18.2",
        "flake8-bugbear>=23.11.28",
        "flake8-comprehensions>=3.14.0",
        "flake8-simplify>=0.20.0",
    ]
    
    # Testing tools
    test_tools = [
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.1",
        "pytest-mock>=3.12.0",
    ]
    
    # Pre-commit
    pre_commit = ["pre-commit>=3.6.0"]
    
    all_tools = core_tools + flake8_plugins + test_tools + pre_commit
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install"] + all_tools,
        "Installing linting tools"
    )
    
    if success:
        print("   ✅ All linting tools installed successfully")
        return True
    else:
        print(f"   ❌ Installation failed: {output}")
        return False

def install_dev_requirements() -> bool:
    """Install development requirements if file exists"""
    # Check both locations for requirements-dev.txt
    requirements_locations = [
        Path("tests/lint/requirements-dev.txt"),  # New location
        Path("requirements-dev.txt")              # Legacy location
    ]
    
    for requirements_dev in requirements_locations:
        if requirements_dev.exists():
            print(f"\n📋 Installing from {requirements_dev}...")
            success, output = run_command(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_dev)],
                "Installing development requirements"
            )
            return success
    
    print("   ⚠️  requirements-dev.txt not found in any location, skipping")
    return True

def setup_pre_commit() -> bool:
    """Setup pre-commit hooks"""
    print("\n🪝 Setting up pre-commit hooks...")
    
    # Check if .pre-commit-config.yaml exists
    config_file = Path(".pre-commit-config.yaml")
    if not config_file.exists():
        print("   ⚠️  .pre-commit-config.yaml not found, skipping")
        return True
    
    # Install pre-commit hooks
    success, output = run_command(
        ["pre-commit", "install"],
        "Installing pre-commit hooks"
    )
    
    if success:
        # Update hooks to latest versions
        run_command(
            ["pre-commit", "autoupdate"],
            "Updating pre-commit hooks",
            check=False
        )
        return True
    
    return success

def download_spacy_models() -> bool:
    """Download required spaCy models"""
    print("\n🧠 Downloading spaCy models...")
    
    models = ["en_core_web_sm"]
    
    for model in models:
        success, output = run_command(
            [sys.executable, "-m", "spacy", "download", model],
            f"Downloading {model}"
        )
        if not success:
            return False
    
    return True

def validate_tool_installation() -> Dict[str, bool]:
    """Validate that all linting tools are properly installed"""
    print("\n🔍 Validating tool installation...")
    
    tools = {
        "black": ["black", "--version"],
        "isort": ["isort", "--version"],
        "flake8": ["flake8", "--version"],
        "mypy": ["mypy", "--version"],
        "pylint": ["pylint", "--version"],
        "bandit": ["bandit", "--version"],
        "ruff": ["ruff", "--version"],
        "pytest": ["pytest", "--version"],
        "pre-commit": ["pre-commit", "--version"],
    }
    
    results = {}
    
    for tool_name, command in tools.items():
        success, output = run_command(command, f"Checking {tool_name}", check=False)
        results[tool_name] = success
        
        if success:
            version = output.strip().split('\n')[0] if output else "Unknown version"
            print(f"   ✅ {tool_name}: {version}")
        else:
            print(f"   ❌ {tool_name}: Not available")
    
    return results

def create_vscode_settings() -> None:
    """Create VS Code settings for linting integration"""
    print("\n⚙️  Creating VS Code settings...")
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    settings = {
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.mypyEnabled": True,
        "python.linting.pylintEnabled": True,
        "python.linting.banditEnabled": True,
        "python.formatting.provider": "black",
        "python.sortImports.args": ["--profile", "black"],
        "editor.formatOnSave": True,
        "editor.codeActionsOnSave": {
            "source.organizeImports": True
        },
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            ".mypy_cache": True,
            ".pytest_cache": True,
            "htmlcov": True
        },
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.testing.pytestArgs": [
            "tests"
        ]
    }
    
    settings_file = vscode_dir / "settings.json"
    
    import json
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"   ✅ VS Code settings created: {settings_file}")

def run_initial_lint_check() -> bool:
    """Run initial linting to check project status"""
    print("\n🚀 Running initial lint check...")
    
    # Check if lint script exists
    lint_script = Path("scripts/lint_project.py")
    if not lint_script.exists():
        print("   ⚠️  Lint script not found, skipping initial check")
        return True
    
    success, output = run_command(
        [sys.executable, str(lint_script), "--tools", "flake8", "black"],
        "Running initial lint check",
        check=False
    )
    
    print(f"   📊 Initial lint status: {'✅ PASS' if success else '⚠️  Issues found'}")
    return True

def main():
    """Main setup function"""
    print("🔧 MCP Yggdrasil Linting Setup")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 Project root: {project_root.absolute()}")
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Install Linting Tools", install_linting_tools),
        ("Install Dev Requirements", install_dev_requirements),
        ("Setup Pre-commit Hooks", setup_pre_commit),
        ("Download spaCy Models", download_spacy_models),
        ("Validate Installation", lambda: all(validate_tool_installation().values())),
        ("Create VS Code Settings", lambda: create_vscode_settings() or True),
        ("Initial Lint Check", run_initial_lint_check),
    ]
    
    failed_steps = []
    
    for step_name, step_func in setup_steps:
        print(f"\n🔄 {step_name}...")
        try:
            success = step_func()
            if success:
                print(f"   ✅ {step_name} completed")
            else:
                print(f"   ❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            print(f"   ❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "=" * 50)
    
    if not failed_steps:
        print("🎉 Linting setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run 'make lint' to check code quality")
        print("2. Run 'make lint-fix' to auto-fix formatting issues")
        print("3. Set up your IDE to use the installed linting tools")
        print("4. Consider adding linting to your CI/CD pipeline")
        
        print("\n📚 Available commands:")
        print("• make lint          - Run all linting tools")
        print("• make lint-fix      - Auto-fix formatting issues")
        print("• make lint-report   - Generate detailed report")
        print("• pre-commit run --all-files  - Run pre-commit on all files")
    else:
        print(f"❌ Setup completed with {len(failed_steps)} failures:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\nPlease fix these issues and run the setup again.")
        sys.exit(1)

if __name__ == "__main__":
    main()