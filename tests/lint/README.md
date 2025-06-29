# Linting and Code Quality Tools

This directory contains all linting, formatting, and code quality tools for the MCP Yggdrasil project.

## 📁 Directory Structure

```
tests/lint/
├── README.md              # This file
├── lint_project.py        # Main linting orchestrator script
├── setup_linting.py       # One-click setup for linting infrastructure
└── requirements-dev.txt   # Development dependencies
```

## 🔧 Root Configuration Files

These files remain in the project root because linting tools expect them there:

```
# Project root files (DO NOT MOVE)
├── .flake8                    # Flake8 configuration
├── pyproject.toml            # Modern Python project config (black, isort, mypy, etc.)
├── .pre-commit-config.yaml   # Git hooks configuration
```

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Run this once to set up everything
make setup-lint
```

### 2. Daily Usage
```bash
# Check code quality
make lint

# Auto-fix formatting issues
make lint-fix

# Generate detailed report
make lint-report
```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `make lint` | Run all linting tools |
| `make lint-fix` | Auto-fix formatting issues |
| `make lint-parallel` | Run tools in parallel (faster) |
| `make lint-sequential` | Run tools one by one |
| `make lint-report` | Generate detailed report |
| `make lint-individual` | Run each tool separately |
| `make setup-lint` | Initial setup of linting infrastructure |
| `make setup-dev` | Install dev dependencies + pre-commit |

## 🔍 Linting Tools Included

### Code Formatting
- **black** - Automatic code formatting
- **isort** - Import statement sorting

### Style & Quality
- **flake8** - PEP8 style guide enforcement
- **pylint** - Comprehensive code analysis
- **ruff** - Fast modern Python linter

### Type Checking
- **mypy** - Static type analysis

### Security
- **bandit** - Security vulnerability scanning

### Git Hooks
- **pre-commit** - Automated quality checks on commit

## ⚙️ Configuration Details

### Line Length
- **88 characters** (Black standard)

### Import Organization
- **Google style** import ordering
- **Project-aware** first-party imports

### Docstring Convention
- **Google style** docstrings

### Excluded Directories
```
__pycache__/
.pytest_cache/
venv/
.venv/
build/
dist/
node_modules/
```

## 🎯 Quality Standards

The linting setup enforces:

- ✅ PEP8 compliance
- ✅ Type safety with mypy
- ✅ Security best practices
- ✅ Import organization
- ✅ Docstring presence
- ✅ Code complexity limits
- ✅ No dead code
- ✅ Consistent formatting

## 🛠️ Customization

### Adding New Tools
Edit `tests/lint/lint_project.py` and add to the `linting_tools` dictionary:

```python
"new_tool": {
    "command": ["new_tool", "args"],
    "description": "Tool description",
    "config_file": "config_file_name"
}
```

### Modifying Rules
- **flake8**: Edit `.flake8`
- **black/isort/mypy**: Edit `pyproject.toml`
- **pre-commit**: Edit `.pre-commit-config.yaml`

### Project-Specific Ignores
Add per-file ignores in `.flake8`:

```ini
per-file-ignores =
    specific_file.py: E501, F401
    tests/*.py: E501, F401, F403
```

## 🔧 Integration

### VS Code
The setup script creates `.vscode/settings.json` with optimal linting integration.

### CI/CD
Add to your GitHub Actions or other CI:

```yaml
- name: Lint Code
  run: make lint
```

### Pre-commit Hooks
Automatically installed with `make setup-lint`. Runs on every commit to prevent quality regressions.

## 📊 Reporting

### Console Output
Real-time feedback during linting with:
- ✅/❌ status indicators
- Error/warning counts
- Execution times
- Suggestions for fixes

### Detailed Reports
Generate comprehensive reports with:
```bash
make lint-report
```

Creates `lint_report.txt` with:
- Summary statistics
- Tool-by-tool results
- Sample error output
- Recommendations

## 🐛 Troubleshooting

### Tool Not Found
```bash
# Re-run setup
make setup-lint
```

### Permission Issues
```bash
# Make scripts executable
chmod +x tests/lint/*.py
```

### Configuration Conflicts
Check that root config files (`.flake8`, `pyproject.toml`) haven't been moved.

### Performance Issues
Use parallel execution:
```bash
make lint-parallel
```

## 📚 Resources

- [PEP8 Style Guide](https://pep8.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)

---

**Maintained by the MCP Yggdrasil Team**