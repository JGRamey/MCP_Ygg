#!/usr/bin/env python3
"""
Comprehensive Linting Script for MCP Yggdrasil Project
Runs multiple linting tools to ensure PEP8 compliance and code quality
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class LintResult:
    """Result of a linting operation"""

    tool: str
    success: bool
    output: str
    error_count: int = 0
    warning_count: int = 0
    execution_time: float = 0.0
    files_checked: int = 0


class ProjectLinter:
    """Main linting orchestrator for the MCP Yggdrasil project"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_files = self._find_python_files()
        self.results: List[LintResult] = []

        # Define source directories
        self.source_dirs = [
            "agents",
            "api",
            "streamlit_workspace",
            "scripts",
            "tests",
            "cache",
            "monitoring",
            "tasks",
            "dependencies",
        ]

        # Define linting tools and their configurations
        self.linting_tools = {
            "flake8": {
                "command": ["flake8"] + self.source_dirs,
                "description": "PEP8 style guide enforcement",
                "config_file": ".flake8",
            },
            "black": {
                "command": ["black", "--check", "--diff"] + self.source_dirs,
                "description": "Code formatting check",
                "config_file": "pyproject.toml",
            },
            "isort": {
                "command": ["isort", "--check-only", "--diff"] + self.source_dirs,
                "description": "Import sorting check",
                "config_file": "pyproject.toml",
            },
            "mypy": {
                "command": ["mypy"] + self.source_dirs,
                "description": "Static type checking",
                "config_file": "pyproject.toml",
            },
            "pylint": {
                "command": ["pylint"] + self.source_dirs,
                "description": "Comprehensive code analysis",
                "config_file": "pyproject.toml",
            },
            "bandit": {
                "command": ["bandit", "-r"] + self.source_dirs + ["-f", "json"],
                "description": "Security vulnerability scanning",
                "config_file": "pyproject.toml",
            },
            "ruff": {
                "command": ["ruff", "check"] + self.source_dirs,
                "description": "Fast Python linter (alternative to flake8)",
                "config_file": "pyproject.toml",
            },
        }

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []

        # Directories to search
        search_dirs = [
            "agents",
            "api",
            "streamlit_workspace",
            "scripts",
            "tests",
            "cache",
            "monitoring",
            "tasks",
            "dependencies",
        ]

        for dir_name in search_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                python_files.extend(dir_path.rglob("*.py"))

        # Add root level Python files
        python_files.extend(self.project_root.glob("*.py"))

        return [f for f in python_files if self._should_include_file(f)]

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in linting"""
        exclude_patterns = [
            "__pycache__",
            ".pytest_cache",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
            ".git",
            "node_modules",
            ".mypy_cache",
        ]

        return not any(pattern in str(file_path) for pattern in exclude_patterns)

    def _run_tool(self, tool_name: str, tool_config: Dict) -> LintResult:
        """Run a single linting tool"""
        print(f"🔍 Running {tool_name}: {tool_config['description']}")

        start_time = time.time()

        try:
            # Change to project root directory
            os.chdir(self.project_root)

            # Run the linting tool
            result = subprocess.run(
                tool_config["command"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            execution_time = time.time() - start_time

            # Parse output based on tool
            error_count, warning_count = self._parse_tool_output(
                tool_name, result.stdout, result.stderr
            )

            success = result.returncode == 0
            output = result.stdout if result.stdout else result.stderr

            lint_result = LintResult(
                tool=tool_name,
                success=success,
                output=output,
                error_count=error_count,
                warning_count=warning_count,
                execution_time=execution_time,
                files_checked=len(self.python_files),
            )

            status = "✅ PASS" if success else "❌ FAIL"
            print(
                f"   {status} - {error_count} errors, {warning_count} warnings ({execution_time:.2f}s)"
            )

            return lint_result

        except subprocess.TimeoutExpired:
            return LintResult(
                tool=tool_name,
                success=False,
                output="Tool execution timed out after 5 minutes",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return LintResult(
                tool=tool_name,
                success=False,
                output=f"Error running {tool_name}: {str(e)}",
                execution_time=time.time() - start_time,
            )

    def _parse_tool_output(
        self, tool_name: str, stdout: str, stderr: str
    ) -> Tuple[int, int]:
        """Parse tool output to extract error and warning counts"""
        error_count = 0
        warning_count = 0

        if tool_name == "flake8":
            # Count lines in output (each line is typically an issue)
            lines = stdout.strip().split("\n") if stdout.strip() else []
            error_count = len([line for line in lines if line.strip()])

        elif tool_name == "black":
            # Black returns non-zero exit code if formatting needed
            if "would reformat" in stdout:
                error_count = stdout.count("would reformat")

        elif tool_name == "isort":
            # isort shows files that would be reformatted
            if "would reformat" in stdout or "Fixing" in stdout:
                error_count = stdout.count("would reformat") + stdout.count("Fixing")

        elif tool_name == "mypy":
            # Count error lines in mypy output
            lines = stdout.strip().split("\n") if stdout.strip() else []
            error_count = len([line for line in lines if ": error:" in line])
            warning_count = len([line for line in lines if ": note:" in line])

        elif tool_name == "pylint":
            # Parse pylint score output
            lines = stdout.strip().split("\n") if stdout.strip() else []
            for line in lines:
                if "Your code has been rated" in line:
                    # Extract issues from pylint summary
                    if "error" in line:
                        error_count = 1  # Pylint doesn't give detailed counts easily

        elif tool_name == "bandit":
            # Parse bandit JSON output
            try:
                if stdout.strip():
                    data = json.loads(stdout)
                    error_count = len(data.get("results", []))
            except json.JSONDecodeError:
                pass

        elif tool_name == "ruff":
            # Count ruff issues
            lines = stdout.strip().split("\n") if stdout.strip() else []
            error_count = len(
                [
                    line
                    for line in lines
                    if line.strip() and not line.startswith("Found")
                ]
            )

        return error_count, warning_count

    def _check_tool_installed(self, tool_name: str) -> bool:
        """Check if a linting tool is installed"""
        try:
            result = subprocess.run(
                [tool_name, "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _install_missing_tools(self, missing_tools: List[str]) -> None:
        """Install missing linting tools"""
        if not missing_tools:
            return

        print(f"\n📦 Installing missing tools: {', '.join(missing_tools)}")

        # Create requirements for linting tools
        lint_requirements = [
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pylint>=3.0.0",
            "bandit>=1.7.0",
            "ruff>=0.1.0",
            "flake8-docstrings>=1.7.0",
            "flake8-import-order>=0.18.0",
            "flake8-bugbear>=23.0.0",
        ]

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + lint_requirements,
                check=True,
                capture_output=True,
            )
            print("✅ Successfully installed linting tools")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install tools: {e}")
            sys.exit(1)

    def run_all_tools(
        self, parallel: bool = True, tools: Optional[List[str]] = None
    ) -> List[LintResult]:
        """Run all linting tools"""
        print(
            "🚀 Starting comprehensive code quality analysis for MCP Yggdrasil Project"
        )
        print(f"📁 Project root: {self.project_root}")
        print(f"🐍 Python files found: {len(self.python_files)}")
        print(f"📂 Source directories: {', '.join(self.source_dirs)}")

        # Filter tools if specified
        tools_to_run = tools or list(self.linting_tools.keys())

        # Check which tools are installed
        missing_tools = []
        available_tools = {}

        for tool_name in tools_to_run:
            if tool_name in self.linting_tools:
                if self._check_tool_installed(tool_name):
                    available_tools[tool_name] = self.linting_tools[tool_name]
                else:
                    missing_tools.append(tool_name)

        # Install missing tools
        if missing_tools:
            self._install_missing_tools(missing_tools)
            # Re-check after installation
            for tool_name in missing_tools:
                if self._check_tool_installed(tool_name):
                    available_tools[tool_name] = self.linting_tools[tool_name]

        print(f"\n🔧 Available tools: {', '.join(available_tools.keys())}")
        print("=" * 80)

        if parallel and len(available_tools) > 1:
            # Run tools in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_tool = {
                    executor.submit(self._run_tool, tool_name, tool_config): tool_name
                    for tool_name, tool_config in available_tools.items()
                }

                for future in as_completed(future_to_tool):
                    result = future.result()
                    self.results.append(result)
        else:
            # Run tools sequentially
            for tool_name, tool_config in available_tools.items():
                result = self._run_tool(tool_name, tool_config)
                self.results.append(result)

        return self.results

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive linting report"""
        report_lines = []

        # Header
        report_lines.extend(
            [
                "=" * 80,
                "MCP YGGDRASIL PROJECT - CODE QUALITY REPORT",
                "=" * 80,
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Project Root: {self.project_root}",
                f"Python Files Analyzed: {len(self.python_files)}",
                "",
            ]
        )

        # Summary statistics
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)
        passed_tools = len([r for r in self.results if r.success])
        failed_tools = len([r for r in self.results if not r.success])

        report_lines.extend(
            [
                "📊 SUMMARY",
                "-" * 40,
                f"Tools Run: {len(self.results)}",
                f"✅ Passed: {passed_tools}",
                f"❌ Failed: {failed_tools}",
                f"🚨 Total Errors: {total_errors}",
                f"⚠️  Total Warnings: {total_warnings}",
                "",
            ]
        )

        # Tool-by-tool results
        report_lines.extend(["🔍 DETAILED RESULTS", "-" * 40])

        for result in sorted(self.results, key=lambda x: x.tool):
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.extend(
                [
                    f"{result.tool.upper()}:",
                    f"  Status: {status}",
                    f"  Errors: {result.error_count}",
                    f"  Warnings: {result.warning_count}",
                    f"  Execution Time: {result.execution_time:.2f}s",
                    f"  Files Checked: {result.files_checked}",
                    "",
                ]
            )

            # Add sample output for failed tools
            if not result.success and result.output:
                sample_output = result.output[:500]
                if len(result.output) > 500:
                    sample_output += "... (truncated)"
                report_lines.extend([f"  Sample Output:", f"  {sample_output}", ""])

        # Recommendations
        report_lines.extend(["💡 RECOMMENDATIONS", "-" * 40])

        if total_errors == 0 and total_warnings == 0:
            report_lines.append("🎉 Excellent! Your code meets all quality standards.")
        else:
            if failed_tools > 0:
                report_lines.extend(
                    [
                        "1. Fix failing linting tools first (red ❌ items above)",
                        "2. Address errors before warnings",
                        "3. Consider running tools individually for detailed output",
                    ]
                )

            if total_errors > 50:
                report_lines.append(
                    "4. Consider running 'black' and 'isort' to auto-fix formatting"
                )

            if any(r.tool == "mypy" and not r.success for r in self.results):
                report_lines.append("5. Add type hints to improve mypy compliance")

            if any(r.tool == "bandit" and r.error_count > 0 for r in self.results):
                report_lines.append("6. Review security issues found by bandit")

        report_lines.extend(
            [
                "",
                "🚀 NEXT STEPS",
                "-" * 40,
                "• Run individual tools for detailed output: flake8, black, mypy, etc.",
                "• Use auto-fixing tools: black, isort, autopep8",
                "• Set up pre-commit hooks to prevent quality regressions",
                "• Integrate linting into CI/CD pipeline",
                "",
                "=" * 80,
            ]
        )

        report = "\n".join(report_lines)

        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(report)
            print(f"📄 Report saved to: {output_path}")

        return report

    def fix_auto_fixable_issues(self) -> None:
        """Run auto-fixing tools to resolve formatting issues"""
        print("\n🔧 Running auto-fixing tools...")

        # Run black for code formatting
        try:
            subprocess.run(
                ["black"] + self.source_dirs, check=True, cwd=self.project_root
            )
            print("✅ Black formatting applied")
        except subprocess.CalledProcessError:
            print("❌ Black formatting failed")

        # Run isort for import sorting
        try:
            subprocess.run(
                ["isort"] + self.source_dirs, check=True, cwd=self.project_root
            )
            print("✅ Import sorting applied")
        except subprocess.CalledProcessError:
            print("❌ Import sorting failed")


def main():
    """Main entry point for the linting script"""
    parser = argparse.ArgumentParser(
        description="Comprehensive linting for MCP Yggdrasil Project"
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["flake8", "black", "isort", "mypy", "pylint", "bandit", "ruff"],
        help="Specific tools to run (default: all)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run tools in parallel (default: True)",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run tools sequentially"
    )
    parser.add_argument("--output", type=str, help="Output file for the report")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues with black and isort",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing linting dependencies",
    )

    args = parser.parse_args()

    # Create linter instance
    linter = ProjectLinter(project_root)

    # Auto-fix if requested
    if args.fix:
        linter.fix_auto_fixable_issues()
        print("\n" + "=" * 80)

    # Run linting
    parallel = args.parallel and not args.sequential
    results = linter.run_all_tools(parallel=parallel, tools=args.tools)

    # Generate and display report
    print("\n" + "=" * 80)
    report = linter.generate_report(args.output)
    print(report)

    # Exit with error code if any tools failed
    failed_count = len([r for r in results if not r.success])
    if failed_count > 0:
        print(f"\n❌ {failed_count} tools failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n🎉 All linting tools passed! Your code quality is excellent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
