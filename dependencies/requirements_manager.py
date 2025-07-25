"""Core dependency management functionality."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import DependencyConfig


class RequirementsManager:
    """Manages project dependencies with pip-tools."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = DependencyConfig()

    def create_requirements_in(self) -> None:
        """Create requirements.in file with categorized dependencies."""
        requirements_content = self._build_requirements_content()
        requirements_path = self.project_root / "requirements.in"

        with open(requirements_path, "w") as f:
            f.write(requirements_content)

    def compile_requirements(self) -> bool:
        """Compile requirements.in to requirements.txt."""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pip-tools"], check=True
            )

            subprocess.run(
                ["pip-compile", "requirements.in", "-o", "requirements.txt"],
                check=True,
                cwd=self.project_root,
            )

            return True
        except subprocess.CalledProcessError:
            return False

    def _build_requirements_content(self) -> str:
        """Build the requirements.in content with categorized dependencies."""
        sections = [
            ("# Core server and API", self.config.CORE_DEPS),
            ("# Database connections", self.config.DATABASE_DEPS),
            ("# NLP and ML", self.config.ML_DEPS),
            ("# Web scraping", self.config.SCRAPING_DEPS),
            ("# YouTube processing", self.config.YOUTUBE_DEPS),
            ("# UI", self.config.UI_DEPS),
        ]

        content = []
        for section_name, deps in sections:
            content.append(section_name)
            for package, version in deps.items():
                content.append(f"{package}{version}")
            content.append("")  # Empty line between sections

        return "\n".join(content)
