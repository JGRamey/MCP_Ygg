"""Command-line interface for dependency management."""
import click
from pathlib import Path
from .requirements_manager import RequirementsManager

@click.group()
def cli():
    """Dependency management CLI."""
    pass

@cli.command()
@click.option('--project-root', type=click.Path(exists=True), default='.')
def setup(project_root):
    """Setup dependency management."""
    manager = RequirementsManager(Path(project_root))
    
    click.echo("Creating requirements.in...")
    manager.create_requirements_in()
    
    click.echo("Compiling requirements...")
    if manager.compile_requirements():
        click.echo("✅ Dependencies setup complete!")
    else:
        click.echo("❌ Failed to compile requirements", err=True)

if __name__ == '__main__':
    cli()