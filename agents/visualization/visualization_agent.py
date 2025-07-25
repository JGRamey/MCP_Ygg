"""
Visualization Agent for MCP Server - Modular Architecture
Generates interactive Yggdrasil tree charts and other visualizations for Neo4j graphs.
"""

import argparse
import logging
from typing import List, Optional

import asyncio

from .core.chart_generator import ChartGenerator
from .core.config import VisualizationConfig


async def main():
    """Main CLI interface for chart generation."""

    parser = argparse.ArgumentParser(description="MCP Server Chart Generator")
    parser.add_argument(
        "--chart-type",
        choices=["yggdrasil", "network", "timeline", "domain"],
        default="yggdrasil",
        help="Type of chart to generate",
    )
    parser.add_argument("--title", help="Chart title")
    parser.add_argument("--domain", help="Domain filter")
    parser.add_argument(
        "--format", choices=["html", "svg", "png"], default="html", help="Export format"
    )
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--max-depth", type=int, default=5, help="Maximum depth for Yggdrasil tree"
    )
    parser.add_argument(
        "--node-types", nargs="+", help="Node types to include (for network graph)"
    )
    parser.add_argument(
        "--relationship-types",
        nargs="+",
        help="Relationship types to include (for network graph)",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = VisualizationConfig(args.config) if args.config else VisualizationConfig()
    generator = ChartGenerator(config)

    # Initialize generator
    await generator.initialize()

    try:
        if args.chart_type == "yggdrasil":
            title = args.title or "Yggdrasil Knowledge Tree"
            output_file = await generator.generate_yggdrasil_chart(
                title=title,
                domain_filter=args.domain,
                max_depth=args.max_depth,
                export_format=args.format,
            )
            print(f"Generated Yggdrasil chart: {output_file}")

        elif args.chart_type == "network":
            title = args.title or "Knowledge Network"
            output_file = await generator.generate_network_graph(
                node_types=args.node_types,
                relationship_types=args.relationship_types,
                title=title,
                export_format=args.format,
            )
            print(f"Generated network graph: {output_file}")

        else:
            print(f"Chart type '{args.chart_type}' not yet implemented")
            print("Available types: yggdrasil, network")

    except Exception as e:
        logging.error(f"Error generating chart: {e}")
        raise

    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main())
