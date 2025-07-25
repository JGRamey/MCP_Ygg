"""
Trend Visualization Engine for Trend Analysis

This module provides comprehensive visualization capabilities for trend analysis results,
including time series plots, statistical summaries, comparative visualizations, and
interactive dashboards.

Key Features:
- Multiple visualization types (line plots, scatter plots, heatmaps, distribution plots)
- Statistical summary visualizations
- Seasonal pattern visualization
- Prediction and confidence interval plotting
- Multi-trend comparison plots
- Interactive and static visualization options

Author: MCP Yggdrasil Analytics Team
"""

import base64
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

from ..config import TrendConfig
from ..models import TrendAnalysis, TrendDirection, TrendPoint, TrendType

logger = logging.getLogger(__name__)


class TrendVisualizationEngine:
    """
    Advanced visualization engine for trend analysis.

    Provides multiple visualization types and formats for trend analysis results,
    including static matplotlib plots, interactive plotly visualizations,
    and export capabilities.
    """

    def __init__(
        self, config: Optional[TrendConfig] = None, output_dir: Optional[Path] = None
    ):
        """Initialize the visualization engine."""
        self.config = config or TrendConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Set up output directory
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # Visualization settings
        self.figure_size = getattr(self.config, "figure_size", (12, 8))
        self.dpi = getattr(self.config, "dpi", 300)
        self.style = getattr(self.config, "plot_style", "seaborn-v0_8")

        # Color palette
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "warning": "#F79D00",
            "neutral": "#6B7280",
        }

        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except Exception:
            plt.style.use("default")

    async def generate_trend_visualization(
        self,
        trend_analysis: TrendAnalysis,
        visualization_type: str = "comprehensive",
        save_format: str = "png",
        interactive: bool = False,
    ) -> str:
        """
        Generate visualization for trend analysis results.

        Args:
            trend_analysis: Trend analysis results
            visualization_type: Type of visualization ('comprehensive', 'simple', 'statistical')
            save_format: Output format ('png', 'svg', 'html', 'json')
            interactive: Whether to generate interactive plot

        Returns:
            Path to generated visualization file
        """
        try:
            if interactive:
                return await self._generate_interactive_visualization(
                    trend_analysis, visualization_type, save_format
                )
            else:
                return await self._generate_static_visualization(
                    trend_analysis, visualization_type, save_format
                )

        except Exception as e:
            self.logger.error(f"Error generating trend visualization: {e}")
            return ""

    async def _generate_static_visualization(
        self, trend_analysis: TrendAnalysis, visualization_type: str, save_format: str
    ) -> str:
        """Generate static matplotlib visualization."""
        try:
            if visualization_type == "comprehensive":
                fig = await self._create_comprehensive_plot(trend_analysis)
            elif visualization_type == "simple":
                fig = await self._create_simple_plot(trend_analysis)
            elif visualization_type == "statistical":
                fig = await self._create_statistical_plot(trend_analysis)
            else:
                fig = await self._create_comprehensive_plot(trend_analysis)

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_{trend_analysis.trend_type.value}_{visualization_type}_{timestamp}.{save_format}"
            filepath = self.output_dir / filename

            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", format=save_format)
            plt.close(fig)

            self.logger.info(f"Static visualization saved to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error generating static visualization: {e}")
            return ""

    async def _create_comprehensive_plot(
        self, trend_analysis: TrendAnalysis
    ) -> plt.Figure:
        """Create comprehensive multi-panel visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{trend_analysis.trend_type.value.replace('_', ' ').title()} Trend Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Main trend plot (top-left)
        ax1 = axes[0, 0]
        await self._plot_main_trend(ax1, trend_analysis)

        # Statistical summary (top-right)
        ax2 = axes[0, 1]
        await self._plot_statistical_summary(ax2, trend_analysis)

        # Distribution plot (bottom-left)
        ax3 = axes[1, 0]
        await self._plot_value_distribution(ax3, trend_analysis)

        # Trend characteristics (bottom-right)
        ax4 = axes[1, 1]
        await self._plot_trend_characteristics(ax4, trend_analysis)

        plt.tight_layout()
        return fig

    async def _create_simple_plot(self, trend_analysis: TrendAnalysis) -> plt.Figure:
        """Create simple single-panel trend plot."""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        await self._plot_main_trend(ax, trend_analysis)

        plt.title(
            f"{trend_analysis.trend_type.value.replace('_', ' ').title()} Trend",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    async def _create_statistical_plot(
        self, trend_analysis: TrendAnalysis
    ) -> plt.Figure:
        """Create statistical analysis focused plot."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Statistical Analysis", fontsize=16, fontweight="bold")

        # Statistical summary
        await self._plot_statistical_summary(axes[0], trend_analysis)

        # Distribution
        await self._plot_value_distribution(axes[1], trend_analysis)

        # Trend characteristics
        await self._plot_trend_characteristics(axes[2], trend_analysis)

        plt.tight_layout()
        return fig

    async def _plot_main_trend(self, ax: plt.Axes, trend_analysis: TrendAnalysis):
        """Plot the main trend line with predictions."""
        # Extract data
        timestamps = [point.timestamp for point in trend_analysis.data_points]
        values = [point.value for point in trend_analysis.data_points]

        # Main trend line
        ax.plot(
            timestamps,
            values,
            "o-",
            color=self.colors["primary"],
            linewidth=2,
            markersize=4,
            label="Actual",
        )

        # Add predictions if available
        if trend_analysis.predictions:
            pred_timestamps = [point.timestamp for point in trend_analysis.predictions]
            pred_values = [point.value for point in trend_analysis.predictions]
            ax.plot(
                pred_timestamps,
                pred_values,
                "--",
                color=self.colors["accent"],
                alpha=0.7,
                linewidth=2,
                label="Predicted",
            )

        # Add trend direction indicator
        direction_color = {
            TrendDirection.INCREASING: "green",
            TrendDirection.DECREASING: "red",
            TrendDirection.STABLE: "blue",
            TrendDirection.VOLATILE: "orange",
        }.get(trend_analysis.direction, "gray")

        ax.axhline(
            y=np.mean(values),
            color=direction_color,
            linestyle=":",
            alpha=0.5,
            label=f"Trend: {trend_analysis.direction.value}",
        )

        # Formatting
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, len(timestamps) // 10))
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    async def _plot_statistical_summary(
        self, ax: plt.Axes, trend_analysis: TrendAnalysis
    ):
        """Plot statistical summary as bar chart."""
        stats = trend_analysis.statistics

        # Select key statistics
        key_stats = {
            "Mean": stats.get("mean", 0),
            "Median": stats.get("median", 0),
            "Std Dev": stats.get("std", 0),
            "Min": stats.get("min", 0),
            "Max": stats.get("max", 0),
        }

        bars = ax.bar(
            key_stats.keys(),
            key_stats.values(),
            color=self.colors["secondary"],
            alpha=0.7,
        )

        # Add value labels on bars
        for bar, value in zip(bars, key_stats.values()):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Statistical Summary")
        ax.set_ylabel("Value")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    async def _plot_value_distribution(
        self, ax: plt.Axes, trend_analysis: TrendAnalysis
    ):
        """Plot value distribution as histogram."""
        values = [point.value for point in trend_analysis.data_points]

        ax.hist(
            values,
            bins=min(20, len(values) // 2),
            color=self.colors["accent"],
            alpha=0.7,
            edgecolor="black",
        )

        # Add statistics lines
        mean_val = np.mean(values)
        median_val = np.median(values)

        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax.axvline(
            median_val, color="blue", linestyle="--", label=f"Median: {median_val:.2f}"
        )

        ax.set_title("Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    async def _plot_trend_characteristics(
        self, ax: plt.Axes, trend_analysis: TrendAnalysis
    ):
        """Plot trend characteristics as radar/bar chart."""
        characteristics = {
            "Strength": trend_analysis.strength,
            "Confidence": trend_analysis.confidence,
            "Growth Rate": abs(trend_analysis.statistics.get("growth_rate", 0)),
            "Volatility": min(
                trend_analysis.statistics.get("volatility", 0)
                / trend_analysis.statistics.get("mean", 1),
                1.0,
            ),
            "Data Quality": trend_analysis.statistics.get("reliability_score", 0.5),
        }

        # Normalize all values to 0-1 scale
        normalized_chars = {}
        for key, value in characteristics.items():
            if key == "Growth Rate":
                normalized_chars[key] = min(abs(value), 1.0)
            else:
                normalized_chars[key] = min(max(value, 0.0), 1.0)

        bars = ax.bar(
            normalized_chars.keys(),
            normalized_chars.values(),
            color=self.colors["success"],
            alpha=0.7,
        )

        # Add value labels
        for bar, value in zip(bars, normalized_chars.values()):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Trend Characteristics")
        ax.set_ylabel("Score (0-1)")
        ax.set_ylim(0, 1.1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    async def _generate_interactive_visualization(
        self, trend_analysis: TrendAnalysis, visualization_type: str, save_format: str
    ) -> str:
        """Generate interactive plotly visualization."""
        try:
            # Create subplots
            if visualization_type == "comprehensive":
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Trend Analysis",
                        "Statistical Summary",
                        "Value Distribution",
                        "Trend Characteristics",
                    ),
                    specs=[
                        [{"type": "scatter"}, {"type": "bar"}],
                        [{"type": "histogram"}, {"type": "bar"}],
                    ],
                )
            else:
                fig = make_subplots(rows=1, cols=1)

            # Main trend plot
            timestamps = [point.timestamp for point in trend_analysis.data_points]
            values = [point.value for point in trend_analysis.data_points]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color=self.colors["primary"]),
                ),
                row=1,
                col=1,
            )

            # Add predictions
            if trend_analysis.predictions:
                pred_timestamps = [
                    point.timestamp for point in trend_analysis.predictions
                ]
                pred_values = [point.value for point in trend_analysis.predictions]

                fig.add_trace(
                    go.Scatter(
                        x=pred_timestamps,
                        y=pred_values,
                        mode="lines",
                        name="Predicted",
                        line=dict(color=self.colors["accent"], dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

            if visualization_type == "comprehensive":
                # Statistical summary
                stats = trend_analysis.statistics
                key_stats = ["mean", "median", "std", "min", "max"]
                stat_values = [stats.get(stat, 0) for stat in key_stats]

                fig.add_trace(
                    go.Bar(
                        x=key_stats,
                        y=stat_values,
                        name="Statistics",
                        marker_color=self.colors["secondary"],
                    ),
                    row=1,
                    col=2,
                )

                # Value distribution
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name="Distribution",
                        marker_color=self.colors["accent"],
                    ),
                    row=2,
                    col=1,
                )

                # Trend characteristics
                characteristics = {
                    "Strength": trend_analysis.strength,
                    "Confidence": trend_analysis.confidence,
                    "Growth": abs(trend_analysis.statistics.get("growth_rate", 0)),
                    "Quality": trend_analysis.statistics.get("reliability_score", 0.5),
                }

                fig.add_trace(
                    go.Bar(
                        x=list(characteristics.keys()),
                        y=list(characteristics.values()),
                        name="Characteristics",
                        marker_color=self.colors["success"],
                    ),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(
                title=f"{trend_analysis.trend_type.value.replace('_', ' ').title()} Trend Analysis",
                showlegend=True,
                height=800 if visualization_type == "comprehensive" else 500,
            )

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_{trend_analysis.trend_type.value}_{visualization_type}_{timestamp}.{save_format}"
            filepath = self.output_dir / filename

            if save_format == "html":
                fig.write_html(str(filepath))
            elif save_format == "json":
                fig.write_json(str(filepath))
            else:
                fig.write_image(str(filepath))

            self.logger.info(f"Interactive visualization saved to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error generating interactive visualization: {e}")
            return ""

    async def generate_comparison_visualization(
        self, trend_analyses: Dict[str, TrendAnalysis], save_format: str = "png"
    ) -> str:
        """Generate comparison visualization for multiple trends."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Multi-Trend Comparison", fontsize=16, fontweight="bold")

            # Trend lines comparison
            ax1 = axes[0, 0]
            colors = plt.cm.tab10(np.linspace(0, 1, len(trend_analyses)))

            for i, (label, analysis) in enumerate(trend_analyses.items()):
                timestamps = [point.timestamp for point in analysis.data_points]
                values = [point.value for point in analysis.data_points]
                ax1.plot(
                    timestamps, values, "o-", color=colors[i], label=label, alpha=0.7
                )

            ax1.set_title("Trend Comparison")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Value")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Strength comparison
            ax2 = axes[0, 1]
            labels = list(trend_analyses.keys())
            strengths = [analysis.strength for analysis in trend_analyses.values()]

            bars = ax2.bar(labels, strengths, color=colors[: len(labels)])
            ax2.set_title("Trend Strength Comparison")
            ax2.set_ylabel("Strength")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # Confidence comparison
            ax3 = axes[1, 0]
            confidences = [analysis.confidence for analysis in trend_analyses.values()]

            bars = ax3.bar(labels, confidences, color=colors[: len(labels)])
            ax3.set_title("Confidence Comparison")
            ax3.set_ylabel("Confidence")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

            # Growth rate comparison
            ax4 = axes[1, 1]
            growth_rates = [
                analysis.statistics.get("growth_rate", 0)
                for analysis in trend_analyses.values()
            ]

            bars = ax4.bar(labels, growth_rates, color=colors[: len(labels)])
            ax4.set_title("Growth Rate Comparison")
            ax4.set_ylabel("Growth Rate")
            ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_comparison_{timestamp}.{save_format}"
            filepath = self.output_dir / filename

            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", format=save_format)
            plt.close(fig)

            self.logger.info(f"Comparison visualization saved to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error generating comparison visualization: {e}")
            return ""


# Factory function for easy instantiation
def create_trend_visualization_engine(
    config: Optional[TrendConfig] = None, output_dir: Optional[Path] = None
) -> TrendVisualizationEngine:
    """
    Create and configure a TrendVisualizationEngine instance.

    Args:
        config: Optional configuration object
        output_dir: Optional output directory for visualizations

    Returns:
        Configured TrendVisualizationEngine instance
    """
    return TrendVisualizationEngine(config, output_dir)


# Export main classes and functions
__all__ = ["TrendVisualizationEngine", "create_trend_visualization_engine"]
