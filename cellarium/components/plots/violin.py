"""
Violin plot component for gene expression distributions.

Shows expression distribution of genes across groups.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc


def create_violin_plot(
    expression_data: pd.DataFrame,
    var_names: List[str],
    groupby: str,
    groups: Optional[List[str]] = None,
    rotation: int = 0,
    stripplot: bool = True,
    jitter: float = 0.4,
    scale: str = "width",
    title: str = "Violin Plot",
    height: int = 400,
    config: Optional[Dict] = None,
) -> dcc.Graph:
    """
    Create a violin plot for gene expression distributions.

    Args:
        expression_data: DataFrame with cells as rows, genes as columns
        var_names: List of gene names to display
        groupby: Column name in obs for grouping
        groups: List of group values for each cell
        rotation: X-axis label rotation in degrees
        stripplot: Whether to overlay strip plot points
        jitter: Amount of jitter for strip plot points
        scale: Violin scaling method ('width', 'count', 'area')
        title: Plot title
        height: Plot height in pixels
        config: Additional plot configuration

    Returns:
        dcc.Graph component with the violin plot
    """
    if groups is None or len(groups) == 0 or len(var_names) == 0:
        return _create_empty_violin(title, height)

    unique_groups = sorted(np.unique(groups))
    colors = _get_violin_colors(len(unique_groups))

    fig = go.Figure()

    # Create violins for each gene-group combination
    for i, gene in enumerate(var_names):
        if gene not in expression_data.columns:
            continue

        gene_data = expression_data[gene].values

        for j, group in enumerate(unique_groups):
            group_mask = np.array(groups) == group
            group_expr = gene_data[group_mask]

            if len(group_expr) == 0:
                continue

            # Position on x-axis
            x_pos = i + (j - len(unique_groups) / 2) * 0.15

            # Add violin
            fig.add_trace(
                go.Violin(
                    x=[gene] * len(group_expr),
                    y=group_expr,
                    name=str(group),
                    legendgroup=str(group),
                    showlegend=(i == 0),  # Only show legend for first gene
                    scalemode=scale,
                    side="both",
                    line_color=colors[j],
                    fillcolor=colors[j],
                    opacity=0.6,
                    meanline_visible=True,
                    points="all" if stripplot else False,
                    jitter=jitter if stripplot else 0,
                    pointpos=0,
                    marker=dict(
                        size=3,
                        opacity=0.6,
                    ),
                    hovertemplate=(
                        f"Gene: {gene}<br>"
                        f"Group: {group}<br>"
                        "Expression: %{y:.3f}<extra></extra>"
                    ),
                )
            )

    if len(fig.data) == 0:
        return _create_empty_violin(title, height)

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Genes",
            tickangle=rotation,
        ),
        yaxis=dict(
            title="Expression",
        ),
        violinmode="group",
        violingap=0.3,
        violingroupgap=0.1,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=50, b=80),
        legend=dict(
            title=groupby,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=height,
    )

    default_config = {
        "displayModeBar": True,
        "displaylogo": False,
    }

    if config:
        default_config.update(config)

    return dcc.Graph(
        figure=fig,
        config=default_config,
        style={"height": "100%", "width": "100%"},
    )


def _create_empty_violin(title: str, height: int) -> dcc.Graph:
    """Create an empty placeholder violin plot."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[
            dict(
                text="Select genes and grouping variable to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
        ],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
    )
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "100%", "width": "100%"},
    )


def _get_violin_colors(n: int) -> List[str]:
    """Get a list of colors for violin plots."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return [palette[i % len(palette)] for i in range(n)]
