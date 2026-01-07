"""
Dot plot component for gene expression visualization.

Shows expression levels and fraction of expressing cells
across groups (e.g., cell types).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc


def create_dotplot(
    expression_data: pd.DataFrame,
    var_names: List[str],
    groupby: str,
    groups: Optional[List[str]] = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    colorscale: str = "Reds",
    size_max: int = 15,
    title: str = "Dot Plot",
    height: int = 400,
    config: Optional[Dict] = None,
) -> dcc.Graph:
    """
    Create a dot plot for gene expression across groups.

    Args:
        expression_data: DataFrame with cells as rows, genes as columns
        var_names: List of gene names to display
        groupby: Column name in obs for grouping (e.g., 'cell_type')
        groups: List of group values (from obs[groupby])
        expression_cutoff: Minimum expression to consider a cell as expressing
        mean_only_expressed: Calculate mean only over expressing cells
        colorscale: Plotly colorscale for expression levels
        size_max: Maximum dot size in pixels
        title: Plot title
        height: Plot height in pixels
        config: Additional plot configuration

    Returns:
        dcc.Graph component with the dot plot
    """
    if groups is None or len(groups) == 0:
        # Return empty plot
        return _create_empty_dotplot(title, height)

    # Calculate statistics for each gene and group
    stats = []
    for gene in var_names:
        if gene not in expression_data.columns:
            continue

        for group in np.unique(groups):
            group_mask = np.array(groups) == group
            gene_expr = expression_data[gene].values[group_mask]

            # Calculate fraction expressing
            expressing = gene_expr > expression_cutoff
            frac_expressing = np.mean(expressing) if len(expressing) > 0 else 0

            # Calculate mean expression
            if mean_only_expressed and np.any(expressing):
                mean_expr = np.mean(gene_expr[expressing])
            else:
                mean_expr = np.mean(gene_expr) if len(gene_expr) > 0 else 0

            stats.append({
                "gene": gene,
                "group": group,
                "frac_expressing": frac_expressing,
                "mean_expression": mean_expr,
            })

    if not stats:
        return _create_empty_dotplot(title, height)

    stats_df = pd.DataFrame(stats)

    # Get unique genes and groups for axis ordering
    unique_genes = [g for g in var_names if g in stats_df["gene"].unique()]
    unique_groups = sorted(stats_df["group"].unique())

    # Create the dot plot
    fig = go.Figure()

    # Normalize sizes and colors
    max_frac = stats_df["frac_expressing"].max()
    max_expr = stats_df["mean_expression"].max()

    if max_frac == 0:
        max_frac = 1
    if max_expr == 0:
        max_expr = 1

    fig.add_trace(
        go.Scatter(
            x=stats_df["gene"],
            y=stats_df["group"],
            mode="markers",
            marker=dict(
                size=(stats_df["frac_expressing"] / max_frac) * size_max + 2,
                color=stats_df["mean_expression"],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Mean<br>Expression",
                    thickness=15,
                ),
                line=dict(width=0.5, color="DarkSlateGrey"),
            ),
            hovertemplate=(
                "Gene: %{x}<br>"
                "Group: %{y}<br>"
                "Fraction expressing: %{customdata[0]:.2%}<br>"
                "Mean expression: %{customdata[1]:.3f}<extra></extra>"
            ),
            customdata=np.column_stack([
                stats_df["frac_expressing"],
                stats_df["mean_expression"],
            ]),
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Genes",
            categoryorder="array",
            categoryarray=unique_genes,
            tickangle=45,
        ),
        yaxis=dict(
            title=groupby,
            categoryorder="array",
            categoryarray=unique_groups,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=100, r=80, t=50, b=100),
        height=height,
    )

    # Add size legend annotation
    fig.add_annotation(
        text="Size: Fraction<br>expressing",
        xref="paper",
        yref="paper",
        x=1.15,
        y=0.3,
        showarrow=False,
        font=dict(size=10),
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


def _create_empty_dotplot(title: str, height: int) -> dcc.Graph:
    """Create an empty placeholder dot plot."""
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
