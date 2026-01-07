"""
Heatmap component for gene expression matrices.

Shows expression levels across cells/groups as a color matrix.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc


def create_heatmap(
    expression_data: pd.DataFrame,
    var_names: List[str],
    groupby: Optional[str] = None,
    groups: Optional[List[str]] = None,
    aggregate: bool = True,
    colorscale: str = "RdBu_r",
    center_zero: bool = True,
    title: str = "Expression Heatmap",
    height: int = 400,
    max_cells: int = 500,
    config: Optional[Dict] = None,
) -> dcc.Graph:
    """
    Create a heatmap for gene expression visualization.

    Args:
        expression_data: DataFrame with cells as rows, genes as columns
        var_names: List of gene names to display
        groupby: Column name in obs for grouping (optional)
        groups: List of group values for each cell
        aggregate: If True and groupby provided, show mean per group
        colorscale: Plotly colorscale name
        center_zero: Whether to center colorscale at zero
        title: Plot title
        height: Plot height in pixels
        max_cells: Maximum number of cells to show (if not aggregating)
        config: Additional plot configuration

    Returns:
        dcc.Graph component with the heatmap
    """
    if len(var_names) == 0:
        return _create_empty_heatmap(title, height)

    # Filter to available genes
    available_genes = [g for g in var_names if g in expression_data.columns]
    if not available_genes:
        return _create_empty_heatmap(title, height)

    expr_matrix = expression_data[available_genes].values
    y_labels = None

    # Aggregate by group if specified
    if groupby is not None and groups is not None and aggregate:
        unique_groups = sorted(np.unique(groups))
        agg_matrix = []

        for group in unique_groups:
            group_mask = np.array(groups) == group
            group_mean = np.mean(expr_matrix[group_mask], axis=0)
            agg_matrix.append(group_mean)

        expr_matrix = np.array(agg_matrix)
        y_labels = [str(g) for g in unique_groups]
    else:
        # Limit number of cells for performance
        if expr_matrix.shape[0] > max_cells:
            indices = np.random.choice(expr_matrix.shape[0], max_cells, replace=False)
            indices = np.sort(indices)
            expr_matrix = expr_matrix[indices]
            y_labels = [f"Cell {i}" for i in indices]
        else:
            y_labels = [f"Cell {i}" for i in range(expr_matrix.shape[0])]

    # Z-score normalize for better visualization (if we have variance)
    if expr_matrix.shape[0] > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            z_matrix = (expr_matrix - np.mean(expr_matrix, axis=0)) / np.std(expr_matrix, axis=0)
            z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        z_matrix = expr_matrix

    # Determine colorscale range
    if center_zero:
        max_abs = np.max(np.abs(z_matrix))
        zmin, zmax = -max_abs, max_abs
    else:
        zmin, zmax = np.min(z_matrix), np.max(z_matrix)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=available_genes,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title="Z-score" if expr_matrix.shape[0] > 1 else "Expression",
                thickness=15,
            ),
            hovertemplate=(
                "Gene: %{x}<br>"
                "Row: %{y}<br>"
                "Value: %{z:.3f}<extra></extra>"
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Genes",
            tickangle=45,
            side="bottom",
        ),
        yaxis=dict(
            title=groupby if groupby and aggregate else "Cells",
            autorange="reversed",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=100, r=60, t=50, b=100),
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


def _create_empty_heatmap(title: str, height: int) -> dcc.Graph:
    """Create an empty placeholder heatmap."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[
            dict(
                text="Select genes to display",
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
