"""
Scatter plot component for UMAP/PCA visualization.

Uses Plotly Scattergl (WebGL) for efficient rendering of 500K+ cells.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from dash import dcc


def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    color_values: Optional[Union[np.ndarray, List[str]]] = None,
    color_label: str = "",
    cell_indices: Optional[np.ndarray] = None,
    point_size: int = 3,
    opacity: float = 0.7,
    colorscale: str = "Viridis",
    show_colorbar: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    selected_indices: Optional[np.ndarray] = None,
    height: int = 400,
    config: Optional[Dict] = None,
) -> dcc.Graph:
    """
    Create a WebGL scatter plot for embedding visualization.

    Args:
        x: X coordinates (embedding dim 1)
        y: Y coordinates (embedding dim 2)
        color_values: Values for coloring points (numeric or categorical)
        color_label: Label for the color legend
        cell_indices: Original cell indices for selection tracking
        point_size: Size of scatter points
        opacity: Point opacity (0-1)
        colorscale: Plotly colorscale name for continuous data
        show_colorbar: Whether to show colorbar for continuous data
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        selected_indices: Currently selected cell indices (highlighted)
        height: Plot height in pixels
        config: Additional plot configuration

    Returns:
        dcc.Graph component with the scatter plot
    """
    if cell_indices is None:
        cell_indices = np.arange(len(x))

    # Determine if color is categorical or continuous
    is_categorical = False
    if color_values is not None:
        if isinstance(color_values, list) or (
            hasattr(color_values, "dtype") and not np.issubdtype(color_values.dtype, np.number)
        ):
            is_categorical = True

    traces = []

    if is_categorical and color_values is not None:
        # Create separate trace for each category
        unique_cats = np.unique(color_values)
        colors = _get_categorical_colors(len(unique_cats))

        for i, cat in enumerate(unique_cats):
            mask = np.array(color_values) == cat
            traces.append(
                go.Scattergl(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name=str(cat),
                    marker=dict(
                        size=point_size,
                        color=colors[i],
                        opacity=opacity,
                    ),
                    customdata=cell_indices[mask],
                    hovertemplate=f"{color_label}: {cat}<br>Index: %{{customdata}}<extra></extra>",
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.3)),
                )
            )
    else:
        # Single trace with continuous coloring
        marker_config = dict(
            size=point_size,
            opacity=opacity,
        )

        if color_values is not None:
            marker_config.update(
                color=color_values,
                colorscale=colorscale,
                showscale=show_colorbar,
                colorbar=dict(title=color_label) if show_colorbar else None,
            )

        traces.append(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=marker_config,
                customdata=cell_indices,
                hovertemplate=(
                    f"{color_label}: %{{marker.color:.2f}}<br>Index: %{{customdata}}<extra></extra>"
                    if color_values is not None
                    else "Index: %{customdata}<extra></extra>"
                ),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.3)),
            )
        )

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5) if title else None,
        xaxis=dict(
            title=x_label,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            title=y_label,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        dragmode="lasso",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ) if is_categorical else None,
        height=height,
    )

    # Highlight selected points if any
    if selected_indices is not None and len(selected_indices) > 0:
        fig.update_traces(
            selectedpoints=selected_indices.tolist(),
        )

    # Default config for interactivity
    default_config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["autoScale2d", "hoverClosestCartesian", "hoverCompareCartesian"],
        "displaylogo": False,
        "scrollZoom": True,
    }

    if config:
        default_config.update(config)

    return dcc.Graph(
        figure=fig,
        config=default_config,
        style={"height": "100%", "width": "100%"},
    )


def _get_categorical_colors(n_categories: int) -> List[str]:
    """Get a list of distinct colors for categorical data."""
    # Plotly qualitative color palette
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    if n_categories <= len(palette):
        return palette[:n_categories]

    # If we need more colors, cycle through the palette
    return [palette[i % len(palette)] for i in range(n_categories)]


def update_scatter_selection(
    figure: Dict,
    selected_indices: np.ndarray,
) -> Dict:
    """
    Update scatter plot to highlight selected points.

    Args:
        figure: Current figure dict
        selected_indices: Indices of selected cells

    Returns:
        Updated figure dict
    """
    fig = go.Figure(figure)

    for trace in fig.data:
        if hasattr(trace, "customdata") and trace.customdata is not None:
            # Find which points in this trace are selected
            trace_indices = np.array(trace.customdata)
            selected_mask = np.isin(trace_indices, selected_indices)
            selected_points = np.where(selected_mask)[0].tolist()
            trace.selectedpoints = selected_points if selected_points else None

    return fig.to_dict()
