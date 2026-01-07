"""
Panel wrapper component for dashboard visualizations.

Each panel has a header with controls (config, fullscreen, close)
and a content area for the visualization.
"""

from typing import Any, Dict, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify


def create_panel(
    panel_id: str,
    panel_type: str,
    title: str,
    content: Any,
    config: Optional[Dict] = None,
) -> html.Div:
    """
    Create a panel wrapper with header and content area.

    Args:
        panel_id: Unique identifier for this panel
        panel_type: Type of visualization ('scatter', 'dotplot', 'violin', 'heatmap')
        title: Display title for the panel header
        content: The visualization component to display
        config: Current configuration for this panel

    Returns:
        Panel component wrapped in a div with the panel_id
    """
    return html.Div(
        dmc.Paper(
            [
                # Panel header (draggable handle)
                create_panel_header(panel_id, title),

                # Panel content with loading overlay
                dmc.LoadingOverlay(
                    html.Div(
                        content,
                        style={
                            "height": "calc(100% - 44px)",
                            "padding": "8px",
                            "overflow": "hidden",
                        },
                    ),
                    id={"type": "panel-loading", "index": panel_id},
                    visible=False,
                    loaderProps={"type": "dots"},
                    overlayProps={"blur": 2},
                ),
            ],
            id={"type": "panel-paper", "index": panel_id},
            withBorder=True,
            radius="md",
            style={
                "height": "100%",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            },
            **{"data-panel-type": panel_type, "data-panel-id": panel_id},
        ),
        id=panel_id,
        style={"height": "100%"},
    )


def create_panel_header(panel_id: str, title: str) -> dmc.Group:
    """
    Create the panel header with drag handle and control buttons.

    Args:
        panel_id: Panel identifier for button IDs
        title: Title to display

    Returns:
        Group component with header content
    """
    return dmc.Group(
        [
            # Left side: drag handle and title
            dmc.Group(
                [
                    DashIconify(
                        icon="tabler:grip-vertical",
                        width=16,
                        className="panel-drag-handle",
                        style={"cursor": "grab", "color": "var(--mantine-color-dimmed)"},
                    ),
                    dmc.Text(title, size="sm", fw=500, truncate=True),
                ],
                gap="xs",
                style={"flex": 1, "overflow": "hidden"},
            ),

            # Right side: control buttons
            dmc.Group(
                [
                    # Configure button
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:settings", width=16),
                        id={"type": "panel-config-btn", "index": panel_id},
                        variant="subtle",
                        size="sm",
                        color="gray",
                    ),
                    # Fullscreen button
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:arrows-maximize", width=16),
                        id={"type": "panel-fullscreen-btn", "index": panel_id},
                        variant="subtle",
                        size="sm",
                        color="gray",
                    ),
                    # Close button
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:x", width=16),
                        id={"type": "panel-close-btn", "index": panel_id},
                        variant="subtle",
                        size="sm",
                        color="red",
                    ),
                ],
                gap=4,
            ),
        ],
        justify="space-between",
        p="xs",
        style={
            "borderBottom": "1px solid var(--mantine-color-default-border)",
            "flexShrink": 0,
        },
    )


def create_empty_panel_placeholder() -> dmc.Center:
    """
    Create a placeholder for empty panels.

    Returns:
        Centered placeholder message
    """
    return dmc.Center(
        dmc.Stack(
            [
                DashIconify(
                    icon="tabler:chart-dots-3",
                    width=48,
                    color="var(--mantine-color-dimmed)",
                ),
                dmc.Text(
                    "Configure this panel",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Text(
                    "Click the settings icon to choose visualization type",
                    size="xs",
                    c="dimmed",
                ),
            ],
            align="center",
            gap="xs",
        ),
        style={"height": "100%"},
    )


def get_panel_icon(panel_type: str) -> str:
    """
    Get the icon name for a panel type.

    Args:
        panel_type: Type of visualization

    Returns:
        Tabler icon name
    """
    icons = {
        "scatter": "tabler:chart-dots-3",
        "umap": "tabler:chart-dots-3",
        "pca": "tabler:chart-dots-2",
        "dotplot": "tabler:chart-dots",
        "violin": "tabler:chart-area",
        "heatmap": "tabler:chart-histogram",
    }
    return icons.get(panel_type, "tabler:chart-bar")


def get_panel_title(panel_type: str, config: Optional[Dict] = None) -> str:
    """
    Generate a title for a panel based on its type and config.

    Args:
        panel_type: Type of visualization
        config: Panel configuration

    Returns:
        Descriptive title string
    """
    base_titles = {
        "scatter": "Scatter Plot",
        "umap": "UMAP",
        "pca": "PCA",
        "dotplot": "Dot Plot",
        "violin": "Violin Plot",
        "heatmap": "Heatmap",
    }

    title = base_titles.get(panel_type, "Panel")

    if config:
        # Add color-by info for scatter plots
        if panel_type in ("scatter", "umap", "pca") and config.get("color"):
            title = f"{title} - {config['color']}"

        # Add gene info for expression plots
        elif panel_type in ("dotplot", "violin") and config.get("var_names"):
            n_genes = len(config["var_names"])
            title = f"{title} ({n_genes} genes)"

    return title
