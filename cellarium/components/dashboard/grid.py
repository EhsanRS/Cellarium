"""
Dashboard grid component using dash-draggable.

Provides a responsive, draggable grid layout for organizing
visualization panels on the dashboard.
"""

from typing import Any, Dict, List, Optional

import dash_draggable
import dash_mantine_components as dmc
from dash import html


def create_dashboard_grid(
    panels: List[html.Div],
    layout_config: Optional[Dict[str, List]] = None,
) -> html.Div:
    """
    Create a responsive draggable grid layout for dashboard panels.

    Args:
        panels: List of panel components to display
        layout_config: Dict with layout configurations for each breakpoint
                      Keys: 'lg', 'md', 'sm', 'xs'
                      Values: List of position dicts with 'i', 'x', 'y', 'w', 'h'

    Returns:
        Div containing the draggable grid
    """
    if layout_config is None:
        layout_config = get_default_layout(len(panels))

    return html.Div([
        dash_draggable.ResponsiveGridLayout(
            id="dashboard-grid",
            children=panels,
            layouts=layout_config,
            breakpoints={"lg": 1200, "md": 996, "sm": 768, "xs": 480},
            cols={"lg": 12, "md": 10, "sm": 6, "xs": 4},
            rowHeight=80,
            isDraggable=True,
            isResizable=True,
            draggableHandle=".panel-drag-handle",
            compactType="vertical",
            preventCollision=False,
            margin=[10, 10],
        ),
    ], id="dashboard-container", style={"minHeight": "500px"})


def get_default_layout(n_panels: int) -> Dict[str, List]:
    """
    Generate a default layout configuration for n panels.

    Arranges panels in a 2-column grid on large screens,
    single column on smaller screens.

    Args:
        n_panels: Number of panels

    Returns:
        Layout configuration dict
    """
    layouts = {"lg": [], "md": [], "sm": [], "xs": []}

    for i in range(n_panels):
        panel_id = f"panel-{i}"

        # Large screens: 2 columns
        layouts["lg"].append({
            "i": panel_id,
            "x": (i % 2) * 6,
            "y": (i // 2) * 4,
            "w": 6,
            "h": 4,
            "minW": 3,
            "minH": 2,
        })

        # Medium screens: 2 columns, slightly smaller
        layouts["md"].append({
            "i": panel_id,
            "x": (i % 2) * 5,
            "y": (i // 2) * 4,
            "w": 5,
            "h": 4,
            "minW": 3,
            "minH": 2,
        })

        # Small screens: single column
        layouts["sm"].append({
            "i": panel_id,
            "x": 0,
            "y": i * 4,
            "w": 6,
            "h": 4,
            "minW": 3,
            "minH": 2,
        })

        # Extra small: single column, full width
        layouts["xs"].append({
            "i": panel_id,
            "x": 0,
            "y": i * 4,
            "w": 4,
            "h": 4,
            "minW": 2,
            "minH": 2,
        })

    return layouts


def create_panel_layout(
    panel_id: str,
    x: int = 0,
    y: int = 0,
    w: int = 6,
    h: int = 4,
) -> Dict[str, Any]:
    """
    Create a layout configuration for a single panel.

    Args:
        panel_id: Unique panel identifier
        x: X position in grid units
        y: Y position in grid units
        w: Width in grid units
        h: Height in grid units

    Returns:
        Layout dict for this panel
    """
    return {
        "i": panel_id,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "minW": 3,
        "minH": 2,
    }


def update_layout_for_new_panel(
    existing_layout: Dict[str, List],
    panel_id: str,
) -> Dict[str, List]:
    """
    Add a new panel to existing layout configuration.

    Places the new panel at the bottom of the grid.

    Args:
        existing_layout: Current layout configuration
        panel_id: ID for the new panel

    Returns:
        Updated layout configuration
    """
    new_layout = {k: list(v) for k, v in existing_layout.items()}

    # Find the lowest y position in each breakpoint
    for bp in ["lg", "md", "sm", "xs"]:
        max_y = 0
        for item in new_layout[bp]:
            bottom = item["y"] + item["h"]
            max_y = max(max_y, bottom)

        # Default widths for each breakpoint
        widths = {"lg": 6, "md": 5, "sm": 6, "xs": 4}

        new_layout[bp].append({
            "i": panel_id,
            "x": 0,
            "y": max_y,
            "w": widths[bp],
            "h": 4,
            "minW": 3 if bp != "xs" else 2,
            "minH": 2,
        })

    return new_layout


def remove_panel_from_layout(
    existing_layout: Dict[str, List],
    panel_id: str,
) -> Dict[str, List]:
    """
    Remove a panel from layout configuration.

    Args:
        existing_layout: Current layout configuration
        panel_id: ID of panel to remove

    Returns:
        Updated layout configuration
    """
    new_layout = {}
    for bp, items in existing_layout.items():
        new_layout[bp] = [item for item in items if item["i"] != panel_id]
    return new_layout
