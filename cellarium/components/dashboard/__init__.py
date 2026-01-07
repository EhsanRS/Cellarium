"""
Dashboard components for Cellarium.

Provides the main dashboard area with:
- Draggable/resizable panel grid
- Panel wrapper with controls
- Dashboard toolbar
"""

from cellarium.components.dashboard.grid import create_dashboard_grid, create_panel_layout
from cellarium.components.dashboard.panel import create_panel
from cellarium.components.dashboard.toolbar import create_toolbar

__all__ = [
    "create_dashboard_grid",
    "create_panel_layout",
    "create_panel",
    "create_toolbar",
]
