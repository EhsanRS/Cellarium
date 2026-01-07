"""
UI components for Cellarium.

This package contains all the Dash/Mantine components used in the application:
- app_shell: Main layout with header, sidebar, and content areas
- sidebar: Page navigation and controls
- dashboard: Draggable panel grid and toolbar
- plots: Scatter, dotplot, violin, heatmap visualizations
- config_drawers: Parameter configuration panels
- modals: Dialogs for page creation, layout save/load
"""

# Dashboard components
from cellarium.components.dashboard import (
    create_dashboard_grid,
    create_panel_layout,
    create_panel,
    create_toolbar,
)

# Plot components
from cellarium.components.plots import (
    create_scatter_plot,
    create_dotplot,
    create_violin_plot,
    create_heatmap,
)

# Sidebar components
from cellarium.components.sidebar import create_page_navigator, create_data_info

# Configuration drawers
from cellarium.components.config_drawers import (
    create_scatter_config_drawer,
    create_expression_config_drawer,
)

# Modals
from cellarium.components.modals import (
    create_save_layout_modal,
    create_load_layout_modal,
    create_export_modal,
)

__all__ = [
    # Dashboard
    "create_dashboard_grid",
    "create_panel_layout",
    "create_panel",
    "create_toolbar",
    # Plots
    "create_scatter_plot",
    "create_dotplot",
    "create_violin_plot",
    "create_heatmap",
    # Sidebar
    "create_page_navigator",
    "create_data_info",
    # Config drawers
    "create_scatter_config_drawer",
    "create_expression_config_drawer",
    # Modals
    "create_save_layout_modal",
    "create_load_layout_modal",
    "create_export_modal",
]
