"""
Dash application factory for Cellarium.

This module creates and configures the main Dash application with:
- Mantine UI components
- Multi-panel flexbox dashboard with configurable sizing
- State management stores
- Callback registration
"""

import base64
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update, ALL, MATCH, callback_context
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from cellarium.config import Config
from cellarium.data.manager import DataManager
from cellarium.data.export import export_cell_subset
from cellarium.layouts import serialize_layout, deserialize_layout
from cellarium.db.repository import PageRepository, LayoutRepository, SelectionRepository


# Notification IDs for consistent management
NOTIFICATION_IDS = {
    "page_created": "notif-page-created",
    "page_deleted": "notif-page-deleted",
    "layout_saved": "notif-layout-saved",
    "layout_loaded": "notif-layout-loaded",
    "export_success": "notif-export-success",
    "error": "notif-error",
}

# Default values for scatter plots
DEFAULT_POINT_SIZE = 2
DEFAULT_OPACITY = 0.7
DEFAULT_PANEL_WIDTH = "50%"
DEFAULT_PANEL_HEIGHT = "800px"

# Color scales - includes dark mode friendly options
COLORSCALES = [
    # Sequential - good for expression data (dark mode friendly first)
    {"value": "Viridis", "label": "Viridis (dark friendly)"},
    {"value": "Plasma", "label": "Plasma (dark friendly)"},
    {"value": "Inferno", "label": "Inferno (dark friendly)"},
    {"value": "Magma", "label": "Magma (dark friendly)"},
    {"value": "Cividis", "label": "Cividis (colorblind safe)"},
    {"value": "Turbo", "label": "Turbo (high contrast)"},
    {"value": "Blues", "label": "Blues"},
    {"value": "Reds", "label": "Reds"},
    {"value": "Greens", "label": "Greens"},
    {"value": "Purples", "label": "Purples"},
    {"value": "Oranges", "label": "Oranges"},
    {"value": "YlOrRd", "label": "Yellow-Orange-Red"},
    {"value": "YlGnBu", "label": "Yellow-Green-Blue"},
    {"value": "Hot", "label": "Hot"},
    # Diverging - good for differential expression
    {"value": "RdBu_r", "label": "Red-Blue (diverging)"},
    {"value": "RdYlBu_r", "label": "Red-Yellow-Blue (diverging)"},
    {"value": "PiYG", "label": "Pink-Green (diverging)"},
    {"value": "PRGn", "label": "Purple-Green (diverging)"},
    {"value": "BrBG", "label": "Brown-Teal (diverging)"},
]


def get_default_embedding(data_manager: DataManager) -> Optional[str]:
    """Get the preferred default embedding, preferring UMAP over PCA."""
    embeddings = data_manager.available_embeddings
    if not embeddings:
        return None

    # Prefer UMAP, then tsne, then PCA, then first available
    for preferred in ["X_umap", "X_tsne", "X_pca"]:
        if preferred in embeddings:
            return preferred

    return embeddings[0]


def persist_page_layout(layout_repo: LayoutRepository, page_id: str, page_data: dict) -> None:
    """
    Persist a page's panel configuration and layout to the database.

    This function saves the current panel configs and layout for a page,
    ensuring that changes survive app restarts.

    Args:
        layout_repo: The LayoutRepository instance
        page_id: The page ID to persist
        page_data: The page data dict containing panels and grid_layout
    """
    if page_id == "root":
        # Root page layout is also persisted
        pass

    layout_json = {
        "grid_layout": page_data.get("grid_layout", {}),
        "panels": page_data.get("panels", {}),
    }

    try:
        # Check if layout exists for this page
        existing = layout_repo.get_active_for_page(page_id)
        if existing:
            # Update existing layout
            layout_repo.update(existing.id, layout_json)
        else:
            # Create new layout
            layout_repo.create(
                page_id=page_id,
                layout_json=layout_json,
                name="Default",
                is_active=True,
            )
    except Exception as e:
        print(f"Warning: Could not persist layout for page {page_id}: {e}")


def create_app(config: Config, data_manager: DataManager) -> Dash:
    """
    Create and configure the Dash application.
    """
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="Cellarium - Single-Cell Explorer",
        update_title=None,
    )

    # Initialize repositories for persistence
    page_repo = PageRepository(config.db_path)
    layout_repo = LayoutRepository(config.db_path)
    selection_repo = SelectionRepository(config.db_path)

    # Store references for callbacks (use Flask server config, not Dash config)
    app.server.config["data_manager"] = data_manager
    app.server.config["app_config"] = config
    app.server.config["page_repo"] = page_repo
    app.server.config["layout_repo"] = layout_repo
    app.server.config["selection_repo"] = selection_repo

    # Load existing pages from database
    saved_pages = load_pages_from_db(page_repo, layout_repo, data_manager)

    # Build layout
    app.layout = create_layout(data_manager, saved_pages)

    # Register callbacks
    register_callbacks(app)

    return app


def load_pages_from_db(
    page_repo: PageRepository,
    layout_repo: LayoutRepository,
    data_manager: DataManager,
) -> Dict:
    """Load saved pages and layouts from database."""
    default_embedding = get_default_embedding(data_manager)

    # Default root page - start with half-width panel to show side-by-side is possible
    default_grid_layout = {
        "lg": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
        "md": [{"i": "panel-0", "x": 0, "y": 0, "w": 5, "h": 5, "minW": 3, "minH": 2}],
        "sm": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
        "xs": [{"i": "panel-0", "x": 0, "y": 0, "w": 4, "h": 5, "minW": 2, "minH": 2}],
    }

    default_panels = {
        "panel-0": {
            "type": "scatter",
            "config": {
                "embedding": default_embedding,
                "color_type": "obs",
                "color_obs": None,
                "color_gene": None,
                "colorscale": "Viridis",
                "point_size": DEFAULT_POINT_SIZE,
                "opacity": DEFAULT_OPACITY,
            },
            "width": DEFAULT_PANEL_WIDTH,
            "height": DEFAULT_PANEL_HEIGHT,
        }
    }

    # Check if root page has a saved layout in database
    root_grid_layout = default_grid_layout
    root_panels = default_panels
    try:
        root_layout = layout_repo.get_active_for_page("root")
        if root_layout and root_layout.layout_json:
            layout_data = root_layout.layout_json
            root_grid_layout = layout_data.get("grid_layout", default_grid_layout)
            root_panels = layout_data.get("panels", default_panels)
    except Exception:
        pass  # Use defaults if root layout lookup fails

    pages = {
        "root": {
            "id": "root",
            "name": "All Cells",
            "cell_indices": None,
            "parent_id": None,
            "n_cells": data_manager.n_cells,
            "grid_layout": root_grid_layout,
            "panels": root_panels,
        }
    }
    page_order = ["root"]

    # Load saved pages from database
    try:
        db_pages = page_repo.get_all()
        for db_page in db_pages:
            page_id = db_page.id

            # Skip if page_id already exists (e.g., "root" should not be in DB)
            if page_id in pages:
                continue

            # Get layout for this page
            active_layout = layout_repo.get_active_for_page(page_id)
            if active_layout and active_layout.layout_json:
                layout_data = active_layout.layout_json
                grid_layout = layout_data.get("grid_layout", default_grid_layout)
                panels = layout_data.get("panels", default_panels)
            else:
                grid_layout = default_grid_layout.copy()
                panels = default_panels.copy()

            # Convert cell_indices to list if numpy array
            cell_indices = db_page.cell_indices
            if cell_indices is not None and hasattr(cell_indices, "tolist"):
                cell_indices = cell_indices.tolist()

            pages[page_id] = {
                "id": page_id,
                "name": db_page.name,
                "cell_indices": cell_indices,
                "parent_id": db_page.parent_page_id,
                "n_cells": db_page.n_cells,
                "grid_layout": grid_layout,
                "panels": panels,
            }
            page_order.append(page_id)
    except Exception as e:
        # If DB load fails, just use defaults
        print(f"Warning: Could not load pages from database: {e}")

    return {
        "pages": pages,
        "page_order": page_order,
        "active_page": "root",
        "next_panel_id": 1,
    }


def create_layout(data_manager: DataManager, saved_pages: Optional[Dict] = None) -> dmc.MantineProvider:
    """Create the main application layout."""
    # Use saved pages or create defaults
    if saved_pages:
        pages_data = saved_pages
        root_page = pages_data["pages"]["root"]
        initial_panels = root_page["panels"]
        initial_grid_layout = root_page["grid_layout"]
    else:
        default_embedding = get_default_embedding(data_manager)

        # Auto-detect gene symbols and raw data settings
        detected_symbols_col = data_manager.detect_gene_symbols_column()
        use_raw_default = data_manager.has_raw

        # Initial panel configuration with auto-detected defaults
        initial_panels = {
            "panel-0": {
                "type": "scatter",
                "config": {
                    "embedding": default_embedding,
                    "color_type": "obs",
                    "color_obs": None,
                    "color_gene": None,
                    "gene_symbols_col": detected_symbols_col,
                    "use_raw": use_raw_default,
                    "colorscale": "Viridis",
                    "vmax_type": "auto",
                    "vmax_value": 99.9,
                    "point_size": DEFAULT_POINT_SIZE,
                    "opacity": DEFAULT_OPACITY,
                    "groups_column": None,
                    "groups_values": [],
                },
            }
        }

        initial_grid_layout = {
            "lg": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
            "md": [{"i": "panel-0", "x": 0, "y": 0, "w": 5, "h": 5, "minW": 3, "minH": 2}],
            "sm": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
            "xs": [{"i": "panel-0", "x": 0, "y": 0, "w": 4, "h": 5, "minW": 2, "minH": 2}],
        }

        pages_data = {
            "pages": {
                "root": {
                    "id": "root",
                    "name": "All Cells",
                    "cell_indices": None,
                    "parent_id": None,
                    "n_cells": data_manager.n_cells,
                    "grid_layout": initial_grid_layout,
                    "panels": initial_panels,
                }
            },
            "active_page": "root",
            "page_order": ["root"],
            "next_panel_id": 1,
        }

    return dmc.MantineProvider(
        id="mantine-provider",
        theme={
            "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "primaryColor": "blue",
            "components": {
                "Button": {"defaultProps": {"radius": "md"}},
                "Paper": {"defaultProps": {"radius": "md"}},
                "Select": {"defaultProps": {"radius": "md"}},
            },
        },
        children=[
            # Notification system
            dmc.NotificationProvider(position="top-right"),
            html.Div(id="notifications-container"),

            # Keyboard listener for shortcuts
            dcc.Store(id="keyboard-store", storage_type="memory"),

            # Global state stores
            dcc.Store(
                id="app-state-store",
                storage_type="session",
                data={
                    "filename": str(data_manager.config.data_path.name),
                    "n_cells": data_manager.n_cells,
                    "n_genes": data_manager.n_genes,
                    "obs_columns": data_manager.obs_columns,
                    "embeddings": data_manager.available_embeddings,
                    "layers": data_manager.layers,
                    "gene_names": data_manager.gene_names,  # All genes available
                },
            ),
            dcc.Store(
                id="pages-store",
                storage_type="session",
                data=pages_data,
            ),
            dcc.Store(
                id="selection-store",
                storage_type="memory",
                data={"selected_indices": [], "source_panel": None},
            ),
            # Store mapping from (panel_id, curveNumber, pointIndex) -> cell_index
            # Structure: {panel_id: [[cell_indices for trace 0], [cell_indices for trace 1], ...]}
            dcc.Store(id="scatter-cell-map", storage_type="memory", data={}),
            dcc.Store(id="config-drawer-store", storage_type="memory", data={"panel_id": None, "panel_type": None}),
            dcc.Store(id="layout-upload-store", storage_type="memory", data=None),

            # Modals
            create_new_page_modal(),
            create_subset_page_modal(data_manager),
            create_save_layout_modal(),
            create_load_layout_modal(),
            create_export_modal(),
            create_rename_modal(),
            create_top_genes_modal(),

            # Config Drawer
            create_config_drawer(data_manager),

            # Main layout
            dmc.AppShell(
                id="app-shell",
                children=[
                    dmc.AppShellHeader(create_header(data_manager)),
                    dmc.AppShellNavbar(
                        id="sidebar-navbar",
                        children=create_sidebar(data_manager, pages_data),
                        p="md",
                    ),
                    dmc.AppShellMain(
                        children=create_main_content(data_manager, initial_panels, initial_grid_layout),
                        id="main-content",
                    ),
                    dmc.AppShellFooter(
                        children=create_selection_bar(),
                        p="xs",
                    ),
                ],
                header={"height": 60},
                footer={"height": 60},
                padding="md",
                navbar={"width": 280, "breakpoint": "sm", "collapsed": {"mobile": True}},
            ),
        ],
    )


def create_header(data_manager: DataManager):
    """Create the header content."""
    return dmc.Group(
        [
            dmc.Group([
                dmc.Title("Cellarium", order=3, c="blue"),
                dmc.Badge(f"{data_manager.n_cells:,} cells", color="gray", variant="light", size="lg"),
            ], gap="md"),
            dmc.Group([
                dmc.Text(
                    f"{data_manager.n_genes:,} genes | {len(data_manager.available_embeddings)} embeddings",
                    size="sm", c="dimmed",
                ),
                dmc.Switch(
                    id="theme-toggle",
                    offLabel=DashIconify(icon="radix-icons:sun", width=15),
                    onLabel=DashIconify(icon="radix-icons:moon", width=15),
                    size="lg", color="gray",
                    persistence=True, persistence_type="local",
                ),
            ], gap="md"),
        ],
        justify="space-between", h="100%", px="md",
    )


def create_sidebar(data_manager: DataManager, pages_data: Optional[Dict] = None):
    """Create the sidebar content."""
    # Build initial page list from saved pages
    if pages_data:
        page_items = [
            create_page_item(
                pid,
                pages_data["pages"][pid]["name"],
                pages_data["pages"][pid]["n_cells"],
                pid == pages_data.get("active_page", "root"),
                pages_data["pages"][pid].get("parent_id") is None,
            )
            for pid in pages_data.get("page_order", ["root"])
        ]
    else:
        page_items = [
            create_page_item("root", "All Cells", data_manager.n_cells, True, True),
        ]

    return dmc.Stack([
        # Pages section
        dmc.Group([
            dmc.Text("Pages", fw=600, size="sm"),
            dmc.ActionIcon(
                DashIconify(icon="tabler:plus", width=16),
                id="add-page-btn", variant="light", size="sm",
            ),
        ], justify="space-between"),

        dmc.ScrollArea(
            h=180,
            children=html.Div(id="page-list", children=page_items),
        ),

        dmc.Divider(),

        # Quick controls
        dmc.Text("Quick Color", fw=600, size="sm"),
        dmc.Select(
            id="global-color-select",
            data=[{"value": col, "label": col} for col in data_manager.obs_columns],
            placeholder="Color all plots by...",
            searchable=True, clearable=True,
        ),

        dmc.Divider(),

        # Dataset info
        dmc.Text("Dataset", fw=600, size="sm"),
        dmc.Stack([
            dmc.Text(f"📁 {data_manager.config.data_path.name}", size="xs", c="dimmed"),
            dmc.Text(f"🧬 {data_manager.n_genes:,} genes", size="xs", c="dimmed"),
            dmc.Text(f"📊 {len(data_manager.available_embeddings)} embeddings", size="xs", c="dimmed"),
        ], gap=4),
    ], gap="md")


def create_page_item(page_id: str, name: str, n_cells: int, is_active: bool, is_root: bool):
    """Create a page item for the sidebar."""
    # Use html.Div as the clickable wrapper since dmc.Paper doesn't track n_clicks properly
    return html.Div(
        dmc.Paper(
            dmc.Group([
                dmc.Stack([
                    dmc.Group([
                        DashIconify(
                            icon="tabler:file" if is_root else "tabler:file-arrow-right",
                            width=14,
                            color="var(--mantine-color-blue-6)" if is_active else "var(--mantine-color-dimmed)",
                        ),
                        dmc.Text(
                            name, size="sm", fw=500 if is_active else 400, truncate=True,
                            id={"type": "page-name-text", "index": page_id},
                        ),
                    ], gap="xs"),
                    dmc.Text(f"{n_cells:,} cells", size="xs", c="dimmed", pl="xl"),
                ], gap=2),
                dmc.Group([
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:pencil", width=12),
                        id={"type": "rename-page-btn", "index": page_id},
                        variant="subtle", size="xs", color="gray",
                        style={"visibility": "hidden" if is_root else "visible"},
                    ) if not is_root else None,
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:x", width=12),
                        id={"type": "delete-page-btn", "index": page_id},
                        variant="subtle", size="xs", color="red",
                        style={"visibility": "hidden" if is_root else "visible"},
                    ) if not is_root else None,
                ], gap=2),
            ], justify="space-between"),
            p="xs", withBorder=True,
            style={
                "backgroundColor": "var(--mantine-color-blue-light)" if is_active else None,
            },
        ),
        id={"type": "page-item", "index": page_id},
        n_clicks=0,
        style={"cursor": "pointer"},
    )


def create_main_content(data_manager: DataManager, panels: Dict, grid_layout: Dict):
    """Create the main content area with toolbar and dashboard grid."""
    return dmc.Stack([
        # Toolbar
        create_toolbar(data_manager.available_embeddings),

        # Dashboard grid
        html.Div(
            id="dashboard-wrapper",
            children=[create_dashboard_grid(panels, grid_layout)],
            style={"minHeight": "500px", "flex": 1},
        ),
    ], gap="md", style={"height": "100%"})


def create_toolbar(available_embeddings: List[str]):
    """Create the dashboard toolbar."""
    embedding_items = []
    for emb in available_embeddings:
        name = emb.replace("X_", "").upper()
        embedding_items.append(
            dmc.MenuItem(
                f"{name} Plot",
                id={"type": "add-panel-btn", "panel_type": "scatter", "embedding": emb},
                leftSection=DashIconify(icon="tabler:chart-dots-3", width=16),
            )
        )

    return dmc.Group([
        dmc.Menu([
            dmc.MenuTarget(
                dmc.Button("Add Panel", leftSection=DashIconify(icon="tabler:plus", width=16), variant="light")
            ),
            dmc.MenuDropdown([
                dmc.MenuLabel("Embeddings"),
                *embedding_items,
                dmc.MenuDivider(),
                dmc.MenuLabel("Expression Plots"),
                dmc.MenuItem(
                    "Dot Plot",
                    id={"type": "add-panel-btn", "panel_type": "dotplot", "embedding": ""},
                    leftSection=DashIconify(icon="tabler:chart-dots", width=16),
                ),
                dmc.MenuItem(
                    "Violin Plot",
                    id={"type": "add-panel-btn", "panel_type": "violin", "embedding": ""},
                    leftSection=DashIconify(icon="tabler:chart-area", width=16),
                ),
                dmc.MenuItem(
                    "Heatmap",
                    id={"type": "add-panel-btn", "panel_type": "heatmap", "embedding": ""},
                    leftSection=DashIconify(icon="tabler:chart-histogram", width=16),
                ),
                dmc.MenuDivider(),
                dmc.MenuLabel("Comparison"),
                dmc.MenuItem(
                    "Crosstab Heatmap",
                    id={"type": "add-panel-btn", "panel_type": "crosstab", "embedding": ""},
                    leftSection=DashIconify(icon="tabler:table", width=16),
                ),
            ]),
        ], position="bottom-start"),

        # Layout templates menu
        dmc.Menu([
            dmc.MenuTarget(
                dmc.Button("Layout", leftSection=DashIconify(icon="tabler:layout", width=16), variant="light")
            ),
            dmc.MenuDropdown([
                dmc.MenuLabel("Arrange Panels"),
                dmc.MenuItem("1 Column", id={"type": "layout-template", "template": "1-col"},
                           leftSection=DashIconify(icon="tabler:square", width=16)),
                dmc.MenuItem("2 Columns (50/50)", id={"type": "layout-template", "template": "2-col-equal"},
                           leftSection=DashIconify(icon="tabler:columns-2", width=16)),
                dmc.MenuItem("2 Columns (60/40)", id={"type": "layout-template", "template": "2-col-60-40"},
                           leftSection=DashIconify(icon="tabler:columns-2", width=16)),
                dmc.MenuItem("2 Columns (40/60)", id={"type": "layout-template", "template": "2-col-40-60"},
                           leftSection=DashIconify(icon="tabler:columns-2", width=16)),
                dmc.MenuItem("3 Columns", id={"type": "layout-template", "template": "3-col-equal"},
                           leftSection=DashIconify(icon="tabler:columns-3", width=16)),
                dmc.MenuItem("2x2 Grid", id={"type": "layout-template", "template": "2x2-grid"},
                           leftSection=DashIconify(icon="tabler:layout-grid", width=16)),
            ]),
        ], position="bottom-start"),

        dmc.Group([
            dmc.Button("Save Layout", id="save-layout-btn", leftSection=DashIconify(icon="tabler:device-floppy", width=16), variant="outline", size="sm"),
            dmc.Button("Load Layout", id="load-layout-btn", leftSection=DashIconify(icon="tabler:folder-open", width=16), variant="outline", size="sm"),
            dmc.Menu([
                dmc.MenuTarget(
                    dmc.ActionIcon(DashIconify(icon="tabler:dots-vertical", width=16), variant="subtle", size="lg")
                ),
                dmc.MenuDropdown([
                    dmc.MenuItem("Reset Layout", id="reset-layout-btn", leftSection=DashIconify(icon="tabler:refresh", width=16)),
                    dmc.MenuItem("Clear All Panels", id="clear-panels-btn", leftSection=DashIconify(icon="tabler:trash", width=16), color="red"),
                ]),
            ], position="bottom-end"),
        ], gap="xs"),
    ], justify="space-between", mb="md")


def create_dashboard_grid(panels: Dict, grid_layout: Dict = None):
    """Create a flexbox-based dashboard grid with panels.

    Each panel has width (percentage) and height (pixels) in its config.
    Panels flow in a flex-wrap container.
    """
    if not panels:
        return dmc.Center(
            dmc.Stack([
                DashIconify(icon="tabler:layout-dashboard", width=64, color="var(--mantine-color-dimmed)"),
                dmc.Text("No panels yet", size="lg", c="dimmed"),
                dmc.Text("Click 'Add Panel' to get started", size="sm", c="dimmed"),
            ], align="center", gap="md"),
            style={"height": "400px"},
        )

    panel_components = []
    for panel_id, panel_config in panels.items():
        # Get layout config from panel
        width = panel_config.get("width", "50%")
        height = panel_config.get("height", "400px")

        panel_components.append(
            html.Div(
                create_panel_wrapper(panel_id, panel_config),
                style={
                    "width": f"calc({width} - 10px)",
                    "height": height,
                    "padding": "5px",
                    "boxSizing": "border-box",
                },
            )
        )

    return html.Div(
        id="dashboard-grid",
        children=panel_components,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignContent": "flex-start",
            "minHeight": "400px",
            "width": "100%",
        },
    )


def create_panel_wrapper(panel_id: str, panel_config: Dict):
    """Create a panel wrapper with header and content."""
    panel_type = panel_config.get("type", "scatter")
    config = panel_config.get("config", {})

    # Generate title
    title = get_panel_title(panel_type, config)

    return html.Div(
        dmc.Paper([
            # Panel header (no drag handle)
            dmc.Group([
                dmc.Text(title, size="sm", fw=500, truncate=True, style={"flex": 1}),
                dmc.Group([
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:settings", width=16),
                        id={"type": "panel-config-btn", "index": panel_id},
                        variant="subtle", size="sm", color="gray",
                    ),
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:x", width=16),
                        id={"type": "panel-close-btn", "index": panel_id},
                        variant="subtle", size="sm", color="red",
                    ),
                ], gap=4),
            ], justify="space-between", p="xs",
               style={"borderBottom": "1px solid var(--mantine-color-default-border)", "flexShrink": 0}),

            # Panel content
            html.Div(
                dcc.Graph(
                    id={"type": "panel-graph", "index": panel_id},
                    config={"displaylogo": False, "scrollZoom": True},
                    style={"height": "100%", "width": "100%"},
                ),
                style={"height": "calc(100% - 44px)", "padding": "4px"},
            ),
        ], withBorder=True, radius="md",
           style={"height": "100%", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
           **{"data-panel-type": panel_type, "data-panel-id": panel_id}),
        id=panel_id,
        style={"height": "100%"},
    )


def get_panel_title(panel_type: str, config: Dict) -> str:
    """Generate panel title from type and config."""
    if panel_type == "scatter":
        emb = config.get("embedding", "")
        color = config.get("color_obs") or config.get("color_gene") or ""
        name = emb.replace("X_", "").upper() if emb else "Scatter"
        return f"{name}" + (f" - {color}" if color else "")
    elif panel_type == "dotplot":
        genes = config.get("var_names", [])
        return f"Dot Plot ({len(genes)} genes)" if genes else "Dot Plot"
    elif panel_type == "violin":
        genes = config.get("var_names", [])
        return f"Violin ({len(genes)} genes)" if genes else "Violin Plot"
    elif panel_type == "heatmap":
        genes = config.get("var_names", [])
        return f"Heatmap ({len(genes)} genes)" if genes else "Heatmap"
    elif panel_type == "crosstab":
        groupby = config.get("groupby", "")
        compare = config.get("compare_col", "")
        if groupby and compare:
            return f"Crosstab: {groupby} vs {compare}"
        return "Crosstab Heatmap"
    return "Panel"


def create_selection_bar():
    """Create the selection info bar."""
    return dmc.Paper(
        dmc.Group([
            html.Div(id="selection-info", children=[
                dmc.Text("No cells selected", size="sm", c="dimmed"),
            ]),
            dmc.Group([
                dmc.Button(
                    "Top Genes", id="top-genes-btn",
                    leftSection=DashIconify(icon="tabler:dna", width=16),
                    size="sm", variant="light", color="violet", disabled=True,
                ),
                dmc.Button(
                    "Create Page", id="create-page-btn",
                    leftSection=DashIconify(icon="tabler:file-plus", width=16),
                    size="sm", disabled=True,
                ),
                dmc.Button(
                    "Export", id="export-selection-btn",
                    leftSection=DashIconify(icon="tabler:download", width=16),
                    variant="outline", size="sm", disabled=True,
                ),
                dmc.Button(
                    "Clear", id="clear-selection-btn",
                    leftSection=DashIconify(icon="tabler:x", width=16),
                    variant="subtle", size="sm", color="gray",
                ),
            ], gap="xs"),
        ], justify="space-between"),
        withBorder=True, p="sm",
    )


def create_config_drawer(data_manager: DataManager):
    """Create the configuration drawer for panels."""
    # Get gene symbol column options and auto-detect
    gene_symbol_cols = data_manager.get_gene_symbol_column_options()
    detected_symbols_col = data_manager.detect_gene_symbols_column()
    var_names_type = data_manager.detect_var_names_type()

    # Use gene display names (symbols from raw if available)
    # Prefer raw data if available for more complete gene list
    use_raw_default = data_manager.has_raw
    gene_display_names = data_manager.get_gene_display_names(use_raw=use_raw_default)

    # Build info text about detection
    detection_info = []
    if var_names_type == 'ensembl':
        detection_info.append("var_names appear to be Ensembl IDs")
        if detected_symbols_col:
            detection_info.append(f"Auto-detected symbols column: {detected_symbols_col}")
    elif var_names_type == 'symbols':
        detection_info.append("var_names appear to be gene symbols")

    return dmc.Drawer(
        id="config-drawer",
        title=dmc.Group([
            DashIconify(icon="tabler:settings", width=20),
            dmc.Text("Panel Settings", fw=500),
        ]),
        position="right",
        size="md",
        children=[
            dmc.ScrollArea(
                h="calc(100vh - 100px)",
                children=[
                    dmc.Stack([
                        # Scatter config section
                        html.Div(id="scatter-config-section", children=[
                            dmc.Select(
                                id="config-embedding",
                                label="Embedding",
                                data=[{"value": e, "label": e.replace("X_", "").upper()} for e in data_manager.available_embeddings],
                                value=data_manager.available_embeddings[0] if data_manager.available_embeddings else None,
                            ),
                            dmc.Divider(label="Color By", labelPosition="center", my="md"),
                            dmc.SegmentedControl(
                                id="config-color-type",
                                data=[{"value": "obs", "label": "Metadata"}, {"value": "gene", "label": "Gene"}],
                                value="obs", fullWidth=True,
                            ),
                            dmc.Select(
                                id="config-color-obs",
                                label="Metadata Column",
                                data=[{"value": c, "label": c} for c in data_manager.obs_columns],
                                searchable=True, clearable=True,
                            ),
                            dmc.Select(
                                id="config-color-gene",
                                label="Gene",
                                description=f"Type to search {len(gene_display_names):,} genes" + (" from raw" if use_raw_default else ""),
                                data=[],  # Populated dynamically via callback
                                searchable=True, clearable=True,
                                placeholder="Type to search genes...",
                                nothingFoundMessage="No genes found",
                            ),

                            # Gene symbol column selector (for when var_names are Ensembl IDs)
                            dmc.Select(
                                id="config-gene-symbols-col",
                                label="Gene Symbols Column",
                                description="; ".join(detection_info) if detection_info else "If var_names are Ensembl IDs, select column with gene symbols",
                                data=[{"value": c, "label": c} for c in gene_symbol_cols] if gene_symbol_cols else [],
                                value=detected_symbols_col,  # Auto-select detected column
                                searchable=True, clearable=True,
                                placeholder="Auto-detected" if detected_symbols_col else "None (use var_names)",
                                disabled=not gene_symbol_cols,
                            ),

                            # Use raw toggle - default to True if raw is available
                            dmc.Switch(
                                id="config-use-raw",
                                label="Use Raw Data",
                                description=f"Raw has {data_manager.raw_n_genes:,} genes" if data_manager.has_raw else "No raw data available",
                                checked=use_raw_default,  # Default to True if raw is available
                                disabled=not data_manager.has_raw,
                            ),

                            dmc.Divider(label="Appearance", labelPosition="center", my="md"),

                            dmc.Select(
                                id="config-colorscale",
                                label="Color Scale",
                                data=COLORSCALES,
                                value="Viridis",
                                searchable=True,
                            ),

                            # vmax control with percentile support
                            dmc.Text("Max Value (vmax)", size="sm", fw=500, mt="md"),
                            dmc.Group([
                                dmc.SegmentedControl(
                                    id="config-vmax-type",
                                    data=[
                                        {"value": "auto", "label": "Auto"},
                                        {"value": "percentile", "label": "Percentile"},
                                        {"value": "fixed", "label": "Fixed"},
                                    ],
                                    value="auto",
                                    size="xs",
                                ),
                            ], gap="xs"),
                            dmc.NumberInput(
                                id="config-vmax-value",
                                label="Value",
                                description="Percentile (e.g., 99.9) or fixed value",
                                value=99.9,
                                min=0,
                                step=0.1,
                                decimalScale=2,
                            ),

                            dmc.Text("Point Size", size="sm", fw=500, mt="md"),
                            dmc.Slider(id="config-point-size", value=DEFAULT_POINT_SIZE, min=1, max=10, step=1),
                            dmc.Text("Opacity", size="sm", fw=500, mt="md"),
                            dmc.Slider(id="config-opacity", value=0.7, min=0.1, max=1.0, step=0.1),

                            # Groups filter (highlight only selected groups)
                            dmc.Divider(label="Group Highlight", labelPosition="center", my="md"),
                            dmc.Select(
                                id="config-groups-column",
                                label="Groups Column",
                                description="Select column to filter groups",
                                data=[{"value": c, "label": c} for c in data_manager.obs_columns],
                                searchable=True, clearable=True,
                            ),
                            dmc.MultiSelect(
                                id="config-groups-values",
                                label="Highlight Groups",
                                description="Other groups will be shown in grey",
                                data=[],  # Populated dynamically based on groups column
                                searchable=True, clearable=True,
                                placeholder="All groups (no filter)",
                            ),
                        ]),

                        # Expression config section (dotplot, violin, heatmap)
                        html.Div(id="expression-config-section", style={"display": "none"}, children=[
                            dmc.MultiSelect(
                                id="config-genes",
                                label="Genes",
                                description=f"Type to search {len(gene_display_names):,} genes" + (" from raw" if use_raw_default else ""),
                                data=[],  # Populated dynamically via callback
                                searchable=True, clearable=True,
                                placeholder="Type to search genes...",
                                maxDropdownHeight=300,
                                nothingFoundMessage="No genes found",
                            ),
                            dmc.Select(
                                id="config-groupby",
                                label="Group By",
                                data=[{"value": c, "label": c} for c in data_manager.obs_columns],
                                searchable=True, clearable=True,
                            ),

                            # Gene symbol column selector
                            dmc.Select(
                                id="config-expr-gene-symbols-col",
                                label="Gene Symbols Column",
                                description="; ".join(detection_info) if detection_info else "If var_names are Ensembl IDs, select column with gene symbols",
                                data=[{"value": c, "label": c} for c in gene_symbol_cols] if gene_symbol_cols else [],
                                value=detected_symbols_col,  # Auto-select detected column
                                searchable=True, clearable=True,
                                placeholder="Auto-detected" if detected_symbols_col else "None (use var_names)",
                                disabled=not gene_symbol_cols,
                            ),

                            # Use raw toggle - default to True if raw is available
                            dmc.Switch(
                                id="config-expr-use-raw",
                                label="Use Raw Data",
                                description=f"Raw has {data_manager.raw_n_genes:,} genes" if data_manager.has_raw else "No raw data available",
                                checked=use_raw_default,  # Default to True if raw is available
                                disabled=not data_manager.has_raw,
                            ),

                            dmc.Select(
                                id="config-expr-colorscale",
                                label="Color Scale",
                                data=COLORSCALES,
                                value="Reds",
                                searchable=True,
                            ),
                        ]),

                        # Crosstab config section
                        html.Div(id="crosstab-config-section", style={"display": "none"}, children=[
                            dmc.Select(
                                id="config-crosstab-groupby",
                                label="Groupby (Rows)",
                                description="Primary grouping variable (rows)",
                                data=[{"value": c, "label": c} for c in data_manager.get_categorical_columns()],
                                searchable=True, clearable=False,
                            ),
                            dmc.Select(
                                id="config-crosstab-compare",
                                label="Compare Column (Columns)",
                                description="Secondary grouping variable (columns)",
                                data=[{"value": c, "label": c} for c in data_manager.get_categorical_columns()],
                                searchable=True, clearable=False,
                            ),
                            dmc.Select(
                                id="config-crosstab-use-rep",
                                label="Representation for Dendrogram",
                                description="Used for computing category distances",
                                data=[{"value": r, "label": r.replace("X_", "")} for r in data_manager.get_available_representations()],
                                value=data_manager.get_default_representation(),
                                searchable=True, clearable=False,
                            ),
                            dmc.Select(
                                id="config-crosstab-colorscale",
                                label="Color Scale",
                                data=COLORSCALES,
                                value="YlOrRd",
                                searchable=True,
                            ),
                            dmc.Switch(
                                id="config-crosstab-normalize",
                                label="Normalize Rows",
                                description="Show proportions (row sums = 1)",
                                checked=True,
                            ),
                            dmc.Switch(
                                id="config-crosstab-dendrogram",
                                label="Show Dendrogram",
                                description="Display hierarchical clustering tree",
                                checked=True,
                            ),
                        ]),

                        # Layout section (always visible for all panel types)
                        html.Div(id="layout-config-section", children=[
                            dmc.Divider(label="Panel Size", labelPosition="center", my="md"),
                            dmc.Select(
                                id="config-layout-preset",
                                label="Size Preset",
                                description="Choose a common panel size",
                                data=[
                                    {"group": "Standard", "items": [
                                        {"value": "half", "label": "Half Width (50% × 400px)"},
                                        {"value": "third", "label": "Third Width (33% × 400px)"},
                                        {"value": "quarter", "label": "Quarter Width (25% × 350px)"},
                                    ]},
                                    {"group": "Wide", "items": [
                                        {"value": "full", "label": "Full Width (100% × 400px)"},
                                        {"value": "full-tall", "label": "Full Width Tall (100% × 600px)"},
                                        {"value": "two-thirds", "label": "Two-Thirds (66% × 450px)"},
                                    ]},
                                    {"group": "Large", "items": [
                                        {"value": "half-tall", "label": "Half Tall (50% × 600px)"},
                                    ]},
                                    {"group": "Compact", "items": [
                                        {"value": "half-short", "label": "Half Compact (50% × 300px)"},
                                        {"value": "third-short", "label": "Third Compact (33% × 300px)"},
                                    ]},
                                    {"group": "Advanced", "items": [
                                        {"value": "custom", "label": "Custom..."},
                                    ]},
                                ],
                                value="half",
                            ),
                            # Advanced options (collapsible)
                            dmc.Accordion(
                                id="layout-advanced-accordion",
                                chevronPosition="left",
                                variant="contained",
                                mt="sm",
                                children=[
                                    dmc.AccordionItem(
                                        value="advanced",
                                        children=[
                                            dmc.AccordionControl(
                                                dmc.Text("Advanced Options", size="sm"),
                                                icon=DashIconify(icon="tabler:adjustments", width=16),
                                            ),
                                            dmc.AccordionPanel([
                                                dmc.Select(
                                                    id="config-panel-width",
                                                    label="Width",
                                                    data=[
                                                        {"value": "25%", "label": "25% (Quarter)"},
                                                        {"value": "33.33%", "label": "33% (Third)"},
                                                        {"value": "50%", "label": "50% (Half)"},
                                                        {"value": "66.66%", "label": "66% (Two-thirds)"},
                                                        {"value": "75%", "label": "75% (Three-quarters)"},
                                                        {"value": "100%", "label": "100% (Full)"},
                                                    ],
                                                    value="50%",
                                                    size="sm",
                                                ),
                                                dmc.Select(
                                                    id="config-panel-height",
                                                    label="Height",
                                                    data=[
                                                        {"value": "250px", "label": "250px"},
                                                        {"value": "300px", "label": "300px"},
                                                        {"value": "350px", "label": "350px"},
                                                        {"value": "400px", "label": "400px"},
                                                        {"value": "450px", "label": "450px"},
                                                        {"value": "500px", "label": "500px"},
                                                        {"value": "600px", "label": "600px"},
                                                        {"value": "800px", "label": "800px"},
                                                    ],
                                                    value="400px",
                                                    size="sm",
                                                    mt="xs",
                                                ),
                                            ]),
                                        ],
                                    ),
                                ],
                            ),
                        ]),

                        dmc.Button(
                            "Apply", id="config-apply-btn",
                            leftSection=DashIconify(icon="tabler:check", width=16),
                            fullWidth=True, mt="xl",
                        ),
                    ], gap="sm", p="md"),
                ],
            ),
        ],
        opened=False,
    )


def create_new_page_modal():
    """Create modal for new page creation."""
    return dmc.Modal(
        id="new-page-modal",
        title=dmc.Group([DashIconify(icon="tabler:file-plus", width=20), dmc.Text("Create New Page", fw=500)]),
        children=[
            dmc.Stack([
                dmc.TextInput(id="new-page-name", label="Page Name", placeholder="e.g., T Cells"),
                dmc.Text(id="new-page-info", size="sm", c="dimmed"),
                dmc.Group([
                    dmc.Button("Cancel", id="new-page-cancel-btn", variant="light", color="gray"),
                    dmc.Button("Create", id="new-page-create-btn", leftSection=DashIconify(icon="tabler:check", width=16)),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_save_layout_modal():
    """Create save layout modal."""
    return dmc.Modal(
        id="save-layout-modal",
        title=dmc.Group([DashIconify(icon="tabler:device-floppy", width=20), dmc.Text("Save Layout", fw=500)]),
        children=[
            dmc.Stack([
                dmc.TextInput(id="save-layout-name", label="Layout Name", placeholder="My Layout"),
                dmc.Textarea(id="save-layout-description", label="Description", placeholder="Optional description...", autosize=True, minRows=2),
                dmc.Group([
                    dmc.Button("Cancel", id="save-layout-cancel-btn", variant="light", color="gray"),
                    dmc.Button("Download", id="save-layout-confirm-btn", leftSection=DashIconify(icon="tabler:download", width=16)),
                ], justify="flex-end"),
                dcc.Download(id="layout-download"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_load_layout_modal():
    """Create load layout modal."""
    return dmc.Modal(
        id="load-layout-modal",
        title=dmc.Group([DashIconify(icon="tabler:folder-open", width=20), dmc.Text("Load Layout", fw=500)]),
        children=[
            dmc.Stack([
                dcc.Upload(
                    id="layout-upload",
                    children=dmc.Paper([
                        dmc.Stack([
                            DashIconify(icon="tabler:upload", width=40, color="var(--mantine-color-blue-6)"),
                            dmc.Text("Drop a layout file here or click to upload", size="sm"),
                        ], align="center", gap="xs"),
                    ], p="xl", withBorder=True, radius="md", style={"borderStyle": "dashed", "cursor": "pointer"}),
                    accept=".json",
                ),
                html.Div(id="layout-upload-preview"),
                dmc.Group([
                    dmc.Button("Cancel", id="load-layout-cancel-btn", variant="light", color="gray"),
                    dmc.Button("Apply", id="load-layout-apply-btn", leftSection=DashIconify(icon="tabler:check", width=16), disabled=True),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_export_modal():
    """Create export modal."""
    return dmc.Modal(
        id="export-modal",
        title=dmc.Group([DashIconify(icon="tabler:download", width=20), dmc.Text("Export Cells", fw=500)]),
        children=[
            dmc.Stack([
                dmc.Text(id="export-info", size="sm"),
                dmc.TextInput(id="export-filename", label="File Name", placeholder="subset", rightSection=dmc.Text(".h5ad", size="sm", c="dimmed")),
                dmc.Checkbox(id="export-include-raw", label="Include raw counts", checked=True),
                dmc.Checkbox(id="export-include-obsm", label="Include embeddings", checked=True),
                dmc.Group([
                    dmc.Button("Cancel", id="export-cancel-btn", variant="light", color="gray"),
                    dmc.Button("Export", id="export-confirm-btn", leftSection=DashIconify(icon="tabler:download", width=16)),
                ], justify="flex-end"),
                dcc.Download(id="export-download"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_rename_modal():
    """Create page rename modal."""
    return dmc.Modal(
        id="rename-page-modal",
        title=dmc.Group([DashIconify(icon="tabler:pencil", width=20), dmc.Text("Rename Page", fw=500)]),
        size="sm",
        children=[
            dmc.Stack([
                dcc.Store(id="rename-page-store", data={"page_id": None}),
                dmc.TextInput(id="rename-page-input", label="New Name", placeholder="Enter page name..."),
                dmc.Group([
                    dmc.Button("Cancel", id="rename-page-cancel-btn", variant="light", color="gray"),
                    dmc.Button("Rename", id="rename-page-confirm-btn", leftSection=DashIconify(icon="tabler:check", width=16)),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_top_genes_modal():
    """Create top genes analysis modal."""
    return dmc.Modal(
        id="top-genes-modal",
        title=dmc.Group([DashIconify(icon="tabler:dna", width=20), dmc.Text("Top Expressed Genes", fw=500)]),
        size="lg",
        children=[
            dmc.Stack([
                # Info and controls
                dmc.Group([
                    dmc.Text(id="top-genes-info", size="sm", c="dimmed"),
                    dmc.Group([
                        dmc.NumberInput(
                            id="top-genes-count",
                            label="Top N",
                            value=100,
                            min=10,
                            max=500,
                            step=10,
                            w=100,
                            size="xs",
                        ),
                        dmc.Switch(
                            id="top-genes-use-raw",
                            label="Use raw",
                            checked=True,
                            size="sm",
                        ),
                    ], gap="md"),
                ], justify="space-between"),

                # Loading state and table (LoadingOverlay in DMC 2.4+ doesn't take children)
                dmc.Box(
                    pos="relative",
                    children=[
                        dmc.LoadingOverlay(
                            id="top-genes-loading",
                            visible=False,
                            overlayProps={"radius": "sm", "blur": 2},
                        ),
                        html.Div(
                            id="top-genes-table-container",
                            style={"maxHeight": "400px", "overflowY": "auto"},
                        ),
                    ],
                ),

                # Action buttons
                dmc.Group([
                    dmc.Button(
                        "Refresh",
                        id="top-genes-refresh-btn",
                        leftSection=DashIconify(icon="tabler:refresh", width=16),
                        variant="light",
                    ),
                    dmc.Button(
                        "Download CSV",
                        id="top-genes-download-btn",
                        leftSection=DashIconify(icon="tabler:download", width=16),
                        variant="outline",
                    ),
                    dmc.Button("Close", id="top-genes-close-btn", variant="light", color="gray"),
                ], justify="flex-end"),
                dcc.Download(id="top-genes-download"),
                dcc.Store(id="top-genes-data-store", storage_type="memory", data=None),
            ], gap="md"),
        ],
        opened=False,
    )


def create_subset_page_modal(data_manager: DataManager):
    """Create modal for creating pages via variable or gene expression subsetting."""
    # Get available columns for variable subsetting
    obs_columns = data_manager.obs_columns

    return dmc.Modal(
        id="subset-page-modal",
        title=dmc.Group([DashIconify(icon="tabler:filter-plus", width=20), dmc.Text("Create Subset Page", fw=500)]),
        size="lg",
        children=[
            dmc.Stack([
                dmc.Tabs(
                    id="subset-page-tabs",
                    value="variable",
                    children=[
                        dmc.TabsList([
                            dmc.TabsTab("By Variable", value="variable", leftSection=DashIconify(icon="tabler:list", width=16)),
                            dmc.TabsTab("By Gene Expression", value="gene", leftSection=DashIconify(icon="tabler:dna", width=16)),
                        ]),

                        # Variable-based subsetting tab
                        dmc.TabsPanel(
                            dmc.Stack([
                                dmc.Select(
                                    id="subset-var-column",
                                    label="Select Variable",
                                    description="Choose a metadata column to filter by",
                                    data=[{"value": c, "label": c} for c in obs_columns],
                                    searchable=True,
                                    placeholder="Select a column...",
                                ),
                                dmc.MultiSelect(
                                    id="subset-var-values",
                                    label="Select Values",
                                    description="Choose which values to include",
                                    data=[],
                                    searchable=True,
                                    placeholder="First select a variable above...",
                                    disabled=True,
                                ),
                                dmc.Text(id="subset-var-info", size="sm", c="dimmed"),
                            ], gap="md"),
                            value="variable",
                            pt="md",
                        ),

                        # Gene expression-based subsetting tab
                        dmc.TabsPanel(
                            dmc.Stack([
                                dmc.Select(
                                    id="subset-gene-name",
                                    label="Select Gene",
                                    description="Type to search genes",
                                    data=[],
                                    searchable=True,
                                    placeholder="Type gene name...",
                                ),
                                dmc.Group([
                                    dmc.Select(
                                        id="subset-gene-operator",
                                        label="Operator",
                                        data=[
                                            {"value": ">", "label": "> (greater than)"},
                                            {"value": ">=", "label": ">= (greater or equal)"},
                                            {"value": "<", "label": "< (less than)"},
                                            {"value": "<=", "label": "<= (less or equal)"},
                                            {"value": "==", "label": "== (equal to)"},
                                        ],
                                        value=">",
                                        style={"flex": 1},
                                    ),
                                    dmc.NumberInput(
                                        id="subset-gene-threshold",
                                        label="Threshold",
                                        value=0,
                                        step=0.1,
                                        decimalScale=2,
                                        style={"flex": 1},
                                    ),
                                ], grow=True),
                                dmc.Checkbox(
                                    id="subset-gene-use-raw",
                                    label="Use raw expression data",
                                    checked=data_manager.has_raw,
                                ),
                                dmc.Text(id="subset-gene-info", size="sm", c="dimmed"),
                            ], gap="md"),
                            value="gene",
                            pt="md",
                        ),
                    ],
                ),

                dmc.TextInput(
                    id="subset-page-name",
                    label="Page Name",
                    placeholder="e.g., T Cells, CD68+ Macrophages",
                ),

                dmc.Group([
                    dmc.Button("Cancel", id="subset-page-cancel-btn", variant="light", color="gray"),
                    dmc.Button(
                        "Create Page", id="subset-page-create-btn",
                        leftSection=DashIconify(icon="tabler:check", width=16),
                        disabled=True,
                    ),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )


def create_notification(title: str, message: str, color: str = "blue", icon: str = "tabler:check") -> dmc.Notification:
    """Create a notification component."""
    return dmc.Notification(
        title=title,
        message=message,
        color=color,
        icon=DashIconify(icon=icon),
        action="show",
        autoClose=4000,
    )


# =============================================================================
# CALLBACKS
# =============================================================================

def register_callbacks(app: Dash):
    """Register all Dash callbacks."""

    # -------------------------------------------------------------------------
    # Theme Toggle
    # -------------------------------------------------------------------------
    app.clientside_callback(
        """
        function(switchOn) {
            document.documentElement.setAttribute('data-mantine-color-scheme', switchOn ? 'dark' : 'light');
            return window.dash_clientside.no_update;
        }
        """,
        Output("theme-toggle", "id"),
        Input("theme-toggle", "checked"),
    )

    # -------------------------------------------------------------------------
    # Panel Graph Rendering
    # -------------------------------------------------------------------------
    @app.callback(
        Output({"type": "panel-graph", "index": ALL}, "figure"),
        Output("scatter-cell-map", "data"),
        Input("pages-store", "data"),
        Input("theme-toggle", "checked"),
        State({"type": "panel-graph", "index": ALL}, "id"),
    )
    def render_all_panels(pages_state, dark_mode, panel_ids):
        """Render all panel graphs based on their configuration."""
        if not panel_ids:
            return [], {}

        dm: DataManager = app.server.config["data_manager"]
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        panels = active_page.get("panels", {})
        cell_indices = active_page.get("cell_indices")

        figures = []
        scatter_cell_map = {}  # Maps panel_id -> [[cell indices per trace], ...]

        for panel_id_dict in panel_ids:
            panel_id = panel_id_dict["index"]
            panel_config = panels.get(panel_id, {})
            panel_type = panel_config.get("type", "scatter")
            config = panel_config.get("config", {})

            if panel_type == "scatter":
                # render_scatter returns (figure, trace_cell_indices)
                fig, trace_cell_indices = render_scatter(dm, config, cell_indices, [], dark_mode)
                scatter_cell_map[panel_id] = trace_cell_indices
            elif panel_type == "dotplot":
                fig = render_dotplot(dm, config, cell_indices, dark_mode)
            elif panel_type == "violin":
                fig = render_violin(dm, config, cell_indices, dark_mode)
            elif panel_type == "heatmap":
                fig = render_heatmap(dm, config, cell_indices, dark_mode)
            elif panel_type == "crosstab":
                fig = render_crosstab(dm, config, cell_indices, dark_mode)
            else:
                fig = go.Figure()

            figures.append(fig)

        return figures, scatter_cell_map

    # -------------------------------------------------------------------------
    # Add Panel
    # -------------------------------------------------------------------------
    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input({"type": "add-panel-btn", "panel_type": ALL, "embedding": ALL}, "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def add_panel(n_clicks, pages_state):
        """Add a new panel to the dashboard."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        panel_type = triggered.get("panel_type", "scatter")
        embedding = triggered.get("embedding", "")

        # Generate new panel ID
        next_id = pages_state.get("next_panel_id", 1)
        panel_id = f"panel-{next_id}"
        pages_state["next_panel_id"] = next_id + 1

        # Create panel config with auto-detected defaults
        dm: DataManager = app.server.config["data_manager"]
        detected_symbols_col = dm.detect_gene_symbols_column()
        use_raw_default = dm.has_raw

        if panel_type == "scatter":
            config = {
                "embedding": embedding or get_default_embedding(dm),
                "color_type": "obs",
                "color_obs": None,
                "color_gene": None,
                "gene_symbols_col": detected_symbols_col,
                "use_raw": use_raw_default,
                "colorscale": "Viridis",
                "vmax_type": "auto",
                "vmax_value": 99.9,
                "point_size": DEFAULT_POINT_SIZE,
                "opacity": DEFAULT_OPACITY,
                "groups_column": None,
                "groups_values": [],
            }
        elif panel_type == "crosstab":
            # Get default categorical columns
            categorical_cols = dm.get_categorical_columns()
            default_groupby = categorical_cols[0] if categorical_cols else None
            default_compare = categorical_cols[1] if len(categorical_cols) > 1 else default_groupby
            config = {
                "groupby": default_groupby,
                "compare_col": default_compare,
                "use_rep": dm.get_default_representation(),
                "colorscale": "YlOrRd",
                "normalize": True,
                "show_dendrogram": True,
            }
        else:
            # Expression plots (dotplot, violin, heatmap)
            config = {
                "var_names": [],
                "groupby": None,
                "gene_symbols_col": detected_symbols_col,
                "use_raw": use_raw_default,
                "colorscale": "Reds",
            }

        # Add to active page with default width/height
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state["pages"][active_page_id]
        active_page["panels"][panel_id] = {
            "type": panel_type,
            "config": config,
            "width": "50%",   # Default to half width
            "height": "400px",  # Default height
        }

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        # Rebuild grid
        grid = create_dashboard_grid(active_page["panels"], active_page.get("grid_layout"))
        return pages_state, grid

    # -------------------------------------------------------------------------
    # Layout Templates
    # -------------------------------------------------------------------------
    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input({"type": "layout-template", "template": ALL}, "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def apply_layout_template(n_clicks, pages_state):
        """Apply a layout template to rearrange panels."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        template = triggered.get("template", "1-col")
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state["pages"][active_page_id]
        panel_ids = list(active_page["panels"].keys())
        n_panels = len(panel_ids)

        if n_panels == 0:
            raise PreventUpdate

        # Define layout templates (widths out of 12 columns)
        templates = {
            "1-col": [(12,)],  # All panels full width, stacked
            "2-col-equal": [(6, 6)],  # Two equal columns per row
            "2-col-60-40": [(7, 5)],  # 60/40 split
            "2-col-40-60": [(5, 7)],  # 40/60 split
            "3-col-equal": [(4, 4, 4)],  # Three equal columns
            "2x2-grid": [(6, 6), (6, 6)],  # 2x2 grid
        }

        col_pattern = templates.get(template, [(12,)])

        # Generate new grid layout
        new_layout = {"lg": [], "md": [], "sm": [], "xs": []}
        panel_idx = 0
        row_y = 0
        row_height = 5

        while panel_idx < n_panels:
            # Cycle through row patterns
            row_pattern = col_pattern[row_y % len(col_pattern)] if template == "2x2-grid" else col_pattern[0]

            for col_idx, width in enumerate(row_pattern):
                if panel_idx >= n_panels:
                    break

                panel_id = panel_ids[panel_idx]
                x_pos = sum(row_pattern[:col_idx])

                # Scale widths for different breakpoints
                new_layout["lg"].append({
                    "i": panel_id, "x": x_pos, "y": row_y * row_height,
                    "w": width, "h": row_height, "minW": 3, "minH": 2
                })
                # For md, scale down proportionally
                md_width = max(3, int(width * 10 / 12))
                md_x = max(0, int(x_pos * 10 / 12))
                new_layout["md"].append({
                    "i": panel_id, "x": md_x, "y": row_y * row_height,
                    "w": md_width, "h": row_height, "minW": 3, "minH": 2
                })
                # For sm/xs, stack vertically
                new_layout["sm"].append({
                    "i": panel_id, "x": 0, "y": panel_idx * row_height,
                    "w": 6, "h": row_height, "minW": 3, "minH": 2
                })
                new_layout["xs"].append({
                    "i": panel_id, "x": 0, "y": panel_idx * row_height,
                    "w": 4, "h": row_height, "minW": 2, "minH": 2
                })

                panel_idx += 1

            row_y += 1

        active_page["grid_layout"] = new_layout

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        return pages_state, grid

    # -------------------------------------------------------------------------
    # Close Panel
    # -------------------------------------------------------------------------
    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input({"type": "panel-close-btn", "index": ALL}, "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def close_panel(n_clicks, pages_state):
        """Close/remove a panel."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        panel_id = triggered["index"]
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state["pages"][active_page_id]

        # Remove panel
        if panel_id in active_page["panels"]:
            del active_page["panels"][panel_id]

        # Remove from grid layout
        for bp in ["lg", "md", "sm", "xs"]:
            if bp in active_page.get("grid_layout", {}):
                active_page["grid_layout"][bp] = [
                    item for item in active_page["grid_layout"][bp] if item["i"] != panel_id
                ]

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        return pages_state, grid

    # -------------------------------------------------------------------------
    # Gene Search (Server-side filtering)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("config-color-gene", "data", allow_duplicate=True),
        Input("config-color-gene", "searchValue"),
        State("config-use-raw", "checked"),
        prevent_initial_call=True,
    )
    def search_genes_scatter(search_value, use_raw):
        """Search genes for scatter plot color dropdown."""
        if not search_value or len(search_value) < 2:
            return []

        dm: DataManager = app.server.config["data_manager"]
        gene_names = dm.get_gene_display_names(use_raw=use_raw)

        # Case-insensitive search, limit to 100 results
        search_lower = search_value.lower()
        matches = [g for g in gene_names if search_lower in g.lower()][:100]

        return [{"value": g, "label": g} for g in matches]

    @app.callback(
        Output("config-genes", "data", allow_duplicate=True),
        Input("config-genes", "searchValue"),
        State("config-expr-use-raw", "checked"),
        State("config-genes", "value"),
        prevent_initial_call=True,
    )
    def search_genes_expression(search_value, use_raw, current_values):
        """Search genes for expression plot dropdown."""
        dm: DataManager = app.server.config["data_manager"]
        gene_names = dm.get_gene_display_names(use_raw=use_raw)

        # Always include currently selected genes so they remain visible
        selected_genes = current_values or []
        selected_options = [{"value": g, "label": g} for g in selected_genes if g in gene_names]

        if not search_value or len(search_value) < 2:
            return selected_options

        # Case-insensitive search, limit to 100 results
        search_lower = search_value.lower()
        matches = [g for g in gene_names if search_lower in g.lower() and g not in selected_genes][:100]

        return selected_options + [{"value": g, "label": g} for g in matches]

    # -------------------------------------------------------------------------
    # Config Drawer
    # -------------------------------------------------------------------------
    @app.callback(
        Output("config-drawer", "opened"),
        Output("config-drawer-store", "data"),
        Output("scatter-config-section", "style"),
        Output("expression-config-section", "style"),
        Output("crosstab-config-section", "style"),
        # Scatter config outputs
        Output("config-embedding", "value"),
        Output("config-color-type", "value"),
        Output("config-color-obs", "value"),
        Output("config-color-gene", "value"),
        Output("config-color-gene", "data"),  # Gene dropdown data
        Output("config-gene-symbols-col", "value"),
        Output("config-use-raw", "checked"),
        Output("config-colorscale", "value"),
        Output("config-vmax-type", "value"),
        Output("config-vmax-value", "value"),
        Output("config-point-size", "value"),
        Output("config-opacity", "value"),
        Output("config-groups-column", "value"),
        Output("config-groups-values", "value"),
        Output("config-groups-values", "data"),
        # Expression config outputs
        Output("config-genes", "value"),
        Output("config-genes", "data"),  # Genes dropdown data
        Output("config-groupby", "value"),
        Output("config-expr-gene-symbols-col", "value"),
        Output("config-expr-use-raw", "checked"),
        Output("config-expr-colorscale", "value"),
        # Crosstab config outputs
        Output("config-crosstab-groupby", "value"),
        Output("config-crosstab-compare", "value"),
        Output("config-crosstab-use-rep", "value"),
        Output("config-crosstab-colorscale", "value"),
        Output("config-crosstab-normalize", "checked"),
        Output("config-crosstab-dendrogram", "checked"),
        # Layout config outputs
        Output("config-panel-width", "value", allow_duplicate=True),
        Output("config-panel-height", "value", allow_duplicate=True),
        Output("config-layout-preset", "value"),
        Input({"type": "panel-config-btn", "index": ALL}, "n_clicks"),
        Input("config-apply-btn", "n_clicks"),
        State("pages-store", "data"),
        State("config-drawer-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_config_drawer(config_clicks, apply_click, pages_state, drawer_state):
        """Open/close config drawer and populate with panel settings."""
        triggered = ctx.triggered_id
        dm: DataManager = app.server.config["data_manager"]

        # Keep drawer open on apply - just prevent update to allow changes to take effect
        if triggered == "config-apply-btn":
            raise PreventUpdate

        if not any(config_clicks):
            raise PreventUpdate

        panel_id = triggered["index"]
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        panel = active_page.get("panels", {}).get(panel_id, {})
        panel_type = panel.get("type", "scatter")
        config = panel.get("config", {})

        # Determine section visibility based on panel type
        is_scatter = panel_type == "scatter"
        is_expression = panel_type in ["dotplot", "violin", "heatmap"]
        is_crosstab = panel_type == "crosstab"

        scatter_style = {} if is_scatter else {"display": "none"}
        expr_style = {} if is_expression else {"display": "none"}
        crosstab_style = {} if is_crosstab else {"display": "none"}

        # Get groups column options if set
        groups_col = config.get("groups_column")
        groups_data = []
        if groups_col and groups_col in dm.obs_columns:
            unique_vals = dm.get_obs_unique_values(groups_col)
            groups_data = [{"value": str(v), "label": str(v)} for v in unique_vals]

        # Auto-detect defaults for gene symbols and raw data
        detected_symbols_col = dm.detect_gene_symbols_column()
        use_raw_default = dm.has_raw

        # Use config values if set, otherwise use auto-detected defaults
        gene_symbols_col = config.get("gene_symbols_col") if config.get("gene_symbols_col") is not None else detected_symbols_col
        use_raw = config.get("use_raw") if config.get("use_raw") is not None else use_raw_default

        # Determine preset from width/height
        panel_width = panel.get("width", "50%")
        panel_height = panel.get("height", "400px")
        preset_map = {
            ("50%", "400px"): "half",
            ("33.33%", "400px"): "third",
            ("25%", "350px"): "quarter",
            ("100%", "400px"): "full",
            ("100%", "600px"): "full-tall",
            ("66.66%", "450px"): "two-thirds",
            ("50%", "600px"): "half-tall",
            ("50%", "300px"): "half-short",
            ("33.33%", "300px"): "third-short",
        }
        preset_value = preset_map.get((panel_width, panel_height), "custom")

        # Build gene dropdown data - include current value if set
        color_gene = config.get("color_gene")
        color_gene_data = [{"value": color_gene, "label": color_gene}] if color_gene else []

        var_names = config.get("var_names", [])
        genes_data = [{"value": g, "label": g} for g in var_names] if var_names else []

        return (
            True,
            {"panel_id": panel_id, "panel_type": panel_type},
            scatter_style,
            expr_style,
            crosstab_style,
            # Scatter config values
            config.get("embedding"),
            config.get("color_type", "obs"),
            config.get("color_obs"),
            color_gene,
            color_gene_data,  # Gene dropdown data with current value
            gene_symbols_col,
            use_raw,
            config.get("colorscale", "Viridis"),
            config.get("vmax_type", "auto"),
            config.get("vmax_value", 99.9),
            config.get("point_size", 5),
            config.get("opacity", 0.7),
            config.get("groups_column"),
            config.get("groups_values", []),
            groups_data,
            # Expression config values
            var_names,
            genes_data,  # Genes dropdown data with current values
            config.get("groupby"),
            gene_symbols_col,  # Same auto-detected value for expression config
            use_raw,  # Same auto-detected value for expression config
            config.get("colorscale", "Reds"),
            # Crosstab config values
            config.get("groupby"),
            config.get("compare_col"),
            config.get("use_rep", dm.get_default_representation()),
            config.get("colorscale", "YlOrRd"),
            config.get("normalize", True),
            config.get("show_dendrogram", True),
            # Layout config values (from panel, not config)
            panel_width,
            panel_height,
            preset_value,
        )

    # Callback to update groups values dropdown when column changes
    @app.callback(
        Output("config-groups-values", "data", allow_duplicate=True),
        Input("config-groups-column", "value"),
        prevent_initial_call=True,
    )
    def update_groups_values_options(groups_column):
        """Update the groups values dropdown based on selected column."""
        if not groups_column:
            return []
        dm: DataManager = app.server.config["data_manager"]
        if groups_column not in dm.obs_columns:
            return []
        unique_vals = dm.get_obs_unique_values(groups_column)
        return [{"value": str(v), "label": str(v)} for v in unique_vals]

    # Callback to update width/height when preset is selected
    @app.callback(
        Output("config-panel-width", "value"),
        Output("config-panel-height", "value"),
        Output("layout-advanced-accordion", "value"),
        Input("config-layout-preset", "value"),
        prevent_initial_call=True,
    )
    def update_layout_from_preset(preset):
        """Update width/height values when a preset is selected."""
        # Define preset mappings
        presets = {
            "half": ("50%", "400px"),
            "third": ("33.33%", "400px"),
            "quarter": ("25%", "350px"),
            "full": ("100%", "400px"),
            "full-tall": ("100%", "600px"),
            "two-thirds": ("66.66%", "450px"),
            "half-tall": ("50%", "600px"),
            "half-short": ("50%", "300px"),
            "third-short": ("33.33%", "300px"),
        }

        if preset == "custom":
            # Open the accordion for custom settings
            return no_update, no_update, "advanced"

        if preset in presets:
            width, height = presets[preset]
            return width, height, None  # None closes the accordion

        return no_update, no_update, no_update

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input("config-apply-btn", "n_clicks"),
        State("config-drawer-store", "data"),
        # Scatter config states
        State("config-embedding", "value"),
        State("config-color-type", "value"),
        State("config-color-obs", "value"),
        State("config-color-gene", "value"),
        State("config-gene-symbols-col", "value"),
        State("config-use-raw", "checked"),
        State("config-colorscale", "value"),
        State("config-vmax-type", "value"),
        State("config-vmax-value", "value"),
        State("config-point-size", "value"),
        State("config-opacity", "value"),
        State("config-groups-column", "value"),
        State("config-groups-values", "value"),
        # Expression config states
        State("config-genes", "value"),
        State("config-groupby", "value"),
        State("config-expr-gene-symbols-col", "value"),
        State("config-expr-use-raw", "checked"),
        State("config-expr-colorscale", "value"),
        # Crosstab config states
        State("config-crosstab-groupby", "value"),
        State("config-crosstab-compare", "value"),
        State("config-crosstab-use-rep", "value"),
        State("config-crosstab-colorscale", "value"),
        State("config-crosstab-normalize", "checked"),
        State("config-crosstab-dendrogram", "checked"),
        # Layout config states
        State("config-panel-width", "value"),
        State("config-panel-height", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def apply_config(n_clicks, drawer_state,
                     # Scatter config
                     embedding, color_type, color_obs, color_gene,
                     gene_symbols_col, use_raw, colorscale, vmax_type, vmax_value,
                     point_size, opacity, groups_column, groups_values,
                     # Expression config
                     genes, groupby, expr_gene_symbols_col, expr_use_raw, expr_colorscale,
                     # Crosstab config
                     crosstab_groupby, crosstab_compare, crosstab_use_rep,
                     crosstab_colorscale, crosstab_normalize, crosstab_dendrogram,
                     # Layout config
                     panel_width, panel_height,
                     pages_state):
        """Apply configuration changes to a panel."""
        if not n_clicks or not drawer_state.get("panel_id"):
            raise PreventUpdate

        panel_id = drawer_state["panel_id"]
        panel_type = drawer_state["panel_type"]
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state["pages"][active_page_id]

        if panel_id not in active_page["panels"]:
            raise PreventUpdate

        if panel_type == "scatter":
            active_page["panels"][panel_id]["config"] = {
                "embedding": embedding,
                "color_type": color_type,
                "color_obs": color_obs if color_type == "obs" else None,
                "color_gene": color_gene if color_type == "gene" else None,
                "gene_symbols_col": gene_symbols_col,
                "use_raw": use_raw,
                "colorscale": colorscale,
                "vmax_type": vmax_type,
                "vmax_value": vmax_value,
                "point_size": point_size,
                "opacity": opacity,
                "groups_column": groups_column,
                "groups_values": groups_values or [],
            }
        elif panel_type == "crosstab":
            active_page["panels"][panel_id]["config"] = {
                "groupby": crosstab_groupby,
                "compare_col": crosstab_compare,
                "use_rep": crosstab_use_rep,
                "colorscale": crosstab_colorscale,
                "normalize": crosstab_normalize,
                "show_dendrogram": crosstab_dendrogram,
            }
        else:
            # Expression plots (dotplot, violin, heatmap)
            active_page["panels"][panel_id]["config"] = {
                "var_names": genes or [],
                "groupby": groupby,
                "gene_symbols_col": expr_gene_symbols_col,
                "use_raw": expr_use_raw,
                "colorscale": expr_colorscale,
            }

        # Apply layout settings (width/height) to panel
        active_page["panels"][panel_id]["width"] = panel_width or "50%"
        active_page["panels"][panel_id]["height"] = panel_height or "400px"

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        grid = create_dashboard_grid(active_page["panels"], active_page.get("grid_layout"))
        return pages_state, grid

    # -------------------------------------------------------------------------
    # Selection Handling
    # -------------------------------------------------------------------------
    @app.callback(
        Output("selection-store", "data"),
        Output("selection-info", "children"),
        Output("create-page-btn", "disabled"),
        Output("export-selection-btn", "disabled"),
        Output("top-genes-btn", "disabled"),
        Input({"type": "panel-graph", "index": ALL}, "selectedData"),
        State("pages-store", "data"),
        State("scatter-cell-map", "data"),
        State({"type": "panel-graph", "index": ALL}, "id"),
    )
    def handle_selection(selected_data_list, pages_state, scatter_cell_map, panel_ids):
        """Handle selection from any panel.

        Note: Scattergl (WebGL) doesn't return customdata in selectedData,
        so we use the scatter-cell-map store to look up cell indices via
        (panel_id, curveNumber, pointIndex).
        """
        all_indices = []
        source_panel = None

        if not scatter_cell_map:
            scatter_cell_map = {}

        for i, selected_data in enumerate(selected_data_list or []):
            if selected_data and "points" in selected_data:
                # Get the panel_id for this graph
                panel_id = panel_ids[i]["index"] if i < len(panel_ids) else None
                panel_cell_map = scatter_cell_map.get(panel_id, [])

                for p in selected_data["points"]:
                    curve_num = p.get("curveNumber", 0)
                    point_idx = p.get("pointIndex", p.get("pointNumber"))

                    if point_idx is not None and curve_num < len(panel_cell_map):
                        trace_indices = panel_cell_map[curve_num]
                        if point_idx < len(trace_indices):
                            all_indices.append(trace_indices[point_idx])

                if all_indices:
                    source_panel = i

        if not all_indices:
            return (
                {"selected_indices": [], "source_panel": None},
                dmc.Text("No cells selected", size="sm", c="dimmed"),
                True, True, True,
            )

        # Deduplicate
        unique_indices = list(set(all_indices))
        n_selected = len(unique_indices)

        active_page = pages_state.get("pages", {}).get(pages_state.get("active_page", "root"), {})
        n_total = active_page.get("n_cells", 1)
        pct = (n_selected / n_total) * 100

        return (
            {"selected_indices": unique_indices, "source_panel": source_panel},
            dmc.Group([
                dmc.Text(f"{n_selected:,} cells selected", size="sm", fw=500),
                dmc.Text(f"({pct:.1f}%)", size="sm", c="dimmed"),
            ], gap="xs"),
            False, False, False,
        )

    @app.callback(
        Output("selection-store", "data", allow_duplicate=True),
        Input("clear-selection-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_selection(n_clicks):
        """Clear selection."""
        return {"selected_indices": [], "source_panel": None}

    # -------------------------------------------------------------------------
    # Page Management
    # -------------------------------------------------------------------------
    @app.callback(
        Output("new-page-modal", "opened"),
        Output("new-page-info", "children"),
        Input("create-page-btn", "n_clicks"),
        Input("new-page-cancel-btn", "n_clicks"),
        Input("new-page-create-btn", "n_clicks"),
        State("selection-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_new_page_modal(create_click, cancel_click, confirm_click, selection_data):
        """Toggle new page modal (for creating from selection)."""
        triggered = ctx.triggered_id
        if triggered in ["new-page-cancel-btn", "new-page-create-btn"]:
            return False, ""
        if triggered == "create-page-btn":
            n = len(selection_data.get("selected_indices", []))
            return True, f"Create page with {n:,} selected cells"
        raise PreventUpdate

    # -------------------------------------------------------------------------
    # Subset Page Modal Callbacks
    # -------------------------------------------------------------------------
    @app.callback(
        Output("subset-page-modal", "opened"),
        Input("add-page-btn", "n_clicks"),
        Input("subset-page-cancel-btn", "n_clicks"),
        Input("subset-page-create-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_subset_page_modal(add_click, cancel_click, create_click):
        """Toggle the subset page modal."""
        triggered = ctx.triggered_id
        if triggered == "add-page-btn":
            return True
        return False

    @app.callback(
        Output("subset-var-values", "data"),
        Output("subset-var-values", "disabled"),
        Output("subset-var-values", "value"),
        Output("subset-var-info", "children"),
        Input("subset-var-column", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def update_variable_values(column, pages_state):
        """Update available values when a variable column is selected."""
        if not column:
            return [], True, [], ""

        dm: DataManager = app.server.config["data_manager"]

        # Get the current page's cell indices
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        cell_indices = active_page.get("cell_indices")

        # Get unique values from the column for the current page subset
        if cell_indices is not None:
            values = dm.adata.obs.iloc[cell_indices][column].unique()
        else:
            values = dm.adata.obs[column].unique()

        # Convert to list and sort, handling different dtypes
        values_list = sorted([str(v) for v in values])
        options = [{"value": str(v), "label": str(v)} for v in values_list]

        return options, False, [], f"{len(values_list)} unique values available"

    @app.callback(
        Output("subset-var-info", "children", allow_duplicate=True),
        Output("subset-page-create-btn", "disabled", allow_duplicate=True),
        Input("subset-var-values", "value"),
        State("subset-var-column", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def preview_variable_subset(values, column, pages_state):
        """Preview how many cells match the variable selection."""
        if not values or not column:
            return "", True

        dm: DataManager = app.server.config["data_manager"]

        # Get the current page's cell indices
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        cell_indices = active_page.get("cell_indices")

        # Get the obs dataframe for current subset
        if cell_indices is not None:
            obs_df = dm.adata.obs.iloc[cell_indices]
        else:
            obs_df = dm.adata.obs

        # Count matching cells (convert to string for comparison since values are stored as strings)
        col_values = obs_df[column].astype(str)
        mask = col_values.isin(values)
        n_matching = mask.sum()
        total = len(obs_df)

        return f"{n_matching:,} / {total:,} cells match ({n_matching/total*100:.1f}%)", n_matching == 0

    @app.callback(
        Output("subset-gene-name", "data", allow_duplicate=True),
        Input("subset-gene-name", "searchValue"),
        State("subset-gene-use-raw", "checked"),
        prevent_initial_call=True,
    )
    def search_genes_subset(search_value, use_raw):
        """Search genes for subset gene selection."""
        if not search_value or len(search_value) < 2:
            return []

        dm: DataManager = app.server.config["data_manager"]
        gene_names = dm.get_gene_display_names(use_raw=use_raw)

        search_lower = search_value.lower()
        matches = [g for g in gene_names if search_lower in g.lower()][:100]

        return [{"value": g, "label": g} for g in matches]

    @app.callback(
        Output("subset-gene-info", "children"),
        Output("subset-page-create-btn", "disabled", allow_duplicate=True),
        Input("subset-gene-name", "value"),
        Input("subset-gene-operator", "value"),
        Input("subset-gene-threshold", "value"),
        State("subset-gene-use-raw", "checked"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def preview_gene_subset(gene_name, operator, threshold, use_raw, pages_state):
        """Preview how many cells match the gene expression filter."""
        if not gene_name or threshold is None:
            return "", True

        dm: DataManager = app.server.config["data_manager"]

        # Get the current page's cell indices
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        cell_indices = active_page.get("cell_indices")

        try:
            # Get gene expression (use gene_symbols_column for proper mapping)
            gene_symbols_col = dm.detect_gene_symbols_column()
            expr = dm.get_gene_expression(gene_name, use_raw=use_raw, gene_symbols_column=gene_symbols_col)

            # Apply current page subset if any
            if cell_indices is not None:
                expr = expr[cell_indices]
            total = len(expr)

            # Apply operator
            if operator == ">":
                mask = expr > threshold
            elif operator == ">=":
                mask = expr >= threshold
            elif operator == "<":
                mask = expr < threshold
            elif operator == "<=":
                mask = expr <= threshold
            elif operator == "==":
                mask = expr == threshold
            else:
                return "Invalid operator", True

            n_matching = mask.sum()
            return f"{n_matching:,} / {total:,} cells match ({n_matching/total*100:.1f}%)", n_matching == 0

        except Exception as e:
            return f"Error: {str(e)}", True

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("subset-page-modal", "opened", allow_duplicate=True),
        Input("subset-page-create-btn", "n_clicks"),
        State("subset-page-tabs", "value"),
        State("subset-var-column", "value"),
        State("subset-var-values", "value"),
        State("subset-gene-name", "value"),
        State("subset-gene-operator", "value"),
        State("subset-gene-threshold", "value"),
        State("subset-gene-use-raw", "checked"),
        State("subset-page-name", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def create_subset_page(n_clicks, tab, var_column, var_values, gene_name, gene_operator,
                           gene_threshold, gene_use_raw, page_name, pages_state):
        """Create a new page from variable or gene expression subset."""
        if not n_clicks:
            raise PreventUpdate

        dm: DataManager = app.server.config["data_manager"]

        # Get the current page's cell indices
        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state.get("pages", {}).get(active_page_id, {})
        parent_cell_indices = active_page.get("cell_indices")

        # Get the obs dataframe for current subset
        if parent_cell_indices is not None:
            obs_df = dm.adata.obs.iloc[parent_cell_indices]
            base_indices = np.array(parent_cell_indices)
        else:
            obs_df = dm.adata.obs
            base_indices = np.arange(dm.n_cells)

        # Determine which cells to include based on active tab
        if tab == "variable":
            if not var_column or not var_values:
                raise PreventUpdate
            col_values = obs_df[var_column].astype(str)
            mask = col_values.isin(var_values)
            auto_name = f"{var_column}: {', '.join(var_values[:3])}{'...' if len(var_values) > 3 else ''}"
        else:  # gene expression
            if not gene_name or gene_threshold is None:
                raise PreventUpdate
            gene_symbols_col = dm.detect_gene_symbols_column()
            expr = dm.get_gene_expression(gene_name, use_raw=gene_use_raw, gene_symbols_column=gene_symbols_col)
            if parent_cell_indices is not None:
                expr = expr[parent_cell_indices]

            if gene_operator == ">":
                mask = expr > gene_threshold
            elif gene_operator == ">=":
                mask = expr >= gene_threshold
            elif gene_operator == "<":
                mask = expr < gene_threshold
            elif gene_operator == "<=":
                mask = expr <= gene_threshold
            elif gene_operator == "==":
                mask = expr == gene_threshold
            else:
                raise PreventUpdate

            auto_name = f"{gene_name} {gene_operator} {gene_threshold}"

        # Get the actual cell indices
        subset_indices = base_indices[mask].tolist()
        n_subset = len(subset_indices)

        if n_subset == 0:
            raise PreventUpdate

        # Determine page name
        final_name = page_name if page_name else auto_name

        # Create the page in the database
        page_repo: PageRepository = app.server.config["page_repo"]
        layout_repo: LayoutRepository = app.server.config["layout_repo"]

        try:
            db_page = page_repo.create(
                name=final_name,
                cell_indices=np.array(subset_indices),
                parent_page_id=active_page_id,
            )
            page_id = db_page.id
        except Exception as e:
            print(f"Warning: Could not create page in database: {e}")
            page_id = f"page_{uuid.uuid4().hex[:8]}"

        # Get default embedding
        default_emb = get_default_embedding(dm)

        new_page = {
            "name": final_name,
            "cell_indices": subset_indices,
            "n_cells": n_subset,
            "parent_id": active_page_id,
            "panels": {
                "panel-1": {
                    "type": "scatter",
                    "config": {
                        "embedding": default_emb,
                        "color_type": "obs",
                        "color_obs": None,
                        "colorscale": "Viridis",
                        "point_size": DEFAULT_POINT_SIZE,
                        "opacity": DEFAULT_OPACITY,
                    },
                    "width": DEFAULT_PANEL_WIDTH,
                    "height": DEFAULT_PANEL_HEIGHT,
                }
            },
            "grid_layout": {"lg": [{"i": "panel-1", "x": 0, "y": 0, "w": 6, "h": 5}]},
        }

        # Save layout to DB
        try:
            layout_repo.save_layout(
                page_id=page_id,
                panels=new_page["panels"],
                grid_layout=new_page["grid_layout"],
            )
        except Exception as e:
            print(f"Warning: Could not save layout to database: {e}")

        # Update pages store
        updated_pages = dict(pages_state)
        updated_pages["pages"][page_id] = new_page
        updated_pages["active_page"] = page_id

        # Generate page list
        page_list_children = []
        for pid, page in updated_pages["pages"].items():
            is_active = pid == page_id
            is_root = pid == "root"
            page_list_children.append(
                create_page_item(pid, page["name"], page["n_cells"], is_active, is_root)
            )

        # Render dashboard for new page
        grid = create_dashboard_grid(new_page["panels"], new_page["grid_layout"])

        notification = create_notification(
            f"Page Created: {final_name}",
            f"Created subset with {n_subset:,} cells",
            color="green",
        )

        return updated_pages, page_list_children, grid, notification, False

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children"),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("new-page-create-btn", "n_clicks"),
        State("new-page-name", "value"),
        State("selection-store", "data"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def create_new_page(n_clicks, name, selection_data, pages_state):
        """Create a new page from selection."""
        if not n_clicks or not selection_data.get("selected_indices"):
            raise PreventUpdate

        dm: DataManager = app.server.config["data_manager"]
        indices = selection_data["selected_indices"]

        # Handle nested indices
        active_page_id = pages_state.get("active_page", "root")
        parent_page = pages_state["pages"][active_page_id]
        parent_indices = parent_page.get("cell_indices")

        if parent_indices:
            actual_indices = [parent_indices[i] for i in indices if i < len(parent_indices)]
        else:
            actual_indices = indices

        page_name = name or f"Subset ({len(actual_indices):,} cells)"
        default_emb = get_default_embedding(dm)

        # Create page in DB first to get the correct page_id
        page_repo: PageRepository = app.server.config["page_repo"]
        layout_repo: LayoutRepository = app.server.config["layout_repo"]

        try:
            db_page = page_repo.create(
                name=page_name,
                cell_indices=np.array(actual_indices),
                parent_page_id=active_page_id,
            )
            page_id = db_page.id  # Use the DB-generated page_id
        except Exception as e:
            print(f"Warning: Could not create page in database: {e}")
            # Fallback to local ID if DB fails
            page_id = f"page_{uuid.uuid4().hex[:8]}"

        new_page = {
            "id": page_id,
            "name": page_name,
            "cell_indices": actual_indices,
            "parent_id": active_page_id,
            "n_cells": len(actual_indices),
            "grid_layout": {},  # Kept for backward compatibility
            "panels": {
                "panel-0": {
                    "type": "scatter",
                    "config": {"embedding": default_emb, "colorscale": "Viridis", "point_size": DEFAULT_POINT_SIZE, "opacity": DEFAULT_OPACITY},
                    "width": DEFAULT_PANEL_WIDTH,
                    "height": DEFAULT_PANEL_HEIGHT,
                }
            },
        }

        pages_state["pages"][page_id] = new_page
        pages_state["page_order"].append(page_id)
        pages_state["active_page"] = page_id

        # Save initial layout for the page
        try:
            layout_repo.create(
                page_id=page_id,
                layout_json={
                    "grid_layout": new_page["grid_layout"],
                    "panels": new_page["panels"],
                },
                name="Default",
                is_active=True,
            )
        except Exception as e:
            print(f"Warning: Could not save layout to database: {e}")

        # Rebuild page list
        page_list = [
            create_page_item(
                pid,
                pages_state["pages"][pid]["name"],
                pages_state["pages"][pid]["n_cells"],
                pid == page_id,
                pages_state["pages"][pid].get("parent_id") is None,
            )
            for pid in pages_state["page_order"]
        ]

        grid = create_dashboard_grid(new_page["panels"], new_page["grid_layout"])
        notification = create_notification(
            "Page Created",
            f"Created '{page_name}' with {len(actual_indices):,} cells",
            color="green",
            icon="tabler:file-plus",
        )
        return pages_state, page_list, grid, notification

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input({"type": "page-item", "index": ALL}, "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def switch_page(n_clicks, pages_state):
        """Switch to a different page."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or "index" not in triggered:
            raise PreventUpdate

        new_active = triggered["index"]
        if new_active == pages_state.get("active_page"):
            raise PreventUpdate

        pages_state["active_page"] = new_active
        active_page = pages_state["pages"][new_active]

        page_list = [
            create_page_item(
                pid,
                pages_state["pages"][pid]["name"],
                pages_state["pages"][pid]["n_cells"],
                pid == new_active,
                pages_state["pages"][pid].get("parent_id") is None,
            )
            for pid in pages_state["page_order"]
        ]

        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        return pages_state, page_list, grid

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "delete-page-btn", "index": ALL}, "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def delete_page(n_clicks, pages_state):
        """Delete a page."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        page_id = triggered["index"]

        # Can't delete root
        if page_id == "root":
            raise PreventUpdate

        # Get page name before deletion
        page_name = pages_state["pages"].get(page_id, {}).get("name", "Page")

        # Delete from database
        try:
            page_repo: PageRepository = app.server.config["page_repo"]
            page_repo.delete(page_id)
        except Exception as e:
            print(f"Warning: Could not delete page from database: {e}")

        # Remove page from state
        if page_id in pages_state["pages"]:
            del pages_state["pages"][page_id]
        if page_id in pages_state["page_order"]:
            pages_state["page_order"].remove(page_id)

        # Switch to root if deleted active page
        if pages_state.get("active_page") == page_id:
            pages_state["active_page"] = "root"

        active_page = pages_state["pages"][pages_state["active_page"]]

        page_list = [
            create_page_item(
                pid,
                pages_state["pages"][pid]["name"],
                pages_state["pages"][pid]["n_cells"],
                pid == pages_state["active_page"],
                pages_state["pages"][pid].get("parent_id") is None,
            )
            for pid in pages_state["page_order"]
        ]

        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        notification = create_notification(
            "Page Deleted",
            f"'{page_name}' has been removed",
            color="red",
            icon="tabler:trash",
        )
        return pages_state, page_list, grid, notification

    # -------------------------------------------------------------------------
    # Page Rename
    # -------------------------------------------------------------------------
    @app.callback(
        Output("rename-page-modal", "opened"),
        Output("rename-page-store", "data"),
        Output("rename-page-input", "value"),
        Input({"type": "rename-page-btn", "index": ALL}, "n_clicks"),
        Input("rename-page-cancel-btn", "n_clicks"),
        Input("rename-page-confirm-btn", "n_clicks"),
        State("pages-store", "data"),
        State("rename-page-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_rename_modal(rename_clicks, cancel_click, confirm_click, pages_state, rename_store):
        """Toggle rename page modal."""
        triggered = ctx.triggered_id

        # Close modal on cancel or confirm
        if triggered in ["rename-page-cancel-btn", "rename-page-confirm-btn"]:
            return False, {"page_id": None}, ""

        # Open modal when rename button clicked
        if not any(rename_clicks or []):
            raise PreventUpdate

        page_id = triggered["index"]
        current_name = pages_state.get("pages", {}).get(page_id, {}).get("name", "")
        return True, {"page_id": page_id}, current_name

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Input("rename-page-confirm-btn", "n_clicks"),
        State("rename-page-input", "value"),
        State("rename-page-store", "data"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def apply_page_rename(n_clicks, new_name, rename_store, pages_state):
        """Apply page rename."""
        if not n_clicks or not new_name:
            raise PreventUpdate

        page_id = rename_store.get("page_id")
        if not page_id or page_id not in pages_state.get("pages", {}):
            raise PreventUpdate

        # Update in database
        try:
            page_repo: PageRepository = app.server.config["page_repo"]
            page_repo.update_name(page_id, new_name)
        except Exception as e:
            print(f"Warning: Could not update page name in database: {e}")

        pages_state["pages"][page_id]["name"] = new_name

        page_list = [
            create_page_item(
                pid,
                pages_state["pages"][pid]["name"],
                pages_state["pages"][pid]["n_cells"],
                pid == pages_state.get("active_page"),
                pages_state["pages"][pid].get("parent_id") is None,
            )
            for pid in pages_state["page_order"]
        ]

        return pages_state, page_list

    # -------------------------------------------------------------------------
    # Layout Save/Load
    # -------------------------------------------------------------------------
    @app.callback(
        Output("save-layout-modal", "opened"),
        Input("save-layout-btn", "n_clicks"),
        Input("save-layout-cancel-btn", "n_clicks"),
        Input("save-layout-confirm-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_save_modal(open_click, cancel, confirm):
        triggered = ctx.triggered_id
        return triggered == "save-layout-btn"

    @app.callback(
        Output("layout-download", "data"),
        Input("save-layout-confirm-btn", "n_clicks"),
        State("save-layout-name", "value"),
        State("save-layout-description", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def download_layout(n_clicks, name, desc, pages_state):
        if not n_clicks:
            raise PreventUpdate
        layout = serialize_layout(pages_state.get("pages", {}), pages_state.get("active_page", "root"), name or "layout", desc or "")
        filename = f"{name or 'layout'}.json".replace(" ", "_").lower()
        return dict(content=json.dumps(layout, indent=2), filename=filename)

    @app.callback(
        Output("load-layout-modal", "opened"),
        Input("load-layout-btn", "n_clicks"),
        Input("load-layout-cancel-btn", "n_clicks"),
        Input("load-layout-apply-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_load_modal(open_click, cancel, apply):
        triggered = ctx.triggered_id
        return triggered == "load-layout-btn"

    @app.callback(
        Output("layout-upload-store", "data"),
        Output("layout-upload-preview", "children"),
        Output("load-layout-apply-btn", "disabled"),
        Input("layout-upload", "contents"),
        State("layout-upload", "filename"),
        prevent_initial_call=True,
    )
    def parse_uploaded_layout(contents, filename):
        if not contents:
            raise PreventUpdate
        try:
            content_string = contents.split(",")[1]
            decoded = base64.b64decode(content_string).decode("utf-8")
            layout_data = json.loads(decoded)
            preview = dmc.Alert(f"Loaded: {layout_data.get('name', filename)} ({len(layout_data.get('pages', {}))} pages)", color="green")
            return layout_data, preview, False
        except Exception as e:
            return None, dmc.Alert(f"Error: {str(e)}", color="red"), True

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Output("load-layout-modal", "opened", allow_duplicate=True),
        Input("load-layout-apply-btn", "n_clicks"),
        State("layout-upload-store", "data"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def apply_loaded_layout(n_clicks, layout_data, pages_state):
        if not n_clicks or not layout_data:
            raise PreventUpdate

        try:
            result = deserialize_layout(layout_data)
            pages_state["pages"] = result["pages"]
            pages_state["active_page"] = result.get("active_page", "root")
            pages_state["page_order"] = list(result["pages"].keys())

            active_page = pages_state["pages"][pages_state["active_page"]]

            # Persist all loaded pages to database
            layout_repo: LayoutRepository = app.server.config["layout_repo"]
            for page_id, page_data in pages_state["pages"].items():
                persist_page_layout(layout_repo, page_id, page_data)

            page_list = [
                create_page_item(
                    pid,
                    pages_state["pages"][pid]["name"],
                    pages_state["pages"][pid].get("n_cells", 0),
                    pid == pages_state["active_page"],
                    pages_state["pages"][pid].get("parent_id") is None,
                )
                for pid in pages_state["page_order"]
            ]

            grid = create_dashboard_grid(active_page.get("panels", {}), active_page.get("grid_layout", {}))
            return pages_state, page_list, grid, False
        except Exception:
            raise PreventUpdate

    # -------------------------------------------------------------------------
    # Reset/Clear Layout
    # -------------------------------------------------------------------------
    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("page-list", "children", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input("reset-layout-btn", "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def reset_layout(n_clicks, pages_state):
        if not n_clicks:
            raise PreventUpdate

        dm: DataManager = app.server.config["data_manager"]
        default_emb = get_default_embedding(dm)

        # Reset to single panel
        active_page_id = pages_state.get("active_page", "root")
        pages_state["pages"][active_page_id]["panels"] = {
            "panel-0": {
                "type": "scatter",
                "config": {"embedding": default_emb, "colorscale": "Viridis", "point_size": DEFAULT_POINT_SIZE, "opacity": DEFAULT_OPACITY},
            }
        }
        pages_state["pages"][active_page_id]["grid_layout"] = {
            "lg": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
            "md": [{"i": "panel-0", "x": 0, "y": 0, "w": 5, "h": 5, "minW": 3, "minH": 2}],
            "sm": [{"i": "panel-0", "x": 0, "y": 0, "w": 6, "h": 5, "minW": 3, "minH": 2}],
            "xs": [{"i": "panel-0", "x": 0, "y": 0, "w": 4, "h": 5, "minW": 2, "minH": 2}],
        }
        pages_state["next_panel_id"] = 1

        active_page = pages_state["pages"][active_page_id]

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        page_list = [
            create_page_item(pid, pages_state["pages"][pid]["name"], pages_state["pages"][pid]["n_cells"],
                           pid == active_page_id, pages_state["pages"][pid].get("parent_id") is None)
            for pid in pages_state["page_order"]
        ]
        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        return pages_state, page_list, grid

    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input("clear-panels-btn", "n_clicks"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def clear_all_panels(n_clicks, pages_state):
        if not n_clicks:
            raise PreventUpdate

        active_page_id = pages_state.get("active_page", "root")
        pages_state["pages"][active_page_id]["panels"] = {}
        pages_state["pages"][active_page_id]["grid_layout"] = {"lg": [], "md": [], "sm": [], "xs": []}

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, pages_state["pages"][active_page_id])

        grid = create_dashboard_grid({}, {"lg": [], "md": [], "sm": [], "xs": []})
        return pages_state, grid

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    @app.callback(
        Output("export-modal", "opened"),
        Output("export-info", "children"),
        Input("export-selection-btn", "n_clicks"),
        Input("export-cancel-btn", "n_clicks"),
        Input("export-confirm-btn", "n_clicks"),
        State("selection-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_export_modal(export_click, cancel, confirm, selection_data):
        triggered = ctx.triggered_id
        if triggered == "export-selection-btn":
            n = len(selection_data.get("selected_indices", []))
            return True, f"Export {n:,} selected cells to a new .h5ad file"
        return False, ""

    @app.callback(
        Output("export-download", "data"),
        Input("export-confirm-btn", "n_clicks"),
        State("export-filename", "value"),
        State("export-include-raw", "checked"),
        State("export-include-obsm", "checked"),
        State("selection-store", "data"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def do_export(n_clicks, filename, include_raw, include_obsm, selection_data, pages_state):
        if not n_clicks:
            raise PreventUpdate

        dm: DataManager = app.server.config["data_manager"]
        indices = selection_data.get("selected_indices", [])

        if not indices:
            raise PreventUpdate

        # Handle nested indices
        active_page = pages_state["pages"][pages_state.get("active_page", "root")]
        parent_indices = active_page.get("cell_indices")
        if parent_indices:
            actual_indices = [parent_indices[i] for i in indices if i < len(parent_indices)]
        else:
            actual_indices = indices

        # Export to temp file
        import tempfile
        fname = filename or "subset"
        temp_path = Path(tempfile.gettempdir()) / f"{fname}.h5ad"

        export_cell_subset(
            dm.adata,
            np.array(actual_indices),
            temp_path,
            include_raw=include_raw,
            include_obsm=include_obsm,
        )

        # Read and return for download
        with open(temp_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        return dict(content=content, filename=f"{fname}.h5ad", base64=True)

    # -------------------------------------------------------------------------
    # Top Genes Analysis
    # -------------------------------------------------------------------------
    @app.callback(
        Output("top-genes-modal", "opened"),
        Output("top-genes-info", "children"),
        Output("top-genes-table-container", "children"),
        Output("top-genes-data-store", "data"),
        Input("top-genes-btn", "n_clicks"),
        Input("top-genes-close-btn", "n_clicks"),
        Input("top-genes-refresh-btn", "n_clicks"),
        State("selection-store", "data"),
        State("pages-store", "data"),
        State("top-genes-count", "value"),
        State("top-genes-use-raw", "checked"),
        prevent_initial_call=True,
    )
    def handle_top_genes_modal(open_click, close_click, refresh_click, selection_data, pages_state, top_n, use_raw):
        """Open modal and compute top genes for selected cells."""
        triggered = ctx.triggered_id

        if triggered == "top-genes-close-btn":
            return False, "", no_update, no_update

        if triggered in ["top-genes-btn", "top-genes-refresh-btn"]:
            indices = selection_data.get("selected_indices", [])
            if not indices:
                return True, "No cells selected", dmc.Text("Select cells first", c="dimmed"), None

            dm: DataManager = app.server.config["data_manager"]

            # Handle nested indices (same as export)
            active_page = pages_state["pages"][pages_state.get("active_page", "root")]
            parent_indices = active_page.get("cell_indices")
            if parent_indices:
                actual_indices = [parent_indices[i] for i in indices if i < len(parent_indices)]
            else:
                actual_indices = indices

            # Compute top genes
            top_n_val = top_n or 100
            results_df = dm.compute_top_expressed_genes(
                cell_indices=actual_indices,
                top_n=top_n_val,
                use_raw=use_raw if use_raw is not None else True,
            )

            n_cells = len(actual_indices)
            info_text = f"Analyzing {n_cells:,} cells | Showing top {len(results_df)} genes"

            # Build table
            if len(results_df) == 0:
                table = dmc.Text("No expression data found", c="dimmed")
                data_for_download = None
            else:
                # Create a nice table using Mantine
                table_rows = [
                    html.Tr([
                        html.Th("Rank", style={"width": "60px"}),
                        html.Th("Gene Symbol"),
                        html.Th("Gene ID"),
                        html.Th("Mean Expr", style={"textAlign": "right"}),
                        html.Th("% Expressing", style={"textAlign": "right"}),
                    ])
                ]
                for idx, row in results_df.iterrows():
                    table_rows.append(
                        html.Tr([
                            html.Td(str(idx + 1), style={"color": "var(--mantine-color-dimmed)"}),
                            html.Td(dmc.Text(row['gene_symbol'], fw=500)),
                            html.Td(dmc.Text(row['gene_id'], size="sm", c="dimmed")),
                            html.Td(f"{row['mean_expression']:.4f}", style={"textAlign": "right"}),
                            html.Td(f"{row['pct_expressing']:.1f}%", style={"textAlign": "right"}),
                        ])
                    )

                table = dmc.Table(
                    children=[html.Tbody(table_rows)],
                    striped=True,
                    highlightOnHover=True,
                    withTableBorder=True,
                    withColumnBorders=True,
                )

                # Store data for download
                data_for_download = results_df.to_dict('records')

            return True, info_text, table, data_for_download

        raise PreventUpdate

    @app.callback(
        Output("top-genes-download", "data"),
        Input("top-genes-download-btn", "n_clicks"),
        State("top-genes-data-store", "data"),
        prevent_initial_call=True,
    )
    def download_top_genes(n_clicks, data):
        """Download top genes as CSV."""
        if not n_clicks or not data:
            raise PreventUpdate

        # Convert to CSV
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)

        return dict(content=csv_content, filename="top_genes.csv")

    # -------------------------------------------------------------------------
    # Keyboard Shortcuts
    # -------------------------------------------------------------------------
    app.clientside_callback(
        """
        function(id) {
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    // Trigger clear selection button click
                    const btn = document.getElementById('clear-selection-btn');
                    if (btn) btn.click();
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("keyboard-store", "data"),
        Input("keyboard-store", "id"),
    )

    # -------------------------------------------------------------------------
    # Global Color Quick Select
    # -------------------------------------------------------------------------
    @app.callback(
        Output("pages-store", "data", allow_duplicate=True),
        Output("dashboard-wrapper", "children", allow_duplicate=True),
        Input("global-color-select", "value"),
        State("pages-store", "data"),
        prevent_initial_call=True,
    )
    def apply_global_color(color_column, pages_state):
        """Apply color to all scatter plots on the page."""
        if not color_column:
            raise PreventUpdate

        active_page_id = pages_state.get("active_page", "root")
        active_page = pages_state["pages"][active_page_id]

        # Update all scatter panels
        for panel_id, panel in active_page.get("panels", {}).items():
            if panel.get("type") == "scatter":
                panel["config"]["color_type"] = "obs"
                panel["config"]["color_obs"] = color_column

        # Persist layout changes to database
        layout_repo: LayoutRepository = app.server.config["layout_repo"]
        persist_page_layout(layout_repo, active_page_id, active_page)

        grid = create_dashboard_grid(active_page["panels"], active_page["grid_layout"])
        return pages_state, grid


# =============================================================================
# PLOT RENDERING FUNCTIONS
# =============================================================================

def render_scatter(dm: DataManager, config: Dict, cell_indices: Optional[List], selected_indices: List, dark_mode: bool) -> tuple:
    """Render a scatter plot with support for groups filtering, vmax, and gene symbol mapping.

    Returns:
        tuple: (figure, cell_index_map) where cell_index_map is a list of lists,
               one list per trace, containing the cell indices for each point.
    """
    embedding = config.get("embedding")
    if not embedding:
        return go.Figure(), []

    # Track cell indices per trace for selection mapping
    trace_cell_indices = []

    color_type = config.get("color_type", "obs")
    color_obs = config.get("color_obs")
    color_gene = config.get("color_gene")
    colorscale = config.get("colorscale", "Viridis")
    point_size = config.get("point_size", 5)
    opacity = config.get("opacity", 0.7)

    # New options
    gene_symbols_col = config.get("gene_symbols_col")
    use_raw = config.get("use_raw", False)
    vmax_type = config.get("vmax_type", "auto")
    vmax_value = config.get("vmax_value", 99.9)
    groups_column = config.get("groups_column")
    groups_values = config.get("groups_values", [])

    # Get color column
    color_by = color_obs if color_type == "obs" else color_gene
    metadata_cols = [color_obs] if color_obs and color_type == "obs" else []

    # Also get groups column if set
    if groups_column and groups_column not in metadata_cols:
        metadata_cols.append(groups_column)

    df = dm.get_embedding_dataframe(embedding, cell_indices, metadata_cols)

    # Add gene expression if needed
    if color_type == "gene" and color_gene:
        try:
            expr = dm.get_gene_expression(
                color_gene, cell_indices,
                use_raw=use_raw,
                gene_symbols_column=gene_symbols_col
            )
            df["gene_expr"] = expr
            color_by = "gene_expr"
        except ValueError as e:
            # Gene not found
            df["gene_expr"] = 0
            color_by = "gene_expr"

    fig = go.Figure()
    bg_color = "#1a1b1e" if dark_mode else "white"
    grid_color = "#373a40" if dark_mode else "#e9ecef"
    font_color = "#c1c2c5" if dark_mode else "#212529"
    grey_color = "#888888"

    is_categorical = False
    if color_by and color_by in df.columns:
        is_categorical = dm.is_categorical(color_by) if color_type == "obs" else False
        if not is_categorical and color_type == "obs":
            is_categorical = len(df[color_by].unique()) < 50

    # Handle groups filtering (grey out non-selected groups)
    has_groups_filter = groups_column and groups_values and groups_column in df.columns

    if is_categorical and color_by:
        colors = px.colors.qualitative.Plotly
        unique_cats = df[color_by].unique()

        # If groups filter is active, determine which cells to grey out
        if has_groups_filter:
            # First add grey background for cells NOT in selected groups
            grey_mask = ~df[groups_column].astype(str).isin(groups_values)
            if grey_mask.any():
                grey_cell_indices = df.loc[grey_mask, "cell_index"].values.tolist()
                fig.add_trace(go.Scattergl(
                    x=df.loc[grey_mask, "x"], y=df.loc[grey_mask, "y"],
                    mode="markers", name="Other",
                    marker=dict(size=point_size, opacity=opacity * 0.3, color=grey_color),
                    customdata=df.loc[grey_mask, "cell_index"].values,
                    hovertemplate=f"Index: %{{customdata}}<extra></extra>",
                    showlegend=True,
                ))
                trace_cell_indices.append(grey_cell_indices)

            # Then add colored points only for selected groups
            highlight_mask = df[groups_column].astype(str).isin(groups_values)
            df_highlight = df[highlight_mask]

            for i, cat in enumerate(df_highlight[color_by].unique()):
                mask = df_highlight[color_by] == cat
                color = colors[i % len(colors)]
                cat_cell_indices = df_highlight.loc[mask, "cell_index"].values.tolist()
                fig.add_trace(go.Scattergl(
                    x=df_highlight.loc[mask, "x"], y=df_highlight.loc[mask, "y"],
                    mode="markers", name=str(cat),
                    marker=dict(size=point_size, opacity=opacity, color=color),
                    customdata=df_highlight.loc[mask, "cell_index"].values,
                    hovertemplate=f"{color_by}: {cat}<br>Index: %{{customdata}}<extra></extra>",
                    showlegend=True,
                ))
                trace_cell_indices.append(cat_cell_indices)
        else:
            # No groups filter - show all with colors
            for i, cat in enumerate(unique_cats):
                mask = df[color_by] == cat
                color = colors[i % len(colors)]
                cat_cell_indices = df.loc[mask, "cell_index"].values.tolist()
                fig.add_trace(go.Scattergl(
                    x=df.loc[mask, "x"], y=df.loc[mask, "y"],
                    mode="markers", name=str(cat),
                    marker=dict(size=point_size, opacity=opacity, color=color),
                    customdata=df.loc[mask, "cell_index"].values,
                    hovertemplate=f"{color_by}: {cat}<br>Index: %{{customdata}}<extra></extra>",
                    showlegend=True,
                ))
                trace_cell_indices.append(cat_cell_indices)

    elif color_by and color_by in df.columns:
        # Continuous color (e.g., gene expression)
        color_values = df[color_by].values

        # Calculate vmax based on settings
        if vmax_type == "percentile" and vmax_value:
            vmax = np.percentile(color_values[color_values > 0], vmax_value) if np.any(color_values > 0) else color_values.max()
        elif vmax_type == "fixed" and vmax_value:
            vmax = vmax_value
        else:  # auto
            vmax = color_values.max()

        vmin = 0  # Expression typically starts at 0

        if has_groups_filter:
            # Grey out cells not in selected groups
            grey_mask = ~df[groups_column].astype(str).isin(groups_values)
            if grey_mask.any():
                grey_cell_indices = df.loc[grey_mask, "cell_index"].values.tolist()
                fig.add_trace(go.Scattergl(
                    x=df.loc[grey_mask, "x"], y=df.loc[grey_mask, "y"],
                    mode="markers", name="Other",
                    marker=dict(size=point_size, opacity=opacity * 0.3, color=grey_color),
                    customdata=df.loc[grey_mask, "cell_index"].values,
                    hovertemplate=f"Index: %{{customdata}}<extra></extra>",
                    showlegend=False,
                ))
                trace_cell_indices.append(grey_cell_indices)

            # Colored points for selected groups
            highlight_mask = df[groups_column].astype(str).isin(groups_values)
            df_highlight = df[highlight_mask]
            highlight_cell_indices = df_highlight["cell_index"].values.tolist()
            fig.add_trace(go.Scattergl(
                x=df_highlight["x"], y=df_highlight["y"], mode="markers",
                marker=dict(
                    size=point_size,
                    color=df_highlight[color_by],
                    colorscale=colorscale,
                    cmin=vmin, cmax=vmax,
                    colorbar=dict(title=color_gene if color_type == "gene" else color_obs),
                    opacity=opacity
                ),
                customdata=df_highlight["cell_index"].values,
                hovertemplate=f"Value: %{{marker.color:.2f}}<br>Index: %{{customdata}}<extra></extra>",
            ))
            trace_cell_indices.append(highlight_cell_indices)
        else:
            all_cell_indices = df["cell_index"].values.tolist()
            fig.add_trace(go.Scattergl(
                x=df["x"], y=df["y"], mode="markers",
                marker=dict(
                    size=point_size,
                    color=color_values,
                    colorscale=colorscale,
                    cmin=vmin, cmax=vmax,
                    colorbar=dict(title=color_gene if color_type == "gene" else color_obs),
                    opacity=opacity
                ),
                customdata=df["cell_index"].values,
                hovertemplate=f"Value: %{{marker.color:.2f}}<br>Index: %{{customdata}}<extra></extra>",
            ))
            trace_cell_indices.append(all_cell_indices)
    else:
        # No coloring - just points
        if has_groups_filter:
            grey_mask = ~df[groups_column].astype(str).isin(groups_values)
            if grey_mask.any():
                grey_cell_indices = df.loc[grey_mask, "cell_index"].values.tolist()
                fig.add_trace(go.Scattergl(
                    x=df.loc[grey_mask, "x"], y=df.loc[grey_mask, "y"],
                    mode="markers", name="Other",
                    marker=dict(size=point_size, opacity=opacity * 0.3, color=grey_color),
                    customdata=df.loc[grey_mask, "cell_index"].values,
                    hovertemplate="Index: %{customdata}<extra></extra>",
                    showlegend=False,
                ))
                trace_cell_indices.append(grey_cell_indices)

            highlight_mask = df[groups_column].astype(str).isin(groups_values)
            df_highlight = df[highlight_mask]
            highlight_cell_indices = df_highlight["cell_index"].values.tolist()
            fig.add_trace(go.Scattergl(
                x=df_highlight["x"], y=df_highlight["y"], mode="markers",
                marker=dict(size=point_size, color="steelblue", opacity=opacity),
                customdata=df_highlight["cell_index"].values,
                hovertemplate="Index: %{customdata}<extra></extra>",
            ))
            trace_cell_indices.append(highlight_cell_indices)
        else:
            all_cell_indices = df["cell_index"].values.tolist()
            fig.add_trace(go.Scattergl(
                x=df["x"], y=df["y"], mode="markers",
                marker=dict(size=point_size, color="steelblue", opacity=opacity),
                customdata=df["cell_index"].values,
                hovertemplate="Index: %{customdata}<extra></extra>",
            ))
            trace_cell_indices.append(all_cell_indices)

    dims = dm.get_embedding_dims(embedding)
    fig.update_layout(
        title=f"{embedding.replace('X_', '').upper()} ({len(df):,} cells)",
        dragmode="lasso", hovermode="closest",
        paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=font_color),
        xaxis=dict(title=dims[0] if dims else "Dim 1", showgrid=True, gridcolor=grid_color, zeroline=False),
        yaxis=dict(title=dims[1] if len(dims) > 1 else "Dim 2", showgrid=True, gridcolor=grid_color, zeroline=False, scaleanchor="x"),
        margin=dict(l=40, r=120, t=40, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            itemsizing="constant",
        ) if is_categorical or has_groups_filter else None,
    )

    # Configure selection appearance - selected points stay visible, unselected dim
    fig.update_traces(
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.1)),
    )

    return fig, trace_cell_indices


def render_dotplot(dm: DataManager, config: Dict, cell_indices: Optional[List], dark_mode: bool) -> go.Figure:
    """Render a dot plot."""
    var_names = config.get("var_names", [])
    groupby = config.get("groupby")
    colorscale = config.get("colorscale", "Reds")
    use_raw = config.get("use_raw", False)
    gene_symbols_col = config.get("gene_symbols_col")

    if not var_names or not groupby:
        fig = go.Figure()
        fig.add_annotation(text="Configure genes and groupby", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get expression data with new options
    expr_df = dm.get_genes_expression(var_names, cell_indices, use_raw=use_raw, gene_symbols_column=gene_symbols_col)
    groups = dm.get_obs_column(groupby, cell_indices)

    if expr_df.empty or groups is None:
        return go.Figure()

    # Calculate stats
    stats = []
    for gene in var_names:
        if gene not in expr_df.columns:
            continue
        for group in np.unique(groups):
            mask = groups == group
            gene_expr = expr_df[gene].values[mask]
            frac = np.mean(gene_expr > 0)
            mean_expr = np.mean(gene_expr)
            stats.append({"gene": gene, "group": group, "frac": frac, "mean": mean_expr})

    if not stats:
        return go.Figure()

    stats_df = pd.DataFrame(stats)
    max_frac = stats_df["frac"].max() or 1
    max_mean = stats_df["mean"].max() or 1

    bg_color = "#1a1b1e" if dark_mode else "white"
    font_color = "#c1c2c5" if dark_mode else "#212529"

    fig = go.Figure(go.Scatter(
        x=stats_df["gene"], y=stats_df["group"], mode="markers",
        marker=dict(
            size=(stats_df["frac"] / max_frac) * 20 + 3,
            color=stats_df["mean"], colorscale=colorscale,
            colorbar=dict(title="Mean Expr"),
        ),
        hovertemplate="Gene: %{x}<br>Group: %{y}<br>Frac: %{customdata[0]:.2%}<br>Mean: %{customdata[1]:.2f}<extra></extra>",
        customdata=np.column_stack([stats_df["frac"], stats_df["mean"]]),
    ))

    fig.update_layout(
        title="Dot Plot", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=font_color),
        xaxis=dict(title="Genes", tickangle=45), yaxis=dict(title=groupby),
        margin=dict(l=100, r=60, t=40, b=80),
    )
    return fig


def render_violin(dm: DataManager, config: Dict, cell_indices: Optional[List], dark_mode: bool) -> go.Figure:
    """Render a violin plot."""
    var_names = config.get("var_names", [])
    groupby = config.get("groupby")
    colorscale = config.get("colorscale", "Reds")
    use_raw = config.get("use_raw", False)
    gene_symbols_col = config.get("gene_symbols_col")

    if not var_names or not groupby:
        fig = go.Figure()
        fig.add_annotation(text="Configure genes and groupby", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    expr_df = dm.get_genes_expression(var_names, cell_indices, use_raw=use_raw, gene_symbols_column=gene_symbols_col)
    groups = dm.get_obs_column(groupby, cell_indices)

    if expr_df.empty or groups is None:
        return go.Figure()

    bg_color = "#1a1b1e" if dark_mode else "white"
    font_color = "#c1c2c5" if dark_mode else "#212529"

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    unique_groups = np.unique(groups)

    for i, gene in enumerate(var_names[:5]):  # Limit genes
        if gene not in expr_df.columns:
            continue
        for j, group in enumerate(unique_groups):
            mask = groups == group
            fig.add_trace(go.Violin(
                x=[gene] * mask.sum(), y=expr_df[gene].values[mask],
                name=str(group), legendgroup=str(group), showlegend=(i == 0),
                line_color=colors[j % len(colors)], fillcolor=colors[j % len(colors)],
                opacity=0.6, meanline_visible=True,
            ))

    fig.update_layout(
        title="Violin Plot", violinmode="group",
        paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=font_color),
        xaxis=dict(title="Genes"), yaxis=dict(title="Expression"),
        margin=dict(l=60, r=20, t=40, b=60),
        legend=dict(title=groupby),
    )
    return fig


def render_heatmap(dm: DataManager, config: Dict, cell_indices: Optional[List], dark_mode: bool) -> go.Figure:
    """Render a heatmap."""
    var_names = config.get("var_names", [])
    groupby = config.get("groupby")
    colorscale = config.get("colorscale", "RdBu_r")
    use_raw = config.get("use_raw", False)
    gene_symbols_col = config.get("gene_symbols_col")

    if not var_names:
        fig = go.Figure()
        fig.add_annotation(text="Configure genes", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    expr_df = dm.get_genes_expression(var_names, cell_indices, use_raw=use_raw, gene_symbols_column=gene_symbols_col)

    if expr_df.empty:
        return go.Figure()

    bg_color = "#1a1b1e" if dark_mode else "white"
    font_color = "#c1c2c5" if dark_mode else "#212529"

    # Aggregate by group if specified
    if groupby:
        groups = dm.get_obs_column(groupby, cell_indices)
        if groups is not None:
            expr_df["_group"] = groups
            agg = expr_df.groupby("_group")[var_names].mean()
            z = agg.values
            y_labels = agg.index.tolist()
        else:
            z = expr_df[var_names].values[:100]  # Limit cells
            y_labels = [f"Cell {i}" for i in range(len(z))]
    else:
        z = expr_df[var_names].values[:100]
        y_labels = [f"Cell {i}" for i in range(len(z))]

    # Z-score normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
        z = np.nan_to_num(z)

    fig = go.Figure(go.Heatmap(
        z=z, x=var_names, y=y_labels, colorscale=colorscale,
        colorbar=dict(title="Z-score"),
    ))

    fig.update_layout(
        title="Heatmap", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=font_color),
        xaxis=dict(title="Genes", tickangle=45), yaxis=dict(title=groupby or "Cells", autorange="reversed"),
        margin=dict(l=100, r=60, t=40, b=80),
    )
    return fig


def render_crosstab(dm: DataManager, config: Dict, cell_indices: Optional[List], dark_mode: bool) -> go.Figure:
    """Render a crosstab heatmap with dendrogram."""
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    groupby = config.get("groupby")
    compare_col = config.get("compare_col")
    use_rep = config.get("use_rep")
    colorscale = config.get("colorscale", "YlOrRd")
    normalize = config.get("normalize", True)
    show_dendrogram = config.get("show_dendrogram", True)

    bg_color = "rgb(30,30,30)" if dark_mode else "white"
    font_color = "white" if dark_mode else "black"
    grid_color = "rgba(128,128,128,0.3)"

    # Empty figure if not configured
    if not groupby or not compare_col:
        fig = go.Figure()
        fig.add_annotation(
            text="Configure groupby and comparison columns",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=font_color),
        )
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color)
        return fig

    try:
        # Compute crosstab data
        crosstab_result = dm.compute_crosstab_data(
            groupby=groupby,
            compare_col=compare_col,
            cell_indices=cell_indices,
            use_rep=use_rep,
            normalize=normalize,
        )

        crosstab_data = crosstab_result["crosstab_normalized" if normalize else "crosstab"]
        row_order = crosstab_result["row_order"]
        col_order = crosstab_result["col_order"]
        dendro_info = crosstab_result["dendrogram"]

        z_values = crosstab_data.values
        y_labels = [str(r) for r in row_order]
        x_labels = [str(c) for c in col_order]

        # Create figure with or without dendrogram
        if show_dendrogram and len(dendro_info.get("linkage", [])) > 0:
            # Create subplot with heatmap on left, dendrogram on right
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.85, 0.15],
                horizontal_spacing=0.02,
                specs=[[{"type": "heatmap"}, {"type": "xy"}]],
            )

            # Compute dendrogram coordinates for plotting
            linkage_matrix = dendro_info["linkage"]
            dendro_data = scipy_dendrogram(
                linkage_matrix,
                orientation="right",
                no_plot=True,
                color_threshold=0,
            )

            # Add heatmap first (col=1)
            fig.add_trace(
                go.Heatmap(
                    z=z_values,
                    x=x_labels,
                    y=y_labels,
                    colorscale=colorscale,
                    colorbar=dict(
                        title="Proportion" if normalize else "Count",
                        title_side="right",
                        len=0.8,
                        x=1.15,
                    ),
                    hovertemplate=f"{groupby}: %{{y}}<br>{compare_col}: %{{x}}<br>Value: %{{z:.3f}}<extra></extra>",
                ),
                row=1, col=1,
            )

            # Add dendrogram traces (col=2)
            for i, (xs, ys) in enumerate(zip(dendro_data["dcoord"], dendro_data["icoord"])):
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys,
                        mode="lines",
                        line=dict(color="gray" if dark_mode else "black", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=2,
                )

            # Style heatmap subplot - show ALL labels
            fig.update_xaxes(
                title=compare_col, tickangle=45, row=1, col=1,
                tickfont=dict(size=10),
                tickmode="array",
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels,
            )
            fig.update_yaxes(
                title=groupby, row=1, col=1,
                tickfont=dict(size=10),
                autorange="reversed",
                tickmode="array",
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
            )

            # Style dendrogram subplot
            fig.update_xaxes(
                showticklabels=False, showgrid=False, zeroline=False,
                showline=False, row=1, col=2,
            )
            fig.update_yaxes(
                showticklabels=False, showgrid=False, zeroline=False,
                showline=False, row=1, col=2,
                autorange="reversed",
            )

        else:
            # Simple heatmap without dendrogram
            fig = go.Figure(
                go.Heatmap(
                    z=z_values,
                    x=x_labels,
                    y=y_labels,
                    colorscale=colorscale,
                    colorbar=dict(
                        title="Proportion" if normalize else "Count",
                        title_side="right",
                    ),
                    hovertemplate=f"{groupby}: %{{y}}<br>{compare_col}: %{{x}}<br>Value: %{{z:.3f}}<extra></extra>",
                )
            )

            # Show ALL labels
            fig.update_xaxes(
                title=compare_col, tickangle=45, tickfont=dict(size=10),
                tickmode="array",
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels,
            )
            fig.update_yaxes(
                title=groupby, tickfont=dict(size=10), autorange="reversed",
                tickmode="array",
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
            )

        # Common layout
        fig.update_layout(
            title=f"Crosstab: {groupby} vs {compare_col}",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=font_color),
            margin=dict(l=80, r=60, t=50, b=100),
        )

        return fig

    except Exception as e:
        # Error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)[:100]}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color="red"),
        )
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color)
        return fig
