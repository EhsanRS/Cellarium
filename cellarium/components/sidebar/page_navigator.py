"""
Page navigator component for sidebar.

Displays list of pages and allows creating/switching between them.
"""

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def create_page_navigator(
    pages: Optional[List[Dict]] = None,
    active_page_id: Optional[str] = None,
) -> dmc.Stack:
    """
    Create the page navigator for the sidebar.

    Args:
        pages: List of page dicts with 'id', 'name', 'n_cells', 'parent_id'
        active_page_id: Currently active page ID

    Returns:
        Stack component with page list
    """
    if pages is None:
        pages = []

    return dmc.Stack([
        # Header with add button
        dmc.Group([
            dmc.Text("Pages", fw=600, size="sm"),
            dmc.ActionIcon(
                DashIconify(icon="tabler:plus", width=16),
                id="add-page-btn",
                variant="light",
                size="sm",
                disabled=True,  # Enabled when selection exists
            ),
        ], justify="space-between"),

        # Page list
        dmc.Stack(
            [
                _create_page_item(page, active_page_id == page["id"])
                for page in pages
            ] if pages else [
                dmc.Text(
                    "No pages yet. Load data to get started.",
                    size="xs",
                    c="dimmed",
                    ta="center",
                    py="md",
                )
            ],
            gap="xs",
            id="page-list",
        ),
    ], gap="xs")


def _create_page_item(page: Dict, is_active: bool) -> dmc.Paper:
    """Create a single page item in the navigator."""
    page_id = page["id"]
    is_root = page.get("parent_id") is None

    return dmc.Paper(
        dmc.Group([
            # Page info
            dmc.Stack([
                dmc.Group([
                    DashIconify(
                        icon="tabler:file" if is_root else "tabler:file-arrow-right",
                        width=14,
                        color="var(--mantine-color-blue-6)" if is_active else "var(--mantine-color-dimmed)",
                    ),
                    dmc.Text(
                        page["name"],
                        size="sm",
                        fw=500 if is_active else 400,
                        truncate=True,
                    ),
                ], gap="xs"),
                dmc.Text(
                    f"{page['n_cells']:,} cells",
                    size="xs",
                    c="dimmed",
                    pl="xl",
                ),
            ], gap=2),

            # Actions menu
            dmc.Menu([
                dmc.MenuTarget(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:dots", width=14),
                        variant="subtle",
                        size="sm",
                        color="gray",
                    )
                ),
                dmc.MenuDropdown([
                    dmc.MenuItem(
                        "Rename",
                        id={"type": "rename-page-btn", "index": page_id},
                        leftSection=DashIconify(icon="tabler:edit", width=14),
                    ),
                    dmc.MenuItem(
                        "Duplicate",
                        id={"type": "duplicate-page-btn", "index": page_id},
                        leftSection=DashIconify(icon="tabler:copy", width=14),
                    ),
                    dmc.MenuItem(
                        "Export Cells",
                        id={"type": "export-page-btn", "index": page_id},
                        leftSection=DashIconify(icon="tabler:download", width=14),
                    ),
                    dmc.MenuDivider(),
                    dmc.MenuItem(
                        "Delete",
                        id={"type": "delete-page-btn", "index": page_id},
                        leftSection=DashIconify(icon="tabler:trash", width=14),
                        color="red",
                        disabled=is_root,  # Can't delete root page
                    ),
                ]),
            ], position="bottom-end"),
        ], justify="space-between", wrap="nowrap"),
        id={"type": "page-item", "index": page_id},
        p="xs",
        withBorder=True,
        radius="sm",
        style={
            "cursor": "pointer",
            "backgroundColor": "var(--mantine-color-blue-light)" if is_active else None,
        },
    )


def create_new_page_modal() -> dmc.Modal:
    """Create modal for creating a new page from selection."""
    return dmc.Modal(
        id="new-page-modal",
        title=dmc.Group([
            DashIconify(icon="tabler:file-plus", width=20),
            dmc.Text("Create New Page", fw=500),
        ]),
        children=[
            dmc.Stack([
                dmc.TextInput(
                    id="new-page-name",
                    label="Page Name",
                    placeholder="e.g., T Cells Subset",
                    leftSection=DashIconify(icon="tabler:file", width=16),
                ),
                dmc.Text(
                    id="new-page-cell-count",
                    size="sm",
                    c="dimmed",
                ),
                dmc.Group([
                    dmc.Button(
                        "Cancel",
                        id="new-page-cancel-btn",
                        variant="light",
                        color="gray",
                    ),
                    dmc.Button(
                        "Create Page",
                        id="new-page-create-btn",
                        leftSection=DashIconify(icon="tabler:check", width=16),
                    ),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )
