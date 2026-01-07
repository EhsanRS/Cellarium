"""
Load layout modal component.
"""

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify


def create_load_layout_modal() -> dmc.Modal:
    """
    Create modal for loading layout from JSON file.

    Returns:
        Modal component
    """
    return dmc.Modal(
        id="load-layout-modal",
        title=dmc.Group([
            DashIconify(icon="tabler:folder-open", width=20),
            dmc.Text("Load Layout", fw=500),
        ]),
        children=[
            dmc.Stack([
                dmc.Text(
                    "Upload a previously saved layout file (.json)",
                    size="sm",
                    c="dimmed",
                ),

                # File upload
                dcc.Upload(
                    id="layout-upload",
                    children=dmc.Paper([
                        dmc.Stack([
                            DashIconify(
                                icon="tabler:upload",
                                width=40,
                                color="var(--mantine-color-blue-6)",
                            ),
                            dmc.Text("Drag and drop or click to upload", size="sm"),
                            dmc.Text(".json files only", size="xs", c="dimmed"),
                        ], align="center", gap="xs"),
                    ], p="xl", withBorder=True, radius="md",
                    style={"borderStyle": "dashed", "cursor": "pointer"}),
                    accept=".json",
                    multiple=False,
                ),

                # Preview area (shown after upload)
                html.Div(id="layout-preview", style={"display": "none"}),

                # Error message area
                html.Div(id="layout-load-error"),

                dmc.Group([
                    dmc.Button(
                        "Cancel",
                        id="load-layout-cancel-btn",
                        variant="light",
                        color="gray",
                    ),
                    dmc.Button(
                        "Apply Layout",
                        id="load-layout-apply-btn",
                        leftSection=DashIconify(icon="tabler:check", width=16),
                        disabled=True,
                    ),
                ], justify="flex-end"),
            ], gap="md"),
        ],
        opened=False,
    )
