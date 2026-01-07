"""
Export subset modal component.
"""

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify


def create_export_modal() -> dmc.Modal:
    """
    Create modal for exporting cell subset as .h5ad file.

    Returns:
        Modal component
    """
    return dmc.Modal(
        id="export-modal",
        title=dmc.Group([
            DashIconify(icon="tabler:download", width=20),
            dmc.Text("Export Cell Subset", fw=500),
        ]),
        size="lg",
        children=[
            dmc.Stack([
                # Export info
                dmc.Paper([
                    dmc.Group([
                        DashIconify(icon="tabler:cells", width=24, color="var(--mantine-color-blue-6)"),
                        dmc.Stack([
                            dmc.Text(id="export-cell-count", fw=500),
                            dmc.Text("cells will be exported", size="sm", c="dimmed"),
                        ], gap=0),
                    ], gap="md"),
                ], p="md", withBorder=True, radius="md"),

                dmc.Divider(label="Export Options", labelPosition="center"),

                # File name
                dmc.TextInput(
                    id="export-filename",
                    label="File Name",
                    placeholder="subset",
                    description="File will be saved as {name}.h5ad",
                    leftSection=DashIconify(icon="tabler:file", width=16),
                    rightSection=dmc.Text(".h5ad", size="sm", c="dimmed"),
                ),

                # Data options
                dmc.Stack([
                    dmc.Checkbox(
                        id="export-include-raw",
                        label="Include raw counts (adata.raw)",
                        checked=True,
                    ),
                    dmc.Checkbox(
                        id="export-include-layers",
                        label="Include all layers",
                        checked=True,
                    ),
                    dmc.Checkbox(
                        id="export-include-obsm",
                        label="Include embeddings (obsm)",
                        checked=True,
                    ),
                    dmc.Checkbox(
                        id="export-include-obsp",
                        label="Include cell graphs (obsp)",
                        checked=False,
                    ),
                ], gap="xs"),

                dmc.Alert(
                    children=[
                        dmc.Text("The exported file will contain:", size="sm"),
                        dmc.List([
                            dmc.ListItem("Expression matrix (X)", size="sm"),
                            dmc.ListItem("Cell metadata (obs)", size="sm"),
                            dmc.ListItem("Gene metadata (var)", size="sm"),
                            dmc.ListItem("Selected optional data", size="sm"),
                        ], size="sm"),
                    ],
                    icon=DashIconify(icon="tabler:info-circle", width=16),
                    color="blue",
                ),

                # Progress indicator (shown during export)
                html.Div(
                    id="export-progress",
                    children=[
                        dmc.Progress(
                            id="export-progress-bar",
                            value=0,
                            animated=True,
                            striped=True,
                        ),
                        dmc.Text(
                            id="export-progress-text",
                            size="sm",
                            c="dimmed",
                            ta="center",
                        ),
                    ],
                    style={"display": "none"},
                ),

                dmc.Group([
                    dmc.Button(
                        "Cancel",
                        id="export-cancel-btn",
                        variant="light",
                        color="gray",
                    ),
                    dmc.Button(
                        "Export",
                        id="export-confirm-btn",
                        leftSection=DashIconify(icon="tabler:download", width=16),
                    ),
                ], justify="flex-end"),

                # Download trigger
                dcc.Download(id="export-download"),
            ], gap="md"),
        ],
        opened=False,
    )
