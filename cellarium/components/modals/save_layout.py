"""
Save layout modal component.
"""

import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify


def create_save_layout_modal() -> dmc.Modal:
    """
    Create modal for saving layout to JSON file.

    Returns:
        Modal component
    """
    return dmc.Modal(
        id="save-layout-modal",
        title=dmc.Group([
            DashIconify(icon="tabler:device-floppy", width=20),
            dmc.Text("Save Layout", fw=500),
        ]),
        children=[
            dmc.Stack([
                dmc.TextInput(
                    id="save-layout-name",
                    label="Layout Name",
                    placeholder="My Custom Layout",
                    leftSection=DashIconify(icon="tabler:file", width=16),
                    value="",
                ),
                dmc.Textarea(
                    id="save-layout-description",
                    label="Description (optional)",
                    placeholder="Describe this layout...",
                    autosize=True,
                    minRows=2,
                ),
                dmc.Alert(
                    "Layout will be saved as a JSON file that you can load later.",
                    icon=DashIconify(icon="tabler:info-circle", width=16),
                    color="blue",
                ),
                dmc.Group([
                    dmc.Button(
                        "Cancel",
                        id="save-layout-cancel-btn",
                        variant="light",
                        color="gray",
                    ),
                    dmc.Button(
                        "Save Layout",
                        id="save-layout-confirm-btn",
                        leftSection=DashIconify(icon="tabler:download", width=16),
                    ),
                ], justify="flex-end"),

                # Hidden download component
                dcc.Download(id="layout-download"),
            ], gap="md"),
        ],
        opened=False,
    )
