"""
Dashboard toolbar component.

Provides controls for adding panels, saving/loading layouts,
and other dashboard-level operations.
"""

from typing import List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def create_toolbar(
    available_embeddings: Optional[List[str]] = None,
) -> dmc.Group:
    """
    Create the dashboard toolbar with panel controls.

    Args:
        available_embeddings: List of available embedding keys (X_umap, X_pca, etc.)

    Returns:
        Group component with toolbar controls
    """
    # Build embedding menu items
    embedding_items = []
    if available_embeddings:
        for emb in available_embeddings:
            name = emb.replace("X_", "").upper()
            embedding_items.append(
                dmc.MenuItem(
                    f"{name} Plot",
                    id={"type": "add-panel-btn", "plot_type": "scatter", "embedding": emb},
                    leftSection=DashIconify(icon="tabler:chart-dots-3", width=16),
                )
            )

    return dmc.Group(
        [
            # Left side: Add panel menu
            dmc.Menu(
                [
                    dmc.MenuTarget(
                        dmc.Button(
                            "Add Panel",
                            leftSection=DashIconify(icon="tabler:plus", width=16),
                            variant="light",
                        )
                    ),
                    dmc.MenuDropdown(
                        [
                            dmc.MenuLabel("Embeddings"),
                            *embedding_items,
                            dmc.MenuDivider(),
                            dmc.MenuLabel("Expression Plots"),
                            dmc.MenuItem(
                                "Dot Plot",
                                id={"type": "add-panel-btn", "plot_type": "dotplot", "embedding": None},
                                leftSection=DashIconify(icon="tabler:chart-dots", width=16),
                            ),
                            dmc.MenuItem(
                                "Violin Plot",
                                id={"type": "add-panel-btn", "plot_type": "violin", "embedding": None},
                                leftSection=DashIconify(icon="tabler:chart-area", width=16),
                            ),
                            dmc.MenuItem(
                                "Heatmap",
                                id={"type": "add-panel-btn", "plot_type": "heatmap", "embedding": None},
                                leftSection=DashIconify(icon="tabler:chart-histogram", width=16),
                            ),
                        ]
                    ),
                ],
                position="bottom-start",
            ),

            # Right side: Layout controls
            dmc.Group(
                [
                    dmc.Button(
                        "Save Layout",
                        id="save-layout-btn",
                        leftSection=DashIconify(icon="tabler:device-floppy", width=16),
                        variant="outline",
                        size="sm",
                    ),
                    dmc.Button(
                        "Load Layout",
                        id="load-layout-btn",
                        leftSection=DashIconify(icon="tabler:folder-open", width=16),
                        variant="outline",
                        size="sm",
                    ),
                    dmc.Menu(
                        [
                            dmc.MenuTarget(
                                dmc.ActionIcon(
                                    DashIconify(icon="tabler:dots-vertical", width=16),
                                    variant="subtle",
                                    size="lg",
                                )
                            ),
                            dmc.MenuDropdown(
                                [
                                    dmc.MenuItem(
                                        "Reset Layout",
                                        id="reset-layout-btn",
                                        leftSection=DashIconify(icon="tabler:refresh", width=16),
                                    ),
                                    dmc.MenuItem(
                                        "Clear All Panels",
                                        id="clear-panels-btn",
                                        leftSection=DashIconify(icon="tabler:trash", width=16),
                                        color="red",
                                    ),
                                ]
                            ),
                        ],
                        position="bottom-end",
                    ),
                ],
                gap="xs",
            ),
        ],
        justify="space-between",
        mb="md",
    )


def create_selection_toolbar() -> dmc.Paper:
    """
    Create the selection info and action toolbar.

    Displays information about current cell selection and provides
    actions like creating a new page from selection.

    Returns:
        Paper component with selection toolbar
    """
    return dmc.Paper(
        dmc.Group(
            [
                # Selection info (updated by callback)
                html.Div(
                    id="selection-info",
                    children=[
                        dmc.Text("No cells selected", size="sm", c="dimmed"),
                    ],
                ),

                # Selection actions
                dmc.Group(
                    [
                        dmc.Button(
                            "Create Page from Selection",
                            id="create-page-btn",
                            leftSection=DashIconify(icon="tabler:file-plus", width=16),
                            color="blue",
                            size="sm",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Export Selection",
                            id="export-selection-btn",
                            leftSection=DashIconify(icon="tabler:download", width=16),
                            variant="outline",
                            size="sm",
                            disabled=True,
                        ),
                        dmc.Button(
                            "Clear",
                            id="clear-selection-btn",
                            leftSection=DashIconify(icon="tabler:x", width=16),
                            variant="subtle",
                            size="sm",
                            color="gray",
                        ),
                    ],
                    gap="xs",
                ),
            ],
            justify="space-between",
        ),
        withBorder=True,
        p="sm",
        mt="md",
    )
