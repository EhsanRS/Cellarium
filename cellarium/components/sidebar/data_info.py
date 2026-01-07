"""
Data info component for sidebar.

Displays dataset metadata and statistics.
"""

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def create_data_info(
    filename: str = "",
    n_cells: int = 0,
    n_genes: int = 0,
    obs_columns: Optional[List[str]] = None,
    embeddings: Optional[List[str]] = None,
    layers: Optional[List[str]] = None,
) -> dmc.Stack:
    """
    Create the data info display for the sidebar.

    Args:
        filename: Name of the loaded file
        n_cells: Number of cells in the dataset
        n_genes: Number of genes
        obs_columns: Available observation columns
        embeddings: Available embeddings (X_umap, X_pca, etc.)
        layers: Available expression layers

    Returns:
        Stack component with data info
    """
    if obs_columns is None:
        obs_columns = []
    if embeddings is None:
        embeddings = []
    if layers is None:
        layers = []

    return dmc.Stack([
        dmc.Text("Dataset Info", fw=600, size="sm"),

        # File info
        dmc.Paper([
            dmc.Stack([
                _info_row("tabler:file", "File", filename or "No data loaded"),
                _info_row("tabler:cells", "Cells", f"{n_cells:,}"),
                _info_row("tabler:dna-2", "Genes", f"{n_genes:,}"),
            ], gap="xs"),
        ], p="xs", withBorder=True, radius="sm"),

        # Embeddings
        dmc.Accordion(
            children=[
                dmc.AccordionItem(
                    value="embeddings",
                    children=[
                        dmc.AccordionControl(
                            dmc.Group([
                                DashIconify(icon="tabler:chart-dots-3", width=16),
                                dmc.Text("Embeddings", size="sm"),
                                dmc.Badge(str(len(embeddings)), size="sm", variant="light"),
                            ], gap="xs"),
                        ),
                        dmc.AccordionPanel(
                            dmc.Stack([
                                dmc.Text(
                                    emb.replace("X_", "").upper(),
                                    size="xs",
                                    c="dimmed",
                                )
                                for emb in embeddings
                            ] if embeddings else [
                                dmc.Text("No embeddings found", size="xs", c="dimmed")
                            ], gap=2),
                        ),
                    ],
                ),
                dmc.AccordionItem(
                    value="metadata",
                    children=[
                        dmc.AccordionControl(
                            dmc.Group([
                                DashIconify(icon="tabler:table", width=16),
                                dmc.Text("Metadata", size="sm"),
                                dmc.Badge(str(len(obs_columns)), size="sm", variant="light"),
                            ], gap="xs"),
                        ),
                        dmc.AccordionPanel(
                            dmc.Stack([
                                dmc.Text(col, size="xs", c="dimmed")
                                for col in obs_columns[:20]  # Limit display
                            ] + ([
                                dmc.Text(f"... and {len(obs_columns) - 20} more", size="xs", c="dimmed", fs="italic")
                            ] if len(obs_columns) > 20 else []), gap=2),
                        ),
                    ],
                ),
                dmc.AccordionItem(
                    value="layers",
                    children=[
                        dmc.AccordionControl(
                            dmc.Group([
                                DashIconify(icon="tabler:layers-subtract", width=16),
                                dmc.Text("Layers", size="sm"),
                                dmc.Badge(str(len(layers)), size="sm", variant="light"),
                            ], gap="xs"),
                        ),
                        dmc.AccordionPanel(
                            dmc.Stack([
                                dmc.Text(layer, size="xs", c="dimmed")
                                for layer in layers
                            ] if layers else [
                                dmc.Text("No additional layers", size="xs", c="dimmed")
                            ], gap=2),
                        ),
                    ],
                ),
            ],
            variant="separated",
            chevronPosition="right",
        ),
    ], gap="xs")


def _info_row(icon: str, label: str, value: str) -> dmc.Group:
    """Create an info row with icon, label, and value."""
    return dmc.Group([
        DashIconify(icon=icon, width=14, color="var(--mantine-color-dimmed)"),
        dmc.Text(label, size="xs", c="dimmed", style={"minWidth": "50px"}),
        dmc.Text(value, size="xs", fw=500, truncate=True),
    ], gap="xs", wrap="nowrap")
