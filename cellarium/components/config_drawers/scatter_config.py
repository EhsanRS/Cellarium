"""
Configuration drawer for scatter plot (UMAP/PCA) parameters.
"""

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def create_scatter_config_drawer(
    panel_id: str,
    obs_columns: List[str],
    gene_names: List[str],
    embeddings: List[str],
    current_config: Optional[Dict] = None,
) -> dmc.Drawer:
    """
    Create a configuration drawer for scatter plot settings.

    Args:
        panel_id: Panel identifier
        obs_columns: Available observation columns for coloring
        gene_names: Available gene names for expression coloring
        embeddings: Available embeddings (X_umap, X_pca, etc.)
        current_config: Current panel configuration

    Returns:
        Drawer component with scatter plot settings
    """
    config = current_config or {}

    return dmc.Drawer(
        id={"type": "scatter-config-drawer", "index": panel_id},
        title=dmc.Group([
            DashIconify(icon="tabler:chart-dots-3", width=20),
            dmc.Text("Scatter Plot Settings", fw=500),
        ]),
        position="right",
        size="md",
        children=[
            dmc.Stack([
                # Embedding selection
                dmc.Select(
                    id={"type": "scatter-embedding", "index": panel_id},
                    label="Embedding",
                    description="Choose the dimensionality reduction to display",
                    data=[{"value": e, "label": e.replace("X_", "").upper()} for e in embeddings],
                    value=config.get("embedding", embeddings[0] if embeddings else None),
                    leftSection=DashIconify(icon="tabler:dimensions", width=16),
                ),

                dmc.Divider(label="Color By", labelPosition="center"),

                # Color type selection
                dmc.SegmentedControl(
                    id={"type": "scatter-color-type", "index": panel_id},
                    data=[
                        {"value": "obs", "label": "Metadata"},
                        {"value": "gene", "label": "Gene Expression"},
                    ],
                    value=config.get("color_type", "obs"),
                    fullWidth=True,
                ),

                # Observation column selector
                dmc.Select(
                    id={"type": "scatter-color-obs", "index": panel_id},
                    label="Metadata Column",
                    description="Color by observation metadata",
                    data=[{"value": c, "label": c} for c in obs_columns],
                    value=config.get("color_obs"),
                    searchable=True,
                    clearable=True,
                    leftSection=DashIconify(icon="tabler:table", width=16),
                ),

                # Gene selector
                dmc.Select(
                    id={"type": "scatter-color-gene", "index": panel_id},
                    label="Gene",
                    description="Color by gene expression",
                    data=[{"value": g, "label": g} for g in gene_names[:1000]],  # Limit for performance
                    value=config.get("color_gene"),
                    searchable=True,
                    clearable=True,
                    leftSection=DashIconify(icon="tabler:dna-2", width=16),
                    nothingFoundMessage="No genes found",
                ),

                dmc.Divider(label="Appearance", labelPosition="center"),

                # Colorscale selector
                dmc.Select(
                    id={"type": "scatter-colorscale", "index": panel_id},
                    label="Color Scale",
                    data=[
                        {"value": "Viridis", "label": "Viridis"},
                        {"value": "Plasma", "label": "Plasma"},
                        {"value": "Inferno", "label": "Inferno"},
                        {"value": "Magma", "label": "Magma"},
                        {"value": "Cividis", "label": "Cividis"},
                        {"value": "Blues", "label": "Blues"},
                        {"value": "Reds", "label": "Reds"},
                        {"value": "RdBu_r", "label": "Red-Blue"},
                    ],
                    value=config.get("colorscale", "Viridis"),
                ),

                # Point size slider
                dmc.Stack([
                    dmc.Text("Point Size", size="sm", fw=500),
                    dmc.Slider(
                        id={"type": "scatter-point-size", "index": panel_id},
                        value=config.get("point_size", 3),
                        min=1,
                        max=10,
                        step=1,
                        marks=[
                            {"value": 1, "label": "1"},
                            {"value": 5, "label": "5"},
                            {"value": 10, "label": "10"},
                        ],
                    ),
                ], gap="xs"),

                # Opacity slider
                dmc.Stack([
                    dmc.Text("Opacity", size="sm", fw=500),
                    dmc.Slider(
                        id={"type": "scatter-opacity", "index": panel_id},
                        value=config.get("opacity", 0.7),
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        marks=[
                            {"value": 0.1, "label": "0.1"},
                            {"value": 0.5, "label": "0.5"},
                            {"value": 1.0, "label": "1.0"},
                        ],
                    ),
                ], gap="xs"),

                dmc.Divider(label="Interaction", labelPosition="center"),

                # Drag mode selection
                dmc.Select(
                    id={"type": "scatter-dragmode", "index": panel_id},
                    label="Selection Tool",
                    data=[
                        {"value": "lasso", "label": "Lasso Select"},
                        {"value": "select", "label": "Box Select"},
                        {"value": "pan", "label": "Pan"},
                        {"value": "zoom", "label": "Zoom"},
                    ],
                    value=config.get("dragmode", "lasso"),
                    leftSection=DashIconify(icon="tabler:lasso", width=16),
                ),

                # Apply button
                dmc.Button(
                    "Apply Changes",
                    id={"type": "scatter-apply-btn", "index": panel_id},
                    leftSection=DashIconify(icon="tabler:check", width=16),
                    fullWidth=True,
                    mt="md",
                ),
            ], gap="md", p="md"),
        ],
        opened=False,
    )


def get_scatter_config_from_inputs(
    embedding: str,
    color_type: str,
    color_obs: Optional[str],
    color_gene: Optional[str],
    colorscale: str,
    point_size: int,
    opacity: float,
    dragmode: str,
) -> Dict[str, Any]:
    """
    Collect scatter plot configuration from input values.

    Returns:
        Configuration dict
    """
    config = {
        "embedding": embedding,
        "color_type": color_type,
        "colorscale": colorscale,
        "point_size": point_size,
        "opacity": opacity,
        "dragmode": dragmode,
    }

    if color_type == "obs" and color_obs:
        config["color_obs"] = color_obs
        config["color"] = color_obs
    elif color_type == "gene" and color_gene:
        config["color_gene"] = color_gene
        config["color"] = color_gene

    return config
