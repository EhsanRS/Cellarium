"""
Configuration drawer for expression plots (dotplot, violin, heatmap).
"""

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def create_expression_config_drawer(
    panel_id: str,
    panel_type: str,
    obs_columns: List[str],
    gene_names: List[str],
    current_config: Optional[Dict] = None,
) -> dmc.Drawer:
    """
    Create a configuration drawer for expression plot settings.

    Args:
        panel_id: Panel identifier
        panel_type: Type of plot ('dotplot', 'violin', 'heatmap')
        obs_columns: Available observation columns for grouping
        gene_names: Available gene names
        current_config: Current panel configuration

    Returns:
        Drawer component with expression plot settings
    """
    config = current_config or {}

    # Get plot-specific icon and title
    plot_info = {
        "dotplot": ("tabler:chart-dots", "Dot Plot Settings"),
        "violin": ("tabler:chart-area", "Violin Plot Settings"),
        "heatmap": ("tabler:chart-histogram", "Heatmap Settings"),
    }
    icon, title = plot_info.get(panel_type, ("tabler:chart-bar", "Plot Settings"))

    # Build plot-specific options
    specific_options = _get_plot_specific_options(panel_id, panel_type, config)

    return dmc.Drawer(
        id={"type": "expression-config-drawer", "index": panel_id},
        title=dmc.Group([
            DashIconify(icon=icon, width=20),
            dmc.Text(title, fw=500),
        ]),
        position="right",
        size="lg",
        children=[
            dmc.Stack([
                # Gene selection (multi-select with search)
                dmc.MultiSelect(
                    id={"type": "expression-genes", "index": panel_id},
                    label="Genes",
                    description="Select genes to display (searchable)",
                    data=[{"value": g, "label": g} for g in gene_names[:1000]],
                    value=config.get("var_names", []),
                    searchable=True,
                    clearable=True,
                    maxDropdownHeight=300,
                    leftSection=DashIconify(icon="tabler:dna-2", width=16),
                    nothingFoundMessage="No genes found",
                ),

                # Quick gene selection helpers
                dmc.Group([
                    dmc.Button(
                        "Top 10 Variable",
                        id={"type": "expression-top-variable", "index": panel_id},
                        variant="light",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:chart-line", width=14),
                    ),
                    dmc.Button(
                        "Clear All",
                        id={"type": "expression-clear-genes", "index": panel_id},
                        variant="light",
                        color="gray",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:x", width=14),
                    ),
                ], gap="xs"),

                dmc.Divider(label="Grouping", labelPosition="center"),

                # Group by selection
                dmc.Select(
                    id={"type": "expression-groupby", "index": panel_id},
                    label="Group By",
                    description="Group cells by this observation column",
                    data=[{"value": c, "label": c} for c in obs_columns],
                    value=config.get("groupby"),
                    searchable=True,
                    clearable=True,
                    leftSection=DashIconify(icon="tabler:category", width=16),
                ),

                dmc.Divider(label="Appearance", labelPosition="center"),

                # Colorscale selector
                dmc.Select(
                    id={"type": "expression-colorscale", "index": panel_id},
                    label="Color Scale",
                    data=[
                        {"value": "Reds", "label": "Reds"},
                        {"value": "Blues", "label": "Blues"},
                        {"value": "Viridis", "label": "Viridis"},
                        {"value": "RdBu_r", "label": "Red-Blue (diverging)"},
                        {"value": "RdYlBu_r", "label": "Red-Yellow-Blue"},
                    ],
                    value=config.get("colorscale", "Reds"),
                ),

                # Plot-specific options
                *specific_options,

                # Apply button
                dmc.Button(
                    "Apply Changes",
                    id={"type": "expression-apply-btn", "index": panel_id},
                    leftSection=DashIconify(icon="tabler:check", width=16),
                    fullWidth=True,
                    mt="md",
                ),
            ], gap="md", p="md"),
        ],
        opened=False,
    )


def _get_plot_specific_options(
    panel_id: str,
    panel_type: str,
    config: Dict,
) -> List[Any]:
    """Get plot-type specific configuration options."""
    options = []

    if panel_type == "dotplot":
        options.extend([
            dmc.Divider(label="Dot Plot Options", labelPosition="center"),

            # Expression cutoff
            dmc.NumberInput(
                id={"type": "dotplot-cutoff", "index": panel_id},
                label="Expression Cutoff",
                description="Minimum expression to count as 'expressing'",
                value=config.get("expression_cutoff", 0.0),
                min=0,
                step=0.1,
                precision=2,
            ),

            # Mean only expressed
            dmc.Switch(
                id={"type": "dotplot-mean-expressed", "index": panel_id},
                label="Mean only over expressing cells",
                checked=config.get("mean_only_expressed", False),
            ),

            # Size slider
            dmc.Stack([
                dmc.Text("Max Dot Size", size="sm", fw=500),
                dmc.Slider(
                    id={"type": "dotplot-size-max", "index": panel_id},
                    value=config.get("size_max", 15),
                    min=5,
                    max=25,
                    step=1,
                ),
            ], gap="xs"),
        ])

    elif panel_type == "violin":
        options.extend([
            dmc.Divider(label="Violin Plot Options", labelPosition="center"),

            # Strip plot toggle
            dmc.Switch(
                id={"type": "violin-stripplot", "index": panel_id},
                label="Show strip plot (individual points)",
                checked=config.get("stripplot", True),
            ),

            # Jitter slider
            dmc.Stack([
                dmc.Text("Point Jitter", size="sm", fw=500),
                dmc.Slider(
                    id={"type": "violin-jitter", "index": panel_id},
                    value=config.get("jitter", 0.4),
                    min=0,
                    max=1,
                    step=0.1,
                ),
            ], gap="xs"),

            # Scale mode
            dmc.Select(
                id={"type": "violin-scale", "index": panel_id},
                label="Violin Scale",
                data=[
                    {"value": "width", "label": "Width (all same width)"},
                    {"value": "count", "label": "Count (scale by n cells)"},
                    {"value": "area", "label": "Area (same area)"},
                ],
                value=config.get("scale", "width"),
            ),

            # Rotation
            dmc.NumberInput(
                id={"type": "violin-rotation", "index": panel_id},
                label="X-axis Label Rotation",
                value=config.get("rotation", 0),
                min=0,
                max=90,
                step=15,
            ),
        ])

    elif panel_type == "heatmap":
        options.extend([
            dmc.Divider(label="Heatmap Options", labelPosition="center"),

            # Aggregate toggle
            dmc.Switch(
                id={"type": "heatmap-aggregate", "index": panel_id},
                label="Aggregate by group (show mean)",
                checked=config.get("aggregate", True),
            ),

            # Center at zero
            dmc.Switch(
                id={"type": "heatmap-center-zero", "index": panel_id},
                label="Center color scale at zero",
                checked=config.get("center_zero", True),
            ),

            # Max cells (when not aggregating)
            dmc.NumberInput(
                id={"type": "heatmap-max-cells", "index": panel_id},
                label="Max Cells (if not aggregating)",
                description="Random sample if more cells",
                value=config.get("max_cells", 500),
                min=100,
                max=2000,
                step=100,
            ),
        ])

    return options


def get_expression_config_from_inputs(
    panel_type: str,
    genes: List[str],
    groupby: Optional[str],
    colorscale: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Collect expression plot configuration from input values.

    Returns:
        Configuration dict
    """
    config = {
        "var_names": genes,
        "groupby": groupby,
        "colorscale": colorscale,
    }

    # Add plot-specific options
    if panel_type == "dotplot":
        config["expression_cutoff"] = kwargs.get("expression_cutoff", 0.0)
        config["mean_only_expressed"] = kwargs.get("mean_only_expressed", False)
        config["size_max"] = kwargs.get("size_max", 15)

    elif panel_type == "violin":
        config["stripplot"] = kwargs.get("stripplot", True)
        config["jitter"] = kwargs.get("jitter", 0.4)
        config["scale"] = kwargs.get("scale", "width")
        config["rotation"] = kwargs.get("rotation", 0)

    elif panel_type == "heatmap":
        config["aggregate"] = kwargs.get("aggregate", True)
        config["center_zero"] = kwargs.get("center_zero", True)
        config["max_cells"] = kwargs.get("max_cells", 500)

    return config
