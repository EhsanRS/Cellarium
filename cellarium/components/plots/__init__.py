"""
Plot components for Cellarium.

Provides visualization components using Plotly:
- Scatter plots (UMAP/PCA) using Scattergl for WebGL performance
- Dot plots for gene expression across groups
- Violin plots for expression distributions
- Heatmaps for expression matrices
"""

from cellarium.components.plots.scatter import create_scatter_plot
from cellarium.components.plots.dotplot import create_dotplot
from cellarium.components.plots.violin import create_violin_plot
from cellarium.components.plots.heatmap import create_heatmap

__all__ = [
    "create_scatter_plot",
    "create_dotplot",
    "create_violin_plot",
    "create_heatmap",
]
