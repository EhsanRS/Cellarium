"""
Data export utilities for Cellarium.

Handles exporting cell subsets as new .h5ad files.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


def export_cell_subset(
    adata,
    cell_indices: np.ndarray,
    output_path: Path,
    include_raw: bool = True,
    include_layers: bool = True,
    include_obsm: bool = True,
    include_obsp: bool = False,
) -> Path:
    """
    Export a subset of cells to a new .h5ad file.

    Args:
        adata: Original AnnData object
        cell_indices: Array of cell indices to export
        output_path: Path to save the new file
        include_raw: Whether to include adata.raw
        include_layers: Whether to include all layers
        include_obsm: Whether to include embeddings
        include_obsp: Whether to include cell-cell graphs

    Returns:
        Path to the exported file
    """
    # Create subset view
    subset = adata[cell_indices, :].copy()

    # Handle optional components
    if not include_raw:
        subset.raw = None

    if not include_layers:
        subset.layers = {}

    if not include_obsm:
        subset.obsm = {}

    if not include_obsp:
        subset.obsp = {}

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add suffix if not present
    if output_path.suffix != ".h5ad":
        output_path = output_path.with_suffix(".h5ad")

    # Write file
    subset.write_h5ad(output_path)

    return output_path


def get_export_size_estimate(
    adata,
    n_cells: int,
    include_raw: bool = True,
    include_layers: bool = True,
    include_obsm: bool = True,
) -> int:
    """
    Estimate the file size of an export in bytes.

    Args:
        adata: Original AnnData object
        n_cells: Number of cells to export
        include_raw: Whether raw is included
        include_layers: Whether layers are included
        include_obsm: Whether obsm is included

    Returns:
        Estimated size in bytes
    """
    n_genes = adata.n_vars

    # Base estimate: X matrix (assuming sparse, ~10% density typical)
    density = 0.1
    x_size = n_cells * n_genes * density * 8  # 8 bytes per float64

    # obs: estimate based on number of columns
    obs_size = n_cells * len(adata.obs.columns) * 50  # rough estimate per cell

    # var: relatively small
    var_size = n_genes * 100

    total = x_size + obs_size + var_size

    if include_raw and adata.raw is not None:
        total += x_size

    if include_layers:
        total += x_size * len(adata.layers)

    if include_obsm:
        for key, arr in adata.obsm.items():
            if hasattr(arr, "shape"):
                total += n_cells * (arr.shape[1] if len(arr.shape) > 1 else 1) * 8

    return int(total)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
