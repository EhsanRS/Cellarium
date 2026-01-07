"""
Data management module for Cellarium.

Provides AnnData loading, caching, and subset management.
"""

from cellarium.data.manager import DataManager
from cellarium.data.export import export_cell_subset, get_export_size_estimate, format_size

__all__ = [
    "DataManager",
    "export_cell_subset",
    "get_export_size_estimate",
    "format_size",
]
