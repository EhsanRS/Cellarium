"""
Database module for Cellarium.

Provides SQLite-based persistence for pages, selections, layouts, and app state.
"""

from cellarium.db.schema import init_database, get_connection
from cellarium.db.repository import (
    Page,
    PageRepository,
    Selection,
    SelectionRepository,
    Layout,
    LayoutRepository,
    Visualization,
    VisualizationRepository,
    GeneListRepository,
)

__all__ = [
    "init_database",
    "get_connection",
    "Page",
    "PageRepository",
    "Selection",
    "SelectionRepository",
    "Layout",
    "LayoutRepository",
    "Visualization",
    "VisualizationRepository",
    "GeneListRepository",
]
