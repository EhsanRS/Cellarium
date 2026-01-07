"""
SQLite schema definition and initialization for Cellarium.

This module defines the database schema for persisting:
- Pages (data subsets with cell indices)
- Layouts (panel arrangements for each page)
- Selections (saved cell selections)
- Visualizations (plot configurations)
- App state (user preferences)
"""

import sqlite3
from pathlib import Path
from typing import Optional

# SQL schema definition
SCHEMA_SQL = """
-- Pages represent different views/subsets of the data
-- Each page can have its own subset of cells and dashboard layout
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,                    -- UUID or slug
    name TEXT NOT NULL,                     -- Human-readable name
    parent_page_id TEXT,                    -- NULL for root page
    cell_indices BLOB,                      -- Numpy array serialized (pickle)
    n_cells INTEGER NOT NULL,               -- Number of cells in this subset
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (parent_page_id) REFERENCES pages(id) ON DELETE CASCADE
);

-- Create root page if not exists
INSERT OR IGNORE INTO pages (id, name, parent_page_id, cell_indices, n_cells)
VALUES ('root', 'All Cells', NULL, NULL, 0);

-- Layouts store the arrangement of panels on a page
-- Each page can have multiple saved layouts
CREATE TABLE IF NOT EXISTS layouts (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'Default',
    layout_json TEXT NOT NULL,              -- JSON: panel positions, sizes for each breakpoint
    is_active BOOLEAN DEFAULT FALSE,        -- Currently active layout for this page
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);

-- Cell selections (temporary or saved)
-- Used for creating new pages from selections
CREATE TABLE IF NOT EXISTS selections (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    name TEXT,                              -- NULL for temporary selections
    selection_type TEXT NOT NULL,           -- 'lasso', 'box', 'manual'
    cell_indices BLOB NOT NULL,             -- Selected cell indices (numpy array)
    n_cells INTEGER NOT NULL,
    source_embedding TEXT,                  -- e.g., 'X_umap'
    source_dims TEXT,                       -- e.g., '0,1' for UMAP1 vs UMAP2
    selection_coords TEXT,                  -- JSON: lasso/box coordinates for recreation
    is_saved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);

-- Visualization/panel configurations
-- Stores the configuration for each plot panel
CREATE TABLE IF NOT EXISTS visualizations (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    layout_id TEXT,                         -- Associated layout
    viz_type TEXT NOT NULL,                 -- 'scatter', 'dotplot', 'violin', 'heatmap'
    config_json TEXT NOT NULL,              -- Full configuration as JSON
    position_json TEXT,                     -- Grid position if using layout
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE,
    FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE SET NULL
);

-- User preferences and app state
-- Key-value store for application settings
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,                    -- JSON encoded
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gene lists (saved for quick access)
-- Users can save commonly used gene sets
CREATE TABLE IF NOT EXISTS gene_lists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    genes TEXT NOT NULL,                    -- JSON array of gene names
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_pages_parent ON pages(parent_page_id);
CREATE INDEX IF NOT EXISTS idx_selections_page ON selections(page_id);
CREATE INDEX IF NOT EXISTS idx_visualizations_page ON visualizations(page_id);
CREATE INDEX IF NOT EXISTS idx_layouts_page ON layouts(page_id);
CREATE INDEX IF NOT EXISTS idx_layouts_active ON layouts(page_id, is_active);
"""


def init_database(db_path: Path) -> sqlite3.Connection:
    """
    Initialize SQLite database with schema.

    Creates the database file if it doesn't exist and applies the schema.
    Uses WAL mode for better concurrent read performance.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        sqlite3.Connection: Database connection
    """
    db_path = Path(db_path)

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect and initialize
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better performance
    conn.execute("PRAGMA journal_mode=WAL")

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys=ON")

    # Apply schema
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    return conn


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Get a database connection.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        sqlite3.Connection: Database connection with row factory set
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def update_root_page_count(db_path: Path, n_cells: int) -> None:
    """
    Update the cell count for the root page.

    Called after loading data to set the total cell count.

    Args:
        db_path: Path to the SQLite database file
        n_cells: Total number of cells in the dataset
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE pages SET n_cells = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 'root'",
            (n_cells,)
        )
        conn.commit()


def get_app_state(db_path: Path, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an app state value.

    Args:
        db_path: Path to the SQLite database file
        key: State key
        default: Default value if key not found

    Returns:
        The value as a string, or default if not found
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (key,)
        ).fetchone()

    return row["value"] if row else default


def set_app_state(db_path: Path, key: str, value: str) -> None:
    """
    Set an app state value.

    Args:
        db_path: Path to the SQLite database file
        key: State key
        value: Value to store (should be JSON-serializable string)
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value)
        )
        conn.commit()
