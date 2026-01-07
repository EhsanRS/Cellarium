"""
Repository classes for database CRUD operations.

This module provides repository pattern implementations for:
- Pages (data subsets)
- Selections (cell selections)
- Layouts (dashboard configurations)
- Visualizations (plot configurations)
"""

import json
import pickle
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cellarium.db.schema import get_connection


@dataclass
class Page:
    """Page entity representing a data subset."""

    id: str
    name: str
    parent_page_id: Optional[str]
    cell_indices: Optional[np.ndarray]
    n_cells: int
    created_at: datetime
    updated_at: datetime


@dataclass
class Selection:
    """Selection entity representing a cell selection."""

    id: str
    page_id: str
    name: Optional[str]
    selection_type: str
    cell_indices: np.ndarray
    n_cells: int
    source_embedding: Optional[str]
    source_dims: Optional[str]
    selection_coords: Optional[Dict]
    is_saved: bool
    created_at: datetime


@dataclass
class Layout:
    """Layout entity representing a dashboard arrangement."""

    id: str
    page_id: str
    name: str
    layout_json: Dict
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class Visualization:
    """Visualization entity representing a plot configuration."""

    id: str
    page_id: str
    layout_id: Optional[str]
    viz_type: str
    config_json: Dict
    position_json: Optional[Dict]
    created_at: datetime
    updated_at: datetime


class PageRepository:
    """CRUD operations for pages."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def create(
        self,
        name: str,
        cell_indices: np.ndarray,
        parent_page_id: str = "root",
    ) -> Page:
        """Create a new page from cell selection."""
        page_id = f"page_{uuid.uuid4().hex[:8]}"
        indices_blob = pickle.dumps(cell_indices)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO pages (id, name, parent_page_id, cell_indices, n_cells)
                VALUES (?, ?, ?, ?, ?)
                """,
                (page_id, name, parent_page_id, indices_blob, len(cell_indices)),
            )
            conn.commit()

        return self.get_by_id(page_id)

    def get_by_id(self, page_id: str) -> Optional[Page]:
        """Get a page by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE id = ?", (page_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_page(row)

    def get_all(self) -> List[Page]:
        """Get all pages ordered by creation time."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pages ORDER BY created_at"
            ).fetchall()

        return [self._row_to_page(row) for row in rows]

    def get_children(self, parent_id: str) -> List[Page]:
        """Get child pages of a parent."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pages WHERE parent_page_id = ? ORDER BY created_at",
                (parent_id,),
            ).fetchall()

        return [self._row_to_page(row) for row in rows]

    def update_name(self, page_id: str, name: str) -> bool:
        """Update page name."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE pages SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (name, page_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, page_id: str) -> bool:
        """Delete a page (cascades to child pages)."""
        if page_id == "root":
            return False

        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM pages WHERE id = ?", (page_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_page(self, row: sqlite3.Row) -> Page:
        """Convert database row to Page object."""
        return Page(
            id=row["id"],
            name=row["name"],
            parent_page_id=row["parent_page_id"],
            cell_indices=pickle.loads(row["cell_indices"]) if row["cell_indices"] else None,
            n_cells=row["n_cells"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class SelectionRepository:
    """CRUD operations for selections."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def create(
        self,
        page_id: str,
        cell_indices: np.ndarray,
        selection_type: str,
        source_embedding: Optional[str] = None,
        source_dims: Optional[str] = None,
        selection_coords: Optional[Dict] = None,
        name: Optional[str] = None,
        is_saved: bool = False,
    ) -> Selection:
        """Create a new selection."""
        selection_id = f"sel_{uuid.uuid4().hex[:8]}"
        indices_blob = pickle.dumps(cell_indices)
        coords_json = json.dumps(selection_coords) if selection_coords else None

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO selections
                (id, page_id, name, selection_type, cell_indices, n_cells,
                 source_embedding, source_dims, selection_coords, is_saved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selection_id, page_id, name, selection_type, indices_blob,
                    len(cell_indices), source_embedding, source_dims, coords_json, is_saved
                ),
            )
            conn.commit()

        return self.get_by_id(selection_id)

    def get_by_id(self, selection_id: str) -> Optional[Selection]:
        """Get selection by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM selections WHERE id = ?", (selection_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_selection(row)

    def get_by_page(self, page_id: str, saved_only: bool = False) -> List[Selection]:
        """Get all selections for a page."""
        query = "SELECT * FROM selections WHERE page_id = ?"
        params = [page_id]

        if saved_only:
            query += " AND is_saved = 1"

        query += " ORDER BY created_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_selection(row) for row in rows]

    def save_selection(self, selection_id: str, name: str) -> bool:
        """Mark a selection as saved with a name."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE selections SET is_saved = 1, name = ? WHERE id = ?",
                (name, selection_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, selection_id: str) -> bool:
        """Delete a selection."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM selections WHERE id = ?", (selection_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_unsaved(self, page_id: str) -> int:
        """Delete all unsaved selections for a page."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM selections WHERE page_id = ? AND is_saved = 0",
                (page_id,),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_selection(self, row: sqlite3.Row) -> Selection:
        """Convert database row to Selection object."""
        return Selection(
            id=row["id"],
            page_id=row["page_id"],
            name=row["name"],
            selection_type=row["selection_type"],
            cell_indices=pickle.loads(row["cell_indices"]),
            n_cells=row["n_cells"],
            source_embedding=row["source_embedding"],
            source_dims=row["source_dims"],
            selection_coords=json.loads(row["selection_coords"]) if row["selection_coords"] else None,
            is_saved=bool(row["is_saved"]),
            created_at=row["created_at"],
        )


class LayoutRepository:
    """CRUD operations for layouts."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def create(
        self,
        page_id: str,
        layout_json: Dict,
        name: str = "Default",
        is_active: bool = True,
    ) -> Layout:
        """Create a new layout."""
        layout_id = f"layout_{uuid.uuid4().hex[:8]}"

        # Deactivate other layouts if this one is active
        if is_active:
            self._deactivate_page_layouts(page_id)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO layouts (id, page_id, name, layout_json, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (layout_id, page_id, name, json.dumps(layout_json), is_active),
            )
            conn.commit()

        return self.get_by_id(layout_id)

    def get_by_id(self, layout_id: str) -> Optional[Layout]:
        """Get layout by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM layouts WHERE id = ?", (layout_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_layout(row)

    def get_active_for_page(self, page_id: str) -> Optional[Layout]:
        """Get the active layout for a page."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM layouts WHERE page_id = ? AND is_active = 1",
                (page_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_layout(row)

    def get_by_page(self, page_id: str) -> List[Layout]:
        """Get all layouts for a page."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM layouts WHERE page_id = ? ORDER BY created_at",
                (page_id,),
            ).fetchall()

        return [self._row_to_layout(row) for row in rows]

    def update(self, layout_id: str, layout_json: Dict) -> bool:
        """Update layout configuration."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE layouts SET layout_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(layout_json), layout_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_active(self, layout_id: str) -> bool:
        """Set a layout as active for its page."""
        layout = self.get_by_id(layout_id)
        if layout is None:
            return False

        self._deactivate_page_layouts(layout.page_id)

        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE layouts SET is_active = 1 WHERE id = ?",
                (layout_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, layout_id: str) -> bool:
        """Delete a layout."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM layouts WHERE id = ?", (layout_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _deactivate_page_layouts(self, page_id: str) -> None:
        """Deactivate all layouts for a page."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE layouts SET is_active = 0 WHERE page_id = ?",
                (page_id,),
            )
            conn.commit()

    def _row_to_layout(self, row: sqlite3.Row) -> Layout:
        """Convert database row to Layout object."""
        return Layout(
            id=row["id"],
            page_id=row["page_id"],
            name=row["name"],
            layout_json=json.loads(row["layout_json"]),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class VisualizationRepository:
    """CRUD operations for visualizations (plot configurations)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def create(
        self,
        page_id: str,
        viz_type: str,
        config_json: Dict,
        layout_id: Optional[str] = None,
        position_json: Optional[Dict] = None,
    ) -> Visualization:
        """Create a new visualization configuration."""
        viz_id = f"viz_{uuid.uuid4().hex[:8]}"

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO visualizations (id, page_id, layout_id, viz_type, config_json, position_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    viz_id, page_id, layout_id, viz_type,
                    json.dumps(config_json),
                    json.dumps(position_json) if position_json else None
                ),
            )
            conn.commit()

        return self.get_by_id(viz_id)

    def get_by_id(self, viz_id: str) -> Optional[Visualization]:
        """Get visualization by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM visualizations WHERE id = ?", (viz_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_visualization(row)

    def get_by_page(self, page_id: str) -> List[Visualization]:
        """Get all visualizations for a page."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM visualizations WHERE page_id = ? ORDER BY created_at",
                (page_id,),
            ).fetchall()

        return [self._row_to_visualization(row) for row in rows]

    def get_by_layout(self, layout_id: str) -> List[Visualization]:
        """Get all visualizations for a layout."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM visualizations WHERE layout_id = ? ORDER BY created_at",
                (layout_id,),
            ).fetchall()

        return [self._row_to_visualization(row) for row in rows]

    def update(
        self,
        viz_id: str,
        config_json: Optional[Dict] = None,
        position_json: Optional[Dict] = None,
    ) -> bool:
        """Update visualization configuration."""
        updates = []
        params = []

        if config_json is not None:
            updates.append("config_json = ?")
            params.append(json.dumps(config_json))

        if position_json is not None:
            updates.append("position_json = ?")
            params.append(json.dumps(position_json))

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(viz_id)

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"UPDATE visualizations SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, viz_id: str) -> bool:
        """Delete a visualization."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM visualizations WHERE id = ?", (viz_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_visualization(self, row: sqlite3.Row) -> Visualization:
        """Convert database row to Visualization object."""
        return Visualization(
            id=row["id"],
            page_id=row["page_id"],
            layout_id=row["layout_id"],
            viz_type=row["viz_type"],
            config_json=json.loads(row["config_json"]),
            position_json=json.loads(row["position_json"]) if row["position_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class GeneListRepository:
    """CRUD operations for saved gene lists."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def create(
        self,
        name: str,
        genes: List[str],
        description: Optional[str] = None,
    ) -> str:
        """Create a new gene list."""
        list_id = f"genelist_{uuid.uuid4().hex[:8]}"

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO gene_lists (id, name, genes, description)
                VALUES (?, ?, ?, ?)
                """,
                (list_id, name, json.dumps(genes), description),
            )
            conn.commit()

        return list_id

    def get_by_name(self, name: str) -> Optional[List[str]]:
        """Get genes in a list by name."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT genes FROM gene_lists WHERE name = ?", (name,)
            ).fetchone()

        if row is None:
            return None

        return json.loads(row["genes"])

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all gene lists."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, name, genes, description, created_at FROM gene_lists ORDER BY name"
            ).fetchall()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "genes": json.loads(row["genes"]),
                "description": row["description"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, name: str) -> bool:
        """Delete a gene list by name."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM gene_lists WHERE name = ?", (name,)
            )
            conn.commit()
            return cursor.rowcount > 0
