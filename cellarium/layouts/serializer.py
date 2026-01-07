"""
Layout serializer for JSON save/load.

Handles exporting and importing dashboard layouts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


LAYOUT_VERSION = "1.0"


def serialize_layout(
    pages: Dict[str, Dict],
    active_page_id: str,
    name: str = "Untitled Layout",
    description: str = "",
) -> Dict[str, Any]:
    """
    Serialize the current layout state to a JSON-compatible dict.

    Args:
        pages: Dict of page_id -> page configuration
        active_page_id: Currently active page ID
        name: Layout name
        description: Layout description

    Returns:
        JSON-serializable layout dict
    """
    layout = {
        "version": LAYOUT_VERSION,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "active_page": active_page_id,
        "pages": {},
    }

    for page_id, page_data in pages.items():
        layout["pages"][page_id] = {
            "name": page_data.get("name", "Untitled"),
            "parent_id": page_data.get("parent_id"),
            "grid_layout": page_data.get("grid_layout", {}),
            "panels": {},
        }

        # Serialize panel configurations
        panels = page_data.get("panels", {})
        for panel_id, panel_config in panels.items():
            layout["pages"][page_id]["panels"][panel_id] = {
                "type": panel_config.get("type"),
                "config": _sanitize_config(panel_config.get("config", {})),
            }

    return layout


def deserialize_layout(
    layout_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deserialize a layout from JSON data.

    Args:
        layout_data: Layout dict from JSON

    Returns:
        Dict with 'pages' and 'active_page' keys

    Raises:
        ValueError: If layout version is incompatible
    """
    version = layout_data.get("version", "0.0")

    # Check version compatibility
    major_version = version.split(".")[0]
    if major_version != LAYOUT_VERSION.split(".")[0]:
        raise ValueError(
            f"Incompatible layout version: {version}. "
            f"Expected version {LAYOUT_VERSION.split('.')[0]}.x"
        )

    pages = {}
    for page_id, page_data in layout_data.get("pages", {}).items():
        pages[page_id] = {
            "name": page_data.get("name", "Untitled"),
            "parent_id": page_data.get("parent_id"),
            "grid_layout": page_data.get("grid_layout", {}),
            "panels": {},
        }

        for panel_id, panel_data in page_data.get("panels", {}).items():
            pages[page_id]["panels"][panel_id] = {
                "type": panel_data.get("type"),
                "config": panel_data.get("config", {}),
            }

    return {
        "name": layout_data.get("name", "Untitled"),
        "pages": pages,
        "active_page": layout_data.get("active_page"),
    }


def save_layout_to_file(
    layout: Dict[str, Any],
    filepath: Path,
) -> None:
    """
    Save a layout to a JSON file.

    Args:
        layout: Serialized layout dict
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)


def load_layout_from_file(filepath: Path) -> Dict[str, Any]:
    """
    Load a layout from a JSON file.

    Args:
        filepath: Path to the layout file

    Returns:
        Deserialized layout data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    return deserialize_layout(layout_data)


def _sanitize_config(config: Dict) -> Dict:
    """
    Sanitize configuration for JSON serialization.

    Removes non-serializable values and converts numpy types.
    """
    sanitized = {}

    for key, value in config.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                v if isinstance(v, (str, int, float, bool, type(None)))
                else str(v)
                for v in value
            ]
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_config(value)
        elif hasattr(value, "item"):  # numpy scalar
            sanitized[key] = value.item()
        elif hasattr(value, "tolist"):  # numpy array
            sanitized[key] = value.tolist()
        else:
            sanitized[key] = str(value)

    return sanitized


def get_default_layout() -> Dict[str, Any]:
    """
    Get a default layout configuration for a new session.

    Returns:
        Default layout dict
    """
    return {
        "version": LAYOUT_VERSION,
        "name": "Default Layout",
        "pages": {
            "root": {
                "name": "All Cells",
                "parent_id": None,
                "grid_layout": {
                    "lg": [],
                    "md": [],
                    "sm": [],
                    "xs": [],
                },
                "panels": {},
            }
        },
        "active_page": "root",
    }


def create_page_layout(
    page_id: str,
    name: str,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new page configuration.

    Args:
        page_id: Unique page identifier
        name: Page display name
        parent_id: Parent page ID (for subsets)

    Returns:
        Page configuration dict
    """
    return {
        "name": name,
        "parent_id": parent_id,
        "grid_layout": {
            "lg": [],
            "md": [],
            "sm": [],
            "xs": [],
        },
        "panels": {},
    }
