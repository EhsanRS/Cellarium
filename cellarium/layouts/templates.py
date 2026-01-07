"""
Predefined layout templates for quick setup.
"""

from typing import Any, Dict


def get_template_layouts() -> Dict[str, Dict[str, Any]]:
    """
    Get available layout templates.

    Returns:
        Dict of template_name -> template config
    """
    return {
        "empty": {
            "name": "Empty Layout",
            "description": "Start with a blank canvas",
            "panels": [],
        },
        "umap_only": {
            "name": "UMAP Overview",
            "description": "Single UMAP plot",
            "panels": [
                {"type": "scatter", "embedding": "X_umap", "w": 12, "h": 6},
            ],
        },
        "umap_pca": {
            "name": "UMAP + PCA",
            "description": "Side-by-side embedding views",
            "panels": [
                {"type": "scatter", "embedding": "X_umap", "w": 6, "h": 5},
                {"type": "scatter", "embedding": "X_pca", "w": 6, "h": 5},
            ],
        },
        "expression_focus": {
            "name": "Expression Analysis",
            "description": "UMAP with dotplot and violin",
            "panels": [
                {"type": "scatter", "embedding": "X_umap", "w": 6, "h": 4},
                {"type": "dotplot", "w": 6, "h": 4},
                {"type": "violin", "w": 12, "h": 4},
            ],
        },
        "comprehensive": {
            "name": "Comprehensive View",
            "description": "All plot types",
            "panels": [
                {"type": "scatter", "embedding": "X_umap", "w": 4, "h": 4},
                {"type": "scatter", "embedding": "X_pca", "w": 4, "h": 4},
                {"type": "heatmap", "w": 4, "h": 4},
                {"type": "dotplot", "w": 6, "h": 4},
                {"type": "violin", "w": 6, "h": 4},
            ],
        },
    }


def generate_layout_from_template(
    template_name: str,
    available_embeddings: list,
) -> Dict[str, Any]:
    """
    Generate a layout configuration from a template.

    Args:
        template_name: Name of the template to use
        available_embeddings: List of available embedding keys

    Returns:
        Layout configuration dict
    """
    templates = get_template_layouts()
    template = templates.get(template_name, templates["empty"])

    grid_layout = {"lg": [], "md": [], "sm": [], "xs": []}
    panels = {}

    y_position = 0
    for i, panel_spec in enumerate(template.get("panels", [])):
        panel_id = f"panel-{i}"
        panel_type = panel_spec.get("type", "scatter")
        w = panel_spec.get("w", 6)
        h = panel_spec.get("h", 4)

        # Adjust embedding if specified one not available
        embedding = panel_spec.get("embedding")
        if embedding and embedding not in available_embeddings:
            embedding = available_embeddings[0] if available_embeddings else None

        # Create panel config
        panels[panel_id] = {
            "type": panel_type,
            "config": {
                "embedding": embedding,
            } if panel_type == "scatter" else {},
        }

        # Calculate grid position (stack vertically for lg)
        if w == 12:
            x = 0
        else:
            x = (i * 6) % 12

        grid_layout["lg"].append({
            "i": panel_id,
            "x": x,
            "y": y_position if x == 0 or w == 12 else grid_layout["lg"][-1]["y"] if grid_layout["lg"] else 0,
            "w": w,
            "h": h,
            "minW": 3,
            "minH": 2,
        })

        # Simpler layouts for smaller screens
        grid_layout["md"].append({
            "i": panel_id,
            "x": 0,
            "y": i * h,
            "w": 10,
            "h": h,
            "minW": 3,
            "minH": 2,
        })

        grid_layout["sm"].append({
            "i": panel_id,
            "x": 0,
            "y": i * h,
            "w": 6,
            "h": h,
            "minW": 3,
            "minH": 2,
        })

        grid_layout["xs"].append({
            "i": panel_id,
            "x": 0,
            "y": i * h,
            "w": 4,
            "h": h,
            "minW": 2,
            "minH": 2,
        })

        if x == 0 or w == 12:
            y_position += h

    return {
        "grid_layout": grid_layout,
        "panels": panels,
    }
