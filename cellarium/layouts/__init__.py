"""
Layout management for Cellarium.

This package handles:
- Layout templates (predefined dashboard arrangements)
- Layout serialization (JSON save/load)
- Grid layout configuration
"""

from cellarium.layouts.serializer import (
    serialize_layout,
    deserialize_layout,
    save_layout_to_file,
    load_layout_from_file,
    get_default_layout,
    create_page_layout,
)
from cellarium.layouts.templates import get_template_layouts, generate_layout_from_template

__all__ = [
    "serialize_layout",
    "deserialize_layout",
    "save_layout_to_file",
    "load_layout_from_file",
    "get_default_layout",
    "create_page_layout",
    "get_template_layouts",
    "generate_layout_from_template",
]
