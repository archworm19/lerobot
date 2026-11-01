"""The VLM harness: objects, long-term tools, rendering and the workspace DAG. Design: `docs/harness.md`."""

from .objects import (
    ArmPath,
    Capture,
    Focus,
    Grid,
    HeldObject,
    Plan,
    Points,
    Poses,
    Reach,
    TableEstimate,
    View,
)
from .render import contact_sheet, render
from .scene import Scene
from .tools import focus, full_view, locate_grid, locate_pixels, make_grid, table_height, zoom
from .workspace import Node, Workspace

__all__ = [
    "ArmPath",
    "Capture",
    "Focus",
    "Grid",
    "HeldObject",
    "Node",
    "Plan",
    "Points",
    "Poses",
    "Reach",
    "Scene",
    "TableEstimate",
    "View",
    "Workspace",
    "contact_sheet",
    "focus",
    "full_view",
    "locate_grid",
    "locate_pixels",
    "make_grid",
    "render",
    "table_height",
    "zoom",
]
