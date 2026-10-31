"""The VLM harness: objects, long-term tools, rendering and the workspace DAG. Design: `docs/harness.md`."""

from .objects import Capture, Focus, Grid, Plan, Points, Reach, TableEstimate, View
from .render import contact_sheet, render
from .scene import BLOCKED, FREE, UNKNOWN, Scene, SegmentCheck
from .tools import check_path, focus, full_view, locate_grid, locate_pixels, make_grid, table_height, zoom
from .workspace import Node, Workspace

__all__ = [
    "BLOCKED",
    "FREE",
    "UNKNOWN",
    "Capture",
    "Focus",
    "Grid",
    "Node",
    "Plan",
    "Points",
    "Reach",
    "Scene",
    "SegmentCheck",
    "TableEstimate",
    "View",
    "Workspace",
    "check_path",
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
