"""Rebuild the depth-alignment images from a `realsense_check --save-dir` folder (no camera, no root).

Needs `color.png`, `depth_raw.png` and `calibration.json` in the folder; writes `alignment.png` and
`alignment_zoom.png` next to them. Useful for trying changes to `felix_jr/depth.py` on a captured frame.

Usage: uv run python -m lerobot.felix_jr.diagnostics.alignment_from_saved DIR
"""

import sys
from pathlib import Path

import cv2

from lerobot.felix_jr.depth import alignment_panel, load_calibration

folder = Path(sys.argv[1])
color = cv2.cvtColor(cv2.imread(str(folder / "color.png")), cv2.COLOR_BGR2RGB)
depth = cv2.imread(str(folder / "depth_raw.png"), cv2.IMREAD_UNCHANGED)  # 16-bit, raw units
panel, zoom = alignment_panel(color, depth, load_calibration(folder / "calibration.json"))
cv2.imwrite(str(folder / "alignment.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(folder / "alignment_zoom.png"), cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
print(f"Wrote {folder / 'alignment.png'} and {folder / 'alignment_zoom.png'}")
