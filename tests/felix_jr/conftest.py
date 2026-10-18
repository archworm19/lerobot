"""Shared synthetic data for the felix_jr tests (no camera or robot needed).

The synthetic scene is a pinhole camera (fx = fy = 100 px, principal point (80, 60), 160 x 120 image)
looking at a flat wall 1.0 m away, with a box face 0.6 m away covering rows 40-79 / columns 60-99 and a hole
with no depth at rows 10-19 / columns 10-29.
"""

import numpy as np
import pytest

from lerobot.felix_jr.scene import Scene

H, W = 120, 160
FX = FY = 100.0
PPX, PPY = 80.0, 60.0
WALL_Z, BOX_Z = 1.0, 0.6


def make_depth() -> np.ndarray:
    depth = np.full((H, W), WALL_Z, dtype=np.float32)
    depth[40:80, 60:100] = BOX_Z
    depth[10:20, 10:30] = np.nan
    return depth


@pytest.fixture
def scene() -> Scene:
    return Scene(np.zeros((H, W, 3), dtype=np.uint8), make_depth(), FX, FY, PPX, PPY)


def make_calibration(translation=(0.0, 0.0, 0.0)) -> dict:
    """Calibration dict in the format of calibration.json, with the same intrinsics for both cameras."""
    intrinsics = {"width": W, "height": H, "fx": FX, "fy": FY, "ppx": PPX, "ppy": PPY}
    return {
        "depth_scale_m_per_unit": 0.001,
        "color_intrinsics": dict(intrinsics),
        "depth_intrinsics": dict(intrinsics),
        "depth_to_color_rotation_col_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "depth_to_color_translation_m": list(translation),
    }
