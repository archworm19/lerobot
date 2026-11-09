"""A wrist scan (`harness.wrist_scan`): frames of a synthetic table, their contact sheet and the scale set by the table.

The frames are `test_sfm`'s wrist-roll sweep over a textured table, the model's poses off by 2 mm and 0.5 deg per
frame. The scale is the model's (the images fix the motion up to one factor) until the table's height sets it.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_sfm import CAMERA, TABLE_Z, perturbed, render, table_texture, wrist_sweep  # noqa: E402

from lerobot.felix_jr.harness.objects import Object  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.sfm import _pose  # noqa: E402
from lerobot.felix_jr.harness.wrist import WristView  # noqa: E402
from lerobot.felix_jr.harness.wrist_scan import (  # noqa: E402
    render as render_scan,
    rescale,
    scale_from_table,
    scan_views,
)

CAMERA_FROM_PINCH = _pose(np.zeros(3), np.array([0.0, 0.03, 0.10]))  # the jaws 10 cm ahead of the camera
CURRENT = 2  # the unrolled frame: where the arm is
TOLERANCE_M = 0.006  # the model's scale: ~3% of the distances


@pytest.fixture(scope="module")
def scene():
    texture = table_texture()
    truth = wrist_sweep()
    model = perturbed(truth, mm=2.0, deg=0.5)
    views = [
        WristView("right", render(texture, t), CAMERA, {}, m @ CAMERA_FROM_PINCH, m)
        for t, m in zip(truth, model, strict=True)
    ]
    scan = scan_views(views, "light", CURRENT, [f"roll {r:+d}" for r in (-30, -15, 0, 15, 30)])
    return scan, truth


def test_the_contact_sheet_and_a_frame(scene):
    scan, _ = scene
    bottle = Object(
        "bottle", {"body": Cylinder.upright(np.array([-0.37, 0.07, TABLE_Z]), 0.06, 0.018)}, None, ""
    )
    sheet = render_scan(scan, objects=[bottle])
    assert sheet.shape == (2 * 240, 3 * 320, 3)
    one = render_scan(scan, frames=[CURRENT], objects=[bottle])
    assert one.shape == (480, 640, 3) and (one != scan.views[CURRENT].color).any()
    assert "the arm is at frame 2" in scan.describe()


def _camera_height(scan):
    """The current camera's height over the scan's table plane."""
    normal, offset = scan.sfm.plane
    return float(normal @ scan.camera(scan.current)[:3, 3] - offset)


def test_the_scale_set_by_the_table(scene):
    """A scan whose scale is 10% off (as the model's moves on the robot) is set right by the table's height."""
    scan, truth = scene
    true_height = truth[CURRENT][2, 3] - TABLE_Z
    off = rescale(scan, 0.9, "10% short")
    assert abs(_camera_height(off) - true_height) > 0.01  # 10% of ~15 cm
    by_table = scale_from_table(off, TABLE_Z)
    assert "the table's height" in by_table.scale_note
    assert abs(_camera_height(by_table) - true_height) < TOLERANCE_M
