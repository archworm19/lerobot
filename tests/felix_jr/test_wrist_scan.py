"""A wrist scan (`harness.scan`): points and an upright cylinder from frames of a synthetic table, in the jaws' frame.

The frames are `test_sfm`'s wrist-roll sweep over a textured table, the model's poses off by 2 mm and 0.5 deg per
frame. Answers are checked in the current pinch frame against the truth; the scale is the model's (the images fix the
motion up to one factor), so a few mm of the ~15-20 cm distances is allowed.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_sfm import CAMERA, TABLE_Z, perturbed, render, table_texture, wrist_sweep  # noqa: E402

from lerobot.felix_jr.harness.sfm import _pose  # noqa: E402
from lerobot.felix_jr.harness.wrist import WristView  # noqa: E402
from lerobot.felix_jr.harness.wrist_scan import (  # noqa: E402
    find,
    fit_upright_cylinder,
    locate,
    render as render_scan,
    rescale,
    scale_from_table,
    scale_from_width,
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


def true_pinch(truth, point):
    pinch = truth[CURRENT] @ CAMERA_FROM_PINCH
    inv = np.linalg.inv(pinch)
    return inv[:3, :3] @ point + inv[:3, 3]


def pixel(truth, frame, point):
    inv = np.linalg.inv(truth[frame])
    return CAMERA.project((inv[:3, :3] @ point + inv[:3, 3])[None, :])[0]


def test_a_table_point_by_the_scene_points_around_it(scene):
    scan, truth = scene
    point = np.array([-0.37, 0.03, TABLE_Z])
    found = locate(scan, 1, pixel(truth, 1, point), label="table")
    assert "points around it" in found.how, found.describe()
    assert np.linalg.norm(found.pinch_xyz - true_pinch(truth, point)) < TOLERANCE_M, found.describe()


def test_a_point_off_the_table_from_two_marks(scene):
    scan, truth = scene
    point = np.array([-0.38, 0.06, TABLE_Z + 0.05])  # say, a bottle's cap: no scene points of its own
    found = locate(scan, 0, pixel(truth, 0, point), 4, pixel(truth, 4, point), "cap")
    assert "two marks" in found.how, found.describe()
    assert np.linalg.norm(found.pinch_xyz - true_pinch(truth, point)) < TOLERANCE_M, found.describe()


def test_a_point_found_in_another_frame(scene):
    scan, truth = scene
    point = np.array([-0.42, 0.07, TABLE_Z])
    found = find(scan, CURRENT, pixel(truth, CURRENT, point), label="grain")
    assert "found in frame" in found.how, found.describe()
    assert np.linalg.norm(found.pinch_xyz - true_pinch(truth, point)) < TOLERANCE_M, found.describe()


def test_a_smooth_patch_is_not_found(scene):
    scan, truth = scene
    blank = [v.color.copy() for v in scan.views]
    for image in blank:
        image[150:330, 220:420] = 128  # a featureless patch where the point is
    from dataclasses import replace

    smooth = replace(scan, views=tuple(replace(v, color=b) for v, b in zip(scan.views, blank, strict=True)))
    with pytest.raises(ValueError, match="not found|more than once|by hand"):
        find(smooth, CURRENT, (320.0, 240.0))


def tangent_sides(truth, frame, center_xy, radius, height):
    """The silhouette of an upright cylinder at `height` over the table, as frame `frame` sees it: two pixels."""
    pose = truth[frame]
    camera = pose[:3, 3]
    axis_point = np.array([*center_xy, TABLE_Z + height])
    across = camera - axis_point
    across[2] = 0.0
    d = np.linalg.norm(across)
    e = across / d
    w = np.cross([0.0, 0.0, 1.0], e)
    sides = [
        axis_point + (radius**2 / d) * e + s * radius * np.sqrt(1 - radius**2 / d**2) * w for s in (1, -1)
    ]
    return [pixel(truth, frame, p) for p in sides]


def test_an_upright_cylinder_from_its_sides(scene):
    scan, truth = scene
    center, radius, height = (-0.37, 0.07), 0.018, 0.06
    sides = {f: tangent_sides(truth, f, center, radius, h) for f, h in ((0, 0.02), (2, 0.04), (4, 0.03))}
    bottle = fit_upright_cylinder(scan, sides, height=height, label="bottle")
    body = bottle.part()
    # where it is relative to the jaws now: the axis's foot in the pinch frame
    root_from_pinch = scan.views[CURRENT].root_from_pinch
    foot_pinch = np.linalg.inv(root_from_pinch) @ np.array([*body.bottom_root, 1.0])
    expected = true_pinch(truth, np.array([*center, TABLE_Z]))
    assert np.linalg.norm(foot_pinch[:3] - expected) < TOLERANCE_M, (foot_pinch[:3], expected, bottle.note)
    assert body.radius == pytest.approx(radius, abs=0.003), bottle.note
    assert body.axis_root[2] > 0.999 and "wrist scan" in bottle.note
    # given the radius (the head's), the same place
    given = fit_upright_cylinder(scan, sides, radius=radius, height=height).part()
    np.testing.assert_allclose(given.bottom_root, body.bottom_root, atol=0.004)


def test_the_contact_sheet_and_a_frame(scene):
    scan, truth = scene
    point = locate(scan, 1, pixel(truth, 1, np.array([-0.37, 0.03, TABLE_Z])), label="table")
    sides = {f: tangent_sides(truth, f, (-0.37, 0.07), 0.018, 0.03) for f in (0, 2, 4)}
    bottle = fit_upright_cylinder(scan, sides, height=0.06)
    sheet = render_scan(scan, points=[point], objects=[bottle])
    assert sheet.shape == (2 * 240, 3 * 320, 3)
    one = render_scan(scan, frames=[CURRENT], points=[point], objects=[bottle])
    assert one.shape == (480, 640, 3) and (one != scan.views[CURRENT].color).any()
    assert "the arm is at frame 2" in scan.describe()


def test_the_scale_set_by_the_table_or_the_objects_width(scene):
    """A scan whose scale is 10% off (as the model's moves on the robot) is set right by the table or a width."""
    scan, truth = scene
    off = rescale(scan, 0.9, "10% short")
    point = np.array([-0.37, 0.03, TABLE_Z])
    expected = true_pinch(truth, point)
    short = locate(off, 1, pixel(truth, 1, point))
    assert np.linalg.norm(short.pinch_xyz - expected) > 0.01  # 10% of ~15 cm
    by_table = scale_from_table(off, TABLE_Z)
    assert "the table's height" in by_table.scale_note
    assert np.linalg.norm(locate(by_table, 1, pixel(truth, 1, point)).pinch_xyz - expected) < TOLERANCE_M
    sides = {
        f: tangent_sides(truth, f, (-0.37, 0.07), 0.018, h) for f, h in ((0, 0.02), (2, 0.04), (4, 0.03))
    }
    by_width = scale_from_width(off, sides, 0.036)
    assert "the object's width" in by_width.scale_note
    assert fit_upright_cylinder(by_width, sides, height=0.06).part().radius == pytest.approx(0.018, abs=1e-6)
    assert np.linalg.norm(locate(by_width, 1, pixel(truth, 1, point)).pinch_xyz - expected) < TOLERANCE_M
