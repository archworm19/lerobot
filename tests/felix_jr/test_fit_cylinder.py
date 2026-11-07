"""`tools.fit_cylinder`: a cylinder from the depth where a view is marked, on a rendered scene (`docs/object_grasping.md`).

The scene is rendered here (sphere-traced against the primitives' own `surface_distance` and a table plane, 1 mm of
depth noise), so the truth is known; the recorded sessions' captures are gitignored. On them, the fit was checked by
hand (2026-11-06): bowl1's jar, white container and chapstick within 0.4 cm of the session's widths and 0.1 cm of
its tops; pen2's pen, free, 52-55 deg from vertical (56) with its axis 0.36 cm behind the marked surface.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from lerobot.felix_jr.harness.objects import Capture  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.scene import Scene  # noqa: E402
from lerobot.felix_jr.harness.tools import fit_cylinder, full_view  # noqa: E402

W, H, F = 320, 240, 300.0
TABLE_Z = 0.715
JAR = Cylinder.upright(np.array([-0.35, -0.10, TABLE_Z]), 0.078, 0.02)  # a 4 cm jar, its top at 79.3 cm
PEN_AXIS = np.array([-0.192, 0.802, 0.566]) / np.linalg.norm([-0.192, 0.802, 0.566])  # 55 deg from vertical
PEN = Cylinder.between(np.array([-0.40, 0.05, 0.75]), np.array([-0.40, 0.05, 0.75]) + 0.10 * PEN_AXIS, 0.0045)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """root_from_camera (optical: x right, y down, z forward) for a camera at `eye` looking at `target`."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    pose[:3, 3] = eye
    return pose


def render(shapes: list[Cylinder], eye: np.ndarray, target: np.ndarray, seed: int = 0) -> Capture:
    """A capture of `shapes` on the table, from `eye`: depth by sphere tracing, 1 mm of noise."""
    pose = look_at(eye, target)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_cam = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], axis=-1).reshape(-1, 3)
    rays = (rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)) @ pose[:3, :3].T
    t = np.full(len(rays), 0.05)
    for _ in range(120):
        points = eye + t[:, None] * rays
        distance = points[:, 2] - TABLE_Z
        for shape in shapes:
            distance = np.minimum(distance, shape.surface_distance(points))
        t += np.where(distance > 1e-5, distance, 0.0)  # a ray on a surface stays there
    points = eye + t[:, None] * rays
    z_cam = (points - eye) @ pose[:3, 2]
    depth = (
        (z_cam + np.random.default_rng(seed).normal(0, 0.001, len(z_cam))).reshape(H, W).astype(np.float32)
    )
    depth[depth > 3.0] = np.nan
    scene = Scene(np.zeros((H, W, 3), np.uint8), depth, F, F, W / 2, H / 2)
    return Capture(scene, pose, {}, {}, 0.0)


EYE = np.array([-0.13, -0.03, 1.16])  # about the head camera's place, looking down at the table in front


def pixels(capture: Capture, points_root: np.ndarray) -> np.ndarray:
    return capture.project(points_root)


def marks_around(capture: Capture, cylinder: Cylinder, pad: float = 0.015) -> np.ndarray:
    """Marks a little outside an upright cylinder, around it: the region takes in some table, as a person's would."""
    b, top = cylinder.bottom_root, cylinder.top_root
    r = cylinder.radius + pad
    ring = [np.array([np.cos(a), np.sin(a), 0.0]) * r for a in np.radians(np.arange(0, 360, 45))]
    return pixels(capture, np.array([b + d for d in ring] + [top + d for d in ring]))


def test_an_upright_jar_among_the_table():
    capture = render([JAR], EYE, JAR.bottom_root)
    jar = fit_cylinder(full_view(capture), marks_around(capture, JAR), "up", "jar").part()
    np.testing.assert_allclose(jar.bottom_root[:2], JAR.bottom_root[:2], atol=0.003)
    assert jar.radius == pytest.approx(JAR.radius, abs=0.002)
    assert jar.top_root[2] == pytest.approx(JAR.top_root[2], abs=0.003)  # the lid counts
    assert abs(jar.axis_root[2]) == pytest.approx(1.0)


def test_the_note_says_what_it_fitted():
    capture = render([JAR], EYE, JAR.bottom_root)
    obj = fit_cylinder(full_view(capture), marks_around(capture, JAR), "up", "jar")
    assert obj.label == "jar" and obj.capture is capture
    assert "fitted to" in obj.note and "residual" in obj.note and "deg of it" in obj.note
    assert "jar: body: cylinder" in obj.describe()


def test_a_view_from_straight_above_is_refused():
    above = render([JAR], JAR.top_root + np.array([0.001, 0.0, 0.40]), JAR.bottom_root)
    with pytest.raises(ValueError, match="does not show the object's side"):
        fit_cylinder(full_view(above), marks_around(above, JAR), "up", "jar")


def test_a_leaning_pen_with_its_radius_given():
    capture = render([PEN], EYE, PEN.bottom_root)
    along = PEN.bottom_root + np.linspace(0.02, 0.08, 6)[:, None] * PEN.axis_root
    pen = fit_cylinder(full_view(capture), pixels(capture, along), "free", "pen", radius=0.0045).part()
    assert pen.radius == 0.0045
    angle = np.degrees(np.arccos(min(abs(float(pen.axis_root @ PEN.axis_root)), 1.0)))
    assert angle < 3.0
    middle = PEN.bottom_root + 0.05 * PEN.axis_root
    rel = middle - pen.bottom_root
    assert np.linalg.norm(rel - (rel @ pen.axis_root) * pen.axis_root) < 0.003
    assert pen.top_root[2] > pen.bottom_root[2]  # oriented up


def test_too_few_marks_with_depth():
    capture = render([JAR], EYE, JAR.bottom_root)
    far = np.array([[1.0, 1.0]])  # one mark: a few pixels
    with pytest.raises(ValueError, match="only"):
        fit_cylinder(full_view(capture), far, "up", "jar", pad_px=1)
    with pytest.raises(ValueError, match="'up' or 'free'"):
        fit_cylinder(full_view(capture), marks_around(capture, JAR), "sideways", "jar")  # type: ignore[arg-type]
