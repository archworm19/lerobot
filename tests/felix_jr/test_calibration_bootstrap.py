"""Starting a calibration session: the head tilt from the table plane, the first pose, and prior overrides."""

import numpy as np
import pytest

from lerobot.felix_jr.calibration.bootstrap import (
    Plane,
    estimate_tilt_zero,
    estimate_up_vector,
    table_height,
    up_from_pitch,
)
from lerobot.felix_jr.calibration.model import (
    CalibrationModel,
    load_observations,
    parameter_set,
    save_observations,
)
from lerobot.felix_jr.scene import Scene

W, H, FX, FY, PPX, PPY = 640, 480, 610.0, 610.0, 316.0, 247.0
INTRINSICS = {"fx": FX, "fy": FY, "ppx": PPX, "ppy": PPY, "width": W, "height": H}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1520, "head_eyes": 1324}


def table_scene(up_cam, height=0.6, wall=1.2, noise=0.003, with_table=True, seed=0):
    """A depth image of a horizontal table `height` m below the camera, with a wall behind it `wall` m away."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - PPX) / FX, (v - PPY) / FY, np.ones_like(u, dtype=float)], axis=-1)
    depth = np.full((H, W), np.inf)
    up_dot = rays @ up_cam
    if with_table:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(up_dot < -1e-6, -height / up_dot, np.inf)
        depth = np.minimum(depth, t)
    # the wall: a vertical plane facing the camera, perpendicular to the horizontal forward direction
    forward = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ up_cam) * up_cam
    forward /= np.linalg.norm(forward)
    facing = rays @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.minimum(depth, np.where(facing > 1e-6, wall / facing, np.inf))
    depth = np.where(depth < 3.0, depth + rng.normal(0, noise, depth.shape), np.nan)
    return Scene(np.zeros((H, W, 3), np.uint8), depth.astype(np.float32), FX, FY, PPX, PPY)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


@pytest.mark.parametrize("pitch,roll", [(10.0, 0.0), (17.0, -2.0), (30.0, 3.0), (45.0, 0.0)])
def test_the_up_direction_is_found_from_the_table_even_with_a_wall_in_view(pitch, roll):
    up = up_from_pitch(np.radians(pitch), np.radians(roll))
    found = estimate_up_vector(table_scene(up))
    assert found is not None
    estimated, fraction = found.normal, found.fraction
    assert angle_deg(estimated, up) < 2.5  # shallow views are noisier; the fit's prior (5 deg) absorbs this
    assert 0.05 < fraction < 1.0


def test_no_table_means_no_estimate():
    up = up_from_pitch(np.radians(17.0))
    assert estimate_up_vector(table_scene(up, with_table=False)) is None  # only a wall: it faces sideways
    noise = Scene(
        np.zeros((H, W, 3), np.uint8),
        np.random.default_rng(1).uniform(0.3, 1.4, (H, W)).astype(np.float32),
        FX,
        FY,
        PPX,
        PPY,
    )
    assert estimate_up_vector(noise) is None
    empty = Scene(np.zeros((H, W, 3), np.uint8), np.full((H, W), np.nan, np.float32), FX, FY, PPX, PPY)
    assert estimate_up_vector(empty) is None


def test_the_head_tilt_zero_is_recovered_from_the_up_direction():
    model = CalibrationModel("left", INTRINSICS, SIGNS, REFS)
    ticks = {"head_neck": 1500, "head_eyes": 1300}
    for true_tilt in (-0.2, 0.0, 0.3, 0.6):
        camera = model.camera_pose(ticks, {"zero:head_tilt_joint": true_tilt})
        up_cam = camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        tilt, miss = estimate_tilt_zero(model, ticks, up_cam)
        assert tilt == pytest.approx(true_tilt, abs=np.radians(0.2)) and miss < 0.2
    # a pitch given by hand goes the same way: the camera really does look that far down
    tilt, miss = estimate_tilt_zero(model, REFS, up_from_pitch(np.radians(17.0)))
    camera = model.camera_pose(REFS, {"zero:head_tilt_joint": tilt})
    assert miss < 2.5
    assert np.degrees(np.arcsin(-(camera[:3, :3].T @ np.array([0.0, 0.0, 1.0]))[2])) == pytest.approx(
        17.0, abs=2.5
    )
    # an up vector the head cannot produce shows up as a large miss
    assert estimate_tilt_zero(model, REFS, np.array([1.0, 0.0, 0.0]))[1] > 20


def test_prior_overrides_shift_the_prior_and_survive_the_saved_clicks(tmp_path):
    overrides = {"zero:head_tilt_joint": (0.3, np.radians(5.0))}
    params = {p.name: p for p in parameter_set("left", overrides)}
    assert params["zero:head_tilt_joint"].mean == 0.3 and params[
        "zero:head_tilt_joint"
    ].sigma == pytest.approx(np.radians(5))
    assert params["zero:head_pan_joint"].mean == 0.0
    model = CalibrationModel("left", INTRINSICS, SIGNS, REFS, prior_overrides=overrides)
    save_observations(tmp_path / "o.json", model, [])
    reloaded, _ = load_observations(tmp_path / "o.json")
    assert reloaded.prior_overrides == {"zero:head_tilt_joint": (0.3, pytest.approx(np.radians(5.0)))}


def test_the_table_height_comes_from_the_plane_and_the_cameras_height():
    model = CalibrationModel("left", INTRINSICS, SIGNS, REFS)
    theta = {"zero:head_tilt_joint": 0.3}
    ticks = {"head_neck": 1520, "head_eyes": 1324}
    camera = model.camera_pose(ticks, theta)
    up_cam = camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    plane = estimate_up_vector(table_scene(up_cam, height=0.55))
    assert plane is not None and plane.offset < 0
    assert table_height(model, ticks, theta, plane) == pytest.approx(camera[2, 3] - 0.55, abs=0.02)
    exact = Plane(up_cam, -0.4, 0.2)  # 40 cm below the camera
    assert table_height(model, ticks, theta, exact) == pytest.approx(camera[2, 3] - 0.4, abs=1e-6)
