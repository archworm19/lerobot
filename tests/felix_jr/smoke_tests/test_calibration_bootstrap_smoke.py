"""Starting a calibration session: the head tilt from the table plane, the first pose, and prior overrides."""

import numpy as np

from lerobot.felix_jr.calibration.bootstrap import (
    seed_pose_for_view,
)
from lerobot.felix_jr.calibration.fit import FitConfig, fit
from lerobot.felix_jr.calibration.model import (
    ARM_JOINTS,
    CalibrationModel,
    Observation,
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


def test_the_first_pose_puts_the_predicted_tip_near_the_middle_of_the_picture():
    model = CalibrationModel("left", INTRINSICS, SIGNS, REFS)
    theta = {"zero:head_tilt_joint": 0.3}
    ticks = {"head_neck": 1520, "head_eyes": 1324}
    pose = seed_pose_for_view(model, theta, ticks)
    assert pose is not None and pose["gripper.pos"] == 1.0
    assert set(pose) == {f"{j}.pos" for j in ARM_JOINTS} | {"gripper.pos"}
    prediction = model.predict(Observation(arm=pose, head_ticks=ticks, pixel=(0, 0)), theta)
    assert np.linalg.norm(prediction.uv - [W / 2, H / 2]) < 60
    assert 0.25 <= prediction.z <= 0.6
    base_z = model.head_tree.link_pose(model.base_link)[2, 3]
    assert model.tip_in_root(pose, theta)[2] - base_z >= 0.05  # not below the arm base
    assert seed_pose_for_view(model, theta, ticks) == pose  # deterministic
    assert seed_pose_for_view(model, theta, ticks, target_px=(5000.0, 5000.0)) is None
    right = seed_pose_for_view(CalibrationModel("right", INTRINSICS, SIGNS, REFS), theta, ticks)
    assert right is not None and right["shoulder_pan.pos"] * pose["shoulder_pan.pos"] <= 0  # mirrored sides


def test_a_good_tilt_prior_makes_a_handful_of_clicks_enough():
    plain = CalibrationModel("left", INTRINSICS, SIGNS, REFS)
    informed = CalibrationModel(
        "left", INTRINSICS, SIGNS, REFS, prior_overrides={"zero:head_tilt_joint": (0.31, np.radians(5.0))}
    )
    truth = {"zero:head_tilt_joint": 0.30, "zero:head_pan_joint": 0.05}
    rng = np.random.default_rng(0)
    clicks = []
    while len(clicks) < 8:
        arm = {f"{j}.pos": float(rng.uniform(-20, 20)) for j in ARM_JOINTS}
        arm["shoulder_pan.pos"] += 10
        ticks = {
            "head_neck": int(1520 + rng.integers(-250, 250)),
            "head_eyes": int(1324 + rng.integers(-150, 150)),
        }
        obs = Observation(arm=arm, head_ticks=ticks, pixel=(0.0, 0.0))
        prediction = plain.predict(obs, truth)
        if prediction.z > 0.1 and plain.in_view(prediction.uv, margin=15):
            obs.pixel = tuple(prediction.uv)
            clicks.append(obs)
    cfg = FitConfig(sigma_px=1.0)
    weak = fit(plain, clicks, cfg=cfg)
    strong = fit(informed, clicks, cfg=cfg)
    assert strong.sigma_post["zero:head_tilt_joint"] < weak.sigma_post["zero:head_tilt_joint"]
    assert abs(strong.theta["zero:head_tilt_joint"] - 0.30) < np.radians(1.5)


def test_the_first_pose_keeps_the_tip_above_the_table_when_its_height_is_known():
    model = CalibrationModel("left", INTRINSICS, SIGNS, REFS)
    theta = {"zero:head_tilt_joint": 0.3}
    ticks = {"head_neck": 1520, "head_eyes": 1324}
    free = seed_pose_for_view(model, theta, ticks)
    free_height = model.tip_in_root(free, theta)[2]
    table_z = free_height + 0.02  # a table that would have been in the way
    pose = seed_pose_for_view(model, theta, ticks, table_z=table_z)
    assert pose is None or model.tip_in_root(pose, theta)[2] >= table_z + 0.04 - 1e-9
    low_table = seed_pose_for_view(model, theta, ticks, table_z=free_height - 0.3)
    assert low_table is not None and model.tip_in_root(low_table, theta)[2] >= free_height - 0.3 + 0.04
