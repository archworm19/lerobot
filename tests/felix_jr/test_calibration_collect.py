"""Pose planning, head centering and the click-collection session (fake robot behind a real ZeroMQ socket)."""

import threading

import cv2
import numpy as np
import pytest

from lerobot.felix_jr.calibration import collect as collect_module, design
from lerobot.felix_jr.calibration.collect import CollectSession, decode_key, parse_seed
from lerobot.felix_jr.calibration.design import (
    JOINT_LIMITS,
    LIMIT_MARGIN,
    best_roll,
    center_head,
    plan_poses,
)
from lerobot.felix_jr.calibration.model import ARM_JOINTS, CalibrationModel, Observation
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware
from lerobot.felix_jr.server.server import RobotServer

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 2048, "head_eyes": 2048}


def make_model():
    return CalibrationModel("left", INTRINSICS, SIGNS, REFS)


def test_plan_poses_spreads_poses_around_the_seed_inside_the_joint_limits():
    poses = plan_poses(30, rng_seed=1)
    assert len(poses) == 30
    assert poses[0] == {**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, "gripper.pos": 1.0}  # the seed pose first
    assert all(p["gripper.pos"] == 1.0 for p in poses)  # always closed: the closed tip is the target
    for pose in poses:
        for joint in ARM_JOINTS:
            low, high = JOINT_LIMITS[joint]
            assert low * LIMIT_MARGIN <= pose[f"{joint}.pos"] <= high * LIMIT_MARGIN
            assert abs(pose[f"{joint}.pos"]) <= design.DEFAULT_DELTA[joint]
    matrix = np.array([[p[f"{j}.pos"] / design.DEFAULT_DELTA[j] for j in ARM_JOINTS] for p in poses])
    gaps = np.linalg.norm(matrix[:, None] - matrix[None], axis=2) + np.eye(len(poses)) * 99
    assert gaps.min() > 0.25  # spread out, not clustered
    assert plan_poses(30, rng_seed=1) == poses  # deterministic


def test_plan_poses_respects_a_seed_pose_and_a_scale_and_the_limits():
    seed = {"shoulder_lift": -40.0, "elbow_flex": 60.0}
    poses = plan_poses(10, seed, delta_scale=0.5)
    assert poses[0]["shoulder_lift.pos"] == -40.0 and poses[0]["elbow_flex.pos"] == 60.0
    assert all(abs(p["shoulder_lift.pos"] + 40.0) <= 10.0 + 1e-9 for p in poses)
    near_limit = plan_poses(10, {"elbow_flex": 90.0})  # a seed beyond the margin is pulled inside it
    assert max(p["elbow_flex.pos"] for p in near_limit) <= 97.0 * LIMIT_MARGIN + 1e-9


def test_center_head_puts_the_predicted_tip_in_the_middle_of_the_image():
    model = make_model()
    theta = {"zero:head_tilt_joint": 0.1}
    arm = {f"{j}.pos": 0.0 for j in ARM_JOINTS}
    ticks = {"head_neck": 2048, "head_eyes": 2048}
    target = center_head(model, theta, arm, ticks)
    assert target is not None
    pred = model.predict(Observation(arm=arm, head_ticks=target, pixel=(0, 0)), theta)
    np.testing.assert_allclose(pred.uv, [320, 240], atol=3.0)
    other = center_head(model, theta, arm, ticks, target=(100.0, 400.0))
    pred = model.predict(Observation(arm=arm, head_ticks=other, pixel=(0, 0)), theta)
    np.testing.assert_allclose(pred.uv, [100, 400], atol=3.0)


def test_center_head_stays_inside_the_head_limits_and_gives_up_when_the_tip_is_behind_the_camera():
    model = make_model()
    arm = {f"{j}.pos": 0.0 for j in ARM_JOINTS}
    free = center_head(model, {}, arm, {"head_neck": 2048, "head_eyes": 2048})
    limited = center_head(
        model,
        {},
        arm,
        {"head_neck": 2048, "head_eyes": 2048},
        limits={"head_eyes": (2040, 2050), "head_neck": (0, 4095)},
    )
    assert 2040 <= limited["head_eyes"] <= 2050 and abs(free["head_eyes"] - 2048) > 12
    behind = center_head(
        model, {}, arm, {"head_neck": 2048 + 1500, "head_eyes": 2048}
    )  # head turned far away
    assert behind is None


def test_decode_key_and_parse_seed():
    assert decode_key(63232) == "up" and decode_key(65364) == "down" and decode_key(63234) == "left"
    assert decode_key(ord("j")) == "left" and decode_key(ord("l")) == "right"
    assert decode_key(ord("n")) == "next" and decode_key(ord("u")) == "undo" and decode_key(27) == "quit"
    assert decode_key(ord("z")) is None
    assert parse_seed("shoulder_pan=10, elbow_flex=-5") == {"shoulder_pan": 10.0, "elbow_flex": -5.0}
    assert parse_seed(None) is None


@pytest.fixture
def robot(tmp_path):
    fake = FakeHardware(arm_time_scale=0.02, head_range_path=tmp_path / "range.json", session_timeout_s=5.0)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=5.0) as client:
        yield fake, client
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def make_session(client, tmp_path, n_poses=12, **kwargs):
    shown = []
    session = CollectSession(client, "left", tmp_path / "cal", shown.append, plan_poses(n_poses), **kwargs)
    return session, shown


def test_arrow_keys_turn_the_head_and_a_fresh_frame_is_shown(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, tmp_path, head_step_deg=2.0)
        eyes0, neck0 = fake.head.backend.read_ticks("head_eyes"), fake.head.backend.read_ticks("head_neck")
        frames = len(shown)
        assert session.handle("down")
        assert fake.head.backend.read_ticks("head_eyes") == eyes0 + round(
            2.0 * 4096 / 360
        )  # eyes: positive = down
        assert session.handle("left")
        assert fake.head.backend.read_ticks("head_neck") == neck0 - round(2.0 * 4096 / 360)  # neck sign is -1
        assert session.handle("up") and session.handle("right")
        assert (
            fake.head.backend.read_ticks("head_eyes") == eyes0
            and fake.head.backend.read_ticks("head_neck") == neck0
        )
        assert len(shown) == frames + 4
        assert session.handle("quit") is False


def test_next_pose_moves_the_arm_closed_and_runs_out_of_poses(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, tmp_path, n_poses=2)
        session.next_pose()
        assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == 1.0
        assert fake.arm_backend.trajectories[-1][1:] == (30.0, 60.0)  # slow
        assert set(fake.arm_backend.trajectories[-1][0]) == {"left_arm"}
        session.next_pose()
        session.next_pose()
        assert "no more planned poses" in session.message


def test_the_overlay_marks_the_prediction_and_the_clicks(robot, tmp_path):
    from lerobot.felix_jr.calibration.collect import annotate

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    out = annotate(image, ["hello"], np.array([160.0, 120.0]), [(40.0, 40.0)])
    assert out.shape == image.shape and out.any() and not image.any()  # a copy, with something drawn
    assert annotate(image, [], None, []).sum() == 0
    assert cv2 is not None


def test_plan_poses_can_veto_candidates():
    allowed = plan_poses(20, keep=lambda pose: pose["shoulder_lift.pos"] <= 0.0, rng_seed=3)
    assert allowed[0]["shoulder_lift.pos"] == 0.0 and all(p["shoulder_lift.pos"] <= 0.0 for p in allowed)
    assert len(allowed) == 20
    assert len(plan_poses(20, keep=lambda pose: False)) == 1  # only the seed pose survives


def test_bootstrap_can_be_switched_off(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client, "left", tmp_path / "cal", lambda image: None, None, n_poses=4, bootstrap=False
        )
        assert not any("tilt" in note or "first pose" in note for note in session.notes)
        assert set(session.model.prior_overrides) == {"zero:head_pan_joint"}  # only the neck's missing home
        assert all(abs(session.poses[0][f"{j}.pos"]) < 1e-9 for j in ARM_JOINTS)  # the plain center seed


def test_head_limits_use_recorded_limits_or_a_range_around_the_present_position():
    head = {
        "head_eyes": {"ticks": 1324, "limit_neg": 1013, "limit_pos": 1620},
        "head_neck": {"ticks": 1520, "limit_neg": None, "limit_pos": None},
    }
    limits = collect_module.head_limits(head)
    assert limits["head_eyes"] == (1013, 1620)
    assert limits["head_neck"] == (
        1520 - collect_module.NECK_RANGE_TICKS,
        1520 + collect_module.NECK_RANGE_TICKS,
    )
    head["head_neck"]["ticks"] = 100  # near the end of the raw range
    assert collect_module.head_limits(head)["head_neck"][0] == 0


def test_the_hinge_axis_angle_depends_on_the_wrist_roll_and_best_roll_minimizes_it():
    model = make_model()
    theta = {"zero:head_tilt_joint": 0.4}
    ticks = {"head_neck": 2048, "head_eyes": 2048}
    pose = {**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, "shoulder_pan.pos": 30.0, "gripper.pos": 1.0}
    angles = [
        model.hinge_axis_angle({**pose, "wrist_roll.pos": r}, ticks, theta) for r in range(-100, 101, 10)
    ]
    assert all(0.0 <= a <= 90.0 for a in angles) and max(angles) - min(angles) > 20  # the roll matters a lot
    roll, angle = best_roll(model, theta, pose, ticks, max_roll=100)
    assert -100 <= roll <= 100 and angle <= min(angles) + 1e-9
    narrow_roll, narrow = best_roll(model, theta, pose, ticks, max_roll=20)
    assert abs(narrow_roll) <= 20 and narrow >= angle
    # the tip barely moves with the roll (it is about 8 mm off the roll axis)
    moved = model.tip_in_root({**pose, "wrist_roll.pos": 90.0}, theta) - model.tip_in_root(pose, theta)
    assert np.linalg.norm(moved) < 0.02


def test_plan_poses_can_adjust_candidates():
    rolled = plan_poses(10, adjust=lambda pose: {**pose, "wrist_roll.pos": 77.0}, rng_seed=2)
    assert rolled[0]["wrist_roll.pos"] == 0.0  # the seed is left as it is
    assert all(p["wrist_roll.pos"] == 77.0 for p in rolled[1:])
    assert len(plan_poses(10, adjust=lambda pose: None)) == 1


def test_a_neck_without_a_recorded_home_gets_a_wide_pan_prior(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client, "left", tmp_path / "cal", lambda image: None, None, n_poses=3, bootstrap=False
        )
        assert session.model.prior_overrides["zero:head_pan_joint"] == (
            0.0,
            collect_module.NECK_PRIOR_NO_HOME,
        )
        assert any("no recorded home" in note for note in session.notes)


def earlier_fit(folder, signs=None):
    """A `fit` output folder (only the JSON `collect --head-from` reads), as if the left arm had been calibrated."""
    from lerobot.felix_jr.calibration.model import parameter_set

    folder.mkdir(parents=True, exist_ok=True)
    params = parameter_set("left")
    values = {p.name: 0.001 * (i + 1) for i, p in enumerate(params)}
    values["zero:head_tilt_joint"], values["pos:fixed_head_camera_link:y"] = 0.58, 0.021
    sigma = {p.name: 0.5 * p.sigma for p in params}
    sigma["rot:fixed_head_camera_link:roll"] = 1e-6  # absurdly sure: the floor applies
    data = {
        "settings": {
            "arm": "left",
            "intrinsics": INTRINSICS,
            "head_signs": signs or SIGNS,
            "head_refs": {"head_neck": 1900, "head_eyes": 2000},
            "prior_overrides": {},
        },
        "parameters": values,
        "sigma_post": sigma,
    }
    (folder / "kinematic_calibration.json").write_text(__import__("json").dumps(data))
    return data


def test_head_priors_take_the_head_and_camera_from_an_earlier_fit_and_nothing_else(tmp_path):
    from lerobot.felix_jr.calibration.model import head_priors

    data = earlier_fit(tmp_path / "left")
    priors = head_priors(data)
    assert set(priors) == {
        "zero:head_pan_joint",
        "zero:head_tilt_joint",
        *{
            f"pos:{j}:{a}"
            for j in ("head_pan_joint", "head_tilt_joint", "fixed_head_camera_link")
            for a in "xyz"
        },
        *{f"rot:fixed_head_camera_link:{a}" for a in ("roll", "pitch", "yaw")},
    }  # no arm mount, arm joint or tip parameters: those belong to the arm
    assert priors["zero:head_tilt_joint"] == (0.58, data["sigma_post"]["zero:head_tilt_joint"])
    assert priors["rot:fixed_head_camera_link:roll"][1] == pytest.approx(np.radians(0.3))  # floored
    assert priors["pos:fixed_head_camera_link:y"][0] == 0.021


def test_head_from_refuses_a_fit_made_with_other_head_directions(robot, tmp_path):
    fake, client = robot
    earlier_fit(tmp_path / "left", signs={"head_neck": 1, "head_eyes": 1})
    with client.session(), pytest.raises(ValueError, match="directions changed"):
        CollectSession(
            client,
            "right",
            tmp_path / "right",
            lambda image: None,
            None,
            n_poses=2,
            head_from=tmp_path / "left",
        )


def test_the_hinge_axis_cosine_says_which_side_faces_the_camera():
    model = make_model()
    theta = {"zero:head_tilt_joint": 0.4}
    ticks = {"head_neck": 2048, "head_eyes": 2048}
    pose = {**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, "shoulder_pan.pos": 30.0, "gripper.pos": 1.0}
    for roll in (-90.0, -30.0, 0.0, 45.0, 120.0):
        p = {**pose, "wrist_roll.pos": roll}
        cosine = model.hinge_axis_cosine(p, ticks, theta)
        positive = model.hinge_axis_angle(p, ticks, theta, "positive")
        negative = model.hinge_axis_angle(p, ticks, theta, "negative")
        assert positive + negative == pytest.approx(180.0)
        assert model.hinge_axis_angle(p, ticks, theta) == pytest.approx(min(positive, negative))
        assert (cosine > 0) == (positive < 90)
    for side, sign in (("positive", 1), ("negative", -1)):
        roll, _ = best_roll(model, theta, pose, ticks, max_roll=130, side=side)
        assert sign * model.hinge_axis_cosine({**pose, "wrist_roll.pos": roll}, ticks, theta) > 0
