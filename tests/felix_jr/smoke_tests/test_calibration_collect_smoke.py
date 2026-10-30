"""Pose planning, head centering and the click-collection session (fake robot behind a real ZeroMQ socket)."""

import threading

import numpy as np
import pytest

from lerobot.felix_jr.calibration import collect as collect_module
from lerobot.felix_jr.calibration.bootstrap import Plane, up_from_pitch
from lerobot.felix_jr.calibration.collect import CollectSession
from lerobot.felix_jr.calibration.design import (
    center_head,
    plan_poses,
)
from lerobot.felix_jr.calibration.model import ARM_JOINTS, CalibrationModel, Observation, load_observations
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware
from lerobot.felix_jr.server.server import RobotServer

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 2048, "head_eyes": 2048}


def make_model():
    return CalibrationModel("left", INTRINSICS, SIGNS, REFS)


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


def test_the_session_collects_clicks_refits_and_saves_everything(robot, tmp_path):
    fake, client = robot
    theta_true = {"zero:head_tilt_joint": 0.12, "zero:head_pan_joint": -0.05}
    rng = np.random.default_rng(0)
    with client.session():
        session, shown = make_session(client, tmp_path, n_poses=14)
        assert len(shown) == 1 and shown[0].shape == (240, 320, 3)
        # the fake camera is 320x240, so the model's intrinsics come from the capture, not from INTRINSICS
        assert session.model.camera.width == 320
        for _ in range(14):
            session.next_pose()
            true_pred = session.model.predict(session.current_observation(), theta_true)
            if session.model.in_view(true_pred.uv, margin=8):
                session.click(*(true_pred.uv + rng.normal(0, 0.7, 2)))
        count = len(session.observations)
        assert count >= 8, f"only {count} of 14 poses had the true tip in view"
        assert session.last_rms is not None and session.last_rms < 6.0
        tilt = session.theta["zero:head_tilt_joint"]
        assert abs(tilt - 0.12) < np.radians(5)  # the head's unknown zero was found from the clicks
        assert len(shown) > count

        session.undo()
        assert len(session.observations) == count - 1
    # the folder: json + images, readable again, with measured joints and head ticks from the server
    model2, observations = load_observations(tmp_path / "cal" / "observations.json")
    assert len(observations) == count - 1
    assert model2.camera.width == 320 and model2.head_signs == SIGNS
    first = observations[0]
    assert first.image == "images/000.png" and (tmp_path / "cal" / first.image).is_file()
    assert (tmp_path / "cal" / "images" / "000_depth.png").is_file()
    assert set(first.arm) >= {f"{j}.pos" for j in ARM_JOINTS} and set(first.head_ticks) == {
        "head_neck",
        "head_eyes",
    }
    assert first.depth_m is None or 0.3 < first.depth_m < 1.5  # from the fake scene
    assert all(abs(o.arm["gripper.pos"] - 1.0) < 1e-6 for o in observations)  # closed in every planned pose
    assert fake.arm_backend.violations == 0
    # the session ended: the robot went to rest
    assert all(pose == fake.arm_backend.HOME for pose in fake.arm_backend.pos.values())


def test_resume_continues_an_earlier_folder(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, _ = make_session(client, tmp_path)
        for u, v in [(100, 100), (150, 120), (200, 140), (250, 160), (120, 180)]:
            session.click(u, v)
        assert len(session.observations) == 5 and session.last_rms is not None
        resumed, _ = make_session(client, tmp_path, resume=True)
        assert len(resumed.observations) == 5 and "resumed with 5" in resumed.message
        resumed.click(60, 60)
        assert len(resumed.observations) == 6
        _, observations = load_observations(tmp_path / "cal" / "observations.json")
        assert len(observations) == 6 and observations[-1].index == 5


def test_a_given_head_pitch_sets_the_tilt_prior_and_the_first_pose_shows_the_tip(robot, tmp_path):
    fake, client = robot
    with client.session():
        shown = []
        session = CollectSession(
            client, "left", tmp_path / "cal", shown.append, None, n_poses=8, head_tilt_deg=15.0
        )
        assert any("head tilt zero set" in note for note in session.notes)
        mean, sigma = session.model.prior_overrides["zero:head_tilt_joint"]
        assert session.theta["zero:head_tilt_joint"] == mean and sigma == pytest.approx(np.radians(5.0))
        assert len(session.poses) == 8 and session.poses[0]["gripper.pos"] == 1.0
        assert any(note.startswith("first pose:") for note in session.notes)
        ticks = {n: fake.head.backend.read_ticks(n) for n in ("head_neck", "head_eyes")}
        predicted = session.model.predict(
            Observation(arm=session.poses[0], head_ticks=ticks, pixel=(0, 0)), session.theta
        )
        assert np.linalg.norm(predicted.uv - [160, 120]) < 60  # the fake camera is 320x240
        session.click(100, 100)  # the prior travels with the saved clicks
        _, observations = load_observations(tmp_path / "cal" / "observations.json")
        import json

        saved = json.loads((tmp_path / "cal" / "observations.json").read_text())["settings"]
        assert "zero:head_tilt_joint" in saved["prior_overrides"]


def test_without_a_table_plane_the_session_says_so_and_still_plans_poses(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(client, "left", tmp_path / "cal", lambda image: None, None, n_poses=5)
        assert any("no table plane" in note for note in session.notes)  # the fake scene is a flat wall
        assert "zero:head_tilt_joint" not in session.model.prior_overrides and session.table_z is None
        assert len(session.poses) == 5


def test_a_known_table_keeps_every_planned_pose_above_it(robot, tmp_path, monkeypatch):
    fake, client = robot
    monkeypatch.setattr(
        collect_module, "estimate_up_vector", lambda scene: Plane(up_from_pitch(np.radians(15.0)), -0.5, 0.2)
    )
    with client.session():
        session = CollectSession(client, "left", tmp_path / "cal", lambda image: None, None, n_poses=10)
        assert session.table_z is not None
        assert any("below the arm base" in note for note in session.notes)
        floor = session.table_z + collect_module.TABLE_CLEARANCE
        assert all(session.model.tip_in_root(p, session.theta)[2] >= floor - 1e-9 for p in session.poses)


def test_every_planned_pose_can_be_brought_into_view_by_the_head_within_its_limits(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client, "left", tmp_path / "cal", lambda image: None, None, n_poses=12, head_tilt_deg=15.0
        )
        head = client.head_state()
        limits = collect_module.head_limits(head)
        ticks = {n: head[n]["ticks"] for n in ("head_neck", "head_eyes")}
        for pose in session.poses[1:]:  # the seed pose is checked at the head's current pose
            target = center_head(session.model, session.theta, pose, ticks, limits)
            assert target is not None
            tip = session.model.predict(Observation(arm=pose, head_ticks=target, pixel=(0, 0)), session.theta)
            assert session.model.in_view(tip.uv, margin=collect_module.VIEW_MARGIN - 1)
            assert all(limits[n][0] <= target[n] <= limits[n][1] for n in target)


def test_planned_poses_turn_the_gripper_so_the_camera_sees_the_pinch_point(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client,
            "left",
            tmp_path / "cal",
            lambda image: None,
            None,
            n_poses=10,
            head_tilt_deg=30.0,
            max_roll=100,
        )
        head = client.head_state()
        limits = collect_module.head_limits(head)
        ticks = {n: head[n]["ticks"] for n in ("head_neck", "head_eyes")}
        for pose in session.poses[1:]:
            target = center_head(session.model, session.theta, pose, ticks, limits)
            assert (
                session.model.hinge_axis_angle(pose, target, session.theta)
                <= collect_module.FACING_MAX_DEG + 1e-6
            )
            assert abs(pose["wrist_roll.pos"]) <= 100
        seed_angle = session.model.hinge_axis_angle(session.poses[0], ticks, session.theta)
        unrolled = session.model.hinge_axis_angle(
            {**session.poses[0], "wrist_roll.pos": 0.0}, ticks, session.theta
        )
        assert seed_angle <= unrolled  # the first pose is rolled towards the camera too


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


def test_the_right_arm_session_starts_from_the_left_arms_head(robot, tmp_path):
    fake, client = robot
    earlier_fit(tmp_path / "left")
    with client.session():
        session = CollectSession(
            client,
            "right",
            tmp_path / "right",
            lambda image: None,
            None,
            n_poses=4,
            head_from=tmp_path / "left",
        )
        assert session.model.arm == "right"
        assert session.model.head_refs == {
            "head_neck": 1900,
            "head_eyes": 2000,
        }  # the left session's references
        assert session.theta["zero:head_tilt_joint"] == 0.58
        assert session.model.prior_overrides["pos:fixed_head_camera_link:y"][0] == 0.021
        assert not any(name.startswith("so101:") for name in session.model.prior_overrides)
        assert any("head and camera from" in note for note in session.notes)
        assert not any("no recorded home" in note for note in session.notes)  # the references came with it
        session.click(100, 100)
    import json

    saved = json.loads((tmp_path / "right" / "observations.json").read_text())["settings"]
    assert saved["arm"] == "right" and saved["prior_overrides"]["zero:head_tilt_joint"][0] == 0.58


@pytest.mark.parametrize("side,sign", [("positive", 1), ("negative", -1)])
def test_a_chosen_side_is_the_one_every_planned_pose_shows(robot, tmp_path, side, sign):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client,
            "right",
            tmp_path / "cal",
            lambda image: None,
            None,
            n_poses=8,
            head_tilt_deg=30.0,
            side=side,
        )
        head = client.head_state()
        limits = collect_module.head_limits(head)
        ticks = {n: head[n]["ticks"] for n in ("head_neck", "head_eyes")}
        for pose in session.poses[1:]:
            target = center_head(session.model, session.theta, pose, ticks, limits)
            assert sign * session.model.hinge_axis_cosine(pose, target, session.theta) > 0
        assert session.side == side


def test_with_either_side_the_side_is_learned_from_the_clicks_and_the_rest_replanned(robot, tmp_path):
    fake, client = robot
    with client.session():
        session = CollectSession(
            client,
            "left",
            tmp_path / "cal",
            lambda image: None,
            None,
            n_poses=40,
            head_tilt_deg=30.0,
            side="either",
        )
        clicked = 0
        while clicked < collect_module.SIDE_MIN_CLICKS and session.pose_index < len(session.poses):
            session.next_pose()
            obs = session.current_observation()
            if (
                session.model.hinge_axis_cosine(obs.arm, obs.head_ticks, session.theta) < 0
            ):  # only these show it
                session.click(100, 100)
                clicked += 1
        assert clicked == collect_module.SIDE_MIN_CLICKS
        assert session.side == "negative" and any("negative side" in note for note in session.notes)
        head = client.head_state()
        limits = collect_module.head_limits(head)
        ticks = {n: head[n]["ticks"] for n in ("head_neck", "head_eyes")}
        for pose in session.poses[session.pose_index :]:
            target = center_head(session.model, session.theta, pose, ticks, limits)
            assert session.model.hinge_axis_cosine(pose, target, session.theta) < 0
    with client.session():  # resuming learns it again from the saved clicks
        resumed = CollectSession(
            client, "left", tmp_path / "cal", lambda image: None, None, n_poses=4, resume=True, side="either"
        )
        assert resumed.side == "negative"
