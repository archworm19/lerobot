"""`calibration.arm_shapes`: the parts' boxes from measurements, and sessions against the fake robot."""

import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.arm_shapes import (  # noqa: E402
    PARTS,
    ArmShapesSession,
    box_outline,
    measured_box,
    measures_of,
    parse_numbers,
)
from lerobot.felix_jr.calibration.xle_robot_model import ArmBox, XleRobotModel, arm_record  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


def test_a_gripper_body_runs_from_the_wrist_toward_the_tips(model):
    box = measured_box(model, "right", "gripper_body", [0.05, 0.07, 0.03])
    assert box.parent == "right_gripper_link" and not box.moving
    # length along the link's -z (toward the pinch point), from its origin (the wrist roll's axis); width across the
    # jaws (x), depth along the hinge (y); centered across
    np.testing.assert_allclose(box.size, [0.07, 0.03, 0.05])
    np.testing.assert_allclose(box.center, [0.0, 0.0, -0.025])
    q = model.joint_vector()
    toward_tips = (
        np.linalg.inv(model.frame_pose("right_gripper_link", q)) @ np.r_[model.pinch_point("right", q), 1.0]
    )
    assert toward_tips[2] < -0.05  # the tips are along -z
    shifted = measured_box(model, "right", "gripper_body", [0.05, 0.07, 0.03], [0.01, -0.005, 0.002])
    np.testing.assert_allclose(shifted.center, [-0.005, 0.002, -0.035])  # along the arrows: -z, +x, +y


def test_a_forearm_is_centered_between_the_elbow_and_the_wrist(model):
    box = measured_box(model, "left", "forearm", [0.18, 0.05, 0.045])
    assert box.parent == "left_lower_arm_link"
    np.testing.assert_allclose(box.size, [0.18, 0.045, 0.05])  # length along x, height along y, width along z
    q = model.joint_vector()
    lower = model.frame_pose("left_lower_arm_link", q)
    wrist = lower[:3, :3].T @ (model.frame_pose("left_wrist_link", q)[:3, 3] - lower[:3, 3])
    # between the joints, its top on their line (y = 0), hanging down (-y: up in the world, with the arm reaching)
    np.testing.assert_allclose(box.center, [wrist[0] / 2, -0.0225, 0.0], atol=1e-12)
    reach = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 40.0,
    }
    up = model.frame_pose("left_lower_arm_link", model.joint_vector({"left": reach}))[:3, 1]
    assert up[2] > 0.9
    lower_box = measured_box(model, "left", "forearm", [0.18, 0.05, 0.045], [0.0, 0.0, 0.01])
    assert lower_box.center[1] == pytest.approx(-0.0325)  # a positive height shift: further down


@pytest.mark.parametrize("part", list(PARTS))
def test_measures_come_back_from_a_box(model, part):
    sizes, shifts = [0.06, 0.04, 0.03], [0.01, -0.02, 0.005]
    back = measures_of(model, "right", part, measured_box(model, "right", part, sizes, shifts))
    np.testing.assert_allclose(back[0], sizes, atol=1e-12)
    np.testing.assert_allclose(back[1], shifts, atol=1e-12)


def test_parse_numbers():
    assert parse_numbers("7 3.5, 3") == [7.0, 3.5, 3.0]
    assert parse_numbers("7 3") is None and parse_numbers("a b c") is None
    assert parse_numbers("5 -2", 2) == [5.0, -2.0]


def test_a_box_outline_surrounds_its_center(model):
    import cv2

    box = measured_box(model, "right", "forearm", [0.15, 0.05, 0.05])
    q = model.joint_vector()
    frame = model.box_frame(box, "right", q)
    polygon = box_outline(model, q, frame, 0.5 * np.asarray(box.size))
    if polygon is None:  # behind the camera at the zero pose: nothing to draw
        return
    assert 4 <= len(polygon) <= 6  # a box seen from anywhere: 4 to 6 sides
    uv, _ = model.project(frame[:3, 3], q)
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    assert cv2.pointPolygonTest(contour, tuple(map(float, uv[0])), False) > 0


# ---- a session against the fake robot -------------------------------------------------------------------------


FAKE_INTRINSICS = {"fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0, "width": 320, "height": 240}
SIGNS = {"head_neck": -1, "head_eyes": 1}


@pytest.fixture
def robot(tmp_path):
    fake = FakeHardware(arm_time_scale=0.02, head_range_path=tmp_path / "range.json", session_timeout_s=5.0)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=5.0) as client:
        for name in SIGNS:
            client.head_record(name, "home")
        yield fake, client
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def calibration_for(client, path: Path) -> Path:
    """A calibration made for the fake robot (its head homes, looking down at the arms), saved at `path`."""
    refs = {name: servo["home"] for name, servo in client.head_state().items()}
    model = XleRobotModel.from_urdf(
        head_signs=SIGNS,
        head_refs=refs,
        intrinsics=FAKE_INTRINSICS,
        arm_calibration={"left": arm_record(client.arm_calibration()["left_arm"])},
    )
    model.set_estimates({"zero:head_tilt_joint": np.radians(33.0)})
    model.save(path)
    return path


def test_a_session_is_refused_for_another_robots_calibration(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    model = XleRobotModel.load(start)
    model.arm_calibration["left"]["elbow_flex"]["homing_offset"] += 40
    model.save(start)
    with pytest.raises(Exception, match="elbow_flex recalibrated"):
        ArmShapesSession(client, start, "left", tmp_path / "out", lambda image: None, lambda question: "q")
    assert fake.arm_backend.trajectories == []


def answers(replies: Iterable[str]):
    """An `ask` that gives `replies` in turn and records the questions."""
    it = iter(replies)
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return next(it)

    ask.asked = asked  # type: ignore[attr-defined]
    return ask


def test_a_session_measures_both_parts_and_writes_them(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    out = tmp_path / "out"
    out.mkdir()
    shown: list[np.ndarray] = []
    ask = answers(["7 6.5 3", "x", "m 0 0.5 0", "s 7 6 3", "y", "18 5 4.5", "y"])
    with client.session():
        session = ArmShapesSession(client, start, "left", out, shown.append, ask)
        path = session.run()
    assert path == out / "calibration.urdf"
    written = XleRobotModel.load(path).boxes["left"]
    model = XleRobotModel.load(start)
    assert written["gripper_body"] == measured_box(
        model, "left", "gripper_body", [0.07, 0.06, 0.03], [0, 0.005, 0]
    )
    assert written["forearm"] == measured_box(model, "left", "forearm", [0.18, 0.05, 0.045])
    assert "length (from the wrist toward the tips)" in ask.asked[0]  # type: ignore[attr-defined]
    report = (out / "arm_shapes.md").read_text()
    assert "| gripper_body | length 7.0, width 6.0, depth 3.0 | +0.0, +0.5, +0.0 |" in report
    assert "| forearm | length 18.0, width 5.0, height 4.5 |" in report
    assert (out / "left_gripper_body.png").is_file() and (out / "left_forearm.png").is_file()
    assert len(shown) >= 4 and shown[-1].shape == session.capture.color.shape
    XleRobotModel.load(path).check_robot(client)  # still this robot's calibration


def test_enter_keeps_the_calibrations_box_and_q_writes_nothing(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    body = ArmBox("left_gripper_link", (0.004, -0.002, -0.03), (0.065, 0.03, 0.05))
    XleRobotModel.load(start).with_boxes("left", {"gripper_body": body}).save(start)
    out = tmp_path / "out"
    out.mkdir()
    ask = answers(["", "y"])
    with client.session():
        path = ArmShapesSession(
            client, start, "left", out, lambda image: None, ask, parts=["gripper_body"]
        ).run()
    assert path is not None
    kept = XleRobotModel.load(path).boxes["left"]["gripper_body"]
    np.testing.assert_allclose(kept.center, body.center, atol=1e-12)
    np.testing.assert_allclose(kept.size, body.size, atol=1e-12)
    assert "[5.0 6.5 3.0]" in ask.asked[0]  # type: ignore[attr-defined]  # length, width, depth shown
    quit_out = tmp_path / "quit"
    quit_out.mkdir()
    with client.session():
        stopped = ArmShapesSession(
            client, start, "left", quit_out, lambda image: None, answers(["7 6 3", "q"])
        )
        assert stopped.run() is None
    assert not (quit_out / "calibration.urdf").exists()
    with pytest.raises(ValueError, match="parts must be"):
        ArmShapesSession(client, start, "left", out, lambda image: None, parts=["elbow"])


def test_another_view_moves_the_arm_and_the_head_turns(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    out = tmp_path / "out"
    out.mkdir()
    with client.session():
        session = ArmShapesSession(
            client,
            start,
            "left",
            out,
            lambda image: None,
            answers(["7 6 3", "v", "h 4 2", "y"]),
            parts=["gripper_body"],
        )
        head_before = {n: s["ticks"] for n, s in client.head_state().items()}
        assert session.run() is not None
        assert session.target_index >= 1  # a planned view was taken
        assert fake.arm_backend.trajectories  # the arm moved for it
        assert {n: s["ticks"] for n, s in client.head_state().items()} != head_before
