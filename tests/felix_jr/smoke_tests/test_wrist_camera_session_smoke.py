"""`calibration.wrist_camera` end to end: a session against the fake robot whose wrist camera renders a checkerboard
on the table from the arm's pose (the truth is known), driven by keys: the lens, the board put down, the views, the
camera's pose on the gripper, the calibration written."""

import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from test_wrist_camera import (
    BOARD,
    GRIPPER_FROM_CAMERA,
    TRUE_CAMERA,
    URDF,
    _with_pose,
    board_under,
    render_board,
)  # noqa: E402

from lerobot.felix_jr.calibration.wrist_camera import WristCameraSession, refit, write  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_record  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


# ---- a session against the fake robot -------------------------------------------------------------------------


@pytest.fixture
def robot(tmp_path, model):
    root_from_board = board_under(model, "right")
    truth = model.with_wrist_camera("right", _with_pose(TRUE_CAMERA))

    def render(arm: str, state: dict) -> np.ndarray:
        q = truth.joint_vector({arm: state[f"{arm}_arm"]})
        camera_from_board = np.linalg.inv(truth.wrist_camera_pose(arm, q)) @ root_from_board
        return render_board(TRUE_CAMERA, camera_from_board)

    fake = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json", wrist_renderer=render)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=10.0) as client:
        yield fake, client, root_from_board
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


# key moves for the session test, each undone after (UNDO), so the board stays in view
MOVES = [  # each 2-4 cm or 10-20 deg from the start: a new view of the board
    ("left", "left"),
    ("right", "right"),
    ("up", "up"),
    ("down", "down"),
    ("up_z", "up_z"),
    ("pitch_up",),
    ("pitch_up", "pitch_up"),
    ("left", "left", "pitch_up"),
    ("right", "right", "pitch_up"),
    ("up", "up", "roll_left"),
    ("down", "down", "roll_right"),
    ("left", "up", "pitch_up", "roll_left"),
    ("right", "down", "pitch_up", "roll_right"),
    ("up_z", "up_z", "left", "left"),
    ("up_z", "up_z", "right", "right"),
    ("pitch_up", "roll_left", "roll_left"),
    ("pitch_up", "roll_right", "roll_right"),
    ("down_z", "pitch_up", "up"),
    ("roll_left", "roll_left", "roll_left"),
    ("roll_right", "roll_right", "roll_right"),
    ("pitch_up", "pitch_up", "roll_left", "roll_left", "roll_left"),
    ("pitch_up", "pitch_up", "roll_right", "roll_right", "roll_right"),
]
UNDO = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
    "up_z": "down_z",
    "down_z": "up_z",
    "pitch_up": "pitch_down",
    "pitch_down": "pitch_up",
    "roll_left": "roll_right",
    "roll_right": "roll_left",
}


def test_a_session_calibrates_the_lens_and_the_pose_by_keys(robot, model, tmp_path, monkeypatch):
    fake, client, root_from_board = robot
    # the arm alone over a fixed board gives fewer distinct views than a board moved by hand
    monkeypatch.setattr("lerobot.felix_jr.calibration.wrist_camera.MIN_INTRINSIC_VIEWS", 8)
    monkeypatch.setattr(
        "lerobot.felix_jr.calibration.wrist_camera.MIN_SPREAD_DEG", 10.0
    )  # its keys tilt one way
    start = model
    start.arm_calibration["right"] = arm_record(client.arm_calibration()["right_arm"])
    out = tmp_path / "out"
    out.mkdir()
    shown: list[np.ndarray] = []
    with client.session():
        session = WristCameraSession(client, start, "right", BOARD, out, shown.append, step_m=0.01)
        assert session.start() and session.corners is not None  # the board under the start pose
        # intrinsics: the board found in new poses is kept by itself; each move is undone before the next, so the
        # board stays in view (on the robot the person moves the board instead)
        for move in MOVES:
            for action in move:
                session.handle(action)
            for action in reversed(move):
                session.handle(UNDO[action])
            if len(session.record.intrinsic_views) >= 8:
                break
        assert len(session.record.intrinsic_views) >= 8
        session.handle("calibrate")
        assert session.phase == "extrinsics" and session.camera is not None
        assert session.camera.fx == pytest.approx(400.0, rel=0.03)
        session.handle("calibrate")  # the board is down
        session.handle("take")
        for move in MOVES:
            for action in move:
                session.handle(action)
            session.handle("take")
            for action in reversed(move):
                session.handle(UNDO[action])
            if session.can_fit_extrinsics():
                break
        assert session.can_fit_extrinsics(), (len(session.record.extrinsic_views), session.spread())
        session.handle("calibrate")
        assert session.phase == "check" and session.camera.gripper_from_camera is not None
        session.handle("calibrate")
        assert session.done
    fitted = np.array(session.camera.gripper_from_camera)
    np.testing.assert_allclose(fitted[:3, 3], GRIPPER_FROM_CAMERA[:3, 3], atol=0.004)
    path = write(start, [session.record], out)
    written = XleRobotModel.load(path).wrist_cameras["right"]
    assert written.gripper_from_camera is not None and (out / "wrist_cameras.md").is_file()
    assert (out / "right_views.json").is_file() and any((out / "images").iterdir())
    assert shown and shown[-1].shape == (480, 640, 3)
    # the saved views refit with no robot to the same camera
    start_path = tmp_path / "start.urdf"
    start.save(start_path)
    again = XleRobotModel.load(refit(start_path, out)).wrist_cameras["right"]
    assert again.extrinsics_views == written.extrinsics_views
    np.testing.assert_allclose(
        np.array(again.gripper_from_camera), np.array(written.gripper_from_camera), atol=1e-6
    )
