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
    _pose,
    _with_pose,
    board_under,
    render_board,
)  # noqa: E402

from lerobot.felix_jr.calibration.wrist_camera import (  # noqa: E402
    POSE_VIEWS,
    WristCameraSession,
    lens_coverage_missing,
    refit,
    write,
)
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
    """The fake robot, its right wrist camera rendering the board. `held["camera_from_board"]` set: the person holds
    the board there in front of the camera; None: it lies on the table (`root_from_board`)."""
    root_from_board = board_under(model, "right")
    truth = model.with_wrist_camera("right", _with_pose(TRUE_CAMERA))
    held: dict[str, np.ndarray | None] = {"camera_from_board": None}

    def render(arm: str, state: dict) -> np.ndarray:
        camera_from_board = held["camera_from_board"]
        if camera_from_board is None:
            q = truth.joint_vector({arm: state[f"{arm}_arm"]})
            camera_from_board = np.linalg.inv(truth.wrist_camera_pose(arm, q)) @ root_from_board
        return render_board(TRUE_CAMERA, camera_from_board)

    fake = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json", wrist_renderer=render)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=10.0) as client:
        yield fake, client, held
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def test_a_session_calibrates_the_lens_and_the_pose_by_keys(robot, model, tmp_path, monkeypatch):
    fake, client, held = robot
    start = model
    start.arm_calibration["right"] = arm_record(client.arm_calibration()["right_arm"])
    out = tmp_path / "out"
    out.mkdir()
    shown: list[np.ndarray] = []
    with client.session():
        session = WristCameraSession(client, start, "right", BOARD, out, shown.append, step_m=0.01)
        left = WristCameraSession(client, start, "left", BOARD, out, shown.append)
        assert (
            left.go_to_start() and session.go_to_start()
        )  # both arms out of the way first, as the script does
        moved = len(fake.arm_backend.trajectories)
        assert session.start() and session.corners is not None  # the board under the start pose
        assert len(fake.arm_backend.trajectories) - moved <= 1  # already there: at most a tiny move
        assert session.phase == "focus" and session.best_sharpness > 0
        session.handle("take")  # nothing is kept while focusing
        assert session.record.intrinsic_views == []
        session.handle("accept")  # Enter: the lens as it is
        assert session.phase == "intrinsics" and session.record.focus_sharpness is not None
        # intrinsics: the person holds the board in front of the camera, tilted, turned and moved about; each new
        # pose found is kept by itself
        rng = np.random.default_rng(3)
        for _ in range(150):
            turn = rng.uniform([-0.6, -0.6, -0.8], [0.6, 0.6, 0.8])
            center = rng.uniform([-0.06, -0.04, 0.22], [0.06, 0.04, 0.38])
            pose = _pose(turn, center)
            pose[:3, 3] -= pose[:3, :3] @ [4 * BOARD.square_m, 2.5 * BOARD.square_m, 0.0]  # centered there
            held["camera_from_board"] = pose
            session.refresh()
            if len(session.record.intrinsic_views) >= 12 and not lens_coverage_missing(session.signatures):
                break
        held["camera_from_board"] = None  # put down flat on the table
        assert len(session.record.intrinsic_views) >= 12
        session.handle("calibrate")
        assert session.phase == "extrinsics" and session.camera is not None
        assert session.camera.fx == pytest.approx(400.0, rel=0.02)
        session.handle("next")  # before the board is down: nothing moves
        assert session.record.extrinsic_views == [] and "once the board lies" in session.message
        session.handle("calibrate")  # the board is down
        for _ in POSE_VIEWS:  # n: the suggested views, each taken where the board is found
            session.handle("next")
            if session.can_fit_extrinsics() and len(session.record.extrinsic_views) >= 10:
                break
        assert session.can_fit_extrinsics(), (len(session.record.extrinsic_views), session.spread())
        assert session.next_view <= len(POSE_VIEWS)  # the suggested views were enough
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
    # --resume keeps the saved views
    resumed = WristCameraSession(client, start, "right", BOARD, out, shown.append, resume=True)
    assert len(resumed.record.intrinsic_views) == len(session.record.intrinsic_views)
    assert len(resumed.signatures) == len(resumed.record.intrinsic_views) and "resumed" in resumed.message
    # the saved views refit with no robot to the same camera
    start_path = tmp_path / "start.urdf"
    start.save(start_path)
    again = XleRobotModel.load(refit(start_path, out)).wrist_cameras["right"]
    assert again.extrinsics_views == written.extrinsics_views
    np.testing.assert_allclose(
        np.array(again.gripper_from_camera), np.array(written.gripper_from_camera), atol=1e-6
    )
