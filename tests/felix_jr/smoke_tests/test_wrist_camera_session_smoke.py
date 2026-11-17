"""`calibration.wrist_camera` end to end: a session against the fake robot whose wrist camera renders the ChArUco board
a person holds in front of it (the truth is known), driven by keys: the focus, the lens's views kept by themselves,
the lens fitted, the calibration written with the camera's pose kept, resumed and refitted."""

import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from test_wrist_camera import (  # noqa: E402
    BOARD,
    GRIPPER_FROM_CAMERA,
    TRUE_CAMERA,
    URDF,
    held,
    render_board,
)

from lerobot.felix_jr.calibration.wrist_camera import (  # noqa: E402
    MIN_INTRINSIC_VIEWS,
    WristCameraSession,
    lens_coverage_missing,
    refit,
    write,
)
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_record  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

TABLE = np.full((480, 640, 3), 90, np.uint8)


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


@pytest.fixture
def robot(tmp_path):
    """The fake robot, its right wrist camera rendering the board where `holding["camera_from_board"]` puts it (None:
    the board out of view, the table only)."""
    holding: dict[str, np.ndarray | None] = {"camera_from_board": None}

    def render(arm: str, state: dict) -> np.ndarray:
        pose = holding["camera_from_board"]
        return TABLE.copy() if pose is None else render_board(TRUE_CAMERA, pose)

    fake = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json", wrist_renderer=render)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=10.0) as client:
        yield fake, client, holding
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def test_a_session_calibrates_the_lens_by_keys_and_keeps_the_cameras_pose(robot, model, tmp_path):
    fake, client, holding = robot
    start = model.with_wrist_camera(
        "right", TRUE_CAMERA, GRIPPER_FROM_CAMERA
    )  # a pose on the gripper (a joint fit's)
    start.arm_calibration["right"] = arm_record(client.arm_calibration()["right_arm"])
    out = tmp_path / "out"
    out.mkdir()
    shown: list[np.ndarray] = []
    said: list[str] = []
    with client.session():
        session = WristCameraSession(client, start, "right", BOARD, out, shown.append, say=said.append)
        holding["camera_from_board"] = held([0.0, 0.0, 0.0], [0.0, 0.0, 0.25])
        assert session.start() and session.phase == "focus" and session.best_sharpness > 0
        assert said and said[0].startswith("right FOCUS")
        session.handle("take")  # nothing is kept while focusing
        assert session.record.intrinsic_views == []
        session.handle("accept")  # Enter: the lens as it is
        assert session.phase == "intrinsics" and session.record.focus_sharpness is not None
        assert said[-1].startswith("right LENS")
        # the person holds the board still at a pose (a few frames), then another: varied, some off the edge
        rng = np.random.default_rng(3)
        for _ in range(60):
            pose = held(
                rng.uniform([-0.6, -0.6, -0.8], [0.6, 0.6, 0.8]),
                rng.uniform([-0.14, -0.1, 0.24], [0.14, 0.1, 0.42]),
            )
            holding["camera_from_board"] = pose
            for _ in range(3):
                session.refresh()
            n = len(session.record.intrinsic_views)
            if n >= 2 * MIN_INTRINSIC_VIEWS and not lens_coverage_missing(session.views.signatures):
                break
        n = len(session.record.intrinsic_views)
        assert MIN_INTRINSIC_VIEWS <= n <= 60  # one per pose at most
        assert any(len(v) < 48 for v in session.record.intrinsic_views)  # partial boards kept too
        holding["camera_from_board"] = None
        session.refresh()
        assert "NOT found" in session.seen
        session.handle("undo")
        assert len(session.record.intrinsic_views) == n - 1
        session.handle("calibrate")
        assert session.done and session.camera is not None, session.message
        assert session.camera.fx == pytest.approx(400.0, rel=0.02)
    path = write(start, [session.record], out)
    written = XleRobotModel.load(path).wrist_cameras["right"]
    assert written.fx == pytest.approx(400.0, rel=0.02)
    np.testing.assert_allclose(
        XleRobotModel.load(path).gripper_from_camera("right"), GRIPPER_FROM_CAMERA, atol=1e-12
    )
    report = (out / "wrist_cameras.md").read_text()
    assert "corrections to its design (kept)" in report and "corners each" in report
    assert (out / "right_views.json").is_file() and any((out / "images").iterdir())
    assert shown and shown[-1].shape == (480, 640, 3)
    # --resume keeps the saved views
    resumed = WristCameraSession(client, start, "right", BOARD, out, shown.append, resume=True)
    assert len(resumed.record.intrinsic_views) == n - 1 and "resumed" in resumed.message
    assert resumed.views is not None and len(resumed.views.signatures) == n - 1
    # the saved views refit with no robot to the same lens
    start_path = tmp_path / "start.urdf"
    start.save(start_path)
    again = XleRobotModel.load(refit(start_path, out)).wrist_cameras["right"]
    assert again.fx == pytest.approx(written.fx, rel=1e-6) and again.coeffs == pytest.approx(written.coeffs)


def test_a_skip_takes_a_second_x(robot, model, tmp_path):
    """calib1008_full3 (2026-11-08): the left camera skipped by mistake (x read as "on to the right camera")."""
    _, client, _ = robot
    start = model
    start.arm_calibration["right"] = arm_record(client.arm_calibration()["right_arm"])
    out = tmp_path / "out"
    out.mkdir()
    with client.session():
        session = WristCameraSession(
            client,
            start,
            "right",
            BOARD,
            out,
            lambda image: None,
            focus=False,
            next_arm="left",
            say=lambda s: None,
        )
        session.handle("skip")  # one x only asks
        assert not session.done and "x again" in session.message and "next the left" in session.message
        session.handle("undo")  # anything else in between: the x is asked for again
        session.handle("skip")
        assert not session.done
        session.handle("skip")
        assert session.done and session.skipped
        session.handle("calibrate")  # (nothing to fit: done already)
        last = WristCameraSession(client, start, "right", BOARD, out, lambda image: None, focus=False)
        last.handle("skip")
        assert "then the calibration is written" in last.message
