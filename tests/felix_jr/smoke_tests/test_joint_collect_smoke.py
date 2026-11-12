"""A joint-calibration session (`calibration.joint_collect`) against the fake robot, its cameras showing a board.

A true robot (the start's estimates moved by 30% of their priors, the wrist camera off on the gripper) renders a
checkerboard on the table into the head's picture and the wrist camera's; the test drives the session as a person
would (`n`, the fingertips clicked where the truth shows them) and checks the sightings, the session file and the
fit: the calibrated model is nearer the truth than the start.
"""

import threading

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import joint_views as jv  # noqa: E402
from lerobot.felix_jr.calibration.boards import Checkerboard  # noqa: E402
from lerobot.felix_jr.calibration.joint_collect import JointCollect  # noqa: E402
from lerobot.felix_jr.calibration.joint_fit import HEAD, load_session  # noqa: E402
from lerobot.felix_jr.calibration.joint_sim import draw_truth  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

ARM = "right"
BOARD = Checkerboard(8, 6, 0.025)
PX_PER_M = 8000.0  # the printout's resolution (blurred before the warp: no aliasing)
HEAD_SIZE, HEAD_F = (640, 480), 610.0


def render_board(color_size, matrix, camera_from_board):
    """The board on gray, seen through a pinhole (`matrix`) at `camera_from_board`."""
    printed = cv2.GaussianBlur(BOARD.image(PX_PER_M), (0, 0), PX_PER_M / 4000.0).astype(np.float32)
    offset = PX_PER_M * BOARD.square_m  # the printout's pixel of the board frame's origin (id 0)
    printout_from_board = np.array([[PX_PER_M, 0, offset], [0, PX_PER_M, offset], [0, 0, 1.0]])
    pose = camera_from_board
    if pose[2, 3] <= 0:
        return np.full((color_size[1], color_size[0], 3), 128, np.uint8)
    homography = (
        matrix @ np.column_stack([pose[:3, 0], pose[:3, 1], pose[:3, 3]]) @ np.linalg.inv(printout_from_board)
    )
    gray = cv2.warpPerspective(printed, homography, color_size, flags=cv2.INTER_LINEAR, borderValue=128)
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


@pytest.fixture
def setup(tmp_path):
    start = XleRobotModel.load(api.MODEL)
    lens = start.wrist_cameras[ARM]
    from dataclasses import replace

    start = start.with_wrist_camera(ARM, replace(lens, coeffs=tuple(0.0 for _ in lens.coeffs)))  # a pinhole
    start.intrinsics = {"fx": HEAD_F, "fy": HEAD_F, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
    start_path = tmp_path / "start.urdf"
    start.save(start_path)
    rng = np.random.default_rng(5)
    truth = draw_truth(start, rng, scale=0.3)
    base = start.frame_pose(f"{ARM}_base_link", start.joint_vector())[:3, 3]
    table_z = base[2] - 0.06
    root_board = jv.suggest_placements(start, ARM, BOARD, 1, table_z)[0]
    root_board[:3, 3] += [0.02, -0.015, 0.0]  # placed roughly on the outline
    head_matrix = np.array([[HEAD_F, 0, 320.0], [0, HEAD_F, 240.0], [0, 0, 1.0]])

    def true_q(state):
        ticks = {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        return start.joint_vector({ARM: state[f"{ARM}_arm"]}, ticks)

    def head(state):
        camera = start.camera_pose(true_q(state), truth.theta)
        return render_board(HEAD_SIZE, head_matrix, np.linalg.inv(camera) @ root_board)

    def wrist(arm, state):
        camera = jv.wrist_pose(start, ARM, true_q(state), truth.theta, truth.gripper_from_camera)
        lens = start.wrist_cameras[ARM]
        return render_board((lens.width, lens.height), lens.matrix(), np.linalg.inv(camera) @ root_board)

    fake = FakeHardware(
        arm_time_scale=0.01, head_range_path=tmp_path / "range.json", wrist_renderer=wrist, head_renderer=head
    )
    fake.WIDTH, fake.HEIGHT, fake.FX = 640, 480, HEAD_F
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=20.0) as client:
        yield client, start, start_path, truth, true_q, tmp_path
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def test_a_session_collects_sightings_and_clicks_and_fits(setup):
    client, start, start_path, truth, true_q, tmp_path = setup
    shown = []
    out = tmp_path / "session"
    with client.session():
        session = JointCollect(
            client,
            ARM,
            out,
            shown.append,
            start_path,
            BOARD,
            placements=1,
            arm_views=5,
            head_views=3,
            low_views=2,
        )
        assert session.phase == "place" and shown, session.message

        def click_truth():
            """Click the fingertips where the true robot shows them (each camera its own face)."""
            state = client.get_state()
            q = true_q(state)
            head_tip, wrist_tip = jv.fingertips(start, ARM, q, truth.theta, truth.thickness)
            uv_head = start.project(head_tip[None], q, truth.theta)[0][0]
            uv_wrist = jv.wrist_pixels(
                start, ARM, q, wrist_tip[None], truth.theta, truth.gripper_from_camera
            )[0]
            if jv.in_picture(uv_head[None], HEAD_SIZE):
                session.click("head", *uv_head)
            if jv.in_picture(uv_wrist[None], (640, 480)):
                session.click("wrist", *uv_wrist)

        # the live view while placing the board; the arrows if it is cut off, as a person would
        for _ in range(8):
            session.live_at = 0.0
            session.refresh()
            if session.live is not None:
                break
            session.nudge("down")
        assert session.live is not None and not session.session.sightings, session.message
        session.next()  # the board found by the head, then seen from around it
        assert session.phase == "arm", session.message
        head_sightings = sum(s.camera == HEAD for s in session.session.sightings)
        assert head_sightings >= 2, session.message
        while session.phase == "arm":
            session.next()
            if session.phase == "arm":
                click_truth()
        assert sum(s.camera == ARM for s in session.session.sightings) >= 3, session.message
        while session.phase == "low":
            session.next()
            if session.phase == "low":
                click_truth()
        assert session.phase == "done"
        path = session.finish(holdout_every=0)
    saved = load_session(out)
    assert len(saved.sightings) == len(session.session.sightings) and len(saved.clicks) >= 6
    assert all((out / s.image).is_file() for s in saved.sightings)
    calibrated = XleRobotModel.load(path)
    assert (out / "calibration.md").is_file()
    # nearer the truth: the wrist camera's pose relative to the pinch frame, at a pose like the views'
    q = start.joint_vector({ARM: session.views[0]})

    def camera_in_pinch(model, theta, gc):
        pinch = model.frame_pose(f"{ARM}_gripper_frame_link", q, theta)
        return np.linalg.inv(pinch) @ jv.wrist_pose(model, ARM, q, theta, gc)

    true_rel = camera_in_pinch(start, truth.theta, truth.gripper_from_camera)
    before = camera_in_pinch(start, None, None)
    after = camera_in_pinch(calibrated, None, None)
    error_before = np.linalg.norm(before[:3, 3] - true_rel[:3, 3])
    error_after = np.linalg.norm(after[:3, 3] - true_rel[:3, 3])
    assert error_after < 0.002 and error_after < 0.5 * error_before, (
        error_before,
        error_after,
    )  # 9 mm -> 0.4 here
    assert {s.camera for s in saved.sightings} == {HEAD, ARM}
