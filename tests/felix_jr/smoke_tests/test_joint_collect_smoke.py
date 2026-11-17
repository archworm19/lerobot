"""A joint-calibration session (`calibration.joint_collect`) against the fake robot, its cameras showing a board.

A true robot (the start's estimates moved by 30% of their priors, the wrist camera off on the gripper) renders the
ChArUco board on the table into the head's picture and the wrist camera's; the test drives the session as a person
would (`n`, the fingertips clicked where the truth shows them) and checks the sightings, the session file and the
fit: the calibrated model is nearer the truth than the start.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import joint_views as jv  # noqa: E402
from lerobot.felix_jr.calibration.boards import CharucoBoard  # noqa: E402
from lerobot.felix_jr.calibration.joint_collect import JointCollect  # noqa: E402
from lerobot.felix_jr.calibration.joint_fit import (  # noqa: E402
    HEAD,
    MOVING,
    MOVING_PINCH,
    Look,
    TipClick,
    load_session,
)
from lerobot.felix_jr.calibration.joint_sim import draw_truth  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

# the simulations start from (and draw their true robots around) this calibration, not `api.MODEL`: a new default
# calibration must not change what these tests check
SIM_START = Path(__file__).parents[1] / "reference" / "xle_calib_1008_wrist2.urdf"

ARM = "right"
BOARD = CharucoBoard(9, 7, 0.025, 0.018, "DICT_4X4_50")  # the one printed
PX_PER_M = 8000.0  # the printout's resolution (blurred before the warp: no aliasing)
MARGIN_PX = int(PX_PER_M * 0.01)  # the printout's white margin, 1 cm
HEAD_SIZE, HEAD_F = (640, 480), 610.0


def render_board(color_size, matrix, camera_from_board):
    """The board on gray, seen through a pinhole (`matrix`) at `camera_from_board`."""
    printed = cv2.GaussianBlur(BOARD.image(PX_PER_M, MARGIN_PX), (0, 0), PX_PER_M / 4000.0).astype(np.float32)
    offset = MARGIN_PX  # the printout's pixel of the board frame's origin (its outer top-left corner)
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
    start = XleRobotModel.load(SIM_START)
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
    # laid face up: a board's frame (OpenCV's) has z into the board, the plan's z up, which a camera above sees from
    # behind (mirrored: ArUco markers do not read). Turned half a turn about its x through its center, same footprint
    center = BOARD.points().mean(axis=0)
    face_up = np.diag([1.0, -1.0, -1.0, 1.0])
    face_up[:3, 3] = center - face_up[:3, :3] @ center
    root_board = root_board @ face_up
    root_board[:3, 3] += [0.02, -0.015, 0.0]  # placed roughly on the outline
    head_matrix = np.array([[HEAD_F, 0, 320.0], [0, HEAD_F, 240.0], [0, 0, 1.0]])

    def true_q(state):
        ticks = {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        return start.joint_vector({ARM: state[f"{ARM}_arm"]}, ticks)

    def head(state):
        camera = start.camera_pose(true_q(state), truth.theta)
        return render_board(HEAD_SIZE, head_matrix, np.linalg.inv(camera) @ root_board)

    def wrist(arm, state):
        camera = jv.wrist_pose(start, ARM, true_q(state), truth.theta)
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


@pytest.mark.parametrize("measured_map", [True, False], ids=["measured map", "no map"])
def test_a_session_collects_sightings_and_clicks_and_fits(setup, measured_map):
    """Also from a start with no gripper map (`measured_map` False): the session takes the servo's
    (`GripperMap.from_servo`) and the fit puts where the tips meet where the truth has them."""
    client, start, start_path, truth, true_q, tmp_path = setup
    if not measured_map:
        bare = XleRobotModel.load(start_path)
        bare.gripper_maps.pop(ARM)
        start_path = tmp_path / "start_no_map.urdf"
        bare.save(start_path)
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
            wrist_tip="click",  # (clicked here: the fake robot's wrist frames have no fingers for a sweep to find)
        )
        assert session.phase == "place" and shown, session.message

        def click_truth():
            """Click the fingertips where the true robot shows them: the wrist its own face, the head both tips on
            the face the session names (the jaws open: the moving one at the true opening)."""
            state = client.get_state()
            q = true_q(state)
            side = 0 if session.head_face == "pinch" else 1
            fixed = jv.fingertips(start, ARM, q, truth.theta, truth.thickness)
            uv_head = start.project(fixed[side][None], q, truth.theta)[0][0]
            uv_wrist = jv.wrist_pixels(start, ARM, q, fixed[1][None], truth.theta)[0]
            if jv.in_picture(uv_head[None], HEAD_SIZE):
                session.click("head:fixed", *uv_head)
            if jv.in_picture(uv_wrist[None], (640, 480)):
                session.click("wrist", *uv_wrist)
            pos = state[f"{ARM}_arm"]["gripper.pos"]
            opening = start.opening_deg(ARM, pos) + truth.opening_off_deg
            moving = jv.moving_fingertips(start, ARM, q, opening, truth.theta, truth.thickness)[side]
            uv_moving = start.project(moving[None], q, truth.theta)[0][0]
            if jv.in_picture(uv_moving[None], HEAD_SIZE):
                session.click("head:moving", *uv_moving)

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
        assert not session.board_warning  # the board stayed where it was placed
        while session.phase == "low":
            session.next()
            if session.phase == "low":
                click_truth()
        assert session.phase == "done"
        path = session.finish(holdout_every=0)
    saved = load_session(out)
    assert len(saved.sightings) == len(session.session.sightings) and len(saved.clicks) >= 6
    assert all((out / s.image).is_file() for s in saved.sightings)
    # the jaws opened at the views in turn (the start has the arm's gripper map), the opening read with each look
    openings = {round(c.look.arms[ARM]["gripper.pos"]) for c in saved.clicks}
    assert openings <= {round(pos) for pos in jv.VIEW_OPENINGS} and len(openings) >= 2, openings
    # the moving tip clicked by the head, on the face it sees: the wrist camera's side at the arm views (the head looks
    # down on it), the pinch face low
    moving = [c.on_face for c in saved.clicks if c.camera == HEAD and c.on_face in (MOVING, MOVING_PINCH)]
    assert len(moving) >= 4 and MOVING_PINCH in moving, moving
    calibrated = XleRobotModel.load(path)
    fitted, measured = calibrated.gripper_maps[ARM], start.gripper_maps[ARM]
    assert not fitted.default  # fitted, whichever it started from
    # the truth's tips meet where the start's measured map has them, moved by the true opening's offset
    true_closed = measured.pos_closed - truth.opening_off_deg / measured.deg_per_unit
    assert fitted.pos_closed == pytest.approx(true_closed, abs=0.4), (fitted.pos_closed, true_closed)
    assert (out / "calibration.md").is_file()
    # nearer the truth: the wrist camera's pose relative to the pinch frame, at a pose like the views'
    q = start.joint_vector({ARM: session.views[0]})

    def camera_in_pinch(model, theta):
        pinch = model.frame_pose(f"{ARM}_gripper_frame_link", q, theta)
        return np.linalg.inv(pinch) @ jv.wrist_pose(model, ARM, q, theta)

    true_rel = camera_in_pinch(start, truth.theta)
    before = camera_in_pinch(start, None)
    after = camera_in_pinch(calibrated, None)
    error_before = np.linalg.norm(before[:3, 3] - true_rel[:3, 3])
    error_after = np.linalg.norm(after[:3, 3] - true_rel[:3, 3])
    assert error_after < 0.002 and error_after < 0.5 * error_before, (
        error_before,
        error_after,
    )  # 9 mm -> 0.4 here
    assert {s.camera for s in saved.sightings} == {HEAD, ARM}


def test_by_default_the_sweep_comes_first_and_the_wrist_fingertip_is_not_clicked(setup):
    """The default (`wrist_tip="sweep"`): the session waits on the sweep (here declined, and the session told to go on
    without its tips); then a click in the wrist pane is declined, with a message, and the head's two are recorded,
    the moving tip on its own face."""
    client, start, start_path, truth, true_q, tmp_path = setup
    with client.session():
        session = JointCollect(
            client,
            ARM,
            tmp_path / "sweep_session",
            lambda _: None,
            start_path,
            BOARD,
            placements=1,
            arm_views=2,
        )
        assert session.wrist_tip == "sweep" and session.phase == "sweep" and session.sweep_pending
        session.next()
        assert session.phase == "sweep"  # `n` does not skip it
        asked = []

        def ask(prompt):
            asked.append(prompt)
            return "q" if "skip" in prompt else "y"  # the sweep skipped, then go on without its tips

        assert session.sweep_first(window=False, ask=ask)
        assert len(asked) == 2 and session.phase == "place" and not session.sweep_pending
        for _ in range(8):
            session.live_at = 0.0
            session.refresh()
            if session.live is not None:
                break
            session.nudge("down")
        session.next()  # the board found, then seen from around it
        session.next()  # the first arm view: a look to click in
        assert session.phase == "arm" and session.look is not None, session.message
        assert session.click("wrist", 320.0, 240.0) is None and "sweep" in session.message
        assert not session.session.clicks
        assert session.click("head:fixed", 320.0, 240.0).on_face == session.head_face
        assert session.opening() > 5.0  # the first view's jaws: open (20)
        moving_face = (
            MOVING_PINCH if session.head_face == "pinch" else MOVING
        )  # the head's face, the moving tip's
        assert session.click("head:moving", 330.0, 240.0).on_face == moving_face
        assert len(session.session.clicks) == 2


def test_a_fixed_fingertip_moved_since_the_start_stops_the_session_unless_told_to_go_on(setup):
    """The sweep's drift check (`sweep_first`): its fixed fingertip `SWEEP_DRIFT_PX` or less from the start's goes on
    unasked; farther, the person is asked, and no stops before the board."""
    client, start, start_path, truth, true_q, tmp_path = setup

    def measured(session, drift):
        def measure(window=True, ask=input):
            session.sweep_drift_px = drift
            look = Look({ARM: {}}, {})  # (never evaluated here)
            session.session.clicks.append(
                TipClick(ARM, ARM, look, np.zeros(2), "wrist_tip_sweep/frame_00.png", "wrist")
            )
            return 1  # the sweep's one tip

        return measure

    with client.session():
        # (a misread, 70 px as on the fake robot's board: its tips dropped, and asked to go on without them)
        for drift, answer, goes_on in (
            (2.0, None, True),
            (9.0, "n", False),
            (9.0, "y", True),
            (70.0, "y", True),
        ):
            session = JointCollect(
                client,
                ARM,
                tmp_path / f"drift_{drift}_{answer}",
                lambda _: None,
                start_path,
                BOARD,
                placements=1,
            )
            session.measure_wrist_tip = measured(session, drift)
            asked: list[str] = []

            def ask(prompt, asked=asked, answer=answer):
                asked.append(prompt)
                return answer

            assert session.sweep_first(ask=ask) is goes_on
            assert len(asked) == (answer is not None)
            assert session.phase == ("place" if goes_on else "sweep")
            assert len(session.session.clicks) == (drift < 30.0)  # a misread's tips dropped
