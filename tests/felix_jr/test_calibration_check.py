"""`calibration.check`: the views, the gripper openings and the drawn fingertips, against the fake robot."""

import json
import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.check import (  # noqa: E402
    MOVING_COLOR,
    PINCH_COLOR,
    CheckSession,
    annotate,
)
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    GripperMap,
    RobotMismatchError,
    XleRobotModel,
    arm_record,
)
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

# The fake robot: a 320x240 camera with fx = fy = 200, head servos at raw 2048.
FAKE_INTRINSICS = {"fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0, "width": 320, "height": 240}
SIGNS = {"head_neck": -1, "head_eyes": 1}
OPENINGS = (0.0, 30.0, 60.0)


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


def model_for(client, gripper: bool = True) -> XleRobotModel:
    """A calibration made for the fake robot: its head homes, looking down at the arms, the left gripper mapped
    (the tips meet at gripper.pos 0)."""
    refs = {name: servo["home"] for name, servo in client.head_state().items()}
    record = arm_record(client.arm_calibration()["left_arm"])
    maps = {"left": GripperMap(0.0, 1.27, record["gripper"], radius_m=0.085)} if gripper else {}
    model = XleRobotModel.from_urdf(
        head_signs=SIGNS,
        head_refs=refs,
        intrinsics=FAKE_INTRINSICS,
        gripper_maps=maps,
        arm_calibration={"left": record},
    )
    model.set_estimates({"zero:head_tilt_joint": np.radians(33.0)})
    return model


def make_session(client, model, **kwargs):
    shown: list[np.ndarray] = []
    session = CheckSession(
        client, model, "left", shown.append, openings=OPENINGS, n_targets=6, floor_z=0.5, **kwargs
    )
    return session, shown


def test_a_calibration_without_the_arms_gripper_map_or_for_another_robot_is_refused(robot):
    fake, client = robot
    with pytest.raises(ValueError, match="no gripper map for the left arm"):
        make_session(client, model_for(client, gripper=False))
    stale = model_for(client)
    stale.arm_calibration["left"]["elbow_flex"]["homing_offset"] += 40  # the arm recalibrated since
    with pytest.raises(RobotMismatchError, match="left arm: elbow_flex recalibrated"):
        make_session(client, stale)
    assert fake.arm_backend.trajectories == []  # nothing moved


def test_each_view_opens_the_gripper_to_the_next_opening_and_draws_both_tips(robot):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, model_for(client))
        assert len(shown) == 1
        apart = []
        for expected in (*OPENINGS, OPENINGS[0]):  # in turn, then around again
            session.handle("next")
            assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == expected
            opening, pinch, moving = session.predictions()
            assert opening == pytest.approx(1.27 * expected)
            assert session.model.in_view(pinch.uv, margin=30)  # the view centers the pinch point
            apart.append(float(np.linalg.norm(moving.point - pinch.point)))
        assert apart[0] == pytest.approx(0.0, abs=1e-9)  # closed: the tips meet
        # the moving tip turns on the circle through the pinch point (not the map's caliper radius)
        radius = session.model.jaw_radius("left")
        assert apart[1] == pytest.approx(2 * radius * np.sin(np.radians(1.27 * 30) / 2))
        assert apart[0] < apart[1] < apart[2]
        assert len(shown) == 5 and session.message.startswith("view 4:")


def test_the_gripper_opens_and_closes_by_a_step_at_the_same_view(robot):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, model_for(client), gripper_step=10.0)
        session.handle("next")  # closed
        arm = {k: v for k, v in fake.arm_backend.pos["left_arm"].items() if k != "gripper.pos"}
        session.handle("open")
        session.handle("open")
        assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == pytest.approx(20.0)
        session.handle("close")
        assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == pytest.approx(10.0)
        for _ in range(3):
            session.handle("close")
        assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == 0.0  # not below 0
        assert {k: v for k, v in fake.arm_backend.pos["left_arm"].items() if k != "gripper.pos"} == arm
        assert len(shown) == 8


def test_every_frame_is_saved_with_what_the_model_said(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, _ = make_session(client, model_for(client), out_dir=tmp_path / "check")
        session.handle("next")
        session.handle("next")
        session.handle("open")
    frames = json.loads((tmp_path / "check" / "frames.json").read_text())
    assert [f["index"] for f in frames] == [0, 1, 2, 3] and [f["view"] for f in frames] == [0, 1, 2, 2]
    second = frames[2]
    assert second["state"]["left_arm"]["gripper.pos"] == OPENINGS[1]
    assert second["opening_deg"] == pytest.approx(1.27 * OPENINGS[1])
    assert set(second["pinch"]) == {"uv", "z", "point", "visibility"} and len(second["moving_tip"]["uv"]) == 2
    for f in frames:
        assert (tmp_path / "check" / f["image"]).is_file()
        assert (tmp_path / "check" / f["image"].replace(".png", "_color.png")).is_file()


def has_color(image: np.ndarray, color: tuple, row: int, col: int, half: int = 3) -> bool:
    """Whether a pixel near (row, col) is close to `color` (circles are anti-aliased)."""
    window = image[row - half : row + half + 1, col - half : col + half + 1].astype(int)
    return bool((np.abs(window - np.array(color)).max(axis=2) < 60).any())


def test_annotate_draws_both_tips_and_skips_one_behind_the_camera():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    tips = [
        (np.array([40.0, 60.0]), PINCH_COLOR, "pinch"),
        (np.array([110.0, 60.0]), MOVING_COLOR, "moving tip"),
    ]
    out = annotate(image, ["hi"], tips)
    assert has_color(out, PINCH_COLOR, 60, 40 + 12) and has_color(out, MOVING_COLOR, 60, 110 - 12)
    assert has_color(out, (255, 255, 255), 60, 75, half=0)  # the line between them
    assert not image.any()  # a copy
    alone = annotate(image, [], [(None, PINCH_COLOR, "pinch"), tips[1]])
    assert not has_color(alone, (255, 255, 255), 60, 75, half=0)  # no line with one point
    assert not has_color(alone, PINCH_COLOR, 60, 40 + 12)


def test_the_wrist_camera_plates_are_drawn_where_the_calibration_has_them(robot):
    from lerobot.felix_jr.calibration.check import BOARD_COLOR
    from lerobot.felix_jr.calibration.xle_robot_model import WristBoard

    fake, client = robot
    with client.session():
        bare, _ = make_session(client, model_for(client))
        assert bare.board_outlines() == []  # none in the calibration: nothing drawn
        model = model_for(client)
        for arm in ("left", "right"):  # a 4 cm plate beside each gripper
            plate = WristBoard(
                f"{arm}_gripper_link",
                ((0.02, -0.03, 0.06), (0.02, -0.03, 0.02), (0.02, 0.01, 0.02)),
                (1.0, 0, 0),
            )
            model = model.with_wrist_board(arm, plate)
        session, shown = make_session(client, model)
        session.handle("next")
        outlines = session.board_outlines()
        q = session.current_q()
        uv, _ = session.model.project(session.model.board_corners("left", q), q)
        assert any(np.allclose(o, uv) for o in outlines)  # the checked arm's, at its measured joints
        assert len(outlines) == 2  # the other arm's too, at its measured joints
        assert session.model.in_view(uv[0], margin=5)
        corner = np.round(uv[0]).astype(int)
        assert has_color(shown[-1], BOARD_COLOR, corner[1], corner[0])  # drawn on the frame
