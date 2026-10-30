"""The click-collection session against a fake robot behind a real ZeroMQ socket."""

import threading
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import collect as collect_module  # noqa: E402
from lerobot.felix_jr.calibration.collect import CollectSession, decode_key, head_limits, parse_seed  # noqa: E402
from lerobot.felix_jr.calibration.xle_fit import Session, load_session, save_session  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

# The fake robot: a 320x240 camera with fx = fy = 200, head servos at raw 2048 with no recorded home.
FAKE_INTRINSICS = {"fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0, "width": 320, "height": 240}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 2048, "head_eyes": 2048}


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


@pytest.fixture
def calibration(tmp_path):
    """A saved calibration whose head looks down at the arms, as the real one does."""
    model = XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=FAKE_INTRINSICS)
    model.set_estimates({"zero:head_tilt_joint": np.radians(33.0)})
    path = tmp_path / "start.urdf"
    model.save(path)
    return path


def make_session(client, tmp_path, **kwargs):
    shown = []
    session = CollectSession(client, "left", tmp_path / "cal", shown.append, n_targets=6, **kwargs)
    return session, shown


def test_decode_key_and_parse_seed():
    assert decode_key(63232) == "up" and decode_key(65364) == "down" and decode_key(63234) == "left"
    assert decode_key(ord("j")) == "left" and decode_key(ord("l")) == "right"
    assert decode_key(ord("n")) == "next" and decode_key(ord("u")) == "undo" and decode_key(27) == "quit"
    assert decode_key(ord("z")) is None
    assert parse_seed("shoulder_pan=10, elbow_flex=-5") == {"shoulder_pan.pos": 10.0, "elbow_flex.pos": -5.0}
    assert parse_seed(None) is None


def test_head_limits_use_recorded_limits_or_a_range_around_the_present_position():
    head = {
        "head_neck": {"ticks": 2000, "limit_pos": 2600, "limit_neg": 1500, "home": 2000, "sign": -1},
        "head_eyes": {"ticks": 100, "limit_pos": None, "limit_neg": None, "home": None, "sign": 1},
    }
    limits = head_limits(head)
    assert limits["head_neck"] == (1500, 2600)
    assert limits["head_eyes"] == (0, 100 + collect_module.NECK_RANGE_TICKS)  # clipped at the encoder's 0


def test_a_fresh_session_builds_the_model_from_the_head_and_keeps_above_the_arm_base(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, tmp_path)
    assert session.model.head_refs == REFS and session.model.intrinsics == FAKE_INTRINSICS
    pan = next(p for p in session.model.parameters if p.name == "zero:head_pan_joint")
    assert pan.sigma == collect_module.NECK_PRIOR_NO_HOME  # the fake neck has no recorded home
    base_z = session.model.frame_pose("left_base_link", session.model.joint_vector())[2, 3]
    assert session.model.floor_z == pytest.approx(base_z)
    assert any("no floor height" in note for note in session.notes)
    assert len(session.targets) == 6 and len(shown) == 1 and shown[0].shape == (240, 320, 3)


def test_next_moves_the_arm_closed_and_the_head_to_the_view(robot, tmp_path, calibration):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, tmp_path, start_from=calibration, floor_z=0.5)
        session.handle("next")
        assert fake.arm_backend.pos["left_arm"]["gripper.pos"] == 1.0
        assert fake.arm_backend.trajectories[-1][1:] == (30.0, 60.0)  # slow
        assert set(fake.arm_backend.trajectories[-1][0]) == {"left_arm"}
        # the pinch point is where the view put it: in the middle of the picture
        predicted = session.model.predict("left", session.current_q())
        assert session.model.in_view(predicted.uv, margin=30)
        assert "view 1" in session.message and len(shown) == 2


def test_clicks_are_saved_refit_and_undone(robot, tmp_path, calibration):
    fake, client = robot
    with client.session():
        session, _ = make_session(client, tmp_path, start_from=calibration, floor_z=0.5)
        for u in (150.0, 160.0, 170.0, 165.0):
            session.click(u, 120.0)
        assert session.last_rms is not None  # four clicks: the model was refit
        assert not np.array_equal(session.model.estimates, XleRobotModel.load(calibration).estimates)
        saved = load_session(tmp_path / "cal")
        assert [c.pixel for c in saved.clicks] == [(u, 120.0) for u in (150.0, 160.0, 170.0, 165.0)]
        assert saved.clicks[0].arm_pose["gripper.pos"] == fake.arm_backend.pos["left_arm"]["gripper.pos"]
        assert (tmp_path / "cal" / "images" / "000.png").is_file()
        assert (tmp_path / "cal" / "images" / "000_depth.png").is_file()
        session.undo()
        assert len(load_session(tmp_path / "cal").clicks) == 3 and session.last_rms is None


def test_arrow_keys_turn_the_head_and_a_fresh_frame_is_shown(robot, tmp_path):
    fake, client = robot
    with client.session():
        session, shown = make_session(client, tmp_path, head_step_deg=2.0)
        eyes0, neck0 = fake.head.backend.read_ticks("head_eyes"), fake.head.backend.read_ticks("head_neck")
        frames = len(shown)
        assert session.handle("down")
        assert fake.head.backend.read_ticks("head_eyes") == eyes0 + round(2.0 * 4096 / 360)  # positive = down
        assert session.handle("left")
        assert fake.head.backend.read_ticks("head_neck") == neck0 - round(2.0 * 4096 / 360)  # neck sign is -1
        assert session.handle("up") and session.handle("right")
        assert fake.head.backend.read_ticks("head_eyes") == eyes0
        assert len(shown) == frames + 4
        assert session.handle("quit") is False


def test_other_sessions_join_the_fit_and_finish_writes_one_calibration(robot, tmp_path, calibration):
    fake, client = robot
    other = Session(tmp_path / "cal_right", "right", [], SIGNS, REFS, FAKE_INTRINSICS)
    with client.session():
        session, _ = make_session(client, tmp_path, start_from=calibration, floor_z=0.5)
        for u in (150.0, 160.0, 170.0):
            session.click(u, 120.0)
        other.clicks = [replace(c, arm="right", session="cal_right") for c in session.session.clicks]
        save_session(other)
        joined, _ = make_session(
            client, tmp_path / "second", start_from=calibration, other_sessions=[other.path]
        )
        assert len(joined.all_clicks()) == 3 and joined.last_rms is None  # no clicks of its own yet
        joined.click(155.0, 118.0)
        assert joined.last_rms is not None  # 4 clicks with the other session's
        path = joined.finish(holdout_every=0)
    loaded = XleRobotModel.load(path)
    assert loaded.fit_info["clicks"] == 4 and sorted(loaded.fit_info["sessions"]) == ["cal", "cal_right"]
    assert path.with_suffix(".md").is_file()


def test_sessions_with_another_head_reference_are_refused(robot, tmp_path):
    fake, client = robot
    elsewhere = Session(tmp_path / "old", "right", [], SIGNS, {**REFS, "head_neck": 1605}, FAKE_INTRINSICS)
    save_session(elsewhere)
    with client.session(), pytest.raises(ValueError, match="head_refs"):
        make_session(client, tmp_path, other_sessions=[elsewhere.path])


def test_a_calibration_made_with_other_head_directions_is_refused(robot, tmp_path):
    fake, client = robot
    model = XleRobotModel.from_urdf(head_signs={"head_neck": 1, "head_eyes": 1}, head_refs=REFS)
    model.save(tmp_path / "flipped.urdf")
    with client.session(), pytest.raises(ValueError, match="directions changed"):
        make_session(client, tmp_path, start_from=tmp_path / "flipped.urdf")


def test_resume_continues_an_earlier_folder(robot, tmp_path, calibration):
    fake, client = robot
    with client.session():
        session, _ = make_session(client, tmp_path, start_from=calibration, floor_z=0.5)
        session.click(150.0, 120.0)
        session.click(160.0, 121.0)
        resumed, _ = make_session(client, tmp_path, start_from=calibration, floor_z=0.5, resume=True)
        assert [c.pixel for c in resumed.session.clicks] == [(150.0, 120.0), (160.0, 121.0)]
        assert "resumed with 2 clicks" in resumed.message
        resumed.click(170.0, 122.0)
        assert resumed.session.clicks[-1].index == 2 and resumed.session.clicks[-1].image == "images/002.png"


def test_the_overlay_marks_the_prediction_and_the_clicks():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    out = collect_module.annotate(image, ["hello"], np.array([100.0, 100.0]), [(200.0, 50.0)])
    assert out[100, 100 - 14].any() and out[50, 200].any() and not image.any()
