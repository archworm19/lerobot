"""Head controller, the head commands of the server, `capture(after=)`, and the head range script (no hardware)."""

import threading
import time

import numpy as np
import pytest

from lerobot.felix_jr.server import hardware
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware, FakeHeadBackend, _LatestFrames
from lerobot.felix_jr.server.head import SUB_STEP_DEG, TICKS_PER_DEG, HeadController, RangeStore
from lerobot.felix_jr.server.protocol import ServerError
from lerobot.felix_jr.server.server import RobotServer
from lerobot.felix_jr.setup.head_range import HeadRangeSession, decode_key

EYES, NECK = "head_eyes", "head_neck"
SIGNS = {EYES: 1, NECK: -1}


@pytest.fixture
def head():
    return HeadController(FakeHeadBackend(), SIGNS, RangeStore())


def test_move_in_the_positive_direction_follows_the_sign(head):
    start_eyes, start_neck = head.backend.read_ticks(EYES), head.backend.read_ticks(NECK)
    head.move(EYES, delta_deg=2.0)
    head.move(NECK, delta_deg=2.0)
    step = round(2.0 * TICKS_PER_DEG)
    assert head.backend.read_ticks(EYES) == start_eyes + step  # eyes: raw positive is down
    assert head.backend.read_ticks(NECK) == start_neck - step  # neck: raw positive is right, so left is -1


def test_limits_clamp_in_the_right_raw_direction(head):
    head.move(EYES, delta_deg=10.0)
    head.record(EYES, "limit_pos")  # the down limit, raw high end for the eyes
    limit_pos = head.backend.read_ticks(EYES)
    head.move(EYES, delta_deg=-20.0)
    head.record(EYES, "limit_neg")
    limit_neg = head.backend.read_ticks(EYES)
    assert limit_neg < limit_pos
    assert head.bounds(EYES) == (limit_neg, limit_pos)

    result = head.move(EYES, delta_deg=-5.0)  # past the up limit
    assert result["clamped"] and result["ticks"] == limit_neg
    result = head.move(EYES, delta_deg=90.0)  # past the down limit
    assert result["clamped"] and result["ticks"] == limit_pos
    assert head.move(EYES, delta_deg=-1.0)["clamped"] is False


def test_limit_on_a_servo_with_negative_sign_bounds_the_other_end(head):
    head.move(NECK, delta_deg=10.0)  # raw goes down
    head.record(NECK, "limit_pos")  # "left" limit = raw low end
    low = head.backend.read_ticks(NECK)
    assert head.bounds(NECK) == (low, 4095)
    assert head.move(NECK, delta_deg=5.0)["ticks"] == low


def test_go_to_recorded_positions_and_errors(head):
    head.record(EYES, "home")
    home = head.backend.read_ticks(EYES)
    head.move(EYES, delta_deg=15.0)
    result = head.move(EYES, to="home")
    assert result["ticks"] == home and result["reached"]
    with pytest.raises(ValueError, match="no level recorded"):
        head.move(EYES, to="level")
    with pytest.raises(ValueError, match="exactly one"):
        head.move(EYES)
    with pytest.raises(ValueError, match="exactly one"):
        head.move(EYES, delta_deg=1.0, to="home")
    with pytest.raises(ValueError, match="unknown head servo"):
        head.move("head_tail", delta_deg=1.0)
    with pytest.raises(ValueError, match="`what`"):
        head.record(EYES, "favourite")
    with pytest.raises(ValueError, match="torque limit"):
        head.torque_limit(EYES, 0)
    assert head.torque_limit(EYES, 300) == {"name": EYES, "limit": 300}
    assert head.backend.torque_limits[EYES] == 300


def test_a_long_move_is_made_in_small_steps():
    class Recording(FakeHeadBackend):
        writes: list[int] = []

        def write_goal(self, name, ticks):
            self.writes.append(ticks)
            super().write_goal(name, ticks)

    backend = Recording()
    Recording.writes = []
    controller = HeadController(backend, SIGNS, RangeStore())
    controller.move(EYES, delta_deg=22.0)
    steps = np.diff([2048, *backend.writes])
    assert np.all(np.abs(steps) <= round(SUB_STEP_DEG * TICKS_PER_DEG))
    assert backend.read_ticks(EYES) == 2048 + round(22.0 * TICKS_PER_DEG)


def test_a_blocked_servo_is_told_to_hold_and_the_move_stops():
    class Blocked(FakeHeadBackend):
        """Moves only up to raw 2100, then ignores goals."""

        writes: list[int] = []

        def write_goal(self, name, ticks):
            self.writes.append(ticks)
            super().write_goal(name, min(ticks, 2100))

    now = [0.0]
    backend = Blocked()
    Blocked.writes = []
    controller = HeadController(
        backend, SIGNS, RangeStore(), sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0]
    )
    result = controller.move(EYES, delta_deg=30.0)
    assert result["reached"] is False
    assert result["ticks"] == 2100
    assert backend.writes[-1] == 2100  # the last command is "stay where you are"


def test_range_store_persists(tmp_path):
    path = tmp_path / "sub" / "left_arm_head_range.json"
    controller = HeadController(FakeHeadBackend(), SIGNS, RangeStore(path))
    controller.move(EYES, delta_deg=4.0)
    controller.record(EYES, "home")
    ticks = controller.backend.read_ticks(EYES)
    reloaded = HeadController(FakeHeadBackend(), SIGNS, RangeStore(path))
    assert reloaded.state()[EYES]["home"] == ticks
    assert reloaded.state()[EYES]["level"] is None


@pytest.fixture
def served(tmp_path):
    fake = FakeHardware(head_range_path=tmp_path / "range.json")
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake, address
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


@pytest.fixture
def client(served):
    with FelixClient(served[1], timeout_s=5.0) as c:
        yield c


def test_head_commands_over_the_wire(client):
    state = client.head_state()
    assert state[EYES]["ticks"] == 2048 and state[EYES]["home"] is None
    result = client.head_move(EYES, delta_deg=2.0)
    assert result["ticks"] == 2048 + round(2.0 * TICKS_PER_DEG) and result["reached"]
    assert client.head_record(EYES, "home")["home"] == result["ticks"]
    assert client.get_state()["head"][EYES]["home"] == result["ticks"]
    assert client.head_torque_limit(EYES, 300)["limit"] == 300
    with pytest.raises(ServerError, match="no level recorded"):
        client.head_move(EYES, to="level")


def test_the_picture_changes_when_the_head_moves(client):
    before = client.capture(after=time.time()).color
    done = client.head_move(EYES, delta_deg=5.0)["done_at"]
    after = client.capture(after=done).color
    assert not np.array_equal(before, after)


class StubFrames:
    """A pipeline delivering numbered frames every 20 ms."""

    def __init__(self):
        self.count = 0

    def wait_for_frames(self, timeout_ms):
        time.sleep(0.02)
        self.count += 1
        n = self.count
        frame = type("Frame", (), {"get_data": lambda self, n=n: np.full((2, 2), n, dtype=np.uint16)})()
        return type("Frames", (), {"get_color_frame": lambda s: frame, "get_depth_frame": lambda s: frame})()


def test_capture_after_waits_for_a_frame_taken_after_the_move():
    frames = _LatestFrames(StubFrames())
    try:
        _, _, first = frames.get()
        asked_at = time.time()
        _, _, arrived = frames.get(after=asked_at)
        assert arrived > asked_at + hardware.FRAME_LATENCY_S
        with pytest.raises(RuntimeError):
            frames.get(after=time.time() + 60, wait_s=0.3)
        assert first <= arrived
    finally:
        frames.stop()


def test_decode_key():
    assert decode_key(63232) == "up" and decode_key(65362) == "up" and decode_key(ord("i")) == "up"
    assert decode_key(63233) == "down" and decode_key(ord("k")) == "down"
    assert decode_key(ord("u")) == "limit_neg" and decode_key(ord("d")) == "limit_pos"
    assert decode_key(ord("h")) == "home" and decode_key(ord("z")) == "level"
    assert decode_key(27) == "quit" and decode_key(ord("q")) == "quit"
    assert decode_key(ord("x")) is None and decode_key(-1) is None


def test_range_session_walks_through_setting_a_range(client, served):
    fake, _ = served
    shown = []
    session = HeadRangeSession(client, EYES, 2.0, shown.append)
    session.refresh()
    assert len(shown) == 1 and shown[0].shape == (240, 320, 3)

    assert session.handle("level")
    assert "saved level" in session.message
    for _ in range(3):
        session.handle("down")
    session.handle("limit_pos")
    assert session.handle("down")  # at the down limit now
    assert session.message == "at the recorded limit"
    for _ in range(4):
        session.handle("up")
    session.handle("limit_neg")
    session.handle("home")
    session.handle("up")  # clamped by the up limit
    assert session.message == "at the recorded limit"
    state = client.head_state()[EYES]
    assert state["level"] == 2048
    assert state["limit_pos"] == 2048 + 3 * round(2.0 * TICKS_PER_DEG)
    assert state["limit_neg"] < state["level"] < state["limit_pos"]
    assert len(shown) == 1 + 13  # the first image, then one per handled action

    assert session.handle("go_level") and session.message == "moved to level"
    assert fake.head.backend.read_ticks(EYES) == 2048
    session.handle("go_home")  # fine: home is recorded
    session.handle("go_level")
    fake.head.store.data[EYES].pop("home")
    session.handle("go_home")
    assert "no home recorded" in session.message
    assert session.handle("quit") is False


def test_range_session_finish_goes_home_and_lifts_the_torque_cap(client, served):
    fake, _ = served
    session = HeadRangeSession(client, EYES, 2.0, lambda image: None)
    client.head_torque_limit(EYES, 300)
    session.handle("down")
    client.head_record(EYES, "home")
    session.handle("down")
    session.finish()
    assert fake.head.backend.read_ticks(EYES) == client.head_state()[EYES]["home"]
    assert fake.head.backend.torque_limits[EYES] == 1000
