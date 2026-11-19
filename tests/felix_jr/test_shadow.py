"""`sim2real.shadow`: a move replayed as the server runs it: its trapezoid's duration, the servos stopping short on the
side they came from, the jaws' closing marking the moves after it loaded."""

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.shadow import DWELL_S, NO_TABLE_Z, Shadow  # noqa: E402
from lerobot.felix_jr.trajectory import move_time_trap  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")

# pen2's attack on the right arm (as `test_shortfalls.py`), the jaws open
PEN2_ATTACK = {
    "shoulder_pan.pos": -10.68,
    "shoulder_lift.pos": -13.16,
    "elbow_flex.pos": 43.28,
    "wrist_flex.pos": 11.34,
    "wrist_roll.pos": -126.71,
    "gripper.pos": 14.2,
}


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


def test_a_move_takes_its_trapezoid_and_stops_short_on_the_side_it_came_from(model):
    """The elbow down its range by 3 deg then back up: ~1 deg short going down (the real attacks: +1.1), within
    ~0.5 going up; each reached (within the server's 1.5 deg), in the trapezoid's time plus the dwell at least."""
    start = {**PEN2_ATTACK, "elbow_flex.pos": PEN2_ATTACK["elbow_flex.pos"] + 3.0}
    shadow = Shadow(model, "right", start, table_z=NO_TABLE_Z)
    down = shadow.move("down", {"elbow_flex.pos": PEN2_ATTACK["elbow_flex.pos"]}, 20.0, 40.0)
    assert down.reached and not down.loaded
    assert down.seconds >= move_time_trap(3.0, 20.0, 40.0) + DWELL_S - 1e-9
    short_down = down.joints["elbow_flex.pos"] - PEN2_ATTACK["elbow_flex.pos"]
    assert 0.5 < short_down < 1.6, short_down
    up = shadow.move("up", {"elbow_flex.pos": PEN2_ATTACK["elbow_flex.pos"] + 3.0}, 20.0, 40.0)
    short_up = up.joints["elbow_flex.pos"] - (PEN2_ATTACK["elbow_flex.pos"] + 3.0)
    assert up.reached and -0.3 < short_up < 0.6 and short_up < short_down, (short_up, short_down)


def test_a_close_marks_the_moves_after_it_loaded_until_the_jaws_open(model):
    shadow = Shadow(model, "right", PEN2_ATTACK, table_z=NO_TABLE_Z)
    close = shadow.move("close", {"gripper.pos": 2.0}, 20.0, 40.0)
    lift = shadow.move("lift", {"shoulder_lift.pos": PEN2_ATTACK["shoulder_lift.pos"] - 3.0}, 20.0, 40.0)
    opened = shadow.move("open", {"gripper.pos": 14.0}, 20.0, 40.0)
    assert close.loaded and lift.loaded and not opened.loaded
    assert close.gripper_pos < 5.0  # (nothing in the jaws: they close)


def test_a_logged_execution_is_replayed_compared_and_reported_through_the_harness(model):
    """An `ExecutionLog` as `execute` records it (the attack reached from 3 deg up the elbow's range, read ~1 deg short
    as pen2's was): `replay` records the simulated ends, `compare` lists the move real against simulated, and
    `shadow_report` sums the session (each log replayed once)."""
    from lerobot.felix_jr.harness import api
    from lerobot.felix_jr.harness.objects import ExecutionLog, MoveReading

    start = {**PEN2_ATTACK, "elbow_flex.pos": PEN2_ATTACK["elbow_flex.pos"] + 3.0}
    attack = {k: v for k, v in PEN2_ATTACK.items() if k != "gripper.pos"}
    read = {**PEN2_ATTACK, "elbow_flex.pos": PEN2_ATTACK["elbow_flex.pos"] + 1.29}
    log = ExecutionLog(
        "right", 20.0, 40.0, start, (MoveReading("pick.back", attack, 1.5, 10.0, "done", True, read),), "held"
    )
    h = api.Harness(model=model, policy=api.UNATTENDED)
    h.ws.add(log, "execute")
    record = h.call("replay", log)
    assert h.ws.id_of(record) is not None and len(record.moves) == 1 and record.settings["kp"]
    text = h.call("compare", record)
    assert "pick.back [attack]" in text and "+1.29/" in text and "mm apart" in text
    report = h.call("shadow_report")
    assert report.startswith("1 moves of 1 executions") and "attack (1)" in report
    assert sum(1 for c in h.calls if c["tool"] == "replay") == 1  # (the recorded replay reused)
