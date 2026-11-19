"""`sim2real.shortfalls`: a held move's kind, direction and shortfall, and the simulator's arm stopping short as the
real one does (the servos as fitted, `robot.ARM_KP_NM_PER_RAD` / `robot.ARM_FRICTIONLOSS_NM`)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import shortfalls as sf  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402

# pen2's attack on the right arm (2026-11-05): commanded, and read there (+0.35, +0.10, +1.29, +0.83, +0.34 deg short)
PEN2_ATTACK = {
    "shoulder_pan": -10.68,
    "shoulder_lift": -13.16,
    "elbow_flex": 43.28,
    "wrist_flex": 11.34,
    "wrist_roll": -126.71,
}
PEN2_SHORT = (0.35, 0.10, 1.29, 0.83, 0.34)


def _move(label, previous, commanded=PEN2_ATTACK, short=PEN2_SHORT):
    read = {j: commanded[j] + s for j, s in zip(sf.JOINTS, short, strict=True)}
    return sf.HeldMove("pen2", "capture12", "right", label, dict(commanded), read, previous, 14.2)


def test_a_moves_kind_direction_and_shortfall():
    previous = {**PEN2_ATTACK, "elbow_flex": 46.28, "wrist_flex": 11.32}
    move = _move("pick.back [-43.0, 7.6, ...]", previous)
    assert move.kind == "attack"
    assert [_move(label, previous).kind for label in ("pick [x]", "lift4 [x]", "stage", "place")] == [
        "grasp",
        "loaded",
        "loaded",
        "other",
    ]
    assert move.direction.tolist() == [0.0, 0.0, -1.0, 0.0, 0.0]  # (the wrist's 0.02 deg: no direction)
    assert move.shortfall == pytest.approx(PEN2_SHORT)


@pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")
def test_the_simulated_arm_stops_short_on_the_side_it_came_from_as_the_real_one_did():
    """The elbow moved down its range into pen2's attack stops ~1 deg short (real: +1.29); moved up, ~0.2 (the real
    holds that moved up: +0.21 on average); the lift, carrying the arm, a little short either way."""
    model = XleRobotModel.load(api.MODEL)
    down = _move("pick.back", {**PEN2_ATTACK, "elbow_flex": PEN2_ATTACK["elbow_flex"] + 3.0})
    up = _move("pick.back", {**PEN2_ATTACK, "elbow_flex": PEN2_ATTACK["elbow_flex"] - 3.0})
    short = {
        name: np.array([r[j] - PEN2_ATTACK[j] for j in sf.JOINTS])
        for name, m in (("down", down), ("up", up))
        for r in [sf.replay_hold(model, m)]
    }
    elbow = sf.JOINTS.index("elbow_flex")
    assert 0.6 < short["down"][elbow] < 1.6
    assert -0.2 < short["up"][elbow] < 0.5
    assert short["down"][elbow] - short["up"][elbow] > 0.5  # (the friction's band)
