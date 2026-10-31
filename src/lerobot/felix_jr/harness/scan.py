"""The startup scan: a harness phase, not a tool (`docs/harness.md`, 1.A: no tool moves the robot).

The head steps through a grid of poses relative to its recorded home (tilts down x pans left/right, in a serpentine
order so each step is small), takes a capture after each move has settled, and goes back home, even if a step fails.
Only the head moves. Every capture gets its camera pose from its own measured head ticks, so a pose the head could
not reach (clamped to its recorded limits) still gives a correct capture.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import Capture

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

TILTS_DEG = (0.0, 15.0, 30.0)  # down from home
PANS_DEG = (20.0, 0.0, -20.0)  # left (+) / right (-) of home
TILT_SERVO, PAN_SERVO = "head_eyes", "head_neck"


def scan_poses(
    tilts: Sequence[float] = TILTS_DEG, pans: Sequence[float] = PANS_DEG
) -> list[tuple[float, float]]:
    """(tilt, pan) in degrees from home, in the order the scan visits them (each row of pans reverses the last)."""
    return [(t, p) for i, t in enumerate(tilts) for p in (pans if i % 2 == 0 else list(reversed(pans)))]


def scan(
    client: "FelixClient",
    model: XleRobotModel,
    tilts: Sequence[float] = TILTS_DEG,
    pans: Sequence[float] = PANS_DEG,
) -> list[Capture]:
    """One capture per pose of `scan_poses(tilts, pans)`, in that order. The head starts and ends at home.

    Before anything moves, every servo the scan uses must have a recorded home and limits (`setup.head_range`):
    without limits the server would not clamp the moves.
    """
    used = {TILT_SERVO: any(tilts), PAN_SERVO: any(pans)}
    head = client.head_state()
    missing = [
        f"{name} has no recorded {what}"
        for name, moves in used.items()
        if moves
        for what in ("home", "limit_pos", "limit_neg")
        if head[name].get(what) is None
    ]
    if missing:
        raise RuntimeError("cannot scan: " + "; ".join(missing) + " (set them with setup.head_range)")
    servos = [name for name, moves in used.items() if moves]  # a servo the scan doesn't use stays put
    captures = []
    try:
        for name in servos:
            client.head_move(name, to="home")
        tilt = pan = 0.0
        moved = None
        for t, p in scan_poses(tilts, pans):
            if used[TILT_SERVO]:
                moved = client.head_move(TILT_SERVO, delta_deg=t - tilt)
            if used[PAN_SERVO]:
                moved = client.head_move(PAN_SERVO, delta_deg=p - pan)
            tilt, pan = t, p
            after = moved["done_at"] if moved is not None else None
            captures.append(Capture.from_snapshot(client.capture(after=after), model))
    finally:
        for name in reversed(servos):
            client.head_move(name, to="home")
    return captures
