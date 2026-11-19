"""The calibration windows' controls: the keys, and the head's limits and moves through the robot server.

What `joint_collect` and `wrist_camera` share (from the old click calibration's `live_view`, gone 2026-11-19).
"""

from typing import Any

from lerobot.felix_jr.calibration.xle_robot_model import TICKS_PER_TURN
from lerobot.felix_jr.server.client import FelixClient

TICKS_PER_DEG = TICKS_PER_TURN / 360.0
NECK_RANGE_TICKS = (
    450  # +/- ticks (about 40 deg) the neck may be turned from where it is, if no limit is recorded
)
INTRINSICS_KEYS = ("fx", "fy", "ppx", "ppy", "width", "height")
# cv2.waitKeyEx codes differ by platform (macOS, Linux GTK, Windows, Qt).
UP_KEYS = {63232, 65362, 2490368, 0, ord("i")}
DOWN_KEYS = {63233, 65364, 2621440, 1, ord("k")}
LEFT_KEYS = {63234, 65361, 2424832, 2, ord("j")}
RIGHT_KEYS = {63235, 65363, 2555904, 3, ord("l")}
KEY_ACTIONS = {ord("n"): "next", ord("u"): "undo", ord("q"): "quit", 27: "quit"}


def decode_key(key: int, actions: dict[int, str] = KEY_ACTIONS) -> str | None:
    """The action for a `cv2.waitKeyEx` code: up, down, left, right, or one of `actions`; None if none."""
    for keys, action in ((UP_KEYS, "up"), (DOWN_KEYS, "down"), (LEFT_KEYS, "left"), (RIGHT_KEYS, "right")):
        if key in keys:
            return action
    return actions.get(key)


def head_limits(head: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """Allowed raw ticks per head servo from `client.head_state()`: the recorded limits, else a range around now.

    Limits are recorded as the servo's positive / negative direction (down / left, up / right), in raw ticks, so
    the lower and upper bound are whichever of the two is smaller. Without recorded limits the servo may move
    `NECK_RANGE_TICKS` either side of where it is now (nothing is known about the cable or the mount).
    """
    limits = {}
    for name, servo in head.items():
        if servo["limit_neg"] is not None and servo["limit_pos"] is not None:
            low, high = sorted((servo["limit_neg"], servo["limit_pos"]))
        else:
            low, high = servo["ticks"] - NECK_RANGE_TICKS, servo["ticks"] + NECK_RANGE_TICKS
        limits[name] = (max(low, 0), min(high, TICKS_PER_TURN - 1))
    return limits


def move_head_to(client: FelixClient, targets: dict[str, int]) -> float | None:
    """Move each head servo to raw ticks `targets` (within 2 ticks: left). Returns the finish time of the last move."""
    head = client.head_state()
    done = None
    for name, target in targets.items():
        delta = head[name]["sign"] * (target - head[name]["ticks"]) / TICKS_PER_DEG
        if abs(delta) * TICKS_PER_DEG >= 2:
            done = client.head_move(name, delta_deg=delta)["done_at"]
    return done
