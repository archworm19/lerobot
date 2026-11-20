"""An object stood where a session wants it: its outline drawn on the head camera's live picture (`docs/sim_harness.md`).

For a session whose objects must stand at known places (the first robot session of the simulated grasps): the head
turns to the spot (`look.aim_head`) and shows its live picture with the object's outline where it should stand, its
circle on the table and at its top (`bottle_rings`, projected through the calibration: `Capture.project`, as good as
it is, ~3 mm); the person moves the object until it matches, Enter (`place_object`). `placement_spots`: spots for an
upright cylinder over the arm's reach where the harness finds a grasp (`grasping.first_feasible`).
"""

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.controls import decode_key, head_limits, move_head_to
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel

from .look import aim_head
from .objects import Capture

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

RING_POINTS = 48
RING_COLORS = {"bottom": (255, 160, 0), "top": (0, 255, 255)}  # RGB: on the table orange, its top cyan
KEYS = {13: "done", 10: "done", 32: "done", ord("q"): "quit", 27: "quit"}
REFRESH_MS = 150
# the spots tried by `placement_spots`, from the arm's base: ahead (the robot faces -x) and to the side (m)
SPOT_AHEAD_M = (0.24, 0.28, 0.32)
SPOT_SIDE_M = (-0.10, -0.05, 0.0, 0.05, 0.10)


def bottle_rings(xy: Sequence[float], radius: float, height: float, table_z: float) -> dict[str, np.ndarray]:
    """An upright cylinder standing at `xy` (root frame, m): its circle on the table and at its top, (N, 3) each."""
    angles = np.linspace(0.0, 2 * np.pi, RING_POINTS)
    ring = np.c_[xy[0] + radius * np.cos(angles), xy[1] + radius * np.sin(angles)]
    return {
        name: np.c_[ring, np.full(len(ring), z)]
        for name, z in (("bottom", table_z), ("top", table_z + height))
    }


def draw_rings(
    picture: np.ndarray, capture: Capture, rings: Mapping[str, np.ndarray], label: str = ""
) -> np.ndarray:
    """`rings` drawn on `picture` (the capture's colour, RGB) through the capture's camera pose."""
    out = picture.copy()
    for name, points in rings.items():
        uv = capture.project(points)
        ok = np.isfinite(uv).all(axis=1)
        if ok.sum() > 2:
            cv2.polylines(out, [np.round(uv[ok]).astype(np.int32)], True, RING_COLORS[name], 2, cv2.LINE_AA)
    lines = [label, "orange: on the table  cyan: its top    Enter: placed  q: stop"]
    for k, text in enumerate(line for line in lines if line):
        for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(
                out, text, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, width, cv2.LINE_AA
            )
    return out


def place_object(
    client: "FelixClient",
    model: XleRobotModel,
    xy: Sequence[float],
    radius: float,
    height: float,
    table_z: float,
    label: str = "",
    show: Callable[[np.ndarray], int] | None = None,
) -> Capture | None:
    """Turn the head to `xy`, show its live picture with the object's outline there until the person presses Enter.

    Returns a capture once the person accepted the placement (full quality, the head still aimed: what to fit the
    object in), None if they stopped (`q`). `show(picture)` shows a
    picture and returns the key pressed (`cv2.waitKeyEx`'s code, -1 for none); default: an OpenCV window.
    """
    head = client.head_state()
    ticks = {n: int(head[n]["ticks"]) for n in HEAD_JOINTS}
    aimed, _ = aim_head(model, ticks, np.array([xy[0], xy[1], table_z + height / 2]))
    limits = head_limits(head)
    done_at = move_head_to(client, {n: int(np.clip(t, *limits[n])) for n, t in aimed.items()})
    rings = bottle_rings(xy, radius, height, table_z)
    window = "place the object on its outline (Enter: placed, q: stop)"

    def in_window(picture: np.ndarray) -> int:
        cv2.imshow(window, cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        return cv2.waitKeyEx(REFRESH_MS)

    show = in_window if show is None else show
    after = done_at
    try:
        while True:
            capture = Capture.from_snapshot(client.capture(color_format="jpeg", after=after), model)
            after = None
            action = decode_key(show(draw_rings(capture.scene.color, capture, rings, label)), KEYS)
            if action == "done":  # (a full-quality capture of the object as placed: what the fit works on)
                return Capture.from_snapshot(client.capture(), model)
            if action == "quit":
                return None
    finally:
        if show is in_window:
            cv2.destroyWindow(window)


def placement_spots(
    model: XleRobotModel,
    radius: float,
    height: float,
    table_z: float,
    arm: str = "right",
    rest: Mapping[str, float] | None = None,
    n: int = 5,
    settings: Any = None,
) -> list[tuple[float, float]]:
    """Up to `n` spots (root frame x, y) for an upright cylinder where the harness finds a grasp from the side.

    Tried over a grid ahead of and beside the arm's base (`SPOT_AHEAD_M`, `SPOT_SIDE_M`), spread: the farthest from
    those already taken first.
    """
    from lerobot.felix_jr.calibration.joint_sim import REST

    from .grasping import GraspSettings, first_feasible
    from .primitives import Cylinder

    if settings is None:  # (the arm's plane checked before any IK: a spot nothing reaches is told in seconds)
        settings = GraspSettings(rotation_bounds_deg=(15.0, 10.0, 10.0), arm_plane_tol_deg=15.0)

    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    capture: Any = SimpleNamespace(arms={arm: dict(rest or {**REST, "gripper.pos": 2.0})}, head_ticks=None)
    reachable = []
    for ahead in SPOT_AHEAD_M:
        for side in SPOT_SIDE_M:
            xy = (float(base[0] - ahead), float(base[1] + side))
            target = Cylinder.upright(np.array([*xy, table_z]), height, radius)
            found = first_feasible(
                target,
                model,
                capture,
                np.random.default_rng(0),
                arms=(arm,),
                where={"psi": (0.0, 30.0)},
                settings=settings,
                max_tries=3000,
            )
            if found.candidate is not None:
                reachable.append(xy)
    if not reachable:
        return []
    chosen = [reachable[len(reachable) // 2]]
    while len(chosen) < min(n, len(reachable)):
        far = max(
            (p for p in reachable if p not in chosen),
            key=lambda p: min(np.hypot(p[0] - c[0], p[1] - c[1]) for c in chosen),
        )
        chosen.append(far)
    return chosen
