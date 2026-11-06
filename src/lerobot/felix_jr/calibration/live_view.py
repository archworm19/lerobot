"""A live camera view of one arm through the robot server, driven through planned views.

It is what `collect` (clicking the pinch point) and `check` (drawing the model's gripper points) share. `LiveView` plans target views over the workspace (`views.plan_targets`), moves the arm and the head to the next one
that shows the pinch point (`views.find_view`), turns the head on request, and keeps the latest capture with the
joint angles measured with it (`current_q`). What is drawn over the frame is the subclass's (`render`).
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG, Target, find_view, plan_targets
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, TICKS_PER_TURN, XleRobotModel
from lerobot.felix_jr.server.client import Capture, FelixClient
from lerobot.felix_jr.server.protocol import ServerError

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
KEY_ACTIONS = {ord("n"): "next", ord("u"): "undo", ord("r"): "refit", ord("q"): "quit", 27: "quit"}


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


def parse_seed(text: str | None) -> dict[str, float] | None:
    """`"shoulder_pan=10,elbow_flex=-5"` -> `{"shoulder_pan.pos": 10.0, "elbow_flex.pos": -5.0}` (None for no text)."""
    if not text:
        return None
    return {f"{k.strip()}.pos": float(v) for k, v in (part.split("=") for part in text.split(","))}


def set_floor(model: XleRobotModel, arm: str, floor_z: float | None, note: Callable[[str], None]) -> None:
    """Give `model` a floor: `floor_z`, else its own, else the arm base's height (said through `note`)."""
    if floor_z is not None:
        model.floor_z = floor_z
    elif model.floor_z is None:
        base = model.frame_pose(f"{arm}_base_link", model.joint_vector())
        model.floor_z = float(base[2, 3])
        note("no floor height known: keeping the pinch point above the arm base's height (--floor-z)")


class LiveView:
    """One arm seen through the head camera, moved view after view. See the module docstring."""

    no_more_views = "no more views: q to finish"  # the message when the targets run out

    def __init__(
        self,
        client: FelixClient,
        model: XleRobotModel,
        arm: str,
        show: Callable[[np.ndarray], None],
        capture: Capture,
        n_targets: int = 40,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        color_format: str = "png",
        seed_pose: dict[str, float] | None = None,
        delta_scale: float = 1.0,
        max_roll: float = MAX_ROLL_DEG,
    ):
        """Plan `n_targets` views around `seed_pose` (LeRobot degrees; default the center pose) with `model`.

        `capture` is the frame to start from; nothing is shown until the subclass calls `render`.
        """
        self.client, self.model, self.arm, self.show, self.capture = client, model, arm, show, capture
        self.arm_id = f"{arm}_arm"
        self.max_vel, self.max_accel, self.head_step_deg = max_vel, max_accel, head_step_deg
        self.color_format, self.max_roll = color_format, max_roll
        self.message = ""
        self.targets: list[Target] = plan_targets(model, arm, n_targets, seed_pose, delta_scale)
        self.target_index = 0

    def render(self) -> None:
        """Show the current frame, with whatever the subclass draws over it."""
        raise NotImplementedError

    def before_view(self, index: int) -> None:
        """Called before a view of target `index` is looked for; nothing by default.

        A subclass may set up the model for it, e.g. the side of the gripper the camera must see.
        """

    # ---- the current frame ----------------------------------------------------------------------------------

    def current_q(self) -> np.ndarray:
        """The robot's measured joints in the current frame, as the model's `q`."""
        state = self.capture.state
        return self.model.joint_vector(
            {self.arm: state[self.arm_id]}, {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        )

    def refresh(self, after: float | None = None) -> None:
        """Take a new frame (taken after server time `after`, if given) and show it."""
        self.capture = self.client.capture(color_format=self.color_format, after=after)
        self.render()

    # ---- moving ---------------------------------------------------------------------------------------------

    def _move_head_to(self, targets: dict[str, int]) -> float | None:
        """Move each head servo to raw ticks `targets`. Returns the finish time of the last move."""
        head = self.client.head_state()
        done = None
        for name, target in targets.items():
            delta = head[name]["sign"] * (target - head[name]["ticks"]) / TICKS_PER_DEG
            if abs(delta) * TICKS_PER_DEG >= 2:
                done = self.client.head_move(name, delta_deg=delta)["done_at"]
        return done

    def next_pose(self, gripper_pos: float | None = None) -> bool:
        """Find a view of the next target that shows the pinch point, move the arm and the head there, show it.

        Targets whose best view does not show the pinch point (`View.ok`) are skipped. `gripper_pos` sets the
        gripper (LeRobot's 0-100) in the same motion; by default it closes. Returns False if there are no more views
        or the arm did not get there.
        """
        head = self.client.head_state()
        limits = head_limits(head)
        ticks = {n: int(head[n]["ticks"]) for n in HEAD_JOINTS}
        skipped = 0
        while self.target_index < len(self.targets):
            target = self.targets[self.target_index]
            self.before_view(self.target_index)
            self.target_index += 1
            seed = self.model.joint_vector({self.arm: target.arm_pose}, ticks)
            view = find_view(self.model, self.arm, seed, target.point, limits, max_roll_deg=self.max_roll)
            if view.ok:
                break
            skipped += 1
        else:
            self.message = self.no_more_views
            self.render()
            return False
        pose = dict(view.arm_pose)
        if gripper_pos is not None:
            pose["gripper.pos"] = float(gripper_pos)
        status = self.client.move_joints({self.arm_id: pose}, max_vel=self.max_vel, max_accel=self.max_accel)
        if status["state"] != "done":
            self.message = f"the arm motion ended as {status['state']}: {status['reason']}"
            self.refresh(status["finished_at"])
            return False
        done = self._move_head_to(view.head_ticks) or status["finished_at"]
        skip_text = f" ({skipped} targets skipped: no view showed them)" if skipped else ""
        self.message = (
            f"view {self.target_index}: gripper facing the camera within {view.facing_deg:.0f} deg{skip_text}"
        )
        self.refresh(done)
        return True

    def jog(self, neck_deg: float = 0.0, eyes_deg: float = 0.0) -> None:
        """Turn the head by the given degrees (positive = left / down) and show a new frame."""
        done = None
        try:
            for name, delta in (("head_neck", neck_deg), ("head_eyes", eyes_deg)):
                if delta:
                    result = self.client.head_move(name, delta_deg=delta)
                    done = result["done_at"]
                    self.message = "" if not result["clamped"] else f"{name} is at its recorded limit"
        except ServerError as e:
            self.message = str(e)
        self.refresh(done)

    def handle(self, action: str) -> bool:
        """Do one of the shared actions (next, the head's arrows, quit). Returns False when it is time to quit."""
        step = self.head_step_deg
        if action == "quit":
            return False
        if action == "next":
            self.next_pose()
        elif action in ("up", "down"):
            self.jog(eyes_deg=step if action == "down" else -step)  # the eyes' positive direction is down
        elif action in ("left", "right"):
            self.jog(neck_deg=step if action == "left" else -step)  # the neck's positive direction is left
        return True
