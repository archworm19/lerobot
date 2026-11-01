"""Running solved poses on the arm: `prepare_execution` (software) and `run_execution` (moves the arm).

`prepare_execution` turns `Poses` into an `Execution`, the exact moves to send: the path through the poses is cut
into straight steps of at most `step` (so the joint-space moves between them stay close to the checked lines), each
solved with `check_poses` at the poses' pitch, width and jaw direction. It refuses if a step is unreachable or a
joint jumps more than `max_jump_deg` between steps. With the jaws free, pointing down, the roll is nearly free and
drifts from step to step, so it is pinned to the first pose's (or `roll_deg`). Gripper actions go at the poses they
name: by default, for `approach_poses` output, open before the first move, close at `at1`, open at `at2`.

`run_execution` sends the moves one by one inside `client.session()`, so any error or Ctrl+C sends the robot home;
checks first that the arm is where the poses' capture saw it; goes home at the end; and returns a capture taken
after. It does not ask anyone: the caller (`api.Harness`) does, before it runs.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import Capture, Execution, Move, Poses
from .reach import check_poses

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

GRIPPER_ACTIONS = ("open", "close")
START = "start"  # the grip key for "before the first move"


def densify(points: np.ndarray, step: float) -> tuple[np.ndarray, list[int]]:
    """`points` with each leg cut into pieces of at most `step`, and the index of each point in the result."""
    out, marks = [points[0]], [0]
    for a, b in zip(points[:-1], points[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
        marks.append(len(out) - 1)
    return np.array(out), marks


def default_grip(poses: Poses) -> dict[str, str]:
    """Open before the first move, close at `at1`, open at `at2`, when the poses have those labels."""
    if "at1" not in poses.labels:
        return {}
    grip = {START: "open", "at1": "close"}
    if "at2" in poses.labels:
        grip["at2"] = "open"
    return grip


def prepare_execution(
    model: XleRobotModel,
    poses: Poses,
    grip: Mapping[str, str] | None = None,
    gripper_open: float = 60.0,
    gripper_closed: float = 1.0,
    step: float = 0.01,
    roll_deg: float | None = None,
    max_jump_deg: float = 20.0,
) -> Execution:
    """The moves through `poses` (see the module docstring); raises ValueError if they cannot be run.

    `grip` maps a pose label (or "start") to "open" / "close"; default `default_grip`.
    """
    if not len(poses.points) or not poses.ok.all():
        raise ValueError(f"the {poses.arm} arm does not reach every pose: {poses.describe()}")
    grip = dict(default_grip(poses) if grip is None else grip)
    unknown = [k for k in grip if k != START and k not in poses.labels]
    if unknown or any(a not in GRIPPER_ACTIONS for a in grip.values()):
        raise ValueError(
            f"bad grip {grip}: keys are pose labels {poses.labels} or {START!r}, actions open/close"
        )
    dense, marks = densify(poses.points, step)
    if roll_deg is None and poses.across is None:  # jaws free: pin the roll, else it drifts from step to step
        roll_deg = poses.joints[0]["wrist_roll.pos"]
    steps = check_poses(
        model, poses.arm, poses.capture, dense, poses.pitch_deg, poses.width, poses.across, roll_deg
    )
    if not steps.ok.all():
        raise ValueError(
            f"steps {(np.flatnonzero(~steps.ok) + 1).tolist()} are unreachable: {steps.describe()}"
        )
    for i in range(1, len(steps.joints)):
        jump = max(abs(steps.joints[i][k] - steps.joints[i - 1][k]) for k in steps.joints[i])
        if jump > max_jump_deg:
            raise ValueError(f"step {i + 1}: a joint jumps {jump:.0f} deg (more than {max_jump_deg:.0f})")
    settings = {
        "open": (gripper_open, 10.0, True),
        "close": (gripper_closed, 2.0, False),
    }  # a closed grip stalls

    def gripper(key: str, where: str) -> Move:
        value, timeout, settle = settings[grip[key]]
        return Move(f"{where}: gripper {grip[key]}", {"gripper.pos": value}, timeout, settle)

    moves = [gripper(START, "start")] if START in grip else []
    for i, joints in enumerate(steps.joints):
        label = poses.labels[marks.index(i)] if i in marks else None
        where = f"{label} {np.round(100 * dense[i], 1).tolist()} cm" if label else f"step {i + 1}"
        moves.append(Move(where, dict(joints), 10.0, True))
        if label in grip:
            moves.append(gripper(label, label))
    return Execution(poses, steps, tuple(moves))


def run_execution(
    client: "FelixClient",
    model: XleRobotModel,
    execution: Execution,
    max_vel: float = 20.0,
    max_drift_deg: float = 3.0,
    confirm: Callable[[str], bool] | None = None,
) -> Capture:
    """Send `execution`'s moves (see the module docstring) and return a capture taken after, at home.

    `confirm(label)` is asked before each move when given; False stops (and the robot goes home).
    """
    arm, poses = execution.poses.arm, execution.poses
    key = f"{arm}_arm"
    now = client.get_state()[key]
    seen = poses.capture.arms[arm]
    first = execution.steps.joints[0]
    drift = max(abs(now[k] - seen[k]) for k in first)
    if drift > max_drift_deg:
        raise RuntimeError(
            f"the {arm} arm is {drift:.1f} deg from where the poses' capture saw it: capture again"
        )
    with client.session():
        for move in execution.moves:
            if confirm is not None and not confirm(move.label):
                raise RuntimeError(f"stopped before {move.label!r}: the robot goes home")
            status = client.move_joints(
                {key: move.targets}, max_vel=max_vel, max_accel=2 * max_vel, timeout=move.timeout
            )
            reached = bool((status.get("result") or {}).get("reached"))
            if status["state"] != "done" or (move.settle and not reached):
                raise RuntimeError(f"{move.label}: {status['state']}, did not settle; the robot goes home")
        status = client.home()
    return Capture.from_snapshot(client.capture(after=status.get("finished_at")), model)
