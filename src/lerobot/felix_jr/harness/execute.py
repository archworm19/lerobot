"""Running solved poses on the arm: `prepare_execution` (software) and `run_execution` (moves the arm).

`prepare_execution` turns `Poses` into an `Execution`, the exact moves to send: the path through the poses is cut
into straight steps of at most `step` (so the joint-space moves between them stay close to the drawn lines), each
solved from the last one's joints with its pitch and roll interpolated between the poses (`reach.solve_path`). It
refuses if a step misses or a joint jumps more than `max_jump_deg` between steps. Each move settles within a
tolerance: `pose_tol_deg` at the labeled poses, the looser `step_tol_deg` at the steps between them (they are there to
stay near the drawn line, not to be exact; a loaded, extended arm does not get within 1.5 deg). Gripper actions go at
the poses `grip` names (and "start", before the first move); there are none by default.

`run_execution` sends the moves one by one, after checking that the arm is where the poses' capture saw it. Two kinds
of failure, two answers (`docs/sessions.md`, pen1):

- **A move does not settle** (the client is alive and in control): the arm stops and holds where it is, a capture is
  taken there, and `StoppedError` is raised with it and each joint's remaining error, for the model to look and re-plan.
  Going home instead would be an unplanned move, starting next to whatever the arm was handling.
- **Nobody is in control** (an error, Ctrl+C, a dropped connection, a crashed client): the moves run inside
  `client.session()`, so the robot goes home, from the client or, when the heartbeats stop, from the server.

At the end it goes home, or with `hold` stays at the last pose (to look at where the gripper really is), and returns a
capture taken there. It does not ask anyone: the caller (`api.Harness`) does, before it runs.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import ArmPath, Capture, Execution, HeldObject, Move, Poses
from .reach import solve_path

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

SKELETON = (
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
)
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


def prepare_execution(
    model: XleRobotModel,
    poses: Poses,
    grip: Mapping[str, str] | None = None,
    gripper_open: float = 60.0,
    gripper_closed: float = 1.0,
    step: float = 0.01,
    max_jump_deg: float = 20.0,
    pose_tol_deg: float = 2.5,
    step_tol_deg: float = 4.0,
) -> Execution:
    """The moves through `poses` (see the module docstring); raises ValueError if they cannot be run.

    `grip` maps a pose label (or "start") to "open" / "close".
    """
    if not len(poses.points) or not poses.ok.all():
        raise ValueError(f"the {poses.arm} arm does not reach every pose: {poses.describe()}")
    grip = dict(grip or {})
    unknown = [k for k in grip if k != START and k not in poses.labels]
    if unknown or any(a not in GRIPPER_ACTIONS for a in grip.values()):
        raise ValueError(
            f"bad grip {grip}: keys are pose labels {poses.labels} or {START!r}, actions open/close"
        )
    dense, marks = densify(poses.points, step)
    steps = solve_path(model, poses, dense, marks)
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
        moves.append(Move(where, dict(joints), 10.0, True, pose_tol_deg if label else step_tol_deg))
        if label in grip:
            moves.append(gripper(label, label))
    return Execution(poses, steps, tuple(moves))


class StoppedError(RuntimeError):
    """A move did not settle: the arm stopped and holds there.

    `capture` was taken there; `errors` is each joint's measured minus target (deg) when it stopped, `at` the move's
    label.
    """

    def __init__(self, at: str, errors: Mapping[str, float], capture: Capture, tol: float):
        """Record where it stopped, by how much each joint missed, and the capture taken there."""
        worst = ", ".join(f"{k.removesuffix('.pos')} {v:+.1f}" for k, v in errors.items() if abs(v) > tol)
        super().__init__(
            f"{at}: did not settle within {tol:g} deg ({worst or 'no joint over it now'}); the arm stopped and "
            "holds there: look at the capture taken there and re-plan from it"
        )
        self.at, self.errors, self.capture = at, dict(errors), capture


def run_execution(
    client: "FelixClient",
    model: XleRobotModel,
    execution: Execution,
    max_vel: float = 20.0,
    max_drift_deg: float = 3.0,
    confirm: Callable[[str], bool] | None = None,
    hold: bool = False,
) -> Capture:
    """Send `execution`'s moves (see the module docstring) and return a capture taken after.

    The capture is at home, or with `hold` at the last pose, which the arm then holds. A move that does not settle
    raises `StoppedError` (the arm holds there). `confirm(label)` is asked before each move when given; False stops, and the
    robot goes home.
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
    with client.session():  # home if nobody is in control: an error, Ctrl+C, a lost connection
        for move in execution.moves:
            if confirm is not None and not confirm(move.label):
                raise RuntimeError(f"stopped before {move.label!r}: the robot goes home")
            status = client.move_joints(
                {key: move.targets},
                max_vel=max_vel,
                max_accel=2 * max_vel,
                tol=move.tol,
                timeout=move.timeout,
            )
            if status["state"] != "done":  # stopped or failed on the server: not ours to re-plan
                raise RuntimeError(
                    f"{move.label}: {status['state']} ({status.get('reason')}); the robot goes home"
                )
            if move.settle and not (status.get("result") or {}).get("reached"):
                client.stop()
                state = client.get_state()[key]
                errors = {k: state[k] - v for k, v in move.targets.items() if k in state}
                client.keep_holding()  # the session ends here without going home
                capture = Capture.from_snapshot(client.capture(), model)
                raise StoppedError(move.label, errors, capture, move.tol)
        if hold:
            client.keep_holding()
        else:
            status = client.home()
    return Capture.from_snapshot(client.capture(after=status.get("finished_at")), model)


def arm_path(model: XleRobotModel, execution: Execution, every: int = 5, first_samples: int = 8) -> ArmPath:
    """The arm's skeleton along `execution`: sampled over the first move, at every pose, and every `every`-th step.

    The first move is from where the poses' capture saw the arm to the first step, all joints together (as
    `move_joints` runs it: a straight line in joint space), sampled `first_samples` times. The return home at the end
    is not drawn: the home pose is recorded on the robot.
    """
    arm, capture = execution.poses.arm, execution.poses.capture
    steps = execution.steps
    start = capture.arms[arm]
    joints: list[Mapping[str, float]] = []
    kinds: list[str] = []
    for t in np.linspace(0.0, 1.0, first_samples, endpoint=False):
        joints.append({k: start[k] + t * (v - start[k]) for k, v in steps.joints[0].items()})
        kinds.append("first move")
    for i, (point, step) in enumerate(zip(steps.points, steps.joints, strict=True)):
        at = [
            label
            for label, p in zip(execution.poses.labels, execution.poses.points, strict=True)
            if np.allclose(p, point)
        ]
        if at or i % every == 0:
            joints.append(step)
            kinds.append(at[0] if at else "step")
    skeletons = []
    for pose in joints:
        q = model.joint_vector({arm: pose}, capture.head_ticks or None)
        skeletons.append([model.frame_pose(f"{arm}_{link}", q)[:3, 3] for link in SKELETON])
    return ArmPath(arm, np.array(skeletons), tuple(kinds))


def held_object(
    poses: Poses,
    length: float,
    grasp: str,
    release: str,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> HeldObject:
    """The object taken at pose `grasp` and let go at `release`, carried rigidly with the jaws in between.

    At `grasp` its center is the tool point and its axis is `axis` (the object's long direction, the end that is up
    first; default vertical, an upright object). Whatever the gripper does to it on the way shows: a roll of the
    wrist turns it.
    """
    if grasp not in poses.labels or release not in poses.labels:
        raise ValueError(f"no pose {grasp!r} or {release!r}; have {poses.labels}")
    first, last = poses.labels.index(grasp), poses.labels.index(release)
    if last < first:
        raise ValueError(f"{release!r} comes before {grasp!r}")
    up = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    frame = poses.frames[first]
    center_local = np.array([-poses.width / 2, 0.0, 0.0])  # the tool point, in the pinch frame
    axis_local = frame[:3, :3].T @ up
    m = len(poses.labels)
    centers, axes, tilt = np.full((m, 3), np.nan), np.full((m, 3), np.nan), np.full(m, np.nan)
    held: np.ndarray = np.zeros(m, dtype=bool)
    for i in range(first, last + 1):
        f = poses.frames[i]
        centers[i] = f[:3, 3] + f[:3, :3] @ center_local
        axes[i] = f[:3, :3] @ axis_local
        tilt[i] = np.degrees(np.arccos(np.clip(axes[i] @ up, -1.0, 1.0)))
        held[i] = True
    return HeldObject(poses, length, centers, axes, held, tilt)
