"""Where can an arm put its gripper? One constrained optimization per pose (`solve_pose`); the rest is built on it.

The reasoning is in `docs/harness.md` (1.J). In short:

**The target** is a full gripper pose in the root frame: the tool point's position, and an orientation given as
pitch, heading and roll (`orientation`). The **tool point** is the midpoint between the open jaws, where the object's
center goes: the pinch frame's origin is where the closed jaws meet, the fixed jaw, and the moving jaw opens along
the pinch frame's -x, so with the jaws `width` apart the tool point is `width / 2` along -x from the pinch point.
**Pitch** is the gripper axis' angle above horizontal (the axis is the pinch frame's +z, wrist to fingertips): -90
straight down, 0 horizontal. **Heading** is the axis' direction about the vertical (deg from root +x toward +y).
**Roll** turns the jaws about the axis: 0 has the jaw axis horizontal, along heading + 90 deg; 180 swaps the jaws.

**The optimization.** The arm has five joints, a pose has six numbers. So one of them is held and the rest are traded
off: the pitch is held (it decides reach on this arm, the wrist flex limit, and it is the approach's main choice);
position, heading and roll are soft, each weighted by a tolerance the caller gives (an error of one tolerance costs one
unit). Joint limits are hard bounds. `scipy.optimize.least_squares` solves it once, locally, from `init` (default: the
arm's pose in the capture): the start decides the branch (elbow up or down, the jaws on one side or the other). There
are no restarts and no sweeps: trying another pitch, start or tolerance is the caller's (the model's) choice.

Holding the pitch is a stiff penalty (`PITCH_TOL_DEG`) checked afterwards, not an equality constraint: pointing
straight down, an equality on the pitch pins both horizontal components of the axis and its gradient vanishes, which
equality-constrained solvers handle badly. The stiff penalty is the same constraint in the limit and stays smooth.

On this arm the heading mostly follows from the position: the gripper stays in the vertical plane through the
shoulder. So by default the heading is free; asking for one that the position does not allow trades position for it.

This checks poses only: not the motion between them, and not collisions (`docs/harness.md`, 1.K).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG, arm_pose_degrees
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame
from lerobot.utils.import_utils import _scipy_available, require_package

from .objects import VERTICAL_COS, Capture, Poses, angles_of

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

LIMIT_MARGIN = 0.95  # joints stay within this fraction of their URDF limits
PITCH_TOL_DEG = 0.1  # the held pitch: its penalty scale
PITCH_OK_DEG = 0.5  # and how far off it may end up and still count as held
POSITION_TOL_M = 0.005
ROLL_TOL_DEG = 10.0
AT_LIMIT_RAD = 1e-3  # a joint this close to a bound is reported as at its limit


def _angle(a: float, b: float) -> float:
    """|a - b| in degrees, wrapped to 0..180."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


@dataclass(frozen=True)
class Solution:
    """One solved pose."""

    x: np.ndarray  # the arm's five joints (rad)
    joints: dict[str, float]  # the same in LeRobot degrees
    frame: np.ndarray  # (4, 4) X_root_pinch
    position_error_m: float  # of the tool point
    pitch_error_deg: float
    heading_error_deg: float  # NaN if the heading was free
    roll_error_deg: float  # NaN if the roll was free
    at_limit: tuple[str, ...]  # joints that ended at a bound
    ok: bool  # within every tolerance, the pitch held


class ArmIK:
    """One arm's joints, their limits, the tool point and the rest of the robot's pose; solves one pose at a time."""

    def __init__(
        self,
        model: XleRobotModel,
        arm: str,
        arms: Mapping[str, Mapping[str, float]],
        head_ticks: Mapping[str, int] | None = None,
        width: float = 0.0,
    ):
        """`arms` / `head_ticks` fix the rest of the robot and give the default start; `width` places the tool point."""
        require_package("scipy", extra="felix_jr")
        self.model, self.arm = model, arm
        self.index = [model.q_index(arm_joint(arm, j)) for j in SO101_JOINTS]
        limits = [model.limits(arm_joint(arm, j)) for j in SO101_JOINTS]
        self.lower = np.array([lo for lo, _ in limits]) * LIMIT_MARGIN
        self.upper = np.array([hi for _, hi in limits]) * LIMIT_MARGIN
        roll = SO101_JOINTS.index(
            "wrist_roll"
        )  # the wrist camera's cable limits the roll (as in calibration)
        self.lower[roll] = max(self.lower[roll], -np.radians(MAX_ROLL_DEG))
        self.upper[roll] = min(self.upper[roll], np.radians(MAX_ROLL_DEG))
        self.offset = np.array([-width / 2, 0.0, 0.0])  # the tool point in the pinch frame
        self.q = model.joint_vector(arms, head_ticks or None)
        self.start = self.clip(self.q[self.index])

    def clip(self, x: np.ndarray) -> np.ndarray:
        """`x` strictly inside the bounds (least_squares wants a strictly feasible start)."""
        return np.clip(x, self.lower + 1e-6, self.upper - 1e-6)

    def radians(self, joints: Mapping[str, float]) -> np.ndarray:
        """LeRobot degrees -> the five joints in radians, inside the bounds."""
        return self.clip(np.radians([joints[f"{j}.pos"] for j in SO101_JOINTS]))

    def frame(self, x: np.ndarray) -> np.ndarray:
        """X_root_pinch with the arm's joints at `x`."""
        q = self.q.copy()
        q[self.index] = x
        return self.model.frame_pose(pinch_frame(self.arm), q)

    def tool(self, frame: np.ndarray) -> np.ndarray:
        """The tool point of a pinch frame."""
        return frame[:3, :3] @ self.offset + frame[:3, 3]

    def solve(
        self,
        position: np.ndarray,
        pitch_deg: float,
        heading_deg: float | None = None,
        roll_deg: float | None = None,
        position_tol: float = POSITION_TOL_M,
        heading_tol_deg: float = 10.0,
        roll_tol_deg: float = ROLL_TOL_DEG,
        init: np.ndarray | None = None,
    ) -> Solution:
        """The pose closest to the target from `init` (see the module docstring); None heading / roll: free."""
        position = np.asarray(position, dtype=np.float64)
        pitch = np.radians(pitch_deg)
        vertical = abs(np.cos(pitch)) < VERTICAL_COS
        heading_goal = None if heading_deg is None else np.radians(heading_deg)
        roll_goal = None if roll_deg is None else np.radians(roll_deg)

        def residuals(x: np.ndarray) -> np.ndarray:
            frame = self.frame(x)
            rotation, axis = frame[:3, :3], frame[:3, 2]
            horizontal = np.hypot(axis[0], axis[1])
            direction = axis[:2] / horizontal if horizontal > 1e-9 else np.array([1.0, 0.0])
            rows = [(self.tool(frame) - position) / position_tol]
            # the held pitch: the axis vs the nearest axis at that pitch (same heading); smooth, even at -90
            goal = np.array([*(np.cos(pitch) * direction), np.sin(pitch)])
            rows.append((axis - goal) / np.radians(PITCH_TOL_DEG))
            if heading_goal is not None and not vertical:
                wanted = np.array([np.cos(heading_goal), np.sin(heading_goal)])
                rows.append(np.cos(pitch) * (direction - wanted) / np.radians(heading_tol_deg))
            if roll_goal is not None:
                if vertical:  # heading and roll are one angle: the jaws' direction is set by both
                    reference = heading_goal if heading_goal is not None else 0.0
                else:
                    reference = np.arctan2(direction[1], direction[0])
                level = np.array([-np.sin(reference), np.cos(reference), 0.0])
                roll = np.arctan2(rotation[:, 0] @ np.cross(axis, level), rotation[:, 0] @ level)
                rows.append(
                    np.array([np.cos(roll) - np.cos(roll_goal), np.sin(roll) - np.sin(roll_goal)])
                    / np.radians(roll_tol_deg)
                )
            return np.concatenate(rows)

        start = self.clip(self.start if init is None else init)
        result = least_squares(residuals, start, jac="2-point", bounds=(self.lower, self.upper), max_nfev=400)
        return self._solution(
            result.x, position, pitch_deg, heading_deg, roll_deg, position_tol, heading_tol_deg, roll_tol_deg
        )

    def _solution(
        self,
        x: np.ndarray,
        position: np.ndarray,
        pitch_deg: float,
        heading_deg: float | None,
        roll_deg: float | None,
        position_tol: float,
        heading_tol_deg: float,
        roll_tol_deg: float,
    ) -> Solution:
        frame = self.frame(x)
        pitch, heading, roll = angles_of(frame[:3, :3], heading_deg if heading_deg is not None else 0.0)
        error = float(np.linalg.norm(self.tool(frame) - position))
        pitch_error = abs(pitch - pitch_deg)
        vertical = abs(np.cos(np.radians(pitch_deg))) < VERTICAL_COS
        heading_error = float("nan") if heading_deg is None or vertical else _angle(heading, heading_deg)
        roll_error = float("nan") if roll_deg is None else _angle(roll, roll_deg)
        q = self.q.copy()
        q[self.index] = x
        names = [
            j
            for j, v, lo, hi in zip(SO101_JOINTS, x, self.lower, self.upper, strict=True)
            if min(v - lo, hi - v) < AT_LIMIT_RAD
        ]
        ok = (
            error <= position_tol
            and pitch_error <= PITCH_OK_DEG
            and not heading_error > heading_tol_deg
            and not roll_error > roll_tol_deg
        )
        return Solution(
            x=x,
            joints={k: v for k, v in arm_pose_degrees(self.model, self.arm, q).items() if k != "gripper.pos"},
            frame=frame,
            position_error_m=error,
            pitch_error_deg=pitch_error,
            heading_error_deg=heading_error,
            roll_error_deg=roll_error,
            at_limit=tuple(names),
            ok=bool(ok),
        )


def solve_pose(
    model: XleRobotModel,
    arm: str,
    capture: Capture,
    position: np.ndarray,
    pitch_deg: float,
    heading_deg: float | None = None,
    roll_deg: float | None = 0.0,
    width: float = 0.0,
    position_tol: float = POSITION_TOL_M,
    heading_tol_deg: float = 10.0,
    roll_tol_deg: float = ROLL_TOL_DEG,
    init: Mapping[str, float] | Poses | None = None,
    label: str = "1",
) -> Poses:
    """Solve one pose (see the module docstring) and return it as `Poses` of one.

    `init` is the start: joints in LeRobot degrees, or `Poses` (its last pose's joints); default the arm's pose in
    `capture`.
    """
    ik = ArmIK(model, arm, capture.arms, capture.head_ticks, width)
    start = None
    if isinstance(init, Poses):
        start = ik.radians(init.joints[-1])
    elif init is not None:
        start = ik.radians(init)
    solution = ik.solve(
        position, pitch_deg, heading_deg, roll_deg, position_tol, heading_tol_deg, roll_tol_deg, start
    )
    target = (float(pitch_deg), heading_deg, roll_deg)
    return _poses(
        arm,
        capture,
        width,
        [np.asarray(position, dtype=np.float64)],
        [label],
        [solution],
        [target],
        (position_tol, heading_tol_deg, roll_tol_deg),
    )


def join_poses(*parts: Poses) -> Poses:
    """Several `Poses` (same arm, capture and jaw width) as one path, in the order given; labels must be unique."""
    if not parts:
        raise ValueError("nothing to join")
    first = parts[0]
    for part in parts[1:]:
        if part.arm != first.arm or part.capture is not first.capture or part.width != first.width:
            raise ValueError("only poses of the same arm, capture and jaw width can be joined")
    labels = tuple(label for part in parts for label in part.labels)
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels repeat: {labels}")
    return Poses(
        first.arm,
        first.capture,
        np.vstack([p.points for p in parts]),
        labels,
        np.concatenate([p.frames for p in parts]),
        tuple(j for p in parts for j in p.joints),
        np.concatenate([p.position_error_m for p in parts]),
        np.concatenate([p.pitch_error_deg for p in parts]),
        np.concatenate([p.heading_error_deg for p in parts]),
        np.concatenate([p.roll_error_deg for p in parts]),
        tuple(a for p in parts for a in p.at_limit),
        np.concatenate([p.ok for p in parts]),
        tuple(t for p in parts for t in p.targets),
        first.width,
        first.tolerances,
    )


def select_poses(poses: Poses, *labels: str) -> Poses:
    """The poses with these labels, in the order given (e.g. one pose of a path, to go only there)."""
    missing = [label for label in labels if label not in poses.labels]
    if missing or not labels:
        raise ValueError(f"no poses {missing or labels}; have {poses.labels}")
    index = [poses.labels.index(label) for label in labels]
    return Poses(
        poses.arm,
        poses.capture,
        poses.points[index],
        tuple(labels),
        poses.frames[index],
        tuple(poses.joints[i] for i in index),
        poses.position_error_m[index],
        poses.pitch_error_deg[index],
        poses.heading_error_deg[index],
        poses.roll_error_deg[index],
        tuple(poses.at_limit[i] for i in index),
        poses.ok[index],
        tuple(poses.targets[i] for i in index),
        poses.width,
        poses.tolerances,
    )


def approach(
    model: XleRobotModel,
    pose: Poses,
    back: float = 0.05,
    up: float = 0.05,
    order: Sequence[str] = ("back", "at", "up"),
) -> Poses:
    """The pose's neighbors: `back` m back along its gripper axis and `up` m above, at the same orientation.

    `pose` is a solved single pose (its label L); the result has poses "L.back", "L" and "L.up" in `order` (e.g.
    ("up", "at", "back") to come down and leave along the axis), each solved from the pose's own joints.
    """
    if len(pose.labels) != 1:
        raise ValueError("approach takes one solved pose")
    label, frame = pose.labels[0], pose.frames[0]
    ik = ArmIK(model, pose.arm, pose.capture.arms, pose.capture.head_ticks, pose.width)
    tool = ik.tool(frame)
    pitch, heading, roll = angles_of(frame[:3, :3])
    points = {"back": tool - back * frame[:3, 2], "at": tool, "up": tool + np.array([0.0, 0.0, up])}
    names = {"back": f"{label}.back", "at": label, "up": f"{label}.up"}
    position_tol, _, roll_tol = pose.tolerances
    start = ik.radians(pose.joints[0])
    parts = []
    for key in order:
        if key == "at":
            parts.append(pose)
            continue
        solution = ik.solve(points[key], pitch, heading, roll, position_tol, 10.0, roll_tol, start)
        parts.append(
            _poses(
                pose.arm,
                pose.capture,
                pose.width,
                [points[key]],
                [names[key]],
                [solution],
                [(pitch, heading, roll)],
                pose.tolerances,
            )
        )
    return join_poses(*parts)


def solve_path(model: XleRobotModel, poses: Poses, points: np.ndarray, marks: Sequence[int]) -> Poses:
    """Solve `points` (the path through `poses` cut into steps; pose i is at `points[marks[i]]`) in order.

    Each step starts from the last one's joints; its pitch and roll are interpolated between the poses it lies
    between (as they were achieved), its heading is free (`execute.prepare_execution`).
    """
    ik = ArmIK(model, poses.arm, poses.capture.arms, poses.capture.head_ticks, poses.width)
    position_tol, heading_tol, roll_tol = poses.tolerances
    achieved = [angles_of(f[:3, :3]) for f in poses.frames]
    start, solutions, targets = ik.radians(poses.joints[0]), [], []
    for i, point in enumerate(points):
        k = next(k for k, m in enumerate(marks) if i <= m)
        a, b = achieved[max(k - 1, 0)], achieved[k]
        t = 1.0 if k == 0 else (i - marks[k - 1]) / (marks[k] - marks[k - 1])
        pitch = a[0] + t * (b[0] - a[0])
        roll = a[2] + t * ((b[2] - a[2] + 180.0) % 360.0 - 180.0)
        solution = ik.solve(point, pitch, None, roll, position_tol, heading_tol, roll_tol, start)
        solutions.append(solution)
        targets.append((pitch, None, roll))
        if solution.ok:
            start = solution.x
    labels = [str(i + 1) for i in range(len(points))]
    return _poses(
        poses.arm, poses.capture, poses.width, list(points), labels, solutions, targets, poses.tolerances
    )


def _poses(
    arm: str,
    capture: Capture,
    width: float,
    points: Sequence[np.ndarray],
    labels: Sequence[str],
    solutions: Sequence[Solution],
    targets: Sequence[tuple[float, float | None, float | None]],
    tolerances: tuple[float, float, float],
) -> Poses:
    return Poses(
        arm,
        capture,
        np.array(points, dtype=np.float64).reshape(-1, 3),
        tuple(labels),
        np.array([s.frame for s in solutions]).reshape(-1, 4, 4),
        tuple(s.joints for s in solutions),
        np.array([s.position_error_m for s in solutions]),
        np.array([s.pitch_error_deg for s in solutions]),
        np.array([s.heading_error_deg for s in solutions]),
        np.array([s.roll_error_deg for s in solutions]),
        tuple(s.at_limit for s in solutions),
        np.array([s.ok for s in solutions], dtype=bool),
        tuple(targets),
        width,
        tolerances,
    )
