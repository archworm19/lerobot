"""Where can an arm put its gripper? Inverse kinematics on the calibrated model: `check_poses` and `approach_poses`.

**The tool point** is the midpoint between the open jaws, where the object's center goes. The pinch frame's origin is
where the closed jaws meet, which is the fixed jaw; the moving jaw opens along the pinch frame's -x. So with the jaws
`width` apart (the object's width), the tool point is `width / 2` along -x from the pinch point. Width 0 makes it the
pinch point itself.

**Pitch** is the gripper axis' angle above the horizontal (the axis is the pinch frame's +z, from the wrist to the
fingertips): -90 deg points straight down, 0 is horizontal, positive points up. The SO-101 keeps its gripper in the
vertical plane through the shoulder (lift, elbow and wrist flex all bend in it), so the heading follows from the
shoulder pan and is not a choice. **Jaws across** a direction makes the jaw axis (pinch x) perpendicular to it: across
the vertical (the default), the jaws close horizontally, as an upright object or one lying on the table needs.

**Solve.** `scipy.optimize.least_squares` puts the tool point on the target, the axis at the pitch and the jaws
across the direction: 3 + 1 + 1 constraints on the arm's five joints, so the solutions are isolated (elbow up or
down, ...), except pointing straight down, where the jaws are horizontal at any roll. A weak pull picks among them:
toward the previous solution, so consecutive poses get nearby joints; for a first pose, with none to follow, also
toward the neutral pose (all joints at 0, mid-range; the roll at 0 keeps the wrist camera's cable slack). Within the
tolerances, staying close wins over a tighter fit (the pinch axis is not quite the roll axis, so the roll trades a
fraction of a degree of pitch; chasing it would swing the roll between neighbors). When that pulled solve misses,
a second one from there with the pulls almost off meets the constraints, so the pulls never cost reach at the edge
of the workspace. After the seed, random starts (without the pulls, which would steer them all alike) run only
until one works. The Jacobian is analytic (`XleRobotModel.frame_jacobian`). A pose is reachable when the solution is
within
`POSITION_TOL_M`, `PITCH_TOL_DEG` and `JAW_TOL_DEG`, inside the joint limits (times `LIMIT_MARGIN`; the wrist roll
within `MAX_ROLL_DEG`).

**The tools.** `check_poses` solves explicit points at one pitch, each from the last solution. `approach_poses` finds
the pitch for a set of targets (say, an object to take and where it goes): it sweeps the pitch range from the
preferred end and, at each pitch and for each target, solves the target, derives the points `back` along the gripper
axis and `up` from the solved pose, and solves those. The first pitch where every target works and whose neighbors
in the sweep work too (a margin from the edge of what the arm can do) wins; failing that, the first that works at
all. The derived points depend on the solved heading, so they are checked at every candidate pitch, not after.

This checks poses only: not the motion between them, and not collisions of the arm's links.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG, arm_pose_degrees
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame
from lerobot.utils.import_utils import _scipy_available, require_package

from .objects import Capture, Poses

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

LIMIT_MARGIN = 0.95  # joints stay within this fraction of their URDF limits
POSITION_TOL_M = 0.005
PITCH_TOL_DEG = 5.0
JAW_TOL_DEG = 5.0
DOWN_DEG = -90.0
VERTICAL = (0.0, 0.0, 1.0)
FULL_RANGE_DEG = (-90.0, 30.0)  # swept to say which pitches would work, after `approach_poses` finds none
# Residual scales: how far from its goal each term may be before it costs as much as one unit.
POSITION_SCALE_M = 0.002
AXIS_SCALE = 0.02  # on differences of unit vectors (about 1 degree)
SEED_SCALE_RAD = 3.0  # a weak pull toward the previous solution
NEUTRAL_SCALE_RAD = 3.0  # a weak pull toward the neutral pose
POLISH_PULL = 0.01  # the pulls' weight in the second, polishing solve
STARTS = 12  # solves per pose at most: the seed, then random ones until one works (6 missed ~5%)


def pitch_of(pose: np.ndarray) -> float:
    """The pitch (deg) of a pinch frame pose's +z axis: -90 = straight down, 0 = horizontal."""
    axis = pose[:3, 2]
    return float(np.degrees(np.arctan2(axis[2], np.hypot(axis[0], axis[1]))))


def heading_of(pose: np.ndarray) -> float:
    """The heading (deg, root frame, from +x toward +y) of a pinch frame pose's +z axis; NaN when it is vertical."""
    axis = pose[:3, 2]
    return (
        float(np.degrees(np.arctan2(axis[1], axis[0]))) if np.hypot(axis[0], axis[1]) > 1e-3 else float("nan")
    )


@dataclass(frozen=True)
class Solution:
    """One solved pose."""

    x: np.ndarray  # the arm's five joints (rad)
    joints: dict[str, float]  # the same in LeRobot degrees
    frame: np.ndarray  # (4, 4) X_root_pinch
    position_error_m: float  # of the tool point
    pitch_error_deg: float  # from the requested pitch (0 when none was requested)
    jaw_error_deg: (
        float  # how far the jaws are from across the requested direction (0 when none was requested)
    )
    ok: bool


class ArmIK:
    """The solver for one arm: its joints, their limits, the tool point and the rest of the robot's pose."""

    def __init__(
        self,
        model: XleRobotModel,
        arm: str,
        arms: Mapping[str, Mapping[str, float]],
        head_ticks: Mapping[str, int] | None = None,
        width: float = 0.0,
        roll_deg: float | None = None,
    ):
        """Set up `arm`'s solver.

        `arms` / `head_ticks` fix the rest of the robot and give the first seed; `width` places the tool point (see
        the module docstring); `roll_deg` pins the wrist roll.
        """
        require_package("scipy", extra="felix_jr")
        self.model, self.arm = model, arm
        self.joint_names = [arm_joint(arm, j) for j in SO101_JOINTS]
        self.index = [model.q_index(j) for j in self.joint_names]
        limits = [model.limits(j) for j in self.joint_names]
        self.lower = np.array([lo for lo, _ in limits]) * LIMIT_MARGIN
        self.upper = np.array([hi for _, hi in limits]) * LIMIT_MARGIN
        roll = SO101_JOINTS.index(
            "wrist_roll"
        )  # the wrist camera's cable limits the roll (as in calibration)
        self.lower[roll] = max(self.lower[roll], -np.radians(MAX_ROLL_DEG))
        self.upper[roll] = min(self.upper[roll], np.radians(MAX_ROLL_DEG))
        if roll_deg is not None:
            pinned = np.radians(roll_deg)
            if not self.lower[roll] <= pinned <= self.upper[roll]:
                raise ValueError(f"roll {roll_deg} deg is outside the allowed wrist roll")
            self.lower[roll] = max(self.lower[roll], pinned - 1e-4)
            self.upper[roll] = min(self.upper[roll], pinned + 1e-4)
        self.offset = np.array([-width / 2, 0.0, 0.0])  # the tool point in the pinch frame
        self.q = model.joint_vector(arms, head_ticks or None)
        self.start = self.clip(self.q[self.index])
        self.neutral = self.clip(np.zeros(len(SO101_JOINTS)))

    def clip(self, x: np.ndarray) -> np.ndarray:
        """`x` strictly inside the limits (least_squares wants a strictly feasible start)."""
        return np.clip(x, self.lower + 1e-6, self.upper - 1e-6)

    def frame(self, x: np.ndarray) -> np.ndarray:
        """X_root_pinch with the arm's joints at `x`."""
        return self.model.frame_pose(pinch_frame(self.arm), self._q(x))

    def tool(self, frame: np.ndarray) -> np.ndarray:
        """The tool point of a pinch frame."""
        return frame[:3, :3] @ self.offset + frame[:3, 3]

    def _q(self, x: np.ndarray) -> np.ndarray:
        q = self.q.copy()
        q[self.index] = x
        return q

    def solve(
        self,
        target: np.ndarray,
        pitch_deg: float | None,
        seed: np.ndarray,
        rng: np.random.Generator,
        starts: int = STARTS,
        across: np.ndarray | None = None,
        neutral: bool = False,
    ) -> Solution:
        """The first of up to `starts` solves that works, from `seed` then random ones; else the closest.

        The tool point at `target`, the axis at `pitch_deg` (None: any), the jaws across `across` (a unit vector;
        None: any). The seeded solve stays near `seed`, and with `neutral` (for a first pose, with no previous one
        to follow) near the neutral pose too.
        """
        to_neutral = 1.0 if neutral else 0.0
        pitch = None if pitch_deg is None else np.radians(pitch_deg)
        n = len(SO101_JOINTS)

        def terms(x: np.ndarray, pull: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
            frame, jacobian = self.model.frame_jacobian(pinch_frame(self.arm), self._q(x), self.joint_names)
            rotation, linear, angular = frame[:3, :3], jacobian[:3], jacobian[3:]
            lever = rotation @ self.offset
            rows = [(frame[:3, 3] + lever - target) / POSITION_SCALE_M]
            jac = [(linear + np.cross(angular.T, lever).T) / POSITION_SCALE_M]
            if (
                pitch is not None
            ):  # the axis vs the nearest axis at `pitch` (same heading): smooth, even at -90
                axis = rotation[:, 2]
                d_axis = np.cross(angular.T, axis).T
                horizontal = np.hypot(axis[0], axis[1])
                heading = axis[:2] / horizontal if horizontal > 1e-9 else np.array([1.0, 0.0])
                rows.append((axis - np.array([*(np.cos(pitch) * heading), np.sin(pitch)])) / AXIS_SCALE)
                d_goal = np.zeros((3, 3))
                if horizontal > 1e-9:
                    d_goal[:2, :2] = np.cos(pitch) * (np.eye(2) - np.outer(heading, heading)) / horizontal
                jac.append((np.eye(3) - d_goal) @ d_axis / AXIS_SCALE)
            if across is not None:
                jaw = rotation[:, 0]
                rows.append(np.array([jaw @ across]) / AXIS_SCALE)
                jac.append((across @ np.cross(angular.T, jaw).T)[None] / AXIS_SCALE)
            rows += [
                pull * (x - seed) / SEED_SCALE_RAD,
                to_neutral * pull * (x - self.neutral) / NEUTRAL_SCALE_RAD,
            ]
            jac += [pull * np.eye(n) / SEED_SCALE_RAD, to_neutral * pull * np.eye(n) / NEUTRAL_SCALE_RAD]
            return np.concatenate(rows), np.vstack(jac)

        best: tuple[float, Solution] | None = None

        def run(x0: np.ndarray, pull: float) -> Any:
            return least_squares(
                lambda x: terms(x, pull)[0],
                self.clip(x0),
                jac=lambda x: terms(x, pull)[1],
                bounds=(self.lower, self.upper),
                max_nfev=200,
            )

        for start in range(starts):
            if start == 0:
                # from the seed, with the pulls: within the tolerances, staying close beats a tighter fit
                result = run(seed, 1.0)
                solution = self._solution(result.x, target, pitch_deg, across)
                if solution.ok:
                    return solution
                result = run(
                    result.x, POLISH_PULL
                )  # else polish: nearly without the pulls, to meet the constraints
            else:  # random starts look for any solution: pulls toward the seed or neutral would steer them all alike
                result = run(np.asarray(rng.uniform(self.lower, self.upper)), POLISH_PULL)
            solution = self._solution(result.x, target, pitch_deg, across)
            if solution.ok:  # the first that works: from the seed when it can, so neighbors stay close
                return solution
            if best is None or result.cost < best[0]:
                best = (result.cost, solution)
        assert best is not None
        return best[1]

    def _solution(
        self, x: np.ndarray, target: np.ndarray, pitch_deg: float | None, across: np.ndarray | None
    ) -> Solution:
        frame = self.frame(x)
        error = float(np.linalg.norm(self.tool(frame) - target))
        pitch_error = 0.0 if pitch_deg is None else abs(pitch_of(frame) - pitch_deg)
        jaw_error = (
            0.0 if across is None else float(np.degrees(np.arcsin(min(1.0, abs(frame[:3, 0] @ across)))))
        )
        joints = arm_pose_degrees(self.model, self.arm, self._q(x))
        return Solution(
            x=x,
            joints={k: v for k, v in joints.items() if k != "gripper.pos"},
            frame=frame,
            position_error_m=error,
            pitch_error_deg=pitch_error,
            jaw_error_deg=jaw_error,
            ok=error <= POSITION_TOL_M and pitch_error <= PITCH_TOL_DEG and jaw_error <= JAW_TOL_DEG,
        )


def check_poses(
    model: XleRobotModel,
    arm: str,
    capture: Capture,
    points: np.ndarray,
    pitch_deg: float | None = DOWN_DEG,
    width: float = 0.0,
    across: Sequence[float] | np.ndarray | None = VERTICAL,
    roll_deg: float | None = None,
    labels: Sequence[str] | None = None,
    starts: int = STARTS,
    rng_seed: int = 0,
) -> Poses:
    """Solve `arm` for the tool point at each of `points` (M, 3), at `pitch_deg`, each pose seeded by the last.

    `capture` fixes the rest of the robot and seeds the first pose; `roll_deg` pins the wrist roll.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ik = ArmIK(model, arm, capture.arms, capture.head_ticks, width, roll_deg)
    jaws = _unit(across)
    rng = np.random.default_rng(rng_seed)
    previous, solutions = ik.start, []
    for i, point in enumerate(points):
        solution = ik.solve(point, pitch_deg, previous, rng, starts, jaws, neutral=i == 0)
        solutions.append(solution)
        if solution.ok:
            previous = solution.x
    names = tuple(labels) if labels is not None else tuple(str(i + 1) for i in range(len(points)))
    return _poses(arm, capture, points, names, solutions, pitch_deg, width, jaws)


def approach_poses(
    model: XleRobotModel,
    arm: str,
    capture: Capture,
    targets: np.ndarray,
    pitch_deg: Sequence[float] = (0.0, -90.0),
    width: float = 0.0,
    back: float = 0.05,
    up: float = 0.05,
    across: Sequence[float] | np.ndarray | None = VERTICAL,
    step_deg: float = 5.0,
    order: Sequence[str] | None = None,
) -> Poses:
    """Find one pitch at which `arm` reaches every target, from `back` along the gripper axis and `up` above.

    See the module docstring. `pitch_deg` is (preferred end, other end). Each target k gives the poses `back{k}`,
    `at{k}`, `up{k}`. `order` lists them as the path goes; by default the first target is entered along the axis
    and left upward (`back1, at1, up1`) and the others are entered from above and left along the axis
    (`up2, at2, back2`): a pick and its places.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    ik = ArmIK(model, arm, capture.arms, capture.head_ticks, width)
    jaws = _unit(across)
    rng = np.random.default_rng(0)
    requested = (float(pitch_deg[0]), float(pitch_deg[1]))
    pitches = _sweep_grid(requested, step_deg)
    found = _sweep(ik, targets, pitches, back, up, jaws, rng)
    every = [all(s is not None for s in row) for row in found]
    chosen = _choose(every)
    if chosen is None:  # say which pitches would work
        pitches = _sweep_grid(FULL_RANGE_DEG, step_deg)
        found = _sweep(ik, targets, pitches, back, up, jaws, rng)
        every = [all(s is not None for s in row) for row in found]
    sweep: dict[str, Any] = {
        "targets": targets,
        "requested_deg": requested,
        "reachable_deg": _runs(pitches, every),
        "target_reachable_deg": tuple(
            _runs(pitches, [row[k] is not None for row in found]) for k in range(len(targets))
        ),
        "back": back,
        "up": up,
    }
    if chosen is None:
        return _poses(arm, capture, np.zeros((0, 3)), (), [], None, width, jaws, **sweep)
    by_label: dict[str, tuple[np.ndarray, Solution]] = {}
    for k, poses in enumerate(found[chosen], start=1):
        assert poses is not None
        for role, pose in zip(("back", "at", "up"), poses, strict=True):
            by_label[f"{role}{k}"] = pose
    if order is None:
        order = ["back1", "at1", "up1"]
        for k in range(2, len(targets) + 1):
            order += [f"up{k}", f"at{k}", f"back{k}"]
    unknown = [label for label in order if label not in by_label]
    if unknown:
        raise ValueError(f"unknown poses {unknown}; have {sorted(by_label)}")
    points = np.array([by_label[label][0] for label in order])
    solutions = [by_label[label][1] for label in order]
    return _poses(arm, capture, points, tuple(order), solutions, float(pitches[chosen]), width, jaws, **sweep)


def pitch_ranges(pitches_deg: Sequence[float], reachable: Sequence[bool]) -> list[tuple[float, float]]:
    """The reachable pitches as (low, high) runs of consecutive ones."""
    order = np.argsort(pitches_deg)
    runs: list[list[float]] = []
    last_ok = False
    for i in order:
        if reachable[i]:
            if last_ok:
                runs[-1][1] = float(pitches_deg[i])
            else:
                runs.append([float(pitches_deg[i]), float(pitches_deg[i])])
        last_ok = bool(reachable[i])
    return [(lo, hi) for lo, hi in runs]


def _unit(direction: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if direction is None:
        return None
    vector = np.asarray(direction, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _poses(
    arm: str,
    capture: Capture,
    points: np.ndarray,
    labels: tuple[str, ...],
    solutions: Sequence[Solution],
    pitch_deg: float | None,
    width: float,
    across: np.ndarray | None,
    **sweep: Any,
) -> Poses:
    return Poses(
        arm,
        capture,
        points,
        labels,
        np.array([s.frame for s in solutions]).reshape(-1, 4, 4),
        tuple(s.joints for s in solutions),
        np.array([s.position_error_m for s in solutions]),
        np.array([s.pitch_error_deg for s in solutions]),
        np.array([s.jaw_error_deg for s in solutions]),
        np.array([s.ok for s in solutions], dtype=bool),
        pitch_deg,
        width,
        across,
        **sweep,
    )


def _sweep_grid(span: tuple[float, float], step: float) -> np.ndarray:
    """From `span[0]` toward `span[1]` (inclusive) every `step`."""
    first, last = span
    n = int(np.floor(abs(last - first) / step + 1e-9))
    return first + np.sign(last - first or 1.0) * step * np.arange(n + 1)


PoseSet = list[tuple[np.ndarray, Solution]]  # (point, solution) for back, at, up


def _sweep(
    ik: ArmIK,
    targets: np.ndarray,
    pitches: np.ndarray,
    back: float,
    up: float,
    across: np.ndarray | None,
    rng: np.random.Generator,
) -> list[list[PoseSet | None]]:
    """Per pitch, per target: its three poses, or None. Each target's solves are seeded by its last success."""
    # TODO: solve the pitches in parallel. They are independent apart from the warm start, which the random starts
    # stand in for. Options: a process pool (each worker loads its own `XleRobotModel` from the URDF, rather than
    # pickling one), or batched solves on all pitches x starts at once.
    seeds = [ik.start] * len(targets)
    first = [True] * len(targets)
    found: list[list[PoseSet | None]] = []
    for pitch in pitches:
        row: list[PoseSet | None] = []
        for k, target in enumerate(targets):
            at = ik.solve(target, float(pitch), seeds[k], rng, STARTS, across, neutral=first[k])
            poses: PoseSet | None = None
            if at.ok:
                tool = ik.tool(at.frame)
                derived = (tool - back * at.frame[:3, 2], tool + np.array([0.0, 0.0, up]))
                others = [ik.solve(point, float(pitch), at.x, rng, STARTS, across) for point in derived]
                if all(s.ok for s in others):
                    poses = [(derived[0], others[0]), (target, at), (derived[1], others[1])]
                seeds[k], first[k] = at.x, False
            row.append(poses)
        found.append(row)
    return found


def _choose(reachable: Sequence[bool]) -> int | None:
    """The first reachable index whose neighbors are reachable too (or lie outside the sweep); else the first one."""
    n = len(reachable)
    for i, ok in enumerate(reachable):
        if ok and all(reachable[j] for j in (i - 1, i + 1) if 0 <= j < n):
            return i
    return next((i for i, ok in enumerate(reachable) if ok), None)


def _runs(pitches: np.ndarray, reachable: Sequence[bool]) -> tuple[tuple[float, float], ...]:
    return tuple(pitch_ranges(list(pitches), reachable))
