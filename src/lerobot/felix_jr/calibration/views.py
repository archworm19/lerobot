"""Choosing what the camera looks at during calibration: head and arm joints that show an arm's pinch point.

`find_view` solves for the head's pan and tilt and the arm's five joints together, with `scipy.optimize.least_squares`
over the model (`XleRobotModel.predict`), so that the pinch point:

- is near the middle of the picture (measured as the angle off the optical axis, which stays smooth even when a
  trial pose puts the point behind the camera);
- is at a target position (optional: from `plan_targets`);
- is seen from the arm's visible side: the gripper's hinge axis points along the line of sight, so the jaws meet
  in profile instead of hiding behind each other or behind the wrist camera's board;
- stays a clearance above the floor (the model's `floor_z`, if known);
- and the joints stay near a seed pose, within the joint limits (bounds; the head's recorded limits if given).

`plan_targets` spreads target pinch points over the workspace: random arm poses around a seed pose, kept above the
floor, then the most spread-out pinch points (farthest-point selection in Cartesian space). Each target comes with
the arm pose that reaches it, a good seed for `find_view`.

Everything here uses the model's current estimates (or a `theta` passed in), so views get better as the clicks
pin the model down. Nothing checks for collisions: the region is small, the arm moves slowly, a person watches.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.xle_robot_model import (
    HEAD_JOINTS,
    Prediction,
    XleRobotModel,
    arm_joint,
)
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

CLOSED = 1.0  # gripper.pos of a closed gripper (0-100): the closed tip is the calibration target
LIMIT_MARGIN = 0.8  # the arm's joints stay within this fraction of their URDF limits
MAX_ROLL_DEG = 130.0  # how far the wrist may roll to show the pinch point (watch the wrist camera's cable)
FACING_MAX_DEG = 50.0  # the hinge axis must be within this of the line of sight for the pinch point to show
VIEW_MARGIN_PX = 60  # pixels from the image border the pinch point must stay
FLOOR_CLEARANCE = (
    0.04  # meters the pinch point stays above the floor (the model is good to a centimeter or two)
)
# Residual scales: how far from its goal each term may be before it costs as much as one unit.
CENTER_SCALE_PX = 20.0
TARGET_SCALE_M = 0.02
FACING_SCALE = 0.2
FLOOR_SCALE_M = 0.005
SEED_SCALE_RAD = 1.0
# +/- degrees around the seed pose that `plan_targets` samples arm poses from.
DEFAULT_DELTA = {
    "shoulder_pan": 25.0,
    "shoulder_lift": 20.0,
    "elbow_flex": 20.0,
    "wrist_flex": 20.0,
    "wrist_roll": 30.0,
}


@dataclass
class Target:
    """A pinch point position to look at (root frame, meters) and an arm pose that reaches it (LeRobot degrees)."""

    point: np.ndarray
    arm_pose: dict[str, float]


@dataclass
class View:
    """Joints that show `arm`'s pinch point: what to command, and what the model expects to see."""

    arm: str
    arm_pose: dict[str, float]  # LeRobot degrees, gripper closed
    head_ticks: dict[str, int]
    prediction: Prediction
    facing_deg: float  # angle between the hinge axis and the line of sight, on the visible side
    ok: bool  # in the picture with a margin, facing the camera, above the floor
    reason: str  # why not, if not ok


def arm_pose_degrees(model: XleRobotModel, arm: str, q: np.ndarray) -> dict[str, float]:
    """LeRobot's `{"shoulder_pan.pos": deg, ..., "gripper.pos": closed}` for `arm`'s joints in `q`."""
    pose = {f"{j}.pos": float(np.degrees(q[model.q_index(arm_joint(arm, j))])) for j in SO101_JOINTS}
    return {**pose, "gripper.pos": CLOSED}


def head_bounds(
    model: XleRobotModel, head_limits: Mapping[str, tuple[int, int]] | None
) -> dict[str, tuple[float, float]]:
    """(lower, upper) measured angles of the head's URDF joints: from raw tick limits per servo if given (see
    `collect.head_limits`), else the URDF's."""
    bounds = {}
    for servo, joint in HEAD_JOINTS.items():
        if head_limits is None or servo not in head_limits:
            bounds[joint] = model.limits(joint)
            continue
        ends = [model.head_angles({**model.head_refs, servo: ticks})[joint] for ticks in head_limits[servo]]
        bounds[joint] = (min(ends), max(ends))
    return bounds


def find_view(
    model: XleRobotModel,
    arm: str,
    seed: np.ndarray,
    target: np.ndarray | None = None,
    head_limits: Mapping[str, tuple[int, int]] | None = None,
    theta: np.ndarray | Mapping[str, float] | None = None,
    max_roll_deg: float = MAX_ROLL_DEG,
    starts: int = 4,
    rng_seed: int = 0,
) -> View:
    """Head and arm joints that show `arm`'s pinch point (see the module docstring), starting from `seed` (a `q`).

    The side the camera must see is `model.visible_side[arm]`, the floor `model.floor_z`. Besides `seed`, `starts - 1`
    random starts near it are tried and the best solution is kept.
    """
    require_package("scipy", extra="felix_jr")
    joints = [*HEAD_JOINTS.values(), *(arm_joint(arm, j) for j in SO101_JOINTS)]
    index = [model.q_index(j) for j in joints]
    bounds = head_bounds(model, head_limits)
    for j in SO101_JOINTS:
        low, high = model.limits(arm_joint(arm, j))
        bounds[arm_joint(arm, j)] = (low * LIMIT_MARGIN, high * LIMIT_MARGIN)
    roll = arm_joint(arm, "wrist_roll")
    max_roll = np.radians(max_roll_deg)
    bounds[roll] = (max(bounds[roll][0], -max_roll), min(bounds[roll][1], max_roll))
    lower = np.array([bounds[j][0] for j in joints])
    upper = np.array([bounds[j][1] for j in joints])
    seed = np.asarray(seed, dtype=float)
    x_seed = np.clip(seed[index], lower, upper)
    sign = 1.0 if model.visible_side[arm] == "positive" else -1.0
    c = model.intrinsics

    def configuration(x: np.ndarray) -> np.ndarray:
        q = seed.copy()
        q[index] = x
        return q

    def residuals(x: np.ndarray) -> np.ndarray:
        pred = model.predict(arm, configuration(x), theta)
        x_c, y_c, z_c = pred.p_camera
        rows = [
            c["fx"] * np.arctan2(x_c, z_c) / CENTER_SCALE_PX,
            c["fy"] * np.arctan2(y_c, z_c) / CENTER_SCALE_PX,
            (1.0 - sign * pred.visibility) / FACING_SCALE,
        ]
        if target is not None:
            rows += list((pred.point - target) / TARGET_SCALE_M)
        if model.floor_z is not None:
            rows.append(max(0.0, model.floor_z + FLOOR_CLEARANCE - pred.point[2]) / FLOOR_SCALE_M)
        return np.concatenate([rows, (x - x_seed) / SEED_SCALE_RAD])

    rng = np.random.default_rng(rng_seed)
    best = None
    for start in range(starts):
        x0 = x_seed if start == 0 else np.clip(x_seed + rng.normal(0.0, 0.3, len(x_seed)), lower, upper)
        x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)  # least_squares wants a strictly feasible start
        result = least_squares(residuals, x0, jac="2-point", bounds=(lower, upper), max_nfev=200)
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    q = configuration(best.x)
    pred = model.predict(arm, q, theta)
    facing = float(np.degrees(np.arccos(np.clip(sign * pred.visibility, -1.0, 1.0))))
    reasons = []
    if pred.z <= 0 or not model.in_view(pred.uv, margin=VIEW_MARGIN_PX):
        reasons.append("the pinch point is not in the picture")
    if facing > FACING_MAX_DEG:
        reasons.append(f"the gripper faces the camera only within {facing:.0f} deg")
    if model.floor_z is not None and pred.point[2] < model.floor_z + FLOOR_CLEARANCE - 0.005:
        reasons.append("the pinch point is too close to the floor")
    head_angles = {joint: float(q[model.q_index(joint)]) for joint in HEAD_JOINTS.values()}
    return View(
        arm,
        arm_pose_degrees(model, arm, q),
        model.head_ticks(head_angles),
        pred,
        facing,
        not reasons,
        "; ".join(reasons),
    )


def plan_targets(
    model: XleRobotModel,
    arm: str,
    n: int,
    seed_pose: Mapping[str, float] | None = None,
    delta_scale: float = 1.0,
    theta: np.ndarray | Mapping[str, float] | None = None,
    candidates: int = 400,
    rng_seed: int = 0,
) -> list[Target]:
    """`n` spread-out pinch point positions around the one `seed_pose` reaches (LeRobot degrees; default: the
    wiggle's center pose, all joints at 0), the seed's first.

    Candidates are arm poses within `DEFAULT_DELTA * delta_scale` degrees of the seed (and the joint limits with
    their margin), kept if their pinch point is above the floor; then farthest-point selection on the pinch points.
    """
    seed = {j: float((seed_pose or {}).get(f"{j}.pos", 0.0)) for j in SO101_JOINTS}
    limits = model.joint_limits(arm)
    rng = np.random.default_rng(rng_seed)

    def pose(values: Mapping[str, float]) -> dict[str, float]:
        clipped = {
            j: float(
                np.clip(v, np.degrees(limits[j][0]) * LIMIT_MARGIN, np.degrees(limits[j][1]) * LIMIT_MARGIN)
            )
            for j, v in values.items()
        }
        return {**{f"{j}.pos": v for j, v in clipped.items()}, "gripper.pos": CLOSED}

    poses = [pose(seed)]
    for _ in range(candidates):
        poses.append(
            pose({j: seed[j] + rng.uniform(-1, 1) * DEFAULT_DELTA[j] * delta_scale for j in SO101_JOINTS})
        )
    points = np.array([model.pinch_point(arm, model.joint_vector({arm: p}), theta) for p in poses])
    keep: np.ndarray = np.ones(len(poses), dtype=bool)
    if model.floor_z is not None:
        keep = points[:, 2] >= model.floor_z + FLOOR_CLEARANCE
        keep[0] = True  # the seed stays, even if the model puts it low: find_view lifts it
    chosen = [0]
    distance = np.where(keep, np.linalg.norm(points - points[0], axis=1), -np.inf)
    while len(chosen) < min(n, int(keep.sum())):
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, np.where(keep, np.linalg.norm(points - points[i], axis=1), -np.inf))
    return [Target(points[i], poses[i]) for i in chosen]
