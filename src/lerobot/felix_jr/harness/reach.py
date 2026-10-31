"""Can an arm reach a plan's waypoints? Inverse kinematics on the calibrated model, one waypoint after the other.

For each waypoint, `scipy.optimize.least_squares` solves for the arm's five joints so that the pinch point is at the
waypoint and, with `down`, the gripper points straight down (the pinch frame's +z axis, which runs from the wrist to
the pinch point, along the root frame's -z). Five joints for three position and two pointing constraints leaves the
yaw about the vertical to the wrist roll; it is not constrained yet. Each solve starts from the previous waypoint's
solution (the first from the arm's pose in the plan's capture), so consecutive waypoints get nearby joint angles,
plus a few random starts. A waypoint is reachable when the solution is within `POSITION_TOL_M` and
`DOWN_TOL_DEG`, inside the joint limits (times `LIMIT_MARGIN`; the wrist roll within `MAX_ROLL_DEG`, for the wrist
camera's cable).

This checks the waypoints only, not the motion between them, and not collisions of the arm's links.
"""

from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG, arm_pose_degrees
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame
from lerobot.utils.import_utils import _scipy_available, require_package

from .objects import Plan, Reach

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

LIMIT_MARGIN = 0.95  # joints stay within this fraction of their URDF limits
POSITION_TOL_M = 0.005
DOWN_TOL_DEG = 5.0
# Residual scales: how far from its goal each term may be before it costs as much as one unit.
POSITION_SCALE_M = 0.002
DOWN_SCALE = 0.02  # on the difference of unit vectors (about 1 degree)
SEED_SCALE_RAD = 3.0  # a weak pull toward the previous solution


def check_reach(
    model: XleRobotModel,
    arm: str,
    plan: Plan,
    down: bool = True,
    starts: int = 6,
    rng_seed: int = 0,
    roll_deg: float | None = None,
) -> Reach:
    """Solve `arm`'s joints for every waypoint of `plan` (see the module docstring).

    `roll_deg` pins the wrist roll, so the gripper keeps one yaw along the whole plan (left free, the roll drifts
    from waypoint to waypoint and can flip between neighbors).
    """
    require_package("scipy", extra="felix_jr")
    index = [model.q_index(arm_joint(arm, j)) for j in SO101_JOINTS]
    limits = [model.limits(arm_joint(arm, j)) for j in SO101_JOINTS]
    lower = np.array([lo for lo, _ in limits]) * LIMIT_MARGIN
    upper = np.array([hi for _, hi in limits]) * LIMIT_MARGIN
    roll = SO101_JOINTS.index("wrist_roll")  # the wrist camera's cable limits the roll (as in calibration)
    lower[roll] = max(lower[roll], -np.radians(MAX_ROLL_DEG))
    upper[roll] = min(upper[roll], np.radians(MAX_ROLL_DEG))
    if roll_deg is not None:
        pinned = np.radians(roll_deg)
        if not lower[roll] <= pinned <= upper[roll]:
            raise ValueError(f"roll {roll_deg} deg is outside the allowed wrist roll")
        lower[roll], upper[roll] = max(lower[roll], pinned - 1e-4), min(upper[roll], pinned + 1e-4)
    capture = plan.capture
    q = model.joint_vector(capture.arms, capture.head_ticks or None)
    previous = np.clip(q[index], lower + 1e-6, upper - 1e-6)
    rng = np.random.default_rng(rng_seed)
    poses, errors, tilts = [], [], []

    def pinch(x: np.ndarray) -> np.ndarray:
        qq = q.copy()
        qq[index] = x
        return model.frame_pose(pinch_frame(arm), qq)

    for target in plan.waypoints:
        seed = previous

        def residuals(x: np.ndarray, target: np.ndarray = target, seed: np.ndarray = seed) -> np.ndarray:
            pose = pinch(x)
            rows = [(pose[:3, 3] - target) / POSITION_SCALE_M]
            if down:
                rows.append((pose[:3, 2] - np.array([0.0, 0.0, -1.0])) / DOWN_SCALE)
            rows.append((x - seed) / SEED_SCALE_RAD)
            return np.concatenate(rows)

        best = None
        for start in range(starts):
            x0 = seed if start == 0 else rng.uniform(lower, upper)
            x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)  # least_squares wants a strictly feasible start
            result = least_squares(residuals, x0, jac="2-point", bounds=(lower, upper), max_nfev=200)
            if best is None or result.cost < best.cost:
                best = result
        assert best is not None
        pose = pinch(best.x)
        error = float(np.linalg.norm(pose[:3, 3] - target))
        tilt = float(np.degrees(np.arccos(np.clip(-pose[2, 2], -1.0, 1.0))))
        qq = q.copy()
        qq[index] = best.x
        poses.append({k: v for k, v in arm_pose_degrees(model, arm, qq).items() if k != "gripper.pos"})
        errors.append(error)
        tilts.append(tilt)
        if error <= POSITION_TOL_M and (not down or tilt <= DOWN_TOL_DEG):
            previous = best.x
    errors_m, tilts_deg = np.array(errors), np.array(tilts)
    ok = (errors_m <= POSITION_TOL_M) & ((tilts_deg <= DOWN_TOL_DEG) | (not down))
    return Reach(plan, arm, tuple(poses), errors_m, tilts_deg, ok, down)
