"""Aiming the head: the pan and tilt that point the camera at a root-frame point (`api.look_at` moves there).

The camera's optical axis should pass through the target. The head has two joints (pan, tilt) and the camera is not
on their axes, so `least_squares` solves the URDF head angles on the calibrated model (`XleRobotModel.camera_pose`),
from the head's current angles, within the URDF limits. The servos' recorded limits may be tighter: the server clamps
to them, and the new capture shows where the head really ended up.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

TICKS_PER_DEG = 4096 / 360.0


def aim_head(
    model: XleRobotModel, head_ticks: Mapping[str, int], target: np.ndarray
) -> tuple[dict[str, int], float]:
    """Servo ticks that point the camera at root-frame `target` (3,), from the head at `head_ticks`.

    Returns the ticks per servo and how far (deg) the optical axis misses the target there (0 when reachable).
    """
    require_package("scipy", extra="felix_jr")
    target = np.asarray(target, dtype=np.float64)
    q0 = model.joint_vector(None, head_ticks)
    joints = list(HEAD_JOINTS.values())
    index = [model.q_index(j) for j in joints]
    lower = np.array([model.limits(j)[0] for j in joints])
    upper = np.array([model.limits(j)[1] for j in joints])

    def miss(x: np.ndarray) -> np.ndarray:
        q = q0.copy()
        q[index] = x
        pose = model.camera_pose(q)
        to_target = target - pose[:3, 3]
        return pose[:3, 2] - to_target / np.linalg.norm(to_target)

    result = least_squares(miss, np.clip(q0[index], lower + 1e-6, upper - 1e-6), bounds=(lower, upper))
    off = float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(miss(result.x)) / 2))))
    return model.head_ticks(dict(zip(joints, result.x, strict=True))), off


def head_moves(state: Mapping[str, Mapping[str, int]], goal: Mapping[str, int]) -> dict[str, float]:
    """The `head_move` deltas (deg, the server's convention: positive = down / left) from `state` to `goal` ticks."""
    return {
        name: (goal[name] - int(state[name]["ticks"])) / (int(state[name].get("sign", 1)) * TICKS_PER_DEG)
        for name in goal
    }
