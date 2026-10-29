"""Choosing arm poses to click, and steering the head so the tip is in the picture.

`plan_poses` makes a list of arm poses spread over a region around a seed pose (the wiggle test's center pose by
default): random poses within +/- `delta` degrees of the seed, then the `n` most spread-out of them (farthest-point
selection in joint space), so the clicks cover the workspace instead of clustering. The gripper is closed in every
pose, because the closed gripper's tip is the calibration target. Poses stay inside the SO-101's joint limits with
a margin. Nothing here checks for collisions: the region is small, the arm moves slowly, and a person watches it.

`center_head` finds the head pan / tilt (raw ticks) that put the tip, as the current model predicts it, in the
middle of the image. At first the prediction is poor (the head's zero offsets are unknown to about 20 degrees), so
the tip may land elsewhere and the person nudges the head; after a few clicks the fit has pinned the head down and
the tip lands where it was predicted.
"""

from collections.abc import Callable

import numpy as np

from lerobot.felix_jr.calibration.model import (
    ARM_JOINTS,
    HEAD_JOINTS,
    TICKS_PER_TURN,
    CalibrationModel,
    Observation,
)

CENTER_POSE = dict.fromkeys(
    ARM_JOINTS, 0.0
)  # the wiggle test's center: the arm up and forward, tip ~35 cm ahead
CLOSED = 1.0  # gripper.pos of a closed gripper (0-100)
# SO-101 joint limits in degrees (from the URDF, shrunk by `LIMIT_MARGIN`).
JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-97.0, 97.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.0, 163.0),
}
LIMIT_MARGIN = 0.8
DEFAULT_DELTA = {  # +/- degrees around the seed pose
    "shoulder_pan": 25.0,
    "shoulder_lift": 20.0,
    "elbow_flex": 20.0,
    "wrist_flex": 20.0,
    "wrist_roll": 30.0,
}


def plan_poses(
    n: int,
    seed_pose: dict[str, float] | None = None,
    delta_scale: float = 1.0,
    candidates: int = 600,
    rng_seed: int = 0,
    keep: Callable[[dict[str, float]], bool] | None = None,
) -> list[dict[str, float]]:
    """`n` arm poses (`{"shoulder_pan.pos": deg, ..., "gripper.pos": 1.0}`) spread around `seed_pose`.

    The first pose is the seed itself. `delta_scale` scales the region (`DEFAULT_DELTA`). Selection is
    farthest-point: each next pose is the candidate farthest from the ones already chosen, in joint space with
    each joint divided by its region size. `keep(pose)` can veto candidates (e.g. a tip that would be below the
    table); the seed pose itself is always first. If too few candidates pass, fewer than `n` poses come back.
    """
    rng = np.random.default_rng(rng_seed)
    seed = {**CENTER_POSE, **{k.removesuffix(".pos"): v for k, v in (seed_pose or {}).items()}}
    seed = {  # only the arm joints (a pose from `seed_pose_for_view` also carries the gripper), inside the limits
        j: min(max(seed[j], JOINT_LIMITS[j][0] * LIMIT_MARGIN), JOINT_LIMITS[j][1] * LIMIT_MARGIN)
        for j in ARM_JOINTS
    }
    scale = np.array([DEFAULT_DELTA[j] * delta_scale for j in ARM_JOINTS])
    lows = np.array(
        [max(seed[j] - s, JOINT_LIMITS[j][0] * LIMIT_MARGIN) for j, s in zip(ARM_JOINTS, scale, strict=True)]
    )
    highs = np.array(
        [min(seed[j] + s, JOINT_LIMITS[j][1] * LIMIT_MARGIN) for j, s in zip(ARM_JOINTS, scale, strict=True)]
    )
    pool = rng.uniform(lows, highs, size=(candidates, len(ARM_JOINTS)))
    if keep is not None:
        allowed = [
            keep(
                {
                    **{f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, row, strict=True)},
                    "gripper.pos": CLOSED,
                }
            )
            for row in pool
        ]
        pool = pool[np.array(allowed, dtype=bool)]
    chosen = [np.array([seed[j] for j in ARM_JOINTS])]
    distance = np.linalg.norm((pool - chosen[0]) / scale, axis=1) if len(pool) else np.array([])
    while len(chosen) < n and len(pool):
        pick = int(np.argmax(distance))
        chosen.append(pool[pick])
        distance = np.minimum(distance, np.linalg.norm((pool - pool[pick]) / scale, axis=1))
    return [
        {**{f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, pose, strict=True)}, "gripper.pos": CLOSED}
        for pose in chosen
    ]


def center_head(
    model: CalibrationModel,
    theta: dict[str, float],
    arm: dict[str, float],
    head_ticks: dict[str, int],
    limits: dict[str, tuple[int, int]] | None = None,
    target: tuple[float, float] | None = None,
    iterations: int = 6,
) -> dict[str, int] | None:
    """Head ticks that put the predicted tip at `target` (default: the image center), or None if it cannot.

    Gauss-Newton on the two head servos' raw ticks, starting from `head_ticks`, with the model's prediction
    under `theta`. `limits` (`{servo: (low, high)}` raw ticks) keeps the result inside the head's recorded range.
    Returns None when the tip starts behind the camera and no step brings it in front.
    """
    goal = np.array(target if target is not None else (model.camera.width / 2, model.camera.height / 2))
    names = list(HEAD_JOINTS)
    ticks = np.array([float(head_ticks[n]) for n in names])
    bounds = [(limits or {}).get(n, (0, TICKS_PER_TURN - 1)) for n in names]

    def project(vec: np.ndarray) -> np.ndarray | None:
        obs = Observation(arm=arm, head_ticks=dict(zip(names, vec, strict=True)), pixel=(0.0, 0.0))
        pred = model.predict(obs, theta)
        return pred.uv if pred.z > 0.05 and np.isfinite(pred.uv).all() else None

    for _ in range(iterations):
        here = project(ticks)
        if here is None:
            return None
        error = goal - here
        if np.linalg.norm(error) < 2.0:
            break
        jac = np.zeros((2, 2))
        for i in range(2):
            step = np.zeros(2)
            step[i] = 15.0
            plus = project(ticks + step)
            if plus is None:
                return None
            jac[:, i] = (plus - here) / 15.0
        if abs(np.linalg.det(jac)) < 1e-9:
            return None
        move = np.linalg.solve(jac, error)
        move = np.clip(move, -250.0, 250.0)  # never a huge jump on a poor prediction
        ticks = np.array([min(max(t, lo), hi) for t, (lo, hi) in zip(ticks + move, bounds, strict=True)])
    return {n: int(round(t)) for n, t in zip(names, ticks, strict=True)}
