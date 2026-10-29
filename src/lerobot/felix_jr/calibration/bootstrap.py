"""Getting the first frames of a calibration session right, before any click exists.

The head's tilt at its recorded home is not "level": it is looking some unknown number of degrees down, and a
model that assumes zero sends the head to the wrong place (it overshoots downwards and shows the elbow instead of
the gripper). Three helpers fix the start:

- `estimate_up_vector` finds the dominant upward-facing plane in the depth image (the table) by RANSAC. Its normal
  is the direction of "up" in the camera's frame, which is what gravity would tell an accelerometer. Planes that
  face sideways (a wall behind the table) are ignored: the normal must point up in the image.
- `estimate_tilt_zero` turns that into the head tilt zero offset: the value for which the model's own camera pose
  has the same "up". It is good to a degree or two (the camera mount's own rotation error is the limit), so it is
  used as a *prior* (mean and a 5 degree sigma) for the fit, not as the answer.
- `seed_pose_for_view` searches for an arm pose whose tip, as the model predicts it, is near the image center at
  the head's current pose and not too close or too low, so the first pose already shows the gripper.
"""

from dataclasses import dataclass

import numpy as np

from lerobot.felix_jr.calibration.design import CLOSED, JOINT_LIMITS, LIMIT_MARGIN
from lerobot.felix_jr.calibration.model import ARM_JOINTS, CalibrationModel, Observation
from lerobot.felix_jr.scene import Scene

TILT_PRIOR_SIGMA = np.radians(5.0)  # how much we trust the plane-based tilt estimate
MAX_PLANE_DEPTH = 2.5  # meters; farther depth is too noisy to fit a plane
MIN_UP_Y = (
    -0.5
)  # an upward normal points up in the image (y is down): n_y below this (pitch under ~60 degrees)
SEED_RANGE = {"shoulder_pan": 50.0, "shoulder_lift": 35.0, "elbow_flex": 35.0, "wrist_flex": 35.0}


@dataclass
class Plane:
    """A plane found in the depth image, in the camera's optical frame.

    `normal`: unit normal pointing towards the camera (and up, for a table). `offset`: `normal . p` for points p on
    the plane (negative: the plane is on the far side of the camera origin, `-offset` meters away). `fraction`: its
    share of the valid depth points.
    """

    normal: np.ndarray
    offset: float
    fraction: float


def up_from_pitch(pitch_down: float, roll: float = 0.0) -> np.ndarray:
    """The up direction in the camera's optical frame, for a pitch and a roll.

    The camera is pitched `pitch_down` radians below horizontal and rolled `roll` radians about its axis
    (optical frame: x right, y down, z forward).
    """
    up = np.array([0.0, -np.cos(pitch_down), -np.sin(pitch_down)])
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ up


def estimate_up_vector(
    scene: Scene,
    step: int = 4,
    iterations: int = 600,
    min_inliers: int = 600,
    min_fraction: float = 0.06,
    rng_seed: int = 0,
) -> Plane | None:
    """The dominant upward-facing plane in the depth image (the table), as a `Plane`.

    The normal is in the camera's optical frame and points towards the camera (and up). RANSAC over planes whose
    normal points up in the image, refined by a least-squares fit to the inliers. A point is an inlier within
    `0.005 + 0.004 z^2` meters of the plane (stereo depth noise grows with the square of the distance). Returns
    None if no plane has at least `min_inliers` points and `min_fraction` of all valid ones (no table in view).
    """
    depth = scene.depth_m[::step, ::step]
    v, u = np.mgrid[0 : scene.height : step, 0 : scene.width : step]
    ok = np.isfinite(depth) & (depth > 0.15) & (depth < MAX_PLANE_DEPTH)
    z = depth[ok]
    points = np.stack([(u[ok] - scene.ppx) / scene.fx * z, (v[ok] - scene.ppy) / scene.fy * z, z], axis=1)
    if len(points) < min_inliers:
        return None
    tolerance = 0.005 + 0.004 * points[:, 2] ** 2
    rng = np.random.default_rng(rng_seed)
    best_count, best_inliers = 0, None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal @ a > 0:
            normal = -normal  # towards the camera
        if normal[1] > MIN_UP_Y:
            continue  # not an upward-facing plane: a wall, or a steep slope
        inliers = np.abs(points @ normal - normal @ a) < tolerance
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    if best_inliers is None or best_count < min_inliers or best_count < min_fraction * len(points):
        return None
    cloud = points[best_inliers]
    centroid = cloud.mean(axis=0)
    normal = np.linalg.svd(cloud - centroid)[2][2]
    if normal @ centroid > 0:
        normal = -normal
    if normal[1] > MIN_UP_Y:
        return None
    normal = normal / np.linalg.norm(normal)
    return Plane(normal, float(normal @ centroid), best_count / len(points))


def table_height(
    model: CalibrationModel, head_ticks: dict[str, int], theta: dict[str, float], plane: Plane
) -> float:
    """The height of the table plane in the robot's root frame (meters, z up).

    The camera sits at a known height in the model; the plane is `-plane.offset` meters below it along the plane's
    normal, which is nearly vertical for a table.
    """
    camera = model.camera_pose(head_ticks, theta)
    normal_z = float(np.clip((camera[:3, :3] @ plane.normal)[2], 0.5, 1.0))
    return float(camera[2, 3] + plane.offset / normal_z)


def estimate_tilt_zero(
    model: CalibrationModel,
    head_ticks: dict[str, int],
    up_cam: np.ndarray,
    theta: dict[str, float] | None = None,
) -> tuple[float, float]:
    """The head tilt zero offset (radians) for which the model's camera has `up_cam` as its up.

    Also returns the angle (degrees) by which the best match still misses `up_cam`, as a check on the estimate:
    a large miss means the plane was not the table, or the head is rolled. `theta` supplies the other parameters
    (default: all zero).
    """
    theta = dict(theta or {})
    up_cam = up_cam / np.linalg.norm(up_cam)

    def miss(tilt: float) -> float:
        camera = model.camera_pose(head_ticks, {**theta, "zero:head_tilt_joint": tilt})
        predicted = camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # the root frame's z is up
        return float(np.degrees(np.arccos(np.clip(predicted @ up_cam, -1.0, 1.0))))

    grid = np.linspace(-1.3, 1.3, 261)
    errors = np.array([miss(t) for t in grid])
    i = int(np.argmin(errors))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(40):  # golden-section refinement inside the bracket
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        lo, hi = (lo, b) if miss(a) < miss(b) else (a, hi)
    best = 0.5 * (lo + hi)
    return float(best), miss(best)


def seed_pose_for_view(
    model: CalibrationModel,
    theta: dict[str, float],
    head_ticks: dict[str, int],
    target_px: tuple[float, float] | None = None,
    depth_range: tuple[float, float] = (0.25, 0.6),
    min_tip_height: float = 0.05,
    table_z: float | None = None,
    clearance: float = 0.04,
    samples: int = 6000,
    rng_seed: int = 0,
    max_error_px: float = 120.0,
) -> dict[str, float] | None:
    """An arm pose whose tip lands near `target_px` (default: the image center) at the head's current pose.

    Random search then local refinement over the pan, lift, elbow and wrist flex within `SEED_RANGE` of the
    wiggle's center pose (which is known to be safe), preferring poses close to it. The tip must be between
    `depth_range` meters from the camera and at least `min_tip_height` above the arm base, or, if the table's height
    `table_z` (root frame, from `table_height`) is known, at least `clearance` above the table. Returns the pose (closed gripper), or None if nothing qualifies or the best pose still misses
    `target_px` by more than `max_error_px`.
    """
    target = np.array(
        target_px if target_px is not None else (model.camera.width / 2, model.camera.height / 2)
    )
    base_z = model.head_tree.link_pose(model.base_link)[2, 3]
    joints = list(SEED_RANGE)
    scale = np.array([SEED_RANGE[j] for j in joints])

    def cost(vector: np.ndarray) -> float:
        pose = dict(zip(joints, vector, strict=True))
        for joint in joints:
            low, high = JOINT_LIMITS[joint]
            if not low * LIMIT_MARGIN <= pose[joint] <= high * LIMIT_MARGIN:
                return np.inf
        arm = {f"{j}.pos": pose.get(j, 0.0) for j in ARM_JOINTS}
        prediction = model.predict(Observation(arm=arm, head_ticks=head_ticks, pixel=(0.0, 0.0)), theta)
        if not np.isfinite(prediction.uv).all() or not depth_range[0] <= prediction.z <= depth_range[1]:
            return np.inf
        floor = base_z + min_tip_height if table_z is None else table_z + clearance
        if model.tip_in_root(arm, theta)[2] < floor:
            return np.inf
        return float(np.linalg.norm(prediction.uv - target) / 50.0 + 0.3 * np.linalg.norm(vector / scale))

    rng = np.random.default_rng(rng_seed)
    candidates = rng.uniform(-scale, scale, size=(samples, len(joints)))
    costs = np.array([cost(c) for c in candidates])
    if not np.isfinite(costs.min()):
        return None
    best, best_cost = candidates[int(np.argmin(costs))], float(costs.min())
    for step in (0.5, 0.25, 0.1, 0.05):  # pattern search around the best sample, in units of the range
        for _ in range(150):
            trial = best + rng.normal(0.0, step, len(joints)) * scale * 0.3
            c = cost(trial)
            if c < best_cost:
                best, best_cost = trial, c
    pose = {
        **{f"{j}.pos": 0.0 for j in ARM_JOINTS},
        **{f"{j}.pos": float(v) for j, v in zip(joints, best, strict=True)},
    }
    landed = model.predict(Observation(arm=pose, head_ticks=head_ticks, pixel=(0.0, 0.0)), theta).uv
    if np.linalg.norm(landed - target) > max_error_px:
        return None
    return {**pose, "gripper.pos": CLOSED}
