"""Planning a joint calibration's views (`joint_collect` on the robot, `joint_sim` in simulation).

`docs/camera_calibration.md`, "The rework". Everything is planned on a model: the calibration's start for a session,
the true robot in a simulation (`theta`, `gripper_from_camera`). Per board placement:

- `suggest_placements`: where to put the board: on the table ahead of the arm, toward the robot's middle, where the
  head sees it whole with the arms folded.
- `head_aims`: points around the board for the head-only views (the arms folded).
- `arm_views`: arm poses whose wrist camera sees the whole board, its center up and right in the picture
  (`AIM_PX`), clear of the fingers (lower left): around a few looks down at it (`look_at`), the joints varied.
- `low_poses`: the gripper low over the table, as for a grasp: only the fingertips clicked there.
"""

from collections.abc import Callable, Mapping, Sequence

import cv2
import numpy as np

from lerobot.felix_jr.calibration.boards import Board
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    XleRobotModel,
    pinch_frame,
    wrist_camera_link,
)
from lerobot.utils.import_utils import _scipy_available, require_package

if _scipy_available:
    from scipy.optimize import least_squares

AIM_PX = (
    400.0,
    170.0,
)  # the board's center in the wrist picture: up and right, away from the fingers (lower left)
FINGER_BOX_PX = (
    80.0,
    40.0,
)  # the fingers in the wrist picture: this far either side of the tips, and above them
HEAD_ONLY_SPREAD_M = (
    0.12  # the head aimed up to this far around the board's center (the board must stay in view)
)
# camera looks at the board: (height m, tilt from straight down deg, toward yaw deg; 0 = from the robot's side)
VIEW_SEEDS = (
    (0.30, 25.0, 0.0),
    (0.35, 30.0, 35.0),
    (0.30, 30.0, -35.0),
    (0.40, 15.0, 0.0),
    (0.28, 35.0, 20.0),
)
VIEW_SPREAD_DEG = 10.0  # the joints varied around the seeds by this much (1 sigma)
PLACEMENT_AHEAD_M = 0.36  # the board's center ahead of the arm's base (hil3, jnt1, jnt2: 35-50 cm)
PLACEMENT_SIDES_M = (-0.24, -0.06)  # ... and toward the robot's middle (negative y for the right arm)
LOW_AHEAD_M, LOW_SIDE_M, LOW_UP_M = (0.15, 0.30), (-0.12, 0.12), (0.03, 0.12)  # low poses' pinch point


def board_points(board: Board, root_board: np.ndarray) -> np.ndarray:
    """The board's points (all of them) in the root frame."""
    return board.points() @ root_board[:3, :3].T + root_board[:3, 3]


def wrist_pose(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    theta: np.ndarray | None = None,
    gc: np.ndarray | None = None,
) -> np.ndarray:
    """root_from_camera of `arm`'s wrist camera at `q` (the model's estimates and camera pose unless given)."""
    gc = np.asarray(model.wrist_cameras[arm].gripper_from_camera if gc is None else gc, dtype=np.float64)
    return model.frame_pose(wrist_camera_link(arm), q, theta) @ gc


def wrist_pixels(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    points: np.ndarray,
    theta: np.ndarray | None = None,
    gc: np.ndarray | None = None,
) -> np.ndarray:
    """Root-frame points in the wrist camera's picture at `q`."""
    pose = wrist_pose(model, arm, q, theta, gc)
    return model.wrist_cameras[arm].project((np.atleast_2d(points) - pose[:3, 3]) @ pose[:3, :3])


def fingertips(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    theta: np.ndarray | None = None,
    thickness: float = JAW_THICKNESS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """The fixed fingertip on the head's face (the pinch point) and on the wrist camera's face, root frame."""
    pinch = model.frame_pose(pinch_frame(arm), q, theta)
    face = model.wrist_face(arm) * (thickness / JAW_THICKNESS_M)
    return pinch[:3, 3].copy(), pinch[:3, :3] @ face + pinch[:3, 3]


def in_picture(uv: np.ndarray, size: tuple[int, int], margin: float = 10.0) -> bool:
    """Whether every pixel is in a picture of `size` (width, height), `margin` from its edges."""
    return bool(
        np.isfinite(uv).all()
        and (uv[:, 0] > margin).all()
        and (uv[:, 0] < size[0] - margin).all()
        and (uv[:, 1] > margin).all()
        and (uv[:, 1] < size[1] - margin).all()
    )


def wrist_sees(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    points: np.ndarray,
    theta: np.ndarray | None = None,
    gc: np.ndarray | None = None,
) -> np.ndarray | None:
    """The pixels of `points` if the wrist camera sees all of them, clear of the fingers, else None."""
    camera = model.wrist_cameras[arm]
    uv = wrist_pixels(model, arm, q, points, theta, gc)
    if not in_picture(uv, (camera.width, camera.height)):
        return None
    tips = wrist_pixels(model, arm, q, np.array(fingertips(model, arm, q, theta)), theta, gc)
    if np.isfinite(tips).all():
        left, right = tips[:, 0].min() - FINGER_BOX_PX[0], tips[:, 0].max() + FINGER_BOX_PX[0]
        top = tips[:, 1].min() - FINGER_BOX_PX[1]
        if ((uv[:, 0] > left) & (uv[:, 0] < right) & (uv[:, 1] > top)).any():
            return None
    return uv


def _arm_solve(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    residual: Callable[[np.ndarray], np.ndarray],
    start_deg: Sequence[float] = (0.0, 0.0, 30.0, 60.0, 0.0),
) -> dict[str, float]:
    """The arm's joints (LeRobot degrees) minimizing `residual(q)`, within 95% of the joints' limits."""
    require_package("scipy", extra="felix_jr")
    index = [model.q_index(f"{arm}_{j}") for j in SO101_JOINTS]
    lower = np.array([model.limits(f"{arm}_{j}")[0] for j in SO101_JOINTS]) * 0.95
    upper = np.array([model.limits(f"{arm}_{j}")[1] for j in SO101_JOINTS]) * 0.95
    q0 = model.joint_vector({arm: dict(rest)})

    def wrapped(x: np.ndarray) -> np.ndarray:
        q = q0.copy()
        q[index] = x
        return residual(q)

    start = np.clip(np.radians(np.asarray(start_deg, dtype=np.float64)), lower + 1e-6, upper - 1e-6)
    x = least_squares(wrapped, start, bounds=(lower, upper)).x
    return {f"{j}.pos": float(np.degrees(v)) for j, v in zip(SO101_JOINTS, x, strict=True)}


def look_at(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    target: np.ndarray,
    height: float,
    tilt_deg: float,
    yaw_deg: float,
    pixel: Sequence[float] = AIM_PX,
    theta: np.ndarray | None = None,
    gc: np.ndarray | None = None,
) -> dict[str, float]:
    """The arm's joints putting the wrist camera `height` from `target`, with `target` at `pixel`.

    The camera is tilted from straight above by `tilt_deg` toward `yaw_deg` (root frame); `pixel` keeps the target
    clear of the fingers.
    """
    yaw, tilt = np.radians(yaw_deg), np.radians(tilt_deg)
    eye = target + height * np.array([np.cos(yaw) * np.sin(tilt), np.sin(yaw) * np.sin(tilt), np.cos(tilt)])
    camera = model.wrist_cameras[arm]

    def residual(q: np.ndarray) -> np.ndarray:
        pose = wrist_pose(model, arm, q, theta, gc)
        uv = camera.project(((target - pose[:3, 3]) @ pose[:3, :3])[None])[0]
        return np.r_[(pose[:3, 3] - eye) / 0.01, np.nan_to_num((uv - np.asarray(pixel)) / 20.0, nan=50.0)]

    return _arm_solve(model, arm, rest, residual)


def within_limits(model: XleRobotModel, arm: str, pose: Mapping[str, float]) -> bool:
    """Whether every joint of `pose` (LeRobot degrees) is inside the model's limits."""
    return all(
        model.limits(f"{arm}_{j}")[0] < np.radians(pose[f"{j}.pos"]) < model.limits(f"{arm}_{j}")[1]
        for j in SO101_JOINTS
    )


def arm_views(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    points: np.ndarray,
    n: int,
    rng: np.random.Generator,
    accept: Callable[[dict[str, float]], bool] | None = None,
    tries: int = 400,
) -> list[dict[str, float]]:
    """`n` arm poses whose wrist camera sees all of `points` (the board) clear of the fingers.

    The `VIEW_SEEDS` looks first, then the joints varied around them. `accept(pose)` decides (default: the model
    sees the board there); a simulation passes the truth's.
    """
    center = points.mean(axis=0)
    seeds = [look_at(model, arm, rest, center, *seed) for seed in VIEW_SEEDS]
    if accept is None:

        def accept(pose: dict[str, float]) -> bool:
            return wrist_sees(model, arm, model.joint_vector({arm: pose}), points) is not None

    out: list[dict[str, float]] = []
    for i in range(tries):
        if len(out) >= n:
            break
        spread = 0.0 if i < len(seeds) else VIEW_SPREAD_DEG
        pose = {k: v + rng.normal(0.0, spread) for k, v in seeds[i % len(seeds)].items()}
        if within_limits(model, arm, pose) and accept(pose):
            out.append(pose)
    return out


def suggest_placements(
    model: XleRobotModel,
    arm: str,
    board: Board,
    n: int,
    table_z: float,
    yaws_rad: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """root_from_board of `n` placements on the table at `table_z`, ahead of the arm, spread toward the middle."""
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    sign = -1.0 if arm == "right" else 1.0  # toward the robot's middle
    center = board.points().mean(axis=0)
    out = []
    for k in range(n):
        inward = (
            np.mean(np.abs(PLACEMENT_SIDES_M))
            if n == 1
            else np.interp(k, [0, n - 1], np.abs(PLACEMENT_SIDES_M)[::-1])
        )
        ahead = PLACEMENT_AHEAD_M + (0.08 * ((k % 2) - 0.5) if n > 2 else 0.0)
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, 0.0 if yaws_rad is None else yaws_rad[k]]))[0]
        pose[:3, 3] = np.array([base[0] - ahead, base[1] + sign * inward, table_z]) - pose[:3, :3] @ center
        out.append(pose)
    return out


def head_aims(center: np.ndarray, n: int) -> list[np.ndarray]:
    """`n` points around the board's `center` (root frame) to aim the head at for its views of the board."""
    out = []
    for j in range(n):
        angle = 2 * np.pi * j / max(1, n)
        out.append(
            center + HEAD_ONLY_SPREAD_M * np.array([np.cos(angle), np.sin(angle), 0.0]) * (1 + j % 2) / 2
        )
    return out


def low_poses(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    n: int,
    table_z: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """`n` poses with the pinch point low over the table ahead of the arm, the gripper tilted down, as a grasp's."""
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    out = []
    for _ in range(n):
        target = np.array(
            [
                base[0] - rng.uniform(*LOW_AHEAD_M),
                base[1] + rng.uniform(*LOW_SIDE_M),
                table_z + rng.uniform(*LOW_UP_M),
            ]
        )
        pose = look_at(model, arm, rest, target, 0.15, rng.uniform(20, 45), rng.uniform(-40, 40))
        pinch = model.frame_pose(pinch_frame(arm), model.joint_vector({arm: pose}))[:3, 3]
        if pinch[2] > table_z + 0.02 and within_limits(
            model, arm, pose
        ):  # never lower than 2 cm over the table
            out.append(pose)
    return out
