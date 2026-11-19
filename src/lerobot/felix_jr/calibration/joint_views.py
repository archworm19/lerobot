"""Planning a joint calibration's views (`joint_collect` on the robot, `joint_sim` in simulation).

`docs/camera_calibration.md`, "The rework". Everything is planned on a model: the calibration's start for a session,
the true robot in a simulation (`theta`, the wrist camera's pose among its parameters). Per board placement:

- `suggest_placements`: where to put the board: on the table ahead of the arm, toward the robot's middle, where the
  head sees it whole with the arms folded.
- `head_aims`: points around the board for the head-only views (the arms folded).
- `arm_views`: arm poses whose wrist camera sees the board clear of the fingers (lower left; part of it, for a
  ChArUco board), spread over the arm's joints within the region it works in (near `low_poses`): many looks at it
  drawn at random (`look_at`), the farthest apart kept. A fit from views that differ little learns the arm near one
  pose only (jnt3: 23 mm apart at a held-out pose unlike the others); views far from the grasps spend the fit there.
- `low_poses`: the gripper low over the table, as for a grasp: the region the arm views are kept near.
- `head_views`: where the head clicks the fingertips: the fingers about level, the wrist rolled so the head looks
  down onto the jaws' face (`head_sees`), at heights from just over the table up. The head cannot see the tips at the
  arm views (the wrist camera sits between) or at `low_poses` (the fingers point down, away from it: 2026-11-17).
"""

from collections.abc import Callable, Mapping, Sequence

import cv2
import numpy as np

from lerobot.felix_jr.calibration.boards import Board
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS, wrist_camera_frame
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    XleRobotModel,
    pinch_frame,
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
# the wrist camera's looks at the board drawn from: height (m), tilt from straight down (deg), toward yaw (deg; 0 =
# from the robot's side), and where the board's center is in the picture (AIM_PX +- this, px)
VIEW_HEIGHT_M, VIEW_TILT_DEG, VIEW_YAW_DEG = (0.15, 0.25), (0.0, 70.0), (-90.0, 90.0)
REGION_VIEW_HEIGHT_M = (0.12, 0.25)  # ... nearer the board when the views keep to the working region
# the arm views' joints kept from the extremes (LeRobot deg; None: the joint's own limit). Picked farthest apart, the
# views went to the limits: the gripper bent fully back (the wrist camera's mount hit the arm), the elbow folded the
# other way, the roll near its stops, the camera 30+ cm back (2026-11-12, joint_left_full1)
VIEW_JOINT_LIMITS_DEG = {"elbow_flex": (0.0, None), "wrist_flex": (None, 70.0), "wrist_roll": (-120.0, 120.0)}
# ... and a view whose solve ended on a bound is dropped (the farthest-apart pick otherwise lands on the bounds), as is
# one with the camera farther than this from the board's center
BOUND_MARGIN_DEG = 3.0
VIEW_MAX_DISTANCE_M = 0.25
# ... and none grazing or close: the camera at most this far off looking straight down, no board point nearer than
# this (joint_left_full3: views 6-9 cm from the board, looking across it from below its edge, fitted 10-17 mm off)
VIEW_MAX_OBLIQUE_DEG = 50.0
VIEW_MIN_DISTANCE_M = 0.12
# the wrist camera kept off the arm's body (to the upper arm's axis, shoulder to elbow, and to the arm's base): folded
# poses put its mount within 5-8 cm of the upper arm's axis and 12 cm of the base, against the arm (joint_left_full3)
CAMERA_ARM_CLEARANCE_M = 0.12
CAMERA_BASE_CLEARANCE_M = 0.20
LOW_TRIES_PER_POSE = 30  # `low_poses` draws up to this many per pose it returns
LOW_JOINT_LIMITS_DEG = {
    "elbow_flex": (0.0, None),
    "wrist_flex": (None, 70.0),
}  # the low poses' (their roll free)
REGION_DEG = 40.0  # a view within the working region: this near one of its poses (the first four joints, deg)
FAR_FRACTION = 0.3  # of the arm views with a working region, this part planned anywhere (`arm_views`)
REGION_START_DEG = 15.0  # the solver started this far (1 sigma) around a working-region pose
VIEW_PIXEL_SPREAD_PX = (60.0, 50.0)
VIEW_START_SPREAD_DEG = 40.0  # the solver started this far (1 sigma) around its usual start: other solutions
ARM_LINKS = ("upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link")
LINK_CLEARANCE_M = 0.05  # a view keeps every link's origin this far above the board's table
PLAN_MARGIN_PX = (
    30.0  # a planned view keeps the board inside the wrist picture by this: room for the model's errors
)
BOARD_YAW_RAD = 0.0  # the board turned on the table; pi / 2 (long side across): no better in joint_sim
PLACEMENT_AHEAD_M = (
    0.26  # the board's center ahead of the arm's base: where grasps are (35 cm before 2026-11-11)
)
PLACEMENT_SIDES_M = (-0.24, -0.06)  # ... and toward the robot's middle (negative y for the right arm)
# the jaws at the arm views and low poses, `gripper.pos` in turn (`with_openings`): the moving tip's arc about the
# hinge, clicked by the head, fixes the hinge and the opening's offset (`joint_sim`: the hinge 1.7 -> 0.8 mm)
VIEW_OPENINGS = (20.0, 35.0, 50.0)
# the wrist camera's board (45 mm square around the lens: `hardware_risks.md`) kept this far over the table: at a tip
# view the pinch face is up, so the camera, on the gripper's other side, faces the table (it hit it: joint_right_1117b)
CAMERA_BOARD_HALF_M = 0.0225
CAMERA_TABLE_CLEARANCE_M = 0.02
PATH_STEPS = 12  # a move's straight path in the joints, checked at this many poses
TIP_CLEARANCE_M = (
    0.02  # a pose keeps both fingertips this far above the table (the moving one at its opening)
)
# the head sees the fingertips only with the face it clicks turned this near to it: past it, the gripper's body and the
# wrist camera hide them (the click calibration's views, 30-60 clicks an arm; joint_right_1117's low poses, 56-72 deg
# from the head, showed the tips hardly at all)
HEAD_SEES_DEG = 50.0
# the head views' pinch point (ahead of the arm's base, to its side, over the table) and the fingers' tilt below level
HEAD_VIEW_AHEAD_M, HEAD_VIEW_SIDE_M, HEAD_VIEW_UP_M = (0.10, 0.32), (-0.25, 0.25), (0.04, 0.28)
HEAD_VIEW_TILT_DEG = (-10.0, 30.0)
HEAD_VIEW_MIN_TRIES = 100  # ... and this many at least (a few views from few tries: none, at times)
HEAD_VIEW_TRIES = 15  # targets drawn per view returned (one in ten is kept: 8 of 10 views from 8 a view)
HEAD_VIEW_LIMITS_DEG = {"elbow_flex": (0.0, None)}
HEAD_FACE_MAX_DEG = 75.0  # a low pose shows the head the fingertip's face it clicks within this (90: edge on)
LOW_TILT_DEG = (
    45.0,
    80.0,
)  # low poses' gripper tilt from straight down: toward level, where a roll shows that face
LOW_AHEAD_M, LOW_SIDE_M, LOW_UP_M = (0.15, 0.30), (-0.12, 0.12), (0.03, 0.12)  # low poses' pinch point


def board_points(board: Board, root_board: np.ndarray) -> np.ndarray:
    """The board's points (all of them) in the root frame."""
    return board.points() @ root_board[:3, :3].T + root_board[:3, 3]


def wrist_pose(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    """root_from_camera of `arm`'s wrist camera at `q` (the model's estimates unless given: its pose among them)."""
    return model.frame_pose(wrist_camera_frame(arm), q, theta)


def wrist_pixels(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    points: np.ndarray,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    """Root-frame points in the wrist camera's picture at `q`."""
    pose = wrist_pose(model, arm, q, theta)
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


def moving_fingertips(
    model: XleRobotModel,
    arm: str,
    q: np.ndarray,
    opening_deg: float,
    theta: np.ndarray | None = None,
    thickness: float = JAW_THICKNESS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """The moving fingertip on the head's face and on the wrist camera's, root frame, the jaws `opening_deg` open.

    `fingertips`' two points turned about the hinge by the opening (`XleRobotModel.moving_jaw`): the faces of
    `joint_fit.MOVING_PINCH` and `joint_fit.MOVING`.
    """
    pinch = model.frame_pose(pinch_frame(arm), q, theta)
    turned = model.moving_jaw(arm, opening_deg, theta)
    face = model.wrist_face(arm) * (thickness / JAW_THICKNESS_M)
    on_pinch, on_wrist = (
        pinch[:3, :3] @ (turned[:3, :3] @ point + turned[:3, 3]) + pinch[:3, 3]
        for point in (np.zeros(3), face)
    )
    return on_pinch, on_wrist


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
    margin: float = 10.0,
    min_fraction: float = 1.0,
) -> np.ndarray | None:
    """The pixels of `points` the wrist camera sees clear of the fingers, NaN the others; None below `min_fraction`.

    `margin`: how far inside the picture (and beyond the fingers' box) a point must be.
    """
    camera = model.wrist_cameras[arm]
    uv = wrist_pixels(model, arm, q, points, theta)
    seen = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] > margin)
        & (uv[:, 0] < camera.width - margin)
        & (uv[:, 1] > margin)
        & (uv[:, 1] < camera.height - margin)
    )
    tips = wrist_pixels(model, arm, q, np.array(fingertips(model, arm, q, theta)), theta)
    if np.isfinite(tips).all():
        extra = (
            margin - 10.0
        )  # a margin above the default keeps the board that much farther from the fingers too
        left = tips[:, 0].min() - FINGER_BOX_PX[0] - extra
        right = tips[:, 0].max() + FINGER_BOX_PX[0] + extra
        top = tips[:, 1].min() - FINGER_BOX_PX[1] - extra
        seen &= ~((uv[:, 0] > left) & (uv[:, 0] < right) & (uv[:, 1] > top))
    if not seen.any() or seen.mean() < min_fraction:
        return None
    return np.where(seen[:, None], uv, np.nan)


def _arm_solve(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    residual: Callable[[np.ndarray], np.ndarray],
    start_deg: Sequence[float] = (0.0, 0.0, 30.0, 60.0, 0.0),
    limits_deg: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, float]:
    """The arm's joints (LeRobot degrees) minimizing `residual(q)`, within 95% of the joints' limits.

    `limits_deg`: tighter bounds for some joints (`{joint: (low, high)}`, None for the joint's own).
    """
    require_package("scipy", extra="felix_jr")
    index = [model.q_index(f"{arm}_{j}") for j in SO101_JOINTS]
    lower = np.array([model.limits(f"{arm}_{j}")[0] for j in SO101_JOINTS]) * 0.95
    upper = np.array([model.limits(f"{arm}_{j}")[1] for j in SO101_JOINTS]) * 0.95
    for i, joint in enumerate(SO101_JOINTS):
        low, high = (limits_deg or {}).get(joint, (None, None))
        lower[i] = lower[i] if low is None else max(lower[i], np.radians(low))
        upper[i] = upper[i] if high is None else min(upper[i], np.radians(high))
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
    start_deg: Sequence[float] = (0.0, 0.0, 30.0, 60.0, 0.0),
    limits_deg: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, float]:
    """The arm's joints putting the wrist camera `height` from `target`, with `target` at `pixel`.

    The camera is tilted from straight above by `tilt_deg` toward `yaw_deg` (root frame); `pixel` keeps the target
    clear of the fingers.
    """
    yaw, tilt = np.radians(yaw_deg), np.radians(tilt_deg)
    eye = target + height * np.array([np.cos(yaw) * np.sin(tilt), np.sin(yaw) * np.sin(tilt), np.cos(tilt)])
    camera = model.wrist_cameras[arm]

    def residual(q: np.ndarray) -> np.ndarray:
        pose = wrist_pose(model, arm, q, theta)
        uv = camera.project(((target - pose[:3, 3]) @ pose[:3, :3])[None])[0]
        return np.r_[(pose[:3, 3] - eye) / 0.01, np.nan_to_num((uv - np.asarray(pixel)) / 20.0, nan=50.0)]

    return _arm_solve(model, arm, rest, residual, start_deg, limits_deg)


def head_face_angle(model: XleRobotModel, arm: str, q: np.ndarray) -> float:
    """How far (deg) the face of the fixed fingertip the head clicks is turned from the head camera.

    That face is the fingers' side away from the wrist camera (`XleRobotModel.wrist_face`); at 90 or more the head
    sees the fingers' edge or their back, the tip hidden behind the gripper.
    """
    pinch = model.frame_pose(pinch_frame(arm), q)
    face = -pinch[:3, :3] @ model.wrist_face(arm)  # the head's face's outward direction
    to_head = model.camera_pose(q)[:3, 3] - pinch[:3, 3]
    cos = face @ to_head / (np.linalg.norm(face) * np.linalg.norm(to_head))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def head_sees(model: XleRobotModel, arm: str, q: np.ndarray) -> bool:
    """Whether the head can see `arm`'s fingertips at `q`: their pinch face within `HEAD_SEES_DEG` of facing it."""
    return head_face_angle(model, arm, q) < HEAD_SEES_DEG


def camera_clear_of_arm(model: XleRobotModel, arm: str, pose: Mapping[str, float]) -> bool:
    """Whether the wrist camera is clear of the arm's body.

    `CAMERA_ARM_CLEARANCE_M` from the upper arm's axis and `CAMERA_BASE_CLEARANCE_M` from the arm's base (link origins:
    the links' bodies are a few cm thick about them).
    """
    q = model.joint_vector({arm: dict(pose)})
    camera = wrist_pose(model, arm, q)[:3, 3]
    shoulder, elbow = (
        model.frame_pose(f"{arm}_{link}", q)[:3, 3] for link in ("upper_arm_link", "lower_arm_link")
    )
    base = model.frame_pose(f"{arm}_base_link", q)[:3, 3]
    along = np.clip(
        np.dot(camera - shoulder, elbow - shoulder) / np.dot(elbow - shoulder, elbow - shoulder), 0.0, 1.0
    )
    to_upper_arm = np.linalg.norm(camera - (shoulder + along * (elbow - shoulder)))
    return bool(
        to_upper_arm >= CAMERA_ARM_CLEARANCE_M and np.linalg.norm(camera - base) >= CAMERA_BASE_CLEARANCE_M
    )


def clear_of_table(model: XleRobotModel, arm: str, pose: Mapping[str, float], table_z: float) -> bool:
    """Whether the arm is clear of the table: its links and its fingertips.

    Every link (`ARM_LINKS`, their origins) `LINK_CLEARANCE_M` above it, the fingertips `TIP_CLEARANCE_M` (`tips_clear`),
    the wrist camera's board `CAMERA_TABLE_CLEARANCE_M` (`camera_clear_of_table`).
    """
    q = model.joint_vector({arm: dict(pose)})
    return (
        all(model.frame_pose(f"{arm}_{link}", q)[2, 3] > table_z + LINK_CLEARANCE_M for link in ARM_LINKS)
        and tips_clear(model, arm, pose, table_z)
        and camera_clear_of_table(model, arm, pose, table_z)
    )


def camera_clear_of_table(model: XleRobotModel, arm: str, pose: Mapping[str, float], table_z: float) -> bool:
    """Whether `arm`'s wrist camera's board is `CAMERA_TABLE_CLEARANCE_M` over the table at `pose`.

    Its four corners, `CAMERA_BOARD_HALF_M` around the lens.
    """
    q = model.joint_vector({arm: dict(pose)})
    camera = model.frame_pose(wrist_camera_frame(arm), q)
    h = CAMERA_BOARD_HALF_M
    corners = np.array([[x, y, 0.0] for x in (-h, h) for y in (-h, h)]) @ camera[:3, :3].T + camera[:3, 3]
    return bool((corners[:, 2] > table_z + CAMERA_TABLE_CLEARANCE_M).all())


def table_margin(model: XleRobotModel, arm: str, pose: Mapping[str, float], table_z: float) -> float:
    """How far (m) the arm at `pose` is inside its clearances over the table, at the tightest (negative: within one).

    Its links (`LINK_CLEARANCE_M`), both fingertips at every opening (`TIP_CLEARANCE_M`, `tips_clear`) and the wrist
    camera's board (`CAMERA_TABLE_CLEARANCE_M`, `camera_clear_of_table`).
    """
    q = model.joint_vector({arm: dict(pose)})
    margins = [model.frame_pose(f"{arm}_{link}", q)[2, 3] - table_z - LINK_CLEARANCE_M for link in ARM_LINKS]
    tips = [model.pinch_point(arm, q)]
    if arm in model.gripper_maps:
        tips += [model.moving_tip(arm, q, model.opening_deg(arm, pos)) for pos in VIEW_OPENINGS]
    margins += [tip[2] - table_z - TIP_CLEARANCE_M for tip in tips]
    camera = model.frame_pose(wrist_camera_frame(arm), q)
    h = CAMERA_BOARD_HALF_M
    corners = np.array([[x, y, 0.0] for x in (-h, h) for y in (-h, h)]) @ camera[:3, :3].T + camera[:3, 3]
    margins += list(corners[:, 2] - table_z - CAMERA_TABLE_CLEARANCE_M)
    return float(min(margins))


def path_clear(
    model: XleRobotModel, arm: str, start: Mapping[str, float], end: Mapping[str, float], table_z: float
) -> bool:
    """Whether the arm's straight move in the joints from `start` to `end` stays clear of the table.

    As `move_joints` makes it, at `PATH_STEPS` poses: none within the clearances (`table_margin`) more than both its
    ends are (the rest pose folds the fingertips within 2 cm of the table: a move may start there; a planned view may sit
    within the clearances too, its own checks passed: a move may end there; neither may dip below them on the way, as the
    wrist camera did into the table, 2026-11-17). Was the start's alone, which sent the arm around through rest to
    every view lower than the one before (the left arm's session, 2026-11-19: every other view of a placement).
    """
    joints = [f"{j}.pos" for j in SO101_JOINTS]
    ends = min(table_margin(model, arm, start, table_z), table_margin(model, arm, end, table_z))
    floor = min(0.0, ends) - 1e-3
    for t in np.linspace(0.0, 1.0, PATH_STEPS + 1)[1:]:
        pose = {k: start[k] + t * (end[k] - start[k]) for k in joints}
        if table_margin(model, arm, pose, table_z) < floor:
            return False
    return True


def tips_clear(model: XleRobotModel, arm: str, pose: Mapping[str, float], table_z: float) -> bool:
    """Whether both fingertips are `TIP_CLEARANCE_M` above the table at `pose`, the jaws at any of `VIEW_OPENINGS`.

    The moving tip turns down past the fixed one as the jaws open, for a gripper tilted down; without the arm's
    gripper map (the openings unknown) only the fixed tip is checked, the jaws then closed (`with_openings`).
    """
    q = model.joint_vector({arm: dict(pose)})
    tips = [model.pinch_point(arm, q)]
    if arm in model.gripper_maps:
        tips += [model.moving_tip(arm, q, model.opening_deg(arm, pos)) for pos in VIEW_OPENINGS]
    return all(tip[2] > table_z + TIP_CLEARANCE_M for tip in tips)


def with_openings(
    model: XleRobotModel,
    arm: str,
    poses: Sequence[Mapping[str, float]],
    openings: Sequence[float] = VIEW_OPENINGS,
) -> list[dict[str, float]]:
    """`poses` with the jaws at `openings` (`gripper.pos`) in turn; closed (0) without the arm's gripper map or openings."""
    if not openings or arm not in model.gripper_maps:
        return [{**pose, "gripper.pos": 0.0} for pose in poses]
    return [{**pose, "gripper.pos": float(openings[i % len(openings)])} for i, pose in enumerate(poses)]


def on_a_bound(
    model: XleRobotModel,
    arm: str,
    pose: Mapping[str, float],
    limits_deg: Mapping[str, tuple[float | None, float | None]] | None = None,
    margin_deg: float = BOUND_MARGIN_DEG,
) -> bool:
    """Whether a joint of `pose` is within `margin_deg` of its bound: `limits_deg`'s, else 95% of its own (`_arm_solve`)."""
    for joint in SO101_JOINTS:
        low, high = (np.degrees(v) * 0.95 for v in model.limits(f"{arm}_{joint}"))
        own_low, own_high = (limits_deg or {}).get(joint, (None, None))
        low = low if own_low is None else max(low, own_low)
        high = high if own_high is None else min(high, own_high)
        if not low + margin_deg < pose[f"{joint}.pos"] < high - margin_deg:
            return True
    return False


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
    tries: int = 800,
    margin: float = PLAN_MARGIN_PX,
    min_fraction: float = 1.0,
    region: Sequence[Mapping[str, float]] | None = None,
    far_fraction: float = FAR_FRACTION,
) -> list[dict[str, float]]:
    """Up to `n` arm poses whose wrist camera sees the board (`points`) clear of the fingers, spread apart.

    `tries` looks drawn at random (`VIEW_HEIGHT_M`, ...; `look_at`), each at a point of the board (its center if the
    whole board must show, `min_fraction` 1). With `region` (poses the arm works in: `low_poses`), the solver starts
    from one of them and a look is kept only within `REGION_DEG` of one (the first four joints): the views cover
    where the calibration is used, not the arm's far reaches; `far_fraction` of the `n` are planned without it. Of the looks `accept(pose)` passes (default: the model
    sees `min_fraction` of the board, `margin` px inside the picture), the `n` farthest apart in the joints
    (degrees, all five alike), picked one by one. Every link stays clear of the table (`clear_of_table`). A
    simulation passes the truth's `accept`.
    """
    if region and far_fraction > 0.0 and round(far_fraction * n) > 0:
        # the most in the working region; a few anywhere, which pin the camera's pose on the gripper (joint_sim:
        # views kept near the grasps leave it ~2x worse)
        far = int(round(far_fraction * n))
        near = arm_views(
            model,
            arm,
            rest,
            points,
            n - far,
            rng,
            accept,
            tries,
            margin,
            min_fraction,
            region,
            far_fraction=0.0,
        )
        return near + arm_views(model, arm, rest, points, far, rng, accept, tries, margin, min_fraction)
    table_z = float(points[:, 2].min())  # the board lies on the table
    if accept is None:

        def accept(pose: dict[str, float]) -> bool:
            q = model.joint_vector({arm: pose})
            return wrist_sees(model, arm, q, points, margin=margin, min_fraction=min_fraction) is not None

    working = None if not region else np.array([[p[f"{j}.pos"] for j in SO101_JOINTS] for p in region])
    candidates: list[dict[str, float]] = []
    for _ in range(tries):
        if working is None:
            start = np.array([0.0, 0.0, 30.0, 60.0, 0.0])
            height = rng.uniform(*VIEW_HEIGHT_M)
        else:
            start = working[rng.integers(len(working))]
            height = rng.uniform(*REGION_VIEW_HEIGHT_M)
        target = points.mean(axis=0) if min_fraction >= 1.0 else points[rng.integers(len(points))]
        pose = look_at(
            model,
            arm,
            rest,
            target,
            height,
            rng.uniform(*VIEW_TILT_DEG),
            rng.uniform(*VIEW_YAW_DEG),
            np.asarray(AIM_PX) + rng.uniform(-1.0, 1.0, 2) * np.asarray(VIEW_PIXEL_SPREAD_PX),
            start_deg=start
            + rng.normal(0.0, VIEW_START_SPREAD_DEG if working is None else REGION_START_DEG, 5),
            limits_deg=VIEW_JOINT_LIMITS_DEG,
        )
        joints = np.array([pose[f"{j}.pos"] for j in SO101_JOINTS])
        if working is not None and np.linalg.norm(working[:, :4] - joints[:4], axis=1).min() > REGION_DEG:
            continue
        if on_a_bound(model, arm, pose, VIEW_JOINT_LIMITS_DEG):  # the look was not reached freely: an extreme
            continue
        camera = wrist_pose(model, arm, model.joint_vector({arm: pose}))
        if np.linalg.norm(camera[:3, 3] - points.mean(axis=0)) > VIEW_MAX_DISTANCE_M:
            continue
        if np.linalg.norm(points - camera[:3, 3], axis=1).min() < VIEW_MIN_DISTANCE_M:
            continue
        if (
            np.degrees(np.arccos(np.clip(-camera[2, 2], -1.0, 1.0))) > VIEW_MAX_OBLIQUE_DEG
        ):  # optical axis vs down
            continue
        if not camera_clear_of_arm(model, arm, pose):
            continue
        if within_limits(model, arm, pose) and clear_of_table(model, arm, pose, table_z) and accept(pose):
            candidates.append(pose)
    if not candidates:
        return []
    joints = np.array([[pose[f"{j}.pos"] for j in SO101_JOINTS] for pose in candidates])
    picked = [0]
    nearest = np.linalg.norm(joints - joints[0], axis=1)
    while len(picked) < min(n, len(candidates)):
        k = int(np.argmax(nearest))
        picked.append(k)
        nearest = np.minimum(nearest, np.linalg.norm(joints - joints[k], axis=1))
    return [candidates[k] for k in picked]


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
        yaw = BOARD_YAW_RAD + (0.0 if yaws_rad is None else yaws_rad[k])
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]
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
    """Up to `n` poses with the pinch point low over the table ahead of the arm, the gripper tilted down, as a grasp's.

    The wrist rolled to show the head the fingertip's face it clicks best (`head_face_angle`, within
    `HEAD_FACE_MAX_DEG`, short of edge-on; the roll in 15 deg steps): tilted down, that face looks sideways unless the
    wrist turns it up. A pose no roll shows it from, a fingertip near the table (`tips_clear`, the jaws at any of
    `VIEW_OPENINGS`), or that puts the wrist camera against the arm (`camera_clear_of_arm`) or a joint on
    a bound (`LOW_JOINT_LIMITS_DEG`), is dropped and another drawn, `LOW_TRIES_PER_POSE` per pose at most.
    """
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    out: list[dict[str, float]] = []
    for _ in range(LOW_TRIES_PER_POSE * n):
        if len(out) >= n:
            break
        target = np.array(
            [
                base[0] - rng.uniform(*LOW_AHEAD_M),
                base[1] + rng.uniform(*LOW_SIDE_M),
                table_z + rng.uniform(*LOW_UP_M),
            ]
        )
        planned = look_at(
            model,
            arm,
            rest,
            target,
            0.15,
            rng.uniform(*LOW_TILT_DEG),
            rng.uniform(-40, 40),
            limits_deg=LOW_JOINT_LIMITS_DEG,
        )
        best: tuple[float, dict[str, float]] | None = None
        for turn in np.arange(-180.0, 181.0, 15.0):
            pose = {**planned, "wrist_roll.pos": planned["wrist_roll.pos"] + turn}
            q = model.joint_vector({arm: pose})
            if not tips_clear(model, arm, pose, table_z) or not within_limits(model, arm, pose):
                continue
            if not camera_clear_of_table(model, arm, pose, table_z):
                continue
            if not camera_clear_of_arm(model, arm, pose) or on_a_bound(
                model, arm, pose, LOW_JOINT_LIMITS_DEG
            ):
                continue  # (a bound: the arm folded up against its stops, the camera by its base)
            angle = head_face_angle(model, arm, q)
            if angle < HEAD_FACE_MAX_DEG and (best is None or angle < best[0]):
                best = (angle, pose)
        if best is not None:
            out.append(best[1])
    return out


def head_views(
    model: XleRobotModel,
    arm: str,
    rest: Mapping[str, float],
    n: int,
    table_z: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Up to `n` arm poses at which the head sees the fingertips (`head_sees`), spread apart in the joints.

    Each solved for (`_arm_solve`): the pinch point at a target drawn ahead of the arm (`HEAD_VIEW_*`: from just over
    the table to ~30 cm up), the fingers at a tilt below level, the pinch face turned to the head. With 5 joints the
    fingers point in the arm's vertical plane, so the head (above the arm's base) sees them face-on only near level,
    the wrist rolled so the hinge points up at it. Kept: the target reached (1.5 cm), the head sees the tips, within the
    joints' limits and off their bounds, the wrist camera clear of the arm, both tips clear of the table at every
    opening (`tips_clear`) and the wrist camera's board too (`camera_clear_of_table`: it faces the table here). `HEAD_VIEW_TRIES` targets per view, the `n` farthest apart in the joints kept.
    """
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    face = model.wrist_face(arm) / np.linalg.norm(model.wrist_face(arm))
    candidates: list[dict[str, float]] = []
    for _ in range(max(HEAD_VIEW_TRIES * n, HEAD_VIEW_MIN_TRIES)):
        target = np.array(
            [
                base[0] - rng.uniform(*HEAD_VIEW_AHEAD_M),
                base[1] + rng.uniform(*HEAD_VIEW_SIDE_M),
                table_z + rng.uniform(*HEAD_VIEW_UP_M),
            ]
        )
        down = np.radians(rng.uniform(*HEAD_VIEW_TILT_DEG))

        def residual(q: np.ndarray, target: np.ndarray = target, down: float = down) -> np.ndarray:
            pinch = model.frame_pose(pinch_frame(arm), q)
            to_head = model.camera_pose(q)[:3, 3] - pinch[:3, 3]
            to_head /= np.linalg.norm(to_head)
            return np.r_[
                (pinch[:3, 3] - target) / 0.01,
                (pinch[2, 2] + np.sin(down)) / 0.05,  # the pinch z (toward the tips) that far below level
                (-pinch[:3, :3] @ face - to_head) / 0.3,  # the pinch face's outward normal toward the head
            ]

        start = np.array([0.0, 0.0, 30.0, 0.0, 0.0]) + rng.normal(0.0, 30.0, 5)
        pose = _arm_solve(model, arm, rest, residual, start, HEAD_VIEW_LIMITS_DEG)
        q = model.joint_vector({arm: pose})
        if np.linalg.norm(model.frame_pose(pinch_frame(arm), q)[:3, 3] - target) > 0.015:
            continue
        if not head_sees(model, arm, q) or not within_limits(model, arm, pose):
            continue
        if on_a_bound(model, arm, pose, HEAD_VIEW_LIMITS_DEG) or not camera_clear_of_arm(model, arm, pose):
            continue
        if tips_clear(model, arm, pose, table_z) and camera_clear_of_table(model, arm, pose, table_z):
            candidates.append(pose)
    if not candidates:
        return []
    joints = np.array([[pose[f"{j}.pos"] for j in SO101_JOINTS] for pose in candidates])
    picked = [0]
    nearest = np.linalg.norm(joints - joints[0], axis=1)
    while len(picked) < min(n, len(candidates)):
        k = int(np.argmax(nearest))
        picked.append(k)
        nearest = np.minimum(nearest, np.linalg.norm(joints - joints[k], axis=1))
    return [candidates[k] for k in picked]
