"""Fit an arm's shapes into the calibration: its wrist camera's plate and its parts as capsules, from clicks.

  uv run python -m lerobot.felix_jr.calibration.arm_shapes tcp://<pi>:5555 --arm right
      --start-from <calibration.urdf> --out-dir ~/shapes_right [--parts plate,fixed_finger,...] [--n 12]
      [--n-other 8] [--step 2] [--max-vel 30] [--seed-pose "shoulder_lift=-10"] [--delta-scale 1] [--floor-z Z]
      [--max-roll 130] [--resume]

Seen from the head camera, the wrist camera's plate can hide the jaws (`docs/sessions.md`, pen2), and the clearance
checks of the grasping (`docs/object_grasping.md`) need the gripper's and the forearm's shape, so both go into the
model. `--parts` picks what to fit (default all, in this order):

  plate           the wrist camera's mount plate, a thin box: three of its corners, clicked in order
  fixed_finger    the fixed jaw's finger            \
  moving_finger   the moving jaw's finger            |  capsules: click points on each part's outline (its
  gripper_body    the gripper's body and its servo   |  silhouette edge against the background), along both
  wrist           the wrist (the wrist flex servo)   |  sides and around its ends
  forearm         the forearm, with its sleeve      /

For the plate, put small numbered marks (1, 2, 3) on three corners of its back (the side the head camera sees),
going around it, so that corner 2 is between 1 and 3. The plate is assumed 1 cm thick (`BOARD_THICKNESS_M`).

Start the server on the Pi, clear the workspace, as for `collect`. A window shows the head camera. `n` moves the arm
and the head to the next view: `--n` views from the plate's side of the gripper (the head camera sees its back),
then, when capsules are fitted, `--n-other` from the other side (the jaws' side, as `collect`'s), so each part is seen
from both. With capsules, the jaws are opened about 3 cm (if the calibration has a gripper map), so the two fingers'
outlines do not merge. In each view the status line asks for one thing at a time:

  left mouse click    a corner: record the corner asked for (1, then 2, then 3)
                      a part: record a point on its outline (2-4 per part and view, both sides)
  s                   skip the corner asked for (hidden in this view); for a part, go on to the next part (with no
                      clicks: skip it here)
  n                   go to the next view (back to the first thing to click)
  arrow keys          turn the head by --step degrees
  u                   undo the last click
  q or Esc            finish: fit and write `calibration.urdf` (the robot is sent to rest)

**The fits** use the measured joints and leave the rest of the calibration (`--start-from`) as it is. The plate:
each corner is a point fixed in `<arm>_gripper_link`, 9 numbers for the three, and every click asks the corner's
predicted pixel to land on it (the depth there is saved, but not fitted: at a corner it is often what is behind).
A capsule: a segment fixed in its part's link (the fingers: the pinch frame, the moving one turned by the measured
opening) and a radius, 7 numbers; every outline click is a ray from the camera that grazes it, so the ray's distance
to the segment is the radius, and the fit makes that so (the misses scaled to pixels), from the link's geometry and a
weak pull toward it. Fitted shapes are drawn as they come: the plate yellow, the capsules orange; they should sit on the
real ones. The report (`arm_shapes.md`) gives the pixel RMS per shape, how square the plate is (two edges of equal
length at 90 deg), and each capsule's length and radius (check them with a ruler).

**The output.** `--out-dir/calibration.urdf` is `--start-from` with the shapes added (`XleRobotModel.with_wrist_board`:
a box link on a fixed joint to the gripper, and the fitted corners in `<felix_calibration>`; `with_capsules`: in
`<felix_calibration>`); for the other arm, run again with `--start-from` that file. The clicks and their images are
kept in `--out-dir` (`corners.json`, `outlines.json`, `images/`); `--resume` continues. The script runs inside a client
session, so on Ctrl+C or a crash the robot goes to rest.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.live_view import (
    INTRINSICS_KEYS,
    LiveView,
    decode_key,
    parse_seed,
    set_floor,
)
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG
from lerobot.felix_jr.calibration.xle_robot_model import (
    BOARD_THICKNESS_M,
    CAPSULE_PARTS,
    HEAD_JOINTS,
    SIDES,
    Capsule,
    WristBoard,
    XleRobotModel,
    arm_record,
    pinch_frame,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.harness.look import aim_head
from lerobot.felix_jr.harness.scene import Scene
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

PARTS = ("plate", *CAPSULE_PARTS)
CORNERS = (1, 2, 3)
CLICKS_FILE = "corners.json"
OUTLINES_FILE = "outlines.json"
CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "arm_shapes.md"
PIXEL_SIGMA = 2.0  # px: a click's expected error
MIN_CLICKS_PER_CORNER = 2  # before the plate is fitted and drawn
MIN_OUTLINE_CLICKS = 6  # before a capsule is fitted and drawn
MIN_OUTLINE_VIEWS = 2  # and from this many views
PRIOR_POINT_M = (
    0.03  # the pull of a capsule's ends toward the link's geometry: this far costs as much as one click
)
PRIOR_RADIUS_M = 0.015  # and of its radius
RADIUS_BOUNDS_M = (0.002, 0.06)
OPEN_APART_M = 0.03  # the jaws this far apart in the views, when capsules are fitted
AIM_MARGIN_PX = 60  # a part whose middle is not this far inside the picture: the head turns to it
AIM_MISS_DEG = 5.0  # the head cannot point closer than this: say so
BOARD_COLOR = (255, 220, 0)  # RGB: yellow
CAPSULE_COLOR = (255, 140, 0)  # RGB: orange
KEY_ACTIONS = {ord("n"): "next", ord("s"): "skip", ord("u"): "undo", ord("q"): "quit", 27: "quit"}


def board_parent(arm: str) -> str:
    """The link the plate is fixed to: the gripper's body (not the pinch frame, whose placement is calibrated)."""
    return f"{arm}_gripper_link"


@dataclass
class CornerClick:
    """One clicked corner: which, the pixel, the depth there, and the robot's measured joints with the frame."""

    arm: str
    corner: int
    arm_pose: dict[str, float]  # LeRobot's degrees, {"shoulder_pan.pos": ..., ...}
    head_ticks: dict[str, int]
    pixel: tuple[float, float]
    depth_m: float | None = None
    image: str | None = None  # relative to the session folder
    index: int = 0
    timestamp: float = 0.0


def save_clicks(directory: Path, arm: str, record: dict[str, Any], clicks: Sequence[CornerClick]) -> None:
    """Write `corners.json`: the arm, its LeRobot calibration and the clicks."""
    data = {"arm": arm, "arm_calibration": record, "clicks": [asdict(c) for c in clicks]}
    (directory / CLICKS_FILE).write_text(json.dumps(data, indent=1) + "\n")


def load_clicks(directory: Path) -> tuple[str, dict[str, Any], list[CornerClick]]:
    """Read `corners.json` back: the arm, its LeRobot calibration and the clicks."""
    data = json.loads((directory / CLICKS_FILE).read_text())
    clicks = [CornerClick(**c | {"pixel": tuple(c["pixel"])}) for c in data["clicks"]]
    return data["arm"], data["arm_calibration"], clicks


# ---- the fit ------------------------------------------------------------------------------------------------


def _q(model: XleRobotModel, click: CornerClick) -> np.ndarray:
    return model.joint_vector({click.arm: click.arm_pose}, click.head_ticks or None)


def parent_in_camera(model: XleRobotModel, clicks: Sequence[CornerClick]) -> np.ndarray:
    """Per click, the parent link's pose in the camera's optical frame (N, 4, 4): all the fit needs of the robot."""
    poses = []
    for click in clicks:
        q = _q(model, click)
        poses.append(np.linalg.inv(model.camera_pose(q)) @ model.frame_pose(board_parent(click.arm), q))
    return np.array(poses)


def predicted(
    model: XleRobotModel, clicks: Sequence[CornerClick], corners: np.ndarray, frames: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Each click's corner (`corners` (3, 3), the parent link's frame) projected: pixels (N, 2) and depths (N,).

    `frames` are `parent_in_camera(model, clicks)`, if computed already.
    """
    frames = parent_in_camera(model, clicks) if frames is None else frames
    points = corners[[c.corner - 1 for c in clicks]]  # (N, 3)
    p = np.einsum("nij,nj->ni", frames[:, :3, :3], points) + frames[:, :3, 3]
    z = p[:, 2]
    safe_z = np.where(z > 1e-6, z, np.nan)
    k = model.intrinsics
    uv = np.stack([k["fx"] * p[:, 0] / safe_z + k["ppx"], k["fy"] * p[:, 1] / safe_z + k["ppy"]], axis=1)
    return uv, z


def initial_corners(model: XleRobotModel, clicks: Sequence[CornerClick]) -> np.ndarray:
    """A start for the fit: the pinch point (the plate is a few cm from it), a centimeter apart per corner."""
    frames = [
        (model.frame_pose(board_parent(c.arm), _q(model, c)), model.pinch_point(c.arm, _q(model, c)))
        for c in clicks
    ]
    pinch = np.mean([f[:3, :3].T @ (p - f[:3, 3]) for f, p in frames], axis=0)
    return pinch + np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])


def fit_corners(
    model: XleRobotModel, clicks: Sequence[CornerClick], start: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The three corners (3, 3) in the parent link's frame that best explain `clicks`, and each click's pixel error.

    Only the pixels are fitted (`PIXEL_SIGMA` each, a Huber loss so a misclick does not pull the rest): the depth at
    a corner is often the background behind the plate's edge, so it is kept with the click but not used.
    """
    require_package("scipy", extra="felix_jr")
    pixels = np.array([c.pixel for c in clicks], dtype=float)
    frames = parent_in_camera(model, clicks)

    def residuals(x: np.ndarray) -> np.ndarray:
        uv, _ = predicted(model, clicks, x.reshape(3, 3), frames)
        return np.nan_to_num(((uv - pixels) / PIXEL_SIGMA).ravel(), nan=1e3)

    x0 = (initial_corners(model, clicks) if start is None else start).ravel()
    result = least_squares(residuals, x0, loss="huber", x_scale=0.01)
    corners = result.x.reshape(3, 3)
    uv, _ = predicted(model, clicks, corners, frames)
    return corners, np.linalg.norm(uv - pixels, axis=1)


def board_from_fit(
    model: XleRobotModel, clicks: Sequence[CornerClick], corners: np.ndarray, errors: np.ndarray
) -> WristBoard:
    """The plate from fitted corners: thick away from where the head camera saw it from (on average)."""
    arm = clicks[0].arm
    center = (corners[0] + corners[2]) / 2
    away = np.zeros(3)
    for click in clicks:
        q = _q(model, click)
        frame = model.frame_pose(board_parent(arm), q)
        camera = frame[:3, :3].T @ (model.camera_pose(q)[:3, 3] - frame[:3, 3])
        away += (center - camera) / np.linalg.norm(center - camera)
    away /= np.linalg.norm(away)
    return WristBoard(
        parent=board_parent(arm),
        corners=tuple((float(x), float(y), float(z)) for x, y, z in corners),
        away=(float(away[0]), float(away[1]), float(away[2])),
        thickness=BOARD_THICKNESS_M,
        rms_px=float(np.sqrt(np.mean(errors**2))),
        n_clicks=len(clicks),
    )


def squareness(board: WristBoard) -> tuple[float, float, float]:
    """The two fitted edges' lengths (m) and the angle between them (deg): a square plate has them equal, at 90."""
    c1, c2, c3 = board.face_corners()[:3]
    e1, e2 = c1 - c2, c3 - c2
    angle = np.degrees(np.arccos(np.clip(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2)), -1.0, 1.0)))
    return float(np.linalg.norm(e1)), float(np.linalg.norm(e2)), float(angle)


def plate_report(board: WristBoard, clicks: Sequence[CornerClick], errors: np.ndarray) -> list[str]:
    """The plate's part of the report (markdown): the plate, how square it came out, the pixel errors per corner."""
    edge1, edge2, angle = squareness(board)
    pose, size = board.box()
    lines = [
        "## Wrist camera plate",
        "",
        f"{len(clicks)} clicks; pixel RMS {board.rms_px:.1f} px.",
        "",
        f"- edges 1-2 {edge1 * 1000:.1f} mm, 2-3 {edge2 * 1000:.1f} mm, at {angle:.1f} deg (a square: equal, 90)",
        f"- box center in `{board.parent}` ({', '.join(f'{v * 1000:.1f}' for v in pose[:3, 3])}) mm, size "
        f"{' x '.join(f'{v * 1000:.0f}' for v in size)} mm (thickness assumed)",
        "",
        "| corner | clicks | RMS (px) | max (px) |",
        "|---|---|---|---|",
    ]
    for corner in CORNERS:
        mine = errors[[c.corner == corner for c in clicks]]
        if len(mine):
            lines.append(f"| {corner} | {len(mine)} | {np.sqrt(np.mean(mine**2)):.1f} | {mine.max():.1f} |")
    return lines


# ---- the capsules -------------------------------------------------------------------------------------------


@dataclass
class OutlineClick:
    """One clicked point on a part's outline: the part, the pixel, and the robot's measured joints with the frame.

    `opening_deg` is the jaws' opening then (from the gripper map; 0 without one): it places the moving finger.
    """

    arm: str
    part: str
    arm_pose: dict[str, float]  # LeRobot's degrees, with "gripper.pos"
    head_ticks: dict[str, int]
    pixel: tuple[float, float]
    opening_deg: float = 0.0
    image: str | None = None  # relative to the session folder
    index: int = 0
    timestamp: float = 0.0


def save_outlines(directory: Path, arm: str, record: dict[str, Any], clicks: Sequence[OutlineClick]) -> None:
    """Write `outlines.json`: the arm, its LeRobot calibration and the outline clicks."""
    data = {"arm": arm, "arm_calibration": record, "clicks": [asdict(c) for c in clicks]}
    (directory / OUTLINES_FILE).write_text(json.dumps(data, indent=1) + "\n")


def load_outlines(directory: Path) -> tuple[str, dict[str, Any], list[OutlineClick]]:
    """Read `outlines.json` back: the arm, its LeRobot calibration and the outline clicks."""
    data = json.loads((directory / OUTLINES_FILE).read_text())
    clicks = [OutlineClick(**c | {"pixel": tuple(c["pixel"])}) for c in data["clicks"]]
    return data["arm"], data["arm_calibration"], clicks


def _point(p: np.ndarray) -> tuple[float, float, float]:
    return float(p[0]), float(p[1]), float(p[2])


def _origin_in(model: XleRobotModel, parent: str, child: str) -> np.ndarray:
    """Where `child`'s origin is in `parent`'s frame, at the model's zero pose."""
    q = model.joint_vector()
    parent_frame, child_frame = model.frame_pose(parent, q), model.frame_pose(child, q)
    return parent_frame[:3, :3].T @ (child_frame[:3, 3] - parent_frame[:3, 3])


def initial_capsule(model: XleRobotModel, arm: str, part: str) -> Capsule:
    """A start for a part's fit, from the URDF's links: rough, the fit moves it.

    The fingers run back from the tips (the pinch frame's origin) along -z, the fixed one on +x, the moving one on -x
    (the side it opens to); the gripper body from its link's origin toward the pinch point; the wrist and the forearm
    from their link's origin to the next joint.
    """
    pinch = pinch_frame(arm)
    if part == "fixed_finger":
        return Capsule(pinch, (0.008, 0.0, -0.005), (0.012, 0.0, -0.05), 0.008)
    if part == "moving_finger":
        return Capsule(pinch, (-0.008, 0.0, -0.005), (-0.012, 0.0, -0.05), 0.008, moving=True)
    if part == "gripper_body":
        parent = f"{arm}_gripper_link"
        tip = _origin_in(model, parent, pinch)
        return Capsule(parent, _point(0.2 * tip), _point(0.6 * tip), 0.022)
    if part == "wrist":
        parent = f"{arm}_wrist_link"
        return Capsule(
            parent, (0.0, 0.0, 0.0), _point(_origin_in(model, parent, f"{arm}_gripper_link")), 0.02
        )
    if part == "forearm":
        parent = f"{arm}_lower_arm_link"
        return Capsule(parent, (0.0, 0.0, 0.0), _point(_origin_in(model, parent, f"{arm}_wrist_link")), 0.025)
    raise ValueError(f"unknown part {part!r}: one of {CAPSULE_PARTS}")


def click_rays(
    model: XleRobotModel, capsule: Capsule, clicks: Sequence[OutlineClick]
) -> tuple[np.ndarray, np.ndarray]:
    """Each click's ray from the camera through its pixel, in the capsule's parent frame.

    Returns the origins and unit directions (N, 3). The parent's pose is the one measured with the click (the moving
    finger turned by its opening).
    """
    k = model.intrinsics
    origins, directions = [], []
    for click in clicks:
        q = model.joint_vector({click.arm: click.arm_pose}, click.head_ticks or None)
        parent = model.capsule_frame(capsule, click.arm, q, click.opening_deg)
        camera = model.camera_pose(q)
        u, v = click.pixel
        ray = camera[:3, :3] @ np.array([(u - k["ppx"]) / k["fx"], (v - k["ppy"]) / k["fy"], 1.0])
        origins.append(parent[:3, :3].T @ (camera[:3, 3] - parent[:3, 3]))
        directions.append(parent[:3, :3].T @ (ray / np.linalg.norm(ray)))
    return np.array(origins), np.array(directions)


def ray_segment_distance(
    origins: np.ndarray, directions: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each ray's (unit `directions`) closest distance to the segment `a`-`b`, and how far along the ray it is."""
    e = b - a
    w0 = a - origins  # (N, 3)
    ee, ed = float(e @ e), directions @ e
    ew, dw = w0 @ e, np.einsum("ij,ij->i", directions, w0)
    denom = ee - ed**2
    s = np.where(denom > 1e-12, (ed * dw - ew) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = dw + s * ed
    gap = w0 + s[:, None] * e - t[:, None] * directions
    return np.linalg.norm(gap, axis=1), t


def fit_capsule(
    model: XleRobotModel, clicks: Sequence[OutlineClick], start: Capsule, prior: Capsule | None = None
) -> tuple[Capsule, np.ndarray]:
    """The capsule (in `start`'s parent frame) whose surface every click's ray grazes, and each click's miss (px).

    A ray's miss is its distance to the segment minus the radius, scaled to pixels at the depth where it passes
    (`PIXEL_SIGMA` each, a Huber loss so a misclick does not pull the rest). `prior` (default `start`) pulls the ends
    and the radius weakly toward it (`PRIOR_POINT_M`, `PRIOR_RADIUS_M`): an end no click is near stays near it.
    """
    require_package("scipy", extra="felix_jr")
    prior = start if prior is None else prior
    origins, directions = click_rays(model, start, clicks)
    focal = 0.5 * (model.intrinsics["fx"] + model.intrinsics["fy"])
    x_prior = np.array([*prior.a, *prior.b, prior.radius])
    sigma_prior = np.array([PRIOR_POINT_M] * 6 + [PRIOR_RADIUS_M])

    def misses_px(x: np.ndarray) -> np.ndarray:
        distance, along = ray_segment_distance(origins, directions, x[:3], x[3:6])
        return (distance - x[6]) * focal / np.maximum(along, 0.05)

    def residuals(x: np.ndarray) -> np.ndarray:
        return np.concatenate([misses_px(x) / PIXEL_SIGMA, (x - x_prior) / sigma_prior])

    x0 = np.array([*start.a, *start.b, np.clip(start.radius, *RADIUS_BOUNDS_M)])
    lower = np.r_[np.full(6, -np.inf), RADIUS_BOUNDS_M[0]]
    upper = np.r_[np.full(6, np.inf), RADIUS_BOUNDS_M[1]]
    result = least_squares(residuals, x0, loss="huber", x_scale=0.01, bounds=(lower, upper))
    x, errors = result.x, np.abs(misses_px(result.x))
    capsule = Capsule(
        start.parent,
        (float(x[0]), float(x[1]), float(x[2])),
        (float(x[3]), float(x[4]), float(x[5])),
        float(x[6]),
        start.moving,
        rms_px=float(np.sqrt(np.mean(errors**2))),
        n_clicks=len(clicks),
    )
    return capsule, errors


def capsule_report(
    arm: str, fits: dict[str, tuple[Capsule, np.ndarray]], clicks: Sequence[OutlineClick]
) -> list[str]:
    """The capsules' part of the report (markdown): per part, the clicks and views, the pixel errors, the size."""
    lines = [
        "## Capsules",
        "",
        "| part | clicks | views | RMS (px) | max (px) | length (mm) | radius (mm) | link |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for part, (capsule, errors) in fits.items():
        views = len({c.timestamp for c in clicks if c.part == part})
        length = float(np.linalg.norm(np.subtract(capsule.b, capsule.a)))
        lines.append(
            f"| {part} | {capsule.n_clicks} | {views} | {capsule.rms_px:.1f} | {errors.max():.1f} | "
            f"{length * 1000:.0f} | {capsule.radius * 1000:.1f} | `{capsule.parent}`"
            f"{' (turns with the jaw)' if capsule.moving else ''} |"
        )
    return lines


def capsule_outline(
    model: XleRobotModel, q: np.ndarray, a_root: np.ndarray, b_root: np.ndarray, radius: float, n: int = 8
) -> np.ndarray | None:
    """A capsule's outline in the image at `q`, as a closed polygon (2n + 2, 2) of pixels; None if it is behind.

    The ends are projected and each widened by the radius at its depth: a sketch for checking the fit by eye, not the
    exact silhouette.
    """
    uv, z = model.project(np.array([a_root, b_root]), q)
    if not (z > 0).all() or not np.isfinite(uv).all():
        return None
    focal = 0.5 * (model.intrinsics["fx"] + model.intrinsics["fy"])
    r_a, r_b = radius * focal / z[0], radius * focal / z[1]
    along = uv[1] - uv[0]
    length = float(np.linalg.norm(along))
    along = along / length if length > 1e-6 else np.array([1.0, 0.0])
    angle = np.arctan2(along[1], along[0])
    around_b = angle + np.linspace(-np.pi / 2, np.pi / 2, n + 1)
    around_a = angle + np.linspace(np.pi / 2, 3 * np.pi / 2, n + 1)
    return np.vstack(
        [
            uv[1] + r_b * np.column_stack([np.cos(around_b), np.sin(around_b)]),
            uv[0] + r_a * np.column_stack([np.cos(around_a), np.sin(around_a)]),
        ]
    )


# ---- the session --------------------------------------------------------------------------------------------


def annotate(
    image: np.ndarray,
    lines: list[str],
    shapes: Sequence[tuple[np.ndarray, tuple[int, int, int]]],
    clicks: Sequence[tuple[tuple[float, float], str]],
) -> np.ndarray:
    """Copy of the RGB `image` with the fitted shapes, this frame's clicks and text.

    The shapes are closed polygons, each in its color; the clicks green crosses, labelled.
    """
    out = image.copy()
    for polygon, color in shapes:
        if np.isfinite(polygon).all():
            points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [points], True, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(out, [points], True, color, 1, cv2.LINE_AA)
    for (u, v), label in clicks:
        centre = (int(round(u)), int(round(v)))
        cv2.drawMarker(out, centre, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (centre[0] + 6, centre[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def flip_side(side: str) -> str:
    """The other side of the gripper (`SIDES`)."""
    return SIDES[1 - SIDES.index(side)]


Step = int | str  # a corner (1, 2, 3) or a part (`CAPSULE_PARTS`)


class ArmShapesSession(LiveView):
    """One fit of an arm's shapes: moves the robot, shows frames, asks for clicks, refits. See the module docstring."""

    no_more_views = "no more views: click what is left, then q"

    def __init__(
        self,
        client: FelixClient,
        start_from: Path,
        arm: str,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        parts: Sequence[str] = PARTS,
        n_targets: int = 12,
        n_other: int = 8,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        color_format: str = "png",
        seed_pose: dict[str, float] | None = None,
        delta_scale: float = 1.0,
        floor_z: float | None = None,
        max_roll: float = MAX_ROLL_DEG,
    ):
        """Load the calibration, check it is for this robot, plan the views, show a frame.

        `n_targets` views come from the plate's side, then `n_other` from the other side if capsules are fitted
        (without the plate, all `n_targets + n_other` alternate sides). Raises `RobotMismatchError` (from
        `check_robot`) before anything moves.
        """
        unknown = set(parts) - set(PARTS)
        if unknown or not parts:
            raise ValueError(f"parts must be some of {PARTS}, not {sorted(unknown) or 'none'}")
        self.parts = [p for p in PARTS if p in parts]
        self.capsule_parts = [p for p in self.parts if p != "plate"]
        self.start_from, self.out_dir = start_from, out_dir
        self.calibration = XleRobotModel.load(start_from)  # what is written out, with the shapes added
        self.calibration.check_robot(client)
        model = XleRobotModel.load(start_from)  # the one the views are planned with
        self.own_side = model.visible_side[arm]  # the jaws' side: `collect`'s views
        self.plate_side = flip_side(self.own_side)  # the plate's back
        if "plate" in self.parts:
            model.wrist_boards.pop(
                arm, None
            )  # an earlier fit of this plate: it is what the views are meant to show
        self.notes: list[str] = []
        capture = client.capture(color_format=color_format)
        intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        if model.intrinsics != intrinsics:
            self.notes.append("the camera's intrinsics differ from the calibration's: using the camera's")
            model.intrinsics = self.calibration.intrinsics = intrinsics
        set_floor(model, arm, floor_z, self.notes.append)
        self.record = arm_record(client.arm_calibration()[f"{arm}_arm"])
        self.open_pos: float | None = None  # the views' gripper.pos: None closes it
        if self.capsule_parts:
            if arm in model.gripper_maps:
                self.open_pos = model.gripper_pos_for(arm, model.opening_for_width(arm, OPEN_APART_M))
            else:
                self.notes.append(
                    "no gripper map: the jaws stay closed (the fingers' outlines merge at the tips)"
                )
        others = n_other if self.capsule_parts else 0
        if "plate" in self.parts:
            self.target_sides = [self.plate_side] * n_targets + [self.own_side] * others
        else:
            self.target_sides = [(self.own_side, self.plate_side)[i % 2] for i in range(n_targets + others)]
        model.visible_side[arm] = self.target_sides[0] if self.target_sides else self.own_side
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        super().__init__(
            client,
            model,
            arm,
            show,
            capture,
            len(self.target_sides),
            max_vel,
            max_accel,
            head_step_deg,
            color_format,
            seed_pose,
            delta_scale,
            max_roll,
        )
        self.clicks: list[CornerClick] = []
        self.outline_clicks: list[OutlineClick] = []
        self.history: list[str] = []  # "corner" / "outline", the order clicked (for undo)
        if resume:
            for name, load in ((CLICKS_FILE, load_clicks), (OUTLINES_FILE, load_outlines)):
                if not (out_dir / name).is_file():
                    continue
                saved_arm, record, saved = load(out_dir)
                if saved_arm != arm or record != self.record:
                    raise ValueError(f"{out_dir / name} holds clicks of another arm or arm calibration")
                if name == CLICKS_FILE:
                    self.clicks = saved  # type: ignore[assignment]
                else:
                    self.outline_clicks = saved  # type: ignore[assignment]
            self.history = ["corner"] * len(self.clicks) + ["outline"] * len(self.outline_clicks)
        self.view_side = self.target_sides[0] if self.target_sides else self.own_side
        self.step_index = 0  # into `steps()`: what the next click is
        self.aimed: tuple[int, int] | None = None  # the (view, step) the head was last turned for
        self.frame_clicks: list[tuple[tuple[float, float], str, int]] = []  # pixel, label, step index
        self.corners: np.ndarray | None = None
        self.errors: np.ndarray | None = None
        self.capsules: dict[str, tuple[Capsule, np.ndarray]] = {}  # part -> (fitted, errors px)
        self.message = self.notes[-1] if self.notes else "n = first view"
        self._refit()
        self.render()

    # ---- what to click --------------------------------------------------------------------------------------

    def steps(self) -> list[Step]:
        """What this view asks for, in order: the plate's corners (from the plate's side), then the parts."""
        corners: list[Step] = (
            list(CORNERS) if "plate" in self.parts and self.view_side == self.plate_side else []
        )
        return corners + list(self.capsule_parts)

    def asking(self) -> Step | None:
        """What the next click is: a corner (1, 2, 3) or a part's name; None when this view is done."""
        steps = self.steps()
        return steps[self.step_index] if self.step_index < len(steps) else None

    def before_view(self, index: int) -> None:
        """Look for target `index`'s view from its side of the gripper."""
        self.model.visible_side[self.arm] = self.target_sides[index]

    def next_pose(self, gripper_pos: float | None = None) -> bool:
        """The next view (`LiveView.next_pose`), the jaws opened for the capsules; it sets what to click there."""
        moved = super().next_pose(self.open_pos if gripper_pos is None else gripper_pos)
        if moved:  # the view's own side (targets with no view may have been skipped)
            self.view_side = self.target_sides[self.target_index - 1]
            self.step_index = 0
            self.render()
            self._aim_at_asked()
        return moved

    def _aim_at_asked(self) -> None:
        """Turn the head to the part asked for, if its middle is not well inside the picture; take a new frame.

        Its middle as fitted so far, else its start; still asking for it after. Once per part and view: the arm, the
        forearm especially, is often out of a view aimed at the pinch point, and close to the camera.
        """
        step = self.asking()
        if not isinstance(step, str) or (self.target_index, self.step_index) == self.aimed:
            return
        self.aimed = (self.target_index, self.step_index)
        capsule = (
            self.capsules[step][0]
            if step in self.capsules
            else initial_capsule(self.calibration, self.arm, step)
        )
        q = self.current_q()
        frame = self.model.capsule_frame(capsule, self.arm, q, self.opening_deg())
        middle = frame[:3, :3] @ (0.5 * (np.asarray(capsule.a) + np.asarray(capsule.b))) + frame[:3, 3]
        uv, z = self.model.project(middle, q)
        if z[0] > 0 and self.model.in_view(uv[0], margin=AIM_MARGIN_PX):
            return
        head = {n: int(self.capture.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        ticks, miss_deg = aim_head(self.model, head, middle)
        done = self._move_head_to(ticks)
        self.frame_clicks = []  # drawn on the frame before
        LiveView.refresh(self, done)  # not ours: that would ask for the first thing again
        part = step.replace("_", " ")
        self.message = f"turned the head to the {part}" + (
            f" (it points {miss_deg:.0f} deg off: arrows to find it, or s to skip it)"
            if miss_deg > AIM_MISS_DEG
            else ""
        )
        self.render()

    # ---- showing --------------------------------------------------------------------------------------------

    def outline(self) -> np.ndarray | None:
        """The fitted plate's four corners in the current frame (4, 2), if fitted and in front of the camera."""
        if self.corners is None:
            return None
        q = self.current_q()
        frame = self.model.frame_pose(board_parent(self.arm), q)
        c1, c2, c3 = self.corners
        points = np.array([c1, c2, c3, c1 + c3 - c2]) @ frame[:3, :3].T + frame[:3, 3]
        uv, z = self.model.project(points, q)
        return uv if (z > 0).all() else None

    def opening_deg(self) -> float:
        """The jaws' opening (degrees) measured with the current frame; 0 without a gripper map."""
        if self.arm not in self.calibration.gripper_maps:
            return 0.0
        return max(
            self.calibration.opening_deg(self.arm, float(self.capture.state[self.arm_id]["gripper.pos"])), 0.0
        )

    def capsule_outlines(self) -> list[np.ndarray]:
        """The fitted capsules in the current frame, as outlines (`capsule_outline`)."""
        q, opening = self.current_q(), self.opening_deg()
        outlines = []
        for capsule, _ in self.capsules.values():
            frame = self.model.capsule_frame(capsule, self.arm, q, opening)
            a_root, b_root = (frame[:3, :3] @ np.asarray(p) + frame[:3, 3] for p in (capsule.a, capsule.b))
            polygon = capsule_outline(self.model, q, a_root, b_root, capsule.radius)
            if polygon is not None:
                outlines.append(polygon)
        return outlines

    def prompt(self) -> str:
        """The status line's request."""
        step = self.asking()
        if step is None:
            return "this view is done: n next view | u undo | q quit"
        if isinstance(step, int):
            return f"click plate corner {step} | s skip it | n next view | arrows head | u undo | q quit"
        here = sum(1 for _, label, index in self.frame_clicks if index == self.step_index)
        return (
            f"click the {step.replace('_', ' ')}'s outline ({here} here; 2-4, both sides) | s next part | "
            "n next view | u undo | q quit"
        )

    def render(self) -> None:
        """Show the current frame with the fitted shapes, this frame's clicks and the status text."""
        fitted = []
        if self.errors is not None:
            fitted.append(f"plate {np.sqrt(np.mean(self.errors**2)):.1f} px")
        fitted += [
            f"{part} {capsule.rms_px:.1f} px" if capsule.rms_px is not None else part
            for part, (capsule, _) in self.capsules.items()
        ]
        side = "plate's side" if self.view_side == self.plate_side else "jaws' side"
        lines = [
            f"{self.arm} arm: {len(self.clicks)} corner + {len(self.outline_clicks)} outline clicks, view "
            f"{self.target_index}/{len(self.targets)} ({side}); fitted: {', '.join(fitted) or 'none yet'}",
            self.prompt(),
        ]
        if self.message:
            lines.append(self.message)
        shapes = [(polygon, CAPSULE_COLOR) for polygon in self.capsule_outlines()]
        plate = self.outline()
        if plate is not None:
            shapes.append((plate, BOARD_COLOR))
        self.show(
            annotate(self.capture.color, lines, shapes, [(px, label) for px, label, _ in self.frame_clicks])
        )

    def refresh(self, after: float | None = None) -> None:
        """Take a new frame and show it, asking for the first thing again."""
        self.frame_clicks, self.step_index = [], 0
        super().refresh(after)

    # ---- clicks ---------------------------------------------------------------------------------------------

    def _state(self) -> tuple[dict[str, float], dict[str, int]]:
        state = self.capture.state
        return (
            {k: float(value) for k, value in state[self.arm_id].items()},
            {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS},
        )

    def _save_image(self, index: int, kind: str) -> str:
        image = f"images/{kind}_{index:03d}.png"
        cv2.imwrite(str(self.out_dir / image), cv2.cvtColor(self.capture.color, cv2.COLOR_RGB2BGR))
        return image

    def click(self, u: float, v: float) -> CornerClick | OutlineClick | None:
        """Record pixel (u, v) as what is asked for, save it with its image, refit. None if this view is done."""
        step = self.asking()
        if step is None:
            self.message = "this view is done: n for the next"
            self.render()
            return None
        arm_pose, head_ticks = self._state()
        record: CornerClick | OutlineClick
        if isinstance(step, int):
            scene = Scene.from_capture(
                self.capture.color, self.capture.raw_depth, with_matrices(self.capture.calibration)
            )
            depth = float(scene.pixel_to_xyz(np.array([[u, v]]), window=5, max_spread_m=0.03)[0, 2])
            index = len(self.clicks)
            record = CornerClick(
                arm=self.arm,
                corner=step,
                arm_pose=arm_pose,
                head_ticks=head_ticks,
                pixel=(float(u), float(v)),
                depth_m=depth if np.isfinite(depth) else None,
                image=self._save_image(index, "corner"),
                index=index,
                timestamp=self.capture.timestamp,
            )
            self.clicks.append(record)
            self.history.append("corner")
            save_clicks(self.out_dir, self.arm, self.record, self.clicks)
            self.frame_clicks.append((record.pixel, str(step), self.step_index))
            self.step_index += 1
            self.message = f"corner {step} saved"
            self._refit()
            self.render()
            self._aim_at_asked()  # the corners done: the first part
            return record
        else:
            index = len(self.outline_clicks)
            record = OutlineClick(
                arm=self.arm,
                part=step,
                arm_pose=arm_pose,
                head_ticks=head_ticks,
                pixel=(float(u), float(v)),
                opening_deg=self.opening_deg(),
                image=self._save_image(index, "outline"),
                index=index,
                timestamp=self.capture.timestamp,
            )
            self.outline_clicks.append(record)
            self.history.append("outline")
            save_outlines(self.out_dir, self.arm, self.record, self.outline_clicks)
            self.frame_clicks.append((record.pixel, step.split("_")[0][:4], self.step_index))
            self.message = f"{step.replace('_', ' ')} point saved"
        self._refit()
        self.render()
        return record

    def skip(self) -> None:
        """Go on from what is asked for: a corner hidden in this view, or a part done (or hidden) here."""
        step = self.asking()
        if step is not None:
            here = any(index == self.step_index for _, _, index in self.frame_clicks)
            if isinstance(step, int):
                self.message = f"skipped corner {step}"
            else:
                self.message = (
                    f"{step.replace('_', ' ')} done here" if here else f"skipped the {step.replace('_', ' ')}"
                )
            self.step_index += 1
        self.render()
        self._aim_at_asked()

    def undo(self) -> None:
        """Drop the last click (its image stays in the folder), ask for it again if it was in this view, refit."""
        if self.history:
            kind = self.history.pop()
            if kind == "corner":
                self.clicks.pop()
                save_clicks(self.out_dir, self.arm, self.record, self.clicks)
            else:
                self.outline_clicks.pop()
                save_outlines(self.out_dir, self.arm, self.record, self.outline_clicks)
            if self.frame_clicks:
                _, _, step_index = self.frame_clicks.pop()
                self.step_index = step_index
            self.message = f"undid the last click ({len(self.clicks) + len(self.outline_clicks)} left)"
            self._refit()
        self.render()

    def _refit(self) -> None:
        """Refit what has enough clicks, each from its last fit (if any).

        The plate once every corner has `MIN_CLICKS_PER_CORNER`; a capsule once its part has `MIN_OUTLINE_CLICKS`
        from `MIN_OUTLINE_VIEWS` views.
        """
        counts = [sum(c.corner == corner for c in self.clicks) for corner in CORNERS]
        if min(counts) < MIN_CLICKS_PER_CORNER:
            self.corners = self.errors = None
        else:
            self.corners, self.errors = fit_corners(self.calibration, self.clicks, self.corners)
        for part in self.capsule_parts:
            mine = [c for c in self.outline_clicks if c.part == part]
            if len(mine) < MIN_OUTLINE_CLICKS or len({c.timestamp for c in mine}) < MIN_OUTLINE_VIEWS:
                self.capsules.pop(part, None)
                continue
            prior = initial_capsule(self.calibration, self.arm, part)
            start = self.capsules[part][0] if part in self.capsules else prior
            self.capsules[part] = fit_capsule(self.calibration, mine, start, prior)

    def finish(self) -> Path | None:
        """Fit every click and write the calibration with the shapes fitted (`calibration.urdf`) and the report.

        Returns the calibration's path, or None if nothing has enough clicks to fit.
        """
        self._refit()
        model = self.calibration
        report = [f"# Arm shapes: {self.arm} arm", ""]
        if self.corners is not None and self.errors is not None:
            board = board_from_fit(self.calibration, self.clicks, self.corners, self.errors)
            model = model.with_wrist_board(self.arm, board)
            report += [*plate_report(board, self.clicks, self.errors), ""]
        if self.capsules:
            model = model.with_capsules(
                self.arm, {part: capsule for part, (capsule, _) in self.capsules.items()}
            )
            report += [*capsule_report(self.arm, self.capsules, self.outline_clicks), ""]
        missing = [
            p
            for p in self.parts
            if (p == "plate" and self.corners is None) or (p != "plate" and p not in self.capsules)
        ]
        if len(missing) == len(self.parts):
            return None
        if missing:
            report += [f"Not fitted (too few clicks): {', '.join(missing)}.", ""]
        path = self.out_dir / CALIBRATION_FILE
        model.save(path)
        (self.out_dir / REPORT_FILE).write_text("\n".join(report))
        return path

    # ---- key handling ---------------------------------------------------------------------------------------

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        if action == "skip":
            self.skip()
        elif action == "undo":
            self.undo()
        else:
            return super().handle(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], required=True)
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibration.urdf to add the shapes to"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--parts", default=",".join(PARTS), help=f"what to fit, comma-separated ({','.join(PARTS)})"
    )
    parser.add_argument("--n", type=int, default=12, help="number of views from the plate's side (12)")
    parser.add_argument(
        "--n-other", type=int, default=8, help="and from the jaws' side, for the capsules (8)"
    )
    parser.add_argument("--step", type=float, default=2.0, help="head jog per arrow key, degrees (2)")
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="scale of the target region")
    parser.add_argument("--seed-pose", help='center of the target region, e.g. "shoulder_lift=-10"')
    parser.add_argument("--floor-z", type=float, help="floor height in the robot's root frame, meters")
    parser.add_argument("--max-roll", type=float, default=MAX_ROLL_DEG, help="largest wrist roll, deg (130)")
    parser.add_argument("--resume", action="store_true", help="continue an earlier --out-dir")
    args = parser.parse_args()
    if (args.out_dir / CALIBRATION_FILE).resolve() == args.start_from.resolve():
        parser.error("--out-dir would overwrite --start-from: pick another folder")
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not parts or set(parts) - set(PARTS):
        parser.error(f"--parts must be some of {','.join(PARTS)}")

    window = "arm shapes (click what is asked for; s = next; n = next view; q = quit)"
    pending: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = ArmShapesSession(
            client,
            args.start_from,
            args.arm,
            args.out_dir,
            show,
            parts=parts,
            n_targets=args.n,
            n_other=args.n_other,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            resume=args.resume,
            seed_pose=parse_seed(args.seed_pose),
            delta_scale=args.delta_scale,
            floor_z=args.floor_z,
            max_roll=args.max_roll,
        )
        for note in session.notes:
            print(note)
        cv2.setMouseCallback(window, on_mouse)
        try:
            while True:
                key = cv2.waitKeyEx(30)
                while pending:
                    session.click(*pending.pop(0))
                action = decode_key(key, KEY_ACTIONS)
                if action is not None and not session.handle(action):
                    break
        finally:
            cv2.destroyAllWindows()
    print(
        f"{len(session.clicks)} corner clicks and {len(session.outline_clicks)} outline clicks saved in {args.out_dir}"
    )
    path = session.finish()
    if path is None:
        print(
            f"too few clicks to fit anything: each corner needs {MIN_CLICKS_PER_CORNER}, each part "
            f"{MIN_OUTLINE_CLICKS} from {MIN_OUTLINE_VIEWS} views"
        )
    else:
        print(f"Wrote {path} and {args.out_dir / REPORT_FILE}")
        print((args.out_dir / REPORT_FILE).read_text())


if __name__ == "__main__":
    main()
