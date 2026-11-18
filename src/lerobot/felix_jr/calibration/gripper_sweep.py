"""Check the gripper map against the wrist camera: the real moving finger against the simulated one. Runs on your computer.

  uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.gripper_sweep \
      tcp://<pi>:5555 --arm left [--positions 0,10,20,30,40,50] [--table-z <m>] [--stay] [--urdf <calibration.urdf>]
      [--out <new calibration.urdf>] [--no-window]
  ... gripper_sweep tcp://<pi>:5555 --arm both --out <new calibration.urdf>   (left, home, then right; one URDF)
  ... gripper_sweep tcp://<pi>:5555 --arm both --repeats 3 --out <new calibration.urdf>   (3 sweeps per arm, pooled)
  ... gripper_sweep --refit <sweep folder> [<sweep folder> ...] [--out ...]   (saved sweeps fitted again, no robot;
      several of an arm pooled into one fit)

The arm goes to its sweep pose (`SWEEP_POSES`: the wrist camera looking down at the table, both fingers in view, the
gripper well above it), the gripper steps through `--positions` (LeRobot's `gripper.pos`), a wrist frame at each once
it is still (undistorted onto the calibrated pinhole, as the sim renders) with the joints read with it, and the arms go
home at the end, however the run ends. `--stay`: the arm stays wherever it is; only the gripper moves, and goes back to
where it started.

Once at the pose, a live preview (`preview`): the gripper half open, the fingers detected every second, the pixels
taken for them tinted, a pass or a fail by the limits the fit holds frames to (`preview_check`); fix the scene (the
light, anything white or bluish near the fingers) until it passes, then Enter sweeps (f forces a failing one, q stops,
the arms going home). Each arm has its own (`--arm both`).

Before anything moves, the sim checks the pose (and asks): the gripper's lowest point (the fingers, the jaws, the
camera's mount and module) stays `MIN_CLEARANCE_M` above the table over every position (`clearance`), and on the
way there (`path_clearance`, the joints moving together) it never goes lower than that or than where it starts,
whichever is lower (the arm rests lower than the margin, beside the table's edge); where it is over the table only (the scene's table, the robot's
side of it starting just ahead of the arm's base). The table's height: `--table-z`, else the higher of the scene's
(ruler2's) and the arm's base (the table is set up about level with it). And the wrist camera mostly sees the table
(`table_view`): a bluish background passes for the fingers.

**The fit.** The calibrated model turns the moving finger by the arm's `GripperMap` (`opening_deg`), and puts its tip,
the corner the wrist camera sees, in the frame (`tip_pixel`). The fingers' pixels are told from the table's by this
frame's colors (`classify`): learned from inside the fixed finger, where the sim puts it, and from away from both
fingers (fixed thresholds failed when the light changed: the fingers bluish in one sweep, near white in the next).
They are looked for only where the sim (`sim2real`) says each finger can be: its outline, widened. The detected moving
finger's tip is its farthest point from the hinge, toward where the map puts the tip, over all its pieces (a patchy
detection splits it); the opening that puts the model's tip there is the fitted one. The fitted opening minus the
map's is the map's error there. (A tip, not the outline: only the finger's lit face has its color, its shaded side and
black base are dark, so an outline's overlap drifted with the light.) No fit where the tip cannot be seen: the jaws
nearly closed, the tip out of the frame, too little of the finger.

The fixed finger never moves in the wrist camera's view: its outline's `agreement` with what was detected (the finger
pixels it covers less the table's, over the finger's) checks the fingers' placement and the colors.

**The geometry** (`fit_geometry`, from three tips seen): the moving tip turns on a circle about the hinge, through the
fixed tip with the jaws closed, so the tips over the sweep give where the hinge is (in the plane the tip turns in: a
calibrated parameter, `pos:<arm>_gripper:*`, `XleRobotModel.with_gripper_hinge`) with the opening's offset (`pos_closed`),
by least squares on their pixels (`fit_hinge`). The model's tip there is the XLeRobot finger's corner the wrist camera
sees (`sim2real.robot.FINGER_MOVING_WRIST_TIP_M`), and the detected one is taken from the finger's raw pixels (the
clean-up blunts it by a few pixels): either alone moved the hinge 1-2 mm on synthetic frames. Then the moving finger's
mount, its turn on its jaw about its tip (`XleRobotModel.finger_mount_deg`, a parameter), by its outline's agreement over the frames
(`fit_mount`). Only frames whose colors were read right (`trusted`), and the hinge's fit robust: a tip it still misses
by more than `OUTLIER_PX` is left out and the fit done again (a misread tip took a hinge to 10 mm before). The sweep is
fitted again with the corrected model (`corrected_*.png`), and `--out` writes it, a new calibrated URDF (`--urdf` is
never overwritten), once every arm's fit holds (`MAX_GEOMETRY_RMS_PX`). `--arm both`: the left arm's sweep, the arms
home, then the right's (it asks before each), both into the one URDF. `--repeats N`: N sweeps per arm, the arm back
where it started and to the pose again between (each sweep arrives anew, as the arm settles: the left hinge moved
~2 mm between runs), all pooled into the arm's one fit; each sweep's own hinge is reported, the spread.

A window shows each frame as it is fitted: the detected fingers' outline and the moving tip in green, the sim's
fingers and the model's tip at the fitted opening in orange, at the map's in blue (`--no-window` leaves it out). Each frame, its fit and a contact sheet of all of them are
saved in `--out-dir` (default `calibration/scratch/gripper_sweep/<arm>_<time>/`, gitignored); `results.json` holds the
numbers.
"""

import argparse
import copy
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.felix_jr.harness import api
from lerobot.felix_jr.server.protocol import DEFAULT_PORT
from lerobot.felix_jr.sim2real import robot
from lerobot.felix_jr.sim2real.scene import TABLE_Z, Scene, build
from lerobot.felix_jr.sim2real.view import undistort
from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

if TYPE_CHECKING or _mujoco_available:
    import mujoco
if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

# 0: the fixed finger's check (the tips meet); past ~50 the moving tip leaves the frame at the sweep poses
DEFAULT_POSITIONS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
DEFAULT_OUT = Path(__file__).parent / "scratch" / "gripper_sweep"
MIN_CLEARANCE_M = 0.03  # the gripper's lowest point over the table, over the whole sweep
MIN_TABLE_VIEW = 0.8  # the share of the wrist camera's view that is table, at least
# the sweep poses (LeRobot's degrees, the gripper apart): ruler2's left attack pose, and its mirror (the pan) for the
# right, the wrist rolled alike (the cameras sit alike on the grippers). In the sim, over the default positions: the
# gripper 6.4 (left) / 5.3 (right) cm above a table level with the base, the table ~95% of the view
SWEEP_POSES = {
    "left": {
        "shoulder_pan.pos": 12.0,
        "shoulder_lift.pos": 29.0,
        "elbow_flex.pos": -16.8,
        "wrist_flex.pos": 44.7,
        "wrist_roll.pos": -76.5,
    },
    "right": {
        "shoulder_pan.pos": -12.0,
        "shoulder_lift.pos": 29.0,
        "elbow_flex.pos": -16.8,
        "wrist_flex.pos": 44.7,
        "wrist_roll.pos": -76.5,
    },
}
PATH_STEPS = 20  # the way to the sweep pose, checked at this many points
MIN_GEOMETRY_FRAMES = 3  # tips seen, at least, to fit the hinge (two coordinates and the opening offset)
MIN_FIXED_AGREEMENT = 0.7  # a frame trusted for the geometry (`trusted`)
OUTLIER_PX = 6.0  # a tip the fitted geometry misses by more: left out, the fit done again
ROBUST_SCALE_PX = 2.0  # the hinge's fit: misses past this count less than their square
CAMERA_MOVED_PX = (
    3.0  # the fixed finger's tip moved this much between sweeps, or off the model's: the camera may have
)
FIXED_BACK_M = 0.04  # the fixed finger's direction in the frame: from this far back along it to its tip
MAX_GEOMETRY_RMS_PX = 3.0  # the corrected tips' miss, at most, for `--out` to write the fit
MOUNT_SEARCH_DEG, MOUNT_COARSE_DEG, MOUNT_FINE_DEG = 20.0, 2.0, 0.5  # the finger's mount, searched about 0
# what the sweep moves, or moves near the table: the gripper's bodies and the camera's
SWEPT_BODIES = (
    "gripper",
    "moving_jaw_so101_v1",
    robot.CAMERA_MOUNT,
    robot.CAMERA_MODULE,
    *robot.FINGER_BODIES.values(),
)
SEARCH_DEG, COARSE_DEG, FINE_DEG = 15.0, 0.5, 0.05  # the moving finger's search about the map's opening
EDGE_PX = 3  # a detected tip this close to the frame's edge is cut off: no fit
MIN_PIECE_PX = 100  # the finger's pieces smaller than this are specks
TIP_RAW_PX = 4  # the tip: the finger's raw pixels this close to its cleaned piece (the clean-up blunts it)
REGION_PX = 25  # how far beyond the sim's outlines a finger can be (the calibration's error, the fit's range)
# the fingers' color in the wrist cameras, OpenCV's HSV (H 0-180): ruler2's frames, the finger H ~107, S 85-100, V
# 115-170; the wood around it H ~170, S ~27. The black finger bases fall under the value floor
FINGER_HUE, FINGER_MIN_SATURATION, FINGER_MIN_VALUE = (95, 125), 40, 60  # (the fallback: `classify`)
LIT_MIN_VALUE = 120  # the finger's color samples: its lit face only
MIN_SAMPLE_PX = 200  # fewer samples of a color: the fixed thresholds instead
COLOR_NOISE = 4.0  # added to each color's covariance (Lab units squared): a flat color still has a spread
TABLE_SAMPLE_PX = 15  # the table's samples: this far beyond where a finger can be
WOOD_HUE_BELOW, WOOD_HUE_ABOVE, WOOD_MIN_VALUE = 35, 150, 80  # the wood: warm (red to orange), not in shadow
MIN_FINGER_PX = 500  # less of the moving finger in the frame: no fit
STILL_DEG, STILL_WINDOW_S, STILL_POLL_S, STILL_TIMEOUT_S = (
    0.3,
    0.4,
    0.1,
    4.0,
)  # the gripper's reading, settled
GREEN, ORANGE, BLUE = (0, 220, 0), (255, 140, 0), (60, 120, 255)  # RGB
MAGENTA, YELLOW = (255, 0, 255), (255, 230, 0)
PREVIEW_POS = 30.0  # the gripper's position in the preview: both fingers and the moving tip in view
PREVIEW_PERIOD_S = 1.0  # the preview's refresh


@dataclass
class SweepFrame:
    """A wrist frame (RGB, as the camera gives it) at a gripper position, with the arm's readings."""

    arm: str
    pose: dict[str, float]  # LeRobot's readings, the gripper's with them
    color: np.ndarray
    sweep: int = 0  # which of the arm's sweeps (`--repeats`, or pooled folders)


@dataclass
class FingerFit:
    """A frame's fit.

    The map's opening and the fitted one (deg), how far the model's tip is from the detected one at each (px), and the
    fixed finger's `agreement` (1 at best).
    """

    gripper_pos: float
    map_deg: float
    fitted_deg: float | None  # None: the moving finger's tip not seen
    tip_px: float | None  # the model's tip from the detected one, at the fitted opening
    tip_px_at_map: float | None
    fixed_agreement: float

    @property
    def error_deg(self) -> float | None:
        """The fitted opening minus the map's (positive: the jaws are wider than the map says)."""
        return None if self.fitted_deg is None else self.fitted_deg - self.map_deg


class FingerRenderer:
    """The sim's finger outlines in the wrist camera, the arm at a frame's joints and the jaws at any opening."""

    def __init__(self, model: XleRobotModel, arm: str):
        """The scene of `model`'s `arm` (its fingers), and a segmentation renderer for its wrist camera."""
        require_package("mujoco", extra="felix_jr_sim")
        self.model, self.arm = model, arm
        self.scene: Scene = build(model, arm)
        m = self.scene.m
        width, height = (int(v) for v in m.cam_resolution[m.camera(robot.WRIST_CAMERA).id])
        self.renderer = mujoco.Renderer(m, height, width)
        self.renderer.enable_segmentation_rendering()
        self.ids = (m.geom(robot.FINGER_FIXED).id, m.geom(robot.FINGER_MOVING).id)
        self.gripper = m.joint(robot.GRIPPER).qposadr[0]

    def set_arm(self, pose: dict[str, float]) -> None:
        """The arm at the frame's readings (the gripper's is set per opening)."""
        self.scene.set_pose({k: v for k, v in pose.items() if k != "gripper.pos"})

    def masks(self, opening_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """(fixed, moving) finger masks (H, W bool) with the jaws `opening_deg` open."""
        self.scene.d.qpos[self.gripper] = np.radians(opening_deg)
        mujoco.mj_forward(self.scene.m, self.scene.d)
        self.renderer.update_scene(self.scene.d, camera=robot.WRIST_CAMERA)
        ids = self.renderer.render()[..., 0]
        return ids == self.ids[0], ids == self.ids[1]

    def close(self) -> None:
        """Free the renderer."""
        self.renderer.close()


def finger_pixels(image: np.ndarray, clean: bool = True) -> np.ndarray:
    """The pixels of the fingers' color (H, W bool; `FINGER_*`), and of anything else that color.

    `clean`: specks and pinholes taken out (an opening and a closing), which also blunts a sharp tip by a few pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (
        (hue >= FINGER_HUE[0])
        & (hue <= FINGER_HUE[1])
        & (saturation >= FINGER_MIN_SATURATION)
        & (value >= FINGER_MIN_VALUE)
    )
    if not clean:
        return mask
    return _clean(mask)


def _clean(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def classify(
    image: np.ndarray, finger_sample: np.ndarray, table_sample: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The finger's and the table's pixels (H, W bool each, the finger's raw: before `_clean`), by this frame's colors.

    The colors are learned from the frame: `finger_sample` (inside the fixed finger, where the sim puts it: its
    placement is the calibration's, reliable) and `table_sample` (away from both fingers), the lit pixels of each a
    Gaussian in Lab; each pixel goes to the likelier. Dark pixels are neither (the fingers' shaded sides, their black
    bases, shadows). The light, the white balance and the table change between sessions (2026-11-15: the fingers bluish
    in one, near white in the next), so fixed thresholds (`finger_pixels`, `wood_pixels`) are only the fallback, with
    too few samples.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    fingers = finger_sample & (value >= LIT_MIN_VALUE)
    table = table_sample & (value >= WOOD_MIN_VALUE)
    if fingers.sum() < MIN_SAMPLE_PX or table.sum() < MIN_SAMPLE_PX:
        return finger_pixels(image, clean=False), wood_pixels(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

    def cost(sample: np.ndarray) -> np.ndarray:
        x = lab[sample.reshape(-1)]
        mean, cov = x.mean(axis=0), np.cov(x.T) + COLOR_NOISE * np.eye(3)
        d = lab - mean
        return (np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d) + np.log(np.linalg.det(cov))).reshape(
            value.shape
        )

    finger_cost, table_cost = cost(fingers), cost(table)
    return (finger_cost < table_cost) & (value >= FINGER_MIN_VALUE), (table_cost <= finger_cost) & (
        value >= WOOD_MIN_VALUE
    )


def wood_pixels(image: np.ndarray) -> np.ndarray:
    """The pixels of the wood's color (H, W bool; `WOOD_*`): warm, out of shadow."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, value = hsv[..., 0], hsv[..., 2]
    return ((hue <= WOOD_HUE_BELOW) | (hue >= WOOD_HUE_ABOVE)) & (value >= WOOD_MIN_VALUE)


def agreement(outline: np.ndarray, finger: np.ndarray, wood: np.ndarray) -> float:
    """How well a sim outline fits what was seen.

    The finger pixels it covers less the wood it covers, over the finger pixels (1: it covers all of the finger and no
    wood; dark pixels count neither way).
    """
    n = finger.sum()
    return float(((outline & finger).sum() - (outline & wood).sum()) / n) if n else 0.0


def _project(model: XleRobotModel, arm: str, q: np.ndarray, world: np.ndarray) -> np.ndarray:
    """A root-frame point's pixel in the undistorted wrist frame (the calibrated pinhole)."""
    camera = np.linalg.inv(model.wrist_camera_pose(arm, q)) @ np.r_[world, 1.0]
    return (model.wrist_cameras[arm].matrix() @ camera[:3])[:2] / camera[2]


def tip_pixel(model: XleRobotModel, arm: str, q: np.ndarray, opening_deg: float) -> np.ndarray:
    """The moving finger's tip in the undistorted wrist frame, `opening_deg` open: the corner the wrist camera sees.

    The XLeRobot finger's (`sim2real.robot.FINGER_MOVING_WRIST_TIP_M`), turned by its mount: not the model's generic
    `wrist_face`, 1.1 mm short of it along the finger (the tip's edge leans), which the fit took up in the hinge.
    """
    mount = model.moving_jaw(arm, model.finger_mount_deg(arm, "moving"))[:3, :3]
    corner = mount @ robot.FINGER_MOVING_WRIST_TIP_M
    on_the_jaw = model.moving_jaw(arm, opening_deg) @ np.r_[corner, 1.0]  # (pinch frame)
    return _project(model, arm, q, (model.frame_pose(pinch_frame(arm), q) @ on_the_jaw)[:3])


def hinge_pixel(model: XleRobotModel, arm: str, q: np.ndarray) -> np.ndarray:
    """A point of the gripper's hinge axis in the undistorted wrist frame (the moving jaw's origin is on it)."""
    return _project(model, arm, q, model.frame_pose(f"{arm}_moving_jaw_so101_v1_link", q)[:3, 3])


def detected_tip(
    moving: np.ndarray, hinge: np.ndarray, toward: np.ndarray, raw: np.ndarray | None = None
) -> np.ndarray | None:
    """The detected moving finger's tip (None if cut off by the frame's edge).

    The farthest pixel from the hinge, along the direction to `toward` (where the map puts it), of the finger's pieces
    (specks left out: a patchy detection splits the finger, and its largest piece can be its base); with `raw` (the
    finger's pixels before the clean-up, which blunts the tip), the farthest of those within a few pixels of them.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(moving.astype(np.uint8))
    pieces = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_PIECE_PX]
    if not pieces:
        return None
    piece = np.isin(labels, pieces)
    if raw is not None:
        piece = raw & _widen(piece, TIP_RAW_PX)
    ys, xs = np.nonzero(piece)
    direction = (toward - hinge) / np.linalg.norm(toward - hinge)
    k = int(np.argmax((xs - hinge[0]) * direction[0] + (ys - hinge[1]) * direction[1]))
    tip = np.array([xs[k], ys[k]], dtype=float)
    height, width = moving.shape
    if min(tip[0], tip[1], width - 1 - tip[0], height - 1 - tip[1]) < EDGE_PX:
        return None
    return tip


def _widen(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


@dataclass
class Detected:
    """What a frame's fit saw.

    The detected fingers and moving tip, and the sim's fingers and the model's tip at the fitted and at the map's
    opening.
    """

    fixed: np.ndarray
    moving: np.ndarray
    tip: np.ndarray | None
    sim_fitted: tuple[np.ndarray, np.ndarray]
    sim_map: tuple[np.ndarray, np.ndarray]
    tip_fitted: np.ndarray | None
    tip_map: np.ndarray
    wood: np.ndarray  # the wood's pixels where the moving finger could be
    samples: tuple[
        np.ndarray, np.ndarray
    ]  # where the colors were learned: the finger's, the table's (`classify`)
    fixed_wood: np.ndarray  # the wood's pixels where the fixed finger could be
    raw: np.ndarray  # the fingers' pixels before the clean-up (which blunts the tips: `detected_tip`)


def fit_frame(fingers: FingerRenderer, frame: SweepFrame, image: np.ndarray) -> tuple[FingerFit, Detected]:
    """Fit the moving finger's opening in an undistorted frame by its tip (the module docstring)."""
    model, arm = fingers.model, frame.arm
    fingers.set_arm(frame.pose)
    q = model.joint_vector({arm: frame.pose})
    pos = frame.pose["gripper.pos"]
    map_deg = model.opening_deg(arm, pos)
    fixed_sim, moving_map = fingers.masks(map_deg)
    reach = np.logical_or.reduce(
        [
            fingers.masks(float(a))[1]
            for a in np.arange(map_deg - SEARCH_DEG, map_deg + SEARCH_DEG + 1e-9, 2.5)
        ]
    )
    fixed_region = _widen(fixed_sim, REGION_PX)
    moving_region = _widen(reach, REGION_PX) & ~fixed_region  # (where the regions meet, the fixed finger's)
    finger_sample = cv2.erode(fixed_sim.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    table_sample = ~_widen(fixed_sim | reach, REGION_PX + TABLE_SAMPLE_PX)
    raw_colored, wood = classify(image, finger_sample, table_sample)
    colored = _clean(raw_colored)
    fixed = colored & fixed_region
    moving = colored & moving_region
    fixed_agreement = agreement(fixed_sim, fixed, wood & fixed_region)
    tip_map = tip_pixel(model, arm, q, map_deg)
    u, v = (int(round(c)) for c in tip_map)
    on_the_fixed = 0 <= v < fixed_region.shape[0] and 0 <= u < fixed_region.shape[1] and fixed_region[v, u]
    enough = (
        moving.sum() >= MIN_FINGER_PX and not on_the_fixed
    )  # nearly closed: the tips cannot be told apart
    raw = raw_colored & moving_region
    tip = detected_tip(moving, hinge_pixel(model, arm, q), tip_map, raw) if enough else None
    if tip is None:
        fit = FingerFit(pos, map_deg, None, None, None, fixed_agreement)
        sims = (fixed_sim, moving_map)
        return fit, Detected(
            fixed,
            moving,
            None,
            sims,
            sims,
            None,
            tip_map,
            wood & moving_region,
            (finger_sample, table_sample),
            wood & fixed_region,
            raw_colored,
        )

    def miss(angle: float) -> float:
        return float(np.linalg.norm(tip_pixel(model, arm, q, angle) - tip))

    coarse = np.arange(map_deg - SEARCH_DEG, map_deg + SEARCH_DEG + 1e-9, COARSE_DEG)
    best = float(min(coarse, key=miss))
    best = float(min(np.arange(best - COARSE_DEG, best + COARSE_DEG + 1e-9, FINE_DEG), key=miss))
    fit = FingerFit(pos, map_deg, best, miss(best), miss(map_deg), fixed_agreement)
    seen = Detected(
        fixed,
        moving,
        tip,
        (fixed_sim, fingers.masks(best)[1]),
        (fixed_sim, moving_map),
        tip_pixel(model, arm, q, best),
        tip_map,
        wood & moving_region,
        (finger_sample, table_sample),
        wood & fixed_region,
        raw_colored,
    )
    return fit, seen


def _outline(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, color, thickness)


def overlay(image: np.ndarray, fit: FingerFit, seen: Detected, arm: str, title: str = "") -> np.ndarray:
    """The frame with the outlines and tips, `title` above (which model drew them).

    Detected in green, the sim's and the model's at the fitted opening in orange, at the map's in blue.
    """
    out = image.copy()
    for mask in seen.sim_map:
        _outline(out, mask, BLUE, 1)
    for mask in seen.sim_fitted:
        _outline(out, mask, ORANGE, 2)
    _outline(out, seen.fixed | seen.moving, GREEN, 2)
    for point, color, radius in ((seen.tip_map, BLUE, 5), (seen.tip_fitted, ORANGE, 7), (seen.tip, GREEN, 4)):
        if point is not None:
            cv2.circle(out, tuple(int(round(v)) for v in point), radius, color, 2)
    lines = [title] if title else []
    lines += [
        f"{arm} gripper.pos {fit.gripper_pos:.1f}: map {fit.map_deg:.1f} deg",
        "moving finger's tip: not seen"
        if fit.fitted_deg is None
        else f"fitted {fit.fitted_deg:.1f} deg (error {fit.error_deg:+.1f}); tip {fit.tip_px:.0f} px off (at the "
        f"map's {fit.tip_px_at_map:.0f})",
        f"fixed finger agreement {fit.fixed_agreement:.2f}",
        "green: detected  orange: sim, fitted  blue: sim, the map's",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(out, line, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def summarize(fits: Sequence[FingerFit], model: XleRobotModel, arm: str) -> dict[str, Any]:
    """The map's error over the sweep: its mean, and the slope of the fitted opening against the map's."""
    good = [f for f in fits if f.fitted_deg is not None]
    out: dict[str, Any] = {"arm": arm, "frames": [asdict(f) | {"error_deg": f.error_deg} for f in fits]}
    if not good:
        return out
    errors = np.array([f.error_deg for f in good])
    out |= {"mean_error_deg": float(errors.mean()), "spread_deg": float(errors.std())}
    if len(good) >= 2:
        slope, intercept = np.polyfit([f.map_deg for f in good], [f.fitted_deg for f in good], 1)
        out |= {"slope": float(slope), "intercept_deg": float(intercept)}
    gripper_map = model.gripper_maps[arm]
    # a constant error: where the tips meet is off by it (in gripper.pos)
    out["pos_closed_now"] = gripper_map.pos_closed
    out["pos_closed_if_constant"] = gripper_map.pos_closed - float(errors.mean()) / gripper_map.deg_per_unit
    return out


def _settled_frame(client, arm: str) -> SweepFrame:
    """A wrist frame once the gripper's reading is still (`STILL_*`), taken after it was."""
    arm_id = f"{arm}_arm"
    history: list[tuple[float, float]] = []
    deadline = time.monotonic() + STILL_TIMEOUT_S
    while time.monotonic() < deadline:
        now = time.monotonic()
        history = [(t, p) for t, p in history if now - t <= STILL_WINDOW_S]
        history.append((now, float(client.get_state()[arm_id]["gripper.pos"])))
        if now - history[0][0] >= 0.9 * STILL_WINDOW_S and np.ptp([p for _, p in history]) <= STILL_DEG:
            break
        time.sleep(STILL_POLL_S)
    first = client.wrist_capture(arm)
    shot = client.wrist_capture(arm, after=first.timestamp)
    return SweepFrame(arm, {k: float(v) for k, v in shot.state[arm_id].items()}, shot.color)


def table_height(model: XleRobotModel, arm: str, table_z: float | None = None) -> float:
    """The table's height (m, root frame): `table_z`, else the higher of the scene's and the arm's base."""
    if table_z is not None:
        return table_z
    return max(TABLE_Z, float(model.frame_pose(f"{arm}_base_link", model.joint_vector())[2, 3]))


def _lowest_over_table(fingers: FingerRenderer, table_z: float) -> float:
    """The swept bodies' lowest point over the table's footprint, above `table_z` (m; inf: none over it), as posed."""
    m, d = fingers.scene.m, fingers.scene.d
    center, half = d.geom("table").xpos, m.geom("table").size
    bodies = {m.body(name).id for name in SWEPT_BODIES}
    low = np.inf
    for i in range(m.ngeom):
        if m.geom_bodyid[i] not in bodies or m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = m.geom_dataid[i]
        verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        world = d.geom_xpos[i] + verts @ d.geom_xmat[i].reshape(3, 3).T
        over = (np.abs(world[:, 0] - center[0]) <= half[0]) & (np.abs(world[:, 1] - center[1]) <= half[1])
        if over.any():
            low = min(low, float(world[over, 2].min()))
    return low - table_z


def clearance(
    fingers: FingerRenderer, pose: dict[str, float], positions: Sequence[float], table_z: float
) -> list[float]:
    """At each position, the swept bodies' lowest point over the table, above it (m; module docstring).

    The arm at `pose` (its gripper's reading ignored), the jaws at each position's opening (the gripper map).
    """
    model, arm = fingers.model, fingers.arm
    fingers.set_arm(pose)
    lowest = []
    for pos in positions:
        fingers.scene.d.qpos[fingers.gripper] = np.radians(model.opening_deg(arm, pos))
        mujoco.mj_forward(fingers.scene.m, fingers.scene.d)
        lowest.append(_lowest_over_table(fingers, table_z))
    return lowest


def path_clearance(
    fingers: FingerRenderer, start: dict[str, float], end: dict[str, float], table_z: float
) -> tuple[float, float]:
    """The swept bodies' lowest over the table (m above it) at `start`, and on the way from it to `end`.

    The joints moving together, the jaws as at `start`.
    """
    opening = fingers.model.opening_deg(fingers.arm, start["gripper.pos"])
    lowest = []
    for t in np.linspace(0.0, 1.0, PATH_STEPS + 1):
        fingers.set_arm({k: start[k] + t * (end[k] - start[k]) for k in end if k != "gripper.pos"})
        fingers.scene.d.qpos[fingers.gripper] = np.radians(opening)
        mujoco.mj_forward(fingers.scene.m, fingers.scene.d)
        lowest.append(_lowest_over_table(fingers, table_z))
    return float(lowest[0]), float(min(lowest[1:]))


def table_view(fingers: FingerRenderer, pose: dict[str, float]) -> float:
    """The share of the wrist camera's view that is table at `pose` (the fingers aside)."""
    fingers.set_arm(pose)
    mujoco.mj_forward(fingers.scene.m, fingers.scene.d)
    fingers.renderer.update_scene(fingers.scene.d, camera=robot.WRIST_CAMERA)
    ids = fingers.renderer.render()[..., 0]
    rest = ~np.isin(ids, fingers.ids)
    return float((ids[rest] == fingers.scene.m.geom("table").id).mean())


def sweep(client, arm: str, positions: Sequence[float], on_frame, index: int = 0) -> list[SweepFrame]:
    """Step the gripper through `positions`, a settled frame at each (`on_frame(frame)` as it comes; sweep `index`).

    The gripper goes back to where it started, however it ends.
    """
    arm_id = f"{arm}_arm"
    start = float(client.get_state()[arm_id]["gripper.pos"])
    frames: list[SweepFrame] = []
    try:
        for target in positions:
            client.move_joints({arm_id: {"gripper.pos": float(target)}})
            frames.append(replace(_settled_frame(client, arm), sweep=index))
            on_frame(frames[-1])
        return frames
    finally:
        client.move_joints({arm_id: {"gripper.pos": start}})


def save_frame(folder: Path, i: int, frame: SweepFrame) -> None:
    """Save a frame as `frame_<i>.npz` (its image, readings, arm and sweep)."""
    np.savez_compressed(
        folder / f"frame_{i:02d}.npz",
        color=frame.color,
        pose=json.dumps(frame.pose),
        arm=frame.arm,
        sweep=frame.sweep,
    )


def load_frames(folder: Path) -> list[SweepFrame]:
    """A saved sweep's frames, in order."""
    frames = []
    for path in sorted(folder.glob("frame_*.npz")):
        data = np.load(path)
        sweep = int(data["sweep"]) if "sweep" in data.files else 0  # (saved before repeats: one sweep)
        frames.append(SweepFrame(str(data["arm"]), json.loads(str(data["pose"])), data["color"], sweep))
    return frames


class Fitter:
    """Fits frames as they come, shows and saves them (the module docstring)."""

    def __init__(self, model: XleRobotModel, arm: str, folder: Path, window: bool, prefix: str = ""):
        """Fit `arm`'s frames with `model`, saving into `folder` (file names after `prefix`); `window`: show each."""
        self.model, self.arm, self.folder, self.window, self.prefix = model, arm, folder, window, prefix
        self.fingers = FingerRenderer(model, arm)
        self.fits: list[FingerFit] = []
        self.sheets: list[np.ndarray] = []
        self.seen: list[tuple[SweepFrame, Detected, FingerFit]] = []

    def __call__(self, frame: SweepFrame) -> None:
        """Fit a frame, save its overlay, print and show it."""
        image = undistort(frame.color, self.model.wrist_cameras[self.arm])
        fit, seen = fit_frame(self.fingers, frame, image)
        title = (
            "WITH THE FITTED GEOMETRY" if self.prefix == "corrected_" else "as calibrated (before the fit)"
        )
        shown = overlay(image, fit, seen, self.arm, title)
        self.fits.append(fit)
        self.sheets.append(shown)
        self.seen.append((frame, seen, fit))
        i = len(self.fits) - 1
        cv2.imwrite(
            str(self.folder / f"{self.prefix}fit_{i:02d}.png"), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
        )
        print(shown_line(fit))
        if self.window:
            cv2.imshow("gripper sweep", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def finish(self) -> dict[str, Any]:
        """The sweep's summary (`summarize`), saved with a contact sheet of the frames."""
        self.fingers.close()
        result = summarize(self.fits, self.model, self.arm)
        if self.sheets:
            cols = 3
            blank = np.zeros_like(self.sheets[0])
            rows = [
                np.hstack((self.sheets[i : i + cols] + [blank] * cols)[:cols])
                for i in range(0, len(self.sheets), cols)
            ]
            sheet = self.folder / f"{self.prefix}sheet.png"
            cv2.imwrite(str(sheet), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        return result


@dataclass
class GripperGeometry:
    """The gripper's geometry, fitted over a sweep (the module docstring)."""

    hinge_shift_mm: tuple[
        float, float, float
    ]  # the hinge's move, in the gripper's frame (its joint's origin)
    opening_offset_deg: float  # the real opening less the map's: `pos_closed` moves by -it / deg_per_unit
    finger_mount_deg: float  # the moving finger's turn on its jaw (`XleRobotModel.finger_mount_deg`)
    fixed_finger_mount_deg: float  # the fixed finger's turn about its tip (likewise)
    tip_rms_px: (
        float  # the model's tips from the detected ones, over the sweep, with the fitted hinge and offset
    )
    tip_rms_px_before: float  # the same with the model as it was
    mount_scores: dict[
        float, float
    ]  # the moving finger's mount's score (agreement, summed over the frames) by angle
    fixed_mount_scores: dict[
        float, float
    ]  # the fixed finger's mount's score (agreement, mean over the frames)
    used: tuple[float, ...]  # the frames' gripper positions fitted to
    dropped: tuple[str, ...]  # the frames left out, and why
    sweep_hinges_mm: tuple[
        tuple[float, float, float], ...
    ] = ()  # each sweep's own hinge (enough tips), the spread


def corrected_model(
    model: XleRobotModel,
    arm: str,
    hinge_shift_m: np.ndarray,
    offset_deg: float,
    mount_deg: float,
    fixed_mount_deg: float | None = None,
) -> XleRobotModel:
    """A copy of `model` with `arm`'s gripper corrected (`with_gripper_hinge`, `with_finger_mounts`; the rest as it was).

    Its hinge moved by `hinge_shift_m` (the gripper's frame), its map's `pos_closed` by the opening offset, and its
    fingers' mounts set to `mount_deg` (the moving one) and `fixed_mount_deg` (the fixed one; None: as it was).
    """
    corrected = model.with_gripper_hinge(arm).with_finger_mounts(arm, mount_deg, fixed_mount_deg)
    estimates = corrected.estimate_dict()
    for axis, shift in zip(("x", "y", "z"), hinge_shift_m, strict=True):
        estimates[f"pos:{arm}_gripper:{axis}"] += float(shift)
    corrected.set_estimates(estimates)
    gripper_map = corrected.gripper_maps[arm]
    corrected.gripper_maps = dict(corrected.gripper_maps) | {
        arm: replace(
            gripper_map,
            pos_closed=gripper_map.pos_closed - offset_deg / gripper_map.deg_per_unit,
        )
    }
    return corrected


def fit_hinge(
    model: XleRobotModel, arm: str, tips: Sequence[tuple[np.ndarray, float, np.ndarray]]
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """The hinge's shift (m, the gripper's frame) and the opening offset (deg) that put the model's tips on the detected.

    `tips`: (q, the map's opening, the detected tip) per frame. Also returned: each tip's miss (px) after and before.

    The hinge moves in the plane its tip turns in (along its own axis it changes nothing); the tips' circle passes
    through the fixed tip at an opening of 0 whatever the hinge (the tips meet there), so its radius follows.
    """
    require_package("scipy", extra="felix_jr")
    q0 = tips[0][0]
    jaw = model.frame_pose(f"{arm}_moving_jaw_so101_v1_link", q0)[:3, :3]
    axis = jaw[:, 2]  # the hinge's axis (the URDF joint's z)
    e1 = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    gripper = model.frame_pose(f"{arm}_gripper_link", q0)[:3, :3]
    trial = model.with_gripper_hinge(arm)
    trial = copy.copy(trial) if trial is model else trial
    base = trial.estimate_dict()

    def shift_of(x: np.ndarray) -> np.ndarray:
        return gripper.T @ (x[0] * e1 + x[1] * e2)

    def residuals(x: np.ndarray) -> np.ndarray:
        shift = shift_of(x)
        trial.set_estimates(
            base
            | {
                f"pos:{arm}_gripper:{a}": base[f"pos:{arm}_gripper:{a}"] + shift[i]
                for i, a in enumerate("xyz")
            }
        )
        return np.concatenate([tip_pixel(trial, arm, q, opening + x[2]) - tip for q, opening, tip in tips])

    before = residuals(np.zeros(3))
    # robust: a tip detected wrong pulls the fit by its miss's square root, past a few pixels
    result = least_squares(
        residuals, np.zeros(3), x_scale=[0.003, 0.003, 1.0], loss="soft_l1", f_scale=ROBUST_SCALE_PX
    )
    miss = lambda r: np.linalg.norm(r.reshape(-1, 2), axis=1)  # noqa: E731
    return shift_of(result.x), float(result.x[2]), miss(result.fun), miss(before)


def fit_mount(
    model: XleRobotModel,
    arm: str,
    seen: Sequence[tuple[SweepFrame, Detected, FingerFit]],
    angles: Sequence[float],
) -> dict[float, float]:
    """The moving finger's mount's score by angle.

    Its sim outline's `agreement` with the detected finger, summed over the frames whose tip was seen (each at its
    map's opening, the model's hinge and map as given).
    """
    scores = {}
    for angle in angles:
        trial = model.with_finger_mounts(arm, moving_deg=float(angle))
        fingers = FingerRenderer(trial, arm)
        total = 0.0
        for frame, detected, _ in seen:
            fingers.set_arm(frame.pose)
            moving = fingers.masks(trial.opening_deg(arm, frame.pose["gripper.pos"]))[1]
            total += agreement(moving, detected.moving, detected.wood)
        fingers.close()
        scores[float(angle)] = total
    return scores


def fit_fixed_mount(
    model: XleRobotModel,
    arm: str,
    seen: Sequence[tuple[SweepFrame, Detected, FingerFit]],
    angles: Sequence[float],
) -> tuple[dict[float, float], dict[float, list[float]]]:
    """The fixed finger's mount's score by angle, and each frame's.

    Its sim outline's `agreement` with the detected fixed finger, the mean over the frames with the moving tip seen
    (with the jaws closed the moving finger lies against it).
    """
    open_frames = [(frame, detected) for frame, detected, fit in seen if fit.fitted_deg is not None]
    scores, per_frame = {}, {}
    for angle in angles:
        trial = model.with_finger_mounts(arm, fixed_deg=float(angle))
        fingers = FingerRenderer(trial, arm)
        each = []
        for frame, detected in open_frames:
            fingers.set_arm(frame.pose)
            fixed = fingers.masks(trial.opening_deg(arm, frame.pose["gripper.pos"]))[0]
            each.append(agreement(fixed, detected.fixed, detected.fixed_wood))
        fingers.close()
        scores[float(angle)] = float(np.mean(each)) if each else 0.0
        per_frame[float(angle)] = each
    return scores, per_frame


def trusted(
    seen: Sequence[tuple[SweepFrame, Detected, FingerFit]], fixed_agreements: Sequence[float] | None = None
) -> tuple[list, list[str]]:
    """The frames to fit the geometry to, and why each other one is left out.

    A tip seen, and the fixed finger's `agreement` at least `MIN_FIXED_AGREEMENT` (lower: the colors misread): with its
    mount fitted when `fixed_agreements` are given (one per frame with the tip seen, in order). (A tip detected wrong is
    found by the geometry's fit instead: `fit_geometry`.)
    """
    kept, dropped = [], []
    agreements = iter(fixed_agreements) if fixed_agreements is not None else None
    for frame, detected, fit in seen:
        pos = f"gripper.pos {fit.gripper_pos:.1f}"
        if detected.tip is None or fit.tip_px is None:
            dropped.append(f"{pos}: the tip not seen")
            continue
        fixed_agreement = next(agreements) if agreements is not None else fit.fixed_agreement
        if fixed_agreement < MIN_FIXED_AGREEMENT:
            dropped.append(f"{pos}: the fixed finger's agreement {fixed_agreement:.2f}")
        else:
            kept.append((frame, detected, fit))
    return kept, dropped


def fit_geometry(
    model: XleRobotModel, arm: str, seen: Sequence[tuple[SweepFrame, Detected, FingerFit]]
) -> GripperGeometry | None:
    """The fixed finger's mount, the hinge and the opening offset, then the moving finger's mount (`trusted` frames).

    The hinge's fit is robust, and a frame whose tip it still misses by more than `OUTLIER_PX` is left out and the fit
    done again (something else taken for the tip: 2026-11-15, the finger's base). None with fewer than
    `MIN_GEOMETRY_FRAMES` frames.
    """
    fixed_scores, fixed_each = fit_fixed_mount(
        model, arm, seen, np.arange(-MOUNT_SEARCH_DEG, MOUNT_SEARCH_DEG + 1e-9, MOUNT_COARSE_DEG)
    )
    best_fixed = max(fixed_scores, key=fixed_scores.__getitem__)
    fine_scores, fine_each = fit_fixed_mount(
        model,
        arm,
        seen,
        np.arange(best_fixed - MOUNT_COARSE_DEG, best_fixed + MOUNT_COARSE_DEG + 1e-9, MOUNT_FINE_DEG),
    )
    fixed_scores = dict(sorted((fixed_scores | fine_scores).items()))
    fixed_mount = max(fixed_scores, key=fixed_scores.__getitem__)
    model = corrected_model(model, arm, np.zeros(3), 0.0, model.finger_mount_deg(arm, "moving"), fixed_mount)
    kept, dropped = trusted(seen, (fixed_each | fine_each)[fixed_mount])
    while True:
        if len(kept) < MIN_GEOMETRY_FRAMES:
            return None
        tips = [
            (
                model.joint_vector({arm: frame.pose}),
                model.opening_deg(arm, frame.pose["gripper.pos"]),
                detected.tip,
            )
            for frame, detected, _ in kept
        ]
        shift, offset, misses, misses_before = fit_hinge(model, arm, tips)
        worst = int(np.argmax(misses))
        if misses[worst] <= OUTLIER_PX:
            break
        fit = kept.pop(worst)[2]
        dropped.append(
            f"gripper.pos {fit.gripper_pos:.1f}: its tip {misses[worst]:.0f} px from the fitted geometry's"
        )
    rms, before = float(np.sqrt(np.mean(misses**2))), float(np.sqrt(np.mean(misses_before**2)))
    hinged = corrected_model(model, arm, shift, offset, 0.0)  # (the fixed finger's mount as fitted, above)
    coarse = fit_mount(
        hinged, arm, kept, np.arange(-MOUNT_SEARCH_DEG, MOUNT_SEARCH_DEG + 1e-9, MOUNT_COARSE_DEG)
    )
    best = max(coarse, key=coarse.__getitem__)
    fine = fit_mount(
        hinged, arm, kept, np.arange(best - MOUNT_COARSE_DEG, best + MOUNT_COARSE_DEG + 1e-9, MOUNT_FINE_DEG)
    )
    scores = dict(sorted((coarse | fine).items()))
    best = max(scores, key=scores.__getitem__)
    x, y, z = (float(v) * 1000.0 for v in shift)
    used = tuple(fit.gripper_pos for _, _, fit in kept)
    # each sweep alone: how far its hinge lands from the others' (the pooled fit is all of them at once)
    sweeps_kept = sorted({frame.sweep for frame, _, _ in kept})
    sweep_hinges = []
    for index in sweeps_kept if len(sweeps_kept) > 1 else []:
        own = [tip for tip, (frame, _, _) in zip(tips, kept, strict=True) if frame.sweep == index]
        if len(own) >= MIN_GEOMETRY_FRAMES:
            sx, sy, sz = (float(v) * 1000.0 for v in fit_hinge(model, arm, own)[0])
            sweep_hinges.append((sx, sy, sz))
    return GripperGeometry(
        (x, y, z),
        offset,
        best,
        fixed_mount,
        rms,
        before,
        scores,
        fixed_scores,
        used,
        tuple(dropped),
        tuple(sweep_hinges),
    )


def shown_line(fit: FingerFit) -> str:
    """A frame's fit in a line."""
    if fit.fitted_deg is None:
        return f"gripper.pos {fit.gripper_pos:5.1f}: map {fit.map_deg:6.1f} deg; the moving finger's tip not seen"
    return (
        f"gripper.pos {fit.gripper_pos:5.1f}: map {fit.map_deg:6.1f} deg, fitted {fit.fitted_deg:6.1f} "
        f"(error {fit.error_deg:+5.1f}); tip {fit.tip_px:4.1f} px off (map's {fit.tip_px_at_map:4.1f}); fixed finger "
        f"agreement {fit.fixed_agreement:.2f}"
    )


def report(result: dict[str, Any]) -> None:
    """Print a sweep's summary."""
    if "mean_error_deg" not in result:
        print("no frame had enough of the moving finger to fit")
        return
    print(
        f"{result['arm']}: the map's error {result['mean_error_deg']:+.1f} deg on average (spread "
        f"{result['spread_deg']:.1f})"
        + (
            f"; fitted = {result['slope']:.3f} x map {result['intercept_deg']:+.1f} deg"
            if "slope" in result
            else ""
        )
    )
    print(
        f"if the error is the same at every opening: pos_closed {result['pos_closed_now']:.2f} -> "
        f"{result['pos_closed_if_constant']:.2f}"
    )


def report_geometry(geometry: GripperGeometry, model: XleRobotModel, arm: str) -> None:
    """Print the fitted geometry."""
    x, y, z = geometry.hinge_shift_mm
    gripper_map = model.gripper_maps[arm]
    print(
        f"the hinge moves ({x:+.1f}, {y:+.1f}, {z:+.1f}) mm ({np.linalg.norm(geometry.hinge_shift_mm):.1f}; the "
        f"gripper's frame); the opening is off by {geometry.opening_offset_deg:+.2f} deg (pos_closed "
        f"{gripper_map.pos_closed:.2f} -> {gripper_map.pos_closed - geometry.opening_offset_deg / gripper_map.deg_per_unit:.2f}); "
        f"the tips {geometry.tip_rms_px_before:.1f} -> {geometry.tip_rms_px:.1f} px RMS"
    )
    print(
        f"the fingers' mounts: the moving one {geometry.finger_mount_deg:+.1f} deg, the fixed one "
        f"{geometry.fixed_finger_mount_deg:+.1f} deg (its agreement {geometry.fixed_mount_scores[0.0]:.2f} -> "
        f"{geometry.fixed_mount_scores[geometry.fixed_finger_mount_deg]:.2f})"
    )
    print(f"fitted to the frames at gripper.pos {', '.join(f'{p:.1f}' for p in geometry.used)}")
    if geometry.sweep_hinges_mm:
        hinges = np.array(geometry.sweep_hinges_mm)
        spread = np.linalg.norm(hinges - np.array(geometry.hinge_shift_mm), axis=1)
        each = "; ".join("(" + ", ".join(f"{v:+.1f}" for v in h) + ")" for h in hinges)
        print(f"each sweep's hinge alone (mm): {each} -- {spread.max():.1f} mm from the pooled one at most")
    for why in geometry.dropped:
        print(f"  left out: {why}")


def preview_check(fit: FingerFit) -> list[str]:
    """Why a preview frame fails (empty: it passes): the limits the geometry's fit holds frames to."""
    problems = []
    if fit.fixed_agreement < MIN_FIXED_AGREEMENT:
        problems.append(
            f"the fixed finger's agreement {fit.fixed_agreement:.2f} (at least {MIN_FIXED_AGREEMENT})"
        )
    if fit.tip_px is None:
        problems.append("the moving finger's tip not seen")
    elif fit.tip_px > OUTLIER_PX:
        problems.append(f"the moving tip {fit.tip_px:.0f} px from the model's (at most {OUTLIER_PX:g})")
    return problems


def preview_view(
    image: np.ndarray, fit: FingerFit, seen: Detected, arm: str, problems: Sequence[str]
) -> np.ndarray:
    """A preview frame, with the verdict.

    The fit's overlay, the pixels taken for the fingers tinted, and where the colors were learned (yellow).
    """
    tinted = image.copy()
    fingers = seen.fixed | seen.moving
    tinted[fingers] = (0.5 * tinted[fingers] + 0.5 * np.array(MAGENTA)).astype(np.uint8)
    out = overlay(tinted, fit, seen, arm)
    for sample in seen.samples:
        _outline(out, sample, YELLOW, 1)
    lines = [
        "PASS: Enter to sweep, q to stop"
        if not problems
        else "FAIL: fix the scene; f to sweep anyway, q to stop"
    ]
    lines += list(problems) + ["magenta: taken for the fingers  yellow: where the colors were learned"]
    for i, line in enumerate(lines):
        y = out.shape[0] - 12 - 22 * (len(lines) - 1 - i)
        color = (
            (0, 255, 0)
            if not problems and i == 0
            else (255, 80, 80)
            if i <= len(problems)
            else (255, 255, 255)
        )
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return out


def preview(client, arm: str, fitter: "Fitter", window: bool, ask: Callable[[str], str] = input) -> bool:
    """Before the sweep: the gripper at `PREVIEW_POS`, its fingers detected live, until the person starts or stops.

    True to sweep: on a passing frame (`preview_check`) when asked, or forced on a failing one; False to stop. In a
    window it refreshes every `PREVIEW_PERIOD_S` (Enter, f, q); without one it asks between frames (r looks again).
    The last frame is saved as `preview.png` in the sweep's folder.
    """
    client.move_joints({f"{arm}_arm": {"gripper.pos": PREVIEW_POS}})
    while True:
        shot = client.wrist_capture(arm)
        frame = SweepFrame(arm, {k: float(v) for k, v in shot.state[f"{arm}_arm"].items()}, shot.color)
        image = undistort(frame.color, fitter.model.wrist_cameras[arm])
        fit, seen = fit_frame(fitter.fingers, frame, image)
        problems = preview_check(fit)
        shown = preview_view(image, fit, seen, arm, problems)
        cv2.imwrite(str(fitter.folder / "preview.png"), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        if window:
            cv2.imshow("gripper sweep", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(int(PREVIEW_PERIOD_S * 1000)) & 0xFF
            answer = {13: "", 10: "", ord("f"): "f", ord("q"): "q"}.get(key)
            if answer is None:
                continue
        else:
            print(f"preview ({fitter.folder / 'preview.png'}): {shown_line(fit)}")
            for problem in problems:
                print(f"  {problem}")
            answer = (
                ask(
                    "Enter to sweep, r to look again, f to sweep anyway, q to stop: "
                    if not problems
                    else "FAIL: fix the scene; r to look again, f to sweep anyway, q to stop: "
                )
                .strip()
                .lower()
            )
        if answer == "q":
            return False
        if answer == "f" or (answer == "" and not problems):
            return True


def pooled(folders: Sequence[Path], out_dir: Path, stamp: str) -> list[tuple[str, Path, list[SweepFrame]]]:
    """Saved sweeps grouped by arm (in the order they come): (arm, folder, frames) each.

    One folder for an arm: fitted again in it. Several: their frames pooled (each folder's sweeps numbered after the
    one before's) into a new folder, `<arm>_pooled_<stamp>` in `out_dir`, fitted at once (`fit_geometry`).
    """
    groups: dict[str, list[tuple[Path, list[SweepFrame]]]] = {}
    for folder in folders:
        frames = load_frames(folder)
        if not frames:
            raise SystemExit(f"no frames in {folder}")
        groups.setdefault(frames[0].arm, []).append((folder, frames))
    out = []
    for arm, saved in groups.items():
        if len(saved) == 1:
            out.append((arm, saved[0][0], saved[0][1]))
            continue
        frames, offset = [], 0
        for _, own in saved:
            frames += [replace(frame, sweep=frame.sweep + offset) for frame in own]
            offset += max(frame.sweep for frame in own) + 1
        out.append((arm, out_dir / f"{arm}_pooled_{stamp}", frames))
    return out


def check_pose(
    client,
    model: XleRobotModel,
    arm: str,
    fitter: "Fitter",
    positions: Sequence[float],
    table_z: float | None,
    stay: bool,
    repeats: int,
    ask: Callable[[str], str] = input,
) -> dict[str, float] | None:
    """Before a sweep: the sim's checks of the pose (the module docstring) printed, then the person asked.

    The arm's readings where it is now; None if the checks refuse it or the person declines.
    """
    table = table_height(model, arm, table_z)
    start = {k: float(v) for k, v in client.get_state()[f"{arm}_arm"].items()}
    target = dict(start) if stay else {**SWEEP_POSES[arm], "gripper.pos": start["gripper.pos"]}
    at_start, on_the_way = (np.inf, np.inf) if stay else path_clearance(fitter.fingers, start, target, table)
    lowest = clearance(fitter.fingers, target, positions, table)
    view = table_view(fitter.fingers, target)
    print(f"{arm} arm: the table taken at {100 * table:.1f} cm (--table-z to set it)")
    if not stay:
        print(
            f"  the gripper's lowest over the table: {100 * at_start:.1f} cm where it is, {100 * on_the_way:.1f} cm "
            "on the way to the sweep pose"
        )
    for pos, low in zip(positions, lowest, strict=True):
        print(f"  gripper.pos {pos:5.1f}: the lowest over the table {100 * low:5.1f} cm")
    print(f"  the wrist camera's view: {100 * view:.0f}% table")
    if min(lowest) < MIN_CLEARANCE_M or on_the_way < min(MIN_CLEARANCE_M, at_start):
        print(f"refused: the gripper would come within {100 * MIN_CLEARANCE_M:.0f} cm of the table")
        return None
    if view < MIN_TABLE_VIEW:
        print("refused: the wrist camera would see too little of the table (the fingers' color is ambiguous)")
        return None
    if stay:
        print(
            f"The {arm} gripper will step through {list(positions)} and come back; the arm stays where it is."
        )
    else:
        print(
            f"The {arm} arm will move to its sweep pose {SWEEP_POSES[arm]}, the gripper step through "
            f"{list(positions)}"
            + (f" ({repeats} times, the arm back where it started between)" if repeats > 1 else "")
            + ", and the arms go home at the end."
        )
    print("Keep the jaws clear.")
    if ask("Enter to start, q to skip: ").strip().lower() == "q":
        return None
    return start


def run_sweeps(
    client,
    arm: str,
    fitter: "Fitter",
    positions: Sequence[float],
    repeats: int,
    stay: bool,
    start: dict[str, float],
) -> bool:
    """The sweeps (`check_pose` first), in the caller's session; False if stopped at the preview.

    To the sweep pose (unless `stay`), the preview, then `repeats` sweeps, away and back between; each frame saved and
    fitted as it comes.
    """

    def on_frame(frame: SweepFrame) -> None:
        save_frame(fitter.folder, len(fitter.fits), frame)
        fitter(frame)

    if not stay:
        client.move_joints({f"{arm}_arm": SWEEP_POSES[arm]}, max_vel=30.0, max_accel=60.0, timeout=15.0)
    if not preview(client, arm, fitter, fitter.window):
        print(f"{arm}: stopped at the preview")
        return False
    back = {k: v for k, v in start.items() if k != "gripper.pos"}
    for index in range(repeats):
        if index and not stay:  # away and back: each sweep arrives at the pose anew, as the arm settles
            print(f"{arm}: sweep {index + 1} of {repeats}: back where it started, then to the pose again")
            client.move_joints({f"{arm}_arm": back}, max_vel=30.0, max_accel=60.0, timeout=15.0)
            client.move_joints({f"{arm}_arm": SWEEP_POSES[arm]}, max_vel=30.0, max_accel=60.0, timeout=15.0)
        sweep(client, arm, positions, on_frame, index)
    return True


def run_on_robot(client, model: XleRobotModel, arm: str, fitter: "Fitter", args: argparse.Namespace) -> bool:
    """One arm's sweep on the robot (the module docstring), in a session of its own.

    Checks the pose in the sim and asks first (`check_pose`); False if refused or declined.
    """
    positions = [float(p) for p in args.positions.split(",")]
    start = check_pose(client, model, arm, fitter, positions, args.table_z, args.stay, args.repeats)
    if start is None:
        return False
    with client.session(on_disconnect="hold" if args.stay else "home"):
        if args.stay:
            client.keep_holding()  # the arm stays put when the run ends; only the gripper moved
        return run_sweeps(client, arm, fitter, positions, args.repeats, args.stay, start)


def fixed_tip(
    model: XleRobotModel, arm: str, pose: dict[str, float], detected: Detected
) -> tuple[np.ndarray, np.ndarray] | None:
    """The fixed finger's tip in one undistorted wrist frame, as detected and as the model puts it (px; None: unseen).

    Fixed to the gripper as the camera is, it stays put in the wrist frame unless the camera moves on its mount
    (`docs/hardware_risks.md`). Detected: the far end of the detected fixed finger, along the finger (from a point
    `FIXED_BACK_M` back from its tip toward it), its corner the wrist camera sees (`FINGER_FIXED_WRIST_TIP_M`). With the
    jaws closed the moving finger's tip lies against it, at about the same place.
    """
    mount = model.moving_jaw(arm, model.finger_mount_deg(arm, "fixed"))[:3, :3]
    corner = mount @ robot.FINGER_FIXED_WRIST_TIP_M
    back = mount @ (corner + np.array([0.0, 0.0, -FIXED_BACK_M]))  # (the pinch z points to the tips)
    q = model.joint_vector({arm: pose})
    pinch = model.frame_pose(pinch_frame(arm), q)
    expected = _project(model, arm, q, (pinch @ np.r_[corner, 1.0])[:3])
    base = _project(model, arm, q, (pinch @ np.r_[back, 1.0])[:3])
    found = detected_tip(detected.fixed, base, expected, detected.raw)
    return None if found is None else (found, expected)


def fixed_tips(
    model: XleRobotModel, arm: str, seen: Sequence[tuple[SweepFrame, Detected, FingerFit]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per sweep, the fixed finger's tip as detected and as the model puts it (`fixed_tip`).

    The mean over the sweep's frames with the jaws open (closed, the moving finger lies against it).
    """
    tips: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for frame, detected, fit in seen:
        if fit.fitted_deg is None:
            continue
        found = fixed_tip(model, arm, frame.pose, detected)
        if found is not None:
            tips.setdefault(frame.sweep, []).append(found)
    return {
        k: (np.mean([f for f, _ in v], axis=0), np.mean([e for _, e in v], axis=0)) for k, v in tips.items()
    }


def report_fixed_tips(tips: dict[int, tuple[np.ndarray, np.ndarray]], arm: str) -> dict[str, Any]:
    """Print each sweep's fixed tip, their spread and their distance from the model's; warn past `CAMERA_MOVED_PX`."""
    if not tips:
        return {}
    found = np.array([tips[k][0] for k in sorted(tips)])
    expected = np.array([tips[k][1] for k in sorted(tips)])
    spread = float(np.linalg.norm(found.max(axis=0) - found.min(axis=0)))
    off = float(np.linalg.norm(found - expected, axis=1).max())
    each = "; ".join(f"({x:.1f}, {y:.1f})" for x, y in found)
    print(
        f"{arm}: the fixed finger's tip in the wrist frame, per sweep: {each} px -- {spread:.1f} px between sweeps, "
        f"{off:.1f} px from the model's at most"
    )
    if spread > CAMERA_MOVED_PX or off > CAMERA_MOVED_PX:
        print(
            f"  WARNING: more than {CAMERA_MOVED_PX:g} px: the wrist camera may have moved on its mount "
            "(docs/hardware_risks.md): check it, and recalibrate the camera"
        )
    return {
        "per_sweep_px": found.tolist(),
        "model_px": expected.tolist(),
        "spread_px": spread,
        "off_model_px": off,
    }


def fit_arm(model: XleRobotModel, arm: str, fitter: "Fitter") -> tuple[dict[str, Any], XleRobotModel | None]:
    """An arm's sweep fitted (`fitter`'s frames): its summary and the model corrected by its geometry.

    The model is None when the geometry could not be fitted or does not hold (`MAX_GEOMETRY_RMS_PX`).
    """
    result = fitter.finish()
    report(result)
    result["fixed_tip"] = report_fixed_tips(fixed_tips(model, arm, fitter.seen), arm)
    geometry = fit_geometry(model, arm, fitter.seen)
    if geometry is None:
        kept, dropped = trusted(fitter.seen)
        print(f"{arm}: too few frames to fit the geometry ({len(kept)}; at least {MIN_GEOMETRY_FRAMES})")
        for why in dropped:
            print(f"  left out: {why}")
        return result, None
    result["geometry"] = asdict(geometry) | {
        "mount_scores": {f"{k:g}": v for k, v in geometry.mount_scores.items()},
        "fixed_mount_scores": {f"{k:g}": v for k, v in geometry.fixed_mount_scores.items()},
    }
    report_geometry(geometry, model, arm)
    corrected = corrected_model(
        model,
        arm,
        np.array(geometry.hinge_shift_mm) / 1000.0,
        geometry.opening_offset_deg,
        geometry.finger_mount_deg,
        geometry.fixed_finger_mount_deg,
    )
    print("the sweep again, with the corrected geometry:")
    again = Fitter(corrected, arm, fitter.folder, window=False, prefix="corrected_")
    for frame, _, _ in fitter.seen:
        again(frame)
    result["corrected"] = again.finish()
    (fitter.folder / "results.json").write_text(json.dumps(result, indent=2))
    if geometry.tip_rms_px > MAX_GEOMETRY_RMS_PX:
        print(f"{arm}: the corrected tips still miss by {geometry.tip_rms_px:.1f} px RMS: not kept")
        return result, None
    return result, corrected


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", default="left", choices=["left", "right", "both"])
    parser.add_argument("--positions", default=",".join(f"{p:g}" for p in DEFAULT_POSITIONS))
    parser.add_argument("--urdf", type=Path, default=api.MODEL)
    parser.add_argument(
        "--out", type=Path, default=None, help="write the corrected calibration here (a new file)"
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="where the sweeps go (one folder per arm)")
    parser.add_argument(
        "--refit",
        type=Path,
        nargs="+",
        default=None,
        help="fit saved sweeps again, no robot (several of an arm: pooled into one fit)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="sweeps per arm, the arm away and back between"
    )
    parser.add_argument("--table-z", type=float, default=None, help="the table's height (m, root frame)")
    parser.add_argument("--stay", action="store_true", help="sweep where the arm is; only the gripper moves")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    if args.out is not None and args.out.resolve() == args.urdf.resolve():
        raise SystemExit("--out must be a new file: --urdf is never overwritten")
    model = XleRobotModel.load(args.urdf)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweeps: list[tuple[str, Path, list[SweepFrame] | None]] = []
    if args.refit is not None:
        sweeps = list(pooled(args.refit, args.out_dir or DEFAULT_OUT, stamp))
    else:
        arms = ["left", "right"] if args.arm == "both" else [args.arm]
        sweeps = [(arm, (args.out_dir or DEFAULT_OUT) / f"{arm}_{stamp}", None) for arm in arms]

    corrected, fitters, complete = model, [], True
    client = None
    if args.refit is None:
        from lerobot.felix_jr.server.client import FelixClient

        client = FelixClient(args.endpoint, timeout_s=15.0)
    try:
        for arm, folder, saved in sweeps:
            folder.mkdir(parents=True, exist_ok=True)
            fitter = Fitter(corrected, arm, folder, window=not args.no_window)
            fitters.append(fitter)
            if saved is not None:
                for frame in saved:
                    fitter(frame)
            elif not run_on_robot(client, corrected, arm, fitter, args):
                complete = False
                continue
            _, fitted = fit_arm(corrected, arm, fitter)
            if fitted is None:
                complete = False
            else:
                corrected = fitted  # the next arm is fitted on top
            print(f"wrote {folder}")
    finally:
        if client is not None:
            client.close()
    print("the views (green: detected; orange: the sim and the model, fitted; blue: at the map's):")
    for fitter in fitters:
        corrected_sheet, sheet = fitter.folder / "corrected_sheet.png", fitter.folder / "sheet.png"
        if corrected_sheet.exists():
            print(
                f"  {fitter.arm}, WITH THE FITTED GEOMETRY (the hinge, the mount, the map): {corrected_sheet.resolve()}"
            )
        if sheet.exists():
            print(f"  {fitter.arm}, before, as calibrated (--urdf; for comparison): {sheet.resolve()}")
    if args.out is not None:
        if complete:
            corrected.save(args.out)
            print(f"wrote {args.out}")
        else:
            print(f"not writing {args.out}: an arm's geometry was not fitted (above)")
    if not args.no_window and any(f.sheets for f in fitters):
        print("any key in the window to finish")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
