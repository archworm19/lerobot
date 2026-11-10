"""Simulated joint calibrations (`joint_fit`): how many views and board placements, and of what, are enough.

`docs/camera_calibration.md`, "The rework" (open questions). A "true" robot is the model with its estimates moved by
draws from their priors (`SIGMAS`: the head, the camera mount, the arm's mount, joint zeros and origins, the tip), the
wrist camera moved on its gripper and the fingers' thickness off. Views are planned on the model as a session would
plan them, kept if the true robot would see them (the board whole in the wrist camera, clear of the fingers; whole and
not behind the arm in the head's, for a checkerboard), and observed through the true robot with noise (the board's
points, the clicks, the joints' readings). The fit starts from the model; held-out grasp-like poses score it:

- `head px`: the pinch point in the head's picture, the head aimed at it (what the harness draws and plans from);
- `wrist px`: the wrist-side fingertip in the wrist camera's picture;
- `camera mm` / `camera deg`: the wrist camera's pose relative to the pinch frame (what `wrist_contacts` measures in);
- `move mm`: a 5 cm move of the pinch point along the approach as the fitted model predicts it, against the truth.

Usage: uv run python -m lerobot.felix_jr.calibration.joint_sim [--seeds 3] [--quick]
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from lerobot.felix_jr.calibration.boards import Board, Checkerboard, Detection
from lerobot.felix_jr.calibration.joint_fit import HEAD, BoardSighting, JointProblem, Look, TipClick
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    XleRobotModel,
    pinch_frame,
    wrist_camera_link,
)
from lerobot.felix_jr.harness.look import TICKS_PER_DEG, aim_head
from lerobot.utils.import_utils import _scipy_available, require_package

if _scipy_available:
    from scipy.optimize import least_squares

ARM = "right"
REST = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": 70.0,
    "wrist_roll.pos": 0.0,
}
TABLE_BELOW_BASE_M = 0.06
ARM_RADIUS_M = 0.035  # the arm's links as capsules this thick, for what hides the board from the head
FINGER_BOX_PX = (80.0, 40.0)
AIM_PX = (
    400.0,
    170.0,
)  # the board's center in the wrist picture: up and right, away from the fingers (lower left)  # the fingers in the wrist picture: this far either side of the tips, and above them
JOINT_NOISE_DEG = 0.2
BOARD_NOISE_PX = 0.3
CLICK_NOISE_PX = 2.0
CAMERA_SHIFT_M, CAMERA_TURN_DEG = 0.010, 4.0  # the true wrist camera's pose off the model's, 1 sigma
THICKNESS_OFF_M = 0.0005
NUDGE_DEG = (0.0, -4.0, 4.0, -8.0, 8.0, -12.0, 12.0)  # the head's nudges tried, each servo
HEAD_ONLY_SPREAD_M = (
    0.12  # the head aimed up to this far around the board's center (the board must stay in view)
)


@dataclass(frozen=True)
class Scenario:
    """What a session collects, per board placement.

    Wrist views (each with a head view of the board if it can, and clicks), and head views of the board with the arms
    folded.
    """

    placements: int = 2
    arm_views: int = 10
    head_only_views: int = 3
    clicks: bool = True
    label: str = ""
    low_views: int = (
        0  # poses low over the table, as a grasp's (once, not per placement): the fingertips clicked
    )


@dataclass
class Truth:
    """The true robot: its estimates, its wrist camera's pose on the gripper, its fingers' thickness."""

    theta: np.ndarray
    gripper_from_camera: np.ndarray
    thickness: float


def _turn(deg_sigma: float, rng: np.random.Generator) -> np.ndarray:
    return cv2.Rodrigues(np.radians(rng.normal(0.0, deg_sigma, 3)))[0]


def draw_truth(model: XleRobotModel, rng: np.random.Generator, scale: float = 1.0) -> Truth:
    """The true robot: the estimates moved by their priors (times `scale`), the wrist camera and the thickness off."""
    theta = model.estimates.copy()
    for i, p in enumerate(model.parameters):
        if p.arm in (None, ARM):
            theta[i] += rng.normal(0.0, scale * p.sigma)
    gc = np.array(model.wrist_cameras[ARM].gripper_from_camera, dtype=np.float64)
    true_gc = gc.copy()
    true_gc[:3, :3] = gc[:3, :3] @ _turn(scale * CAMERA_TURN_DEG, rng)
    true_gc[:3, 3] += rng.normal(0.0, scale * CAMERA_SHIFT_M, 3)
    return Truth(theta, true_gc, JAW_THICKNESS_M + rng.normal(0.0, scale * THICKNESS_OFF_M))


class World:
    """The model (what is planned with) and the truth (what is seen), and what each camera sees."""

    def __init__(self, model: XleRobotModel, truth: Truth, board: Board, rng: np.random.Generator):
        """The model to plan with, the truth to observe, the board, the random draws (views, noise)."""
        self.model, self.truth, self.board, self.rng = model, truth, board, rng
        self.joint_noise_deg, self.board_noise_px, self.click_noise_px = (
            JOINT_NOISE_DEG,
            BOARD_NOISE_PX,
            CLICK_NOISE_PX,
        )
        self.camera = model.wrist_cameras[ARM]
        self.gc0 = np.array(self.camera.gripper_from_camera, dtype=np.float64)
        base = model.frame_pose(f"{ARM}_base_link", model.joint_vector())
        self.base = base[:3, 3]
        self.table_z = self.base[2] - TABLE_BELOW_BASE_M
        self.home_ticks = model.head_ticks(dict.fromkeys(("head_pan_joint", "head_tilt_joint"), 0.0))
        self.boards: dict[str, np.ndarray] = {}  # root_from_board of each placement `collect` made

    # ---- the cameras, true or modeled ------------------------------------------------------------------------

    def wrist_pose(self, q: np.ndarray, true: bool) -> np.ndarray:
        """root_from_camera of the wrist camera at `q`: the truth's, or the model's."""
        theta, gc = (self.truth.theta, self.truth.gripper_from_camera) if true else (None, self.gc0)
        return self.model.frame_pose(wrist_camera_link(ARM), q, theta) @ gc

    def wrist_pixels(self, q: np.ndarray, points: np.ndarray, true: bool = True) -> np.ndarray:
        """Root-frame points in the wrist camera's picture at `q`."""
        pose = self.wrist_pose(q, true)
        return self.camera.project((points - pose[:3, 3]) @ pose[:3, :3])

    def head_pixels(self, q: np.ndarray, points: np.ndarray, true: bool = True) -> np.ndarray:
        """Root-frame points in the head camera's picture at `q`."""
        return self.model.project(points, q, self.truth.theta if true else None)[0]

    def tips(self, q: np.ndarray, true: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The head-side (pinch) and wrist-side fingertips, root frame."""
        theta = self.truth.theta if true else None
        thickness = self.truth.thickness if true else JAW_THICKNESS_M
        pinch = self.model.frame_pose(pinch_frame(ARM), q, theta)
        face = self.model.wrist_face(ARM) * (thickness / JAW_THICKNESS_M)
        return pinch[:3, 3].copy(), pinch[:3, :3] @ face + pinch[:3, 3]

    def in_picture(self, uv: np.ndarray, size: tuple[int, int], margin: float = 10.0) -> bool:
        """Whether every pixel is in a picture of `size`, `margin` from its edges."""
        return bool(
            np.isfinite(uv).all()
            and (uv[:, 0] > margin).all()
            and (uv[:, 0] < size[0] - margin).all()
            and (uv[:, 1] > margin).all()
            and (uv[:, 1] < size[1] - margin).all()
        )

    def arm_hides(self, q: np.ndarray, points: np.ndarray) -> bool:
        """Whether the true arm's links (capsules) cut any line of sight from the head camera to `points`."""
        eye = self.model.camera_pose(q, self.truth.theta)[:3, 3]
        links = (
            "shoulder_link",
            "upper_arm_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "gripper_frame_link",
        )
        joints = [self.model.frame_pose(f"{ARM}_{link}", q, self.truth.theta)[:3, 3] for link in links]
        for a, b in zip(joints[:-1], joints[1:], strict=True):
            for p in points:
                if _segments_distance(eye, p, a, b) < ARM_RADIUS_M:
                    return True
        return False

    # ---- planning views ------------------------------------------------------------------------------------

    def placement(self, k: int, n: int) -> np.ndarray:
        """root_from_board of placement `k` of `n`: on the table ahead of the arm, toward the robot's middle.

        Where hil3, jnt1 and jnt2 had it (35-50 cm ahead of the arm's base, 21-26 cm in), spread side to side and near
        to far.
        """
        side = -0.16 if n == 1 else np.interp(k, [0, n - 1], [-0.24, -0.06])
        ahead = 0.36 + (0.08 * ((k % 2) - 0.5) if n > 2 else 0.0)
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, self.rng.uniform(-0.6, 0.6)]))[0]
        center = self.board.points().mean(axis=0)
        position = np.array([self.base[0] - ahead, self.base[1] + side, self.table_z])
        pose[:3, 3] = position - pose[:3, :3] @ center
        return pose

    def look_down_at(
        self, target: np.ndarray, height: float, tilt_deg: float, yaw_deg: float, pixel=AIM_PX
    ) -> dict[str, float]:
        """The arm's joints putting the (modeled) wrist camera `height` from `target`, `target` at `pixel`.

        The camera is tilted from straight above by `tilt_deg` toward `yaw_deg` (root frame); `pixel` keeps the target
        clear of the fingers.
        """
        require_package("scipy", extra="felix_jr")
        index = [self.model.q_index(f"{ARM}_{j}") for j in SO101_JOINTS]
        lower = np.array([self.model.limits(f"{ARM}_{j}")[0] for j in SO101_JOINTS]) * 0.95
        upper = np.array([self.model.limits(f"{ARM}_{j}")[1] for j in SO101_JOINTS]) * 0.95
        yaw, tilt = np.radians(yaw_deg), np.radians(tilt_deg)
        direction = np.array([np.cos(yaw) * np.sin(tilt), np.sin(yaw) * np.sin(tilt), np.cos(tilt)])
        eye = target + height * direction
        q0 = self.model.joint_vector({ARM: REST})

        def residual(x: np.ndarray) -> np.ndarray:
            q = q0.copy()
            q[index] = x
            pose = self.wrist_pose(q, true=False)
            uv = self.camera.project(((target - pose[:3, 3]) @ pose[:3, :3])[None])[0]
            return np.r_[(pose[:3, 3] - eye) / 0.01, np.nan_to_num((uv - np.asarray(pixel)) / 20.0, nan=50.0)]

        start = np.clip(np.radians([0.0, 0.0, 30.0, 60.0, 0.0]), lower + 1e-6, upper - 1e-6)
        x = least_squares(residual, start, bounds=(lower, upper)).x
        return {f"{j}.pos": float(np.degrees(v)) for j, v in zip(SO101_JOINTS, x, strict=True)}

    def wrist_sees(self, q: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """The true pixels of `points` if the wrist camera sees all of them clear of the fingers, else None."""
        uv = self.wrist_pixels(q, points)
        if not self.in_picture(uv, (self.camera.width, self.camera.height)):
            return None
        tips = self.wrist_pixels(q, np.array(self.tips(q)))
        if np.isfinite(tips).all():
            left, right = tips[:, 0].min() - FINGER_BOX_PX[0], tips[:, 0].max() + FINGER_BOX_PX[0]
            top = tips[:, 1].min() - FINGER_BOX_PX[1]
            if ((uv[:, 0] > left) & (uv[:, 0] < right) & (uv[:, 1] > top)).any():
                return None
        return uv

    def nudge_head(self, arm: dict[str, float], ticks: dict[str, int], points: np.ndarray) -> dict[str, int]:
        """`ticks` nudged (as a person would, by the arrow keys) until the true picture shows all of `points`.

        Nearest its middle: the model's aim is off by the head's error, which is what the calibration finds.
        """
        size = (int(self.model.intrinsics["width"]), int(self.model.intrinsics["height"]))
        best, best_off = ticks, np.inf
        for d_pan in NUDGE_DEG:
            for d_tilt in NUDGE_DEG:
                trial = {
                    "head_neck": int(round(ticks["head_neck"] + d_pan * TICKS_PER_DEG)),
                    "head_eyes": int(round(ticks["head_eyes"] + d_tilt * TICKS_PER_DEG)),
                }
                uv = self.head_pixels(self.true_q(arm, trial), points)
                if self.in_picture(uv, size):
                    off = float(np.linalg.norm(uv.mean(axis=0) - np.array(size) / 2))
                    if off < best_off:
                        best, best_off = trial, off
        return best

    def head_sees(self, q: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """The true pixels of `points` if the head sees all of them, not behind the arm, else None."""
        uv = self.head_pixels(q, points)
        size = (int(self.model.intrinsics["width"]), int(self.model.intrinsics["height"]))
        if not self.in_picture(uv, size) or self.arm_hides(q, points):
            return None
        return uv

    def arm_views(self, root_board: np.ndarray, n: int, tries: int = 400) -> list[dict[str, float]]:
        """`n` arm poses whose wrist camera sees the whole board: around a few looks down at it, the joints varied."""
        points = self.board.points() @ root_board[:3, :3].T + root_board[:3, 3]
        center = points.mean(axis=0)
        seeds = [
            self.look_down_at(center, h, t, y)
            for h, t, y in (
                (0.30, 25.0, 0.0),
                (0.35, 30.0, 35.0),
                (0.30, 30.0, -35.0),
                (0.40, 15.0, 0.0),
                (0.28, 35.0, 20.0),
            )
        ]
        out: list[dict[str, float]] = []
        for i in range(tries):
            if len(out) >= n:
                break
            seed = seeds[i % len(seeds)]
            spread = 0.0 if i < len(seeds) else 10.0
            pose = {k: v + self.rng.normal(0.0, spread) for k, v in seed.items()}
            limits = [self.model.limits(f"{ARM}_{j}") for j in SO101_JOINTS]
            if any(
                not (lo < np.radians(pose[f"{j}.pos"]) < hi)
                for j, (lo, hi) in zip(SO101_JOINTS, limits, strict=True)
            ):
                continue
            if self.wrist_sees(self.true_q(pose, self.home_ticks), points) is not None:
                out.append(pose)
        return out

    def true_q(self, arm: dict[str, float], head_ticks: dict[str, int]) -> np.ndarray:
        """Pinocchio's q of the arm's true joints and the head's ticks."""
        return self.model.joint_vector({ARM: arm}, head_ticks)

    # ---- observing ---------------------------------------------------------------------------------------------

    def read(self, arm: dict[str, float]) -> dict[str, float]:
        """The joints as the servos report them: the truth plus noise."""
        return {k: v + self.rng.normal(0.0, self.joint_noise_deg) for k, v in arm.items()}

    def detection(self, uv: np.ndarray) -> Detection:
        """A whole board found at `uv`, with the detector's noise."""
        return Detection(np.arange(len(uv)), uv + self.rng.normal(0.0, self.board_noise_px, uv.shape))

    def collect(self, scenario: Scenario) -> tuple[list[BoardSighting], list[TipClick]]:
        """A simulated session: its board sightings and fingertip clicks."""
        sightings: list[BoardSighting] = []
        clicks: list[TipClick] = []
        for k in range(scenario.placements):
            name = f"p{k}"
            root_board = self.placement(k, scenario.placements)
            self.boards[name] = root_board
            points = self.board.points() @ root_board[:3, :3].T + root_board[:3, 3]
            center = points.mean(axis=0)
            for arm in self.arm_views(root_board, scenario.arm_views):
                reading = self.read(arm)
                # the head aimed between the board and the gripper (planned on the model)
                q_plan = self.model.joint_vector({ARM: arm}, self.home_ticks)
                ticks, _ = aim_head(
                    self.model, self.home_ticks, 0.5 * (center + self.model.pinch_point(ARM, q_plan))
                )
                ticks = self.nudge_head(
                    arm, ticks, np.vstack([points, self.tips(self.true_q(arm, ticks))[0][None]])
                )
                q = self.true_q(arm, ticks)
                look = Look({ARM: reading}, ticks)
                sightings.append(BoardSighting(ARM, name, look, self.detection(self.wrist_pixels(q, points))))
                head_uv = self.head_sees(q, points)
                if head_uv is not None:
                    sightings.append(BoardSighting(HEAD, name, look, self.detection(head_uv)))
                if scenario.clicks:
                    clicks += self.click_tips(q, look)
            for j in range(scenario.head_only_views):  # the arms folded, the head aimed around the board
                angle = 2 * np.pi * j / max(1, scenario.head_only_views)
                offset = HEAD_ONLY_SPREAD_M * np.array([np.cos(angle), np.sin(angle), 0.0]) * (1 + j % 2) / 2
                ticks, _ = aim_head(self.model, self.home_ticks, center + offset)
                ticks = self.nudge_head(REST, ticks, points)
                q = self.true_q(REST, ticks)
                uv = self.head_sees(q, points)
                if uv is not None:
                    sightings.append(
                        BoardSighting(HEAD, name, Look({ARM: self.read(REST)}, ticks), self.detection(uv))
                    )
        for arm in self.grasp_poses(
            scenario.low_views
        ):  # the gripper low over the table: the fingertips only
            q_plan = self.model.joint_vector({ARM: arm}, self.home_ticks)
            ticks, _ = aim_head(self.model, self.home_ticks, self.model.pinch_point(ARM, q_plan))
            ticks = self.nudge_head(arm, ticks, self.tips(self.true_q(arm, ticks))[0][None])
            clicks += self.click_tips(self.true_q(arm, ticks), Look({ARM: self.read(arm)}, ticks))
        return sightings, clicks

    def click_tips(self, q: np.ndarray, look: Look) -> list[TipClick]:
        """The fingertips clicked where each camera sees them at `q` (the head: the pinch point; the wrist: its face)."""
        out = []
        head_tip, wrist_tip = self.tips(q)
        for camera, tip, project, size in (
            (HEAD, head_tip, self.head_pixels, (640, 480)),
            (ARM, wrist_tip, self.wrist_pixels, (self.camera.width, self.camera.height)),
        ):
            uv = project(q, tip[None])
            if self.in_picture(uv, size, margin=5.0):
                out.append(TipClick(camera, ARM, look, uv[0] + self.rng.normal(0.0, self.click_noise_px, 2)))
        return out

    # ---- scoring -----------------------------------------------------------------------------------------------

    def grasp_poses(self, n: int) -> list[dict[str, float]]:
        """Held-out poses: the pinch point 3-12 cm over the table in front of the arm, the gripper tilted down."""
        out = []
        for _ in range(n):
            target = np.array(
                [
                    self.base[0] - self.rng.uniform(0.15, 0.30),
                    self.base[1] + self.rng.uniform(-0.12, 0.12),
                    self.table_z + self.rng.uniform(0.03, 0.12),
                ]
            )
            out.append(self.look_down_at(target, 0.15, self.rng.uniform(20, 45), self.rng.uniform(-40, 40)))
        return out

    def score(
        self, theta: np.ndarray, gc: np.ndarray, thickness: float, poses: list[dict[str, float]]
    ) -> dict[str, float]:
        """The held-out errors of a fitted (theta, gc, thickness) against the truth (the module docstring)."""
        fitted = World(self.model, Truth(theta, gc, thickness), self.board, self.rng)
        head, wrist, cam_mm, cam_deg, move = [], [], [], [], []
        for arm in poses:
            q_plan = self.model.joint_vector({ARM: arm}, self.home_ticks)
            ticks, _ = aim_head(self.model, self.home_ticks, self.model.pinch_point(ARM, q_plan))
            q = self.true_q(arm, ticks)
            true_tips, fit_tips = self.tips(q), fitted.tips(q)
            head.append(
                np.linalg.norm(
                    self.head_pixels(q, true_tips[0][None]) - fitted.head_pixels(q, fit_tips[0][None])
                )
            )
            wrist.append(
                np.linalg.norm(
                    self.wrist_pixels(q, true_tips[1][None]) - fitted.wrist_pixels(q, fit_tips[1][None])
                )
            )
            rel_true = np.linalg.inv(
                self.model.frame_pose(pinch_frame(ARM), q, self.truth.theta)
            ) @ self.wrist_pose(q, True)
            rel_fit = np.linalg.inv(self.model.frame_pose(pinch_frame(ARM), q, theta)) @ fitted.wrist_pose(
                q, True
            )
            cam_mm.append(1000 * np.linalg.norm(rel_true[:3, 3] - rel_fit[:3, 3]))
            cam_deg.append(np.degrees(np.linalg.norm(cv2.Rodrigues(rel_true[:3, :3] @ rel_fit[:3, :3].T)[0])))
            # a 5 cm move along the approach: the joints the fitted model solves for it, run on the truth
            pinch_fit = self.model.frame_pose(pinch_frame(ARM), q, theta)
            goal = pinch_fit[:3, 3] + 0.05 * pinch_fit[:3, 2]
            moved = self._move(arm, goal, pinch_fit[:3, :3], theta)
            q2 = self.true_q(moved, ticks)
            true_shift = self.model.pinch_point(ARM, q2, self.truth.theta) - self.model.pinch_point(
                ARM, q, self.truth.theta
            )
            move.append(1000 * np.linalg.norm(true_shift - (goal - pinch_fit[:3, 3])))
        return {
            "head px": float(np.sqrt(np.mean(np.square(head)))),
            "wrist px": float(np.sqrt(np.mean(np.square(wrist)))),
            "camera mm": float(np.mean(cam_mm)),
            "camera deg": float(np.mean(cam_deg)),
            "move mm": float(np.sqrt(np.mean(np.square(move)))),
        }

    def _move(
        self, arm: dict[str, float], goal: np.ndarray, rotation: np.ndarray, theta: np.ndarray
    ) -> dict[str, float]:
        """The joints the model with `theta` puts the pinch point at `goal` with, its turn kept (from `arm`)."""
        index = [self.model.q_index(f"{ARM}_{j}") for j in SO101_JOINTS]
        q0 = self.model.joint_vector({ARM: arm})

        def residual(x: np.ndarray) -> np.ndarray:
            q = q0.copy()
            q[index] = x
            pose = self.model.frame_pose(pinch_frame(ARM), q, theta)
            turn = cv2.Rodrigues(pose[:3, :3] @ rotation.T)[0].ravel()
            return np.r_[(pose[:3, 3] - goal) / 0.001, turn / 0.05]

        x = least_squares(residual, q0[index]).x
        return {f"{j}.pos": float(np.degrees(v)) for j, v in zip(SO101_JOINTS, x, strict=True)}


def _segments_distance(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """The shortest distance between segments p1-p2 and q1-q2."""
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (q1 + d2 * t)))


def run(
    model: XleRobotModel, scenario: Scenario, seed: int, board: Board, held_out: int = 20
) -> dict[str, float]:
    """One simulated session and fit; the held-out scores, before (the model as it is) and after."""
    rng = np.random.default_rng(seed)
    world = World(model, draw_truth(model, rng), board, rng)
    sightings, clicks = world.collect(scenario)
    problem = JointProblem(model, board, {ARM: world.camera}, sightings, clicks)
    started = time.time()
    fit = problem.fit()
    world.rng = np.random.default_rng(10_000 + seed)  # held-out poses apart from the session's
    poses = world.grasp_poses(held_out)
    before = world.score(model.estimates, world.gc0, JAW_THICKNESS_M, poses)
    after = world.score(fit.theta, fit.gripper_from_camera[ARM], fit.thickness, poses)
    counts = {
        "wrist views": sum(s.camera == ARM for s in sightings),
        "head views": sum(s.camera == HEAD for s in sightings),
        "clicks": len(clicks),
        "fit s": time.time() - started,
    }
    return {**counts, **{f"{k} before": v for k, v in before.items()}, **after}


SCENARIOS = (
    Scenario(1, 10, 6, True, "1 placement, 10 arm + 6 head views"),
    Scenario(2, 10, 6, True, "2 placements, 10 arm + 6 head views each"),
    Scenario(3, 10, 6, True, "3 placements, 10 arm + 6 head views each"),
    Scenario(2, 20, 6, True, "2 placements, 20 arm + 6 head views each"),
    Scenario(2, 10, 12, True, "2 placements, 10 arm + 12 head views each"),
    Scenario(2, 10, 6, False, "2 placements, 10 + 6 views, no clicks"),
    Scenario(2, 10, 0, True, "2 placements, 10 arm views, no head-only"),
    Scenario(2, 10, 6, True, "2 placements, 10 + 6, + 5 low poses clicked", 5),
    Scenario(2, 10, 6, True, "2 placements, 10 + 6, + 10 low poses clicked", 10),
    Scenario(2, 10, 6, True, "2 placements, 10 + 6, + 20 low poses clicked", 20),
)


def main() -> None:
    """The scenarios' table (the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", default=None, help="the calibrated URDF to start from (default: the harness's)"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="only the first two scenarios, one seed")
    args = parser.parse_args()
    from lerobot.felix_jr.harness import api

    model = XleRobotModel.load(args.model or api.MODEL)
    board = Checkerboard(8, 6, 0.025)
    scenarios = SCENARIOS[:2] if args.quick else SCENARIOS
    seeds = 1 if args.quick else args.seeds
    keys = ("head px", "wrist px", "camera mm", "camera deg", "move mm")
    print(
        f"{'scenario':46s} {'views w/h/clicks':>17s} "
        + " ".join(f"{k:>10s}" for k in keys)
        + "   (before -> after)"
    )
    for scenario in scenarios:
        results = [run(model, scenario, seed, board) for seed in range(seeds)]
        mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        counts = f"{mean['wrist views']:.0f}/{mean['head views']:.0f}/{mean['clicks']:.0f}"
        cells = " ".join(f"{mean[k + ' before']:4.1f}>{mean[k]:4.1f}" for k in keys)
        print(f"{scenario.label:46s} {counts:>17s} {cells}   ({mean['fit s']:.0f} s a fit)", flush=True)


if __name__ == "__main__":
    main()
