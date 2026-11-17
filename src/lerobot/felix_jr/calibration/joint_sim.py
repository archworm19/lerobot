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
import itertools
import time
from dataclasses import dataclass

import cv2
import numpy as np

from lerobot.felix_jr.calibration import joint_views as jv
from lerobot.felix_jr.calibration.boards import Board, CharucoBoard, Detection
from lerobot.felix_jr.calibration.joint_fit import (
    HEAD,
    MOVING,
    MOVING_PINCH,
    BoardSighting,
    JointProblem,
    Look,
    TipClick,
    fit_staged,
    uncertainty,
)
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    XleRobotModel,
    pinch_frame,
)
from lerobot.felix_jr.harness.look import TICKS_PER_DEG, aim_head
from lerobot.utils.import_utils import _scipy_available

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
SAG_REACH_M = 0.25  # `World.sag_deg` is the droop with the gripper this far out
ARM_RADIUS_M = 0.035  # the arm's links as capsules this thick, for what hides the board from the head
JOINT_NOISE_DEG = 0.2
BOARD_NOISE_PX = 0.3
CLICK_NOISE_PX = 2.0
CAMERA_SHIFT_M, CAMERA_TURN_DEG = 0.010, 4.0  # the true wrist camera's pose off the model's, 1 sigma
THICKNESS_OFF_M = 0.0005
HINGE_OFF_M = (
    0.003  # the true hinge off the model's, 1 sigma per axis (after a gripper sweep: 3-6 mm, 2026-11-15)
)
OPENING_OFF_DEG = 1.0  # the true opening off the gripper map's, 1 sigma
AUTO_NOISE_PX = 1.0  # a fingertip found in the wrist picture (`gripper_sweep.fit_frame`: ~1 px over a sweep)
SWEEP_POSITIONS = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
)  # the end sweep's `gripper.pos` (open: the closed frame finds no tips)
NUDGE_DEG = (0.0, -4.0, 4.0, -8.0, 8.0, -12.0, 12.0)  # the head's nudges tried, each servo
PLACEMENT_YAW_RAD = 0.6  # each placement turned on the table by up to this much


@dataclass(frozen=True)
class Scenario:
    """What a session collects, per board placement.

    Wrist views (each with a head view of the board if it can, and clicks), and head views of the board with the arms
    folded. As `joint_collect`: in the arm views the head looks at the board; in the low poses at the fingertip. The
    wrist's fingertip is clicked, and with `head_clicks` the head's where it shows.
    """

    placements: int = 2
    arm_views: int = 10
    head_only_views: int = 3
    clicks: bool = True
    label: str = ""
    low_views: int = (
        0  # poses low over the table, as a grasp's (once, not per placement): clicks, and the board if seen
    )
    head_clicks: bool = True
    # the jaws at each arm view and low pose: these `gripper.pos` in turn (empty: closed, as today's sessions)
    openings: tuple[float, ...] = ()
    # the wrist's tips found in its picture (1 px), the moving one too when open; else clicked
    wrist_auto: bool = False
    head_moving: bool = False  # the head clicks the moving tip too, when open
    # a gripper sweep after the views (one pose, `SWEEP_POSITIONS`): the wrist's tips found
    end_sweep: bool = False
    # with `wrist_auto`: the wrist's tips found at the views too (else only in the end sweep)
    wrist_at_views: bool = True


@dataclass
class Truth:
    """The true robot: its estimates (its wrist camera's pose on the gripper among them), its fingers' thickness."""

    theta: np.ndarray
    thickness: float
    opening_off_deg: float = 0.0  # the real opening less the gripper map's


def _turn(deg_sigma: float, rng: np.random.Generator) -> np.ndarray:
    return cv2.Rodrigues(np.radians(rng.normal(0.0, deg_sigma, 3)))[0]


def draw_truth(model: XleRobotModel, rng: np.random.Generator, scale: float = 1.0) -> Truth:
    """The true robot: the estimates moved by their priors (times `scale`), the wrist camera and the thickness off.

    The hinge (`HINGE_OFF_M`) and the opening (`OPENING_OFF_DEG`) are off by as much whatever `scale`: what a gripper
    sweep leaves (`scale` < 1: a robot near its calibration, the hinge still to find).
    """
    theta = model.estimates.copy()
    for i, p in enumerate(model.parameters):
        if p.arm in (None, ARM) and p.group != "wrist camera":  # (the camera's: below)
            hinge = p.name.startswith(f"pos:{ARM}_gripper:")
            theta[i] += rng.normal(0.0, HINGE_OFF_M if hinge else scale * p.sigma)  # (the hinge: unscaled)
    # the wrist camera turned and moved on its gripper (`CAMERA_*`; drawn after the rest, as before it was a parameter)
    gc = model.gripper_from_camera(ARM)
    true_gc = gc.copy()
    true_gc[:3, :3] = gc[:3, :3] @ _turn(scale * CAMERA_TURN_DEG, rng)
    true_gc[:3, 3] += rng.normal(0.0, scale * CAMERA_SHIFT_M, 3)
    index = {p.name: i for i, p in enumerate(model.parameters)}
    for name, value in model.camera_corrections(ARM, true_gc).items():
        theta[index[name]] = value
    return Truth(
        theta,
        JAW_THICKNESS_M + rng.normal(0.0, scale * THICKNESS_OFF_M),
        # its own stream (spawned: `rng`'s draws, and the sessions drawn after, stay as they were before it)
        float(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]).normal(0.0, OPENING_OFF_DEG)),
    )


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
        base = model.frame_pose(f"{ARM}_base_link", model.joint_vector())
        self.base = base[:3, 3]
        self.table_z = self.base[2] - TABLE_BELOW_BASE_M
        self.home_ticks = model.head_ticks(dict.fromkeys(("head_pan_joint", "head_tilt_joint"), 0.0))
        self.boards: dict[str, np.ndarray] = {}  # root_from_board of each placement `collect` made
        # the arm's droop (deg, shoulder lift and elbow) at full reach, what the model cannot say (`true_q`)
        self.sag_deg = (0.0, 0.0)

    # ---- the cameras, true or modeled ------------------------------------------------------------------------

    def truth_or_model(self, true: bool) -> tuple[np.ndarray | None]:
        """(theta,) of the truth, or of the model (None: its estimates)."""
        return (self.truth.theta,) if true else (None,)

    def wrist_pose(self, q: np.ndarray, true: bool) -> np.ndarray:
        """root_from_camera of the wrist camera at `q`: the truth's, or the model's."""
        return jv.wrist_pose(self.model, ARM, q, *self.truth_or_model(true))

    def wrist_pixels(self, q: np.ndarray, points: np.ndarray, true: bool = True) -> np.ndarray:
        """Root-frame points in the wrist camera's picture at `q`."""
        return jv.wrist_pixels(self.model, ARM, q, points, *self.truth_or_model(true))

    def head_pixels(self, q: np.ndarray, points: np.ndarray, true: bool = True) -> np.ndarray:
        """Root-frame points in the head camera's picture at `q`."""
        return self.model.project(points, q, self.truth.theta if true else None)[0]

    def tips(self, q: np.ndarray, true: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The head-side (pinch) and wrist-side fingertips, root frame."""
        if true:
            return jv.fingertips(self.model, ARM, q, self.truth.theta, self.truth.thickness)
        return jv.fingertips(self.model, ARM, q)

    def moving_tip(self, q: np.ndarray, pos: float, face: str, true: bool = True) -> np.ndarray:
        """The moving finger's tip (root frame) on `face` (`MOVING`: the wrist's side; `MOVING_PINCH`), at `gripper.pos`."""
        theta = self.truth.theta if true else None
        thickness = self.truth.thickness if true else JAW_THICKNESS_M
        opening = self.model.opening_deg(ARM, pos) + (self.truth.opening_off_deg if true else 0.0)
        on_pinch, on_wrist = jv.moving_fingertips(self.model, ARM, q, opening, theta, thickness)
        return on_pinch if face == MOVING_PINCH else on_wrist

    def arm_hides(self, q: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Per point: whether the true arm's links (capsules) cut the line of sight from the head camera to it."""
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
        return np.array(
            [
                any(
                    _segments_distance(eye, p, a, b) < ARM_RADIUS_M
                    for a, b in zip(joints[:-1], joints[1:], strict=True)
                )
                for p in points
            ]
        )

    # ---- planning views ------------------------------------------------------------------------------------

    def placement(self, k: int, n: int) -> np.ndarray:
        """root_from_board of placement `k` of `n` (`joint_views.suggest_placements`), turned on the table at random."""
        yaw = self.rng.uniform(-PLACEMENT_YAW_RAD, PLACEMENT_YAW_RAD)
        return jv.suggest_placements(self.model, ARM, self.board, n, self.table_z, [yaw] * n)[k]

    def wrist_sees(self, q: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """The true pixels of the `points` the wrist camera sees clear of the fingers, NaN the others.

        None if too few for the board to be found (`Board.min_view_fraction`).
        """
        return jv.wrist_sees(
            self.model, ARM, q, points, *self.truth_or_model(True), min_fraction=self.board.min_view_fraction
        )

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
                if jv.in_picture(uv, size):
                    off = float(np.linalg.norm(uv.mean(axis=0) - np.array(size) / 2))
                    if off < best_off:
                        best, best_off = trial, off
        return best

    def head_sees(self, q: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """The true pixels of the `points` the head sees, not behind the arm, NaN the others.

        None if too few for the board to be found (`Board.min_view_fraction`).
        """
        uv = self.head_pixels(q, points)
        size = (int(self.model.intrinsics["width"]), int(self.model.intrinsics["height"]))
        seen = np.array([jv.in_picture(p[None], size) for p in uv]) & ~self.arm_hides(q, points)
        if not seen.any() or seen.mean() < self.board.min_view_fraction:
            return None
        return np.where(seen[:, None], uv, np.nan)

    def arm_views(self, root_board: np.ndarray, n: int) -> list[dict[str, float]]:
        """`n` arm poses planned on the model (`joint_views.arm_views`), kept if the true wrist camera sees the board."""
        points = jv.board_points(self.board, root_board)
        return jv.arm_views(
            self.model,
            ARM,
            REST,
            points,
            n,
            self.rng,
            accept=lambda pose: self.wrist_sees(self.true_q(pose, self.home_ticks), points) is not None,
            min_fraction=self.board.min_view_fraction,
            region=jv.low_poses(self.model, ARM, REST, 40, self.table_z, self.rng),
        )

    def true_q(self, arm: dict[str, float], head_ticks: dict[str, int]) -> np.ndarray:
        """Pinocchio's q of the arm's true joints and the head's ticks.

        With `sag_deg`, the joints the servos hold (`arm`) are not where the arm is: the shoulder lift and the elbow
        droop by `sag_deg` times how far out from each the gripper reaches (over `SAG_REACH_M`), a pose-dependent error
        no parameter of the model can take up.
        """
        q = self.model.joint_vector({ARM: arm}, head_ticks)
        if not any(self.sag_deg):
            return q
        grip = self.model.frame_pose(f"{ARM}_gripper_frame_link", q)[:3, 3]
        droop = {}
        for (joint, link), sag in zip(
            (("shoulder_lift", "upper_arm_link"), ("elbow_flex", "lower_arm_link")), self.sag_deg, strict=True
        ):
            reach = np.linalg.norm((grip - self.model.frame_pose(f"{ARM}_{link}", q)[:3, 3])[:2])
            droop[f"{joint}.pos"] = arm[f"{joint}.pos"] + sag * reach / SAG_REACH_M
        return self.model.joint_vector({ARM: {**arm, **droop}}, head_ticks)

    # ---- observing ---------------------------------------------------------------------------------------------

    def read(self, arm: dict[str, float]) -> dict[str, float]:
        """The joints as the servos report them: the truth plus noise."""
        return {k: v + self.rng.normal(0.0, self.joint_noise_deg) for k, v in arm.items()}

    def detection(self, uv: np.ndarray) -> Detection:
        """The board's points found at `uv` (NaN: not found), with the detector's noise."""
        ids = np.flatnonzero(np.isfinite(uv).all(axis=1))
        return Detection(ids, uv[ids] + self.rng.normal(0.0, self.board_noise_px, (len(ids), 2)))

    def collect(self, scenario: Scenario) -> tuple[list[BoardSighting], list[TipClick]]:
        """A simulated session: its board sightings and fingertip clicks."""
        sightings: list[BoardSighting] = []
        clicks: list[TipClick] = []
        jaws = itertools.cycle(scenario.openings)

        def opened(arm: dict[str, float]) -> dict[str, float]:  # the arm with its jaws at the next opening
            return {**arm, "gripper.pos": next(jaws)} if scenario.openings else arm

        for k in range(scenario.placements):
            name = f"p{k}"
            root_board = self.placement(k, scenario.placements)
            self.boards[name] = root_board
            points = jv.board_points(self.board, root_board)
            center = points.mean(axis=0)
            for arm in map(opened, self.arm_views(root_board, scenario.arm_views)):
                reading = self.read(arm)
                ticks, _ = aim_head(self.model, self.home_ticks, center)  # the head on the board
                ticks = self.nudge_head(arm, ticks, points)
                q = self.true_q(arm, ticks)
                look = Look({ARM: reading}, ticks)
                wrist_uv = self.wrist_sees(q, points)
                if wrist_uv is not None:
                    sightings.append(BoardSighting(ARM, name, look, self.detection(wrist_uv)))
                head_uv = self.head_sees(q, points)
                if head_uv is not None:
                    sightings.append(BoardSighting(HEAD, name, look, self.detection(head_uv)))
                if scenario.clicks:
                    clicks += self.click_tips(q, look, scenario.head_clicks, arm.get("gripper.pos"), scenario)
            for aim in jv.head_aims(center, scenario.head_only_views):  # the arms folded
                ticks, _ = aim_head(self.model, self.home_ticks, aim)
                ticks = self.nudge_head(REST, ticks, points)
                q = self.true_q(REST, ticks)
                uv = self.head_sees(q, points)
                if uv is not None:
                    sightings.append(
                        BoardSighting(HEAD, name, Look({ARM: self.read(REST)}, ticks), self.detection(uv))
                    )
        last = f"p{scenario.placements - 1}"  # the board left where the last placement put it
        points = jv.board_points(self.board, self.boards[last]) if scenario.placements else None
        for arm in map(
            opened, self.grasp_poses(scenario.low_views)
        ):  # the gripper low over the table, the head on the fingertip
            q_plan = self.model.joint_vector({ARM: arm}, self.home_ticks)
            ticks, _ = aim_head(self.model, self.home_ticks, self.model.pinch_point(ARM, q_plan))
            ticks = self.nudge_head(arm, ticks, self.tips(self.true_q(arm, ticks))[0][None])
            q, look = self.true_q(arm, ticks), Look({ARM: self.read(arm)}, ticks)
            if points is not None:
                for camera, uv in ((ARM, self.wrist_sees(q, points)), (HEAD, self.head_sees(q, points))):
                    if uv is not None:
                        sightings.append(BoardSighting(camera, last, look, self.detection(uv)))
            clicks += self.click_tips(q, look, scenario.head_clicks, arm.get("gripper.pos"), scenario)
        if scenario.end_sweep:  # one pose low over the table, the jaws opened in steps, the head not used
            for arm in self.grasp_poses(10)[:1]:  # (asked for one, the planner can find none in its tries)
                for pos in SWEEP_POSITIONS:
                    pose = {**arm, "gripper.pos": pos}
                    q, look = (
                        self.true_q(pose, self.home_ticks),
                        Look({ARM: self.read(pose)}, self.home_ticks),
                    )
                    sweep = Scenario(wrist_auto=True)
                    clicks += self.click_tips(q, look, False, pos, sweep)
        return sightings, clicks

    def click_tips(
        self,
        q: np.ndarray,
        look: Look,
        head: bool = False,
        pos: float | None = None,
        scenario: "Scenario | None" = None,
    ) -> list[TipClick]:
        """The fingertips clicked (or found) where the cameras see them at `q`.

        The wrist's face, and with `head` the pinch point. With the jaws open (`pos`, a `gripper.pos`): the moving
        tip too, in the wrist's picture if `scenario.wrist_auto` (found, `AUTO_NOISE_PX`; the fixed one found too) and
        in the head's if `scenario.head_moving`.
        """
        auto = scenario is not None and scenario.wrist_auto
        open_ = pos is not None and pos > 1.0
        head_tip, wrist_tip = self.tips(q)
        wrist_size, head_size = (self.camera.width, self.camera.height), (640, 480)
        noise = AUTO_NOISE_PX if auto else self.click_noise_px
        # (camera, root point, its face, noise)
        # found only with the jaws open, and at the views only if `wrist_at_views`
        wrist = not auto or (open_ and scenario is not None and scenario.wrist_at_views)
        seen = [(ARM, wrist_tip, "wrist", noise)] if wrist else []
        if auto and wrist and pos is not None:
            seen.append((ARM, self.moving_tip(q, pos, MOVING), MOVING, AUTO_NOISE_PX))
        if head and jv.head_face_angle(self.model, ARM, q) < 90.0:  # else the jaw hides its tip from the head
            seen.append((HEAD, head_tip, "pinch", self.click_noise_px))
            if open_ and pos is not None and scenario is not None and scenario.head_moving:
                seen.append((HEAD, self.moving_tip(q, pos, MOVING_PINCH), MOVING_PINCH, self.click_noise_px))
        out = []
        for camera, tip, face, sigma in seen:
            project, size = (
                (self.head_pixels, head_size) if camera == HEAD else (self.wrist_pixels, wrist_size)
            )
            uv = project(q, tip[None])
            if jv.in_picture(uv, size, margin=5.0):
                out.append(TipClick(camera, ARM, look, uv[0] + self.rng.normal(0.0, sigma, 2), "", face))
        return out

    # ---- scoring -----------------------------------------------------------------------------------------------

    def grasp_poses(self, n: int) -> list[dict[str, float]]:
        """Poses low over the table, as a grasp's (`joint_views.low_poses`): the low views, and the held-out ones."""
        return jv.low_poses(self.model, ARM, REST, n, self.table_z, self.rng)

    def score(self, theta: np.ndarray, thickness: float, poses: list[dict[str, float]]) -> dict[str, float]:
        """The held-out errors of a fitted (theta, thickness) against the truth (the module docstring)."""
        fitted = World(self.model, Truth(theta, thickness), self.board, self.rng)
        head, wrist, cam_mm, cam_deg, move = [], [], [], [], []
        for arm in poses:
            q_plan = self.model.joint_vector({ARM: arm}, self.home_ticks)
            ticks, _ = aim_head(self.model, self.home_ticks, self.model.pinch_point(ARM, q_plan))
            q = self.true_q(arm, ticks)  # where the arm is
            q_held = self.model.joint_vector({ARM: arm}, ticks)  # what the fit knows: the joints as held
            true_tips, fit_tips = self.tips(q), fitted.tips(q_held)
            head.append(
                np.linalg.norm(
                    self.head_pixels(q, true_tips[0][None]) - fitted.head_pixels(q_held, fit_tips[0][None])
                )
            )
            wrist.append(
                np.linalg.norm(
                    self.wrist_pixels(q, true_tips[1][None]) - fitted.wrist_pixels(q_held, fit_tips[1][None])
                )
            )
            rel_true = np.linalg.inv(
                self.model.frame_pose(pinch_frame(ARM), q, self.truth.theta)
            ) @ self.wrist_pose(q, True)
            rel_fit = np.linalg.inv(
                self.model.frame_pose(pinch_frame(ARM), q_held, theta)
            ) @ fitted.wrist_pose(q_held, True)
            cam_mm.append(1000 * np.linalg.norm(rel_true[:3, 3] - rel_fit[:3, 3]))
            cam_deg.append(np.degrees(np.linalg.norm(cv2.Rodrigues(rel_true[:3, :3] @ rel_fit[:3, :3].T)[0])))
            # a 5 cm move along the approach: the joints the fitted model solves for it, run on the truth
            pinch_fit = self.model.frame_pose(pinch_frame(ARM), q_held, theta)
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
    model: XleRobotModel,
    scenario: Scenario,
    seed: int,
    board: Board,
    held_out: int = 20,
    verbose: bool = False,
    truth_scale: float = 1.0,
) -> dict[str, float]:
    """One simulated session and fit; the held-out scores, before (the model as it is) and after."""
    model = model.with_gripper_hinge(ARM)  # the hinge off in every truth; fitted where the moving tip is seen
    rng = np.random.default_rng(seed)
    world = World(model, draw_truth(model, rng, truth_scale), board, rng)
    sightings, clicks = world.collect(scenario)
    problem = JointProblem(model, board, {ARM: world.camera}, sightings, clicks)
    started = time.time()
    fit = fit_staged(problem)
    hinge = [  # its position across its axis (along it moves nothing: not fitted, `JointProblem`)
        i
        for i, p in enumerate(model.parameters)
        if p.name.startswith(f"pos:{ARM}_gripper:") and problem.free[i]
    ]
    u = uncertainty(problem, fit)
    hinge_names = {model.parameters[i].name for i in hinge}
    hinge_camera = [
        abs(c) for a, b, c in u["pairs"] if (a in hinge_names) != (b in hinge_names) and "camera" in a + b
    ]
    fitted = {
        "hinge mm": float(1000 * np.linalg.norm(fit.theta[hinge] - world.truth.theta[hinge])),
        "hinge mm before": float(1000 * np.linalg.norm(model.estimates[hinge] - world.truth.theta[hinge])),
        "opening deg": abs(fit.openings.get(ARM, 0.0) - world.truth.opening_off_deg),
        "opening deg before": abs(world.truth.opening_off_deg),
        "hinge sd mm": float(np.linalg.norm([u["sd"][n]["sd"] for n in sorted(hinge_names) if n in u["sd"]])),
        "hinge/camera corr": max(hinge_camera, default=0.0),
        "weak sd": u["weak"][0]["sd"] if u["weak"] else 0.0,
    }
    if verbose:
        print(scenario.label, "pairs:", [(x, y, round(c, 2)) for x, y, c in u["pairs"][:6]])
        print("  hinge sd:", {n: round(u["sd"][n]["sd"], 2) for n in sorted(hinge_names) if n in u["sd"]})
        for w in u["weak"]:
            print("  weak", round(w["sd"], 2), [(n, round(v, 2)) for n, v in w["parts"]])
    world.rng = np.random.default_rng(10_000 + seed)  # held-out poses apart from the session's
    poses = world.grasp_poses(held_out)
    before = world.score(model.estimates, JAW_THICKNESS_M, poses)
    after = world.score(fit.theta, fit.thickness, poses)
    counts = {
        "wrist views": sum(s.camera == ARM for s in sightings),
        "head views": sum(s.camera == HEAD for s in sightings),
        "clicks": len(clicks),
        "moving clicks": sum(c.on_face in (MOVING, MOVING_PINCH) for c in clicks),
        "fit s": time.time() - started,
    }
    return {**counts, **{f"{k} before": v for k, v in before.items()}, **after, **fitted}


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


# the merged procedure (`docs/camera_calibration.md`, "The fingers in the joint session"): 2 placements, 10 arm + 6
# head views each, 10 low poses
OPEN = jv.VIEW_OPENINGS  # `gripper.pos` the jaws take in turn at the views (the session's)
FINGER_SCENARIOS = (
    Scenario(
        2, 10, 6, True, "today: closed, wrist tips from an end sweep", 10, wrist_auto=True, end_sweep=True
    ),
    Scenario(
        2,
        10,
        6,
        True,
        "+ open at every view, the wrist finds both",
        10,
        openings=OPEN,
        wrist_auto=True,
        end_sweep=True,
    ),
    Scenario(
        2,
        10,
        6,
        True,
        "+ the head clicks the moving tip",
        10,
        openings=OPEN,
        wrist_auto=True,
        head_moving=True,
        end_sweep=True,
    ),
    Scenario(
        2,
        10,
        6,
        True,
        "  the wrist's from the end sweep only",
        10,
        openings=OPEN,
        wrist_auto=True,
        head_moving=True,
        end_sweep=True,
        wrist_at_views=False,
    ),
    Scenario(
        2,
        10,
        6,
        True,
        "  the end sweep only, 20 low poses",
        20,
        openings=OPEN,
        wrist_auto=True,
        head_moving=True,
        end_sweep=True,
        wrist_at_views=False,
    ),
    Scenario(
        2,
        10,
        6,
        True,
        "  the wrist's found at the views, no sweep",
        10,
        openings=OPEN,
        wrist_auto=True,
        head_moving=True,
    ),
    Scenario(
        2, 10, 6, True, "  the same, 20 low poses", 20, openings=OPEN, wrist_auto=True, head_moving=True
    ),
)


def main() -> None:
    """The scenarios' table (the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", default=None, help="the calibrated URDF to start from (default: the harness's)"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="only the first two scenarios, one seed")
    parser.add_argument(
        "--truth-scale",
        type=float,
        default=1.0,
        help="the true robot off its model by its priors times this (0.25: near a calibration)",
    )
    parser.add_argument(
        "--fingers", action="store_true", help="the moving finger's scenarios (`FINGER_SCENARIOS`) instead"
    )
    args = parser.parse_args()
    from lerobot.felix_jr.harness import api

    model = XleRobotModel.load(args.model or api.MODEL)
    board = CharucoBoard(9, 7, 0.025, 0.018, "DICT_4X4_50")  # the one printed (2026-11-11)
    scenarios: tuple[Scenario, ...] = FINGER_SCENARIOS if args.fingers else SCENARIOS
    scenarios = scenarios[:2] if args.quick else scenarios
    seeds = 1 if args.quick else args.seeds
    keys: tuple[str, ...] = ("head px", "wrist px", "camera mm", "camera deg", "move mm")
    if args.fingers:
        keys += ("hinge mm", "opening deg")
    # the fit's own say (`joint_fit.uncertainty`): the hinge's sd, its worst correlation with the camera's pose, the
    # weakest direction's sd (in its unknowns' priors)
    said = ("hinge sd mm", "hinge/camera corr", "weak sd") if args.fingers else ()
    print(
        f"{'scenario':46s} {'views w/h/clicks/moving':>23s} "
        + " ".join(f"{k:>11s}" for k in keys + said)
        + "   (before -> after)"
    )
    for scenario in scenarios:
        results = [run(model, scenario, seed, board, truth_scale=args.truth_scale) for seed in range(seeds)]
        mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        counts = f"{mean['wrist views']:.0f}/{mean['head views']:.0f}/{mean['clicks']:.0f}/{mean['moving clicks']:.0f}"
        cells = " ".join(f"{mean[k + ' before']:5.1f}>{mean[k]:5.1f}" for k in keys)
        cells += "".join(f" {mean[k]:11.2f}" for k in said)
        print(f"{scenario.label:46s} {counts:>23s} {cells}   ({mean['fit s']:.0f} s a fit)", flush=True)


if __name__ == "__main__":
    main()
