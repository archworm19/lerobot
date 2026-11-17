"""One fit of the head, an arm and its wrist camera: boards seen by both cameras, fingertips clicked in both.

`docs/camera_calibration.md`, "The rework" (2d). The unknowns: the model's estimates (`XleRobotModel.parameters`:
the head and camera, and each fitted arm's mount, joint zeros and origins, tip offset and wrist camera's pose on its
gripper, the corrections to its design: `merge_urdf.wrist_camera_joint`; with their priors), each board placement's
pose in the root frame, and the fingers' thickness (with a tight prior). The residuals, all in pixels over their noise:

- `BoardSighting`: every point of a board a camera found (`boards.Detection`), against the board's point projected
  through that camera's whole chain: the head's (its joints, the camera mount, the RealSense's lens) or the wrist's
  (the arm's mount and joints, the camera's pose on the gripper, its lens).
- `TipClick`: a fingertip clicked by a person. The head sees one face of the fingers, the wrist camera the other, one
  thickness apart along the hinge (`XleRobotModel.wrist_face`): a head click is the pinch point as calibrated, a
  wrist click that point moved by the thickness.
- The priors, as `xle_fit`'s, and on each moment's joint errors (`joint_sigma_deg`): a reading off by a fraction
  of a degree moves every point seen at that moment the same way, which would otherwise bend the parameters.

The lenses are fixed (the wrist's from `calibration.wrist_camera`, the head's the RealSense's). Huber-weighted least
squares from the model's current estimates (`scipy.optimize.least_squares`, finite differences, grouped by
what each unknown touches: a moment's joint errors only what was seen at it).

**Sessions** (`JointSession`, `joint_collect` writes them): a folder with `joint_session.json` (the board, the head's
servo map and lens, the arm's LeRobot calibration, the moments, the sightings and clicks) and their pictures.
`calibrate` fits sessions, holding out some moments both cameras saw the board at to check the result on
(`agreement`), and writes the calibration.

Usage: uv run python -m lerobot.felix_jr.calibration.joint_fit SESSION [SESSION ...] --start-from calibration.urdf
       --out calibration.urdf [--holdout-every 5] [--head-axes]
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.boards import Board, Detection, board_from_dict, board_pose
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS, wrist_camera_frame
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    WristCamera,
    XleRobotModel,
    arm_joint,
    arm_record,
    pinch_frame,
    wrist_camera_link,
)
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

HEAD = "head"
SIGMA_BOARD_PX = 0.5  # a board point's noise, either camera
SIGMA_CLICK_PX = 2.5  # a person's click on a fingertip
THICKNESS_SIGMA_M = 0.001  # the fingers' thickness, measured (1.0 cm)
MOVING = "moving"  # a click's face: the moving finger's tip (its wrist face), the jaws open (`TipClick`)
MOVING_PINCH = "moving_pinch"  # the moving finger's tip on the pinch face (the head's side), the jaws open
MOVING_FACES = (MOVING, MOVING_PINCH)
ALONG_AXIS = (
    0.99  # a hinge parameter this near the hinge's own axis (its cosine) is not fitted (`JointProblem`)
)
CORRELATED = 0.9  # `uncertainty`: two unknowns this correlated after the fit are flagged
OPENING_SIGMA_DEG = 3.0  # the gripper map's opening, off by this much at most (`gripper_sweep`: within 2 deg)
# ... a default map's (`GripperMap.from_servo`, no calipers): ours were 0.5 and 2.1 deg off, a new arm may be more
DEFAULT_OPENING_SIGMA_DEG = 6.0
JOINT_SIGMA_DEG = 0.2  # a joint's reading against where it is (a servo tick is 0.09 deg; play)
HUBER = 2.0  # in sigmas: beyond, a residual is down-weighted
# a board's start (`JointProblem._first_board_pose`): its sightings' poses this near the most central one are averaged
BOARD_START_AGREE_M, BOARD_START_AGREE_DEG = 0.03, 5.0
# the fit's steps at most, a safety net (`fit_staged` starts it near the answer): from a far start it can creep on for
# an hour (joint_sim, 2026-11-11), and 400 cut off one that needed 461
MAX_NFEV = 1500
BEHIND_CAMERA_PX = 1000.0  # the residual of a point that projects behind the camera
FD_STEP = 1e-7  # the Jacobian's relative step


@dataclass(frozen=True)
class Look:
    """The robot at one moment: the arms' joints (LeRobot degrees) and the head's servo ticks."""

    arms: Mapping[str, Mapping[str, float]]
    head_ticks: Mapping[str, int]


@dataclass(frozen=True)
class BoardSighting:
    """The points of the board at `placement` that `camera` ("head", or an arm's wrist camera) found at `look`."""

    camera: str
    placement: str
    look: Look
    detection: Detection
    image: str = ""  # the picture, in its session's folder


@dataclass(frozen=True)
class TipClick:
    """`arm`'s fixed fingertip clicked at `pixel` in `camera`'s picture, on one of its two faces.

    `face`: "pinch" (the pinch point, the face away from the wrist camera) or "wrist" (the face toward it, one thickness
    along the hinge); or `MOVING`, the moving finger's tip on the wrist's side, the jaws open by the look's
    `gripper.pos` (a gripper sweep's open frames: `auto_tips.sweep_clicks`), or `MOVING_PINCH`, its tip on the pinch
    face (the head's clicks, the jaws open): its arc about the hinge sets the hinge and
    the wrist camera's turn about the roll axis, which the board alone traded for the wrist roll's zero
    (joint_left_full5: 2.7 deg one way, 1.9 the other). Empty (sessions before 2026-11-11): by the camera, the head's clicks on "pinch", the wrist
    camera's on "wrist" (it cannot see the other).
    """

    camera: str
    arm: str
    look: Look
    pixel: np.ndarray
    image: str = ""  # the picture, in its session's folder
    face: str = ""

    @property
    def on_face(self) -> str:
        """The face clicked: `face`, or the camera's default."""
        return self.face or ("pinch" if self.camera == HEAD else "wrist")


@dataclass
class JointFit:
    """The fitted estimates and what else was fitted, with each kind of residual's RMS (pixels)."""

    theta: np.ndarray
    gripper_from_camera: dict[str, np.ndarray]
    boards: dict[str, np.ndarray]  # root_from_board per placement
    thickness: float
    rms_px: dict[str, float] = field(default_factory=dict)
    nfev: int = 0
    message: str = ""
    cost: float = 0.0  # the solver's (Huber) cost at the end: comparable between fits of one problem
    openings: dict[str, float] = field(
        default_factory=dict
    )  # per arm with moving clicks: its map's opening off (deg)
    x: np.ndarray | None = None  # the solution vector (`JointProblem`'s layout), for `uncertainty`
    median_px: dict[str, float] = field(default_factory=dict)  # each kind's median error (px): `fit_failed`


def _pose(x: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64))[0]
    pose[:3, 3] = x[3:6]
    return pose


def _vec(pose: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]]


class JointProblem:
    """The observations and the unknowns; `fit` solves it (the module docstring)."""

    def __init__(
        self,
        model: XleRobotModel,
        board: Board,
        cameras: Mapping[str, WristCamera],
        sightings: Sequence[BoardSighting],
        clicks: Sequence[TipClick] = (),
        joint_sigma_deg: float = JOINT_SIGMA_DEG,
        arms_free: bool = True,
    ):
        """`cameras`: each fitted arm's wrist camera (its lens; its pose on the gripper is among the parameters).

        `joint_sigma_deg`: how far a joint's reading may be from where it really is (0: the readings are taken as
        exact). Each moment's fitted-arm joints get an error of their own, with this prior: the same for every point
        seen at that moment, it would otherwise bend the robot's parameters. `arms_free` False holds the arms'
        parameters where they start (the head's, the cameras' poses, the boards still fitted: `fit_staged`).
        """
        self.model, self.board, self.cameras = model, board, dict(cameras)
        self.sightings, self.clicks = list(sightings), list(clicks)
        self.arms = sorted(self.cameras)
        self.moving_arms = sorted({c.arm for c in self.clicks if c.on_face in MOVING_FACES})
        self.placements = sorted({s.placement for s in self.sightings})
        # the wrist cameras' poses stay free with the arms held (`fit_staged`): they are the cameras', not the arms'.
        # The fingers' mounts never are: no point the fit sees depends on them (`gripper_sweep` fits them)
        self.free = np.array(
            [
                p.group != "finger"
                and (p.arm is None or (p.arm in self.arms and (arms_free or p.group == "wrist camera")))
                for p in model.parameters
            ],
            dtype=bool,
        )
        # a gripper hinge's position along its own axis moves nothing: held where it is (else only its prior holds it)
        for arm in {p.arm for p in model.parameters if p.arm and p.name.startswith(f"pos:{p.arm}_gripper:")}:
            axis = model.hinge_axis_in_gripper(arm)
            for i, p in enumerate(model.parameters):
                if (
                    p.name.startswith(f"pos:{arm}_gripper:")
                    and abs(axis["xyz".index(p.name[-1])]) > ALONG_AXIS
                ):
                    self.free[i] = False
        # each moving arm's opening's prior: a default map's wider (`DEFAULT_OPENING_SIGMA_DEG`)
        self.opening_sigma = np.array(
            [
                DEFAULT_OPENING_SIGMA_DEG
                if arm in model.gripper_maps and model.gripper_maps[arm].default
                else OPENING_SIGMA_DEG
                for arm in self.moving_arms
            ]
        )
        self.mean = np.array([p.mean for p in model.parameters])
        self.sigma = np.array([p.sigma for p in model.parameters])
        self.joint_sigma = np.radians(joint_sigma_deg)
        observations: list[BoardSighting | TipClick] = [*self.sightings, *self.clicks]
        self._q = {id(o): model.joint_vector(o.look.arms, o.look.head_ticks) for o in observations}
        # the moments whose fitted arms' joints get an error: one per Look, shared by everything seen at it
        self.looks: dict[int, int] = {}
        if self.joint_sigma > 0:
            for o in observations:
                if any(a in o.look.arms for a in self.arms):
                    self.looks.setdefault(id(o.look), len(self.looks))
        self.joint_index = [model.q_index(arm_joint(a, j)) for a in self.arms for j in SO101_JOINTS]
        # the vector's layout
        n_theta, n_board = int(self.free.sum()), 6 * len(self.placements)
        self.slices = {
            "theta": slice(0, n_theta),
            "boards": slice(n_theta, n_theta + n_board),
            "thickness": slice(n_theta + n_board, n_theta + n_board + 1),
        }
        start = self.slices["thickness"].stop
        self.slices["openings"] = slice(start, start + len(self.moving_arms))
        start = self.slices["openings"].stop
        self.slices["joints"] = slice(start, start + len(self.looks) * len(self.joint_index))
        self.size = self.slices["joints"].stop

    # ---- the unknowns as one vector -----------------------------------------------------------------------------

    def start(
        self, theta0: np.ndarray | None = None, boards0: Mapping[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """The unknowns at the start: the model's estimates (or `theta0`), the joints as read.

        Each board starts where its first sighting puts it, or at `boards0`'s pose for it.
        """
        theta = self.model.estimates if theta0 is None else np.asarray(theta0, dtype=np.float64)
        given_boards = boards0 or {}
        boards = [
            _vec(given_boards[p] if p in given_boards else self._first_board_pose(p, theta))
            for p in self.placements
        ]
        joints = np.zeros(self.slices["joints"].stop - self.slices["joints"].start)
        openings = np.zeros(len(self.moving_arms))
        return np.concatenate([theta[self.free], *boards, [JAW_THICKNESS_M], openings, joints])

    def openings(self, x: np.ndarray) -> dict[str, float]:
        """Each arm with moving clicks: how far its gripper map's opening is off (deg; the real opening less the map's)."""
        return {a: float(v) for a, v in zip(self.moving_arms, x[self.slices["openings"]], strict=True)}

    def unpack(
        self, x: np.ndarray, theta0: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
        """(theta, root_from_board per placement, thickness) from the vector."""
        theta = (self.model.estimates if theta0 is None else np.asarray(theta0, dtype=np.float64)).copy()
        theta[self.free] = x[self.slices["theta"]]
        boards = x[self.slices["boards"]]
        poses = {p: _pose(boards[6 * i : 6 * i + 6]) for i, p in enumerate(self.placements)}
        return theta, poses, float(x[self.slices["thickness"]][0])

    def with_cameras(self, theta: np.ndarray, gcs: Mapping[str, np.ndarray]) -> np.ndarray:
        """`theta` with the wrist cameras' parameters putting each of `gcs`' cameras there (gripper_from_camera)."""
        out = np.asarray(theta, dtype=np.float64).copy()
        index = {p.name: i for i, p in enumerate(self.model.parameters)}
        for arm, gc in gcs.items():
            for name, value in self.model.camera_corrections(arm, gc).items():
                out[index[name]] = value
        return out

    def q(self, observation: BoardSighting | TipClick, x: np.ndarray) -> np.ndarray:
        """The joints at an observation's moment: the readings plus that moment's fitted errors."""
        q = self._q[id(observation)]
        k = self.looks.get(id(observation.look))
        if k is None:
            return q
        n = len(self.joint_index)
        errors = x[self.slices["joints"]][k * n : (k + 1) * n]
        q = q.copy()
        q[self.joint_index] += errors
        return q

    def _sighting_board_pose(self, sighting: BoardSighting, theta: np.ndarray) -> np.ndarray:
        """root_from_board from one sighting (its PnP) through the model at `theta` and the joints as read."""
        q = self._q[id(sighting)]
        if sighting.camera == HEAD:
            intr = self.model.intrinsics
            matrix = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
            return self.model.camera_pose(q, theta) @ board_pose(self.board, sighting.detection, matrix)
        camera = self.cameras[sighting.camera]
        camera_board = board_pose(
            self.board, sighting.detection, camera.matrix(), camera.coeffs, camera.model == "fisheye"
        )
        return self.model.frame_pose(wrist_camera_frame(sighting.camera), q, theta) @ camera_board

    def _first_board_pose(self, placement: str, theta: np.ndarray) -> np.ndarray:
        """The board's start: from every head sighting of it (or every sighting, if the head has none), robustly.

        Each sighting's PnP pose, through the model at `theta`; the one nearest the others in turn (the medoid) the
        reference; those within `BOARD_START_AGREE_M` and `BOARD_START_AGREE_DEG` of it averaged (`_mean_pose`). A flat
        board's PnP can flip to its mirror solution (joint_left_full3: one head picture of 17 put it 1 m and 180 deg
        off); a single first sighting is also only as good as its one view.
        """
        mine = [s for s in self.sightings if s.placement == placement]
        mine = [s for s in mine if s.camera == HEAD] or mine
        poses = [self._sighting_board_pose(s, theta) for s in mine]
        return _medoid_mean(poses, self.board.points().mean(axis=0))

    def wrist_cameras_from_sightings(
        self, theta: np.ndarray, boards: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Each arm's wrist camera on its gripper, from its own board sightings against `boards` (the arm at `theta`).

        Per sighting with `MIN_POSE_POINTS` or more: the camera's PnP pose of the board, taken back through the board's
        pose and the arm's chain to the gripper; robustly averaged (`_medoid_mean`). An arm without one keeps its
        camera's current pose. A start for when the camera's current pose is far off (`fit_staged`).
        """
        out = {}
        for arm in self.arms:
            camera = self.cameras[arm]
            poses = []
            for s in self.sightings:
                if s.camera != arm or s.placement not in boards or len(s.detection) < MIN_POSE_POINTS:
                    continue
                camera_board = board_pose(
                    self.board, s.detection, camera.matrix(), camera.coeffs, camera.model == "fisheye"
                )
                root_link = self.model.frame_pose(wrist_camera_link(arm), self._q[id(s)], theta)
                poses.append(np.linalg.inv(root_link) @ boards[s.placement] @ np.linalg.inv(camera_board))
            current = self.model.gripper_from_camera(arm, theta)
            out[arm] = _medoid_mean(poses, np.zeros(3)) if poses else current
        return out

    # ---- predictions and residuals ------------------------------------------------------------------------------

    def project(self, camera: str, q: np.ndarray, points_root: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) of root-frame points in `camera`'s picture at `q` (NaN behind it)."""
        if camera == HEAD:
            return self.model.project(points_root, q, theta)[0]
        root_camera = self.model.frame_pose(wrist_camera_frame(camera), q, theta)
        in_camera = (points_root - root_camera[:3, 3]) @ root_camera[:3, :3]
        return self.cameras[camera].project(in_camera)

    def tip(
        self,
        click: TipClick,
        q: np.ndarray,
        theta: np.ndarray,
        thickness: float,
        openings: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """The clicked fingertip in the root frame: the pinch point, the other face's, or the moving finger's.

        (`TipClick.on_face`.) The moving finger's: the fixed one's wrist face turned about the hinge by the jaws'
        opening, the gripper map's at the look's `gripper.pos` plus `openings`' offset for the arm (deg).
        """
        pinch = self.model.frame_pose(pinch_frame(click.arm), q, theta)
        if click.on_face == "pinch":
            return pinch[:3, 3]
        face = self.model.wrist_face(click.arm) * (thickness / JAW_THICKNESS_M)
        if click.on_face == MOVING_PINCH:
            face = np.zeros(3)
        if click.on_face in MOVING_FACES:
            opening = self.model.opening_deg(click.arm, float(click.look.arms[click.arm]["gripper.pos"]))
            opening += (openings or {}).get(click.arm, 0.0)
            turned = self.model.moving_jaw(click.arm, opening, theta)  # (pinch frame)
            face = turned[:3, :3] @ face + turned[:3, 3]
        return pinch[:3, :3] @ face + pinch[:3, 3]

    def residual_parts(self, x: np.ndarray, theta0: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Each kind's residuals over its noise: "head board", "<arm> board", "head click", "<arm> click"."""
        theta, boards, thickness = self.unpack(x, theta0)
        openings = self.openings(x)
        parts: dict[str, list[np.ndarray]] = {}
        for s in self.sightings:
            pose = boards[s.placement]
            points = self.board.points(s.detection.ids) @ pose[:3, :3].T + pose[:3, 3]
            uv = self.project(s.camera, self.q(s, x), points, theta)
            parts.setdefault(f"{s.camera} board", []).append(
                ((uv - s.detection.pixels) / SIGMA_BOARD_PX).ravel()
            )
        for c in self.clicks:
            q = self.q(c, x)
            uv = self.project(c.camera, q, self.tip(c, q, theta, thickness, openings)[None], theta)
            parts.setdefault(f"{c.camera} click", []).append(((uv[0] - c.pixel) / SIGMA_CLICK_PX).ravel())
        return {k: np.nan_to_num(np.concatenate(v), nan=BEHIND_CAMERA_PX) for k, v in parts.items()}

    def priors(self, x: np.ndarray) -> np.ndarray:
        """The estimates', the thickness' and the joints' errors' prior residuals."""
        theta = (x[self.slices["theta"]] - self.mean[self.free]) / self.sigma[self.free]
        thickness = (x[self.slices["thickness"]] - JAW_THICKNESS_M) / THICKNESS_SIGMA_M
        openings = x[self.slices["openings"]] / self.opening_sigma
        joints = x[self.slices["joints"]] / self.joint_sigma if self.looks else np.zeros(0)
        return np.concatenate([theta, thickness, openings, joints])

    def residuals(self, x: np.ndarray, theta0: np.ndarray | None = None) -> np.ndarray:
        """Every residual (in `residual_parts`' order), then the priors."""
        return np.concatenate([*self.residual_parts(x, theta0).values(), self.priors(x)])

    def sparsity(self) -> np.ndarray:
        """Which unknowns each residual depends on (rows as `residuals`).

        A moment's joint errors move only what was seen at it, a board's pose only its sightings.
        """
        rows: dict[str, list[np.ndarray]] = {}
        n = len(self.joint_index)
        dense = np.zeros(self.size, dtype=bool)
        dense[: self.slices["boards"].start] = True  # the estimates and the cameras' poses: everything
        dense[self.slices["thickness"]] = True
        dense[self.slices["openings"]] = True

        def row(observation: BoardSighting | TipClick, board: str | None) -> np.ndarray:
            r = dense.copy()
            if board is not None:
                b = self.slices["boards"].start + 6 * self.placements.index(board)
                r[b : b + 6] = True
            k = self.looks.get(id(observation.look))
            if k is not None:
                j = self.slices["joints"].start + k * n
                r[j : j + n] = True
            return r

        for s in self.sightings:
            rows.setdefault(f"{s.camera} board", []).append(
                np.repeat(row(s, s.placement)[None], 2 * len(s.detection), 0)
            )
        for c in self.clicks:
            rows.setdefault(f"{c.camera} click", []).append(np.repeat(row(c, None)[None], 2, 0))
        priors = np.eye(self.size, dtype=bool)[
            np.r_[
                np.arange(self.slices["theta"].stop),
                self.slices["thickness"].start,
                np.arange(self.slices["openings"].start, self.slices["openings"].stop),
                np.arange(self.slices["joints"].start, self.size) if self.looks else [],
            ].astype(int)
        ]
        return np.vstack([np.vstack(v) for v in rows.values()] + [priors])

    def jacobian(self, x: np.ndarray, theta0: np.ndarray | None = None) -> np.ndarray:
        """Forward differences, the columns that touch disjoint residuals (`sparsity`) stepped together.

        ~60 evaluations however many moments; dense, for the exact trust-region solver.
        """
        if not hasattr(self, "_groups"):
            sparsity = self.sparsity()
            self._sparsity = sparsity
            groups: list[tuple[list[int], np.ndarray]] = []
            for column in range(self.size):
                rows = sparsity[:, column]
                for members, used in groups:
                    if not (used & rows).any():
                        members.append(column)
                        used |= rows
                        break
                else:
                    groups.append(([column], rows.copy()))
            self._groups = [members for members, _ in groups]
        base = self.residuals(x, theta0)
        out = np.zeros((len(base), self.size))
        steps = FD_STEP * np.maximum(np.abs(x), 1.0)
        for members in self._groups:
            stepped = x.copy()
            stepped[members] += steps[members]
            change = self.residuals(stepped, theta0) - base
            for column in members:
                rows = self._sparsity[:, column]
                out[rows, column] = change[rows] / steps[column]
        return out

    def fit(
        self,
        theta0: np.ndarray | None = None,
        max_nfev: int = MAX_NFEV,
        boards0: Mapping[str, np.ndarray] | None = None,
    ) -> JointFit:
        """Solve in `max_nfev` steps at most, from the model's estimates (or `theta0`).

        The boards start at their first sightings' poses (or `boards0`'s). `fit_staged` finds a start near the
        answer first.
        """
        require_package("scipy", extra="felix_jr")
        x0 = self.start(theta0, boards0)
        scale = np.ones(self.size) * 0.01
        scale[self.slices["theta"]] = self.sigma[self.free]
        scale[self.slices["thickness"]] = THICKNESS_SIGMA_M
        scale[self.slices["openings"]] = self.opening_sigma
        scale[self.slices["joints"]] = max(self.joint_sigma, 1e-4)
        result = least_squares(
            lambda x: self.residuals(x, theta0),
            x0,
            jac=lambda x: self.jacobian(x, theta0),
            loss="huber",
            f_scale=HUBER,
            x_scale=scale,
            max_nfev=max_nfev,
        )
        theta, boards, thickness = self.unpack(result.x, theta0)
        gcs = {a: self.model.gripper_from_camera(a, theta) for a in self.arms}
        sigmas = {"board": SIGMA_BOARD_PX, "click": SIGMA_CLICK_PX}
        parts = self.residual_parts(result.x, theta0)
        rms = {
            k: float(np.sqrt(np.mean(np.minimum(np.abs(v), BEHIND_CAMERA_PX) ** 2))) * sigmas[k.split()[-1]]
            for k, v in parts.items()
        }
        median = {
            k: float(np.median(np.linalg.norm(v.reshape(-1, 2), axis=1))) * sigmas[k.split()[-1]]
            for k, v in parts.items()
        }
        return JointFit(
            theta,
            gcs,
            boards,
            thickness,
            rms,
            int(result.nfev),
            str(result.message),
            float(result.cost),
            self.openings(result.x),
            result.x,
            median,
        )


def _medoid_mean(poses: Sequence[np.ndarray], point: np.ndarray) -> np.ndarray:
    """Poses averaged robustly, around their medoid.

    The one nearest the others (by where each puts `point`, and its turn) is the reference; those within
    `BOARD_START_AGREE_M` and `BOARD_START_AGREE_DEG` of it are averaged (`_mean_pose`).
    """
    centers = np.array([p[:3, :3] @ point + p[:3, 3] for p in poses])
    turns = np.array(
        [[np.linalg.norm(cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0]) for b in poses] for a in poses]
    )
    apart = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    spread = apart / BOARD_START_AGREE_M + turns / np.radians(BOARD_START_AGREE_DEG)
    reference = int(np.argmin(spread.sum(axis=1)))
    near = [
        p
        for i, p in enumerate(poses)
        if apart[reference, i] <= BOARD_START_AGREE_M
        and turns[reference, i] <= np.radians(BOARD_START_AGREE_DEG)
    ]
    return _mean_pose(near)


def fit_failed(fit: JointFit) -> bool:
    """Whether a fit is a wrong solution: some kind of residual's median error over `FIT_FAILED_PX`.

    The median, not the RMS: a few mis-clicks (`MISCLICK_PX`) lift a kind's RMS, a wrong solution moves every point.
    """
    return any(v > FIT_FAILED_PX for v in fit.median_px.values())


def _mean_pose(poses: Sequence[np.ndarray]) -> np.ndarray:
    """The mean of rigid poses: their translations averaged, their rotations' mean projected back onto a rotation."""
    rotation = np.mean([p[:3, :3] for p in poses], axis=0)
    u, _, vt = np.linalg.svd(rotation)
    out = np.eye(4)
    out[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    out[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return out


STAGED_TRIAL_NFEV = 150  # each start's trial: the wrong solutions show by then (costs 100x+ the right one's)
# the arm-held start's steps at most: only a start (its trial judges it), and a wrong one crept on for 1500 (579 s)
HELD_START_NFEV = 300
# a fit with some kind of residual's median error over this is a wrong solution (the noise is 0.3-2.5 px; joint_sim's failures,
# 2026-11-15: 12-264 px), retried from more starts (`fit_staged`) and flagged (`calibrate`)
FIT_FAILED_PX = 10.0


def fit_staged(problem: "JointProblem", max_nfev: int = MAX_NFEV) -> JointFit:
    """`problem.fit` from the better of two starts: each tried briefly, the lower cost run to the end.

    A start far from the answer (the head's zeros off by degrees, the mounts by centimeters) can leave the joint fit
    stuck in a wrong solution or crawling (joint_sim, 2026-11-11: 1500 steps to a cost 1800 times the truth's, where a
    start at the truth converged in 132). No one start always works there, but the cost tells which did. The starts:

    - **as given:** the model's estimates, the cameras' current poses, each board from its head sightings
      (`JointProblem._first_board_pose`);
    - **the arm held:** every sighting and click, the arms' parameters held where they start, the rest fitted (the
      head's, the cameras' poses on their grippers, the boards): the arm anchors the root frame, so the head and the
      boards cannot swing together, as they did fitted alone (2026-11-13: the boards 3 cm and 7 deg sideways of the
      final fit's on both robot sessions, against 1.5-2.5 cm for the plain start). Then everything, from there.
      Without the moving finger's clicks (`MOVING_FACES`): the hinge and the opening are the arm's, held, and the
      head's clicks on the moving tip pulled the head off to explain them (joint_sim, the head clicking it: this start
      at 34x the answer's cost, the plain one crawling 1500 steps unconverged; without them, converged in 107 s).

    If the result is a wrong solution (`fit_failed`), each start is tried again with the wrist cameras re-estimated
    from their own board sightings (`JointProblem.wrist_cameras_from_sightings`) and the lowest cost kept: a camera
    far off its current pose put the board behind it, where the residual is flat and the fit stops (joint_sim,
    2026-11-15: stopped in 20-30 steps at 1000x the truth's cost, its wrist board 134-264 px off).
    """
    starts: list[tuple[np.ndarray | None, Mapping[str, np.ndarray] | None]] = [(None, None)]
    if problem.arms:
        fixed_tips = [c for c in problem.clicks if c.on_face not in MOVING_FACES]
        held = JointProblem(
            problem.model, problem.board, problem.cameras, problem.sightings, fixed_tips, arms_free=False
        ).fit(max_nfev=min(HELD_START_NFEV, max_nfev))
        starts.append((held.theta, held.boards))
    trials = [problem.fit(theta0, min(STAGED_TRIAL_NFEV, max_nfev), boards0) for theta0, boards0 in starts]
    best = min(trials, key=lambda t: t.cost)
    if best.nfev >= min(STAGED_TRIAL_NFEV, max_nfev):  # not converged within its trial: on to the end
        best = problem.fit(best.theta, max_nfev, best.boards)
    if not fit_failed(best) or not problem.arms:
        return best
    retries = [best]
    for theta0, boards0 in starts:
        theta = problem.model.estimates if theta0 is None else theta0
        boards = (
            dict(boards0)
            if boards0 is not None
            else {p: problem._first_board_pose(p, theta) for p in problem.placements}
        )
        gcs = problem.wrist_cameras_from_sightings(theta, boards)
        retries.append(problem.fit(problem.with_cameras(theta, gcs), max_nfev, boards))
    return min(retries, key=lambda t: t.cost)


# ---- sessions ---------------------------------------------------------------------------------------------------

SESSION_FILE = "joint_session.json"


@dataclass
class JointSession:
    """One arm's joint-calibration session: the board, what the robot was set up as, what was seen and clicked."""

    path: Path
    arm: str
    board: Board
    head_signs: dict[str, int]
    head_refs: dict[str, int]
    intrinsics: dict[str, float]
    arm_calibration: dict[str, dict[str, int]] | None
    sightings: list[BoardSighting] = field(default_factory=list)
    clicks: list[TipClick] = field(default_factory=list)
    placements: dict[str, list[list[float]]] = field(default_factory=dict)  # where the session put each board


def save_session(session: JointSession) -> Path:
    """Write `joint_session.json` in the session's folder (the moments once each, referred to by index)."""
    looks: list[Look] = []
    index: dict[int, int] = {}

    def look_index(look: Look) -> int:
        if id(look) not in index:
            index[id(look)] = len(looks)
            looks.append(look)
        return index[id(look)]

    sightings = [
        {
            "camera": s.camera,
            "placement": s.placement,
            "look": look_index(s.look),
            "ids": s.detection.ids.tolist(),
            "pixels": s.detection.pixels.tolist(),
            "image": s.image,
        }
        for s in session.sightings
    ]
    clicks = [
        {
            "camera": c.camera,
            "arm": c.arm,
            "look": look_index(c.look),
            "pixel": list(map(float, c.pixel)),
            "image": c.image,
            "face": c.on_face,
        }
        for c in session.clicks
    ]
    data = {
        "version": 1,
        "arm": session.arm,
        "board": session.board.to_dict(),
        "head_signs": session.head_signs,
        "head_refs": session.head_refs,
        "intrinsics": session.intrinsics,
        "arm_calibration": session.arm_calibration,
        "placements": session.placements,
        "looks": [
            {"arms": {a: dict(p) for a, p in lk.arms.items()}, "head_ticks": dict(lk.head_ticks)}
            for lk in looks
        ],
        "sightings": sightings,
        "clicks": clicks,
    }
    session.path.mkdir(parents=True, exist_ok=True)
    path = session.path / SESSION_FILE
    path.write_text(json.dumps(data, indent=1))
    return path


def load_session(directory: str | Path) -> JointSession:
    """Read a session folder's `joint_session.json`."""
    directory = Path(directory).expanduser()
    data = json.loads((directory / SESSION_FILE).read_text())
    looks = [Look(lk["arms"], {k: int(v) for k, v in lk["head_ticks"].items()}) for lk in data["looks"]]
    sightings = [
        BoardSighting(
            s["camera"],
            s["placement"],
            looks[s["look"]],
            Detection(
                np.asarray(s["ids"], dtype=int), np.asarray(s["pixels"], dtype=np.float64).reshape(-1, 2)
            ),
            s.get("image", ""),
        )
        for s in data["sightings"]
    ]
    clicks = [
        TipClick(
            c["camera"],
            c["arm"],
            looks[c["look"]],
            np.asarray(c["pixel"], dtype=np.float64),
            c.get("image", ""),
            c.get("face", ""),
        )
        for c in data["clicks"]
    ]
    return JointSession(
        directory,
        data["arm"],
        board_from_dict(data["board"]),
        {k: int(v) for k, v in data["head_signs"].items()},
        {k: int(v) for k, v in data["head_refs"].items()},
        data["intrinsics"],
        data.get("arm_calibration"),
        sightings,
        clicks,
        data.get("placements", {}),
    )


# ---- fitting sessions --------------------------------------------------------------------------------------------


def split_holdout(
    sightings: Sequence[BoardSighting], every: int
) -> tuple[list[BoardSighting], list[BoardSighting]]:
    """(fitted, held out): sightings, whole moments held out (none if `every` is 0).

    Of the moments at which the head and a wrist camera both saw a board, every `every`-th one's.
    """
    if every <= 0:
        return list(sightings), []
    cameras: dict[int, set[str]] = {}
    for s in sightings:
        cameras.setdefault(id(s.look), set()).add(s.camera)
    both = [look for look, seen in cameras.items() if HEAD in seen and len(seen) > 1]
    held_looks = {look for i, look in enumerate(both) if i % every == every - 1}
    return (
        [s for s in sightings if id(s.look) not in held_looks],
        [s for s in sightings if id(s.look) in held_looks],
    )


# a click this far from where the fit puts its fingertip is a mis-click (`calibrate`)
MISCLICK_PX = 30.0
# a camera's board pose from fewer of the board's points than this is not trusted for the held-out check (`agreement`;
# joint_right_full4: from 6, 91 deg off; from 11, 1 deg)
MIN_POSE_POINTS = 10


def agreement(
    model: XleRobotModel,
    board: Board,
    cameras: Mapping[str, WristCamera],
    sightings: Sequence[BoardSighting],
    theta: np.ndarray,
    min_points: int = MIN_POSE_POINTS,
) -> np.ndarray:
    """How far apart the two chains put the board, per moment the head and a wrist camera both saw it (N, 2).

    Through the head (its joints, the camera mount, its lens) and through the arm (its mount and joints, the wrist
    camera's pose on the gripper, its lens), from the joints as read: the board's center apart (mm) and its turn
    between the two (deg). What a grasp planned in the head's picture and done by the arm needs to be small.

    Each camera's board pose is its own (`board_pose`), so a moment either camera saw fewer than `min_points` of the
    board's points at is left out: from a few points the pose's turn is unreliable (joint_right_full4: the head saw 6
    corners at one held-out moment, 91 deg, the other three 1 deg; the fit, which shares one board pose, was not hurt).
    """
    intr = model.intrinsics
    matrix = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
    center = board.points().mean(axis=0)
    by_look: dict[int, dict[str, BoardSighting]] = {}
    for s in sightings:
        by_look.setdefault(id(s.look), {})[s.camera] = s
    out = []
    for seen in by_look.values():
        head = seen.get(HEAD)
        for camera, s in seen.items():
            if head is None or camera not in cameras:
                continue
            if min(len(head.detection), len(s.detection)) < min_points:
                continue
            q = model.joint_vector(s.look.arms, s.look.head_ticks)
            via_head = model.camera_pose(q, theta) @ board_pose(board, head.detection, matrix)
            lens = cameras[camera]
            camera_board = board_pose(board, s.detection, lens.matrix(), lens.coeffs, lens.model == "fisheye")
            via_arm = model.frame_pose(wrist_camera_frame(camera), q, theta) @ camera_board
            apart = (via_head[:3, :3] - via_arm[:3, :3]) @ center + via_head[:3, 3] - via_arm[:3, 3]
            turn = cv2.Rodrigues(via_head[:3, :3].T @ via_arm[:3, :3])[0]
            out.append((1000 * float(np.linalg.norm(apart)), float(np.degrees(np.linalg.norm(turn)))))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def fitted_pixel(problem: "JointProblem", fit: JointFit, click: TipClick) -> np.ndarray:
    """Where `fit` puts a click's fingertip in its camera's picture, at the joints as read (NaN behind the camera)."""
    q = problem._q[id(click)]
    tip = problem.tip(click, q, fit.theta, fit.thickness, fit.openings)[None]
    return problem.project(click.camera, q, tip, fit.theta)[0]


def click_misses(problem: "JointProblem", fit: JointFit) -> list[float]:
    """Each click's distance (px) from where `fit` puts its fingertip, at the joints as read (inf behind the camera)."""
    out = []
    for c in problem.clicks:
        uv = fitted_pixel(problem, fit, c)
        out.append(float(np.linalg.norm(uv - c.pixel)) if np.isfinite(uv).all() else float("inf"))
    return out


def save_misclicks(numbers: Mapping[str, Any], folder: Path) -> None:
    """Each mis-click's picture into `folder`'s `misclicks/`: the click magenta, the fitted fingertip green.

    Each record gets its picture's name (`shown`), for the report.
    """
    for i, m in enumerate([*numbers.get("misclicks", []), *numbers.get("misclicks_kept", [])]):
        picture = cv2.imread(str(Path(m["session"]) / m["image"])) if m["image"] else None
        if picture is None:
            continue
        for uv, color in ((m["pixel"], (255, 0, 255)), (m["fitted"], (0, 220, 0))):
            if np.isfinite(uv).all():
                cv2.circle(picture, (int(round(uv[0])), int(round(uv[1]))), 10, color, 3)
        (folder / "misclicks").mkdir(parents=True, exist_ok=True)
        name = f"misclicks/{i:02d}.png"
        cv2.imwrite(str(folder / name), picture)
        m["shown"] = name


def unknown_names(problem: JointProblem) -> list[tuple[str, slice | int, float, str]]:
    """The unknowns worth reporting, (name, index in the vector, display factor, unit) each.

    The model's free estimates (each wrist camera's pose on its gripper among them), the thickness and the openings;
    not the boards' poses or the joints' errors (nuisance: marginalized, not shown).
    """
    out: list[tuple[str, slice | int, float, str]] = []
    start = problem.slices["theta"].start
    for p in (p for p, free in zip(problem.model.parameters, problem.free, strict=True) if free):
        out.append((p.name, start, float(np.degrees(1.0)) if p.angular else 1000.0, p.unit))
        start += 1
    out.append(("thickness", problem.slices["thickness"].start, 1000.0, "mm"))
    out += [
        (f"{arm} opening", problem.slices["openings"].start + i, 1.0, "deg")
        for i, arm in enumerate(problem.moving_arms)
    ]
    return out


def uncertainty(problem: JointProblem, fit: JointFit, top: int = 3) -> dict[str, Any]:
    """What the fit determined: each unknown's sd, the pairs it could not tell apart, its weakest directions.

    From the Jacobian at the solution (the residuals are over their noise, the priors included): the covariance is
    (J^T J)^-1, the boards' poses and the joints' errors marginalized out. The robust loss's down-weighting is not
    in it (fine once the mis-clicks are out). `sd`: per unknown, in its unit, with `prior` its prior's. `pairs`: the
    unknowns correlated past `CORRELATED`. `weak`: the `top` directions least determined, each unknown scaled by its
    prior, with the share of each unknown in it (over 0.1) and how far the data narrowed it (`sd`, in those units: 1 is
    the prior's, not narrowed at all).
    """
    if fit.x is None:
        raise ValueError("the fit has no solution vector")
    jac = problem.jacobian(fit.x)
    cov = np.linalg.pinv(jac.T @ jac)
    names = unknown_names(problem)
    index = np.array([i for _, i, _, _ in names])
    sub = cov[np.ix_(index, index)]
    sd = np.sqrt(np.clip(np.diag(sub), 0.0, None))
    corr = sub / np.maximum(np.outer(sd, sd), 1e-300)
    theta_free = [p for p, free in zip(problem.model.parameters, problem.free, strict=True) if free]
    priors = [p.sigma for p in theta_free] + [THICKNESS_SIGMA_M] + list(problem.opening_sigma)
    ref = np.array(priors)  # the scale each unknown's weak directions are measured in (`weak`)
    scaled = sub / np.outer(ref, ref)
    values, vectors = np.linalg.eigh(scaled)
    weak = []
    for k in np.argsort(-values)[:top]:
        v = vectors[:, k]
        share = v**2
        weak.append(
            {
                "sd": float(np.sqrt(max(values[k], 0.0))),
                "parts": [
                    (names[j][0], float(np.sign(v[j]) * share[j]))
                    for j in np.argsort(-share)
                    if share[j] > 0.1
                ],
            }
        )
    pairs = [
        (names[i][0], names[j][0], float(corr[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(corr[i, j]) > CORRELATED
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    return {
        "sd": {
            n: {
                "sd": float(sd[i] * f),
                "prior": float(priors[i] * f),
                "unit": u,
            }
            for i, (n, _, f, u) in enumerate(names)
        },
        "pairs": pairs,
        "weak": weak,
    }


def _uncertainty_lines(u: Mapping[str, Any] | None) -> list[str]:
    if not u:
        return []
    lines = [
        "## What the fit determined (from the Jacobian at the solution)",
        "",
        f"Unknowns correlated past {CORRELATED:g} (the data cannot tell them apart; their priors settle them):",
        "",
    ]
    lines += [f"- {a} / {b}: {c:+.2f}" for a, b, c in u["pairs"][:15]] or ["- none"]
    lines += [
        "",
        "The least determined directions (each unknown over its prior; sd 1: not narrowed at all):",
        "",
    ]
    for w in u["weak"]:
        parts = ", ".join(f"{n} {'+' if s > 0 else '-'}{abs(s):.2f}" for n, s in w["parts"])
        lines.append(f"- sd {w['sd']:.2f}: {parts}")
    unlearned = [(n, v) for n, v in u["sd"].items() if v["sd"] > 0.5 * v["prior"]]
    lines += ["", "Unknowns the data narrowed by less than half (sd over the prior's):", ""]
    lines += [f"- {n}: {v['sd']:.2f} of {v['prior']:.2f} {v['unit']}" for n, v in unlearned] or ["- none"]
    return [*lines, ""]


def calibrate(
    model: XleRobotModel,
    sessions: Sequence[JointSession],
    holdout_every: int = 5,
    drop_misclicks: bool = True,
) -> tuple[XleRobotModel, JointFit, dict[str, Any]]:
    """Fit `sessions` (one board kind) from `model`; the calibrated model, the fit, and a report's numbers.

    Every `holdout_every`-th moment at which the head and a wrist camera both saw the board is left out of the fit,
    and scored by how far apart the two cameras' chains put the board there (`agreement`), before and after.

    `drop_misclicks`: a click more than `MISCLICK_PX` from where the fit puts its fingertip is a mis-click (the wrong
    spot, the wrong pane: joint_left_full5's two head clicks 128 and 237 px off, the others 2-7; the robust loss kept
    them from steering the fit, but they read as 65 px RMS): those are left out and the fit done again, and the report
    names them (`numbers["misclicks"]`).
    """
    model = model.with_current_priors()  # its estimates the start, today's table the priors
    # an arm with the moving finger's tip clicked: its hinge fitted too (the clicks' arc sets it)
    for arm in sorted({c.arm for s in sessions for c in s.clicks if c.on_face in MOVING_FACES}):
        model = model.with_gripper_hinge(arm)
    boards = {json.dumps(s.board.to_dict(), sort_keys=True) for s in sessions}
    if len(boards) != 1:
        raise ValueError("the sessions used different boards: fit them separately")
    for s in sessions:
        if s.head_signs != model.head_signs:
            raise ValueError(f"{s.path}: the head's servo directions differ from the calibration's")
    arms = sorted({s.arm for s in sessions})
    cameras = {}
    for arm in arms:
        camera = model.wrist_cameras.get(arm)
        if camera is None:
            raise ValueError(f"the start has no {arm} wrist camera (its lens: calibration.wrist_camera)")
        cameras[arm] = camera
    board = sessions[0].board
    # a placement belongs to its session: two sessions' "p0" are two places, and one board pose for both tore the fit
    # apart (2026-11-14, the left and right arms' sessions: parameters 300 sigma off)
    every = [
        x if len(sessions) == 1 else replace(x, placement=f"{i}:{x.placement}")
        for i, s in enumerate(sessions)
        for x in s.sightings
    ]
    sightings, held = split_holdout(every, holdout_every)
    clicks = [c for s in sessions for c in s.clicks]
    folder_of = {id(c): s.path for s in sessions for c in s.clicks}
    problem = JointProblem(model, board, cameras, sightings, clicks)
    gc0 = {a: model.gripper_from_camera(a) for a in arms}
    before = agreement(model, board, cameras, held, model.estimates)
    fit = fit_staged(problem)
    misses = click_misses(problem, fit)
    misclicks = [
        {
            "camera": c.camera,
            "face": c.on_face,
            "image": c.image,
            "session": str(folder_of[id(c)]),
            "px": miss,
            "pixel": [float(v) for v in c.pixel],
            "fitted": [float(v) for v in fitted_pixel(problem, fit, c)],
        }
        for c, miss in zip(clicks, misses, strict=True)
        if miss > MISCLICK_PX
    ]
    if misclicks and drop_misclicks:
        clicks = [c for c, miss in zip(clicks, misses, strict=True) if miss <= MISCLICK_PX]
        problem = JointProblem(model, board, cameras, sightings, clicks)
        fit = fit_staged(problem)
    after = agreement(model, board, cameras, held, fit.theta)
    calibrated = model.with_current_priors()  # (a copy: `model` is the start, kept for the report)
    calibrated.set_estimates(fit.theta)
    # the map's opening, off by the fitted offset: where the tips meet (`pos_closed`) moves by it
    maps = dict(calibrated.gripper_maps)
    for arm, offset in fit.openings.items():
        maps[arm] = replace(
            maps[arm], pos_closed=maps[arm].pos_closed - offset / maps[arm].deg_per_unit, default=False
        )
    calibrated.gripper_maps = maps
    calibrated.arm_calibration = {
        **model.arm_calibration,
        **{s.arm: arm_record(s.arm_calibration) for s in sessions if s.arm_calibration},
    }
    numbers: dict[str, Any] = {
        "sessions": [str(s.path) for s in sessions],
        "sightings": {k: sum(x.camera == k for x in sightings) for k in [HEAD, *arms]},
        "clicks": {k: sum(c.camera == k for c in clicks) for k in [HEAD, *arms]},
        "rms_px": fit.rms_px,
        "misclicks": misclicks if drop_misclicks else [],
        "misclicks_kept": [] if drop_misclicks else misclicks,
        "held_out": {
            "moments": len(after),
            "left_out": len({id(s.look) for s in held if s.camera == HEAD}) - len(after),
            **{
                name: {
                    "mm": float(np.sqrt(np.mean(a[:, 0] ** 2))),
                    "deg": float(np.sqrt(np.mean(a[:, 1] ** 2))),
                }
                for name, a in (("before", before), ("after", after))
                if len(a)
            },
        },
        "thickness_mm": 1000 * fit.thickness,
        "openings_deg": dict(fit.openings),
        "camera_moved": {
            arm: {
                "mm": float(1000 * np.linalg.norm(fit.gripper_from_camera[arm][:3, 3] - gc0[arm][:3, 3])),
                "deg": float(
                    np.degrees(
                        np.linalg.norm(
                            cv2.Rodrigues(fit.gripper_from_camera[arm][:3, :3] @ gc0[arm][:3, :3].T)[0]
                        )
                    )
                ),
            }
            for arm in arms
        },
        "nfev": fit.nfev,
        "message": fit.message,
        "uncertainty": uncertainty(problem, fit),
        "fit_failed": fit_failed(fit),
    }
    calibrated.fit_info = {**model.fit_info, "joint_fit": numbers}
    return calibrated, fit, numbers


def _held_out_lines(held: Mapping[str, Any]) -> list[str]:
    if not held["moments"]:
        return ["- none held out (too few moments, or `--holdout-every 0`)"]
    b, a = held["before"], held["after"]
    left_out = held.get("left_out", 0)
    return [
        f"- {held['moments']} moments: {b['mm']:.1f} mm, {b['deg']:.2f} deg before -> "
        f"{a['mm']:.1f} mm, {a['deg']:.2f} deg after"
        + (
            f" ({left_out} more left out: a camera saw fewer than {MIN_POSE_POINTS} of the board's points)"
            if left_out
            else ""
        )
    ]


def _misclick_lines(numbers: Mapping[str, Any]) -> list[str]:
    left_out, kept = numbers.get("misclicks", []), numbers.get("misclicks_kept", [])
    if not left_out and not kept:
        return []
    title = "left out, the fit done again without them" if left_out else "kept (--keep-misclicks)"
    lines = [f"## Mis-clicks: more than {MISCLICK_PX:g} px from the fitted fingertip ({title})", ""]
    lines += [
        f"- {m['camera']}, the {m['face']} face, {m['image'] or '(no picture)'}: {m['px']:.0f} px"
        + (f" (`{m['shown']}`: the click magenta, the fitted fingertip green)" if m.get("shown") else "")
        for m in left_out or kept
    ]
    return [*lines, ""]


def report(model: XleRobotModel, start: XleRobotModel, numbers: Mapping[str, Any]) -> str:
    """A markdown report of a `calibrate` run: what was fitted, how well, what moved."""
    lines = [
        "# Joint calibration",
        "",
        f"Sessions: {', '.join(numbers['sessions'])}",
        "",
        "| | count |",
        "| --- | --- |",
        *[f"| board sightings, {k} (fitted) | {v} |" for k, v in numbers["sightings"].items()],
        *[f"| fingertip clicks, {k} | {v} |" for k, v in numbers["clicks"].items() if v],
        "",
        *(
            [
                f"**THE FIT FAILED: a wrong solution** (a kind of residual's median over {FIT_FAILED_PX:g} px, after every "
                "start). Do not use this calibration: look at the residuals, the mis-clicks and the session's pictures.",
                "",
            ]
            if numbers.get("fit_failed")
            else []
        ),
        "## Residuals (RMS, px)",
        "",
        *[f"- {k}: {v:.2f}" for k, v in numbers["rms_px"].items()],
        "",
        *_misclick_lines(numbers),
        "## Held-out moments: the head and the arm on where the board is (RMS)",
        "",
        "The moments the fit did not see at which both cameras saw the board: how far apart the head's chain and the "
        "arm's put it.",
        "",
        *_held_out_lines(numbers["held_out"]),
        "",
        *(
            [
                f"**The fit stopped at its step limit ({numbers['nfev']} steps), not converged:** the robot has an error "
                "the model cannot say (a sag, play), or the start was far off. Look at the residuals and the held-out "
                "moments before using it.",
                "",
            ]
            if numbers["nfev"] >= MAX_NFEV
            else []
        ),
        f"The fingers' thickness: {numbers['thickness_mm']:.1f} mm ({1000 * JAW_THICKNESS_M:.1f} measured).",
        *[
            f"The {arm} gripper's opening: {offset:+.2f} deg off its map (`pos_closed` moved), from the moving finger's tip"
            + (
                " (the map was the servo's default, `GripperMap.from_servo`: now fitted)."
                if start.gripper_maps[arm].default
                else "."
            )
            for arm, offset in numbers.get("openings_deg", {}).items()
        ],
        *[
            f"The {arm} gripper's map is still the servo's default (`GripperMap.from_servo`): no moving-tip clicks fitted it."
            for arm, m in model.gripper_maps.items()
            if m.default
        ],
        "",
        *[
            f"The {arm} wrist camera moved {v['mm']:.1f} mm and {v['deg']:.2f} deg on its gripper."
            for arm, v in numbers["camera_moved"].items()
        ],
        "",
        *_uncertainty_lines(numbers.get("uncertainty")),
        "## The estimates that moved most (in their prior's sigmas)",
        "",
        "| parameter | start | now | change |",
        "| --- | --- | --- | --- |",
    ]
    # by name: the fit may have added parameters the start lacked (the hinge, for the moving tip): from 0 there
    before = start.estimate_dict()
    started = np.array([before.get(p.name, 0.0) for p in model.parameters])
    change = (model.estimates - started) / np.array([p.sigma for p in model.parameters])
    for i in np.argsort(-np.abs(change))[:12]:
        p = model.parameters[i]
        lines.append(
            f"| {p.name} | {p.display(started[i]):+.2f} {p.unit} | {p.display(model.estimates[i]):+.2f} "
            f"{p.unit} | {change[i]:+.2f} sigma |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", type=Path, help="joint_collect session folders")
    parser.add_argument("--start-from", type=Path, required=True, help="the calibration to start from")
    parser.add_argument("--out", type=Path, required=True, help="the calibrated URDF to write")
    parser.add_argument(
        "--keep-misclicks",
        action="store_true",
        help=f"keep clicks more than {MISCLICK_PX:g} px from the fit (named in the report either way)",
    )
    parser.add_argument(
        "--head-axes",
        action="store_true",
        help="fit the head's pan and tilt axes' tilts too (4 more parameters)",
    )
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=5,
        help="hold out every n-th moment both cameras saw the board at (5; 0 = none)",
    )
    args = parser.parse_args()
    start = XleRobotModel.load(args.start_from)
    if args.head_axes:
        start = start.with_head_axes()
    sessions = [load_session(path) for path in args.sessions]
    if sessions[0].intrinsics:
        start.intrinsics = dict(sessions[0].intrinsics)
    calibrated, _, numbers = calibrate(
        start, sessions, args.holdout_every, drop_misclicks=not args.keep_misclicks
    )
    calibrated.save(args.out)
    save_misclicks(numbers, args.out.parent)
    text = report(calibrated, start, numbers)
    args.out.with_suffix(".md").write_text(text)
    print(text)
    print(f"Wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
