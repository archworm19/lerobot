"""One fit of the head, an arm and its wrist camera: boards seen by both cameras, fingertips clicked in both.

`docs/camera_calibration.md`, "The rework" (2d). The unknowns: the model's estimates (`XleRobotModel.parameters`:
the head and camera, and each fitted arm's mount, joint zeros and origins and tip offset; with their priors), each
fitted arm's wrist camera pose on its gripper (`gripper_from_camera`), each board placement's pose in the root frame,
and the fingers' thickness (with a tight prior). The residuals, all in pixels over their noise:

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
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.felix_jr.calibration.boards import Board, Detection, board_pose
from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
from lerobot.felix_jr.calibration.xle_robot_model import (
    JAW_THICKNESS_M,
    WristCamera,
    XleRobotModel,
    arm_joint,
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
JOINT_SIGMA_DEG = 0.2  # a joint's reading against where it is (a servo tick is 0.09 deg; play)
HUBER = 2.0  # in sigmas: beyond, a residual is down-weighted
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


@dataclass(frozen=True)
class TipClick:
    """`arm`'s fixed fingertip clicked at `pixel` in `camera`'s picture.

    "head": the face the head sees (the pinch point); the arm's own wrist camera: the other face.
    """

    camera: str
    arm: str
    look: Look
    pixel: np.ndarray


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
    ):
        """`cameras`: each fitted arm's wrist camera (its lens; its `gripper_from_camera` is the starting point).

        `joint_sigma_deg`: how far a joint's reading may be from where it really is (0: the readings are taken as
        exact). Each moment's fitted-arm joints get an error of their own, with this prior: the same for every point
        seen at that moment, it would otherwise bend the robot's parameters.
        """
        self.model, self.board, self.cameras = model, board, dict(cameras)
        self.sightings, self.clicks = list(sightings), list(clicks)
        self.arms = sorted(self.cameras)
        self.placements = sorted({s.placement for s in self.sightings})
        self.free = np.array([p.arm is None or p.arm in self.arms for p in model.parameters])
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
        n_theta, n_gc, n_board = int(self.free.sum()), 6 * len(self.arms), 6 * len(self.placements)
        self.slices = {
            "theta": slice(0, n_theta),
            "gc": slice(n_theta, n_theta + n_gc),
            "boards": slice(n_theta + n_gc, n_theta + n_gc + n_board),
            "thickness": slice(n_theta + n_gc + n_board, n_theta + n_gc + n_board + 1),
        }
        start = self.slices["thickness"].stop
        self.slices["joints"] = slice(start, start + len(self.looks) * len(self.joint_index))
        self.size = self.slices["joints"].stop

    # ---- the unknowns as one vector -----------------------------------------------------------------------------

    def start(self, theta0: np.ndarray | None = None) -> np.ndarray:
        """The unknowns at the start: the model's estimates, the cameras' poses, the joints as read.

        Each board starts where its first sighting puts it.
        """
        theta = self.model.estimates if theta0 is None else np.asarray(theta0, dtype=np.float64)
        boards = [_vec(self._first_board_pose(p, theta)) for p in self.placements]
        gcs = [_vec(np.asarray(self.cameras[a].gripper_from_camera, dtype=np.float64)) for a in self.arms]
        joints = np.zeros(self.slices["joints"].stop - self.slices["joints"].start)
        return np.concatenate([theta[self.free], *gcs, *boards, [JAW_THICKNESS_M], joints])

    def unpack(
        self, x: np.ndarray, theta0: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
        """(theta, gripper_from_camera per arm, root_from_board per placement, thickness) from the vector."""
        theta = (self.model.estimates if theta0 is None else np.asarray(theta0, dtype=np.float64)).copy()
        theta[self.free] = x[self.slices["theta"]]
        gc, boards = x[self.slices["gc"]], x[self.slices["boards"]]
        gcs = {a: _pose(gc[6 * i : 6 * i + 6]) for i, a in enumerate(self.arms)}
        poses = {p: _pose(boards[6 * i : 6 * i + 6]) for i, p in enumerate(self.placements)}
        return theta, gcs, poses, float(x[self.slices["thickness"]][0])

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

    def _first_board_pose(self, placement: str, theta: np.ndarray) -> np.ndarray:
        """The board's pose from one sighting through the starting model (PnP): the fit's start for it."""
        sighting = next(s for s in self.sightings if s.placement == placement)
        q = self._q[id(sighting)]
        if sighting.camera == HEAD:
            intr = self.model.intrinsics
            matrix = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
            camera_board = board_pose(self.board, sighting.detection, matrix)
            return self.model.camera_pose(q, theta) @ camera_board
        camera = self.cameras[sighting.camera]
        camera_board = board_pose(
            self.board, sighting.detection, camera.matrix(), camera.coeffs, camera.model == "fisheye"
        )
        root_link = self.model.frame_pose(wrist_camera_link(sighting.camera), q, theta)
        return root_link @ np.asarray(camera.gripper_from_camera, dtype=np.float64) @ camera_board

    # ---- predictions and residuals ------------------------------------------------------------------------------

    def project(
        self, camera: str, q: np.ndarray, points_root: np.ndarray, theta: np.ndarray, gcs
    ) -> np.ndarray:
        """Pixels (N, 2) of root-frame points in `camera`'s picture at `q` (NaN behind it)."""
        if camera == HEAD:
            return self.model.project(points_root, q, theta)[0]
        root_camera = self.model.frame_pose(wrist_camera_link(camera), q, theta) @ gcs[camera]
        in_camera = (points_root - root_camera[:3, 3]) @ root_camera[:3, :3]
        return self.cameras[camera].project(in_camera)

    def tip(self, click: TipClick, q: np.ndarray, theta: np.ndarray, thickness: float) -> np.ndarray:
        """The clicked fingertip in the root frame: the pinch point, or the other face's for the wrist camera."""
        pinch = self.model.frame_pose(pinch_frame(click.arm), q, theta)
        if click.camera == HEAD:
            return pinch[:3, 3]
        face = self.model.wrist_face(click.arm) * (thickness / JAW_THICKNESS_M)
        return pinch[:3, :3] @ face + pinch[:3, 3]

    def residual_parts(self, x: np.ndarray, theta0: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Each kind's residuals over its noise: "head board", "<arm> board", "head click", "<arm> click"."""
        theta, gcs, boards, thickness = self.unpack(x, theta0)
        parts: dict[str, list[np.ndarray]] = {}
        for s in self.sightings:
            pose = boards[s.placement]
            points = self.board.points(s.detection.ids) @ pose[:3, :3].T + pose[:3, 3]
            uv = self.project(s.camera, self.q(s, x), points, theta, gcs)
            parts.setdefault(f"{s.camera} board", []).append(
                ((uv - s.detection.pixels) / SIGMA_BOARD_PX).ravel()
            )
        for c in self.clicks:
            q = self.q(c, x)
            uv = self.project(c.camera, q, self.tip(c, q, theta, thickness)[None], theta, gcs)
            parts.setdefault(f"{c.camera} click", []).append(((uv[0] - c.pixel) / SIGMA_CLICK_PX).ravel())
        return {k: np.nan_to_num(np.concatenate(v), nan=BEHIND_CAMERA_PX) for k, v in parts.items()}

    def priors(self, x: np.ndarray) -> np.ndarray:
        """The estimates', the thickness' and the joints' errors' prior residuals."""
        theta = (x[self.slices["theta"]] - self.mean[self.free]) / self.sigma[self.free]
        thickness = (x[self.slices["thickness"]] - JAW_THICKNESS_M) / THICKNESS_SIGMA_M
        joints = x[self.slices["joints"]] / self.joint_sigma if self.looks else np.zeros(0)
        return np.concatenate([theta, thickness, joints])

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

    def fit(self, theta0: np.ndarray | None = None, max_nfev: int | None = None) -> JointFit:
        """Solve from the model's estimates (or `theta0`) and the cameras' current poses."""
        require_package("scipy", extra="felix_jr")
        x0 = self.start(theta0)
        scale = np.ones(self.size) * 0.01
        scale[self.slices["theta"]] = self.sigma[self.free]
        scale[self.slices["thickness"]] = THICKNESS_SIGMA_M
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
        theta, gcs, boards, thickness = self.unpack(result.x, theta0)
        sigmas = {"board": SIGMA_BOARD_PX, "click": SIGMA_CLICK_PX}
        rms = {
            k: float(np.sqrt(np.mean(np.minimum(np.abs(v), BEHIND_CAMERA_PX) ** 2))) * sigmas[k.split()[-1]]
            for k, v in self.residual_parts(result.x, theta0).items()
        }
        return JointFit(theta, gcs, boards, thickness, rms, int(result.nfev), str(result.message))
