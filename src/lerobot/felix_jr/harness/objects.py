"""The harness's objects: what a reasoning chain makes and refers to (`docs/harness.md`, section 1.B).

All are immutable. Geometry lives in the robot's **root frame** (the URDF root, meters, z up): `Grid`, `Points`,
`Plan`, `TableEstimate.h`. Images live in `Capture` (the camera's pixels) and `View` (a crop of a capture, scaled).
An object holds direct references to the objects it was made from (a `View` holds its `Capture`), so it is usable
without a workspace; the workspace (`workspace.py`) records ids, parents and freshness on top.

Every object has `describe()`: the short text the VLM gets back (section 1.E: text for numbers, images for where).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_fit import load_session
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel
from lerobot.felix_jr.depth import with_matrices

from .scene import BLOCKED, FREE, UNKNOWN, Scene, SegmentCheck, read_image

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import Capture as Snapshot

STATE_NAMES = {FREE: "free", UNKNOWN: "unknown", BLOCKED: "blocked"}
CAPTURE_FILE = "capture.json"


def _cm(value: float) -> str:
    return f"{100 * value:.1f}"


@dataclass(frozen=True, eq=False)
class Capture:
    """One color + depth frame, with where the camera was (root frame) and the robot readings it was taken at.

    `scene` answers depth questions in the camera frame; `to_root` / `to_camera` / `project` connect it to the root
    frame through `x_root_camera` (the calibrated `XleRobotModel.camera_pose(q)`). The readings (`arms`,
    `head_ticks`) are kept rather than the pose alone, so a capture loaded with a newer calibration gets the newer
    pose.
    """

    scene: Scene
    x_root_camera: np.ndarray  # (4, 4) X_root_optical
    # "left" -> {"shoulder_pan.pos": deg, ...}, LeRobot's degrees
    arms: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    head_ticks: Mapping[str, int] = field(default_factory=dict)
    timestamp: float = 0.0

    # ---- construction -------------------------------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        color: np.ndarray,
        raw_depth: np.ndarray,
        calibration: dict[str, Any],
        model: XleRobotModel,
        arms: Mapping[str, Mapping[str, float]],
        head_ticks: Mapping[str, int],
        timestamp: float = 0.0,
    ) -> "Capture":
        """From camera frames (RGB color, raw unaligned depth, the RealSense calibration) and servo readings."""
        scene = Scene.from_capture(color, raw_depth, with_matrices(calibration))
        pose = model.camera_pose(model.joint_vector(arms, head_ticks))
        return cls(scene, pose, dict(arms), dict(head_ticks), timestamp)

    @classmethod
    def from_snapshot(cls, snap: "Snapshot", model: XleRobotModel) -> "Capture":
        """From `FelixClient.capture()`: its state holds `"<arm>_arm"` poses and `"head"` servo ticks."""
        state = snap.state
        if "head" not in state:
            raise ValueError(
                "the capture has no head state, so the camera pose is unknown (is the head connected?)"
            )
        arms = {k.removesuffix("_arm"): dict(v) for k, v in state.items() if k.endswith("_arm")}
        head = {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        return cls.from_frames(
            snap.color, snap.raw_depth, snap.calibration, model, arms, head, snap.timestamp
        )

    @classmethod
    def from_session(
        cls, session_dir: str | Path, index: int, model: XleRobotModel, calibration: dict[str, Any]
    ) -> "Capture":
        """Frame `index` of a calibration session folder (`calibration/collect.py`).

        Sessions save color, raw depth and the joint readings but not the RealSense calibration: pass it
        (`depth.load_calibration`).
        """
        session = load_session(session_dir)
        click = next(c for c in session.clicks if c.index == index)
        if click.image is None:
            raise ValueError(f"click {index} of {session_dir} has no saved image")
        color_path = session.path / click.image
        color = cv2.cvtColor(read_image(color_path), cv2.COLOR_BGR2RGB)
        raw = read_image(color_path.with_name(f"{color_path.stem}_depth.png"), cv2.IMREAD_UNCHANGED)
        return cls.from_frames(
            color, raw, calibration, model, {click.arm: click.arm_pose}, click.head_ticks, click.timestamp
        )

    def save(self, folder: str | Path) -> None:
        """Write the capture to `folder`; `load` reads it back.

        Files: `color.png`, `depth_mm.png` (aligned to color, uint16 millimeters, 0 = no depth) and `capture.json`
        (intrinsics, readings, and the pose as saved).
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        s = self.scene
        cv2.imwrite(str(folder / "color.png"), cv2.cvtColor(s.color, cv2.COLOR_RGB2BGR))
        depth_mm = np.where(np.isfinite(s.depth_m), np.rint(s.depth_m * 1000), 0)
        cv2.imwrite(str(folder / "depth_mm.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
        info = {
            "intrinsics": {"fx": s.fx, "fy": s.fy, "ppx": s.ppx, "ppy": s.ppy},
            "arms": {k: dict(v) for k, v in self.arms.items()},
            "head_ticks": dict(self.head_ticks),
            "timestamp": self.timestamp,
            "x_root_camera": self.x_root_camera.tolist(),
        }
        (folder / CAPTURE_FILE).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, folder: str | Path, model: XleRobotModel | None = None) -> "Capture":
        """Read a folder written by `save`.

        With `model`, the pose is recomputed from the readings (use the current calibration); without, the saved
        pose is used.
        """
        folder = Path(folder)
        info = json.loads((folder / CAPTURE_FILE).read_text())
        color = cv2.cvtColor(read_image(folder / "color.png"), cv2.COLOR_BGR2RGB)
        depth_mm = read_image(folder / "depth_mm.png", cv2.IMREAD_UNCHANGED)
        depth_m = np.where(depth_mm > 0, depth_mm / 1000.0, np.nan).astype(np.float32)
        k = info["intrinsics"]
        scene = Scene(color, depth_m, k["fx"], k["fy"], k["ppx"], k["ppy"])
        arms, head = info["arms"], {n: int(t) for n, t in info["head_ticks"].items()}
        if model is not None:
            pose = model.camera_pose(model.joint_vector(arms, head))
        else:
            pose = np.array(info["x_root_camera"], dtype=float)
        return cls(scene, pose, arms, head, float(info["timestamp"]))

    # ---- frames -------------------------------------------------------------------------------------------

    def to_camera(self, xyz: np.ndarray) -> np.ndarray:
        """Root-frame points (..., 3) in the camera frame."""
        r, t = self.x_root_camera[:3, :3], self.x_root_camera[:3, 3]
        return (np.asarray(xyz, dtype=np.float64) - t) @ r

    def to_root(self, xyz: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) in the root frame."""
        r, t = self.x_root_camera[:3, :3], self.x_root_camera[:3, 3]
        return np.asarray(xyz, dtype=np.float64) @ r.T + t

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Capture pixels (..., 2) of root-frame points (..., 3); NaN at or behind the camera."""
        return self.scene.xyz_to_pixel(self.to_camera(xyz))

    def rays(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Root-frame origin (3,) and unit directions (..., 3) of the viewing rays through pixels `uv` (..., 2)."""
        uv = np.asarray(uv, dtype=np.float64)
        s = self.scene
        d_cam = np.stack(
            [(uv[..., 0] - s.ppx) / s.fx, (uv[..., 1] - s.ppy) / s.fy, np.ones(uv.shape[:-1])], axis=-1
        )
        d = d_cam @ self.x_root_camera[:3, :3].T
        return self.x_root_camera[:3, 3].copy(), d / np.linalg.norm(d, axis=-1, keepdims=True)

    def describe(self) -> str:
        """Size, depth coverage and camera position."""
        s = self.scene
        valid = np.isfinite(s.depth_m).mean()
        cam = [float(c) for c in self.x_root_camera[:3, 3]]
        return (
            f"capture {s.width}x{s.height}, depth on {100 * valid:.0f}% of pixels, camera at "
            f"({_cm(cam[0])}, {_cm(cam[1])}, {_cm(cam[2])}) cm"
        )


@dataclass(frozen=True, eq=False)
class View:
    """A crop of a capture, scaled: capture pixel (u, v) shows at view pixel ((u - u0) * scale, (v - v0) * scale).

    Holds no image data; `render` crops and scales the capture's color when drawing.
    """

    capture: Capture
    u0: int
    v0: int
    width: int  # in capture pixels
    height: int
    scale: float = 1.0

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the rendered image in view pixels."""
        return round(self.width * self.scale), round(self.height * self.scale)

    def to_view(self, uv: np.ndarray) -> np.ndarray:
        """Capture pixels (..., 2) to view pixels."""
        uv = np.asarray(uv, dtype=np.float64)
        return (uv - np.array([self.u0, self.v0])) * self.scale

    def to_capture(self, uv: np.ndarray) -> np.ndarray:
        """View pixels (..., 2) to capture pixels."""
        uv = np.asarray(uv, dtype=np.float64)
        return uv / self.scale + np.array([self.u0, self.v0])

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """View pixels (..., 2) of root-frame points (..., 3)."""
        return self.to_view(self.capture.project(xyz))

    def describe(self) -> str:
        """The crop and the scale."""
        w, h = self.size
        return (
            f"view of capture pixels u {self.u0}..{self.u0 + self.width}, v {self.v0}..{self.v0 + self.height}, "
            f"x{self.scale:g} ({w}x{h} px)"
        )


@dataclass(frozen=True, eq=False)
class TableEstimate:
    """The table's height in the root frame, estimated from one capture's depth (`tools.table_height`)."""

    capture: Capture
    h: float  # root-frame z of the table surface, meters
    n_inliers: int
    spread_m: float  # std of the inliers' z
    tilt_deg: float  # angle between the inliers' plane normal and the root z axis
    inlier_mask: np.ndarray  # (H, W) bool, capture pixels that are on the table
    other_h: float | None = None  # a second surface with many points (a shelf?), if any

    def describe(self) -> str:
        """The height and how much to trust it."""
        text = (
            f"table at z = {_cm(self.h)} cm ({self.n_inliers} points, spread {1000 * self.spread_m:.1f} mm, "
            f"tilt {self.tilt_deg:.1f} deg)"
        )
        if self.other_h is not None:
            text += f"; warning: another surface at z = {_cm(self.other_h)} cm"
        return text


@dataclass(frozen=True, eq=False)
class Grid:
    """A square grid on the horizontal plane z = `h` (root frame), with lines every `cell` meters.

    Lines sit at multiples of `cell` (so grids of different sizes share lines) within `x_range` x `y_range`.
    """

    h: float
    cell: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    @property
    def xs(self) -> np.ndarray:
        """X of the lines of constant x."""
        return _multiples(self.x_range, self.cell)

    @property
    def ys(self) -> np.ndarray:
        """Y of the lines of constant y."""
        return _multiples(self.y_range, self.cell)

    def point(self, xy: np.ndarray) -> np.ndarray:
        """Root-frame points (..., 3) on the grid's plane at (x, y) (..., 2), meters."""
        xy = np.asarray(xy, dtype=np.float64)
        return np.concatenate([xy, np.full(xy.shape[:-1] + (1,), self.h)], axis=-1)

    def describe(self) -> str:
        """Height, cell size and extent."""
        return (
            f"grid at z = {_cm(self.h)} cm, {_cm(self.cell)} cm cells, x {_cm(self.x_range[0])}..{_cm(self.x_range[1])}"
            f" cm, y {_cm(self.y_range[0])}..{_cm(self.y_range[1])} cm"
        )


def _multiples(span: tuple[float, float], step: float) -> np.ndarray:
    lo, hi = np.ceil(span[0] / step - 1e-9), np.floor(span[1] / step + 1e-9)
    return np.arange(lo, hi + 1) * step


@dataclass(frozen=True, eq=False)
class Points:
    """Labeled root-frame points, NaN where a point could not be located.

    Points located in an image keep the capture pixels they came from.
    """

    xyz: np.ndarray  # (N, 3)
    labels: tuple[str, ...]
    uv: np.ndarray | None = None  # (N, 2) capture pixels
    capture: Capture | None = None

    def describe(self) -> str:
        """Each point's xyz."""
        rows = []
        for label, p in zip(self.labels, self.xyz, strict=True):
            where = f"({_cm(p[0])}, {_cm(p[1])}, {_cm(p[2])}) cm" if np.isfinite(p).all() else "no depth"
            rows.append(f"{label}: {where}")
        return "points: " + "; ".join(rows)


@dataclass(frozen=True, eq=False)
class Plan:
    """A path through root-frame waypoints, checked against one capture's depth (`tools.check_path`).

    `check` is `Scene.segment_check`'s result for each leg, with its points (`sample_xyz`, `point_first`) moved to
    the root frame so the plan can be drawn in any view.
    """

    waypoints: np.ndarray  # (M, 3)
    check: SegmentCheck
    capture: Capture
    radius: float

    def describe(self) -> str:
        """Each leg's length and state."""
        rows = []
        lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        legs = zip(self.check.status, self.check.t_first, lengths, strict=True)
        for i, (status, t, length) in enumerate(legs):
            row = f"leg {i + 1}->{i + 2} ({_cm(length)} cm): {STATE_NAMES[int(status)]}"
            if status != FREE:
                row += f" from {100 * t:.0f}% along"
            rows.append(row)
        return f"path, {len(self.waypoints)} waypoints, radius {_cm(self.radius)} cm: " + "; ".join(rows)


@dataclass(frozen=True, eq=False)
class Focus:
    """The views the model chose to work in, after the scan (a selection: nothing moves). `views[0]` is primary."""

    views: tuple[View, ...]
    reason: str = ""

    def describe(self) -> str:
        """How many views and why."""
        text = f"focus on {len(self.views)} view(s)"
        return f"{text}: {self.reason}" if self.reason else text


@dataclass(frozen=True, eq=False)
class Reach:
    """Legacy: whether `arm` reached each of `plan`'s waypoints.

    Replaced by `Poses`; kept so that sessions saved before it still load.
    """

    plan: Plan
    arm: str
    poses: tuple[dict[str, float], ...]  # per waypoint, the solved joints in LeRobot degrees
    position_error_m: np.ndarray  # (M,) how far the solution's pinch point is from the waypoint
    pitch_error_deg: np.ndarray  # (M,) how far the gripper's pitch is from `pitch_deg`
    ok: np.ndarray  # (M,) bool
    pitch_deg: float | None  # the requested gripper pitch (-90 = straight down), None = any

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle, including a `Reach` saved before pitches (`tilt_deg` from straight down, `down`)."""
        if "tilt_deg" in state:
            state = dict(state)
            state["pitch_error_deg"] = state.pop("tilt_deg")
            state["pitch_deg"] = -90.0 if state.pop("down") else None
        self.__dict__.update(state)

    def describe(self) -> str:
        """Which waypoints the arm reaches, and by how much it misses the others."""
        rows = []
        for i, (ok, error, off) in enumerate(
            zip(self.ok, self.position_error_m, self.pitch_error_deg, strict=True)
        ):
            if ok:
                rows.append(f"{i + 1} ok")
            else:
                miss = f"{_cm(error)} cm short" if error > 0.005 else ""
                bent = f"pitch {off:.0f} deg off" if off > 5 else ""
                rows.append(f"{i + 1} unreachable ({', '.join(r for r in (miss, bent) if r)})")
        verdict = "reaches all waypoints" if self.ok.all() else f"reaches {int(self.ok.sum())}/{len(self.ok)}"
        if self.pitch_deg is None:
            how = ""
        elif self.pitch_deg == -90:
            how = " pointing down"
        else:
            how = f" at pitch {self.pitch_deg:+.0f} deg"
        return f"{self.arm} arm {verdict}{how}: " + "; ".join(rows)


def _xyz_cm(p: np.ndarray) -> str:
    return "(" + ", ".join(_cm(float(c)) for c in p) + ") cm"


def _ranges(runs: tuple[tuple[float, float], ...]) -> str:
    return ", ".join(f"{lo:+.0f}..{hi:+.0f}" for lo, hi in runs) if runs else "none"


@dataclass(frozen=True, eq=False)
class Poses:
    """Gripper poses solved for `arm` (`reach.check_poses`, `reach.approach_poses`).

    Each pose puts the tool point (the midpoint of jaws `width` apart; the pinch frame's origin is the fixed jaw) at
    `points[i]`, the gripper axis at `pitch_deg`, the jaws across `across`. `approach_poses` also fills the sweep
    fields; when no pitch reaches every target, it has no poses and `pitch_deg` is None, and the per-target ranges
    say what would work.
    """

    arm: str
    capture: Capture
    points: np.ndarray  # (M, 3) where each pose puts the tool point (root frame)
    labels: tuple[str, ...]  # (M,) "1", "2", ... or "back1", "at1", "up1", ...
    frames: np.ndarray  # (M, 4, 4) the solved X_root_pinch
    joints: tuple[dict[str, float], ...]  # (M,) the solved joints, LeRobot degrees
    position_error_m: np.ndarray  # (M,)
    pitch_error_deg: np.ndarray  # (M,)
    jaw_error_deg: np.ndarray  # (M,)
    ok: np.ndarray  # (M,) bool
    pitch_deg: float | None  # the pitch of every pose (None: any, or none found)
    width: float  # m, between the jaws at the tool point
    across: np.ndarray | None  # (3,) the jaws close across it (None: any)
    targets: np.ndarray | None = None  # (K, 3) approach_poses: the targets
    requested_deg: tuple[float, float] | None = None  # approach_poses: the range swept, preferred end first
    reachable_deg: tuple[tuple[float, float], ...] = ()  # approach_poses: pitch runs that reach every target
    target_reachable_deg: tuple[tuple[tuple[float, float], ...], ...] = ()  # approach_poses: runs per target
    back: float = 0.0  # approach_poses: m along the gripper axis before / after each target
    up: float = 0.0  # approach_poses: m above each target

    def describe(self) -> str:
        """The pitch and which poses work; for a sweep, the reachable ranges and which target limits them."""
        head = f"{self.arm} arm"
        if self.pitch_deg is not None:
            head += " pointing down" if self.pitch_deg == -90 else f" at pitch {self.pitch_deg:+.0f} deg"
        if self.width:
            head += f", jaws {_cm(self.width)} cm apart"
        if self.across is not None:
            head += (
                ", jaws closing horizontally"
                if abs(self.across[2]) > 0.999
                else ", jaws across (" + ", ".join(f"{c:.2f}" for c in self.across) + ")"
            )
        text = head + ": " + self._poses_text()
        if self.targets is None or self.requested_deg is None:
            return text
        lo, hi = self.requested_deg
        per_target = "; ".join(
            f"target {k + 1} {_xyz_cm(t)}: {_ranges(r)}"
            for k, (t, r) in enumerate(zip(self.targets, self.target_reachable_deg, strict=True))
        )
        sweep = (
            f"asked pitch {lo:+.0f}..{hi:+.0f}, preferred first; every target: {_ranges(self.reachable_deg)}"
        )
        sweep += f" ({per_target})"
        if self.pitch_deg is None:
            return f"no pitch in {lo:+.0f}..{hi:+.0f} reaches every target with the {self.arm} arm; {sweep}"
        return f"{text}; {sweep}; back {_cm(self.back)} cm along the gripper, up {_cm(self.up)} cm"

    def _poses_text(self) -> str:
        rows = []
        for label, ok, error, pitch, jaw in zip(
            self.labels, self.ok, self.position_error_m, self.pitch_error_deg, self.jaw_error_deg, strict=True
        ):
            if ok:
                rows.append(f"{label} ok")
                continue
            misses = [
                f"{_cm(error)} cm short" if error > 0.005 else "",
                f"pitch {pitch:.0f} deg off" if pitch > 5 else "",
                f"jaws {jaw:.0f} deg off" if jaw > 5 else "",
            ]
            rows.append(f"{label} unreachable ({', '.join(m for m in misses if m)})")
        if not rows:
            return "no poses"
        verdict = "all reachable" if self.ok.all() else f"{int(self.ok.sum())}/{len(self.ok)} reachable"
        return f"{verdict}: " + "; ".join(rows)


@dataclass(frozen=True)
class Move:
    """One command of an `Execution`: joint targets (LeRobot degrees; `gripper.pos` 0-100) for one arm."""

    label: str
    targets: dict[str, float]
    timeout: float  # s, to settle
    settle: bool  # must it settle? (a gripper closing on an object stalls before its target)


@dataclass(frozen=True, eq=False)
class Execution:
    """What `execute` will run (`execute.prepare_execution`): the moves through `poses`, in order.

    The path through the poses is cut into short straight steps (`steps`, each solved like the poses), so the
    joint-space moves stay close to the checked lines; gripper actions go at the poses they name.
    """

    poses: Poses  # the poses the path goes through
    steps: Poses  # the solved steps along it
    moves: tuple[Move, ...]

    def describe(self) -> str:
        """The poses and gripper actions in order, with how many steps lie between."""
        rows, between = [], 0
        labels = set(self.poses.labels)
        for move in self.moves:
            name = move.label.split(" ")[0]
            if name in labels or "gripper" in move.label:
                if between:
                    rows.append(f"{between} steps")
                    between = 0
                rows.append(move.label)
            else:
                between += 1
        return f"{len(self.moves)} moves for the {self.poses.arm} arm: " + ", ".join(rows)

    def lines(self) -> list[str]:
        """Every move with its targets, for a person to check before it runs."""
        return [
            f"{m.label}: " + ", ".join(f"{j.removesuffix('.pos')} {v:+.0f}" for j, v in m.targets.items())
            for m in self.moves
        ]
