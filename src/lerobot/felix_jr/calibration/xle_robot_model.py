"""`XleRobotModel`: the XLeRobot with two SO-101 arms, its head camera and its calibration, in one object.

The design is in `docs/camera_calibration2.md`. In short:

- **Kinematics** come from pinocchio, loaded from the merged URDF (`merge_urdf.py`). The model built from the URDF
  is kept as `nominal` and never changed; `pin_model` is a copy with the current estimates applied.
- **Parameters** are corrections on top of the URDF, one row each (`Param`): `zero:<joint>` (added to the
  joint's angle), `pos:<joint>:x|y|z` (added to the joint origin's translation, in the parent frame) and
  `rot:<joint>:x|y|z` (a rotation vector `w`: the origin's rotation becomes `R_urdf exp(w)`). Values are
  radians / meters; the prior says how wrong we think the URDF is.
- **Every method that depends on the estimates takes them explicitly** (`theta`, a vector in parameter order or a
  `{name: value}` dict; `None` for the current estimates) and applies them, starting from the nominal model,
  before computing anything. Nothing depends on what an earlier call left in the model.
- **Fixed joints.** Pinocchio folds a fixed joint into a frame and pre-multiplies its origin into the placements
  of everything below it, so changing that frame alone would not move its children. The model therefore keeps,
  for every joint placement and frame placement, the chain of URDF origins it is made of, and rebuilds the ones
  whose chain contains a corrected joint.
- **Pinch points and visibility** (the assumptions specific to this robot): the pinch point of an arm is the
  SO-101's tool center point (`<arm>_gripper_frame_link`), and the camera sees it when the gripper's hinge axis
  (`<arm>_gripper` joint) points along the line of sight, from the arm's visible side (the wrist camera blocks
  the other one).

Joint configurations `q` are pinocchio configuration vectors of measured angles (`joint_vector` builds one from
servo readings); `zero:` corrections are added inside, so the same `q` serves every `theta`.
"""

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import (
    ARMS,
    MERGED_URDF,
    OPTICAL_FRAME,
    SO101_JOINTS,
    attr,
    link_of,
)
from lerobot.utils.import_utils import _pinocchio_available, require_package

if TYPE_CHECKING or _pinocchio_available:
    import pinocchio as pin

HEAD_JOINTS = {"head_neck": "head_pan_joint", "head_eyes": "head_tilt_joint"}  # servo -> URDF joint
CAMERA_MOUNT = "fixed_head_camera_link"
ARM_MOUNTS = {"left": "fixed_Base", "right": "fixed_Base_2"}
TICKS_PER_TURN = 4096
AXES = ("x", "y", "z")
SIDES = ("positive", "negative")
CALIBRATION_TAG = "felix_calibration"  # the element `save` adds to the calibrated URDF
# The D415's factory color intrinsics at 640x480 (diagnostics/realsense_check.py); a capture carries its own.
DEFAULT_INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}

# 1-sigma prior on each kind of parameter: how far the URDF value might be from the real robot. Radians / meters.
SIGMAS = {
    "head zero": np.radians(20.0),  # the head's home is not level
    "head origin": 0.002,
    "camera x": 0.003,
    "camera y": 0.015,  # the RGB sensor's lateral offset in the camera link is the least certain
    "camera z": 0.003,
    "camera rot": np.radians(2.0),
    "mount pos": 0.015,  # the SO-101 placed on the XLeRobot's arm base (merge_urdf.py)
    "mount rot": np.radians(5.0),
    "arm zero": np.radians(2.0),
    "arm origin": 0.002,
    "tip": 0.008,  # where "the tip of the closed gripper" is, relative to the SO-101's tool center point
}


def pinch_frame(arm: str) -> str:
    """The frame whose origin is `arm`'s pinch point."""
    return f"{arm}_gripper_frame_link"


def pinch_joint(arm: str) -> str:
    """The fixed joint placing `arm`'s pinch point; its `pos:` parameters are the tip offset."""
    return f"{arm}_gripper_frame_joint"


def arm_joint(arm: str, joint: str) -> str:
    """URDF name of an SO-101 joint (`shoulder_pan`, ...) of `arm`."""
    return f"{arm}_{joint}"


@dataclass(frozen=True)
class Param:
    """One calibrated parameter: a correction on top of the URDF, with its prior (mean, sigma)."""

    name: str
    sigma: float
    group: str  # head, mount, arm or tip
    mean: float = 0.0
    arm: str | None = None  # None for the head and camera, which both arms share

    @property
    def kind(self) -> str:
        """zero, pos or rot."""
        return self.name.split(":")[0]

    @property
    def joint(self) -> str:
        """The URDF joint it corrects."""
        return self.name.split(":")[1]

    @property
    def axis(self) -> int | None:
        """0, 1, 2 for x, y, z (None for `zero:`)."""
        parts = self.name.split(":")
        return AXES.index(parts[2]) if len(parts) > 2 else None

    @property
    def angular(self) -> bool:
        """True for angles (zeros and rotations), False for positions."""
        return self.kind in ("zero", "rot")

    @property
    def unit(self) -> str:
        """Display unit: deg or mm."""
        return "deg" if self.angular else "mm"

    def display(self, value: float) -> float:
        """`value` (radians / meters) in degrees / millimeters."""
        return float(np.degrees(value) if self.angular else value * 1000.0)


def default_parameters(
    arms: Sequence[str] = ARMS, overrides: Mapping[str, tuple[float, float]] | None = None
) -> list[Param]:
    """The parameters we calibrate, with their priors (`SIGMAS`): the head and camera, then each arm's.

    Per arm, 14 head and camera parameters plus 29 of the arm (mount, joint zeros and origins, tip offset), as in
    `model.parameter_set`. The head tower is not included: moving it and moving the arm mount shift every pixel
    identically, so only one of them can be found, and the mount is it. `overrides` replaces some priors,
    `{name: (mean, sigma)}`.
    """
    params = [Param(f"zero:{j}", SIGMAS["head zero"], "head") for j in HEAD_JOINTS.values()]
    for joint in HEAD_JOINTS.values():
        params += [Param(f"pos:{joint}:{a}", SIGMAS["head origin"], "head") for a in AXES]
    params += [Param(f"pos:{CAMERA_MOUNT}:{a}", SIGMAS[f"camera {a}"], "head") for a in AXES]
    params += [Param(f"rot:{CAMERA_MOUNT}:{a}", SIGMAS["camera rot"], "head") for a in AXES]
    for arm in arms:
        mount = ARM_MOUNTS[arm]
        params += [Param(f"pos:{mount}:{a}", SIGMAS["mount pos"], "mount", arm=arm) for a in AXES]
        params += [Param(f"rot:{mount}:{a}", SIGMAS["mount rot"], "mount", arm=arm) for a in AXES]
        for joint in SO101_JOINTS:
            name = arm_joint(arm, joint)
            params.append(Param(f"zero:{name}", SIGMAS["arm zero"], "arm", arm=arm))
            params += [Param(f"pos:{name}:{a}", SIGMAS["arm origin"], "arm", arm=arm) for a in AXES]
        params += [Param(f"pos:{pinch_joint(arm)}:{a}", SIGMAS["tip"], "tip", arm=arm) for a in AXES]
    if overrides:
        params = [
            Param(p.name, overrides[p.name][1], p.group, overrides[p.name][0], p.arm)
            if p.name in overrides
            else p
            for p in params
        ]
    return params


@dataclass
class Prediction:
    """Where the model puts a point in the image: pixel (u, v), depth along the optical axis, camera-frame point."""

    uv: np.ndarray
    z: float
    p_camera: np.ndarray


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: "pin.SE3"
    axis: np.ndarray

    @property
    def moves(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


class XleRobotModel:
    """The robot's kinematics, camera, pinch points and calibration. See the module docstring."""

    def __init__(
        self,
        urdf: str,
        head_signs: Mapping[str, int],
        head_refs: Mapping[str, int],
        intrinsics: Mapping[str, float] | None = None,
        parameters: Sequence[Param] | None = None,
        floor_z: float | None = None,
        visible_side: Mapping[str, str] | None = None,
    ):
        """Build the model from URDF text (`from_urdf` reads a file).

        `head_signs` / `head_refs`: per head servo, the direction and the tick count at its recorded home (from
        `setup.head`); the URDF angle is `sign * (ticks - ref) * 2 pi / 4096`. `intrinsics`: the color camera's
        `fx, fy, ppx, ppy, width, height`. `parameters`: the rows to calibrate (default `default_parameters()`).
        `floor_z`: the table's height in the root frame, if known. `visible_side`: per arm, the side of the
        gripper the camera must see ("positive" by default: the wrist camera is on the negative side).
        """
        require_package("pin", extra="kinematics", import_name="pinocchio")
        self.urdf = urdf
        self.nominal_model = pin.buildModelFromXML(urdf)
        self.pin_model = self.nominal_model.copy()
        self._data = self.pin_model.createData()
        self.head_signs = {k: int(v) for k, v in head_signs.items()}
        self.head_refs = {k: int(v) for k, v in head_refs.items()}
        self.intrinsics = dict(intrinsics or DEFAULT_INTRINSICS)
        self.parameters = list(parameters if parameters is not None else default_parameters())
        self.floor_z = floor_z
        self.visible_side = {arm: "positive" for arm in ARMS} | dict(visible_side or {})
        if not set(self.visible_side.values()) <= set(SIDES):
            raise ValueError(f"visible_side must be one of {SIDES}: {self.visible_side}")
        self.covariance: np.ndarray | None = None  # of the estimates, from the last fit
        self.fit_info: dict[str, Any] = {}  # what the last fit reports about itself (sessions, pixel errors)
        self._index = {p.name: i for i, p in enumerate(self.parameters)}
        if len(self._index) != len(self.parameters):
            raise ValueError("duplicate parameter names")
        self._estimates = np.zeros(len(self.parameters))
        self._read_tree()
        self._zero = np.zeros(self.pin_model.nq)

    @classmethod
    def from_urdf(cls, path: str | Path = MERGED_URDF, **kwargs: Any) -> "XleRobotModel":
        """The model of the URDF file at `path` (default: the merged URDF). See `__init__` for the rest."""
        return cls(Path(path).read_text(), **kwargs)

    # ---- the URDF structure behind pinocchio's placements --------------------------------------------------

    def _read_tree(self) -> None:
        """For every joint and frame placement pinocchio holds, the chain of URDF joint origins it is made of."""
        root = ET.fromstring(self.urdf)
        self._joints: dict[str, _UrdfJoint] = {}
        for element in root.findall("joint"):
            origin, axis = element.find("origin"), element.find("axis")
            xyz = _floats(origin, "xyz")
            rpy = _floats(origin, "rpy")
            name = attr(element, "name")
            self._joints[name] = _UrdfJoint(
                name=name,
                type=attr(element, "type"),
                parent=link_of(element, "parent"),
                child=link_of(element, "child"),
                origin=pin.SE3(pin.rpy.rpyToMatrix(*rpy), xyz),
                axis=_floats(axis, "xyz", default=(1.0, 0.0, 0.0)),
            )
        into = {j.child: j for j in self._joints.values()}

        def fixed_path(link: str) -> list[str]:
            """The fixed joints from the nearest moving joint (or the root) down to `link`."""
            path = []
            while link in into and not into[link].moves:
                path.append(into[link].name)
                link = into[link].parent
            return path[::-1]

        model = self.nominal_model
        for p in self.parameters:
            if p.joint not in self._joints:
                raise KeyError(f"{p.name}: no joint {p.joint!r} in the URDF")
            if p.kind == "zero" and not (
                model.existJointName(p.joint) and model.joints[model.getJointId(p.joint)].nq == 1
            ):
                raise ValueError(f"{p.name}: zero offsets need a one-dimensional moving joint")
        corrected = {p.joint for p in self.parameters if p.kind in ("pos", "rot")}
        # (is_frame, index, chain): placements that depend on a corrected origin, rebuilt by `_apply`.
        self._placements: list[tuple[bool, int, list[str]]] = []
        self._all_placements: list[tuple[bool, int, list[str]]] = []  # every placement, for `check_structure`
        for joint_id in range(1, model.njoints):
            joint = self._joints[model.names[joint_id]]
            self._all_placements.append((False, joint_id, fixed_path(joint.parent) + [joint.name]))
        for frame_id, frame in enumerate(model.frames):
            if frame.type == pin.FrameType.BODY:
                self._all_placements.append((True, frame_id, fixed_path(frame.name)))
            elif frame.type == pin.FrameType.FIXED_JOINT and frame.name in self._joints:
                self._all_placements.append((True, frame_id, fixed_path(self._joints[frame.name].child)))
        self._placements = [entry for entry in self._all_placements if corrected & set(entry[2])]
        self._frame_ids = {f.name: i for i, f in enumerate(model.frames) if f.type == pin.FrameType.BODY}

    def check_structure(self, atol: float = 1e-9) -> None:
        """Raise if rebuilding the placements from the URDF origins does not give pinocchio's own (a self-test)."""
        for is_frame, index, chain in self._all_placements:
            rebuilt = pin.SE3.Identity()
            for name in chain:
                rebuilt = rebuilt * self._joints[name].origin
            original = (
                self.nominal_model.frames[index].placement
                if is_frame
                else self.nominal_model.jointPlacements[index]
            )
            if not np.allclose(rebuilt.homogeneous, original.homogeneous, atol=atol):
                kind = "frame" if is_frame else "joint"
                raise AssertionError(f"{kind} {index} ({chain}) does not match pinocchio's placement")

    # ---- parameters --------------------------------------------------------------------------------------------

    @property
    def estimates(self) -> np.ndarray:
        """The current estimates, in parameter order (a copy)."""
        return self._estimates.copy()

    def estimate_dict(self, theta: np.ndarray | Mapping[str, float] | None = None) -> dict[str, float]:
        """`{name: value}` of `theta` (default: the current estimates)."""
        return {p.name: float(v) for p, v in zip(self.parameters, self._vector(theta), strict=True)}

    def set_estimates(self, theta: np.ndarray | Mapping[str, float]) -> None:
        """Make `theta` the current estimates (a dict leaves missing parameters at 0)."""
        self._estimates = self._vector(theta)

    def nominal(self, name: str) -> float:
        """The URDF value behind parameter `name`: the origin's translation (m) or rotation vector (rad), 0 for a zero."""
        p = self.parameters[self._index[name]]
        if p.kind == "zero":
            return 0.0
        origin = self._joints[p.joint].origin
        values = origin.translation if p.kind == "pos" else pin.log3(origin.rotation)
        return float(values[p.axis])

    def _vector(self, theta: np.ndarray | Mapping[str, float] | None) -> np.ndarray:
        if theta is None:
            return self._estimates.copy()
        if isinstance(theta, Mapping):
            vector = np.zeros(len(self.parameters))
            for name, value in theta.items():
                if name not in self._index:
                    raise KeyError(f"unknown parameter {name!r}")
                vector[self._index[name]] = value
            return vector
        vector = np.asarray(theta, dtype=float)
        if vector.shape != (len(self.parameters),):
            raise ValueError(f"expected {len(self.parameters)} estimates, got shape {vector.shape}")
        return vector.copy()

    def _corrected_origins(
        self, theta: np.ndarray | Mapping[str, float] | None
    ) -> tuple[dict[str, "pin.SE3"], np.ndarray]:
        """The corrected origin of every joint with a `pos:` / `rot:` correction, and the zero offsets as a q vector."""
        shifts: dict[str, np.ndarray] = {}
        rotations: dict[str, np.ndarray] = {}
        zero = np.zeros(self.nominal_model.nq)
        for p, value in zip(self.parameters, self._vector(theta), strict=True):
            if p.kind == "zero":
                zero[self._idx_q(p.joint)] += value
            else:
                target = shifts if p.kind == "pos" else rotations
                target.setdefault(p.joint, np.zeros(3))[p.axis] += value
        origins = {}
        for name in shifts.keys() | rotations.keys():
            nominal = self._joints[name].origin
            rotation = nominal.rotation @ pin.exp3(rotations.get(name, np.zeros(3)))
            origins[name] = pin.SE3(rotation, nominal.translation + shifts.get(name, np.zeros(3)))
        return origins, zero

    def _apply(self, theta: np.ndarray | Mapping[str, float] | None) -> None:
        """Rebuild the working model's placements from the nominal origins and `theta` (never from earlier ones)."""
        origins, self._zero = self._corrected_origins(theta)
        for is_frame, index, chain in self._placements:
            placement = pin.SE3.Identity()
            for name in chain:
                placement = placement * origins.get(name, self._joints[name].origin)
            if is_frame:
                self.pin_model.frames[index].placement = placement
            else:
                self.pin_model.jointPlacements[index] = placement

    def _evaluate(self, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None) -> None:
        self._apply(theta)
        pin.framesForwardKinematics(self.pin_model, self._data, np.asarray(q, dtype=float) + self._zero)

    # ---- joints ------------------------------------------------------------------------------------------------

    def head_angles(self, head_ticks: Mapping[str, int]) -> dict[str, float]:
        """URDF head joint angles (radians, before the zero corrections) from raw servo ticks."""
        return {
            joint: self.head_signs[servo]
            * (head_ticks[servo] - self.head_refs[servo])
            * 2
            * np.pi
            / TICKS_PER_TURN
            for servo, joint in HEAD_JOINTS.items()
        }

    def joint_vector(
        self,
        arms: Mapping[str, Mapping[str, float]] | None = None,
        head_ticks: Mapping[str, int] | None = None,
    ) -> np.ndarray:
        """Pinocchio `q` from servo readings: `arms` maps "left" / "right" to LeRobot's `{"shoulder_pan.pos": deg, ...}`
        (the SO-101 URDF's angles one to one), `head_ticks` the head servos' raw ticks. Joints not given are at 0
        (the head at its recorded home); the gripper stays at 0, which does not move the pinch point."""
        q = pin.neutral(self.nominal_model)
        angles = self.head_angles(head_ticks) if head_ticks is not None else {}
        for arm, pose in (arms or {}).items():
            angles |= {
                arm_joint(arm, j): float(np.radians(pose[f"{j}.pos"]))
                for j in SO101_JOINTS
                if f"{j}.pos" in pose
            }
        for name, angle in angles.items():
            q[self._idx_q(name)] = angle
        return q

    def joint_limits(self, arm: str) -> dict[str, tuple[float, float]]:
        """`{joint: (lower, upper)}` in radians for `arm`'s SO-101 joints, from the URDF."""
        model = self.nominal_model
        return {
            j: (
                float(model.lowerPositionLimit[self._idx_q(arm_joint(arm, j))]),
                float(model.upperPositionLimit[self._idx_q(arm_joint(arm, j))]),
            )
            for j in SO101_JOINTS
        }

    def _idx_q(self, joint: str) -> int:
        return int(self.nominal_model.joints[self.nominal_model.getJointId(joint)].idx_q)

    # ---- geometry and measurement ----------------------------------------------------------------------------

    def frame_pose(
        self, frame: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """X_root_frame (4x4) of a link's frame."""
        self._evaluate(q, theta)
        return self._pose(frame)

    def camera_pose(self, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None) -> np.ndarray:
        """X_root_optical (4x4): the color camera's optical frame (x right, y down, z forward)."""
        return self.frame_pose(OPTICAL_FRAME, q, theta)

    def pinch_point(
        self, arm: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Root-frame position (m) of `arm`'s estimated pinch point. Everything that needs it goes through here."""
        self._evaluate(q, theta)
        return self._pose(pinch_frame(arm))[:3, 3]

    def project(
        self, points: np.ndarray, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of root-frame points (N, 3); NaN pixels at or behind the camera."""
        self._evaluate(q, theta)
        return self._project(np.atleast_2d(points))

    def predict(
        self, arm: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> Prediction:
        """Where `arm`'s pinch point lands in the image."""
        self._evaluate(q, theta)
        tip = self._pose(pinch_frame(arm))[:3, 3]
        uv, z = self._project(tip[None])
        x_root_camera = self._pose(OPTICAL_FRAME)
        return Prediction(uv[0], float(z[0]), x_root_camera[:3, :3].T @ (tip - x_root_camera[:3, 3]))

    def predict_many(
        self, arm: str, qs: Sequence[np.ndarray], theta: np.ndarray | Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of `arm`'s pinch point at each of `qs`, applying `theta` once (for fits)."""
        self._apply(theta)
        uv, z = np.full((len(qs), 2), np.nan), np.full(len(qs), np.nan)
        for i, q in enumerate(qs):
            pin.framesForwardKinematics(self.pin_model, self._data, np.asarray(q, dtype=float) + self._zero)
            uv_i, z_i = self._project(self._pose(pinch_frame(arm))[:3, 3][None])
            uv[i], z[i] = uv_i[0], z_i[0]
        return uv, z

    def visibility(
        self, arm: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> float:
        """Signed cosine between `arm`'s gripper hinge axis and the line of sight from its pinch point to the camera.

        Near +1 / -1 the camera sees the jaws meet from the positive / negative side; near 0 it looks along the
        plane the jaws open in, and the pinch point hides.
        """
        self._evaluate(q, theta)
        hinge = self._joints[arm_joint(arm, "gripper")]
        axis = self._pose(hinge.child)[:3, :3] @ (hinge.axis / np.linalg.norm(hinge.axis))
        sight = self._pose(OPTICAL_FRAME)[:3, 3] - self._pose(pinch_frame(arm))[:3, 3]
        return float(axis @ sight) / float(np.linalg.norm(sight))

    def in_view(self, uv: np.ndarray, margin: float = 0.0) -> bool:
        """True if pixel `uv` is inside the image, at least `margin` pixels from the border."""
        return bool(
            np.isfinite(uv).all()
            and margin <= uv[0] < self.intrinsics["width"] - margin
            and margin <= uv[1] < self.intrinsics["height"] - margin
        )

    def _pose(self, frame: str) -> np.ndarray:
        return self._data.oMf[self._frame_ids[frame]].homogeneous.copy()

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_root_camera = self._pose(OPTICAL_FRAME)
        p_camera = (points - x_root_camera[:3, 3]) @ x_root_camera[:3, :3]
        z = p_camera[:, 2]
        safe_z = np.where(z > 1e-6, z, np.nan)
        c = self.intrinsics
        uv = np.stack(
            [c["fx"] * p_camera[:, 0] / safe_z + c["ppx"], c["fy"] * p_camera[:, 1] / safe_z + c["ppy"]],
            axis=1,
        )
        return uv, z

    # ---- export and persistence -----------------------------------------------------------------------------

    def to_urdf(self, theta: np.ndarray | Mapping[str, float] | None = None) -> str:
        """The URDF with `theta` (default: the current estimates) written into the joint origins.

        Zero offsets become a fixed rotation about the joint's axis folded into its origin, so a model of the
        result with no corrections gives the same poses for the same measured `q`. Joint limits are untouched:
        they bound the measured position, which does not change.
        """
        origins, _ = self._corrected_origins(theta)
        zeros = {
            p.joint: v for p, v in zip(self.parameters, self._vector(theta), strict=True) if p.kind == "zero"
        }
        root = ET.fromstring(self.urdf)
        for element in root.findall("joint"):
            name = attr(element, "name")
            if name not in origins and not zeros.get(name):
                continue
            joint = self._joints[name]
            origin = origins.get(name, joint.origin)
            rotation, translation = origin.rotation, origin.translation
            if zeros.get(name):
                axis = joint.axis / np.linalg.norm(joint.axis)
                if joint.type == "prismatic":
                    translation = translation + rotation @ (axis * zeros[name])
                else:
                    rotation = rotation @ pin.exp3(axis * zeros[name])
            element_origin = element.find("origin")
            if element_origin is None:
                element_origin = ET.SubElement(element, "origin")
            element_origin.set("xyz", _format(translation))
            element_origin.set("rpy", _format(pin.rpy.matrixToRpy(rotation)))
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"

    def save(self, path: str | Path) -> None:
        """Write the calibrated URDF: `to_urdf()` plus a `<felix_calibration>` element holding everything else.

        Any URDF reader (pinocchio, placo, a viewer) uses the file as it is and ignores the element. `load` reads
        the element back: the servo map, intrinsics, visible sides, floor, the parameter table, the estimates,
        their covariance, `fit_info`, and the URDF's original origins of every joint a parameter corrects, so the
        nominal model, and with it the priors, come back exactly.
        """
        root = ET.fromstring(self.to_urdf())
        nominal_root = ET.fromstring(self.urdf)
        corrected = {p.joint for p in self.parameters}
        origins = {}
        for element in nominal_root.findall("joint"):
            if attr(element, "name") in corrected:
                origin = element.find("origin")
                origins[attr(element, "name")] = None if origin is None else dict(origin.attrib)
        data = {
            "version": 1,
            "head_signs": self.head_signs,
            "head_refs": self.head_refs,
            "intrinsics": self.intrinsics,
            "floor_z": self.floor_z,
            "visible_side": self.visible_side,
            "parameters": [asdict(p) for p in self.parameters],
            "estimates": self._estimates.tolist(),
            "covariance": None if self.covariance is None else np.asarray(self.covariance).tolist(),
            "fit_info": self.fit_info,
            "nominal_origins": origins,
        }
        element = ET.SubElement(root, CALIBRATION_TAG, {"version": "1"})
        element.text = "\n" + json.dumps(data, indent=1) + "\n"
        Path(path).write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> "XleRobotModel":
        """Read a calibrated URDF written by `save`."""
        root = ET.fromstring(Path(path).read_text())
        element = root.find(CALIBRATION_TAG)
        if element is None or element.text is None:
            raise ValueError(f"{path} has no <{CALIBRATION_TAG}> element: not written by XleRobotModel.save")
        data = json.loads(element.text)
        root.remove(element)
        for joint in root.findall("joint"):
            name = attr(joint, "name")
            if name not in data["nominal_origins"]:
                continue
            origin = joint.find("origin")
            if origin is not None:
                joint.remove(origin)
            if data["nominal_origins"][name] is not None:
                ET.SubElement(joint, "origin", data["nominal_origins"][name])
        model = cls(
            '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n",
            data["head_signs"],
            data["head_refs"],
            data["intrinsics"],
            [Param(**p) for p in data["parameters"]],
            data["floor_z"],
            data["visible_side"],
        )
        model.set_estimates(np.array(data["estimates"], dtype=float))
        if data["covariance"] is not None:
            model.covariance = np.array(data["covariance"], dtype=float)
        model.fit_info = data["fit_info"]
        return model


def estimates_from_v1(theta: Mapping[str, float], arm: str) -> dict[str, float]:
    """Translate a `CalibrationModel` (version 1) parameter dict for `arm` into `XleRobotModel` names.

    Version 1 prefixes the arm's parameters with `so101:` and uses roll-pitch-yaw for rotations; here the arm's
    joints carry the arm's prefix and rotations are rotation vectors. The translation is exact.
    """
    out: dict[str, float] = {}
    rpy: dict[str, np.ndarray] = {}
    for key, value in theta.items():
        on_arm = key.startswith("so101:")
        kind, joint, *rest = key.removeprefix("so101:").split(":")
        if on_arm:
            joint = arm_joint(arm, joint)
        if kind == "rot":
            rpy.setdefault(joint, np.zeros(3))[("roll", "pitch", "yaw").index(rest[0])] = value
        elif kind == "pos":
            out[f"pos:{joint}:{rest[0]}"] = float(value)
        else:
            out[f"zero:{joint}"] = float(value)
    for joint, angles in rpy.items():
        w = pin.log3(pin.rpy.rpyToMatrix(*angles))
        out |= {f"rot:{joint}:{a}": float(v) for a, v in zip(AXES, w, strict=True)}
    return out


def _floats(
    element: ET.Element | None, key: str, default: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    value = None if element is None else element.get(key)
    if value is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()], dtype=float)


def _format(values: np.ndarray) -> str:
    return " ".join(f"{v:.15g}" for v in values)
