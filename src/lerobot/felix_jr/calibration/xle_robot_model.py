"""`XleRobotModel`: the XLeRobot with two SO-101 arms, its head camera and its calibration, in one object.

The design is in `docs/camera_calibration.md`. In short:

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
- **The gripper** is a claw: the moving jaw turns about the hinge (`<arm>_gripper`) and its tip meets the fixed jaw's
  at the pinch point. Its opening is measured from there (`GripperMap`, recorded by `calibration.gripper_step`), not as
  the URDF's joint angle: in the SO-101 URDF the jaws meet at about -11.4 deg (from the moving jaw's mesh), past the
  joint's -10 deg limit. The moving tip turns on a circle of `jaw_radius` about the hinge axis.
- **Gripper points.** Everything that predicts a point (`point`, `predict`, `predict_many`, `pixel_jacobian`,
  `visibility`) takes which one (`POINTS`): `pinch`, the fixed jaw's tip (the calibration's), or `moving_tip`, the
  moving jaw's tip: the pinch point turned about the hinge axis by the jaws' opening (`opening_deg`, degrees from
  closed; `XleRobotModel.opening_deg` reads it off `gripper.pos`). Both are fixed to the gripper body for a given
  opening. `pinch_point` and `moving_tip` are the two by name.
- **Grasping.** An object `width` wide is held between the tips: the jaws open until the moving tip is `width` from
  the pinch point (`opening_for_width`), and the object's center goes midway between the tips (`grasp_offset`, in the
  pinch frame; with a margin, aimed that much toward the moving jaw, so the fixed one, which does not move, clears
  the object). `moving_jaw` is the turn that takes the fixed jaw to the moving one at an opening (to draw it), and
  `gripper_pos_for` the `gripper.pos` that opens the jaws that far (the arm's `GripperMap`).

Joint configurations `q` are pinocchio configuration vectors of measured angles (`joint_vector` builds one from
servo readings); `zero:` corrections are added inside, so the same `q` serves every `theta`.
"""

import json
import logging
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

log = logging.getLogger(__name__)

HEAD_JOINTS = {"head_neck": "head_pan_joint", "head_eyes": "head_tilt_joint"}  # servo -> URDF joint
CAMERA_MOUNT = "fixed_head_camera_link"
ARM_MOUNTS = {"left": "fixed_Base", "right": "fixed_Base_2"}
TICKS_PER_TURN = 4096
AXES = ("x", "y", "z")
SIDES = ("positive", "negative")
CALIBRATION_TAG = "felix_calibration"  # the element `save` adds to the calibrated URDF
# The fields of a LeRobot motor calibration that decide what its readings mean (not `id`, the bus address)
MOTOR_CALIBRATION_FIELDS = ("drive_mode", "homing_offset", "range_min", "range_max")
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


POINTS = ("pinch", "moving_tip")  # the gripper points the model predicts (module docstring)


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

    Per arm, 14 head and camera parameters plus 29 of the arm (mount, joint zeros and origins, tip offset). Joint
    origins and most of the rest stay at their priors in practice (the fit's report says which were learned), but
    keeping them in costs little and lets the uncertainty show. The head tower is not included: moving it and moving
    the arm mount shift every pixel identically, so only one of them can be found, and the mount is it.
    `overrides` replaces some priors, `{name: (mean, sigma)}`.
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
    """What the model says about one of an arm's gripper points (`POINTS`).

    Its pixel (u, v), depth along the optical axis, position in the camera frame and in the root frame,
    `visibility` (see `XleRobotModel.visibility`) and `hidden_by` (`XleRobotModel.hidden_by`).
    """

    uv: np.ndarray
    z: float
    p_camera: np.ndarray
    point: np.ndarray
    visibility: float
    hidden_by: tuple[str, ...] = ()  # the arms whose wrist camera plate is between the camera and the point


def motor_record(motor: Mapping[str, int]) -> dict[str, int]:
    """The fields of one motor's LeRobot calibration that decide what its readings mean (`MOTOR_CALIBRATION_FIELDS`)."""
    return {k: int(motor[k]) for k in MOTOR_CALIBRATION_FIELDS}


def arm_record(motors: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    """`motor_record` of every motor of an arm, from `FelixClient.arm_calibration()[<arm>_arm]`."""
    return {name: motor_record(motor) for name, motor in motors.items()}


@dataclass(frozen=True)
class GripperMap:
    """How far an arm's `gripper.pos` opens the jaws (`calibration.gripper_step`).

    The opening is the moving jaw's angle about the hinge from where the tips meet:
    `deg_per_unit * (gripper.pos - pos_closed)`, negative below `pos_closed` (the servo squeezes closed jaws).
    `gripper.pos` is LeRobot's 0-100 over the gripper's calibrated tick range, so the map holds only under the
    gripper's LeRobot calibration it was measured with (`calibration`, a `motor_record` as the robot reported it):
    recalibrating the arm with `lerobot-calibrate` invalidates it (`XleRobotModel.check_robot`). `measurements` are
    the `(gripper.pos, gap m)` pairs it was fit to; `radius_m` the radius of the circle the moving tip turns on, where
    the gaps were measured (fitted with the map).
    """

    pos_closed: float
    deg_per_unit: float
    calibration: dict[str, int]
    measurements: tuple[tuple[float, float], ...] = ()
    radius_m: float | None = None

    def opening_deg(self, pos: float) -> float:
        """The jaws' opening angle (degrees) at `gripper.pos` `pos`."""
        return self.deg_per_unit * (pos - self.pos_closed)

    def pos_for(self, opening_deg: float) -> float:
        """The `gripper.pos` that opens the jaws `opening_deg` degrees; `opening_deg`'s inverse."""
        return self.pos_closed + opening_deg / self.deg_per_unit


BOARD_THICKNESS_M = 0.01  # the wrist camera's mount plate, with the board on it (assumed, not measured)
PLATE_MARGIN_M = 0.005  # a plate hides what passes this close to its box (its fit and the arm's error)
SIGHT_END_M = (
    0.005  # the line of sight stops this short of the point: a point on the plate's edge is not hidden
)


def segment_hits_box(a: np.ndarray, b: np.ndarray, pose: np.ndarray, half: np.ndarray) -> bool:
    """Whether the segment from `a` to `b` (root frame) passes through the box at `pose` (4x4) of half-size `half`."""
    rotation, center = pose[:3, :3], pose[:3, 3]
    start, direction = rotation.T @ (a - center), rotation.T @ (b - a)
    low, high = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if abs(start[axis]) > half[axis]:
                return False
            continue
        t1, t2 = sorted(
            ((-half[axis] - start[axis]) / direction[axis], (half[axis] - start[axis]) / direction[axis])
        )
        low, high = max(low, t1), min(high, t2)
        if low > high:
            return False
    return True


def wrist_board_link(arm: str) -> str:
    """The URDF link of `arm`'s wrist camera mount plate (`XleRobotModel.with_wrist_board`)."""
    return f"{arm}_wrist_camera_link"


@dataclass(frozen=True)
class WristBoard:
    """An arm's wrist camera mount plate, fixed to its gripper (`calibration.arm_shapes`): a thin box.

    `corners` are three corners of the plate's face the head camera sees (its back), in turn around it (the second
    between the other two), in `parent`'s frame (m); the fourth follows. The box is `thickness` deep (assumed),
    from that face toward `away` (unit, parent frame: the side away from the head camera, where the lens is). `rms_px`
    and `n_clicks` say what the corners were fitted to.
    """

    parent: str
    corners: tuple[tuple[float, float, float], ...]
    away: tuple[float, float, float]
    thickness: float = BOARD_THICKNESS_M
    rms_px: float | None = None
    n_clicks: int = 0

    def face_corners(self) -> np.ndarray:
        """The seen face's four corners in turn (4, 3), `parent`'s frame: the three fitted and the fourth."""
        c1, c2, c3 = (np.asarray(c, dtype=float) for c in self.corners)
        return np.array([c1, c2, c3, c1 + c3 - c2])

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """The box in `parent`'s frame: its pose (4x4) and its size (3,), m.

        The pose's origin is the box's center, its x and y run along the plate's edges, its z is the normal toward
        `away`.
        """
        c1, c2, c3 = self.face_corners()[:3]
        e1, e2 = c1 - c2, c3 - c2
        x = e1 / np.linalg.norm(e1)
        y = e2 - (e2 @ x) * x
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        if z @ np.asarray(self.away) < 0:  # keep it right-handed
            y, z = -y, -z
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([x, y, z])
        pose[:3, 3] = (c1 + c3) / 2 + z * self.thickness / 2
        return pose, np.array([np.linalg.norm(e1), np.linalg.norm(e2), self.thickness])


# an arm's parts, as boxes (`ArmBox`), in the order clicked; no wrist (the wrist roll servo): mostly hidden, and covered by
# its neighbours for the line of sight
BOX_PARTS = ("fixed_finger", "moving_finger", "gripper_body", "forearm")


@dataclass(frozen=True)
class ArmBox:
    """A part of an arm as a box (`calibration.arm_shapes`), square to its parent link's axes.

    `center` is in `parent`'s frame, `size` (m) along its x, y and z: six numbers, the part being bolted square to the
    link (the gripper's servo housing to the gripper link, the forearm along its link). With `moving`, the parent is
    the pinch frame and the box turns with the moving jaw: `center` is where it is with the jaws closed, and
    `XleRobotModel.box_frame` turns it by the opening (`moving_jaw`). `rms_px` and `n_clicks` say what it was fitted to
    (none for a stand-in).
    """

    parent: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    moving: bool = False
    rms_px: float | None = None
    n_clicks: int = 0


class RobotMismatchError(RuntimeError):
    """A calibration's copy of the robot's settings no longer matches the robot (`XleRobotModel.check_robot`)."""


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
        gripper_maps: Mapping[str, GripperMap] | None = None,
        arm_calibration: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None,
        wrist_boards: Mapping[str, WristBoard] | None = None,
        boxes: Mapping[str, Mapping[str, ArmBox]] | None = None,
    ):
        """Build the model from URDF text (`from_urdf` reads a file).

        `head_signs` / `head_refs`: per head servo, the direction and the tick count at its recorded home (from
        `setup.head`); the URDF angle is `sign * (ticks - ref) * 2 pi / 4096`. `intrinsics`: the color camera's
        `fx, fy, ppx, ppy, width, height`. `parameters`: the rows to calibrate (default `default_parameters()`).
        `floor_z`: the table's height in the root frame, if known. `visible_side`: per arm, the side of the
        gripper the camera must see ("positive" by default: the wrist camera is on the negative side).
        `gripper_maps`: per arm, how `gripper.pos` opens the jaws, where measured. `arm_calibration`: per arm, the
        LeRobot calibration (`arm_record`) its clicks were taken under, where known: the arm's estimates hold only
        under it. `wrist_boards`: per arm, its wrist camera's mount plate, where fitted (the URDF also carries it as a
        link: `with_wrist_board`). `boxes`: per arm, its parts as boxes (`BOX_PARTS`), where fitted.
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
        self.visible_side = dict.fromkeys(ARMS, "positive") | dict(visible_side or {})
        self.gripper_maps = dict(gripper_maps or {})
        self.arm_calibration = {arm: arm_record(motors) for arm, motors in (arm_calibration or {}).items()}
        self.wrist_boards = dict(wrist_boards or {})
        self.boxes = {arm: dict(parts) for arm, parts in (boxes or {}).items()}
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
        self._into = into
        self._ancestor_cache: dict[str, set[str]] = {}
        self._unit_axes = {
            name: j.axis / np.linalg.norm(j.axis) for name, j in self._joints.items() if j.moves
        }

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
    ) -> tuple[dict[str, "pin.SE3"], dict[str, np.ndarray], np.ndarray]:
        """The corrections of `theta` as the kinematics use them.

        Returns the corrected origin of every joint with a `pos:` / `rot:` correction, the rotation vector of each
        `rot:` correction, and the zero offsets as a q vector.
        """
        shifts: dict[str, np.ndarray] = {}
        rotations: dict[str, np.ndarray] = {}
        zero = np.zeros(self.nominal_model.nq)
        for p, value in zip(self.parameters, self._vector(theta), strict=True):
            if p.kind == "zero":
                zero[self.q_index(p.joint)] += value
            else:
                target = shifts if p.kind == "pos" else rotations
                target.setdefault(p.joint, np.zeros(3))[p.axis] += value
        origins = {}
        for name in shifts.keys() | rotations.keys():
            nominal = self._joints[name].origin
            rotation = nominal.rotation @ pin.exp3(rotations.get(name, np.zeros(3)))
            origins[name] = pin.SE3(rotation, nominal.translation + shifts.get(name, np.zeros(3)))
        return origins, rotations, zero

    def _apply(self, theta: np.ndarray | Mapping[str, float] | None) -> None:
        """Rebuild the working model's placements from the nominal origins and `theta` (never from earlier ones)."""
        origins, self._rotation_vectors, self._zero = self._corrected_origins(theta)
        self._origins = origins
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
        self._forward(q)

    def _forward(self, q: np.ndarray) -> None:
        """Forward kinematics at measured `q` with the applied estimates (call `_apply` first)."""
        self._q_effective = np.asarray(q, dtype=float) + self._zero
        pin.framesForwardKinematics(self.pin_model, self._data, self._q_effective)

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
        """Pinocchio `q` (measured angles) from servo readings.

        `arms` maps "left" / "right" to LeRobot's `{"shoulder_pan.pos": deg, ...}` (the SO-101 URDF's angles one to
        one), `head_ticks` the head servos' raw ticks. Joints not given are at 0 (the head at its recorded home); the
        gripper stays at 0, which does not move the pinch point.
        """
        q = pin.neutral(self.nominal_model)
        angles = self.head_angles(head_ticks) if head_ticks is not None else {}
        for arm, pose in (arms or {}).items():
            angles |= {
                arm_joint(arm, j): float(np.radians(pose[f"{j}.pos"]))
                for j in SO101_JOINTS
                if f"{j}.pos" in pose
            }
        for name, angle in angles.items():
            q[self.q_index(name)] = angle
        return q

    def joint_limits(self, arm: str) -> dict[str, tuple[float, float]]:
        """`{joint: (lower, upper)}` in radians for `arm`'s SO-101 joints, from the URDF."""
        return {j: self.limits(arm_joint(arm, j)) for j in SO101_JOINTS}

    def limits(self, joint: str) -> tuple[float, float]:
        """(lower, upper) position limits of a URDF joint (radians / meters), from the URDF."""
        i = self.q_index(joint)
        return float(self.nominal_model.lowerPositionLimit[i]), float(
            self.nominal_model.upperPositionLimit[i]
        )

    def q_index(self, joint: str) -> int:
        """Where a one-dimensional URDF joint's position sits in `q`."""
        return int(self.nominal_model.joints[self.nominal_model.getJointId(joint)].idx_q)

    def head_ticks(self, angles: Mapping[str, float]) -> dict[str, int]:
        """Raw servo ticks for URDF head angles: `head_angles`'s inverse, rounded to whole ticks.

        The angles are measured ones (radians, before the zero corrections).
        """
        return {
            servo: int(
                round(
                    self.head_refs[servo]
                    + angles[joint] * TICKS_PER_TURN / (2 * np.pi) / self.head_signs[servo]
                )
            )
            for servo, joint in HEAD_JOINTS.items()
        }

    # ---- geometry and measurement ----------------------------------------------------------------------------

    def frame_pose(
        self, frame: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """X_root_frame (4x4) of a link's frame."""
        self._evaluate(q, theta)
        return self._pose(frame)

    def frame_jacobian(
        self,
        frame: str,
        q: np.ndarray,
        joints: Sequence[str],
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """X_root_frame (4x4) and its Jacobian (6, len(joints)) w.r.t. the angles of `joints`.

        Rows 0-2: the velocity of the frame's origin, rows 3-5: its angular velocity, both in the root frame, per
        unit change of each joint (zero for a joint that does not move the frame).
        """
        self._evaluate(q, theta)
        pose = self._pose(frame)
        linear, angular = self._twists([("zero", joint, None) for joint in joints])
        chain = self._ancestors(frame)
        moves = np.array([[joint in chain] for joint in joints], dtype=float)
        point = (linear + np.cross(angular, pose[:3, 3])) * moves
        return pose, np.vstack([point.T, (angular * moves).T])

    def camera_pose(self, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None) -> np.ndarray:
        """X_root_optical (4x4): the color camera's optical frame (x right, y down, z forward)."""
        return self.frame_pose(OPTICAL_FRAME, q, theta)

    def point(
        self,
        arm: str,
        q: np.ndarray,
        point: str = "pinch",
        opening_deg: float | None = None,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Root-frame position (m) of one of `arm`'s gripper points (`POINTS`); `opening_deg` for the moving tip."""
        self._evaluate(q, theta)
        return self._point(arm, point, opening_deg)

    def pinch_point(
        self, arm: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Root-frame position (m) of `arm`'s estimated pinch point: the fixed jaw's tip, where the tips meet."""
        return self.point(arm, q, "pinch", theta=theta)

    def moving_tip(
        self,
        arm: str,
        q: np.ndarray,
        opening_deg: float,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Root-frame position (m) of `arm`'s moving jaw's tip, with the jaws `opening_deg` degrees open."""
        return self.point(arm, q, "moving_tip", opening_deg, theta)

    def opening_deg(self, arm: str, gripper_pos: float) -> float:
        """The jaws' opening (degrees from closed) at LeRobot's `gripper.pos`, from the arm's `GripperMap`."""
        if arm not in self.gripper_maps:
            raise ValueError(f"no gripper map for the {arm} arm: measure it with calibration.gripper_step")
        return self.gripper_maps[arm].opening_deg(gripper_pos)

    def opening_for_width(
        self, arm: str, width_m: float, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> float:
        """The jaws' opening (degrees) that puts the moving tip `width_m` from the pinch point (a chord of its circle)."""
        radius = self.jaw_radius(arm, theta)
        if not 0.0 <= width_m <= 2 * radius:
            raise ValueError(
                f"the {arm} jaws cannot be {width_m * 100:.1f} cm apart (at most {2 * radius * 100:.1f} cm)"
            )
        return float(np.degrees(2 * np.arcsin(width_m / (2 * radius))))

    def moving_jaw(
        self, arm: str, opening_deg: float, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """X_pinch (4x4): where a point of the fixed jaw (pinch frame) is on the moving jaw, `opening_deg` open.

        The turn about the hinge axis, in the pinch frame: it takes the pinch point to the moving tip. Both jaws are
        fixed to the gripper body, so it does not depend on the joints.
        """
        self._evaluate(self.joint_vector(), theta)
        pinch = self._pose(pinch_frame(arm))
        turn = np.eye(4)
        turn[:3, :3] = self._moving_turn(arm, opening_deg)
        origin = self._hinge(arm)[0]
        turn[:3, 3] = origin - turn[:3, :3] @ origin
        return np.linalg.inv(pinch) @ turn @ pinch

    def grasp_offset(
        self,
        arm: str,
        width_m: float,
        margin_m: float = 0.0,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Where the center of an object `width_m` wide is, in the pinch frame, when grasped.

        Midway between the tips, with the moving one `width_m` from the pinch point (the origin): where it is held.
        With `margin_m`, the point `margin_m` farther from the fixed tip, toward the moving one: where to aim so the
        fixed jaw clears the object by that much, the moving jaw then sweeping the object onto it as it closes.
        """
        tip = self.moving_jaw(arm, self.opening_for_width(arm, width_m, theta), theta)[:3, 3]
        if not margin_m:
            return tip / 2
        # the direction toward the moving tip (for no width, that of a tip just apart)
        toward = (
            tip if width_m else self.moving_jaw(arm, self.opening_for_width(arm, 1e-3, theta), theta)[:3, 3]
        )
        return toward / np.linalg.norm(toward) * (width_m / 2 + margin_m)

    def gripper_pos_for(self, arm: str, opening_deg: float) -> float:
        """The `gripper.pos` that opens `arm`'s jaws `opening_deg` degrees (its `GripperMap`; `opening_deg`'s inverse)."""
        if arm not in self.gripper_maps:
            raise ValueError(f"no gripper map for the {arm} arm: measure it with calibration.gripper_step")
        return self.gripper_maps[arm].pos_for(opening_deg)

    def _point(self, arm: str, point: str, opening_deg: float | None) -> np.ndarray:
        """`point` at the evaluated configuration."""
        pinch = self._pose(pinch_frame(arm))[:3, 3]
        if point == "pinch":
            return pinch
        if point != "moving_tip":
            raise ValueError(f"unknown gripper point {point!r}; one of {POINTS}")
        origin = self._hinge(arm)[0]
        return origin + self._moving_turn(arm, opening_deg) @ (pinch - origin)

    def _hinge(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """A point on `arm`'s gripper hinge axis and its unit direction (positive opens), as now evaluated.

        Neither depends on the gripper joint's own angle.
        """
        hinge = self._joints[arm_joint(arm, "gripper")]
        frame = self._pose(hinge.child)  # its origin is on the axis
        return frame[:3, 3], frame[:3, :3] @ (hinge.axis / np.linalg.norm(hinge.axis))

    def _moving_turn(self, arm: str, opening_deg: float | None) -> np.ndarray:
        """The rotation (3x3, root frame) that takes the pinch point to the moving tip, `opening_deg` open."""
        if opening_deg is None:
            raise ValueError("the moving tip needs the jaws' opening (opening_deg)")
        return pin.exp3(self._hinge(arm)[1] * np.radians(opening_deg))

    def project(
        self, points: np.ndarray, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of root-frame points (N, 3); NaN pixels at or behind the camera."""
        self._evaluate(q, theta)
        return self._project(np.atleast_2d(points))

    def predict(
        self,
        arm: str,
        q: np.ndarray,
        theta: np.ndarray | Mapping[str, float] | None = None,
        point: str = "pinch",
        opening_deg: float | None = None,
    ) -> Prediction:
        """Where one of `arm`'s gripper points lands in the image, where it is, and whether the camera can see it.

        The point is the pinch point by default; `opening_deg` places the moving tip.
        """
        self._evaluate(q, theta)
        xyz = self._point(arm, point, opening_deg)
        uv, z = self._project(xyz[None])
        x_root_camera = self._pose(OPTICAL_FRAME)
        p_camera = x_root_camera[:3, :3].T @ (xyz - x_root_camera[:3, 3])
        return Prediction(uv[0], float(z[0]), p_camera, xyz, self._visibility(arm, xyz), self._hidden_by(xyz))

    def predict_many(
        self,
        arm: str,
        qs: Sequence[np.ndarray],
        theta: np.ndarray | Mapping[str, float] | None = None,
        point: str = "pinch",
        opening_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of a gripper point at each of `qs`, applying `theta` once (for fits)."""
        self._apply(theta)
        uv, z = np.full((len(qs), 2), np.nan), np.full(len(qs), np.nan)
        for i, q in enumerate(qs):
            self._forward(q)
            uv_i, z_i = self._project(self._point(arm, point, opening_deg)[None])
            uv[i], z[i] = uv_i[0], z_i[0]
        return uv, z

    # ---- derivatives -------------------------------------------------------------------------------------------

    def pixel_jacobian(
        self,
        arm: str,
        q: np.ndarray,
        theta: np.ndarray | Mapping[str, float] | None = None,
        joints: Sequence[str] = (),
        point: str = "pinch",
        opening_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derivatives of the pixel of one of `arm`'s gripper points (default the pinch point).

        (2, P) w.r.t. the estimates and (2, len(joints)) w.r.t. the measured angles of `joints` (URDF names, e.g.
        `head_pan_joint`, `left_elbow_flex`). Zeros if the pinch point is at or behind the camera.

        Analytic: every parameter and joint moves everything below its URDF joint by a twist (a translation for
        `pos:` and prismatic joints, a rotation about an axis through the joint's origin for `rot:`, `zero:` and
        revolute joints); the pixel changes with the point's motion relative to the camera's. The moving tip moves
        with the gripper body like the pinch point, except under the pinch point's own offset (`pos:` / `rot:` of
        `pinch_joint`), which moves it turned by the opening.
        """
        self._evaluate(q, theta)
        return self._jacobians(arm, joints, point, opening_deg)

    def pixel_jacobian_many(
        self,
        arm: str,
        qs: Sequence[np.ndarray],
        theta: np.ndarray | Mapping[str, float] | None = None,
        point: str = "pinch",
        opening_deg: float | None = None,
    ) -> np.ndarray:
        """(N, 2, P): `pixel_jacobian` w.r.t. the estimates at each of `qs`, applying `theta` once (for fits)."""
        self._apply(theta)
        out = np.zeros((len(qs), 2, len(self.parameters)))
        for i, q in enumerate(qs):
            self._forward(q)
            out[i] = self._jacobians(arm, (), point, opening_deg)[0]
        return out

    def _jacobians(
        self, arm: str, joints: Sequence[str], point: str = "pinch", opening_deg: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`pixel_jacobian` at the evaluated configuration."""
        x_root_camera = self._pose(OPTICAL_FRAME)
        rotation, center = x_root_camera[:3, :3], x_root_camera[:3, 3]
        tip = self._point(arm, point, opening_deg)
        x, y, z = rotation.T @ (tip - center)
        d_parameters, d_joints = np.zeros((2, len(self.parameters))), np.zeros((2, len(joints)))
        if z <= 1e-6:
            return d_parameters, d_joints
        c = self.intrinsics
        projection = np.array(
            [[c["fx"] / z, 0.0, -c["fx"] * x / z**2], [0.0, c["fy"] / z, -c["fy"] * y / z**2]]
        )
        coordinates = [(p.kind, p.joint, p.axis) for p in self.parameters] + [
            ("zero", j, None) for j in joints
        ]
        linear, angular = self._twists(coordinates)
        tip_chain, camera_chain = self._ancestors(pinch_frame(arm)), self._ancestors(OPTICAL_FRAME)
        moves_tip = np.array([[joint in tip_chain] for _, joint, _ in coordinates], dtype=float)
        moves_camera = np.array([[joint in camera_chain] for _, joint, _ in coordinates], dtype=float)
        d_tip = (linear + np.cross(angular, tip)) * moves_tip
        if point == "moving_tip":  # the pinch point's own offset moves it turned, not rigidly
            hinge_chain = self._ancestors(self._joints[arm_joint(arm, "gripper")].child)
            offset = np.array([[joint in tip_chain - hinge_chain] for _, joint, _ in coordinates])
            pinch = self._pose(pinch_frame(arm))[:3, 3]
            turned = (linear + np.cross(angular, pinch)) @ self._moving_turn(arm, opening_deg).T
            d_tip = np.where(offset, turned, d_tip)
        d_center = (linear + np.cross(angular, center)) * moves_camera
        d_camera_frame = (
            d_tip - d_center - np.cross(angular * moves_camera, tip - center)
        )  # rows: coordinates
        columns = projection @ rotation.T @ d_camera_frame.T
        return columns[:, : len(self.parameters)], columns[:, len(self.parameters) :]

    def _twists(self, coordinates: Sequence[tuple[str, str, int | None]]) -> tuple[np.ndarray, np.ndarray]:
        """The twist of each coordinate's subtree per unit change, in the root frame.

        `coordinates` are (kind, joint, axis); kinds: `zero` (the joint's angle), or axis `axis` of `pos` / `rot`.
        Returns rows of linear velocity at the root's origin and rows of angular velocity.
        """
        linear, angular = np.zeros((len(coordinates), 3)), np.zeros((len(coordinates), 3))
        points = np.zeros((len(coordinates), 3))  # a point on each rotation's axis
        origin_poses: dict[str, np.ndarray] = {}
        for i, (kind, joint, axis) in enumerate(coordinates):
            if joint not in origin_poses:  # the joint's frame after its origin, before its own motion
                origin_poses[joint] = self._origin_pose(joint)
            rotation, points[i] = origin_poses[joint][:3, :3], origin_poses[joint][:3, 3]
            urdf_joint = self._joints[joint]
            if kind == "zero":
                direction = rotation @ self._unit_axes[joint]
                if urdf_joint.type == "prismatic":
                    linear[i] = direction
                else:
                    angular[i] = direction
            elif kind == "pos":  # along the parent frame's axis
                origin = self._origins.get(joint, urdf_joint.origin)
                linear[i] = rotation @ origin.rotation.T[:, axis]
            else:  # R_origin = R_urdf exp(w): the right Jacobian of exp
                w = self._rotation_vectors.get(joint, np.zeros(3))
                angular[i] = rotation @ pin.Jexp3(w)[:, axis]
        return linear - np.cross(angular, points), angular  # a rotation about an axis through `points`

    def _origin_pose(self, joint: str) -> np.ndarray:
        """X_root of `joint`'s frame after its (corrected) origin and before its own motion."""
        urdf_joint = self._joints[joint]
        if not urdf_joint.moves:
            return self._pose(urdf_joint.child)
        joint_id = self.pin_model.getJointId(joint)
        angle = self._q_effective[self.q_index(joint)]
        axis = self._unit_axes[joint]
        undo = (
            pin.SE3(np.eye(3), -axis * angle)
            if urdf_joint.type == "prismatic"
            else pin.SE3(pin.exp3(-axis * angle), np.zeros(3))
        )
        return (self._data.oMi[joint_id] * undo).homogeneous

    def _ancestors(self, link: str) -> set[str]:
        """Names of the URDF joints between the root and `link`."""
        if link not in self._ancestor_cache:
            out = set()
            name = link
            while name in self._into:
                out.add(self._into[name].name)
                name = self._into[name].parent
            self._ancestor_cache[link] = out
        return self._ancestor_cache[link]

    def visibility(
        self,
        arm: str,
        q: np.ndarray,
        theta: np.ndarray | Mapping[str, float] | None = None,
        point: str = "pinch",
        opening_deg: float | None = None,
    ) -> float:
        """Signed cosine between `arm`'s gripper hinge axis and the line of sight from a gripper point to the camera.

        Near +1 / -1 the camera sees the jaws from the positive / negative side; near 0 it looks along the plane the
        jaws open in, and the point hides behind them.
        """
        self._evaluate(q, theta)
        return self._visibility(arm, self._point(arm, point, opening_deg))

    def hidden_by(
        self,
        arm: str,
        q: np.ndarray,
        point: str = "pinch",
        opening_deg: float | None = None,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> tuple[str, ...]:
        """The arms whose wrist camera plate (`wrist_boards`) is between the camera and one of `arm`'s gripper points.

        Each plate is its box, `PLATE_MARGIN_M` larger; an arm without a fitted plate hides nothing.
        """
        self._evaluate(q, theta)
        return self._hidden_by(self._point(arm, point, opening_deg))

    def view_of(
        self, arm: str, q: np.ndarray, xyz: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> tuple[float, tuple[str, ...]]:
        """How the camera sees a root-frame point held by `arm`'s jaws (a grasp point).

        Returns the jaws' `visibility` there and the arms whose plate hides it (as `hidden_by`).
        """
        self._evaluate(q, theta)
        xyz = np.asarray(xyz, dtype=float)
        return self._visibility(arm, xyz), self._hidden_by(xyz)

    def _hidden_by(self, xyz: np.ndarray) -> tuple[str, ...]:
        camera = self._pose(OPTICAL_FRAME)[:3, 3]
        sight = xyz - camera
        end = camera + sight * max(0.0, 1.0 - SIGHT_END_M / max(float(np.linalg.norm(sight)), 1e-9))
        hits = []
        for arm, board in self.wrist_boards.items():
            link = wrist_board_link(arm)
            if link not in self._frame_ids:
                continue
            _, size = board.box()
            if segment_hits_box(camera, end, self._pose(link), size / 2 + PLATE_MARGIN_M):
                hits.append(arm)
        return tuple(hits)

    def _visibility(self, arm: str, xyz: np.ndarray) -> float:
        sight = self._pose(OPTICAL_FRAME)[:3, 3] - xyz
        return float(self._hinge(arm)[1] @ sight) / float(np.linalg.norm(sight))

    def jaw_radius(self, arm: str, theta: np.ndarray | Mapping[str, float] | None = None) -> float:
        """Distance (m) from `arm`'s pinch point to its gripper's hinge axis: the circle the moving jaw's tip turns on.

        Both are fixed to the gripper body, so it does not depend on the joints.
        """
        self._evaluate(self.joint_vector(), theta)
        origin, axis = self._hinge(arm)
        return float(np.linalg.norm(np.cross(self._pose(pinch_frame(arm))[:3, 3] - origin, axis)))

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

    # ---- the robot it was made for -----------------------------------------------------------------------

    def check_robot(self, client: Any) -> None:
        """Raise `RobotMismatchError` unless the robot behind `client` still has the settings this model copied.

        A calibration is made under settings that live on the robot and change only there: the head servos'
        directions (`head_signs`, from `setup.head`), which turn head readings into angles; each arm's LeRobot
        calibration (`arm_calibration`), which turns servo ticks into the joint angles the estimates correct; and the
        gripper's, for each gripper map. After any of them changes, the model's geometry is silently wrong. The head's
        recorded home is not among them: the model turns raw head ticks into angles with its own references
        (`head_refs`) and fitted zeros, so re-recording the home (`setup.head_range`) only changes where the head
        rests. Parts the robot does not report (no head servos, an arm that is not connected) are not checked; nor are
        arms the model has no snapshot of (calibrations made before snapshots), which is logged.
        """
        problems = []
        head = client.get_state().get("head") or {}
        for servo in HEAD_JOINTS:
            if servo in head and head[servo]["sign"] != self.head_signs[servo]:
                problems.append(
                    f"{servo}: sign {head[servo]['sign']} on the robot, {self.head_signs[servo]} in the calibration "
                    "(setup.head redone? recalibrate the camera, docs/README.md section 6)"
                )
        robot = {arm_id.removesuffix("_arm"): arm_record(m) for arm_id, m in client.arm_calibration().items()}
        for arm, motors in robot.items():
            if arm in self.arm_calibration:
                problems += _record_differences(
                    f"{arm} arm",
                    self.arm_calibration[arm],
                    motors,
                    "recalibrate the camera, docs/README.md section 6",
                )
            elif any(p.arm == arm and v for p, v in zip(self.parameters, self._estimates, strict=True)):
                log.warning(
                    "the calibration has no snapshot of the %s arm's LeRobot calibration: not checked", arm
                )
            if arm in self.gripper_maps:
                problems += _record_differences(
                    f"{arm} gripper map",
                    {"gripper": self.gripper_maps[arm].calibration},
                    motors,
                    "rerun calibration.gripper_step, docs/README.md section 7",
                )
        if problems:
            raise RobotMismatchError("the calibration does not match the robot:\n  " + "\n  ".join(problems))

    # ---- export and persistence -----------------------------------------------------------------------------

    def to_urdf(self, theta: np.ndarray | Mapping[str, float] | None = None) -> str:
        """The URDF with `theta` (default: the current estimates) written into the joint origins.

        Zero offsets become a fixed rotation about the joint's axis folded into its origin, so a model of the
        result with no corrections gives the same poses for the same measured `q`. Joint limits are untouched:
        they bound the measured position, which does not change.
        """
        origins, _, _ = self._corrected_origins(theta)
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

    def with_wrist_board(self, arm: str, board: WristBoard) -> "XleRobotModel":
        """A copy of the model with `arm`'s wrist camera plate, replacing an earlier one.

        The plate is kept in `wrist_boards`, and in the URDF as a link (`wrist_board_link`, a box) on a fixed joint
        to `board.parent`. The joint is not calibrated, so the estimates carry over as they are.
        """
        link, joint = wrist_board_link(arm), f"{arm}_wrist_camera_joint"
        root = ET.fromstring(self.urdf)
        for element in [*root.findall("link"), *root.findall("joint")]:
            if element.get("name") in (link, joint):
                root.remove(element)
        pose, size = board.box()
        rpy = _format(pin.rpy.matrixToRpy(pose[:3, :3]))
        link_element = ET.SubElement(root, "link", {"name": link})
        visual = ET.SubElement(link_element, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(ET.SubElement(visual, "geometry"), "box", {"size": _format(size)})
        joint_element = ET.SubElement(root, "joint", {"name": joint, "type": "fixed"})
        ET.SubElement(joint_element, "origin", {"xyz": _format(pose[:3, 3]), "rpy": rpy})
        ET.SubElement(joint_element, "parent", {"link": board.parent})
        ET.SubElement(joint_element, "child", {"link": link})
        model = type(self)(
            '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n",
            self.head_signs,
            self.head_refs,
            self.intrinsics,
            self.parameters,
            self.floor_z,
            self.visible_side,
            self.gripper_maps,
            self.arm_calibration,
            self.wrist_boards | {arm: board},
            self.boxes,
        )
        model.set_estimates(self._estimates.copy())
        model.covariance = self.covariance
        model.fit_info = dict(self.fit_info)
        return model

    def without_wrist_board(self, arm: str) -> "XleRobotModel":
        """A copy of the model with no wrist camera plate on `arm` (taken off the robot).

        Its link and joint go from the URDF and from `wrist_boards`. The same model if it has none.
        """
        if arm not in self.wrist_boards:
            return self
        link, joint = wrist_board_link(arm), f"{arm}_wrist_camera_joint"
        root = ET.fromstring(self.urdf)
        for element in [*root.findall("link"), *root.findall("joint")]:
            if element.get("name") in (link, joint):
                root.remove(element)
        model = type(self)(
            '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n",
            self.head_signs,
            self.head_refs,
            self.intrinsics,
            self.parameters,
            self.floor_z,
            self.visible_side,
            self.gripper_maps,
            self.arm_calibration,
            {other: board for other, board in self.wrist_boards.items() if other != arm},
            self.boxes,
        )
        model.set_estimates(self._estimates.copy())
        model.covariance = self.covariance
        model.fit_info = dict(self.fit_info)
        return model

    def board_corners(
        self, arm: str, q: np.ndarray, theta: np.ndarray | Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Root-frame positions (4, 3) of the four corners of `arm`'s wrist camera plate's seen face, in turn."""
        if arm not in self.wrist_boards:
            raise ValueError(f"no wrist camera plate for the {arm} arm: run calibration.arm_shapes")
        board = self.wrist_boards[arm]
        frame = self.frame_pose(board.parent, q, theta)
        return board.face_corners() @ frame[:3, :3].T + frame[:3, 3]

    def with_boxes(self, arm: str, parts: Mapping[str, ArmBox]) -> "XleRobotModel":
        """A copy of the model with `arm`'s boxes `parts` added, replacing earlier ones of the same parts.

        They live in `boxes` only (saved in `<felix_calibration>`): each is fixed to a link the URDF has.
        """
        unknown = set(parts) - set(BOX_PARTS)
        if unknown:
            raise ValueError(f"unknown parts {sorted(unknown)}: one of {BOX_PARTS}")
        model = type(self)(
            self.urdf,
            self.head_signs,
            self.head_refs,
            self.intrinsics,
            self.parameters,
            self.floor_z,
            self.visible_side,
            self.gripper_maps,
            self.arm_calibration,
            self.wrist_boards,
            self.boxes | {arm: self.boxes.get(arm, {}) | dict(parts)},
        )
        model.set_estimates(self._estimates.copy())
        model.covariance = self.covariance
        model.fit_info = dict(self.fit_info)
        return model

    def box_frame(
        self,
        box: ArmBox,
        arm: str,
        q: np.ndarray,
        opening_deg: float = 0.0,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """root_from_box (4x4) of one of `arm`'s boxes: its center, its axes its parent's.

        The parent's pose; for the moving finger, the pinch frame turned by the jaws' opening (`opening_deg`, 0 closed).
        """
        frame = self.frame_pose(box.parent, q, theta)
        if box.moving:
            frame = frame @ self.moving_jaw(arm, max(opening_deg, 0.0), theta)
        out = frame.copy()
        out[:3, 3] = frame[:3, :3] @ np.asarray(box.center) + frame[:3, 3]
        return out

    def boxes_root(
        self,
        arm: str,
        q: np.ndarray,
        opening_deg: float = 0.0,
        theta: np.ndarray | Mapping[str, float] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """`arm`'s boxes at `q` (the moving finger at `opening_deg`): part -> (root_from_box, half its size (3,))."""
        return {
            part: (self.box_frame(box, arm, q, opening_deg, theta), 0.5 * np.asarray(box.size, dtype=float))
            for part, box in self.boxes.get(arm, {}).items()
        }

    def save(self, path: str | Path) -> None:
        """Write the calibrated URDF: `to_urdf()` plus a `<felix_calibration>` element holding everything else.

        Any URDF reader (pinocchio, placo, a viewer) uses the file as it is and ignores the element. `load` reads
        the element back: the servo map, intrinsics, visible sides, floor, the gripper maps, the arms' LeRobot
        calibrations, the parameter table, the estimates, their covariance, `fit_info`, and the URDF's original origins of every joint a parameter corrects,
        so the nominal model, and with it the priors, come back exactly.
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
            "gripper_maps": {arm: asdict(m) for arm, m in self.gripper_maps.items()},
            "arm_calibration": self.arm_calibration,
            "wrist_boards": {arm: asdict(b) for arm, b in self.wrist_boards.items()},
            "boxes": {
                arm: {part: asdict(b) for part, b in parts.items()} for arm, parts in self.boxes.items()
            },
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
            {  # a calibration may have none (none measured yet)
                arm: GripperMap(**m | {"measurements": tuple(map(tuple, m["measurements"]))})
                for arm, m in data.get("gripper_maps", {}).items()
            },
            data.get("arm_calibration", {}),  # none in calibrations made before snapshots
            {  # none before the wrist camera boards were fitted
                arm: WristBoard(**b | {"corners": tuple(map(tuple, b["corners"])), "away": tuple(b["away"])})
                for arm, b in data.get("wrist_boards", {}).items()
            },
            {  # none before the arms' shapes were fitted
                arm: {
                    part: ArmBox(**b | {"center": tuple(b["center"]), "size": tuple(b["size"])})
                    for part, b in parts.items()
                }
                for arm, parts in data.get("boxes", {}).items()
            },
        )
        if data.get("capsules"):  # the arms' shapes as capsules, before they were boxes (2026-11-06)
            log.warning(
                "%s: its capsules are left out (the arms' shapes are boxes now): fit them again", path
            )
        model.set_estimates(np.array(data["estimates"], dtype=float))
        if data["covariance"] is not None:
            model.covariance = np.array(data["covariance"], dtype=float)
        model.fit_info = data["fit_info"]
        return model


def _record_differences(
    what: str, ours: Mapping[str, Mapping[str, int]], robot: Mapping[str, Mapping[str, int]], fix: str
) -> list[str]:
    """One line per motor whose LeRobot calibration on the robot differs from `ours` (the fields that differ)."""
    lines = []
    for motor, fields in ours.items():
        if motor not in robot:
            lines.append(f"{what}: no {motor} on the robot ({fix})")
            continue
        changed = [f"{k} {robot[motor][k]} (was {v})" for k, v in fields.items() if robot[motor][k] != v]
        if changed:
            lines.append(f"{what}: {motor} recalibrated on the robot: {', '.join(changed)} ({fix})")
    return lines


def _floats(
    element: ET.Element | None, key: str, default: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    value = None if element is None else element.get(key)
    if value is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()], dtype=float)


def _format(values: np.ndarray) -> str:
    return " ".join(f"{v:.15g}" for v in values)
