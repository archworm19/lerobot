"""The measurement model for camera-to-robot calibration: where does the closed gripper's tip land in the image?

Given the measured joint positions of the head and one arm, plus a set of kinematic parameters (the "fudge
factors": joint zero offsets, link offsets, the camera mount), `CalibrationModel.predict` gives the pixel of the tip
in the head camera's color image. Calibration (`fit.py`) is finding the parameters that make these predictions
agree with where a person clicked the tip.

**Two URDFs.** The head and the camera come from `xlerobot.urdf`. The arm comes from the official SO-101 URDF
(`so101_new_calib.urdf`), not from the arm inside `xlerobot.urdf`: that one is an older gripper (its wrist roll joint
lacks the SO-101's 18 mm lateral offset, and its shoulder offsets differ by about 13 mm), while the hardware is an
SO-101, and the SO-101 URDF's joint angles are LeRobot's calibrated degrees one to one (the teleop relies on this).
The SO-101 chain is mounted on the XLeRobot's `Base` frame with a constant transform `T` (fitted from matching the
two URDFs' shoulder, elbow and wrist pivots), and the mount is then calibrated like any other parameter.

**Parameters** are a dict `{name: value}` (radians / meters; missing = 0). Names are the `UrdfTree` perturbation
keys (`zero:<joint>`, `pos:<joint>:x|y|z`, `rot:<joint>:roll|pitch|yaw`); keys of the SO-101 arm chain carry the
prefix `so101:`. `parameter_set` lists the ones we fit, each with a prior sigma (how wrong we think the URDF value
could be), which is what keeps a fit with few clicks sensible.

**Head angles.** The head servos report raw ticks. The URDF joint angle is `sign * (ticks - ref) * 2pi / 4096`,
where `sign` is the direction saved by `setup.head` (+1: the servo's positive direction is down / left, as in the
URDF) and `ref` is the tick count at the head's recorded home. The true angle at `ref` is unknown (home is not
"level"), so `zero:head_pan_joint` and `zero:head_tilt_joint` are parameters with wide priors.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.urdf_kinematics import PinholeCamera, UrdfTree, invert, pose

FELIX_DIR = Path(__file__).resolve().parent.parent
XLEROBOT_URDF = FELIX_DIR / "xlerobot.urdf"
SO101_URDF = FELIX_DIR / "so101_new_calib.urdf"

ARM_PREFIX = "so101:"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TIP_FRAME, TIP_JOINT = "gripper_frame_link", "gripper_frame_joint"  # the SO-101's tool center point
HEAD_JOINTS = {"head_neck": "head_pan_joint", "head_eyes": "head_tilt_joint"}  # servo -> URDF joint
CAMERA_LINK, CAMERA_MOUNT = "head_camera_link", "fixed_head_camera_link"
TICKS_PER_TURN = 4096

# The arm inside xlerobot.urdf, used only to place the SO-101 chain on its Base frame.
XLE_ARMS = {
    "left": {"mount": "fixed_Base", "joints": ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L"]},
    "right": {"mount": "fixed_Base_2", "joints": ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R"]},
}
# SO-101 joint q <-> xlerobot.urdf joint: q_xle = sign * q_so101 + offset. Signs and the lift / elbow / wrist-flex
# offsets were found by matching the pivots' positions over random poses (the residual that remains, about 8-14 mm,
# is the shoulder geometry difference). The pan offset is the heading: LeRobot's pan = 0 points the arm forward,
# while the XLeRobot URDF's pan = 0 points it sideways (left arm: 90 deg off, right arm: -90 deg). Checked against
# the wiggle test's camera frames: under it the arms' wrist and gripper land below the frame at the left / right,
# where the arms appear, and a +10 deg pan moves them to the right in the image, as seen.
ALIGN_SIGNS = (1.0, -1.0, 1.0, 1.0)
ALIGN_PAN_OFFSET = {"left": np.pi / 2, "right": -np.pi / 2}
ALIGN_OFFSETS = (np.nan, np.pi / 2, np.pi / 2, 0.0)  # the pan entry is ALIGN_PAN_OFFSET[arm]

# 1-sigma prior on each kind of parameter: how far the URDF value might be from the real robot. Radians / meters.
SIGMAS = {
    "head zero": np.radians(20.0),  # the head's home is not level
    "head origin": 0.002,
    "camera x": 0.003,
    "camera y": 0.015,  # the RGB sensor's lateral offset in the camera link is the least certain
    "camera z": 0.003,
    "camera rot": np.radians(2.0),
    "mount pos": 0.015,  # the SO-101 chain placed on the XLeRobot's Base (see above)
    "mount rot": np.radians(5.0),
    "arm zero": np.radians(2.0),
    "arm origin": 0.002,
    "tip": 0.008,  # where "the tip of the closed gripper" is, relative to the SO-101's tool center point
}


@dataclass(frozen=True)
class Param:
    """One fitted parameter: its name, prior sigma (radians or meters) and group (head, mount, arm, tip).

    `mean` is the prior mean (0 unless an estimate from the data, e.g. the head tilt from the table plane,
    replaces it).
    """

    name: str
    sigma: float
    group: str
    mean: float = 0.0

    @property
    def angular(self) -> bool:
        """True for angles (zeros and rotations), False for positions."""
        return self.name.removeprefix(ARM_PREFIX).startswith(("zero:", "rot:"))

    @property
    def unit(self) -> str:
        """Display unit: deg or mm."""
        return "deg" if self.angular else "mm"

    def display(self, value: float) -> float:
        """`value` (radians / meters) in degrees / millimeters."""
        return float(np.degrees(value) if self.angular else value * 1000.0)


def parameter_set(arm: str = "left", overrides: dict[str, tuple[float, float]] | None = None) -> list[Param]:
    """The parameters fitted for `arm` ("left" / "right"), with their priors (see `SIGMAS`).

    `overrides` maps a parameter name to a replacement `(prior mean, prior sigma)`, for estimates made before the
    fit (see `bootstrap.py`).

    The head tower is not included: moving the head tower and moving the arm mount shift every pixel identically
    (a unit test of the sensitivity analysis asserts it), so only one of them can be found, and the mount is it.
    """
    mount = XLE_ARMS[arm]["mount"]
    xyz, rpy = ("x", "y", "z"), ("roll", "pitch", "yaw")
    params = [Param(f"zero:{j}", SIGMAS["head zero"], "head") for j in HEAD_JOINTS.values()]
    for joint in HEAD_JOINTS.values():
        params += [Param(f"pos:{joint}:{a}", SIGMAS["head origin"], "head") for a in xyz]
    params += [Param(f"pos:{CAMERA_MOUNT}:{a}", SIGMAS[f"camera {a}"], "head") for a in xyz]
    params += [Param(f"rot:{CAMERA_MOUNT}:{a}", SIGMAS["camera rot"], "head") for a in rpy]
    params += [Param(f"pos:{mount}:{a}", SIGMAS["mount pos"], "mount") for a in xyz]
    params += [Param(f"rot:{mount}:{a}", SIGMAS["mount rot"], "mount") for a in rpy]
    for joint in ARM_JOINTS:
        params.append(Param(f"{ARM_PREFIX}zero:{joint}", SIGMAS["arm zero"], "arm"))
        params += [Param(f"{ARM_PREFIX}pos:{joint}:{a}", SIGMAS["arm origin"], "arm") for a in xyz]
    params += [Param(f"{ARM_PREFIX}pos:{TIP_JOINT}:{a}", SIGMAS["tip"], "tip") for a in xyz]
    for i, p in enumerate(params):
        if overrides and p.name in overrides:
            params[i] = Param(p.name, overrides[p.name][1], p.group, overrides[p.name][0])
    return params


@dataclass
class Observation:
    """One click: the tip's pixel, with the robot's measured joints at the moment of the frame.

    `arm`: LeRobot joint positions in degrees (`{"shoulder_pan.pos": ..., ...}`, from `get_state`, not the
    commanded values). `head_ticks`: raw ticks of `head_neck` and `head_eyes`. `pixel`: (u, v) of the click.
    `depth_m`: the depth at the click in meters, if known.
    """

    arm: dict[str, float]
    head_ticks: dict[str, int]
    pixel: tuple[float, float]
    depth_m: float | None = None
    image: str | None = None
    index: int = 0
    timestamp: float = 0.0


@dataclass
class Prediction:
    """Where the model puts the tip.

    `uv`: its pixel; `z`: its depth in the camera frame (meters); `tip_cam`: the point in the camera frame
    (x right, y down, z forward).
    """

    uv: np.ndarray
    z: float
    tip_cam: np.ndarray


class CalibrationModel:
    """Forward model from (joints, parameters) to the tip's pixel. See the module docstring."""

    def __init__(
        self,
        arm: str,
        intrinsics: dict[str, float],
        head_signs: dict[str, int],
        head_refs: dict[str, int],
        head_tree: UrdfTree | None = None,
        arm_tree: UrdfTree | None = None,
        mount_transform: np.ndarray | None = None,
        prior_overrides: dict[str, tuple[float, float]] | None = None,
    ):
        """Build the model.

        `intrinsics`: color `fx, fy, ppx, ppy, width, height` (from a capture's calibration). `head_signs` /
        `head_refs`: per head servo, the saved sign and the reference tick count (see the module docstring).
        `mount_transform` is X_Base_so101; it is computed from the two URDFs unless given. Pass the saved one
        when the trees are calibrated copies (their pivots moved, so recomputing it would give another answer).
        `prior_overrides` replaces the prior of some parameters, `{name: (mean, sigma)}`, and is saved with the
        clicks so the fit uses the same priors.
        """
        self.prior_overrides: dict[str, tuple[float, float]] = dict(prior_overrides or {})
        self.arm = arm
        self.head_tree = head_tree or UrdfTree(XLEROBOT_URDF)
        self.arm_tree = arm_tree or UrdfTree(SO101_URDF)
        self.camera = PinholeCamera(
            CAMERA_LINK,
            *(intrinsics[k] for k in ("fx", "fy", "ppx", "ppy")),
            int(intrinsics["width"]),
            int(intrinsics["height"]),
        )
        self.head_signs, self.head_refs = dict(head_signs), dict(head_refs)
        self.mount_joint = XLE_ARMS[arm]["mount"]
        self.base_link = self.head_tree.joints[self.mount_joint].child
        self.alignment_rms = float("nan")
        self.T = (
            np.array(mount_transform, dtype=float)
            if mount_transform is not None
            else self._align_arm_to_base()
        )

    # ---- placing the SO-101 chain on the XLeRobot's Base frame ---------------------------------------------

    def _align_arm_to_base(self) -> np.ndarray:
        """Rigid transform X_Base_so101, by matching the arm's pivots in the two URDFs (Kabsch).

        It aligns the SO-101's shoulder, elbow and wrist pivots with the same pivots of the arm in
        xlerobot.urdf, over random poses.
        """
        rng = np.random.default_rng(0)
        so_pivots = [self.arm_tree.joints[j].child for j in ARM_JOINTS[1:4]]
        xle_joints = XLE_ARMS[self.arm]["joints"]
        xle_pivots = [self.head_tree.joints[j].child for j in xle_joints[1:4]]
        offsets = (ALIGN_PAN_OFFSET[self.arm], *ALIGN_OFFSETS[1:])
        x_root_base = self.head_tree.link_pose(self.base_link)
        to_base = invert(x_root_base)
        src, dst = [], []
        for _ in range(16):
            q_so = {j: rng.uniform(-1.0, 1.0) for j in ARM_JOINTS}
            q_xle = {
                xj: sign * q_so[sj] + offset
                for xj, sj, sign, offset in zip(xle_joints, ARM_JOINTS[:4], ALIGN_SIGNS, offsets, strict=True)
            }
            for so_link, xle_link in zip(so_pivots, xle_pivots, strict=True):
                src.append(self.arm_tree.point(so_link, np.zeros(3), q_so))
                dst.append((to_base @ np.append(self.head_tree.point(xle_link, np.zeros(3), q_xle), 1.0))[:3])
        src, dst = np.array(src), np.array(dst)
        centroid_src, centroid_dst = src.mean(0), dst.mean(0)
        u, _, vt = np.linalg.svd((src - centroid_src).T @ (dst - centroid_dst))
        rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        self.alignment_rms = float(
            np.sqrt(
                np.mean(np.sum((((src - centroid_src) @ rotation.T) - (dst - centroid_dst)) ** 2, axis=1))
            )
        )
        return pose(rotation, centroid_dst - rotation @ centroid_src)

    # ---- joints -> poses -------------------------------------------------------------------------------------

    def head_angles(self, head_ticks: dict[str, int]) -> dict[str, float]:
        """URDF head joint angles (radians) from raw servo ticks: `sign * (ticks - ref) * 2pi / 4096`."""
        return {
            joint: self.head_signs[servo]
            * (head_ticks[servo] - self.head_refs[servo])
            * 2
            * np.pi
            / TICKS_PER_TURN
            for servo, joint in HEAD_JOINTS.items()
        }

    def arm_angles(self, arm: dict[str, float]) -> dict[str, float]:
        """SO-101 joint angles (radians) from LeRobot's degrees (same numbers, same signs)."""
        return {j: float(np.radians(arm[f"{j}.pos"])) for j in ARM_JOINTS}

    @staticmethod
    def split(theta: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """Split a parameter dict into the xlerobot.urdf part and the SO-101 part (prefix removed)."""
        head = {k: v for k, v in theta.items() if not k.startswith(ARM_PREFIX)}
        arm = {k.removeprefix(ARM_PREFIX): v for k, v in theta.items() if k.startswith(ARM_PREFIX)}
        return head, arm

    def camera_pose(self, head_ticks: dict[str, int], theta: dict[str, float]) -> np.ndarray:
        """X_root_optical: the camera's optical frame in the robot's root frame."""
        head_params, _ = self.split(theta)
        return self.camera.pose_in_root(self.head_tree, self.head_angles(head_ticks), head_params)

    def tip_in_root(self, arm: dict[str, float], theta: dict[str, float]) -> np.ndarray:
        """The closed gripper's tip (the SO-101 tool center point) in the root frame, meters."""
        head_params, arm_params = self.split(theta)
        x_root_base = self.head_tree.link_pose(self.base_link, {}, head_params)
        x_arm_tip = self.arm_tree.link_pose(TIP_FRAME, self.arm_angles(arm), arm_params)
        return (x_root_base @ self.T @ x_arm_tip)[:3, 3]

    def hinge_axis_angle(
        self, arm: dict[str, float], head_ticks: dict[str, int], theta: dict[str, float] | None = None
    ) -> float:
        """How far (degrees, 0-90) the gripper's hinge axis is from the line of sight between the tip and the camera.

        The hinge axis is the axis the moving jaw turns about, i.e. the normal of the plane the jaws open in. At 0 the
        camera looks at the gripper from the side and sees the jaws meet; at 90 it looks along the jaw plane, and
        the jaws hide the pinch point behind each other or behind the gripper body.
        """
        theta = theta or {}
        head_params, arm_params = self.split(theta)
        hinge = self.arm_tree.joints["gripper"]
        x_root_base = self.head_tree.link_pose(self.base_link, {}, head_params)
        x_jaw = x_root_base @ self.T @ self.arm_tree.link_pose(hinge.child, self.arm_angles(arm), arm_params)
        axis = x_jaw[:3, :3] @ (hinge.axis / np.linalg.norm(hinge.axis))
        sight = self.camera_pose(head_ticks, theta)[:3, 3] - self.tip_in_root(arm, theta)
        cosine = abs(float(axis @ sight)) / np.linalg.norm(sight)
        return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))

    def predict(self, obs: Observation, theta: dict[str, float] | None = None) -> Prediction:
        """The tip's pixel, depth and camera-frame position for one observation's joints."""
        theta = theta or {}
        x_root_cam = self.camera_pose(obs.head_ticks, theta)
        tip = self.tip_in_root(obs.arm, theta)
        tip_cam = (invert(x_root_cam) @ np.append(tip, 1.0))[:3]
        uv = self.camera.project(x_root_cam, tip[None])[0]
        return Prediction(uv, float(tip_cam[2]), tip_cam)

    def in_view(self, uv: np.ndarray, margin: float = 0.0) -> bool:
        """True if pixel `uv` is inside the image, at least `margin` pixels from the border."""
        return bool(
            np.isfinite(uv).all()
            and margin <= uv[0] < self.camera.width - margin
            and margin <= uv[1] < self.camera.height - margin
        )

    # ---- (de)serialization -------------------------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """What is needed to rebuild this model: the arm, intrinsics and head signs / refs."""
        c = self.camera
        return {
            "arm": self.arm,
            "intrinsics": {
                "fx": c.fx,
                "fy": c.fy,
                "ppx": c.ppx,
                "ppy": c.ppy,
                "width": c.width,
                "height": c.height,
            },
            "head_signs": self.head_signs,
            "head_refs": self.head_refs,
            "prior_overrides": {k: list(v) for k, v in self.prior_overrides.items()},
        }

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "CalibrationModel":
        """Inverse of `settings`."""
        return cls(
            settings["arm"],
            settings["intrinsics"],
            settings["head_signs"],
            settings["head_refs"],
            prior_overrides={k: tuple(v) for k, v in settings.get("prior_overrides", {}).items()},
        )


def load_calibrated_model(directory: Path) -> tuple[CalibrationModel, dict[str, float]]:
    """The model rebuilt from a `fit.py` output folder, and its fitted parameters.

    The model uses the calibrated URDFs, so predicting with an empty parameter dict reproduces the fit; the
    returned parameters are for inspection (and for comparing with the uncalibrated model).
    """
    directory = Path(directory)
    data = json.loads((directory / "kinematic_calibration.json").read_text())
    s = data["settings"]
    model = CalibrationModel(
        s["arm"],
        s["intrinsics"],
        s["head_signs"],
        s["head_refs"],
        head_tree=UrdfTree(xml=(directory / "xlerobot_calibrated.urdf").read_text()),
        arm_tree=UrdfTree(xml=(directory / "so101_calibrated.urdf").read_text()),
        mount_transform=np.array(data["X_Base_so101"]),
        prior_overrides={k: tuple(v) for k, v in s.get("prior_overrides", {}).items()},
    )
    return model, data["parameters"]


def save_observations(path: Path, model: CalibrationModel, observations: list[Observation]) -> None:
    """Write the model settings and observations to a JSON file (`load_observations` reads it back)."""
    data = {
        "version": 1,
        "settings": model.settings(),
        "observations": [
            {
                "index": o.index,
                "image": o.image,
                "timestamp": o.timestamp,
                "pixel": list(o.pixel),
                "depth_m": o.depth_m,
                "arm": o.arm,
                "head_ticks": o.head_ticks,
            }
            for o in observations
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_observations(path: Path) -> tuple[CalibrationModel, list[Observation]]:
    """Read a file written by `save_observations`: the model and its observations."""
    data = json.loads(Path(path).read_text())
    model = CalibrationModel.from_settings(data["settings"])
    observations = [
        Observation(
            arm=o["arm"],
            head_ticks=o["head_ticks"],
            pixel=tuple(o["pixel"]),
            depth_m=o.get("depth_m"),
            image=o.get("image"),
            index=o.get("index", i),
            timestamp=o.get("timestamp", 0.0),
        )
        for i, o in enumerate(data["observations"])
    ]
    return model, observations
