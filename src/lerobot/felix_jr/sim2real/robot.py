"""One calibrated arm in MuJoCo: TheRobotStudio's SO-101 (meshes, servos), posed by our calibrated model.

The SO-101's bodies are our URDF's arm links one to one (`BODIES`), so each body's pose in its parent is taken from
`XleRobotModel` at all-zero readings: the calibrated joint origins with the joints' zero corrections folded in (the
model adds them to the readings). MuJoCo's joint angles are then LeRobot's readings in radians, and its frames are
the calibrated model's (`tests/felix_jr/test_sim2real.py`). The world frame is the model's root frame; the arm's base
is placed where the calibration puts it.

TheRobotStudio's wrist camera mount and camera are removed (ours sits elsewhere on the gripper: our camera would
look out from inside theirs); our camera's housing is a box behind its lens, for collisions only. Added to the
SO-101: a `pinch` site (the pinch frame: the fixed jaw's tip, `pinch_frame`), the arm's wrist camera
(`wrist_camera`: its calibrated pose on the gripper and its lens, without the distortion: the renders are as our
undistorted frames) and the head camera (`head_camera`, fixed in the world: the head is not simulated, only where it
looks from at the given head ticks).

The moving jaw: MuJoCo's gripper joint is the opening from where the tips meet (`GripperMap.opening_deg`), taking
the SO-101's gripper zero as closed. That is not calibrated: compare the jaws in a render with a real wrist frame
(`sim2real.view --robot`).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera, XleRobotModel, pinch_frame
from lerobot.utils.import_utils import _mujoco_available, require_package

from .assets import so101_model

if TYPE_CHECKING or _mujoco_available:
    import mujoco

# TheRobotStudio's body -> our URDF's link (`<arm>_<link>`), parent first
BODIES = {
    "base": "base_link",
    "shoulder": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "lower_arm": "lower_arm_link",
    "wrist": "wrist_link",
    "gripper": "gripper_link",
    "moving_jaw_so101_v1": "moving_jaw_so101_v1_link",
}
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)  # MuJoCo's names, LeRobot's too
GRIPPER = "gripper"
WRIST_CAMERA, HEAD_CAMERA, PINCH = "wrist_camera", "head_camera", "pinch"
# an optical frame (x right, y down, z forward) to MuJoCo's camera frame (x right, y up, looking along -z)
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
PIXEL_M = 1e-6  # a nominal pixel pitch for MuJoCo's sensor size
# TheRobotStudio's wrist camera mount and camera (one body and its child): removed, ours is elsewhere on the gripper
THEIR_CAMERA_MOUNT = "wrist_camera_mount"
CAMERA_HOUSING = "wrist_camera_housing"
CAMERA_HOUSING_M = (0.032, 0.032, 0.02)  # the UVC module (x, y across the lens, z along it)
COLLISION_GROUP = 3  # the SO-101's group for what collides (drawn only in a collision view)


def _pos_quat(pose: np.ndarray) -> tuple[list[float], list[float]]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pose[:3, :3]).ravel())
    return pose[:3, 3].tolist(), quat.tolist()


def _camera(
    parent, name: str, pose: np.ndarray, width: int, height: int, fx: float, fy: float, ppx: float, ppy: float
):
    """A pinhole camera at `pose` (its optical frame, in `parent`'s frame) with the lens's pixels."""
    flipped = pose.copy()
    flipped[:3, :3] = pose[:3, :3] @ OPTICAL_TO_MUJOCO
    pos, quat = _pos_quat(flipped)
    camera = parent.add_camera(name=name, pos=pos, quat=quat)
    camera.resolution = [width, height]
    camera.sensor_size = [
        width * PIXEL_M,
        height * PIXEL_M,
    ]  # MuJoCo's lens needs one; the pixels are what count
    camera.focal_pixel = [fx, fy]
    # MuJoCo's principal point: the image's center minus ours, in pixels (`test_sim2real.py` renders a point through it)
    camera.principal_pixel = [width / 2 - ppx, height / 2 - ppy]
    return camera


def arm_spec(
    model: XleRobotModel, arm: str, head_ticks: Mapping[str, int] | None = None, mjcf: Path | None = None
) -> "mujoco.MjSpec":
    """The SO-101's spec, posed as `model`'s `arm`, with the pinch site and the wrist and head cameras (module docstring).

    `head_ticks`: where the head looks from (its servos' ticks; None: its recorded home).
    """
    require_package("mujoco", extra="felix_jr_sim")
    path = mjcf or so101_model()
    spec = mujoco.MjSpec.from_file(str(path))
    spec.meshdir = str(path.parent / "assets")  # absolute: the spec may be compiled from elsewhere
    q = model.joint_vector(head_ticks=head_ticks)
    poses = {body: model.frame_pose(f"{arm}_{link}", q) for body, link in BODIES.items()}
    parents = list(BODIES)
    spec.body("base").pos, spec.body("base").quat = _pos_quat(poses["base"])
    for body in parents[1:]:
        parent = "gripper" if body == "moving_jaw_so101_v1" else parents[parents.index(body) - 1]
        spec.body(body).pos, spec.body(body).quat = _pos_quat(np.linalg.inv(poses[parent]) @ poses[body])
    gripper = spec.body("gripper")
    spec.delete(spec.body(THEIR_CAMERA_MOUNT))  # not where ours is: our camera would look out from inside it
    pos, quat = _pos_quat(np.linalg.inv(poses["gripper"]) @ model.frame_pose(pinch_frame(arm), q))
    gripper.add_site(name=PINCH, pos=pos, quat=quat, size=[0.003, 0, 0], rgba=[1, 0, 1, 1], group=4)
    camera: WristCamera | None = model.wrist_cameras.get(arm)
    if camera is not None and camera.gripper_from_camera is not None:
        _camera(
            gripper,
            WRIST_CAMERA,
            np.array(camera.gripper_from_camera),
            camera.width,
            camera.height,
            camera.fx,
            camera.fy,
            camera.ppx,
            camera.ppy,
        )
        # its housing, for collisions only (not drawn in the renders): behind the lens, square to its axis
        housing = np.array(camera.gripper_from_camera) @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -CAMERA_HOUSING_M[2] / 2 - 0.002], [0, 0, 0, 1]]
        )
        pos, quat = _pos_quat(housing)
        gripper.add_geom(
            name=CAMERA_HOUSING,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[h / 2 for h in CAMERA_HOUSING_M],
            pos=pos,
            quat=quat,
            group=COLLISION_GROUP,
            rgba=[0.1, 0.1, 0.1, 1],
            contype=1,
            conaffinity=1,
        )
    k = model.intrinsics
    _camera(
        spec.worldbody,
        HEAD_CAMERA,
        model.camera_pose(q),
        int(k["width"]),
        int(k["height"]),
        k["fx"],
        k["fy"],
        k["ppx"],
        k["ppy"],
    )
    # the offscreen buffer: the largest camera's frame
    spec.visual.global_.offwidth = max(int(k["width"]), camera.width if camera is not None else 0)
    spec.visual.global_.offheight = max(int(k["height"]), camera.height if camera is not None else 0)
    return spec


def set_pose(
    model: XleRobotModel, arm: str, m: "mujoco.MjModel", d: "mujoco.MjData", pose: Mapping[str, float]
) -> None:
    """Put the arm at LeRobot's readings `pose`, its servos told to hold there, and update the frames.

    `pose`: `{"shoulder_pan.pos": deg, ..., "gripper.pos": 0-100}`; the gripper through the arm's `GripperMap` (as
    degrees without one).
    """
    angles = {j: np.radians(pose[f"{j}.pos"]) for j in JOINTS if f"{j}.pos" in pose}
    if "gripper.pos" in pose:
        g = pose["gripper.pos"]
        opening = model.opening_deg(arm, g) if arm in model.gripper_maps else g
        angles[GRIPPER] = np.radians(opening)
    for joint, angle in angles.items():
        d.qpos[m.joint(joint).qposadr[0]] = angle
        d.ctrl[m.actuator(joint).id] = angle
    mujoco.mj_forward(m, d)


def camera_pose(m: "mujoco.MjModel", d: "mujoco.MjData", name: str) -> np.ndarray:
    """A camera's optical frame (x right, y down, z forward) in the world (the root frame), 4x4."""
    pose = np.eye(4)
    pose[:3, :3] = d.camera(name).xmat.reshape(3, 3) @ OPTICAL_TO_MUJOCO
    pose[:3, 3] = d.camera(name).xpos
    return pose
