"""One calibrated arm in MuJoCo: TheRobotStudio's SO-101 (meshes, servos), posed by our calibrated model.

The SO-101's bodies are our URDF's arm links one to one (`BODIES`), so each body's pose in its parent is taken from
`XleRobotModel` at all-zero readings: the calibrated joint origins with the joints' zero corrections folded in (the
model adds them to the readings). MuJoCo's joint angles are then LeRobot's readings in radians, and its frames are
the calibrated model's (`tests/felix_jr/test_sim2real.py`). The world frame is the model's root frame; the arm's base
is placed where the calibration puts it.

The wrist camera's mount and module are TheRobotStudio's (the hex-nut 32x32 UVC mount, the robot's), at their design
pose: our calibrated optical frames are within 2 deg of its lens axis, their centers inside its lens barrel (both arms
alike: the robot's mounts sit as designed). Our camera is inside its own module, so the module is drawn in groups of
its own (`OWN_MODULE_GROUPS`) that the wrist camera's renders leave out (`view.render`). Added to the SO-101: a `pinch`
site (the pinch frame: the fixed jaw's tip, `pinch_frame`), the arm's wrist camera
(`wrist_camera`: its calibrated pose on the gripper and its lens, without the distortion: the renders are as our
undistorted frames) and the head camera (`head_camera`, fixed in the world: the head is not simulated, only where it
looks from at the given head ticks).

The fingers (`fingers`, default `FINGER_STL`): the XLeRobot's Fin Ray finger (`meshes/README.md`) on both
jaws, drawn only (not collided with yet: the contact pads are the next step). On the robot they replace the SO-101's
jaw tips, so with them the jaws' meshes (drawn and collided with) are cut off at the fingers' base. Placed from the
calibration: the fixed finger's tip on the pinch point, tip to base back along the gripper (the pinch -z), its
straight face toward the moving jaw (the pinch -x), its depth across the hinge toward the wrist camera's side; the
moving finger the same part turned half a turn about the gripper's axis, the tips meeting with the jaws closed, and turned on its jaw about its
tip by the arm's `GripperMap.finger_mount_deg`, and the fixed one about its tip by `fixed_finger_mount_deg`
(`calibration.gripper_sweep` fits both).
Against ruler2's frames both fingers sat on the real ones (within ~10 px; the gripper closed and ~55 deg open).

The moving jaw: MuJoCo's gripper joint is the opening from where the tips meet (`GripperMap.opening_deg`), taking
the SO-101's gripper zero as closed (the fingers are placed closed there, so the stock jaw's own zero does not show).
Whether the real moving finger is at the map's opening: `calibration.gripper_sweep`.
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
CAMERA_MOUNT, CAMERA_MODULE = (
    "wrist_camera_mount",
    "wrist_camera",
)  # TheRobotStudio's bodies (the module: the camera)
COLLISION_GROUP = 3  # the SO-101's group for what collides (drawn only in a collision view)
VISUAL_GROUP = 2  # the SO-101's group for what is drawn
# the camera module's drawn and collided geoms, moved out of those groups: the wrist camera's renders leave them out
OWN_MODULE_GROUPS = {VISUAL_GROUP: 4, COLLISION_GROUP: 5}
FINGER_FIXED, FINGER_MOVING = "finger_fixed", "finger_moving"
FINGER_STL = Path(__file__).parent / "meshes" / "xlerobot_finger.stl"  # the XLeRobot's (`meshes/README.md`)
LFS_POINTER = b"version https://git-lfs"  # what a git LFS file holds until `git lfs pull`
# the XLeRobot finger's STL, in its own coordinates (mm, an assembly's): x from the tip to the base, y out of the
# straight face, z its depth (11 mm at the tip, 23 at the base) from its flat side (z = 0)
FINGER_TIP_MM = (628.0218, 201.2773, 0.0)  # the tip's vertex on the flat side
FINGER_TIP_FAR_MM = (626.8879, 201.2773, 11.115)  # and on the other side (the tip's edge leans 1 mm)
FINGER_BASE_MM = 694.1  # its base's x: the SO-101's jaws are cut off ahead of it
# the SO-101's meshes that hold its jaw tips, by body (the fixed jaw is part of the gripper's frame)
JAW_MESHES = {"gripper": "wrist_roll_follower_so101_v1", "moving_jaw_so101_v1": "moving_jaw_so101_v1"}
# its axes in the pinch frame: the fixed finger's x to -z (back along the gripper), y to -x (toward the moving jaw),
# z to +y (across the hinge); the moving one turned half a turn about the gripper's axis
FINGER_FIXED_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
FINGER_MOVING_AXES = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
# the moving finger's tip on the wrist camera's side (its flat side's corner), from the pinch point with the jaws closed,
# in the pinch frame: 11 mm across the hinge and, the tip's edge leaning, 1.1 mm back (before its mount's turn)
FINGER_MOVING_WRIST_TIP_M = (
    FINGER_MOVING_AXES @ (np.array(FINGER_TIP_MM) - np.array(FINGER_TIP_FAR_MM)) / 1000.0
)


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


def _finger_pose(axes: np.ndarray, tip_mm: tuple[float, float, float]) -> np.ndarray:
    """pinch_from_finger (4x4, the STL in meters): `axes` the STL's axes in the pinch frame, its point `tip_mm` on the pinch."""
    pose = np.eye(4)
    pose[:3, :3] = axes
    pose[:3, 3] = -axes @ (np.asarray(tip_mm) / 1000.0)
    return pose


def _read_stl(path: Path) -> np.ndarray:
    """A binary STL's triangles (N, 3, 3), in its own units."""
    data = Path(path).read_bytes()
    n = int.from_bytes(data[80:84], "little")
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data[84 : 84 + record.itemsize * n], dtype=record)["v"].astype(float)


def _pose_of(pos, quat) -> np.ndarray:
    pose = np.eye(4)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    pose[:3, :3], pose[:3, 3] = mat.reshape(3, 3), pos
    return pose


def _cut_jaw(spec: "mujoco.MjSpec", body: str, body_from_finger: np.ndarray) -> None:
    """Cut `body`'s jaw mesh off ahead of the finger's base (keep the triangles wholly behind it)."""
    name = JAW_MESHES[body]  # the SO-101's meshes are named after their files (their own names are empty)
    mesh = next(m for m in spec.meshes if Path(m.file).stem == name)
    triangles = _read_stl(Path(spec.meshdir) / mesh.file) * np.asarray(mesh.scale)
    geoms = [g for g in spec.body(body).geoms if g.meshname == name]
    body_from_geom = _pose_of(geoms[0].pos, geoms[0].quat)  # the drawn and the collided geom are placed alike
    finger_from_geom = np.linalg.inv(body_from_finger) @ body_from_geom
    flat = triangles.reshape(-1, 3)
    along = (flat @ finger_from_geom[:3, :3].T + finger_from_geom[:3, 3])[:, 0].reshape(
        -1, 3
    )  # the finger's x
    kept = triangles[(along >= FINGER_BASE_MM / 1000.0).all(axis=1)]
    cut = spec.add_mesh(name=f"{name}_cut")
    cut.uservert = kept.reshape(-1).tolist()
    cut.userface = np.arange(kept.shape[0] * 3).tolist()
    for g in geoms:
        g.meshname = cut.name


def _add_fingers(spec: "mujoco.MjSpec", model: XleRobotModel, arm: str, q: np.ndarray, stl: Path) -> None:
    """The finger STL on both jaws, the jaws cut off at its base (module docstring); the fingers drawn only."""
    if Path(stl).read_bytes()[: len(LFS_POINTER)] == LFS_POINTER:
        raise FileNotFoundError(f"{stl} is a git LFS pointer, not the mesh: run `git lfs pull`")
    spec.add_mesh(name="finger", file=str(Path(stl).resolve()), scale=[0.001] * 3)
    pinch = model.frame_pose(pinch_frame(arm), q)
    # each finger's mount: turned about its tip (the pinch point, the jaws closed) about the hinge's axis
    gripper_map = model.gripper_maps.get(arm)

    def mount(deg: float) -> np.ndarray:
        turn = np.eye(4)
        if deg:
            turn[:3, :3] = model.moving_jaw(arm, deg)[:3, :3]
        return turn

    fixed_deg = gripper_map.fixed_finger_mount_deg if gripper_map is not None else 0.0
    moving_deg = gripper_map.finger_mount_deg if gripper_map is not None else 0.0
    # the tip on the head's side goes on the pinch point: the moving finger is turned over, its other side there
    for body, link, name, placed in (
        (
            "gripper",
            "gripper_link",
            FINGER_FIXED,
            mount(fixed_deg) @ _finger_pose(FINGER_FIXED_AXES, FINGER_TIP_MM),
        ),
        (
            "moving_jaw_so101_v1",
            "moving_jaw_so101_v1_link",
            FINGER_MOVING,
            mount(moving_deg) @ _finger_pose(FINGER_MOVING_AXES, FINGER_TIP_FAR_MM),
        ),
    ):
        body_pose = model.frame_pose(f"{arm}_{link}", q)
        body_from_finger = np.linalg.inv(body_pose) @ pinch @ placed
        _cut_jaw(spec, body, body_from_finger)
        pos, quat = _pos_quat(body_from_finger)
        spec.body(body).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="finger",
            pos=pos,
            quat=quat,
            group=VISUAL_GROUP,
            contype=0,
            conaffinity=0,
            rgba=[0.95, 0.95, 0.95, 1.0],
        )


def arm_spec(
    model: XleRobotModel,
    arm: str,
    head_ticks: Mapping[str, int] | None = None,
    mjcf: Path | None = None,
    fingers: Path | None = FINGER_STL,
) -> "mujoco.MjSpec":
    """The SO-101's spec, posed as `model`'s `arm`, with the pinch site and the wrist and head cameras (module docstring).

    `head_ticks`: where the head looks from (its servos' ticks; None: its recorded home). `fingers`: the XLeRobot
    finger's STL, drawn on both jaws in place of their tips (None: the SO-101's own jaws).
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
    for geom in spec.body(CAMERA_MODULE).geoms:  # our camera is inside it (module docstring)
        geom.group = OWN_MODULE_GROUPS[geom.group]
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
    if fingers is not None:
        _add_fingers(spec, model, arm, q, fingers)
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
