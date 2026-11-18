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
jaws, drawn, and collided as pads on its straight inner face (`finger_pads`, `<name>_pad<k>`, the collision group,
`FINGER_FRICTION`; rigid: the fingers' give is not modeled yet). On the robot they replace the SO-101's
jaw tips, so with them the jaws' meshes (drawn and collided with) are cut off at the fingers' base. Placed from the
calibration: the fixed finger's tip on the pinch point, tip to base back along the gripper (the pinch -z), its
straight face toward the moving jaw (the pinch -x), its depth across the hinge toward the wrist camera's side; the
moving finger the same part turned half a turn about the gripper's axis, the tips meeting with the jaws closed (the
design's finger frames, `merge_urdf.FINGER_AXES`), each turned about its tip by its calibrated mount
(`XleRobotModel.finger_pose`; `calibration.gripper_sweep` fits both).
Against ruler2's frames both fingers sat on the real ones (within ~10 px; the gripper closed and ~55 deg open).

The moving jaw: MuJoCo's gripper joint is the opening from where the tips meet (`GripperMap.opening_deg`), taking
the SO-101's gripper zero as closed (the fingers are placed closed there, so the stock jaw's own zero does not show).
Whether the real moving finger is at the map's opening: `calibration.gripper_sweep`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.merge_urdf import FINGER_AXES
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
# The gripper's servo and fingers, from what the real gripper did (`calibration.grip_test`, 2026-11-17: pyri 45 mm and
# feno 53 mm bottles, closed in steps past contact, held 30 s, lifted; `docs/sim2real_grasp.md`):
# - its position gain at the jaw (N m/rad): the jaw stalled 1.9-3.8 gripper.pos units short of each target with
#   12-23 % load, ~4.6 % of the stall torque a degree (as `tuning_notes.md` 3.6 has it at P 16). The SO-101 model's
#   998 is ~100 times stiffer;
GRIPPER_KP_NM_PER_RAD = 8.8
# - its torque, capped at the servo's torque limit (`Max_Torque_Limit` 500: 50 % of the STS3215's 3.35 N m stall torque,
#   the model's forcerange; the SO-101's anti-burnout setup, `base_follower.py`). The load rose past 20 % on a close
#   (23 % at 15 past contact, 33 % at 0) before the overload protection tripped (over ~20 % for ~2 s: not modeled; what
#   it can keep is `GRIPPER_SUSTAINED_NM`; 2026-11-17);
GRIPPER_STALL_NM = 3.35
GRIPPER_FORCE_NM = 0.5 * GRIPPER_STALL_NM
GRIPPER_SUSTAINED_NM = 0.2 * GRIPPER_STALL_NM
# - the fingers' sliding friction, fitted to the lifts (`grip_sim`, replaying the trials as run, the stiffness below):
#   the 80 g bottle held through a 4 cm lift and the 116 g, slipperier one slipped after a couple of seconds. In the
#   simulator the 116 g one (pushed in 40 mm) slips at 0.30 and holds from 0.33; the 80 g one holds at 0.30 and up:
#   0.31, near the slipperier bottle's limit as it was. Two bottles: a first value;
FINGER_FRICTION = 0.31
# - the fingers collide as pads on their straight inner faces (`finger_pads`): `PAD_SEGMENTS` boxes along each,
#   `PAD_THICKNESS_M` thick (the face: 67 mm long, 11 to 24 mm deep, flat; the convex hull of the whole finger, the first
#   try, tipped the bottles over);
PAD_SEGMENTS = 4
PAD_THICKNESS_M = 0.002
PAD_TORSIONAL_FRICTION, PAD_ROLLING_FRICTION = 0.005, 0.0001
# - the fingers' give: each finger pivots at its base about the hinge's axis on a spring (`FINGER_STIFFNESS_NM_PER_RAD`;
#   one joint for the Fin Ray's bending). The real stall moved ~13 units (~16 deg of the jaw) from first touch to 15
#   past contact as the load rose 12 -> 23 % (0.37 N m): with the force ~6 cm from the hinge and ~3.8 cm from each
#   finger's base, two fingers in series, ~1 N m/rad each. Fitted (`grip_sim`) to the 45 mm bottle, the stiffer one: at
#   1.0-1.4 its five stalls within 0.2-0.3 gripper.pos units (RMS) and its loads 1.4-2.5 %; the 53 mm one gave more at
#   less load (its own give, a softer bottle: an object's, not the gripper's);
FINGER_STIFFNESS_NM_PER_RAD = 1.2
FINGER_DAMPING = 0.005  # (about critical for a finger's ~15 g, 3 cm)
FINGER_GIVE_RANGE_RAD = 0.6
FINGER_PIVOT_Z_MM = (
    12.0  # the pivot across the hinge: the face's middle at the finger's base (it is 24 mm deep there)
)
# - not modeled: the Fin Ray's wrapping along its length (one joint, not a chain). The real stall moved ~13 units (~16 deg) from first touch to 15 past contact as
#   the load rose 12 -> 23 % (the Fin Ray fingers bending, the bottle squeezed); the hulls here are rigid.
VISUAL_GROUP = 2  # the SO-101's group for what is drawn
# the camera module's drawn and collided geoms, moved out of those groups: the wrist camera's renders leave them out
OWN_MODULE_GROUPS = {VISUAL_GROUP: 4, COLLISION_GROUP: 5}
FINGER_FIXED, FINGER_MOVING = "finger_fixed", "finger_moving"
FINGER_BODIES = {
    FINGER_FIXED: "finger_fixed_body",
    FINGER_MOVING: "finger_moving_body",
}  # (on the jaws, sprung)
FINGER_STL = Path(__file__).parent / "meshes" / "xlerobot_finger.stl"  # the XLeRobot's (`meshes/README.md`)
LFS_POINTER = b"version https://git-lfs"  # what a git LFS file holds until `git lfs pull`
# the XLeRobot finger's STL, in its own coordinates (mm, an assembly's): x from the tip to the base, y out of the
# straight face, z its depth (11 mm at the tip, 23 at the base) from its flat side (z = 0)
FINGER_TIP_MM = (628.0218, 201.2773, 0.0)  # the tip's vertex on the flat side
FINGER_TIP_FAR_MM = (626.8879, 201.2773, 11.115)  # and on the other side (the tip's edge leans 1 mm)
FINGER_BASE_MM = 694.1  # its base's x: the SO-101's jaws are cut off ahead of it
# the SO-101's meshes that hold its jaw tips, by body (the fixed jaw is part of the gripper's frame)
JAW_MESHES = {"gripper": "wrist_roll_follower_so101_v1", "moving_jaw_so101_v1": "moving_jaw_so101_v1"}
# its axes in the pinch frame, the design's (`merge_urdf.FINGER_AXES`)
FINGER_FIXED_AXES, FINGER_MOVING_AXES = FINGER_AXES["fixed"], FINGER_AXES["moving"]
# the moving finger's tip on the wrist camera's side (its flat side's corner), from the pinch point with the jaws closed,
# in the pinch frame: 11 mm across the hinge and, the tip's edge leaning, 1.1 mm back (before its mount's turn)
FINGER_MOVING_WRIST_TIP_M = (
    FINGER_MOVING_AXES @ (np.array(FINGER_TIP_MM) - np.array(FINGER_TIP_FAR_MM)) / 1000.0
)
# and the fixed finger's (its other side's corner: its flat side is on the pinch point)
FINGER_FIXED_WRIST_TIP_M = (
    FINGER_FIXED_AXES @ (np.array(FINGER_TIP_FAR_MM) - np.array(FINGER_TIP_MM)) / 1000.0
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


def _finger_pose(tip_frame: np.ndarray, tip_mm: tuple[float, float, float]) -> np.ndarray:
    """pinch_from_finger (4x4, the STL in meters): the STL's point `tip_mm` at `tip_frame`'s origin, in its axes."""
    pose = np.array(tip_frame, dtype=float)
    pose[:3, 3] = tip_frame[:3, 3] - tip_frame[:3, :3] @ (np.asarray(tip_mm) / 1000.0)
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


def finger_pads(stl: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """The finger's contact pads, in the STL's coordinates (m): (center, half sizes along its x, y, z) each.

    Its straight inner face (the plane through `FINGER_TIP_MM`, square to the STL's y) cut into `PAD_SEGMENTS` along
    the finger (x, tip to base), each as deep (z) as the face is there (11 mm at the tip to 24 at the base),
    `PAD_THICKNESS_M` thick with its outer side on the face.
    """
    v = _read_stl(stl).reshape(-1, 3)
    face_y = FINGER_TIP_MM[1]
    face = v[np.abs(v[:, 1] - face_y) < 0.3]
    if len(face) < 3:  # (another mesh than the XLeRobot finger's: no such face, no pads)
        return []
    edges = np.linspace(face[:, 0].min(), face[:, 0].max(), PAD_SEGMENTS + 1)
    pads = []
    for x0, x1 in zip(edges[:-1], edges[1:], strict=True):
        band = face[(face[:, 0] >= x0) & (face[:, 0] <= x1)]
        if len(band) < 2:
            continue
        z0, z1 = float(band[:, 2].min()), float(band[:, 2].max())
        t = 1000.0 * PAD_THICKNESS_M
        center = np.array([(x0 + x1) / 2, face_y - t / 2, (z0 + z1) / 2]) / 1000.0
        half = np.array([(x1 - x0) / 2, t / 2, (z1 - z0) / 2]) / 1000.0
        pads.append((center, half))
    return pads


def _add_fingers(spec: "mujoco.MjSpec", model: XleRobotModel, arm: str, q: np.ndarray, stl: Path) -> None:
    """The finger STL on both jaws, the jaws cut off at its base (module docstring); the fingers drawn only."""
    if Path(stl).read_bytes()[: len(LFS_POINTER)] == LFS_POINTER:
        raise FileNotFoundError(f"{stl} is a git LFS pointer, not the mesh: run `git lfs pull`")
    spec.add_mesh(name="finger", file=str(Path(stl).resolve()), scale=[0.001] * 3)
    pinch = model.frame_pose(pinch_frame(arm), q)
    # each finger at its frame, turned about its tip by its mount (the jaws closed)
    # the tip on the head's side goes on the pinch point: the moving finger is turned over, its other side there
    for body, link, name, placed in (
        (
            "gripper",
            "gripper_link",
            FINGER_FIXED,
            _finger_pose(model.finger_pose(arm, "fixed"), FINGER_TIP_MM),
        ),
        (
            "moving_jaw_so101_v1",
            "moving_jaw_so101_v1_link",
            FINGER_MOVING,
            _finger_pose(model.finger_pose(arm, "moving"), FINGER_TIP_FAR_MM),
        ),
    ):
        body_pose = model.frame_pose(f"{arm}_{link}", q)
        body_from_finger = np.linalg.inv(body_pose) @ pinch @ placed
        _cut_jaw(spec, body, body_from_finger)
        # the finger on its own body, pivoting at its base about the hinge's axis (the STL's z) on a spring: the Fin Ray's
        # give (`FINGER_STIFFNESS_NM_PER_RAD`); its geoms in that body's frame, the STL's axes, the pivot at its origin
        pivot = np.array([FINGER_BASE_MM, FINGER_TIP_MM[1], FINGER_PIVOT_Z_MM]) / 1000.0
        hinge = body_from_finger.copy()
        hinge[:3, 3] = (body_from_finger @ np.r_[pivot, 1.0])[:3]
        hinge_pos, hinge_quat = _pos_quat(hinge)
        finger = spec.body(body).add_body(name=FINGER_BODIES[name], pos=hinge_pos, quat=hinge_quat)
        finger.add_joint(
            name=f"{name}_give",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0.0, 0.0, 1.0],
            stiffness=FINGER_STIFFNESS_NM_PER_RAD,
            damping=FINGER_DAMPING,
            range=[-FINGER_GIVE_RANGE_RAD, FINGER_GIVE_RANGE_RAD],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        finger.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="finger",
            pos=(-pivot).tolist(),
            quat=[1.0, 0.0, 0.0, 0.0],
            group=VISUAL_GROUP,
            contype=0,
            conaffinity=0,
            rgba=[0.95, 0.95, 0.95, 1.0],
        )
        # what the finger collides with: pads on its straight inner face (`finger_pads`), not the mesh, whose convex
        # hull takes in the ribbed body behind the face (up to 22 mm thick): bottles tipped over on it (grip_sim,
        # 2026-11-17)
        for k, (center, half) in enumerate(finger_pads(stl)):
            finger.add_geom(
                name=f"{name}_pad{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(half),
                pos=(center - pivot).tolist(),
                quat=[1.0, 0.0, 0.0, 0.0],  # (the STL's axes: the box's half sizes are along them)
                group=COLLISION_GROUP,
                density=0.0,  # (the finger's mass is the drawn mesh's)
                friction=[FINGER_FRICTION, PAD_TORSIONAL_FRICTION, PAD_ROLLING_FRICTION],
                condim=4,  # torsional friction too: a bottle held between two pads cannot spin in place
                priority=1,  # the pads' friction and softness govern their contacts (else the larger of the two)
                rgba=[0.9, 0.3, 0.3, 0.6],
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
    if camera is not None:
        _camera(
            gripper,
            WRIST_CAMERA,
            model.gripper_from_camera(arm),
            camera.width,
            camera.height,
            camera.fx,
            camera.fy,
            camera.ppx,
            camera.ppy,
        )
    if fingers is not None:
        _add_fingers(spec, model, arm, q, fingers)
    # friction that holds without creeping, for a grasp (MuJoCo's advice: an elliptic cone, a high impratio, no-slip)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.noslip_iterations = 3
    # the gripper's servo as measured (`GRIPPER_KP_NM_PER_RAD`, `GRIPPER_FORCE_NM`)
    gripper = spec.actuator(GRIPPER)
    gripper.gainprm[0] = GRIPPER_KP_NM_PER_RAD
    gripper.biasprm[1] = -GRIPPER_KP_NM_PER_RAD
    gripper.forcerange = [-GRIPPER_FORCE_NM, GRIPPER_FORCE_NM]
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
