"""`felix_jr.sim2real`: the calibrated arm in MuJoCo (skipped without mujoco, pinocchio or the SO-101 model cached:
tests never download it; `sim2real.assets.so101_model()` does, once)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import robot  # noqa: E402
from lerobot.felix_jr.sim2real.assets import MESHES, cache_dir, is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.scene import TABLE_Z, Cylinder, Scene, build, scene_spec  # noqa: E402
from lerobot.felix_jr.sim2real.view import render  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


def _random_pose(rng) -> dict[str, float]:
    return {f"{j}.pos": float(rng.uniform(-60.0, 60.0)) for j in robot.JOINTS}


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_frames_are_the_calibrated_models(model, arm):
    """Every link, the pinch point and both cameras where `XleRobotModel` puts them, at any readings."""
    scene = build(model, arm)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pose = _random_pose(rng)
        scene.set_pose(pose)
        q = model.joint_vector({arm: pose})
        for body, link in robot.BODIES.items():
            if body == "moving_jaw_so101_v1":  # its angle is the opening, not a calibrated joint
                continue
            expected = model.frame_pose(f"{arm}_{link}", q)
            np.testing.assert_allclose(scene.d.body(body).xpos, expected[:3, 3], atol=1e-9)
            np.testing.assert_allclose(scene.d.body(body).xmat.reshape(3, 3), expected[:3, :3], atol=1e-6)
        np.testing.assert_allclose(
            scene.d.site(robot.PINCH).xpos, model.frame_pose(pinch_frame(arm), q)[:3, 3], atol=1e-9
        )
        np.testing.assert_allclose(
            robot.camera_pose(scene.m, scene.d, robot.WRIST_CAMERA),
            model.wrist_camera_pose(arm, q),
            atol=1e-6,
        )
    np.testing.assert_allclose(
        robot.camera_pose(scene.m, scene.d, robot.HEAD_CAMERA),
        model.camera_pose(model.joint_vector()),
        atol=1e-9,
    )


def test_the_head_camera_follows_the_head_ticks(model):
    ticks = {"head_neck": 1529, "head_eyes": 1496}
    scene = build(model, "left", head_ticks=ticks)
    expected = model.camera_pose(model.joint_vector(head_ticks=ticks))
    np.testing.assert_allclose(robot.camera_pose(scene.m, scene.d, robot.HEAD_CAMERA), expected, atol=1e-9)


@pytest.mark.parametrize("camera", [robot.WRIST_CAMERA, robot.HEAD_CAMERA])
def test_a_point_renders_where_the_lens_puts_it(model, camera):
    """A small ball off the optical axis renders at its pinhole projection (the calibrated lens, no distortion)."""
    arm, pose = "left", {"shoulder_pan.pos": 10.0, "shoulder_lift.pos": 20.0, "elbow_flex.pos": -10.0}
    probe = build(model, arm)
    probe.set_pose(pose)
    if camera == robot.WRIST_CAMERA:
        root_from_camera = model.wrist_camera_pose(arm, model.joint_vector({arm: pose}))
        lens = model.wrist_cameras[arm]
        k, size = lens.matrix(), (lens.width, lens.height)
    else:
        root_from_camera = model.camera_pose(model.joint_vector())
        i = model.intrinsics
        k = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]])
        size = (int(i["width"]), int(i["height"]))
    in_camera = np.array([0.03, -0.02, 0.15, 1.0])  # right and up of the axis, 15 cm out
    point = (root_from_camera @ in_camera)[:3]
    spec = scene_spec(model, arm)
    spec.worldbody.add_geom(
        name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0], pos=point.tolist()
    )
    m = spec.compile()
    scene = Scene(model, arm, m, mujoco.MjData(m))
    scene.set_pose(pose)
    with mujoco.Renderer(m, size[1], size[0]) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(scene.d, camera=camera)
        ids = renderer.render()[..., 0]
    v, u = np.nonzero(ids == m.geom("ball").id)
    assert len(u) > 10
    expected = (k @ in_camera[:3])[:2] / in_camera[2]
    np.testing.assert_allclose([u.mean() + 0.5, v.mean() + 0.5], expected, atol=1.0)


def test_the_camera_mount_and_the_real_board_collide_and_the_module_is_not_in_its_own_view(model):
    """TheRobotStudio's mount colliding; its 32 mm module not (the real 45 mm board does instead: `_camera_board`); the
    module's geoms and the board in their own groups, drawn from the head and left out of the wrist camera's renders
    (the camera is inside it)."""
    scene = build(model, "left")
    m = scene.m
    module = [m.geom(i) for i in range(m.ngeom) if m.geom(i).bodyid[0] == m.body(robot.CAMERA_MODULE).id]
    assert {g.group[0] for g in module} == set(robot.OWN_MODULE_GROUPS.values())
    mount = [m.geom(i) for i in range(m.ngeom) if m.geom(i).bodyid[0] == m.body(robot.CAMERA_MOUNT).id]
    assert any(g.contype[0] and g.group[0] == robot.COLLISION_GROUP for g in mount)
    assert not any(g.contype[0] for g in module)
    board = m.geom("wrist_camera_board")
    assert board.contype[0] and board.group[0] == robot.OWN_MODULE_GROUPS[robot.COLLISION_GROUP]
    # the camera is inside its module (why its renders leave it out): within the module's mesh's bounds
    for arm in ("left", "right"):
        scene = build(model, arm)
        visual = next(g for g in module if g.group[0] == robot.OWN_MODULE_GROUPS[robot.VISUAL_GROUP])
        inside = _world_vertices(scene.m, scene.d, visual.name or visual.id)
        lens = scene.d.geom(visual.id).xmat.reshape(3, 3)  # the module's axes
        local = (inside - scene.d.geom(visual.id).xpos) @ lens
        cam = (scene.d.camera(robot.WRIST_CAMERA).xpos - scene.d.geom(visual.id).xpos) @ lens
        assert (local.min(0) < cam).all() and (cam < local.max(0)).all(), arm


def test_the_scene_has_the_table_and_objects_rest_on_it(model):
    scene = build(model, "left", (Cylinder("bottle", -0.45, -0.04, 0.028, 0.09),))
    top = scene.d.geom("table").xpos[2] + scene.m.geom("table").size[2]
    assert top == pytest.approx(TABLE_Z)
    assert scene.d.body("bottle").xpos[2] == pytest.approx(TABLE_Z + 0.045)
    for _ in range(200):
        mujoco.mj_step(scene.m, scene.d)
    assert scene.d.body("bottle").xpos[2] == pytest.approx(TABLE_Z + 0.045, abs=2e-3)


def test_the_gripper_opens_by_its_map(model):
    scene = build(model, "left")
    scene.set_pose({"gripper.pos": 30.0})
    angle = scene.d.qpos[scene.m.joint(robot.GRIPPER).qposadr[0]]
    assert np.degrees(angle) == pytest.approx(model.opening_deg("left", 30.0))


def test_the_views_render(model):
    scene = build(model, "left")
    for camera in (robot.WRIST_CAMERA, robot.HEAD_CAMERA):
        for collision in (False, True):
            image = render(scene, camera, collision)
            assert image.shape == (480, 640, 3) and image.std() > 5


def test_the_cache_holds_the_model_and_its_meshes():
    assert (cache_dir() / "assets").is_dir() and len(MESHES) == 15


def test_the_joints_are_lerobots_as_position_servos_that_hold(model):
    """The SO-101's actuators drive LeRobot's joints; told a pose, the servos get there and hold it (3 s)."""
    scene = build(model, "left")
    names = tuple(scene.m.actuator(i).name for i in range(scene.m.nu))
    assert names == (*robot.JOINTS, robot.GRIPPER)
    assert all(scene.m.actuator(i).trntype == mujoco.mjtTrn.mjTRN_JOINT for i in range(scene.m.nu))
    pose = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -30.0,
        "elbow_flex.pos": 30.0,
        "wrist_flex.pos": 45.0,
    }
    scene.set_pose(pose)
    for _ in range(1500):
        mujoco.mj_step(scene.m, scene.d)
    joints = [np.degrees(scene.d.qpos[scene.m.joint(j).qposadr[0]]) for j in robot.JOINTS[:4]]
    assert (
        np.abs(np.array(joints) - [pose[f"{j}.pos"] for j in robot.JOINTS[:4]]).max() < 5.0
    )  # gravity, backlash


def _stl(path, vertices, faces) -> None:
    """A binary STL of `faces` (index triples into `vertices`, mm)."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + len(faces).to_bytes(4, "little"))
        for face in faces:
            f.write(
                np.zeros(3, "<f4").tobytes()
                + np.asarray([vertices[i] for i in face], "<f4").tobytes()
                + b"\0\0"
            )


def _world_vertices(m, d, name: str | int) -> np.ndarray:
    """A mesh geom's vertices in the world (N, 3)."""
    mesh = m.geom(name).dataid[0]
    verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
    return d.geom(name).xpos + verts @ d.geom(name).xmat.reshape(3, 3).T


def test_the_fingers_tips_are_on_the_pinch_point(model, tmp_path):
    """A wedge in the finger STL's coordinates, its tip at `FINGER_TIP_MM`: on both jaws, closed, the tips (the moving
    one from its other side) are on the pinch point, and the fixed finger runs back along the gripper."""
    tip, top = np.array(robot.FINGER_TIP_MM), np.array(robot.FINGER_TIP_FAR_MM)
    vertices = [tip, top, [694.0, 180.0, 0.0], [694.0, 201.3, 0.0], [694.0, 201.3, 23.0]]
    _stl(tmp_path / "finger.stl", vertices, [(0, 2, 3), (0, 3, 4), (0, 4, 2), (2, 4, 3), (1, 4, 3)])
    scene = build(model, "left", fingers=tmp_path / "finger.stl")
    closed = model.gripper_maps["left"].pos_closed
    scene.set_pose({"shoulder_pan.pos": 10.0, "elbow_flex.pos": 20.0, "gripper.pos": closed})
    m, d = scene.m, scene.d
    pinch = d.site(robot.PINCH).xpos
    for name in (robot.FINGER_FIXED, robot.FINGER_MOVING):
        assert np.linalg.norm(_world_vertices(m, d, name) - pinch, axis=1).min() < 1e-6, name
        assert (
            m.geom(name).contype[0] == 0 and m.geom(name).group[0] == robot.VISUAL_GROUP
        )  # drawn only, for now
    stock = build(model, "left", fingers=None).m
    for (
        body,
        mesh,
    ) in robot.JAW_MESHES.items():  # the SO-101's jaw tips are cut off: fewer faces, drawn and collided
        cut = [
            m.geom(i)
            for i in range(m.ngeom)
            if m.geom(i).bodyid[0] == m.body(body).id and m.geom(i).dataid[0] == m.mesh(f"{mesh}_cut").id
        ]
        assert (
            len(cut) == 2
            and m.mesh_facenum[m.mesh(f"{mesh}_cut").id] < stock.mesh_facenum[stock.mesh(mesh).id]
        )
    toward_the_tips = d.site(robot.PINCH).xmat.reshape(3, 3)[:, 2]  # the pinch z
    behind = (_world_vertices(m, d, robot.FINGER_FIXED) - pinch) @ toward_the_tips
    # behind the tip (but its other side's corner: the tip's edge leans 1.1 mm forward), the base ~67 mm back
    assert behind.max() < 1.2e-3 and behind.min() < -0.06


@pytest.mark.skipif(
    robot.FINGER_STL.read_bytes()[: len(robot.LFS_POINTER)] == robot.LFS_POINTER,
    reason="git lfs pull not run",
)
def test_the_xlerobot_fingers_are_the_default_and_their_tips_meet_on_the_pinch_point(model):
    scene = build(model, "right")
    scene.set_pose({"gripper.pos": model.gripper_maps["right"].pos_closed})
    pinch = scene.d.site(robot.PINCH).xpos
    for name in (robot.FINGER_FIXED, robot.FINGER_MOVING):
        assert np.linalg.norm(_world_vertices(scene.m, scene.d, name) - pinch, axis=1).min() < 1e-6, name
