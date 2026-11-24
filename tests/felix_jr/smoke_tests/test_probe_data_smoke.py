"""The vision probe's labels against its frames (`sim2real/probe_data.py`): the object the labels describe, in the pinch
frame, drawn through the model's wrist camera, is the object in the rendered frame."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from test_start_pool_smoke import _pool  # noqa: E402  (the same folder: pytest puts it on the path)

from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    WristCamera,
    XleRobotModel,
    pinch_frame,
    wrist_camera_frame,
)
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.silhouette import silhouette  # noqa: E402
from lerobot.felix_jr.sim2real import mjlab_env as me, mjlab_object as obj  # noqa: E402
from lerobot.felix_jr.sim2real.probe_data import MJ_OBJ_GEOM, object_share  # noqa: E402
from lerobot.felix_jr.sim2real.render_bench import SENSOR, camera_cfg  # noqa: E402


def _in_view(env, cyl) -> None:
    """Each world's object moved (upright, on the table) to where the wrist camera's optical axis meets the table: its
    foot in the middle of the frame, its sides clear of the frame's edges; rendered again."""
    cam = env.scene.sensors[SENSOR].camera_idx
    pos, mat = env.sim.data.cam_xpos[:, cam], env.sim.data.cam_xmat[:, cam]
    look = -mat[:, :, 2]  # (MuJoCo's cameras look along their -z)
    center, _ = cyl.pose(env)
    foot = center[:, 2] - 0.5 * cyl.size(env)[:, 1]
    t = (foot - pos[:, 2]) / look[:, 2]
    assert (t > 0).all(), "the camera looks up"
    hit = pos + t[:, None] * look
    entity = env.scene[cyl.entity]
    root = torch.zeros(env.num_envs, 13)
    root[:, :2] = hit[:, :2]
    root[:, 2] = entity.data.root_link_pos_w[:, 2]
    root[:, 3] = 1.0  # (upright)
    entity.write_root_state_to_sim(root)
    env.sim.forward()
    env.sim.sense()


def test_the_labelled_object_is_the_rendered_one(tmp_path):
    """Two worlds, the object moved into view, then after the arm moves: the cylinder the labels describe (`in_pinch`,
    `size`), put in the camera through the model's pinch-to-camera transform and drawn by the calibrated lens without
    its distortion (the renders are undistorted frames), covers the object's rendered pixels (IoU > 0.95)."""
    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "pool.npz"
    _pool(model).save(path)
    cfg = me.env_cfg(model, num_envs=2, start_pool=path, randomize=False)
    w, h = 224, 168
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg(w, h, data_types=("rgb", "segmentation")))
    env = me.make_env(cfg, "cpu")
    env.reset()
    cyl = obj.of(env)
    _in_view(env, cyl)
    q = model.joint_vector()
    cam_from_pinch = np.linalg.inv(
        np.linalg.inv(model.frame_pose(pinch_frame("right"), q))
        @ model.frame_pose(wrist_camera_frame("right"), q)
    )
    real = model.wrist_cameras["right"]
    s = w / real.width
    lens = WristCamera(w, h, real.fx * s, real.fy * s, real.ppx * s, real.ppy * s, "rational", (0.0,) * 8)
    ids = env.scene[cyl.entity].indexing.geom_ids.numpy()
    for moves in (0, 4):
        for _ in range(moves):
            env.step(0.3 * torch.ones(2, env.action_manager.total_action_dim))
        assert (object_share(env, cyl).numpy() > 0.01).all()
        seg = env.scene.sensors[SENSOR].data.segmentation.numpy()
        pose, size = cyl.in_pinch(env).numpy(), cyl.size(env).numpy()
        for i in range(len(pose)):
            center = cam_from_pinch[:3, :3] @ pose[i, :3] + cam_from_pinch[:3, 3]
            axis = cam_from_pinch[:3, :3] @ pose[i, 3:]
            radius, height = size[i]
            labelled = Cylinder.between(center - axis * height / 2, center + axis * height / 2, float(radius))
            outline = silhouette(labelled, np.eye(4), lens)
            drawn = np.zeros((h, w), np.uint8)
            cv2.fillPoly(drawn, [np.round(outline).astype(np.int32)], 1)
            rendered = np.isin(seg[i, ..., 0], ids) & (seg[i, ..., 1] == MJ_OBJ_GEOM)
            iou = (drawn.astype(bool) & rendered).sum() / (drawn.astype(bool) | rendered).sum()
            assert iou > 0.95, (i, moves, iou)
