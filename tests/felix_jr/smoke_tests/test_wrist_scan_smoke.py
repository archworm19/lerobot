"""`wrist_scan` end to end against the fake robot: the wrist swept, the frames' poses refined, the table found.

The fake robot's joints read 1-2 deg off the truth (the arm's error), and its wrist camera renders a textured table
(`test_sfm`'s) from the true camera pose. The sweep moves only the wrist and comes back; structure from motion should
say how the camera really moved (the truth, not the joints read) and where the table is.
"""

import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from test_sfm import motion_errors, render, table_texture  # noqa: E402
from test_wrist_camera import TABLE_BELOW_BASE_M, TRUE_CAMERA, URDF, _with_pose, start_joints  # noqa: E402

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

ARM = "right"
JOINT_ERROR_DEG = {
    "shoulder_pan.pos": 1.5,
    "elbow_flex.pos": -1.0,
    "wrist_flex.pos": 2.0,
    "wrist_roll.pos": 1.0,
}


def true_q(model, reading):
    return model.joint_vector({ARM: {k: v + JOINT_ERROR_DEG.get(k, 0.0) for k, v in reading.items()}})


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF).with_wrist_camera(ARM, _with_pose(TRUE_CAMERA))


def test_a_wrist_scan_measures_the_cameras_real_motion_and_the_table(model, tmp_path):
    joints, _, _ = start_joints(model, ARM)
    start_q = true_q(model, {**joints, "gripper.pos": 30.0})
    camera = model.wrist_camera_pose(ARM, start_q)
    table_z = model.frame_pose(f"{ARM}_base_link", model.joint_vector())[2, 3] - TABLE_BELOW_BASE_M
    look = (
        camera[:3, 3] + (table_z - camera[2, 3]) / camera[2, 2] * camera[:3, 2]
    )  # under the start's optical axis
    texture = table_texture()

    def wrist(arm, state):
        pose = model.wrist_camera_pose(arm, true_q(model, state[f"{arm}_arm"]))
        return render(texture, pose, TRUE_CAMERA, (float(look[0]), float(look[1])), float(table_z))

    fake = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json", wrist_renderer=wrist)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with FelixClient(address, timeout_s=10.0) as client:
            with client.session():
                client.move_joints({f"{ARM}_arm": {**joints, "gripper.pos": 30.0}}, tol=0.5)
                client.keep_holding()
            h = api.Harness(ask=lambda spec, preview: True, client=client, model=model)
            scan = h.call("wrist_scan", ARM, method="light")
            truth = [model.wrist_camera_pose(ARM, true_q(model, v.arm_pose)) for v in scan.views]
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.close()
    assert scan.labels[0] == "start" and scan.labels[-1] == "back" and scan.current == len(scan.views) - 1
    assert len(scan.views) == 1 + 4 + 2 + 1  # start, four rolls, two flexes, back
    model_turns, model_dirs, _ = motion_errors(scan.sfm.model_root_from_camera, truth)
    turns, dirs, _ = motion_errors(scan.sfm.root_from_camera, truth)
    assert scan.sfm.rms_after_px < 1.0, scan.describe()
    moved = np.array([np.linalg.norm((np.linalg.inv(truth[0]) @ t)[:3, 3]) for t in truth[1:]]) > 0.002
    assert turns.max() < 0.4 and turns.max() < model_turns.max(), (model_turns, turns)
    assert dirs[moved].max() < 3.0, (model_dirs, dirs)  # the moves' directions (the frame back has none)
    back = np.linalg.inv(scan.sfm.root_from_camera[0]) @ scan.sfm.root_from_camera[-1]
    assert (
        np.linalg.norm(back[:3, 3]) < 0.001
    )  # the wrist came back where it started (the fake repeats exactly)
    assert scan.sfm.plane is not None, scan.describe()
    normal, offset = scan.sfm.plane
    assert abs(offset / normal[2] - table_z) < 0.01, (
        offset / normal[2],
        table_z,
    )  # the model's scale: within 1 cm
    assert "over it" in scan.describe()
    image = h.call("render_scan", scan)
    assert image.shape[1] == 3 * 320
