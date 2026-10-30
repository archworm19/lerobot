"""A whole calibration session against the fake robot: views, clicks where a "true" model sees the pinch point,
refits, and the final calibration recovers the truth."""

import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.collect import CollectSession  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

FAKE_INTRINSICS = {"fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0, "width": 320, "height": 240}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 2048, "head_eyes": 2048}
TRUE = {
    "zero:head_tilt_joint": np.radians(33.0),
    "zero:head_pan_joint": np.radians(-6.0),
    "rot:fixed_head_camera_link:z": np.radians(3.0),
    "pos:fixed_head_camera_link:y": 0.02,
    "pos:fixed_Base:y": 0.02,
    "zero:left_elbow_flex": np.radians(-3.0),
}


@pytest.fixture
def robot(tmp_path):
    fake = FakeHardware(arm_time_scale=0.02, head_range_path=tmp_path / "range.json", session_timeout_s=10.0)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=10.0) as client:
        yield fake, client
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def test_a_session_from_a_rough_start_calibrates_the_head_and_the_arm(robot, tmp_path):
    fake, client = robot
    truth = XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=FAKE_INTRINSICS)
    truth.set_estimates(TRUE)
    rough = XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=FAKE_INTRINSICS)
    rough.set_estimates({"zero:head_tilt_joint": np.radians(28.0)})  # e.g. an old calibration, 5 degrees off
    rough.save(tmp_path / "start.urdf")
    rng = np.random.default_rng(0)
    with client.session():
        session = CollectSession(
            client,
            "left",
            tmp_path / "cal",
            lambda image: None,
            start_from=tmp_path / "start.urdf",
            n_targets=25,
        )
        first_miss = None
        for _ in range(25):
            session.handle("next")
            q = session.current_q()
            seen = truth.predict("left", q)
            if seen.z > 0 and truth.in_view(seen.uv, margin=5) and seen.visibility > 0:
                if first_miss is None:
                    first_miss = np.linalg.norm(session.model.predict("left", q).uv - seen.uv)
                session.click(*(seen.uv + rng.normal(0, 0.5, 2)))
        last_miss = np.linalg.norm(session.model.predict("left", q).uv - seen.uv)
        path = session.finish(holdout_every=5)
    assert len(session.session.clicks) >= 15, "too few views showed the pinch point"
    assert first_miss > 10 and last_miss < 3  # the circle converges on the pinch point
    calibrated = XleRobotModel.load(path)
    held = calibrated.fit_info["holdout"]["cal"]
    assert held["after"] < 2.0 < held["before"]
    found = calibrated.estimate_dict()
    assert found["zero:head_tilt_joint"] == pytest.approx(TRUE["zero:head_tilt_joint"], abs=np.radians(3.0))
    assert calibrated.floor_z is not None and calibrated.fit_info["sessions"] == ["cal"]
