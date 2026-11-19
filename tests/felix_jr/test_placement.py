"""`harness.placement`: an object's outline on the head's live picture, the head turned to it, and spots to place it."""

import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api, placement  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402


def test_the_rings_are_the_cylinders_circles_on_the_table_and_at_its_top():
    rings = placement.bottle_rings((-0.3, 0.1), 0.0225, 0.13, 0.77)
    assert set(rings) == {"bottom", "top"} and rings["bottom"].shape == (placement.RING_POINTS, 3)
    assert np.allclose(np.hypot(rings["top"][:, 0] + 0.3, rings["top"][:, 1] - 0.1), 0.0225)
    assert np.allclose(rings["bottom"][:, 2], 0.77) and np.allclose(rings["top"][:, 2], 0.90)


def test_the_head_turns_to_the_spot_and_the_capture_after_enter_has_the_outline_in_view(tmp_path):
    """Against the fake robot: the head aimed at the spot (its picture's center near the outline), the outline drawn
    on each live frame, Enter on the second: a full capture returned; q: None."""
    hardware = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json")
    server = RobotServer(hardware)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = XleRobotModel.load(api.MODEL)
    base = model.frame_pose("right_base_link", model.joint_vector())[:3, 3]
    xy, table_z = (float(base[0] - 0.28), float(base[1])), float(base[2] - 0.06)
    shown, keys = [], iter([-1, 13])
    try:
        with FelixClient(address, timeout_s=10.0) as client, client.session():
            capture = placement.place_object(
                client,
                model,
                xy,
                0.0225,
                0.13,
                table_z,
                "spot 1",
                show=lambda p: (shown.append(p), next(keys))[1],
            )
            stopped = placement.place_object(
                client, model, xy, 0.0225, 0.13, table_z, show=lambda p: ord("q")
            )
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.close()
    assert capture is not None and stopped is None and len(shown) == 2
    center = capture.project(np.array([*xy, table_z + 0.065]))
    w, h = capture.scene.color.shape[1], capture.scene.color.shape[0]
    assert abs(center[0] - w / 2) < w / 4 and abs(center[1] - h / 2) < h / 4  # (the head aimed at it)
    assert not np.array_equal(shown[0], capture.scene.color)  # (the outline drawn on the live frames)
