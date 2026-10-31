"""Server + client round trips against the fake hardware over a real local ZeroMQ socket (no robot needed)."""

import threading

import numpy as np
import pytest

from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware, add_librealsense_to_path
from lerobot.felix_jr.server.protocol import PROTOCOL_VERSION, ServerError, decode_image, encode_image
from lerobot.felix_jr.server.server import RobotServer


@pytest.fixture
def endpoint():
    server = RobotServer(FakeHardware())
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield address
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


@pytest.fixture
def client(endpoint):
    with FelixClient(endpoint, timeout_s=5.0) as c:
        yield c


def test_hello(client):
    info = client.hello()
    assert info["protocol"] == PROTOCOL_VERSION
    assert info["arms"] == ["left_arm", "right_arm"]
    assert info["camera"] is True
    assert isinstance(info["commit"], str)
    assert client.server_info == info


def test_get_state(client):
    state = client.get_state()
    assert state["moving"] is False
    assert set(state["left_arm"]) == {
        f"{j}.pos"
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    }


def test_capture_round_trip_is_lossless_with_png(client):
    expected = FakeHardware().capture()
    snap = client.capture()
    np.testing.assert_array_equal(snap.color, expected.color)
    np.testing.assert_array_equal(snap.raw_depth, expected.raw_depth)
    assert snap.raw_depth.dtype == np.uint16
    assert snap.calibration == expected.calibration
    assert snap.state["moving"] is False
    assert snap.timestamp > 0


def test_capture_jpeg_is_smaller_and_close(client):
    png = encode_image(FakeHardware().capture().color, "png")
    jpeg = encode_image(FakeHardware().capture().color, "jpeg")
    assert len(jpeg) < len(png)
    snap = client.capture(color_format="jpeg")
    assert snap.color.shape == (240, 320, 3)
    assert np.abs(snap.color.astype(int) - FakeHardware().capture().color).mean() < 5


def test_capture_gives_a_usable_scene(client):
    scene = client.capture().scene()
    center = scene.pixel_to_xyz(np.array([[160, 120]]))
    assert center[0, 2] == pytest.approx(0.6, abs=0.01)  # the box in the middle of the fake scene


def test_unknown_command_is_an_error_and_the_server_survives(client):
    with pytest.raises(ServerError, match="unknown command"):
        client._call("fly")
    assert client.hello()["protocol"] == PROTOCOL_VERSION


def test_bad_arguments_are_an_error_and_the_server_survives(client):
    with pytest.raises(ServerError):
        client._call("get_state", nonsense=1)
    assert client.get_state()["moving"] is False


def test_unreachable_server_times_out():
    with pytest.raises(TimeoutError):
        FelixClient("tcp://127.0.0.1:9", timeout_s=0.3)


def test_client_recovers_after_a_timeout(endpoint):
    with FelixClient(endpoint, timeout_s=5.0) as c:
        with pytest.raises(TimeoutError):
            c._call("hello", timeout_s=0.0)  # reply can't arrive in time
        assert c.get_state()["moving"] is False  # the late hello reply must not be mistaken for this one


def test_image_codec_round_trips():
    depth = np.random.default_rng(0).integers(0, 65535, (8, 6), dtype=np.uint16)
    np.testing.assert_array_equal(decode_image(encode_image(depth)), depth)
    with pytest.raises(ValueError):
        encode_image(depth, "jpeg")


def test_add_librealsense_to_path(tmp_path, monkeypatch):
    import sys

    module_dir = tmp_path / "build" / "Release"
    module_dir.mkdir(parents=True)
    (module_dir / "pyrealsense2.cpython-312-aarch64-linux-gnu.so.2.55").touch()
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert add_librealsense_to_path(tmp_path / "build") == module_dir
    assert str(module_dir) in sys.path
    with pytest.raises(FileNotFoundError):
        add_librealsense_to_path(tmp_path / "empty")


def test_server_check_reports_and_saves_a_loadable_capture(client, tmp_path):
    from lerobot.felix_jr.checks.server_check import run
    from lerobot.felix_jr.harness.scene import Scene

    lines = []
    durations = run(client, captures=3, save_dir=tmp_path / "cap", say=lines.append)
    assert len(durations) == 3
    assert any("fake=True" in line for line in lines)
    scene = Scene.from_saved(tmp_path / "cap")
    assert scene.pixel_to_xyz(np.array([[160, 120]]))[0, 2] == pytest.approx(0.6, abs=0.01)
