"""Wrist camera identification and the saved-camera cache, with a scripted list of video devices (no hardware)."""

import argparse
import logging

import pytest

from lerobot.felix_jr.server.server import wrist_devices
from lerobot.felix_jr.setup.find_wrist_cameras import identify
from lerobot.felix_jr.wrist_cameras import (
    list_video_devices,
    load_wrist_cameras,
    resolve_wrist_camera,
    save_wrist_cameras,
)

LEFT = "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0"
RIGHT = "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0"


class Unplugger:
    """list_fn that drops `gone` after the first `ask`, and brings it back after `replug_after` more polls."""

    def __init__(self, present, gone, replug_after=2):
        self.present, self.gone, self.replug_after = present, gone, replug_after
        self.unplugged = False
        self.polls = 0

    def ask(self, prompt):
        self.unplugged = True
        return ""

    def __call__(self):
        if not self.unplugged:
            return list(self.present)
        self.polls += 1
        if self.polls > self.replug_after:
            return list(self.present)
        return [d for d in self.present if d not in self.gone]


def test_identify_finds_the_camera_that_vanishes():
    fake = Unplugger([LEFT, RIGHT], [RIGHT])
    said = []
    assert identify("right", fake.ask, said.append, list_fn=fake, sleep=lambda s: None) == RIGHT
    assert "right" in said[0] and "same USB port" in said[0]


def test_identify_rejects_nothing_or_several_unplugged_and_no_cameras():
    for gone in ([], [LEFT, RIGHT]):
        fake = Unplugger([LEFT, RIGHT], gone)
        with pytest.raises(RuntimeError, match="exactly one"):
            identify("left", fake.ask, print, list_fn=fake, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="no USB camera"):
        identify("left", lambda p: "", print, list_fn=list, sleep=lambda s: None)


def test_identify_times_out_if_not_replugged_into_the_same_port():
    fake = Unplugger([LEFT, RIGHT], [LEFT], replug_after=10**9)
    with pytest.raises(TimeoutError, match="same port"):
        identify("left", fake.ask, print, list_fn=fake, sleep=lambda s: None, replug_timeout=2.0)


def test_the_cache_and_resolve(tmp_path):
    path = tmp_path / "wrist_cameras.json"
    assert load_wrist_cameras(path) == {}
    save_wrist_cameras({"left": LEFT, "right": RIGHT}, path)
    assert load_wrist_cameras(path) == {"left": LEFT, "right": RIGHT}
    assert resolve_wrist_camera("left", path, present=[LEFT, RIGHT]) == LEFT
    with pytest.raises(LookupError, match="not plugged in"):
        resolve_wrist_camera("right", path, present=[LEFT])
    save_wrist_cameras({"left": LEFT}, path)
    with pytest.raises(KeyError, match="find_wrist_cameras"):
        resolve_wrist_camera("right", path, present=[LEFT, RIGHT])


def test_video_devices_are_the_capture_nodes(tmp_path):
    for name in ("usb-0:1:1.0-video-index0", "usb-0:1:1.0-video-index1", "usb-0:2:1.0-video-index0"):
        (tmp_path / name).touch()
    assert list_video_devices(tmp_path) == [
        str(tmp_path / "usb-0:1:1.0-video-index0"),
        str(tmp_path / "usb-0:2:1.0-video-index0"),
    ]
    assert list_video_devices(tmp_path / "missing") == []


def test_the_server_opens_the_given_cameras_else_the_saved_ones(monkeypatch, caplog):
    import lerobot.felix_jr.wrist_cameras as cameras

    def resolve(arm):
        if arm == "left":
            return LEFT
        raise LookupError(f"the {arm} wrist camera is not plugged in")

    monkeypatch.setattr(cameras, "resolve_wrist_camera", resolve)
    args = argparse.Namespace(left_wrist_camera=None, right_wrist_camera=None)
    with caplog.at_level(logging.WARNING):
        assert wrist_devices(args) == {"left": LEFT}  # the saved left one; the right one missing: left out
    assert "starting without it" in caplog.text
    given = argparse.Namespace(left_wrist_camera="2", right_wrist_camera=RIGHT)
    assert wrist_devices(given) == {"left": 2, "right": RIGHT}
