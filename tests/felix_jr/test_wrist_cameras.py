"""Wrist camera identification, the saved-camera cache, video device numbers and reopening a camera whose frames
stop, with scripted video devices (no hardware)."""

import argparse
import logging
import os

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


def test_video_devices_are_one_name_per_usb_camera(tmp_path):
    """As on the Pi: each USB camera under a usb- and a usbv2- name (one device), its metadata node, the Pi's own
    codec node, and the RealSense's nodes."""
    devices = tmp_path / "dev"
    devices.mkdir()
    for n in range(8):
        (devices / f"video{n}").touch()
    by_path = tmp_path / "by-path"
    by_path.mkdir()
    for name, target in [
        ("platform-xhci-hcd.0-usb-0:1.1:1.0-video-index0", "video0"),
        ("platform-xhci-hcd.0-usbv2-0:1.1:1.0-video-index0", "video0"),
        ("platform-xhci-hcd.0-usb-0:1.1:1.0-video-index1", "video1"),
        ("platform-xhci-hcd.0-usb-0:1.2:1.0-video-index0", "video2"),
        ("platform-xhci-hcd.0-usbv2-0:1.2:1.0-video-index0", "video2"),
        ("platform-1000800000.codec-video-index0", "video3"),
        ("platform-xhci-hcd.1-usbv3-0:1:1.0-video-index0", "video4"),  # the RealSense: only a usbv3- name
    ]:
        (by_path / name).symlink_to(devices / target)
    assert list_video_devices(by_path) == [
        str(by_path / "platform-xhci-hcd.0-usb-0:1.1:1.0-video-index0"),
        str(by_path / "platform-xhci-hcd.0-usb-0:1.2:1.0-video-index0"),
        str(by_path / "platform-xhci-hcd.1-usbv3-0:1:1.0-video-index0"),
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


def test_a_by_path_link_opens_by_its_video_number(tmp_path, monkeypatch):
    from lerobot.felix_jr.server.hardware import video_index

    assert video_index(3) == 3
    link = tmp_path / "platform-xhci-hcd.0-usb-0:1.1:1.0-video-index0"
    link.symlink_to("/dev/video7")  # no /dev/video7 here: its target is faked as there
    monkeypatch.setattr(
        "os.path.realpath", lambda path: "/dev/video7" if str(path) == str(link) else str(path)
    )
    with monkeypatch.context() as patched:
        real_exists = os.path.exists
        patched.setattr("os.path.exists", lambda path: str(path) == str(link) or real_exists(path))
        assert video_index(str(link)) == 7
    with pytest.raises(ValueError, match="no camera at"):  # the link is gone: unplugged, or another port
        video_index(str(tmp_path / "not-a-camera"))
    (tmp_path / "a-file").touch()
    with pytest.raises(ValueError, match="not a video device"):
        video_index(str(tmp_path / "a-file"))


def test_a_camera_whose_frames_stop_is_reopened(monkeypatch):
    """A USB drop (bowl4's wrist camera, 2026-11-07): the reader reopens the camera and frames come again."""
    import time

    import cv2
    import numpy as np

    from lerobot.felix_jr.server import hardware

    opened: list = []

    class FakeCapture:
        def __init__(self, *args):
            self.frames = 3 if not opened else 10**9  # the first one stops after 3 frames
            opened.append(self)

        def isOpened(self):  # noqa: N802 (OpenCV's name)
            return True

        def set(self, *args):
            return True

        def read(self):
            if self.frames <= 0:
                return False, None
            self.frames -= 1
            time.sleep(0.005)
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(hardware, "REOPEN_AFTER_S", 0.05)
    camera = hardware.UsbCamera(0)
    try:
        deadline = time.time() + 2.0
        while camera.reopened == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert camera.reopened == 1 and len(opened) == 2
        frame, stamp = camera.get(after=time.time())  # frames again, from the reopened camera
        assert frame.shape == (4, 4, 3)
    finally:
        camera.stop()
