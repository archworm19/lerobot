"""The server wiggle script against a fake robot behind a real local ZeroMQ socket."""

import argparse
import threading

import pytest

from lerobot.felix_jr.checks.server_wiggle import CENTER, run
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeArms, FakeHardware
from lerobot.felix_jr.server.server import RobotServer

HOME = FakeArms.HOME


def make_args(**overrides):
    args = {
        "arms": None,
        "delta": 10.0,
        "pause": 0.0,
        "max_vel": 60.0,
        "max_accel": 120.0,
        "tol": 1.5,
        "timeout": 10.0,
        "no_head": False,
        "no_base": False,
        "head_delta": 8.0,
        "wheel_speed": 40.0,
        "wheel_seconds": 0.3,
        "capture_dir": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.fixture
def robot(tmp_path):
    fake = FakeHardware(
        arm_time_scale=0.02,
        head_range_path=tmp_path / "range.json",
        session_timeout_s=0.6,
        watch_period_s=0.05,
    )
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=5.0) as client:
        yield fake, client
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def at_home(fake):
    return all(pose == HOME for pose in fake.arm_backend.pos.values())


def test_unknown_arms_and_missing_homes_are_refused_before_anything_moves(robot):
    fake, client = robot
    with pytest.raises(SystemExit, match="not on this server"):
        run(client, make_args(arms=["middle_arm"]), ask=lambda p: "", say=print)

    class NoHome(FakeArms):
        def homes(self):
            return {}

    fake.arm_backend.homes = NoHome(time_scale=0.02).homes
    with pytest.raises(SystemExit, match="no recorded home"):
        run(client, make_args(), ask=lambda p: "", say=print)
    assert fake.arm_backend.trajectories == []


def test_center_is_the_same_pose_as_the_local_wiggle_test():
    assert CENTER == {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
