"""The server wiggle script against a fake robot behind a real local ZeroMQ socket."""

import argparse
import json
import threading

import pytest

from lerobot.felix_jr.checks.server_wiggle import run
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


def test_the_whole_wiggle_runs_and_ends_at_rest(robot, tmp_path):
    fake, client = robot
    fake.head.record("head_eyes", "home")  # so the head goes home first
    fake.head.move("head_eyes", delta_deg=7.0)
    lines, prompts = [], []

    def ask(prompt):
        prompts.append(prompt)
        return ""

    assert run(client, make_args(capture_dir=tmp_path / "frames"), ask=ask, say=lines.append)
    text = "\n".join(lines)
    assert at_home(fake)
    assert fake.arm_backend.violations == 0
    assert fake.head.backend.read_ticks("head_eyes") == 2048  # its recorded home
    assert "head_eyes: going to home" in text and "head_eyes: back at home" in text
    assert "WARNING: no home recorded for head_neck" in text
    assert text.count("read back") == 2 * 6  # +/- for each of the six joints
    assert (
        "both backward" in text and "WARNING: left wheel" not in text and "WARNING: right wheel" not in text
    )
    assert text.rstrip().endswith("Done; the robot is at rest.")
    assert len(prompts) == 2 and "Wheels" in prompts[1]
    assert fake.arm_backend.wheels == (0.0, 0.0)
    # it started at the first prompt's pose, went to center, and read each +/- 10 deg back
    assert "shoulder_pan.pos: +10.0" in text and "gripper.pos: +60.0" in text

    log = [json.loads(line) for line in (tmp_path / "frames" / "log.jsonl").read_text().splitlines()]
    assert log[0]["label"] == "center" and log[-1]["label"] == "head"
    assert all((tmp_path / "frames" / entry["image"]).is_file() for entry in log)
    assert len(log) >= 1 + 12 + 6 + 1 + 1  # center, 12 nudges + 6 returns, home, head


def test_options_choose_the_arms_and_skip_the_head_and_wheels(robot):
    fake, client = robot
    lines = []
    assert run(
        client,
        make_args(arms=["left_arm"], no_head=True, no_base=True, delta=5.0),
        ask=lambda prompt: "",
        say=lines.append,
    )
    text = "\n".join(lines)
    assert "[right_arm] Start pose" not in text and "[left_arm] Start pose" in text
    assert "head_eyes" not in text and "wheel" not in text
    # 19 wiggle moves and the explicit return home touch only the left arm; the rest at the end covers both
    touched = [set(targets) for targets, *_ in fake.arm_backend.trajectories]
    assert touched[:20] == [{"left_arm"}] * 20 and touched[-1] == {"left_arm", "right_arm"}


def test_the_wheel_prompt_can_be_skipped(robot):
    fake, client = robot
    lines = []
    assert run(client, make_args(no_head=True), ask=lambda p: "s" if "Wheels" in p else "", say=lines.append)
    assert "commanded" not in "\n".join(lines)


def test_ctrl_c_part_way_through_sends_the_robot_to_rest(robot):
    fake, client = robot
    asked = []

    def ask(prompt):
        asked.append(prompt)
        if "Wheels" in prompt:
            raise KeyboardInterrupt
        return ""

    with pytest.raises(KeyboardInterrupt):
        run(client, make_args(), ask=ask, say=lambda line: None)
    assert at_home(fake)
    assert fake.arm_backend.wheels == (0.0, 0.0)
