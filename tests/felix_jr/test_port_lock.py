"""One process at a time on a serial port, and the follower's retry of a glitched configure."""

import subprocess
import sys
import textwrap

import pytest

from lerobot.felix_jr.port_lock import PortInUseError, acquire_port_lock, lock_path, release_port_lock
from lerobot.robots.so_follower.so_follower import SOFollower
from tests.felix_jr.test_base_follower import FakeBus, make_robot  # noqa: I001


def test_a_held_port_is_refused_until_it_is_released_and_other_ports_are_free(tmp_path):
    first = acquire_port_lock("/dev/ttyACM1", tmp_path)
    with pytest.raises(PortInUseError, match=r"/dev/ttyACM1 is in use by another process \(pid \d+\)"):
        acquire_port_lock("/dev/ttyACM1", tmp_path)
    other = acquire_port_lock("/dev/ttyACM0", tmp_path)  # a different port is independent
    release_port_lock(first)
    again = acquire_port_lock("/dev/ttyACM1", tmp_path)
    release_port_lock(again)
    release_port_lock(other)
    release_port_lock(None)  # ignored
    assert lock_path("/dev/ttyACM1", tmp_path).name == "felix_jr_dev_ttyACM1.lock"


def test_another_process_holding_the_port_is_refused_and_a_dead_one_releases_it(tmp_path):
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from lerobot.felix_jr.port_lock import acquire_port_lock
                acquire_port_lock("/dev/ttyACM9", Path({str(tmp_path)!r}))
                print("held", flush=True)
                time.sleep(60)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(PortInUseError, match=f"pid {holder.pid}"):
            acquire_port_lock("/dev/ttyACM9", tmp_path)
    finally:
        holder.kill()
        holder.wait()
    release_port_lock(acquire_port_lock("/dev/ttyACM9", tmp_path))  # the OS dropped the dead process's lock


def test_the_follower_holds_the_port_from_connect_to_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.felix_jr.base_follower.acquire_port_lock", lambda port: acquire_port_lock(port, tmp_path)
    )
    monkeypatch.setattr(SOFollower, "connect", lambda self, calibrate=True: None)
    monkeypatch.setattr(SOFollower, "disconnect", lambda self: None)
    robot, rival = make_robot(tmp_path), make_robot(tmp_path)
    robot.connect()
    robot.connect()  # connecting again does not trip over its own lock
    with pytest.raises(PortInUseError):
        rival.connect()
    robot.disconnect()
    rival.connect()  # free again
    rival.disconnect()


def test_a_failed_connect_gives_the_port_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.felix_jr.base_follower.acquire_port_lock", lambda port: acquire_port_lock(port, tmp_path)
    )

    def failing(self, calibrate=True):
        raise ConnectionError("no servo")

    monkeypatch.setattr(SOFollower, "connect", failing)
    robot = make_robot(tmp_path)
    with pytest.raises(ConnectionError):
        robot.connect()
    monkeypatch.setattr(SOFollower, "connect", lambda self, calibrate=True: None)
    make_robot(tmp_path).connect()  # not blocked by the failed attempt


class GlitchyBus(FakeBus):
    """Fails the first `failures` writes with the error seen on the Pi."""

    def __init__(self, motors, failures):
        super().__init__(motors)
        self.failures = failures

    def write(self, register, motor, value, **kwargs):
        if self.failures:
            self.failures -= 1
            raise ConnectionError(
                "Failed to write 'Lock' on id_=2 with '0' after 1 tries. [TxRxResult] Incorrect status packet!"
            )
        super().write(register, motor, value, **kwargs)


def test_configure_retries_a_bus_glitch_and_gives_up_after_a_few(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.felix_jr.base_follower.time.sleep", lambda s: None)
    robot = make_robot(tmp_path, with_head=True)
    robot.bus = GlitchyBus(robot.bus.motors, failures=2)
    robot.configure()  # two glitches, then success
    assert robot.bus.failures == 0 and robot.bus.writes("Operating_Mode")

    robot = make_robot(tmp_path)
    robot.bus = GlitchyBus(robot.bus.motors, failures=10)
    with pytest.raises(ConnectionError, match="Incorrect status packet"):
        robot.configure()
    assert robot.bus.failures == 10 - 3  # tried three times
