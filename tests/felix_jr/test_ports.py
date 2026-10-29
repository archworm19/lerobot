"""Port identification and the saved-port cache, with a scripted list of USB devices (no hardware)."""

import pytest

from lerobot.felix_jr.ports import PortInfo, load_ports, resolve, save_ports
from lerobot.felix_jr.setup.find_ports import identify

LEFT = [PortInfo("/dev/cu.usbmodemL", "L", "0-1"), PortInfo("/dev/tty.usbmodemL", "L", "0-1")]
RIGHT = [PortInfo("/dev/cu.usbmodemR", "R", "0-2"), PortInfo("/dev/tty.usbmodemR", "R", "0-2")]


class Unplugger:
    """list_fn that drops `adapter` after the first `ask`, and brings it back after `replug_after` more polls."""

    def __init__(self, present, adapter, replug_after=2):
        self.present, self.adapter, self.replug_after = present, adapter, replug_after
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
        return [p for p in self.present if p not in self.adapter]


def test_identify_finds_the_adapter_that_vanishes_and_prefers_the_tty_name():
    fake = Unplugger(LEFT + RIGHT, RIGHT)
    said = []
    info = identify("right_arm", fake.ask, said.append, list_fn=fake, sleep=lambda s: None)
    assert (info.device, info.serial_number, info.location) == ("/dev/tty.usbmodemR", "R", "0-2")
    assert "right_arm" in said[0]


def test_identify_rejects_nothing_or_several_unplugged():
    none = Unplugger(LEFT + RIGHT, [])
    with pytest.raises(RuntimeError, match="exactly one"):
        identify("left_arm", none.ask, print, list_fn=none, sleep=lambda s: None)
    both = Unplugger(LEFT + RIGHT, LEFT + RIGHT)
    with pytest.raises(RuntimeError, match="exactly one"):
        identify("left_arm", both.ask, print, list_fn=both, sleep=lambda s: None)


def test_identify_times_out_if_not_replugged():
    fake = Unplugger(LEFT + RIGHT, LEFT, replug_after=10**9)
    with pytest.raises(TimeoutError):
        identify("left_arm", fake.ask, print, list_fn=fake, sleep=lambda s: None, replug_timeout=2.0)


def test_adapters_without_serial_number_are_identified_by_location():
    a = [PortInfo("/dev/ttyACM0", None, "1-1.2")]
    b = [PortInfo("/dev/ttyACM1", None, "1-1.3")]
    fake = Unplugger(a + b, b)
    info = identify("right_arm", fake.ask, print, list_fn=fake, sleep=lambda s: None)
    assert info.device == "/dev/ttyACM1"
    assert info.key == "location:1-1.3"


def test_save_load_and_resolve_follow_the_adapter_when_the_port_name_changes(tmp_path):
    path = tmp_path / "ports.json"
    save_ports({"left_arm": LEFT[1], "right_arm": RIGHT[1]}, path)
    assert load_ports(path)["left_arm"] == LEFT[1]
    # after a replug the left adapter has a different port name
    moved = [PortInfo("/dev/tty.usbmodemL2", "L", "0-1"), *RIGHT]
    assert resolve("left_arm", path, present=moved) == "/dev/tty.usbmodemL2"
    assert resolve("right_arm", path, present=moved) == "/dev/tty.usbmodemR"


def test_resolve_errors(tmp_path):
    path = tmp_path / "ports.json"
    with pytest.raises(KeyError, match="find_ports"):
        resolve("left_arm", path, present=[])
    save_ports({"left_arm": LEFT[1]}, path)
    with pytest.raises(LookupError, match="not plugged in"):
        resolve("left_arm", path, present=RIGHT)
    assert load_ports(tmp_path / "missing.json") == {}
