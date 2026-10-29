"""Which serial port belongs to which arm, remembered across runs.

`setup/find_ports.py` fills the cache (by asking you to unplug each arm in turn). The cache stores, per robot id,
the USB device's serial number (or, if it has none, its USB location) next to the port name seen at the time.
`resolve` looks the arm up again by that identity, so it still works if the port name changed (e.g. `/dev/ttyACM0`
becoming `/dev/ttyACM1` on the Pi after a replug).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lerobot.utils.import_utils import require_package

PORTS_FILE = HF_LEROBOT_CALIBRATION / "felix_jr" / "ports.json"


@dataclass(frozen=True)
class PortInfo:
    """One USB serial device: its port name and what identifies the physical adapter."""

    device: str
    serial_number: str | None
    location: str | None

    @property
    def key(self) -> str:
        """The device's identity: its USB serial number, or (without one) its USB port location."""
        return f"serial:{self.serial_number}" if self.serial_number else f"location:{self.location}"


def list_serial_ports() -> list[PortInfo]:
    """USB serial devices currently present (pyserial).

    On macOS each adapter shows up as both `/dev/cu.*` and `/dev/tty.*`; both are listed.
    """
    require_package("pyserial", extra="feetech", import_name="serial")
    from serial.tools import list_ports

    return [
        PortInfo(p.device, p.serial_number, p.location)
        for p in list_ports.comports()
        if p.vid is not None  # skips built-in non-USB ports
    ]


def preferred_device(candidates: list[PortInfo]) -> str:
    """The port name to use for one adapter.

    That is the `/dev/tty.*` name on macOS (what the existing scripts use), otherwise the first one.
    """
    for info in candidates:
        if info.device.startswith("/dev/tty."):
            return info.device
    return candidates[0].device


def save_ports(ports: dict[str, PortInfo], path: Path = PORTS_FILE) -> None:
    """Write `{robot_id: PortInfo}` to the cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        robot_id: {"device": p.device, "serial_number": p.serial_number, "location": p.location}
        for robot_id, p in ports.items()
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_ports(path: Path = PORTS_FILE) -> dict[str, PortInfo]:
    """Read the cache file; empty if there is none."""
    if not path.is_file():
        return {}
    return {robot_id: PortInfo(**fields) for robot_id, fields in json.loads(path.read_text()).items()}


def resolve(robot_id: str, path: Path = PORTS_FILE, present: list[PortInfo] | None = None) -> str:
    """The current port name of `robot_id`'s adapter, found by its remembered identity.

    Raises `KeyError` if the id was never saved, and `LookupError` if its adapter is not plugged in now.
    """
    saved = load_ports(path)
    if robot_id not in saved:
        raise KeyError(f"no saved port for {robot_id!r}; run setup.find_ports (known: {sorted(saved)})")
    present = list_serial_ports() if present is None else present
    matches = [p for p in present if p.key == saved[robot_id].key]
    if not matches:
        raise LookupError(
            f"{robot_id}'s USB adapter ({saved[robot_id].key}) is not plugged in; "
            f"present: {sorted({p.key for p in present})}"
        )
    return preferred_device(matches)
