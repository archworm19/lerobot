"""Which USB camera is which arm's wrist camera, remembered across runs.

`setup/find_wrist_cameras.py` fills the cache (by asking you to unplug each camera in turn). Generic USB camera boards
often have no serial number (or the same one), so a camera is known by the USB port it is plugged into: its
`/dev/v4l/by-path/...-video-index0` name (Linux, the Pi), stable across reboots and replugs into the same port,
unlike `/dev/videoN`. Move a camera to another port and it must be found again. The server opens the saved cameras
(`server.server`, unless told otherwise).
"""

import json
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

WRIST_CAMERAS_FILE = HF_LEROBOT_CALIBRATION / "felix_jr" / "wrist_cameras.json"
BY_PATH = Path("/dev/v4l/by-path")
ARMS = ("left", "right")


def list_video_devices(by_path: Path = BY_PATH) -> list[str]:
    """The USB cameras present, each by its port's stable name.

    That is the `...-video-index0` entry (the capture node; a camera also shows a metadata node, `-video-index1`).
    Empty where there is no `/dev/v4l/by-path` (not Linux, or no camera).
    """
    if not by_path.is_dir():
        return []
    return sorted(str(p) for p in by_path.iterdir() if p.name.endswith("-video-index0"))


def save_wrist_cameras(cameras: dict[str, str], path: Path = WRIST_CAMERAS_FILE) -> None:
    """Write `{arm: device}` to the cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cameras, indent=2) + "\n")


def load_wrist_cameras(path: Path = WRIST_CAMERAS_FILE) -> dict[str, str]:
    """Read the cache file; empty if there is none."""
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text()))


def resolve_wrist_camera(arm: str, path: Path = WRIST_CAMERAS_FILE, present: list[str] | None = None) -> str:
    """`arm`'s wrist camera device, as saved.

    Raises `KeyError` if it was never saved, and `LookupError` if it is not plugged in (into the same port) now.
    """
    saved = load_wrist_cameras(path)
    if arm not in saved:
        raise KeyError(
            f"no saved wrist camera for the {arm} arm; run setup.find_wrist_cameras (known: {sorted(saved)})"
        )
    present = list_video_devices() if present is None else present
    if saved[arm] not in present:
        raise LookupError(
            f"the {arm} wrist camera ({saved[arm]}) is not plugged in (into that USB port); present: {present}"
        )
    return saved[arm]
