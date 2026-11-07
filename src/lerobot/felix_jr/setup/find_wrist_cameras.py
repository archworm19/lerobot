"""Find which USB camera is each arm's wrist camera, and remember it (`felix_jr/wrist_cameras.py`). Run on the Pi.

For each arm in turn, you unplug that arm's wrist camera; the script sees which camera vanished, waits for you to
plug it back into the same port, then takes a frame from it to check it works. The result is saved, and the server
opens the saved cameras by itself.

  uv run python -m lerobot.felix_jr.setup.find_wrist_cameras                    # left, then right
  uv run python -m lerobot.felix_jr.setup.find_wrist_cameras --arms right       # just one
  uv run python -m lerobot.felix_jr.setup.find_wrist_cameras --show             # print the saved cameras
  uv run python -m lerobot.felix_jr.setup.find_wrist_cameras --snapshot-dir ~/wrist_check   # keep the test frames

A camera is known by its USB port (`wrist_cameras.py`): keep each camera in the port it was found in. Stop the server
first (it holds the cameras open).
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lerobot.felix_jr.wrist_cameras import (
    ARMS,
    WRIST_CAMERAS_FILE,
    list_video_devices,
    load_wrist_cameras,
    resolve_wrist_camera,
    save_wrist_cameras,
)


def identify(
    arm: str,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    list_fn: Callable[[], list[str]] = list_video_devices,
    sleep: Callable[[float], None] = time.sleep,
    replug_timeout: float = 60.0,
) -> str:
    """`arm`'s wrist camera: the one device that disappears when the person unplugs it.

    Waits for it to come back (into the same port: the same name) before returning, so the next arm starts from a
    full list.
    """
    before = list_fn()
    if not before:
        raise RuntimeError(
            "no USB camera found (nothing in /dev/v4l/by-path): plug the wrist cameras into the Pi"
        )
    ask(f"Unplug the {arm} arm's wrist camera (only that one), then press Enter. ")
    sleep(1.0)  # let the kernel drop the device
    gone = sorted(set(before) - set(list_fn()))
    if len(gone) != 1:
        raise RuntimeError(
            f"expected exactly one camera to disappear, saw {len(gone)} ({gone}). Check that only one camera was "
            "unplugged."
        )
    (device,) = gone
    say(f"{arm}: {device}. Plug it back into the same USB port.")
    waited = 0.0
    while device not in list_fn():
        if waited >= replug_timeout:
            raise TimeoutError(
                f"the {arm} wrist camera did not come back within {replug_timeout:.0f} s (into the same port?)"
            )
        sleep(0.5)
        waited += 0.5
    sleep(1.0)  # let it finish enumerating
    return device


def grab_frame(device: str) -> np.ndarray:
    """One RGB frame from `device`, through the server's own camera reader (`UsbCamera`)."""
    from lerobot.felix_jr.server.hardware import UsbCamera

    camera = UsbCamera(device)
    try:
        frame, _ = camera.get()
    finally:
        camera.stop()
    return frame


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arms", nargs="+", choices=ARMS, default=list(ARMS), help="which arms' cameras (left right)"
    )
    parser.add_argument("--show", action="store_true", help="print the saved cameras and exit")
    parser.add_argument(
        "--snapshot-dir", type=Path, help="save each camera's test frame here (wrist_<arm>.jpg)"
    )
    args = parser.parse_args()

    if args.show:
        saved = load_wrist_cameras()
        if not saved:
            print(f"nothing saved yet ({WRIST_CAMERAS_FILE})")
        for arm in saved:
            try:
                status = f"present: {resolve_wrist_camera(arm)}"
            except LookupError as e:
                status = f"NOT PLUGGED IN ({e})"
            print(f"{arm}: {status}")
        return

    print(f"USB cameras present: {list_video_devices() or 'none'}")
    saved = load_wrist_cameras()
    for arm in args.arms:
        saved[arm] = identify(arm, input, print)
        frame = grab_frame(saved[arm])
        print(
            f"  {arm}: a {frame.shape[1]} x {frame.shape[0]} frame, mean brightness {frame.mean():.0f} (0-255)"
        )
        if args.snapshot_dir is not None:
            import cv2

            args.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = args.snapshot_dir / f"wrist_{arm}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"  saved {path}: check it shows the {arm} gripper's view")
    if len(set(saved.values())) != len(saved):
        raise SystemExit("two arms ended up with the same camera; nothing saved")
    save_wrist_cameras(saved)
    print(f"\nSaved to {WRIST_CAMERAS_FILE}:")
    for arm, device in saved.items():
        print(f"  {arm}: {device}")


if __name__ == "__main__":
    main()
