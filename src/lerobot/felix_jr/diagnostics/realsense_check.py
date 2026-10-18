"""Check that a RealSense camera (D415) can be opened and streams color + depth through LeRobot.

Stage 1 enumerates devices with librealsense directly. Stage 2 opens the first one with LeRobot's own
`RealSenseCamera`, reads color and depth for a few seconds, and reports shapes, measured frame rate and basic
depth statistics. With --save-dir it also writes a color PNG and a colorized depth PNG.

On Apple Silicon the camera usually can't be opened without root (macOS USB/UVC restrictions), and the
install is the prebuilt `pyrealsense2-macosx` wheel. Run it with the project's venv python, not `uv run`
(which would leave root-owned files in the cache), and without writing .pyc files:

  sudo env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m lerobot.felix_jr.diagnostics.realsense_check \\
      [--width 640] [--height 480] [--fps 30] [--seconds 3] [--save-dir DIR]
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--seconds", type=float, default=3.0, help="how long to stream (default: 3)")
parser.add_argument("--save-dir", help="write color.png and depth.png here")
args = parser.parse_args()
print = functools.partial(print, flush=True)  # the failure path below can crash the process on exit


def fail(message: str) -> None:
    """Print and leave without interpreter teardown: pyrealsense2's macOS build segfaults at exit when a
    device could not be opened, which would otherwise swallow the message."""
    print(message)
    os._exit(1)


root_note = "" if os.getuid() == 0 else " (not root; macOS may refuse USB access)"
print(f"Running as uid {os.getuid()}{root_note}")

# Stage 1: what does librealsense see?
try:
    cameras = RealSenseCamera.find_cameras()
except Exception as e:
    fail(
        f"Could not read the RealSense device: {e!r}\n"
        "'failed to set power state' means macOS blocked USB access: rerun with sudo (see the docstring)."
    )
if not cameras:
    fail("No RealSense device found. Check the USB-C cable/port (use a USB 3 port), then replug.")
for cam in cameras:
    print(
        f"Found: {cam['name']} | serial {cam['id']} | firmware {cam['firmware_version']} | "
        f"USB {cam['usb_type_descriptor']} | product id {cam['product_id']}"
    )
if not str(cameras[0]["usb_type_descriptor"]).startswith("3"):
    print("WARNING: the camera is on a USB 2 link; high resolutions/frame rates will likely fail.")

# Stage 2: stream through LeRobot's camera class
config = RealSenseCameraConfig(
    serial_number_or_name=cameras[0]["id"],
    fps=args.fps,
    width=args.width,
    height=args.height,
    use_depth=True,
)
camera = RealSenseCamera(config)
camera.connect()
try:
    color = depth = None
    frames = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        color = camera.async_read(timeout_ms=1000)  # blocks until a new color frame arrives
        depth = camera.read_latest_depth()
        frames += 1
    elapsed = time.perf_counter() - t0
    print(f"Color: shape {color.shape}, dtype {color.dtype}; depth: shape {depth.shape}, dtype {depth.dtype}")
    fps = frames / elapsed
    print(f"Delivered {frames} color frames in {elapsed:.1f} s = {fps:.1f} fps (asked {args.fps})")
    valid = depth > 0
    print(
        f"Depth: {100 * valid.mean():.0f}% of pixels valid; of those, min {depth[valid].min()} / "
        f"median {np.median(depth[valid]):.0f} / max {depth[valid].max()} raw units "
        "(D415 default is 1 raw unit = 1 mm)"
    )
    # What is needed to turn (pixel, depth) into metric 3D points. LeRobot returns raw z16 depth that is NOT
    # aligned to the color image, so the depth-to-color extrinsics matter.
    profile = camera.rs_profile
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    streams = {
        name: profile.get_stream(kind).as_video_stream_profile()
        for name, kind in (("color", rs.stream.color), ("depth", rs.stream.depth))
    }

    def intrinsics(stream) -> dict:
        i = stream.get_intrinsics()
        return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model), "coeffs": list(i.coeffs)}

    ext = streams["depth"].get_extrinsics_to(streams["color"])
    calibration = {
        "depth_scale_m_per_unit": depth_scale,
        "color_intrinsics": intrinsics(streams["color"]),
        "depth_intrinsics": intrinsics(streams["depth"]),
        "depth_to_color_rotation_row_major": list(ext.rotation),
        "depth_to_color_translation_m": list(ext.translation),
    }
    ci, di = calibration["color_intrinsics"], calibration["depth_intrinsics"]
    shift_mm = np.round(np.array(ext.translation) * 1000, 1)
    print(f"Depth scale: {depth_scale} m per raw unit")
    print(f"Color fx/fy {ci['fx']:.1f}/{ci['fy']:.1f}; depth fx/fy {di['fx']:.1f}/{di['fy']:.1f}")
    print(f"Depth-to-color translation: {shift_mm} mm (depth is not aligned to color)")
    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        scale = 255.0 / max(1, np.percentile(depth, 98))
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=scale), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "depth.png"), depth_vis)
        (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
        if os.getuid() == 0 and os.environ.get("SUDO_UID"):  # don't leave root-owned files behind
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ.get("SUDO_GID", "-1"))
            for p in (out, out / "color.png", out / "depth.png", out / "calibration.json"):
                os.chown(p, uid, gid)
        print(f"Saved color.png, depth.png and calibration.json to {out}")
finally:
    camera.disconnect()
