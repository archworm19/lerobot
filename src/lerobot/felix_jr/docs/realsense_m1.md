# RealSense D415 on an Apple Silicon Mac

What works, what doesn't, and what to watch for when using an Intel RealSense D415 with LeRobot on an M1 Mac
(macOS, arm64). Written from our own bring-up. **Verified** = seen on this machine. **Not verified** = reasoned
or taken from documentation, not tested here.

## Summary

- The camera streams **640x480 color + depth at 30 fps** through LeRobot's `RealSenseCamera`, **but only as
  root**. Plain user processes fail with `failed to set power state`.
- No source build of librealsense was needed. LeRobot's `intelrealsense` extra already pulls a prebuilt macOS
  wheel, `pyrealsense2-macosx`.
- The camera negotiated only a **USB 2.1** link. That is enough for 640x480 @ 30 fps but will limit higher
  modes.
- LeRobot's RealSense class returns **raw, unaligned depth** and does not expose the depth scale or camera
  intrinsics. Metric 3D from depth needs those (see "Getting metric depth to a VLM").

## What we ran (verified)

Install, into the project environment (the extra is `pyrealsense2-macosx>=2.54,<2.57.0` on macOS; we got
2.56.5 with Python 3.13 arm64):

```bash
uv pip install "pyrealsense2-macosx>=2.54,<2.57.0"      # or: uv sync --extra intelrealsense
```

Check (needs root, see "Pitfalls"). Use the venv's Python directly so root does not leave files in the uv cache
or the repo:

```bash
sudo env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m lerobot.felix_jr.diagnostics.realsense_check \
    --save-dir /tmp/rs_check
```

Result of the run:

| | |
|---|---|
| Device | Intel RealSense D415, serial 244222060589, firmware 5.15.1.55, product id 0AD3 |
| USB link | 2.1 |
| Color | 480x640x3 `uint8` |
| Depth | 480x640x1 `uint16`, raw `z16` |
| Frame rate | 91 frames in 3.0 s = 30.2 fps (asked 30) |
| Depth values | 89 % of pixels valid; min 376, median 969, max 13329 raw units |

We looked at the saved color and depth images and they agree with each other (near objects blue, the far wall
red, no-depth pixels dark blue at edges and on one side). The D415's minimum working distance is roughly
30 to 40 cm, which fits the 376 minimum. The 13 m maximum is stray far-range pixels.

`diagnostics/realsense_check.py` also prints, and with `--save-dir` writes to `calibration.json`, the depth
scale, color and depth intrinsics and the depth-to-color extrinsics. **That part has not been run yet** (it was
added after the first run).

## Pitfalls

1. **Root is required (verified).** Any process that opens the camera must run as root on this Mac. As a normal
   user, enumeration works (one device is listed) but the first `get_info` call raises
   `RuntimeError: failed to set power state`. This matches the macOS USB/UVC restrictions described in
   RealSense's docs. Everything that uses the camera directly, including teleop and recording, would need
   `sudo`.
2. **Crash on exit after a failed open (verified).** After that error the process segfaults (exit code 139)
   during interpreter shutdown, and stdout may be lost if it was not flushed. The check script flushes and
   leaves with `os._exit(1)` so the message survives.
3. **USB 2.1 link (verified).** The D415 supports USB 3 but linked at USB 2. Usual causes: a USB 2-only cable or
   adapter, or a USB 2 port or hub in the path. Higher resolutions and frame rates will likely fail until this
   is fixed. Try a direct USB-C 3.x data cable to the Mac and rerun the check, which reports the link.
4. **`sudo` and files (not verified).** We have not checked whether `sudo` on this Mac keeps your `HOME`, which is
   where calibration and home-pose files live (`sudo env | grep HOME`). Files written by a root run are
   root-owned; the check script hands its output back to the invoking user with `SUDO_UID`.
5. **No depth alignment, scale or intrinsics in LeRobot's class (verified by reading the code).**
   `camera_realsense.py` enables `z16` depth and `rgb8` color but never creates an `rs.align`, and does not
   return the depth scale or intrinsics. On the D415 the depth and color sensors have different fields of view
   and positions, so depth pixel (u, v) is not the color pixel (u, v).
6. **Depth units (not verified).** We assumed 1 raw unit = 1 mm (the D415 default). The check script now reads the
   real depth scale from the device; confirm it.
7. **Connection drops (not verified here).** RealSense's macOS notes say the stream can freeze and that
   unplugging and replugging the cable or power cycling the port is the usual fix. We have not hit it. IMU data
   is not available on macOS builds, which does not matter for the D415 (it has none).

## Camera server over ZMQ (plan, not built)

Because the camera needs root, the idea is a small server that runs as root and publishes frames to normal,
non-root clients over ZMQ. LeRobot already has the pattern, but the existing server is not a drop-in:

- `cameras/zmq/image_server.py` uses a PUB socket with `SNDHWM 20`, sends with `NOBLOCK`, and encodes color as
  base64 JPEG. It only wraps LeRobot's OpenCV camera, so it cannot serve RealSense or depth.
- `cameras/zmq/camera_zmq.py` (the client) uses a SUB socket with `CONFLATE` enabled, which keeps only the
  newest message, and a receive timeout.

**Bounding memory.** ZMQ queue limits count messages, not bytes:

- `SNDHWM` on the server and `RCVHWM` on the client cap queued messages. The ZMQ default is 1000, which at the
  frame size below would be over 1 GB per slow subscriber, so set it explicitly. With PUB/SUB, a subscriber that
  falls behind makes the server drop messages instead of blocking the camera.
- `CONFLATE` on the subscriber keeps only the latest message. It only works with single-part messages, so send
  color and depth together as one message, not as separate parts.
- Frame sizes at 640x480: color 921,600 B, depth 614,400 B, together about 1.5 MB per frame, about 46 MB/s at
  30 fps uncompressed. A depth of 3 messages is about 5 MB. At 1280x720 it is about 4.6 MB per frame.
- Compression reduces both memory and bandwidth. Color can be JPEG. **Depth must stay lossless** (16-bit PNG or
  raw plus zlib); JPEG would corrupt the metric values.

## Getting metric depth to a VLM (goal)

Long-term goal: test whether VLMs such as Qwen can outline a trajectory for the arm's tool point from an image
and a depth map. A colorized depth image is not metric, and VLMs read colorbars unreliably. Options:

1. **Let the VLM point in 2D and lift to 3D ourselves (recommended).** VLMs are comparatively good at marking
   pixel locations. Ask for waypoints as image coordinates, then back-project them with the depth and the
   intrinsics into camera coordinates, and transform to the robot base frame. The VLM never needs metric depth.
2. **Give it numbers.** Overlay a grid or object markers and attach the measured distance (meters) at each marker.
   Fewer points, but the model sees real values.
3. **Colorized depth with a labeled scale.** Cheapest, least precise.

Whichever is used, we need, in order:

- depth aligned to color (add `rs.align`, or use the depth-to-color extrinsics), which LeRobot's class does not
  do;
- the depth scale and intrinsics (`calibration.json` from the check script);
- a camera-to-robot-base transform. That is a hand-eye calibration we have not done: the extrinsics between the
  camera mount and the robot base frame (URDF base frame; x forward, y left, z up).

## Open items

- Run the check again for the depth scale, intrinsics and extrinsics (`calibration.json`).
- Get a USB 3 link and recheck the link type, frame rates and higher resolutions.
- Confirm that `sudo` keeps `HOME`, or decide to run only the camera server as root.
- Build the RealSense ZMQ server (color + aligned depth in one message, small `SNDHWM`, depth lossless) and a
  matching client.
- Decide where the camera sits relative to the arm base and do the hand-eye calibration.
