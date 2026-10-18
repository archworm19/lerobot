# RealSense D415 on an Apple Silicon Mac

What works, what doesn't, and what to watch for when using an Intel RealSense D415 with LeRobot on an M1 Mac
(macOS, arm64). Written from our own bring-up. **Verified** = seen on this machine. **Not verified** = reasoned
or taken from documentation, not tested here.

## Summary

- The camera streams **640x480 color + depth** through LeRobot's `RealSenseCamera` (30 fps once, 15 fps on a
  later run), **but only as root**. Plain user processes fail with `failed to set power state`.
- It is **flaky**: several runs in a row failed in different ways before one worked (pitfall 7). Plan for
  retries and replugging.
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

Results:

| | |
|---|---|
| Device | Intel RealSense D415, serial 244222060589, firmware 5.15.1.55, product id 0AD3 |
| USB link | 2.1 |
| Color | 480x640x3 `uint8` |
| Depth | 480x640x1 `uint16`, raw `z16` |
| Frame rate, run 1 | 91 frames in 3.0 s = 30.2 fps (asked 30) |
| Frame rate, later run | 46 frames in 3.0 s = 15.3 fps (asked 15) |
| Depth values | 89 to 92 % of pixels valid; min 376 to 397, median 969 to 986, max about 12 to 13 m |

We looked at the saved color and depth images and they agree with each other (near objects blue, the far wall
red, no-depth pixels dark blue at edges and on one side). The D415's minimum working distance is roughly
30 to 40 cm, which fits the 376 minimum. The 13 m maximum is stray far-range pixels.

`diagnostics/realsense_check.py` also prints, and with `--save-dir` writes to `calibration.json`, the camera's
own calibration. Values read from this camera at 640x480 (the intrinsics depend on the stream resolution, so
re-read them if you change it):

| | |
|---|---|
| Depth scale | 0.001 m per raw unit (so 1 unit = 1 mm) |
| Color intrinsics | fx 611.0, fy 610.0, principal point (315.9, 247.0), model inverse Brown-Conrady, all distortion coefficients 0 |
| Depth intrinsics | fx 606.7, fy 606.7, principal point (323.1, 246.6), all distortion coefficients 0 |
| Depth to color | rotation almost identity (off-diagonal terms below 0.001), translation (15.1, -0.1, -0.2) mm. librealsense stores the rotation **column-major**; the first saved `calibration.json` mislabeled it `..._row_major`, and `depth.py` reads both labels as column-major |

So the depth sensor sits about 15 mm to one side of the color sensor. That offset is small but not zero, and the
two focal lengths and principal points differ, so depth pixels do not line up with color pixels without
alignment.

## Aligning depth to color

Terminology: mapping the depth map into the color camera's pixel grid is **alignment** (also called
**registration**: "depth registered to color"; RealSense's own tool is `rs.align`). **Rectification** is a
different step, re-projecting a stereo pair so matching points share an image row; the D415 does that internally
to compute depth. Removing lens distortion is a third, separate step (the D415 reports zero distortion here).

`felix_jr/depth.py` does the alignment with the saved `calibration.json`, in numpy, because LeRobot's camera class
only hands back raw arrays, not RealSense framesets: depth pixel -> 3D point (depth intrinsics) -> color frame
(extrinsics) -> color pixel (color intrinsics), nearest point wins, then one-pixel cracks from the resampling
are filled where the neighbors agree. The same calibration is what a later pixels-to-meters step needs.

### Why it is not a plain shift

Alignment is a 3D re-projection, so each pixel moves by an amount that depends on its own depth (parallax):

shift in pixels along x is about `fx * 0.0151 / Z - 7.1`, where 0.0151 m is the sensor offset and 7.1 px is the
difference between the two principal points (`ppx` color 315.9, depth 323.1).

| Distance Z | 0.28 m (the hand in our test) | 0.4 m | 1.0 m | 1.5 m | 3 m |
|---|---|---|---|---|---|
| shift | about 26 px | about 16 px | about 2 px | about -1 px | about -4 px |

So near objects slide a lot and the far wall barely moves. Consequences:

- Neighboring pixels at different depths end up in different places: a near object slides over the background,
  and gaps (no depth) and new borders appear. The depth image's gradients are therefore *not* just the old ones
  moved; they are recomputed on a different arrangement of pixels. To see "the same edges, moved", `depth.py`
  finds the edges once on the as-is depth and then moves those pixels with their own depth.
- The D415 leaves a band of missing depth next to near objects (a stereo shadow), so strong depth jumps are
  often hidden behind invalid pixels in the as-is map.
- Depth edges are marked on the nearer of the two pixels (the object's outline), since that is what shifts.
- A far, noisy wall produces many small centimeter-level gradients; edges therefore use a median filter and a
  10 cm minimum step.

### What the check writes

`diagnostics/realsense_check.py --save-dir DIR` writes `alignment.png` and `alignment_zoom.png`;
`diagnostics/alignment_from_saved.py DIR` regenerates them from the saved `color.png`, `depth_raw.png` and
`calibration.json` without the camera or root (handy when changing `depth.py`).

`alignment.png` is a 2x2 panel. Top row: depth as it comes from the camera. Bottom row: aligned to color.

- **Left column, green lines:** depth jumps of 10 cm or more painted on the color image. Top, where they sit in
  the depth image; bottom, the same edges moved to their place in the color image. What to look for: in the
  bottom row the green should sit on the object outlines in the photo (here: the top of the hand and pen, the
  edge of the box, the headboard top). In the top row the lines of *near* objects float off the object.
- **Right column, colored overlay:** the depth map colored near = blue, far = red, black/gray where there is no
  depth, drawn 50 % transparent over the photo, with the same color scale in both rows. What to look for: whether
  the colored shapes line up with the objects underneath. Where depth spills past an object's outline onto the
  background, or stops short of it, the image is not aligned. The fringe is biggest around near objects.
  Compare the same object in the top and bottom panel.

`alignment_zoom.png` stacks three 3x enlargements of the area around the nearest objects (the biggest shift):
color only, depth as-is, depth aligned. Compare the finger and pen outlines in the color crop with the colored
shapes below it: as-is, blue patches spill off to one side of the hand; aligned, they sit on it.

### Verification and caveats

- **Synthetic scene (verified):** a box in front of a wall rendered from both cameras with the real calibration.
  Aligned depth matched the ground truth in the color frame for 99.9 % of filled pixels within 15 mm and 99.7 %
  of color pixels received depth.
- **Real frame (verified visually, 640x480, hand holding a pen at about 0.28 m, wall at 1.5 to 3 m):** after
  alignment the depth edges follow the outline of the hand and pen, where as-is they do not.
- **A number was tried and dropped.** We first printed the share of depth-edge pixels within 2 px of a color
  edge, then the mean distance to the nearest color edge. On the real scene they barely moved or even got worse
  depending on the choices (edge threshold, which pixel of a pair is marked), because depth edges from stereo are
  blurry and a color photo is full of edges. They were measuring noise, so we removed them. The synthetic scene
  with known ground truth is the quantitative test; the real-scene check is visual.
- The nearest-point-wins warp and one-pixel crack fill are simple. Occlusion shadows stay empty (no depth), which
  is correct, not a bug.

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
6. **Depth units (verified).** The device reports a depth scale of 0.001 m per raw unit, so raw values are
   millimeters.
7. **Opening is unreliable, even as root (verified, cause not established).** The first run as root worked.
   Later attempts failed in different ways: `failed to set power state` at `pipeline.start` under `sudo`; a
   segfault right after the device was listed; a start where the read thread did not terminate and warmup
   failed (`read failed (status=False)`), then a segfault; and finally enumeration itself failing as root. The
   camera stayed listed on USB throughout (its registry id changed once, so it re-enumerated) and no stray
   RealSense process was running. Observations:
   - Once a run has crashed, later runs tend to fail until the camera is unplugged and replugged. Treat a
     replug as part of recovery after any failure (not yet confirmed to restore the first-run behavior).
   - **Possible cause (inferred, untested): the USB 2.1 link is saturated.** 640x480 @ 30 fps color plus depth
     is about 18 MB/s each if both travel at roughly 2 bytes per pixel, so about 37 MB/s, near what USB 2
     delivers in practice (roughly 35 to 40 MB/s). The first run may have been lucky. Test with `--fps 15`,
     a smaller size, or one stream at a time, and fix the link with a USB 3 cable and port.
   - `LRS_LOG_LEVEL=DEBUG` (set inside `sudo env ...`) makes librealsense print its own log, which would show
     where a failing start dies.
   - What worked: after three failed runs, the fourth run (with `--fps 15`, no replug mentioned) opened and
     streamed normally. Whether the lower frame rate or simply trying again made the difference is not known, so
     the bandwidth theory above is still unconfirmed.
   - After a failed open the process segfaults when the failed camera object is torn down, so an in-process retry
     never runs. The check script therefore retries by re-executing itself in a fresh process (`--retries`, default
     3, 2 s apart) and says to replug if all attempts fail.
8. **Connection drops (not verified here).** RealSense's macOS notes say the stream can freeze and that
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

- Get a USB 3 link and recheck the link type; then test whether 30 fps and larger resolutions open reliably.
- Find out whether 15 fps really is more reliable than 30 on the USB 2 link, or whether the successful run was
  luck.
- Confirm that `sudo` keeps `HOME`, or decide to run only the camera server as root.
- Build the RealSense ZMQ server (color + aligned depth in one message, small `SNDHWM`, depth lossless) and a
  matching client.
- Decide where the camera sits relative to the arm base and do the hand-eye calibration.
