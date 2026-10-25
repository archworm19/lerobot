# Scene tools: asking geometric questions about a color + depth capture

Helpers for the planning experiments (a VLM proposes a path from an RGB-D view; geometry checks it). They take
and return numpy arrays and work on batches. Code: `felix_jr/scene.py` (queries) and `felix_jr/overlay.py`
(drawing). Tests: `tests/felix_jr/`.

## Conventions

- Coordinates are in the **color camera's frame**: x right, y down, z forward (along the optical axis), meters.
  A transform to the robot base frame comes later (needs the hand-eye calibration, not done yet).
- Pixels are `(u, v)` = (column, row).
- A `Scene` holds the color image, depth **aligned to color** in meters (`NaN` = no depth, see the shadow
  bands in `realsense_m1.md`) and the color intrinsics. Build one with `Scene.from_capture(color, raw_depth,
  calib)` or, from a folder written by `realsense_check --save-dir`, `Scene.from_saved(folder)`.

## Functions

| Call | What it does |
|---|---|
| `scene.pixel_to_xyz(uv, window=5, min_valid=3, max_spread_m=0.10, standoff=0.0)` | `(..., 2)` pixels to `(..., 3)` points. Depth is the median of a patch. Returns NaN if the pixel is off-image, the patch has too few valid pixels, or it straddles a depth edge. `standoff` moves the point toward the camera along its ray. |
| `scene.xyz_to_pixel(xyz)` | `(..., 3)` to `(..., 2)`. NaN for points at or behind the camera. |
| `scene.in_image(uv)` | Which pixels fall inside the image. |
| `scene.segment_check(p1, p2, radius=0.02, tolerance=0.01, step=0.01)` | For N segments, sample every `step` m and give each sample FREE, UNKNOWN or BLOCKED. Returns `SegmentCheck(status, t_first, point_first, sample_xyz, sample_state)`. |
| `draw_points(image, uv, labels)` | Numbered markers. |
| `render_plan(scene, waypoints, check=None)` | The path over the photo: dots colored by state, numbered waypoints with xyz, each leg's length in meters, a legend. |

### What FREE / UNKNOWN / BLOCKED mean

A ball of `radius` meters is imagined at each sample and compared with the depth image around its pixel:

- **BLOCKED:** an observed surface lies within `radius + tolerance` of the sample's depth, inside a disc of the
  ball's apparent size. The ball touches something.
- **UNKNOWN:** cannot be said to be free: outside the image or behind the camera, the pixel has no depth (a hole
  or a shadow), or an observed surface is in front of the sample, which hides it. A depth camera does not see
  behind surfaces, so hidden space is unknown, not free.
- **FREE:** the observed surfaces around it are all farther than the ball: visible empty space.

A segment's status is the worst of its samples. `t_first` / `point_first` locate the first non-free sample.

Limits worth remembering: the test is done in the depth image (distances along the viewing ray), so it is an
approximation good to about `tolerance`; far points are noisier (D415 error grows roughly with distance
squared); and a point that sits **on** a surface is "touching" it, so start and goal points picked on a surface
should use `standoff` (a few cm) to be in the free space in front of it.

## Checked on a real frame (`/tmp/rs_check6`, 640x480, a hand holding a pen)

`pixel_to_xyz` gave plausible metric points: pen body 0.28 m from the camera, hand 0.30 m, box top 0.60 m,
headboard 0.89 m, far wall 2.83 m. The pen tip pixel returned NaN (no depth on that dark tip), which is the
intended behavior. Round trip `xyz_to_pixel(pixel_to_xyz(uv))` was exact for the valid points. A straight path
from in front of the pen to the box was reported blocked/unknown (it passes behind the hand), and the rendered
plan shows where along the path each state occurs. Timing on this machine: 5 pixel queries in a few
milliseconds; 200 random segments at radius 3 cm in about 0.6 s. Not tuned.

## Tests

```bash
uv run pytest tests/felix_jr -q       # needs pytest: uv sync --extra test (or uv pip install pytest)
```

53 tests, no camera or robot needed. `tests/felix_jr/conftest.py` builds a synthetic scene with known geometry (a
wall at 1.0 m, a box face at 0.6 m, a hole with no depth) that the expected values are computed from by hand.
They cover back-projection, shapes and batching, NaN handling, median and depth-edge rejection, projection
round trips, the three segment states (including near misses decided by `radius`, and chunking independence),
standoff, the depth alignment (identical cameras, a known sideways offset, nearest point wins, column-major
rotation), crack filling, edge detection, mask warping, and the overlay drawing.

Writing them found a real bug: a NaN pixel coordinate was cast to an integer, which is undefined and gives 0 on
Apple Silicon, so a NaN pixel was treated as the image corner and returned a valid depth. Non-finite inputs are
now rejected explicitly in both `pixel_to_xyz` and `segment_check`.

## Not built yet

- `clearance(p)`: distance to the nearest observed surface (a distance field over the point cloud).
- A top-down view with a metric grid, and height above the support plane.
- Iso-depth contour overlays with labeled depths.
- The transform to the robot base frame.
- The VLM harness: prompt, parse waypoints from pixel coordinates, lift with `pixel_to_xyz`, check, retry.
