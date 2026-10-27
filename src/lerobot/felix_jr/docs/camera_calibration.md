# Camera calibration plan

**Goal:** draw proposed arm trajectories into the head camera's RGB image accurately enough for a VLM to judge
them. The XLeRobot has loose tolerances (3D-printed parts, servo backlash, a hand-measured RealSense mount), so
a point computed from the URDF and the joint angles can land some distance from where the camera actually sees
it.

**Plan:** measure before calibrating. Three steps decide what to calibrate and how; the calibration itself comes
after.

1. **Sensitivity analysis (URDF only):** how many pixels each length or angle error costs.
2. **Repeatability test (hardware):** the error floor that no calibration can remove.
3. **Choose the main path:** see the camera's pose from tags on the robot each frame, or calibrate the head
   chain, or both.

Rough scale to keep in mind (estimates, not measurements): at 0.5 m from the camera, 1 cm is about 12 px
(color fx = 611 px), and a 1 deg error in the head tilt shifts the whole image by about 10 px. A VLM's own
pointing is probably tens of pixels, so the target may be around 5 mm, not sub-millimeter.

## Notation

Drake / Tedrake monogram notation, used in code too (`X_BC`, `p_BT_B`), because sign and ordering mistakes are
the usual calibration bugs:

- `p_AB_C`: position of point B relative to point A, expressed in frame C. `p_AB` means `p_AB_A`.
- `R_AB`: rotation of frame B in frame A; it takes vectors in B's coordinates to A's: `v_A = R_AB v_B`.
- `X_AB`: pose (rotation + translation) of frame B in frame A. Adjacent subscripts cancel:
  `X_AC = X_AB X_BC`, `p_AT = X_AB p_BT`.

Frames: `B` robot base (`top_base_link` in the URDF), `N` neck (`head_pan_link`), `E` eyes (`head_tilt_link`),
`C` color camera (optical frame: x right, y down, z forward), `T` a target point on the arm (a tag corner or the
tool point), `A` the arm's end link.

The camera sees `T` at `p_CT = X_CB p_BT`, and its pixel is `K p_CT` (intrinsics `K`, then divide by depth).
Two routes give `X_BC` and `p_BT`:

- head chain: `X_BC = X_BN(q_neck) X_NE(q_eyes) X_EC`
- arm chain: `p_BT = X_BA(q_arm) p_AT`

## What we already have

- **`xlerobot.urdf`:** the whole tree (base, both arms, head pan and tilt), so `X_BC` and `p_BT` can be
  computed in one frame. Findings so far:
  - It does not mention the RealSense. The head camera is a generic massless `head_camera_link`, 2.5 cm forward
    and 3 cm up from `head_tilt_link`. **The measured RealSense mount offset is exactly what is missing.**
  - Use `head_camera_link`'s x axis as the view direction. The URDF's `head_camera_rgb_frame` /
    `head_camera_depth_frame` rotation (rpy 0, -pi/2, -pi/2) puts their optical axis sideways at rest, so it
    looks wrong.
  - Joint conventions: a positive pan turns left (counter-clockwise from above), a positive tilt looks down.
    `setup/head.py` records the servo signs to match.
- **RealSense factory calibration** (read by `diagnostics/realsense_check.py`, at 640x480): color intrinsics
  fx 611.0, fy 610.0, principal point (315.9, 247.0), zero distortion; depth intrinsics; the depth-to-color
  transform (15.1 mm sideways). These describe the camera itself, not where it sits on the robot. Verify the
  color intrinsics once with a checkerboard (reprojection error should be well under 1 px), then freeze them.
- **Joint readings:** measured servo positions (4096 ticks per turn, about 0.09 deg). Use measured angles, not
  commanded ones: under gravity the commanded and real angles differ by a few degrees (see `tuning_notes.md`),
  but the encoder reports where the joint actually is.

## Step 1: Sensitivity analysis (URDF only)

**Question:** which parameters move the projected arm the most, and how wrong is the projection with no
calibration at all?

**Method:**

- Forward kinematics in plain numpy from the URDF's joint origins and axes (no meshes needed).
- A set of representative configurations: head poses that look at the workspace, and arm poses across the
  reach where trajectories will be (a table-height workspace in front of the robot, to start; easy to change).
- Target points on the arm: a point near the arm base, the wrist, and the tool tip.
- For each parameter, perturb it, re-project every target point and record the pixel shift. Parameters: each
  joint's zero offset (neck, eyes, the arm joints), each link length, and the 6-DoF camera mount offset
  relative to `head_tilt_link`.

**Assumed tolerances** (guesses; the per-unit results below do not depend on them):

| Parameter | Assumed 1-sigma |
|---|---|
| Joint zero offset | 2 deg |
| Link length | 3 mm |
| Camera mount position | 5 mm |
| Camera mount rotation | 2 deg |

**Outputs:**

1. Pixels per degree / per millimeter for each parameter, averaged over the configurations, with the worst
   case. This ranks the parameters.
2. A Monte Carlo run: perturb everything at once by the assumed tolerances and report the distribution of total
   pixel error at each target point. This is "how bad is it uncalibrated".
3. Split by target point, to see whether head errors (which move everything in the image) or arm errors (which
   move only the arm) dominate.

**Cannot tell us:** whether the assumed tolerances are realistic, or anything about backlash and play. That is
Step 2.

### Step 1 results on the default URDF

Run: `uv run python -m lerobot.felix_jr.calibration.sensitivity --out <report.md>` (code in `calibration/`,
full report in [sensitivity_default_urdf.md](sensitivity_default_urdf.md)). Default `xlerobot.urdf`, so the
camera sits on the generic `head_camera_link`, not the measured RealSense mount; rerun with `--urdf` once the
new URDF exists. Left arm, 54 configurations (head tilted 40 to 69 deg down, pan -14 to +14 deg, random arm
poses with the gripper in view).

- **Geometry:** the head camera can only see the arm close up. The jaw pivot was 0.16 to 0.51 m from the
  camera (median 0.35 m), and the arm only appears in a band about 25 to 45 cm in front of the robot. The
  shoulder is never in view (it is beside and below the camera), and the elbow only in 15 % of the poses.
- **Uncalibrated error, all tolerances at once** (jaw pivot): mean 55 px, 95th percentile 113 px. Head-chain
  parameters alone: mean 45 px; arm-chain parameters alone: mean 26 px. At 0.35 m, 55 px is about 3 cm.
  These totals depend on the guessed tolerances; the per-unit numbers below do not.
- **Biggest single parameters** (px at the assumed tolerance): head tilt zero 24 (12 px/deg), camera mount yaw
  24 and pitch 22 (about 11 px/deg), head pan zero 16, the arm's shoulder rotation zero 16 (8 px/deg) and
  shoulder pitch zero 12. Every position error costs about 1.6 to 1.9 px/mm (fx / 0.35 m).
- **So:** the head joint zeros and the camera mount rotation matter most, which fits option A or a head-first
  calibration stage. Arm joint zeros come next; link lengths matter least (about 3 px at 2 mm).
- **Identifiability check:** moving the head tower and moving the arm mount by the same amount give exactly
  the same pixel shifts (a unit test asserts this). A calibration can only find their difference, so fit one
  of them, not both.
- **Checks on the numbers:** a head-tilt error costs about 12 px/deg, close to the 10.7 px/deg (fx times 1 deg
  in radians) a pure camera rotation should cost; a translation costs about fx / depth per meter.

**Tooling note.** LeRobot has a generic `RobotKinematics` (`lerobot/model/kinematics.py`, built on placo) that
the SpaceMouse teleop uses for the SO-101 arm, but nothing specific to the XLeRobot. For this analysis we use
our own small numpy forward kinematics (`calibration/urdf_kinematics.py`): placo loads the meshes, which this
URDF references but we do not have, and the calibration needs to perturb joint origins and mount rotations,
which `RobotKinematics` does not expose (only joint angles). For runtime FK and IK on the real robot,
placo / `RobotKinematics` remain the natural choice once meshes are available.

## Step 2: Repeatability test (hardware)

**Question:** how far does the same commanded pose land from itself? This is the floor: constant offsets
cannot fix play, backlash or hysteresis.

**Method:**

- One fiducial tag (ArUco or AprilTag) fixed to a rigid part of the robot body in the head camera's view, e.g.
  the base or an arm's base.
- Head: command the same pan/tilt many times, arriving from different directions (from left and right, from
  above and below). Record the tag's corner pixels each time, with the head stationary.
- Arm: a tag on a rigid link near the tool (the fixed jaw or wrist, not the moving jaw). With the head still,
  command the same arm pose from different directions and record the tag's pixels.
- Also log the measured joint angles each time, to separate "the servo reports a different angle" from "the
  servo reports the same angle but the link is somewhere else" (play after the encoder).

**Outputs:** pixel scatter (RMS and worst case) for head and arm repeatability, and whether the error depends on
the approach direction. If it does, trajectories should approach their poses consistently, or per-frame
correction (Step 3, option A) becomes more attractive.

## Step 3: Choose the main path

Decide with the numbers from Steps 1 and 2.

**Option A: camera pose from tags on the robot, every frame.** Tags on rigid parts of the robot that the head
camera usually sees (base, arm bases). Each frame, PnP on the detected corners gives `X_CB` directly, so errors
in the neck, eyes and camera mount stop mattering. Only the arm chain still needs calibrating.

- Good when: head errors dominate Step 1 and head repeatability is poor in Step 2.
- Costs: tags must be in view; it fails when the head looks away from the base. Tag size matters: a 2 cm tag
  at 0.5 m is only about 24 px across, which gives poor depth and orientation; larger tags work better.

**Option B: calibrate the kinematic chain** (the original "fudge factor" idea). Each length and joint angle
becomes `URDF value + correction`, with the corrections pulled toward 0 by a prior. That is a maximum a
posteriori fit with Gaussian priors set from the assumed tolerances.

- Good when: errors are mostly constant offsets (Step 2 shows good repeatability).
- Costs: a calibration session, and identifiability care (below).

**Hybrid:** calibrate the chain (B) as the baseline, and use tags (A) to refine the camera pose whenever they are
visible.

## Calibration procedure (after Step 3)

Applies to Option B and to the arm part of Option A.

- **Residual: pixels, not depth.** Minimize the pixel distance between the predicted and the detected tag
  corners: predicted = `K X_CB(head params) X_BA(arm params) p_AT`. Corners are detected to a fraction of a
  pixel, and pixel error is what the VLM overlay needs. Depth from the D415 is the noisiest input (a few mm at
  0.5 m, growing quickly with distance), so use it at most as a weak extra term.
- **Parameters: a small set.** Joint zero offsets for every joint involved, the 6-DoF camera mount offset from
  `head_tilt_link`, the tag's pose on its link if it cannot be measured well, and link-length corrections only
  if Step 1 shows they matter. Many of these trade off against each other (a joint-zero error and a link offset
  can produce the same image), so fewer is better.
- **Stages.** Head first: tags fixed on the base, many head poses, solve the camera mount and the neck / eyes
  offsets. Then the arm: with the head part fixed, many arm poses, solve the arm offsets.
- **Pose design.** Spread poses over the ranges actually used, and approach them consistently. Check the
  singular values of the Jacobian and report each parameter's uncertainty from `J^T J`. A parameter whose
  estimate stays at its prior is not observable from the data.
- **Data capture.** Arm and head stationary for each sample; record the image and the measured joint angles
  together.
- **Solver.** A numpy forward-kinematics chain with the parameters as inputs, and
  `scipy.optimize.least_squares` with a robust loss (Huber) for occasional bad tag detections. No pinocchio or
  sklearn needed.
- **Validation.** Hold out some poses and report the pixel RMS before and after calibration. Add a physical
  check: touch the tool tip to a marked point on the table and compare the predicted pixel with the observed one.

## Open questions

- Where to put the tags: which rigid links are visible from the head poses we use, and what tag size fits.
- Whether `cv2.aruco` is available in this environment's OpenCV build.
- Measuring the RealSense mount relative to `head_tilt_link` (ruler and calipers for the prior; the calibration
  refines it).
- Moving camera capture to a Linux machine (Raspberry Pi) changes nothing in the math, but image and joint
  timestamps then come from two machines; for calibration this is fine as long as everything is stationary.
