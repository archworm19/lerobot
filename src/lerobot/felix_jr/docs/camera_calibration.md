# Camera calibration

**Goal:** draw proposed arm trajectories into the head camera's RGB image accurately enough for a VLM to judge
them, which means a few pixels at the gripper's pinch point. The XLeRobot has loose tolerances (3D-printed parts,
servo backlash, a hand-measured RealSense mount), so a point computed from the URDF and the joint angles can land
hundreds of pixels from where the camera actually sees it.

**Approach:** a person clicks the pinch point of the closed gripper in the camera image, view after view, and a
least-squares fit finds the corrections to the URDF (joint zero offsets, joint and mount origins, the camera mount,
the pinch point's offset) that make the model's predicted pixel agree with the clicks. One model,
`XleRobotModel`, is the single source of truth for the kinematics, the corrections and their uncertainty, the camera
and the pinch points; the fit and the choice of views are built on it. Kinematics come from pinocchio, the
optimization from `scipy.optimize.least_squares`.

**Run it:** [README.md](README.md), section 6. In short, `calibration.collect` per arm (each refits with the other
arm's clicks), or `calibration.xle_fit SESSION ... --out calibration.urdf` to refit any set of sessions. The result
is one file, `calibration.urdf`: a URDF any tool can load, which also carries what a refit needs.

**Results so far:** the first two sessions (40 clicks per arm, collected with version 1, see "History"), fitted
together with one head: held-out pixel error 2.07 px (left) and 2.05 px (right), from about 500 px uncalibrated,
with click noise of about 2 to 3 px. Head tilt zero +33.5 deg (+/-2.1), pan zero -9.3 deg (+/-3.5), the camera's
lateral offset +23.7 mm (+/-5.5) (as suspected from the RealSense adapter), camera roll +0.6 and yaw +3.0 deg, both
arm mounts by millimeters to centimeters, the elbow zeros by -2 to -3.5 deg, and the pinch point offsets.

## Code map

| File | What |
|---|---|
| `calibration/merge_urdf.py` | builds `xlerobot_so101.urdf` (committed): the XLeRobot's head and base with two SO-101 arms |
| `calibration/xle_robot_model.py` | `XleRobotModel`: kinematics, parameter table, estimates, pinch points, derivatives, saving |
| `calibration/xle_fit.py` | sessions of clicks, the fit, held-out validation, the report; CLI to refit sessions |
| `calibration/views.py` | `find_view` (head and arm joints that show a pinch point), `plan_targets` |
| `calibration/collect.py` | the interactive session: views, clicks, refits, `calibration.urdf` at the end |

Tests: `tests/felix_jr/test_xle_robot_model.py` (including agreement with version 1's frozen predictions,
`reference/v1_reference.json`), `test_xle_robot_model_vs_placo.py`, `test_xle_fit.py`, `test_views.py`,
`test_calibration_collect.py`, and `smoke_tests/test_calibration_collect_smoke.py` (a whole session against the
fake robot, clicking where a "true" model sees the pinch point; the calibration recovers it).

## Notation and frames

Drake / Tedrake monogram notation, in code too (`X_AB`, `p_AB_C`), because sign and ordering mistakes are the usual
calibration bugs: `p_AB_C` is the position of point B relative to A, expressed in frame C (`p_AB` means `p_AB_A`);
`R_AB` the rotation of frame B in A (`v_A = R_AB v_B`); `X_AB` the pose of B in A, and `X_AC = X_AB X_BC`.

Frames: `W` the robot's root (the URDF's `world` link; the chassis is 0.38 m above it, robot forward is -x, +y the
robot's right), `C` the color camera's optical frame (`head_camera_optical`: x right, y down, z forward), `T` a pinch
point (`<arm>_gripper_frame_link`). The camera sees `T` at `p_CT = X_CW p_WT`, at pixel `K p_CT / z`.

## The robot model

### The merged URDF

`merge_urdf.py` builds `xlerobot_so101.urdf` once, offline:

- from `xlerobot.urdf`: the head, camera, base and wheels; its root `joint_0` made `fixed` (pinocchio and placo add
  their own free base, and a second floating joint scrambles placo's joint indexing: `set_joint` then moves
  nothing); the mesh-only `*_geom_*` links and the `head_camera_rgb_frame` / `head_camera_depth_frame` links (their
  rpy puts the optical axis sideways) dropped; meshes, transmissions and materials removed (we do not have the
  meshes; only the tree is needed);
- two copies of the official SO-101 URDF (`so101_new_calib.urdf`), every link and joint prefixed `left_` / `right_`,
  attached to the arm bases `Base` / `Base_2` by fixed `<arm>_arm_mount` joints;
- a fixed `head_camera_optical` frame under `head_camera_link`, whose x axis is the view direction.

**Why the SO-101 URDF for the arms.** The arm inside `xlerobot.urdf` is an older gripper (its wrist roll joint lacks
the SO-101's 18 mm lateral offset, and its shoulder offsets differ by about 13 mm), while the hardware is an SO-101;
and the SO-101 URDF's joint angles are LeRobot's calibrated degrees one to one (the teleop relies on this), so no
sign or offset is guessed for the arm. The mount transform is the rigid fit of the SO-101's shoulder, elbow and
wrist pivots onto those of the old arm over random poses (residual about 8 mm, the shoulder geometry difference).
The XLeRobot URDF's pan zero points the arm sideways where LeRobot's points it forward (left arm +90 deg, right arm
-90); checked against the wiggle test's camera frames (a +10 deg pan moves the arms to the right in the image, and
at the center pose the tip is about 35 cm forward and 23 cm up from the arm base). The mount is then calibrated like
everything else.

A test checks the committed file is what the script builds; another, that the model reproduces the predictions of
version 1 (which joined the two URDFs at runtime) to about 1e-6 px, with and without corrections.

### Parameters, nominal values and estimates

Parameters are corrections on top of the URDF, one row each (`Param`: name, prior mean and sigma, group, arm):

- `zero:<joint>`: added to the joint's angle. For the head, the unknown angle at its recorded home; for an arm
  joint, the servo's calibration offset.
- `pos:<joint>:x|y|z`: added to the joint origin's translation, in the parent frame.
- `rot:<joint>:x|y|z`: a rotation vector `w`: the origin's rotation becomes `R_urdf exp(w)`. Unlike roll-pitch-yaw
  (version 1), its Jacobian is exact at any value and the order of combined corrections does not matter.

`default_parameters()`: the head and camera once (both head joints' zeros and origins, the camera mount's position
and rotation: 14), then per arm the mount's position and rotation, each SO-101 joint's zero and origin, and the
pinch point's offset (29); 72 in all. Priors (`SIGMAS`): head zeros 20 deg (the head's home is not level), head
joint origins 2 mm, camera mount 3 mm in x and z and **15 mm laterally** (the RGB sensor's position in the camera
link is the least certain), 2 deg in rotation; arm mounts 15 mm and 5 deg; arm joint zeros 2 deg and origins 2 mm;
pinch point offset 8 mm. The head tower is not a parameter: moving it and moving the arm mount shift every pixel
identically, so only one of them can be found, and the mount is it. Most joint origins stay at their priors (the
clicks do not move the pinch point enough to see them); the fit's report says which parameters were learned.

`XleRobotModel` builds the pinocchio model from the URDF once and keeps it as `nominal_model`, never modified.
`pin_model` is a copy (`model.copy()` is deep) with the estimates applied. **Every application starts from the URDF
values**, so hundreds of updates during a fit cannot drift, and zero estimates give back the URDF exactly.

**Fixed joints.** Pinocchio folds a fixed joint into a frame and pre-multiplies its origin into the placements of
everything below it, so editing that frame alone moves nothing below it (checked: editing the `fixed_Base` frame
does not move the arm). Several important parameters sit on fixed joints: the camera mount, the arm mounts, the
pinch points. So the model records, for every joint and frame placement pinocchio holds, the chain of URDF origins
it is the product of (the fixed joints from the nearest moving joint down, then the joint itself), and rebuilds
every placement whose chain contains a corrected joint. `check_structure()` verifies the chains reproduce
pinocchio's placements, and the predictions are off by about 40 px if either joint or frame placements are not
rebuilt.

**Explicit estimates.** Every method that depends on the estimates takes them (`theta`: a vector in parameter order
or a `{name: value}` dict; `None` for the current ones) and applies them before computing. Nothing depends on what
an earlier call left in the model; there is no caching. Applying rebuilds 28 placements in about 0.17 ms, so the
batch methods (`predict_many`, `pixel_jacobian_many`) apply once for a whole set of clicks.

**Joints.** `q` is pinocchio's configuration vector of *measured* angles: `joint_vector(arms, head_ticks)` takes
LeRobot's arm degrees and the head servos' raw ticks, through the servo map `angle = sign * (ticks - ref) * 2 pi /
4096` (`sign` and `ref`, the recorded home, from `setup.head`); the `zero:` corrections are added inside. So a
click's `q` serves every `theta`. Measured, not commanded: under gravity the two differ by a few degrees (see
`tuning_notes.md`), and the encoder reports where the joint is.

### Pinch points and visibility

The assumptions specific to this robot, hence the name:

- **Pinch point:** `<arm>_gripper_frame_link`, the SO-101's tool center point, about 98 mm along the closed jaws;
  its offset (the `pos:` parameters of `<arm>_gripper_frame_joint`) is calibrated. `pinch_point(arm, q)` is the one
  place that says where it is; `predict`, `visibility`, the fit and the view finder go through it.
- **Visibility:** the gripper hides its own pinch point when the camera looks along the plane the jaws open in.
  `visibility(arm, q)` is the signed cosine between the jaw hinge axis (the SO-101 `gripper` joint's axis) and the
  line of sight; near +/-1 the camera sees the jaws meet from one side. At wiggle-like poses with the wrist roll near
  0 the hinge axis is 75 to 86 deg off the line of sight, which is why the first sessions showed the gripper but
  not its tip.
- **Visible side:** the wrist camera sits on the negative side of both grippers, and its board hides the pinch point
  from there (all 40 first left-arm clicks were seen from the positive side; right-arm frames from the negative side
  showed only the board). `model.visible_side` is "positive" for both arms by default.
- **Floor:** `model.floor_z`, the table's height in the root frame; views keep the pinch point 4 cm above it.

### Derivatives

Pinocchio differentiates frame poses with respect to the joint angles, not the model's geometry, which is most of
what we fit. `pixel_jacobian` is analytic and covers both: every parameter and joint moves everything below its URDF
joint by a twist (a translation for `pos:`, a rotation about an axis through the joint's origin for `rot:`, `zero:`
and revolute joints; `rot:` through the SO(3) right Jacobian of `exp`), and the pixel changes with the pinch point's
motion relative to the camera's (both chains can move). It agrees with central finite differences to about 1e-10
(relative), large rotation corrections included, and the fit's Jacobian for 40 clicks and 72 parameters takes
18 ms.

## The fit (`xle_fit.py`)

The maximum a posteriori problem is plain nonlinear least squares once each Gaussian prior is a residual row:

```
residuals(theta) = [ (uv_predicted_i - uv_clicked_i) / sigma_px    for each click i ]
                   [ (theta_j - mean_j) / sigma_j                   for each fitted parameter j ]
```

solved by `least_squares` (trust region) with the analytic Jacobian.

- **Robust loss:** Huber at 2 sigma, so a mis-click is down-weighted instead of bending the model. scipy applies
  it to each residual (u, v and each prior row separately) rather than to a click's pixel distance; prior rows
  within 2 sigma are in the quadratic part, so this rarely matters.
- **Covariance:** `inv(J^T J)` at the solution, over all rows; a parameter is *learned* if its posterior sigma is
  below half its prior's ("partly" below 0.85).
- **Held-out clicks:** every 5th click of each session is left out, the fit runs on the rest, and the pixel RMS on
  the left-out clicks before and after is reported per session. That number, not the training RMS, says whether
  the calibration generalizes. The saved calibration is then fitted on all clicks.
- **One fit for everything.** Every click of every session goes into one fit with the parameter table's wide priors:
  the head and camera get one estimate from both arms' clicks, and each arm's parameters are fitted only when that
  arm has clicks. An earlier calibration is only a starting point, never a prior, so no click counts twice (version 1
  fed the left fit's head posteriors to the right arm as independent priors, which also gave two head estimates).
  All sessions must share the head's servo map and the intrinsics, since clicks store raw head ticks;
  `check_compatible` refuses sessions that do not.
- **Sessions** are folders: `observations.json` (the clicks, with measured joints, head ticks, pixel, depth at the
  click) and `images/`. `load_session` also reads version 1's folders.

## Choosing views (`views.py`)

`find_view(model, arm, seed, target, head_limits)` solves for the head's pan and tilt and the arm's five joints
together, with `least_squares` (scipy's finite differences: 7 variables, cheap), from the seed and three random
starts. Residuals:

- the pinch point's *angle* off the optical axis times the focal length (about its pixel offset from the middle;
  / 20 px). An angle stays smooth when a trial pose puts the point behind the camera, where a pixel would jump;
- `1 - visibility` on the visible side (/ 0.2);
- the target position minus the pinch point (/ 2 cm), if a target is given;
- a one-sided floor clearance (/ 5 mm);
- a weak pull towards the seed pose (/ 1 rad).

Bounds: the head's recorded limits (or +/-40 deg around where it is), the arm's URDF limits with a 0.8 margin, the
wrist roll within 130 deg (the wrist camera's cable). `View.ok` says whether the result is usable: in the picture
with a 60 px margin, the hinge axis within 50 deg of the line of sight, above the floor. Showing the pinch point beats
reaching the target: a target the head cannot center (the neck at its limit) gets a view with the arm moved towards
what the head can see.

`plan_targets` samples arm poses within +/-20 to 30 deg per joint of a seed pose (the center pose by default),
keeps those whose pinch point is above the floor, and picks the most spread-out pinch points (farthest-point
selection in Cartesian space); each target keeps the pose that reaches it as `find_view`'s seed.

On the joint calibration, 40 of 40 planned targets per arm get a usable view (median hinge angle 35 deg; wrist rolls
49 to 66 deg for the left arm, 86 to 106 deg for the right), in 0.15 to 0.2 s each.

## A session (`collect.py`)

`collect` plans the targets (0.08 s), and on each `n` solves the next view with the latest estimates, moves the arm
and then the head, and shows the frame with the predicted pinch point. Each click is saved and refits the model with
every click (this session's and those of `--with` sessions), so the circle closes in on the pinch point. `q` runs
the final fit with held-out validation and writes `calibration.urdf` and `calibration.md`.

- **Starting point:** `--start-from` an earlier calibration, and the first view shows the gripper. On a fresh robot
  the head's zero angles are unknown (the tilt by about 20 deg), so the first views may miss: the person nudges the
  head with the arrow keys and clicks, and from 4 clicks the refits pin the head down. (Version 1 estimated the
  tilt from the table plane in the depth image; that is gone.)
- **Floor:** `--floor-z`, else the calibration's, else the arm base's height (conservative: on this robot the table
  is about 3 cm below it).
- **Vary the head:** nudging it a few degrees before some clicks keeps the head's parameters from being confounded
  with the arm's.
- Everything runs inside a client session: Ctrl+C or a crash sends the robot to rest.

## Saving (`calibration.urdf`)

`model.save(path)` writes `to_urdf()`, the merged URDF with the estimates written into the joint origins (zero
offsets folded in as a rotation about the joint's axis, so a model of the file with no corrections predicts the same
pixels), plus a `<felix_calibration>` element with what a URDF cannot hold: the servo map, intrinsics, visible sides,
floor, the parameter table with its priors, the estimates and their covariance, `fit_info` (sessions, pixel errors,
held-out errors) and the URDF's original origins of the corrected joints. Pinocchio and placo ignore the element, and
placo's view of the exported URDF is tested against the model, so any URDF reader gets the calibrated robot.
`XleRobotModel.load` restores the original origins and gives back the exact model, ready to refit.

## Trusting a result

1. **Held-out error.** It should be near the click noise (2 to 3 px). If it stays well above it, the remainder is
   play, backlash, gravity sag the measured joint angles do not capture, or a model error, not something more
   clicks fix. This replaces a separate repeatability test.
2. **Largest remaining errors** in `calibration.md`: a mis-click shows up there; `u` in `collect` undoes the last
   click.
3. **Learned column:** a parameter marked "no" was not determined by the clicks; do not read meaning into its value.
4. **Poses you did not collect:** try the result where the VLM will plan, by clicking a few more and checking the
   error.
5. **Simulation is optimistic:** with the right model and random poses, 20 to 80 clicks with 1.5 px noise bring the
   held-out error from about 220 px to about 2 px. Real errors also include backlash and play.

## Background: what the numbers say

From the version 1 sensitivity analysis (default URDF, left arm, 54 configurations with the gripper in view):

- **Geometry:** the head camera sees the arm only close up: the jaw pivot 0.16 to 0.51 m from the camera (median
  0.35 m), in a band about 25 to 45 cm in front of the robot. The shoulder is never in view, the elbow rarely.
- **Rules of thumb:** a head tilt error costs about 12 px/deg (fx times 1 deg in radians is 10.7 px), camera mount
  yaw and pitch about 11 px/deg, the arm's shoulder joint zeros 6 to 8 px/deg, and every position error about fx /
  depth: 1.6 to 1.9 px/mm at 0.35 m. At 0.5 m, 1 cm is about 12 px.
- **So:** the head's joint zeros and the camera mount's rotation matter most, the arm's joint zeros next, link
  lengths least; which is what the fits found.
- **Identifiability:** moving the head tower and moving an arm mount give exactly the same pixel shifts; fit one.

The camera: the D415's factory color intrinsics at 640x480 are fx 611.0, fy 610.0, principal point (315.9, 247.0),
zero distortion (`diagnostics/realsense_check.py`), and every capture carries them, so intrinsics are not fitted.
RealSense's self-calibration tools (on-chip, tare) improve the depth stereo pair, not the color camera's place on the
robot; run them only if depth looks wrong.

The head: a positive pan turns left (counter-clockwise from above), a positive tilt looks down; `setup.head`
records the servo signs to match. The URDF's camera is a generic `head_camera_link` 2.5 cm forward and 3 cm up from
`head_tilt_link`; the RealSense's actual mount is what the camera mount parameters learn.

## History

- **Version 1** (the first left and right sessions): two URDFs joined at runtime (`CalibrationModel`), our own
  numpy kinematics, roll-pitch-yaw corrections, a hand-written Levenberg-Marquardt fit with finite differences,
  poses from random sampling with a wrist-roll scan (8 to 26 s to start a session), the head tilt bootstrapped from
  the table plane in the depth image, and the right arm fitted separately with the left fit's head as priors. Its
  left fit: 1.99 px held out; right: 2.22 px. The current model reproduces its predictions (frozen in
  `tests/felix_jr/reference/v1_reference.json`), and refitting the left session alone gives the same 1.99 px.
- **Tags** (ArUco / AprilTag on the robot, to see the camera's pose every frame or to measure repeatability) were
  considered and dropped: clicks need nothing printed, measure what matters (where the pinch point lands), and the
  held-out error measures repeatability for free.
- **The sensitivity analysis** decided the parameters and priors (numbers above); the fit's posterior sigmas now
  answer the same question on real clicks.

## Dependencies

- `pin` (pinocchio) and `placo` through the `placo-dep` extra (it pins the cmeel libraries their wheels need), `scipy`
  through `scipy-dep`; the `felix_jr` extra pulls in both (`uv sync --extra felix_jr`). Imports are guarded with
  `_pinocchio_available` and `_scipy_available` (`utils/import_utils.py`).
- Lock with the uv version CI pins (`UV_VERSION` in `.github/workflows`, 0.11.30), e.g.
  `uvx --from uv==0.11.30 uv lock`: other versions rewrite the whole lock file's format.

## Not done yet, and open questions

- **Runtime:** projecting planned trajectories into the image with the calibration (`model.project`), and a
  `robot_kinematics(arm)` helper that builds LeRobot's `RobotKinematics` from `to_urdf()` (LeRobot's
  `RobotKinematics` on the merged URDF is tested to agree with the model).
- A marker on the gripper instead of clicks (the fit does not change), and the depth at the click as an extra
  residual (recorded, unused: dark shiny parts give poor depth).
- The floor: measure it once by hand, or keep a one-off depth measurement as a setup step?
- The right arm's rolls (86 to 106 deg now) and the wrist camera's cable; the view finder could penalize large rolls.
- Backlash and gravity sag: if the held-out error stays well above the click noise, try an approach-direction
  convention for the views before adding parameters.
