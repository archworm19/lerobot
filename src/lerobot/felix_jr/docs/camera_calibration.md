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
| `calibration/xle_robot_model.py` | `XleRobotModel`: kinematics, parameter table, estimates, gripper points (the pinch point, which the calibration uses, and the moving jaw's tip), derivatives, saving |
| `calibration/xle_fit.py` | sessions of clicks, the fit, held-out validation, the report; CLI to refit sessions |
| `calibration/views.py` | `find_view` (head and arm joints that show a pinch point), `plan_targets` |
| `calibration/live_view.py` | `LiveView`: one arm seen through the head camera, moved through planned views (shared) |
| `calibration/collect.py` | the interactive session: views, clicks, refits, `calibration.urdf` at the end |
| `calibration/check.py` | checking a calibration on the robot: both fingertips drawn at views and openings, no clicks |

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
  All sessions must share the head's servo map and the intrinsics, since clicks store raw head ticks, and the
  sessions of one arm its LeRobot calibration, since clicks store its joint angles (which `lerobot-calibrate`
  defines); `check_compatible` refuses sessions that do not.
- **Sessions** are folders: `observations.json` (the clicks, with measured joints, head ticks, pixel, depth at the
  click, and the arm's LeRobot calibration as the robot reported it) and `images/`. `load_session` also reads
  version 1's folders, and sessions from before the arm's calibration was kept.

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
floor, the gripper maps, a snapshot of each clicked arm's LeRobot calibration, the parameter table with its priors,
the estimates and their covariance, `fit_info` (sessions, pixel errors, held-out errors) and the URDF's original
origins of the corrected joints. The snapshot and the head servos' directions are what `check_robot` compares
with the robot when the harness connects (not the head's references: they are the model's own, and the fitted zeros
are relative to them): the estimates hold only under the settings they were fit with. Pinocchio and placo ignore the element, and
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
   error, or with `calibration.check` (`README.md` section 9): it draws both fingertips at new views and gripper
   openings, with nothing to click, which tests the gripper map and the hinge as well.
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

## The rework: one joint calibration of the head, the arms and the wrist cameras (design, 2026-11-09)

Branch `calibration_rework`. **Status: in use.** Both arms calibrated together (2026-11-14: `joint_left_full4` +
`joint_right_full2`, fitted as one: held-out head vs arm 21.6 mm / 3.9 deg before, 5.2 mm / 0.7 deg after), checked on
the robot with `calibration.check`, and `urdfs/xle_calib_1114_joint.urdf` is now `api.MODEL`. Built: the boards (`calibration/boards.py`), the fit (`calibration/joint_fit.py`),
a simulator to size a session (`calibration/joint_sim.py`), the views' planning (`calibration/joint_views.py`) and the
session on the robot (`calibration/joint_collect.py`); below.

**Run it** (one arm per session; the start must have the arm's wrist camera lens, `calibration.wrist_camera`):

```bash
uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.joint_collect tcp://<pi>:5555 --arm right \
    --start-from <calibration.urdf> --out-dir <scratch folder> --board charuco:9x7:0.025:0.018:DICT_4X4_50
uv run python -m lerobot.felix_jr.calibration.joint_fit <session> [<session> ...] --start-from <calibration.urdf> \
    --out <new calibration.urdf>   # a refit, e.g. both arms' sessions together
```

**The wrist fingertip from a gripper sweep, not by eye (2026-11-15).** The wrist camera and the fixed finger are both
fixed to the gripper, so the fingertip is at one pixel in every wrist frame and a session's wrist clicks all measure
that one point. With the jaws closed the two tips lie side by side, and joint_right_full4's 27 clicks sat in the notch
between them (2-3 px apart, ~10 px from the fixed finger's end): the calibration fitted to them moved the camera 3.4 mm
/ 2.6 deg on the gripper, put the fixed finger off in the simulation, and `gripper_sweep` read 7.5 px. Replaced by the
fingertip a sweep measures with the jaws open (to 0.5 px; `calibration.auto_tips <session> --from-sweep <sweep> --out
<session copy>`, then `joint_fit` on the copy): the camera moved 1.2 mm / 0.7 deg, the board residuals and the held-out
check unchanged, the fixed finger then on the real one (`urdfs/xle_calib_1115_grippers4.urdf`, now `api.MODEL`, with
both grippers' geometry from `gripper_sweep`). **`joint_collect` now does this itself** (`--wrist-tip sweep`, the
default): the wrist pane takes no clicks, and after `q` a gripper sweep (the sim's checks, the person asked, the live
preview) measures the wrist's fingertip into the session before the fit. `--wrist-tip click` keeps clicking by eye.

The board is the printed ChArUco (9 x 7 squares of 25 mm, 18 mm markers, `DICT_4X4_50`; the PDF came from
`CharucoBoard.image`, so its layout is the detector's by construction): partial views count. Per placement (2 by
default; the board 26 cm ahead of the arm's base, where grasps are): the arms rest and the head looks where the board
should go (an outline), its picture live with the found corners dotted green; lay the board there, `n`: the head
looks at it from 6 directions. Then 10 arm views (`n` each, the arm moves slowly), the head looking at the board:
click the fixed fingertip in the head's picture where it shows (the wrist's comes from the sweep after `q`). Then 10 low
poses, the wrist rolled to show the head the fingertip's face it clicks and the head looking at it: click it in both
(a picture showing the board is one more sighting). `q` fits and writes `calibration.urdf` and `calibration.md`.
Each click is on one of the fingertip's two faces (`TipClick.face`). The window has three panes: the head's picture
twice, one per face (the pinch face, magenta; the wrist camera's side, a finger's thickness away, cyan), and the
wrist camera's (cyan: it sees only its own side), each with that face's predicted tip circled; a click is on the face
of the pane it is in, and the head pane the model expects to show the fingertip is marked (the 1 cm between the faces
is ~10 px at 60 cm, about the model's error before a fit: the eye tells them apart, the model cannot). Once a
placement is measured, every pane also draws the board where the starting model has it (orange: its outline and
corners) beside the corners found (green): the gap is that model's error, in the wrist's pane the arm chain's; a head
sighting more than 2 cm from the placement's board raises a "was it moved?" warning (`BOARD_MOVED_M`). `--head-axes` (session and refit) also fits
the head's pan and tilt axes' tilts (`XleRobotModel(head_axes=True)`, `with_head_axes`; 4 parameters, off by default):
in the simulator, tilts of 0.5-3 deg left the head's error at the noise without them, only the tilt axis about x and
partly the pan axis about x were recovered, and the other two traded off with the head's other parameters; turn it on
if the held-out agreement shows an error that depends on the head's pose. Tested end to end on the fake robot with a rendered board (`smoke_tests/test_joint_collect_smoke.py`). It replaces how the model is calibrated, so it is
built beside the current pipeline (`collect` + `xle_fit`, then `wrist_camera`'s pose step) and replaces it only once
it passes the checks at the end of this section.

### Why

The current pipeline calibrates in two steps with the first frozen: the clicks fit the arm and head, then
`wrist_camera` fits each wrist camera's pose on the gripper from views of a board, holding the arm's model fixed
(`fit_extrinsics`: only the camera's pose and the board's are free). The arm's errors had nowhere to go but into the
camera's pose:

- **On the robot:** the right camera's pose came out 7.8 deg off (bowl5: the drawn fingertips 34-51 px from the
  real ones), and the script's own numbers (settled, ~1 px) did not show it. A fingertip fit at one arm pose
  (`urdfs/xle_calib_1008_wrist2.urdf`) fixed it; the board's fit, refitted on hil3's views, landed 18 mm and 5.7 deg
  from it (`object_grasping.md` 6.3).
- **On the saved board data** (a scratch test, 2026-11-09: three board placements, each seen once by the head with
  the arms at rest and by the right wrist camera from 5-22 arm poses, 32 in all, with the calib1007 clicks):
  - With the current model, the board as the wrist sees it is tilted 5.2-5.7 deg from the board as the head sees it,
    in all three placements.
  - With the arm frozen, no camera pose and board pose satisfy both cameras: the head's corners stay 5.4 px off.
  - With the arm's parameters free, both fit (the head 0.6 px, the wrist 1.4 px): the cameras agree once the arm can
    move.
  - But that data cannot calibrate. The October clicks then fit at ~4 px instead of 2 (a month apart). The free
    parameters trade off, the mounts and the wrist roll's origin moving 7-25 mm from one variant of the fit to the
    next (one head view per placement, from nearly one head pose). And the camera's pose came out 20 mm and 4 deg
    from the fingertip fit in every variant: the board places the camera against the arm's chain, the fingertips
    against the fingers, and the two disagree when the gripper's geometry between them is off (the fixed tip's
    offset moved ~6 mm in the free fits).

So: one fit with everything free, on data collected for it in one session, with observations that tie the camera to
the fingers as well as to the arm.

### The stages

1. **The gripper's opening** (`calibration.gripper_step`, as now: calipers on the fingertips at several
   `gripper.pos`): the map from `gripper.pos` to the jaw's angle. It depends on no camera and not on the arm's model
   (only the hinge's radius is compared with the model's, as a warning), so it comes first, and the joint fit uses it
   to place the moving tip at any opening.
2. **The robot model, jointly, one arm at a time** (the head and the board shared; each arm's session adds its own
   unknowns, and a refit takes every session, as `xle_fit` does with the clicks of both arms now):
   - a. **Each wrist camera's focus and lens** (`calibration.wrist_camera`'s first two steps, as now). The lens fits
     289 hand-held views to 0.64 px and passed every check (`object_grasping.md` 6.3); it stays fixed in the joint
     fit. The head camera's lens is the RealSense's factory one.
   - b. **Board views by both cameras** (new): see below.
   - c. **The fingertips, seen by both cameras at once** (new; replaces the clicks of `collect`): see below.
   - d. **One fit** of everything (an extension of `xle_fit`), then the checks.

### 2b. The board, seen by the head and a wrist camera

- **Any board, any number of points.** The detection is behind one interface: a board kind (a checkerboard, a
  ChArUco board, an ArUco grid, ...) gives, per picture, the points it found as (id, pixel), and the board gives each
  id's position on it. The fit takes whatever subset each view found: a partial board, or a different number of
  points in each view and camera, needs nothing special (the residuals are per point, not per view).
  **Built** (`calibration/boards.py`, tests `test_boards.py`): `Detection` (ids, pixels), `Checkerboard` (the
  current one's detection, moved; `wrist_camera.Board` is it) and `CharucoBoard`, each with its points, `detect`,
  `readings` (two for a symmetric checkerboard), a printable `image`, and `parse_board` / `board_from_dict`;
  `board_pose` (PnP from any detection). On synthetic views (30-45 cm, tilted) the points land within 0.2-0.3 px of
  the truth, a ChArUco board half hidden still gives its corners. OpenCV's ChArUco corners alone sat 0.44 px off in
  x and y at any square size (as if half a pixel): they are refined again on the picture (`cornerSubPix`, 5 x 5).
- **Every point is a residual**, in each camera that sees it: its pixel against the projection of the board's
  corner through the whole chain (the head: root -> head joints -> camera mount -> RealSense lens; the wrist: root ->
  arm mount -> joints -> `gripper_from_camera` -> wrist lens), with the board's pose a shared unknown. Not the
  board's pose solved per view and compared (the hand-eye way): pixels keep the information a pose throws away (how
  well each direction is seen, the distortion at the picture's edges, partial boards), weigh every view by what it
  really constrains, and are what both cameras measure. The test above already fitted corners this way.
- **The board stays put** for a set of views (one placement), and the head need not see it at the same instant as the
  wrist: the board's pose is one unknown for the placement, pinned down by every view of it. What has to be
  simultaneous is only what moves: the gripper's pose, seen by the head (2c) and set by the joints at that instant.
- **The arm in the head's way.** The head sees the board over or beside the arm, not through it, and the wrist
  camera looks down at it from 15-30 cm: a board right under the gripper is hidden from the head. So:
  - **A ChArUco board** (a checkerboard with an ArUco marker in each white square; OpenCV's `cv2.aruco`, in our
    OpenCV 4.13), as the first board kind after the checkerboard: each corner is identified by the markers around
    it, so a board half hidden by the arm still gives the corners it shows, and the symmetric board's two readings
    (`orient_views`) go away. A new board to print.
  - **The views planned for both cameras** (as `views.py` plans the clicks): arm poses where the wrist camera sees
    the board, aimed so the gripper is off to the side of it from the head's side, and the head turned at each so
    the board and the gripper are both in its picture; the model (and the arm's skeleton, `arm_path`) predicts what
    the arm hides. Views where the head sees too little of the board still count for the wrist.
  - **Head views of the board on their own:** with the arm moved away, from several head poses (the pan and tilt
    varied): they fix the board's pose and the head's joints apart from the arm's.
  - **Two or three placements**, across where the arm works (in front, to each side, near and far), so the arm's
    parameters are seen from different directions (the test's single head pose per placement let them trade off).
- **Depth:** the RealSense's depth on the board is a check, not a residual (dark, shiny and near parts read poorly;
  under ~35 cm it reads nothing: jnt1).

### 2c. The fingertips, by both cameras at once

Clicks are still worth it: the board ties the cameras to the arm's chain, but only something on the fingers ties
them to the fingers (where the grasp is), and the gripper's geometry between the chain and the fingertips is where
the test found disagreement. What changes:

- **Each camera clicks the tip on the face it sees, at the same instant.** The two cameras cannot see the same
  point: the fingers are ~1 cm thick, the head sees the bottom face's tip (the pinch point as calibrated now) and the
  wrist camera the top face's. So the person clicks the fixed tip (and the moving tip, at a few openings: it tests
  the hinge and stage 1's map) on the face each camera sees, in the head's picture and the wrist camera's, shown
  side by side. The two points are tied by the gripper's geometry: the same tip, one thickness apart along the
  hinge, the axis square to the grasp plane (the pinch frame's y; `JAW_THICKNESS_M`, `wrist_face`). The thickness
  is a parameter with a tight prior (1.0 +- 0.1 cm, measured). Each click is a residual through its own chain: the
  head's (the arm's model) and the wrist's (its pose on the gripper). Together they tie the two chains at the
  fingers, which the board alone does not.
- **The model's prediction drawn in every view** as the fit improves (as `collect` does now), so a bad click or a
  wrong model shows at once in all of them.
- **A gripper sweep at a few poses** in the wrist camera alone (the moving tip's arc, the fixed tip, points along the
  fixed finger's face): the fingertip fit that fixed the right camera, now one more set of residuals.
- **Later, markers:** small colored dots or tiny ArUco markers on the fingers' faces, found automatically in both
  cameras; the fit does not change.

### 2d. The fit

**The order:** one joint fit of everything, always; what varies is where it starts (`fit_staged`). Three starts are
tried for 150 steps each and the lowest cost run to convergence: as given (the previous calibration); the head first
(the head's sightings alone, the arms left out, give the head's parameters and the boards' poses); and that plus each
wrist camera's pose re-estimated from its sightings against those boards. No one start always worked in the
simulator; the cost always told which did (a wrong solution sat at 100x+ the right one's).

The unknowns: the model's estimates (`default_parameters`: the head, the camera mount, each arm's mount, joint zeros
and origins and tip offset), each wrist camera's pose on its gripper (6), each board placement's pose (6) and the fingers' thickness, with
the current priors (`SIGMAS`); the lenses fixed. The residuals: board points (head, wrist), fingertip clicks (head:
the bottom face, wrist: the top face, one thickness apart), the sweep's tips (wrist), the priors. Huber, as now. Identifiability to watch:

- **The wrist roll's zero against the camera's turn on the gripper:** both turn the camera about the roll axis, and
  the fixed tip sits on that axis. Only the moving tip and points along the finger (off the axis) and the head's view
  of the fingers tell them apart.
- **The arm's mount against the head camera's mount:** both shift what the head sees of the arm; head views at varied
  head poses and board views separate them (the test, from one head pose, could not).
- The posterior covariance and the learned column (as now) say what was determined; poses the views missed show up
  in the checks, not in the fit's residuals.

**Built** (`calibration/joint_fit.py`, tests `test_joint_fit.py`, `smoke_tests/test_joint_fit_smoke.py`):
`JointProblem` takes `BoardSighting`s (a camera, a placement, the robot's state, a `Detection`) and `TipClick`s and
fits all of the above. One more unknown per moment: **the joints' reading errors** (5 per arm pose, 0.2 deg prior). A
reading off by a fraction of a degree moves every point seen at that moment the same way; without them the fit bent
the robot's parameters to explain it (in a simulated session its cost came out below the cost at the true robot:
25 px in the head and 5.4 mm / 3.2 deg on the camera, against 7 px and 2.7 mm / 2.0 deg with them). The Jacobian is
finite differences grouped by what each unknown touches (~60 evaluations however many moments): a fit of 2
placements is ~15 s when started near the answer, 1-2 min with `fit_staged`'s three starts. The fit stops at 1500
steps (`MAX_NFEV`) and the report says so: from a far start, or with an error the model cannot say, it otherwise
creeps on for an hour.

**The check** (`calibrate`'s report): every 5th moment at which the head and the wrist camera both saw the board is
left out of the fit, and scored by how far apart the head's chain and the arm's put the board there (`agreement`, mm
and deg). It replaced held-out clicks: a wrist click always lands on the same pixel (the camera rides on the gripper),
so holding one out checked nothing. That needed `XleRobotModel` to cache its applied estimates (90% of the time was rebuilding them
on every call; the module docstring says what would make the cache stale).

### Sizing a session: the simulator

`calibration/joint_sim.py`: a true robot drawn from the priors (the head's zeros +-20 deg, the mounts +-1.5 cm, ...),
the wrist camera 1 cm and 4 deg off on the gripper, the thickness off; views planned on the model (the board's center
up and right in the wrist picture, the fingers being lower left; the head nudged until the board is in its picture,
as a person would), kept if the true robot sees them (a checkerboard whole; for the head, not behind the arm's links);
noise on the board's points (0.3 px), the clicks (2 px) and the joints' readings (0.2 deg). Scored on held-out
grasp-like poses (the pinch point 3-12 cm over the table). 3 seeds each, before -> after the fit (with the first
planner and the checkerboard, 2026-11-09; what changed since: "First trials on the robot" below):

| per placement | views (wrist / head / clicks) | head px | wrist px | camera mm | camera deg | 5 cm move mm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 placement, 10 arm + 6 head views | 10 / 8 / 20 | 167 -> 4.1 | 49 -> 0.6 | 17 -> 2.7 | 4.5 -> 1.7 | 6.2 -> 2.1 |
| 2 placements, 10 + 6 each | 20 / 18 / 40 | 164 -> 1.8 | 49 -> 0.7 | 17 -> 0.9 | 4.5 -> 0.2 | 6.7 -> 2.1 |
| 3 placements, 10 + 6 each | 30 / 24 / 60 | 163 -> 1.4 | 49 -> 0.2 | 17 -> 1.4 | 4.5 -> 0.9 | 6.1 -> 1.9 |
| 2 placements, 20 + 6 each | 40 / 23 / 80 | 166 -> 1.9 | 49 -> 0.3 | 17 -> 1.4 | 4.5 -> 0.8 | 6.4 -> 2.0 |
| 2 placements, 10 + 12 each | 20 / 28 / 40 | 165 -> 2.3 | 49 -> 0.7 | 17 -> 1.5 | 4.5 -> 0.7 | 6.3 -> 1.9 |
| 2 placements, no clicks | 20 / 18 / 0 | 164 -> 23 | 49 -> 40 | 17 -> 15 | 4.5 -> 0.9 | 6.6 -> 2.7 |
| 2 placements, no head-only views | 20 / 6 / 40 | 163 -> 2.4 | 49 -> 0.5 | 17 -> 1.3 | 4.5 -> 0.9 | 6.2 -> 2.1 |

- **Two placements, ~10 arm views each, with the clicks,** bring the head to ~2 px (the clicks' noise) and the camera's
  pose on the fingers to ~1-1.5 mm and under 1 deg. More placements or views add little (with 3 seeds, the rows
  within ~0.5 mm / 0.5 deg of each other are the same).
- **The clicks are essential:** without them the board places the camera against the arm's chain but nothing places
  the fingers (the camera 15 mm off the fingers, the fingertips 40 px off in the wrist picture), as the test on the
  saved data found.
- **With a checkerboard the head sees the board whole only with the arm out of the way:** in every arm view the arm
  hides part of it (the wrist camera looks down at it from the head's side). The head-only views (arms folded) are
  then most of the head's board views; the clicks carry the head's link to the arm.
- **A 5 cm move along the approach stays ~2 mm off** whatever the views. Not extrapolation: poses low over the table
  with only the fingertips clicked (no board) bring the head from 1.7 to 0.8 px with 20 of them, but the move only from
  2.0 to 1.7 mm (the held-out poses drawn apart from them). The fit leaves the arm's joint origins (the links'
  millimeters, prior +-2 mm) nearly where it started them: the views do not turn the joints through enough different
  configurations to tell them apart, and each left ~2 mm off adds to a move. It is far below the robot's ~1 cm short
  over a few cm (bowl7, bowl8), which the model's parameters at their priors' size cannot produce: that is something
  the model does not have (sag under load, backlash), for check 3 below.

### First trials on the robot, and what they changed (2026-11-11)

Left arm, one placement each, from `xle_calib_1008_wrist2.urdf` with the gripper step redone:

- **The checkerboard (trial 1, 4 arm views):** the wrist found it in 2 of 4: the views, planned on a model ~1 cm off,
  put it off the picture or behind the fingers, and the 8 x 6 board barely fits the wrist's picture. The simulator
  showed the same (~50%). Hence **ChArUco** (partial views count, `Board.min_view_fraction` 0.5). Also found: the
  detector's histogram normalization lost the board in the head's views under a lamp (plain first now, then
  normalized, then exhaustive); `--resume` overwrote images and the saved images were mislabeled (keyed by `id()`).
- **ChArUco (trial 2, 6 arm views):** the wrist read the board in every view (13-35 corners), the head in all 11.
  The head and the arm put the board **12.9 mm and 4.6 deg apart with the old calibration, 1.5 mm and 0.7 deg after**,
  and 1.6 mm / 0.3 deg on the first trial's data, which the fit never saw. But the four arm views were near copies
  (elbow +27 to +44 deg): held out, the one view in another pose (elbow -56) was 13.8 mm off. Hence the planner below.
- **Head clicks:** 1 in 11 views. In the arm views the fingertips were off the head's picture or behind the wrist
  camera's mount; in the low poses the jaw hid its tip. They are needed (a wrist click fixes only the direction from
  its camera to the tip; the head's fixes the distance: without them the camera's pose came out ~9 mm off in the
  simulator), so the low poses now roll the wrist to show the head that face (`head_face_angle` < 75 deg; tilted
  45-80 deg from straight down, where a roll can).

**The arm views** (`arm_views`): 400 looks drawn at random (the camera's height, tilt, direction; where the board
lands in the picture; the solver started from different arm configurations), every link 5 cm over the table, the
board 30 px inside the picture on the model. Of those: 70% planned from and kept within 40 deg of the working region
(poses like `low_poses`'), 30% anywhere; each set picked farthest apart in the joints. In the simulator (held-out
grasp poses, medians of 6 seeds; 2 seeds left out where the head never saw the board):

| arm views per placement | planner | sag | head px | camera mm | camera deg | 5 cm move mm |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | working region only | none | 0.65 | 1.79 | 0.86 | 1.76 |
| 6 | mix (70 / 30) | none | 0.72 | 0.98 | 0.20 | 1.84 |
| 10 | mix | none | 0.61 | 0.73 | 0.38 | 1.77 |
| 6 | working region only | 1.5 / 1.0 deg | 1.69 | 1.28 | 1.05 | 2.15 |
| 10 | mix | 1.5 / 1.0 deg | 1.86 | 0.77 | 0.37 | 2.03 |

- The head's error is flat from 4 to 10 views; the wrist camera's pose is what the far views and the 10 views buy.
- The views clustered as before (5 looks, jittered) and max spread (anywhere) each did worse on one of the two.
- Seeds scatter by 1-2 mm: the medians give the direction, not the size.

**Sag** (`World.sag_deg`: the shoulder and elbow drooping 1.5 / 1.0 deg with the gripper 25 cm out, where the
encoders cannot see it) sets a floor on the head's error (~1.6-1.9 px against ~0.6). It first looked much worse: most
of that was the fit stuck from far starts, which `fit_staged` fixed. The servos' own droop under P-only control (the
elbow ~3 deg short, `tuning_notes.md` 3.1) is read by the output-shaft encoders, so it is in the calibration's data;
it matters for motion (a move lands short), not for the fit. A compliance term (q + C tau_g(q), C per pitch joint, the
gravity torque from the URDF's masses) is designed, not built: first measure commanded against read across poses, and
the session's residuals against the gravity torque.

**joint_left_full4 (2026-11-14, safe poses, 2 placements):** the head and the arm agree on the board to 0.7-3.6 mm
at the first placement (held out: 2.1 and 1.6 mm, under 0.4 deg), 13.5 mm and 5.7 deg before. The second placement,
with the arm swung further out (shoulder pan 10-39 deg), stays 3-7 mm off even for fitted views: not the board moving
(fitted as two placements split mid-way, its two poses 1.3 mm apart and no better), not the head (wider head priors
and the head axes' tilts change nothing): an error on the arm's side the model cannot say. Next to try: the arm joints'
axes' tilts. Both sessions moved the head camera's z and the tilt joint's x by 14-20 mm and 8-15 mm (6 sigma): their
priors are now 10 mm (and the camera's turn 5 deg), and a fit takes its priors from the table, not from the
calibration it starts from (`XleRobotModel.with_current_priors`).

### Checks before it replaces the current calibration

1. **Held-out views:** the fingertips predicted in both cameras at poses left out, near the click noise (2-3 px in
   the head; a few px in the wrist), and the board's corners likewise.
2. **The board's tilt:** the wrist's and the head's agree within ~0.5 deg (5.2-5.7 deg now).
3. **Motion:** a move of a few cm along the approach, planned and run, measured by the head's depth and the wrist
   camera; today it falls ~1 cm short (bowl7, bowl8). The wrist roll's travel against the head's ICP (4-6% more
   than the model now).
4. **A session:** `wrist_contacts` and a pick on the robot with the new calibration, nothing patched by hand.

### What it replaces, once it passes

`collect`'s pinch-point clicks (2c does them in more views), `wrist_camera`'s pose step and `fit_extrinsics`
(`calibrate_extrinsics`, `orient_views`, and their tests), the hand patches of the camera's pose
(`urdfs/xle_calib_1008_wrist*.urdf`), and `checks/hil_wrist_sfm.py`'s `handeye-*`. The output stays one
`calibration.urdf` (`XleRobotModel.save`), so the harness does not change.

### Open questions

- The weighting: each corner counts the same, so a view showing more of the board counts more. In the simulator
  weighting per view instead changed the scores by ~0.1 (less than the seeds' scatter): left as is.
- Whether the head's lens needs fitting too (the RealSense's factory intrinsics, unverified at the picture's edges).
- Gravity sag: the board views have the arm still and unloaded; a pick's arm is loaded. Separate (`open_issues.md`
  item 16).

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
