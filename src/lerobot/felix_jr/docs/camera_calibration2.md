# Camera calibration, version 2: `XleRobotModel`

**Status:** migration steps 1 to 5 are done (`calibration/merge_urdf.py`, `xlerobot_so101.urdf`,
`calibration/xle_robot_model.py`, `calibration/xle_fit.py`, `calibration/views.py`, and `calibration/collect.py`
rewritten on them; tests in `tests/felix_jr/test_xle_robot_model.py`, `test_xle_fit.py`, `test_views.py`,
`test_calibration_collect.py` and the smoke test of a whole session). Step 6, deleting version 1, is left. How to
run a calibration: [README.md](README.md), section 6.

**Calibrate** (both arms, one head):
`uv run python -m lerobot.felix_jr.calibration.xle_fit ~/cal_left ~/cal_right --out calibration.urdf`
writes the calibrated URDF and `calibration.md`, the report. The first joint fit of the two existing sessions
gives a held-out error of 2.07 px (left) and 2.05 px (right), against 1.99 and 2.22 px for the two separate
version 1 fits, with one head: pan zero -9.3 deg (+/-3.5), between the separate fits' -5.8 and -10.0. [camera_calibration.md](camera_calibration.md)
records how we got here (sensitivity analysis, the tag-based options, the first click sessions) and describes
the code as it is today.

**Goal (unchanged):** draw proposed arm trajectories into the head camera's RGB image accurately enough for a VLM
to judge them, which means a few pixels at the pinch point.

**What changed since version 1:**

- **Click calibration is the only workflow.** No ArUco or AprilTags, and no separate sensitivity analysis. The
  fit's posterior sigmas and its "learned" column tell us, on real clicks, what the sensitivity analysis guessed
  at from assumed tolerances. The first left-arm session (40 clicks, 2.0 px held-out error) showed the workflow
  works; the work now is making it robust and simple.
- **One model is the single source of truth.** `XleRobotModel` holds the kinematics, every calibrated parameter
  (its URDF value, its current estimate and how sure we are), the camera, and the pinch points of both grippers.
  The fit, the pose finder, the overlay and the runtime all read from it.
- **Off-the-shelf kinematics and optimization.** Pinocchio does the forward kinematics and joint Jacobians,
  `scipy.optimize.least_squares` does the fit and the pose search. Our own numpy kinematics, our hand-written
  Levenberg-Marquardt loop and our random-sampling pose planner go away.
- **Poses are chosen from the model alone.** Visibility of the pinch point was already decided from the URDF (the
  jaw hinge axis versus the line of sight). The depth camera was only used to guess the head's tilt zero and
  the table height. Both are replaced (see "Starting a session" below).

Notation as in version 1 (Drake monogram): `X_AB` is the pose of frame B in frame A, `p_AB_C` the position of B
relative to A expressed in C. Frames: `W` the robot's root (`top_base_link`), `C` the color camera's optical frame,
`T` a pinch point.

## Overview

```
                  merged URDF (built once, committed)
                               |
                     pinocchio.Model (nominal, never changed)
                               |  copy() + current estimates
                     pinocchio.Model (working)
                               |
  XleRobotModel ---------------+-----------------------------------------
    parameters: name, nominal, estimate, prior (mean, sigma), learned
    covariance of the estimates
    head servo map: ticks -> URDF angle (sign, reference)
    camera: intrinsics + optical frame
    pinch points: left / right, with hinge axis and blocked side
    |
    +-- pinch_point(arm, q)        world position of the estimated pinch point
    +-- predict(arm, q)            its pixel
    +-- pixel_jacobian(arm, q)     d pixel / d estimates, d pixel / d joints
    +-- to_urdf(), robot_kinematics(arm), save(), load()

  fit.py    (uses the model)     scipy least_squares over the estimates, MAP with priors
  views.py  (uses the model)     scipy least_squares over the joints: find_view(arm, target)
```

**The model supplies primitives; it does not optimize.** It holds the robot (kinematics, estimates, camera,
pinch points) and functions of joints and estimates. The fit and the pose finder are separate modules built on
those primitives: they hold policy (priors as residual rows, robust loss, held out clicks, seeds and thresholds)
that will change more often than the robot.

## Primitives

| Group | Method | Returns |
|---|---|---|
| Construction | `from_urdf(path, head_signs=, head_refs=, intrinsics=, parameters=, floor_z=, visible_side=)`, `load(path)` | an uncalibrated / a saved model |
| Parameters | `parameters`, `estimates`, `set_estimates(vec)`, `nominal(name)` | the table, the current vector, the URDF value |
| Joints | `joint_vector(arms, head_ticks)`, `head_angles(ticks)`, `joint_limits(arm)` | pinocchio `q` of measured angles; bounds |
| Geometry | `frame_pose(frame, q)`, `camera_pose(q)`, `pinch_point(arm, q)` | poses / position in the root frame |
| Measurement | `project(points, q)`, `predict(arm, q)`, `predict_many(arm, qs)`, `visibility(arm, q)` | pixels and depth; signed hinge cosine |
| Derivatives | `pixel_jacobian(arm, q, joints=...)`, `pixel_jacobian_many(arm, qs)` | (2 x P) w.r.t. the estimates, (2 x J) w.r.t. named joints |
| Export | `to_urdf()`, `robot_kinematics(arm)` (not yet), `save(path)` | |

**`q` is measured angles only.** `joint_vector` turns servo readings into pinocchio's configuration vector (head
ticks through the servo map, LeRobot degrees to radians for the arms); the `zero:` corrections are added inside
the kinematics. So a click's `q` is computed once and serves every `theta` the fit tries.

**`pinch_point(arm, q)` is the one place that knows where a pinch point is.** It returns the world (root frame)
position of the estimated left or right pinch point. `predict`, `visibility`, the overlay and the pose finder all
go through it, so the embodiment-specific choice (which frame, which offset) lives in one method.

**Every function that depends on the estimates takes them explicitly**, as `theta=None` meaning the current
ones. With `theta` given, the call applies it (`set_estimates`, always from the nominal model) before computing
anything. Optimizers always pass `theta`, so the result can never depend on what an earlier call left in the
model: no caching of "the estimates were already set to this `x`". Applying the estimates rebuilds 28
placements and takes about 0.17 ms, so `predict_many` applies them once for a whole set of clicks: 40 clicks take
0.6 ms, and a finite-difference Jacobian over all 72 parameters about 45 to 90 ms.

The pinocchio models are reachable as `model.nominal_model` and `model.pin_model` for debugging, read only. Editing it directly would skip the
fixed-joint rewrite (below).

## The merged URDF

Today the head comes from `xlerobot.urdf` and the arm from `so101_new_calib.urdf`, joined at runtime by a Kabsch
fit (`CalibrationModel._align_arm_to_base`), with an `so101:` prefix to tell the two trees' parameters apart.
Version 2 does this join **once, offline**, with a script that writes `xlerobot_so101.urdf`:

- the xlerobot head, camera, base and wheels;
- the old arms inside `xlerobot.urdf` removed, and two SO-101 chains (every link and joint prefixed `left_` /
  `right_`) attached to the arm bases `Base` / `Base_2` by fixed `left_arm_mount` / `right_arm_mount` joints, with
  the Kabsch transform as their origin (computed with pinocchio, same poses and fit as version 1);
- the root `joint_0` changed from `floating` to `fixed` (placo and pinocchio both need this, see version 1);
- mesh references, the mesh-only `*_geom_*` links, transmissions and materials removed, and the misleading
  `head_camera_rgb_frame` / `head_camera_depth_frame` links dropped;
- a fixed `head_camera_optical` frame (x right, y down, z forward), so the projection needs no extra rotation.

The pinch point is the SO-101's tool center point, `<arm>_gripper_frame_link`; the tip offset is the `pos:`
parameter of the fixed `<arm>_gripper_frame_joint` above it. The arm mount parameters stay on `fixed_Base` /
`fixed_Base_2` (as in version 1), so version 1 parameter names translate one to one (`estimates_from_v1`).

The script and its output are committed; a test checks the file is what the script builds. Another test checks
that `XleRobotModel` gives the same pinch point, camera pose, pixel and hinge cosine as today's `CalibrationModel`
at random joints, with no corrections and with random corrections of all 43 parameters (agreement to about 1e-12
px), which ties version 2 to the calibration we already trust.

## Nominal values and estimates

`XleRobotModel.__init__` builds the pinocchio model from the merged URDF **once** and keeps it as `nominal_model`.
It is never modified. The working model `pin_model` is `nominal_model.copy()` (a deep copy; checked), and the
estimates are applied to it.

**Every update starts from the nominal model.** `set_estimates(theta)` recomputes each affected placement as
`nominal placement composed with correction(theta)` and writes it into the working model. Corrections are never
applied on top of earlier corrections, so repeated updates (hundreds per fit) cannot drift, and setting the
estimates back to zero gives back the URDF exactly.

**The parameter table.** One row per parameter:

| Field | Meaning |
|---|---|
| `name` | `zero:<joint>`, `pos:<joint>:x|y|z`, `rot:<joint>:x|y|z` (joint names from the merged URDF) |
| `nominal` | the URDF value behind it: the origin's translation or rotation vector component (0 for a zero) |
| `estimate` | the current correction (radians / meters) |
| `prior` | (mean, sigma): how wrong we think the URDF is, before any clicks |
| `posterior_sigma` | from the last fit |
| `learned` | posterior sigma below 0.5 x prior sigma ("partly" below 0.85 x), as in today's report |

The full **covariance** of the estimates is stored next to the table (43 x 43 per arm). Per-parameter sigmas are
not enough to carry a fit forward: the head tilt zero and the camera pitch, for example, are strongly correlated,
and reusing their sigmas as independent priors would be overconfident along exactly the direction the clicks
could not separate. Reusing the head and camera from one arm's fit for the other arm (today's `--head-from`)
uses the joint Gaussian of those parameters.

**Kinds of correction:**

- `zero:<joint>`: added to the joint angle before forward kinematics. For the head this is the unknown angle at
  the recorded home; for the arm, the calibration offset of the servo.
- `pos:<joint>:<axis>`: added to the joint origin's translation, in the parent frame (as today).
- `rot:<joint>:<axis>`: a **rotation vector** `w`, applied as `R_origin = R_urdf exp([w])`. Version 1 used
  roll-pitch-yaw, which makes the Jacobian exact only at zero and makes the order of combined corrections
  matter. With a rotation vector, the Jacobian is exact at any value (via the SO(3) right Jacobian), and the
  values stay small (a few degrees), where the two agree anyway.

**Pinocchio folds fixed joints into frames.** This is the one thing about pinocchio that shapes the design.
Several of the parameters that matter most sit on fixed joints: the camera mount (`fixed_head_camera_link`), the
arm mounts (`fixed_Base`, `fixed_Base_2`) and the pinch points. Pinocchio turns a fixed joint into a frame and
pre-multiplies its placement into the placements of everything below it. Checked: editing the `fixed_Base` frame's
placement does not move `Rotation_L`. So the init reads the URDF and records, for every joint placement and
frame placement pinocchio holds, the chain of URDF origins it is the product of (the fixed joints from the nearest
moving joint down, then the joint itself). Applying the estimates rebuilds every placement whose chain contains a
corrected joint as the product of the corrected origins, always from the URDF values. `check_structure()`
verifies that rebuilding with no corrections gives pinocchio's own placements, and the comparison with version 1
fails by about 40 px if either the joint or the frame placements are not rebuilt.

**Head servos.** `angle = sign * (ticks - ref) * 2 pi / 4096 + zero`, with `sign` and `ref` from `setup.head` (as
today). The servo map is part of the model and is saved with it.

## Pinch points and visibility

These are the assumptions that make this model specific to the XLeRobot with SO-101 arms, hence the name.

- **Pinch point:** `<arm>_gripper_frame_link`, the SO-101 tool center point, returned by `pinch_point(arm, q)`.
  Its offset (the `pos:` parameters of `<arm>_gripper_frame_joint`) is calibrated (prior 8 mm, as today).
- **Hinge axis:** the SO-101 `gripper` joint axis. The camera sees the jaws meet when the hinge axis points along
  the line of sight. `visibility(arm, joints)` returns the signed cosine between the two, as
  `hinge_axis_cosine` does today.
- **Blocked side:** the wrist camera sits on the negative side of both grippers and hides the pinch point from
  that side, so the visible side is `positive` by default. This is a per-arm attribute of the model, not a
  command-line option. If the sign rule ever fails, the next step is a box for the wrist camera in the model and
  a line-of-sight test against it.
- **Workspace floor:** the table height in the root frame, measured once and stored with the model (not read from
  the depth image each session). The pose finder keeps the pinch point a clearance (4 cm) above it.

## Forward kinematics and derivatives

**Pinocchio** (`pin` 3.4, brought in by placo) loads the merged URDF with `buildModelFromXML`. On today's two URDFs
it agrees with our numpy kinematics to about 1e-15 and computes the forward kinematics plus a frame Jacobian in
about 1 microsecond, against 35 to 65 microseconds for our forward kinematics alone.

**What pinocchio differentiates and what it does not.** Pinocchio gives the Jacobian of any frame's pose with
respect to the **joint angles** `q` (`computeFrameJacobian`). It does not differentiate with respect to the
model's **geometry** (joint origins), which is most of what we fit. So:

- **Joints and zero offsets:** a `zero:` correction adds to `q`, so its Jacobian column is the joint's column. The
  pose finder only needs these.
- **Origin corrections:** a small change of a joint's origin moves everything below it exactly as a virtual joint
  placed at that origin would. Its column is that virtual joint's twist (translation along an axis for `pos:`,
  rotation about an axis through the origin for `rot:`), applied to the pinch point and to the camera. About 30
  lines on top of the frame placements pinocchio already computed.
- **Projection:** `p_CT = X_CW p_WT`, `uv = K p_CT / z`. The pixel Jacobian is the 2 x 3 pinhole derivative times
  the motion of the pinch point relative to the camera, so a parameter that moves the camera (head chain) and
  one that moves the arm both come out right.

**Staged.** First, scipy's finite differences on top of pinocchio's forward kinematics, with no derivative code of
our own: 43 parameters x 2 x about 40 clicks of microsecond evaluations is tens of milliseconds per Jacobian, fast
enough for a refit after every click. Then the analytic `pixel_jacobian`, tested against central finite
differences at random parameter values and joints. The analytic version is what makes the pose finder fast and
exact, and is a good test of the model itself.

## The fit

`scipy.optimize.least_squares` (trust region reflective). The maximum a posteriori problem with Gaussian priors is a
plain nonlinear least squares problem once each prior is a residual row:

```
residuals(theta) = [ (uv_predicted_i - uv_clicked_i) / sigma_px    for each click i ]
                   [ (theta_j - mean_j) / sigma_j                   for each parameter j ]
```

- **Robust loss:** `loss="huber"` with `f_scale` at the current 2 sigma threshold, so a mis-click is down-weighted.
  scipy applies the loss to each residual, so the prior rows and the u and v of a click are treated separately
  (today's fit applies it to the click's pixel distance). Prior rows stay within about 2 sigma, where Huber is
  quadratic, so the difference is small. Note it in the report, do not work around it.
- **Covariance:** `inv(J^T J)` at the solution, over all rows (including the priors), as today.
- **Held out clicks:** every 5th click is held out and reported before and after, as today. That number, not the
  training RMS, says whether the calibration generalizes.
- **Depth at the click:** an optional extra residual row per click (off by default, as today).

**One fit over every session.** Version 1 calibrated the left arm, then the right arm with the left fit's head
posteriors as priors. That gave two head estimates and counted the left clicks twice (as independent priors, so
overconfident where the clicks cannot separate parameters, like the pan zero and the camera yaw). Now:

- every click of every session goes into one fit, with the parameter table's wide priors; the head and camera
  get one estimate, and each arm's parameters are fitted only when that arm has clicks (others stay as they are);
- an earlier calibration is only a starting point (`--start-from`), never a prior;
- while collecting the second arm, each refit includes the first arm's saved clicks, so the head stays consistent
  throughout and there is no separate final step;
- all sessions must share the head's servo map and the intrinsics, since clicks store raw head ticks;
  `model_for_sessions` refuses sessions that do not (re-recording the head's home starts a new calibration).

## Choosing poses: `find_view`

`views.find_view(model, arm, seed, target, head_limits)` returns head and arm joints that put the pinch point near
the middle of the picture, seen from the visible side, above the floor, as a `View` (what to command, what the model
expects, and whether it is good enough). It is the same machinery as the fit, over joints instead of parameters:

- **Variables:** head pan and tilt, the arm's five joints. Bounds: the head's recorded limits (or +/-40 deg around
  where it is), the arm's URDF limits with a 0.8 margin, the wrist roll within `--max-roll`.
- **Residuals:** the pinch point's *angle* off the optical axis times the focal length (about its pixel offset from
  the middle, / 20 px; an angle stays smooth when a trial pose puts the point behind the camera, where a pixel
  would jump), `1 - visibility` on the visible side (/ 0.2), the target position minus the pinch point (/ 2 cm), a
  one-sided floor clearance (/ 5 mm), and a weak pull towards the seed pose (/ 1 rad).
- **Solver:** `least_squares` with scipy's finite differences (7 variables: a few milliseconds per Jacobian), from
  the seed and three random starts; the best is kept. `View.ok` says whether it is usable: in the picture with a
  60 px margin, hinge axis within 50 deg of the line of sight, above the floor.
- **Priorities:** showing the pinch point beats reaching the target. A target the head cannot center (the neck at
  its limit) gets a view with the arm moved towards what the head can see.

**Spreading the clicks:** `views.plan_targets` samples arm poses around a seed pose (the center pose by default,
+/-20 to 30 deg per joint), keeps those whose pinch point is above the floor, and picks the most spread-out pinch
points (farthest-point selection in Cartesian space); each target keeps the pose that reaches it as `find_view`'s
seed. Planning takes 0.08 s; each view is solved when `n` is pressed, with the latest estimates. This replaces
version 1's 6000-sample search for the first pose, its 600 candidates and the wrist-roll scan, which made a session
take 8 to 26 s to start.

**Starting a session.** The head's tilt zero is unknown to about 20 deg on a fresh robot.

- After the first calibration, the saved head and camera estimates (with their covariance) are the starting
  point, so the first pose already shows the pinch point.
- On a fresh robot: the first view is chosen with the priors, the person nudges the head with the arrow keys until
  the gripper is in view and clicks; the first refits (from 4 clicks) pin the tilt zero. The depth-based table
  plane estimate is dropped.
- The floor: `--floor-z`, else the calibration's `floor_z`, else the arm base's height (conservative: on this robot
  the table is about 3 cm below it). The depth at each click is still recorded, for an optional depth residual.

## Saving and interfacing

- **`save(path)` / `load(path)`: the calibration is one URDF file.** `save` writes `to_urdf()` (the estimates
  baked into the joint origins) plus a `<felix_calibration>` element holding everything a URDF cannot: the
  servo map, intrinsics, visible sides, floor, the parameter table with its priors, the estimates, their
  covariance, `fit_info` (sessions, pixel errors, held-out errors) and the URDF's original origins of the
  corrected joints. Pinocchio and placo ignore the element (checked), so any URDF reader gets the calibrated
  robot; `load` restores the original origins and gives back the exact model, ready to refit. The clicks stay in
  their session folders: they are inputs, not calibration.
- **`to_urdf()`:** the URDF with the estimates written into the joint origins, zero offsets folded in as a rotation
  about the joint axis (pinocchio cannot write URDF, so the model does it). A model of the result with no
  corrections predicts the same pixels (tested).
- **`estimates_from_v1(theta, arm)`:** translates a version 1 parameter dict (the saved left-arm calibration).
- **`robot_kinematics(arm)`:** a LeRobot `RobotKinematics` (placo) for one arm, built from `to_urdf()`, for the
  teleop and runtime code that already uses it.
- **`project(points, joints)`:** pixels of 3D points for the trajectory overlay.

## Dependencies

- `pin` (pinocchio) and `placo`, through the existing `placo-dep` extra (it pins the cmeel libraries the wheels
  need, see the note in `pyproject.toml`).
- `scipy`, through `scipy-dep`.
- The `felix_jr` extra pulls in both (`uv sync --extra felix_jr`). Imports guarded per `CLAUDE.md`:
  `_scipy_available`, `_placo_available` and `_pinocchio_available` (added) in `utils/import_utils.py`.
- Lock the dependencies with the uv version CI pins (`UV_VERSION` in `.github/workflows`, 0.11.30), e.g.
  `uvx --from uv==0.11.30 uv lock`: other versions rewrite the whole lock file's format.

## Migration

1. **Merged URDF** and the script that builds it, with the test against today's `CalibrationModel`. *Done.*
2. **`XleRobotModel`:** loading, parameter table, `set_estimates` (including the fixed-joint rewrite),
   `predict`, `visibility`, `to_urdf`, save and load. Tests against today's model. *Done;* `robot_kinematics` and
   porting the placo tests are left for step 6.
3. **The fit** with `least_squares` and finite differences, over all sessions and both arms at once
   (`xle_fit.py`). *Done:* the left session alone reproduces version 1 (1.99 px held out, 2.70 px overall); the
   joint fit is above. Version 1 session folders are read directly (`load_v1_session`).
4. **`pixel_jacobian`** (analytic), tested against finite differences. *Done:* it agrees with central
   differences to about 1e-10 (relative), large rotation corrections included. With it the fit's Jacobian for 40
   clicks and 72 parameters takes 18 ms (finite differences: 45 to 90 ms), and the joint fit of both sessions,
   held-out validation included, runs in 2 s.
5. **`views.find_view`** and the target spreading (through `pinch_point`); switch `collect` over to `XleRobotModel`.
   *Done:* on the joint calibration, 40 of 40 planned targets per arm get a view that shows the pinch point (median
   hinge angle 35 deg; wrist rolls 49 to 66 deg left, 86 to 106 deg right, lower than version 1's +120 to +130 for
   the right arm), in about 0.15 to 0.2 s each, solved when `n` is pressed so each uses the latest estimates.
   `find_view` uses scipy's finite differences: 7 variables, cheap enough. `collect` starts from a calibration
   (`--start-from`), fits jointly with other sessions (`--with`) after every click, and writes `calibration.urdf`
   at the end.
6. **Delete:** `urdf_kinematics.py` (once the placo tests use pinocchio), `sensitivity.py` and its report,
   `design.py`'s sampling, `bootstrap.py`, the fitting loop in `fit.py`, and `CalibrationModel`. Fold what is still
   true from [camera_calibration.md](camera_calibration.md) into this document and retire it.

## Open questions

- The workspace floor: measure it once by hand, or keep a one-off depth measurement as a setup step?
- Should `find_view` move the head at all during normal use (runtime), or only during calibration?
- Whether the right arm's poses (rolls of +120 to +130 deg today) are kind to the wrist camera's cable; the pose
  finder could penalize large rolls.
- Backlash and gravity sag: the held-out error is the measure. If it stays well above the click noise, consider
  an approach-direction convention for the poses before adding parameters.
