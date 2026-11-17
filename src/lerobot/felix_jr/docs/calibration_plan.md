# Calibration plan: where the calibration is going, and the work to get there

The working plan for the calibration's development. Newest decisions first; items are struck from "The work" when
done (their story goes into [camera_calibration.md](camera_calibration.md), the design and its history). The run book
is [README.md](README.md) (sections 6-9, still the old click calibration: work item 7).

## The procedure we are building toward (decided 2026-11-16)

From the simulator's comparison and two robot sessions (`camera_calibration.md`, "Not done yet": the head clicking the
moving tip halved the hinge's and the wrist camera's errors; finding the fingers automatically works against the table
and is untested anywhere else, so the person clicks what the head sees).

1. **The wrist cameras' lens, alone, once per lens** (`calibration.wrist_camera`): focus, then the intrinsics from a
   ChArUco board held in front of the camera. Redone only when the lens changes (refocused, a new camera), not when a
   mount moves. No camera pose on the gripper any more: the joint session finds it.
2. **The joint session, per arm, each time the robot needs it** (`calibration.joint_collect`):
   1. **The gripper sweep first**, the table clear (no board yet): the arm over the table, the jaws opened in steps;
      both fingertips found automatically in the wrist picture (`gripper_sweep`, `auto_tips`), the fixed tip's pixel
      compared with the last calibration's (the mount moved? stop here, before any clicking).
   2. **The board placements** (a ChArUco board): arm views and head-only views as now; at every arm view the jaws
      open (`gripper.pos` 20, 35, 50 in turn) and the person clicks **both fingertips in the head's picture**: one
      pane for the fixed tip, one for the moving tip, on the face the model says the head sees; poses that show the
      fingers edge-on to the head are not planned.
   3. **Low poses** over the table (10: 20 added little in the simulator), the same two clicks.
   4. **The fit** (`joint_fit.calibrate`): everything above together, with the hinge and the opening; the report says
      what was determined, names the mis-clicks (with pictures) and flags a failed fit.
3. **Optional: the gripper's widths with calipers** (`calibration.gripper_step`): no longer needed (the servo's range
   gives the degrees per unit; the joint session fits where the jaws close and the hinge), kept as the one measurement
   of the jaws' width in millimeters: a check, and the ground truth for the grip deformation work.
4. **The check** (`calibration.check`), as now.

## The work, in order

### 1. ChArUco in the lens step, the default board everywhere: done (2026-11-16)

- **ChArUco only, everywhere:** the checkerboard is gone from `boards` (one kind of detection; its sessions, before
  2026-11-11 for the joint ones and the 2026-10-08 lens records, are refused with a message), `boards.PRINTED_BOARD`
  is every script's default (`joint_collect`'s was still the checkerboard). `checks/hil_wrist_sfm.py`, built on the
  checkerboard and the pose step, is retired.
- **The lens's views** (`wrist_camera.LensViews`): a frame's board is kept when it is **held still** (its corners
  within 1.5 px of the frame before's), shows **30%** of its corners, and is a **new pose** (`view_signature`, now from
  the homography of whatever corners were found: the same for any part of the board) **or covers a picture cell**
  (8 x 6, two corners at least) no kept view does; at most **100**. Simulated (`test_wrist_camera`): a board held still
  200 frames, kept once; swept at ~6 px a frame, never; a slow 600-frame pass, 25; a person's 60 poses over two
  minutes (3600 frames), 63, covering 45 of the 48 cells, the lens then fitted to 0.03% (fx 400.1 for 400). A board
  pushed into each corner of the picture, running off it, is kept for the cells only it covers.
- Not yet run on the robot: the next lens calibration is the test (does the still check keep up with a real hand?
  the status line says why a frame was not kept).

### 2. The lens step alone; the camera's start from its design

- **Done (2026-11-16):** the camera-pose step (extrinsics, `fit_extrinsics`, `POSE_VIEWS`) is out of `wrist_camera`;
  a new lens leaves the camera's pose on the gripper as it was (`with_lens`). The design pose is in (item 3): a robot
  never calibrated starts from it.
- **The design pose** (measured 2026-11-16, `hardware_risks.md`): TheRobotStudio's module pose as drawn (its axes,
  the lens on its barrel's axis 15 mm out), not adjusted for our 45 mm boards: the calibration's correction says how
  far ours are (~9 mm, within 1-3 deg), with priors that allow it (1.5 cm, 5 deg).
- **Checked in the simulator (2026-11-16):** the joint fit started from the design's camera, the true one where ours
  are (`calib_20261115_grippers.urdf`'s corrections, plus a quarter-prior draw: 12-14 mm and 1.5-2.8 deg off), the
  procedure decided (the head clicking both tips, the wrist's tips from the end sweep), 3 seeds: the camera to
  0.4-1.0 mm and 0.2-0.4 deg, converged in 25-107 steps, no fit failed.
- The joint session then starts the camera from its **design pose**: the hex-nut mount and the UVC module's STLs
  (SO-ARM100's `SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`, as `sim2real.robot` places them), the lens at the
  module's optical center.
- **Check first in the simulator** (`joint_sim`): the camera started at the design pose with the truth off by a
  mount's tolerance (a few mm, a few degrees, and the mounts' drift, `hardware_risks.md`): the fit's retry
  (`wrist_cameras_from_sightings`) should bring it home every time.

### 3. Design values and learned values, kept apart

The question: keep the "naive" geometry (the URDFs, the STLs) apart from what calibration learned. For the arm, the
head and the mounts it already is: `design_xlerobot_so101.urdf` is the design, generated once (`merge_urdf`), and a
calibration stores only **corrections** to it (`zero:` / `pos:` / `rot:` parameters, with their priors), applied on
top of the design every time (`XleRobotModel`, "Parameters, nominal values and estimates"). A model with no
corrections *is* the naive robot. What does not follow that yet:

- ~~the wrist camera's pose on the gripper: stored whole (`WristCamera.gripper_from_camera`), not as a correction to a
  design pose~~ **done (2026-11-16)**, below;
- ~~the fingers' mount angles (`GripperMap.finger_mount_deg`, `fixed_finger_mount_deg`)~~ **done (2026-11-17)**,
  below;
- the gripper map itself (item 5).

Decided (2026-11-16): the design URDF stays **static** once the design values are in it, and the file names say
which is which, and `urdfs/` keeps only the two that matter. **Done (2026-11-16):** `xlerobot_so101.urdf` ->
`design_xlerobot_so101.urdf`; `xle_calib_1115_grippers4.urdf` -> `calib_20261115_grippers.urdf` (`api.MODEL`; new
calibrations `calib_<date>_<what>.urdf`); `xle_calib_1008_wrist2` and `xle_calib_1114_joint`, which tests start
from, moved to `tests/felix_jr/reference/`; `xle_calib_1008` and `xle_calib_1008_wrist` removed (git's history).
The docs' session logs keep the names used at the time.

Plan: put the design values in the design: the wrist camera's design pose as a fixed `<arm>_wrist_camera` frame in
`design_xlerobot_so101.urdf` (from the STLs, item 2; `merge_urdf` writes it), the fingers' design mounts likewise; the
calibration stores corrections to them as parameters with priors (the mount's tolerance). One design file, and every
calibration a set of corrections on it: no separate "naive URDF" to keep in step, and the corrections themselves say
how far the robot is from its design (a report column). Old calibrations keep loading (their whole poses read as
corrections from the design).

**Done for the wrist cameras (2026-11-16):** `design_xlerobot_so101.urdf` has, per arm, a fixed
`<arm>_wrist_camera_joint` from the gripper link to an `<arm>_wrist_camera_optical` frame at the design pose
(`merge_urdf.design_gripper_from_camera`: TheRobotStudio's mount and module, their numbers in `merge_urdf`); the
camera's pose is six parameters on it (`wrist_camera_parameters`, group "wrist camera", 1.5 cm and 5 deg), corrected
like the arm's mount. `WristCamera` is the lens only; `XleRobotModel.wrist_camera_pose` / `gripper_from_camera` read
the frame. `XleRobotModel.load` turns an older file's whole pose into these corrections (exactly: 6e-16 on
`calib_20261115_grippers.urdf`, its cameras 3.7 / 9.1 / -2.6 mm and 2.0 deg (left), 2.9 / 10.1 / 1.8 mm and 1.0 deg
(right) from the design). `joint_fit`'s separate camera unknowns are gone: the cameras' parameters are among the
estimates, free also while the arm is held (`fit_staged`).

**Done for the fingers (2026-11-17):** `design_xlerobot_so101.urdf` has, per arm, fixed `<arm>_fixed_finger` and
`<arm>_moving_finger` frames under the pinch frame (joints `<arm>_<finger>_finger_joint`): each finger's tip on the
pinch point in its STL's axes, the jaws closed (`merge_urdf.FINGER_AXES`, the placement `sim2real.robot` drew them by;
ours, not a drawing's: no drawing puts the finger on the SO-101's jaws). The hinge's axis is each frame's z, so a
mount is one parameter, `rot:<arm>_<finger>_finger_joint:z` (`finger_mount_parameters`, group "finger"; priors 5 deg
for the fixed finger, 20 for the moving one, which sits 16-18 deg off). Off by default like the hinge: `gripper_sweep`
adds and fits them (`with_finger_mounts`), and `joint_fit` never frees them (no point it sees depends on them).
`XleRobotModel.finger_mount_deg` / `finger_pose` read them; the sim draws the fingers by `finger_pose`, where it drew
them before (within 1 um on `calib_20261115_grippers.urdf`). `GripperMap` lost its two mount fields: `load` turns an
older map's into the parameters (left -16.5 / +3.0 deg, right -18.0 / +0.5 deg, moving / fixed).

### 4. The joint session's new flow

- ~~The sweep first (item 2.1 above), its drift check against the start's fixed-tip pixel.~~ **Done (2026-11-17):**
  `JointCollect.sweep_first`, before the window opens (phase "sweep"; `n` does not skip it; a resumed session keeps
  its sweep): `measure_wrist_tip`, then the fixed tip found against the start's (`sweep_drift_px`, the median over the
  frames, `gripper_sweep.fixed_tip`): more than `SWEEP_DRIFT_PX` (5), or no tips found, and the person is asked
  whether to go on. The sweep after `q` is gone.
- ~~The jaws open at the arm views and low poses (`gripper.pos` 20 / 35 / 50 in turn).~~ **Done (2026-11-17):**
  `joint_views.VIEW_OPENINGS` (`joint_sim.OPEN` is it), `with_openings` (closed without the arm's gripper map, with a
  message; `joint_collect --openings`, none: closed); both tips kept 2 cm over the table at every opening
  (`tips_clear`, in `low_poses` and `clear_of_table`: the moving tip turns down past the fixed one; no low pose lost,
  40 of 40 planned on either arm).
- ~~The head's panes: "fixed tip" and "moving tip" (`TipClick` faces `pinch` / `moving_pinch`; the face from the model:
  `joint_views.head_face_angle`), replacing today's two face panes; the wrist pane shows the live picture only.~~
  **Done (2026-11-17):** `PANES` `head:fixed`, `head:moving`, `wrist`; `JointCollect.face_of` (the fixed tip
  `pinch` / `wrist`, the moving `moving_pinch` / `moving`, by the head's face), the moving pane only with the jaws
  `MIN_MOVING_OPENING_DEG` (5) open, its tip predicted (`joint_views.moving_fingertips`, which `joint_sim` now uses
  too). The wrist pane shows the moment's picture and, with the sweep, takes no clicks. At the arm views the head sees
  the wrist camera's side (it looks down on it), so their clicks are `wrist` / `moving`: the smoke test's were.
  `joint_fit.report` compares the estimates by name (the fit adds the hinge for the moving tip: it crashed).
- ~~The low-pose planner drops poses with the fingers edge-on to the head (`EDGE_ON_DEG`).~~ **Already so:**
  `low_poses` keeps a roll only within `HEAD_FACE_MAX_DEG` (75, where `EDGE_ON_DEG` starts); now a test says so
  (`test_joint_views`). The arm views do not (the board's views come first; their head clicks are optional).
- Rehearsed against the fake robot (`server.server --fake`) and the smoke test (`test_joint_collect_smoke`). **The
  smoke tests done (2026-11-17)**: a session clicking both tips (4+ moving-tip clicks, the fit as near the truth), the
  sweep first (declined, gone on), the drift check's three answers. **Rehearsed by hand on the fake (2026-11-17)**
  (the board rendered, no fingers): the sweep first is right (the board not yet down); a misread sweep went into the
  fit (fixed: `SWEEP_FAILED_PX`), the planning's 35 s looked frozen (fixed: its message). Not yet run on the robot.

### 5. The gripper map without the calipers

**Done (2026-11-17).**

- A first calibration has no `GripperMap` (`opening_deg` refuses): build one from the gripper's servo range (degrees
  per unit, as `gripper_step` already takes it) and where it closes, with a wide prior on the opening's offset for that
  first fit (`OPENING_SIGMA_DEG` is 3 deg, for a map that was measured). **Done:** `GripperMap.from_servo` (degrees
  per unit `servo_deg_per_unit`, moved into the model; the tips meeting at `gripper.pos` 0: LeRobot's range starts at
  the jaws closed, ours measured -1.65 and -0.37, 2.1 and 0.5 deg off it), marked `default`; `joint_collect` builds
  it when the start has none; `JointProblem.opening_sigma` 6 deg for a default map (`DEFAULT_OPENING_SIGMA_DEG`);
  `calibrate` clears `default` once it fits the opening, and the report says which maps were the default (and any
  still are: no moving-tip clicks). `gripper_step` marked optional.
- Check in the simulator: the map's closing point off by several degrees, recovered. **Done** (`joint_sim
  --opening-off DEG`: the truth that far off, + / - by seed, the map the default; the procedure decided, scenario 3
  of `FINGER_SCENARIOS`, truth scale 0.25, 3 seeds): from 3 deg off, the opening to 0.11-0.24 deg; from 6, to
  0.08-0.21; the hinge (0.6-1.4 mm) and the camera (0.5-1.4 mm, 0.2-0.9 deg) as from a measured map (0.7-1.3 mm,
  0.3-1.2 mm). And end to end: the collection smoke test from a start with no map, `pos_closed` fitted to the
  truth's within 0.4 units.

### 6. Loose ends of the fit

- The hinge's z stays at ~2 mm sd in every procedure tried: more varied rolls in the head's clicks?
- Some fits crawl near the answer (1500 steps, `camera_calibration.md`); each restart resets the openings and the
  joints' errors.
- `gripper_sweep`'s hinge fit still fits the hinge along its own axis; it should say what it determined too.

### 7. The run book

**Done (2026-11-17):** `docs/README.md` sections 6-9 in the procedure's order: 6 the lenses (was 8), 7 the joint
session (new: the sweep first and its drift check, the board placements, the two head panes, the low poses, reading
the report, the other arm and `joint_fit` refitting both), 8 the calipers, optional (was 7), 9 the check; the click
calibration (`collect`, `xle_fit`) and the camera-pose step marked retired. The section pointers in the code's
messages (`check_robot`, `gripper_step`) and the other docs moved with them. Found writing it: a session started
from a map made under another LeRobot calibration of the gripper kept it; it now takes the servo's
(`joint_collect`), tested end to end (the collection smoke test: measured, none, stale).

## Changes, by commit

Newest first. Each commit that changes the calibration adds its entry here, titled by its subject; a commit cannot
name its own hash, so the next one fills it in. Before the plan (2026-11-16) the work is in `camera_calibration.md`
and git's log.

- **The run book for the new procedure; a stale gripper map replaced at a session's start** (2026-11-17, hash in the
  next entry). Item 7: `README.md` sections 6-9 in the procedure's order (the lenses, the joint session, the
  calipers optional, the check), the section pointers in `check_robot`'s and `gripper_step`'s messages and the other
  docs. `joint_collect`: a start's map made under another LeRobot calibration of the gripper is replaced by the
  servo's. Tests: the collection smoke test from a measured, a missing and a stale map.
- **`965154cb2`: The gripper map from the servo when there is none; the calipers optional** (2026-11-17).
  Item 5. `GripperMap.from_servo` and `default`, `servo_deg_per_unit` in the model; `joint_collect` takes the default
  for an arm with no map; `joint_fit`'s opening prior 6 deg for it, `default` cleared once fitted, the report saying
  so; `joint_sim --opening-off`; `gripper_step` optional. Tests: the default map and its save, its prior, the
  collection smoke test with no map.
- **`6efe16755`: A misread sweep's tips left out; the sweep's pictures saved; the planning said** (2026-11-17). From the rehearsal on the fake robot. `joint_collect`: the sweep's fixed tip more than `SWEEP_FAILED_PX`
  (30) from the start's is a misread (the fake's board taken for fingers: 70 px), its tips dropped and the person
  asked to go on without them; each sweep frame saved (`wrist_tip_sweep/frame_NN.png`) for its tips' mis-click
  pictures; "PLANNING the arm views (about 30 s)" before the 30 s of planning, the window painted at once, and keys
  and clicks made during a step dropped (a second `n` no longer moves the arm on). Tests: the drift test's misread
  case.
- **`c4ea9160a`: The joint session's head clicks both fingertips; the sweep first, with its drift check** (2026-11-17). Item 4's last two points. `joint_collect`: `PANES` `head:fixed` / `head:moving` / `wrist`, the face
  from the model (`face_of`), the moving pane only with the jaws open; `sweep_first` before the window (the fixed
  tip's drift from the start's, `SWEEP_DRIFT_PX`; asked to go on past it, or with no tips), the sweep after `q`
  gone. `joint_views.moving_fingertips` (`joint_sim` uses it). `joint_fit.report` compares estimates by name (it
  crashed when the fit added the hinge). Tests: the collection smoke tests click both tips, take the sweep first, and
  check the drift's three answers.
- **`c4793f371`: The jaws open at the joint session's views, both tips clear of the table** (2026-11-17).
  Item 4, its first two points. `joint_views`: `VIEW_OPENINGS`, `with_openings`, `tips_clear` (in `low_poses` and
  `clear_of_table`); `joint_collect --openings`; `joint_sim.OPEN` the same openings. Tests: `test_joint_views` (new),
  the collection smoke test with the jaws opening.
- **`26afef53c`: The wrist cameras and the fingers as corrections to design frames** (2026-11-17; tagged
  `felix-calib-design-frames-20261117`). Items 2
  and 3. `merge_urdf`: per arm, the wrist camera's design frame (`<arm>_wrist_camera_optical`, TheRobotStudio's mount
  and module) and the fingers' (`<arm>_fixed_finger`, `<arm>_moving_finger`) in `design_xlerobot_so101.urdf`.
  `XleRobotModel`: the camera's pose six parameters on its frame (`wrist_camera_parameters`, in the defaults: 35 per
  arm), each finger's mount one (`finger_mount_parameters`, opt-in); `WristCamera` the lens only, `GripperMap` without
  the mounts; `load` turns older files' whole camera poses and map mounts into these corrections (exactly), and pads
  the covariance with the priors. `joint_fit`: the separate camera unknowns gone (the camera's parameters free also
  with the arm held), the fingers' never freed. `joint_sim`, `joint_views`, `check`, `wrist_camera`, the harness,
  `sim2real.robot` and `gripper_sweep` read the frames. Checked in the simulator: from the design's camera, the truth
  12-14 mm and 1.5-2.8 deg off, to 0.4-1.0 mm and 0.2-0.4 deg (3 seeds). Tests: the camera and finger parameters,
  the migrations; the pixel-Jacobian test's search bounded (it hung when the parameters shifted its draws).
- **`3f651cc00`: The URDFs named for what they are; only the design and the current calibration kept** (2026-11-16). Item 3's naming: `xlerobot_so101.urdf` -> `design_xlerobot_so101.urdf` (`merge_urdf.MERGED_URDF`),
  `xle_calib_1115_grippers4.urdf` -> `calib_20261115_grippers.urdf` (`api.MODEL`); the two older calibrations tests
  start from moved to `tests/felix_jr/reference/`, `xle_calib_1008` and `xle_calib_1008_wrist` removed. The wrist
  camera's measured board (45 mm, its lens roughly centered, not the 32 mm module the design's mount holds) recorded
  in `hardware_risks.md`: the calibrated cameras' ~9 mm from the design's lens axis is the hardware.
- **`c3fb33048`: ChArUco only; the lens step alone, its views chosen out of the stream** (2026-11-16).
  Work item 1, and the removal half of item 2. `boards`: the checkerboard gone, `Board` the ChArUco board,
  `PRINTED_BOARD` every script's default (`joint_collect` too); checkerboard sessions refused with a message.
  `wrist_camera`: the lens only (focus, then the views), `LensViews` (held still, 30% of the corners, a new pose or a
  new picture cell, at most 100), the view signature from the corners' homography (any part of the board), a new
  lens keeps the camera's pose on the gripper (`with_lens`); the camera-pose step (`fit_extrinsics`, `POSE_VIEWS`,
  `orient_views`, `--only`, `--pattern`, `--square-mm`) removed. `checks/hil_wrist_sfm.py` retired. Tests:
  `test_wrist_camera` and its smoke test rewritten on synthetic ChArUco views, `test_boards` ChArUco only. Docs: README
  section 8, this plan (with the URDF naming decision under item 3).
- **`a871fc48b`: the moving fingertip in the joint fit, what the fit determined, and wrong solutions retried**
  (2026-11-15). Before the plan; what led to it: the moving tip's clicks (`MOVING`, `MOVING_PINCH`) with an opening
  offset and the hinge fitted, its along-axis part held; mis-clicks named with pictures; the uncertainty report;
  wrong solutions retried and flagged; `joint_sim --fingers --truth-scale` and the procedures compared
  (`camera_calibration.md`, "Not done yet").
