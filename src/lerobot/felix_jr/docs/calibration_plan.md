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
  a new lens keeps `--start-from`'s pose on the gripper (`with_lens`). A robot with no pose yet cannot start a joint
  session until the design pose below is in.
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

- the wrist camera's pose on the gripper: stored whole (`WristCamera.gripper_from_camera`), not as a correction to a
  design pose;
- the fingers' mount angles (`GripperMap.finger_mount_deg`, `fixed_finger_mount_deg`) and the gripper map itself.

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

### 4. The joint session's new flow

- The sweep first (item 2.1 above), its drift check against the start's fixed-tip pixel.
- The jaws open at the arm views and low poses (`gripper.pos` 20 / 35 / 50 in turn).
- The head's panes: "fixed tip" and "moving tip" (`TipClick` faces `pinch` / `moving_pinch`; the face from the model:
  `joint_views.head_face_angle`), replacing today's two face panes; the wrist pane shows the live picture only.
- The low-pose planner drops poses with the fingers edge-on to the head (`EDGE_ON_DEG`).
- Rehearsed against the fake robot (`server.server --fake`) and the smoke test (`test_joint_collect_smoke`).

### 5. The gripper map without the calipers

- A first calibration has no `GripperMap` (`opening_deg` refuses): build one from the gripper's servo range (degrees
  per unit, as `gripper_step` already takes it) and where it closes, with a wide prior on the opening's offset for that
  first fit (`OPENING_SIGMA_DEG` is 3 deg, for a map that was measured).
- Check in the simulator: the map's closing point off by several degrees, recovered.

### 6. Loose ends of the fit

- The hinge's z stays at ~2 mm sd in every procedure tried: more varied rolls in the head's clicks?
- Some fits crawl near the answer (1500 steps, `camera_calibration.md`); each restart resets the openings and the
  joints' errors.
- `gripper_sweep`'s hinge fit still fits the hinge along its own axis; it should say what it determined too.

### 7. The run book

`docs/README.md` sections 6-9 rewritten for the procedure above: the lens step, the joint session, the optional
caliper step (marked optional, what it is still good for), the check; the old click calibration (`collect`,
`xle_fit`) and the camera-pose step marked retired.

## Changes, by commit

Newest first. Each commit that changes the calibration adds its entry here, titled by its subject; a commit cannot
name its own hash, so the next one fills it in. Before the plan (2026-11-16) the work is in `camera_calibration.md`
and git's log.

- **The URDFs named for what they are; only the design and the current calibration kept** (2026-11-16, hash in the
  next entry). Item 3's naming: `xlerobot_so101.urdf` -> `design_xlerobot_so101.urdf` (`merge_urdf.MERGED_URDF`),
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
