# Sim-to-real grasping: the last centimeters from the wrist camera

**Status (2026-11-14): research, branch `sim2real_grasp`.** Built: **our calibrated arm in MuJoCo with its wrist camera
and the head camera** (`sim2real/`, below), checked against ruler2's real frames. Not yet: the jaws' zero and contacts, a grasp
environment, any policy. Plain MuJoCo on this Mac for now; later mjlab on rented GPUs (RunPod or similar).

## The idea

1. **The head plans and moves** to an attack pose near the object, as now (`object_grasping.md`: it worked well).
2. **A policy trained in simulation does the grasp**, closed loop on the wrist camera: small moves that line the jaws
   up on the object, then the close.

Every failure so far was in the last centimeters (bowl5-bowl8: the alignment, the depth, moves along the approach ~1
cm short, the calibration's errors): what a closed loop on the wrist camera absorbs and an open loop (measure, move,
measure again) does not. Starting close keeps the policy's job, and the gap between simulation and the robot, small:
no navigation, no search, no object detection; only small adjustments and the close.

**How: a teacher, then a student** (the structure of Success Guided Sampling, SGS: a state-based RL policy distilled
into a vision policy, there at a large scale; ours is a smaller problem, and GPUs can be rented if needed):

- **Teacher:** RL on the simulator's state (the object's pose against the jaws), with the robot's errors randomized
  (the calibration's, the servos' lag and sag, the object's size and friction). State-based RL is cheap here (CPU).
- **Student:** the teacher distilled (behavior cloning / DAgger) into a policy on the wrist camera. **A depth model as
  the bridge:** depth models (DA3) were trained largely on synthetic data, so they read rendered and real frames
  alike; a small action head on their depth or features, the depth model run on the simulator's renders too (not
  the simulator's exact depth), so the student sees the same kind of input in both. DA3 takes seconds a frame on
  this Mac: fine for distillation's tens of thousands of frames, not for RL's millions.
- **The baseline to beat:** the current closed loop by hand (`wrist_contacts`: DA3 at a hold, a small move, measure
  again).

## The simulator on this Mac (M1, 2026-11-09)

| | |
| --- | --- |
| MuJoCo 3.8-4.0, CPU, the SO-101 + a table + 2 objects (8 contacts) | 55k steps/s on one core, **200k/s on 8** (`mujoco.rollout`) |
| the same, the XLeRobot's whole model, no contacts | 77k / 400k steps/s |
| offscreen rendering, 640 x 480 | 36-50 frames/s |
| MJX (MuJoCo on JAX), JAX's CPU backend | still compiling after 10 min (the meshes' collisions); stopped: it made the machine unstable |
| MJX on the GPU (Metal) | not tried: reported broken (`jax-metal` cannot lower `mhlo.reduce`) |

- **Plain MuJoCo on the CPU is enough for the teacher:** at a 2 ms step and 20-50 Hz control, 200k steps/s is
  ~8-20k control steps/s, a million in a minute or two.
- **Rendering (and DA3) bound the student**, which is why it learns from the teacher rather than by RL. Smaller
  renders than 640 x 480 will do.
- MJX's GPU path needs NVIDIA (MuJoCo Warp) or a TPU: rent one if the teacher needs more than the CPU gives.

## The robot model

- **TheRobotStudio's SO-101** (`SO-ARM100/Simulation/SO101`, Apache-2.0; `sim2real/assets.py` downloads it once at a
  pinned commit, ~16 MB of meshes, not vendored): our arm, in LeRobot's "new" calibration (each joint's zero
  mid-range, as our readings are), STS3215 servos (position control, kp, damping, armature, friction from the Open
  Duck Mini's measurements), a backlash class, and the wrist camera's mount. **Not** the XLeRobot repository's
  MuJoCo model: its arms are the older SO-100 (`Rotation`, `Pitch`, ...), with no cameras, the jaw free over +-180
  deg, and the jaws colliding as whole meshes.
- **What MuJoCo collides with** (`sim2real.view --collision` draws it): each mesh's convex hull. For the arm that is fine;
  for the claw it fills the jaw's inner curve, so an object cannot seat as deep as on the robot and contacts land on
  the hulls' faces.

## Decisions (2026-11-14)

- **The model learns in simulation.** Real frames are only its test set (wrist frames at holds, labelled by the head's
  depth before the arm hides the object, or by the person). Learning from real observations is a separate line of
  experiments, for another branch.
- **Both outputs:** the object's offset in the pinch frame (checkable on real frames; the existing moves act on it)
  and an action. The offset is the check and the fallback until the action is trusted.
- **DA3 as the bridge is in doubt:** on ruler2's plain white bottle its depth was the table's from both wrists
  (`sessions.md`). Plain objects need handling whatever the input.

## Our calibrated arm in MuJoCo (`sim2real/`, 2026-11-14)

- **`robot.arm_spec(model, arm)`:** TheRobotStudio's SO-101 (meshes, servos), each body posed from `XleRobotModel`
  at all-zero readings: the calibrated joint origins with the joints' zero corrections folded in, so MuJoCo's joint
  angles are LeRobot's readings in radians. World = the model's root frame; the arm's base where the calibration
  puts it. Added: a `pinch` site, the wrist camera on the gripper (its calibrated pose and lens: `focal_pixel`,
  `principal_pixel`, no distortion, as our undistorted frames), the head camera fixed at the given head ticks.
- **The wrist camera's mount** is the SO-101's hex-nut mount for a 32x32 UVC module (`SO-ARM100/Optional/
  SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`: the same mesh, triangle for triangle, as TheRobotStudio's model
  uses), kept with its module at the model's design pose. Our calibrated optical frames are within 2 deg (left) and
  1.4 deg (right) of the design lens axis, their centers inside the lens barrel, 10-13 mm up from the board, the same
  ~9 mm sideways on both arms (the module mesh's origin is off its lens): the robot's mounts sit as designed. The
  camera is inside its own module, so the module's geoms are in groups of their own that the wrist camera's renders
  leave out (they still cast shadows). From the head, mount and module sit on the real ones in ruler2's frame.
- **`scene.build`:** the arm over a table (top at 0.747 m, ruler2's), objects (`Cylinder`, `Box`) on it. One arm, no head or chassis
  meshes yet (the XLeRobot repository's MuJoCo model has them: Apache-2.0).
- **`view`:** renders from the head and the wrist camera, the visual meshes and (`--collision`) the convex hulls
  MuJoCo collides with; with `--robot` the robot's frames next to the renders at the same joints (nothing moves).
- **Tested** (`tests/felix_jr/test_sim2real.py`): every link, the pinch point and both cameras where the calibrated
  model puts them (to 1e-9 m), and a ball rendering where the lens projects it (within 1 px; this caught MuJoCo's
  principal point convention: the image's center minus ours).
- **Against ruler2's frames** (the joints and frames recorded there): the head view's gripper and bottle within a
  few tens of pixels; the left wrist's fixed jaw on the real one and the bottle centered within ~10 px (its size was a
  guess). **The moving jaw is off:** opened by the gripper map from the SO-101's zero taken as closed, it is wider
  than the real one in both wrists (the right gripper at 1.9 is closed on the robot, open in the sim). The jaw's zero
  needs calibrating (but see the fingers, below).
- **The fingers** (the default; `view --no-fingers` for the SO-101's own jaws): the robot's jaws are the XLeRobot's
  Fin Ray fingers (white, ribbed; 67 mm long, a wedge 11-23 mm deep), which replace the SO-101's jaw tips: the
  SO-101's jaw meshes are cut off at the fingers' base (drawn and collided with). Placed from the calibration alone
  (the fixed tip on the pinch point, the straight face toward the other jaw; the moving finger the same part turned
  over, the tips meeting with the jaws closed), **both fingers sat on the real ones in ruler2's wrist frames** (the
  right gripper closed, the left ~55 deg open: within ~10 px). The moving jaw's error above was the stock jaw's: with
  the fingers, the gripper map is to be checked, not the stock jaw (`gripper_sweep`, below). Drawn only for now: they may serve as the contact
  pads (their convex hull is close to their shape), or the pads go on their faces. The STL is in `sim2real/meshes/`
  (the XLeRobot's, Apache-2.0: `meshes/README.md`; git LFS).

## The gripper map against the wrist camera (`calibration.gripper_sweep`, 2026-11-15)

The planned jaw's zero (an angle per arm for the SO-101's stock jaw) dropped out: with the fingers on, the stock jaw is
out of sight and the moving finger is placed so the tips meet at the map's zero, so what the wrist camera can check is
the gripper map itself: is the real moving finger at the opening `GripperMap.opening_deg` says? The map's slope is the
servo's (its calibrated range); only `pos_closed`, where the tips meet, was fitted, from caliper gaps
(`gripper_step`).

- **`gripper_sweep`:** moves the arm to its sweep pose (the wrist camera looking down at the table; ruler2's left
  attack pose and its mirror), steps the gripper through a few positions, a wrist frame at each once settled, and sends
  the arms home. Before anything moves, the sim checks the gripper clears the table (taken level with the arm's base
  unless `--table-z`) at every position and on the way there, and that the wrist camera mostly sees table; then it
  asks. At the pose, a live preview first (the gripper half open, the pixels taken for the fingers tinted, a pass or
  a fail by the fit's limits): fix the scene until it passes, then Enter sweeps. `--stay`: only the gripper moves, the
  arm where it is. The fingers are found by color (cool blue in the wrist
  cameras; the wood warm), only where the sim says each can be; the moving finger's tip (the corner the wrist camera
  sees) is the farthest detected point from the hinge; the opening that puts the calibrated model's tip there is the
  fitted one. A live window shows each frame: the detected fingers and tip (green), the sim's fingers and the model's
  tip at the fitted opening (orange) and at the map's (blue); a contact sheet and `results.json` at the end.
  `--refit <folder>` fits a saved sweep again.
- **The tip, not the outline:** only the fingers' lit faces have their color (their shaded sides and black bases are
  dark), so scoring the sim's whole outline against the detected pixels drifted with the light (+8 deg on ruler2's
  left frame, where the tip says -1). The fixed finger's outline still checks the fingers' placement (its
  `agreement`: the finger pixels it covers less the wood it covers).
- **No fit** with the jaws nearly closed (the tips cannot be told apart), the tip out of the frame, or too little of
  the finger seen. Aim the wrist camera at the table: a bluish background (ruler2's window blinds) passes for the
  fingers.
- **The geometry** (`fit_geometry`): the hinge's position (a calibrated parameter, `pos:<arm>_gripper:*`) and the
  opening's offset from the moving tip's circle, then the moving finger's mount on its jaw
  (`GripperMap.finger_mount_deg`) from its outline; `--out` writes them into a new calibrated URDF.
- **On ruler2's frames** (offline): the left at `gripper.pos` 43.7, the map's 55.1 deg, fitted 54.2 (-0.9; the model's
  tip 5 px from the real one, the fixed finger's agreement 0.73); the right closed, no fit (and the blinds in view).
- **The left arm's sweep on the robot** (2026-11-15, `gripper.pos` 0-50): the map's opening right to 0.1 deg
  (slope 1.000), but the moving finger's base off in every frame. **The geometry fit** (from the same sweep): the
  tips' circle puts the **hinge 3.9 mm** from the SO-101's design (-3.7, 0, -1.1 mm in the gripper's frame), the
  opening's offset +0.2 deg, the tips 5.1 -> 1.5 px RMS; and the **moving finger turned -16.5 deg** on its jaw (closed,
  the fingers touch at the tips only, a V toward their bases, as on the robot). With them the sim's fingers sit on the
  real ones at every opening, bases and all. **A second sweep** (brighter light, the fingers near white): the fixed
  color thresholds lost the fingers and a misread tip (the finger's base) took the hinge to 10 mm, silently; now the
  colors are learned from each frame and misread tips are left out of a robust fit. Both sweeps then: the mount
  -16.5 deg both times, the hinge 3.9 and 2.1 mm (the same way; it repeats to ~2 mm: average a few sweeps), the
  opening within 0.4 deg. **Both arms** (`--arm both`, 2026-11-15): the moving fingers -15.5 (left) and -17.0 deg
  (right), the hinges 1.2 and 4.6 mm, the openings -0.2 and -1.0 deg; and **the fixed fingers turned too**, +4.0
  and +2.0 deg about their tips (`GripperMap.fixed_finger_mount_deg`; their outlines' agreement 0.74 -> 0.93, 0.79
  -> 0.88). Only the turn about the hinge's axis is fitted: the wrist camera sees the fingers' outlines in that plane;
  their offset along the hinge's axis and their tilt out of it are the SO-101's design (the head's views could find
  them). **The left camera moved on its mount** between the three left sweeps (the fixed finger's tip 6 px over an
  hour: `hardware_risks.md`), which the hinge took up (-3.5 to +1.1 mm); pooled, one hinge missed the tips by 3.2 px
  and was refused. Both mounts tightened; `gripper_sweep` now reports the fixed tip per sweep, and `--repeats N` /
  several `--refit` folders pool an arm's sweeps into one fit. The hinge also moves the harness's moving tip (`moving_tip`): its radius
  about the hinge was 83 mm in the model, ~87 by the tips' circle (85.6 by `gripper_step`'s calipers).
- **Not calibrated: the contact pads** (next step 2): design geometry, the same for both arms, constants in
  `sim2real/robot.py` on the jaw bodies, checked with `view --collision`.

## Next

1. **The gripper sweep on both arms** (`gripper_sweep --arm both --out <urdf>`): the left arm's fitted twice, the
   right's to run; a few sweeps per arm to average the hinge.
2. **The jaws' contacts:** pads (boxes, capsules) where the real claw touches, the meshes' hulls left to the rest;
   `view --collision` shows them.
3. **A grasp check before any RL:** open, approach a cylinder, close, lift: does it hold, as the real claw holds a
   bottle? Tune the friction, the contacts' softness (`solref`, `solimp`) and the gripper servo's force.
4. **The grasp environment:** starts near an attack pose (as the head's plan leaves it), the robot's errors
   randomized, the wrist camera's pose on the gripper among them (about half a degree, a millimeter or two: the mounts
   move, `hardware_risks.md`); then the teacher.
