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
  (`XleRobotModel.finger_mount_deg`, a parameter on the design's finger frame since 2026-11-17) from its outline;
  `--out` writes them into a new calibrated URDF.
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
  and +2.0 deg about their tips (then `GripperMap.fixed_finger_mount_deg`, now a parameter; their outlines' agreement 0.74 -> 0.93, 0.79
  -> 0.88). Only the turn about the hinge's axis is fitted: the wrist camera sees the fingers' outlines in that plane;
  their offset along the hinge's axis and their tilt out of it are the SO-101's design (the head's views could find
  them). **The left camera moved on its mount** between the three left sweeps (the fixed finger's tip 6 px over an
  hour: `hardware_risks.md`), which the hinge took up (-3.5 to +1.1 mm); pooled, one hinge missed the tips by 3.2 px
  and was refused. Both mounts tightened; `gripper_sweep` now reports the fixed tip per sweep, and `--repeats N` /
  several `--refit` folders pool an arm's sweeps into one fit. The hinge also moves the harness's moving tip (`moving_tip`): its radius
  about the hinge was 83 mm in the model, ~87 by the tips' circle (85.6 by `gripper_step`'s calipers).
- **Not calibrated: the contact pads** (next step 2): design geometry, the same for both arms, constants in
  `sim2real/robot.py` on the jaw bodies, checked with `view --collision`.

## The grip, measured and simulated (2026-11-17)

**Measured on the robot** (`calibration.grip_test`, the right arm, results in `calibration/scratch/grip_right_1117/`):
upright bottles gripped from the side (the fingers 30-45 deg below level, the tips at half the bottle's height), the
bottle stood where the wrist picture's outline showed (`--depth-mm`, or wholly between the fingers by default), the jaw
closed to the map's contact and 2, 5, 10, 15 `gripper.pos` units past it (2 s each, the servo's position, load and
current sampled: the server's `grip`, protocol 6), a 30 s hold, a 4 cm lift.

| bottle | first stall (map's contact) | stalls, 0 -> 15 past | load, 0 -> 15 past | 30 s hold at 10 past | lift |
| --- | --- | --- | --- | --- | --- |
| pyri, 45 mm, 80 g, 27.5 mm in (5 trials) | 25.1 (23.2) | 25.1, 23.5, 20.7, 16.3, 12.0 | 12 -> 23 % | 16.4 %, steady | held |
| feno, 53 mm, 116 g, pushed fully in (~40 mm) | 29.4 (27.7) | 29.4, 27.9, 25.1, 20.8, 16.7 | 11 -> 24 % | 16.0 %, steady | slipped after ~2 s |

- **The stalls repeat** to 0.1-0.3 units over five trials of one bottle: the curve is the gripper's and the bottle's.
- **The fingers give:** the jaw goes on closing ~13 units (~16 deg) past first touch as the load rises 12 -> 23 %.
- **The overload protection** (`tuning_notes.md` 3.10): over ~20 % load for ~2 s it trips (the close to 0 at ~33 %
  tripped in 2.2 s; 15 past contact, 23 %, tripped 0.2 s into its hold); 10 past contact (~20 %) holds.
- **A grip near the tips pushes the bottle out:** feno at 15 mm in, mostly beyond the tips, popped out at the close
  to 0; at 27.5 mm in, pyri held through every close.

**In the simulator** (`sim2real/robot.py` has every value with what set it; `sim2real.grip_sim` replays a trial,
`sim2real.finger_models` compares the finger models):

- **The fingers collide as pads on their straight inner faces** (`finger_pads`: 4 boxes 2 mm thick on the 67 mm face,
  11 to 24 mm deep), not the mesh: its convex hull takes in the ribbed body (22 mm behind the face), and a bottle
  gripped on it tipped over. Torsional friction (`condim` 4), the pads' contact parameters first (`priority`), an
  elliptic friction cone, `impratio` 10 (no-slip iterations until 2026-11-20: MuJoCo Warp has none; the fingers'
  give and the joints' friction refitted without them, `sim_deploy.md`).
- **The gripper's servo as measured:** its gain 8.8 N m/rad (the stalls 1.9-3.8 units short at 12-23 %; the stock
  SO-101 model's 998 is ~100 times stiffer), its torque capped at the servo's limit, 50 % of 3.35 N m (what it keeps:
  20 %; the protection's trip is not modeled).
- **The fingers' give:** each finger on a hinge at its base with a spring, 1.6 N m/rad (1.2 with no-slip), fitted to
  pyri's stalls. The
  joints' armature is the SO-101 default the fit was made with (0.005 kg m^2: an explicit 0 moved the stalls 0.8).
- **Friction 0.31**, fitted to the lifts (feno slips below 0.33, pyri holds from 0.30). Two bottles: a first value.

The four finger models (`robot.FINGER_MODELS`), each fitted to pyri's stalls, both trials replayed (single
environment, CPU, the scene's build included):

| model | pyri stall / load RMS | feno stall / load RMS | lifts right | x real time |
| --- | --- | --- | --- | --- |
| **sprung** (default): one sprung hinge a finger | **0.1** / 2.1 | 0.7 / 4.5 | both | **15-20** |
| chain: the face's 4 pads hinged base to tip, 2.0 N m/rad each | 0.3 / **1.3** | 0.8 / 4.7 | both | 12-14 |
| soft: rigid pads, soft contact (best of 8: `solimp` width 3 cm) | 0.8 / 5.3 | 2.2 / 11.5 | feno wrong | 8-13 |
| rigid: no give (the jaw stops at first touch, at the cap) | 10.0 / 28.6 | 9.6 / 28.6 | feno wrong | 16-18 |

(stall RMS in `gripper.pos` units, load RMS in %.) The chain fits as well as the one hinge, slower: nothing in the
numbers yet says the fingers' wrapping matters for these bottles. **But by eye the chain looks closest to the real
fingers** (the video: its pads curl around the bottle as the Fin Ray does; the one hinge swings the face stiff). In
every model only the pads move: the drawn finger is one rigid body (a skin over the pads, as dexdesigner blends one,
would make it follow). To make that a number: each model's pads against the real fingers' inner edge in the wrist
frames saved at every close (the sweep's outline agreement, `calibration.gripper_sweep`). Feno's loads are off by ~5 % in every model: a softer bottle (an object's
give, which none of them has). Videos: `finger_models --video`, `grip_sim`'s replays (`on_frame`).

**Not tried** (from a survey of Fin Ray models in MuJoCo, 2026-11-17), and what each would take:

- **The full beam lattice** (rails and ribs as hinged box chains, closed with weld or connect equalities; stiffness
  `E I / l` per joint: RakshinEgor/FinRay_MuJoCo, BE2R-Lab-RND-AI-Grasping/finray_synthesis, and the cleaner
  uksangyoo/dexdesigner `hands/finray/`, TPU's E ~9 MPa "provisional"). The real wrapping, still rigid bodies, but
  stiff loop closures: a 1e-5 s step in one, chatter at 1 ms in another. Worth it if a trial that stresses wrapping
  (below) is missed by "sprung" and "chain".
- **A 2D flex skeleton** (MuJoCo's own `model/flex/gripper_2d.xml`, "Gripper 2D Fin Ray": a `flexcomp` `dim=2`,
  `dof="2d"`, inextensible edges, the base vertices pinned, at 2 ms). The most native; planar. Needs a newer MuJoCo
  than ours (3.8.1; the survey ran 4.0.1) and a training pipeline that supports flex: a dependency change to ask about.
- **Volumetric FEM or elastic rods** (a `flexcomp dim=3` from the finger's STL with `<elasticity young poisson>`, as
  `gripper_trilinear.xml`; or the rails as 1D flex / the cable plugin with the ribs on connect equalities). The most
  faithful (out-of-plane twist too), costly with thin ribs; no Fin Ray example found. A reference to calibrate the
  others against, not for training.
- **A Cosserat-rod solver outside MuJoCo** (bleebron/finray-sim, after Sheikh Sofla et al. 2025): an indentation-curve
  reference for the finger alone.
- For training at scale, the common practice (real-stanford/scalingup and others) is the soft-contact model with its
  contact parameters randomized; here the sprung hinge fits better at about the same cost, so it is the one randomized.

**To tell the models apart next:** trials where the wrapping should matter: a small bottle (2-3 cm), a grip near the
tips (`grip_test --depth-mm 15`), an off-center grip; and the friction pinned directly: pyri with water added until it
slips. Each replays into `finger_models` as it is.

## The grasp environment's starts (plan, 2026-11-18)

**What the episodes start from.** The task: from an attack pose near an object, line the jaws up, close, lift. Success:
the object 3-4 cm up and held 2 s (`grip_test`'s lift). Objects: upright cylinders first (the sizes and masses spread
around the measured ones), through the harness's primitives (`harness/primitives.py`: any `Primitive` with antipodal
grasps), so nothing below is specific to a cylinder.

**Diverse starts** (after OmniReset, "Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning",
arXiv 2603.15789, ICLR 2026: starts drawn from every stage of the task, not only its beginning; their policies learned to
retry). Four pools, their shares a parameter (`pool_weights`; a first guess in brackets):

| pool | starts at | teaches |
| --- | --- | --- |
| attack (0.5) | the attack pose, the gripper-to-object offset perturbed as the real robot misses | the whole grasp |
| at the object (0.2) | the grasp pose, a few mm off, the jaws open | when and where to close |
| in the jaws (0.2) | closed on the object, poor grips too (near the tips, off center) | the lift; a slip, a re-grip |
| lifted (0.1) | the object partway up | holding through the lift |

**The real misses, for the attack pool** (`sessions.md`, bowl5-bowl8, pen1-pen2): the gripper-to-object offset ~1 cm
(up to ~2), biased ~1 cm short along the approach (moves fell short twice; bottles 0.3-2.3 cm ahead of the aim point)
and ~0.5 cm low (the arm's sag at the attack holds: 0.4-0.9 cm); across the jaws 0.5-1.5 cm; the object's heading a few
degrees (a pushed pen: ~15). A starting point to randomize around, not a fit.

**The bank: the feasibility work offline, once per arm, table and object family** (OmniReset filters its resets
offline too, but on a 6-joint arm; ours has 5, so reachability is most of the work):

1. **Placements:** objects sampled over the reachable table, their sizes and masses.
2. **Feasible grasp chains,** the harness's filters (`harness/grasping.py`: the jaws, attack and grasp by IK, joint
   margins, the wrist camera's clearance), with four changes, the first three shared with the harness:
   - **the jaws' width from the fingers' real faces:** the closed jaws are a V, narrower deeper in (`grip_sim`: a
     bottle "wide enough" at the tips started inside the fingers); the open jaws checked at the depth the grasp puts
     the object;
   - **the lift, and the paths between** (attack -> grasp -> lift): the lift by IK from the grasp's joints, the table's
     clearance along each move in the joints, the camera's board corners and the arm's links included
     (`joint_views.table_margin`, `path_clear`: what hit the table in the joint session);
   - **the object in the wrist camera's view** at the attack pose (in frame, not behind the fingers): the policy's
     input;
   - **reachable around the start, not just at it** (bank only, its cost to be measured): the attack pose solvable
     ~2 cm off along a few axes (warm-started IK), or the Jacobian's manipulability bounded, so a start is not next
     to a limit or a singularity the policy cannot correct from.

   The harness's ranking (how antipodal, the approach's preference, the joint margin) is heuristics: not used. The
   candidates are sampled from the filtered set, weighted only for coverage (mostly side grasps for cylinders).
3. **Each grasp simulated before training** (`grip_sim`'s scripted grasp from the exact grasp pose: close 10 past
   contact, hold, lift 4 cm; ~1 s each now, parallel over the cores), under a few draws of the physics around the
   fitted values (friction, the fingers' stiffness): a success rate, not a pass. It answers whether the grasp is
   physically possible at all, the setup's question; the real robot will not reach that ideal pose, and the attack
   pool's offsets (the policy's job to close) and the in-jaws pool's poor grips are the real arrival. The same
   rollouts are the bank's later pools: their states snapshotted (`mj_getState`) open at the grasp, closed on the
   object, partway up. Every candidate keeps its label, successes and failures: the data for a learned grasp-quality
   model, which could replace the harness's ranking at deployment.
4. **The bank on disk:** per anchor the object, its pose, the grasp, the attack, grasp and lift joints, the success
   rate, the snapshots.

**What the first builds found** (2026-11-18): `harness.grasping.first_feasible` draws a feasible grasp at random (the
targets shuffled, each through the filters until one passes: a uniform draw, ~0.16 s against the full search's 60 s),
its `accept` the simulation's check (`sim2real.grasp_check.SimAccept`: the grasp from its exact pose under physics
draws; a one-build check first that the open gripper does not start inside the object: the harness checks the
fingers' faces, not the gripper's body). Only ~1 % of a cylinder's generated grasps are reachable by this 5-joint arm:
most need the gripper turned 55-147 deg about the closing line (no sixth joint), the rest a tilt the arm cannot reach
there. Starting the IK from the sessions' real configurations (8 per arm, clustered from 2300 moves) or from random
ones passed exactly the same targets: the failures are the arm's, not the search's (dropped). The gripper's axis stays
in the vertical plane through the pan axis (`GraspSettings.arm_plane_tol_deg`): with the solves held near the ideal pose
(rotation bounds 15/10/10 deg; loose ones put the gripper's body in the object), every reachable grasp was within 9 deg
of it, and a 15 deg check before any IK halves the time (0.77 against 1.47 s a placement, the same grasps).

**The bank's generator** (`sim2real/bank.py`, 2026-11-18; `uv run --extra felix_jr --extra felix_jr_sim python -m
lerobot.felix_jr.sim2real.bank <out.jsonl> --placements N --workers 6`): per placement an upright cylinder drawn
(40-60 mm since the first run, 10-20 cm, 40-120 g; 0.22-0.35 m ahead of the arm's base, +-0.15 m to the side), `first_feasible` with
`SimAccept` (side grasps, psi 0-30 deg; depths 0, 10, 20 mm and just past the radius; 3 physics draws, 2 to hold), the
anchor's states at the grasp, in the jaws and partway up from a draw that held (`grasp_check.SNAPSHOTS`; they load back
into the same scene with the object staying in the jaws: `test_bank_smoke.py`). JSON lines, one a placement, every
grasp simulated kept with its trials; `--resume` continues a stopped run. The first 24 placements:

- **13 of 24 anchored, ~3 s a placement on 6 workers** (4 performance cores; ~14 s a placement of work): ~1000 anchors
  in about an hour and a half.
- **Wide cylinders do not hold in the simulator:** of the 11 placements 59 mm and up, 3 anchored (59-61 mm), the rest
  simulated 7-38 grasps each with none holding (the object not coming up); below 58 mm all but 3 anchored, and those 3
  reached nothing to simulate. Untested on the robot (the grip trials were 45 and 52 mm): a real trial with a 60-70 mm
  bottle would tell a simulator's limit from the claw's; taken as the claw's (bottles past 60 mm are hard for it), the
  bank's diameters now 40-60 mm. The failed placements are most of the time (25-45 s each).
- **Depth:** accepted 9 of 96 at 0 mm, 4 of 85 at 10 mm, none of 54 at 20 mm.
- **Where the arm reaches** (a 50 mm cylinder, the filters only): side grasps none closer than ~0.22 m ahead and all of
  +-0.15 m to the side from 0.25 m out; closer in only from well above (psi to 90 deg: the grasps that put the gripper's
  body inside a tall cylinder). Hence the region; nearer placements would need top grasps.
- **The masses:** with 110-260 g one placement in six anchored, with 40-100 g three (the real claw: 80 g held, 116 g
  slipped). Most of a placement's 3000 tries fail the arm plane (~1 ms each, before any IK): 400 tries anchored none.
- Not yet: a cap on the grasps simulated per placement. The neighbourhood check is dropped: the attack pool's starts
  perturb the arm's joints instead of moving the object (below), so the grasp stays the anchor's, reachable.

**At a reset** (no IK): a pool by `pool_weights`, an anchor; the attack pool's start the anchor's attack joints
perturbed (decided 2026-11-19, replacing moving the object: the grasp stays the anchor's, reachable by construction;
any joints are a real pose; the misses are the shapes this 5-joint arm makes): the execution error in the joints (the
sessions' commanded against read joints, its bias too) plus the perception error, a Cartesian offset mapped through
the Jacobian's pseudo-inverse at the attack pose; both fitted from the sessions, sampled wider than fitted, the widths
parameters (the servos now sag as the real ones do, above: the start's command the attack joints, the arm settling
short on its own, so the execution's bias is not drawn, only its spread); a cheap check (one forward: the arm not in the object, the wrist camera no nearer the table than the
attack's), wide; the other
pools a snapshot with a few mm of noise; a few settling steps, and the rare start that intersects or falls rejected.

**The reset sampler** (`sim2real/starts.py`, 2026-11-19: `StartSampler(model, bank).sample(rng)`, `StartConfig` its
parameters): the attack pool's joints as above, the offset's means and spreads `perception_mean_m` /
`perception_std_m` (along -1.0 +- 0.7 cm, across 0 +- 0.7, up -0.4 +- 0.3) and the joints' `joint_std_deg` (the
sessions' 0.3-0.5 deg), each sampled `*_width` 1.5 times as wide. **The arm arrives:** started at the command, the
servos never sag (the friction's band holds them there), so the attack is ramped into from `arrival_deg` (3 deg) back,
each joint from the side the sessions' attacks came from (`ARRIVAL_UP`: the lift always up its range, wrist_flex
almost always down, the elbow either), at the harness's 20 deg/s. A bank records the servos it was built with
(`bank.servos()`); the sampler refuses one built under others (its snapshots would not match). On the first bank (13
anchors):

- ~65 ms a start (the scene's build, the arrival, the check's 0.3 s settle), every pool; the check passed at the first
  draw nearly always (1 redraw in 120).
- The gripper lands where the drawn offset puts it to within a few mm (the Jacobian's map; a 4 cm draw 3-4 mm off).
- The servos' stop, at the bank's attacks: elbow_flex +0.7 deg, wrist_flex +1.0 (the sessions' +0.65, +0.87); the
  lift -0.46 (the sessions' +0.24: the bank's attacks load the lift less, so moving up it stops past its target, as
  the fitted band has it; this rests on the lift's ridge, `robot.ARM_KP_NM_PER_RAD`); the pan's and the roll's spread
  0.46 and 0.75 deg (the sessions' 0.33, 0.36; ~2 mm at the gripper).
- The check's limit: a geom passed right through the object reads shallow in MuJoCo (a pad shut through a bottle:
  0.1 mm); the attack's offsets cannot get there.
- **Seen** (`python -m lerobot.felix_jr.sim2real.starts <bank> <out.mp4> --pairs 6 --clips at_object=1 ...`: each
  start played forward with its anchor's script, a view from beside and the wrist camera's; an attack start below the
  same anchor unperturbed, the miss kept through the script): every unperturbed attack held (2.6-3.1 cm up), every
  perturbed one missed with no correction, even a small offset (-0.3 along, -0.5 across, -0.9 up, cm): most of this
  bank's anchors grasp at depth 0, the object's center at the tips, where a few mm short or low loses it. Deeper
  anchors would forgive the real misses more; for the environment, prefer them or let the policy learn to go deeper.
  The other pools held as scripted. **Not perturbed yet:** the in-jaws and lifted pools are the anchor's snapshot
  exactly (the at-object pool's nudge only a few mm): the poor grips the plan has for the in-jaws pool (near the tips,
  off center: re-simulated closes from a nudged pre-grasp, a snapshot cannot be shifted while gripped) and a joint
  perturbation for the lifted one are for the environment's branch.

**The arm's misses, from the sessions** (2026-11-19: every `execute(..., hold=True)` of 13 sessions, bowl1-8,
container1, pen1-2, ruler1-2, both arms; the last move's commanded joints against the capture's read ones, the gripper's
offset by the model; 50 moves). What the encoders see, so the execution part of the miss only:

| pose (moves) | read - commanded, mean (deg): pan, lift, elbow, wrist flex, roll | std (deg) | gripper (cm) |
| --- | --- | --- | --- |
| attack, `pick.back` (28) | +0.1, +0.2, +0.7, +0.9, +0.2 | 0.3, 0.3, 0.5, 0.3, 0.4 | 0.6 low (std 0.3, to 1.35); across 0.0 +- 0.2, along 0.0 +- 0.15 |
| grasp, `pick` (10) | -0.0, +0.3, +1.2, +0.1, +0.2 | 0.5, 0.1, 0.3, 0.5, 0.3 | 0.7 low (std 0.2); across 0.05 +- 0.4 |
| loaded, lift / stage (8) | -0.1, +1.4, +1.1, -0.1, -0.1 | 0.5, 0.9, 0.6, 0.4, 0.3 | 1.05 low (std 0.4, to 1.9) |

- **Gravity, not the direction of the move:** every pose sags (the read joints all on the same side), more loaded (the
  shoulder +1.4 deg with an object, +0.2 without). The joints barely correlate at the attack (|r| <= 0.4): independent
  draws per joint will do.
- **The encoders see most of it, not all:** pen2's camera-measured sag was 1.3 cm against its encoders' 0.9 (link flex).
- **The rest of the miss is perception** (`sessions.md`: not in the joints): the jaws ~1-1.5 cm short along the
  approach (twice), bottles 0.3-2.3 cm ahead of the aim point, 0.5-1.5 cm across, a pushed pen ~15 deg.
- **The simulator's arm, fitted to them** (`sim2real/shortfalls.py --fit`; `robot.ARM_KP_NM_PER_RAD`,
  `robot.ARM_FRICTIONLOSS_NM`): the direction a joint moved into its hold matters more than gravity (elbow_flex +1.1
  deg short moving down its range, +0.2 moving up; wrist_flex +0.9 and -0.2; the lift, loaded, +1.9 down): a servo
  stops short on the side it came from, a friction band about a center gravity sets. Per joint a gain and a
  `frictionloss`: ~25 N m/rad and 0.2 N m for the P 16 joints, ~50 for the P 32 ones (the elbow 0.45 N m, the lift
  0.65), against the SO-101 model's 998 and 0.052. The replays (from the move before, at the harness's 20 deg/s) within
  0.19-0.30 deg RMS of the holds that moved; ~0.35-0.4 on all, the rest moved too little to say which side. The holds
  alone fit the lift as well at 150 N m/rad (its unloaded holds all moved one way), but only ~50 sags as the loaded
  lifts did. The grip trials' replays barely moved (pyri6's loads 2.6 % RMS against 2.1, feno2's 4.3 against 4.6; both
  lifts as on the robot); the bank's yield the same (13 of 24).

**Randomized around the fits:** the friction, the fingers' stiffness, the gripper's gain and torque cap, the objects'
size, mass and softness (feno: a softer bottle), the attack offsets, the wrist camera's pose on the gripper (~0.5 deg,
1-2 mm). How widely is the open question: the fits rest on two bottles.

**Order of work:**

1. The shared filter fixes in the harness (the jaws' width from the faces, the lift and the paths, the object in the
   wrist camera's view), each tested on the harness's grasp tests and timed.
2. ~~The bank's generator: placements, the filters, the simulated grasps under physics draws, the snapshots, the
   file.~~ **Built (2026-11-18, above);** the neighbourhood check dropped (the starts perturb the joints).
3. ~~The reset sampler and its pools.~~ **Built (2026-11-19, above).**
4. The environment around it (observations, actions, reward), then the teacher.

## The wrist camera in mjlab, and a frozen DINO's cost (2026-11-23)

The direction (2026-11-23): a student on the wrist camera's images through a frozen DINOv2-S, distilled from the
privileged teacher (`rl_experiments.md`), instead of hand-built perception (the silhouettes and DA3 at a hold did not
hold up: `object_grasping.md` 7). The order: (1) render and cost, (2) a probe: a small attention head over the frozen
patch tokens, trained on the teacher's rollouts to predict the object in the pinch frame, held-out error in the
simulator, overlays on the real wrist frames (the background test: randomize the table and background only if it
fails there), (3) the student by DAgger.

**Step 1, `sim2real/render_bench.py`.** mjlab's `CameraSensorCfg` wraps the arm's calibrated `wrist_camera` (its
intrinsics scaled with the resolution; the camera's own module left out, geom groups 0-2, as `view.render`). **224 x
168** (the real frames' 4:3; 16 x 12 patches of 14 px; a square image would stretch it). The frames look as expected:
the fingers at the bottom, the object between them; plain colours, and **black past the table's far edge** (no floor
or room: the first thing a background would fix). On an L40S (run `20261123_205000`), random actions, the
`multi1p` pool:

| worlds | step | with the camera | the render alone | a ViT-S/14 (DINOv2-S's shape, bf16) | env steps/s, all | peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 33 ms | 38 ms | 3.9 ms (65k frames/s) | 47 ms (5.5k frames/s) | 3.0k | 0.7 GB |
| 1024 | 49 ms | 57 ms | 15.7 ms (65k frames/s) | 198 ms (5.2k frames/s) | 4.0k | 2.5 GB |
| 4096 | 60 ms | 128 ms | 63 ms (65k frames/s) | 792 ms (5.2k frames/s) | 4.5k | 9.8 GB |

- **Rendering is cheap:** 65k frames/s, linear in the worlds; at 4096 it doubles the step (60 -> 128 ms).
- **The ViT is the cost:** 5.2k frames/s whatever the batch, 6x the rest of the step at 4096; with it the environment
  runs ~4.5k steps/s against ~68k without the camera. For distillation that is ~40 min per 10M frames.
- **Its weights in bf16, not autocast, double it: 10.2k frames/s** (`--vit-only`, 4096 frames: autocast 797 ms, bf16
  403 ms, bf16 compiled 400 ms). `torch.compile` adds ~1% (the bf16 model already runs PyTorch's fused attention);
  not worth it, and the image cannot compile anyway (no C compiler nor Python headers for Triton: `gcc` and
  `python3.12-dev` installed by hand on the pod to try it). With bf16 weights: ~530 ms a step at 4096 worlds,
  **~7.8k env steps/s, ~20 min per 10M frames**. To check with DINOv2-S's real code (its attention should be
  as fused). Not yet tried: the image at a lower rate than the control (every 2nd or 3rd step), fewer tokens
  (112 x 84: 6 x 8 patches).

**On the robot (the M1, measured 2026-11-22):** DINOv2 ViT-S/14 at 224 px 17 ms on MPS (25 ms CPU), 448 px 98 ms,
4 frames at 224 px 58 ms; CoTracker3 online 141 ms a frame on MPS; small PointNets (11k-239k parameters, 256 points)
0.14-0.76 ms. MLX has DINOv2 Base and Large (mlx-image), not Small; nothing for CoTracker.

## Next

1. **The gripper sweep on both arms** (`gripper_sweep --arm both --out <urdf>`): the left arm's fitted twice, the
   right's to run; a few sweeps per arm to average the hinge.
2. ~~**The jaws' contacts:** pads (boxes, capsules) where the real claw touches, the meshes' hulls left to the rest;
   `view --collision` shows them.~~ **Done (2026-11-17):** pads on the fingers' faces, the fingers sprung (above).
3. **A grasp check before any RL:** open, approach a cylinder, close, lift: does it hold, as the real claw holds a
   bottle? Tune the friction, the contacts' softness (`solref`, `solimp`) and the gripper servo's force. **The real
   data for it: `calibration.grip_test`** (2026-11-17): bottles of known diameter and mass, gripped from the side,
   the closes sampled (where the jaw stalls past the map's contact, the servo's load and current: the server's `grip`,
   protocol 6), a 30 s hold (the overload protection), a lift (held, slipped or dropped), wrist frames throughout
   (the fingers' give, against the sim's rigid outline). The simulator's grip is then fitted to it. **Done for two bottles
   (2026-11-17, above):** the stalls within 0.1-0.7 units, both lifts as on the robot. Still: the trials that stress
   the fingers' wrapping, and the friction pinned directly (above).
4. **The grasp environment:** starts near an attack pose (as the head's plan leaves it), the robot's errors
   randomized, the wrist camera's pose on the gripper among them (about half a degree, a millimeter or two: the mounts
   move, `hardware_risks.md`); then the teacher. **Planned (2026-11-18): "The grasp environment's starts", above.**
