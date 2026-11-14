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
  `principal_pixel`, no distortion, as our undistorted frames), the head camera fixed at the given head ticks. Their
  wrist camera mount is removed (ours is elsewhere: our camera looked out from inside theirs); our camera's housing
  is a box behind its lens, for collisions only.
- **`scene.build`:** the arm over a table (top at 0.747 m, ruler2's), objects (`Cylinder`, `Box`) on it. One arm, no head or chassis meshes yet (the XLeRobot repository's MuJoCo model has them; its license to check
  before copying).
- **`view`:** renders from the head and the wrist camera, the visual meshes and (`--collision`) the convex hulls
  MuJoCo collides with; with `--robot` the robot's frames next to the renders at the same joints (nothing moves).
- **Tested** (`tests/felix_jr/test_sim2real.py`): every link, the pinch point and both cameras where the calibrated
  model puts them (to 1e-9 m), and a ball rendering where the lens projects it (within 1 px; this caught MuJoCo's
  principal point convention: the image's center minus ours).
- **Against ruler2's frames** (the joints and frames recorded there): the head view's gripper and bottle within a
  few tens of pixels; the left wrist's fixed jaw on the real one and the bottle centered within ~10 px (its size was a
  guess). **The moving jaw is off:** opened by the gripper map from the SO-101's zero taken as closed, it is wider
  than the real one in both wrists (the right gripper at 1.9 is closed on the robot, open in the sim). The jaw's zero
  needs calibrating. The wrist camera's raised mount is not modeled.

## Next

1. **The jaw's zero:** where the SO-101's moving jaw meets the fixed one, from a real wrist frame at a known
   `gripper.pos` (`view --robot`).
2. **The jaws' contacts:** pads (boxes, capsules) where the real claw touches, the meshes' hulls left to the rest;
   `view --collision` shows them.
3. **A grasp check before any RL:** open, approach a cylinder, close, lift: does it hold, as the real claw holds a
   bottle? Tune the friction, the contacts' softness (`solref`, `solimp`) and the gripper servo's force.
4. **The grasp environment:** starts near an attack pose (as the head's plan leaves it), the robot's errors
   randomized; then the teacher.
