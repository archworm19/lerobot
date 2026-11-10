# Sim-to-real grasping: the last centimeters from the wrist camera

**Status (2026-11-09): research, branch `sim2real_grasp`.** Built: the SO-101 in MuJoCo on a table with objects
(`sim/`), a render of random motion. Not yet: the jaws' contacts, the wrist camera, a grasp environment, any policy.

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

- **TheRobotStudio's SO-101** (`SO-ARM100/Simulation/SO101`, Apache-2.0; `sim/assets.py` downloads it once at a
  pinned commit, ~16 MB of meshes, not vendored): our arm, in LeRobot's "new" calibration (each joint's zero
  mid-range, as our readings are), STS3215 servos (position control, kp, damping, armature, friction from the Open
  Duck Mini's measurements), a backlash class, and the wrist camera's mount. **Not** the XLeRobot repository's
  MuJoCo model: its arms are the older SO-100 (`Rotation`, `Pitch`, ...), with no cameras, the jaw free over +-180
  deg, and the jaws colliding as whole meshes.
- **What MuJoCo collides with** (`sim/render_random.py` draws it): each mesh's convex hull. For the arm that is fine;
  for the claw it fills the jaw's inner curve, so an object cannot seat as deep as on the robot and contacts land on
  the hulls' faces.

## Next

1. **The jaws' contacts:** pads (boxes, capsules) where the real claw touches, the meshes' hulls left to the rest.
2. **A grasp check before any RL:** open, approach a cylinder, close, lift: does it hold, as the real claw holds a
   bottle? Tune the friction, the contacts' softness (`solref`, `solimp`) and the gripper servo's force.
3. **The wrist camera** from our calibration (its pose on the gripper, its lens), the fingers in its view; and the arm
   placed as the calibrated model has it.
4. **The grasp environment:** starts near an attack pose (as the head's plan leaves it), the robot's errors
   randomized; then the teacher.
