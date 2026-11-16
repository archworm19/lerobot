# Hardware risks

What on the robot can change without anything saying so, and quietly spoil a calibration, a measurement or a
simulation: what was seen, with the numbers, how to tell, and what to do. Newest first. Tuning findings (gains,
limits) are in [tuning_notes.md](tuning_notes.md); a session's own story in [sessions.md](sessions.md).

## The wrist cameras are not the module their mount is drawn for (2026-11-16)

**Seen.** TheRobotStudio's wrist camera mount (`SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`, the one in their
MuJoCo model and so in `sim2real`) holds a 32 x 32 mm UVC module with its lens in the middle. Both calibrated wrist
cameras (`calib_20261115_grippers.urdf`) look along that module's axis to within 1-3 deg, but their optical centers
sit 8.4-9.3 mm off its lens axis (away from the mount's bracket, toward the side away from the gripper's body) and
~3 mm toward the moving jaw. Measured on the robot (the right arm; the left alike): the camera is a **45 x 45 mm**
board, its lens about centered on it (25 mm from one edge), its front 8-10 mm past the plastic (with the focus).
A 45 mm board cannot put its lens where a 32 mm module's would be: (45 - 32) / 2 = 6.5 mm for a board seated against
one side, the rest the mount's own make.

**What it means.** The design's wrist camera pose (the 32 mm module's) is ~1 cm off the real one: a calibration's
correction to it is that large, and its prior has to allow it (`calibration_plan.md`, item 3). If the cameras or their
mounts are changed, recalibrate; a drawing of the mount actually used would let the design say where they are.

## The wrist camera's mount catches the arm going home with the wrist rolled (2026-11-15)

**Seen.** At the end of `joint_right_full4` the arm went home from its last low pose (the wrist rolled ~140 deg to show
the head the fingertip): folding with the roll still turning, the wrist camera's mount came into the upper arm and
caught its sleeve. In the simulation (`sim2real`, the mount's own meshes, from the session's 39 poses): all joints
together, the mount came within 0.0-0.3 cm of the arm from every pose rolled more than ~140 deg (the sleeve takes
more); the roll home first, the arm still out, then the rest: never under 2.0 cm, its gap at rest.

**Done.** The server's going to rest (`MotionController.home`: every script's end, Ctrl+C, a lost client) unrolls an
arm first when its wrist is more than `UNROLL_FIRST_DEG` (20) from home (`server/motion.py`). The Pi's server has to
run the new code.

## The wrist cameras move on their mounts (2026-11-15)

**Seen.** Three gripper sweeps of the left arm over about an hour (`calibration.gripper_sweep`, the arm at the same
sweep pose each time): the fixed finger's tip in the wrist frame drifted steadily, (252.8, 314.0) -> (252.8, 311.2) ->
(250.8, 308.0) px, 6 px over the three, while within each sweep it held to about 1 px. The fixed finger is fixed to the
gripper, as the camera is: in its frame it can move only if the camera moves on the gripper. 6 px is about half a
degree, or a millimeter or two. The mounts (the SO-101 hex-nut mount for the 32x32 UVC module, printed) were found
loose and tightened (both arms).

**What it spoils.**
- The wrist camera's calibration (`WristCamera.gripper_from_camera`): everything seen from the wrist (`wrist_look`,
  `align_step`, `wrist_contacts`, the gripper sweep) assumes it.
- The gripper sweep's hinge: fitted with the camera taken as fixed, the hinge took up the camera's moves (-3.5, -1.8,
  +1.1 mm along x over the three sweeps; pooled, the tips missed by 3.2 px and the fit was refused).
- The simulation: the wrist camera is placed by its calibration (`sim2real.robot`).

**How to tell.** The fixed finger's tip in the wrist frame: it should stay where it was, and where the calibrated
model puts it. `gripper_sweep` reports both for each sweep. A move of more than a few pixels between sessions, or
from the model's, means the camera has moved: tighten the mount, then recalibrate the wrist camera.

**For the simulation.** A mount that settles can shift by about this much at any time: train with the wrist camera's
pose varied by about as much (about half a degree, a millimeter or two; `docs/sim2real_grasp.md`).

## Arm moves fall short under load (2026-11-08 on)

The servos are position-controlled without an integral term, so a joint holding against gravity settles short of its
target: the elbow 1-3 deg (bowl6, `tuning_notes.md`), moves along the approach ~1 cm short over a few centimeters
(bowl8). Plans aim a little high (`SAG_LIFT_M`), and the last centimeters are planned from a fresh view.

## The wrist camera hangs below the gripper when it is upside down (ruler1, 2026-11-14)

With the gripper rolled over, the wrist camera hangs ~5 cm below the pinch point, and on ruler1 it hit the table at an
attack point the plan thought clear. Grasp candidates keep the camera `CAMERA_CLEARANCE_M` from every obstacle
(`harness.grasping`), and the gripper sweep checks the table's clearance in the simulation before moving.
