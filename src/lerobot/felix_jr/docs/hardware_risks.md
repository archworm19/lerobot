# Hardware risks

What on the robot can change without anything saying so, and quietly spoil a calibration, a measurement or a
simulation: what was seen, with the numbers, how to tell, and what to do. Newest first. Tuning findings (gains,
limits) are in [tuning_notes.md](tuning_notes.md); a session's own story in [sessions.md](sessions.md).

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
