# Layout of felix_jr

What is where in `src/lerobot/felix_jr/`, which computer each part runs on, and where the state it keeps lives. How
to run things is in [README.md](README.md); why things are the way they are is in the design documents listed at the
end.

## Two computers

- **The robot's computer** is the one the arms are plugged into: the Raspberry Pi on the robot in normal use, or
  your laptop when you plug the USB cables in there (teleop, the local checks). On the Pi, `server/` runs the robot
  server, which owns the serial ports and the camera.
- **Your computer** (the laptop) runs the experiments: the harness, the camera calibration, and anything that talks
  to the server through `FelixClient`.

Most scripts run on your computer against the server. A few talk to the servos directly (`setup.find_ports`,
`setup.record_home`, `setup.wheels`, `setup.head`, `checks.wiggle_test`, `checks.wheel_test`, `teleop/`, most of
`diagnostics/`), so they run on whichever computer the arm is plugged into.

## Three kinds of state

Each kind is written by one part of the code and lives in one place:

| Kind | Written by | Lives | Used by |
|---|---|---|---|
| **Robot settings**: what the servos' readings mean, and where the robot may go | `lerobot-calibrate`, `setup/` | the robot's computer, next to each arm's LeRobot calibration (below) | the server and the local scripts |
| **Calibration**: the robot's geometry as the camera sees it | `calibration/` | a calibrated URDF on your computer (`urdfs/xlerobot_calib<date>.urdf` once committed) | the harness, reach and the renders |
| **Recordings**: captures, renders, logs, clicks | the harness, `collect`, the checks | a `scratch/` folder (gitignored) or a folder you name | you, later; write-ups in `sessions.md` |

The robot settings, in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` on the robot's computer:

| File | Written by | What it holds |
|---|---|---|
| `<id>.json` | `lerobot-calibrate` (LeRobot's own) | per motor: homing offset, direction, calibrated tick range; turns ticks into joint angles and `gripper.pos` |
| `<id>_home.json` | `setup.record_home` | the arm's rest pose (where it goes home to, and when a client goes silent) |
| `<id>_base.json` | `setup.wheels` | the wheel servos' forward directions |
| `<id>_head.json` | `setup.head` | the head servos' directions (down / left) |
| `<id>_head_range.json` | `setup.head_range` | the head servos' limits and home, in raw ticks |

The serial port of each arm is remembered per computer, in `~/.cache/huggingface/lerobot/calibration/felix_jr/
ports.json` (`setup.find_ports`). A computer that talks to the arms directly needs a copy of the arm files
([README.md](README.md) section 2c).

## setup/ and calibration/

The two folders are split by **where their output lives**, not by where they run (`setup.head_range` runs on your
computer, but what it records is kept on the robot):

- **`setup/` writes robot settings.** The robot uses them itself, at run time: the server clamps head moves to the
  recorded range and sends the arms to their recorded homes, with no calibration file involved. The Pi does not
  even install what reading a calibrated URDF needs.
- **`calibration/` writes the calibrated URDF.** Only the planner on your computer uses it.

The calibration is built on top of the robot settings:

```
servo encoder ticks
   │  robot settings (LeRobot calibration, head directions), on the robot's computer
   ▼
joint angles, gripper.pos, head angles                 ← what the server reports
   │  calibration (the URDF's corrections, the gripper map), on your computer
   ▼
geometry: where the gripper is, how wide the jaws open ← what the planner uses
```

So changing a robot setting silently invalidates the calibration made on top of it: rerunning `lerobot-calibrate`
moves what every joint angle means, and redoing `setup.head` can flip what a head angle means. To catch that, a
calibrated URDF records the robot settings it was made under (each clicked arm's LeRobot calibration, the head's
directions, the gripper's LeRobot calibration for each gripper map), and when the harness connects,
`XleRobotModel.check_robot` compares them with what the robot reports and refuses a mismatch, naming what changed
and what to redo. Anything new that depends on a robot setting should be recorded and checked the same way.

Not every robot setting is under the calibration. The head's recorded home (`setup.head_range`) is only where the
head rests: the calibration turns the raw head ticks into angles with its own references and fitted zeros, so
re-recording the home changes nothing it computes. Ask what a setting changes in what the server reports before
recording it.

## The directory

| Path | What it is | Runs on |
|---|---|---|
| `home.py` | Follower construction (`make_follower`), home pose files, the trapezoid-profile `move_all`. | shared |
| `base_follower.py` | `SO101FollowerWithBase`: an SO-101 follower whose bus can also carry the wheels and the head. | shared |
| `ports.py`, `port_lock.py` | The remembered arm-to-port map; one process at a time on a serial port. | shared |
| `depth.py` | Aligning RealSense depth to color, and checking the alignment. | shared |
| `setup/` | Robot settings, once per robot: `find_ports`, `record_home`, `wheels`, `head` (servo IDs and directions), `head_range` (limits and home, through the server). | robot's computer, `head_range` yours |
| `checks/` | Sign-of-life tests: `wiggle_test`, `wheel_test` (direct), `server_check`, `server_wiggle` (through the server). | either |
| `diagnostics/` | Tuning and debugging: `lift_test`, `dump_servo_config`, `set_servo_id`, `realsense_check`, `alignment_from_saved`. | where the hardware is plugged in |
| `teleop/` | `spacemouse_teleop`. | robot's computer |
| `server/` | The robot server (`server`, `hardware`, `motion`, `head`), its wire format (`protocol`) and its client (`client`, `FelixClient`). `server --fake` runs a fake robot anywhere. | the Pi; the client yours |
| `calibration/` | The calibrated URDF: the robot model (`xle_robot_model`), the merged URDF (`merge_urdf`), choosing views (`views`), clicking (`collect`), the fit (`xle_fit`), the gripper's opening (`gripper_step`), the wrist cameras' plates (`wrist_camera_board`), checking it on the robot (`check`), and what `collect` and `check` share (`live_view`: driving the arm and the head through planned views). | yours |
| `harness/` | The VLM harness: objects and the workspace DAG (`objects`, `workspace`), geometry (`scene`, `tools`, `reach`, `look`), drawing (`render`), the startup scan (`scan`), execution (`execute`), and every tool the VLM can call with the runner that asks a person before a move (`api`). | yours |
| `harness/scratch/` | Gitignored: `session.py`, the CLI that drives the harness by hand, and `sessions/<name>/`, what each session recorded. | yours |
| `urdfs/` | `xlerobot.urdf` (the XLeRobot) and `so101_new_calib.urdf` (the official SO-101, Apache 2.0), merged into `xlerobot_so101.urdf` (the head and base with two SO-101 arms); `xlerobot_calib<date>.urdf`, committed calibrations. | — |
| `docs/` | The documents below. | — |

Tests are in `tests/felix_jr/`, one file per module or flow, none needing hardware; end-to-end tests are in
`tests/felix_jr/smoke_tests/`, and frozen reference data in `tests/felix_jr/reference/`.

## The documents

- **Run books:** [README.md](README.md) (first-time setup, the server, calibration, checks, teleop, troubleshooting,
  tests), [current_ports.md](current_ports.md), [realsense_m1.md](realsense_m1.md) (the RealSense on the Mac).
- **Design:** this file, [harness.md](harness.md), [camera_calibration.md](camera_calibration.md),
  [server_plan.md](server_plan.md), [head.md](head.md).
- **Experiments:** [sessions.md](sessions.md) (each harness session on the robot), [tuning_notes.md](tuning_notes.md)
  (hardware tuning), [open_issues.md](open_issues.md) (what is missing or wrong).
