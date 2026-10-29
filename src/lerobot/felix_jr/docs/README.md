# felix_jr

Bring-up and teleop tools for two SO-101 follower arms (`left_arm`, `right_arm`), an optional two-wheel base
on one arm's servo bus, and SpaceMouse teleop. Everything runs as a module:

```bash
uv run python -m lerobot.felix_jr.<folder>.<script> ...
```

Serial ports for each arm are in [current_ports.md](current_ports.md). Findings from tuning the hardware are in
[tuning_notes.md](tuning_notes.md); the RealSense camera on this Mac is covered in [realsense_m1.md](realsense_m1.md).

## Layout

| Path | What it is |
|---|---|
| `home.py` | Shared: follower construction (`make_follower`), home pose files, the trapezoid-profile `move_all`. |
| `depth.py` | Shared: align RealSense depth to the color image, depth-edge overlays and checks. |
| `scene.py`, `overlay.py` | Shared: ask a color + depth capture for 3D points, projections and path checks; draw paths on it. |
| `base_follower.py` | Shared: `SO101FollowerWithBase` (per-joint P gains, optional wheel servos). |
| `setup/` | One-time, per-robot: `record_home`, `wheels`, `head`. |
| `checks/` | Hardware sign-of-life tests: `wiggle_test`, `wheel_test`. |
| `diagnostics/` | Tuning and debugging: `lift_test`, `dump_servo_config`, `set_servo_id`, `realsense_check`, `alignment_from_saved`. |
| `server/` | Robot server for the Raspberry Pi and its client (`FelixClient`); plan in `server_plan.md`. |
| `teleop/` | `spacemouse_teleop`. |
| `calibration/` | Camera-to-robot calibration work: numpy URDF kinematics (`urdf_kinematics`) and the URDF sensitivity analysis (`sensitivity`); plan in `camera_calibration.md`. |
| `docs/` | This file, `tuning_notes.md`, `realsense_m1.md`, `scene_tools.md`, `head.md`, `camera_calibration.md`, `sensitivity_default_urdf.md`, `server_plan.md`, `current_ports.md`. |

## First-time order

Use each arm's own port and id (`left_arm`, `right_arm`).

1. **Calibrate each arm** with LeRobot's own tool (writes the calibration file this code relies on):
   `uv run lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>`
2. **Record each arm's home pose.** Hold the arm in its rest pose, then:
   `uv run python -m lerobot.felix_jr.setup.record_home <PORT> <ID>`
3. **Set up the wheels** (only if the base is fitted). Needs the arm unplugged from the board and one wheel
   disconnected from the other for the first step; the script prompts through it. Pass the port and id of the
   arm whose board the wheels plug into (the right arm here):
   `uv run python -m lerobot.felix_jr.setup.wheels <PORT> <ID>`
3b. **Set up the head** (only if it is fitted; details in [head.md](head.md)). Same pattern: the arm unplugged
   from the board and the neck disconnected from the eyes for the first step. Pass the port and id of the arm
   whose board the head plugs into (the left arm here):
   `uv run python -m lerobot.felix_jr.setup.head <PORT> <ID>`
4. **Check the arms move.** `uv run python -m lerobot.felix_jr.checks.wiggle_test --left <PORT> --right <PORT>`
   (`--no-base` / `--no-head` ignore the wheel / head servos).
5. **Check the wheels** with the robot lifted:
   `uv run python -m lerobot.felix_jr.checks.wheel_test <PORT> <ID>`
6. **Teleop.** `uv run python -m lerobot.felix_jr.teleop.spacemouse_teleop <PORT> <ID> --max-step 15`.
   See the script's docstring for every option.

## When something looks wrong

- `diagnostics.lift_test` drives only `shoulder_lift` and prints the servo's load, current and voltage per
  phase. `--p-lift N` and `--max-step DEG` override the P gain and the per-command cap for one run.
- `diagnostics.dump_servo_config --left <PORT> --right <PORT>` prints the limit and gain registers of every arm
  servo without changing anything.
- `teleop.spacemouse_teleop --log run.csv` records commanded and measured joints and tip position every tick.

## Setup notes

- Teleop needs `pyspacemouse`, the hidapi system library (`brew install hidapi` on macOS) and `placo`. They
  are installed in the project environment with `uv pip install`, not tracked in `pyproject.toml`, so a later
  `uv sync` removes them.
- The IK reads the SO-101 URDF from `~/.cache/felix_jr/SO-ARM100/Simulation/SO101/` (clone of
  `TheRobotStudio/SO-ARM100` with its `assets/`).
- 3Dconnexion's own driver can block the SpaceMouse from being opened; see the tips in the teleop script's
  docstring and `tuning_notes.md`, section 3.8.
- State is saved next to each arm's calibration, in
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`: `<id>_home.json`, and `<id>_base.json` for
  the arm with the base.

## Tests

`uv run pytest tests/felix_jr -q` runs the tests for the geometry helpers (`depth.py`, `scene.py`, `overlay.py`). They
need no hardware.
