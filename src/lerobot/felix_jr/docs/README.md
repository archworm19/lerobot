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
| `ports.py` | Shared: the saved arm-to-port cache (`setup.find_ports` writes it, `resolve(robot_id)` reads it). |
| `so101_new_calib.urdf` | The official SO-101 URDF (Apache 2.0), used by the calibration for the arm. |
| `base_follower.py` | Shared: `SO101FollowerWithBase` (per-joint P gains, optional wheel servos). |
| `setup/` | One-time, per-robot: `find_ports`, `record_home`, `wheels`, `head`. |
| `checks/` | Hardware sign-of-life tests: `wiggle_test`, `wheel_test`. |
| `diagnostics/` | Tuning and debugging: `lift_test`, `dump_servo_config`, `set_servo_id`, `realsense_check`, `alignment_from_saved`. |
| `server/` | Robot server for the Raspberry Pi and its client (`FelixClient`); plan in `server_plan.md`. |
| `teleop/` | `spacemouse_teleop`. |
| `calibration/` | Camera-to-robot calibration: numpy URDF kinematics (`urdf_kinematics`), the sensitivity analysis (`sensitivity`), and the click calibration (`model`, `design`, `collect`, `fit`, `urdf_patch`); plan in `camera_calibration.md`. |
| `docs/` | This file, `tuning_notes.md`, `realsense_m1.md`, `scene_tools.md`, `head.md`, `camera_calibration.md`, `sensitivity_default_urdf.md`, `server_plan.md`, `current_ports.md`. |

## First-time order

Two kinds of setup. **A new arm** is done once per arm, and its results stay with the arm. **A new computer**
is done once per computer that will talk to the arms. A brand-new robot on its first computer needs both: do the
new-computer steps 2a-2b first (the arm steps need a port), then the new-arm steps, then the checks.

Arms are called `left_arm` and `right_arm`; use each arm's own port and id.

### 1. Set up a new arm (once per arm)

Needs the arm's port: run `setup.find_ports` (step 2b) first if this computer doesn't know it yet.

1. **Calibrate** with LeRobot's own tool. It writes the calibration into the servos and into
   `<id>.json` on this computer, which this code relies on:
   `uv run lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>`
2. **Record the home pose.** Hold the arm in its rest pose, then:
   `uv run python -m lerobot.felix_jr.setup.record_home <PORT> <ID>`
3. **Set up the wheels** (only if the base is fitted). Needs the arm unplugged from the board and one wheel
   disconnected from the other for the first step; the script prompts through it. Pass the port and id of the
   arm whose board the wheels plug into (the right arm here):
   `uv run python -m lerobot.felix_jr.setup.wheels <PORT> <ID>`
4. **Set up the head** (only if it is fitted; details in [head.md](head.md)). Same pattern: the arm unplugged
   from the board and the neck disconnected from the eyes for the first step. Pass the port and id of the arm
   whose board the head plugs into (the left arm here):
   `uv run python -m lerobot.felix_jr.setup.head <PORT> <ID>`

Then run the checks below. Redo the calibration (steps 1-2) if an arm is taken apart or a servo is replaced.

### 2. Set up on a new computer

The servos keep their calibration and IDs, so nothing needs recalibrating. This computer needs the software, the
port names, and a copy of the arm files.

a. **Install.** `uv sync --locked --extra feetech` (arms, wheels and head). On the Pi add `--extra pyzmq-dep`
   for the server (see [server_plan.md](server_plan.md)). A plain `uv sync` lacks `pyserial`, so the scripts
   would fail with `No module named 'serial'`.
b. **Find each arm's port.** Plug in both arms' USB cables, then:
   `uv run python -m lerobot.felix_jr.setup.find_ports`
   It asks you to unplug one arm's USB cable at a time (the servo power can stay on), sees which port vanished,
   waits for you to plug it back in, and prints the result, e.g. `/dev/tty.usbmodem5AE60837301 right_arm`. Use
   those names wherever a `<PORT>` appears, and copy them into [current_ports.md](current_ports.md). The ports
   are remembered by the USB adapter's serial number, so `find_ports --show` still reports the right port name
   if it changes after a replug (on the Pi `/dev/ttyACM0` and `/dev/ttyACM1` can swap). Without the script,
   LeRobot's `uv run lerobot-find-port` does the same for one cable at a time, without saving anything.
c. **Copy the arm files** from a computer where the arms were already set up. They are everything in
   `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`: `<id>.json` (calibration), `<id>_home.json`,
   and `<id>_base.json` / `<id>_head.json` / `<id>_head_range.json` where fitted. Keep the same ids and the same
   folder, e.g.
   `rsync -av ~/.cache/huggingface/lerobot/calibration/robots/so_follower/ <host>:.cache/huggingface/lerobot/calibration/robots/so_follower/`
   Note the `robots/` level: the folder is `calibration/robots/so_follower`, not `calibration/so_follower`.
   Files in the wrong folder are silently ignored (the scripts report a missing home pose).
   Without the calibration file LeRobot would run a fresh calibration, which gives slightly different ranges
   and no longer matches the saved home poses and sign files. If it asks "Press ENTER to use provided
   calibration file", pressing ENTER is the right answer: it writes the file to the servos.
d. **Check** with the arm checks below.

### 3. Test the camera server (Pi, with the laptop)

The server (`server/`, plan in [server_plan.md](server_plan.md)) runs on the Pi; experiments talk to it from the
laptop. Test it camera-only first, then with the arms. Section 2a (install, with `--extra pyzmq-dep`) comes first.

a. **Rehearse on the laptop with no robot.** Start a fake server in one terminal and check it from another:
   `uv run python -m lerobot.felix_jr.server.server --fake`
   `uv run python -m lerobot.felix_jr.checks.server_check`
   It should print the server info, the joint positions of each arm, ten timed captures and a median time.
b. **Start the server on the Pi, camera only:**
   `uv run python -m lerobot.felix_jr.server.server`
   It finds the source-built `pyrealsense2` under `/home/zcecere/CodeRepository/librealsense/build` (change it
   with `--librealsense-build`), starts the camera and logs `listening on tcp://0.0.0.0:5555`. If it fails:
   - `no pyrealsense2*.so under ...`: the bindings were not built there (`-DBUILD_PYTHON_BINDINGS=ON`).
   - an import error naming the Python version: the venv's Python (3.12 here) must match the one the bindings were
     built for.
   - a USB permission error opening the camera: install librealsense's udev rules
     (`config/99-realsense-libusb.rules` in the librealsense repo, then `sudo udevadm control --reload-rules &&
     sudo udevadm trigger`) and replug the camera.
c. **Check it from the laptop** (use the Pi's hostname or IP):
   `uv run python -m lerobot.felix_jr.checks.server_check tcp://<pi>:5555 --save-dir ~/pi_capture`

   Typical: `uv run python -m lerobot.felix_jr.checks.server_check tcp://zcecere@ikeabot.local:5555 --save-dir ~/pi_c
apture`

   Expect `camera True`, the arms list empty (camera only), and ten captures with their image shapes and times. Compare
   `--jpeg` against the default PNG: PNG is lossless but larger, so slower over Wi-Fi. A `TimeoutError` means
   the laptop can't reach port 5555 (wrong host, or a firewall on the Pi).
d. **Look at the capture.** `uv run python -m lerobot.felix_jr.diagnostics.alignment_from_saved ~/pi_capture`
   writes `alignment.png` next to the saved color and depth; check that the depth looks sensible and matches
   the color. The folder also loads with `Scene.from_saved` (see [scene_tools.md](scene_tools.md)).
e. **Add the arms.** Stop the server, then start it with their ports (from `setup.find_ports --show`):
   `uv run python -m lerobot.felix_jr.server.server --left-port <LEFT_PORT> --right-port <RIGHT_PORT>`
   and rerun the `server_check` from step c. It should now list both arms and print their joint positions.
   The server doesn't move the arms, but connecting switches their torque on (they hold where they are), and
   stopping the server (Ctrl+C) switches it off again, so the arms go limp: support them if they are not resting.
   The positions are only meaningful if the arm files from section 2c are in place, so compare them against a
   pose you know (e.g. the arm resting at home).

### 4. Set the head's tilt range and home (once, needs the server)

The eyes servo (it tilts the camera) has no range yet, so it could be driven into its mechanical stop or the camera
cable. This script lets you tilt the head in small steps while watching the camera, and record the limits and home.

1. Start the server on the Pi with the left arm's port (the head plugs into that board), as in section 3e:
   `uv run python -m lerobot.felix_jr.server.server --left-port <LEFT_PORT>`
2. On your computer:
   `uv run python -m lerobot.felix_jr.setup.head_range tcp://<pi>:5555`
   A window shows the camera image. Click it, then: up / down arrow tilts the head by `--step` degrees (default 2);
   `u` records the **up limit**, `d` the **down limit**, `h` **home**; `g` goes to home; `q` quits (going home
   first).
3. Approach each end of the range slowly and record the limit *before* the camera cable or the mount is strained.
   The script caps the servo's torque while it runs (`--torque-limit`, default 300 of 1000), so a blocked servo
   only pushes gently.

Everything recorded is saved on the Pi in `<id>_head_range.json` next to the arm's calibration, and later head
moves are clamped to the limits. The script's docstring lists every option.

### 5. Wiggle the robot through the server (Pi, with the laptop)

The same wiggle as `checks.wiggle_test`, but the commands come from your computer through the server. It tests
the motion commands, the stop and rest behaviour, and (optionally) the camera while the robot moves.

1. On the Pi, stop anything else that uses the arms' ports, then start the server with both arms:
   `uv run python -m lerobot.felix_jr.server.server --left-port <LEFT_PORT> --right-port <RIGHT_PORT>`
   Each arm needs a recorded home (`setup.record_home`). On startup the server connects the arms; when it exits
   (Ctrl+C or a stop from systemd) it sends them to their home poses first, then releases the servos. Use
   `--no-home-on-exit` to skip that.
2. Rehearse on your computer with a fake robot first: `server.server --fake`, then
   `uv run python -m lerobot.felix_jr.checks.server_wiggle` (the fake arms move in a quarter of real time).
3. On your computer, with the arms supported and the workspace clear:
   `uv run python -m lerobot.felix_jr.checks.server_wiggle tcp://<pi>:5555 --no-base --capture-dir ~/wiggle_frames`
   It prints the same lines as the local test, nudges each joint, goes home, then wiggles the head. Drop
   `--no-base` for the wheels (robot lifted). `--capture-dir` saves a camera frame after each step, taken after
   the move finished.
4. Test the safety behaviour: press **Ctrl+C** part way through. The motion should stop and the robot go to
   rest. Then try closing the terminal or pulling the laptop off the network mid-wiggle: after about 5 seconds
   without heartbeats the server stops the motion and sends the robot to rest by itself, slowly.

How it behaves, in short: `move_joints` takes one target pose per arm and returns at once, the client polls until
it ends; `stop` aborts and holds in place; only one motion runs at a time; while arms move, `get_state` and
`capture` answer from a snapshot (nothing else may use the serial bus then). Scripts that move the robot wrap
themselves in `client.session()`, which sends it to rest on any exit; scripts that only read open no session and
are never moved. Details are in `server/motion.py` and the module docstring of `server/client.py`.

### 6. Calibrate the camera to the arm (clicks, needs the server)

Finds the corrections to the URDFs (head and camera offsets, arm joint zeros, the tip offset) so a trajectory can be
projected into the camera image. You click the tip of the closed gripper in about 30 to 60 frames. Background and
how to judge the result are in [camera_calibration.md](camera_calibration.md), "Click calibration".

1. Server on the Pi with the arm (the left arm also carries the head): `server.server --left-port <PORT> ...`.
   Record the head's home (`setup.head_range`, for the eyes and, ideally, the neck) first, since the head angles
   are measured from it.
2. On your computer, with the workspace in front of the robot clear and the arm free to move:
   `uv run python -m lerobot.felix_jr.calibration.collect tcp://<pi>:5555 --arm left --out-dir ~/cal_left`
   A window shows the camera with a cyan circle where the model predicts the tip. `n` moves the arm (slowly) to the
   next planned pose and turns the head towards the circle; click the very end of the closed gripper; arrow keys
   nudge the head (do this now and then, so the head's parameters are not confounded with the arm's); `u` undoes
   a click; `q` finishes and the robot goes to rest. The first few clicks are hard: the circle is far off until the
   head's zero offsets are learned. Watch the arm: poses are within about +/-20 degrees of the center pose, but
   nothing checks for collisions. Ctrl+C is safe (the robot stops and goes to rest).
3. Fit: `uv run python -m lerobot.felix_jr.calibration.fit ~/cal_left/observations.json --out-dir ~/cal_left`
   Read `fit_report.md`: the held-out pixel error before and after, which parameters were learned, and the
   largest remaining errors (re-click or undo those). It also writes the calibrated URDFs.
4. Do the right arm with `--arm right` (and the same head), then compare.

`calibration.design` only plans poses; `collect --resume` continues an earlier folder.

## Checks and teleop (after either setup)

1. **Check the arms move.** `uv run python -m lerobot.felix_jr.checks.wiggle_test --left <PORT> --right <PORT>`
   (`--no-base` / `--no-head` ignore the wheel / head servos).
2. **Check the wheels** with the robot lifted:
   `uv run python -m lerobot.felix_jr.checks.wheel_test <PORT> <ID>`
3. **Teleop.** `uv run python -m lerobot.felix_jr.teleop.spacemouse_teleop <PORT> <ID> --max-step 15`.
   See the script's docstring for every option.

## When something looks wrong

- `ConnectionError ... Incorrect status packet!` while a script or the server starts: a garbled reply on the servo
  bus. The followers retry the startup configuration three times, so this should now only appear if the bus is
  really unhealthy (check the servo power supply and the cable). `PortInUseError: ... in use by another process
  (pid N)` means another felix_jr program (often the previous server, which takes several seconds to shut down
  because it goes to rest first) still holds the port; wait for it or stop it. Two programs on one bus corrupt each
  other's packets, so every follower locks its serial port while connected (`port_lock.py`; LeRobot's own tools do
  not take the lock).

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
