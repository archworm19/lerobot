# felix_jr

Bring-up and teleop tools for two SO-101 follower arms (`left_arm`, `right_arm`), an optional two-wheel base
on one arm's servo bus, and SpaceMouse teleop. Everything runs as a module:

```bash
uv run python -m lerobot.felix_jr.<folder>.<script> ...
```

Serial ports for each arm are in [current_ports.md](current_ports.md). Findings from tuning the hardware are in
[tuning_notes.md](tuning_notes.md); the RealSense camera on this Mac is covered in [realsense_m1.md](realsense_m1.md).

## Layout

What is in each folder, which computer it runs on, and where the state it keeps lives (robot settings on the robot,
the calibrated URDF on your computer, recordings in `scratch/`) are in [layout.md](layout.md).

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
   for the server (see [server_plan.md](server_plan.md)); on the computer that runs the camera calibration
   (section 6) add `--extra felix_jr` (pinocchio, placo and scipy). A plain `uv sync` lacks `pyserial`, so the
   scripts would fail with `No module named 'serial'`.
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
   the color. The folder also loads with `harness.Scene.from_saved` (see [harness.md](harness.md)).
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

### 6. Calibrate the camera to the arms (clicks, needs the server)

Finds the corrections to the robot's URDF (head and camera offsets, arm mounts, arm joint zeros, the pinch point
offset) so a trajectory can be projected into the camera image. You click the pinch point of the closed gripper in
about 30 to 60 frames per arm. The result is one file, `calibration.urdf`: a URDF any tool can load (placo,
`RobotKinematics`, a viewer), which also carries what a refit needs. Background and how to judge the result are in
[camera_calibration.md](camera_calibration.md).

1. Server on the Pi with the arm (the left arm also carries the head): `server.server --left-port <PORT> ...`.
   Record the head's home first, for the eyes **and the neck** (`setup.head_range`, then
   `setup.head_range --name head_neck` with the head looking straight ahead). The servos report their position in
   raw ticks; the home is what says which tick count means "straight ahead", and every session that is fitted
   together must use the same one. Without a neck home the script takes wherever the neck happens to be as its
   reference, and the circle starts off sideways.
2. On your computer, with the workspace in front of the robot clear and the arm free to move:
   `uv run python -m lerobot.felix_jr.calibration.collect tcp://<pi>:5555 --arm left --out-dir ~/cal_left
   [--start-from <an earlier calibration.urdf>]`
   A window shows the camera with a cyan circle where the model predicts the pinch point. `n` moves the arm
   (slowly) and the head to the next view: the pinch point in the middle of the picture, seen from the side the
   wrist camera does not block. Click where the closed fingertips meet; arrow keys nudge the head (do this now and
   then, so the head's parameters are not confounded with the arm's); `u` undoes a click; `r` refits and prints the
   error; `q` finishes. Each click refits the model, so the circle closes in on the pinch point. Without
   `--start-from` the first views may miss (the head's zero angles are unknown): nudge the head until the gripper
   shows, and click. Views keep the pinch point above `--floor-z` (default: the arm base's height), but nothing
   checks for other collisions: clear the space and watch the arm. Ctrl+C is safe (the robot stops and goes to
   rest).
3. On `q` it fits everything and writes `~/cal_left/calibration.urdf` and `calibration.md`: the held-out pixel
   error before and after, which parameters were learned, and the largest remaining errors (re-click or undo
   those).
4. The right arm, in the same fit as the left one, so the head and camera get one estimate:
   `uv run python -m lerobot.felix_jr.calibration.collect tcp://<pi>:5555 --arm right --out-dir ~/cal_right
   --start-from ~/cal_left/calibration.urdf --with ~/cal_left`
   Its `calibration.urdf` covers both arms. To refit any set of sessions later (e.g. after more clicks):
   `uv run python -m lerobot.felix_jr.calibration.xle_fit ~/cal_left ~/cal_right --out calibration.urdf`.

An earlier calibration (`--start-from`) is only a starting point, so the first view already shows the gripper; the
fit itself uses every session's clicks with the wide default priors, so no click counts twice. The sessions must
share the head's servo map (the script refuses otherwise), so do not re-record the head's home between them, and
recalibrate from scratch if the camera was remounted or the head taken apart.

**The pinch point and the wrist camera.** The pinch point is only visible when the camera looks at the gripper
from the side, and from the side *without* the wrist camera, whose board otherwise covers it. Views roll the wrist
to show that side (`--side positive`, the default; `--side negative` if your wrist camera is on the other side).
Rolls stay within `--max-roll` (130 degrees): keep an eye on the wrist camera's cable.

`collect --resume` continues an earlier folder. Folders from the first version of `collect` (with
`kinematic_calibration.json`) are read by `xle_fit` and `--with` as they are.

### 7. Measure the gripper's opening (calipers, needs the server)

Finds how far `gripper.pos` opens each arm's jaws, so plans can open the gripper to a width instead of guessing. The
gripper is a claw (the moving jaw turns about a hinge ~8 cm behind the tips), so each gap you measure is a chord of
the circle the tip turns on: `gap = 2 r sin(theta / 2)`, with the opening angle theta at the servo's own scale
(degrees per unit from the gripper's calibrated range, about 1.27). The script fits where the tips meet and the
radius r. It needs a calibrated URDF from section 6 (the model's radius is the starting point and the check).

1. Server on the Pi with the arm: `server.server --right-port <PORT>` (or `--left-port`).
2. On your computer, with calipers at hand:
   `uv run python -m lerobot.felix_jr.calibration.gripper_step tcp://<pi>:5555 --arm right
   --start-from <calibration.urdf> --out <new calibration.urdf>`
   The gripper steps through `--positions` (default 10, 25, ..., 85). At each, measure the gap between the
   fingertips, inner faces at the very tips, and type it in mm (Enter skips, q stops). A window plots the gaps
   and the opening angles they give, with the fit, after each one, so a misread gap stands out (`--no-plot` for
   none).
3. It prints each measurement against the fit, where the tips meet (`gripper.pos`, about 0-1 if the arm was
   calibrated with the jaws closed), the radius, and how far the gaps scatter around the fit, and saves the plot next
   to `--out` (`.png`). The radius should be within a few mm of the model's distance from the pinch point to the hinge
   (more if the calipers touch beyond the pinch point: 86.8 mm against 80.9 on the right arm); a warning means it is
   not, or that the gaps scatter more than 2.5 mm: check where the calipers touch, and measure again.
4. `--out` is the `--start-from` calibration plus the arm's map; `--start-from` is never overwritten. Do the other arm
   with `--start-from` the file just written. Redo it after `lerobot-calibrate` on that arm: the map is tied to the
   gripper's LeRobot calibration (as the robot reports it), and the file records which one. The script refuses a
   `--start-from` made under an older calibration of the arm (recalibrate the camera first). `--refit` refits the
   stored measurements with no robot. A camera calibration started from this file (`collect` /
   `xle_fit --start-from`) keeps the maps.

### 8. Measure the arm's shapes: the gripper body and the forearm as boxes (a ruler, needs the server)

The grasping's line-of-sight checks (`object_grasping.md`) need the gripper's and the forearm's shape. Each is a box
square to its link: measure it with a ruler, type the sizes, and check it drawn over the head camera's picture (the
rest of the calibration stays as it is). The fingers are not measured: the grasping puts 1 cm x 1 inch stand-in boxes
where the calibration puts the tips. (The boxes were first fitted to clicked outlines: fragile, 2026-11-06.)

1. Measure each part with a ruler, in cm:
   - **gripper body:** its length from the wrist toward the tips, its width across the jaws (the way they close),
     its depth along the jaws' hinge;
   - **forearm** (with its sleeve): its length from the elbow toward the wrist, its width side to side (along the
     elbow's axis), its height from the line through the elbow's and the wrist's axes down to its bottom.
2. Server on the Pi with the arm (and the head): as for section 6.
3. On your computer, with the workspace clear:
   `uv run python -m lerobot.felix_jr.calibration.arm_shapes tcp://<pi>:5555 --start-from <calibration.urdf>
   --out-dir <a scratch folder>` (`--arm right` to skip the question). For each part, the head turns to it and the
   terminal asks for its sizes (Enter keeps the ones shown: the calibration's box, if it has one). The box is then
   drawn orange over the frame, with an arrow from its center along each direction (also saved as
   `<arm>_<part>.png`), and the terminal asks: `y` it sits on the part, keep it; `s L W D` new sizes; `m L W D`
   shift it (cm, along the arrows); `v` another view (moves the arm and the head); `h N E` turn the head (deg);
   `q` stop without writing. The gripper body starts at the wrist roll's axis and runs toward the tips; the forearm
   is centered between the elbow and the wrist, its top on the joints' line: shift them onto the real parts.
4. It writes `calibration.urdf` (the start with the boxes) and `arm_shapes.md` (the sizes and shifts) in the folder.
5. The other arm: run again with `--start-from` the file just written. Then check the whole calibration (section 9).

### 8b. Calibrate the wrist cameras (a checkerboard, needs the server)

The wrist cameras (plain USB / UVC boards, one per arm, plugged into the Pi) are read by the server. Their lens
(intrinsics) and their pose on the gripper (extrinsics) go into the calibration, so a wrist picture can be measured in
the jaws' frame (`object_grasping.md`, section 6).

0. **Let your user read the cameras** (once per Pi): `groups` must list `video`; if not,
   `sudo usermod -aG video $USER`, then log in again (or reboot, if the server runs as a service). Without it a
   camera is found but does not open. (The RealSense works without it: librealsense reaches it over USB directly.)
1. **Find which camera is which arm's** (once, and again if a camera moves to another USB port): on the Pi, with
   the server stopped, `uv run python -m lerobot.felix_jr.setup.find_wrist_cameras` (`--arms right` for one;
   `--snapshot-dir ~/wrist_check` keeps a test frame from each to look at). For each arm it asks you to unplug that
   arm's camera, sees which one vanished, waits for you to plug it back into the same port and takes a frame from it.
   A camera is known by its USB port (generic camera boards have no serial number): keep each in its port. `--show`
   prints what is saved.
2. **Server:** started as usual, it opens the saved wrist cameras (640 x 480 MJPG at 30 fps); a saved camera that is
   unplugged is left out with a warning. `--left-wrist-camera` / `--right-wrist-camera <device>` override,
   `--no-wrist-cameras` skips them.
3. **The board:** a printed checkerboard, taped flat on something stiff. Count its inner corners, the corners where
   four squares meet: one fewer each way than the squares (9 x 7 squares are 8 x 6), and measure one square
   (printers scale). Asked for a wrong count, the detector can still "find" the board with a row on the paper's edge,
   so the status line warns when the board looks like another count. Any count works; with one odd and one even
   (e.g. 9 x 6) the board's two ends look different, otherwise the script tells them apart from the arm's motion.
4. **On your computer, with the workspace clear:**
   `uv run python -m lerobot.felix_jr.calibration.wrist_camera tcp://<pi>:5555 --start-from <calibration.urdf>
   --out-dir <a scratch folder>` (`--arms right` for one arm; `--pattern 9x6 --square-mm 25` to skip the
   questions; `--lens fisheye` for a lens wider than about 120 deg). For each arm in turn, after a `y` in the terminal,
   the arm goes to a start pose (the gripper tilted 70 deg down, about 18 cm over the table) and a window shows its
   wrist camera. Keys: arrows, `w` / `s` move the arm; `[` / `]` tilt it; `,` / `.` turn it about its axis;
   space takes a view; `u` drops the last one; `c` goes on; `q` stops.
   - **The focus:** the window streams the camera with a sharpness score and the best so far. Hold the board about
     20 cm in front of the camera and turn the lens (it screws in and out of its holder) until the score peaks:
     "in focus" from about 100 (out-of-focus lenses read ~2), and the board's corners are drawn once it is found.
     Enter or a click in the window accepts (`--no-focus` skips this, once the lenses are focused).
   - **The lens:** pick the board up and hold it in front of the camera, at 20-40 cm, tilted 20-45 deg every way
     (toward each side, top and bottom), turned, and into each part of the picture. A view is kept by itself when the
     board is found in a pose unlike the kept ones (moved, resized, tilted or turned); space keeps one anyway. Views
     of the board flat on the table differ only in where it is: they leave the focal length trading off with the
     distance, and the camera's pose then comes out a few mm off along its axis. `c` once there are 12 (the RMS error
     should be well under 1 px).
   - **The camera's pose:** lay the board flat on the table under the camera (its corners drawn), where it stays
     from now on, and press `c`. Then `n`, again and again: each moves the arm to the next of 12 suggested views
     around the start pose (tilted, turned, shifted) and takes it if the board is in view (if not, move the arm by
     the keys and press space, or `n` to skip). The status line counts the views and how far they turn; `c` once it
     says ENOUGH (8 views turned 15 deg about two axes).
   - **The check:** the model's fingertips are drawn on the live picture (cyan the fixed tip, magenta the moving
     one): they should sit on the real ones. `c` to finish the arm (`u` for more views).
5. It writes `calibration.urdf` (the start with the wrist cameras) and `wrist_cameras.md` (the lens, the pose, the
   errors) in the folder, and keeps the views: `wrist_camera --refit --start-from <urdf> --out-dir <folder>` fits
   them again with no robot.

### 9. Check the calibration on the robot (no clicks, needs the server)

Draws where the calibrated URDF puts both fingertips over the head camera, view after view, with the gripper at
different openings: the pinch point (cyan, placed by the camera calibration) and the moving jaw's tip (magenta,
placed by the gripper map from section 7 as well), and the wrist cameras' plates (yellow outlines) and the arms'
boxes (orange outlines), from section 8, where the URDF has them, each named above its outline (the other
arm's as, e.g., "left forearm"). It needs a URDF with the arm's gripper map, made for this robot
(the same check as the harness's: a recalibrated arm or flipped head directions are refused).

1. Server on the Pi with the arm (and the head): as for section 6.
2. On your computer, with the workspace clear:
   `uv run python -m lerobot.felix_jr.calibration.check tcp://<pi>:5555 --arm right --model <calibration.urdf>
   --out-dir <a scratch folder>`
   `n` moves to the next view (the same views `collect` uses) with the gripper at the next of `--openings`
   (default 0, 15, 30, 50, 70); `[` / `]` close / open it by `--gripper-step` at the same view; the arrow keys turn
   the head; `q` quits and sends the robot to rest.
3. Read the result: both circles should sit on the fingertips everywhere. An offset of the pinch point that changes
   with the pose or the head is the camera calibration (recalibrate, section 6); a pinch point on target with the
   moving tip off, growing with the opening, is the gripper map (re-measure, section 7); a plate's or a box's
   outline off the real one is its fit (section 8). With `--out-dir`, every
   frame is saved as drawn and raw, with `frames.json` (the measured state and both predictions per frame).

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

- `RobotMismatchError: the calibration does not match the robot` when the harness connects: the calibrated URDF
  was made under settings that have changed on the robot since. The message lists each mismatch and its fix:
  - `head_<servo>: sign`: the head's directions changed (`setup.head` redone), so every head angle is inverted;
    recalibrate the camera (section 6). A re-recorded head home is not a mismatch: it only changes where the head
    rests.
  - `<arm> arm: <motor> recalibrated on the robot`: the arm was recalibrated with `lerobot-calibrate`, so its joint
    angles mean something else than when you clicked; recalibrate the camera (section 6).
  - `<arm> gripper map: gripper recalibrated`: rerun `calibration.gripper_step` (section 7).

  Calibrations made before the URDF kept a snapshot of the arms' LeRobot calibration cannot be checked for the arms
  (a warning is logged); recalibrating the camera adds one. Older servers (protocol 3) lack the `arm_calibration`
  command the check uses: update the Pi's checkout.

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

`uv run pytest tests/felix_jr -q` runs everything; none of it needs hardware (the server and calibration tests
use a fake robot behind a real socket). The calibration tests need `--extra felix_jr` and are skipped without it.
For a quick run, `-m "not smoke and not slow"` skips the end-to-end tests in `tests/felix_jr/smoke_tests/` and the
tests that wait on real timeouts (watchdogs, heartbeats).
