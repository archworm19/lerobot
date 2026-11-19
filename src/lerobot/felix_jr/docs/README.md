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
   for the server (see [server_plan.md](server_plan.md)); on the computer that runs the calibration
   (sections 6-9) add `--extra felix_jr` (pinocchio, placo and scipy) and `--extra felix_jr_sim` (MuJoCo, for the
   gripper sweep). A plain `uv sync` lacks `pyserial`, so the
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

The calibration, in order (the procedure and why: [calibration_plan.md](calibration_plan.md); the fit's design:
[camera_calibration.md](camera_calibration.md)): **the wrist cameras' lenses** once per lens (section 6), then **a
joint session per arm** (section 7) each time the robot needs it: a mount moved, a servo recalibrated, the head taken
apart. The calipers (section 8) are optional. Then **the check** (section 9). Every step writes a new calibrated URDF
(`urdfs/calib_<date>_<what>.urdf` once kept); none overwrites its `--start-from`. A robot never calibrated starts
from the design, `urdfs/design_xlerobot_so101.urdf`, with its corrections at 0 (`XleRobotModel`).

### 6. Calibrate the wrist cameras' lenses (a ChArUco board, needs the server)

The wrist cameras (plain USB / UVC boards, one per arm, plugged into the Pi) are read by the server. Their lens
(intrinsics) goes into the calibration here, once per lens: redo it when a lens is refocused or a camera replaced, not
when a mount moves. Their pose on the gripper is the joint session's (section 7, which ties it to the head's view of the
fingers), a correction to the design's (`design_xlerobot_so101.urdf`): this step leaves it as
it was. (The camera-pose step, and the checkerboard, were
retired on 2026-11-16: `calibration_plan.md`.)

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
3. **The board:** the printed ChArUco (`boards.PRINTED_BOARD`: 9 x 7 squares of 25 mm, 18 mm markers,
   `DICT_4X4_50`), taped flat on something stiff; measure a square (printers scale) and pass `--board
   charuco:9x7:<square m>:<marker m>:DICT_4X4_50` if it is not 25 mm. Its markers name every corner, so a board
   partly out of the picture or partly hidden still counts: that is how the picture's corners get covered.
4. **On your computer:**
   `uv run python -m lerobot.felix_jr.calibration.wrist_camera tcp://<pi>:5555 --start-from <calibration.urdf>
   --out-dir <a scratch folder>` (`--arms right` for one arm; `--lens fisheye` for a lens wider than about 120 deg).
   After a `y` in the terminal both arms go to a start pose (the gripper tilted down, about 18 cm over the table),
   and for each arm in turn a window shows its wrist camera. Keys: arrows, `w` / `s` move the arm; `[` / `]` tilt
   it; `,` / `.` turn it about its axis; space keeps this view; `u` drops the last one; `c` fits; `x x` skips the
   arm's camera; `q` stops. The picture shows the arm and step (1/2 focus, 2/2 lens) with how far along, the key to
   press next and the last message; the terminal prints each step's keys as it starts.
   - **The focus:** the window streams the camera with a sharpness score and the best so far. Hold the board about
     20 cm in front of the camera and turn the lens (it screws in and out of its holder) until the score peaks:
     "in focus" from about 100 (out-of-focus lenses read ~2). Enter or a click in the window accepts (`--no-focus`
     skips this, once the lenses are focused).
   - **The lens:** hold the board in front of the camera at 20-40 cm and move it slowly from pose to pose, holding
     each a moment: tilted 20-45 deg every way (toward each side, top and bottom), turned, near and far, and pushed
     into each corner of the picture (it may run off the edges). The corners found are dotted green; the picture's
     cells some kept view covers are framed. A view is kept by itself when the board is **held still** (a moving
     board's corners blur), shows at least 30% of its corners, and is a new pose or covers a cell no kept view
     does: about one view per pose held, at most 100 (simulated, 2026-11-16: 60 poses over two minutes, 3600 frames,
     kept 63). The status line says why a frame was not kept. Views of the board flat at one distance leave the
     focal length trading off with the distance (wrist_cams5: anywhere from 314 to 3700 px): the status line suggests
     what would help. `c` once there are 12: the lens is fitted on random halves of the views first and accepted
     when they agree within 3%; if not, it says so and wants more views. That ends the arm's camera.
   - **Resuming:** the views are saved as they are kept; `--resume` (same `--out-dir`) keeps them and adds to them.
5. It writes `calibration.urdf` (the start with the new lenses, each camera's pose on the gripper as it was) and
   `wrist_cameras.md` (the lens, the errors) in the folder, and keeps the views: `wrist_camera --refit --start-from
   <urdf> --out-dir <folder>` fits them again with no robot. Views recorded with the checkerboard (before 2026-11-16)
   are no longer read.

### 7. The joint session: the head, an arm and its wrist camera (a ChArUco board, needs the server)

Finds the corrections to the design (the head's zeros and offsets and its camera's mount, the arm's mount, its joints'
zeros and origins, the pinch point, the gripper's hinge and where its jaws close, the wrist camera's pose on the
gripper) from one session per arm: the board seen by both cameras, both fingertips clicked in the head's picture, and
the fingertips found in the wrist camera's by a gripper sweep. About 20 minutes an arm. It needs the arm's wrist
camera lens in `--start-from` (section 6); a gripper map (section 8) is optional: without one the session starts from
the servo's and fits it.

1. **Server** on the Pi with the arm and the head (the left arm carries the head):
   `server.server --left-port <PORT> --right-port <PORT>`. The head's home recorded (section 4, the neck's too).
2. **On your computer**, the table in front of the arm clear, the board at hand (`boards.PRINTED_BOARD`, flat and
   stiff, as in section 6):
   `uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.joint_collect
   tcp://<pi>:5555 --arm right --out-dir <a scratch folder> --start-from <calibration.urdf>`
   (`--placements 2 --arm-views 10 --tip-views 10` by default; `--max-vel` the arm's speed, 20 deg/s).
3. **The gripper sweep, first**, in the terminal, the board not yet down: it prints the sim's checks of the sweep
   pose (how near the table the gripper comes), asks (Enter to start, `q` to skip), moves the arm over the table and
   shows a live preview of the fingers found (Enter sweeps, `q` stops), then opens the jaws in steps. Then the **drift
   check**: the fixed fingertip it found against where `--start-from` has it.
   - Within 5 px: on without asking.
   - 5 to 30 px: **the wrist camera moved on its mount** since that calibration (`hardware_risks.md`). If the mount is
     loose, tighten it and start again (it will move again during the session); `y` goes on.
   - Past 30 px, or no tips found: the sweep misread the fingers (its frames are in `wrist_tip_sweep/`); its tips are
     left out and `y` goes on with the head's clicks only for them. Clear the table and try again if you can.
4. **Place the board.** The window opens; the arms rest and the head looks where the board should go: lay it flat on
   the drawn outline (roughly). Its corners show green where the head finds them; arrows turn the head if it is cut
   off. `n` when it is found whole: the head looks at it from a few directions, then **plans the arm views (about 30
   s: the status line says PLANNING; wait, keys pressed meanwhile are dropped)**.
5. **Arm views** (`n` each: the arm moves slowly, watch it). The jaws open to `gripper.pos` 20, 35, 50 in turn. The
   window has three panes, the pictures of one moment:
   - **head: FIXED fingertip** (magenta): click the fixed finger's tip;
   - **head: MOVING fingertip** (yellow): click the moving finger's tip (declined with the jaws closed);
   - **wrist camera** (cyan): no clicks (the sweep has its tips).

   Each head pane circles where the model has its tip and names the face to click on (the pane's title). At the arm
   views the head looks down on the wrist camera, which mostly hides the tips: click only a tip you can see; most
   views have none, and that is expected (the tip views, 7, are for them). `u` undoes the last click.
6. **The next placement** (`--placements`, 2): the arm rests, lay the board on the new outline, as in 4-5.
7. **Tip views** (`n` each; `--tip-views`, 10): the arm turns the fingers to the head, about level, the wrist rolled
   so the head looks down onto the jaws' face (the pinch face, the side away from the wrist camera), at heights from
   just over the table to ~30 cm up; the head on the fingertips. Click both tips, every view. Leave the board where it
   is. (These replaced poses low over the table on 2026-11-17: the fingers pointed down, away from the head, which
   hardly saw the tips; the simulator, made to see only what the head can, agrees: the camera 3.4 mm off with those,
   0.8 mm with 10 tip views, 0.6 with 20.)
8. **`q`** fits and writes `calibration.urdf` and `calibration.md` in the folder (a minute or a few). Read the report:
   - **the residuals**: the boards ~0.1-0.5 px, the clicks ~1-2 px;
   - **mis-clicks**: clicks more than 30 px from the fitted tip are named, with pictures (`misclicks/`), and left
     out;
   - **the held-out moments**: the head's and the arm's chains on where the board is, at moments the fit did not
     see (a few mm);
   - **what moved**: the wrist camera on its gripper, the gripper's opening (and whether its map was the servo's
     default), the estimates that moved most;
   - **what the fit determined**: unknowns it could not tell apart (their priors settle them) and the least
     determined directions. A **failed fit** (a kind of residual far off at its median) is flagged at the top:
     do not use it; look at the mis-clicks and the board sightings, or collect again.
9. **The other arm**: its own session, `--start-from` the first's `calibration.urdf`; then both fitted together, so
   the head gets one estimate:
   `uv run --extra felix_jr python -m lerobot.felix_jr.calibration.joint_fit <left folder> <right folder>
   --start-from <the first's calibration.urdf> --out <calibration.urdf>` (`--head-axes` also fits the head axes'
   tilts). The same refits any sessions later with no robot.

`--resume` (same `--out-dir`) continues a session: its clicks and sightings are kept, a sweep it already has is not
redone. `--wrist-tip click` clicks the wrist camera's fingertip by eye instead of the sweep (~10 px worse);
`--openings` alone keeps the jaws closed. The script runs in a client session: Ctrl+C or a crash sends the robot to
rest.

**Retired** (2026-11-16; its code removed 2026-11-19): the click calibration (`collect`, `xle_fit`: the closed pinch
point clicked in 30-60 frames per arm, no board) and the wrist camera's separate pose step. Older calibrations still load (their
wrist cameras' poses and fingers' mounts read as corrections to the design).

### 8. Optional: measure the gripper's opening with calipers (needs the server)

**Optional since 2026-11-17**: the joint session (section 7) fits where the jaws close from the moving fingertip's
clicks, from the servo's map when there is none. This is still the one measurement of the jaws' width in
millimeters: a check of that fit, and the ground truth for grip work. Run it after a joint session, `--start-from`
its calibration.

Finds how far `gripper.pos` opens each arm's jaws, so plans can open the gripper to a width instead of guessing. The
gripper is a claw (the moving jaw turns about a hinge ~8 cm behind the tips), so each gap you measure is a chord of
the circle the tip turns on: `gap = 2 r sin(theta / 2)`, with the opening angle theta at the servo's own scale
(degrees per unit from the gripper's calibrated range, about 1.27). The script fits where the tips meet and the
radius r. It needs a calibrated URDF (section 7: the model's radius is the starting point and the check).

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
   `--start-from` made under an older calibration of the arm (a new joint session first). `--refit` refits the
   stored measurements with no robot. A joint session started from this file keeps the map as its start (and fits
   where the jaws close again).

### 9. Check the calibration on the robot (needs the server)

A short joint session that fits nothing (`joint_collect --check`): the same steps and the same safety as section 7
(the gripper sweep's checks, the views kept clear of the table, a move around through rest when the straight way is
not), shorter (1 placement, 6 arm views, 3 head views, 4 tip views by default), the clicks optional. At the end the
calibration is scored as it is (`joint_fit.check`), each part against its limit. The URDF must match the robot (the
same check as the harness's: a recalibrated arm, a recalibrated gripper or flipped head directions are refused).

1. Server on the Pi with the arm and the head: as for section 7.
2. On your computer, the table clear, the board at hand:
   `uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.joint_collect
   tcp://<pi>:5555 --arm right --out-dir <a scratch folder> --start-from <calibration.urdf> --check`
3. As section 7: the sweep, the board on its outline, `n` through the views. Click the fingertips in the head's panes
   if you like (scored if you do); `q` stops early and scores what there is.
4. Read `check.md` (printed at the end; `check.json` the numbers):
   - **the head and the arm on where the board is** (every moment both saw it; limit 6 mm, 1.5 deg): off, the head's
     chain or the arm's is: a joint session (section 7);
   - **the wrist camera's fingertips** (the sweep's, against the model's; limit 3 px): off, the wrist camera moved on
     its mount (`hardware_risks.md`) or the gripper map is off; the sweep's drift is printed too;
   - **the head's fingertip clicks** (if any; limit 8 px): off, the arm's joints, its gripper or the head are.

   The limits are from the recorded sessions (`joint_fit.CHECK_*`): good calibrations scored 2.5-5.0 mm, 1.3-2.3 px
   and 2.8-4.9 px; an arm never calibrated, 17 mm and 36 px.

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
    a new joint session (section 7). A re-recorded head home is not a mismatch: it only changes where the head
    rests.
  - `<arm> arm: <motor> recalibrated on the robot`: the arm was recalibrated with `lerobot-calibrate`, so its joint
    angles mean something else than when you clicked; a new joint session (section 7).
  - `<arm> gripper map: gripper recalibrated`: a new joint session (section 7: it starts from the servo's map and fits
    it), or the calipers (section 8).

  Calibrations made before the URDF kept a snapshot of the arms' LeRobot calibration cannot be checked for the arms
  (a warning is logged); recalibrating the camera adds one. Older servers (protocol 3) lack the `arm_calibration`
  command the check uses: update the Pi's checkout.

- `diagnostics.lift_test` drives only `shoulder_lift` and prints the servo's load, current and voltage per
  phase. `--p-lift N` and `--max-step DEG` override the P gain and the per-command cap for one run.
- `diagnostics.dump_servo_config --left <PORT> --right <PORT>` prints the limit and gain registers of every arm
  servo without changing anything.
- `teleop.spacemouse_teleop --log run.csv` records commanded and measured joints and tip position every tick.

## Branches and CI

- **`main`** (the fork's default, `archworm19/lerobot`) is ours: changes reach it by pull request, its CI passed
  (`.github/workflows/felix_ci.yml`: `tests`, the felix_jr fast tests; `guard`, no images, captures or `scratch/`
  files added, and ruff on the Python a change touches). No force pushes, no deletion; the owner may bypass.
- **`upstream-main`** mirrors Hugging Face's `main`. To take upstream's changes: `git fetch upstream`,
  `git push origin upstream/main:upstream-main`, then a pull request from `upstream-main` into `main`.
- The repository is public: GitHub-hosted Linux runners cost nothing. Upstream's own workflows are disabled here;
  a new one arriving with a sync shows in the pull request: disable it (`gh workflow disable`).
- `smoke` (every test, smoke and slow too) runs nightly and from the Actions tab (`workflow_dispatch`).

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
