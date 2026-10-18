# felix_jr tuning notes

What we measured while bringing up the two SO-101 followers (`left_arm`, `right_arm`), the wheel base and
SpaceMouse teleop, what we concluded, and what is still unverified. Numbers come from our own runs on this
hardware (Feetech STS3215 servos, 12 V wall supply). Most of the raw CSVs are no longer around (only
`rotate_y2.csv` and `rotate_y3.csv` sit untracked in the repo root), so the values quoted here are what we
have. Re-run the commands in each section to regenerate them.

**Status key:** *Measured* = seen in logged data. *Inferred* = fits the data, not isolated by a direct test.
*Open* = not resolved.

## Contents

1. [Scripts and where state lives](#1-scripts-and-where-state-lives)
2. [Current defaults](#2-current-defaults)
3. [Experiments and findings](#3-experiments-and-findings)
4. [Open questions](#4-open-questions)

## 1. Scripts and where state lives

Run all of these with `uv run python -m lerobot.felix_jr.<name>`.

| Script | Purpose |
|---|---|
| `record_home` | Save the arm's current pose as its "home" pose (`<id>_home.json`). |
| `wiggle_test` | Go to center, nudge each joint, return home. `--left` / `--right`, several arms at once. |
| `lift_test` | Drive only `shoulder_lift` down / up (against gravity) / back and sample the servo's load, current, voltage. |
| `dump_servo_config` | Read-only dump of limit and gain registers for every arm servo. |
| `setup_base` | One-time wheel setup: servo IDs and per-wheel forward direction. |
| `wheel_test` | Spin each wheel, then both, then spin in place. |
| `spacemouse_teleop` | Drive an arm's tip pose with a 3Dconnexion SpaceMouse, with optional per-tick logging. |

Shared code is in `home.py` (follower construction, home pose, the trapezoid-profile `move_all`) and
`base_follower.py` (`SO101FollowerWithBase`: per-joint P gains and the optional wheel servos).

Files written next to each arm's calibration (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`):
`<id>.json` (calibration, from `lerobot-calibrate`), `<id>_home.json`, and for the arm with the base
`<id>_base.json` (wheel direction signs).

Environment extras that are not in `pyproject.toml` / `uv.lock` (installed with `uv pip install`, so a later
`uv sync` removes them): `pyspacemouse`, `placo` (+ `cmeel-urdfdom<5`, `cmeel-tinyxml2<11`). The IK uses the
SO-101 URDF from `TheRobotStudio/SO-ARM100` (`Simulation/SO101/so101_new_calib.urdf`, with its `assets/`),
cloned to `~/.cache/felix_jr/SO-ARM100/`.

## 2. Current defaults

| Setting | Value | Where | Why (section) |
|---|---|---|---|
| `shoulder_lift` P gain | 32 (lerobot default is 16) | `DEFAULT_JOINT_P_GAINS` in `base_follower.py` | 3.5 |
| `elbow_flex` P gain | 32 | same | 3.7 |
| Other joints' P gain | 16 | lerobot default | 3.5 |
| Per-command cap (`max_relative_target`) | 5 deg in the scripts; teleop `--max-step` (we tried 15) | `make_follower`, `--max-step` | 3.6 |
| Teleop joint speed limit | 60 deg/s, `--max-joint-speed` | `spacemouse_teleop.py` | 3.7 |
| Trajectory limits for `move_all` | 60 deg/s, 120 deg/s^2 | `DEFAULT_MAX_VEL`, `DEFAULT_MAX_ACCEL` | 3.2 |
| Wheel servo IDs | left 9, right 10 (7-8 reserved for head servos) | `BASE_MOTOR_IDS` | 3.3 |

Servo registers were identical on all six joints of both arms when we dumped them (`dump_servo_config`):
P 16, I 0, D 32, `Max_Torque_Limit` / `Torque_Limit` 1000, `Protection_Current` 310 (about 2 A),
`Protective_Torque` 20, `Overload_Torque` 80, `Acceleration` 254. The gripper differs by design (torque 500,
`Protection_Current` 250, `Overload_Torque` 25). So the arm-to-arm and joint-to-joint differences below are
not settings mismatches.

## 3. Experiments and findings

### 3.1 Pause between wiggle moves was a tolerance wait *(measured)*

Symptom: after the fast joints moved, the non-wrist joints sat still for a long time before the next move.

Cause: `move_all` waited until every joint was within `tol` (1.5 deg) of target or timed out (10 s). With P-only
position control (I = 0), `elbow_flex` settles about 3 deg short of a target under gravity, so the wait never
ended. The timeout message showed `elbow_flex` 3.1 to 3.5 deg off with torque on, no fault flags, 30 C.

Fix: a larger `--tol` (4 to 5 deg works), and later a dwell-based completion check. The steady gravity
offset is a property of the controller, not a fault.

### 3.2 Jump when one move followed another *(measured, then fixed)*

Symptom: a visible lurch when a new trajectory started.

Cause: trajectories started from the *measured* pose. At rest the arm sits a few degrees off its goal under
gravity, so the first waypoint snapped the servo's goal onto the sagging position.

Fix: `move_all` starts from the last *commanded* goal (`_last_goal` in `home.py`), falling back to the
measured pose when an arm is more than 10 deg from it (first move, or moved by hand). Teleop starts from the
commanded center for the same reason. Trapezoid profiles (shared duration, per-joint peak speed scaled to
match) replaced the old "resend the target every tick" loop; see the derivation in `move_time_trap` /
`schedule_trapezoid`.

### 3.3 Wheel base: IDs and directions *(measured)*

- Wheels leave the factory as ID 1, which collides with the arm's `shoulder_pan` and with each other, so the
  first wheel must be set up alone on the board. Wiring: board -> right wheel -> left wheel, arm on the same
  board. `setup_base` sets the right wheel (ID 10) alone, then finds the left wheel by scanning the chain for
  the one servo that is not the right wheel (ID 9).
- The stock `setup_motor` finds its target by broadcast ping, which is unreliable (and can pick the wrong
  servo) when two servos answer, hence the per-ID scan.
- Directions: both wheels were first recorded as "forward", which cannot be right for wheels mounted on
  opposite sides. The left wheel's sign was flipped by hand in `right_arm_base.json` afterwards.
- The wheel IDs live on the right arm's bus, so `right_arm` is the one created with `with_base=True`.
  `make_follower` auto-detects this from the existence of `<id>_base.json`.

### 3.4 USB drop during a two-arm wiggle *(open)*

One run crashed with `OSError: [Errno 6] Device not configured`: the serial device disappeared mid-move while
both arms were moving. We never established the cause (cable, hub or power dip are the candidates). The
scripts now stop and disconnect each arm independently so one dead port does not leave the other holding
torque. Not seen again, but not explained.

### 3.5 Dip in tip height on a -x push: `shoulder_lift` under-tracking *(measured; mechanism inferred)*

Logged with `spacemouse_teleop --log`. Commanded tip height stayed within about 9 mm, but the measured tip
dropped up to 110 mm and recovered 1 to 2 s later. Per joint, `elbow_flex` and `wrist_flex` stayed within about
3 deg of their commands; `shoulder_lift` trailed by 10 to 20 deg whenever its command decreased (raising the
arm against gravity) and crept back slowly after the input stopped.

`lift_test` isolated it (shoulder only, down 18 deg, up 14 deg, back to 0):

| Run | Peak lag in the "up" phase | Final error after hold | Peak load | Peak current |
|---|---|---|---|---|
| left arm, P 16, cap 5 | 28.1 deg | +9.8 deg | 26 % | 221 mA |
| left arm, P 32, cap 5 | 9.0 deg | +1.1 deg | 35.6 % | 260 mA |
| right arm, P 16, cap 20 | 10.9 deg | +3.3 deg | 42.4 % | 364 mA |

What was ruled out *(measured)*:

- **Power.** Servo-reported voltage stayed 12.2 to 12.4 V through every phase; temperature peaked at 34 C.
- **Overload protection.** `Torque_Limit` stayed at 1000 and `Status` at 0 throughout, so the firmware never
  cut torque.
- **Configuration mismatch.** Registers were identical across joints and arms.

With P 16 and cap 5, the load sat on a flat plateau of 22 to 24 % while the position error was anywhere from
12 to 28 deg, and the joint climbed at 5 to 15 deg/s with the speed rising as the arm came upright. A
position loop that is simply too soft would push harder as the error grows; this one did not.

### 3.6 The per-command cap limits the torque *(inferred)*

`max_relative_target` does not only limit the step size. In `SOFollower.send_action` the goal sent to the
servo is clamped to within that many degrees of the arm's *present* position, so the servo's position error
can never exceed the cap, however far behind the arm is. The servo's torque is then roughly P times the cap,
which fits the data:

- at P 16 the load plateaued near 23 % while the goal was pinned 5 deg ahead (about 4.6 % per degree), and the
  P 32 hold showed 11.2 % of load for a 1.1 deg error (about 10 % per degree), roughly double per degree;
- raising the cap from 5 to 20 at P 16 lifted the peak load from 26 % to 42 % and cut the lag from 28 to
  about 11 deg, similar to what doubling P did.

The two knobs do different jobs: the cap sets how hard the servo can push while catching up (transient lag);
P sets how closely it holds once there (steady error: about 3.3 deg at P 16, about 1.1 deg at P 32 under the
same gravity load).

Caveats: the P 16 / cap 20 run was on the right arm, the others on the left, and we have no right-arm
baseline at P 16 / cap 5. We did not run the combination cap 20 with P 32 as a controlled test. Safety
tradeoff: the servo applies about P times the cap, so a larger cap or gain means more force into an obstacle
(P 32 with cap 15 is roughly six times the limit of P 16 with cap 5). Keep a hand near the arm.

### 3.7 Pitch pushes: `elbow_flex` lag, then joint-speed blow-up *(measured / reproduced in simulation)*

Pitching the tool about a fixed tip (SpaceMouse pitch) with the shoulder fixed at P 32:

- `elbow_flex` (P 16, cap 5) trailed its command by up to 35 deg, and the measured tip height fell up to
  166 mm below the commanded height before recovering. It is the same gravity-loaded failure as the shoulder,
  so `elbow_flex` was given P 32 as well.
- With the elbow at P 32 and cap 15 (right arm), tracking was good: measured minus commanded stayed within
  about 9 deg for the whole run, tip-height error max 19 mm with a constant mean of about -8 mm (gravity sag
  at rest, about -12 mm at the start).
- The operator still reported "unresponsive, then violent". A light pitch deflection of 0.15 is only about
  5 deg/s of tool rotation, which feels dead. But a pure pitch about a fixed tip needs the lift, elbow and
  wrist to move in a coordinated way whose size grows as the arm folds up. In the log, with a deflection of
  about 0.25, the commanded elbow went from -58 to -96 deg in 0.6 s (roughly 60 to 130 deg/s) and ran into its
  limit near -96 deg.
- We reproduced this in simulation (no hardware): holding a pitch of 0.25 for 8 s from the center pose gave
  peak commanded joint speeds of 261 deg/s (lift), **489 deg/s (elbow)** and 237 deg/s (wrist). Nothing in the
  teleop limited joint speed; only tip speed was limited.
- Fix: `--max-joint-speed` (default 60 deg/s). If any joint would exceed it in a tick, the whole step is
  scaled down so the joints stay coordinated, and the position target is pulled back to where the arm is so
  the part that could not be followed does not accumulate. In the same simulated push, peak speeds became
  32 / 60 / 38 deg/s with a similar final pose.
- Hardware check with the limit on (a 45 s run covering pitch up to 0.94, roll, yaw and full-deflection x/z
  pushes, elbow command from -72 to +52 deg): peak commanded joint speeds were 36 / 69 / 69 deg/s (the 60
  deg/s limit is per nominal 33 ms tick, so ticks that run late show a little above it), peak measured speeds
  36 / 73 / 61 deg/s, and measured minus commanded stayed within 4 / 6 / 7 deg. Loop period averaged 33.3 ms
  (max 42.8 ms).

Whether the motion now *feels* right to the operator has not been written down here; add it when known.

### 3.8 SpaceMouse setup on macOS *(measured)*

- `pyspacemouse` needs the native hidapi library (`brew install hidapi`). Its HID layer looks the library up
  by name, which misses Homebrew's `/opt/homebrew/lib`, then silently falls back to the wrong place and
  reports "HID API is probably not installed". `spacemouse_teleop.preload_hidapi()` loads the library first.
- Even with the library loaded, the device would not open ("Failed to open device") while 3Dconnexion's own
  driver was running (its DriverKit extension and helper apps). Quitting/disabling it fixed it. Terminal
  Input Monitoring permission can give the same symptom.

## 4. Open questions

- **Joint-speed limit feel.** Logged numbers look good (section 3.7); the subjective result is not recorded.
- **P 32 and the caps are a compromise.** We did not sweep P between 16 and 32 for chatter, did not control
  for arm-to-arm differences, and did not test cap 20 with P 32. If the arm hums at rest or hits something,
  lower P or `--max-step` first.
- **Wrist joints** carry little load and tracked fine at P 16; they were not tuned.
- **USB drop** (section 3.4) is unexplained.
- **IK limits.** The 5-joint arm cannot follow all six puck axes: x/y/z move the tip, pitch tilts the tool in
  the arm's plane, roll twists the tool, yaw swings the whole arm about the shoulder pan. Large pitch near a
  folded pose runs into the elbow limit (about -96 deg); the 20 deg-per-tick IK-jump rejection and the 2 cm
  unreachable-target rule are the only other guards.
- **The Python packages** for teleop are not tracked in `pyproject.toml`; adding an extra means regenerating
  `uv.lock`.
