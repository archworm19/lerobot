# The simulator in the harness (plan, 2026-11-19)

A harness protocol that plans and checks its grasps in the simulator (`sim2real/`), and, first, a **shadow**: every
real execution replayed in the simulator and logged beside it. Branch `sim_harness`.

## The protocol we are building toward

**Planning from the head:**

1. **The candidate:** the greedy draw (`harness.grasping.first_feasible`: the claw targets shuffled, the first through
   every filter taken; the ranking's heuristics not used), its simulated grasp the last check
   (`sim2real.grasp_check.SimAccept`). A joint's margin from its limit, which the ranking weighed, belongs in the
   filters if we keep it.
2. **The commands, through the servos:** the simulator's kinematics are the IK's own model, so what it adds is the
   servos (`robot.ARM_KP_NM_PER_RAD`, `ARM_FRICTIONLOSS_NM`: they stop short of their command, on the side they came
   from, `sim2real_grasp.md`). The attack's commanded joints are found so that the simulated servos settle at the
   target: command, simulate, correct by the miss, 2-3 rollouts. The move into the attack from a fixed side (the
   friction's band depends on it). Replaces `GraspSettings.sag_lift` and `align_step`'s pushing again. Its reach: the
   sessions' execution error is ~0.6 cm of a 1-2 cm miss; the rest is perception.
3. **Move to the attack** with those commands.

**The grasp, from the wrist camera:**

1. Capture with the head and the wrist camera.
2. **The cylinder's pose by render and compare** (if needed): the wrist camera is rigid on the gripper, so what
   matters is the object relative to it: for an upright cylinder of known radius 2-3 numbers, searched by the
   rendered outline against the object's outline in the picture (outlines, not pixels: the renders are not
   photographs). Built and scored offline first, on the sessions' and `grip_test`'s recorded wrist frames.
3. **The grasp optimized in the simulator:** a few parameters (the grasp point's offset along and across the jaws,
   its height, the depth) by the cross-entropy method: sample ~30 around the plan, simulate each from the attack
   (approach, close, lift), keep the best ~20 %, refit the sampling's mean and spread to them, 3 rounds. The score
   from the rollout: held, then smooth terms (the object centered between the pads, depth margin, how far it rose).
   The final spread says how forgiving the grasp is (the bank's depth-0 grasps fail with a few mm of miss).
4. Execute.

The same planner is the "teacher" the grasp environment's plan needs (`sim2real_grasp.md`): an expert with the true
state whose rollouts a policy learns from.

## The shadow simulator (first)

**What:** every real execution replayed in the simulator from the same start with the same commands, and the two
logged side by side. It changes nothing the robot does. For: where the simulator is wrong (the servos, the grip,
contacts, the cameras), a dataset that refits the servo and grip models from every session (not only the special
trials), and the yardstick for the steps above.

**Recorded always, replayed on demand** (decided 2026-11-19): execution only records; what to do with the record is
the VLM's or a protocol's. Every `execute` adds an `ExecutionLog` to the workspace beside its capture (its parent the
execution; also when a move stopped): the start reading, and per move its targets, `tol`, `timeout`, the server's
status and the arm read after it (`MoveReading`: one `client.get_state()` a move, in `run_execution`). Nothing is
simulated then: the robot's timing untouched, and a log from today can be replayed after next month's refit. The
shadow is three software tools (`api.py`; the simulator imported only when one is called): `replay(log)` (a
`ShadowRecord`, recorded), `compare(record)` (move by move, real against simulated), `shadow_report()` (the session,
per kind of pose; each log replayed once). The recorded sessions, from before the logs, become logs too
(`shadow.session_logs`: their held executions, read at the hold only). Not logged yet: the moves that go to the server
without `execute` (`align_step`, the wrist moves, `home`).

**The replay** (`sim2real/shadow.py`):

1. **Scene:** the arm from the calibrated model (`scene.build`: the fitted servos, the sprung fingers' pads), the table
   at the model's floor, the objects from the plan's primitives (an upright `Cylinder` maps onto `scene.Cylinder`).
2. **Start:** the real joints and `gripper.pos` of the capture the plan was solved from (`robot.set_pose`).
3. **Each move as the server runs it, with the server's own code** (`trajectory.py`, used by `home.move_all` on the
   robot and by the replay: where a move starts, `move_starts`, its last commanded goal unless the arm is 10 deg from
   it; its plan, `MovePlan`, every joint on its trapezoid, starting and stopping together; a waypoint each
   1/`WAYPOINT_HZ` s; arrived when every target, the jaw's too, is within `tol` for `DWELL_S`, `Settler`), at the
   same `max_vel` / `max_accel`. A change to how the server moves changes the replay with it.
4. **The gripper:** its commands to the sim gripper, stalling on the object as the real jaw does (`grip_sim`).
5. **At each capture:** the sim's wrist and head cameras rendered through the calibrated lenses (`view.render`; the
   real wrist frame through `view.undistort`).

**Records** (per execution, in the session's scratch folder: `shadow/<execution>.json`, never in git; a `shadow`
entry in the harness log): per move, the joints at its end (read; settled in the sim), the pinch point (forward
kinematics of each), `gripper.pos` (where each jaw stalled), reached or not; the object's sim pose after each move
(pushed? knocked over?); the grasp's outcome, sim against real (the wrist picture, or the person, as `grip_test` asks).
Pictures: the real captures beside the sim's renders, the sim's object outline drawn on the real picture (the start
of the render-and-compare estimate).

**Used for:** `shadow_report()` (per session: the joints' differences per joint and kind of pose, the pinch point's
miss in mm, the outcomes agreed or not); the records feed `shortfalls`' fit (every move, both directions, loaded
and not) and extend `grip_sim`'s two bottles. Each record keeps the simulator's settings (servos, friction, the
finger model), so it stays readable after a refit.

**Reused:** `sim2real.scene`, `robot` (servos, pads, `set_pose`), `home`'s trapezoids, `shortfalls.replay_hold`
(generalized to a sequence of moves), `grip_sim` / `grasp_check` (closing, held), `view.render` / `undistort`. New: the
per-move read, the multi-move replay, a primitive-to-sim-object converter, the overlays, the report.

**Phases:**

1. **Kinematics and servos**, no objects: the replay of every move, the joints' and the pinch point's comparison.
   First on the recorded sessions (bowl1-8, pen1-2, ruler, container: their executions and held captures are in
   `workspace.pkl`), then live.
2. **Objects and the grip:** the converter, contacts, the jaw's stall, the outcome against the real one.
3. **Pictures:** the renders and overlays at the captures.

**Phase 1, the recorded sessions** (2026-11-19, `python -m lerobot.felix_jr.sim2real.shadow`; `trajectory.py` holds the
server's trapezoids, out of `home`, so the replay needs no servo SDK): the 50 held executions of 13 sessions, each
replayed whole from the capture its plan was solved from (12 s for all), against the joints read at the hold:

| pose (holds) | real, mean (deg: pan, lift, elbow, wrist flex, roll) | simulated | RMS apart (deg) | pinch points apart |
| --- | --- | --- | --- | --- |
| attack (28) | +0.09 +0.24 +0.65 +0.87 +0.21 | +0.13 +0.25 +0.66 +0.95 +0.20 | 0.13-0.26 | 1.4 mm mean, 8.4 max |
| grasp (10) | -0.03 +0.27 +1.19 +0.13 +0.16 | +0.01 +0.41 +1.22 +0.35 +0.18 | 0.08-0.29 | 1.8 mm mean |
| loaded (6) | -0.19 +1.60 +0.94 -0.06 -0.04 | -0.03 +1.49 +0.86 +0.06 -0.02 | 0.06-0.41 | 2.3 mm mean |

Nearer than the servos' fit's own replays (0.35-0.43 deg RMS at the attack, from the move before only, a hold that
moved little saying nothing of its side): the whole execution says it. The loaded moves (the jaws closed on an object
on the robot, empty here) come near too: the side they arrived from is most of their sag. In-sample: the servos were
fitted to these holds (`shortfalls`); the live sessions are the test.

**Open:** which object pose seeds the sim (the plan's, what the robot acted on, first); scenes of several objects (the
target alone first); the records' versioning (the settings saved with each, as the bank's).

## The grasp refined in the simulator (built 2026-11-19)

`sim2real/refine.py`: one function for both phases (`phase="head"` / `"wrist"`). **The parameters** are the object's,
from its primitive (`Primitive.grasp_space`, `grasp_at`, new on the protocol: a cylinder's `t`, `phi`, `psi`;
`antipodal_grasps` now `grasp_at` on its grid, the 6033 grasps checked identical), and the gripper's: `depth` and
`offset` (across the jaws). The stage, the attack and the lift follow from the grasp (`plan_pick`): a sample counts
only if its whole chain works. From the wrist, `phi` and `psi` are kept (the arm committed to its approach).

**Each sample** runs the harness's own chain (the filters and IK, `plan_pick`, `prepare_execution`: what `execute`
sends), and the shadow plays it with the object in the scene (`scene.from_primitive`) under 8 draws, the same for
every sample: the object moved by the perception's error (7 mm; the first draw the estimate), the physics as the
bank's. Lifted: up 3/4 of the planned lift, held 2 s. One rollout (a whole pick, 29 s simulated) takes 0.65 s, ~45
times real time.

**The search is greedy** (decided over the cross-entropy method: satisfice with a robustness threshold, as
`first_feasible` does): batches of 6 around the greedy candidate, in 6 processes; the best of a batch lifted in 7 of 8
draws is taken; else the next batch around the best so far, 1.5 times wider; at most 5. (The cross-entropy method,
tried first, converged but its rounds cost ~2 minutes; it climbed to a ceiling its 3 draws could not see past.)

**Two changes it needed:**

- **The wrist camera's clearance at the attack only** (`GraspSettings.camera_clearance_at_grasp`: the harness keeps
  both): the 8 cm is for the camera to see, which matters at the attack (the wrist phase's input); at the grasp only
  collision does, which the simulator judges. With it at both, every deeper grasp was refused.
- **The camera's real board collides in the simulator** (`robot._camera_board`: 45 x 45 mm about the lens, 12 mm
  deep, at the calibrated pose; the SO-101 model's 32 mm module's mesh no longer collides). Clear of the gripper's
  own parts at every opening.

**First result** (seed 0's placement 4: a 45 mm x 158 mm, 54 g bottle; the arm from rest): the bank's grasp, run as
the robot would run it (through the stage and the attack, the servos sagging: the pinch point 8-9 mm low, as the
sessions' 0.6-0.9 cm), failed with the bottle at its estimate and held only with it 5-15 mm nearer: its working region
beside the estimate. The refine took a grasp lifting in 8 of 8 draws in 3 batches, 28 s: 14 mm deep (the camera's
clearance had refused it), 41 deg down (was 30), 2 cm higher. Over the bottle moved -15..+15 mm along and across the
approach: held at 38 of 49 offsets (the start: 15), the estimate inside with 1-1.5 cm to spare on most sides.

## Then

The servo-compensated commands (protocol step 2), judged by the shadow's records (does the arm land nearer?); the
greedy and simulated candidates in the harness; the wrist's render-and-compare, offline first; the grasp optimized by
the cross-entropy method.
