# Open issues

What is known to be missing or wrong in the harness's tools, reach, planning and execution (`harness/api.py`,
`harness/reach.py`, `harness/execute.py`, `harness/scratch/session.py`), with where it shows up and what would fix
it. Close an item by deleting it (git keeps
the history); add new ones at the end of their section.

Other areas keep their own lists: [camera_calibration.md](camera_calibration.md#not-done-yet-and-open-questions),
[head.md](head.md#not-done-yet), [realsense_m1.md](realsense_m1.md#open-items),
[tuning_notes.md](tuning_notes.md#4-open-questions), and the "Not done" line in [server_plan.md](server_plan.md).

Last updated 2026-11-01.

## Before the first real run

1. **The motion commands have never run on the hardware.** `move_joints`, `home` and `stop` are tested against
   `FakeHardware` only. First run: a free-air test in `client.session()` (one arm, ~10 cm above the table, gripper
   open/close, `stop` mid-move), before any grasp.
2. **Gripper values are guesses.** `execute --open 60` is not measured: `gripper.pos` (0-100 over the calibrated
   tick range) has no known mapping to jaw opening. Closing to 1 on an object stalls the servo against it for as
   long as it holds (torque and heat). Measure opening vs `gripper.pos`, and close to "object width minus a bit"
   rather than to 1.
3. **Not everything that moves is drawn.** `arm_path` draws the arm's skeleton along an execution, including the
   first joint-space move from where the arm is (`held_object` describes the carried object in text). Still missing:
   the return home at the end (the home pose is recorded on the robot; `get_homes` would give it) and the gripper
   open / closed at each pose. The fallback when a run fails (stop, then `home`) is also an unplanned joint-space
   move, and it can start right beside the objects: on pen1 a step backing away from the pen did not settle, and the
   arm went home from next to both cups. And grippers are drawn over everything: a gripper behind an object (from the camera)
   looks in front of it. TODO: hide or dim the parts the depth shows a surface in front of, as the grid does. A render is also flat: an arm segment drawn over an object may be well above it.
   A top or side view drawn from the root-frame geometry (no camera) would show heights.

## Seeing the grasp

13. **Whether a grasp holds is hard to see.** On pen1 (a thin pen, ~0.8 cm) the gripper closed to `gripper.pos` 2,
    which reads as "nothing between the jaws", and the close-up showed the pen at the very tips of the closed jaws,
    yet the person watching saw that it was held. The head camera sees the jaws from above and behind, at ~40 cm:
    a thin object between the jaw faces is mostly hidden by the jaws themselves, and depth on it is patchy. Ideas:
    the wrist camera (it looks along the jaws); the gripper's load or current after closing; a capture after a small
    lift (is the object still in the cup?). Also: the tool point is the closed fingertip, so aiming it at a thin
    object puts the object at the tips; aim it a finger-depth (~1.5-2 cm) into the jaws instead.

## Motion on the robot

14. **The gripper is a claw; the tool point assumes a parallel gripper.** The moving jaw rotates about a pivot ~8 cm
    behind the tips: the gap is a wedge, an object's grip center is not `width / 2` from the pinch point along the jaw
    axis, and closing pushes the object along the jaws too. Model the moving face from the gripper joint (it is in
    the URDF) and define a grasp as an object of width w at depth d into the jaws, centered in the gap there; measure
    the `gripper.pos` -> jaw angle mapping (see 2).
15. **No wrist camera in the loop.** The head camera cannot see whether a thin object is between the jaws
    (`sessions.md`, pen1); the wrist cameras look along the jaws. Needs their extrinsics (the gripper frame) and their
    frames in `Capture` (the person is setting them up).

## Collisions (decided: not checked by the harness)

The harness does not check paths or the arm against the scene: no depth path check, no link collision geometry, no
motion planner (`harness.md` 1.K). Avoiding obstacles is the model's job, from the renders, and the person approving
`execute` is the check. The removed path check (`check_path`) also judged every surface in the depth as an obstacle,
including the object being grasped. Revisit only if the model, given good renders, keeps planning into things.

## IK and pose choice

4. **A free roll can land anywhere.** `solve_pose` with `roll_deg=None` puts the roll anywhere along its free
   direction; between neighboring poses it can differ a lot (85 deg on the cable plan), and `prepare_execution`
   interpolates that as a swing of the wrist (its 20 deg / step check catches the worst). The model has to hold the
   roll (solve the first pose free, keep its roll); the tool docs say so. Solving a whole path at once, with
   smoothness, would make this automatic.
5. **Poses sit on joint limits.** The steepest reachable pitches have `wrist_flex` at its cap (+90.2 deg, the URDF's
   95 x `LIMIT_MARGIN`) and often the roll at `MAX_ROLL_DEG` (130). `solve_pose` reports the joints at a limit; the
   model can back off the pitch. A cost for closeness to the limits would make the solution comfortable by itself.
6. **The limits themselves are unverified.** `LIMIT_MARGIN` scales the limits toward zero (a fraction, not a fixed
   distance from each end). `MAX_ROLL_DEG = 130` protects the wrist camera's cable but was not measured. The URDF's
   wrist flex is +-95 deg where the right arm's calibrated range is ~+-97.7; the rest pose has `shoulder_lift` at
   -103 deg, outside the URDF's +-100. Compare the URDF limits with each arm's calibration file.
7. **A local solve depends on its start.** `solve_pose` runs once from `init`, without restarts: from the arm's rest
    pose it finds the experiments' poses, from the neutral pose (all joints 0) it misses every one. The model chooses
    the start (the previous pose, usually); offering a few named starts (rest, previous, elbow up / down) would help.
8. **Only the pitch and jaws are chosen, not the approach direction.** The heading follows from the shoulder pan,
    so a side grasp always comes from the shoulder's side of the object. Reaching around an obstacle, or taking an
    object from a chosen side, would need the base to move (not planned) or the other arm.

## Tooling

9. **The session CLI is a gitignored file.** `harness/scratch/` is ignored (`**/scratch/` in the root
    `.gitignore`). Execution now lives in tracked modules (`execute.py`, the `execute` tool in `api.py`), but the
    CLI that drives a session by hand (`session.py`) does not. Move it to a tracked module once its commands settle.
10. **Legacy code and data from the first sessions.** Kept only so the sessions recorded before the current design
    (`cable1`, `chapstick1`, `desk1`, gitignored under `harness/scratch/sessions/`) still load, and so the smoke test
    has data. New recordings will replace them; then remove, each marked `TODO` in the code:
    - `objects.Plan` and `scene.SegmentCheck` (the removed path check) and `objects.Reach` with its `__setstate__`
      (renamed fields), in `harness/objects.py` and `harness/scene.py`;
    - `Plan` and `Reach` in `OBJECT_TYPES` (`harness/workspace.py`) and in the package exports (`harness/__init__.py`);
    - the defaults for a missing call log in `api.Harness.restore` (sessions saved before the runner);
    - their tests in `tests/felix_jr/test_harness.py` (`test_reach_pickled_before_pitches_still_loads`, the legacy-plan
      check in `TestRender.test_errors`, the `Plan` line in `TestWorkspace.test_unknown_objects`);
    - the smoke test's data (`tests/felix_jr/smoke_tests/test_ik_smoke.py`: the waypoints, the rest pose and the
      pinned verdicts come from `cable1` / `chapstick1`): replace with the new recordings, re-pinned.
11. **Placo needs a patched URDF.** Placo mishandles the URDF's continuous wheel joints (its joint limits land on the
    wrong dofs, and the QP turns infeasible). The IK smoke test loads a copy with the wheels fixed. Worth reporting
    upstream if placo is used for more than cross-checks.
12. **Approval is enforced in Python only.** `api.Harness` asks a person before a tool moves the head or arm, but the
    model's Python snippets run in-process (`Harness.run_snippet`) and could import the robot client and send
    motion commands directly. A guardrail while a person watches; before any unattended run, the robot server must
    enforce it (motion only with a short-lived approval token the person's "yes" hands out), and the model should
    act through tool calls only (name + JSON arguments), with snippets, if kept, in a sandbox without the client.
