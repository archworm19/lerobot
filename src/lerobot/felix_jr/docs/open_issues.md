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
3. **The first move is unchecked.** `execute` goes from wherever the arm is (home) to the first pose with one
   joint-space move; nothing checks that sweep for collisions. `--step` lets a person watch it.

## Collisions

4. **The held object counts as an obstacle.** `check_path` puts a 2 cm ball around the tool point and checks it
   against the depth, so the object being grasped, and then carried, shows as "blocked" on the approach, lift and
   carry legs. Fix: exclude the object's own points (those near the grasp target) from the check, and model the held
   object as part of the gripper while carrying.
5. **Only the tool point is checked, not the arm.** The wrist, forearm and elbow are never checked against the scene
   or the robot's own body. Needs collision geometry for the arm links (the merged URDF has no meshes) and the depth
   as obstacles (point cloud or voxels); pinocchio + coal can do the distance queries.
6. **No motion planner.** Moves through free space follow hand-written straight lines (densified into 1 cm IK
   steps). OMPL (`pip install ompl`, 2.0.1 has wheels for macOS arm64 and Linux aarch64; tried RRTConnect with a
   Python validity callback) is the candidate for the transit moves (home to the first pose, lift to above the
   place), once 5 exists. VAMP has no wheels and generates its collision code per robot; pybullet does not build.

## IK and pose choice

7. **With the jaws free and pointing down, the roll drifts.** `check_poses` solves one pose at a time, and pointing
   down the roll is nearly free, so it relaxes where it can and must swing back where it cannot (~50 deg in 1 cm on
   the cable plan). `execute` pins the roll to the first pose's when the jaws are free (`--roll-deg` to choose).
   Fix: solve the whole path at once (a trajectory-level objective with smoothness), not pose by pose.
8. **Poses sit on joint limits.** The steepest reachable pitches have `wrist_flex` at its cap (+90.2 deg, the URDF's
   95 x `LIMIT_MARGIN`) and often the roll at `MAX_ROLL_DEG` (130). `approach_poses` keeps a margin in pitch only
   (one grid step). Add a cost for closeness to the joint limits, so the chosen pose is comfortable, not just
   feasible.
9. **The limits themselves are unverified.** `LIMIT_MARGIN` scales the limits toward zero (a fraction, not a fixed
   distance from each end). `MAX_ROLL_DEG = 130` protects the wrist camera's cable but was not measured. The URDF's
   wrist flex is +-95 deg where the right arm's calibrated range is ~+-97.7; the rest pose has `shoulder_lift` at
   -103 deg, outside the URDF's +-100. Compare the URDF limits with each arm's calibration file.
10. **`approach_poses` is slow when it fails.** It sweeps the full range (-90..+30) to say which pitches would work:
    ~9 s for two targets. TODO in `reach._sweep`: solve the pitches in parallel (a process pool, or batched solves).
11. **Only the pitch and jaws are chosen, not the approach direction.** The heading follows from the shoulder pan,
    so a side grasp always comes from the shoulder's side of the object. Reaching around an obstacle, or taking an
    object from a chosen side, would need the base to move (not planned) or the other arm.

## Tooling

12. **The session CLI is a gitignored file.** `harness/scratch/` is ignored (`**/scratch/` in the root
    `.gitignore`). Execution now lives in tracked modules (`execute.py`, the `execute` tool in `api.py`), but the
    CLI that drives a session by hand (`session.py`) does not. Move it to a tracked module once its commands settle.
13. **Legacy `Reach` objects.** `objects.Reach` is kept only so sessions pickled before `Poses` still load. Delete it
    once those sessions (`cable1`, `chapstick1`, `desk1`) are no longer needed.
14. **Placo needs a patched URDF.** Placo mishandles the URDF's continuous wheel joints (its joint limits land on the
    wrong dofs, and the QP turns infeasible). The IK smoke test loads a copy with the wheels fixed. Worth reporting
    upstream if placo is used for more than cross-checks.
15. **Approval is enforced in Python only.** `api.Harness` asks a person before a tool moves the head or arm, but the
    model's Python snippets run in-process (`Harness.run_snippet`) and could import the robot client and send
    motion commands directly. A guardrail while a person watches; before any unattended run, the robot server must
    enforce it (motion only with a short-lived approval token the person's "yes" hands out), and the model should
    act through tool calls only (name + JSON arguments), with snippets, if kept, in a sandbox without the client.
