# Open issues

What is known to be missing or wrong in the harness's tools, reach, planning and execution (`harness/api.py`,
`harness/reach.py`, `harness/execute.py`, `harness/scratch/session.py`), with where it shows up and what would fix
it. Close an item by deleting it (git keeps the history); add new ones at the end of their section with the next
free number. Item numbers are stable ids, cited from code and other docs (`open_issues.md`, 10), so never renumber.

Other areas keep their own lists: [camera_calibration.md](camera_calibration.md#not-done-yet-and-open-questions),
[head.md](head.md#not-done-yet), [realsense_m1.md](realsense_m1.md#open-items),
[tuning_notes.md](tuning_notes.md#4-open-questions), and the "Not done" line in [server_plan.md](server_plan.md).

**Where things stand.** The harness runs on the real robot: `cable1` and `chapstick1` (2026-10-31) scanned and
planned on it, and `pen1` (2026-11-01) moved the arm, every move approved; session folders under
`harness/scratch/sessions/`, write-ups in [sessions.md](sessions.md). On pen1 the gripper landed ~1 cm from plan, in line with the XLeRobot's inexpensive hardware: close enough everywhere except
across the jaws, which is the first item.

Last updated 2026-11-02 (10 closed: the legacy objects are gone; the IK smoke test is built on pen1).

## Fix the gripper model (the biggest issue)

14. **The gripper is a claw; the tool point assumes a parallel gripper.** One jaw is fixed (its tip is the pinch
    point); the moving jaw rotates about a pivot ~8 cm behind the tips, so its tip moves along an arc ((-1.3, 0, +0.4)
    cm at 10 deg, (-2.7, 0, +0.6) at 20, (-4.2, 0, +0.6) at 30, in the pinch frame). The gap is a wedge, widest at the
    tips; an object's grip center is not `width / 2` from the pinch point along the jaw axis, as the tool point
    assumes; and closing pushes the object along the jaws as well as across.

    This is what made pen1's first grasp miss. The plan put the pen within a couple of centimeters of the pinch point,
    which a parallel gripper opening around its center would have forgiven. Here only the moving jaw opens and the
    fixed jaw stays at the pinch point, so the direction of the ~2 cm error decided it: toward the moving jaw, the pen
    is inside the grip; toward the fixed jaw, it is just outside (pen1 measured it +0.4..+1.4 cm outside the fixed jaw,
    where the plan put it 0.5 cm inside). The tool point is also the closed fingertip, so aiming it at a thin object
    puts the object at the very tips; it should sit a finger-depth (~1.5-2 cm) into the jaws.

    **a. Add a grasp point to `XleRobotModel`, next to the pinch point.** The grasp point is where an object of width
    w is held: centered in the gap at depth d into the jaws. With the moving jaw opened by theta, the gap's
    bisector is at theta / 2 from the fixed jaw, so roughly `r * (cos(theta / 2), sin(theta / 2))` in the jaw plane,
    offset from the **gripper hinge** (the `<arm>_gripper` joint's origin, the pivot), not from the pinch point. The
    moving tip is the pinch point turned about the hinge axis by the opening: `jaw_radius` gives the circle (~8.1 cm
    on the right arm) and the map from b the opening. Do not use the URDF's joint angle for it: in the SO-101 URDF
    the jaws meet at about -11.4 deg (from the moving jaw's mesh), past the joint's -10 deg limit, so angle 0 is
    already ~16 mm open. `joint_vector` still leaves the gripper joint at 0.

    **b. Measure the gripper's range of motion: `gripper.pos` -> jaw opening.** Built: `calibration.gripper_step`
    (`README.md` section 7) steps the gripper, takes caliper gaps, and writes a `GripperMap` (where the tips meet,
    degrees per unit, the calibrated range it holds for, the raw measurements) into the calibrated URDF's
    `<felix_calibration>` element; `XleRobotModel.gripper_maps` reads it. The harness refuses a map whose range no
    longer matches the robot's (`XleRobotModel.check_robot`, when it connects). Not yet done: running it on the robot
    (both arms), and committing the resulting URDF.

    **c. Decide how the pinch point helpers coexist with the grasp point.** `XleRobotModel`'s pinch point methods
    (`pinch_point`, the projection and its derivatives, `pinch_frame`, `pinch_joint`, visibility) are what the
    calibration needs: the closed pinch point is what a person can see and click. Options: parameterize them (pinch
    or grasp point), or keep them for calibration and add separate grasp point methods. Suggestion: keep the pinch
    methods as they are and add a general "point in a frame" accessor that both build on, so the calibration code
    does not change.

    **d. Move the planning and rendering from the pinch point to the grasp point.** Everything that places the
    object at `width / 2` along the pinch frame's -x:
    - `harness/reach.py`: `Reach.offset` (the tool point in the pinch frame), and the module docstring's definition;
    - `harness/execute.py`: `held_object`'s `center_local`;
    - `harness/render.py`: the gripper drawing (the moving jaw drawn parallel to the fixed one, `width` apart);
    - `harness/objects.py` (`Poses.width`, `position_error_m`) and the tool docs in `harness/api.py` (`solve_pose`,
      `approach`, the render legend) that describe the tool point.

    Also from item 2 (merged here): close to "object width minus a bit" rather than to 1. Closing to 1 on an object
    stalls the servo against it for as long as it holds (torque and heat); this needs the mapping from b.

## Set up the wrist cameras

15. **No wrist camera in the loop, so whether a grasp holds is hard to see.** On pen1 (a thin pen, ~0.8 cm) the
    gripper closed to `gripper.pos` 2, which reads as "nothing between the jaws", and the close-up showed the pen at
    the very tips of the closed jaws, yet the person watching saw that it was held. The head camera sees the jaws
    from above and behind, at ~40 cm: a thin object between the jaw faces is mostly hidden by the jaws themselves, and
    depth on it is patchy. The wrist cameras look along the jaws (the person is setting them up). TODO:

    **a. Intrinsics calibration procedure for the wrist cameras.**

    **b. Extrinsics calibration procedure** (the camera in the gripper frame). Open question: make it an optional
    part of the current click calibration (`calibration/collect.py`), or a separate step?

    **c. Carry it in the URDF:** the wrist cameras' frames (and intrinsics, or a pointer to them) in
    `calibration.urdf`, so the model and `Capture` can use them like the head camera.

    Then their frames in `Capture`. Signals besides the camera, worth trying: the gripper's load or current after
    closing; a capture after a small lift (is the object still in the cup?).

## Rendering

3. **Not everything that moves is drawn.** `arm_path` draws the arm's skeleton along an execution, including the
   first joint-space move from where the arm is (`held_object` describes the carried object in text). Still missing:
   the return home at the end (the home pose is recorded on the robot; `get_homes` would give it) and the gripper
   open / closed at each pose. The fallback when a run fails (stop, then `home`) is also an unplanned joint-space
   move, and it can start right beside the objects: on pen1 a step backing away from the pen did not settle, and the
   arm went home from next to both cups. And grippers are drawn over everything: a gripper behind an object (from
   the camera) looks in front of it. TODO: hide or dim the parts the depth shows a surface in front of, as the grid
   does. A render is also flat: an arm segment drawn over an object may be well above it. A top or side view drawn
   from the root-frame geometry (no camera) would show heights.

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
11. **Placo needs a patched URDF.** Placo mishandles the URDF's continuous wheel joints (its joint limits land on the
    wrong dofs, and the QP turns infeasible). The IK smoke test loads a copy with the wheels fixed. Worth reporting
    upstream if placo is used for more than cross-checks.
12. **Approval is enforced in Python only.** `api.Harness` asks a person before a tool moves the head or arm, but the
    model's Python snippets run in-process (`Harness.run_snippet`) and could import the robot client and send
    motion commands directly. A guardrail while a person watches; before any unattended run, the robot server must
    enforce it (motion only with a short-lived approval token the person's "yes" hands out), and the model should
    act through tool calls only (name + JSON arguments), with snippets, if kept, in a sandbox without the client.
