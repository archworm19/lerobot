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

Last updated 2026-11-05 (16 added: the arm's sag, from pen2). 2026-11-02: 10 closed (the legacy objects are gone;
the IK smoke test is built on pen1).

## Fix the gripper model (the biggest issue)

14. **The gripper is a claw.** One jaw is fixed (its tip is the pinch point); the moving jaw turns about a hinge
    ~8 cm behind the tips, so its tip moves along an arc ((-1.3, 0, +0.4) cm at 10 deg, (-2.7, 0, +0.6) at 20,
    (-4.2, 0, +0.6) at 30, in the pinch frame), the gap is a wedge, widest at the tips, and closing pushes the object
    along the jaws as well as across. This is what made pen1's first grasp miss: only the moving jaw opens, so a
    ~2 cm error toward the fixed jaw put the pen just outside the grip (pen1 measured it +0.4..+1.4 cm outside the
    fixed jaw, where the plan put it 0.5 cm inside).

    **Built (2026-11-04).** The gripper map (`calibration.gripper_step`, `README.md` section 7; both arms in
    `urdfs/xle_calib_11042026.urdf`), the moving tip (`XleRobotModel.moving_tip`, `point="moving_tip"` on every
    predicting method; not the URDF's joint angle, which is ~16 mm open at 0), checked on the robot
    (`calibration.check`, `README.md` section 9), and the harness on top of it: `width` is the object's, the jaws open until the
    tips are `width` apart, and the **grasp point** (where `solve_pose` puts the target and `held_object` the
    object's center) is midway between the tips (`XleRobotModel.grasp_offset`); `prepare_execution` opens and closes
    the gripper by the object's width through the map; `render` draws the moving finger turned about the hinge.

    Still open:
    - **TODO: how far to open and how hard to close.** `reach.OPEN_CLEARANCE_M` (the tips 2 cm wider apart than the
      object for the approach) and `reach.SQUEEZE_M` (close to 3 mm narrower, to hold without stalling the servo for
      long) are first guesses. The clearance trades the margin for the ~1-2 cm aim error (on the moving jaw's side
      only: the fixed jaw does not move) against hitting neighbors. **The squeeze is too weak and in the wrong
      unit** (pen2, 2026-11-05: a pen slipped out of the jaws at once; `tuning_notes.md` 3.9): the grip force is P
      times how far the close target is past contact, and 3 mm of width was ~1 deg, a few percent of load. Make it
      an overshoot in servo units / degrees past the object (or close to the tips meeting), with the servo's torque
      cap setting the force; weigh it against overload protection (torque drops to 25 % after a long hard hold).
      Worth measuring: the load after closing at a few overshoots, and how long until protection trips.
    - **Grasping at the tips.** The grasp point is between the fingertips, so a thin object is held at the very tips,
      where pen1's pen sat (and pen2's, which slipped). Holding it a finger-depth (~1.5-2 cm) into the jaws would be a `depth` on the grasp: the
      points of both jaws that far back from the tips, and the opening that leaves `width` between them there.
    - **The fixed jaw's margin.** Only the moving jaw opens, so with no margin the fixed fingertip touches the
      object's side and an aim error toward it misses (pen1's first attempt). `solve_pose(..., margin=)` (default 0)
      aims the grasp point that far toward the moving jaw instead, opens that much wider, and lets closing sweep the
      object onto the fixed jaw (`Poses.held`). Which margin works, and what the sweep does to the object, is
      untested.
    - **Try it on the robot:** a pen1-like pick with the new grasp point and gripper commands (a new session), once
      at margin 0 and once at ~1 cm.

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
16. **The arm sags below the plan, and nothing corrects for it (a major issue).** On pen2 (2026-11-05) the left arm,
    reaching out level at ~96 cm, held its attack pose **1.2 cm below** the plan (elbow_flex 1.6 deg and wrist_flex
    0.9 deg off; across the jaws 0.1 cm). The model given the *measured* joints matched the real fingertips in the
    image (~8 px), so the error is the servos not reaching their targets under the arm's own weight, not the
    calibration; pen1 saw 1.2 deg at rest and more loaded and extended. It is about the size of a thin object and of
    the clearances the grasps rely on (a jaw ~1 cm above a book's edge, a rim). What worked by hand: hold at the attack
    point, measure the error (planned vs measured-joint pinch point), aim the grasp that much higher (1.0 of the 1.2
    cm). What would fix it: a tool that returns that error at a hold and re-solves the next poses with it; then a
    gravity model of the sag (per joint, by load and extension) so the plan is right the first time, or a higher
    servo P gain / a closer settle tolerance (`tuning_notes.md`).
    **Loaded, it is worse:** on container1 (2026-11-05) the right arm carrying a plastic tub by its rim held
    shoulder_lift 2.5-8 deg short of its targets, and `run_execution` stopped the carry at the first labeled pose
    (2.5 deg). `prepare_execution(..., pose_tol_deg=4.0)` (`session.py execute --pose-tol 4`) loosens it for a
    carry; the fix is the same as above (a sag model by load).
    **Built (2026-11-05): the `arm_error` tool.** At a hold, the gripper's error against the pose it was sent to
    (the model at the measured joints vs the plan; vertical, across the jaws, along the axis) and the joints that
    missed, with `.offset` to aim the next short move from there at its target minus it (by hand, this made pen2's
    book grasp land and container1's rim grasp). Checked on container1's attack hold: (0.25, 0.23, -0.52) cm, as
    measured by hand. **Used on bowl1** (four attack holds, 0.4-0.8 cm; each grasp aimed to cancel the full error
    landed). TODO: a couple more sessions, then add it to `protocols.md` (rule 1's "measure the arm" step); and log
    its measurements (pose, load, error) toward the sag model.

17. **Seeing a grasp from the head camera: the plates are checked, the forearm is not.** pen2 chose two picks the
    camera could not check: the jaws edge-on (`XleRobotModel.visibility` +0.03), then hidden by the forearm (its
    black sleeve: the replay against the fitted plates found the plate off to the side). Built (2026-11-05): `calibration.wrist_camera_board` (now `arm_shapes`) fits each plate into the calibration
    (`WristBoard`, a box link); `XleRobotModel.hidden_by` / `view_of` test the line of sight against both plates;
    `find_view` refuses views a plate hides; `solve_pose` reports per pose "jaws seen / edge-on / hidden by the ...
    plate"; `calibration.check` draws the plates. Replayed on pen2 and container1: the right plate's fit sits on the
    real plate (2.3 px RMS, 38 x 43 mm) and hides none of those grasps.
    TODO:
    - **The sleeved forearm:** a cylinder along the lower arm with the sleeve's real radius, tested like the plates.
      It is what hid pen2's second pick. Built (2026-11-06), not yet run on the robot: `calibration.arm_shapes` fits
      it (and the fingers, the gripper body, the wrist) as capsules from outline clicks (`Capsule`,
      `XleRobotModel.capsules_root`); still to do: fit them on both arms, then test the line of sight against them
      in `hidden_by` / `view_of` like the plates.
    - **Re-check the left plate's fit:** 38 x 54 mm at 3.3 px RMS (the right: 38 x 43), not square; likely a corner
      clicked out of order or a misclick (the report, now `arm_shapes.md`, has the per-corner errors; `calibration.check` shows
      whether its outline sits on the plate). Re-run if it is off.
    - Using the report in a session: prefer a grasp the camera sees before contact.

18. **A rolled grasp on a vertical object is wider than its `width`.** `solve_pose` takes `width` along the jaws'
    closing line; rolled by `roll`, that line crosses a vertical cylinder over `width / cos(roll)` (bowl1: a 4 cm jar
    at roll -60 needed 8 cm, and the plan put the fixed jaw inside it; the swept move-in check caught it). Fix:
    describe the object (an upright cylinder of diameter `d`) and let the tool compute the opening, or warn when a
    roll is far from level for an upright object.

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
