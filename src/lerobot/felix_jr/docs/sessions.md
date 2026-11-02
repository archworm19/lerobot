# Sessions

What each session on the robot tried, what worked, what failed (with the numbers), and what it taught. Newest first.
The design these sessions test is in [harness.md](harness.md); what is still missing or wrong is in
[open_issues.md](open_issues.md). Session folders are `harness/scratch/sessions/<name>/` (gitignored): captures,
renders and the workspace log; `session.py log <name>` prints the calls and objects.

## pen1 (2026-11-01): pick a pen out of a glass cup, put it in the other cup

**Setup.** Two clear glass mugs about 14 cm apart on the desk; a pen (~0.8 cm thick, ~15 cm below the grasp point)
standing in one of them, leaning ~35 deg away from the robot against the far rim. Right arm, the 10/31 calibration
(`urdfs/xlerobot_calib10312026.urdf`), the harness of `api.py` driven through `session.py`, Claude Code in the
model's seat, every head and arm move approved by the person watching.

**Outcome.** The pen was picked on the second attempt (the person saw it held), but the lift stopped after 1 cm and
the arm went home; the place was never reached. Most of the session was spent finding why the first grasp missed.

### What happened

1. **Scan.** 9 captures; table at 72.2-74.0 cm, tilt 1.3-2.9 deg (the desk). capture4 / capture8 (later capture25 /
   capture29) showed both cups and the pen.
2. **Perception.**
   - The pen, from depth: first three good points, later a line fit through 21-22 samples along it in two views
     (0.25 cm mean residual), 35-36 deg from vertical. Samples on a thin object miss often: the pixel reads the table
     behind it.
   - The cups are clear glass and **invisible to depth**: the grid draws straight through them (it hides only where
     depth sees a surface), and a pixel on a rim reads the table behind it. Triangulating a rim from two head poses
     **failed**: the camera barely translates when the head turns (~1.5 cm baseline), so the rays are nearly
     parallel. What worked, from one view: the ray through the base ellipse's center meets the table plane at the
     cup's center; the rim center's ray passes the cup's vertical axis at the rim's height. Empty cup (-37.8, -1.3) /
     (-36.7, 0.0) cm in two views, rim 13.4-13.5 cm above the table; full cup (-39.4, 12.2) / (-38.5, 14.2), rim
     10.9 cm. Repeating it after the first attempt gave the same numbers.
3. **Planning** (`solve_pose`, the model sweeping pitch in snippets).
   - Grasp across the pen, roll 0 (jaws horizontal). A gripper perpendicular to the pen (pitch -36) reached the pen
     but not the lift and the place. A pitch sweep found -30..-10 reaching all three; -15 (later -12) was used.
   - The pen turns with the gripper: the place's heading differs from the pick's by ~20 deg, which moves the pen's
     bottom ~3 cm sideways. Solved by computing where the bottom really lands (the pen fixed in the gripper's frame)
     and correcting the target until it is over the cup (3 iterations to 0.0 cm).
   - Clearance: the empty cup's rim is higher (87.2 cm) than the full one's (84.6); the first carry passed it with
     0.2 cm. The lift was raised to 14.2 cm (the pen's bottom 1.3 cm over the higher rim), the carry at ~101 cm, at
     the edge of reach.
   - Renders: the place was above the focused views (they look down at the table); the scan's level captures and
     `arm_path` showed the whole motion.
4. **Attempt 1** (the full execution, the person stepping through): **missed the pen** and pushed it (its lean turned
   ~12 deg). The run ended early, without an after-capture.
5. **Diagnosis**, staged with `execute(..., hold=True)` and `look_at`:
   - The pen had not moved (4 mm) and was located correctly.
   - At `pick.back` (5 cm short), the servos settled within 1.2 deg of their targets (~0.5 cm at the gripper), and the
     real jaw tip was within 0.1-0.3 cm of the plan forward and across, +0.9 cm "up" (probably the finger's
     thickness: depth sees its top).
   - At `pick`, jaws open: the pen was **on the outside of the fixed jaw**, +0.4..+1.4 cm across the jaws where the
     plan put it at -0.5 (between the jaws). Small errors that were each fine had added up across the jaws, the one
     direction a thin object does not forgive. Nothing measured that direction until the camera looked at the jaws.
   - Backing out, the first 1 cm step **did not settle** within 1.5 deg, the run stopped, and the fallback sent the
     arm home: an unplanned joint-space move starting next to both cups.
6. **Attempt 2.** Rescan; the target shifted +1.4 cm across the jaws; jaws opened to 80 (from 60).
   - Staged approach: to `pick.back`, hold, look: the pen predicted 1.2 cm inside the fixed jaw; nudged 0.6 cm; to
     `pick`, hold, look: the pen measured 1.3 cm inside (the nudge did not show one for one: the arm's own offset
     changed between the two poses, which is the noise level of the whole chain at this distance).
   - Closed: `gripper.pos` read 2 (fully closed, as if empty) and the close-up showed the pen at the very tips of the
     jaws, **but the person watching saw it held**.
   - Lift: the first 1 cm step **did not settle**; the arm went home again (with or without the pen). The server was
     shut down there.

### Takeaways

1. **The settle tolerance is too strict, and a failed step should not send the arm home.** `move_joints` counts a
   move as done within 1.5 deg (server default), and `run_execution` stops on any step that is not; then
   `client.session()` goes home. Unloaded at `pick.back` the joints were already 1.2 deg off; loaded and extended
   they were more, and both "did not settle" stops came on the first step away from the pen. Decided: two kinds of
   failure, two responses. A step that does not settle (the client is alive) stops, **holds**, captures and returns
   the per-joint errors so the model can re-plan; intermediate 1 cm steps get a looser tolerance than the labeled
   poses. Going home stays for when nobody is in control (a lost connection, a crashed client, the watchdog).
   Done (2026-11-01, laptop side only, the server unchanged): `prepare_execution` gives the labeled poses 2.5 deg and
   the steps 4 deg, `run_execution` passes each move's own tolerance, and a move that does not settle raises
   `StoppedError` with the per-joint errors and a capture taken where the arm holds (`client.keep_holding()` ends the
   session without going home).
2. **The head camera cannot see a grasp well enough; the wrist cameras are needed.** It looks at the jaws from above
   and behind at ~40 cm, the jaws hide a thin object, and depth on it is patchy. The wrist camera looks along the
   jaws. Being set up (the person).
3. **Closed loop at the last centimeters works.** Stop short, look, measure the real jaw against the plan and the
   object against the real jaw, correct, continue: this found the miss and put the pen between the jaws. It should be
   the standard pattern for small objects, and "where is the gripper really" is a measurement a tool can make.
4. **The gripper is a claw, not a parallel gripper, and the model treats it as one.** The moving jaw rotates about a
   pivot ~8 cm behind the tips (its tip moves along an arc: (-1.3, 0, +0.4) cm at 10 deg, (-2.7, 0, +0.6) at 20,
   (-4.2, 0, +0.6) at 30, in the pinch frame). The gap is a wedge, widest at the tips; an object's grip center is not
   `width / 2` from the pinch point along the jaw axis, as the tool point assumes; and closing pushes the object along
   the jaws as well as across. Also: the tool point is the closed fingertip, so aiming it at a thin object puts the
   object at the tips. Both need the gripper's geometry from the model and the `gripper.pos` -> jaw angle mapping
   (unmeasured).
5. **`gripper.pos` after closing is not a grasp signal** (it read 2 while the pen was held).
6. **Clear objects:** geometry from the image (an upright object's base ellipse on the table, its rim on the
   vertical axis) works where depth sees nothing; triangulation between head poses does not.
7. **Tall objects in tall containers push the arm to the edge of its reach:** the carry ran at ~101 cm because of
   the pen's length and lean. A lower grasp, or turning the pen upright, would have helped.
8. **Working in snippets, the model makes REPL mistakes:** names reused for other objects (`a`, `b` pointed at the
   located points and were later rebound to poses), plain values (the pen's line) not persisting between snippets,
   a pitch sweep that started each solve from the last *failed* one. Persisting plain values and warning when a name
   changes kind would help; the tool docs should say to chain from the last good solution.
9. **Plans leave the focused views:** lifts and places are above a camera looking down at the table. Level captures
   (the scan's tilt 0) and `look_at` showed them.
10. **Approvals came through chat:** the person approved each move in the conversation, and Claude Code passed that
    approval to the terminal prompt. The gate held (each move approved specifically), but that is how it was used.

## cable1 / chapstick1 (2026-10-31): pick an upright chapstick and move it past an obstacle

**Setup.** An upright chapstick; move it to the other side of a cable (cable1) or past a journal (chapstick1). Driven
by hand with `session.py`, before the current `api.py` and `solve_pose`; nothing was run on the arm then.

**What happened.** The scan, the table height (72.5-74.2 cm over 9 poses, tilt ~2 deg: the desk) and the grids
worked, and `locate_pixels` on a zoomed view gave usable points on the chapstick and the targets. Planning failed on
reach, not perception: every waypoint pointing straight down above ~82 cm, and chapstick1's place at any height, was
out of reach.

**Takeaways.**

1. **Pointing straight down is often out of reach:** the wrist flex limit (+-95 deg), confirmed against placo
   (`tests/felix_jr/smoke_tests/test_ik_smoke.py`). At some tilt every waypoint is reachable (the steepest: -25 to
   -90 deg). This led to choosing a pitch per pose (`harness.md` 1.J).
2. **A free roll can flip the jaws between pick and place,** which would have put the object down upside down; an
   earlier, symmetric gripper glyph hid it. The fix was the model choosing the roll (the same at pick and place), the
   roll in the text of every pose, and fingers drawn in two colors.
3. **The first version of the tools hid too much orchestration** (pitch sweeps, restarts, pulls) and too many
   judgments (a depth path check that called the grasped object an obstacle). They were cut back to one optimization
   per pose and no collision checking (`harness.md` 1.J, 1.K).
