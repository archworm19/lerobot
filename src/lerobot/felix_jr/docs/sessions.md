# Sessions

What each session on the robot tried, what worked, what failed (with the numbers), and what it taught. Newest first.
The design these sessions test is in [harness.md](harness.md); what is still missing or wrong is in
[open_issues.md](open_issues.md). Session folders are `harness/scratch/sessions/<name>/` (gitignored): captures,
renders and the workspace log; `session.py log <name>` prints the calls and objects. The harness state
(`workspace.pkl`) of the sessions before pen1 (`cable1`, `cable1_new`, `chapstick1`, `desk1`) no longer loads: it
holds objects removed since (`Plan`, `Reach`) or lacks the call log. Their images and `log.jsonl` still read.

## container1 (2026-11-05): pick a plastic tub by its rim, move it past a cable

**Setup.** An empty clear plastic tub (rim radius 5.4 cm, ~7.5 cm tall) on the +y side, the USB-C dongle and its
cable between it and the -y side. Right arm, `urdfs/xle_calib_11042026.urdf`, the claw grasp model, Claude Code
driving `session.py` with pen2's proposed rules (below, in pen2), every move approved.

**Outcome.** **Done:** picked by the rim, carried over the cable, set down upright past it.

### What happened

1. **Scan and perception.** Table 71.5 +- 0.5 cm. The rim from depth (clear plastic: the rim reads, the walls
   mostly do not): a circle fit, center (-41.1, 8.1) cm, radius 5.4, 0.2 cm residual (310 points).
2. **Grasp: pinch the rim** (one jaw inside the wall, one outside; the tub is too wide for the jaws) on its +y side,
   right arm, pitch -75, closing across the wall, width 0.4 cm (the close goes to the tips meeting: the hard close of
   `tuning_notes.md` 3.9), fixed-jaw margin 0.8 cm. The left arm cannot reach that side. Staging pose 10 cm back and
   up, then the attack point ~6 cm above the rim, hold.
3. **At the attack point** the arm was 0.5 cm low (less than pen2's 1.2: a more folded pose). From the home head pose
   the jaws were cut off at the frame edge and only the rim's near arc was seen (its fit moved 1.4 cm, radius 5.2: a
   partial-arc bias), so the camera was pointed at the grasp (**rule 1 amended: center the look on the grasp**).
   The centered view fit the whole rim (center (-41.6, 7.5), radius 5.5, 564 points; ~0.8 cm from the scan's,
   most likely the views' head calibration). The straight move in, swept at the measured error, kept the fixed tip
   0.6-0.9 cm inside the wall and the moving tip 2.2-2.4 cm outside: in.
4. **Closed and lifted 3 cm, hold:** held, but **tipping**: the near rim rose 2.5 cm, the far side stayed on the
   table (a one-point rim grip). Lifted to 10 cm at half speed: it came free and hung, little swing. Depth on the
   hanging tub read its bottom ~1 cm over the table (clear plastic; the person saw it held, well up).
5. **Carry:** stopped at its first labeled pose: **loaded, shoulder_lift settled 2.5 deg short** (the labeled
   poses' tolerance), then held as designed. `prepare_execution` got a `pose_tol_deg` (4 for this carry;
   `open_issues.md` 16). Re-planned from the stop capture: over the dongle at ~88 cm, down to 80 at y -8, open,
   hold; up 6 cm off the rim before home (a joint move home with the jaws around the rim could drag it).

### Takeaways

1. **The rules worked** on an easy object: staging, the attack hold, the centered look, the swept move in, a test
   lift with a hold before the carry, and clearing the arm before the last look.
2. **A one-point rim grip tilts and hangs** the object; fine for an empty tub, not for anything with contents.
3. **Load matters for settling:** the carry needs a looser tolerance, or a sag model, or both.

## pen2 (2026-11-05): pick a pen out of a plastic cup, then out of a book; drop it past a cable

**Outcome.** Not picked. Four approaches (two arms, two scenes); the pen was moved by every approach in the cup and
fell from the book on the last. What it produced: visibility and occlusion as grasp criteria, the arm's repeatable
~1.2 cm sag at a level reach (`open_issues.md`, 16), grip force as overshoot past contact (`tuning_notes.md` 3.9),
and the proposed rules at the end of this entry. Goal 2 (click the pen with the other arm) was not started.

**Setup.** A clear plastic cup (rim 79.2 cm, ~7.5 cm above the table); a pen leaning ~55 deg from vertical
toward +y against the far rim, ~1.5-2.5 cm of it above the rim; a USB-C dongle and its cable on the -y side
(body at about (-38, -9), cable end (-49, -11.5) cm). Right arm, `urdfs/xle_calib_11042026.urdf`, the claw
grasp model (width and margin). A "Claude session": Claude Code drives `session.py`, asks before every command
that moves the robot, and keeps everything in `harness/scratch/sessions/pen2/`.

Goals: (1) pick the pen from the cup and drop it on the table on the other side of the cable; (2) pick the pen
from the cup and click its top with the other arm.

### What happened

1. **Scan.** Table 71.7 +- 0.6 cm; tilt 1.6 deg level, 2.9-4.1 deg at the steepest head tilt (a little more
   head-pitch dependence than pen1's). capture8 / capture5 showed the cup, the pen and the dongle.
2. **Perception.** The plastic cup does **not** hide the pen from depth (pen1's glass did): 24 of 25 samples
   along the pen landed on it, a line fit with 0.09 cm mean residual over 10.8 cm. The rim reads where depth sees
   its near side (z 79.0-79.3); its far side reads the table. A later capture from another head pose put the pen
   0.5-0.6 cm off the first line, a constant offset (each view sees another side of a ~0.9 cm pen).
3. **Planning.**
   - Roll 0 (jaws horizontal) would close at 38 deg to this pen: it leans across the robot's view, not away from
     it. A pitch x roll sweep (closing direction vs the pen's line) gave roll +40 (the wrist_roll limit is ~+43
     here) at 75-79 deg; pitch -40 to keep the lower jaw over the rim.
   - The lift as a vertical `approach` would drag the pen's bottom across the cup; instead the pen is pulled out
     along its own line (5 poses, 4 cm apart), its bottom tracked fixed in the gripper's frame (above the rim by
     `pull3`, 4 cm over it by `pull5`). The drop past the cable is at the edge of the right arm's reach:
     (-42, -16, 86) cm reaches, y -20 does not (wrist_roll at its limit).
4. **Staged approach.** To `pick.back`, hold, `look_at` the grasp: the planned pinch point lay on the real
   gripper's front corner (no gross error, ~1 cm in the image), but **the jaws could not be seen**: the head looks
   down the gripper's back. In to `pick` (jaws open), hold, look: the gripper's face at the pen's top, the pen's top
   ~0.7 cm left of where it was (a nudge, or the gripper hiding the tip); whether the pen was between the jaws
   could not be told from the image. (`execute` refused the first try at this step: the poses were solved before
   the arm moved; they were solved again from the newest capture.)

### Findings

1. **We chose a pick the camera cannot check.** `XleRobotModel.visibility` (the cosine between the gripper's hinge
   axis and the line of sight from the grasp point to the camera: +-1 sees the jaws side on, 0 looks along the
   plane they open in) was **+0.03** at that pick, which is why the jaws were hidden. It was chosen for reach and the
   closing angle only.
2. **Visibility and a good closing angle pull against each other**, but both can be met when the closing
   direction is perpendicular to the object's line *and* the line of sight (possible here: the pen lies almost
   perpendicular to the sight line, cos 0.16). Sweeping reachable poses, keeping those closing >= 65 deg across the
   pen, ranking by |visibility|: pitch -40 / roll +40 is +0.03 / 75 deg; **pitch -70 / roll -100 is -0.37 / 86 deg**
   (drawn, it shows an open wedge with the pen between the fingers). About 0.4 is the ceiling here with the jaws
   across the pen; |0.8| is reachable only closing along it.
3. **The head camera's only lever is the gripper's orientation**: turning the head barely moves the camera
   (~1.5 cm). The wrist camera removes the problem rather than working around it (pen1, takeaway 2).

### Recommendations

1. **Pick for visibility, not only for reach and the closing angle**, for small objects: prefer the reachable pose
   with the largest |visibility| among those whose jaws close well across the object, and stage the approach so
   the hold-and-look at `pick` can see the object between the jaws.
2. **Make it a number the model sees, not a hidden sweep** (`harness.md` 1.K): `solve_pose`'s description gives each
   pose's grasp visibility, with a warning below ~0.2, and its docs say to prefer a visible grasp for small objects.
3. **Not covered by the number:** the arm's own body between the camera and the jaws (self-occlusion), and a steep
   approach bringing the jaws nearer a rim (at pitch -70 the lower jaw is ~0.7 cm below the grasp point, the rim
   1.2 cm). Both still need the render and the look.

### The visible pick, tried

5. **Re-approach at the visible pick** (pitch -70, roll -100, visibility -0.37): back out 5 cm, up 6 cm, the turn
   spread over 10 poses on the way down (`prepare_execution` refuses a turn in place: it steps by distance, so a
   140 deg roll at one point is one step), all chained from the arm's pose (solved from the rest pose, the pick had
   landed on the other elbow branch). Held at the pick, jaws open: **the jaws were hidden again, this time by the
   arm itself**: coming in steeply from behind and above, the forearm sat between the head camera and the jaws.
   (First read as the forearm and the wrist camera's plate; replayed against the fitted plates,
   `xle_calib_11052026.urdf`, the plate was off to the side and the jaws were behind the **black sleeve over the
   forearm**, much bulkier than the arm's links.) The person saw the pick very close to the pen.
6. **The pen had moved:** fresh depth put it 1.4-2.0 cm off the scan's line, leaning 61 deg (was 55), its heading
   turned ~15 deg (0.55 cm off after the first `look_at`): pushed during the first pick, the back-out or this
   descent, still leaning on the rim.

**Findings.** The visibility number is the jaws' orientation only (one cosine). It knows nothing about what stands
between them and the camera, and the URDF could not tell it: it has **no wrist camera link** (nor its board) and
**no collision geometry**, only the arm links' visual meshes.

**Recommendations (occlusion).**

1. **Score occlusion when choosing a pose:** z-buffer the URDF's visual meshes from the head camera at the solved
   joints and check the grasp point is the nearest surface at its pixel (gripper body, forearm). A cheaper, coarser
   version tests the `arm_path` skeleton's segments against the grasp pixel; it misses link widths and the wrist
   camera.
2. **Add the wrist camera's mount to the URDF** (a link, measured once): needed for (1), and the first step toward
   using that camera.
3. **At every hold, check the look can be trusted:** compare the measured depth at the grasp pixel with the
   predicted one; much nearer means something is in front of the jaws.
4. **Re-locate the object at each hold**, not only the gripper: here it had moved ~2 cm before the second approach.

### The left arm, tried

7. **The right arm cannot roll 180 deg** to move its wrist camera out of the way: `ArmIK` caps wrist_roll at
   +-130 deg (`MAX_ROLL_DEG`, the wrist camera's cable; the URDF allows ~+-160), and the flip needed ~-166. It would
   not have helped anyway: the forearm, not the roll, is on the line of sight.
8. **The left arm reaching across** scored best, with a coarse occlusion check added (how near the arm's link
   origins come to the line from the grasp point to the camera; the occluded right pick: 3.7 cm): pitch -20, roll
   -160, heading +155, visibility +0.52, closing 89 deg across the pen, links 5.9 cm off the sight line. The right
   arm was first withdrawn (5 cm back along its axis, up, home).
9. **From home, the pen had moved ~4.5 cm** at its top since the scan (now 53 deg from vertical, heading turned ~16
   deg), still on the rim; re-located (20 samples, 0.11 cm residual) and the left pick solved on the new line.
10. **At the left `pick.back`, the jaws were plainly visible** (both fingers, open, side on), the drawn fingertips
    within ~1 cm of the real ones in the image, across the jaws included. **But the pen was gone from view**: most
    likely the gripper clipped its shaft at the end of the move and it slid off the rim into the cup. The arm's
    centerline (wrist, gripper, pinch point) came within 21.7 -> 6.9 cm of the pen's line along the first move and
    **4.6 cm** at `pick.back` (5 cm back along the axis from the grasp, which on this lean is right above the pen's
    lower shaft); the gripper is a few cm thick and the open moving jaw sticks out ~3 cm. Every approach in this
    session moved the pen (0.7, ~2, ~4.5 cm, then off the rim): each plan checked the grasp point and the jaws' gap,
    never the gripper's body against the object being picked.

### The pen in a book (scene reset)

11. **Reset:** the pen stood vertical in the top edge of an upright book (top edge z 92.5 cm, ~4.5 cm of pen above
    it, the pen at (-45.5, -2.9) cm); the cup moved aside. A fresh scan (table 71.9 +- 0.5 cm). With a vertical pen
    any jaws-horizontal grasp closes across it, so the choice was visibility and occlusion: the left arm, pitch 0,
    roll 180, heading +163, grasp 95.8 cm (3.3 cm over the book), visibility +0.52, arm 6.6 cm off the sight line.
    Planned with rules 1-5: a staging pose 12 cm back and 3 cm up for the move from rest, the attack point 5 cm in
    front of the pen, hold.
12. **At the attack point** the jaws and the pen were plainly visible; the pen had not moved (0.4 cm). **The arm sat
    1.2 cm below the plan** (elbow_flex 1.6 deg, wrist_flex 0.9 deg off its targets; 0.1 cm across the jaws), and the
    model given the *measured* joints put both fingertips within ~8 px of the real ones (the fixed jaw where the
    model said). So the model is right; the arm sags under its own weight reaching out level (`open_issues.md`, 16:
    a major issue). Depth on the fingertips read ~1.2 cm farther, but those pixels mix the thin tips with the book
    behind and were not trusted.
13. **The grasp, re-solved from the held capture** on the pen's re-located line, aimed 1.0 cm higher (of the 1.2
    measured): in, closed: the pen between the fingertips ~1.5 cm below its top; `gripper.pos` 3.8 against 3
    commanded (stopped on the pen, a weak signal).

14. **The lift (8 cm straight up, the elbow's limit for a level grasp) did not bring the pen out.** At the hold the
    gripper's body hid the pen from the head camera (between the book's edge and the jaws only), so the arm was
    cleared (rule 2: jaws opened first, since a pen still in the pages would have been dragged by backing out). After:
    the pen still in the book at the same place, **1.4 cm higher** (top 97.4 -> 98.8 cm): the jaws pulled it up a
    little and lost it. The grip was at the very tips (~1.5 cm below the top) of a smooth barrel, with the close at
    width - 3 mm (`SQUEEZE_M`): the pages held harder than the tips. The `SQUEEZE_M` TODO (`open_issues.md`, 14) and
    grasping deeper than the tips are what this asks for.

15. **The re-grasp: deeper, lower, a hard close** (`tuning_notes.md` 3.9): staged and held at the attack point as
    before (the pen had not moved; **the sag repeated: 1.3 cm**, elbow_flex 1.8 deg, wrist_flex 1.1 deg, 0.1 cm
    across the jaws: predictable), then one execution re-solved from that capture: 6 cm in to a grasp point 1.5 cm
    past the pen (the pen ~1.1 cm into the jaws), aimed 1.2 cm high for the sag, the fixed jaw's line ~1 cm from the
    pen's axis (margin widened from 0.65 to 0.9 cm after the first solve left 0.2 cm), close to `gripper.pos` 0, lift
    8 cm, hold. **The jaws did not clear the book's top edge:** pushing forward into the grasp, the gripper hit the
    book and moved it, and the pen fell (the person saw it). At the top of the lift `gripper.pos` read 1.0 (closed on
    nothing) and depth saw nothing between the jaws and the book's edge where the pen had stood. Going 1.5 cm deeper
    brought more of the gripper over the book's edge, with ~3 cm planned between the jaws' line and the edge: not
    enough for the gripper's body below that line plus the sag's spread. The hard close was never tested on the pen.
    **What it changes:** the hold at the attack point measured the jaws and the pen, but not what the straight move
    from there to the grasp would pass through; rule 1 is now explicit about that (below).

### Proposed rules (for driving a session)

Rules 1 and 2 are the person's; 3-5 follow from finding 10. They are written out here while they are tried.
**TODO:** as later sessions succeed with them, make each succinct and move it to [protocols.md](protocols.md)
(for whoever drives a session; to be fed to the model next to `describe_tools`). Rule 1 moved after container1.

1. **Stop at the attack point and replan before the grasp.** Move to the attack point (the pose before the grasp,
   `pick.back`) and hold; **point the camera at the planned grasp (`look_at`, head only) and capture there**;
   re-locate the object *and* the jaws, solve the grasp again from that capture, then go in and close. (Centering
   puts the jaws and the object near the image center at more pixels, and the whole object in view: container1's
   hold at the home head pose had the jaws cut off at the frame edge and only the rim's near arc, which biases its
   fit. It does not see around the arm (the camera barely moves); and the head's own calibration now enters, so
   compare the object across the views. Plan the look with the attack: the head pose that centers the grasp, and a
   check that the arm at `pick.back` will not cover the grasp in that view.) *Why:* this arm is not accurate: its error (0.5-1.5 cm on pen1, mostly across the jaws;
   a 1.2-1.3 cm sag on pen2) is the size of a thin object, and the look at the attack point is what corrects it
   (pen1 attempt 2). **Before going in, the capture at the attack point must show that the straight move forward
   (along the gripper axis, from where the arm really is) enters the grasp without hitting anything:** the gripper's
   body swept along that move, at the measured pose (the sag included), clear of the depth there (the support the
   object stands on or in, neighbors), with only the object between the open jaws. If it cannot be shown, re-plan
   the attack (higher, another pitch or side), do not go in. *Why:* finding 15: the replanned grasp was right
   about the pen and the jaws, and the gripper still hit the book on the way in. (Needs the gripper's shape: the
   URDF meshes, or a measured box around the jaws and the gripper's body in the pinch frame.)
2. **If the scene becomes hard to see, clear the arm before recapturing:** move it up and out of the workspace
   (rest, or a clear pose) rather than looking around it. *Why:* here the forearm hid the jaws, and a capture with
   the arm in front of the object cannot locate the object.
3. **The attack point must be clear of the object, not just 5 cm back along the gripper axis:** check the arm's
   body (links, then meshes) against the object's model (the pen's line as a ~0.45 cm capsule) with ~2 cm
   clearance everywhere except between the open jaws. *Why:* finding 10; without it rule 1's stop is where the
   object gets hit.
4. **Do not end a move from rest near the object:** the first move is a joint-space arc, uncontrolled in between.
   Send it to a staging pose well clear (>= 10 cm back and above), then only straight, stepped moves come closer.
5. **An object that moved is a stop:** at every hold, compare where the object is with where it should be; more than
   ~0.5 cm, or not found, means stop, report and ask. *Why:* the drift here was visible two approaches before the
   pen fell.

## pen1 (2026-11-01): pick a pen out of a glass cup, put it in the other cup

**Setup.** Two clear glass mugs about 14 cm apart on the desk; a pen (~0.8 cm thick, ~15 cm below the grasp point)
standing in one of them, leaning ~35 deg away from the robot against the far rim. Right arm, the 10/31 calibration
(`urdfs/xlerobot_calib10312026.urdf` then; kept as `tests/felix_jr/reference/pen1_calibration_10312026.urdf`), the harness of `api.py` driven through `session.py`, Claude Code in the
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
