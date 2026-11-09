# Sessions

What each session on the robot tried, what worked, what failed (with the numbers), and what it taught. Newest first.
The design these sessions test is in [harness.md](harness.md); what is still missing or wrong is in
[open_issues.md](open_issues.md). Session folders are `harness/scratch/sessions/<name>/` (gitignored): captures,
renders and the workspace log; `session.py log <name>` prints the calls and objects. The harness state
(`workspace.pkl`) of the sessions before pen1 (`cable1`, `cable1_new`, `chapstick1`, `desk1`) no longer loads: it
holds objects removed since (`Plan`, `Reach`) or lacks the call log. Their images and `log.jsonl` still read.

## bowl8 (2026-11-08): the chapstick, by DA3 contact patches, without touching, into the container (done)

**Setup.** The bottle in the container from bowl7; the chapstick (1.75 cm) upright at (-51, 7) cm, near the right arm's
far reach. The pattern: the head plans, the wrist refines (DA3 on a few views at a hold, a cylinder fitted to the
graspable part, the contact patches in the jaws' frame); the person: no nudging, the fixed finger close but not touching.
`harness/scratch/da3_check.py` (`checkpoint`), DA3-SMALL in its isolated environment.

1. **Head:** the body only (below the cap), radius held at the measured 0.875 cm; grasp B (2.7 cm up, pitch -30).
   The approach stopped at "stage" (the elbow 2.7 deg short of 2.5) with the jaws still closed (their "open" was in
   that step): re-planned to the attack point with 4 deg, then the jaws opened by hand.
2. **Checkpoint 1** (a +-15 deg wrist roll, SfM scaled by the head's table, DA3 posed): the whole box first came out
   "above the table" (the table there sits higher than the view's average): fixed with the table from a ring of
   pixels around the box, the object's pixels by color (not the wood), and DA3's flying pixels at depth edges and its
   low-confidence third dropped. The free radius came out 1.3 cm (0.875 measured: the edge filter leaves few points
   on a thin object), so the radius was held: the widest chord 2.1 cm beyond the tips, 1.2 cm outside the fixed jaw.
   Moved 1.5 cm sideways, 1.1 cm in.
3. **Checkpoint 2:** sideways right (the contact 2-3 mm from the fixed jaw, as planned), but the chord 2.05 cm beyond
   the tips where checkpoint 1 predicted 1.0: along the approach, the arm (or the estimate) fell ~1 cm short.
   Moved in 2 cm, then 1 cm: by the joints read, done; by the picture (and the person), the chord still at the tips.
4. **Too close to roll** (the person). Carried through the arm's model from checkpoint 2, DA3's cylinder overlapped the
   fixed finger by 3.5 mm: impossible (the person saw a gap): measurements carried over 3.4 cm of modelled motion are
   wrong. **DA3 on the current frame alone, scaled by 10 points along the fixed finger (6% spread):** the chord at
   the tips, the fixed finger 8 mm from the chapstick. In 5 mm, then horizontally 4.4 mm (along the approach would hit
   the table: the person), then the sideways correction (first applied with the wrong sign, by me: the contact was on
   the moving jaw's side of the fixed finger, unlike at checkpoint 1; corrected).
5. **Close to 2:** the jaws stopped at 3.94 (gripped). Lifted 2 cm: held. Carried at pitch -20 over the container
   (refitted from a fresh head capture), released: in (the person; the server was shut down after).

### Findings

- **The pattern works on a narrow, light object without touching it:** head plan; at a hold, DA3 on a few views posed by
  SfM and scaled by the table; a primitive on the graspable part (the radius held when the free fit disagrees); the
  contact patches; small moves; the person's eye at the last centimeter.
- **Close in, a single frame scaled by the fingers works** (many points along a finger; two tips did not, bowl7):
  the last check needs no roll.
- **Never carry a measurement through the arm's model over centimeters:** re-measure at the pose.
- **Moves along the approach fall ~1 cm short** over a few cm (twice): plan the last centimeters from a fresh view.
- **Sign conventions in the contact step** need care: which side of the fixed finger the contact is on decides the
  shift's direction. Build it into the tool, not by hand.

## bowl7 (2026-11-08): the bottle, by the wrist scan and DA3, into the container (done)

**Setup.** bowl4's scene. Right arm, `urdfs/xle_calib_1008_wrist2.urdf` (the fingers' 1 cm thickness in; plans aimed
1 cm high, at the fingers' middle), every move approved.

1. **Head:** the bottle fitted 3.8 cm wide (the marked region took in the cap: the body is 2.9 cm, bowl4's measure); grasp
   from the robot's side at pitch -41, to the attack point.
2. **`wrist_scan`** (roll +-30, flex +-10, `torch`): 28 of 28 pairs matched, 0.71 px. The scale by the object's width
   came out x1.60 (the width was wrong, and three marked frames too few); by the table, x1.12. `scan_fit_cylinder`
   moved the bottle 1.6-2 cm from the head's fit; with a free radius it measured 1.3 cm (set aside then: it was right).
3. **Revised attack, in by the closing line:** the closing line kept saying "not far enough" and the person agreed. The
   first close (`gripper.pos` 10) caught the bottle's foot: too low, not far enough onto it.
4. **The person's strategy for this gripper:** jaws wide, the closing line on the object, then the **shoulder pan** to
   press the fixed finger onto it (only the moving jaw closes; the wrist roll does not move the fixed tip: it is on the
   roll axis), then close.
5. **Depth Anything 3 (DA3-SMALL, an isolated environment; its packaging conflicts with ours):** on the scan's 8 frames
   with our poses, in 2 s: sharp depth on the bottle, chapstick, jaws, container; the bottle's axis within 5 mm of the
   wrist fit, its radius 1.30 cm. On one frame scaled by the two fingertips: the tips' scales 20% apart and only 50 of
   12,000 bottle pixels in the jaws' slab: no good. **On a 3-frame wrist roll at the current pose (poses from SfM,
   scaled by the table), a cylinder fitted to all 9,238 bottle points: radius 1.52 cm (measured 1.45), the widest chord
   0.9 cm beyond the fingertips, the fixed finger 1.15 cm from its contact:** "advance 1.9 cm, pan to contact".
6. **Advance 1.9 cm, pan +0.8 deg, close to 6:** the jaws stopped at 9.05 (a real squeeze, 3 of `gripper.pos`); the person:
   "a tad low, but will probably work". Lifted 1.65 cm: the bottle kept its place and size in the wrist picture.
7. **The place:** the carry at pitch -20 (-39 and -30 do not reach over the container with this roll); stopped at
   "up" (the loaded shoulder 2.8 deg short of 2.5), re-planned from there with 4 deg: released over the container, in.

### Findings

- **DA3 on a few wrist frames at the current pose, posed by SfM and scaled by the table, plus a primitive fitted to all
  the object's points, gave the contact patches in the jaws' own frame**, and with them the move that worked. Its
  depth does not go through the arm's model over large moves (the scan pose was 6 cm of modelled motion away).
- **The object's width must come from its graspable part**: the head's fit took in the cap (3.8 against 2.9 cm), and every
  overlay inherited it.
- **Shoulder pan to contact, then close** (the person's): reliable with a fixed jaw.
- **A firmer close** (target ~6-8 below contact) held where 1.3 had not.

## bowl5, bowl6 (2026-11-08): the bottle, aligned from the wrist camera (not picked)

**Setup.** bowl4's scene again (the container, the white bottle and the chapstick upright). Right arm; the first
session with the wrist-camera alignment loop (`wrist_look`, `plan_alignment`, `triangulate_wrist`, `align_step`:
[object_grasping.md](object_grasping.md) section 6), `urdfs/xle_calib_1008`, every move approved. The bottle fitted
3.6-3.7 cm wide (a depth from parallax later agreed: 13.8 cm against the width's 13.7); grasp A, from the robot's
side at pitch -40, 4 cm up the bottle; `execute` to the attack point, held there.

### What happened

1. **The fingertips show in the wrist picture** (both arms): the last check can be against the real tips.
2. **The right wrist camera's pose was off ~8 deg (bowl5).** The drawn tips sat 34-51 px from the real ones. A
   gripper sweep at the hold (`gripper.pos` 10, 20, 35, 50: the moving tip's arc and the fixed tip, 5 points) fitted
   a turn of 7.8 deg to 3.2 px RMS; a shift alone needed 13 mm and left 8 px (up to 15). The person's view: the
   extrinsics calibration was bad for this camera (the checkerboard fit did not show it). Patched as
   `urdfs/xle_calib_1008_wrist.urdf` (`api.MODEL`), the session restarted as bowl6 on it: the tips sit on the real
   ones. The calibration's check should fit this from the tips (open issue).
3. **A sidestep along the hinge gives the wrong parallax.** The first sidestep (2 cm along the hinge: up, for an
   upright bottle grasped from the side) and the step after it gave two triangulations 1.8 cm apart in height
   (depths 13.0 and 11.2 cm). Along an upright object's axis nothing marks the same height in both views. Changed
   to a sidestep across the jaws (the object's two sides pin the correspondence): the rays then passed 1 mm apart,
   and the depth agreed with the width's. `hinge_gain=0` leaves the along-hinge offset (the height on the object) to
   the plan.
4. **Centering across the jaws worked:** 23 -> 6 -> (sidestep) -> 13 -> 4 mm, the steps going 90-106% of the
   way predicted when they were sideways.
5. **The first advance went down instead of in.** Asked: the pinch (-0.8, -0.2, -0.9) cm along the approach; by
   the joints read it moved (-0.1, 0.0, -1.5): the shoulder lift reached its target, the elbow stopped 1.2 deg short.
   The servos' deadband (0.5-1.5 deg per joint, 3-8 mm at the gripper) is now as large as the steps. After it, the
   depth carried through the model (11.7 cm) and the width's (13.5 cm) disagree by 1.8 cm, and the approach line runs
   2.6 cm below the marked center (just under the cap): the jaws perhaps 1 cm over the table, by the model.

### Where it stopped: unsure

At this point the measurements disagreed and nothing in the loop could say which was right: the across-the-jaws
offset was well measured (2-4 mm), but the height of the jaws over the table and how far the tips were from the
bottle depended on the arm model through moves the servos had not made as asked.

**Decision:** a second sensor before contact. The head camera (`look_at` the bottle) could not see the jaws: the
sleeve and the wrist camera's board were in front of them (bowl4's problem: from behind the arm it does not see the
grasp). The person chose to raise the gripper first (too high only grasps higher on the body; too low risks the
table).

6. **The elbow stops short under load, every time:** the raise came out 2 mm instead of 7 (the elbow 1.3 deg short of
   43.8 -> 42.5; the advance before, 1.2 deg short). The other joints within 0.5 deg. `align_step` now commands a
   joint more than 0.5 deg short again, pushed past by the shortfall (twice at most): the raise again went as asked
   (the elbow pushed once more, 39.4 for 39.3), and so did the advance after it.
7. **The advance: every measurement said "there", the person said "too far".** After 1.2 cm in along the approach,
   across the jaws 1 mm, the bottle's center 0.3 cm ahead of the aim point, the depth 11.1 cm by the triangulation
   and 11.5 by the width, and the bottle 109 px wide against 108 for its width at the fingertips' depth (the tips as a
   ruler). The person: the gripper is too far in. Not closed.

### Finding (confirmed): the free direction is the object's axis, not the hinge

The marks were at one height on the bottle (just under the cap) and the jaws' line passed 1.6 cm from it along the
hinge; `hinge_gain=0` left that to the plan as "the height on the object". But the approach is pitched 40 deg down,
so the hinge is tilted 50 deg from vertical: a point moved along the hinge also moves along the approach. The free
direction of an upright cylinder is vertical (its axis), so `plan_alignment(..., object_axis=(0, 0, 1))` now slides
the measured center along it to the jaws' line before measuring ahead and across. Measured that way the bottle's
center was 2.3 cm *ahead* of the aim point (against 0.3 cm along the hinge), and so was it: the person's "too far"
meant the jaws had not reached the bottle yet, and the gripper sat a bit low for a sure grasp.

8. **Back and up, then in again, aimed higher.** Pulled back 2 cm along the approach and up 1.5 cm (the person's
   call). Triangulated on the label (the top ends of its two black lines: a point that marks the same in both views,
   unlike the smooth sides), carried 1.8 cm horizontally onto the axis (`triangulate_wrist(..., to_axis_m=)`; a step
   along the nearly vertical ray first went 12 cm down it). The person: aim against gravity (`GraspSettings.sag_lift`,
   1 cm, now in every plan; 1 cm up added to the next step). Two steps in, every joint pushed again once (the elbow
   1.3 deg each time): the jaws' line met the bottle's axis at the planned height, 78.1 cm, and the measurement said
   0.3 cm to go.
9. **The closing line said otherwise, and was right.** `render_wrist` now draws the closing line (the fixed tip to
   where the moving tip touches an object of the plan's width), the moving tip's sweep and where the object's center
   is once held: all rigid to the gripper. It crossed the picture over the bottle's foot, its contacts 107 px apart
   for 3.6 cm against the bottle's 93 px on that row: the jaws ~1.5 cm short. The person confirmed: the gripper had to
   move toward the bottle.

### Finding: triangulation from the arm's moves inherits the arm's motion error

A triangulation's depth is off by about (the baseline's error) / (the baseline). The baseline (how far the camera
moved between the two looks) came from the joints read, through the model; the encoders read true, but backlash,
link flex and a sag that changes as the arm extends or rises are not in them. The baselines were short (1.5 cm
sideways, 2.9 cm up): 3 mm of real-against-modeled motion on 2.9 cm is 10%, 1.5 cm at 14.6 cm, the size of the miss.
The tell was in every step's report: the fraction of the predicted shift that happened came out 157, 90, 129, 69, 112,
110 and 42% (in the simulation, with only fixed joint offsets, 99%: why the arm's small relative moves were thought
accurate). The two sources the arm's motion does not enter agreed with each other and with the person: the depth
from the object's width (15.2 cm against the triangulation's 14.6; the fitted width, 3.6 cm, was right) and the
closing line / fingertip ruler (rigid to the gripper: only the camera's pose on it and the gripper map).

**So:** the depth from the fitted width first, triangulation as a cross-check only, and then from a baseline of 4-5
cm or more; a step whose gain is outside ~0.8-1.2 marks any triangulation across it as suspect; and the closing line
is the go / no-go before closing (the object's two sides on its ends).

10. **In by the closing line, then the close (missed).** Two steps straight along the approach (1.5 then 1 cm; each
    went in less and down more than asked, ~2.5 mm), until the closing line crossed the bottle near its middle (the
    person: move it toward the object's midpoint). Closed to `gripper.pos` 17.0: the jaws stopped at 18.24 (3.5 cm);
    lifted 3 cm: the bottle stayed on the table. The person: the gripper was too far from the bottle's middle line.
    The jaw reading said so, unread: across the diameter a 3.6 cm bottle stops them at 18.7; 3.5 cm is a chord ~4 mm
    off the middle, enough for a round bottle to slip out. ~~A grip is "stopped at the object's width", not "stopped
    short of the command".~~ (Wrong: see the findings below; the reading cannot tell them apart.)
11. The lifted pose (3 cm over the missed grasp) is a good attack pose for a second try (the person).
12. **The midline, anchored on the cap.** `render_wrist` now also draws the object's middle line down from its
    observed top (`midline_top`: the cap's center as seen; its rim's ends give the depth from the width) along the
    vertical as the model has the camera tilted, ticked every cm. The wrist camera looked nearly straight down (its
    rays' vertical component -0.98), so the bottle's whole axis fell on about one point, the cap's center: a grasp
    through the middle puts the "held" circle on it, with no depth. It was 33 px (~1.1 cm) short; measured from the
    cap (the width's depth, the axis slide) 1.8 cm short. Two steps in (the second 134% of the way predicted, the
    picture checked instead) put "held" on the cap's center, 2 mm across and 0.3 cm along by the measurement, the
    jaws' line on the body ~3 cm over the table.
13. **The second close: held, too weakly; the third: missed on the lift.** Closed to 17.0: the jaws stopped at 18.31.
    The person: a grasp, but too weak; open and move closer. Opened, 1 cm further in along the approach (the bottle
    deeper between the jaws, the cap filling the gap between the tips), closed to 17.0: 18.31 again. Lifted 3 cm: the
    bottle stayed on the table. The wrist picture said so (a held object moves with the camera and keeps its size: the
    cap went from ~125 to ~90 px across), as it had after the first lift, which I misread as a pick.
14. **Ended there**, the arm 3 cm over the bottle, the jaws closed on nothing.

### Findings

- **The jaws' reading cannot tell a grip.** A chord 4 mm off a 1.8 cm radius is 0.9 mm shorter than the diameter,
  0.4 of `gripper.pos`: inside the gripper map's accuracy. 18.24 (missed) and 18.31 (held, weakly) read the same;
  the servo just stops there, and reads the same after the object slips out. (Item 10's "a grip is stopped at the
  object's width" was wrong: struck.) The test is the lift, judged by the wrist picture (the held object's size) and
  the person.
- **The grip is weak because the squeeze is small.** The gripper is position controlled: its force comes from how
  far the target is past where the jaws stop. Commanded 17.0, stopped at 18.3: 1.3 of `gripper.pos`, the plan's 3 mm
  (`reach.SQUEEZE_M`); bowl4's chapstick dropped on the same squeeze ("almost no squeeze"). Not tried: a harder close
  (~`gripper.pos` 10 on a 3.6 cm bottle, ~8 of error) (open issue 14).
- **The wrist camera has no depth, and every way of getting it was weaker than the person's eye.** The width (good
  when the fit is: 3.6 cm here), the rigid closing line and fingertip ruler (good, but hard to read from a camera
  looking nearly straight down, where nearer and lower look alike), triangulation over the arm's moves (as poor as
  the arm's motion: 10-20% on 1.5-3 cm baselines), the head camera (blind behind the arm). The checks that worked
  were 2D and rigid to the gripper: the closing line against the object's sides, "held" on the cap's center. The
  person's call: we struggled without a depth camera at the grasp.
- **The arm does not make small moves as asked:** the elbow stops 1.2-1.4 deg short under load every time (fixed in
  `align_step` by pushing again), and steps went 42-157% of the way predicted. Moves along the approach came out
  ~2.5 mm lower and less far in than asked.
- **What worked:** the camera's pose from a gripper sweep (bowl5), centering across the jaws (to 1-4 mm), the
  sidestep across the jaws, the axis slide, aiming against gravity, the closing line and the midline.

### Next

- A grip-force test before the next session: the bottle placed between the open jaws by hand, closed to
  `gripper.pos` 17, 13 and 10, and lifted: what squeeze holds it (open issue 14).
- Depth at the grasp: a depth camera on the wrist, or the head camera placed to see the jaws (from the side, not
  from behind the arm); until then, the person's eye at the last check.
- The wrist camera calibration's check should fit the camera's turn from the fingertips (open issue).

## bowl4 (2026-11-07): put a bottle and a chapstick into a plastic container


**Setup.** The clear plastic container (11.0 cm wide, center (-44.2, -11.5) cm, rim at 79 cm); a white bottle
(2.9 cm wide, top 77.6) and a chapstick (1.75 cm, top 78.3) upright on the table. Right arm, the grasp tools
(`fit_cylinder`, `grasp_candidates`, `plan_pick`, `refine`) with `urdfs/xle_calib_11072026.urdf` (the arms' parts
measured as boxes, no plates), `protocols.md` rule 1, every move approved. (bowl3, the session before, stopped at
its first hold: the arm-shape model was replaced, `arm_shapes`.)

### What happened

1. **The bottle, from above (missed).** The first choice was a top-down grasp (pitch -76) for its square grip
   (antipodal within 1 deg) and in the hope the head camera would see between the jaws. At the hold the view filter
   rejected it anyway (a finger in front of the center: `GraspSettings.see_at_attack` added to turn it off, judging
   the hold by the picture). The jaws closed fully on nothing much (`gripper.pos` 12.85 against 12.7 commanded):
   a slight miss. Top-down also left the joints 19 deg from their bounds, and the right arm cannot carry that tilt
   over the container (only pitch -40 or shallower reaches it).
2. **The bottle, from the side (done).** A side grasp (pitch -30, 3.2 cm up the bottle): attack hold 0.7 cm low,
   refined to cancel it; the jaws stopped at 14.5 for the 2.9 cm bottle (14.4 expected: gripped). Lifted, carried
   at the same pitch (z 85) and released over the container's center: in.
3. **The chapstick, first try (dropped).** A near-level grasp (pitch -13) at the far end of the right arm's reach:
   gripped (jaws stopped at 1.75 cm), but closed only to 1.7 cm (width - 3 mm from a fit 2.0 cm wide): almost no
   squeeze. On the lift the arm sagged 2 cm and the shoulder stalled 3 deg short; the chapstick was on the table.
   Reset 5 cm further out it was out of reach for every grasp; moved closer.
4. **The chapstick, second try (knocked over).** A side grasp (pitch -28, phi 30), the radius held at the measured
   0.875 cm. The arm landed within 1.0 cm of plan (0.8 low, 0.3 across the jaws); the jaws closed fully: the
   chapstick lay on the table, to the left of the left (fixed) finger.

### Hypothesis: the move in goes sideways

What we expected: open the jaws to the object's width plus a tolerance, put the point midway between the fingers on
the object, then move straight forward along the gripper's own axis. What the tools do differs in two ways:

- **Not centered:** the fixed finger is aimed 0.8 cm clear of its contact (`margin`), so closing pushes the object up
  to 0.8 cm onto it. Fine for a heavy object, risky for a light, narrow one.
- **Not straight forward:** `to_claw` lines the tip-to-tip line (tilted 8-19 deg toward the tips by the moving jaw's
  arc) up with the closing line, so the gripper's axis leans off the grasp's approach; `plan_pick` puts the attack
  point back along the approach, not the gripper axis. The move in therefore ran 21 deg off the gripper's own axis,
  crabwise: the fixed finger swept across the front of the chapstick (planned 1.7-2.4 cm clear) and then ran
  alongside it at 0.8 cm. Within the arm's ~1 cm error that clips a light object and pushes it toward the fixed
  finger's outside: what was seen. The arm's orientation itself matched the plan within ~1 deg.

To try: center the jaws on narrow objects (no margin) and move in straight along the gripper's own axis (the attack
point back along it, the grasp pose unchanged).

**Tried (third chapstick attempt):** the fixed finger 1.2 cm clear, the move in built straight along the gripper's
own axis (`solve_pose` from the candidate's pose; 0.0 deg between travel and axis, the fixed finger closing in from
4.6 to 1.2 cm clear in the model). At the hold the chapstick was *not* between the fingers (seen by the person).

### Finding: the camera's pose at the head's home is ~2 cm off

At that hold, the real fingertips (from the depth) were 2.8 cm (left, fixed) and 1.9 cm (right) from where the model
puts them at the measured joints, mostly along the reach; the arm's joints were within 1 deg of the plan
(`arm_error` 0.9 cm, the usual sag). With only the head turned to the grasp (`look_at`, the arm holding), the same
fingertips came within 1.3 cm of the model (mostly vertical). So most of the error was the camera's pose with the
head at home, which the startup scan also showed: the table read 70.3 cm from the home pose against 71.7-72.4 from
the others. Earlier holds were judged with the head aimed (`look_at`), where the model's fingertips sat on the real
ones. Two consequences: judge every hold, and plan every move in, from a capture with the head aimed at the grasp,
not from home; and `arm_error` compares the joints only, so it cannot see a camera error: the fingertips measured in
the image are the check (open_issues).

**Fourth chapstick attempt (between the fingers, fell on the close).** Re-planned from the aimed view: the chapstick
re-fitted there ((-50.4, 7.4), 0.7 cm closer and its base 0.9 cm higher than the home view's fit), the straight-in
path aimed to cancel what the real fingertips would be off by (the joints' sag plus the model-to-depth gap, about
(+0.2, +0.5, +0.4) cm), the fixed finger 1.2 cm clear and the moving one 2.0 cm. The predicted fingertip paths
started on the real fingertips in the image and ended either side of the chapstick, and the arm landed as planned by
its joints (0.8 cm: the 0.7 sag, 0.0 across the jaws). The chapstick *was* between the fingers (seen by the person),
but off center: closed on, it tipped and fell (`gripper.pos` 6.63, closed fully).

### Takeaways

- **Upright cylinders: from the side, not top-down** (the person's guidance): more room for error, and the carry
  keeps the grasp's pitch. The bottle, from the side at pitch -30, went in first time.
- **The move in must be straight along the gripper's own axis.** `plan_pick` moves along the grasp's approach, which
  the tip-line tilt leaves 8-19 deg off the gripper axis: the fingers travel crabwise and sweep across a narrow
  object. Built by hand here; to go into `plan_pick` (open_issues).
- **Judge holds and plan moves in from a view with the head aimed at the grasp.** The home pose's camera is ~2 cm off;
  aimed, the model's fingertips were within 1.3 cm of the real ones, and predicted paths started on them.
- **Check the grip by the jaws' position after the close:** the jaws stopping near the object's width means
  gripped, closing fully means missed (every miss here read that way). Close well past the object, from a measured
  width (holding the radius in `fit_cylinder`), not 3 mm past a fit that may be too wide.
- **Light, narrow objects need centering, and only one jaw moves.** With the object off center, the moving jaw pushes
  it onto the fixed finger and an upright chapstick tips before it is held. Open: center it between the fingers
  (the fixed finger's margin equal to the moving one's clearance), grip it lower (below its middle), or close slowly;
  or nudge it upright against the fixed finger first.
- **Reach:** the right arm reaches the chapstick only within about 50 cm of the root along -x; 5 cm further out
  nothing passed, and near the limit the shoulder stalled on the lift (dropping the first grip).

## bowl1 (2026-11-05): put a chapstick, a small container and a jar into a bowl

**Setup.** The container1 tub as the bowl (rim center (-45.5, -1.0) cm, radius 5.5, z 77.8); in a row in front of
it a jar with a blue lid (4.0 cm wide, top 79.3), a white-capped container (3.3 cm, top 78.4) and a chapstick
(1.6 cm, top 79.1). Right arm for all three, `urdfs/xle_calib_11052026.urdf` (the wrist camera plates fitted, so
`solve_pose` reports when a plate hides a grasp), `protocols.md` rule 1 and pen2's proposed rules, every move approved.

**Outcome.** **Done:** all three picked and placed in the bowl, each on the first grasp: the chapstick, then the
white container, then the jar (it ended on top of the container).

### What happened

1. **Order and arm:** chapstick first (planned first, the white container's approach passed 2.4 cm from it), then
   the white container, then the jar, each with nothing left on the arm's side.
2. **Each pick the same way:** staging 6 cm above the attack point; the attack point 4 cm back along the gripper
   axis; hold; `look_at` the grasp; re-locate the object; `arm_error`; sweep the move in at the measured pose; re-solve
   aimed to cancel the error; in, close, lift 3 cm, hold and look; carry at half speed with `--pose-tol 4`; place,
   open, hold; up 6-7 cm; home; a fresh look before the next.
3. **`arm_error` at the four attack holds:** 0.4, 0.4, 0.8 and 0.5 cm, mostly 0.3-0.6 cm low, 0.0-0.6 across the
   jaws (a more folded pose than pen2's level reach, which sagged 1.2). Every grasp aimed to cancel it landed.
4. **The camera report chose the grasps:** the right plate hid the jar's grasp at pitch -30 roll 0 and the
   chapstick's at pitch -40 roll -30; picks were taken where the jaws were "seen" (-0.37 to -0.66). In every centered
   look the jaws and the object were in view.
5. **Grasps:** the chapstick 1.2 cm into the jaws, closed to the tips meeting (held, slid up 3 cm as planned); the
   white container by its cap (screwed on), closed to width - 3 mm (`gripper.pos` stopped at 15.6: pressing); the jar
   by its lid (stopped at 19.6). Fixed-jaw margins 0.8-1.3 cm, widened at the hold when the sweep left 0.3-0.4 cm.
6. **The jar, first plan:** roll -60 (to keep the jaws visible). The sweep at the hold found the fixed jaw *inside*
   the jar's body: a tilted closing line crosses a vertical cylinder over width / cos(roll) (8 cm at 60 deg), while
   the grasp model takes `width` along the closing line. Re-planned level (roll 0, pitch -40) from the hold: up,
   the roll turned over 5 poses up high, a new attack point.
7. **Locating:** the same object read 0.3-0.7 cm apart across head poses (the views' calibration, as container1);
   one view with the plate over half the jar's lid put its center 1.5 cm off (a partial disc's median); the full-lid
   capture from the same head pose was used instead.

### Takeaways

1. **The procedure works for a sequence:** three picks, no misses, no object pushed. Rule 1's swept move-in check
   caught a real error (the tilted grasp) before contact.
2. **Roll the jaws only for objects that are wide along the tilt;** for a vertical cylinder keep them level, or
   use width / cos(roll). The tools do not know (`open_issues.md`, 18).
3. **Locate from a view that sees the whole object;** a partly hidden object's points are biased.
4. **`arm_error` is useful as built:** four holds, each correction landed.

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
