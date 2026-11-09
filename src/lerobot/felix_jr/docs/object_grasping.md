# Object-centric grasping (experiment)

**Status: phases A-C built and tried on the robot (bowl3, bowl4); the wrist-camera alignment loop built and tried (bowl5/bowl6, 2026-11-08: centered, not picked; section 6); bowl7 and bowl8 picked and placed with the head planning and the wrist refining by DA3 (section 7: what worked and what did not)** (results below the phase table). Built: the primitives
(`harness/primitives.py`), the claw mapping and the thin `grasp_candidates` (`harness/grasping.py`),
`ArmIK.solve_frame` (`harness/reach.py`). It changes how the harness plans a grasp: today the model
names a gripper pose (a point and pitch / heading / roll in the root frame) and searches for a good one itself; here
the model names an *object* and *where* to take it, and a tool generates, checks and ranks grasps. If the experiment
works, it replaces the pose tools listed in section 4 and `harness.md` 1.J; until then both exist side by side.

Why: in the sessions of 2026-11-05 ([sessions.md](sessions.md): pen2, container1, bowl1) every grasp was chosen by
hand-written snippets around `solve_pose`, a few hundred solves per pick: pitch x roll sweeps, the closing angle
against the object, visibility, distances from the jaws to neighbors, the straight move in. Those checks decided the
grasps, and none of them is a tool. The one grasp they missed at first (bowl1's jar at roll -60, the fixed jaw inside
the jar: `open_issues.md` 18) came from describing the grasp by the gripper's angles instead of the object's shape.
Composing heading, pitch and roll across frames is also hard to reason about, for a person or a model.

## 1. High-level plan

**The idea.** Objects are first-class: each is a few **primitives** (a cylinder with an axis, a rim, later a box or
plate) fitted from the depth. A primitive knows its own **antipodal grasps**: two contacts with opposed normals along
a closing line (across a cylinder, through a rim's wall). A grasp is described in the object's terms (which part,
where on it, from which side); the closing direction follows from the shape, so the opening is always right. One tool
turns an object into **ranked grasp candidates**: it generates the primitive's grasps on a grid, maps each to the claw
(which contact gets the fixed jaw), solves the arm's joints, checks it, scores it, and describes the best few in words
and numbers. The model picks one by name and judges from the render (`harness.md` 1.K: tools measure, the model
judges). Heading, pitch and roll stay inside; the model does not compose rotations.

**Three stages, kept apart:**

1. **Generate (object only):** the primitive's antipodal grasps, in its frame, filtered only by physics (the opening
   fits the jaws; not from inside the support). Deterministic: a grid over the family's 2-4 parameters, so the same
   scene gives the same candidates, tests are stable and a report can say why nothing on one side works.
2. **Map to the claw:** each antipodal grasp gives two candidates, one per contact taking the fixed jaw (the fixed jaw
   does not move; closing sweeps the object onto it). The claw's geometry (the moving tip's arc, the depth into the
   jaws, the fixed jaw's margin, the close as an overshoot past contact) lives here only.
3. **Filter and rank (robot and scene):** IK, hard checks (reach, the jaws' opening, the attack point and the straight
   move in clear of the object, its support and its neighbors), soft scores (the jaws visible, not hidden by a plate
   or the forearm, distance from joint limits, depth in the jaws). The same checks for every primitive, and for grasps
   sampled from a point cloud later.

**Phases, each with a test that decides whether to go on.** Planning comes before perception: phases A and B use
cylinders typed in from the sessions' hand fits, so a failed replay cannot be the fit's fault, and the riskiest
question (does ranking an object's grasps choose the ones that worked?) is answered first. Only the cylinder is built
until B passes: the `Primitive` protocol is shaped by what the claw mapping and the checks need from it.

| phase | builds | passes when |
|---|---|---|
| A | `AntipodalGrasp`, the `Primitive` protocol, `Cylinder` (an axis, any direction); `to_claw`; the thinnest `grasp_candidates` (`ArmIK`, two checks: reaches, opening fits; ranked) | unit tests of the geometry; `to_claw` puts the pinch point where `grasp_offset` does today; **offline replay of bowl1's jar** from its hand-fit cylinder: the roll -60 grasp (the fixed jaw inside the jar) is rejected, and the grasp that worked is among the candidates (choosing it over the others is B's, with the scores) |
| B | the other checks (attack point and move in clear, no joint jump) and scores (visibility, `hidden_by`, joint-limit margin, depth); `Object`, `plan_pick`; then `fit_cylinder` last | **offline replay** (no robot) of bowl1, container1's body, pen2: the top candidate is like the grasp that worked (bowl1's jar: `test_grasp_replay_smoke.py::test_the_best_is_seen`, an expected failure until then); pen2's edge-on pick and its forearm-hidden pick are rejected or ranked low, with the reason; `fit_cylinder` fits bowl1's three objects and pen2's pen from the recorded captures within ~0.5 cm of the hand fits |
| C | the attack hold folded in: `refine` (re-locate, `arm_error`, re-solve the chosen candidate, check the view again); the tools model-facing (`api.py`: `fit_cylinder`, `grasp_candidates`, `plan_pick`, `refine`; `render` draws objects and candidates). **Tools built 2026-11-06** (`test_grasp_tools.py`) | a robot session (bowl1-like) with no hand-written snippets, every grasp from `grasp_candidates` |
| D | the rim primitive (container1's pinch); the object-centric place (`plan_place`) | container1 and bowl1 replayed end to end, places included |
| E | `harness.md` 1.L written; section 4's removals done | phases A-D pass |

**Phase A results (2026-11-06).** Tests: `tests/felix_jr/test_primitives.py`, `test_grasping.py`, and the replay
`smoke_tests/test_grasp_replay_smoke.py`.

- **The roll -60 grasp cannot happen.** A cylinder's closing line is square to its axis, so for the upright jar every
  grasp closes level; the solved gripper keeps it within 3.3 deg of level (the approach tolerance is 5: a 4 cm jar
  then reads at most 4.015 cm).
- **The grasp that worked is found, unprompted.** Asked only for the lid (the top 2 cm), from the side to straight
  down: phi 60, psi 45, the fixed jaw on contact 0 solves to pitch -44, heading -129, roll -6; bowl1's was -40, -129,
  0. The pipeline names no pitch, heading or roll.
- **Ranking needs phase B.** Ranked only by how closely the arm meets them (all within a degree or two), the best at
  the lid's middle is pitch -30, the grasp bowl1 turned down because the right wrist camera's plate hid it. The plate check (`view_of`) already sees it: A hidden, B (pitch -59) edge-on, C (the grasp that worked) the first both unhidden and seen (-0.35).
- **The arm's five joints show as rejections.** Of 168 claw targets at the lid's middle, 19 pass; 112 fail because
  the arm cannot turn the gripper to the target (73 about the closing line, 39 about the hinge: the heading follows
  the shoulder, and phi comes every 30 deg), 37 are out of reach.
- **The claw's tip line.** The moving tip's arc tilts the tip-to-tip line 5-19 deg off the pinch -x (8 at 4 cm, 19 at
  1 cm); `to_claw` aligns the real line with the closing line. The roll-based poses ignore it, so they close that
  far off the line intended (a possible part of pen2's nudges: untested).
- **Slow:** 27 ms per solve (two per claw target), 18 s for the lid's 504 targets.

**Phase B progress (2026-11-06).** Built in order: (1) the rejected kept (`GraspCandidates.rejected`: each claw target
with the stage it failed, the reason and its solves; `rejected_by_reason`, `nearest_misses`), and the arm-plane prune;
(2) the camera check and the tiered ranking; (3) the clearance checks: the gripper body and forearm as fitted by
`calibration.arm_shapes` into `urdfs/xle_calib_11062026.urdf` (`api.MODEL` until 11-07's measured boxes), and stand-in fingers
(then capsules; boxes since, below: `finger_boxes`). (4) `Object` and `plan_pick`; (5) pen2 replayed, which led to the redesign below, and
container1's tub; (6) `fit_cylinder`. Phase B is built; its tools are not model-facing yet (`api.py`: phase C).

- **`fit_cylinder`** (`tools.py`, step 6; `test_fit_cylinder.py`, on rendered scenes). The marks' convex hull (a few
  marks around an object, or a line along a thin one) is lifted from depth. `axis="up"`: the vertical side first (the
  points stacked over a range of heights in plan cells: a table or a lid puts one height in a place), then the circle
  with the most of them on it (a search through triples), then a robust fit; the top from anything over its footprint
  (a lid, a cap); a view looking down on the object barely shows its side and is refused. `axis="free"`: from the line
  through the marks' own depths, keeping what is near it. `radius=` holds a known radius (thin objects: their fitted
  radius comes out short). On the sessions' own marks (checked by hand: the captures are gitignored): bowl1's jar,
  white container and chapstick within 0.4 cm of the session's widths and 0.1 cm of its tops, the jar's center 0.4 cm
  from its own; the two looks down from the attack holds refused; pen2's leaning pen, free, 52-55 deg from vertical
  (56), its axis 0.36 cm behind the marked surface; the pen in the book, up with its radius given, its top within
  0.1 cm and its axis 0.3 cm of the session's (free: 8 deg off vertical, and one capture failed: `up` for an object
  standing up).

- **The redesign: the tool measures, the model judges** (2026-11-06, after pen2; `harness.md` 1.K). The tool held
  heuristics that were the model's to choose; now:
  - **What it is told is what may be grasped.** The model describes only the graspable part (the pen above the rim,
    above the book) and, where it matters, from where (`where`: e.g. `{"psi": (0, 90)}`, level or from above). The
    tool knows nothing of the rest of the object or the scene: **no collision checks** (the clearance stage,
    `Surroundings` and the book's `Box` are removed); `protocols.md` rule 1, at the attack hold, is the check.
  - **Generous filters, settable:** `GraspSettings` (printed with every result): the accepted turn about each axis
    (default 45 deg each), the grasp point's miss (0.5 cm), what the camera must see, the margin and the attack
    distance. The solve still *prefers* the exact antipodal pose (`ROTATION_SCALES_DEG`, the closing line kept square
    most); `ArmIK.solve_frame` now separates those weights from the bounds it accepts.
  - **No arm-plane prune** (it was a heuristic tied to the old bounds; the cost is speed: the jar's lid 32 s, pen2's
    cup with both arms 77 s).
  - **The ranking:** the antipodal error (the turn about the approach: the closing line off the object's normal;
    about the hinge: the contacts slid along the approach), in `rank_step_deg` steps (default 5), then the joint
    margin (the closest any joint comes to its bound), then the solve's cost.
  - **Filters kept:** the jaws open wide enough; the arm reaches the attack point and the grasp; the head camera's
    line of sight past both arms' plates and boxes (the stand-in fingers too: they are what catches jaws seen
    edge-on).
- **Replayed under the redesign:** *bowl1's jar lid:* the grasp that worked first (pitch -44, heading -129,
  antipodal within 2 deg, joints >= 50 deg); the one bowl1 turned down rejected for the plate. *pen2's cup* (the pen
  above the rim): 178 pass, the best all left-arm, pitch -26 to -33, heading 148, within 5 deg of antipodal, joints
  >= 45 deg: within about 10 deg of the session's left pick, which clipped the pen's shaft below the rim (what the
  tool no longer knows: rule 1's hold must catch it). *pen2's book* (the pen above it): unrestricted, the best come
  **from below** (pitch +26 to +51, through the book, joints 1-12 deg from their bounds), nearly antipodal: trimming
  the cylinder does not help (the gripper's bulk is behind the jaws, never in the cylinder); with
  `where={"psi": (0, 90)}` the best are the session's own grasp (pitch 1, heading 163, roll 180) and one 13 deg lower,
  joints >= 39 deg (`test_grasp_replay_pen2_smoke.py`). So "from where" is part of what the model must say.
- **container1's tub body** (not a test): 131 of 1560 right-arm targets pass across the 11 cm tub (pitch -30 to -42),
  the jaws open 13.8 cm: near the model's widest (15 cm), which is itself extrapolated (the gripper map was measured to
  `gripper.pos` 85). The session pinched the rim instead, judging the body too wide; whether a fingertip grip across
  a clear 11 cm tub holds is the model's call. TODO: a settable widest opening (`GraspSettings`), and the rim (phase D).
- **The arm's shapes are boxes** (2026-11-06, after bowl2). The gripper body's capsule fitted its box-like servo
  housing poorly, so every part is now an `ArmBox`: square to its link, a center and three sizes. The stand-in fingers
  are boxes too (`finger_boxes`: 1 cm thick (guessed), 1 inch wide, from the tip back to the gripper body's box). A
  calibration with capsules loads without them (a warning). Fitting the boxes to clicked outlines proved fragile:
  clicks on edges inside the outline (the forearm's top ridge) flattened three of four boxes to the size bound, and
  the forearm's soft sleeve, which runs into the upper arm's at the elbow, fits no rigid shape better than ~1 cm. So
  (2026-11-07) `calibration.arm_shapes` takes the sizes from a ruler and checks the box drawn over the head camera's
  frame, shifted by hand until it sits on the part. TODO: the plates as rotated `ArmBox`es, so one shape is left.
- *The entries below describe the design before the redesign (kept for its findings).*

- **The stand-in fingers:** computed, not clicked: each from its tip (where the calibration puts it) back along the
  gripper axis to the gripper body's capsule (7.1 cm on both arms), its inner face on the contact line, the moving
  one turned by the jaws' opening; radius 1.27 cm, half of "about 1 inch across at the widest" (an estimate, not
  measured: generous, a finger is thinner than it is wide). They may come 3 mm from the table (to take something
  lying on it), 1 cm from neighbours. TODO: measure the fingers if a check turns on them; keep contacts half a finger
  in from an object's ends (`t` still runs to the jar's rim).

- **The prune:** a claw target whose gripper axis is more than 25 deg out of the arm's vertical plane (through the
  shoulder pan axis and the target) is not solved: passes reach 18 deg (the jar and a lying pen, both arms), and it
  skips about half the failing solves (225 of 504 at the jar's lid; 18 s to 11.5 s).
- **Filters, then one ranking** (2026-11-06, replacing the visibility tiers: visibility was in three places, a plate
  filter, an edge-on tier and a "seen squarely" tier). Filters, cheapest first where one does not need another's
  result (`STAGES`): the jaws open wide enough; the arm plane; the attack point solves; **attack view**: the head
  camera's line of sight to the grasp's center and both contacts passes no arm surface (both arms' plates, 5 mm
  margin, and capsules, the stand-in fingers too: `sight_blockers`); the grasp solves; **grasp view**: the same for
  the center; **clearance**. Jaws seen edge-on hide the center behind a finger, so they fail a view filter. Then
  **one ranking:** how far the joints stay from their bounds (the IK's cost breaks ties).
- **Timed** (`GraspCandidates.timing`, bowl1's jar, 504 claw targets): the attack point's solve 34 ms each (279 run,
  9.6 s: nearly all of it), the grasp's 17 ms (39), attack view 2.0 ms (66), grasp view 1.8 ms (39), clearance
  6.9 ms (14), the arm plane 0.2 ms (504). The view at the attack point runs before the grasp's solve and spares it
  for those it rejects.
- **bowl1's jar, replayed:** the best is the grasp that worked (pitch -44, heading -129, every joint >= 50 deg from
  its bound); the grasp bowl1 turned down (pitch -30, the right plate in the way) is rejected at the attack view for
  the plate; the low, face-on grasps the visibility tiers had put first (pitch -16, joints >= 18 deg) rank below.
  The clearance checks reject nothing here (the chapstick and the white container were already in the bowl behind
  the jar when it was picked; an earlier note here had them beside it): the jar does not test them.
- **Planned and executable:** bowl1's best, through `plan_pick`, runs through the harness's own
  `prepare_execution` (dense straight steps re-solved by the old solver, no joint jumps, the gripper commands): the
  bridge to phase C. Its attack point and grasp are within 1 cm of bowl1's executed plan
  (`test_grasp_replay_smoke.py`).
- **pen2 replayed** (step 5, 2026-11-06; `test_grasp_replay_pen2_smoke.py`). *The book:* the left arm's first grasp
  (pitch 0, heading 163, roll 180, the pen above the book) is found as a candidate; the same pose lower, the fingers
  over the book's edge, is rejected for the book (what the session's deeper re-grasp hit); the best take the pen above
  the edge, clear of it. The book is a `Box` (only something to clear: its grasps are not built). *The cup* (not a
  test yet): nothing passes for either arm. The right arm's session poses (pitch -40 / roll +40 edge-on, pitch -70 /
  roll -100 behind the forearm) are not even reached: they closed at 75-79 deg to the pen, and the cylinder's family
  closes square to it with 5 deg allowed about the approach; at 20 deg the right arm finds 8 grasps, every one with a
  joint at its bound (the corner pen2 was in), and the session's poses fail about the closing line instead. The left
  arm's poses there are rejected for the cup (the gripper body over its rim: modelled solid). Open: the tolerance about
  the approach (TODO below).
- **The sight margin is on the plates only.** With 5 mm on the capsules too, the gripper body "hid" the center of the
  grasp that worked; a capsule's radius was fitted to the outline as seen.

Out of scope for now: grasps sampled from raw point clouds (the interfaces allow it), multi-part objects beyond a part
plus the whole's extent (a pan: a handle cylinder plus its body), two-arm grasps.

## 2. Interfaces and responsibilities

New modules: `harness/primitives.py` (shapes and their grasp families), `harness/grasping.py` (claw mapping, checks,
scores, ranking, planning). Model-facing tools go in `api.py` like the others.

**Frames and names.** Every name holding geometry says its frame, in the new code (the old pose code is left as it is:
section 4 removes most of it).

- The frames: `root` (the URDF root, on the floor: x forward of the robot's back, the robot faces -x, y its right, z
  up; what the rest of the harness calls the root frame), `obj` (a primitive's own frame: a cylinder's z is its axis),
  `pinch` (the gripper: origin at the fixed jaw's tip, z toward the fingertips, y the hinge), `cam` (the head
  camera's optical frame).
- Points and directions end in their frame: `center_root`, `approach_root`, `aim_pinch`.
- Transforms are named `<to>_from_<from>`: `root_from_pinch` (4x4) maps pinch-frame coordinates to root-frame ones,
  so `root_from_obj @ obj_from_pinch = root_from_pinch` reads by cancelling. Rotations alone: `rot_root_from_pinch`
  (3x3). (Not the Drake-style `X_root_pinch`: ruff's pep8-naming refuses capitals in variables and arguments.)
- Scalars are frame-free: lengths in m, angles in deg unless they end in `_rad`.
- What the model reads (`describe()`) gives coordinates in the root frame and says so; the model never gets another
  frame's numbers.

**Perception (model-facing tools):**

| interface | responsibility |
|---|---|
| `fit_cylinder(view, uv, axis="up" \| "free", label) -> Object` | a cylinder from depth at the pixels the model marks on it (or along it): axis, radius, extent along the axis; its residual and point count in the description. `axis="up"` holds the axis vertical (an upright object on the table), `"free"` fits it (a pen, a handle). |
| `fit_rim(view, uv, label) -> Object` (phase D) | a circle in a horizontal plane (container1's fit, as a tool), with an assumed wall thickness. |
| `Object` | a named object: `parts` (name -> primitive), the whole's extent (for clearance while carried), a rough center of mass, the capture it came from (freshness, `harness.md` 1.C). Drawn by `render`. **Built** (`objects.Object(label, parts, capture)`: parts and capture; not yet the extent or center of mass, for carrying: phase D). `grasp_candidates` takes one (and its `part`); `Surroundings.of(table_z, objects)` makes the rest what the gripper must clear. |

**Primitives (internal):**

| interface | responsibility |
|---|---|
| `Primitive` (protocol) | a shape with a pose in the root frame: `root_from_obj` (4x4), `surface_distance(points)` (signed, for clearance), `antipodal_grasps(where, spacing, max_opening) -> list[AntipodalGrasp]`, `describe()`. `where` and `spacing` are keyed by the family's parameter names (`where={"t": (0.059, 0.079), "psi": (0, 40)}`: ranges, ends on the grid). Pure geometry: knows nothing about the robot. **Built** (`primitives.py`). |
| `Cylinder(root_from_obj, radius, length)`, `Cylinder.upright(bottom_root, height, radius)`, `Cylinder.between(bottom_root, top_root, radius)` | the axis is the object frame's z from the bottom end. The constructors are first guesses at what the model finds natural (section 5). Family: `t` (m from the bottom end), `phi` (deg, the side the jaws come from, around the axis from the frame's x: for an upright cylinder 0 is from the robot's side), `psi` (deg, the approach's tilt down the axis: 0 square to it, 90 from the top end); closing square to the axis and the side, through the axis, so the opening is the diameter. **Built.** |
| `Rim(frame, radius, wall)` (phase D) | family: the angle around the rim; closing line through the wall, radial. |
| `AntipodalGrasp` | in the root frame (what the claw mapping needs): the contacts' center, the closing direction (contact 1 to 2), the approach direction, the opening; the family parameters keep it in the object's terms (for describing it: "from the robot's side, 30 deg down"). **Built.** |

**The claw and the robot (internal):**

| interface | responsibility |
|---|---|
| `to_claw(grasp, model, arm, fixed_side, margin, attack) -> ClawTarget` | `root_from_pinch` putting the fixed tip on contact `fixed_side` (`margin` clear of it) and the moving tip, at the grasp's opening, on the other: the real tip-to-tip line (it tilts 5-19 deg off the pinch -x with the opening, which the roll-based poses ignore) along the closing line, the hinge square to the closing and the approach; `root_from_attack`, `attack` back along the approach; how far apart the tips open and close (`OPEN_CLEARANCE_M`, `SQUEEZE_M`; `prepare_execution` turns them into `gripper.pos`). Uses `XleRobotModel.grasp_offset`, which it agrees with. **Built.** Not yet: `depth` (contacts deeper than the tips: the claw's opening for a width changes there), the close as an overshoot past contact (`tuning_notes.md` 3.9). |
| `ArmIK` (kept, from `reach.py`) | `solve_frame(root_from_pinch, axes_root, rotation_tols_deg)`: the joints for a whole target pinch frame, the rotation's error split about three axes, each weighted by its tolerance; still one local least-squares solve. `grasping` uses the grasp's closing line 20 deg (keeps the contacts: the missing sixth joint gives way here), approach 5 (rolls the closing line off the object), hinge 10. **Built.** |
| `Check` / `Score` | **Redesigned** (phase B progress): filters `STAGES` (the jaws, reach, the head camera's line of sight past the arms' plates and boxes) with generous, settable bounds (`GraspSettings`); the score is the antipodal error, then the joint margin. No collision checks: the model describes only the graspable part, and from where. |
| `Scene` | **Not built, by design** (phase B redesign): the tool is told only what may be grasped and from where; the scene is the model's to judge, and rule 1's hold checks it. (`Surroundings`, built and removed, 2026-11-06.) |

**Planning (model-facing tools):**

| interface | responsibility |
|---|---|
| `grasp_candidates(obj, part=None, arm=None, where=None, n=5) -> GraspCandidates` | **Built, thin** (`grasp_candidates(primitive, model, capture, arms, where, spacing, margin, attack, n)`; the checks and tiers of `Check` / `Score`; every rejected target kept, with its stage and reason; `Object` and `render` not yet). generate -> claw -> IK -> checks -> ranked; `arm=None` tries both. `where` narrows the family in object terms ("top 2 cm", "from above"). Returns the best `n` named A, B, ... plus a count of the rejected by reason. Drawn by `render` (each candidate's jaws, colored by rank). |
| `GraspCandidate` | one ranked grasp: the `ClawTarget`, the solved `Poses` for attack and grasp, every check and score, `describe()`: "A: right arm, from the robot's side 40 deg down, jaws level; seen (-0.40); move in clear >= 0.8 cm; 7.6 cm from `white`; reaches". |
| `plan_pick(candidate, lift=0.03) -> Poses` | stage, attack point, grasp, lift, with the grip actions, ready for `prepare_execution` (kept). **Built** (`plan_pick(model, candidate, capture, stage_up=0.06, lift=0.03)`): "stage" 6 cm above the attack point, "pick.back", "pick", "lift", as the harness's `Poses` (the new solver's frames and joints, its achieved angles as targets), so `prepare_execution` runs it unchanged with `PICK_GRIP`; rule 1 stops at `select_poses(pick, "stage", "pick.back")`. |
| `refine(candidate, capture, error=None) -> GraspCandidates` (phase C) | at the attack hold: the same family member re-solved from the new capture, the object re-fitted, aimed to cancel `arm_error` (kept), the move in re-checked. |
| `plan_place(pick, onto, pose="upright", at=None) -> Poses` (phase D) | carry and place: the object's pose in the jaws is known from the grasp, so the gripper pose comes from where the object should end; lift height from the object's extent over what is between; release height above what is below. |

Kept as they are: `Capture`, `View`, `Grid`, `Points`, `TableEstimate`, `Focus`, the `Workspace`, `render`,
`contact_sheet`, `scan`, `look_at`, `prepare_execution`, `execute`, `arm_path`, `arm_error`, `home`, `history`.

## 3. Grasp solving flow

bowl1's jar, as the model calls it now (`harness.api`, phase C; the tool's work in comments; numbers from the
replays):

```python
# perception: the model marks the part it may grasp in a view (a few marks around it, or a line along a thin one)
jar = fit_cylinder(view, uv=[[214, 250], [262, 252], [238, 300], ...], axis="up", label="jar")
#   -> jar: body: cylinder, radius 2.0 cm, from (-34.3, -11.1, 72.5) cm, 6.7 cm along the vertical
#      (fitted to 2641 of 7835 points where marked, residual 0.16 cm, the side seen over 180 deg of it)

# candidates: generate -> claw -> filters (jaws, reach, the head camera's line of sight) -> rank
cands = grasp_candidates(jar, capture, arms=["right"], where={"psi": (0, 90)})  # level to from above
#   -> settings: accepted: turns up to (closing line 45, approach 45, hinge 45) deg, ...
#      A: right arm, ... gripper at pitch -44, heading -129, roll -6; antipodal error 2 deg ...; joints >= 50 deg
#      ... rejected: ... attack point: the plate hides the grasp from the head camera ...
render(view, jar, cands)                 # the model looks, picks A

# pick: stage and attack, hold
pick = plan_pick(cands, "A")             # stage, pick.back, pick, lift
execute(prepare_execution(select_poses(pick, "stage", "pick.back"), {"start": "open"}), hold=True)

# rule 1 at the hold (protocols.md)
look = look_at(pick.points[2])                                  # the grasp, centered (head only)
jar2 = fit_cylinder(full_view(look), uv=[...], axis="up", label="jar")   # where it is now
err = arm_error(pick, look, "pick.back")                        # where the arm really is
a2 = refine(cands, "A", jar2, look, err)  # the same grasp on jar2, from the arm's pose, aimed to cancel err
pick2 = plan_pick(a2, "A")
execute(prepare_execution(select_poses(pick2, "pick.back", "pick", "lift"), {"pick": "close"}), hold=True)

# carry and place: phase D (plan_place)
```

How the pieces call each other inside `grasp_candidates`:

```
grasp_candidates(obj, where, arm)
  for part in obj.parts:                      Primitive.antipodal_grasps(where, spacing)   [geometry]
    for g in grasps, side in (fixed on contact 1, fixed on contact 2):
      t = to_claw(g, model, arm, side, depth, margin)                                     [claw]
      s = ArmIK(model, arm, ...).solve(t.frame, init=rest or the last good)               [robot]
      checks = [c(t, s, scene) for c in CHECKS]      -> drop on the first failure, keep the reason
      scores = [f(t, s, scene) for f in SCORES]
  rank by scores; describe the best n; count the rejected by reason
```

## 4. What it replaces

If the experiment proves out (phase E), these go. Until then they stay and keep working.

**Model-facing tools (`harness/api.py`):**

Marked deprecated (2026-11-06: `@tool(..., deprecated=...)`: `describe_tools` shows the model what replaces each,
and a call to one is logged as deprecated). `solve_pose`, `join_poses` and `held_object` still plan a carry and a
place until `plan_place` (phase D).


- `solve_pose`: replaced by `grasp_candidates` and `plan_pick` / `plan_place`. The model no longer names a pitch,
  heading and roll.
- `approach`: the attack point and the lift come from `plan_pick`.
- `held_object` and `HeldObject` (`harness/objects.py`): `plan_place` knows the object's pose in the jaws.
- `join_poses`: paths come whole from `plan_pick` / `plan_place` (`select_poses` stays, for staging).

**Code (`harness/`):**

- `reach.py`: `solve_pose`, `approach`, `solve_path` as public functions; `OPEN_CLEARANCE_M`, `SQUEEZE_M` (the open
  and close move to `to_claw`, the close as an overshoot). `ArmIK` stays (internal, with a full-rotation target);
  `join_poses`, `select_poses` move with `Poses`.
- `objects.py`: `HeldObject`; from `Poses`, the grasp fields (`width`, `grasp`, `open_jaw`, `held`, `margin`) move to
  `ClawTarget`, and the camera fields (`visibility`, `hidden_by`, `seen`, `hidden`, `camera_text`,
  `EDGE_ON_VISIBILITY`) to the scores. `orientation` and `angles_of` stay internal (for describing a grasp).
- `execute.py`: `held_object`.
- `calibration/xle_robot_model.py`: `grasp_offset` (width and margin) moves into `to_claw`; `moving_jaw`,
  `opening_for_width`, `jaw_radius`, `visibility`, `hidden_by`, `view_of` stay.
- `scratch/session.py` (gitignored): `propose` (waypoints in cm at a pitch); `execute` keeps working on `Poses`.


**Removed with the wrist-camera redesign** (2026-11-07, section 6): the code for the head camera's line of sight to a
grasp, which the wrist cameras replace. bowl3 and bowl4 turned the view filters off at every hold.

- *The arm's shapes:* `calibration/arm_shapes.py` and its tests; in `calibration/xle_robot_model.py` the parts' boxes
  (`ArmBox`, `with_boxes`, `boxes_root`, ...) and the wrist cameras' plates (`WristBoard`, `with_wrist_board`,
  `board_corners`, `segment_hits_box`, ...); their outlines in `calibration/check.py`. An older calibration loads
  without them (`OLD_SHAPE_KEYS`, its plate links taken out of the URDF, a warning).
- *The head camera's view of the grasp:* in `harness/grasping.py` the stages "attack view" and "grasp view",
  `sight_blockers`, the stand-in fingers, `GraspSettings.see_*`, `refine`'s view check; in the model `hidden_by` and
  `view_of`; in `reach.py` / `objects.py` the poses' `visibility` / `hidden_by` and their "camera: jaws seen /
  edge-on / hidden" reports. The model's `visibility` (the hinge axis against the line of sight) stays: the click
  calibration's views use it to face the pinch point to the head camera.
- *Tests:* `test_arm_model_shapes.py`; the visibility checks in the replays (`test_grasp_replay_smoke.py`,
  `test_grasp_replay_pen2_smoke.py`: its first left grasp no longer ranks among the best 10 without the view
  filters, only among the level ones).

*Also changed or replaced:*

- `plan_pick`'s attack point back along the grasp's approach: replaced by one back along the gripper's own axis (the
  move in went crabwise, bowl4).
- `arm_error` (the joints through the model) as the check at the hold: replaced by the wrist camera's measurement of
  the object in the jaws' frame; `arm_error` stays as a diagnostic.
- `protocols.md` rule 1, steps 1-3 (look with the head, re-locate, measure the arm): the alignment loop (section 6).
- The smoke replays' checks of visibility (`test_grasp_replay_smoke.py::test_the_hidden_grasp_is_rejected_with_its_reason`,
  `test_the_best_is_seen`, `test_the_view_filters_can_be_turned_off`).

**Tests (`tests/felix_jr/`):**

- `test_grasp.py`: rewritten for `to_claw` and the candidates (the claw geometry it checks stays true).
- `test_harness.py`: the `solve_pose` / `approach` / `held_object` cases.
- `smoke_tests/test_ik_smoke.py`: rebuilt on `grasp_candidates` over pen1's and bowl1's recorded scenes.

**Docs:**

- `harness.md` 1.J ("Reach: one constrained optimization per pose"; the search is the model's) is replaced by a new
  1.L (object-centric grasping); 1.K stays.
- `open_issues.md` 18 (a rolled grasp on a vertical object) closes; 17's "use the camera report in a session" and the
  forearm become scores; 8 ("only the pitch and jaws are chosen, not the approach direction") is reframed.
- `sessions.md` pen2's proposed rules 3 (the attack point clear of the object) and 4 (a staging pose) become checks and
  `plan_pick`; rule 1's swept move in becomes a check (`protocols.md` rule 1 keeps the hold and the look).

**Not in the repo, also replaced:** the per-session snippets of 2026-11-05 (the pitch x roll sweeps, the closing-angle
and occlusion scores, the clearance and swept move-in checks) were the prototype of stages 2 and 3.

## 5. Open questions (TODO)

- **Constructors for the model.** `Cylinder.upright(bottom_root, height, radius)` and `Cylinder.between(bottom_root,
  top_root, radius)` are guesses at what a VLM states most naturally and most accurately. To try in sessions: from
  extreme points (the ends of a pen, the top and bottom of a jar's side as two or three points marked in a view),
  from marked pixels plus depth (`fit_cylinder`), from a center, an axis direction and a size. Measure which one the
  model gets right most often against the hand fits, and keep that one (or two) as the model-facing form.
- **Speed.** 27 ms per solve with a finite-difference Jacobian; the model has the analytic one
  (`XleRobotModel.frame_pose_jacobian`), unused by both `ArmIK.solve` and `solve_frame` (`jac="2-point"`). Also
  cheaper: solve once per (phi, psi) and shift along `t`. (The arm-plane prune was removed in the redesign: it
  skipped half the solves, but was a heuristic tied to the bounds.)
- **Families against the arm's five joints.** The heading follows the shoulder, so a 30 deg `phi` grid leaves most
  sides unreachable, each a failed solve. A cylinder is symmetric about its axis, so `phi` could instead be
  solved for (set by the heading the arm reaches); whether a primitive should declare its symmetries for that is open.
- **How far a grip may tilt is the model's call** (settled by the redesign): the width a tilted closing line needs
  (diameter / cos(a)) and the friction cone (tan(a) < mu) depend on the object; the default accepts 45 deg and ranks
  the most antipodal first, and the model narrows `GraspSettings.rotation_bounds_deg` for a slippery one.
- **Ends that are not free.** "From level or above" (pen2's book) could be said as a fact about the object instead:
  a cylinder's end that is buried or attached, ruling out approaches from it. Try if the model finds `where` awkward.
- **Joints at their bounds rank high when antipodal.** On the jar, grasps with a joint at its bound (pitch 0) are in
  the top 8, after the ones that worked, inside the first 5 deg step. If they reach the top in a session, a bound on the
  joint margin (in `GraspSettings`) is the setting to add.
- **Ranking beyond antipodal and the joint margin.** To add when a session shows the need: the arm's expected sag
  (it grows with a level reach: `open_issues.md` 16), depth in the jaws.
- **The camera's pose.** The checks use the capture's head pose; rule 1 looks from the `look_at` pose at the attack
  point, which sees the grasp from a little elsewhere. Check from that pose instead.
- **Contacts at an edge.** `t` runs to the cylinder's very ends (the jar's top rim); the fingers have width, so a
  grasp needs its contacts at least half a finger in from an end (a check, with the finger's size from the model).
- **Frames in model-facing calls.** Whether the model ever needs to give a direction in the object's or the gripper's
  terms (e.g. "from the robot's side"), and how to name it so the frame is obvious, is open until the model uses
  `grasp_candidates`.

## 6. Next: coarse to fine with the wrist cameras (2026-11-07)

**Why.** bowl3 and bowl4 ([sessions.md](sessions.md)) put the error budget at 1-3 cm: the arm's sag (0.7-1 cm), the
calibration under the arm, and a camera pose that depends on the head's (about 2 cm off at the head's home). The
3D-printed parts' slop adds to it. A narrow object leaves about 1 cm of clearance, so planning from the head camera
alone cannot be made accurate enough. Neither can a re-calibration alone (worth doing: check `calibration.check` at
the home head pose first, then collect with views from it). And the head camera, ~60 cm away and behind the
gripper, could not tell whether the jaws were lined up, with or without the visibility filters (bowl4's last two
chapstick attempts: the person saw what the camera did not). So: accept a rough first move, and correct it close up
with a camera that sees the object in the jaws' own frame. That measurement does not go through the URDF, the head
or the arm's model at all.

**The plan:**

1. **Rough, from the head camera:** find the object and fit it (`fit_cylinder`), choose a grasp, go to an attack
   point a few cm back along the gripper's own axis. The head camera's job ends here (and at the place).
2. **Fine, from the wrist camera, at the hold:** find the object in the wrist image and measure it in the jaws' frame:
   how far it sits across the jaws (off the midpoint between the tips), how far ahead, and the turn about the hinge
   (pinch y) that would square the closing line on it. Correct with small joint moves, capture again, repeat until
   it is centered within tolerance.
3. **Straight in along the gripper's axis, close, check the grip** by the jaws' position (stopped near the object's
   width: held; closed fully: missed) and by the wrist image.

**Choosing grasps for it (replacing the head camera's visibility filters):**

- **(a) TODO, an experiment: the object in the wrist camera's view at the attack point** (inside its field of view,
  at a distance it can focus and measure, not behind the fingers). No visibility filter of any kind for now
  (2026-11-08): first see on the robot whether grasps lose the object from the wrist camera's view at all.
- **(b) Corrections that do not fight each other:** prefer grasps whose closing line is square to the arm's vertical
  plane (through the shoulder pan axis and the object). Then the joints split the corrections: the shoulder pan
  moves the gripper across the jaws and, at a level grasp, turns it about the hinge (pinch y), both of which a side
  grasp needs; the shoulder lift, elbow and wrist flex move it in and out and up and down; the wrist roll turns the
  closing line. A preference in the ranking with a tolerance (tens of degrees), not a filter: an upright cylinder
  loses nothing (any closing direction around it works); for an object that sets its closing direction (a lying
  pen, a box) a level side grasp only fits when it lies along the arm's reach, while from above the wrist roll
  turns the closing line anywhere. Otherwise: the other arm, or nudge the object first.
- Kept: reach, the jaws' opening, the joint margin, the antipodal error (with a step: a few degrees off on a round
  object do not matter), and the move in straight along the gripper's axis.

**To build, in order:**

1. **The wrist cameras on the robot** (`open_issues.md` 15): mount them; the server streams them; their
   intrinsics; their pose in the gripper link (the extrinsics: replaces the plate's corner fit; the plate itself need
   not be modeled once nothing checks the head camera's line of sight past it). **Built 2026-11-07, not yet run on
   the robot:** `server --left-wrist-camera / --right-wrist-camera`, `calibration.wrist_camera` (README 8b).
2. **A wrist capture in the harness:** built 2026-11-08: `wrist_look` (a `WristView`) and `render_wrist`.
3. **The measurement:** built 2026-11-08: `plan_alignment`, from the object's two sides marked in the wrist view
   (below, "The alignment loop").
4. **The alignment step:** built 2026-11-08: `align_step` runs a `plan_alignment` (one approved move through the
   model, `ArmIK.solve_frame`, rather than the model composing joint moves). Run on the robot in bowl5/bowl6.
5. **Grasp selection:** (b) in `grasp_candidates` ((a) waits on the experiment), the head camera's view filters
   removed (section 4, done 2026-11-07); the
   attack point along the gripper's axis in `plan_pick`.
6. **A session** (a bowl4-like scene: an upright bottle and a chapstick into a container), every grasp aligned at the
   hold from the wrist camera.

**The alignment loop** (`harness/wrist.py`; smoke test `test_wrist_align_smoke.py`, against the fake robot with its
joints 1-2 deg off the truth). Revised after its first session on the robot (bowl5/bowl6, 2026-11-08: centered to
1-4 mm across the jaws, the bottle not picked; [sessions.md](sessions.md)):

1. At the attack point (`execute(..., hold=True)` of the pick plan up to "pick.back"): `wrist_look`, then
   `render_wrist(view, pick)`. It draws the fingertips (check first that they sit on the real ones: the right
   camera's calibration was 7.8 deg off, fitted again from a gripper sweep), the **approach line** (where the
   object's center appears when it is lined up, a dot every 2 cm ahead of the aim point), and the **closing line**
   (the fixed tip to where the moving tip touches an object the plan's width, its sweep, and where the object's
   center is once held: "held").
2. Mark the object's two sides at one height (and, from a camera looking down at an upright object, its top's
   center and rim: `midline_top`, `midline_sides` draw its middle line); `plan_alignment(..., object_axis=(0, 0, 1))`
   measures its offset from the jaws' line, toward the moving jaw, square to that and how far ahead, after sliding the
   measured center along the object's axis to the jaws' line (not along the hinge: with the approach pitched, the
   hinge is not the axis, and the height of the marks then shifts "ahead"; bowl6 stopped 2 cm short on it).
3. Steps across (80% of the correction each: `STEP_FACTOR`), then in along the approach in parts (`advance`, or a
   `nudge_m` along it), looking after each; `roll_deg` turns the jaws. A step's joints left more than 0.5 deg short
   are commanded again (the elbow stopped 1.2-1.4 deg short under load on every step). Aim against gravity
   (`nudge_m` up; `GraspSettings.sag_lift` in the plan): too high grips higher on the object, too low hits its
   support.
4. **Go / no-go before closing, in the picture, rigid to the gripper:** "held" on the object's middle (an upright
   object seen from above: on its top's center), the closing line's ends on the object's two sides. Then close, and
   judge the grip by a small lift: the held object keeps its size in the wrist picture. The jaws' reading cannot tell
   a grip across the middle from a chord near the front (0.4 of `gripper.pos`).

**Depth is the hard part.** The camera is ~7 cm off the jaws' line on the robot, and the ray to the object crosses
the line at an angle, so a depth off by a fraction e puts the object off the line by e times that offset, at any
range. What bowl6 found about each source:

- **The object's width** (the fitted primitive's): good when the fit is (3.6 cm, right). The default.
- **Triangulation over the arm's moves** (`triangulate_wrist`): off by about (the baseline's error) / (the baseline),
  and the baseline comes from the joints read through the model, without the arm's backlash, flex and changing sag:
  10-20% on the 1.5-3 cm moves tried, as poor as the width or worse; the step's gain (the fraction of the predicted
  shift that happened: 42-157% on the robot, 99% in the simulation) is the tell. A cross-check only, on a baseline of
  4-5 cm or more, across the object (a sidestep across the jaws: the sides pin the correspondence; a move along an
  upright object's axis needs a point that marks the same height twice, like a label's corner, then `to_axis_m`).
- **The head camera:** blind to the jaws behind the arm.
- **The checks rigid to the gripper** (the closing line, "held" on the midline) need no depth and were right when the
  measurements were not; from a camera looking nearly straight down, nearer and lower look alike in them, so read
  them with care.
- Not solved: depth at the grasp. A depth camera on the wrist, or a head view from the side ([open_issues.md](open_issues.md) 15).

In the smoke test (exact marks, fixed joint errors only), the object 25% wider than the plan says: after a sidestep,
26 -> 5.1 -> 0.3 mm off the line; with the width right, 9.4 -> 1.3 -> 0.5 -> 0.3 mm. The robot's steps were far less
exact than the simulation's.

### 6.1 Next: active sensing with the wrist camera (proposal, 2026-11-08, after bowl6)

bowl6 aligned by hand-sized steps, one approved move and one look at a time (a dozen to get in), with depth that was
either as poor as the arm's motion or read by the person. The proposal: let the wrist camera *move* to sense, in
sweeps a tool runs, and let the model judge pictures and choose rather than step.

**The flow:**

1. **Fit primitives to the objects** from the head camera (`fit_cylinder`), as now.
2. **Choose a grasp** from the head (`grasp_candidates`, `plan_pick`), as now: aimed high against the sag
   (`GraspSettings.sag_lift`).
3. **Move to the attack point** and hold (`execute(..., hold=True)` up to "pick.back").
4. **Structure from motion** (`wrist_scan`, to build): the object fitted again, in the jaws' own frame.
5. **The revised attack:** the grasp solved again on the wrist-fitted object, a relative move from where the arm is.
6. **The approach** (`approach_sweep`, to build): in along the gripper's axis to the step the closing-line check
   passes at.
7. **The grasp** (`grasp`, to build): close to contact, squeeze, and test the grip by a small lift.

**4. `wrist_scan` (structure from motion).**

- *The sweep:* only the wrist moves: roll +-30 deg in 15 deg steps and a couple of wrist-flex steps, ~8 frames, each
  with its joints read, then back to the start (checked by comparing a fresh frame with the first). The camera is
  ~7 cm off the gripper's axis, so a roll swings it through a 4-7 cm arc: a baseline 2-4 times bowl6's, by the two
  joints that carry the least load and leave the arm above them still (bowl6's 42-157% step gains were shoulder and
  elbow moves).
- *The camera's real motion, from the images:* features on the table (its grain) matched across the frames, and a
  small bundle adjustment: the camera poses (started from the model's) and the table's points, refined to the
  matches; the scale from the wrist moves' length (or the table's height from the head camera). The arm's motion
  error, which sank bowl6's triangulation, drops out. A by-product: **the table plane in the jaws' frame**, how high
  the jaws are over it (bowl6's open question at the hold).
- *The object, fitted again:* the model marks it in one or two frames (its two sides, or a box); the tool finds its
  silhouette's edges in every frame and fits the same primitive as the head's (an upright cylinder, its radius the
  head's, or free) to all of them, from the refined poses: the object relative to the jaws now, with residuals.
- *Returned:* a contact sheet with the fitted outline drawn on every frame (it should hug the object in all) and the
  table plane; the object in the jaws' frame, the jaws' height over the table, the residuals; and the grasp solved
  again on it (step 5). Not depth images: dense stereo has little to work with on a white bottle.
- *The model's part:* judge the overlays (does the outline hug the object, is the plane the table), approve or choose
  the revised attack, mark more where the edges failed.

**6. `approach_sweep` (the grasp depth by choosing a frame).** In along the gripper's axis in ~5 mm steps from the
(revised) attack point, a wrist frame at each with the joints read, the closing line and "held" drawn on it; the far
end bounded short of contact (the depth from the width plus a margin; stop early if the object's apparent width
passes the closing line's ends). One contact sheet back; the model picks the step where "held" is on the object's
middle and the closing line's ends on its sides; the tool returns to that step's recorded joints. That depends on the
arm repeating a pose (~0.5 deg with the push-again), not on the model or a depth. A by-product: the object's growth
over known steps is a depth by looming, independent of its width.

**7. `grasp` (close by contact, then squeeze).** The gripper is position controlled: step its target closed and watch
the reading; contact is where the reading stops following the target, the realized width. Then the target the model
asks past contact (the squeeze: it sets the force; bowl6's 1.3 of `gripper.pos` was too weak, open issue 14), and a
report: the contact width against the object's (a contact much narrower than the object warns of a chord near its
front, though 4 mm off the middle is only 0.4 of `gripper.pos`). Last, the lift test: up 1-2 cm, the object's size in
the wrist picture before and after (a held object keeps it).

**Protocol: aim against gravity.** Every plan aims `sag_lift` (1 cm) high, the person's rule from bowl6. Toward a
model: log at every hold the pose, the reach, the load and the measured error (`arm_error`, or the wrist scan's
table plane), then fit the sag to the horizontal reach and the load. The joints' own shortfall is already corrected
(`align_step` pushes again); what is left is link flex and gear play, which reach and load should predict.

**Build order:** a grip-force bench test first (the bottle between the jaws by hand, closed to 17, 13 and 10, lifted:
which squeeze holds); then `grasp` and `approach_sweep` (bowl6's failures directly); then `wrist_scan`, the wrist-only
sweep and the object's refit first, the feature-based motion second if the wrist joints' motion is not exact enough.

**Open questions (for 6.1):** the wrist's repeatability on return (backlash); whether the table's grain matches across
a 30 deg roll; how well silhouette edges come out on a white object against a light table; how many frames a contact
sheet can show the model legibly.

### 6.2 Structure from motion: `sfm_light`, `sfm_torch` (built 2026-11-08, not yet on a sweep)

`harness/sfm.py`; unit tests `test_sfm.py` (a textured table rendered exactly from a wrist-roll sweep, the "model"
poses off by 3 mm and 1 deg per frame). Not COLMAP/GLOMAP: few frames, a calibrated lens, every pose known to a few mm,
so a small bundle adjustment from the model's poses with a prior on the corrections (`refine`): features, matches per
pair, an essential matrix per pair (RANSAC), tracks, triangulation, the adjustment (SciPy, sparse), the table plane.
`sfm_light` matches with OpenCV's SIFT; `sfm_torch` with DISK and LightGlue (kornia, in the `felix_jr` extra).

What it does, and does not:

- **It fixes the turns and the directions of the moves; one scale stays the model's.** Images fix a moving camera's
  motion up to one scale factor, here the prior's (the model's translations averaged over the frames). Synthetic
  sweep: the moves' directions to within 0.5-1 deg (7 deg before), the turns to 0.1-0.3 deg (0.5), every length off
  by the same ~3% (and the table's distance with it). 3% of 12 cm is ~4 mm (bowl6's triangulation: 10-20%); the
  object's known width or the table's height (head camera) could set the scale.
- **DISK and LightGlue are not rotation invariant**, and a wrist roll turns the picture: pairs 45 deg or more apart
  did not match at all. The frames are turned back by the model's roll before the features (all pairs then match).
- **On the robot's frames the learned features win by far.** bowl6's held frames (w17-w19, ~1 cm moves): SIFT kept
  22-26 inliers per pair (32 points), DISK+LightGlue 747-883 (800 points); both put the arm's real motion 2-4 mm and
  0.2-0.8 deg off the model's per move. No table plane from moves that small (the points too noisy): a sweep's
  4-7 cm baselines are for that.
- **The jaws are in every frame**, rigid to the camera: matches that did not move between two frames (`STATIC_PX`)
  are dropped before the essential matrix.
- Not yet: the scale from the object's width, a learned server (below).

**The wrist scan (`harness/wrist_scan.py`, built 2026-11-08; tests `test_wrist_scan.py`, smoke
`test_wrist_scan_smoke.py`).** `wrist_scan(arm)` (moves the wrist only: roll -30..30, flex +-10, and back; the arm
holds) or `scan_views(views)` (frames already taken) give a `WristScan`: the frames, their refined poses, and the
frame the arm is at now. The model works on it like a head capture:

- `render_scan(scan, frames=, points=, objects=)`: a contact sheet of the frames (or one, full size) with points and
  fitted outlines drawn in each; what sits right in every frame is right.
- `scan_locate(scan, frame, uv)`: a pixel's 3D point from the scene's points around it (the table, textured
  surfaces) or the table plane; with a second mark (`frame_b`, `uv_b`), triangulated (an object's feature: a label's
  corner, a cap's center). `scan_find`: the second mark found by the tool along the epipolar line (refused on a
  smooth or repeating surface).
- `scan_fit_cylinder(scan, sides={frame: [[u, v], [u, v]]})`: an upright object from its sides in a few frames (each
  a ray tangent to it), on the table plane: an `Object` for `grasp_candidates` (with a fresh `capture` for the arm).
- Answers are in the jaws' (pinch) frame now and in the root frame through it as the model has it at the joints now,
  so a plan from there cancels the model's error; the scale is the model's.

Synthetic (the model's poses off 2 mm, 0.5 deg per frame): a table point by lookup 2.4 mm off, a point 5 cm up from
two marks 2.5 mm, one found by the tool 3.6 mm, a cylinder's foot 3.3 mm (radius 17.6 for 18 mm); against the fake
robot with its joints 1-2 deg off, the sweep's turns refined to under 0.1 deg. Not yet on the robot.

**`sfm_ml_server` (proposed, the person's idea).** `sfm_torch` loads torch and the weights on every call (~3-5 s of
its ~10). A local server holding DISK and LightGlue on the GPU, started by the first `sfm_torch` call (a subprocess on
the laptop, a localhost socket), answering "features and matches for these frames", and shutting itself down after
some idle time (~10 min); `sfm_torch` falls back to running in process if it cannot reach it. The adjustment stays in
the caller.

### 6.3 Hardware-in-the-loop test: the head's depth against the wrist's structure from motion (design, 2026-11-08)

**Question:** how well does each wrist method (`sfm_light`, `sfm_torch`, the model's poses alone) know the camera's
motion and the scene, against the head camera's depth, at attack-like poses; which to build the scan on.

**Ground truth that needs neither calibration:** the wrist calibration's checkerboard (8 x 6 inner corners, 25 mm)
flat on the table, in view of both cameras. Each wrist frame's pose relative to the board comes from the board's
corners (PnP: ~1 mm, a few tenths of a degree at 15-25 cm), so the camera's real motion between frames is known
without the arm, the model or the head. The head sees the board too (its pose from the color image's corners), which
puts the head's depth points in the board's frame, again without the head's calibration (its ~2 cm at home). The board
is masked out of the features the methods use (they work from the table and the objects, as in a session); it is only
the referee.

**Procedure** (a script in `checks/`, each move approved, recordings in `scratch/`):

1. The scene: the board, the bottle and a box of known size on the table.
2. Head: a capture aimed at the scene: the board's pose (head frame), the depth's points in the board's frame, the
   objects fitted (`fit_cylinder`): their positions and the table, the head's numbers.
3. Arm: to an attack-like pose over the objects (the camera 12-20 cm away); a wrist sweep: roll -30..30 deg in 15 deg
   steps, wrist flex +-10 deg; each frame saved with the joints read.
4. For each frame, from the board: the true pose; from the model: its pose; from each method: the refined pose.
   Scored: the relative motion's error against the board's (mm, deg; the turns, the directions, the one scale), the
   table plane's height and tilt against the board's plane, the jaws' height over the table.
5. The objects: marked in a few frames, triangulated with each method's poses (and the board's), compared in the
   board's frame with the head's fit (position, width).
6. Repeat over scenarios: 3-4 attack poses (heights 12-20 cm, pitch -40 to -70), the sweep's size (roll +-15, +-30),
   lighting; the box and bottle moved.

**First run, hil3 (2026-11-08;** `checks/hil_wrist_sfm.py`; the right arm's tip 16 cm over the board, 8 cm toward the
robot from its center so the closed jaws do not cover it (hil1/hil2: they did, the board found in no frame); a sweep
of 8 frames, the board found in 6; the bottle, chapstick, screwdriver and the table's edge in view):

| | turn error, max | direction error, max | scale | table (board frame) |
| --- | --- | --- | --- | --- |
| the model (joints read) | 3.6 deg | 6.1 deg | 0.92 +- 0.05 | - |
| `sfm_light` | 2.5 deg | 40 deg | 0.77 +- 0.26 | wrong |
| `sfm_torch` | 0.9 deg | 13 deg | 0.89 +- 0.04 | wrong (another plane) |
| the head's depth | | | | +2.9 mm, 0.4 deg |

- **The model's relative motion is far off even for wrist-only moves:** turns up to 3.6 deg (likely much of it the
  right camera's pose on the gripper: hil3's board views can fit it, a hand-eye fit).
- **SIFT's matches are wrong on the robot's frames** (16 px against the board's poses); DISK and LightGlue's agree with
  them to 1.3 px: `sfm_torch` is the one to build on.
- **The adjustment did not converge** with SciPy's finite differences (every run stopped at 1000 evaluations, near
  its start); replaced by Levenberg-Marquardt with the Schur complement (`_bundle_adjust`): `sfm_torch` from the
  model's poses now ends 0.9 deg from the truth (2.8 before), at 0.89 px (the true poses: 0.88).
- **Still wrong:** the moves' directions (short moves along the camera's view are poorly fixed by images) and the
  table plane (the points' dominant plane was not the table). Next: score positions after a similarity alignment
  (what matters is mm, not the direction of a 1 cm move), find the table among the points near where the model puts
  it, and the hand-eye fit from the board views.

**The right wrist camera's pose, again (hand-eye from the board, 2026-11-08).** The model's 3.6 deg of relative
turns on wrist-only moves pointed at the camera's pose on the gripper. `checks/hil_wrist_sfm.py handeye-record`
(the calibration's `POSE_VIEWS` by IK: 5 + 3 views with the board, most others out of reach; `--joints`: joint offsets
about the hold, nothing out of reach, 8) and `handeye-fit` (the calibration's `fit_extrinsics`, with hil3's 6 board
frames): 22 views, settled: the camera at (1, 85, -12) mm in the gripper link +- 3-4 mm, 0.7 deg; 18 mm and 5.7 deg
from bowl5's patch.

- **A pinch point moved 3 cm along the fingers fitted the wrist picture, but the head camera refuted it:**
  `calibration.check` with that candidate (5 views, the gripper at 4-65 deg; driven by a script,
  `CheckSession.next_view`) drew its tips ~3 cm beyond the real ones, the current model's **on** them in every view.
- **The answer, from the person: the fingers have two faces.** Both are ~1 cm thick at the tip, along the hinge.
  The pinch point is calibrated (and checked) on the face the head camera sees; the wrist camera sees the other
  face, 1 cm away along the hinge (the pinch frame's +y; its -y is the hinge axis, which points toward the head when
  `visible_side` is "positive"). bowl5's 7.8 deg turn of the camera had absorbed that 1 cm. Built in
  (`JAW_THICKNESS_M`, `XleRobotModel.wrist_face`): the wrist camera's fingertips, the closing line and the sweep are
  drawn on its face; with it, the original calibration's tips are 22 px off (44 without), and a turn of 5.2 deg
  (not 7.8) fits them to 8 px: `urdfs/xle_calib_1008_wrist2.urdf`, now `api.MODEL`. The board's hand-eye fit (18 mm,
  5.7 deg from the patch) relates the camera to the gripper through the arm's model at every pose and absorbs its
  errors: not used. hil3's 3.6 deg of model motion error is then mostly the arm's kinematics, which structure from
  motion measures from the images anyway.
- **Grasps aim at the fingers' middle** (`grasp_offset(..., mid_plane=True)`, the default): half a thickness along
  the hinge from the pinch point, which is on one face. For a side grasp, ~0.5 cm in height.

**hil3 again, with `xle_calib_1008_wrist2.urdf` (the fingers' thickness in), and the table and scale fixed:**

| | turn error, max | direction error, max | scale | table (board frame) |
| --- | --- | --- | --- | --- |
| the model (joints read) | 1.9 deg (3.6 before) | 6.4 deg | 0.92 +- 0.05 | - |
| `sfm_torch` | 0.38 deg | 6.2 deg | 0.88 +- 0.03 | -30 mm, 1.8 deg |
| `sfm_torch`, scaled by the head's table (`scan_scale`) | 0.38 deg | 6.2 deg | **1.005 +- 0.03** | **+2.4 mm**, 1.8 deg |
| the head's depth | | | | +2.9 mm, 0.4 deg |

- Half the model's motion error was the camera's pose on the gripper (the 1 cm thickness fix).
- **The table search** now takes only near-level planes (`TABLE_TILT_DEG`), 4 mm thick: the points' dominant plane had
  been a tilted slab across the table's band (the forearm, objects and the floor beyond the edge in view too).
- **Scale is the one thing structure from motion cannot measure**, and here it came out 12% short: the arm moves the
  camera ~9% further than the model says even for wrist-only moves (the board's squares checked metric: 25.1 mm by
  the head's depth; a joint fit of the camera's pose to the fingertips and hil3's motions (tips 2.5 px, turns 0.42
  deg) did not change it, so not the camera's place on the gripper). `scan_scale` sets it from the scene: the table's
  height by the head's depth (x1.14 here; relies on the model only for the camera's height now) or the object's
  width from the head's fit (independent of the arm). Scaled, the scan's table sits 2.4 mm from the board's plane.
- **The cause of the ~9%, tested (jnt1, jnt2, roll1, lens2; `checks/hil_wrist_sfm.py joint-record`, `lens-record
  --show`, `lens-check`):** the head and the wrist camera together, the arm turning only its wrist roll.
  - Distances to the board agree: the head's depth (on the wrist camera's module; the head must be >= ~35 cm from it,
    jnt1's 25 cm was inside the RealSense's range), the wrist's PnP and the model within 3-4%.
  - The camera's travel per roll: the head's depth (ICP of the module's points, the same answer from starts 13% apart)
    4-6% more than the model; the wrist's PnP 14% more (the board's symmetric corner order chosen consistently),
    10% with the lens refit on 289 hand-held views (the current lens fits them to 0.64 px, the edges included: the
    lens is fine). So the arm's wrist roll is close to the model (~5%), and the wrist's PnP overstates sideways
    motion by another 5-8% for a reason not found: a poor referee for scale (hil3's board scores carry it).
  - Structure from motion's scale, examined offline: the adjustment is unbiased on synthetic data (an exact model:
    1.000; a model off on average: its average). On hil3 the scale depended on the prior's strength (0.96 of the
    model's moves with a 1 mm prior, 0.87 with 20 mm): the images' error is the same at any scale, the prior pulls on
    it weakly, and the solver drifted along it and stopped. Fixed exactly (`sfm._model_scale`: after the adjustment,
    the one factor that best matches the cameras' spread to the model's; the fit is unchanged): 0.958 at any prior.
    What is left on hil3, the table 11% too near before `scan_scale`, is the model's own motion: the head measured
    the wrist's rolls 4-6% longer than modeled, and the flex frame's ratio is the lowest (0.88: the wrist flex is off
    more than the roll). So a scan at the model's scale is short; `scan_scale` sets it from the scene (hil3: 1.005,
    the table within 2.4 mm). A fix at the source: calibrate the wrist joints' geometry (their zero offsets and the
    links' lengths) from roll and flex motions seen by the head.
  - A detour, recorded: the roll motions by PnP alone asked for the camera 1 cm further along the hinge; the person's
    ruler (6.5 cm from the lens to the fingers' camera-side plane; wrist2: 6.7) and the head's ICP said no.

**What comes out:** a table per scenario (each method's motion error, scale, table, object position; the head's
object positions), and the choice of method and sweep. Smoke tests from it: the recorded frames are images, which do
not go in git (`felix_jr/CLAUDE.md`), so the frozen data is numbers: each scenario's tracks (the matched pixels), the
model's and the board's poses, in `tests/felix_jr/reference/` (a few hundred kB as JSON); the tests replay the
adjustment on them and check its error against the board's.

**Open questions:**

- On the robot: the arm's smallest reliable step (the servos' deadband: 1 deg of pan is ~5 mm at the gripper), and
  whether the steps go as predicted (the reported gain).
- The fingertips in the wrist picture: if they show, a last check against them at the object's depth needs no depth
  estimate at all.
- How far the wrist camera sees: its field of view against the attack standoff (5 cm today) and the jaws' opening;
  the fingers in its picture.
- Whether the pan's turn about the shoulder (about 45 cm back) is close enough to a shift across the jaws, or a
  small re-solve per correction is needed.
- The tolerance to stop at (how centered is centered for a narrow, light object), and whether to grip lower or
  close slower as well (the fourth chapstick tipped when closed off center).
- Whether to keep a head-camera check of anything at the hold (the hand's neighbors, the support) that the wrist
  camera cannot see.

## 7. What worked and what did not (2026-11-08, after bowl5-bowl8)

bowl7 (the bottle) and bowl8 (the chapstick, never touched) were picked and placed with the pattern below; bowl5/6
were not. Details and numbers: [sessions.md](sessions.md), sections 6-6.3 above.

### The pattern that worked

0. **The head plans, the wrist refines.** The head camera's depth finds the object and plans the grasp and the attack
   point; at the hold, the wrist camera measures the object in the jaws' own frame and plans the last centimeters.
1. **Primitives fitted to the graspable part** (`fit_cylinder`, and at the wrist a cylinder on DA3's points). The
   graspable part is the model's (the VLM's) choice: the body, not the cap (bowl7's head fit took in the cap, 3.8 cm
   against the body's 2.9, and every plan and overlay inherited it). Hold the radius when it is known and the free
   fit disagrees.
2. **The head's depth for the targets:** the object's place and the initial grasp (within 1-2 cm: good enough to get
   the wrist camera there).
3. **A short wrist roll + DA3 at a hold:** a +-15 deg roll (3 frames, the wrist only), posed by structure from motion
   (`sfm_torch`: DISK + LightGlue) and scaled by the head's table height (`scan_scale`); DA3-SMALL (isolated
   environment: its packaging conflicts with ours) gives dense depth on every frame; a cylinder fitted to all the
   graspable part's points (the table measured locally, the object's pixels by color, DA3's flying pixels at depth
   edges and its least confident third dropped); the contact patches where the jaws' closing plane meets it
   (`harness/scratch/da3_check.py`). bowl7: radius 1.52 cm (1.45 measured), "advance 1.9 cm": it worked.
4. **The final attack from DA3 with the fingers and the table as rulers:** too close to roll, DA3 on the current frame
   alone, scaled by points along the fixed finger (rigid to the camera), placed the chapstick at the fingertips with
   the fixed finger 8 mm clear, as the person saw.
5. **For this gripper (one fixed jaw):** the closing line on the object's middle, then either the shoulder pan to
   press the fixed finger on (bowl7; the wrist roll does not move the fixed tip: it is on the roll axis) or, for an
   object that must not be nudged, the fixed finger a few mm clear (bowl8); then a firm close (the target 6-8 of
   `gripper.pos` past contact: held, where 1.3 had not); the lift test judged by the object's size in the wrist picture.
6. **Supporting fixes that mattered:** the fingers' 1 cm thickness (the wrist camera sees their other face); plans aimed
   1 cm high against the sag; joints left short pushed again (`align_step`); the 4 deg settle tolerance when loaded.

### What did not work

1. **Structure from motion as a depth source:** triangulating the object over the arm's moves (bowl6: 10-20% depth
   errors, the arm's motion error), and sparse-point fits on white objects (`scan_fit_cylinder`, `scale_from_width`:
   wrong widths). Its scale is the arm model's (~5% short for the wrist, more for the elbow). **As a pose provider for
   DA3 it did work** (above), with the learned matcher.
2. **SIFT matching (`sfm_light`)** on the robot's frames: 16 px off the board's truth, 22-26 inliers per pair (the
   learned matcher: ~1 px, 750-1100).
3. **The image and the closing line alone:** rigid to the gripper and right about "not far enough", but from a camera
   looking nearly straight down they could not say how far (nearer and lower look alike); every last move needed
   DA3's depth (or the person) to complete the picture.
4. **Carrying a measurement through the arm's model over centimeters:** the scan's fit 6 cm of moves away (bowl7), the
   checkpoint's fit 3.4 cm away (bowl8: it put the chapstick through the fixed finger). Re-measure at the pose.
5. **Depth from the object's width** (when the width was wrong), **single-frame DA3 scaled by two fingertips** (20%
   apart; many points along a finger worked), **the board hand-eye fit over arm poses** (it absorbs the arm model's
   errors), **the jaws' reading as a grip check** (a chord 4 mm off the middle reads 0.4 of `gripper.pos` less).

### Still open

- **Moves along the approach fall ~1 cm short** over a few cm (bowl7, bowl8 twice): the wrist joints' geometry
  (calibrate from head-observed roll and flex motions), or plan the last centimeters from a fresh view (which works).
- **DA3 in the harness:** a local service in its own environment (the person's idea: started on first use, shut down
  when idle), the checkpoint as a tool with the contact shift's sign computed (bowl8: applied by hand with the wrong
  sign once), and tests.
- **The VLM's part:** choosing the graspable part (the box, the height band) and judging the overlays; the person did
  the final checks in bowl7/8.

### Deprecated by these findings (2026-11-08)

Marked in `harness/api.py` (`@tool(..., deprecated=...)`: shown in the model's tool documentation and logged when
called); kept for now, to be removed once the DA3 checkpoint is a tool:

| Tool | Why | Instead |
| --- | --- | --- |
| `sfm_light` | SIFT's matches failed on the robot's frames | `sfm_torch` (as DA3's pose provider) |
| `triangulate_wrist` | triangulating over the arm's moves inherits its motion error (bowl6) | DA3 on a short wrist roll |
| `scan_fit_cylinder` | a sparse silhouette fit: a wrong radius (bowl7) | a cylinder on DA3's points (`scratch/da3_check.py`) |
| `scan_locate`, `scan_find` | sparse depth; never used on the robot | DA3's dense depth |
| `refine`, `arm_error` | the head's re-measure at the hold, through the joints | the wrist refines (DA3 at the hold) |

In part: `scan_scale`'s width path (bowl7: x1.6 from a wrong width; the table path stays); `plan_alignment`'s
measurement from marked edges (depth from the width, `center`, `sidestep_m`, `hinge_gain`: superseded by the DA3
checkpoint; as the planner of a given move, `nudge_m`, it stays). Outside the harness: `checks/hil_wrist_sfm.py`'s
`handeye-record` / `handeye-fit` (the board's hand-eye fit over arm poses absorbs the arm model's errors: not for the
camera's pose on the gripper; the fingertip fit is). Still current: `wrist_look`, `render_wrist` (the closing line,
the midline), `wrist_scan` / `scan_views` / `sfm_torch` (poses for DA3), `scan_scale` (the table), `align_step`.
