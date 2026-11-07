# Object-centric grasping (experiment)

**Status: phases A-C built and tried on the robot (bowl3, bowl4); next, a coarse-to-fine redesign around the wrist cameras, 2026-11-07 (section 6)** (results below the phase table). Built: the primitives
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


**Deprecated by the wrist-camera redesign** (2026-11-07, section 6). They keep working until the redesign replaces
them; nothing new is built on them.

*The arm's shapes* (they existed for the head camera's line of sight, which the wrist cameras replace; the wrist
camera's own pose is calibrated instead, section 6):

- `calibration/arm_shapes.py` (the whole script) and its tests (`test_arm_shapes.py`).
- `calibration/xle_robot_model.py`: `ArmBox`, `BOX_PARTS`, `with_boxes`, `box_frame`, `boxes_root`, the `boxes`
  record; `WristBoard`, `with_wrist_board`, `without_wrist_board`, `board_corners`, `wrist_board_link`,
  `BOARD_THICKNESS_M`, `PLATE_MARGIN_M`, `segment_hits_box`; the plates' and boxes' outlines in `calibration/check.py`.
  Their tests: `test_arm_model_shapes.py`, `test_calibration_check.py`'s plate and box cases.

*The head camera's view of the grasp:*

- `harness/grasping.py`: the stages "attack view" and "grasp view", `sight_blockers`, `Body`, `arm_bodies`,
  `arm_boxes`, `finger_boxes`, `FINGER_*`, `SIGHT_STOP_M`, `SIGHT_MARGIN_M`; `GraspSettings.see_contacts`,
  `see_at_grasp`, `see_at_attack`; `refine`'s view check. (bowl3 and bowl4 turned them off at every hold.)
- `calibration/xle_robot_model.py`: `hidden_by`, `view_of`, `visibility` (as used for grasps; `views.py`'s check that
  the head camera faces the pinch point stays: the click calibration needs it, without the plates).
- `harness/reach.py` / `objects.py`: `FrameSolution` and `Poses`'s `visibility`, `hidden_by`, `seen`, `hidden`,
  `camera_text`, `EDGE_ON_VISIBILITY` (the "camera: jaws seen / edge-on / hidden" reports).

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
- `test_arm_model_shapes.py` (from `test_arm_shapes.py`, was `test_wrist_camera_board.py`): `test_solved_poses_say_what_the_camera_sees_of_the_grasp` moves to the scores.

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

- **(a) The object in the wrist camera's view at the attack point:** inside its field of view, at a distance it can
  focus and measure, not behind the fingers. This depends on the approach and the standoff, not on the head.
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
2. **A wrist capture in the harness:** `Capture` with the wrist camera's frame, and a tool to take one at a hold.
3. **The measurement:** the object in the jaws' frame from a wrist capture (the same `fit_cylinder` marks, or a
   simpler detector for the centering), as a tool that says how far off and which way.
4. **The alignment step:** from that measurement, the small joint moves (pan across, roll for the closing line, the
   arm's plane for in/out and up/down), each approved, until centered: a tool, so the model asks for it rather than
   composing joint moves.
5. **Grasp selection:** (a) and (b) in `grasp_candidates`, the head camera's view filters removed (section 4); the
   attack point along the gripper's axis in `plan_pick`.
6. **A session** (a bowl4-like scene: an upright bottle and a chapstick into a container), every grasp aligned at the
   hold from the wrist camera.

**Open questions:**

- How far the wrist camera sees: its field of view against the attack standoff (5 cm today) and the jaws' opening;
  the fingers in its picture.
- Whether the pan's turn about the shoulder (about 45 cm back) is close enough to a shift across the jaws, or a
  small re-solve per correction is needed.
- The tolerance to stop at (how centered is centered for a narrow, light object), and whether to grip lower or
  close slower as well (the fourth chapstick tipped when closed off center).
- Whether to keep a head-camera check of anything at the hold (the hand's neighbors, the support) that the wrist
  camera cannot see.
