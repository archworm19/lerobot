# Object-centric grasping (experiment)

**Status: phase A built, phase B steps 1-2 built, 2026-11-06** (results below the phase table). Built: the primitives
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
| C | the attack hold folded in: `refine` (re-locate, `arm_error`, re-solve the chosen candidate, check the move in) | a robot session (bowl1-like) with no hand-written snippets, every grasp from `grasp_candidates` |
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
(2) the camera check and the tiered ranking. To come: (3) the clearance checks and their geometry (capsules for the
gripper and forearm: the URDF has no collision geometry; `calibration.arm_shapes` fits them from outline clicks,
built 2026-11-06, not yet run on the robot), (4) `Object` and `plan_pick`, (5) the pen2 and
container1 replays, (6) `fit_cylinder`.

- **The prune:** a claw target whose gripper axis is more than 25 deg out of the arm's vertical plane (through the
  shoulder pan axis and the target) is not solved: passes reach 18 deg (the jar and a lying pen, both arms), and it
  skips about half the failing solves (225 of 504 at the jar's lid; 18 s to 11.5 s).
- **The camera check is hard:** a wrist camera plate hiding the jaws at the attack point or the grasp rejects it
  (`protocols.md` rule 1 must look there). The pitch -30 grasp, best by the IK alone, is rejected for the right plate,
  bowl1's own reason; three such are the nearest misses.
- **The tiers:** jaws not edge-on; then no joint within 5 deg of its bound; then how squarely the camera sees the jaws
  (steps of 0.1); then the IK cost.
- **Not yet the grasp that worked:** the best now come low (pitch -16 to -31) from phi 30, the fixed jaw on the far
  side (roll near 180), seen at +0.47 to +0.69; the grasp that worked (pitch -44, seen -0.35) ranks next. The low
  ones cross where the jar's neighbours stood, which the clearance checks will judge; whether those or the
  visibility tier should decide is open (`test_grasp_replay_smoke.py::test_the_best_is_like_the_grasp_that_worked`,
  an expected failure).

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
| `Object` | a named object: `parts` (name -> primitive), the whole's extent (for clearance while carried), a rough center of mass, the capture it came from (freshness, `harness.md` 1.C). Drawn by `render`. |

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
| `Check` / `Score` | functions of (candidate, scene) -> (passed or value, a short text). Checks: reaches; opening fits; attack point clear; straight move in clear (swept at the arm's measured error when one is given); no joint jump on the way; not hidden (`hidden_by`, later the forearm). Scores: jaws visible (`XleRobotModel.visibility`), joint-limit margin, depth in the jaws. **Built so far** (inline in `grasp_candidates`, stages `STAGES`): opening fits, the arm plane, reaches (attack, then grasp from its joints: no joint jump), not hidden by a plate; the visibility and joint-limit scores, ranked in tiers (`GraspCandidate.rank_key`). |
| `Scene` | what the checks test against: the table plane, the other `Object`s, the robot model (links, plates), and the capture's depth for anything not modelled. |

**Planning (model-facing tools):**

| interface | responsibility |
|---|---|
| `grasp_candidates(obj, part=None, arm=None, where=None, n=5) -> GraspCandidates` | **Built, thin** (`grasp_candidates(primitive, model, capture, arms, where, spacing, margin, attack, n)`; the checks and tiers of `Check` / `Score`; every rejected target kept, with its stage and reason; `Object` and `render` not yet). generate -> claw -> IK -> checks -> ranked; `arm=None` tries both. `where` narrows the family in object terms ("top 2 cm", "from above"). Returns the best `n` named A, B, ... plus a count of the rejected by reason. Drawn by `render` (each candidate's jaws, colored by rank). |
| `GraspCandidate` | one ranked grasp: the `ClawTarget`, the solved `Poses` for attack and grasp, every check and score, `describe()`: "A: right arm, from the robot's side 40 deg down, jaws level; seen (-0.40); move in clear >= 0.8 cm; 7.6 cm from `white`; reaches". |
| `plan_pick(candidate, lift=0.03) -> Poses` | stage, attack point, grasp, lift, with the grip actions, ready for `prepare_execution` (kept). |
| `refine(candidate, capture, error=None) -> GraspCandidates` (phase C) | at the attack hold: the same family member re-solved from the new capture, the object re-fitted, aimed to cancel `arm_error` (kept), the move in re-checked. |
| `plan_place(pick, onto, pose="upright", at=None) -> Poses` (phase D) | carry and place: the object's pose in the jaws is known from the grasp, so the gripper pose comes from where the object should end; lift height from the object's extent over what is between; release height above what is below. |

Kept as they are: `Capture`, `View`, `Grid`, `Points`, `TableEstimate`, `Focus`, the `Workspace`, `render`,
`contact_sheet`, `scan`, `look_at`, `prepare_execution`, `execute`, `arm_path`, `arm_error`, `home`, `history`.

## 3. Grasp solving flow

bowl1's jar, as it would go (the model's calls; the tool's work in comments):

```python
# perception: the model marks pixels on the jar in a view
jar = fit_cylinder(view, uv=[[214, 250], [262, 252], [238, 300], ...], axis="up", label="jar")
#   -> Object jar: cylinder, axis vertical at (-34.4, -10.7) cm, radius 2.0, z 71.4..79.3; residual 0.2 cm, 812 points
bowl = fit_rim(view, uv=[...], label="bowl")

# candidates: generate -> claw -> IK -> checks -> rank
cands = grasp_candidates(jar, where="top 2 cm", arm="right")
#   Cylinder.antipodal_grasps: t in the top 2 cm x phi every 30 deg x psi in {0, -20, -40} -> 108 grasps
#   to_claw: x 2 fixed sides -> 216 targets (opening = diameter 4.0 cm along the real closing line)
#   ArmIK: 216 solves, ~15 ms each
#   checks: 61 reach; 44 with the attack point and the move in clear of the jar, the table and `white`
#   scores: visibility, hidden_by (the right plate hides 9), joint-limit margin
#   -> A: right, from the robot's side, 40 deg down, jaws level; seen (-0.40); move in clear >= 0.8 cm; ...
#      B: right, from the right, 20 deg down; seen (-0.54); 3.1 cm from the bowl's rim at the attack point
#      ... rejected: 155 out of reach, 17 move in hits the jar or the table, 9 hidden by the right wrist camera plate
render(view, jar, bowl, cands)          # the model looks, picks A

# pick: stage and attack, hold
pick = plan_pick(cands["A"])            # stage, pick.back, pick, lift1, with "start": open, "pick": close
execute(prepare_execution(select_poses(pick, "stage", "pick.back")), hold=True)

# rule 1 at the hold (protocols.md), now three calls
look = look_at(cands["A"])
err = arm_error(pick, look, "pick.back")
a2 = refine(cands["A"], look, error=err)  # re-fit the jar, re-solve A, re-check the move in at the measured pose
execute(prepare_execution(select_poses(plan_pick(a2["A"]), "pick.back", "pick", "lift1")), hold=True)

# carry and place
place = plan_place(pick, onto=bowl, pose="upright")
execute(prepare_execution(place), hold=True)
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

**Tests (`tests/felix_jr/`):**

- `test_grasp.py`: rewritten for `to_claw` and the candidates (the claw geometry it checks stays true).
- `test_harness.py`: the `solve_pose` / `approach` / `held_object` cases.
- `smoke_tests/test_ik_smoke.py`: rebuilt on `grasp_candidates` over pen1's and bowl1's recorded scenes.
- `test_arm_shapes.py` (was `test_wrist_camera_board.py`): `test_solved_poses_say_what_the_camera_sees_of_the_grasp` moves to the scores.

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
  cheaper: solve once per (phi, psi) and shift along `t`. (The arm-plane prune is in: phase B progress.)
- **Families against the arm's five joints.** The heading follows the shoulder, so a 30 deg `phi` grid leaves most
  sides unreachable; the prune skips them unsolved. A cylinder is symmetric about its axis, so `phi` could instead be
  solved for (set by the heading the arm reaches); whether a primitive should declare its symmetries for that is open.
- **Visibility as a tier.** Ranking by how squarely the jaws are seen prefers low, face-on grasps; bowl1's landed at
  -0.37 to -0.66, so past the edge-on threshold it may matter little. Revisit once the clearance checks are in.
- **The camera's pose.** The checks use the capture's head pose; rule 1 looks from the `look_at` pose at the attack
  point, which sees the grasp from a little elsewhere. Check from that pose instead.
- **Contacts at an edge.** `t` runs to the cylinder's very ends (the jar's top rim); the fingers have width, so a
  grasp needs its contacts at least half a finger in from an end (a check, with the finger's size from the model).
- **Frames in model-facing calls.** Whether the model ever needs to give a direction in the object's or the gripper's
  terms (e.g. "from the robot's side"), and how to name it so the frame is obvious, is open until the model uses
  `grasp_candidates`.
