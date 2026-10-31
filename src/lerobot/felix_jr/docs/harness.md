Replaces scene_tools.md


Research questions: what is the ideal harness / tool set to enable a vlm to accomplish quasi static pick and place tasks. Implied assumption = contact rich manipulation tasks would require finetuning and are beyond the scope of this project.


This doc does 3 things:
0. summarizes relevant research
1. discusses design principles for the harness
2. records results of experiments; TODO: maybe this should be in a separate file?


# 0. Research summary

Voxposer is the seminal work for this project. Its main limit is that it came before foundation models could
see well: the LLM never saw an image. We focus on follow-up work that uses the vision of GPT, Claude or Gemini.

Paper status (checked 2026-10-31):

| Paper | Year | Status | Why |
|---|---|---|---|
| VoxPoser | 2023 | baseline | the starting point |
| PIVOT | 2024 | relevant | VLM picks among candidates drawn on the image, iterative |
| MOKA | 2024 | relevant | marks + grid overlay, the VLM picks labels, geometry does the rest |
| ReKep | 2024 | relevant | VoxPoser successor: VLM writes constraint code, an optimizer plans |
| CaP-X | 2026 | relevant | benchmark of how much abstraction coding agents need |
| Gemini 4 Argon robotics demos | 2026 | relevant (blog, not a paper) | world grid overlay + primitives over HTTP, on an SO-101 |
| VIA | 2026 | relevant | agent edits a virtual gripper target and checks renders before executing |
| Show-Harness | 2026 | relevant | VLM in a closed loop of discrete step actions |
| Guava | 2026 | relevant | ablation of harness design: observations, action level, loop |
| World Action Agent | 2026 | relevant, abstract only | rehearse each action as a preview, then execute |
| Gemini Robotics-ER 1.5 | 2025 | reference | pointing/trajectory output format, orchestration through function calls |
| RoboPoint | 2024 | partly relevant | fine-tuned; useful only as numbers on how well zero-shot GPT-4o points |
| ManipBench | 2025 | partly relevant | benchmark of VLM low-level manipulation reasoning; could help pick a model |
| ManipVQA | 2024 | **irrelevant** | needs fine-tuning |
| Sketch-MoMa | 2024 | **irrelevant** | teleoperation: a *human* draws the sketch, the VLM reads it. Not autonomous |
| "Code-as-Costmaps" | — | **not found, probably invented** | no paper by this title; the description reads like a mix of VoxPoser and later work |

Tool calls are written as `name()`. "Fixed pipeline" means the harness calls the step, not the model.


## VoxPoser (Huang et al., CoRL 2023) — baseline

Models: GPT-4, **text only** (the paper lists multimodal LLMs as future work).

a. **Planner LMP** (an LLM prompt): the instruction goes in, a list of sub-tasks comes out. Your guess is right.
   Not used in simulation, where tasks have one step.
b. **Composer LMP**: for each sub-task, it writes Python that calls the value-map LMPs, passing them
   natural-language arguments: `get_affordance_map()`, `get_avoidance_map()`, `get_rotation_map()`,
   `get_velocity_map()`, `get_gripper_map()`. Each of these is **another LLM call** that writes NumPy code to fill a
   100×100×100 voxel grid. So it is right that the low-level LLM writes affordance and avoidance functions, and it
   also writes rotation, velocity and gripper maps.
c. **Perception**: the generated code calls `parse_query_obj()` / `detect(name)`, which runs OWL-ViT (box) → SAM
   (mask) → XMem (tracking) → RGB-D point cloud and returns position, occupancy grid and normal. So yes, the
   map code calls segmentation and a tracker. The tracker is a mask tracker (XMem), not a point tracker. It is
   called through a fixed API, not chosen by the model.
d. **Motion planner**: yes, it is a tool call. The composer code ends with `execute(movable, affordance_map=...,
   avoidance_map=..., ...)`. The planner uses only the affordance and avoidance maps (cost = −(2·aff − 1·avoid),
   normalized), searches greedily for a collision-free path of end-effector positions, sets rotation, velocity and
   gripper at each waypoint from the other maps, runs the first waypoint, and **replans at 5 Hz** (MPC). Pushing
   tasks use a heuristic dynamics model with random shooting.

Results: 88% real-world success (vs 24% for LLM + primitives); most real-world errors came from perception.


## PIVOT (Nasiriany et al., ICML 2024)

Models: GPT-4V, Gemini.

a. Start with a broad Gaussian over the action space.
b. **Sample ~10 candidate actions** (10 was the best trade-off: more samples cover more but hide each other) and
   draw each as an arrow from the end-effector ending in a numbered circle. Height is shown by color or label size.
   The gripper is given as text.
c. The VLM returns the most promising labels → fit a new Gaussian to them → repeat, usually **3 iterations**.
   Optional: run E copies in parallel and merge them.
d. Execute the chosen action **directly** as a relative Cartesian step (Δx, Δy, Δz + gripper, rotation fixed),
   then observe and repeat. For navigation, a pixel is turned into a 3D target with depth.

Answers to the earlier TODOs:
- **How strict was the loop?** Very strict: the harness fixes sampling, drawing and refitting. The VLM only picks
  labels.
- **Could the VLM control the re-rendering?** No. It cannot zoom, change the view or change the candidates.
- **Was there a motion planner?** No: no planner, IK planner or grasp model is described. Actions are small direct
  steps.

Limits they report: unreliable depth reasoning (VLMs could not choose well based on depth), no rotation, occlusion
and low resolution near the grasp, and the model sometimes names the right object but gives the wrong arrow number.
Small trial counts. Pick-coke-can grasp success: 0% → 67% with iterations + parallel calls.

Takeaway: the "draw candidates, have the VLM choose" idea is good. Using it without geometry (no planner, no depth
check) is the weak part.


## MOKA (Liu, Fang et al., RSS 2024)

Models: GPT-4V, GroundedSAM.

a. **Task decomposition** (VLM, top-down image + instruction): a list of sub-tasks, each with `object_grasped`,
   `object_unattached` and a motion direction.
b. **Segmentation** (fixed pipeline): GroundedSAM (GroundingDINO + SAM) masks each named object. K points are picked
   on the contour by farthest-point sampling, plus the mask center. They are drawn as red `P[i]` (in-hand object)
   and blue `Q[j]` (target object).
c. **Grid overlay**: a 5×5 grid with chess names (a–e, 1–5) for free-space waypoints.
d. **VLM picks labels**, returned as JSON: `grasp_keypoint`, `function_keypoint`, `target_keypoint`,
   `pre_contact_tile`, `post_contact_tile`, `pre/post_contact_height ∈ {same, above}`, `target_angle` (from a fixed
   list).
e. **Geometry** (fixed pipeline): keypoints are unprojected with depth. Waypoints are sampled inside the chosen tile,
   with height taken from the same/above choice. A grasp is picked from 30 DexNet-style antipodal proposals, the
   one nearest the chosen point. A smooth trajectory then moves the function point through pre-contact → target →
   post-contact. The planner is not named.
f. Open loop within a sub-task. Between sub-tasks: return to a neutral pose, take a clean picture, ask the VLM again.

Results: zero-shot about the same as Code-as-Policies and VoxPoser. With 2 in-context examples, 9–10/10. Without the
hierarchical decomposition, 0–2/10.
Takeaway: **this is the clearest example of "VLM picks from a discrete set, depth and a grasp model give precision".**
A grid is already in use here, which bears on the RoboPoint TODO below.


## ReKep (Huang et al., CoRL 2024) — the closest successor to VoxPoser

Models: GPT-4o, DINOv2, SAM, AnyGrasp.

a. **Keypoint proposal** (fixed pipeline): DINOv2 features → SAM masks → PCA + k-means inside each mask → candidate
   3D keypoints, merged with Mean Shift.
b. **VLM** (image with numbered keypoints + instruction) writes **Python constraint functions over keypoint
   positions**: the number of stages, then per stage a *sub-goal* constraint and a *path* constraint (distances, dot
   products). This is VoxPoser's "LLM writes cost functions", except the costs are over a few keypoints instead of a
   voxel grid, and the model *sees* the scene.
c. **Optimization** (fixed pipeline, SciPy): the sub-goal solver finds an SE(3) end-effector pose; the path solver
   finds a trajectory. Extra costs: collision with the point cloud, reachability, smoothness. Grasp stages use the
   AnyGrasp grasp nearest the keypoint. PyBullet IK → joint impedance control.
d. **Closed loop**: keypoints are tracked at 20 Hz and the solver re-solves at about 10 Hz. If a path constraint is
   broken, it backtracks to an earlier stage.

Results: 44% with automatic keypoints, 69% with hand-annotated keypoints, vs 10% for VoxPoser on their 7 tasks.
**Failure breakdown: the point tracker causes the most errors (occlusion), then keypoint proposal and the VLM
referring to the wrong keypoint. The optimizer causes few.**
Takeaway: connecting to a trajectory optimizer works. The weak points are perception (tracking, keypoint coverage),
not the optimizer.


## CaP-X (Fu et al., ICML 2026)

Benchmark of coding agents controlling a robot by writing programs over perception and control primitives.

- **Abstraction levels**: S1–S2 use human-written macros (`stack_objs_in_order()`); S3–S4 use raw library calls
  (`solve_ik()`, `sam3_text_prompt()`), with or without usage examples.
- **Tools**: SAM3 (segmentation from text), Molmo 2 (pointing), OpenCV / Open3D, PyRoki (IK + motion planning
  with collision checks).
- **Loop levels**: single turn → multi-turn with stdout/stderr → + raw images → + "visual differencing" (a model
  turns before/after images into text).
- **CaP-Agent0**: multi-turn + visual differencing + a library of 9 synthesized skills + an ensemble of 9 parallel
  drafts (GPT-5.2 / Claude Opus 4.5 / Gemini-3-Pro) merged by a central agent.

Findings: success rises with more human-written abstraction. Multi-turn execution feedback helps every model.
**Adding raw images to the loop *lowered* success compared with text feedback**, but turning images into text
(visual differencing) helped the most. With multi-turn and visual differencing, low-level primitives reach the level
of high-level macros.


## Gemini 4 Argon robotics demos (Google Developers Blog, 2026) — blog post, not a paper

This is the grid demo mentioned under RoboPoint below. The results are Google's own and there is no paper.

- **Observation**: `wrist_cam.png` with calibrated overlays: **a 2 cm world grid with X/Y labels in cm (yellow)**, a
  cyan "+" at the current 3D pinch center, cyan brackets for the jaw opening, a green line for the object's long-axis
  yaw. Plus `obs.json` (`ee_pos`, `holding`, `status`, workspace bounds).
- **Tools**: two HTTP endpoints called with `curl`. `GET /obs`, and `POST /plan` taking Code-as-Policies primitives:
  `look_at`, `deproject`, `grasp`, `place`, `place_until_contact`, `goto` (with `arc_z`), `move_by`,
  `move_until_contact`, `open/close_gripper`, `home`, `done`. Poses take pitch / azimuth / roll + speed. IK and the
  planner live behind the endpoint and are not described.
- **Loop**: on turn 1 the model **chooses a granularity**: coarse (2–4 calls), hover-and-verify before the grasp
  (4–7), or fine visual servoing (8–20). It checks the picture after every motion and recovers on `grasp_failed`.
- **SO-101 demo**: (1) dry run, drawing planned poses on an overhead image, no motion; (2) hover over the brick and
  check that arm and vision agree; (3) run under safety rules in `AGENTS.md`, diagnose failures from logs and video,
  retry, then make it 2× faster. Cycle time 58.9 s → 22.9 s → 17.4 s.

Takeaway: very close to our setup (same arm). The design choices worth copying are the world grid in cm, the overlay of
the current gripper state, the dry run, and the hover check.


## VIA: Visual Interface Agent for robot control (Stanford, 2026)

Models: Claude Code and Codex at several tiers, used unchanged.

a. The agent sees a GUI: third-person + wrist camera images, plus (in sim) a **3D point cloud with a rendered target
   gripper** that it can orbit, pan and zoom.
b. **MCP tools** that edit only the *virtual* target: `hover(u,v)` (returns the 3D point under a pixel),
   `gripper_teleport_via_click`, `gripper_drag`, `gripper_translate`, `gripper_rotate`, `gripper_toggle`, camera
   orbit/pan/zoom, `screenshot`. Each returns a new render.
c. Inner loop: edit the target → look at the render → repeat until satisfied. Then call **`execute_waypoint`**.
   Execution: in sim, a PI controller interpolates linearly to the pose; on the real Stretch, a whole-body controller.
   **No motion planner.**
d. Outer loop: observe after each waypoint.

Results (sim, 6 tasks): Claude Code 43–85%, Codex 38–100% depending on tier. Without the interface (only
`set_pose` / `observe`): 7–25%, except the strongest model at 93%. Real robot: 24/25. Cost $2–16 per success,
51–146 tool calls per episode.
Takeaway: **letting the model check a rendered proposal before it runs is the biggest single gain**, and the VLM
*does* control the view. This answers the PIVOT "re-render" question with a strong yes.


## Show-Harness (Chen et al., 2026)

Models: Gemini-3.1 Pro zero-shot; Qwen3.5-2B fine-tuned.

a. The VLM plans sub-tasks (JSON with completion criteria).
b. Each step it picks **one discrete action**: `MV_FWD/BACK/LEFT/RIGHT/UP/DOWN`, `ROTATE_CW/CCW(axis)`, `GRASP`,
   `RELEASE`, `DONE`.
c. A fixed interpreter turns the action into a bounded pose step (2 cm if the target is visible in the wrist camera,
   4 cm otherwise) with workspace and safety checks. Impedance control or IK. **No planner, no depth, no grid.**
d. Exo + wrist images; proprioception as text; last 5 actions; failure recovery (empty grasp → back to the grasp
   sub-task).

Ablations: removing sub-task planning drops success 96% → 60%, removing failure recovery → 72%. A "visual prompt"
marker on the target affordance raised handle grasping from 40% → 85%. 1 cm steps for precision tasks: 60% → 82%.
Takeaway: visual servoing in small steps works but needs ~30 model calls per episode. Fine for quasi-static tasks,
slow.


## Guava (2026) — ablation of harness design

GPT-5.4 as teacher; Qwen3.5-4B distilled from it. A ReAct loop: one semantic tool call (`grasp(obj)`,
`align(obj)`, ...) → execute → observe → replan.

Ablation (success, GPT-5.4):

| Variant | Success |
|---|---|
| full: images + gripper state + tool response text, object-level tools, one action per turn | 88.6 |
| image only | 56.2 |
| image + gripper state | 77.1 |
| low-level tools (end-effector pose + width) | 73.3 |
| a program of several actions per turn | 82.9 |

Takeaway: object-level tools, text feedback from tools, and replanning after every action each add points.


## World Action Agent (Zhang et al., Sep 2026) — abstract only, PDF not read

A multi-agent harness: views near the contact are chosen automatically; each action is an **editable proposal the
agent previews and revises against planner feedback before execution**; residual offsets are corrected in the same
view; a skill library is learned from videos. 75.6% on LIBERO-Pro. Same idea as VIA. Worth reading in full.


## Gemini Robotics-ER 1.5 (2025) — reference

Outputs points as JSON `[y, x]` normalized 0–1000 with labels, plus boxes and trajectories. Orchestration means the
model calls your robot functions. Google's own precision advice: **crop/zoom small objects, and for precise points
query several times and average.** It does not cover converting to 3D (that is up to us).


## RoboPoint (Yuan et al., CoRL 2024) — partly relevant

Answers to the earlier TODOs:
- **Fine-tuned for pointing?** Yes. Vicuna-13B + CLIP, trained on about 660K synthetic spatial-relation examples
  from procedurally generated scenes. Output: normalized (u,v) → depth → 3D + offset → motion planner.
- **Is zero-shot pointing precise enough?** Their numbers say no, at least for GPT-4o in 2024: on Where2Place, 29%
  of GPT-4o's points land in the target region vs 47% for RoboPoint (RoboRefIt: 15% vs 50%). Frontier models have
  improved since (PointArena reports Gemini-2.5-Pro close to Molmo-72B, and that models trained on pointing beat
  those that weren't), so this should be measured again on our scenes.
- **Grid vs pointing comparisons?** **I found no controlled head-to-head** of world grid vs set-of-marks vs raw
  coordinates. Indirect evidence: MOKA (grid tiles + marks) works zero-shot; Argon uses a cm world grid; PIVOT and
  ReKep use numbered marks; a 2025 wrench-planning paper reports that world-aligned labels beat wrist-frame labels.
  PointArena found chain-of-thought *lowered* pointing accuracy for GPT-4o. → **This is a good first experiment for
  section 2.**


## ManipVQA (Liu et al., 2024) — irrelevant

Fine-tunes a VLM to output keypoints and approach vectors. Out of scope (no fine-tuning).

## Sketch-MoMa (Tanada et al., Toyota, 2024) — irrelevant

The earlier summary was wrong. This is **teleoperation**: a human draws a sketch on the camera image, and the VLM
reads it to choose a task (`pick`, `place`, `move`, `pull`) and a shape for the motion planner. The robot does not
plan on its own. Answer to the earlier TODO: "keyframe trajectories" are the human's sketch.

## "Code-as-Costmaps" — not found

No paper with this title turned up. The description ("VLM outputs 2D costmap functions, unprojected with depth, fed
to MPPI/TrajOpt") looks like a summarizer's mix of VoxPoser and later work. Answers to the earlier TODOs from the
nearest real papers: ReKep gives the VLM depth only indirectly (keypoints are 3D, the model sees 2D marks). VIA and
World Action Agent *do* render proposed motions back to the model before execution; VoxPoser and ReKep do not.


## Patterns across papers (input to section 1)

1. **The VLM picks from a discrete set drawn on the image; geometry gives precision.** MOKA, PIVOT, ReKep and Argon
   all do this. Depth unprojection + a grasp model / optimizer give the last few centimeters.
2. **Check before executing.** A VLM-controlled render or dry run (VIA, WAA, Argon stage 1–2) gives the largest
   gains reported. The VLM should be able to change the view.
3. **Close the loop after each action** with text feedback from tools (Guava, CaP-X, Show-Harness). Turning images
   into text helped in CaP-X more than adding raw images.
4. **Object-level primitives beat raw poses** for frontier models (Guava 88.6 vs 73.3; CaP-X S2 vs S4), unless
   there is a lot of multi-turn feedback.
5. **Perception is the usual bottleneck**, not planning (VoxPoser, ReKep failure breakdowns).
6. Open question: **world grid vs marks vs raw points** has no published comparison. Run it ourselves.


Sources:
- VoxPoser https://arxiv.org/abs/2307.05973
- PIVOT https://arxiv.org/abs/2402.07872
- MOKA https://arxiv.org/abs/2403.03174
- ReKep https://arxiv.org/abs/2409.01652
- RoboPoint https://arxiv.org/abs/2406.10721
- CaP-X https://arxiv.org/abs/2603.22435
- Gemini 4 Argon for Robotics https://developers.googleblog.com/gemini-4-argon-for-robotics/
- VIA https://arxiv.org/abs/2607.11119
- Show-Harness https://arxiv.org/abs/2609.10522
- Guava https://arxiv.org/abs/2606.18363 , https://guava-harness.github.io/
- World Action Agent https://arxiv.org/abs/2609.29964
- Gemini Robotics-ER https://ai.google.dev/gemini-api/docs/robotics-overview , https://arxiv.org/abs/2510.03342
- ManipBench https://arxiv.org/abs/2505.09698
- PointArena https://arxiv.org/abs/2505.09990
- Sketch-MoMa https://arxiv.org/abs/2412.19153



# 1. Harness design principles

## A. Who controls what

The VLM controls looking, drawing, measuring and replanning. In section 0, the systems that did well (VIA, Argon,
WAA, CaP-Agent0) let the agent zoom, re-render and re-check as it chooses. PIVOT, where the harness fixed all of
that, did worst. The harness keeps three jobs:

- **No tool moves the robot.** Tools only read what the harness has already captured. The robot moves only at phase
  boundaries, run by the harness: the **startup scan** (head only) and the **execution of an approved plan** (arm).
  If the model needs a view the scan didn't give, it asks for one at the end of its turn and the harness fulfils it
  between turns (later; head-only requests could be approved by policy).
- **Gating motion.** Nothing moves unless a plan has been previewed and checked (IK, joint limits, workspace,
  collisions) and then **approved by the user**. Approval by the user is the gate for the first experiments; an
  automatic gate can come later.
- **Freshness.** The world changes outside the model's calls (the arm moves, an object slips). The harness marks
  which results are stale (see C).

## B. State: a DAG of immutable objects

A reasoning chain is recorded as an **append-only DAG**. Every object is immutable and has an id, the call that
made it, and the ids of its parents. Calls take ids as arguments. It is a graph, not a stack: the model goes back
to an earlier view, compares two plans made from the same grid, and so on.

| Object | Defined in | Contents |
|---|---|---|
| `Capture` | camera | color, depth aligned to color, intrinsics, camera pose (from `XleRobotModel` + `q`), `q`, timestamp. Wraps a `Scene`. |
| `View` | 2D | `capture_id` + crop rectangle + scale: only a mapping between view pixels and capture pixels, no data |
| `Grid` | world | the plane `z = h` in the base frame (default `h` = table height), origin, cell size, extent |
| `Points` | world | xyz + labels (e.g. from `locate`), and the pixels they came from |
| `Plan` | world | primitives with xyz / yaw, and the checks on them (IK per waypoint, `segment_check` states per leg) |
| `TableEstimate` | world | table height `h` in the base frame, inlier count, spread, tilt, inlier mask (see below) |

Geometry lives in the world frame (`Grid`, `Points`, `Plan`). Images live in `Capture` / `View`. So drawing an
object in any view is projection plus the view's pixel mapping, and nothing is recomputed when the model zooms.

The robot's model and its current state (`XleRobotModel`, `q`, gripper opening, holding) are *not* in the DAG: they
are the live world. A `Capture` records the `q` it was taken at, which is how the DAG refers to the robot.

**Grid projection and grid height.** Drawing a grid in an image is forward kinematics alone: the grid is defined
in the base frame and projected with the calibrated `XleRobotModel.camera_pose(q)` and the intrinsics. No depth
is involved. FK cannot give the grid's height `h`: it says where the camera is relative to the base, not where the
table is (the table is part of the environment, and the robot drives between surfaces). A grid at the wrong `h`
causes parallax: it lines up with the table only directly below the camera, and objects elsewhere read as the
wrong cell. So `h` is a parameter of `Grid`:

- default: the table height, estimated from depth (below).
- the model can ask for a grid at another height (e.g. an object's top) to read positions there without parallax.

**Table height from depth.** Not an average over the scene: that mixes in the floor, walls, objects and the arm, and
lands on no real surface. FK gives the base frame's "up", so no plane fit is needed: finding the table becomes
finding the height most points share.

1. Lift depth to the base frame: valid depth pixels → camera-frame points → base frame with
   `XleRobotModel.camera_pose(q)`. Each pixel now has a height `z`.
2. Crop to where the table can be: within arm reach in front of the robot (out to ~1 m, where D415 noise, which grows
   roughly with distance squared, is still small), excluding points near the robot's own links (projected with the
   same model).
3. Histogram `z` in 1 cm bins. A horizontal surface is a sharp peak; objects, walls and the arm spread over many
   bins. Take the largest peak above the floor (the floor is the other big peak, near the base's height).
4. Refine: `h` = median `z` of the points within ±1 cm of the peak.
5. Check: report the inlier count, the spread (std of inliers) and the **tilt** (angle between the normal of a plane
   fit to the inliers and the base's z axis). More than a degree or two on a level table means the table isn't level
   or the camera pose is off, so this is a running check on the calibration. Render the inlier mask over the image.

Returns `TableEstimate(h, n_inliers, spread_m, tilt_deg, inlier_mask)`, a DAG object derived from its `Capture`
(stale after the robot moves, e.g. to another table).

Failure cases: dark / glossy tables leave depth holes (fine while some patches survive); a crowded table shrinks the
peak but rarely moves it; a base that isn't level makes FK's "up" wrong (the tilt check flags it); a shelf or second
surface gives two peaks (take the one with the most points within reach, warn about the other).

## C. Freshness

Every object inherits the capture(s) it was derived from. After the **world** may have changed (an executed plan:
the arm moved, an object may have moved), objects derived from older captures are **stale**. Moving only the head
does not stale anything: the scan's captures all share one epoch, and a grid or point found in one capture is still
true in the others. `preview` and execute refuse stale inputs; renders of stale objects are labeled stale.

## D. Rendering is one pure function

`render(view_id, overlays=[grid_id, points_id, plan_id, ...]) -> image`. Every image the model sees can be rebuilt
from ids. Overlays are drawn in a fixed style (one legend for FREE / UNKNOWN / BLOCKED, one color per overlay type)
so the model learns one visual language.

## E. Returns: image for *where*, text for *how much*

Each tool returns ids, a short text summary with numbers (xyz in cm, heights, check results) and, only when there
is something new to see, a render. Reasons: in CaP-X, adding raw images to the feedback loop lowered success while
turning images into text helped; in Guava, tool response text added points over image + gripper state. The
model's context holds ids, small renders and summaries, never arrays. The DAG is the full record.

## F. Long-term tools and scratch space (hybrid)

- **Long-term tools:** a Python library over the objects in B (`observe`, `zoom`, `grid`, `locate`, `preview`,
  `render`).
- **Scratch space:** the model works in a **persistent Python session**. Variables hold references to DAG
  objects. Functions the model writes in a session are scratch tools.
- **Promotion:** every executed cell and every object created is logged with its parents. Scratch tools that keep
  being useful across sessions are copied into the library. The logs are also replayable, and are the kind of
  trace Guava / WAA trained small models on.
- Execute is outside the session: the model cannot move the robot from its own code.

## G. Types of tool calls

| Type | Tools | Changes the world? |
|---|---|---|
| look | `zoom`, `focus` (choose captures from the scan) | no (the scan is a harness phase, not a tool) |
| analyze | `grid`, `locate`, (later) `segment` | no |
| render | `render` | no |
| plan | `preview` | no |
| act | execute (user-approved) | yes → stales older captures |

## H. Interfaces (layers)

The VLM does not implement `plan(capture) -> Plan` in one call. The harness implements it as a loop, and the
intermediate objects are what the VLM works with inside that loop.

1. **Objects (data):** `Capture`, `View`, `Grid`, `TableEstimate`, `Points`, `Plan` (B). Immutable. They hold
   direct references to the objects they were made from (a `View` holds its `Capture`), so they are usable without
   a workspace.
2. **Long-term tools (pure functions over objects):** `zoom`, `make_grid`, `table_height`, `locate_pixels`,
   `locate_grid`, `check_path`, (later) `preview`, and `render`. They are both the library and the VLM's tool API.
   They know nothing about the DAG.
3. **Workspace (the session):** the DAG store. `ws.call(fn, *args)` runs any function (a long-term tool or a
   scratch function the VLM wrote), finds which arguments are workspace objects (its parents), and records the
   result with an id. It also tracks freshness (C) and writes the log. In the persistent Python session the VLM
   holds results in variables; the workspace holds the ids, parents and log.
4. **Agent loop (`propose_plan`):**
   ```python
   def propose_plan(task: str, capture: Capture, vlm: VLM, budget: int) -> Episode:
       # ws = Workspace(); add the capture; send the task + render(full view) to the VLM
       # loop: the VLM writes a cell -> run it in the session -> return text summaries + any new renders
       # until the VLM calls submit(plan) or the budget runs out
       # Episode = (plan | None, workspace, transcript)
   ```
   The VLM only chooses the next cell. The harness owns everything else.
5. **Approval:** `render(full view, table, plan)` shown to the user → y / n → execute.

Why keep the intermediates rather than only the final plan: the user can render any step of how the model got to
its plan; Step 3 scores intermediates (`locate` error, reading the grid); and repeated call patterns in the DAG
are the candidates for promotion to long-term tools (F).


## I. One workspace per episode, many captures

A workspace holds every capture of an episode (one task attempt), not one image. Each capture has its own camera
pose from FK, and the geometry (`Grid`, `Points`, `Plan`, the table height) is in the root frame, so anything made
from one capture renders correctly in all the others: find the object in one capture, measure the target in another,
draw one plan in both. Later: `check_path` against several captures at once (blocked if any sees a collision, free
only if one sees the space empty), to shrink UNKNOWN.


# 2. Experiments

## Flow for the first experiments

An episode runs in phases:

0. **Scan (harness):** the head moves through fixed poses (3 tilts x 3 pans); each pose gives a capture with its
   `table_height`; the model gets one **contact sheet** of all captures. Nothing else moves the head.
1. **Focus (model):** choose the captures that show the task (`focus`): no motion, a selection recorded in the DAG.
2. **Analyze (model):** zoom, grids, locate, ... on those views.
3. **Propose → approve (user) → execute (harness).** Execution starts a new epoch and the next round scans again.

Within phases 2–3:

1. The model reasons, using the long-term tools and its scratch space, and proposes a **trajectory plan** (a list
   of primitives: `move_to(xyz, arc_z)`, `grasp(xyz, yaw)`, `place(xyz)`, `open`, `close`, `home`).
2. `preview(plan)` checks it (IK, joint limits, `segment_check` against depth) and renders it on the image with
   possible collisions marked.
3. The **user** looks at the render and decides whether to execute.

Motion: straight-line / arc moves in Cartesian space on pinocchio IK (`XleRobotModel`). No sampling planner and no
VoxPoser-style cost maps to start (section 0: ReKep's optimizer was its smallest source of error; perception was the
largest). Upgrade path if clutter defeats straight moves: PyRoki (CPU, used by CaP-X).

## Development plan (bottom-up)

Each layer of 1.H is built and tested before the one above it, so nothing waits on the VLM:

1. **Objects + `render` + workspace**, offline on saved captures. No VLM, no robot. (Step 1)
2. **Sample renders, driving the workspace by hand** from a script or notebook: zoom, re-grid, locate, check a path,
   render. If the renders don't help a person judge a plan, they won't help the VLM. (Step 2)
3. **Evaluate the renders** against the checks fixed in advance. (Step 3)
4. **Agent loop.** First version: Claude Code itself is the agent, working in the workspace (as VIA did with Claude
   Code / Codex), so no loop is built until the tools settle. Then `propose_plan` with a VLM API.
5. **`preview(primitives)` + execute on the robot**, with the user approving each plan.

## Step 1: objects + render + workspace (offline)

Code in `felix_jr/harness/` (replaces `felix_jr/scene.py` and `felix_jr/overlay.py`). Tests in `tests/felix_jr/`.
Data: the calibration sessions (`~/cal_left`, `~/cal_right`: color, raw depth and joint readings per frame), the
calibrated URDF, and the RealSense calibration (`calibration.json` from `realsense_check --save-dir`) for aligning
depth.

Status: **done** (2026-10-31). `uv run pytest tests/felix_jr/test_harness.py` (synthetic table scene, no robot).

| Module | Contents |
|---|---|
| `scene.py` | `Scene` (moved from `felix_jr/scene.py`): depth queries and `segment_check`, camera frame |
| `objects.py` | `Capture`, `View`, `TableEstimate`, `Grid`, `Points`, `Plan`, each with `describe()` (the text for the VLM) |
| `tools.py` | `full_view`, `zoom`, `table_height`, `make_grid`, `locate_pixels`, `locate_grid`, `check_path` |
| `render.py` | `render(view, *overlays)` (replaces `overlay.py`) |
| `workspace.py` | `Workspace`: `call` / `add` record nodes with parents; `moved()` / `is_stale` for freshness; `save_log` |

```python
model = XleRobotModel.load("~/cal_right/calibration.urdf")
ws = Workspace()
cap = Capture.from_session("~/cal_right", 0, model, load_calibration("~/pi_capture/calibration.json"))
view = ws.call(full_view, cap)                      # cap is recorded as "external"
table = ws.call(table_height, cap)
grid = ws.call(make_grid, view, table, 0.02)        # the estimate, not table.h: the grid is derived from it
close = ws.call(zoom, view, (200, 200, 200, 150))
fine = ws.call(make_grid, close, table, 0.005)      # extent = what the zoomed view shows
image = ws.call(render, close, fine)
ws.ancestors(image)                                 # ['view2', 'grid2', 'view1', 'tableestimate1', 'capture1']
```

Choices made while building it:

- A `Capture` keeps the servo readings, not only the pose: `Capture.load(folder, model)` recomputes the pose with
  the current calibration.
- `make_grid` takes a `View`, and its default extent is the part of the plane that view shows (within 1 m of the
  camera), so a grid made after zooming covers only the zoomed area.
- The grid is **hidden where depth shows a surface more than 2 cm in front of the plane** (objects, the arm), so it is
  drawn only on visible table. Every 5th line is stronger and labeled with its coordinate in cm (`x-60`, `y20`).
- `Points` from `locate_*` are labeled `P1, P2, ...` so they can't be confused with plan waypoints `1, 2, ...`.
- `Workspace.call` treats any function the same, so scratch functions are recorded with their name.

First run on real frames (`~/cal_right` frames 0 and 20, the head looking at the desk):

- `table_height`: 72.0 / 72.2 cm, spread 5 mm, **tilt 1.0–1.1 deg**: the calibration and the depth agree on "up"
  to about a degree. The calibration's `floor_z` is 77.8 cm, 6 cm higher; `collect.py` falls back to the arm base's
  height when no `--floor-z` is given, so it is probably not a measured value. TODO: measure the desk height once
  to settle which is right.
- Both frames warned of a second surface (95 / 89 cm): the arm and the cloth heap, not a shelf. The warning
  threshold (30% of the table's points) may be too low for a cluttered view.
- The table mask covered the desk and a strip of cloth at the same height.

Open questions for Step 2:

- **Units.** The API is in meters (repo convention); everything the VLM sees (labels, `describe()`) is in cm. The VLM
  will probably write cm back. Pick one before the agent loop: likely cm in the VLM-facing tools.
- **Grid label placement** works but is busy at the left edge in oblique views.
- **Robot links** are not yet excluded from `table_height` (they spread over many bins, so the peak held here).

## Step 2: sample renders

Driven by hand with `harness/scratch/session.py` (usage in its docstring), live from the robot, one session folder
per experiment (`harness/scratch/sessions/<name>/`, gitignored):

1. `scan <name>`: the harness phase: the head steps through 3 tilts x 3 pans from home, a capture and its
   `table_height` at each pose, then back home; a contact sheet and the tilt per pose (desk vs calibration).
2. `focus <name> capture4 [capture7]`: name the captures to work in (`view`, `view2`, ...): a 2 cm grid in each.
3. `analyze <name> "<snippet>"`: Python run in the session (tools recorded by the workspace, `show(view, ...)` saves a
   PNG, assigned names persist). Repeated until the renders answer what we need.
4. `propose <name> "x,y,z; ..." --views close --decide`: waypoints in cm, checked against depth, drawn in the full
   view and the named views; approve / reject is logged. Nothing moves.
5. `log <name>`: names, nodes, parents, staleness, decisions.

On a handful of saved scenes (a clear table, one object, an object next to an obstacle, an object near the edge of
the depth image), produce renders of the chain from the reasoning example:

1. full view + 2 cm grid
2. zoom on the object + 0.5 cm grid
3. `locate` points on the object with their xyz
4. a plan: one clean, one passing through an obstacle, one through a depth shadow (UNKNOWN)

## Step 3: is the rendered information valuable?

Decide what counts before looking at renders. Checks, roughly in order:

- **Readable by eye:** does the user judge the plan from the render alone, without other tools? Are the grid labels,
  waypoint numbers and FREE / UNKNOWN / BLOCKED colors legible at the size the model sees?
- **Geometrically right:** the grid lines up with the table (check against tape marks / a ruler); `locate` agrees
  with a measured position (error in cm); renders of the same plan in the full view and in a zoomed view agree.
- **Readable by the VLM:** given only the render, can the model (a) read off the grid cell / xyz of a target,
  (b) say which leg of a plan is blocked, (c) say whether a plan will hit something, matching `segment_check`?
  Compare the image alone, image + text summary, and text summary alone (tests 1.E).
- **Selecting a target (open question from section 0):** ask the model for a target three ways (world grid cell,
  numbered marks, raw pixel) on the same scenes, and score each by the 3D error after `locate`. Only `observe` /
  `zoom` / `locate` are needed.

## Later

- Run the full flow on the robot with the user approving each plan.
- `segment(text)` / candidate keypoints (SAM3 or GroundedSAM, as in MOKA / ReKep) when grid + depth fails on object
  edges or grasp points.
- Grasp yaw from depth (PCA of the object's depth blob), so the model doesn't estimate the angle.
- Automatic gate replacing user approval.

## Results

TODO: maybe this should be in a separate file once there are results.
