# RL experiments (mjlab)

The grasp policy trained with reinforcement learning in mjlab (MuJoCo Warp) on rented GPUs: how the environment was
made to match our simulator and the robot, what was checked, how a run is launched, and each run's results. Newest run
last in the log. The deployment (the image, the pod, the code sync) is in [sim_deploy.md](sim_deploy.md); the
simulator itself in [sim2real_grasp.md](sim2real_grasp.md); the robot sessions it is checked against in
[sessions.md](sessions.md).

Code: `sim2real/mjlab_env.py` (the environment), `sim2real/mjlab_task.py` (the task's terms), `sim2real/start_pool.py`
(the starts), `sim2real/train.py` (training), `cloud/train.py` (a run on a pod).

## Running

```bash
# a start pool from a bank (CPU; ~2 min for 4,096 starts on six workers)
uv run --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.start_pool \
    src/lerobot/felix_jr/sim2real/scratch/banks/pyri1.jsonl src/lerobot/felix_jr/sim2real/scratch/pools/pyri1.npz --workers 6
# a run on a pod (the pool sent along, in the background); then its folder home
uv run python -m lerobot.felix_jr.cloud.train --host root@<ip> --port <port> -- --iterations 1500 --num-envs 4096
uv run python -m lerobot.felix_jr.cloud.run fetch --host root@<ip> --port <port> <run id>
# on the Mac's CPU, to catch wiring before paying for a GPU
uv run --extra felix_jr_sim --extra felix_jr_gpu python -m lerobot.felix_jr.sim2real.train \
    --pool src/lerobot/felix_jr/sim2real/scratch/pools/pyri1.npz --device cpu --num-envs 8 --iterations 3
```

A run's folder (`cloud/scratch/runs/<id>/`): `commit`, `diff.patch`, `command`, `output.log`, `exit_code`, and
`train/` (`settings.json`, tensorboard's events, `model_*.pt`, `eval.json`: the final policy played one episode in every
world, held at the end and at any time, by start pool; `evals/model_<iteration>.json`, `watch.log`: the watcher's).
**The watcher** (`sim2real/watch.py`; `sim2real/train.py` starts it beside itself on a GPU, `--no-watcher` to not): a
process of its own that evaluates each checkpoint as it is saved on the held-out pool (2,048 worlds, the same starts
every time; tensorboard's `held_out/<pool>/held_at_end` and `.../held_ever` curves) and films it. Training never waits
on it; sharing the GPU slows it by about the watcher's share of the work (a held-out pass is ~7-10 training
iterations' simulation: ~5-10 % at a checkpoint every 100 iterations) and takes ~2 GB more of its memory. A finished
run after the fact: `python -m lerobot.felix_jr.sim2real.watch <run>/train --once`. Without the watcher, the trainer
films in its own process (it waits) and evaluates the held-out pool at the end only.
At every checkpoint (`--video-every`, default each) the policy is filmed on a few random starts (`sim2real/videos.py`:
6 worlds, every pool equally likely, a side view of each world's object, each tile labelled with its pool and state)
into `train/videos/model_<iteration>.mp4` (~1 MB a world-episode), on the pod's GPU; a checkpoint after the fact:
`python -m lerobot.felix_jr.sim2real.videos <run>/train [--checkpoint model_1400.pt]`. Each filmed checkpoint is also in tensorboard's
Images tab at its iteration (`policy/episode`: an animated GIF, every other frame at half size, ~0.75 MB; `policy/sheet`:
the tiles at 0, 1.5, 4 and 11.5 s), so the slider steps through training (`--tensorboard` adds one after the fact).
Tensorboard: `ssh -L 6006:localhost:6006 -p <port> root@<ip> 'bash -lc "tensorboard --logdir /workspace/runs/<id>/train
--host 127.0.0.1 --port 6006"'` (live, http://localhost:6006), or `tensorboard --logdir cloud/scratch/runs` on fetched
runs (each a curve).

## The environment: our simulator in mjlab

### Built from our own model, not re-modeled

| mjlab piece | ours |
| --- | --- |
| robot entity | `robot.arm_spec` itself: the calibrated kinematics (`calib_20261119_joint.urdf`), the sprung Fin Ray fingers and their pads, the wrist camera's board, the collision groups |
| actuators | `XmlActuatorCfg`: the servos as fitted, untouched (`ARM_KP_NM_PER_RAD`, the gripper's gain and torque cap, `ARM_FRICTIONLOSS_NM`, `JOINT_FRICTION_SOLIMP`) |
| object entity | the upright cylinder as `scene` builds it; its pose the entity's initial state (mjlab resets a free body to that) |
| table entity | `scene_spec`'s table, in a body of its own |
| physics options | `robot.arm_spec`'s, restated in `MujocoCfg` (`sim_cfg`) |
| control | 17 physics steps a policy step (~30 Hz: the server's waypoint rate) |

**mjlab's traps** (each found and handled):

- **It drops a spec's options at attach, and its defaults differ:** an implicit integrator (ours Euler), a pyramidal
  friction cone (ours elliptic), impratio 1 (ours 10), 50 CCD iterations (MuJoCo's 35). All four change grasp friction
  or contact; restated in `sim_cfg`.
- **A fixed-base entity is wrapped in a mocap body at the origin**, its root body's position overwritten from
  `init_state.pos`: the arm keeps its calibrated pose inside the wrapper; the object's body sits at its origin with
  its pose in the initial state.
- **Its controls are derived from the robot's joint position targets** at every step and reset: a start's servo
  commands go in as targets (`set_joint_position_target`); a write to the controls alone is overwritten.
- **Its relative-position action steps from the joint reading**, so it cannot build a gap past one step: no squeeze,
  no lift against a load. Ours accumulates targets (below).

### Before mjlab: the simulator made MuJoCo Warp's

- **MuJoCo 3.11** (mjlab 1.6's; was 3.8.1): the shadow's 50 executions identical to the digit, the feno replays the same
  11 of 12, the grip trials' stalls the same and loads within 0.6 points: nothing refitted.
- **No noslip** (MuJoCo Warp has none; `put_model` refused it): the joints' friction constraint made stiff
  (`JOINT_FRICTION_SOLIMP` 0.99-0.999) and the fingers' give refitted, 1.2 -> 1.6 N m/rad. Against the model with noslip:

| | with noslip | without (refitted) |
| --- | --- | --- |
| shadow, pinch points mean (attack / grasp / loaded / other) | 2.0 / 1.4 / 1.9 / 1.5 mm | 1.7 / 1.4 / 1.8 / 1.5 mm |
| grip trials, stall RMS (pyri6 / feno2) | 0.23 / 0.39 units | ~0.3 / ~0.5 |
| grip trials, load RMS | 1.7 / 3.3 points | ~2.9 / ~4.4 |
| grip trials, lifts as the robot's | 1 of 2 | 2 of 2 |
| feno replays, lifted or not | 11 of 12 | 11 of 12 |
| physics draws lifted: real lifts / real failures | 4, 8, 3 / 1, 2, 0, 4 of 16 | 7, 7, 5 / 0, 0, 0, 4 |

### Validation: mjlab against the CPU simulator and the robot

| check | how | result |
| --- | --- | --- |
| **the model** | the compiled model against `scene.build`'s, field by field (`model_differences`): every option, joint, degree of freedom, actuator, geom, body; the bodies' positions at a pose (`pose_differences`) | **identical** (smoke test `test_mjlab_env_smoke.py`) |
| **free motion** | the arm driven 25 deg by its servos for 2 s, MuJoCo Warp against CPU MuJoCo (`cloud/warp_check.py`) | 0.0002 deg apart (the Mac's CPU and an L40S alike); the bottle settles 0.7 mm apart |
| **the feno trials, open loop** | each trial's CPU run recorded step by step (`Recording`), its commands played in an mjlab world (`replay_in_env`) | free moves and light contact 0.01-0.4 deg / 2-5 mm apart; loaded and slipping grasps drift to 2.5 deg / 10-15 mm (the contact solver is not CPU MuJoCo's to the bit; open-loop commands never correct a drift); knock-overs far apart, as chaos has it. So: compared statistically |
| **under physics draws** | 12 trials x 8 draws (friction, give, offset; mass nominal), each draw the same in its CPU run and its world (`set_world_physics`) | lifted or not the same in 89 of 96; all 7 differences lifted on CPU only (14 against 7): **mjlab less grippy** |
| **the grip trials against the robot** | pyri6, feno2 (open loop by design: the replay is the trial in mjlab) | stalls as near as CPU MuJoCo's, lifts right, but the loads ~1.5 points lower at every close (stall / load RMS: pyri6 0.43 / 4.2, CPU 0.28 / 2.9; feno2 0.52 / 4.5, CPU 0.45 / 4.4): a lighter squeeze, the lifts' bias |
| **the correction** | mjlab's fingers' give x1.3 (`MJLAB_GIVE_SCALE`, mjlab only; x1.15 and x1.5 worse on one trial or the other) | the grip trials pyri6 0.29 / 2.7, feno2 0.21 / 2.9 (nearer the robot than CPU MuJoCo), both lifts right; the 96 draws lifted 11 against 14, the differences both ways (within the counts' noise). A smoke test checks the give is the one model difference |

### The starts: the bank, through a pool

- **The bank** (`bank --object 45 130 80`: pyri, which the robot holds and we have; `pyri1`, 48 placements): 47
  anchored, 0-27.5 mm deep (most 20-27.5), 26 s. Two fixes first (a test bank anchored 1 of 6): the wrist-view filter
  refused every grasp deeper than 10 mm (off in the bank; `open_issues.md` 20 for the harness), and `grasp_check`
  judged a hold by the planned 30 mm lift while the simulated arm's jaws rise ~25 mm (its sag): now by the jaws' rise.
- **Several grasps a placement** (`bank --anchors N`, 2026-11-21): OmniReset computes 1,000 feasible grasps an object
  and found narrower grasp sampling worse; `pyri1` had one grasp a placement (47 in all), however many starts were
  drawn. Now `grasp_check.SimAccept` keeps accepting until N distinct grasps (by depth, height, tilt, roll), each an
  anchor (`anchors` in the line; `StartSampler` takes them all). A test bank: 4 placements, 16 anchors, 0-27.5 mm deep,
  ~11 s a placement. `pyri2`: 512 placements x up to 4 anchors, its pool 4,096 starts a pool (building).
- **The pool** (`start_pool`): 1,024 starts a pool (attack, at_object, in_jaws, lifted: OmniReset's reaching,
  near-object and stable-grasp resets) drawn ahead on CPU (`starts.StartSampler`: the attack's arrival as the real arm
  arrives), saved by joint and actuator name; a reset event writes one into each world. **Checked:** held a second in
  mjlab, each pool's starts stay put (the object 0.5-2 mm, the lifted ones still up, the arm under 0.8 deg; smoke test
  `test_start_pool_smoke.py`).

## The first experiment: a positive control

The easiest version that should work: if it does not learn, the environment is wrong, not the RL.

- **Actor and critic both privileged** (42 numbers): the joints' positions and velocities; the object's center and axis
  in the root frame and in the pinch frame, noise `object_noise_m` 1 mm (the axis a tenth); the servos' loads (torque
  over stall: the real servos report a load); each finger's pads touching the object; the last action.
- **Action: delta joint targets** (`DeltaJointTargetAction`): targets accumulate, 1 deg a step (30 deg/s), each within
  5 deg of its reading; the jaw 2 units a step within 15. The caps: twice the sessions' settled gaps (0.9-2.7 deg; the
  jaw 3 units on an object), the 50 N m/rad servos at full torque within them. The projects surveyed mostly use delta
  actions (OmniReset: end-effector pose; ManiSkill's SO-100 sim-to-real: joint targets; IndustReal, AnyRotate,
  ClutterDexGrasp's arm); only mjlab's and IsaacLab's lift templates are absolute.
- **Starts:** the pyri pool, `POOL_WEIGHTS` attack 0.2, at_object 0.5, in_jaws 0.2, lifted 0.1; the physics randomized
  at startup (the fingers' friction and give, the object's mass: `grasp_check.physics_draws`' ranges).
- **Rewards** (positive and staged; `REWARD_WEIGHTS`, per step x the step's length): reach the grasp point 0.2 (exp of
  minus the distance over 5 cm), between the jaws 0.3, both fingers on it 0.5, its rise above resting towards the 3 cm
  lift 1.0, held (75 % of the lift, both fingers on) 2.0; action rate -0.01; an episode ended by a failure -5.
  **Terminations:** the arm on the table, the object tipped past 30 deg, 12 s (not success: holding keeps paying).
  The grasp point and the box "between the jaws" are measured from the pool, where the object's axis passes the pinch
  point (the grasps are at every height of the bottle): in the jaws (-16, 1, -11) mm, the jaws open (-30, 2, -12), at
  the attack (-30, -1, +34). On the pool's starts: reach 0.34 / 0.69 / 0.88 / 0.82, between 0 / 0.88 / 0.94 / 0.81
  (attack, at_object, in_jaws, lifted). Driven by hand: a closing jaw takes the object in every world; each in_jaws
  start lifted by its anchor's lift joints keeps it, up 63 % of the lift (the arm's sag): held asks for more than
  the plan.
- **PPO** (rsl_rl through mjlab's runner): its lift-cube task's settings, actor and critic 512-256-128 ELU,
  observations normalized, adaptive learning rate from 1e-3, 24 steps a world an update, 4 mini-batches x 5 epochs,
  entropy 0.005, gamma 0.99, lambda 0.95.

## Log

| run | where | worlds x iterations | time | held ever (all / attack / at_object / in_jaws / lifted) | held at the end |
| --- | --- | --- | --- | --- | --- |
| `train_smoke` | Mac CPU | 8 x 3 | 4 s | 0 (wiring only) | - |
| `20261121_133937` | RTX 4090 | 4,096 x 20 | 37 s (1.8 s an iteration) | 81 / 0.3 / 97 / 98 / 99.8 % | (not measured: the bug below) |
| `20261121_134628` | RTX 4090 | 4,096 x 1,500 | 42 min (1.67 s an iteration) | 99.2 / 99.5 / 98.7 / 99.9 / 100 % (train pool) | 98.0 / 98.7 / 98.7 / 96.9 / 95.4 % (train pool) |
| `20261121_144032` | RTX 4090 | 4,096 x 1,000 | 27 min (1.6 s an iteration) | train 99.7 / 99.4 / 99.5 / 99.8 / 99.9 %; **held out 99.9 / 99.4 / 100 / 100 / 100 %** | train 97.5 / 98.4 / 99.5 / 96.0 / 95.9 %; **held out 98.3 / 98.7 / 100 / 97.5 / 97.1 %** |
| `20261121_205721` | RTX 4090 | 4,096 x 1,000 | 29 min (1.7 s an iteration, the watcher beside it) | train 99.9 / 100 / 99.9 / 99.8 / 99.9 %; **side held out 98.7 / 95.8 / 99 / 100 / 100 %** | train 97.3 / 98.1 / 99 / 96.5 / 95.8 %; **side held out 95.6 / 90.5 / 99 / 96.5 / 96.4 %** |
| `20261122_171622` | RTX 4090 | 4,096 x 700 (stopped) | 27 min (2.25 s an iteration, the watcher beside it) | **multi-size held out 99.3 / 99.6 / 100 / 98.4 / 99.2 %** | **multi-size held out 96.9 / 98.5 / 99.6 / 94.1 / 95.4 %** |
| `20261122_174436` | RTX 4090 | 4,096 x 700 | 25 min (the watcher beside it) | train 96.2 % at the end; **no finger contacts, held out 99.4 / 99.8 / 100 / 99.2 / 98.6 %** | **no finger contacts, held out 97.1 / 98.3 / 99.8 / 95.7 / 94.6 %** |
| `20261122_181122` | RTX 4090 | 4,096 x 700 | 25 min (the watcher beside it) | train 88.0 % at the end; **estimated pose, held out 98.5 / 97.9 / 98.8 / 98.4 / 99.0 %** | **estimated pose, held out 88.9 / 88.8 / 95.4 / 84.1 / 87.1 %** |

### 2026-11-21: the first runs (the positive control)

- **Wiring, Mac CPU** (`train_smoke`, 8 worlds, 3 iterations): end to end, the checkpoints, tensorboard's events and
  `eval.json` written. A smoke test (`test_train_smoke.py`, 4 worlds, 2 iterations) keeps it so.
- **20 iterations, RTX 4090** (`20261121_133937`; pod image `270a85eccbe5`, commit `a7bb64f63`): 1.8 s an iteration
  (4,096 worlds x 24 steps x 17 physics steps: ~0.9 M physics steps a second, learning included). The easy starts
  learned at once: held at some time from 97-100 % of at_object, in_jaws and lifted starts; from the attack 0.3 %.
  Episodes mostly ran their 12 s; ~4 ended on a tipped object and fewer than 1 on the table, each iteration (of
  ~170 episodes). "Held at the end" read 0 everywhere: every evaluation episode times out at its last step and mjlab
  resets the world within it, so the end was read after the reset; fixed (`f0a9ce27f`: held from the step before).
- **1,500 iterations, RTX 4090** (`20261121_134628`, the pool `pyri1`: 47 grasps; mix 0.2 / 0.5 / 0.2 / 0.1): 42 min.
  The curves plateaued by iteration ~400-700 (held 1.84-1.87 of 2.0 a second of episode, both fingers 0.47 of 0.5,
  rise 0.92-0.94, episodes their full 12 s, tipped terminations ~5 -> ~0.25 a batch); PPO's adaptive learning rate
  fell to 1e-5. The evaluation (4,096 worlds, the training pool: not held out): held at the end 98.0 % (attack 98.7,
  at_object 98.7, in_jaws 96.9, lifted 95.4), at some time 99.2 %. **The positive control works**, the attack's starts
  included (0.3 % at iteration 20). **But the reward was too easy:** filmed (checkpoint 1,000, `sim2real/videos.py`
  after the fact), every world holds by 1.5 s, the attack's too, and many bottles hang tilted 20-30 deg, gripped high
  and swinging: "held" asked only 22.5 mm of rise and both fingers on, the episode ending past 30 deg. Changed for the
  next run (`e23497965`): held only upright within 10 deg, and a penalty on the object's angular speed while lifted
  (-0.2 a rad/s).

### Held-out starts (2026-11-21)

The evaluation so far was on the training pool. Now a split by placement (`start_pool --holdout 0.15`: a placement's
starts are near copies of each other, so a split by start would leak): `<pool>_train.npz` and `<pool>_test.npz`, the
held-out placement indices recorded in each. The held-out starts are in an environment of their own, which PPO never
steps: its rollouts, its updates and its observation normalizer (which rsl_rl updates from every world it steps in
training mode) only ever see the training pool's worlds; the held-out evaluation runs the frozen policy (eval mode).
Masking the held-out worlds' updates inside the training batch would still leak through the normalizer. The final
checkpoint is evaluated, none chosen by its held-out score (choosing would need a third split). `eval.json`:
`train_pool` and `held_out`, each by start pool. Since, the watcher (see "Running") evaluates every checkpoint on the
held-out pool as it is saved, beside the trainer, and `eval.json`'s `held_out` is its result for the last.

### 2026-11-21: the bigger bank, held out, upright

- **`20261121_144032`** (RTX 4090, 1,000 iterations x 4,096 worlds, 27 min; commit `b292a4e12`): the pool `pyri2_train`
  (16,384 starts from 1,466 grasps on 405 placements; `pyri2`: 512 placements x up to 4 grasps, 476 anchored, 1,818
  grasps) with the uniform mix; held only upright (10 deg) and the swing penalty; evaluated on `pyri2_test` too
  (4,096 starts from 267 grasps on 71 placements never trained on, in an environment of its own). **Held at the end:
  the training pool 97.5 %, held out 98.3 %** (attack 98.4 / 98.7, at_object 99.5 / 100, in_jaws 96.0 / 97.5, lifted
  95.9 / 97.1): no gap; the policy generalizes to new placements of the same object on the same table (interpolation:
  the held-out placements lie among the trained ones). Filmed (held-out starts): the bottles held upright now, the
  tilted swinging grips gone; the grips still high on the bottle. The checkpoints' films in its tensorboard after the
  fact (`videos --tensorboard`, on the pod). The pod stopped itself after (`runpodctl pod stop` with its pod-scoped key:
  `POD_TTL`'s mechanism works).
- **Read:** the positive control is solved, with privileged observations and 1 mm of noise on one object.

### 2026-11-21: a spatial holdout (the side's high quarter)

- **`20261121_205721`** (RTX 4090, 1,000 iterations x 4,096 worlds, 29 min; commit `a555978ee`): the pool
  `pyri2side_train` (16,384 starts from 1,313 grasps on 357 placements) against `pyri2side_test` (4,096 starts from
  417 grasps on the 119 placements 80-150 mm to the arm's side, `start_pool --split side --holdout 0.25`: where the arm
  never trained, turned further); otherwise as `20261121_144032`. The first run with the watcher: each checkpoint on
  the held-out band as it was saved, training ~8 % slower for it (29 min against 27). **Held at the end: the training
  pool 97.3 %, the side held out 95.6 %**: a small gap, nearly all of it the attack starts (90.5 against 98.1 %: the
  approach from where it never trained); from at the object, in the jaws or lifted the band makes no difference (99.0,
  96.5, 96.4 %).
- **The held-out curve** (the watcher's, held at the end, all / attack, %): 100 it. 91.9 / 83.7, 200 95.4 / 91.3,
  300 96.5 / 95.3, **400 97.4 / 95.1**, 500 96.9 / 95.1, 600 96.4 / 91.1, 800 96.6 / 94.0, 999 95.6 / 90.5. It peaks at
  400 and drifts down while the training pool's holds (2,048 worlds: past the noise, ~0.4 %, overall), some
  overfitting to the trained region; choosing a checkpoint by it would need a third split.
- **Read:** the policy extrapolates in angle, a little worse on the approach. More of the table in training (the bank's
  reach and side as wide as the arm allows) is the cheap fix; the size grid next.

### 2026-11-22: objects of several sizes

- **`20261122_171622`** (RTX 4090, 4,096 worlds, stopped at 700 iterations, 27 min; commit `57342aa30`): the pool
  `multi1_train` (16,384 starts from 2,382 grasps on 791 placements; cylinders 40-60 mm x 100-200 mm, 40-120 g, each
  world's object resized at its reset) against `multi1_test` (4,096 starts on 140 placements never trained on, random
  split); the size observed with a 2 mm error an episode (the critic the truth). **Held at the end on the held-out
  placements: 96.9 %** (attack 98.5, at the object 99.6, in the jaws 94.1, lifted 95.4; held at any point 99.3 %):
  one policy for the whole size range, as good as one object's runs (`pyri2`: 98.3 %). The held-out curve (all, %):
  100 it. 88.9, 200 95.3, 300 96.8, then flat (96.3-96.9 to 700): stopped there. A first attempt
  (`20261122_164732`) crashed at its first reset (a host tensor mixed into a GPU batch's check: fixed, `d4984cc7e`),
  a second ran at 3.0 s an iteration (the constants' refresh: fixed, `57342aa30`; profiled above); stopped at 250.
- **The drops: the policy opens the gripper.** Most of the shortfall is starts already gripping (in the jaws, lifted:
  held at the end ~94-95 % though held at some point 98-99 %). Instrumented on `20261122_164732`'s `model_200`
  (45 in-jaws / lifted worlds, CPU): every one of the 6 held-then-lost episodes lost "held" to a finger coming off,
  with the object still fully lifted (rise 1.00) and upright (2-7 deg); the gripper's target had opened first (0.31-0.46
  rad to 0.66-1.03, the jaws following), mostly late (steps 230-340 of 360); ordinary physics draws (pad friction
  0.31-0.34 in a 0.25-0.39 range, 62-88 g, 42-57 mm). Not slip: the policy's gripper opens.
  **Not a creep, a random walk** (checked on `20261122_171622`'s `model_700`, 128 in-jaws / lifted worlds, episodes
  doubled to 24 s): the gripper target's mean change while held is nil (+0.02 mrad a step in the 114 worlds that kept
  holding, +0.00 in the 13 that lost it) but it jitters (90th percentile +7.5 / +8.4 mrad a step); the targets
  accumulate (the delta action), so a zero-mean jitter wanders, and now and then opens the jaws past contact. Holding
  decays with time: of the worlds held at some point, 98.4 % held at 6 s, 94.5 % at 12 s (the trained episode), 96.9 %
  at 18 s (some regrasp), 89.8 % at 24 s. Nothing pays for keeping it closed (held needs contact, not force; the policy
  sees no time; a time-out is no failure). On the robot, where a hold lasts longer than 12 s, it would drop bottles.
  Fixes to weigh: a squeeze term while held (the gripper servo's load) or a penalty on opening while lifted; an
  absolute or latched gripper command (no random walk).

### 2026-11-22: experiment A, the actor without finger contacts

- **`20261122_174436`** (RTX 4090, 4,096 worlds x 700 iterations, 25 min; commit `b3421f72f`): as `20261122_171622`
  (multi-size, `multi1`) but the actor without `finger_contacts` (`--actor-exclude finger_contacts`; the critic keeps
  them): the claw has no touch sensors. **Held at the end on the held-out placements: 97.1 %** (attack 98.3, at the
  object 99.8, in the jaws 95.7, lifted 94.6; the training pool 96.2 %), against 96.9 % with them. The curve (all, %;
  with contacts in brackets): 100 it. 87.8 (88.9), 200 94.4 (95.3), 300 96.6 (96.8), 400 96.6 (96.3), 500 97.4 (96.4),
  600 95.6 (96.4), 699 97.1 (96.9 at 700).
- **Read:** the contact flags carried nothing the policy could not get otherwise: what the robot does measure (the
  gripper's position against its target, the servos' loads, the joints) tells it as much, with the object's live pose
  still told. Whether that holds without the live pose is experiment B2. The drops (in the jaws, lifted: ~95 %) are
  unchanged: the gripper's random walk is not about contact sensing.

### 2026-11-22: experiment B1, the actor told a pose estimated at the reset (finger contacts kept)

- **`20261122_181122`** (RTX 4090, 4,096 worlds x 700 iterations, 25 min; commit `7319a57cd`): the pools `multi1p`
  (`multi1` drawn again, each start with where perception put its object: an attack start's gripper move), the actor
  told the pose estimated at the episode's reset and frozen (`--object-pose estimated`: the true center plus the
  start's own perception offset plus 3 mm of noise, the axis tilted ~2 deg from vertical; in the root frame and in the
  live pinch frame), the critic the truth; finger contacts kept. **Held at the end on the held-out placements: 88.9 %**
  (attack 88.8, at the object 95.4, in the jaws 84.1, lifted 87.1; the training pool 88.0 %), against 96.9-97.1 % told
  the live pose; held at some point 98.5 % (about as often as with the live pose). The curve (all, %): 100 it. 79.0,
  200 83.9, 300 84.3, 400 87.4, 500 88.0, 600 88.3, 699 88.9: still rising slowly.
- **Read:** without the live pose the policy still gets the object nearly always; it keeps it less: the starts already
  gripping hold to the end 84-87 % (95 % with the live pose). Unable to see the object move in its grip, the gripper's
  random walk (above) costs more. A squeeze term or a latched gripper (the drops' fixes) would likely help most here;
  a live but noisy, delayed pose (the wrist camera's tracking) is the other lever.

### Next (decided 2026-11-21)

- **The object pose's noise stays at 1 mm.** The head camera's error and the kinematics' are already in the starts: each
  attack start is the gripper moved off the object by a perception offset (`perception_mean_m`, `perception_std_m`) and
  the joints by an execution spread (`starts.py`, `attack_joints`). The observation noise is another thing: what the
  actor is told about the object every step, which nothing tells the wrist policy on the robot (it sees the object). It
  matters only for distillation (a teacher leaning on precision the wrist-camera student cannot have); revisit it if
  the student shows that gap.
- **The watcher** (done: "Running").
- **A spatial holdout:** placements split by a region, not at random (`start_pool --split ahead|side|diameter|height
  --holdout-at high|low|middle`): an extrapolation number beside the random split's interpolation one. In `pyri2`'s
  476 anchored placements reach spans only 0.22-0.35 m ahead (its farthest 15 % a 2 cm band), the side -0.15 to
  +0.15 m: the first, `pyri2side`, holds out the side's high quarter (the arm turned past where it trained). `--plot` draws
  the placements (`placement_plot.py`: the table from above and the sizes, training / held out / unanchored; with sizes
  that vary, the table again per diameter and height range), `--plot-only` to preview a split without drawing pools.
  `b1` (24 placements over 40-60 mm x 100-200 mm) already shows the short wide objects mostly unanchored.
- **Objects of several sizes.** A bank holds one object (`--object D H G`); its grasps are in absolute units (depth mm,
  height along the object cm), and its in_jaws and lifted starts valid for that size only, so a bank per size on a
  grid (diameters 40 / 50 / 60 mm x heights 90 / 120 / 150 mm, the corners and the centre first), sizes between held
  out (e.g. 45 x 105). One pool across the grid, each start with its object's size (`StartPool.object_size`); one
  cylinder in the environment, resized per world at reset (`mdp.dr.geom_size`, which recomputes `geom_rbound` and
  `geom_aabb`), its mass and inertia to match (`dr.pseudo_inertia`, not `dr.body_mass`, which leaves the inertia); the
  size in the privileged observations.
  **Built (2026-11-21):** `StartPool.objects` (a size a start), the per-world resize at reset
  (`pool_reset.PoolReset`, a class event term bound to the robot and the object: size, bounding sphere and box, mass times a
  log-uniform draw, a solid cylinder's inertia; `set_const` after; one object's pool declares no field per world). The
  model is where each world's size is kept (`mjlab_object.object_dims`, `rest_z` read it); the term keeps only each
  world's start (`rows`, `pools_of_worlds`) and the size error. Files: the bank's `.jsonl` is `bank.py`'s, the pool's
  `.npz` `StartPool`'s (`start_pool.FORMAT`, checked when read; no mjlab needed), the term reads a pool.
  The resting height and `between_jaws`' height per world, the radius and height observed. The size the actor is told is off by an error drawn at each reset and kept for
  the episode (`SIZE_ERROR_M` 2 mm: a camera's estimate is off by a bias, not noise that averages away; SimToolReal
  perturbs its bounding box alike); the critic is told the true size. The grasp point and the box between the jaws
  refit against the radius over `multi1`'s starts (`mjlab_object.GRASP_POINT_FIT`, `BETWEEN_X_FIT`: in the jaws the axis
  sits x = -0.79 r + 2.0 mm from the pinch point, residual 1.0 mm; the fixed -16 mm was 6 mm off at 30 mm).
  **`multi1`** (1,024 placements, 40-60 x 100-200 mm, 40-120 g, up to 4 grasps each): 931 anchored; the unanchored
  are short objects close in (under ~260 mm ahead, mostly 100-133 mm tall, worst wide and short). Pools (random split,
  15 %): `multi1_train` 16,384 starts from 2,382 grasps on 791 placements, `multi1_test` 4,096 from 459 on 140.
  **The object's responsibilities gathered (2026-11-22):** what it *is* in `harness/primitives.py`: a `Primitive` is
  geometry, batched (N objects: parameters and poses; centroid, volume, bounds, inertia per kg, surface distance), the
  same code on numpy (planning) and torch (mjlab, on the GPU; `array-api-compat`); a `GraspFamily` is planning's grasps
  of one object; an `ObjectRecord` is primitives with masses, written down (the bank's placements and the pools' rows
  both go through it: pool format 3 names the shape and its columns); how it lives in mjlab in
  `sim2real/mjlab_object.py`: `MjlabCylinder`, a stateless configuration built from the object's record (construction:
  `entity_cfg`; contacts: `contact_match`, for the finger pads' sensors; reset: `declaration`, `write`,
  `draw_size_error`; observations; geometry: `pose`, `dims`, `rest_z`, `upright`; and the object in our claw:
  `in_pinch`, `axis_from_pinch`, `grasp_point`, `between_jaws`). `mjlab_task`'s rewards and terminations are generic
  over the object (each takes it in its params and asks it for what its stage needs); `PoolReset` draws starts, writes
  state, and has the object write itself.
  **Profiled (2026-11-22, `sim2real/reset_bench.py`, RTX 4090, 4,096 worlds, random actions):** a multi-size step
  took 104.7 ms against one object's 70.6, nearly all of it the constants' refresh after a reset (`set_const` over
  every world, ~31 ms; every step resets some world at 4,096); the solver's iterations were the same (the slowest
  world's median 8). The object now writes the constants that change with it itself (a free body's closed forms:
  subtree mass, body and dof inverse weights; a test holds them equal to `set_const`'s), and nothing is recomputed:
  69.2 ms against 67.7. The first multi-size run (`20261122_164732`, stopped at ~iteration 250 to profile) ran at
  3.0 s an iteration; held out 90.7 % at 100 iterations, 95.3 % at 200 (on par with one object's runs).
- Then fewer privileged observations (the finger contacts dropped).
