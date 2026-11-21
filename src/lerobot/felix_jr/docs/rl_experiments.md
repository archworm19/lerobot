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
world, held at the end and at any time, by start pool).
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
`train_pool` and `held_out`, each by start pool. Next: a watcher process beside the trainer, evaluating and filming
each checkpoint as it is saved (no pause in the training, no evaluation code in its process).

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
- **Read:** the positive control is solved, with privileged observations and 1 mm of noise on one object. Next,
  away from it: the object pose's noise towards the head's real error, a spatial holdout (a region of the table: an
  extrapolation test), the watcher process (each checkpoint evaluated on the held-out pool as it is saved), then more
  objects (a 40-60 mm range) and fewer privileged observations (the finger contacts dropped).
