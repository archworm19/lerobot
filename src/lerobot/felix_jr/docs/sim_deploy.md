# Simulation on rented GPUs (plan, 2026-11-20)

The simulator (`sim2real/`: the refine, the bank, the shadow's replays) runs on this Mac's CPU, one rollout per
worker. Training a grasp policy, or refining over thousands of draws, needs many environments stepped in parallel
on a GPU: mjlab (MuJoCo Warp, the GPU MuJoCo, with torch) on a rented GPU. Branch `sim_deploy`.

## Decided

**Software: mjlab 1.6** (released 2026-11-15). It brings torch (>=2.7: our lock's), `warp-lang`, `mujoco-warp`
3.11 and MuJoCo 3.11 (mjlab's main has moved to MuJoCo 4.0; we follow its releases). Training needs Linux and an
NVIDIA GPU; on this Mac MuJoCo Warp runs on the CPU (`warp`'s CPU device, Apple Silicon): slow, but whether it takes
our model and agrees with CPU MuJoCo is answered here.

**MuJoCo 3.11 for `felix_jr_sim`** (was 3.8.1). Upstream's `aloha`, `hilserl` and `libero` pin MuJoCo below 3.9
(gym-aloha, gym-hil, hf-libero), so they and `all` are declared conflicting extras (`pyproject.toml`,
`[tool.uv] conflicts`): resolved apart, never installed together. That needs uv 0.11 or later (CI pins 0.11.30;
the lock is written with it).

**Hardware:** the L40S (48 GB) for scale; first an RTX 4090 (the same chip, 24 GB, about half the price): Warp's
physics is FP32 and memory bandwidth, not tensor cores or HBM, so an H100 buys little. Measured, not assumed:
environment steps per dollar on both.

**Provider:** RunPod first (2026-11 list prices: L40S $0.79/h community, $1.09 secure; RTX 4090 $0.34 / $0.69); Vast.ai
cheaper (marketplace: 4090 from ~$0.27, L40S from ~$0.47) and less reliable. SkyPilot on top when there is more than one
pod (it provisions both, stops idle ones and syncs the code). A RunPod pod *is* a container: the image is given at
creation; Docker cannot run inside one (no daemon, no privileged mode), so images are built in GitHub Actions.

**Image:** GitHub's container registry, public (the repository is): public packages are free, storage and transfer
both; RunPod charges no ingress or egress. Dependencies only: a CUDA base, `uv sync --frozen --no-install-project`
with our extras and mjlab, `openssh-server` and `rsync` (sshd started at boot, port 22 exposed: RunPod's own templates
do this, a custom image does not), and the SO-101 model (`sim2real/assets.py`'s download, ~18 MB) baked in.
Rebuilt by CI only when `uv.lock` changes. Built from a CI checkout, so nothing under `scratch/` can reach it.

**Code:** not in the image. Each run rsyncs the tracked and untracked-not-ignored files (`git ls-files -co
--exclude-standard`: `src/` is ~10 MB, the diff seconds after the first) to the pod over SSH and runs
`uv run --frozen --no-sync python -m lerobot.felix_jr...` there; the run records the commit and `git diff`
(reproducible, as Ray's working-directory snapshot). Our own assets (the finger mesh, the banks: <1 MB) go with
the code; a bucket pays from ~1 GB or many pods reading the same data. **Outputs** (checkpoints, videos): a bucket
(Cloudflare R2: no egress fee) or a RunPod network volume; metrics to wandb (mjlab's).

## Steps

1. **MuJoCo 3.11 on the Mac. Done (2026-11-20): nothing moved.** Each replay run under 3.11.0 and 3.8.1
   (`uv run --with mujoco==3.8.1`): the shadow's 50 executions identical to the digit (the pinch points 1.4 mm
   mean apart at the attack, 1.9 loaded); the feno replays the same 11 of 12 (lifted or not; four failures' kind
   swapped between one jaw and slipped, one physics draw's count 1 -> 0 of 16); the grip trials (`grip_sim`, pyri6,
   feno2) the same stalls, the loads within 0.6 points. The felix tests pass (one timing test failed here only:
   macOS stretched its 50 ms sleeps to 90-150 ms with the machine busy; it does not touch MuJoCo). Nothing refitted.
2. **Our scene in MuJoCo Warp on the Mac's CPU:** what it refuses (the fingers' hinges and springs, the soft contacts,
   the mesh collisions), then a few environments' rollouts against CPU MuJoCo on the same replays.
   **First look (2026-11-20):** mjlab 1.6.1 installed (`--extra felix_jr_sim --extra felix_jr_gpu`; Warp's CPU
   device). `put_model` refuses one thing: **noslip** (`robot.py`: 3 iterations, with the elliptic cone and impratio
   10, for friction that holds); with it off the scene loads (meshes, the fingers' hinges and springs, the elliptic
   cone). The arm driven 25 deg by its servos for 2 s: Warp and CPU MuJoCo the same to 0.000 deg; the bottle settles
   on the table 0.7 mm apart. Warp on the Mac's CPU: ~4 ms a step (CPU MuJoCo 0.03 ms): correctness only.
   **Noslip off, on CPU MuJoCo** (the replays patched, nothing else): the grip trials' loads fall 4-6 points below
   the robot's (RMS ~5 and ~6 points, pyri6 and feno2; with noslip ~2 and ~3) and the jaw stalls ~0.5 units further
   closed; the lifts agree better (pyri6 now held, as on the robot; feno2 dropped, as it slipped); the feno replays
   10 of 12 (sim3 spot 2's refined, lifted on the robot, now slips; the physics draws' lifts 6, 6, 3 of 16 for the
   real lifts, 4, 0, 1, 2 for the real failures: as marginal as before). So: refit the contact without noslip
   (the finger model's give or the contact's softness, against the grip trials' loads), then the replays again.
   **Refitted without noslip (2026-11-20; `robot.py` now has none):** noslip also held the joints' friction loss
   (the arm servos' fitted friction) hard; without it they creep (the shadow's pinch points 2.3 / 2.0 / 2.8 mm mean
   at the attacks / grasps / loaded). Two changes: the joints' friction constraint stiff (`JOINT_FRICTION_SOLIMP`
   0.99-0.999; 0.004 s time constants scored the same) and the fingers' give 1.6 N m/rad (was 1.2; impratio 10, 30,
   100 alike at 1.6; 2.2 erratic). Against the model with noslip:

   | | with noslip | without |
   | --- | --- | --- |
   | shadow, pinch points mean (attack / grasp / loaded / other) | 2.0 / 1.4 / 1.9 / 1.5 mm | 1.7 / 1.4 / 1.8 / 1.5 mm |
   | grip trials, stall RMS (pyri6 / feno2) | 0.23 / 0.39 units | ~0.3 / ~0.5 |
   | grip trials, load RMS | 1.7 / 3.3 points | ~2.9 / ~4.4 (low) |
   | grip trials, lifts as the robot's | 1 of 2 | 2 of 2 |
   | feno replays (lifted or not) | 11 of 12 | 11 of 12 |
   | physics draws lifted: real lifts / real failures | 4, 8, 3 / 1, 2, 0, 4 of 16 | 7, 7, 5 / 0, 0, 0, 4 |

   MuJoCo Warp takes it (`put_model`); the arm the same to 0.000 deg, the bottle settling 0.7 mm apart.
3. **The image in CI** (GHCR) and the rsync-and-run script; a CPU smoke test (mjlab imports, our scene steps once).
   **Built (2026-11-20):** `docker/Dockerfile.felix_sim` (CUDA 12.8 base, Python 3.12, `uv sync --frozen
   --no-install-project --extra felix_jr_sim --extra felix_jr_gpu` into `/opt/venv`, sshd and rsync, the SO-101 model
   in `/opt/felix/so101`; the code is not in it: `PYTHONPATH=/workspace/lerobot/src`), `docker/felix_sim_start.sh`
   (RunPod's `$PUBLIC_KEY` into root's keys, sshd, stays up), `.github/workflows/felix_image.yml` (on a push to `main`
   or `sim_deploy` that changes what the image is built from, or by hand: builds unless the tag exists, pushes
   `felix-sim:<tag>` and `:latest`), `cloud/run.py` (below). The tag: a hash of `pyproject.toml`, `uv.lock`, the
   Dockerfile and the start script (`python src/lerobot/felix_jr/cloud/run.py tag`), the same in CI and the run
   script, which refuses a pod on another tag.

## Running on a pod

1. **The image:** `ghcr.io/archworm19/felix-sim:<tag>`, public (it took the repository's visibility through its
   source label: RunPod pulls it without credentials, at no cost). The first build (2026-11-20, tag `7b58f4797478`)
   took 10 minutes on a free runner.
2. **A pod:** RunPod, "Deploy" a GPU pod with a custom image, `ghcr.io/<owner>/felix-sim:<tag>` (`run.py tag` names the
   tag), TCP port 22 exposed, your ssh public key in RunPod's settings (it reaches the pod as `$PUBLIC_KEY`). The
   pod's "Connect" shows `ssh root@<ip> -p <port>`. The dev template (2026-11-21): `felix-sim:latest`, an RTX 4090,
   20 GB container disk and no volume, TCP 22 exposed (on a Public IP host), no start command (the image's starts sshd),
   `NVIDIA_DRIVER_CAPABILITIES=all` (now in the image too), `POD_TTL`, CUDA 12.8 and up (the cu128 wheels' minimum;
   a newer driver runs them).
3. **A run:** `uv run python -m lerobot.felix_jr.cloud.run --host root@<ip> --port <port> -- python -m
   lerobot.felix_jr.sim2real.<module> ...`: the code synced (what git tracks or would: `scratch/` stays home), the
   commit and `git diff` saved with the run, its output streamed, `$RUN_DIR`'s files fetched into
   `cloud/scratch/runs/<id>/`. Again after a code change: only the changes go.
4. **Stop the pod** when done (RunPod's console or `runpodctl pod stop <id>`): it bills until stopped, its disk until
   deleted. **`POD_TTL`** (the template's env, e.g. `4h`; `30m`, `1d`, plain seconds) makes it stop itself that long
   after it starts (`docker/felix_sim_start.sh`: `runpodctl pod stop` with the pod-scoped `$RUNPOD_API_KEY` RunPod
   gives every pod; `runpodctl` 3.0.1 in the image, its checksum checked); the run script prints when
   (`/opt/felix/stop_at`). Stopped, not deleted: the disk bills until the pod is deleted. A backstop: RunPod stops every
   pod when the prepaid balance reaches $0.
4. **One RTX 4090 on RunPod** (deleted after): the replays on CUDA, then environments per second against their count;
   the same on an L40S.
   **First run (2026-11-21, an L40S, `cloud/warp_check.py` through `cloud/run.py`):** the scene (the arm driven 25 deg,
   feno standing; no grasp) on CUDA the same as CPU MuJoCo to 0.0002 deg, the bottle 0.68 mm (as on the Mac's CPU).
   The worlds stepped together (one step a CUDA graph):

   | worlds | steps/s | x real time | GPU memory |
   | --- | --- | --- | --- |
   | 1 | 1,239 | 2 | |
   | 1,024 | 0.90 M | 1,795 | |
   | 4,096 | 2.57 M | 5,139 | 4.7 GB |
   | 8,192 | 3.30 M | 6,608 | 8.8 GB |
   | 16,384 | 3.91 M | 7,816 | 16.9 GB |
   | 24,576 | 4.03 M | 8,067 | 25.0 GB |

   Flat from ~16k worlds at ~4 M steps/s; ~1 MB a world (`NCONMAX` 64, `NJMAX` 400 per world: room to shrink); 32,768
   out of the 46 GB. The Mac's six CPU workers: ~0.2 M steps/s on this scene, so ~20x. An RTX 4090 (24 GB, ~20k
   worlds, the same chip) should reach the plateau at less than half the price: to measure. The grasps' contacts will
   cost more a step. The pod: `PUBLIC_KEY` was not passed (the key added in RunPod's settings after the deploy?), so
   the template now sets it itself.
5. **The simulator as an mjlab environment** (2026-11-21: the GPU is for training a policy with mjlab; the refine and
   the bank stay on CPU MuJoCo): `sim2real/mjlab_env.py`. The robot entity is `robot.arm_spec` itself, its fitted
   servos kept (`XmlActuatorCfg`); the object and the table as `scene` builds them; joint position targets every 17
   steps (~30 Hz, the server's waypoint rate); resets and randomization with `grasp_check.physics_draws`' ranges; the
   reward the object's rise (refine's lifted), the episode ended by a tip past 30 deg. mjlab drops a spec's options at
   attach and its defaults differ (an implicit integrator, a pyramidal cone, impratio 1, 50 CCD iterations): restated
   (`sim_cfg`). A fixed-base entity is wrapped in a mocap body at the origin: the arm keeps its calibrated pose in it.
   **Checked (Warp's CPU device):**
   - **The model:** the compiled model equals `scene.build`'s, every option and every joint, degree of freedom,
     actuator, geom and body (`model_differences`), and the bodies' positions at a pose (`pose_differences`): a smoke
     test (`smoke_tests/test_mjlab_env_smoke.py`).
   - **The feno trials replayed open loop** (each trial's CPU run recorded step by step, `Recording`, its commands
     played in a world of its own, `replay_in_env`): free moves and light contact 0.01-0.4 deg and 2-5 mm apart;
     the loaded and slipping grasps up to 2.5 deg and 10-15 mm (the contact solver is not CPU MuJoCo's to the bit, and
     open-loop commands do not correct a drift); the two knock-overs far apart, as chaos has it.
   - **Under physics draws** (12 trials x 8 draws: the fingers' friction and give, the object's offset; the mass
     nominal: varying it needs mjlab's recompute), each draw the same in its CPU run and its world
     (`set_world_physics`): lifted or not the same in 89 of 96; all 7 differences lifted on CPU, not in mjlab (14 of 96
     against 7). The trials that never lift never do in either. **mjlab's physics a little less grippy** (the replay's
     rises ~1.5 mm lower over the 3 cm lift); these feno grasps sit at the threshold, which magnifies it.
   - **The grip trials in mjlab against the robot** (pyri6, feno2: open loop by design, so the replay is the trial run
     in mjlab): the stalls as near as CPU MuJoCo's, the lifts both right, but the loads ~1.5 points lower at every
     close (stall / load RMS against the robot: pyri6 0.43 units / 4.2 points, CPU 0.28 / 2.9; feno2 0.52 / 4.5, CPU
     0.45 / 4.4): a lighter squeeze, less friction to hold with, the same bias as the lifts'. **Corrected:** mjlab's
     fingers' give x1.3 (`MJLAB_GIVE_SCALE`, mjlab only: `env_cfg`'s default; `give_scale=1.0` is the CPU model as is):
     pyri6 0.29 / 2.7, feno2 0.21 / 2.9 (nearer the robot than CPU MuJoCo), both lifts right (x1.15 and x1.5 scored
     worse on one or the other). The 96 draws again: lifted 11 in mjlab against 14 on CPU (7 before), the same in 89,
     the differences both ways now: within the counts' noise.
   **Next:** training (mjlab's PPO, `rsl_rl`): the policy's observations, the action's scale and the episode, then a
   first run on a pod; the server's streaming mode before a policy runs on the robot.
6. **The first experiment: a positive control** (2026-11-21, decided): actor and critic both privileged (the joints,
   the object's pose in the root and the gripper's frame with a noise setting starting ~1 mm, the servos' loads, the
   fingers' contacts, the last action); **delta joint targets** (the targets accumulate; ~1 deg a tick at 30 Hz, the
   gap to the reading capped) as the prominent sim-to-real projects mostly do (OmniReset: delta end-effector pose;
   ManiSkill's SO-100 sim-to-real: delta joint targets; IndustReal, AnyRotate, ClutterDexGrasp's arm: delta; only
   mjlab's and IsaacLab's lift templates absolute); starts from the bank's pools (OmniReset's four reset kinds are
   ours: reaching = attack, near-object = at_object, stable grasp = in_jaws / lifted; precomputed, no curriculum);
   staged positive rewards. One object: pyri (45 x 130 mm, 80 g), which the robot holds and we have.
   **The bank rebuilt for it** (test banks, 6 placements of pyri): first 1 of 6 anchored, 3 of 214 trials held. Two
   causes: the wrist-view filter refused every grasp deeper than 10 mm (`open_issues.md` 20; the bank turns it off,
   and the camera's clearance at the grasp, as the refine), and `grasp_check` judged a hold by the planned 30 mm lift
   while the simulated arm's jaws rise ~23-28 mm (its sag), so held grasps scored as slips (now: the object up 75 % of
   the jaws' rise). Then 6 of 6 anchored, at 0 to 27.5 mm deep, mostly the first grasp simulated, 6 s.
   **The bank and its starts** (2026-11-21): `pyri1` (48 placements of pyri, `bank --object 45 130 80`): 47 anchored,
   at 0-27.5 mm deep (most 20-27.5), 26 s on six workers. **The start pool** (`sim2real/start_pool.py`): 1,024 starts
   a pool (attack, at_object, in_jaws, lifted) drawn ahead on CPU MuJoCo (`starts.StartSampler`), 110 s on six
   workers, 414 KB: each joint's position and velocity and each actuator's command, by name. In the environment
   (`env_cfg(start_pool=..., pool_weights=...)`) a reset event (`reset_from_pool`) writes a start into each world;
   its commands go in as the robot's joint position targets (mjlab derives the controls from them at every step and
   reset; a write to the controls alone is overwritten). Held a second in mjlab, each pool's starts stay put: the
   object 0.5-2 mm (the table contact's settle, as CPU against Warp), the lifted ones still up, the arm under 0.8 deg,
   nothing unstable (smoke test: `test_start_pool_smoke.py`). The pool is under `scratch/` (gitignored): the run
   script does not send it; draw it on the pod (CPU) or send it along.
