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
5. **The rollout batched** (`refine.rollout` / `outcome` over many environments): the refine and the bank on the GPU.
