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
3. **The image in CI** (GHCR) and the rsync-and-run script; a CPU smoke test (mjlab imports, our scene steps once).
4. **One RTX 4090 on RunPod** (deleted after): the replays on CUDA, then environments per second against their count;
   the same on an L40S.
5. **The rollout batched** (`refine.rollout` / `outcome` over many environments): the refine and the bank on the GPU.
