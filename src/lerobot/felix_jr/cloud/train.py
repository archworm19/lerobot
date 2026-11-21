"""Train the grasp policy on a rented GPU pod: the start pool sent along, the code synced, the run in the background.

  uv run python -m lerobot.felix_jr.cloud.train --host root@<ip> --port <port> [--key <ssh key>] \
      [--pool src/lerobot/felix_jr/sim2real/scratch/pools/pyri1.npz] [--attach] [-- <sim2real.train's options>]

e.g. `-- --iterations 3000 --num-envs 8192 --object-noise-mm 1`. Through `cloud/run.py`: the pod's image checked
against this checkout's, the code rsynced (what git tracks or would), the pool sent (it is under `scratch/`, which the
sync leaves out), the commit and diff recorded with the run, then `python -m lerobot.felix_jr.sim2real.train --pool
<pool> ...` started on the pod: in the background by default (a training run outlives an ssh session; its output in
the run's `output.log`, the logs and checkpoints in its `train/`), or streamed here (`--attach`, fetched at the end).
Then, whenever: `python -m lerobot.felix_jr.cloud.run fetch --host ... --port ... <run id>` brings the run's folder
home (`cloud/scratch/runs/<run id>/`: `output.log`, `train/` with tensorboard's logs, the checkpoints, `eval.json`).
Stop the pod after (`POD_TTL` stops it in any case).
"""

import argparse
import sys
from pathlib import Path

from .run import ROOT, run

DEFAULT_POOL = Path("src/lerobot/felix_jr/sim2real/scratch/pools/pyri1.npz")


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="the pod's ssh user@address (RunPod: root@<ip>)")
    parser.add_argument(
        "--port", type=int, default=22, help="its ssh port (RunPod: the exposed TCP port for 22)"
    )
    parser.add_argument("--key", type=Path, default=None, help="the ssh key RunPod has (default: ssh's)")
    parser.add_argument(
        "--pool", type=Path, default=DEFAULT_POOL, help="the start pool (repository-relative)"
    )
    parser.add_argument(
        "--attach", action="store_true", help="stream the run here instead of in the background"
    )
    parser.add_argument("--any-image", action="store_true", help="run even if the pod's image is another tag")
    parser.add_argument("train", nargs=argparse.REMAINDER, help="after --: `sim2real.train`'s options")
    args = parser.parse_args(argv)
    extra = args.train[1:] if args.train[:1] == ["--"] else args.train
    pool = args.pool if not args.pool.is_absolute() else args.pool.resolve().relative_to(ROOT)
    if not (ROOT / pool).is_file():
        parser.error(f"{pool}: no start pool there (`python -m lerobot.felix_jr.sim2real.start_pool`)")
    command = ["python", "-m", "lerobot.felix_jr.sim2real.train", "--pool", str(pool), *extra]
    sys.exit(
        run(
            args.host,
            args.port,
            command,
            args.key,
            any_image=args.any_image,
            send=[pool],
            detach=not args.attach,
        )
    )


if __name__ == "__main__":
    main()
