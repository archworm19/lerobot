"""Try out a running robot server from another computer: connect, read the state, grab some captures.

  uv run python -m lerobot.felix_jr.checks.server_check tcp://<pi>:5555 [--captures N] [--jpeg] [--save-dir DIR]

Prints what the server reports (commit, protocol, which parts exist), the joint state, then the time each
`capture` took and how much of the depth image is valid. With --save-dir it writes `color.png`, `depth_raw.png`
and `calibration.json` of the last capture, in the format `Scene.from_saved` and
`diagnostics.alignment_from_saved` read.

To try it with no robot, start a fake server on this computer
(`uv run python -m lerobot.felix_jr.server.server --fake`) and use the default endpoint.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT


def run(
    client: FelixClient,
    captures: int = 10,
    color_format: str = "png",
    save_dir: Path | None = None,
    say: Callable[[str], None] = print,
) -> list[float]:
    """Run the check against `client` and return the duration of each capture in seconds."""
    info = client.server_info
    say(f"Server: commit {info['commit']}, protocol {info['protocol']}, fake={info.get('fake')}")
    say(f"  arms {info['arms']}, head {info['head']}, base {info['base']}, camera {info['camera']}")
    state = client.get_state()
    for arm in info["arms"]:
        say(f"  {arm}: " + ", ".join(f"{k.removesuffix('.pos')} {v:.0f}" for k, v in state[arm].items()))
    if not info["camera"]:
        say("This server has no camera, so there is nothing to capture.")
        return []

    durations = []
    snap = None
    for i in range(captures):
        started = time.perf_counter()
        snap = client.capture(color_format=color_format)
        durations.append(time.perf_counter() - started)
        valid = 100 * float(np.mean(snap.raw_depth > 0))
        say(
            f"  capture {i + 1}/{captures}: {1000 * durations[-1]:.0f} ms, color {snap.color.shape}, "
            f"depth {snap.raw_depth.shape}, {valid:.0f}% of depth pixels valid"
        )
    if snap is None:
        return durations
    say(
        f"{captures} captures ({color_format}): median {1000 * statistics.median(durations):.0f} ms, "
        f"slowest {1000 * max(durations):.0f} ms"
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "color.png"), cv2.cvtColor(snap.color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / "depth_raw.png"), snap.raw_depth)
        (save_dir / "calibration.json").write_text(json.dumps(snap.calibration, indent=2))
        say(f"Saved color.png, depth_raw.png and calibration.json to {save_dir}")
    return durations


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--captures", type=int, default=10, help="how many captures to time (10)")
    parser.add_argument("--jpeg", action="store_true", help="ask for JPEG color instead of PNG")
    parser.add_argument("--save-dir", type=Path, help="write the last capture here")
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for each reply (15)")
    args = parser.parse_args()
    with FelixClient(args.endpoint, timeout_s=args.timeout) as client:
        run(client, args.captures, "jpeg" if args.jpeg else "png", args.save_dir)


if __name__ == "__main__":
    main()
