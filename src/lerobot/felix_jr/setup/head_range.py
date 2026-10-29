"""Set the head's tilt range and home by looking through the camera. Runs on your computer, against a robot server.

  uv run python -m lerobot.felix_jr.setup.head_range tcp://<pi>:5555 [--step DEG] [--name head_eyes]
                                                     [--torque-limit N] [--png]

Start the server on the Pi first (`server.server --left-port <PORT>`, the arm whose board the head plugs into). A
window shows the camera image after every move. Click it so it receives the keys, then:

  up / down arrow (or i / k)   tilt the head up / down by --step degrees (default 2); hold to repeat
  u                            record this position as the UP limit
  d                            record this position as the DOWN limit
  h                            record this position as HOME (where the head rests)
  z                            record this position as LEVEL (the camera looks straight ahead; for hand-eye work)
  g / l                        go to HOME / LEVEL
  r                            refresh the image without moving
  q or Esc                     quit (goes to HOME first if one is recorded)

Each recorded value is saved on the Pi at once, in `<id>_head_range.json` next to the arm's calibration, and
later moves are clamped to the limits. Move slowly towards the end of the range and record the limit *before* the
servo or the camera cable is strained. While the script runs the servo's torque is capped (--torque-limit, default
300 of 1000), so a servo that hits something pushes gently; the cap is lifted on exit.

The signs saved by `setup.head` decide which way is "down": a positive move tilts the camera down for the eyes.
With `--name head_neck` the arrows turn the head instead (up arrow = right, down arrow = left).
"""

import argparse
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.head import MAX_TORQUE_LIMIT, TICKS_PER_DEG
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError

# cv2.waitKeyEx codes for the arrow keys differ by platform (macOS, Linux GTK, Windows, Qt).
UP_KEYS = {63232, 65362, 2490368, 0, ord("i")}
DOWN_KEYS = {63233, 65364, 2621440, 1, ord("k")}
KEY_ACTIONS = {
    ord("u"): "limit_neg",
    ord("d"): "limit_pos",
    ord("h"): "home",
    ord("z"): "level",
    ord("g"): "go_home",
    ord("l"): "go_level",
    ord("r"): "refresh",
    ord("q"): "quit",
    27: "quit",
}
RECORD_LABELS = {"limit_neg": "up limit", "limit_pos": "down limit", "home": "home", "level": "level"}


def decode_key(key: int) -> str | None:
    """The action for a `cv2.waitKeyEx` code, or None for a key that does nothing.

    Actions: up, down, limit_neg, limit_pos, home, level, go_home, go_level, refresh, quit.
    """
    if key in UP_KEYS:
        return "up"
    if key in DOWN_KEYS:
        return "down"
    return KEY_ACTIONS.get(key)


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Copy of the RGB `image` with text lines in the top left corner (outlined so they read on any picture)."""
    out = image.copy()
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class HeadRangeSession:
    """The state of one run: moves, records, and shows a fresh image after each."""

    def __init__(
        self,
        client: FelixClient,
        name: str,
        step_deg: float,
        show: Callable[[np.ndarray], None],
        color_format: str = "jpeg",
    ):
        """`show` displays an RGB image (the script passes a cv2 window; tests pass a list's `append`)."""
        self.client, self.name, self.step_deg = client, name, step_deg
        self.show, self.color_format = show, color_format
        self.message = ""

    def status_lines(self) -> list[str]:
        """Text for the image: where the servo is and what has been recorded."""
        s = self.client.head_state()[self.name]
        lines = [f"{self.name}: {s['ticks']} ticks"]
        if s["level"] is not None:
            deg = s["sign"] * (s["ticks"] - s["level"]) / TICKS_PER_DEG
            positive = "down" if self.name == "head_eyes" else "left"
            lines[0] += f", {deg:+.1f} deg from level (+ is {positive})"
        recorded = ", ".join(
            f"{label} {s[key]}" for key, label in RECORD_LABELS.items() if s[key] is not None
        )
        lines.append(f"recorded: {recorded or 'nothing yet'}")
        if self.message:
            lines.append(self.message)
        return lines

    def refresh(self, after: float | None = None) -> None:
        """Fetch an image (taken after `after`, if given) and show it with the status text."""
        snap = self.client.capture(color_format=self.color_format, after=after)
        self.show(annotate(snap.color, self.status_lines()))

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        after = None
        try:
            if action in ("up", "down"):
                # the servo's positive direction is down (eyes) / left (neck)
                result = self.client.head_move(
                    self.name, delta_deg=self.step_deg * (1 if action == "down" else -1)
                )
                after = result["done_at"]
                self.message = (
                    "" if result["reached"] and not result["clamped"] else self._why_stopped(result)
                )
            elif action in ("go_home", "go_level"):
                result = self.client.head_move(self.name, to=action.removeprefix("go_"))
                after = result["done_at"]
                self.message = f"moved to {action.removeprefix('go_')}"
            elif action in RECORD_LABELS:
                self.client.head_record(self.name, action)
                self.message = f"saved {RECORD_LABELS[action]}"
            elif action == "quit":
                return False
        except ServerError as e:
            self.message = str(e)
        self.refresh(after)
        return True

    @staticmethod
    def _why_stopped(result: dict[str, Any]) -> str:
        if result["clamped"]:
            return "at the recorded limit"
        return "the servo stopped short: blocked?"

    def finish(self) -> None:
        """Go home if one is recorded, and lift the torque cap."""
        try:
            if self.client.head_state()[self.name]["home"] is not None:
                self.client.head_move(self.name, to="home")
        finally:
            self.client.head_torque_limit(self.name, MAX_TORQUE_LIMIT)


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--name", default="head_eyes", choices=["head_eyes", "head_neck"])
    parser.add_argument("--step", type=float, default=2.0, help="degrees per key press (2)")
    parser.add_argument(
        "--torque-limit", type=int, default=300, help="servo torque cap while running, 1-1000 (300)"
    )
    parser.add_argument("--png", action="store_true", help="lossless images (slower); default JPEG")
    args = parser.parse_args()

    window = "head range (click here, then use the keys; q quits)"

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    with FelixClient(args.endpoint, timeout_s=15.0) as client:
        if not client.server_info["head"]:
            raise SystemExit("this server has no head servos (start it with the left arm's port)")
        client.head_torque_limit(args.name, args.torque_limit)
        session = HeadRangeSession(client, args.name, args.step, show, "png" if args.png else "jpeg")
        try:
            session.refresh()
            while True:
                action = decode_key(cv2.waitKeyEx(100))
                if action is not None and not session.handle(action):
                    break
        finally:
            session.finish()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
