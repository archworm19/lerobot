"""Set a head servo's range and home by looking through the camera. Runs on your computer, against a robot server.

  uv run python -m lerobot.felix_jr.setup.head_range tcp://<pi>:5555 [--name head_eyes | head_neck] [--step DEG]
                                                     [--torque-limit N] [--png]

Start the server on the Pi first (`server.server --left-port <PORT>`, the arm whose board the head plugs into). A
window shows the camera image after every move, with the keys listed on it. Click it so it receives the keys, then:

  --name head_eyes (tilt, the default)          --name head_neck (turn)
  up / down arrow (or i / k)  tilt up / down     left / right arrow (or j / l)  turn left / right
  u                           record UP limit    [                              record LEFT limit
  d                           record DOWN limit  ]                              record RIGHT limit

  both:  h  record this position as HOME (where the head rests; for the neck: looking straight ahead)
         g  go to HOME        r  refresh the image without moving        q or Esc  quit (goes HOME first)

Moves are --step degrees (default 2); hold a key to repeat.

Each recorded value is saved on the Pi at once, in `<id>_head_range.json` next to the arm's calibration, and
later moves are clamped to the limits. Move slowly towards the end of the range and record the limit *before* the
servo or the camera cable is strained. While the script runs the servo's torque is capped (--torque-limit, default
300 of 1000), so a servo that hits something pushes gently; the cap is lifted on exit.

The signs saved by `setup.head` decide which way is positive: down for the eyes, left for the neck. So the down
limit and the left limit are each servo's `limit_pos`, the up and right limits its `limit_neg`.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.head import MAX_TORQUE_LIMIT, TICKS_PER_DEG
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError

# cv2.waitKeyEx codes for the arrow keys differ by platform (macOS, Linux GTK, Windows, Qt).
UP_KEYS = {63232, 65362, 2490368, 0, ord("i")}
DOWN_KEYS = {63233, 65364, 2621440, 1, ord("k")}
LEFT_KEYS = {63234, 65361, 2424832, 2, ord("j")}
RIGHT_KEYS = {63235, 65363, 2555904, 3, ord("l")}
COMMON_KEYS = {ord("h"): "home", ord("g"): "go_home", ord("r"): "refresh", ord("q"): "quit", 27: "quit"}


@dataclass(frozen=True)
class ServoKeys:
    """How one servo is driven from the keyboard: two move directions and two limit keys."""

    positive: tuple[str, set[int]]  # (action, keys) for the servo's positive direction (down / left)
    negative: tuple[str, set[int]]
    records: dict[int, str]  # key -> limit_pos / limit_neg
    labels: dict[str, str]  # limit_pos / limit_neg / home -> what you see
    help: str


SERVOS = {
    "head_eyes": ServoKeys(
        positive=("down", DOWN_KEYS),
        negative=("up", UP_KEYS),
        records={ord("u"): "limit_neg", ord("d"): "limit_pos"},
        labels={"limit_neg": "up limit", "limit_pos": "down limit", "home": "home"},
        help="up/down (i/k): tilt   u: up limit   d: down limit",
    ),
    "head_neck": ServoKeys(
        positive=("left", LEFT_KEYS),
        negative=("right", RIGHT_KEYS),
        records={ord("["): "limit_pos", ord("]"): "limit_neg"},
        labels={"limit_pos": "left limit", "limit_neg": "right limit", "home": "home"},
        help="left/right (j/l): turn   [: left limit   ]: right limit",
    ),
}
COMMON_HELP = "h: record home   g: go home   r: refresh   q: quit"


def decode_key(key: int, name: str = "head_eyes") -> str | None:
    """The action for a `cv2.waitKeyEx` code when driving servo `name`, or None for a key that does nothing.

    Actions: the servo's two directions (up / down for the eyes, left / right for the neck), limit_neg, limit_pos,
    home, go_home, refresh, quit.
    """
    servo = SERVOS[name]
    for action, keys in (servo.positive, servo.negative):
        if key in keys:
            return action
    return servo.records.get(key) or COMMON_KEYS.get(key)


def annotate(image: np.ndarray, lines: list[str], footer: list[str] | None = None) -> np.ndarray:
    """Copy of the RGB `image` with `lines` at the top left and `footer` lines at the bottom.

    The text is outlined so it reads on any picture.
    """
    out = image.copy()
    rows = [(8, 20 + 20 * i, text) for i, text in enumerate(lines)]
    footer = footer or []
    rows += [(8, out.shape[0] - 10 - 20 * (len(footer) - 1 - i), text) for i, text in enumerate(footer)]
    for x, y, text in rows:
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
        self.keys = SERVOS[name]
        self.show, self.color_format = show, color_format
        self.message = ""

    def status_lines(self) -> list[str]:
        """Text for the image: where the servo is and what has been recorded."""
        s = self.client.head_state()[self.name]
        lines = [f"{self.name}: {s['ticks']} ticks"]
        if s["home"] is not None:
            deg = s["sign"] * (s["ticks"] - s["home"]) / TICKS_PER_DEG
            positive = "down" if self.name == "head_eyes" else "left"
            lines[0] += f", {deg:+.1f} deg from home (+ is {positive})"
        recorded = ", ".join(
            f"{label} {s[key]}" for key, label in self.keys.labels.items() if s[key] is not None
        )
        lines.append(f"recorded: {recorded or 'nothing yet'}")
        if self.message:
            lines.append(self.message)
        return lines

    def help_lines(self) -> list[str]:
        """The keys, drawn at the bottom of every image."""
        return [self.keys.help, COMMON_HELP]

    def refresh(self, after: float | None = None) -> None:
        """Fetch an image (taken after `after`, if given) and show it with the status text."""
        snap = self.client.capture(color_format=self.color_format, after=after)
        self.show(annotate(snap.color, self.status_lines(), self.help_lines()))

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        after = None
        try:
            if action in (self.keys.positive[0], self.keys.negative[0]):
                # the servo's positive direction is down (eyes) / left (neck)
                sign = 1 if action == self.keys.positive[0] else -1
                result = self.client.head_move(self.name, delta_deg=self.step_deg * sign)
                after = result["done_at"]
                self.message = (
                    "" if result["reached"] and not result["clamped"] else self._why_stopped(result)
                )
            elif action == "go_home":
                result = self.client.head_move(self.name, to="home")
                after = result["done_at"]
                self.message = "moved to home"
            elif action in self.keys.labels:
                self.client.head_record(self.name, action)
                self.message = f"saved {self.keys.labels[action]}"
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
        print(f"{args.name}: click the window, then\n  " + "\n  ".join(session.help_lines()))
        try:
            session.refresh()
            while True:
                action = decode_key(cv2.waitKeyEx(100), args.name)
                if action is not None and not session.handle(action):
                    break
        finally:
            session.finish()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
