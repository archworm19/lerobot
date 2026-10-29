"""Find which serial port each arm is on, and remember it (`felix_jr/ports.py`).

For each robot id in turn, you unplug that arm's USB cable; the script sees which adapter vanished, then waits for
you to plug it back in. The result is saved so later runs can look ports up by robot id.

  uv run python -m lerobot.felix_jr.setup.find_ports                      # left_arm, right_arm
  uv run python -m lerobot.felix_jr.setup.find_ports --ids left_arm        # just one
  uv run python -m lerobot.felix_jr.setup.find_ports --show                # print the saved ports, no changes

Only the USB cables are touched; the servos stay powered. The ports are printed at the end, ready to paste into
the other scripts' `<PORT>` arguments (and into `docs/current_ports.md`).
"""

import argparse
import time
from collections.abc import Callable

from lerobot.felix_jr.ports import (
    PORTS_FILE,
    PortInfo,
    list_serial_ports,
    load_ports,
    preferred_device,
    resolve,
    save_ports,
)

DEFAULT_IDS = ("left_arm", "right_arm")


def identify(
    robot_id: str,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    list_fn: Callable[[], list[PortInfo]] = list_serial_ports,
    sleep: Callable[[float], None] = time.sleep,
    replug_timeout: float = 60.0,
) -> PortInfo:
    """Identify `robot_id`'s adapter: it is the one that disappears when the user unplugs it.

    Waits for it to come back before returning, so the next arm starts from a full list.
    """
    before = list_fn()
    ask(f"Unplug the USB cable of {robot_id} (only that one), then press Enter. ")
    sleep(0.5)  # let the OS drop the port
    gone_keys = {p.key for p in before} - {p.key for p in list_fn()}
    if len(gone_keys) != 1:
        raise RuntimeError(
            f"expected exactly one adapter to disappear, saw {len(gone_keys)} ({sorted(gone_keys)}). "
            "Check that only one cable was unplugged, and that the adapter shows up as a USB serial device."
        )
    (key,) = gone_keys
    candidates = [p for p in before if p.key == key]
    chosen = PortInfo(preferred_device(candidates), candidates[0].serial_number, candidates[0].location)
    say(f"{robot_id}: {chosen.device} ({chosen.key}). Plug it back in.")
    waited = 0.0
    while key not in {p.key for p in list_fn()}:
        if waited >= replug_timeout:
            raise TimeoutError(f"{robot_id}'s adapter did not come back within {replug_timeout:.0f} s")
        sleep(0.5)
        waited += 0.5
    sleep(0.5)
    return chosen


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="robot ids to identify")
    parser.add_argument("--show", action="store_true", help="print the saved ports and exit")
    args = parser.parse_args()

    if args.show:
        saved = load_ports()
        if not saved:
            print(f"nothing saved yet ({PORTS_FILE})")
        for robot_id, info in saved.items():
            try:
                now = resolve(robot_id)
            except LookupError:
                now = "NOT PLUGGED IN"
            print(f"{robot_id}: {now}   (saved {info.device}, {info.key})")
        return

    saved = load_ports()
    for robot_id in args.ids:
        saved[robot_id] = identify(robot_id, input, print)
    if len({p.key for p in saved.values()}) != len(saved):
        raise SystemExit("two robot ids ended up with the same adapter; nothing saved")
    save_ports(saved)
    print(f"\nSaved to {PORTS_FILE}:")
    for robot_id, info in saved.items():
        print(f"  {info.device} {robot_id}")


if __name__ == "__main__":
    main()
