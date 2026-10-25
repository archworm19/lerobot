"""Change one Feetech servo's ID, refusing whenever it could hit the wrong servo.

Use it to repair a servo that was given the wrong ID. Only the servo addressed as `--current-id` is touched,
and only if it answers cleanly and `--new-id` is not already in use. Servos that share an ID cannot be told
apart on the bus: a ping of that ID comes back garbled, and this tool refuses. Disconnect all but one of them
first (and the others stay at the shared ID until you fix them one at a time).

Usage: uv run python -m lerobot.felix_jr.diagnostics.set_servo_id <PORT> --current-id N --new-id M [--yes]
"""

import argparse
from collections.abc import Callable

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

BAUDRATE = 1_000_000  # lerobot's default for the STS3215; setup_motor also writes this back


def probe(bus, servo_id: int) -> str:
    """'ok' if the servo answers cleanly, 'garbled' if several answer at once (shared ID), else 'none'."""
    _, comm, error = bus.packet_handler.ping(bus.port_handler, servo_id)
    if comm == bus._comm_success and not error:
        return "ok"
    return "garbled" if "Incorrect" in bus.packet_handler.getTxRxResult(comm) else "none"


def change_servo_id(bus, current_id: int, new_id: int, confirm: Callable[[str], bool]) -> None:
    """Re-address the servo `current_id` as `new_id`. Raises RuntimeError instead of guessing."""
    if current_id == new_id:
        raise RuntimeError("--current-id and --new-id are the same.")
    bus.set_baudrate(BAUDRATE)
    state = probe(bus, current_id)
    if state == "garbled":
        raise RuntimeError(
            f"Several servos answer as id {current_id} at once, so none of them can be addressed. Unplug all "
            "but one of them from the chain (leave only that one connected to the board), then run this "
            "again."
        )
    if state == "none":
        raise RuntimeError(f"No servo answers as id {current_id} at {BAUDRATE} baud. Nothing was changed.")
    if probe(bus, new_id) != "none":
        raise RuntimeError(f"Id {new_id} is already in use on this chain. Nothing was changed.")
    present = [i for i in range(21) if probe(bus, i) == "ok"]
    question = f"Servos answering now: {present}. Change the servo with id {current_id} to id {new_id}?"
    if not confirm(question):
        raise RuntimeError("Cancelled. Nothing was changed.")
    bus.setup_motor("servo", initial_baudrate=BAUDRATE, initial_id=current_id)
    if probe(bus, new_id) != "ok" or probe(bus, current_id) == "ok":
        raise RuntimeError(
            f"The change did not take: id {new_id} should answer and id {current_id} should not."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port")
    parser.add_argument("--current-id", type=int, required=True)
    parser.add_argument("--new-id", type=int, required=True)
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port, motors={"servo": Motor(args.new_id, "sts3215", MotorNormMode.DEGREES)}
    )
    bus._connect(handshake=False)  # the chain holds other servos, so no handshake with a single motor
    try:
        change_servo_id(
            bus,
            args.current_id,
            args.new_id,
            confirm=lambda question: args.yes or input(f"{question} [y/N] ").strip().lower() == "y",
        )
        print(f"Servo {args.current_id} is now servo {args.new_id}.")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
