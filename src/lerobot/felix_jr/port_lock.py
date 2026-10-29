"""One process at a time on a serial port.

Two programs talking on the same servo bus (a second server started while the first is still shutting down, the
server and a wiggle test) interleave their packets, which shows up as `Incorrect status packet!` on random
writes, or worse, as servos receiving commands nobody meant to send. serial ports allow this by default, so every
follower takes an advisory lock (`flock`) on a file named after the port when it connects, and holds it until it
disconnects. The operating system drops the lock if the process dies. The check only works between programs that
use it, i.e. everything in `felix_jr`; LeRobot's own tools do not take it. Not available on Windows (no `fcntl`):
there it does nothing.
"""

import os
import re
import tempfile
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None


class PortInUseError(RuntimeError):
    """Another process already holds the lock for this serial port."""


def lock_path(port: str, directory: Path | None = None) -> Path:
    """The lock file for `port` (e.g. `/dev/ttyACM1` -> `<tmp>/felix_jr_dev_ttyACM1.lock`)."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", port).strip("_")
    return Path(directory or tempfile.gettempdir()) / f"felix_jr_{name}.lock"


def acquire_port_lock(port: str, directory: Path | None = None) -> int | None:
    """Take the lock for `port` and return a handle for `release_port_lock`; raises `PortInUseError` if held.

    The error says which process holds it (its pid is written into the lock file). Returns None where locking
    is not available.
    """
    if fcntl is None:
        return None
    path = lock_path(port, directory)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        holder = os.pread(fd, 32, 0).decode(errors="ignore").strip()
        os.close(fd)
        raise PortInUseError(
            f"{port} is in use by another process"
            + (f" (pid {holder})" if holder else "")
            + ". Stop it first (the robot server? another script?). "
            f"Lock file: {path}"
        ) from e
    os.ftruncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)
    return fd


def release_port_lock(fd: int | None) -> None:
    """Release a lock from `acquire_port_lock` (None is accepted and ignored)."""
    if fd is None or fcntl is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
