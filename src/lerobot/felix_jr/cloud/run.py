"""A command run on a rented GPU pod: the code rsynced to it, run over ssh, its outputs fetched (`docs/sim_deploy.md`).

  uv run python -m lerobot.felix_jr.cloud.run --host root@<ip> --port <port> [--key ~/.ssh/id_ed25519] \
      [--fetch <remote dir>] -- python -m lerobot.felix_jr.sim2real.<module> ...
  python src/lerobot/felix_jr/cloud/run.py tag       # the image tag this checkout's dependencies build (CI uses it)
  uv run python -m lerobot.felix_jr.cloud.run fetch --host root@<ip> --port <port> <run id>   # a run's folder, again
  (`--send <file>`: a file the sync leaves out, e.g. a start pool under scratch/; `--detach`: in the background)

The pod runs the `felix-sim` image (`docker/Dockerfile.felix_sim`: the dependencies, sshd). Each run:

1. **The image checked:** the pod's `/opt/felix/image_tag` against this checkout's (`image_tag`: a hash of the files the
   image is built from); a mismatch (the lock changed since the image was built) stops the run unless `--any-image`.
2. **The code synced:** every file git tracks or would (`git ls-files -co --exclude-standard`: `scratch/` and the
   session images stay home) rsynced to `/workspace/lerobot` (the first time ~10 MB, then the changes).
3. **The run recorded:** `/workspace/runs/<id>/` gets the commit, `git diff HEAD` and the command (reproducible).
4. **Run** over ssh in `/workspace/lerobot` (`bash -lc`: the image's environment), its output streamed here, its
   exit code returned; `RUN_DIR` (that folder) set for the command to write into.
5. **Fetched:** the run's folder into `cloud/scratch/runs/<id>/` (and `--fetch`'s remote folders too).

Standard library only (`tag` runs before anything is installed). Nothing here starts or stops a pod: RunPod's console
or `runpodctl` does, and a pod bills until it is stopped.
"""

import argparse
import hashlib
import shlex
import subprocess  # nosec B404: rsync and ssh, with argument lists
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # the repository
# what the image is built from: its tag changes when one of these does (`.github/workflows/felix_image.yml`)
IMAGE_INPUTS = ("pyproject.toml", "uv.lock", "docker/Dockerfile.felix_sim", "docker/felix_sim_start.sh")
REMOTE_CODE = "/workspace/lerobot"
REMOTE_RUNS = "/workspace/runs"
LOCAL_RUNS = Path(__file__).parent / "scratch" / "runs"


def image_tag(root: Path = ROOT) -> str:
    """The image tag for `root`'s dependencies: the first 12 hex digits of a hash over `IMAGE_INPUTS`, in order."""
    digest = hashlib.sha256()
    for name in IMAGE_INPUTS:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()[:12]


def ssh_base(host: str, port: int, key: Path | None) -> list[str]:
    """The ssh arguments for the pod (no host key prompt: a pod's key is new every time)."""
    args = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    return args + (["-i", str(key)] if key else []) + [host]


def rsync_base(port: int, key: Path | None) -> list[str]:
    """The rsync arguments over that ssh."""
    shell = f"ssh -p {port} -o StrictHostKeyChecking=accept-new -o BatchMode=yes" + (
        f" -i {shlex.quote(str(key))}" if key else ""
    )
    return ["rsync", "-az", "-e", shell]


def tracked_files(root: Path = ROOT) -> bytes:
    """The files git tracks or would (untracked but not ignored), NUL-separated, for rsync's `--files-from`."""
    return subprocess.run(  # nosec B603 B607: git, fixed arguments
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, check=True, capture_output=True
    ).stdout


def remote_command(command: Sequence[str], run_dir: str, detach: bool = False) -> str:
    """`command` as the pod runs it: in the code's folder, with the image's environment and `RUN_DIR` set.

    `detach`: started in the background (`nohup`), its output into `RUN_DIR/output.log` and its exit code into
    `RUN_DIR/exit_code` when it ends; the ssh session returns at once (a training run outlives it).
    """
    inner = f"cd {REMOTE_CODE} && export RUN_DIR={shlex.quote(run_dir)} && {shlex.join(command)}"
    if not detach:
        return f"bash -lc {shlex.quote(inner)}"
    log, code = shlex.quote(f"{run_dir}/output.log"), shlex.quote(f"{run_dir}/exit_code")
    background = f"nohup bash -lc {shlex.quote(inner)} > {log} 2>&1; echo $? > {code}"
    return f"nohup bash -c {shlex.quote(background)} > /dev/null 2>&1 &"


def send_files(host: str, port: int, key: Path | None, paths: Sequence[Path], root: Path = ROOT) -> None:
    """`paths` (files the sync leaves out: under `scratch/`, say) to the same place under the pod's code folder."""
    names = []
    for path in paths:
        full = (root / path).resolve() if not Path(path).is_absolute() else Path(path)
        if not full.is_file():
            raise FileNotFoundError(f"{path}: no such file to send")
        names.append(str(full.relative_to(root.resolve())))
    if names:
        subprocess.run(  # nosec B603: rsync, an argument list
            [*rsync_base(port, key), "--relative", *names, f"{host}:{REMOTE_CODE}/"], cwd=root, check=True
        )


def fetch_run(host: str, port: int, run_id: str, key: Path | None = None) -> Path:
    """Run `run_id`'s folder from the pod into `cloud/scratch/runs/<run_id>/` (again: only what changed)."""
    local = LOCAL_RUNS / run_id
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec B603: rsync, an argument list
        [*rsync_base(port, key), f"{host}:{REMOTE_RUNS}/{run_id}/", f"{local}/"], check=False
    )
    return local


def run(
    host: str,
    port: int,
    command: Sequence[str],
    key: Path | None = None,
    fetch: Sequence[str] = (),
    any_image: bool = False,
    root: Path = ROOT,
    send: Sequence[Path] = (),
    detach: bool = False,
) -> int:
    """Steps 1-5 of the module docstring; the command's exit code (0 once started, `detach`).

    `send`: files to send along (`send_files`). `detach`: started in the background, nothing fetched (`fetch`
    later); the run's id printed.
    """
    ssh = ssh_base(host, port, key)
    remote_tag = subprocess.run(  # nosec B603: ssh, an argument list
        [*ssh, "cat /opt/felix/image_tag"], check=True, capture_output=True, text=True
    ).stdout.strip()
    local_tag = image_tag(root)
    if remote_tag != local_tag:
        message = f"the pod runs felix-sim:{remote_tag}, this checkout's dependencies are {local_tag}"
        if not any_image:
            print(f"{message}: start the pod on felix-sim:{local_tag} (or --any-image)", file=sys.stderr)
            return 2
        print(f"warning: {message}", file=sys.stderr)
    stop_at = subprocess.run(  # nosec B603: ssh, an argument list
        [*ssh, "cat /opt/felix/stop_at 2>/dev/null || true"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"the pod stops itself at {stop_at} (POD_TTL)" if stop_at else "no POD_TTL: stop the pod when done")

    rsync = rsync_base(port, key)
    subprocess.run(  # nosec B603: rsync, an argument list
        [*rsync, "--from0", "--files-from=-", "./", f"{host}:{REMOTE_CODE}/"],
        cwd=root,
        input=tracked_files(root),
        check=True,
    )
    send_files(host, port, key, send, root)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{REMOTE_RUNS}/{run_id}"
    commit = subprocess.run(  # nosec B603 B607: git
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    diff = subprocess.run(  # nosec B603 B607: git
        ["git", "diff", "HEAD"], cwd=root, check=True, capture_output=True
    ).stdout
    record = f"mkdir -p {run_dir} && cat > {run_dir}/diff.patch && printf '%s\\n' {shlex.quote(commit)} > {run_dir}/commit"
    subprocess.run([*ssh, record], input=diff, check=True)  # nosec B603: ssh, an argument list
    subprocess.run(  # nosec B603: ssh, an argument list
        [*ssh, f"printf '%s\\n' {shlex.quote(shlex.join(command))} > {run_dir}/command"], check=True
    )

    print(f"run {run_id} on {host}:{port} (felix-sim:{remote_tag}, {commit[:9]}): {shlex.join(command)}")
    if detach:
        subprocess.run([*ssh, remote_command(command, run_dir, detach=True)], check=True)  # nosec B603: ssh
        where = f"--host {host} --port {port}" + (f" --key {key}" if key else "")
        print(
            f"started in the background: {run_dir}/output.log on the pod; fetch with\n"
            f"  python -m lerobot.felix_jr.cloud.run fetch {where} {run_id}"
        )
        return 0
    code = subprocess.run([*ssh, remote_command(command, run_dir)]).returncode  # nosec B603: ssh

    local = LOCAL_RUNS / run_id
    local.mkdir(parents=True, exist_ok=True)
    for remote in (run_dir, *fetch):
        subprocess.run([*rsync, f"{host}:{remote}/", f"{local}/{Path(remote).name}/"], check=False)  # nosec B603
    print(f"exit {code}; fetched into {local}")
    return code


def main() -> None:
    """Command line entry point; see the module docstring."""
    if sys.argv[1:] == ["tag"]:
        print(image_tag())
        return
    if sys.argv[1:2] == ["fetch"]:
        fetcher = argparse.ArgumentParser(description="a run's folder from the pod")
        fetcher.add_argument("--host", required=True)
        fetcher.add_argument("--port", type=int, default=22)
        fetcher.add_argument("--key", type=Path, default=None)
        fetcher.add_argument("run_id")
        a = fetcher.parse_args(sys.argv[2:])
        print(f"fetched into {fetch_run(a.host, a.port, a.run_id, a.key)}")
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="the pod's ssh user@address (RunPod: root@<ip>)")
    parser.add_argument(
        "--port", type=int, default=22, help="its ssh port (RunPod: the exposed TCP port for 22)"
    )
    parser.add_argument("--key", type=Path, default=None, help="the ssh key RunPod has (default: ssh's)")
    parser.add_argument("--fetch", action="append", default=[], help="a remote folder to fetch after, too")
    parser.add_argument("--any-image", action="store_true", help="run even if the pod's image is another tag")
    parser.add_argument(
        "--send",
        action="append",
        type=Path,
        default=[],
        help="a file the sync leaves out to send (scratch/...)",
    )
    parser.add_argument("--detach", action="store_true", help="start it in the background; `fetch` later")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="after --: what to run in the code's folder"
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command (after --)")
    sys.exit(
        run(
            args.host,
            args.port,
            command,
            args.key,
            args.fetch,
            args.any_image,
            send=args.send,
            detach=args.detach,
        )
    )


if __name__ == "__main__":
    main()
