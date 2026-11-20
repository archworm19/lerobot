"""The pod run script's pieces that need no pod: the image tag, the files sent, the remote command
(`cloud/run.py`)."""

import shlex
import subprocess

from lerobot.felix_jr.cloud.run import (
    IMAGE_INPUTS,
    image_tag,
    remote_command,
    rsync_base,
    ssh_base,
    tracked_files,
)


def _repo(tmp_path):
    for name in IMAGE_INPUTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    return tmp_path


def test_the_image_tag_follows_what_the_image_is_built_from_only(tmp_path):
    root = _repo(tmp_path)
    tag = image_tag(root)
    assert len(tag) == 12 and tag == image_tag(root)
    (root / "README.md").write_text("not an input")
    assert image_tag(root) == tag
    (root / "uv.lock").write_text("a new lock")
    assert image_tag(root) != tag


def test_the_files_sent_are_what_git_tracks_or_would(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("scratch/\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "capture.png").write_bytes(b"\x89PNG")
    sent = set(tracked_files(tmp_path).decode().split("\0")) - {""}
    assert sent == {".gitignore", "code.py"}


def test_the_remote_command_runs_in_the_code_with_the_run_dir_and_quoting_kept():
    line = remote_command(["python", "-m", "mod", "--name", "a b"], "/workspace/runs/1")
    args = shlex.split(line)
    assert args[:2] == ["bash", "-lc"]
    assert (
        args[2] == "cd /workspace/lerobot && export RUN_DIR=/workspace/runs/1 && python -m mod --name 'a b'"
    )


def test_ssh_and_rsync_use_the_pods_port_and_key(tmp_path):
    key = tmp_path / "id"
    assert ssh_base("root@1.2.3.4", 2222, key)[-3:] == ["-i", str(key), "root@1.2.3.4"]
    assert "-p 2222" in rsync_base(2222, key)[-1] and str(key) in rsync_base(2222, key)[-1]
