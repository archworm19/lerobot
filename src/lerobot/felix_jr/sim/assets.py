"""The SO-101's MuJoCo model (TheRobotStudio's, Apache-2.0), downloaded once at a pinned commit and cached.

Not vendored: its meshes are ~18 MB of STL. `so101_model()` returns the path of `so101_new_calib_camera.xml` (the arm
with the wrist camera's mount, LeRobot's "new" calibration: each joint's zero mid-range, as our readings are), its
meshes next to it in `assets/`. The XLeRobot repository's MuJoCo model was not used: its arms are the older SO-100,
with no cameras (`docs/sim2real_grasp.md`).
"""

import os
import urllib.request
from pathlib import Path

REPOSITORY = "TheRobotStudio/SO-ARM100"
COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"  # 2026-11-09's main
FOLDER = "Simulation/SO101"
MODEL = "so101_new_calib_camera.xml"
MESHES = (
    "base_motor_holder_so101_v1.stl",
    "base_so101_v2.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "moving_jaw_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "sts3215_03a_v1.stl",
    "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
    "wrist_camera_mount_so101_v1.stl",
    "wrist_camera_so101_v1.stl",
    "wrist_roll_follower_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
)


def cache_dir() -> Path:
    """Where the model is kept: `$FELIX_JR_SIM_CACHE`, else `~/.cache/felix_jr/so101/<commit>`."""
    root = os.environ.get("FELIX_JR_SIM_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "felix_jr" / "so101" / COMMIT[:12]


def is_cached() -> bool:
    """Whether the model and all its meshes are already downloaded."""
    folder = cache_dir()
    return (folder / MODEL).is_file() and all((folder / "assets" / mesh).is_file() for mesh in MESHES)


def so101_model(download: bool = True) -> Path:
    """The path of the SO-101's MJCF, downloading it and its meshes first if they are not cached."""
    folder = cache_dir()
    if not is_cached():
        if not download:
            raise FileNotFoundError(
                f"the SO-101 model is not in {folder}: call so101_model() once with network"
            )
        (folder / "assets").mkdir(parents=True, exist_ok=True)
        base = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{FOLDER}"
        for name in (MODEL, *(f"assets/{mesh}" for mesh in MESHES)):
            target = folder / name
            if not target.is_file():
                partial = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(f"{base}/{name}", partial)  # noqa: S310  # nosec B310: a fixed https URL
                partial.rename(target)
    return folder / MODEL
