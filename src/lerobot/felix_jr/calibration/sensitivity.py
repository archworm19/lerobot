"""Step 1 of `docs/camera_calibration.md`: how many pixels do URDF errors cost (no hardware needed)?

For a set of head and arm configurations where the arm is in the head camera's view, every kinematic
parameter (joint zero offsets, joint origin positions, mount rotations, the camera mount) is perturbed and
points on the arm are re-projected into the color image:

1. Sensitivity: pixel shift per degree / per millimeter for each parameter, and the shift at its assumed
   tolerance (mean and worst case over the configurations).
2. Monte Carlo: all parameters perturbed at once by their assumed tolerances (Gaussian), giving the distribution
   of total pixel error with no calibration, for the head parameters only, the arm parameters only, and both.

The tolerances are guesses (see TOLERANCES); the per-unit sensitivities do not depend on them.

Usage: uv run python -m lerobot.felix_jr.calibration.sensitivity [--urdf PATH] [--arm left|right]
       [--calibration calibration.json] [--samples N] [--seed S] [--out report.md]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.felix_jr.calibration.urdf_kinematics import PinholeCamera, UrdfTree, invert

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "xlerobot.urdf"
# Measured on our D415 at 640x480 (diagnostics/realsense_check.py); overridden by --calibration.
DEFAULT_INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}

CAMERA_LINK = "head_camera_link"
CAMERA_MOUNT = "fixed_head_camera_link"
HEAD_TOWER = "fixed_top_base_link"
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
ARMS = {
    "left": {
        "mount": "fixed_Base",
        "joints": ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"],
        "tool_joint": "Jaw_L",
        "targets": {"elbow": "Lower_Arm", "wrist": "Wrist_Pitch_Roll", "jaw pivot": "Moving_Jaw"},
    },
    "right": {
        "mount": "fixed_Base_2",
        "joints": ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"],
        "tool_joint": "Jaw_R",
        "targets": {"elbow": "Lower_Arm_2", "wrist": "Wrist_Pitch_Roll_2", "jaw pivot": "Moving_Jaw_2"},
    },
}

# Assumed 1-sigma tolerances (guesses, to be replaced by measurements): radians and meters.
TOLERANCES = {
    "joint zero": np.radians(2.0),
    "joint origin position": 0.002,
    "head tower position": 0.003,
    "head tower rotation": np.radians(1.0),
    "arm mount position": 0.003,
    "arm mount rotation": np.radians(1.0),
    "camera mount position": 0.005,
    "camera mount rotation": np.radians(2.0),
}


@dataclass(frozen=True)
class Parameter:
    """One kinematic parameter to perturb, with the tolerance kind and group (head / arm) it belongs to."""

    name: str  # perturbation key understood by UrdfTree (zero:..., pos:...:x, rot:...:roll)
    kind: str  # key into TOLERANCES
    group: str  # "head" or "arm"

    @property
    def sigma(self) -> float:
        """Assumed 1-sigma tolerance, in radians or meters."""
        return TOLERANCES[self.kind]

    @property
    def angular(self) -> bool:
        """True for angles (joint zeros, rotations), False for positions."""
        return self.name.startswith(("zero:", "rot:"))

    @property
    def unit(self) -> str:
        """Unit used in the report: deg or mm."""
        return "deg" if self.angular else "mm"

    def per_unit(self, value_per_si: float) -> float:
        """Convert a quantity per radian / per meter into per degree / per millimeter."""
        return value_per_si * (np.pi / 180.0 if self.angular else 1e-3)


def parameters(tree: UrdfTree, arm: str) -> list[Parameter]:
    """Every parameter to study for `arm`: the head chain and camera mount, then the arm chain."""
    spec = ARMS[arm]
    xyz, rpy = ("x", "y", "z"), ("roll", "pitch", "yaw")
    out = [Parameter(f"pos:{HEAD_TOWER}:{a}", "head tower position", "head") for a in xyz]
    out += [Parameter(f"rot:{HEAD_TOWER}:{a}", "head tower rotation", "head") for a in rpy]
    for joint in HEAD_JOINTS:
        out.append(Parameter(f"zero:{joint}", "joint zero", "head"))
        out += [Parameter(f"pos:{joint}:{a}", "joint origin position", "head") for a in xyz]
    out += [Parameter(f"pos:{CAMERA_MOUNT}:{a}", "camera mount position", "head") for a in xyz]
    out += [Parameter(f"rot:{CAMERA_MOUNT}:{a}", "camera mount rotation", "head") for a in rpy]
    out += [Parameter(f"pos:{spec['mount']}:{a}", "arm mount position", "arm") for a in xyz]
    out += [Parameter(f"rot:{spec['mount']}:{a}", "arm mount rotation", "arm") for a in rpy]
    for joint in spec["joints"]:
        out.append(Parameter(f"zero:{joint}", "joint zero", "arm"))
        out += [Parameter(f"pos:{joint}:{a}", "joint origin position", "arm") for a in xyz]
    out += [Parameter(f"pos:{spec['tool_joint']}:{a}", "joint origin position", "arm") for a in xyz]
    missing = {p.name.split(":")[1] for p in out} - set(tree.joints)
    if missing:
        raise KeyError(f"joints not in the URDF: {sorted(missing)}")
    return out


def make_configurations(
    tree: UrdfTree,
    camera: PinholeCamera,
    arm: str,
    per_head_pose: int = 6,
    seed: int = 0,
    max_depth: float = 1.2,
) -> list[dict[str, float]]:
    """Head poses looking down at the workspace, each with random arm poses whose jaw pivot is in view."""
    rng = np.random.default_rng(seed)
    spec = ARMS[arm]
    tool_link = spec["targets"]["jaw pivot"]
    configs = []
    for tilt in (0.7, 0.95, 1.2):  # about 40, 54 and 69 deg down
        for pan in (-0.25, 0.0, 0.25):
            head = {"head_pan_joint": pan, "head_tilt_joint": tilt}
            x_root_cam = camera.pose_in_root(tree, head)
            found = 0
            for _ in range(20000):
                q = {j: rng.uniform(tree.joints[j].lower, tree.joints[j].upper) for j in spec["joints"]}
                tool = tree.point(tool_link, np.zeros(3), q)
                depth = (invert(x_root_cam) @ np.append(tool, 1.0))[2]
                if 0.15 < depth < max_depth and camera.in_view(camera.project(x_root_cam, tool))[0]:
                    configs.append({**head, **q})
                    found += 1
                    if found == per_head_pose:
                        break
    if not configs:
        raise RuntimeError(
            "no arm pose puts the jaw pivot in the camera's view; check the URDF and camera link"
        )
    return configs


def project_targets(
    tree: UrdfTree,
    camera: PinholeCamera,
    arm: str,
    config: dict[str, float],
    perturb: dict[str, float] | None,
) -> np.ndarray:
    """Pixels (T, 2) of the arm's target points for one configuration."""
    targets = ARMS[arm]["targets"].values()
    points = np.array([tree.point(link, np.zeros(3), config, perturb) for link in targets])
    return camera.project(camera.pose_in_root(tree, config, perturb), points)


def sensitivities(tree, camera, arm, configs, params, step: float = 1e-4) -> np.ndarray:
    """Pixel shift per radian / meter: array (P, C, T); NaN where a target is out of view."""
    out = np.full((len(params), len(configs), len(ARMS[arm]["targets"])), np.nan)
    for c, config in enumerate(configs):
        nominal = project_targets(tree, camera, arm, config, None)
        visible = camera.in_view(nominal)
        for i, p in enumerate(params):
            plus = project_targets(tree, camera, arm, config, {p.name: step})
            minus = project_targets(tree, camera, arm, config, {p.name: -step})
            shift = np.linalg.norm(plus - minus, axis=1) / (2 * step)
            out[i, c] = np.where(visible, shift, np.nan)
    return out


def monte_carlo(tree, camera, arm, configs, params, samples: int, seed: int = 1) -> np.ndarray:
    """Total pixel error (S, C, T) with every parameter in `params` drawn from N(0, its tolerance)."""
    rng = np.random.default_rng(seed)
    nominal = [project_targets(tree, camera, arm, c, None) for c in configs]
    out = np.full((samples, len(configs), len(ARMS[arm]["targets"])), np.nan)
    for s in range(samples):
        perturb = {p.name: rng.normal(0.0, p.sigma) for p in params}
        for c, config in enumerate(configs):
            visible = camera.in_view(nominal[c])
            err = np.linalg.norm(project_targets(tree, camera, arm, config, perturb) - nominal[c], axis=1)
            out[s, c] = np.where(visible, err, np.nan)
    return out


def report(tree, camera, arm, configs, params, sens, mc, urdf_path) -> str:
    """The markdown report: setup, assumed tolerances, Monte Carlo totals and the parameter ranking."""
    targets = list(ARMS[arm]["targets"])
    lines = [
        "# URDF sensitivity report",
        "",
        f"URDF: `{urdf_path}`, {arm} arm, camera link `{CAMERA_LINK}` (fx {camera.fx:.1f} px, "
        f"{camera.width}x{camera.height}). {len(configs)} configurations: head tilt 40-69 deg down, pan -14..+14 deg, "
        "random arm poses with the jaw pivot in view.",
        "",
    ]
    depths = []
    for config in configs:
        x_root_cam = camera.pose_in_root(tree, config)
        tool = tree.point(ARMS[arm]["targets"]["jaw pivot"], np.zeros(3), config)
        depths.append((invert(x_root_cam) @ np.append(tool, 1.0))[2])
    lines.append(
        f"Jaw pivot distance from the camera: {min(depths):.2f} to {max(depths):.2f} m "
        f"(median {np.median(depths):.2f}). Visible fraction per target: "
        + ", ".join(f"{t} {100 * np.isfinite(sens[0, :, k]).mean():.0f}%" for k, t in enumerate(targets))
    )
    lines += ["", "## Assumed tolerances (1 sigma, guesses)", "", "| Kind | Sigma |", "|---|---|"]
    for kind, sigma in TOLERANCES.items():
        lines.append(
            f"| {kind} | {np.degrees(sigma):.1f} deg |"
            if "zero" in kind or "rotation" in kind
            else f"| {kind} | {sigma * 1000:.0f} mm |"
        )

    lines += [
        "",
        "## Monte Carlo: total pixel error with no calibration",
        "",
        "| Parameters perturbed | Target | Mean px | 95th pct px | Max px |",
        "|---|---|---|---|---|",
    ]
    for label, values in mc.items():
        for k, t in enumerate(targets):
            v = values[:, :, k]
            v = v[np.isfinite(v)]
            if v.size:
                lines.append(
                    f"| {label} | {t} | {v.mean():.1f} | {np.percentile(v, 95):.1f} | {v.max():.1f} |"
                )

    tool = targets.index("jaw pivot")
    per_unit = np.array([p.per_unit(np.nanmean(sens[i, :, tool])) for i, p in enumerate(params)])
    worst = np.array([p.per_unit(np.nanmax(sens[i, :, tool])) for i, p in enumerate(params)])
    at_sigma = np.array([np.nanmean(sens[i, :, tool]) * p.sigma for i, p in enumerate(params)])
    order = np.argsort(-at_sigma)
    lines += [
        "",
        "## Parameters ranked by pixel shift of the jaw pivot at their assumed tolerance",
        "",
        "| Parameter | Group | px per unit (mean / worst) | px at 1 sigma |",
        "|---|---|---|---|",
    ]
    for i in order:
        p = params[i]
        lines.append(
            f"| `{p.name}` | {p.group} | {per_unit[i]:.1f} / {worst[i]:.1f} px per {p.unit} | {at_sigma[i]:.1f} |"
        )

    lines += [
        "",
        "## By kind (jaw pivot, mean over parameters of that kind)",
        "",
        "| Kind | Group | px per unit (mean) | px at 1 sigma (mean) |",
        "|---|---|---|---|",
    ]
    kinds = {}
    for i, p in enumerate(params):
        kinds.setdefault((p.kind, p.group), []).append(i)
    for (kind, group), idx in sorted(kinds.items(), key=lambda kv: -np.mean(at_sigma[kv[1]])):
        unit = params[idx[0]].unit
        lines.append(
            f"| {kind} | {group} | {np.mean(per_unit[idx]):.1f} px per {unit} | {np.mean(at_sigma[idx]):.1f} |"
        )
    return "\n".join(lines) + "\n"


def load_camera(calibration: str | None) -> PinholeCamera:
    """The head camera, with intrinsics from a calibration.json or the measured defaults."""
    intr = dict(DEFAULT_INTRINSICS)
    if calibration:
        ci = json.loads(Path(calibration).read_text())["color_intrinsics"]
        intr = {k: ci[k] for k in ("fx", "fy", "ppx", "ppy", "width", "height")}
    return PinholeCamera(
        CAMERA_LINK, intr["fx"], intr["fy"], intr["ppx"], intr["ppy"], intr["width"], intr["height"]
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--arm", choices=sorted(ARMS), default="left")
    parser.add_argument("--calibration", help="calibration.json from realsense_check (color intrinsics)")
    parser.add_argument("--samples", type=int, default=200, help="Monte Carlo samples (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="also write the report (markdown) to this file")
    args = parser.parse_args()

    tree = UrdfTree(args.urdf)
    camera = load_camera(args.calibration)
    params = parameters(tree, args.arm)
    configs = make_configurations(tree, camera, args.arm, seed=args.seed)
    sens = sensitivities(tree, camera, args.arm, configs, params)
    mc = {
        "head only": monte_carlo(
            tree,
            camera,
            args.arm,
            configs,
            [p for p in params if p.group == "head"],
            args.samples,
            args.seed + 1,
        ),
        "arm only": monte_carlo(
            tree,
            camera,
            args.arm,
            configs,
            [p for p in params if p.group == "arm"],
            args.samples,
            args.seed + 2,
        ),
        "all": monte_carlo(tree, camera, args.arm, configs, params, args.samples, args.seed + 3),
    }
    text = report(tree, camera, args.arm, configs, params, sens, mc, args.urdf)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
