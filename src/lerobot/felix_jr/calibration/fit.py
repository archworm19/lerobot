"""Fit the kinematic parameters so the model's predicted tip pixels match the clicked ones (a MAP estimate).

The unknowns are the parameters of `model.parameter_set` (joint zero offsets, joint and camera mount offsets, the
tip offset). Each has a Gaussian prior from `model.SIGMAS`, so the cost is

    sum over clicks of rho(|pixel residual| / sigma_px)  +  sum over parameters of (value / prior sigma)^2   [/ 2]

with a Huber `rho` so that a mis-click does not drag the fit. Optionally (`use_depth`) the depth at the click is a
third residual per click. It is solved with Levenberg-Marquardt in plain numpy (finite-difference Jacobian), and
the posterior covariance `inv(J^T W J)` at the solution says which parameters the clicks actually determined: a
parameter whose posterior sigma is close to its prior sigma was not learned and stays near zero.

Validation: with `--holdout-every N` every N-th click is left out, the fit is run on the rest, and the pixel RMS on
the left-out clicks before and after calibration is reported, which is the number that says whether calibration
generalizes. The saved calibration is then fitted on all the clicks.

Usage: uv run python -m lerobot.felix_jr.calibration.fit observations.json --out-dir DIR
       [--sigma-px 3] [--holdout-every 5] [--use-depth] [--sigma-depth 0.015] [--huber 2.0]

Writes `kinematic_calibration.json` (parameters, uncertainties, the SO-101 mount transform), `xlerobot_calibrated.urdf`
and `so101_calibrated.urdf` (the parameters folded into the URDFs, see `urdf_patch.py`) and `fit_report.md`.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.felix_jr.calibration.model import (
    SO101_URDF,
    XLEROBOT_URDF,
    CalibrationModel,
    Observation,
    Param,
    load_observations,
    parameter_set,
)
from lerobot.felix_jr.calibration.urdf_patch import patch_urdf


@dataclass
class FitConfig:
    """Noise and optimizer settings.

    `sigma_px`: the click noise (pixels); `huber`: residual size, in sigmas, beyond which a click is
    down-weighted; `use_depth` adds the depth at the click as a residual with `sigma_depth` (meters).
    """

    sigma_px: float = 3.0
    huber: float = 2.0
    use_depth: bool = False
    sigma_depth: float = 0.015
    max_iter: int = 40
    tol: float = 1e-8


@dataclass
class FitResult:
    """The fitted parameters (`theta`, radians / meters) and how good the fit is."""

    theta: dict[str, float]
    params: list[Param]
    covariance: np.ndarray  # (P, P), in the same units as theta
    pixel_errors: np.ndarray  # (N,) |predicted - clicked| in pixels at the solution
    iterations: int
    cost_history: list[float] = field(default_factory=list)

    @property
    def sigma_post(self) -> dict[str, float]:
        """Posterior standard deviation of each parameter."""
        return {p.name: float(np.sqrt(self.covariance[i, i])) for i, p in enumerate(self.params)}


def vector_to_theta(params: list[Param], x: np.ndarray) -> dict[str, float]:
    """`{name: value}` for the parameter vector `x`."""
    return {p.name: float(v) for p, v in zip(params, x, strict=True)}


def observation_residuals(
    model: CalibrationModel, observations: list[Observation], theta: dict[str, float], cfg: FitConfig
) -> np.ndarray:
    """Residuals in units of their noise sigma: (u, v) per click, then the depth if `cfg.use_depth`."""
    out = []
    for o in observations:
        pred = model.predict(o, theta)
        u, v = pred.uv
        if not np.isfinite(pred.uv).all() or pred.z <= 0:
            out.append(
                [1e3, 1e3] + ([1e3] if cfg.use_depth else [])
            )  # tip behind the camera: a huge residual
            continue
        row = [(u - o.pixel[0]) / cfg.sigma_px, (v - o.pixel[1]) / cfg.sigma_px]
        if cfg.use_depth:
            row.append(0.0 if o.depth_m is None else (pred.z - o.depth_m) / cfg.sigma_depth)
        out.append(row)
    return np.array(out)


def pixel_errors(
    model: CalibrationModel, observations: list[Observation], theta: dict[str, float]
) -> np.ndarray:
    """Distance in pixels between the predicted tip and the click, per observation (inf if behind the camera)."""
    out = []
    for o in observations:
        pred = model.predict(o, theta)
        out.append(
            np.linalg.norm(pred.uv - np.array(o.pixel))
            if np.isfinite(pred.uv).all() and pred.z > 0
            else np.inf
        )
    return np.array(out)


def rms(values: np.ndarray) -> float:
    """Root mean square of the finite values (nan if none)."""
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def fit(
    model: CalibrationModel,
    observations: list[Observation],
    params: list[Param] | None = None,
    cfg: FitConfig | None = None,
    theta0: dict[str, float] | None = None,
) -> FitResult:
    """MAP fit of `params` (default: `parameter_set(model.arm)`) to the clicks. See the module docstring."""
    cfg = cfg or FitConfig()
    params = params or parameter_set(model.arm, model.prior_overrides)
    if not observations:
        raise ValueError("no observations to fit")
    prior = np.array([p.sigma for p in params])
    mean = np.array([p.mean for p in params])
    x = np.array([(theta0 or {}).get(p.name, p.mean) for p in params])
    steps = 1e-2 * prior  # finite-difference step per parameter

    def residuals(vec: np.ndarray) -> np.ndarray:
        return observation_residuals(model, observations, vector_to_theta(params, vec), cfg).ravel()

    def robust_cost(r_obs: np.ndarray, vec: np.ndarray) -> float:
        per_obs = np.linalg.norm(r_obs.reshape(len(observations), -1), axis=1)
        huber = np.where(per_obs <= cfg.huber, 0.5 * per_obs**2, cfg.huber * (per_obs - 0.5 * cfg.huber))
        return float(huber.sum() + 0.5 * np.sum(((vec - mean) / prior) ** 2))

    def jacobian(vec: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(len(params)):
            plus, minus = vec.copy(), vec.copy()
            plus[i] += steps[i]
            minus[i] -= steps[i]
            cols.append((residuals(plus) - residuals(minus)) / (2 * steps[i]))
        return np.array(cols).T

    lam, history, weighted_hessian = 1e-3, [], None
    iteration = 0
    for iteration in range(1, cfg.max_iter + 1):  # noqa: B007 (the last value is reported)
        r_obs = residuals(x)
        j_obs = jacobian(x)
        per_obs = np.linalg.norm(r_obs.reshape(len(observations), -1), axis=1)
        weights = np.where(per_obs <= cfg.huber, 1.0, cfg.huber / np.maximum(per_obs, 1e-12))
        w = np.repeat(weights, r_obs.size // len(observations))
        jw = j_obs.T * w
        hessian = jw @ j_obs + np.diag(1.0 / prior**2)
        gradient = jw @ r_obs + (x - mean) / prior**2
        weighted_hessian = hessian
        cost = robust_cost(r_obs, x)
        history.append(cost)
        improved = False
        for _ in range(12):
            step = np.linalg.solve(hessian + lam * np.diag(np.diag(hessian)), -gradient)
            x_new = x + step
            if robust_cost(residuals(x_new), x_new) < cost:
                x, lam, improved = x_new, max(lam / 3.0, 1e-9), True
                break
            lam *= 4.0
        if not improved or np.linalg.norm(step / prior) < cfg.tol:
            break

    theta = vector_to_theta(params, x)
    covariance = np.linalg.inv(weighted_hessian)
    return FitResult(theta, params, covariance, pixel_errors(model, observations, theta), iteration, history)


def split_holdout(observations: list[Observation], every: int) -> tuple[list[Observation], list[Observation]]:
    """(train, held out): every `every`-th observation is held out (none if `every` is 0)."""
    if every <= 0:
        return list(observations), []
    held = [o for i, o in enumerate(observations) if i % every == 0]
    train = [o for i, o in enumerate(observations) if i % every != 0]
    return train, held


def report(
    model: CalibrationModel,
    observations: list[Observation],
    final: FitResult,
    holdout_stats: dict[str, float] | None,
    cfg: FitConfig,
) -> str:
    """The markdown report: errors before and after, the parameters with their uncertainty, the worst clicks."""
    before = pixel_errors(model, observations, {})
    lines = [
        "# Kinematic calibration fit",
        "",
        f"{model.arm} arm, {len(observations)} clicks, click noise {cfg.sigma_px:g} px, "
        f"{final.iterations} iterations. The model's SO-101 mount alignment residual was "
        f"{model.alignment_rms * 1000:.1f} mm (see `model.py`).",
        "",
        "## Pixel error of the tip (clicks vs model)",
        "",
        "| | RMS px | median px | worst px |",
        "|---|---|---|---|",
        f"| all clicks, before calibration | {rms(before):.1f} | {np.nanmedian(before):.1f} | {np.nanmax(before):.1f} |",
        f"| all clicks, after calibration | {rms(final.pixel_errors):.1f} | {np.median(final.pixel_errors):.1f} "
        f"| {np.max(final.pixel_errors):.1f} |",
    ]
    if holdout_stats:
        lines += [
            f"| held out ({holdout_stats['n']:.0f} clicks), before | {holdout_stats['before']:.1f} | | |",
            f"| held out, after (fitted on the rest) | {holdout_stats['after']:.1f} | | |",
        ]
    lines += [
        "",
        "## Parameters",
        "",
        "A parameter is *learned* if its posterior sigma is well below its prior sigma; otherwise the clicks did not "
        "determine it and it stays near its prior mean (0).",
        "",
        "| Parameter | Group | Value | Prior sigma | Posterior sigma | Learned |",
        "|---|---|---|---|---|---|",
    ]
    sigma_post = final.sigma_post
    for p in final.params:
        value, post = final.theta[p.name], sigma_post[p.name]
        learned = "yes" if post < 0.5 * p.sigma else ("partly" if post < 0.85 * p.sigma else "no")
        lines.append(
            f"| `{p.name}` | {p.group} | {p.display(value):+.2f} {p.unit} | {p.display(p.sigma):.2f} | "
            f"{p.display(post):.2f} | {learned} |"
        )
    worst = np.argsort(-np.where(np.isfinite(final.pixel_errors), final.pixel_errors, 1e9))[:5]
    lines += [
        "",
        "## Largest remaining errors (check these clicks)",
        "",
        "| Click | Image | Error px |",
        "|---|---|---|",
    ]
    for i in worst:
        lines.append(
            f"| {observations[i].index} | {observations[i].image or ''} | {final.pixel_errors[i]:.1f} |"
        )
    return "\n".join(lines) + "\n"


def save_result(
    out_dir: Path, model: CalibrationModel, final: FitResult, holdout_stats: dict[str, float] | None
) -> None:
    """Write the calibration JSON, the two patched URDFs and nothing else (the report is written by the caller)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head_params, arm_params = model.split(final.theta)
    data = {
        "settings": model.settings(),
        "parameters": final.theta,
        "sigma_post": final.sigma_post,
        "prior_sigma": {p.name: p.sigma for p in final.params},
        "prior_mean": {p.name: p.mean for p in final.params},
        "X_Base_so101": model.T.tolist(),
        "pixel_rms": {"all": rms(final.pixel_errors), "holdout": holdout_stats},
    }
    (out_dir / "kinematic_calibration.json").write_text(json.dumps(data, indent=2) + "\n")
    (out_dir / "xlerobot_calibrated.urdf").write_text(patch_urdf(XLEROBOT_URDF.read_text(), head_params))
    (out_dir / "so101_calibrated.urdf").write_text(patch_urdf(SO101_URDF.read_text(), arm_params))


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("observations", type=Path, help="observations.json written by calibration.collect")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sigma-px", type=float, default=3.0, help="click noise in pixels (3)")
    parser.add_argument(
        "--holdout-every", type=int, default=5, help="hold out every N-th click to validate (5; 0 = none)"
    )
    parser.add_argument(
        "--huber", type=float, default=2.0, help="down-weight clicks off by more than this many sigmas (2)"
    )
    parser.add_argument("--use-depth", action="store_true", help="also fit the depth at each click")
    parser.add_argument("--sigma-depth", type=float, default=0.015, help="depth noise in meters (0.015)")
    args = parser.parse_args()

    model, observations = load_observations(args.observations)
    cfg = FitConfig(args.sigma_px, args.huber, args.use_depth, args.sigma_depth)
    holdout_stats = None
    train, held = split_holdout(observations, args.holdout_every)
    if held and len(train) >= 5:
        partial = fit(model, train, cfg=cfg)
        holdout_stats = {
            "n": float(len(held)),
            "before": rms(pixel_errors(model, held, {})),
            "after": rms(pixel_errors(model, held, partial.theta)),
        }
    final = fit(model, observations, cfg=cfg)
    text = report(model, observations, final, holdout_stats, cfg)
    print(text)
    save_result(args.out_dir, model, final, holdout_stats)
    (args.out_dir / "fit_report.md").write_text(text)
    print(f"Wrote the calibration, both URDFs and fit_report.md to {args.out_dir}")


if __name__ == "__main__":
    main()
