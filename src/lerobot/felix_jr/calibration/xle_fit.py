"""Fit `XleRobotModel`'s estimates to clicked pinch points: every session, both arms, one head (a MAP estimate).

Each click is the pixel where a person saw an arm's pinch point, with the arm's joints and the head's servo ticks
at that moment. The fit finds the estimates that make the model's predicted pixels agree with the clicks:

    minimize  sum over clicks of rho(|pixel residual| / sigma_px)  +  sum over parameters of ((value - mean) / sigma)^2

with Gaussian priors from the model's parameter table and a Huber `rho`, so a mis-click does not drag the fit.
It is a plain nonlinear least squares problem once each prior is a residual row, solved by
`scipy.optimize.least_squares` (trust region), with the model's analytic Jacobian (`pixel_jacobian_many`; finite
differences on top of `predict_many` with `FitConfig(jacobian="2-point")`).
scipy applies the Huber loss to each residual (u, v and every prior row) rather than to a click's pixel
distance; prior rows within 2 sigma are in the quadratic part, so this rarely matters.

**One fit for everything.** Clicks of both arms are fitted together: the head and camera parameters, which both
arms share, get one estimate from all the clicks, and each arm's parameters are learned from its own clicks. The
parameters of an arm without clicks are left as they are. An earlier calibration is a starting point (`theta0`),
never a prior: the priors stay the wide ones of the parameter table, so no click is counted twice.

**Sessions** are folders of clicks. All sessions in a fit must share the head's servo map (signs and reference
ticks) and the camera intrinsics, since the clicks store raw head ticks. `calibration.collect` writes them
(`save_session`); `load_session` also reads version 1 folders.

Usage: uv run python -m lerobot.felix_jr.calibration.xle_fit SESSION [SESSION ...] --out calibration.urdf
       [--sigma-px 3] [--huber 2] [--holdout-every 5] [--start-from calibration.urdf]
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

BEHIND_CAMERA_RESIDUAL = (
    1e3  # residual (in sigmas) of a click whose predicted pinch point is behind the camera
)


@dataclass
class Click:
    """One clicked pinch point: the pixel, and the robot's measured joints at the moment of the frame."""

    arm: str
    arm_pose: dict[str, float]  # LeRobot's degrees, {"shoulder_pan.pos": ..., ...}
    head_ticks: dict[str, int]
    pixel: tuple[float, float]
    depth_m: float | None = None
    image: str | None = None  # relative to the session folder
    session: str = ""
    index: int = 0
    timestamp: float = 0.0


@dataclass
class Session:
    """A folder of clicks of one arm, with the head servo map and intrinsics they were recorded with."""

    path: Path
    arm: str
    clicks: list[Click]
    head_signs: dict[str, int]
    head_refs: dict[str, int]
    intrinsics: dict[str, float]


SESSION_FILE = "observations.json"


def load_session(directory: str | Path) -> Session:
    """Read a session folder's `observations.json` (written by `save_session`, or by version 1's `collect`)."""
    directory = Path(directory).expanduser()
    data = json.loads((directory / SESSION_FILE).read_text())
    settings = data["settings"]
    arm = settings["arm"]
    clicks = [
        Click(
            arm=arm,
            arm_pose={k: float(v) for k, v in o["arm"].items()},
            head_ticks={k: int(v) for k, v in o["head_ticks"].items()},
            pixel=(float(o["pixel"][0]), float(o["pixel"][1])),
            depth_m=o.get("depth_m"),
            image=o.get("image"),
            session=directory.name,
            index=int(o.get("index", i)),
            timestamp=float(o.get("timestamp", 0.0)),
        )
        for i, o in enumerate(data["observations"])
    ]
    return Session(
        directory,
        arm,
        clicks,
        {k: int(v) for k, v in settings["head_signs"].items()},
        {k: int(v) for k, v in settings["head_refs"].items()},
        dict(settings["intrinsics"]),
    )


def save_session(session: Session) -> None:
    """Write `session` to `observations.json` in its folder (`load_session` reads it back)."""
    data = {
        "version": 2,
        "settings": {
            "arm": session.arm,
            "intrinsics": session.intrinsics,
            "head_signs": session.head_signs,
            "head_refs": session.head_refs,
        },
        "observations": [
            {
                "index": c.index,
                "image": c.image,
                "timestamp": c.timestamp,
                "pixel": list(c.pixel),
                "depth_m": c.depth_m,
                "arm": c.arm_pose,
                "head_ticks": c.head_ticks,
            }
            for c in session.clicks
        ],
    }
    session.path.mkdir(parents=True, exist_ok=True)
    (session.path / SESSION_FILE).write_text(json.dumps(data, indent=2) + "\n")


def check_compatible(sessions: Sequence[Session]) -> None:
    """Raise unless all sessions share the head's servo map and the intrinsics (their clicks store raw ticks)."""
    first = sessions[0]
    for s in sessions[1:]:
        for name in ("head_signs", "head_refs", "intrinsics"):
            if getattr(s, name) != getattr(first, name):
                raise ValueError(
                    f"{s.path} and {first.path} have different {name} ({getattr(s, name)} vs {getattr(first, name)}): "
                    "their clicks cannot be fitted together"
                )


def model_for_sessions(sessions: Sequence[Session], **kwargs: Any) -> XleRobotModel:
    """An uncalibrated `XleRobotModel` with the sessions' servo map and intrinsics (which must all agree)."""
    check_compatible(sessions)
    first = sessions[0]
    return XleRobotModel.from_urdf(
        head_signs=first.head_signs, head_refs=first.head_refs, intrinsics=first.intrinsics, **kwargs
    )


@dataclass
class FitConfig:
    """Settings of the fit.

    `sigma_px`: the click noise (pixels); `huber`: residual size, in sigmas, beyond which it is down-weighted;
    `max_nfev`: a cap on the optimizer's evaluations; `jacobian`: "analytic" (the model's) or a scipy
    finite-difference scheme ("2-point", "3-point").
    """

    sigma_px: float = 3.0
    huber: float = 2.0
    max_nfev: int | None = None
    jacobian: str = "analytic"


@dataclass
class FitResult:
    """The fitted estimates (all parameters, in table order) and how good the fit is."""

    theta: np.ndarray
    free: np.ndarray  # which parameters were fitted (the head's and those of arms with clicks)
    covariance: np.ndarray  # (P, P); a parameter that was not fitted keeps its prior variance
    pixel_errors: np.ndarray  # per click, |predicted - clicked| in pixels (inf behind the camera)
    cost: float
    nfev: int
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sigma_post(self) -> np.ndarray:
        """Posterior standard deviation of each parameter."""
        return np.sqrt(np.diag(self.covariance))


def free_parameters(model: XleRobotModel, clicks: Sequence[Click]) -> np.ndarray:
    """Mask of the parameters the clicks can inform: the shared head and camera, and the clicked arms'."""
    arms = {c.arm for c in clicks}
    return np.array([p.arm is None or p.arm in arms for p in model.parameters])


def pixel_errors(
    model: XleRobotModel, clicks: Sequence[Click], theta: np.ndarray | None = None
) -> np.ndarray:
    """Distance in pixels between the predicted pinch point and the click, per click (inf if behind the camera)."""
    out = np.full(len(clicks), np.inf)
    for arm, (indices, qs, pixels) in _by_arm(model, clicks).items():
        uv, z = model.predict_many(arm, qs, theta)
        ok = np.isfinite(uv).all(axis=1) & (z > 0)
        out[indices[ok]] = np.linalg.norm(uv[ok] - pixels[ok], axis=1)
    return out


def rms(values: np.ndarray) -> float:
    """Root mean square of the finite values (nan if none)."""
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def fit(
    model: XleRobotModel,
    clicks: Sequence[Click],
    cfg: FitConfig | None = None,
    theta0: np.ndarray | None = None,
) -> FitResult:
    """MAP fit of the model's estimates to `clicks`, starting from `theta0` (default: the current estimates).

    Only the parameters `free_parameters` selects are fitted; the others keep their `theta0` values. The model's
    current estimates are not changed (the caller decides whether to `set_estimates(result.theta)`).
    """
    require_package("scipy", extra="felix_jr")
    if not clicks:
        raise ValueError("no clicks to fit")
    cfg = cfg or FitConfig()
    free = free_parameters(model, clicks)
    mean = np.array([p.mean for p in model.parameters])
    sigma = np.array([p.sigma for p in model.parameters])
    start = model.estimates if theta0 is None else np.asarray(theta0, dtype=float).copy()
    groups = _by_arm(model, clicks)

    def full(x: np.ndarray) -> np.ndarray:
        theta = start.copy()
        theta[free] = x
        return theta

    def residuals(x: np.ndarray) -> np.ndarray:
        theta = full(x)
        rows = []
        for arm, (_, qs, pixels) in groups.items():
            uv, z = model.predict_many(arm, qs, theta)
            r = (uv - pixels) / cfg.sigma_px
            r[~(np.isfinite(uv).all(axis=1) & (z > 0))] = BEHIND_CAMERA_RESIDUAL
            rows.append(r.ravel())
        rows.append((x - mean[free]) / sigma[free])
        return np.concatenate(rows)

    def jacobian(x: np.ndarray) -> np.ndarray:
        theta = full(x)
        rows = []
        for arm, (_, qs, _) in groups.items():
            d_pixels = (
                model.pixel_jacobian_many(arm, qs, theta)[:, :, free] / cfg.sigma_px
            )  # zero behind the camera
            rows.append(d_pixels.reshape(-1, int(free.sum())))
        rows.append(np.diag(1.0 / sigma[free]))
        return np.vstack(rows)

    result = least_squares(
        residuals,
        start[free],
        jac=jacobian if cfg.jacobian == "analytic" else cfg.jacobian,
        loss="huber",
        f_scale=cfg.huber,
        x_scale=sigma[free],
        max_nfev=cfg.max_nfev,
    )
    theta = full(result.x)
    covariance = np.diag(sigma**2)
    covariance[np.ix_(free, free)] = np.linalg.pinv(result.jac.T @ result.jac)
    return FitResult(
        theta,
        free,
        covariance,
        pixel_errors(model, clicks, theta),
        float(result.cost),
        result.nfev,
        result.message,
    )


def split_holdout(clicks: Sequence[Click], every: int) -> tuple[list[Click], list[Click]]:
    """(train, held out): in each session, every `every`-th click is held out (none if `every` is 0)."""
    if every <= 0:
        return list(clicks), []
    position: dict[str, int] = {}
    train: list[Click] = []
    held: list[Click] = []
    for c in clicks:
        i = position.get(c.session, 0)
        position[c.session] = i + 1
        (held if i % every == 0 else train).append(c)
    return train, held


def learned(posterior: float, prior: float) -> str:
    """Whether the clicks determined a parameter: posterior sigma well below the prior's."""
    return "yes" if posterior < 0.5 * prior else ("partly" if posterior < 0.85 * prior else "no")


def report(
    model: XleRobotModel,
    clicks: Sequence[Click],
    final: FitResult,
    holdout: dict[str, dict[str, float]] | None,
    cfg: FitConfig,
) -> str:
    """The markdown report: errors per session before and after, the fitted parameters, the worst clicks."""
    before = pixel_errors(model, clicks, np.zeros(len(model.parameters)))
    sessions = sorted({c.session for c in clicks})
    lines = [
        "# Calibration fit",
        "",
        f"{len(clicks)} clicks from {', '.join(sessions)}; click noise {cfg.sigma_px:g} px, Huber at "
        f"{cfg.huber:g} sigma; {final.nfev} evaluations ({final.message}).",
        "",
        "## Pixel error of the pinch point",
        "",
        "| Session | Clicks | RMS before | RMS after | Worst after | Held out | Held-out RMS before | after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for session in sessions:
        mask = np.array([c.session == session for c in clicks])
        h = (holdout or {}).get(session, {})
        lines.append(
            f"| {session} | {mask.sum()} | {rms(before[mask]):.1f} | {rms(final.pixel_errors[mask]):.2f} | "
            f"{np.max(final.pixel_errors[mask]):.1f} | {h.get('n', 0):.0f} | {h.get('before', np.nan):.1f} | "
            f"{h.get('after', np.nan):.2f} |"
        )
    lines += [
        "",
        "## Parameters",
        "",
        "Fitted parameters only. *Learned*: the posterior sigma is well below the prior sigma; otherwise the clicks "
        "did not determine it and it stays near its prior mean.",
        "",
        "| Parameter | Value | Prior mean | Prior sigma | Posterior sigma | Learned |",
        "|---|---|---|---|---|---|",
    ]
    for p, value, post, fitted in zip(
        model.parameters, final.theta, final.sigma_post, final.free, strict=True
    ):
        if fitted:
            lines.append(
                f"| `{p.name}` | {p.display(value):+.2f} {p.unit} | {p.display(p.mean):+.2f} | "
                f"{p.display(p.sigma):.2f} | {p.display(post):.2f} | {learned(post, p.sigma)} |"
            )
    worst = np.argsort(-np.where(np.isfinite(final.pixel_errors), final.pixel_errors, 1e9))[:5]
    lines += [
        "",
        "## Largest remaining errors (check these clicks)",
        "",
        "| Session | Click | Image | Error px |",
    ]
    lines.append("|---|---|---|---|")
    for i in worst:
        c = clicks[i]
        lines.append(f"| {c.session} | {c.index} | {c.image or ''} | {final.pixel_errors[i]:.1f} |")
    return "\n".join(lines) + "\n"


def calibrate(
    model: XleRobotModel,
    clicks: Sequence[Click],
    cfg: FitConfig | None = None,
    holdout_every: int = 5,
    theta0: np.ndarray | None = None,
) -> tuple[FitResult, dict[str, dict[str, float]]]:
    """Validate on held-out clicks, then fit all of them and store the result in the model.

    With `holdout_every` > 0, every N-th click of each session is left out, the fit is run on the rest, and the
    pixel RMS of the left-out clicks before and after calibration is recorded per session: the number that says
    whether the calibration generalizes. The final fit uses every click; its estimates, covariance and the
    held-out numbers go into the model (`estimates`, `covariance`, `fit_info`).
    """
    cfg = cfg or FitConfig()
    holdout: dict[str, dict[str, float]] = {}
    train, held = split_holdout(clicks, holdout_every)
    if held and len(train) >= 5:
        partial = fit(model, train, cfg, theta0)
        before = pixel_errors(model, held, np.zeros(len(model.parameters)))
        after = pixel_errors(model, held, partial.theta)
        for session in sorted({c.session for c in held}):
            mask = np.array([c.session == session for c in held])
            holdout[session] = {
                "n": float(mask.sum()),
                "before": rms(before[mask]),
                "after": rms(after[mask]),
            }
    final = fit(model, clicks, cfg, theta0)
    model.set_estimates(final.theta)
    model.covariance = final.covariance
    model.fit_info = {
        "sessions": sorted({c.session for c in clicks}),
        "clicks": len(clicks),
        "sigma_px": cfg.sigma_px,
        "huber": cfg.huber,
        "pixel_rms": {
            s: rms(final.pixel_errors[np.array([c.session == s for c in clicks])])
            for s in sorted({c.session for c in clicks})
        },
        "holdout": holdout,
    }
    return final, holdout


def _by_arm(
    model: XleRobotModel, clicks: Sequence[Click]
) -> dict[str, tuple[np.ndarray, list[np.ndarray], np.ndarray]]:
    """Per arm: the clicks' positions in `clicks`, their joint vectors and their pixels."""
    groups: dict[str, tuple[np.ndarray, list[np.ndarray], np.ndarray]] = {}
    for arm in sorted({c.arm for c in clicks}):
        indices = np.array([i for i, c in enumerate(clicks) if c.arm == arm])
        qs = [model.joint_vector({arm: clicks[i].arm_pose}, clicks[i].head_ticks) for i in indices]
        groups[arm] = (indices, qs, np.array([clicks[i].pixel for i in indices], dtype=float))
    return groups


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sessions", nargs="+", type=Path, help="session folders (observations.json from collect)"
    )
    parser.add_argument("--out", type=Path, required=True, help="the calibrated URDF to write")
    parser.add_argument("--sigma-px", type=float, default=3.0, help="click noise in pixels (3)")
    parser.add_argument(
        "--huber", type=float, default=2.0, help="down-weight residuals beyond this many sigmas (2)"
    )
    parser.add_argument(
        "--holdout-every", type=int, default=5, help="hold out every N-th click (5; 0 = none)"
    )
    parser.add_argument("--start-from", type=Path, help="a calibrated URDF whose estimates start the fit")
    args = parser.parse_args()

    sessions = [load_session(path) for path in args.sessions]
    model = model_for_sessions(sessions)
    theta0 = None
    if args.start_from:
        previous = XleRobotModel.load(args.start_from)
        theta0 = np.array([previous.estimate_dict().get(p.name, 0.0) for p in model.parameters])
    clicks = [c for s in sessions for c in s.clicks]
    cfg = FitConfig(args.sigma_px, args.huber)
    final, holdout = calibrate(model, clicks, cfg, args.holdout_every, theta0)
    text = report(model, clicks, final, holdout, cfg)
    print(text)
    model.save(args.out)
    args.out.with_suffix(".md").write_text(text)
    print(f"Wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
