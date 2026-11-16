"""Reproducible MCL experiment runner.

Every number this script writes comes from actually running the filter; no
values are typed in by hand. Each trial uses the same map, the same agreed
robot path and the same seed streams, so in a comparison only the factor
under test changes (common random numbers across configurations).

Seed streams for a trial with seed ``s``:
    robot odometry noise  -> ``s``
    lidar measurement noise -> ``s + 1000``
    particle filter         -> ``s + 2000``

Outputs (under ``--out``, default ``results/``):
    tables/<experiment>_raw.csv         one row per (configuration, seed)
    tables/<experiment>_steps.csv       per-step errors for every trial
    tables/<experiment>_comparison.csv  grouped summary per configuration
    figures/<experiment>_rmse.png, _runtime.png, _success_rate.png,
    figures/<experiment>_error_curves.png

Usage::

    python -m scripts.experiments --list
    python -m scripts.experiments baseline --seeds 10
    python -m scripts.experiments resampling sensor_model --seeds 50
    python -m scripts.experiments all --seeds 50

Dependencies on other members' modules (imported lazily, only when a trial
runs): ``src.world.World``, ``src.robot.Robot``, ``src.lidar.Lidar`` and
``src.mcl.ParticleFilter``. The particle-filter calls are confined to
:func:`make_filter` and :func:`filter_step`; if the published interface
differs, only those two functions need changing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for batch runs

import numpy as np  # noqa: E402

from src.metrics import (  # noqa: E402
    convergence_time,
    success_rate,
    trajectory_errors,
)
from src import viz  # noqa: E402

# ---------------------------------------------------------------------------
# Agreed scenario
# ---------------------------------------------------------------------------

#: Simulation time step in seconds.
DT: float = 0.1

#: Start pose; matches the default ``Robot`` pose.
START_POSE: tuple[float, float, float] = (1.5, 1.2, 0.2)

#: Agreed path as ``(linear_velocity m/s, angular_velocity rad/s, steps)``.
#: 253 steps (25.3 s): east along the lower corridor, north, east under the
#: x = 7 m wall, then north up the right-hand room. Checked against the
#: default ``World`` map: every pose is free with >= 0.30 m wall clearance.
AGREED_PATH: tuple[tuple[float, float, int], ...] = (
    (0.3, -1.0, 2),
    (0.6, 0.0, 58),
    (0.3, 0.7854, 20),
    (0.6, 0.0, 18),
    (0.3, -0.7854, 20),
    (0.6, 0.0, 70),
    (0.3, 0.7854, 20),
    (0.6, 0.0, 45),
)

#: Default ``Robot`` odometry noise (x, y, theta); scaled by
#: ``TrialConfig.odometry_noise_scale``.
BASE_ODOMETRY_STD: tuple[float, float, float] = (0.015, 0.015, 0.01)

#: Kidnapping: at ``KIDNAP_STEP`` the true robot is translated by
#: ``KIDNAP_OFFSET`` without the odometry reporting it. The remaining path,
#: shifted by this offset, stays in free space.
KIDNAP_STEP: int = 170
KIDNAP_OFFSET: tuple[float, float] = (0.0, 3.0)

SEED_OFFSETS = {"robot": 0, "lidar": 1000, "filter": 2000}


def path_controls(path: Sequence[tuple[float, float, int]] = AGREED_PATH) -> list[tuple[float, float]]:
    """Expand ``(v, omega, steps)`` segments into one control per step."""
    controls: list[tuple[float, float]] = []
    for v, omega, steps in path:
        if int(steps) != steps or steps < 0:
            raise ValueError(f"segment step count must be a non-negative int; got {steps}")
        controls.extend([(float(v), float(omega))] * int(steps))
    return controls


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialConfig:
    """Everything that defines one experimental condition (seed excluded)."""

    label: str = "baseline"
    num_particles: int = 1000
    sensor_model: str = "beam"  # "beam" | "likelihood_field"
    resampling: str = "systematic"  # "systematic" | "multinomial"
    adaptive: bool = False
    augmented: bool = False
    beams: int = 36
    max_range: float = 8.0
    lidar_noise_std: float = 0.03
    odometry_noise_scale: float = 1.0
    kidnap: bool = False
    threshold: float = 0.3
    stable_steps: int = 10

    def __post_init__(self) -> None:
        if self.num_particles < 1:
            raise ValueError("num_particles must be >= 1")
        if self.sensor_model not in ("beam", "likelihood_field"):
            raise ValueError(f"unknown sensor_model {self.sensor_model!r}")
        if self.resampling not in ("systematic", "multinomial"):
            raise ValueError(f"unknown resampling {self.resampling!r}")
        if self.beams < 1 or self.max_range <= 0:
            raise ValueError("beams must be >= 1 and max_range > 0")
        if self.lidar_noise_std < 0 or self.odometry_noise_scale < 0:
            raise ValueError("noise parameters must be non-negative")


BASELINE = TrialConfig()


def _variants(**by_label: dict[str, Any]) -> list[TrialConfig]:
    return [replace(BASELINE, label=label, **kw) for label, kw in by_label.items()]


#: Experiment name -> list of configurations compared in it.
EXPERIMENTS: dict[str, list[TrialConfig]] = {
    "baseline": [BASELINE],
    "resampling": _variants(
        systematic={"resampling": "systematic"},
        multinomial={"resampling": "multinomial"},
    ),
    "sensor_model": _variants(
        beam={"sensor_model": "beam"},
        likelihood_field={"sensor_model": "likelihood_field"},
    ),
    "adaptive": _variants(
        fixed={"adaptive": False},
        adaptive_kld={"adaptive": True},
    ),
    "kidnapped_robot": _variants(
        standard={"kidnap": True, "augmented": False},
        augmented={"kidnap": True, "augmented": True},
    ),
    "particle_sweep": [
        replace(BASELINE, label=f"N={n}", num_particles=n)
        for n in (100, 250, 500, 1000, 2000)
    ],
    "noise_robustness": [
        replace(BASELINE, label=f"noise x{s:g}", odometry_noise_scale=s,
                lidar_noise_std=0.03 * s)
        for s in (0.5, 1.0, 2.0, 4.0)
    ],
    # Same as baseline; run it with --seeds 50..100 for seed validation.
    "seed_validation": [replace(BASELINE, label="baseline")],
}


# ---------------------------------------------------------------------------
# Particle-filter adapter (the only place that touches src.mcl)
# ---------------------------------------------------------------------------

def make_filter(world, lidar, config: TrialConfig, seed: int):
    """Construct the team's ``ParticleFilter`` for ``config``.

    Assumed interface (from the task allocation; confirm against SPEC.md
    once ``src/mcl.py`` is merged)::

        ParticleFilter(world, lidar, num_particles=..., sensor_model=...,
                       resampling=..., augmented=..., seed=...)
    """
    from src.mcl import ParticleFilter

    return ParticleFilter(
        world,
        lidar,
        num_particles=config.num_particles,
        sensor_model=config.sensor_model,
        resampling=config.resampling,
        augmented=config.augmented,
        seed=seed,
    )


def filter_step(pf, odometry: np.ndarray, scan: np.ndarray, config: TrialConfig) -> np.ndarray:
    """Advance the filter one step and return the pose estimate ``(x, y, theta)``.

    Uses ``pf.adaptive_step`` for adaptive configurations, ``pf.step``
    otherwise. If the step method returns ``None``, ``pf.estimate()`` is used.
    """
    method = pf.adaptive_step if config.adaptive else pf.step
    estimate = method(odometry, scan)
    if estimate is None:
        estimate = pf.estimate()
    est = np.asarray(estimate, dtype=float).ravel()
    if est.shape != (3,):
        raise ValueError(f"filter estimate must be (x, y, theta); got shape {est.shape}")
    return est


def particle_count(pf) -> int:
    """Current number of particles (varies under adaptive resampling)."""
    return int(len(pf.particles))


FilterFactory = Callable[[Any, Any, TrialConfig, int], Any]


# ---------------------------------------------------------------------------
# Running trials
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of one (configuration, seed) run."""

    config: TrialConfig
    seed: int
    true_history: np.ndarray
    estimated_history: np.ndarray
    particle_counts: np.ndarray
    runtime_s: float
    summary: dict[str, Any] = field(default_factory=dict)


def _default_components():
    from src.lidar import Lidar
    from src.robot import Robot
    from src.world import World

    return World, Robot, Lidar


def run_trial(
    config: TrialConfig,
    seed: int,
    filter_factory: FilterFactory = make_filter,
    components=None,
    controls: Sequence[tuple[float, float]] | None = None,
) -> TrialResult:
    """Run one localisation trial along the agreed path.

    ``filter_factory`` and ``components`` (a ``(World, Robot, Lidar)``
    triple of classes) can be injected for testing; by default the team's
    modules are used.
    """
    World, Robot, Lidar = components or _default_components()
    controls = path_controls() if controls is None else list(controls)
    if not controls:
        raise ValueError("controls must not be empty")
    if config.kidnap and not 0 < KIDNAP_STEP < len(controls):
        raise ValueError("KIDNAP_STEP lies outside the path")

    world = World()
    robot = Robot(
        pose=START_POSE,
        seed=seed + SEED_OFFSETS["robot"],
        odometry_std=tuple(config.odometry_noise_scale * s for s in BASE_ODOMETRY_STD),
    )
    lidar = Lidar(
        world,
        beams=config.beams,
        max_range=config.max_range,
        noise_std=config.lidar_noise_std,
        seed=seed + SEED_OFFSETS["lidar"],
    )
    pf = filter_factory(world, lidar, config, seed + SEED_OFFSETS["filter"])

    true_hist, est_hist, counts = [], [], []
    filter_time = 0.0
    for k, control in enumerate(controls):
        if config.kidnap and k == KIDNAP_STEP:
            robot.pose[0] += KIDNAP_OFFSET[0]
            robot.pose[1] += KIDNAP_OFFSET[1]
        true_pose, odometry = robot.step(control, DT)
        scan = lidar.scan(true_pose, noisy=True)
        start = time.perf_counter()
        est = filter_step(pf, odometry, scan, config)
        filter_time += time.perf_counter() - start
        true_hist.append(np.asarray(true_pose, dtype=float).copy())
        est_hist.append(est)
        counts.append(particle_count(pf))

    result = TrialResult(
        config=config,
        seed=seed,
        true_history=np.array(true_hist),
        estimated_history=np.array(est_hist),
        particle_counts=np.array(counts),
        runtime_s=filter_time,
    )
    result.summary = summarise_trial(result)
    return result


def summarise_trial(result: TrialResult) -> dict[str, Any]:
    """Scalar metrics for one trial (one row of the raw CSV)."""
    cfg = result.config
    errs = trajectory_errors(result.true_history, result.estimated_history)
    pos = errs["position_errors"]
    conv = convergence_time(pos, cfg.threshold, cfg.stable_steps)
    steps = errs["steps"]

    # Success: converged, and still within threshold at the end of the run.
    success = conv is not None and errs["final_position_error"] <= cfg.threshold

    row: dict[str, Any] = {
        "label": cfg.label,
        "seed": result.seed,
        "num_particles": cfg.num_particles,
        "sensor_model": cfg.sensor_model,
        "resampling": cfg.resampling,
        "adaptive": cfg.adaptive,
        "augmented": cfg.augmented,
        "odometry_noise_scale": cfg.odometry_noise_scale,
        "lidar_noise_std": cfg.lidar_noise_std,
        "kidnap": cfg.kidnap,
        "steps": steps,
        "position_rmse_m": errs["position_rmse"],
        "heading_rmse_rad": errs["heading_rmse"],
        "mean_position_error_m": errs["mean_position_error"],
        "final_position_error_m": errs["final_position_error"],
        "final_heading_error_rad": errs["final_heading_error"],
        "converged": conv is not None,
        "convergence_step": conv if conv is not None else "",
        "convergence_time_s": conv * DT if conv is not None else "",
        "success": success,
        "runtime_s": result.runtime_s,
        "runtime_per_step_ms": 1000.0 * result.runtime_s / steps,
        "mean_particles": float(np.mean(result.particle_counts)),
    }
    if cfg.kidnap:
        after = pos[KIDNAP_STEP:]
        rec = convergence_time(after, cfg.threshold, cfg.stable_steps)
        recovered = rec is not None and after[-1] <= cfg.threshold
        row["recovered"] = recovered
        row["recovery_steps"] = rec if rec is not None else ""
        row["recovery_time_s"] = rec * DT if rec is not None else ""
        # For a kidnapped trial, success means re-localising after the kidnap.
        row["success"] = recovered
    return row


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def summarise_groups(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group raw rows by ``label`` (in first-seen order) and summarise."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    if not groups:
        raise ValueError("no rows to summarise")

    out = []
    for label, rs in groups.items():
        rmses = np.array([r["position_rmse_m"] for r in rs], dtype=float)
        head = np.array([r["heading_rmse_rad"] for r in rs], dtype=float)
        final = np.array([r["final_position_error_m"] for r in rs], dtype=float)
        runtime = np.array([r["runtime_per_step_ms"] for r in rs], dtype=float)
        conv = np.array([r["convergence_time_s"] for r in rs if r["converged"]], dtype=float)
        summary = {
            "label": label,
            "runs": len(rs),
            "position_rmse_mean_m": float(np.mean(rmses)),
            "position_rmse_std_m": float(np.std(rmses)),
            "position_rmse_median_m": float(np.median(rmses)),
            "heading_rmse_mean_rad": float(np.mean(head)),
            "final_position_error_mean_m": float(np.mean(final)),
            "success_rate": success_rate([r["success"] for r in rs]),
            "converged_runs": int(sum(bool(r["converged"]) for r in rs)),
            "convergence_time_mean_s": float(np.mean(conv)) if conv.size else float("nan"),
            "runtime_per_step_mean_ms": float(np.mean(runtime)),
            "runtime_per_step_std_ms": float(np.std(runtime)),
            "mean_particles": float(np.mean([r["mean_particles"] for r in rs])),
        }
        if "recovered" in rs[0]:
            rec = np.array([r["recovery_time_s"] for r in rs if r["recovered"]], dtype=float)
            summary["recovery_rate"] = success_rate([r["recovered"] for r in rs])
            summary["recovery_time_mean_s"] = float(np.mean(rec)) if rec.size else float("nan")
        out.append(summary)
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write dict rows to CSV; columns are the union of keys in first-seen order."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    columns: list[str] = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in columns})
    return path


def _fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return value


def step_rows(result: TrialResult) -> list[dict[str, Any]]:
    """Per-step rows for the ``_steps.csv`` file."""
    errs = trajectory_errors(result.true_history, result.estimated_history)
    return [
        {
            "label": result.config.label,
            "seed": result.seed,
            "step": k,
            "time_s": (k + 1) * DT,
            "true_x": t[0], "true_y": t[1], "true_theta": t[2],
            "est_x": e[0], "est_y": e[1], "est_theta": e[2],
            "position_error_m": p,
            "heading_error_rad": h,
            "particles": int(n),
        }
        for k, (t, e, p, h, n) in enumerate(zip(
            result.true_history, result.estimated_history,
            errs["position_errors"], errs["heading_errors"], result.particle_counts,
        ))
    ]


def save_plots(name: str, summary: list[dict[str, Any]], results: list[TrialResult],
               fig_dir: Path) -> list[Path]:
    """Write the standard figures for one experiment."""
    labels = [s["label"] for s in summary]
    paths = [
        viz.plot_metric_bars(
            labels, [s["position_rmse_mean_m"] for s in summary],
            fig_dir / f"{name}_rmse.png", "Position RMSE (m)",
            f"{name}: position RMSE (mean ± std over seeds)",
            errors=[s["position_rmse_std_m"] for s in summary],
        ),
        viz.plot_metric_bars(
            labels, [s["runtime_per_step_mean_ms"] for s in summary],
            fig_dir / f"{name}_runtime.png", "Filter time per step (ms)",
            f"{name}: runtime per step (mean ± std over seeds)",
            errors=[s["runtime_per_step_std_ms"] for s in summary],
            value_format="{:.1f}",
        ),
        viz.plot_metric_bars(
            labels, [100.0 * s["success_rate"] for s in summary],
            fig_dir / f"{name}_success_rate.png", "Success rate (%)",
            f"{name}: success rate", value_format="{:.0f}%",
        ),
    ]
    curves, spread = {}, {}
    for label in labels:
        stacked = np.stack([
            trajectory_errors(r.true_history, r.estimated_history)["position_errors"]
            for r in results if r.config.label == label
        ])
        curves[label] = np.median(stacked, axis=0)
        q25, q75 = np.percentile(stacked, [25, 75], axis=0)
        spread[label] = (q75 - q25) / 2
    if len(curves) <= len(viz.SERIES_COLORS):
        paths.append(viz.plot_error_curves(
            curves, fig_dir / f"{name}_error_curves.png",
            f"{name}: median position error over time (band: IQR/2)",
            dt=DT, threshold=results[0].config.threshold, spread=spread,
        ))
    return paths


def run_experiment(
    name: str,
    seeds: Sequence[int],
    out_dir: Path = Path("results"),
    configs: Sequence[TrialConfig] | None = None,
    filter_factory: FilterFactory = make_filter,
    components=None,
    controls: Sequence[tuple[float, float]] | None = None,
    plots: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run every configuration of an experiment over ``seeds`` and save outputs.

    Returns the grouped summary rows.
    """
    out_dir = Path(out_dir)
    configs = EXPERIMENTS[name] if configs is None else list(configs)
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    results: list[TrialResult] = []
    for cfg in configs:
        for seed in seeds:
            res = run_trial(cfg, seed, filter_factory, components, controls)
            results.append(res)
            if verbose:
                s = res.summary
                print(f"[{name}] {cfg.label:>18} seed={seed:<4} "
                      f"rmse={s['position_rmse_m']:.3f} m  success={s['success']}  "
                      f"{s['runtime_per_step_ms']:.1f} ms/step", flush=True)

    raw = [r.summary for r in results]
    summary = summarise_groups(raw)
    tables = out_dir / "tables"
    write_csv(tables / f"{name}_raw.csv", raw)
    write_csv(tables / f"{name}_steps.csv", [row for r in results for row in step_rows(r)])
    write_csv(tables / f"{name}_comparison.csv", summary)
    if plots:
        save_plots(name, summary, results, out_dir / "figures")
    if verbose:
        print_summary(name, summary, seeds)
    return summary


def print_summary(name: str, summary: Sequence[dict[str, Any]], seeds: Sequence[int]) -> None:
    """Print a grouped summary table to stdout."""
    print(f"\n== {name}: {len(seeds)} seeds ({min(seeds)}..{max(seeds)}) ==")
    header = (f"{'label':>18} {'runs':>5} {'RMSE m':>14} {'success':>8} "
              f"{'conv s':>7} {'ms/step':>8} {'particles':>9}")
    print(header)
    print("-" * len(header))
    for s in summary:
        conv = s["convergence_time_mean_s"]
        conv_text = f"{conv:.2f}" if np.isfinite(conv) else "n/a"
        rmse_text = f"{s['position_rmse_mean_m']:.3f}±{s['position_rmse_std_m']:.3f}"
        print(f"{s['label']:>18} {s['runs']:>5} {rmse_text:>14} "
              f"{100 * s['success_rate']:>7.0f}% {conv_text:>7} "
              f"{s['runtime_per_step_mean_ms']:>8.1f} {s['mean_particles']:>9.0f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiments", nargs="*", default=["baseline"],
                        help="experiment names, or 'all'")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of seeds (uses first-seed .. first-seed+N-1)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for name, cfgs in EXPERIMENTS.items():
            print(f"{name:>18}: {', '.join(c.label for c in cfgs)}")
        return 0
    names = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}; use --list", file=sys.stderr)
        return 2
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    for name in names:
        run_experiment(name, seeds, args.out, plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
