"""Live Matplotlib demo of Monte Carlo Localisation on the agreed path.

Shows the occupancy map, particle cloud, true and estimated robot, true and
estimated trajectory, lidar beams and the current error.

Usage::

    python -m scripts.run_live_demo
    python -m scripts.run_live_demo --seed 3 --particles 2000 --sensor likelihood_field
    python -m scripts.run_live_demo --kidnap --augmented
    python -m scripts.run_live_demo --no-show --save results/figures/live_demo_final.png

Uses the same scenario, seed streams and particle-filter adapter as
``scripts/experiments.py``, so a demo run with ``--seed s`` is the same run
as that seed in an experiment with the same configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live MCL visualisation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--sensor", choices=("beam", "likelihood_field"), default="beam")
    p.add_argument("--resampling", choices=("systematic", "multinomial"), default="systematic")
    p.add_argument("--adaptive", action="store_true", help="use KLD adaptive_step")
    p.add_argument("--augmented", action="store_true", help="augmented MCL")
    p.add_argument("--kidnap", action="store_true", help="kidnap the robot mid-run")
    p.add_argument("--every", type=int, default=1, help="redraw every N steps")
    p.add_argument("--pause", type=float, default=0.001, help="seconds per redraw")
    p.add_argument("--no-show", action="store_true",
                   help="run headless (Agg backend), no window")
    p.add_argument("--save", type=Path, default=None, help="save the final frame here")
    args = p.parse_args(argv)
    if args.every < 1 or args.particles < 1:
        p.error("--every and --particles must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Imported after the backend choice (experiments.py would otherwise force Agg).
    from scripts import experiments as ex
    from src.metrics import heading_error, position_error
    from src.viz import LiveView

    cfg = replace(
        ex.BASELINE,
        label="demo",
        num_particles=args.particles,
        sensor_model=args.sensor,
        resampling=args.resampling,
        adaptive=args.adaptive,
        augmented=args.augmented,
        kidnap=args.kidnap,
    )

    World, Robot, Lidar = ex._default_components()
    world = World()
    robot = Robot(
        pose=ex.START_POSE,
        seed=args.seed + ex.SEED_OFFSETS["robot"],
        odometry_std=tuple(cfg.odometry_noise_scale * s for s in ex.BASE_ODOMETRY_STD),
    )
    lidar = Lidar(world, beams=cfg.beams, max_range=cfg.max_range,
                  noise_std=cfg.lidar_noise_std,
                  seed=args.seed + ex.SEED_OFFSETS["lidar"])
    pf = ex.make_filter(world, lidar, cfg, args.seed + ex.SEED_OFFSETS["filter"])

    # Beam angles relative to the heading; Lidar is expected to expose them.
    beam_angles = getattr(lidar, "angles", None)
    title = (f"MCL live demo - {cfg.sensor_model}, {cfg.resampling}"
             f"{', adaptive' if cfg.adaptive else ''}"
             f"{', augmented' if cfg.augmented else ''}, seed {args.seed}")
    view = LiveView(world, beam_angles=beam_angles, title=title)
    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    controls = ex.path_controls()
    true_hist, est_hist = [], []
    note = ""
    for k, control in enumerate(controls):
        if cfg.kidnap and k == ex.KIDNAP_STEP:
            robot.pose[0] += ex.KIDNAP_OFFSET[0]
            robot.pose[1] += ex.KIDNAP_OFFSET[1]
            note = f"kidnapped at step {k}"
        true_pose, odometry = robot.step(control, ex.DT)
        scan = lidar.scan(true_pose, noisy=True)
        est = ex.filter_step(pf, odometry, scan, cfg)
        true_hist.append(np.array(true_pose, dtype=float))
        est_hist.append(est)

        last = k == len(controls) - 1
        if k % args.every == 0 or last:
            view.update(
                true_pose, est, particles=pf.particles,
                true_history=true_hist, estimated_history=est_hist,
                scan=scan, step=k,
                position_error=position_error(true_pose, est),
                heading_error=heading_error(true_pose[2], est[2]),
                extra_text=note,
            )
            if not args.no_show:
                view.fig.canvas.draw_idle()
                plt.pause(args.pause)

    if args.save is not None:
        print(f"saved final frame to {view.save(args.save)}")
    if not args.no_show:
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
