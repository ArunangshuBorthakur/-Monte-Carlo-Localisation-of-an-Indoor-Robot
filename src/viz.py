"""Matplotlib visualisation for Monte Carlo Localisation.

Two kinds of figure live here:

* :class:`LiveView` - an animated view of one run (occupancy map, particle
  cloud, true and estimated robot, true trajectory, lidar beams and an
  error read-out). Used by ``scripts/run_live_demo.py``.
* Result plots (:func:`plot_metric_bars`, :func:`plot_error_curves`) - static
  PNG figures written by ``scripts/experiments.py``.

This module never selects a Matplotlib backend. Batch scripts choose the
non-interactive ``Agg`` backend before importing it; the live demo uses the
default interactive one.

Only these attributes of a ``World`` are read: ``grid`` (bool, ``True`` =
occupied, row 0 at ``y = 0``), ``width``, ``height`` (metres).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "SERIES_COLORS",
    "beam_endpoints",
    "draw_map",
    "LiveView",
    "plot_metric_bars",
    "plot_error_curves",
]

# Categorical palette, used in this fixed order (never cycled or re-ranked).
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
)
_TEXT = "#0b0b0b"
_TEXT_MUTED = "#52514e"
_GRID = "#e4e3df"

# Fixed role -> colour mapping for the live view.
_TRUE_COLOR = SERIES_COLORS[0]
_EST_COLOR = SERIES_COLORS[1]
_BEAM_COLOR = SERIES_COLORS[2]
_PARTICLE_COLOR = SERIES_COLORS[6]


def _style_axes(ax: Axes) -> None:
    """Recessive axes: no top/right spines, light grid, muted tick labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_TEXT_MUTED)
    ax.tick_params(colors=_TEXT_MUTED)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def beam_endpoints(pose, angles, ranges) -> np.ndarray:
    """World-frame end points of lidar beams.

    Parameters
    ----------
    pose: ``(x, y, theta)`` of the sensor.
    angles: beam angles relative to the robot heading, shape ``(B,)``.
    ranges: measured ranges in metres, shape ``(B,)``.

    Returns
    -------
    numpy.ndarray of shape ``(B, 2)``.
    """
    x, y, theta = np.asarray(pose, dtype=float)[:3]
    angles = np.asarray(angles, dtype=float).ravel()
    ranges = np.asarray(ranges, dtype=float).ravel()
    if angles.shape != ranges.shape:
        raise ValueError(
            f"angles and ranges differ in shape: {angles.shape} vs {ranges.shape}"
        )
    world_angles = theta + angles
    return np.column_stack(
        (x + ranges * np.cos(world_angles), y + ranges * np.sin(world_angles))
    )


def draw_map(ax: Axes, world) -> None:
    """Draw the occupancy grid (occupied cells dark) in world coordinates."""
    grid = np.asarray(world.grid, dtype=bool)
    if grid.ndim != 2:
        raise ValueError(f"world.grid must be 2-D; got shape {grid.shape}")
    ax.imshow(
        grid,
        cmap="Greys",
        vmin=0,
        vmax=1.4,  # occupied cells render dark grey rather than pure black
        origin="lower",
        extent=(0.0, float(world.width), 0.0, float(world.height)),
        interpolation="nearest",
    )
    ax.set_xlim(0.0, float(world.width))
    ax.set_ylim(0.0, float(world.height))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=_TEXT)
    ax.set_ylabel("y (m)", color=_TEXT)


class LiveView:
    """Animated view of a single localisation run.

    Create once, then call :meth:`update` every step. All artists are created
    in ``__init__`` and mutated in place, so updates are cheap.

    Parameters
    ----------
    world: the map to draw.
    beam_angles: relative lidar beam angles; ``None`` hides the beams.
    title: figure title.
    arrow_length: length of the heading arrows, metres.
    """

    def __init__(
        self,
        world,
        beam_angles: Sequence[float] | None = None,
        title: str = "Monte Carlo Localisation",
        arrow_length: float = 0.5,
    ) -> None:
        if arrow_length <= 0:
            raise ValueError("arrow_length must be positive")
        self.world = world
        self.beam_angles = None if beam_angles is None else np.asarray(beam_angles, float)
        self.arrow_length = float(arrow_length)

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(9, 7.5))
        draw_map(self.ax, world)
        self.ax.set_title(title, color=_TEXT)

        ax = self.ax
        (self._beams,) = ax.plot([], [], color=_BEAM_COLOR, linewidth=0.6,
                                 alpha=0.6, label="Lidar beams")
        self._particles = ax.scatter([], [], s=4, color=_PARTICLE_COLOR,
                                     alpha=0.35, linewidths=0, label="Particles")
        (self._true_traj,) = ax.plot([], [], color=_TRUE_COLOR, linewidth=2,
                                     label="True trajectory")
        (self._est_traj,) = ax.plot([], [], color=_EST_COLOR, linewidth=1.5,
                                    linestyle="--", label="Estimated trajectory")
        (self._true_dot,) = ax.plot([], [], "o", color=_TRUE_COLOR, markersize=9,
                                    markeredgecolor="white", markeredgewidth=2,
                                    label="True robot")
        (self._est_dot,) = ax.plot([], [], "s", color=_EST_COLOR, markersize=8,
                                   markeredgecolor="white", markeredgewidth=2,
                                   label="Estimated robot")
        (self._true_head,) = ax.plot([], [], color=_TRUE_COLOR, linewidth=2)
        (self._est_head,) = ax.plot([], [], color=_EST_COLOR, linewidth=2)
        self._info = ax.text(
            0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=_TEXT, family="monospace",
            bbox=dict(facecolor="white", edgecolor=_GRID, alpha=0.9),
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.fig.tight_layout()

    def _heading_segment(self, pose) -> tuple[list[float], list[float]]:
        x, y, theta = pose[:3]
        return (
            [x, x + self.arrow_length * np.cos(theta)],
            [y, y + self.arrow_length * np.sin(theta)],
        )

    def update(
        self,
        true_pose,
        estimated_pose=None,
        particles=None,
        true_history=None,
        estimated_history=None,
        scan=None,
        step: int | None = None,
        position_error: float | None = None,
        heading_error: float | None = None,
        extra_text: str = "",
    ) -> None:
        """Redraw with the latest state. Any argument left as ``None`` is hidden.

        ``scan`` holds ranges for ``beam_angles`` and is drawn from the
        *true* pose (it is what the robot actually sees).
        """
        true_pose = np.asarray(true_pose, dtype=float)
        self._true_dot.set_data([true_pose[0]], [true_pose[1]])
        self._true_head.set_data(*self._heading_segment(true_pose))

        if estimated_pose is not None:
            est = np.asarray(estimated_pose, dtype=float)
            self._est_dot.set_data([est[0]], [est[1]])
            self._est_head.set_data(*self._heading_segment(est))
        else:
            self._est_dot.set_data([], [])
            self._est_head.set_data([], [])

        if particles is not None:
            p = np.asarray(particles, dtype=float)
            self._particles.set_offsets(p[:, :2])
        else:
            self._particles.set_offsets(np.empty((0, 2)))

        for line, hist in ((self._true_traj, true_history),
                           (self._est_traj, estimated_history)):
            if hist is not None and len(hist):
                h = np.asarray(hist, dtype=float)
                line.set_data(h[:, 0], h[:, 1])
            else:
                line.set_data([], [])

        if scan is not None and self.beam_angles is not None:
            ends = beam_endpoints(true_pose, self.beam_angles, scan)
            # One polyline with NaN breaks: origin -> end, gap, origin -> end ...
            n = len(ends)
            xs = np.empty(3 * n)
            ys = np.empty(3 * n)
            xs[0::3], ys[0::3] = true_pose[0], true_pose[1]
            xs[1::3], ys[1::3] = ends[:, 0], ends[:, 1]
            xs[2::3] = ys[2::3] = np.nan
            self._beams.set_data(xs, ys)
        else:
            self._beams.set_data([], [])

        lines = []
        if step is not None:
            lines.append(f"step        {step}")
        if particles is not None:
            lines.append(f"particles   {len(particles)}")
        if position_error is not None:
            lines.append(f"pos error   {position_error:.3f} m")
        if heading_error is not None:
            lines.append(f"head error  {np.degrees(heading_error):.1f} deg")
        if extra_text:
            lines.append(extra_text)
        self._info.set_text("\n".join(lines))
        self._info.set_visible(bool(lines))

    def save(self, path: str | Path, dpi: int = 120) -> Path:
        """Save the current frame as an image and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=dpi)
        return path


def plot_metric_bars(
    labels: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    ylabel: str,
    title: str,
    errors: Sequence[float] | None = None,
    value_format: str = "{:.3f}",
) -> Path:
    """Bar chart of one metric per configuration, saved as PNG.

    ``errors`` (e.g. standard deviation across seeds) draws error bars. Each
    bar carries its value as a label. NaN values (metric undefined, e.g. no
    run converged) are drawn as an empty slot labelled ``n/a``.
    """
    labels = list(labels)
    vals = np.asarray(values, dtype=float)
    if len(labels) != vals.size or vals.size == 0:
        raise ValueError("labels and values must be non-empty and equal length")
    errs = None if errors is None else np.asarray(errors, dtype=float)
    if errs is not None and errs.shape != vals.shape:
        raise ValueError("errors must match values in length")

    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(labels) + 2), 4.2))
    x = np.arange(len(labels))
    heights = np.nan_to_num(vals, nan=0.0)
    bars = ax.bar(x, heights, width=0.6, color=SERIES_COLORS[0],
                  edgecolor="white", linewidth=2)
    if errs is not None:
        ok = np.isfinite(vals) & np.isfinite(errs)
        ax.errorbar(x[ok], vals[ok], yerr=errs[ok], fmt="none",
                    ecolor=_TEXT_MUTED, elinewidth=1, capsize=4)
    tops = heights.copy()
    if errs is not None:
        tops += np.where(np.isfinite(errs) & np.isfinite(vals), errs, 0.0)
    for bar, v, top in zip(bars, vals, tops):
        text = "n/a" if not np.isfinite(v) else value_format.format(v)
        # Label sits above the error whisker so the two never overlap.
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, top),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=_TEXT_MUTED)
    ax.set_xticks(x, labels, rotation=20 if len(labels) > 4 else 0,
                  ha="right" if len(labels) > 4 else "center")
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.set_title(title, color=_TEXT)
    _style_axes(ax)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return _save(fig, path)


def plot_error_curves(
    curves: Mapping[str, np.ndarray],
    path: str | Path,
    title: str,
    ylabel: str = "Position error (m)",
    dt: float | None = None,
    threshold: float | None = None,
    spread: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Error-over-time line chart, one line per configuration, saved as PNG.

    ``curves`` maps label -> mean error per step. ``spread`` (same keys)
    optionally adds a shaded band, e.g. the inter-quartile half-range. At most
    eight series are allowed (one per palette slot); colours follow insertion
    order. ``threshold`` draws the convergence threshold.
    """
    if not curves:
        raise ValueError("curves must not be empty")
    if len(curves) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series; got {len(curves)}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (label, values) in zip(SERIES_COLORS, curves.items()):
        y = np.asarray(values, dtype=float)
        t = np.arange(y.size) * (dt if dt else 1.0)
        ax.plot(t, y, color=color, linewidth=2, label=label)
        if spread is not None and label in spread:
            s = np.asarray(spread[label], dtype=float)
            ax.fill_between(t, y - s, y + s, color=color, alpha=0.15, linewidth=0)
    if threshold is not None:
        ax.axhline(threshold, color=_TEXT_MUTED, linewidth=1, linestyle=":")
        ax.annotate(f"threshold {threshold:g} m", (1.0, threshold),
                    xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=_TEXT_MUTED)
    ax.set_xlabel("Time (s)" if dt else "Step", color=_TEXT)
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.set_title(title, color=_TEXT)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
