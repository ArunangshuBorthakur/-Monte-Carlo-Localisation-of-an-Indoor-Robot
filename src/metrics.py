"""Localisation accuracy metrics for Monte Carlo Localisation experiments.

All functions are pure (no global state, no randomness) and work on plain
Python sequences or NumPy arrays. Poses are ``(x, y, theta)`` with ``x`` and
``y`` in metres and ``theta`` in radians, matching ``robot.py``.

Conventions
-----------
* Position error is the Euclidean distance in metres.
* Heading error is the absolute wrapped angular difference, in ``[0, pi]``,
  so an error across the ``-pi/pi`` boundary is measured the short way round.
* Convergence is reported as a step index (0-based). Multiply by the
  simulation time step ``dt`` to get seconds.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "wrap_angle",
    "position_error",
    "heading_error",
    "rmse",
    "trajectory_errors",
    "convergence_time",
    "success_rate",
    "measure_runtime",
]


def wrap_angle(angle):
    """Wrap an angle (scalar or array) into ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _as_poses(poses, name: str) -> np.ndarray:
    """Convert to a float array whose last axis holds at least ``(x, y)``."""
    arr = np.asarray(poses, dtype=float)
    if arr.ndim == 0 or arr.shape[-1] < 2:
        raise ValueError(
            f"{name} must have at least (x, y) on its last axis; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _scalar_or_array(values: np.ndarray):
    """Return a Python float for 0-d results, otherwise the array."""
    return float(values) if values.ndim == 0 else values


def position_error(true_pose, estimated_pose):
    """Euclidean position error in metres.

    Accepts single poses ``(x, y[, theta])`` or arrays of shape ``(N, 2|3)``;
    any heading component is ignored. Returns a float for a single pose and
    an array of shape ``(N,)`` for batches.
    """
    true_arr = _as_poses(true_pose, "true_pose")
    est_arr = _as_poses(estimated_pose, "estimated_pose")
    if true_arr.shape[:-1] != est_arr.shape[:-1]:
        raise ValueError(
            f"pose batches differ in shape: {true_arr.shape} vs {est_arr.shape}"
        )
    diff = true_arr[..., :2] - est_arr[..., :2]
    return _scalar_or_array(np.hypot(diff[..., 0], diff[..., 1]))


def heading_error(true_theta, estimated_theta):
    """Absolute wrapped heading error in radians, in ``[0, pi]``.

    For example, ``heading_error(pi - 0.1, -pi + 0.1)`` is ``0.2``, not
    ``2*pi - 0.2``. Works on scalars or equally shaped arrays.
    """
    t = np.asarray(true_theta, dtype=float)
    e = np.asarray(estimated_theta, dtype=float)
    if t.shape != e.shape:
        raise ValueError(f"heading arrays differ in shape: {t.shape} vs {e.shape}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e))):
        raise ValueError("headings contain non-finite values")
    return _scalar_or_array(np.abs(wrap_angle(t - e)))


def rmse(errors) -> float:
    """Root-mean-square of a 1-D sequence of errors."""
    arr = np.asarray(errors, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("cannot compute RMSE of an empty sequence")
    return float(np.sqrt(np.mean(arr**2)))


def trajectory_errors(true_history, estimated_history) -> dict:
    """Per-step and summary errors for a whole run.

    Parameters
    ----------
    true_history, estimated_history:
        Sequences of ``(x, y, theta)`` poses, shape ``(T, 3)``, aligned by
        time step.

    Returns
    -------
    dict with keys
        ``position_errors`` (T,) metres, ``heading_errors`` (T,) radians,
        ``position_rmse``, ``heading_rmse``, ``mean_position_error``,
        ``final_position_error``, ``final_heading_error``, ``steps``.
    """
    true_arr = np.asarray(true_history, dtype=float)
    est_arr = np.asarray(estimated_history, dtype=float)
    if true_arr.size == 0 or est_arr.size == 0:
        raise ValueError("trajectory histories must not be empty")
    for name, arr in (("true_history", true_arr), ("estimated_history", est_arr)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape (T, 3); got {arr.shape}")
    if true_arr.shape != est_arr.shape:
        raise ValueError(
            f"histories differ in length: {true_arr.shape} vs {est_arr.shape}"
        )

    pos = np.asarray(position_error(true_arr, est_arr), dtype=float)
    head = np.asarray(heading_error(true_arr[:, 2], est_arr[:, 2]), dtype=float)
    return {
        "position_errors": pos,
        "heading_errors": head,
        "position_rmse": rmse(pos),
        "heading_rmse": rmse(head),
        "mean_position_error": float(np.mean(pos)),
        "final_position_error": float(pos[-1]),
        "final_heading_error": float(head[-1]),
        "steps": int(pos.size),
    }


def convergence_time(position_errors, threshold: float = 0.3, stable_steps: int = 10):
    """First step at which the filter has converged, or ``None``.

    The filter counts as converged at step ``k`` when the position error is
    ``<= threshold`` for ``stable_steps`` consecutive steps starting at ``k``.
    A run that dips below the threshold only briefly, or whose final stretch
    below the threshold is shorter than ``stable_steps``, has not converged.
    An empty error sequence returns ``None``.
    """
    if not threshold > 0:
        raise ValueError(f"threshold must be positive; got {threshold}")
    if int(stable_steps) != stable_steps or stable_steps < 1:
        raise ValueError(f"stable_steps must be a positive integer; got {stable_steps}")
    stable_steps = int(stable_steps)

    errors = np.asarray(position_errors, dtype=float).ravel()
    if errors.size < stable_steps:
        return None

    # Errors that are NaN count as "not within threshold".
    within = np.nan_to_num(errors, nan=np.inf) <= threshold
    # Sliding-window count of in-threshold steps via cumulative sum.
    csum = np.concatenate(([0], np.cumsum(within, dtype=int)))
    window_counts = csum[stable_steps:] - csum[:-stable_steps]
    hits = np.flatnonzero(window_counts == stable_steps)
    return int(hits[0]) if hits.size else None


def success_rate(success_flags: Sequence[bool]) -> float:
    """Fraction of successful runs, in ``[0, 1]``.

    Raises ``ValueError`` on an empty input: a success rate over zero runs is
    undefined, and silently returning 0 would hide a broken experiment.
    """
    flags = np.asarray(success_flags)
    if flags.size == 0:
        raise ValueError("success_flags must not be empty")
    if flags.ndim != 1:
        raise ValueError(f"success_flags must be 1-D; got shape {flags.shape}")
    return float(np.mean(flags.astype(bool)))


def measure_runtime(func: Callable[..., Any], *args, **kwargs) -> tuple[Any, float]:
    """Call ``func(*args, **kwargs)`` and return ``(result, elapsed_seconds)``.

    Uses ``time.perf_counter`` (monotonic, high resolution).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start
