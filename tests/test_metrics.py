import math
import time

import numpy as np
import pytest

from src.metrics import (
    convergence_time,
    heading_error,
    measure_runtime,
    position_error,
    rmse,
    success_rate,
    trajectory_errors,
    wrap_angle,
)


# --- position error -------------------------------------------------------

def test_position_error_zero_for_identical_pose():
    assert position_error((1.0, 2.0, 0.5), (1.0, 2.0, -3.0)) == 0.0


def test_position_error_nonzero_3_4_5():
    assert position_error((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_position_error_vectorised():
    true = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
    est = np.array([[0, 1, 0], [1, 1, 2]], dtype=float)
    np.testing.assert_allclose(position_error(true, est), [1.0, 0.0])


def test_position_error_rejects_mismatched_batches():
    with pytest.raises(ValueError):
        position_error(np.zeros((3, 3)), np.zeros((2, 3)))


# --- heading error --------------------------------------------------------

def test_heading_error_zero():
    assert heading_error(1.2, 1.2) == 0.0


def test_heading_error_wraparound():
    assert heading_error(math.pi - 0.1, -math.pi + 0.1) == pytest.approx(0.2)
    assert heading_error(-math.pi + 0.05, math.pi - 0.05) == pytest.approx(0.1)


def test_heading_error_is_within_zero_to_pi():
    rng = np.random.default_rng(0)
    a = rng.uniform(-10, 10, 1000)
    b = rng.uniform(-10, 10, 1000)
    err = heading_error(a, b)
    assert np.all(err >= 0) and np.all(err <= math.pi)


def test_wrap_angle_range():
    np.testing.assert_allclose(wrap_angle([3 * math.pi, -3 * math.pi / 2]),
                               [-math.pi, math.pi / 2])


# --- RMSE and trajectories -----------------------------------------------

def test_rmse_value():
    assert rmse([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))


def test_trajectory_errors_summary():
    true = np.array([[0, 0, 0], [1, 0, math.pi - 0.1], [2, 0, 0]], dtype=float)
    est = np.array([[0, 0, 0], [1, 1, -math.pi + 0.1], [2, 0, 0]], dtype=float)
    out = trajectory_errors(true, est)
    np.testing.assert_allclose(out["position_errors"], [0, 1, 0])
    np.testing.assert_allclose(out["heading_errors"], [0, 0.2, 0], atol=1e-12)
    assert out["position_rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert out["final_position_error"] == 0.0
    assert out["steps"] == 3


def test_trajectory_errors_rejects_length_mismatch():
    with pytest.raises(ValueError):
        trajectory_errors(np.zeros((4, 3)), np.zeros((3, 3)))


# --- convergence ----------------------------------------------------------

def test_convergence_detected_at_first_stable_step():
    errors = [2.0, 1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1]
    assert convergence_time(errors, threshold=0.3, stable_steps=3) == 3


def test_convergence_ignores_brief_dip():
    errors = [1.0, 0.1, 0.9, 0.1, 0.1, 0.1]
    assert convergence_time(errors, threshold=0.3, stable_steps=3) == 3


def test_failed_convergence_returns_none():
    assert convergence_time([1.0] * 50, threshold=0.3, stable_steps=10) is None


def test_convergence_requires_full_stable_window():
    # Only 5 good steps at the end, need 10.
    errors = [1.0] * 20 + [0.1] * 5
    assert convergence_time(errors, threshold=0.3, stable_steps=10) is None


def test_convergence_rejects_bad_parameters():
    with pytest.raises(ValueError):
        convergence_time([0.1], threshold=0.0)
    with pytest.raises(ValueError):
        convergence_time([0.1], stable_steps=0)


# --- success rate ---------------------------------------------------------

def test_success_rate_calculation():
    assert success_rate([True, False, True, True]) == pytest.approx(0.75)
    assert success_rate([False, False]) == 0.0
    assert success_rate([1, 1, 1]) == 1.0


# --- empty input ----------------------------------------------------------

def test_empty_inputs():
    assert convergence_time([]) is None
    with pytest.raises(ValueError):
        success_rate([])
    with pytest.raises(ValueError):
        rmse([])
    with pytest.raises(ValueError):
        trajectory_errors([], [])


# --- runtime --------------------------------------------------------------

def test_measure_runtime_returns_result_and_elapsed():
    result, elapsed = measure_runtime(lambda x: (time.sleep(0.01), x * 2)[1], 21)
    assert result == 42
    assert elapsed >= 0.009
