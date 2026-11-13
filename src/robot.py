"""Planar robot motion and noisy odometry."""

from __future__ import annotations

import numpy as np


def wrap_angle(
    angle: float | np.ndarray,
) -> float | np.ndarray:
    """Wrap an angle to the interval [-pi, pi]."""
    wrapped = (
        np.asarray(angle) + np.pi
    ) % (2.0 * np.pi) - np.pi

    if wrapped.ndim == 0:
        return float(wrapped)

    return wrapped


class Robot:
    """Simple planar robot with noisy odometry."""

    def __init__(
        self,
        pose: tuple[float, float, float] = (
            1.5,
            1.2,
            0.2,
        ),
        seed: int = 1,
        odometry_std: tuple[float, float, float] = (
            0.015,
            0.015,
            0.01,
        ),
    ) -> None:
        self.pose = np.asarray(
            pose,
            dtype=float,
        )

        if self.pose.shape != (3,):
            raise ValueError(
                "pose must contain x, y, and theta"
            )

        self.rng = np.random.default_rng(seed)

        self.odometry_std = np.asarray(
            odometry_std,
            dtype=float,
        )

        if self.odometry_std.shape != (3,):
            raise ValueError(
                "odometry_std must contain three values"
            )

    def step(
        self,
        control: tuple[float, float],
        dt: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move the robot and return true pose and noisy odometry.

        Parameters
        ----------
        control:
            `(linear_velocity, angular_velocity)`.
        dt:
            Time step in seconds.
        """
        if len(control) != 2:
            raise ValueError(
                "control must be (linear_velocity, angular_velocity)"
            )

        linear_velocity, angular_velocity = control

        true_delta = np.array(
            [
                linear_velocity
                * dt
                * np.cos(self.pose[2]),
                linear_velocity
                * dt
                * np.sin(self.pose[2]),
                angular_velocity * dt,
            ],
            dtype=float,
        )

        self.pose += true_delta
        self.pose[2] = wrap_angle(
            self.pose[2]
        )

        noisy_odometry = (
            true_delta
            + self.rng.normal(
                loc=0.0,
                scale=self.odometry_std,
                size=3,
            )
        )

        return self.pose.copy(), noisy_odometry