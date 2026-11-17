import numpy as np

from src.robot import Robot, wrap_angle


def test_wrap_angle():
    assert np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(3.0 * np.pi), -np.pi)


def test_robot_moves_forward():
    robot = Robot(
        pose=(1.0, 1.0, 0.0),
        seed=10,
        odometry_std=(0.0, 0.0, 0.0),
    )

    pose, odometry = robot.step(
        control=(1.0, 0.0),
        dt=1.0,
    )

    assert np.allclose(
        pose,
        [2.0, 1.0, 0.0],
    )

    assert np.allclose(
        odometry,
        [1.0, 0.0, 0.0],
    )


def test_robot_returns_noisy_odometry():
    robot = Robot(seed=10)

    pose, odometry = robot.step(
        control=(0.5, 0.1),
    )

    assert pose.shape == (3,)
    assert odometry.shape == (3,)
    assert np.all(np.isfinite(pose))
    assert np.all(np.isfinite(odometry))