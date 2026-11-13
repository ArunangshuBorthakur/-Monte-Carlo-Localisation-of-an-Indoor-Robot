import numpy as np

from src.world import World


def test_world_has_expected_dimensions():
    world = World()

    assert world.grid.ndim == 2
    assert world.grid.shape[1] == 120
    assert world.grid.shape[0] == 100


def test_interior_point_is_free():
    world = World()

    assert world.is_free(1.5, 1.2)


def test_boundary_is_occupied():
    world = World()

    assert not world.is_free(0.1, 0.1)


def test_vectorised_free_space_query():
    world = World()

    values = world.is_free(
        np.array([1.5, 0.1]),
        np.array([1.2, 0.1]),
    )

    assert values.shape == (2,)
    assert values[0]
    assert not values[1]