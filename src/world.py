"""Known 2D indoor occupancy-grid world."""

from __future__ import annotations

import numpy as np


class World:
    """Indoor 2D occupancy-grid map."""

    def __init__(
        self,
        width: float = 12.0,
        height: float = 10.0,
        resolution: float = 0.1,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution

        rows = int(np.ceil(height / resolution))
        columns = int(np.ceil(width / resolution))

        # True means occupied; False means free.
        self.grid = np.zeros(
            (rows, columns),
            dtype=bool,
        )

        # Outer boundaries.
        self._add_rectangle(0.0, 0.0, width, 0.25)
        self._add_rectangle(0.0, height - 0.25, width, 0.25)
        self._add_rectangle(0.0, 0.0, 0.25, height)
        self._add_rectangle(width - 0.25, 0.0, 0.25, height)

        # Internal walls.
        self._add_rectangle(3.0, 1.5, 0.25, 5.0)
        self._add_rectangle(7.0, 3.5, 0.25, 5.0)
        self._add_rectangle(4.5, 7.0, 3.5, 0.25)
        self._add_rectangle(8.5, 1.5, 2.0, 0.25)

    def _add_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Add an occupied rectangular region."""
        x0 = int(np.floor(x / self.resolution))
        y0 = int(np.floor(y / self.resolution))
        x1 = int(np.ceil((x + width) / self.resolution))
        y1 = int(np.ceil((y + height) / self.resolution))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.grid.shape[1], x1)
        y1 = min(self.grid.shape[0], y1)

        self.grid[y0:y1, x0:x1] = True

    def is_free(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
    ) -> bool | np.ndarray:
        """Return whether world coordinates are free."""
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )

        scalar_input = x_array.ndim == 0

        x_flat = x_array.reshape(-1)
        y_flat = y_array.reshape(-1)

        grid_x = np.floor(
            x_flat / self.resolution
        ).astype(int)

        grid_y = np.floor(
            y_flat / self.resolution
        ).astype(int)

        valid = (
            (grid_x >= 0)
            & (grid_y >= 0)
            & (grid_x < self.grid.shape[1])
            & (grid_y < self.grid.shape[0])
        )

        result = np.zeros(
            x_flat.shape,
            dtype=bool,
        )

        result[valid] = ~self.grid[
            grid_y[valid],
            grid_x[valid],
        ]

        result = result.reshape(x_array.shape)

        if scalar_input:
            return bool(result.item())

        return result

    def ray_cast(
        self,
        poses: np.ndarray,
        angles: np.ndarray,
        max_range: float = 8.0,
        step: float = 0.05,
    ) -> np.ndarray:
        """Cast lidar rays from one or more poses.

        Parameters
        ----------
        poses:
            Pose array with shape `(3,)` or `(N, 3)`.
        angles:
            Relative lidar beam angles in radians.
        max_range:
            Maximum lidar range in metres.
        step:
            Ray-marching resolution in metres.

        Returns
        -------
        numpy.ndarray
            Ranges with shape `(number_of_beams,)` for one pose or
            `(number_of_poses, number_of_beams)` for multiple poses.
        """
        pose_array = np.asarray(poses, dtype=float)
        single_pose = pose_array.ndim == 1

        if single_pose:
            pose_array = pose_array[None, :]

        if pose_array.ndim != 2 or pose_array.shape[1] != 3:
            raise ValueError(
                "poses must have shape (3,) or (N, 3)"
            )

        beam_angles = np.asarray(
            angles,
            dtype=float,
        )

        distances = np.arange(
            0.0,
            max_range + step,
            step,
        )

        ray_angles = (
            pose_array[:, 2, None, None]
            + beam_angles[None, :, None]
        )

        x = (
            pose_array[:, 0, None, None]
            + distances[None, None, :]
            * np.cos(ray_angles)
        )

        y = (
            pose_array[:, 1, None, None]
            + distances[None, None, :]
            * np.sin(ray_angles)
        )

        free = self.is_free(x, y)
        hits = ~free

        first_hit_index = np.argmax(
            hits,
            axis=2,
        )

        has_hit = np.any(
            hits,
            axis=2,
        )

        ranges = np.full(
            (len(pose_array), len(beam_angles)),
            max_range,
            dtype=float,
        )

        rows, columns = np.where(has_hit)

        ranges[rows, columns] = distances[
            first_hit_index[rows, columns]
        ]

        if single_pose:
            return ranges[0]

        return ranges