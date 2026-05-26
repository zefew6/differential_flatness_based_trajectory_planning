"""Pure 2-D occupancy grid and ESDF queries for MINCO planning.

This module intentionally has no MuJoCo imports. Simulator-specific map
construction lives in :mod:`m0.minco_planner.adapters.mujoco`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class GridMap2DParams:
    """Parameters-only constructor for :class:`GridMap2D`."""

    resolution: float = 0.1
    size_x: float = 20.0
    size_y: float = 20.0
    origin_at_center: bool = False
    robot_radius: float = 0.0
    margin: float = 0.0


class GridMap2D:
    """2-D occupancy grid with ESDF distance and gradient queries.

    The grid stores rows as y and columns as x, matching the older
    ``m0.utils.gridmap_2d.GridMap`` interface used by A*.
    """

    def __init__(
        self,
        params: GridMap2DParams | None = None,
        *,
        resolution: float | None = None,
        width: float | None = None,
        height: float | None = None,
        robot_radius: float | None = None,
        margin: float | None = None,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        if params is not None and not isinstance(params, GridMap2DParams):
            raise TypeError(
                "GridMap2D is now pure Python. Use MujocoGridMap2D from "
                "m0.minco_planner.adapters.mujoco for MuJoCo models."
            )

        if params is None:
            params = GridMap2DParams(
                resolution=0.1 if resolution is None else float(resolution),
                size_x=20.0 if width is None else float(width),
                size_y=20.0 if height is None else float(height),
                origin_at_center=False,
                robot_radius=0.0 if robot_radius is None else float(robot_radius),
                margin=0.0 if margin is None else float(margin),
            )

        self.resolution = float(params.resolution)
        self.width = float(params.size_x)
        self.height = float(params.size_y)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        if params.origin_at_center:
            self.origin_x = -0.5 * self.width
            self.origin_y = -0.5 * self.height

        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)
        self.robot_radius = float(params.robot_radius)
        self.margin = float(params.margin)
        self.inflation_radius = self.robot_radius + self.margin

        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.esdf: np.ndarray | None = None
        self._esdf_valid = False
        self._params = params

    @classmethod
    def from_occupancy(
        cls,
        occ: np.ndarray,
        *,
        resolution: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        robot_radius: float = 0.0,
        margin: float = 0.0,
        update_esdf: bool = True,
    ) -> "GridMap2D":
        occ = np.asarray(occ, dtype=np.float32)
        height, width = occ.shape
        grid = cls(
            resolution=resolution,
            width=width * resolution,
            height=height * resolution,
            robot_radius=robot_radius,
            margin=margin,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        grid.set_occupancy(occ, update_esdf=update_esdf)
        return grid

    @property
    def nx(self) -> int:
        return int(self.grid_width)

    @property
    def ny(self) -> int:
        return int(self.grid_height)

    @property
    def occ(self) -> np.ndarray:
        return self.grid

    @property
    def min_boundary(self) -> np.ndarray:
        return np.array([self.origin_x, self.origin_y], dtype=np.float64)

    @property
    def max_boundary(self) -> np.ndarray:
        return np.array(
            [self.origin_x + self.width, self.origin_y + self.height],
            dtype=np.float64,
        )

    def coor_to_index(self, coor) -> tuple[int, int]:
        x, y = float(coor[0]), float(coor[1])
        col = int((x - self.origin_x + self.resolution / 2.0) / self.resolution)
        row = int((y - self.origin_y + self.resolution / 2.0) / self.resolution)
        return row, col

    def index_to_coor(self, ind) -> tuple[float, float]:
        row, col = int(ind[0]), int(ind[1])
        x = self.origin_x + (col - self.resolution / 2.0) * self.resolution
        y = self.origin_y + (row - self.resolution / 2.0) * self.resolution
        return float(x), float(y)

    def index_to_pos(self, idx_xy) -> np.ndarray:
        """Compatibility alias: ``index_to_pos([col, row])``."""

        x, y = self.index_to_coor((int(idx_xy[1]), int(idx_xy[0])))
        return np.array([x, y], dtype=np.float64)

    def is_valid_index(self, index) -> bool:
        row, col = int(index[0]), int(index[1])
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width

    def is_occupied_index(self, index) -> bool:
        row, col = int(index[0]), int(index[1])
        return bool(self.grid[row, col] > 0.0)

    def set_occupancy(self, occ: np.ndarray, *, update_esdf: bool = True) -> None:
        occ = np.asarray(occ, dtype=np.float32)
        if occ.shape == (self.grid_width, self.grid_height):
            self.grid = occ.T.copy()
        elif occ.shape == (self.grid_height, self.grid_width):
            self.grid = occ.copy()
        else:
            raise ValueError(
                f"occ shape {occ.shape} does not match "
                f"({self.grid_height}, {self.grid_width})"
            )
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def clear(self, *, update_esdf: bool = False) -> None:
        self.grid.fill(0.0)
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_circle_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        *,
        update_esdf: bool = True,
    ) -> None:
        cx, cy = float(center[0]), float(center[1])
        r_total = float(radius) + self.inflation_radius
        res = float(self.resolution)
        col_min = max(0, int((cx - r_total - self.origin_x) / res))
        col_max = min(self.grid_width - 1, int((cx + r_total - self.origin_x) / res) + 1)
        row_min = max(0, int((cy - r_total - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((cy + r_total - self.origin_y) / res) + 1)
        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)
        wx = self.origin_x + (cc + 0.5) * res
        wy = self.origin_y + (rr + 0.5) * res
        mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= r_total ** 2
        self.grid[row_min : row_max + 1, col_min : col_max + 1][mask] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_rectangle_obstacle(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        *,
        update_esdf: bool = True,
    ) -> None:
        inf = self.inflation_radius
        xmin -= inf
        xmax += inf
        ymin -= inf
        ymax += inf
        res = float(self.resolution)
        col_min = max(0, int((xmin - self.origin_x) / res))
        col_max = min(self.grid_width - 1, int((xmax - self.origin_x) / res) + 1)
        row_min = max(0, int((ymin - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((ymax - self.origin_y) / res) + 1)
        self.grid[row_min : row_max + 1, col_min : col_max + 1] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_polygon_obstacle(self, verts: np.ndarray, *, update_esdf: bool = True) -> None:
        verts = np.asarray(verts, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("verts must have shape (N, 2)")
        res = float(self.resolution)
        xs = verts[:, 0]
        ys = verts[:, 1]
        col_min = max(0, int((xs.min() - self.origin_x) / res))
        col_max = min(self.grid_width - 1, int((xs.max() - self.origin_x) / res) + 1)
        row_min = max(0, int((ys.min() - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((ys.max() - self.origin_y) / res) + 1)
        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)
        wx = self.origin_x + (cc + 0.5) * res
        wy = self.origin_y + (rr + 0.5) * res
        inside = self._points_in_polygon(wx, wy, verts)
        self.grid[row_min : row_max + 1, col_min : col_max + 1][inside] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def update_esdf(self) -> None:
        """Build an ESDF where free space is positive and obstacles are negative."""

        occ = np.asarray(self.grid, dtype=bool)
        res = float(self.resolution)
        dist_to_obs = distance_transform_edt(~occ) * res
        dist_in_obs = distance_transform_edt(occ) * res
        esdf = dist_to_obs.copy()
        esdf[occ] = -dist_in_obs[occ] + res
        self.esdf = esdf.T.copy()
        self._esdf_valid = True

    def get_distance(self, pos_xy) -> float:
        dist, _ = self.get_distance_and_gradient(pos_xy)
        return float(dist)

    def get_distance_and_gradient(self, pos_xy) -> Tuple[float, np.ndarray]:
        self._ensure_esdf()
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if (
            x < self.origin_x
            or x > self.origin_x + self.width
            or y < self.origin_y
            or y > self.origin_y + self.height
        ):
            return float("inf"), np.zeros(2, dtype=np.float64)

        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)
        col_f = (x - self.origin_x) / res - 0.5
        row_f = (y - self.origin_y) / res - 0.5
        col0 = int(np.clip(int(col_f), 0, nx - 2))
        row0 = int(np.clip(int(row_f), 0, ny - 2))
        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)
        v00 = float(self.esdf[col0, row0])
        v10 = float(self.esdf[col0 + 1, row0])
        v01 = float(self.esdf[col0, row0 + 1])
        v11 = float(self.esdf[col0 + 1, row0 + 1])
        dist = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        grad_x = ((v10 - v00) * (1 - dy) + (v11 - v01) * dy) / res
        grad_y = ((v01 - v00) * (1 - dx) + (v11 - v10) * dx) / res
        return float(dist), np.array([grad_x, grad_y], dtype=np.float64)

    def get_distance_and_gradient_batch(self, positions: np.ndarray):
        self._ensure_esdf()
        positions = np.asarray(positions, dtype=np.float64)
        n_pts = len(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        distances = np.full(n_pts, np.inf, dtype=np.float64)
        gradients = np.zeros((n_pts, 2), dtype=np.float64)
        valid = (
            (x >= self.origin_x)
            & (x <= self.origin_x + self.width)
            & (y >= self.origin_y)
            & (y <= self.origin_y + self.height)
        )
        if not np.any(valid):
            return distances, gradients

        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)
        xv = x[valid]
        yv = y[valid]
        col_f = (xv - self.origin_x) / res - 0.5
        row_f = (yv - self.origin_y) / res - 0.5
        col0 = np.clip(col_f.astype(np.int32), 0, nx - 2)
        row0 = np.clip(row_f.astype(np.int32), 0, ny - 2)
        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)
        v00 = self.esdf[col0, row0]
        v10 = self.esdf[col0 + 1, row0]
        v01 = self.esdf[col0, row0 + 1]
        v11 = self.esdf[col0 + 1, row0 + 1]
        dist_v = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        grad_x = ((v10 - v00) * (1 - dy) + (v11 - v01) * dy) / res
        grad_y = ((v01 - v00) * (1 - dx) + (v11 - v10) * dx) / res
        distances[valid] = dist_v
        gradients[valid, 0] = grad_x
        gradients[valid, 1] = grad_y
        return distances, gradients

    def show_map(self) -> None:
        import matplotlib.pyplot as plt

        plt.imshow(self.grid, cmap="gray")
        plt.title("2D Occupancy Grid")
        plt.show()

    def _ensure_esdf(self) -> None:
        if not self._esdf_valid or self.esdf is None:
            self.update_esdf()

    @staticmethod
    def _points_in_polygon(wx: np.ndarray, wy: np.ndarray, verts: np.ndarray) -> np.ndarray:
        inside = np.zeros(wx.shape, dtype=bool)
        n = len(verts)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            cond = ((yi > wy) != (yj > wy)) & (
                wx < (xj - xi) * (wy - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= cond
            j = i
        return inside
