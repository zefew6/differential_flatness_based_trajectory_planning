"""MuJoCo adapters for the pure MINCO planner."""

from __future__ import annotations

import numpy as np

from ..corridor.sfc import _halfplanes_to_convex_polygon
from ..maps import GridMap2D


class MujocoGridMap2D(GridMap2D):
    """Build a pure :class:`GridMap2D` from MuJoCo geoms.

    The resulting object exposes the same grid/ESDF interface as the pure map
    and can be passed to A*, ESDF obstacle costs, SFC builders, and FIRI.
    """

    def __init__(
        self,
        model_or_params=None,
        data=None,
        resolution=None,
        width=None,
        height=None,
        robot_radius=None,
        margin=None,
        *,
        model=None,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        update_esdf: bool = True,
    ):
        actual_model = model if model is not None else model_or_params
        if actual_model is None or data is None:
            raise ValueError("MujocoGridMap2D requires model and data.")
        if resolution is None or width is None or height is None:
            raise ValueError("resolution, width, and height are required.")

        super().__init__(
            resolution=float(resolution),
            width=float(width),
            height=float(height),
            robot_radius=0.0 if robot_radius is None else float(robot_radius),
            margin=0.0 if margin is None else float(margin),
            origin_x=origin_x,
            origin_y=origin_y,
        )
        self.model = actual_model
        self.data = data
        self.create_grid(update_esdf=update_esdf)

    def create_grid(self, *, update_esdf: bool = True) -> None:
        import mujoco

        self.grid.fill(0.0)
        for geom_id in range(self.model.ngeom):
            geom_type = self.model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self._add_box(geom_id)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                self._add_sphere(geom_id)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self._add_cylinder(geom_id)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self._add_capsule(geom_id)
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def _grid_points(self):
        res = float(self.resolution)
        cols = np.arange(self.grid_width)
        rows = np.arange(self.grid_height)
        x_c = self.origin_x + (cols + 0.5) * res
        y_c = self.origin_y + (rows + 0.5) * res
        return np.meshgrid(x_c, y_c)

    def _add_box(self, geom_id: int) -> None:
        center = self.data.geom_xpos[geom_id]
        lx, ly, _ = self.model.geom_size[geom_id]
        rot = self.data.geom_xmat[geom_id].reshape(3, 3)
        inf = self.inflation_radius
        local_pts = np.array(
            [
                [-lx - inf, -ly - inf],
                [lx + inf, -ly - inf],
                [lx + inf, ly + inf],
                [-lx - inf, ly + inf],
            ],
            dtype=np.float64,
        )
        world_pts = local_pts @ rot[:2, :2].T + center[:2]
        xx, yy = self._grid_points()
        inside = self._points_in_polygon(xx, yy, world_pts)
        self.grid[inside] = 1.0

    def _add_sphere(self, geom_id: int) -> None:
        center = self.data.geom_xpos[geom_id]
        radius = float(self.model.geom_size[geom_id][0]) + self.inflation_radius
        xx, yy = self._grid_points()
        self.grid[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2] = 1.0

    def _add_cylinder(self, geom_id: int) -> None:
        center = self.data.geom_xpos[geom_id]
        radius = float(self.model.geom_size[geom_id][0]) + self.inflation_radius
        xx, yy = self._grid_points()
        self.grid[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2] = 1.0

    def _add_capsule(self, geom_id: int) -> None:
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        radius = float(self.model.geom_size[geom_id][0]) + self.inflation_radius
        half_len = float(self.model.geom_size[geom_id][1])
        rot = self.data.geom_xmat[geom_id].reshape(3, 3)
        axis_xy = rot[:2, 2] * half_len
        p0 = center[:2] - axis_xy
        p1 = center[:2] + axis_xy

        xx, yy = self._grid_points()
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        seg = p1 - p0
        denom = float(np.dot(seg, seg))
        if denom <= 1e-12:
            dist_sq = np.sum((pts - p0) ** 2, axis=1)
        else:
            alpha = np.clip(((pts - p0) @ seg) / denom, 0.0, 1.0)
            nearest = p0 + alpha[:, None] * seg
            dist_sq = np.sum((pts - nearest) ** 2, axis=1)
        self.grid[dist_sq.reshape(self.grid.shape) <= radius**2] = 1.0


def draw_sfc_in_mujoco(
    viewer,
    hPolys_per_piece,
    grid_map=None,
    z=0.02,
    edge_rgba=np.array([0.0, 0.6, 1.0, 1.0]),
    center_rgba=np.array([0.0, 0.6, 1.0, 0.25]),
    edge_width=0.003,
    center_size=0.03,
):
    """Draw SFC corridor boundaries on a ``MujocoViewer``-like object."""

    clip_box = None
    if grid_map is not None and hasattr(grid_map, "min_boundary") and hasattr(grid_map, "max_boundary"):
        mn = grid_map.min_boundary
        mx = grid_map.max_boundary
        clip_box = (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    for hpoly in hPolys_per_piece:
        if hpoly is None:
            continue
        poly = _halfplanes_to_convex_polygon(hpoly, clip_box=clip_box)
        if poly is None or len(poly) < 2:
            continue

        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            try:
                viewer.draw_line_segment(
                    [p0[0], p0[1], z],
                    [p1[0], p1[1], z],
                    width=edge_width,
                    rgba=edge_rgba,
                )
            except Exception:
                pass

        centroid = np.mean(poly, axis=0)
        try:
            viewer.draw_point([centroid[0], centroid[1], z], size=center_size, rgba=center_rgba)
        except Exception:
            pass
