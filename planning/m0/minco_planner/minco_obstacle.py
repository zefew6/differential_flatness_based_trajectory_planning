"""Backward-compatible facade for map, obstacle-cost, and corridor modules.

New code should prefer:

* ``m0.minco_planner.maps`` for pure grid maps
* ``m0.minco_planner.costs`` for cost terms
* ``m0.minco_planner.corridor`` for SFC/FIRI builders
* ``m0.minco_planner.adapters.mujoco`` for MuJoCo integration
"""

from __future__ import annotations

from .corridor import (
    build_corridor_for_segment,
    build_corridors,
    build_corridors_inflated_cubes,
    build_sfc_from_gridmap,
    draw_sfc_corridors,
    extract_obs_points_from_gridmap,
)
from .costs import ObstacleConstraint, SFCObstacleConstraint
from .maps import GridMap2D as _PureGridMap2D
from .maps import GridMap2DParams


class GridMap2D(_PureGridMap2D):
    """Compatibility wrapper.

    If a MuJoCo model/data pair is supplied, this lazily creates
    ``MujocoGridMap2D``. Otherwise it behaves like the pure Python grid map.
    """

    def __new__(
        cls,
        model_or_params=None,
        data=None,
        resolution=None,
        width=None,
        height=None,
        robot_radius=None,
        margin=None,
        *,
        model=None,
        **kwargs,
    ):
        actual_model = model if model is not None else model_or_params
        if actual_model is not None and not isinstance(actual_model, GridMap2DParams):
            from .adapters.mujoco import MujocoGridMap2D

            return MujocoGridMap2D(
                model_or_params,
                data,
                resolution,
                width,
                height,
                robot_radius,
                margin,
                model=model,
                **kwargs,
            )
        return super().__new__(cls)

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
        **kwargs,
    ):
        actual_model = model if model is not None else model_or_params
        if actual_model is not None and not isinstance(actual_model, GridMap2DParams):
            return
        if isinstance(model_or_params, GridMap2DParams):
            super().__init__(model_or_params, origin_x=origin_x, origin_y=origin_y)
        else:
            super().__init__(
                resolution=resolution,
                width=width,
                height=height,
                robot_radius=robot_radius,
                margin=margin,
                origin_x=origin_x,
                origin_y=origin_y,
            )


def draw_sfc_in_mujoco(*args, **kwargs):
    from .adapters.mujoco import draw_sfc_in_mujoco as _draw

    return _draw(*args, **kwargs)


__all__ = [
    "GridMap2D",
    "GridMap2DParams",
    "ObstacleConstraint",
    "SFCObstacleConstraint",
    "build_corridor_for_segment",
    "build_corridors",
    "build_corridors_inflated_cubes",
    "build_sfc_from_gridmap",
    "draw_sfc_corridors",
    "draw_sfc_in_mujoco",
    "extract_obs_points_from_gridmap",
]
