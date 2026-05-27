"""High-level MINCO planner orchestration.

This module turns a geometric path into a MINCO trajectory. It is intentionally
thin: path tools, time allocation, corridor construction, and numerical
optimization each live in their own modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .corridor import build_sfc_from_gridmap
from .minco import MINCO
from .path_tools import (
    check_line_collision,
    preprocess_path,
    push_waypoints_to_clearance,
    resample_path,
    uniform_resample_path,
)
from .time_allocation import allocate_time, trapezoid_duration
from .trajectory_optimizer import TrajectoryOptimizer


@dataclass
class SFCOptions:
    """Safe-flight-corridor options used by the high-level planner."""

    build_method: str = "firi"
    search_radius: float = 6.0
    subsample: int = 2
    n_bins: int = 36
    inflate_step_cells: int = 1
    push_to_clearance: bool = True
    max_iters: int = 60
    step_size: float = 0.05
    target_clearance: float = 0.40
    safe_margin: float = 0.0
    weight: float = 5e5


@dataclass
class MincoPlannerConfig:
    """Reusable configuration for :class:`MincoPlanner`."""

    obstacle_method: str = "esdf"
    max_seg_len: float = 1.2
    sfc: SFCOptions = field(default_factory=SFCOptions)
    optimizer_params: dict = field(default_factory=dict)
    debug_print_every: int | None = None


@dataclass
class MincoPlanResult:
    """Result returned by :class:`MincoPlanner`."""

    minco: MINCO
    path: np.ndarray
    resampled: np.ndarray
    waypoints: np.ndarray
    success: bool
    final_cost: float
    optimize_time_ms: float
    corridors: list | None = None

    @property
    def total_time(self) -> float:
        return float(np.sum(self.minco.T))

    @property
    def cost_time_ms(self) -> float:
        return self.optimize_time_ms

    def eval(self, t: float):
        return self.minco.eval(t)

    def sample_xy(self, n: int = 800) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        ts = np.linspace(0.0, self.total_time, int(n))
        return np.vstack([self.minco.eval(t)[0] for t in ts])

    def as_online_dict(self) -> dict:
        """Compatibility dictionary for older online-replan examples."""

        return {
            "path": self.path,
            "resampled": self.resampled,
            "waypoints": self.waypoints,
            "minco": self.minco,
            "cost_time_ms": self.optimize_time_ms,
            "success": self.success,
            "final_cost": self.final_cost,
            "corridors": self.corridors,
        }


class MincoPlanner:
    """Reusable high-level MINCO planner.

    The planner can be used in two common ways:

    - ``plan(grid_map=..., start_xy=..., goal_xy=...)`` runs A* then MINCO.
    - ``plan(path=...)`` optimizes a supplied geometric path directly.
    """

    def __init__(
        self,
        obstacle_method: str | MincoPlannerConfig = "esdf",
        optimizer: TrajectoryOptimizer | None = None,
        config: MincoPlannerConfig | None = None,
    ):
        if isinstance(obstacle_method, MincoPlannerConfig):
            if config is not None:
                raise ValueError("Pass either obstacle_method as a config or config=, not both.")
            config = obstacle_method
        elif config is None:
            config = MincoPlannerConfig(obstacle_method=obstacle_method)
        elif obstacle_method != "esdf":
            config = replace(config, obstacle_method=obstacle_method)

        self.config = config
        self.optimizer = optimizer if optimizer is not None else TrajectoryOptimizer(config.obstacle_method)
        self.obstacle_method = self.optimizer.obstacle_method
        if self.obstacle_method != self.config.obstacle_method:
            self.config = replace(self.config, obstacle_method=self.obstacle_method)
        self.grid_map = None
        self.last_result: MincoPlanResult | None = None
        if self.config.optimizer_params:
            self.optimizer.set_params(**self.config.optimizer_params)
        if self.config.debug_print_every is not None:
            self.optimizer.set_debug_print_every(self.config.debug_print_every)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def set_grid_map(self, grid_map) -> None:
        self.grid_map = grid_map
        self.optimizer.set_grid_map(grid_map)

    def set_sfc_corridors(self, hpolys_per_traj: list) -> None:
        self.optimizer.set_sfc_corridors(hpolys_per_traj)

    def set_params(self, *args, **kwargs) -> None:
        self.optimizer.set_params(*args, **kwargs)

    def set_debug_print_every(self, n: int) -> None:
        self.optimizer.set_debug_print_every(n)

    def build_sfc_corridors(
        self,
        waypoints,
        search_radius: float | None = None,
        subsample: int | None = None,
        n_bins: int | None = None,
        method: str | None = None,
        inflate_step_cells: int | None = None,
    ) -> list:
        if self.grid_map is None:
            raise RuntimeError("Call set_grid_map() before build_sfc_corridors().")

        sfc = self.config.sfc
        search_radius = sfc.search_radius if search_radius is None else float(search_radius)
        subsample = sfc.subsample if subsample is None else int(subsample)
        n_bins = sfc.n_bins if n_bins is None else int(n_bins)
        method = sfc.build_method if method is None else method
        inflate_step_cells = (
            sfc.inflate_step_cells if inflate_step_cells is None else int(inflate_step_cells)
        )

        t0 = time.perf_counter()
        hpolys = build_sfc_from_gridmap(
            self.grid_map,
            waypoints,
            search_radius=search_radius,
            subsample=subsample,
            n_bins=n_bins,
            method=method,
            inflate_step_cells=inflate_step_cells,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[SFC] Built {len(hpolys)} corridor segments by '{method}' in {elapsed_ms:.1f} ms")
        self.set_sfc_corridors([hpolys])
        return hpolys

    def plan_from_path(
        self,
        path: np.ndarray,
        head_pva: np.ndarray | None = None,
        tail_pva: np.ndarray | None = None,
        max_seg_len: float | None = None,
        waypoints: np.ndarray | None = None,
        sfc_push_to_clearance: bool | None = None,
        sfc_max_iters: int | None = None,
        sfc_step_size: float | None = None,
        sfc_target_clearance: float | None = None,
        sfc_search_radius: float | None = None,
        sfc_subsample: int | None = None,
        sfc_n_bins: int | None = None,
        sfc_build_method: str | None = None,
        sfc_inflate_step_cells: int | None = None,
        sfc_safe_margin: float | None = None,
        sfc_wei: float | None = None,
    ) -> MincoPlanResult:
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            raise ValueError("path needs at least two points")

        sfc = self.config.sfc
        max_seg_len = self.config.max_seg_len if max_seg_len is None else float(max_seg_len)
        sfc_push_to_clearance = sfc.push_to_clearance if sfc_push_to_clearance is None else bool(sfc_push_to_clearance)
        sfc_max_iters = sfc.max_iters if sfc_max_iters is None else int(sfc_max_iters)
        sfc_step_size = sfc.step_size if sfc_step_size is None else float(sfc_step_size)
        sfc_target_clearance = (
            sfc.target_clearance if sfc_target_clearance is None else float(sfc_target_clearance)
        )
        sfc_search_radius = sfc.search_radius if sfc_search_radius is None else float(sfc_search_radius)
        sfc_subsample = sfc.subsample if sfc_subsample is None else int(sfc_subsample)
        sfc_n_bins = sfc.n_bins if sfc_n_bins is None else int(sfc_n_bins)
        sfc_build_method = sfc.build_method if sfc_build_method is None else sfc_build_method
        sfc_inflate_step_cells = (
            sfc.inflate_step_cells if sfc_inflate_step_cells is None else int(sfc_inflate_step_cells)
        )
        sfc_safe_margin = sfc.safe_margin if sfc_safe_margin is None else float(sfc_safe_margin)
        sfc_wei = sfc.weight if sfc_wei is None else float(sfc_wei)

        resampled = uniform_resample_path(path, max_seg_len=max_seg_len)
        full_pts = np.asarray(waypoints, dtype=float) if waypoints is not None else resampled
        if len(full_pts) < 2:
            raise ValueError("waypoints needs at least two points")
        if len(full_pts) == 2:
            full_pts = np.vstack([full_pts[0], 0.5 * (full_pts[0] + full_pts[1]), full_pts[1]])

        corridors = None
        if self.obstacle_method == "sfc":
            if self.grid_map is None:
                raise RuntimeError("SFC planning requires set_grid_map().")
            if sfc_push_to_clearance:
                full_pts = push_waypoints_to_clearance(
                    self.grid_map,
                    full_pts,
                    max_iters=sfc_max_iters,
                    step_size=sfc_step_size,
                    target_clearance=sfc_target_clearance,
                )
            corridors = self.build_sfc_corridors(
                full_pts,
                search_radius=sfc_search_radius,
                subsample=sfc_subsample,
                n_bins=sfc_n_bins,
                method=sfc_build_method,
                inflate_step_cells=sfc_inflate_step_cells,
            )
            self.set_params(sfc_safe_margin=sfc_safe_margin, wei_sfc=sfc_wei)

        head_pva = _default_pva(full_pts[0]) if head_pva is None else np.asarray(head_pva, dtype=float)
        tail_pva = _default_pva(full_pts[-1]) if tail_pva is None else np.asarray(tail_pva, dtype=float)
        inner_pts = full_pts[1:-1]
        durations = allocate_time(full_pts, self.optimizer.max_vel, self.optimizer.max_acc, self.optimizer.mini_T)

        t0 = time.perf_counter()
        success, final_cost = self.optimizer.optimize(
            iniStates=[head_pva],
            finStates=[tail_pva],
            initInnerPts=[inner_pts],
            initTs=np.array([np.sum(durations)]),
            initSegTs=[durations],
        )
        optimize_time_ms = (time.perf_counter() - t0) * 1000.0

        minco = _minco_from_optimized(self.optimizer.get_optimized_trajectories()[0], head_pva, tail_pva)
        result = MincoPlanResult(
            minco=minco,
            path=path,
            resampled=resampled,
            waypoints=full_pts,
            success=bool(success),
            final_cost=float(final_cost),
            optimize_time_ms=optimize_time_ms,
            corridors=corridors,
        )
        self.last_result = result
        return result

    def plan_grid(
        self,
        grid_map,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        max_seg_len: float | None = None,
        head_pva: np.ndarray | None = None,
        tail_pva: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        start_acc: np.ndarray | None = None,
        goal_vel: np.ndarray | None = None,
        goal_acc: np.ndarray | None = None,
        **kwargs,
    ) -> MincoPlanResult:
        from ..planning.a_star import graph_search

        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        self.set_grid_map(grid_map)
        path = graph_search(start=start_xy, goal=goal_xy, gridmap=grid_map)
        if path is None or len(path) < 2:
            raise RuntimeError("A* failed to find a feasible path")

        if head_pva is None:
            head_pva = np.array([
                start_xy,
                np.zeros(2) if start_vel is None else np.asarray(start_vel, dtype=float),
                np.zeros(2) if start_acc is None else np.asarray(start_acc, dtype=float),
            ], dtype=float)
        if tail_pva is None:
            tail_pva = np.array([
                goal_xy,
                np.zeros(2) if goal_vel is None else np.asarray(goal_vel, dtype=float),
                np.zeros(2) if goal_acc is None else np.asarray(goal_acc, dtype=float),
            ], dtype=float)

        return self.plan_from_path(
            path,
            head_pva=head_pva,
            tail_pva=tail_pva,
            max_seg_len=max_seg_len,
            **kwargs,
        )

    def plan(
        self,
        *,
        grid_map=None,
        start_xy: np.ndarray | None = None,
        goal_xy: np.ndarray | None = None,
        path: np.ndarray | None = None,
        head_pva: np.ndarray | None = None,
        tail_pva: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        start_acc: np.ndarray | None = None,
        goal_vel: np.ndarray | None = None,
        goal_acc: np.ndarray | None = None,
        **kwargs,
    ) -> MincoPlanResult:
        """Plan from either a grid map or an already computed path."""

        if grid_map is not None:
            self.set_grid_map(grid_map)

        if path is not None:
            path = np.asarray(path, dtype=float)
            if len(path) < 2:
                raise ValueError("path needs at least two points")
            if start_xy is None:
                start_xy = path[0]
            if goal_xy is None:
                goal_xy = path[-1]
            if head_pva is None:
                head_pva = _make_pva(start_xy, start_vel, start_acc)
            if tail_pva is None:
                tail_pva = _make_pva(goal_xy, goal_vel, goal_acc)
            return self.plan_from_path(path, head_pva=head_pva, tail_pva=tail_pva, **kwargs)

        if grid_map is None and self.grid_map is None:
            raise ValueError("plan() needs either path=... or grid_map=... with start_xy/goal_xy.")
        if start_xy is None or goal_xy is None:
            raise ValueError("start_xy and goal_xy are required when planning from a grid map.")

        return self.plan_grid(
            self.grid_map,
            start_xy,
            goal_xy,
            head_pva=head_pva,
            tail_pva=tail_pva,
            start_vel=start_vel,
            start_acc=start_acc,
            goal_vel=goal_vel,
            goal_acc=goal_acc,
            **kwargs,
        )

    def replan(
        self,
        grid_map,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        **kwargs,
    ) -> MincoPlanResult:
        """Online-replan convenience wrapper returning :class:`MincoPlanResult`."""

        return self.plan(grid_map=grid_map, start_xy=start_xy, goal_xy=goal_xy, **kwargs)

    # Backward-compatible surface used by existing examples.
    setGridMap = set_grid_map
    setSFCCorridors = set_sfc_corridors
    setParam = set_params
    setDebugPrintEvery = set_debug_print_every
    buildSFCCorridors = build_sfc_corridors

    def _check_line_collision(self, p1, p2) -> bool:
        return check_line_collision(self.grid_map, p1, p2)

    def preprocessPath(self, path) -> np.ndarray:
        if self.grid_map is None:
            raise RuntimeError("Call setGridMap() before preprocessPath().")
        return preprocess_path(self.grid_map, path)

    def resamplePath(self, path, max_seg_len: float = 1.5, dense_path: np.ndarray | None = None) -> np.ndarray:
        return resample_path(path, max_seg_len=max_seg_len, dense_path=dense_path)

    def uniform_resample_path(self, path: np.ndarray, max_seg_len: float = 3.0) -> np.ndarray:
        return uniform_resample_path(path, max_seg_len=max_seg_len)

    def push_waypoints_to_clearance(
        self,
        waypoints: np.ndarray,
        max_iters: int = 30,
        step_size: float = 0.08,
        target_clearance: float = 0.30,
    ) -> np.ndarray:
        if self.grid_map is None:
            raise RuntimeError("Call setGridMap() before push_waypoints_to_clearance().")
        return push_waypoints_to_clearance(
            self.grid_map,
            waypoints,
            max_iters=max_iters,
            step_size=step_size,
            target_clearance=target_clearance,
        )

    def allocateTime(self, waypoints) -> np.ndarray:
        return allocate_time(waypoints, self.optimizer.max_vel, self.optimizer.max_acc, self.optimizer.mini_T)

    def _trapezoid_duration(self, length: float) -> float:
        return trapezoid_duration(length, self.optimizer.max_vel, self.optimizer.max_acc, self.optimizer.mini_T)

    def astar_path_to_follower_path(self, astar_path: np.ndarray, **kwargs):
        result = self.plan_from_path(astar_path, **kwargs)
        return result.minco, result.resampled

    def online_replan_once(self, grid_map, start_xy: np.ndarray, goal_xy: np.ndarray, **kwargs) -> dict:
        result = self.replan(grid_map, start_xy, goal_xy, **kwargs)
        return result.as_online_dict()

    def OptimizeTrajectory(self, *args, **kwargs):
        return self.optimizer.optimize(*args, **kwargs)

    def getOptimizedTrajectories(self):
        return self.optimizer.get_optimized_trajectories()


class PolyTrajOptimizer(MincoPlanner):
    """Compatibility alias for older examples."""


def _default_pva(position) -> np.ndarray:
    return _make_pva(position)


def _make_pva(position, velocity=None, acceleration=None) -> np.ndarray:
    return np.array([
        np.asarray(position, dtype=float),
        np.zeros(2) if velocity is None else np.asarray(velocity, dtype=float),
        np.zeros(2) if acceleration is None else np.asarray(acceleration, dtype=float),
    ], dtype=float)


def _minco_from_optimized(opt_traj, head_pva: np.ndarray, tail_pva: np.ndarray) -> MINCO:
    coeffs = opt_traj.getCoeffs()
    durations = np.asarray(opt_traj.T, dtype=float)
    waypoints = []
    for i in range(len(durations) - 1):
        c = coeffs[6 * i : 6 * (i + 1), :]
        Ti = durations[i]
        waypoints.append(c[0] + c[1] * Ti + c[2] * Ti**2 + c[3] * Ti**3 + c[4] * Ti**4 + c[5] * Ti**5)
    return MINCO(head_pva, tail_pva, np.asarray(waypoints), durations)
