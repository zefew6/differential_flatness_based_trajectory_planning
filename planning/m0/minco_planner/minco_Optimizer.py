import numpy as np
from scipy.optimize import minimize
from .minco_MinJerkOpt import MinJerkOpt
from .minco_FeasibilityConstraint import FeasibilityConstraint
from .minco_obstacle import ObstacleConstraint, SFCObstacleConstraint


class PolyTrajOptimizer:
    """
    多项式轨迹优化器
    基于 L-BFGS 优化算法，优化航点位置和时间分配
    
    Cost Function:
        Total Cost = Smoothness Cost + Time Cost + Penalty Cost
        
        1. Smoothness Cost: 轨迹的 Jerk 平方积分
        2. Time Cost: 总时间 * wei_time
        3. Penalty Cost: 约束违反的惩罚（障碍物、动力学等）

    障碍物方法（obstacle_method）
    ----------------------------
    - 'esdf' （默认）：基于 ESDF/SDF 距离场（GridMap2D），参考 ST-opt-tools。
                       需调用 setGridMap(grid_map)。
    - 'sfc'           ：基于安全飞行走廊（凸多面体半平面约束），参考 Dftpav。
                        需调用 setSFCCorridors(hPolys_per_traj)，
                        其中 hPolys_per_traj[i] 为第 i 条轨迹的走廊列表
                        （list of ndarray shape (K,3)，每行 [nx,ny,b]）。
    """
    
    def __init__(self, obstacle_method: str = 'esdf'):
        """初始化优化器

        参数
        ----
        obstacle_method : 障碍物处理方式
            'esdf' - 基于 ESDF 距离场（需调用 setGridMap）
            'sfc'  - 基于安全飞行走廊（需调用 setSFCCorridors）
        """
        # 障碍物方法选择
        assert obstacle_method in ('esdf', 'sfc'), \
            f"obstacle_method must be 'esdf' or 'sfc', got '{obstacle_method}'"
        self.obstacle_method = obstacle_method

        # 优化参数 - 平衡收敛性和约束满足
        self.wei_time = 200.0       # 时间权重（高→规划器倾向短时间轨迹）
        self.wei_feas = 5000.0      # 可行性权重（适中，避免梯度爆炸）
        # 静态障碍物权重：对齐 ST-opt-tools 里常用的 rho_collision=1e5 量级
        self.wei_obs = 1e4
        self.wei_surround = 5000.0 # 动态障碍物权重
        # 障碍安全距离（米）：ESDF 小于此值的点会被惩罚。
        # 需要大于地图中自由点的最小 ESDF（典型值 0.02~0.05m），
        # 通常设为 robot_radius（不含 margin）以在膨胀边界内再留一圈安全余量。
        self.obs_safe_threshold = 0.05
        
        self.mini_T = 0.005         # 最小段时间（Dftpav: 0.1）
        
        # 动力学约束参数 - 参考 Dftpav
        self.max_vel = 4.0         # 最大速度 (m/s)
        self.max_acc = 1.0         # 最大加速度 (m/s²)
        
        # L-BFGS 优化器参数
        self.lbfgs_memsize = 256   # 内存大小（Dftpav: 256）
        self.lbfgs_past = 3        # 过去迭代数（Dftpav: 3）
        self.lbfgs_delta = 2e-2    # 收敛判据（函数值相对变化）
        self.lbfgs_g_epsilon = 1e-4  # 梯度收敛判据（投影梯度范数）
        self.lbfgs_max_iterations = 200  # 最大 L-BFGS 迭代次数（控制 nit）
        self.lbfgs_max_fun = 15000        # 最大函数求值次数（真正的时间上限，控制 nfev）
        
        # 轨迹分辨率
        self.traj_resolution = 16      # 轨迹段分辨率（Dftpav: 16）
        self.destraj_resolution = 32   # 起止段分辨率（Dftpav: 32）
        
        # 轨迹数据
        self.trajnum = 0
        self.piece_num_container = []
        self.jerkOpt_container = []
        self.feasConstr_container = []  # 可行性约束容器
        self.iniState_container = []
        self.finState_container = []
        
        # 优化变量
        self.variable_num = 0
        self.iter_num = 0

        # debug / diagnostics
        self.debug_print_every = 10  # print cost breakdown every N iterations (0 disables)
        self._last_cost_breakdown = None

        # --- obstacle related ---
        # 外部注入的地图对象（m0.minco_planner.minco_obstacle.GridMap2D），需要提供 get_distance_and_gradient(pos)
        self.grid_map = None
        # 静态障碍物约束模块（ESDF 方法）
        self.obsConstr_container = []
        # SFC 走廊约束模块（SFC 方法）
        self.sfcConstr_container = []
        # 外部注入的 SFC 走廊数据（list of list of hPoly）
        # sfc_corridors_data[i] = 第 i 条轨迹的走廊列表（list of ndarray shape (K,3)）
        self.sfc_corridors_data = []
        # SFC 约束权重
        self.wei_sfc = 1e4
        # SFC 安全裕量（距走廊边界的最小距离，通常取 0 或 robot_radius）
        self.sfc_safe_margin = 0.0

    def setGridMap(self, grid_map):
        """设置用于避障项的 SDF/ESDF 地图（ESDF 方法）。

        grid_map 需要提供：
            get_distance_and_gradient(pos)->(dist, grad)
        同时也应提供 coor_to_index / is_valid_index / is_occupied_index
        以支持 preprocessPath 的可见性剪枝。
        """
        self.grid_map = grid_map

    def setSFCCorridors(self, hPolys_per_traj: list):
        """设置 SFC 走廊数据（SFC 方法）。

        参数
        ----
        hPolys_per_traj : list，长度 = trajnum
            hPolys_per_traj[i] 为第 i 条轨迹的走廊列表（list of np.ndarray），
            每个元素 shape (K, 3)，每行 [nx, ny, b]，约束 n^T p <= b（外法向量朝外）。
            长度需等于对应轨迹的 piece_num。

        注意
        ----
        需要在调用 OptimizeTrajectory 之前调用，
        或者在 OptimizeTrajectory 内部通过 initSFCContainers 自动注入。
        """
        self.sfc_corridors_data = list(hPolys_per_traj)

    def buildSFCCorridors(self, waypoints, search_radius: float = 6.0,
                          subsample: int = 2, n_bins: int = 36,
                          method: str = 'cube',
                          inflate_step_cells: int = 1) -> list:
        """从已设置的 grid_map 生成 SFC 走廊并自动注入到优化器。

        委托给 minco_obstacle.build_sfc_from_gridmap 一步完成：
            1. 按 method 选择走廊生成器（默认 cube 膨胀法）
            2. 为每段路径生成凸 hPoly 走廊
            3. 自动调用 setSFCCorridors

        参数
        ----
        waypoints     : np.ndarray shape (N, 2)，包含首尾的完整路点
        search_radius : 走廊搜索半径（米），默认 6.0
        subsample     : legacy 方法下点云下采样步长，默认 2
        n_bins        : legacy 方法下角度分箱数，默认 36
        method        : 'cube'（默认，栅格膨胀）或 'legacy'（点云分箱）
        inflate_step_cells : cube 方法每轮每方向膨胀的格数，默认 1

        返回
        ----
        hPolys : list of np.ndarray，len = piece_num
        """
        from .minco_obstacle import build_sfc_from_gridmap
        import time as _time

        if self.grid_map is None:
            raise RuntimeError("请先调用 setGridMap() 再调用 buildSFCCorridors()。")

        t0 = _time.time()
        hPolys = build_sfc_from_gridmap(
            self.grid_map, waypoints,
            search_radius=search_radius,
            subsample=subsample,
            n_bins=n_bins,
            method=method,
            inflate_step_cells=inflate_step_cells,
        )
        print(f"[SFC] Built {len(hPolys)} corridor segments by '{method}' in "
              f"{(_time.time()-t0)*1000:.1f} ms")

        self.setSFCCorridors([hPolys])
        return hPolys

    # ------------------------------------------------------------------
    # 路径预处理：可见性剪枝  （参考 ST-opt-tools AStar::optimizePath）
    # ------------------------------------------------------------------

    def _check_line_collision(self, p1, p2) -> bool:
        """Bresenham 直线检测：p1→p2 之间是否经过占据格。有碰撞返回 True。"""
        gm = self.grid_map
        r1, c1 = gm.coor_to_index(p1)
        r2, c2 = gm.coor_to_index(p2)
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        sr = 1 if r2 > r1 else -1
        sc = 1 if c2 > c1 else -1
        err = dr - dc
        r, c = r1, c1
        while True:
            idx = (r, c)
            if gm.is_valid_index(idx) and gm.is_occupied_index(idx):
                return True
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return False

    def preprocessPath(self, path) -> np.ndarray:
        """对 A* 原始路径做可见性剪枝，返回仅保留转角点的精简路径（含首尾）。

        参数
        ----
        path : np.ndarray, shape (N, 2)
            A* 输出的稠密路径（每步一个栅格）。

        返回
        ----
        pruned : np.ndarray, shape (M, 2)
            精简后的路径，M << N。内部路点 pruned[1:-1] 即为 MINCO 的 inner_pts。
        """
        if self.grid_map is None:
            raise RuntimeError("Call setGridMap() before preprocessPath().")
        path = np.asarray(path)
        if len(path) <= 2:
            return path
        pruned = [path[0]]
        prev = path[0]
        i = 1
        while i < len(path) - 1:
            # 若能从 prev 直线无碰到 path[i+1]，则跳过 path[i]
            if not self._check_line_collision(prev, path[i + 1]):
                i += 1
            else:
                pruned.append(path[i])
                prev = path[i]
                i += 1
        pruned.append(path[-1])
        return np.array(pruned)

    def resamplePath(self, path, max_seg_len: float = 1.5,
                     dense_path: np.ndarray = None) -> np.ndarray:
        """对路径按固定弧长重采样，使每段长度不超过 max_seg_len。

        目的：剪枝后路点数过少（段过长），MINCO 多项式会产生与 A* 无关的大弧。
        重采样后给 MINCO 足够的形状约束，让优化结果贴近原始路径。

        参数
        ----
        path : np.ndarray, shape (N, 2)
            待重采样路径（通常是 preprocessPath 剪枝后的稀疏路径）。
        max_seg_len : float
            最大段长（米），默认 1.5m
        dense_path : np.ndarray, shape (K, 2), optional
            原始稠密路径（如 A* 输出）。若提供，则在稀疏段之间沿稠密路径
            抽取中间路点，**避免直线插值穿越障碍边界** 而引发蛇形振荡。
            若不提供，则退化为直线插值。

        返回
        ----
        resampled : np.ndarray, shape (M, 2)，M >= N
        """
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return path

        # ── 若提供稠密路径，在稀疏段间沿稠密路径抽取中间点 ──────────────────
        if dense_path is not None:
            dense_path = np.asarray(dense_path, dtype=float)
            result = [path[0]]
            for i in range(len(path) - 1):
                p0, p1 = path[i], path[i + 1]
                seg_len = np.linalg.norm(p1 - p0)
                if seg_len > max_seg_len:
                    # 在 dense_path 中找到属于本段的区间（最近邻索引）
                    d0 = np.linalg.norm(dense_path - p0, axis=1)
                    d1 = np.linalg.norm(dense_path - p1, axis=1)
                    idx0 = int(np.argmin(d0))
                    idx1 = int(np.argmin(d1))
                    if idx0 > idx1:
                        idx0, idx1 = idx1, idx0
                    # 从稠密路径的对应区间中均匀抽取中间点
                    segment = dense_path[idx0:idx1 + 1]
                    if len(segment) > 2:
                        n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
                        # 按弧长均匀抽取 n_insert 个中间点
                        arc = np.cumsum(
                            np.r_[0, np.linalg.norm(np.diff(segment, axis=0), axis=1)]
                        )
                        total_arc = arc[-1]
                        for k in range(1, n_insert + 1):
                            s = total_arc * k / (n_insert + 1)
                            j = np.searchsorted(arc, s, side='right') - 1
                            j = min(j, len(segment) - 2)
                            alpha = (s - arc[j]) / (arc[j + 1] - arc[j] + 1e-12)
                            pt = segment[j] + alpha * (segment[j + 1] - segment[j])
                            result.append(pt)
                result.append(p1)
            return np.array(result)

        # ── 退化模式：直线插值（无稠密路径时使用）────────────────────────────
        result = [path[0]]
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            seg_len = np.linalg.norm(p1 - p0)
            if seg_len > max_seg_len:
                n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
                for k in range(1, n_insert + 1):
                    t = k / (n_insert + 1)
                    result.append(p0 + t * (p1 - p0))
            result.append(p1)
        return np.array(result)

    def uniform_resample_path(self, path: np.ndarray,
                              max_seg_len: float = 3.0) -> np.ndarray:
        """Dftpav 风格：在原始 A* 路径上按等弧长均匀采样。

        参数
        ----
        path : np.ndarray, shape (N, 2)
            A* 输出路径。
        max_seg_len : float
            目标段长（米）。

        返回
        ----
        resampled : np.ndarray, shape (M, 2)
        """
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return path

        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_len = float(np.sum(seg_lens))
        piece_nums = max(int(total_len / max_seg_len + 0.5), 2)

        cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
        sample_dists = np.linspace(0.0, total_len, piece_nums + 1)

        result = []
        for d in sample_dists:
            idx = int(np.searchsorted(cum_dist, d, side='right')) - 1
            idx = np.clip(idx, 0, len(path) - 2)
            denom = cum_dist[idx + 1] - cum_dist[idx]
            alpha = (d - cum_dist[idx]) / (denom + 1e-12)
            result.append(path[idx] + alpha * (path[idx + 1] - path[idx]))
        return np.array(result)

    def push_waypoints_to_clearance(self,
                                    waypoints: np.ndarray,
                                    max_iters: int = 30,
                                    step_size: float = 0.08,
                                    target_clearance: float = 0.30) -> np.ndarray:
        """将内点沿 ESDF 梯度方向推离障碍物（头尾不动）。"""
        if self.grid_map is None:
            raise RuntimeError("Call setGridMap() before push_waypoints_to_clearance().")

        pts = np.asarray(waypoints, dtype=float).copy()
        n_pts = len(pts)
        for k in range(1, n_pts - 1):
            for _ in range(max_iters):
                dist, grad = self.grid_map.get_distance_and_gradient(pts[k])
                if dist >= target_clearance:
                    break
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break
                pts[k] += step_size * grad / gnorm
        return pts

    def astar_path_to_follower_path(
        self,
        astar_path: np.ndarray,
        head_pva: np.ndarray = None,
        tail_pva: np.ndarray = None,
        max_seg_len: float = 1.2,
        waypoints: np.ndarray = None,
        sfc_push_to_clearance: bool = True,
        sfc_max_iters: int = 60,
        sfc_step_size: float = 0.05,
        sfc_target_clearance: float = 0.40,
        sfc_search_radius: float = 6.0,
        sfc_build_method: str = 'legacy',
        sfc_safe_margin: float = 0.0,
        sfc_wei: float = 5e5,
    ):
        """输入 A* path，输出可供 TrajectoryFollower 跟踪的 MINCO 轨迹。

        参数
        ----
        astar_path : np.ndarray, shape (N, 2)
            A* 输出路径。
        head_pva : np.ndarray, optional
            起点状态，shape (3,2)。不提供则使用 astar_path 首点、零速零加。
        tail_pva : np.ndarray, optional
            终点状态，shape (3,2)。不提供则使用 astar_path 末点、零速零加。
        max_seg_len : float
            等弧长重采样段长（米）。
        waypoints : np.ndarray, optional
            若提供，则直接作为优化路点（含首尾）；否则使用重采样结果。
        sfc_push_to_clearance : bool
            当 obstacle_method='sfc' 时，是否在建走廊前推离内点。
        sfc_max_iters, sfc_step_size, sfc_target_clearance
            SFC 推离参数。
        sfc_search_radius, sfc_build_method
            SFC 走廊构建参数，传给 buildSFCCorridors。
        sfc_safe_margin, sfc_wei
            SFC 约束参数，传给 setParam。

        返回
        ----
        opt_minco : MINCO
            可直接传入 TrajectoryFollower 的轨迹对象。
        resampled : np.ndarray
            等弧长重采样后的路径（含首尾）。
        """
        from .minco import MINCO

        astar_path = np.asarray(astar_path, dtype=float)
        if len(astar_path) < 2:
            raise ValueError("astar_path 至少需要 2 个点")

        resampled = self.uniform_resample_path(astar_path, max_seg_len=max_seg_len)

        if waypoints is None:
            full_pts = resampled
        else:
            full_pts = np.asarray(waypoints, dtype=float)
            if len(full_pts) < 2:
                raise ValueError("waypoints 至少需要 2 个点")

        if self.obstacle_method == 'sfc':
            if self.grid_map is None:
                raise RuntimeError("SFC 模式需要先调用 setGridMap()。")

            if sfc_push_to_clearance:
                full_pts = self.push_waypoints_to_clearance(
                    full_pts,
                    max_iters=sfc_max_iters,
                    step_size=sfc_step_size,
                    target_clearance=sfc_target_clearance,
                )

            self.buildSFCCorridors(
                full_pts,
                search_radius=sfc_search_radius,
                method=sfc_build_method,
            )
            self.setParam(sfc_safe_margin=sfc_safe_margin, wei_sfc=sfc_wei)

        if head_pva is None:
            head_pva = np.array([full_pts[0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
        else:
            head_pva = np.asarray(head_pva, dtype=float)

        if tail_pva is None:
            tail_pva = np.array([full_pts[-1], [0.0, 0.0], [0.0, 0.0]], dtype=float)
        else:
            tail_pva = np.asarray(tail_pva, dtype=float)

        inner_pts = full_pts[1:-1]
        durations = self.allocateTime(full_pts)

        self.OptimizeTrajectory(
            iniStates=[head_pva], finStates=[tail_pva],
            initInnerPts=[inner_pts],
            initTs=np.array([np.sum(durations)]),
            initSegTs=[durations],
        )

        opt_traj = self.getOptimizedTrajectories()[0]
        opt_coeffs = opt_traj.getCoeffs()
        opt_T = opt_traj.T
        wpts = []
        for i in range(len(opt_T) - 1):
            c = opt_coeffs[6 * i:6 * (i + 1), :]
            Ti = opt_T[i]
            wpts.append(c[0] + c[1] * Ti + c[2] * Ti ** 2 + c[3] * Ti ** 3 + c[4] * Ti ** 4 + c[5] * Ti ** 5)

        opt_minco = MINCO(head_pva, tail_pva, np.array(wpts), opt_T)
        return opt_minco, resampled

    def online_replan_once(
        self,
        grid_map,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        max_seg_len: float = 1.2,
        head_pva: np.ndarray = None,
        tail_pva: np.ndarray = None,
        start_vel: np.ndarray = None,
        start_acc: np.ndarray = None,
        goal_vel: np.ndarray = None,
        goal_acc: np.ndarray = None,
        sfc_push_to_clearance: bool = True,
        sfc_max_iters: int = 60,
        sfc_step_size: float = 0.05,
        sfc_target_clearance: float = 0.40,
        sfc_search_radius: float = 6.0,
        sfc_build_method: str = 'legacy',
        sfc_safe_margin: float = 0.0,
        sfc_wei: float = 5e5,
    ) -> dict:
        """单次在线重规划：A* -> MINCO，返回可直接用于跟踪/可视化的结果。"""
        from ..planning.a_star import graph_search
        import time as _time

        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        self.setGridMap(grid_map)

        path = graph_search(start=start_xy, goal=goal_xy, gridmap=grid_map)
        if path is None or len(path) < 2:
            raise RuntimeError("A* 无法找到可行路径")

        if head_pva is None:
            start_vel = np.zeros(2, dtype=float) if start_vel is None else np.asarray(start_vel, dtype=float)
            start_acc = np.zeros(2, dtype=float) if start_acc is None else np.asarray(start_acc, dtype=float)
            head_pva = np.array([start_xy, start_vel, start_acc], dtype=float)
        else:
            head_pva = np.asarray(head_pva, dtype=float)

        if tail_pva is None:
            goal_vel = np.zeros(2, dtype=float) if goal_vel is None else np.asarray(goal_vel, dtype=float)
            goal_acc = np.zeros(2, dtype=float) if goal_acc is None else np.asarray(goal_acc, dtype=float)
            tail_pva = np.array([goal_xy, goal_vel, goal_acc], dtype=float)
        else:
            tail_pva = np.asarray(tail_pva, dtype=float)

        t0 = _time.time()
        opt_minco, resampled = self.astar_path_to_follower_path(
            path,
            head_pva=head_pva,
            tail_pva=tail_pva,
            max_seg_len=max_seg_len,
            sfc_push_to_clearance=sfc_push_to_clearance,
            sfc_max_iters=sfc_max_iters,
            sfc_step_size=sfc_step_size,
            sfc_target_clearance=sfc_target_clearance,
            sfc_search_radius=sfc_search_radius,
            sfc_build_method=sfc_build_method,
            sfc_safe_margin=sfc_safe_margin,
            sfc_wei=sfc_wei,
        )
        dt_ms = (_time.time() - t0) * 1000.0

        return {
            "path": path,
            "resampled": resampled,
            "minco": opt_minco,
            "cost_time_ms": dt_ms,
        }

    # ------------------------------------------------------------------
    # 时间分配：梯形速度曲线  （参考 ST-opt-tools AStar::evaluateDuration）
    # ------------------------------------------------------------------

    def allocateTime(self, waypoints) -> np.ndarray:
        """按梯形速度曲线为各段分配飞行时间。

        参数
        ----
        waypoints : np.ndarray, shape (K, 2)
            包含首尾的完整路点序列（head + inner_pts + tail）。

        返回
        ----
        durations : np.ndarray, shape (K-1,)
            每段的时间分配（秒）。
        """
        waypoints = np.asarray(waypoints)
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        durations = np.array([self._trapezoid_duration(d) for d in dists])
        # 保证不低于最小段时间
        return np.maximum(durations, self.mini_T)

    def _trapezoid_duration(self, length: float) -> float:
        """单段梯形速度曲线时间（起止速度均为 0）。"""
        v = self.max_vel
        a = self.max_acc
        if length < 1e-6:
            return self.mini_T
        critical_len = v * v / a          # 刚好能加速到 max_vel 再减速所需路程
        if length >= critical_len:
            return 2.0 * v / a + (length - critical_len) / v
        else:
            v_peak = np.sqrt(a * length)  # 三角形速度曲线
            return 2.0 * v_peak / a

    def setParam(self, wei_time=None, wei_feas=None, mini_T=None,
                 lbfgs_memsize=None, lbfgs_delta=None, lbfgs_max_iterations=None,
                 lbfgs_max_fun=None,
                 wei_obs=None, obs_safe_threshold=None,
                 wei_sfc=None, sfc_safe_margin=None,
                 traj_resolution=None, destraj_resolution=None):
        """
        设置优化参数

        参数:
            wei_time: 时间权重
            wei_feas: 可行性权重
            mini_T: 最小段时间
            lbfgs_memsize: L-BFGS 内存大小
            lbfgs_delta: L-BFGS 收敛判据
            lbfgs_max_iterations: 最大 L-BFGS 迭代次数（nit）
            lbfgs_max_fun: 最大函数求值次数（nfev），直接决定最多执行多少次 cost 回调
            wei_obs: ESDF 方法障碍物权重
            obs_safe_threshold: ESDF 方法安全阈值
            wei_sfc: SFC 方法走廊约束权重
            sfc_safe_margin: SFC 方法安全裕量（距走廊边界最小距离）
            traj_resolution: 中间段 ESDF/SFC 采样点数（越小越快，建议 6~16）
            destraj_resolution: 起止段 ESDF/SFC 采样点数（建议 traj_resolution*2）
        """
        if wei_time is not None:
            self.wei_time = wei_time
        if wei_feas is not None:
            self.wei_feas = wei_feas
        if mini_T is not None:
            self.mini_T = mini_T
        if lbfgs_memsize is not None:
            self.lbfgs_memsize = lbfgs_memsize
        if lbfgs_delta is not None:
            self.lbfgs_delta = lbfgs_delta
        if lbfgs_max_iterations is not None:
            self.lbfgs_max_iterations = lbfgs_max_iterations
        if lbfgs_max_fun is not None:
            self.lbfgs_max_fun = int(lbfgs_max_fun)

        if wei_obs is not None:
            self.wei_obs = float(wei_obs)
            # propagate to already-created containers
            for obsConstr in getattr(self, 'obsConstr_container', []):
                obsConstr.wei_obs = float(wei_obs)

        if obs_safe_threshold is not None:
            self.obs_safe_threshold = float(obs_safe_threshold)  # 存到 self，供创建容器时使用
            # propagate to already-created containers
            for obsConstr in getattr(self, 'obsConstr_container', []):
                obsConstr.safe_threshold = float(obs_safe_threshold)

        if wei_sfc is not None:
            self.wei_sfc = float(wei_sfc)
            for sfcConstr in getattr(self, 'sfcConstr_container', []):
                sfcConstr.wei_sfc = float(wei_sfc)

        if sfc_safe_margin is not None:
            self.sfc_safe_margin = float(sfc_safe_margin)
            for sfcConstr in getattr(self, 'sfcConstr_container', []):
                sfcConstr.safe_margin = float(sfc_safe_margin)

        if traj_resolution is not None:
            self.traj_resolution = int(traj_resolution)
            for c in getattr(self, 'obsConstr_container', []):
                c.traj_resolution = int(traj_resolution)
            for c in getattr(self, 'sfcConstr_container', []):
                c.traj_resolution = int(traj_resolution)
            for c in getattr(self, 'feasConstr_container', []):
                c.traj_resolution = int(traj_resolution)

        if destraj_resolution is not None:
            self.destraj_resolution = int(destraj_resolution)
            for c in getattr(self, 'obsConstr_container', []):
                c.destraj_resolution = int(destraj_resolution)
            for c in getattr(self, 'sfcConstr_container', []):
                c.destraj_resolution = int(destraj_resolution)
            for c in getattr(self, 'feasConstr_container', []):
                c.destraj_resolution = int(destraj_resolution)

    def setDebugPrintEvery(self, n: int):
        """Set how often to print per-iteration cost diagnostics.

        n <= 0 disables printing.
        """
        self.debug_print_every = int(n)
    
    def OptimizeTrajectory(self, iniStates, finStates, initInnerPts, initTs,
                           initSegTs=None):
        """优化轨迹
        
        参数:
            iniStates: 起始状态列表，每个元素 shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
            finStates: 终止状态列表，每个元素 shape (3, 2)
            initInnerPts: 初始内部航点列表，每个元素 shape (N, 2) - N行2列
            initTs: 初始时间分配，shape (trajnum,)，每条轨迹的总时间
            initSegTs: 可选，按段时间列表，list of ndarray shape (piece_num,)
                       若提供则用于初始化各段时间，忽略 initTs 的等分逻辑
        
        返回:
            success: 是否优化成功
            final_cost: 最终代价值
        """
        self.trajnum = len(initInnerPts)
        self.iniState_container = []
        self.finState_container = []
        
        # 裁剪初始/终止状态以满足硬约束
        for i in range(self.trajnum):
            clipped_ini = self.clipStateConstraints(iniStates[i], self.max_vel, self.max_acc)
            clipped_fin = self.clipStateConstraints(finStates[i], self.max_vel, self.max_acc)
            self.iniState_container.append(clipped_ini)
            self.finState_container.append(clipped_fin)
        
        # 检查输入
        if len(initTs) != self.trajnum:
            print("Error: initTs size != trajnum")
            return False, np.inf
        
        if np.min(initTs) < self.mini_T:
            print("Error: mini segment T < mini_T!")
            return False, np.inf
        
        # 初始化优化器容器
        self.jerkOpt_container = []
        self.feasConstr_container = []  # 约束容器
        self.obsConstr_container = []   # 避障约束容器（ESDF 方法）
        self.sfcConstr_container = []   # SFC 走廊约束容器（SFC 方法）
        self.piece_num_container = []
        self.variable_num = 0
        
        for i in range(self.trajnum):
            if initInnerPts[i].shape[0] == 0:  # 检查行数（航点数量）
                print("Error: No inner points!")
                return False, np.inf
            
            piece_num = initInnerPts[i].shape[0] + 1  # 行数 + 1
            self.piece_num_container.append(piece_num)
            
            # 创建 MinJerkOpt
            jerkOpt = MinJerkOpt(piece_num)
            self.jerkOpt_container.append(jerkOpt)
            
            # 创建 FeasibilityConstraint
            feasConstr = FeasibilityConstraint(
                self.max_vel, 
                self.max_acc,
                self.traj_resolution,
                self.destraj_resolution
            )
            self.feasConstr_container.append(feasConstr)

            # 创建 ObstacleConstraint（避障项会在 cost callback 里按需启用）
            obsConstr = ObstacleConstraint(
                safe_threshold=self.obs_safe_threshold,  # 使用 setParam 设置的值
                wei_obs=self.wei_obs,
                traj_resolution=self.traj_resolution,
                destraj_resolution=self.destraj_resolution,
            )
            self.obsConstr_container.append(obsConstr)

            # 创建 SFCObstacleConstraint（SFC 方法）
            sfcConstr = SFCObstacleConstraint(
                safe_margin=self.sfc_safe_margin,
                wei_sfc=self.wei_sfc,
                traj_resolution=self.traj_resolution,
                destraj_resolution=self.destraj_resolution,
            )
            # 注入对应轨迹的走廊数据
            if self.sfc_corridors_data and i < len(self.sfc_corridors_data):
                sfcConstr.set_corridors(self.sfc_corridors_data[i])
            self.sfcConstr_container.append(sfcConstr)
            
            # 变量数：航点数 (2 * (piece_num - 1))
            self.variable_num += 2 * (piece_num - 1)
            
            # 变量数：每段独立的时间 (piece_num 个)
            self.variable_num += piece_num
        
        # 构建初始优化变量
        x0 = np.zeros(self.variable_num)
        offset = 0
        
        # 航点
        for i in range(self.trajnum):
            pts = initInnerPts[i]
            x0[offset:offset + pts.size] = pts.flatten()
            offset += pts.size
        
        # 时间（转换为虚拟时间）- 每段独立
        for i in range(self.trajnum):
            piece_num = self.piece_num_container[i]
            if initSegTs is not None and i < len(initSegTs):
                # 使用外部提供的按段时间（来自梯形速度分配）
                segment_times = np.array(initSegTs[i], dtype=float)
                if len(segment_times) != piece_num:
                    # 长度不匹配时退化为等分
                    dt_init = initTs[i] / piece_num
                    segment_times = np.full(piece_num, dt_init)
            else:
                # 将总时间平均分配给各段作为初始值
                dt_init = initTs[i] / piece_num
                segment_times = np.full(piece_num, dt_init)
            segment_times = np.maximum(segment_times, self.mini_T)

            VT = np.zeros(piece_num)
            self.RealT2VirtualT(segment_times, VT)
            x0[offset:offset + piece_num] = VT
            offset += piece_num
        
        # L-BFGS 优化（使用 Dftpav 的参数）
        self.iter_num = 0
        
        result = minimize(
            fun=self._cost_function_callback,
            x0=x0,
            method='L-BFGS-B',
            jac=True,  # 我们会返回梯度
            options={
                'maxiter': self.lbfgs_max_iterations,  # 12000（Dftpav）
                'ftol': self.lbfgs_delta,              # 1e-4（Dftpav）
                'gtol': self.lbfgs_g_epsilon,          # 1e-16（Dftpav）
                'maxls': 20,                            # 最大线搜索次数
                'maxfun': self.lbfgs_max_fun,           # 最大函数求值次数（对应实时截止）
                'disp': False                          # 不显示scipy优化器的详细信息
            }
        )
        
        success = result.success
        final_cost = result.fun
        
        # print(f"\nOptimization finished!")
        # print(f"Success: {success}")
        # print(f"Message: {result.message}")
        # print(f"Iterations: {result.nit}")
        # print(f"Function evaluations: {result.nfev}")
        # print(f"Final cost: {final_cost:.6f}")
        # print(f"Final gradient norm: {np.linalg.norm(result.jac):.6e}")
        # print(f"Iterations: {self.iter_num}")
        print(f"[MINCO] nit={result.nit}  nfev={result.nfev}  callback_calls={self.iter_num}  msg={result.message}")
        
        return success, final_cost
    
    def _cost_function_callback(self, x):
        """
        代价函数回调
        
        参数:
            x: 优化变量 [P0, P1, ..., t0, t1, ...]
        
        返回:
            cost: 总代价
            grad: 梯度
        """
        total_smcost = 0.0
        total_timecost = 0.0
        total_penalty = 0.0  # 约束惩罚代价

        # breakdown
        total_feas_cost = 0.0
        total_obs_cost = 0.0
        
        # 解析优化变量
        offset = 0
        P_container = []
        gradP_container = []
        
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            size = 2 * (piece_num - 1)
            
            # 重塑为 (N, 2) 格式：N行2列
            P = x[offset:offset + size].reshape(piece_num - 1, 2)
            gradP = np.zeros((piece_num - 1, 2))
            
            P_container.append(P)
            gradP_container.append(gradP)
            offset += size
        
        # 时间 - 每段独立的时间
        T_container = []
        gradT_container = []
        
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            
            # 提取每段的虚拟时间
            t_seg = x[offset:offset + piece_num]
            gradt_seg = np.zeros(piece_num)
            
            # 虚拟时间转真实时间
            T_seg = np.zeros(piece_num)
            self.VirtualT2RealT(t_seg, T_seg)
            
            T_container.append(T_seg)
            gradT_container.append(gradt_seg)
            offset += piece_num
        
        # 计算每条轨迹的代价
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            T_seg = T_container[trajid]
            
            # 检查所有段的时间合理性
            if np.any(T_seg < self.mini_T) or np.any(T_seg > 1000.0) or not np.all(np.isfinite(T_seg)):
                # 时间不合理，返回惩罚代价
                return np.inf, np.zeros_like(x)
            
            # 生成轨迹 - 使用每段独立的时间
            self.jerkOpt_container[trajid].generate(
                P_container[trajid],
                T_seg,  # 传入时间数组而非标量
                self.iniState_container[trajid],
                self.finState_container[trajid]
            )
            
            # 计算平滑代价
            self.jerkOpt_container[trajid].initSmGradCost()
            smoo_cost = self.jerkOpt_container[trajid].getTrajJerkCost()
            
            # 检查Jerk代价是否合理
            if not np.isfinite(smoo_cost) or smoo_cost > 1e8:
                # Jerk代价过大或无穷，返回惩罚
                return 1e10, np.zeros_like(x)
            
            total_smcost += smoo_cost
            
            # 计算动力学约束惩罚代价 (使用 FeasibilityConstraint)
            feasConstr = self.feasConstr_container[trajid]
            feas_cost = feasConstr.addPVAGradCost(
                self.jerkOpt_container[trajid].coeffs,
                self.jerkOpt_container[trajid].T,
                self.piece_num_container[trajid],
                self.wei_feas
            )
            total_penalty += feas_cost
            total_feas_cost += feas_cost
            
            # ⚠️ 重要：在 calGrads_PT() 之前累加约束梯度到 gdC
            # 因为 calGrads_PT() 会用 gdC 计算 gdP (通过伴随方法)
            self.jerkOpt_container[trajid].gdC += feasConstr.get_gdC()
            self.jerkOpt_container[trajid].gdT += feasConstr.get_gdT()

            # 计算避障惩罚代价（按 obstacle_method 选择）
            if self.obstacle_method == 'esdf' and self.grid_map is not None:
                # ── ESDF 方法：基于距离场 ──────────────────────────────────
                obsConstr = self.obsConstr_container[trajid]
                obs_cost = obsConstr.addObstacleGradCost(
                    self.jerkOpt_container[trajid].coeffs,
                    self.jerkOpt_container[trajid].T,
                    self.piece_num_container[trajid],
                    self.grid_map,
                )
                total_penalty += obs_cost
                total_obs_cost += obs_cost
                self.jerkOpt_container[trajid].gdC += obsConstr.get_gdC()
                self.jerkOpt_container[trajid].gdT += obsConstr.get_gdT()

            elif self.obstacle_method == 'sfc' and self.sfcConstr_container:
                # ── SFC 方法：基于凸走廊半平面 ───────────────────────────
                sfcConstr = self.sfcConstr_container[trajid]
                sfc_cost = sfcConstr.addObstacleGradCost(
                    self.jerkOpt_container[trajid].coeffs,
                    self.jerkOpt_container[trajid].T,
                    self.piece_num_container[trajid],
                )
                total_penalty += sfc_cost
                total_obs_cost += sfc_cost
                self.jerkOpt_container[trajid].gdC += sfcConstr.get_gdC()
                self.jerkOpt_container[trajid].gdT += sfcConstr.get_gdT()
        
        # 计算梯度
        offset_P_end = sum([2*(self.piece_num_container[i]-1) for i in range(self.trajnum)])
        
        for trajid in range(self.trajnum):
            # 计算 Jerk 梯度 (此时 gdC 已经包含了约束梯度)
            self.jerkOpt_container[trajid].calGrads_PT()
            
            # 航点梯度 (已经通过伴随方法包含了约束的影响)
            gradP_container[trajid][:] = self.jerkOpt_container[trajid].get_gdP()
            
            # 时间梯度 (包含 Jerk + Constraint) - 每段独立
            piece_num = self.piece_num_container[trajid]
            gdRT_vec = self.jerkOpt_container[trajid].get_gdT()  # 每段的时间梯度向量
            
            # 从优化变量中提取这个轨迹的时间部分
            offset_t = offset_P_end + sum([self.piece_num_container[i] for i in range(trajid)])
            t_seg = x[offset_t:offset_t + piece_num]
            T_seg = T_container[trajid]
            
            # 每段时间的虚拟梯度和时间代价
            for seg_id in range(piece_num):
                gradt_seg, time_cost_seg = self.VirtualTGradCost(
                    np.array([T_seg[seg_id]]), 
                    np.array([t_seg[seg_id]]), 
                    gdRT_vec[seg_id]
                )
                gradT_container[trajid][seg_id] = gradt_seg[0]
                total_timecost += time_cost_seg
        
        # 构建总梯度
        grad = np.zeros_like(x)
        offset = 0
        
        for trajid in range(self.trajnum):
            size = gradP_container[trajid].size
            grad[offset:offset + size] = gradP_container[trajid].flatten()
            offset += size

        # 时间梯度 - 每段独立
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            grad[offset:offset + piece_num] = gradT_container[trajid]
            offset += piece_num

        # 梯度裁剪：防止数值爆炸
        max_grad_norm = 1e4  # 最大梯度范数
        grad_norm = np.linalg.norm(grad)
        grad_norm_scalar = float(grad_norm)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
        
        # 总代价 = 平滑代价 + 时间代价 + 惩罚代价
        # 注意：total_timecost 可能是 numpy array（由 VirtualTGradCost 返回），这里统一转 float
        total_smcost_f = float(total_smcost)
        total_timecost_f = float(np.sum(total_timecost))
        total_penalty_f = float(total_penalty)
        total_feas_cost_f = float(total_feas_cost)
        total_obs_cost_f = float(total_obs_cost)
        total_cost = float(total_smcost_f + total_timecost_f + total_penalty_f)

        # cache breakdown for external inspection (e.g., after optimize)
        self._last_cost_breakdown = {
            'iter': int(self.iter_num),
            'smooth': total_smcost_f,
            'time': total_timecost_f,
            'penalty': total_penalty_f,
            'feas': total_feas_cost_f,
            'obs': total_obs_cost_f,
            'total': float(total_cost),
            'grad_norm_raw': float(grad_norm_scalar),
        }

        # optional per-iteration print
        self.iter_num += 1
        if self.debug_print_every > 0 and (self.iter_num % self.debug_print_every == 0 or self.iter_num == 1):
            msg = (
                f"[MINCO opt] iter={self.iter_num:4d} "
                f"total={total_cost:.6e} "
                f"smooth={total_smcost_f:.3e} time={total_timecost_f:.3e} "
                f"penalty={total_penalty_f:.3e} (feas={total_feas_cost_f:.3e}, obs={total_obs_cost_f:.3e}) "
                f"|grad|={grad_norm_scalar:.3e}"
            )

            # try to report minimum distance along current trajectory (diagnostic only)
            if self.obstacle_method == 'esdf' and self.grid_map is not None and len(self.obsConstr_container) > 0:
                try:
                    min_d = min(oc.min_dist for oc in self.obsConstr_container)
                    if np.isfinite(min_d):
                        msg += f" min_sdf={min_d:.3f}"
                except Exception:
                    pass
            elif self.obstacle_method == 'sfc' and len(self.sfcConstr_container) > 0:
                try:
                    min_d = min(sc.min_dist for sc in self.sfcConstr_container)
                    if np.isfinite(min_d):
                        msg += f" min_corridor_dist={min_d:.3f}"
                except Exception:
                    pass

            print(msg)

        return total_cost, grad
    
    def RealT2VirtualT(self, RT, VT):
        """真实时间转虚拟时间（无约束优化）"""
        for i in range(len(RT)):
            if RT[i] > 1.0 + self.mini_T:
                VT[i] = np.sqrt(2.0 * RT[i] - 1.0 - 2 * self.mini_T) - 1.0
            else:
                VT[i] = 1.0 - np.sqrt(2.0 / (RT[i] - self.mini_T) - 1.0)
    
    def VirtualT2RealT(self, VT, RT):
        """虚拟时间转真实时间"""
        for i in range(len(VT)):
            if VT[i] > 0.0:
                RT[i] = (0.5 * VT[i] + 1.0) * VT[i] + 1.0 + self.mini_T
            else:
                RT[i] = 1.0 / ((0.5 * VT[i] - 1.0) * VT[i] + 1.0) + self.mini_T
    
    def VirtualTGradCost(self, RT, VT, gdRT):
        """计算虚拟时间的梯度和时间代价
        
        参数:
            RT: 真实时间
            VT: 虚拟时间
            gdRT: 关于真实时间的梯度
            
        返回:
            gdVT: 关于虚拟时间的梯度
            costT: 时间代价
        """
        # 计算 dRT/dVT
        if VT > 0:
            gdVT2Rt = VT + 1.0
        else:
            denSqrt = (0.5 * VT - 1.0) * VT + 1.0
            gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt)
        
        # 梯度 (链式法则)
        gdVT = (gdRT + self.wei_time) * gdVT2Rt
        
        # 时间代价
        costT = self.wei_time * RT
        
        return gdVT, costT
    
    def clipStateConstraints(self, state, max_vel, max_acc):
        """
        裁剪状态约束（用于初始/终止状态）
        
        参数:
            state: shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
            max_vel: 最大速度
            max_acc: 最大加速度
            
        返回:
            clipped_state: 裁剪后的状态
        """
        clipped_state = state.copy()
        
        # 裁剪速度
        vel_norm = np.linalg.norm(state[1, :])
        if vel_norm >= max_vel:
            clipped_state[1, :] = state[1, :] / vel_norm * (max_vel - 1e-2)
        
        # 裁剪加速度
        acc_norm = np.linalg.norm(state[2, :])
        if acc_norm >= max_acc:
            clipped_state[2, :] = state[2, :] / acc_norm * (max_acc - 1e-2)
        
        return clipped_state
    
    def getOptimizedTrajectories(self):
        """获取优化后的轨迹"""
        return self.jerkOpt_container