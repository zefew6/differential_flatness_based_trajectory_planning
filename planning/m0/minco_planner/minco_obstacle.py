import numpy as np
from dataclasses import dataclass
from typing import Tuple

# bring in GridMap base class (MuJoCo-backed) for GridMap2D composition
from m0.utils.gridmap_2d import GridMap


# ══════════════════════════════════════════════════════════════════════════════
# ESDF-based obstacle constraint
# ══════════════════════════════════════════════════════════════════════════════

class ObstacleConstraint:
    """静态障碍物约束（ESDF/SDF based）

    目标
    ----
    参考 ST-opt-tools 中 `TrajectoryOptimizer::calculateConstraintCostGrad()` 的碰撞项写法，
    该模块对一条 MINCO 五次多项式轨迹在每段上采样，并累积：

    - cost_obs: 避障惩罚代价（对时间积分）
    - gdC: 关于多项式系数 c 的梯度 (shape: (6*piece_num, 2))
    - gdT: 关于每段时间 T_i 的梯度 (shape: (piece_num,))

    代价形式（对齐 ST-opt-tools）
    ----------------------------
    对每个采样点位置 p(t)：

        d = SDF(p)  (自由空间为正，障碍内部为负)
        viola = safe_threshold - d

    若 viola > 0 且 d < dist_cap，则产生惩罚：

        penalty(viola) =
            { viola^2,  0 < viola < quad_threshold
            { viola,    viola >= quad_threshold

        cost += wei_obs * penalty * dt

    并回传梯度：

        d(cost)/dp = wei_obs * d(penalty)/d(viola) * d(viola)/dp
                  = wei_obs * penaD * (-∇d)

    对系数梯度：

        p(t) = c^T beta0(t)
        => d(cost)/dc += outer(beta0, d(cost)/dp)

    时间梯度（对齐 ST-opt-tools 的离散形式）
    ----------------------------------------
    ST-opt-tools 里对每个采样点的时间梯度使用：

        grad_time += omg * (cost/K + step * alpha * (d(cost)/dp)·v )

    这里采用同样结构，
    其中 v(t)=dp/dt=c^T beta1(t)，alpha=j/K。

    依赖
    ----
    - 需要外部传入 grid_map 对象，提供：
        get_distance_and_gradient(pos)->(d, grad)
      推荐使用 `planning/m0/utils/gridmap_2d_v2.py::GridMap2D`。
    """

    def __init__(
        self,
        safe_threshold: float = 0.5,
        wei_obs: float = 1000.0,
        traj_resolution: int = 16,
        destraj_resolution: int = 32,
        quad_threshold: float = 0.1,
        dist_cap: float = 5.0,
    ):
        self.safe_threshold = float(safe_threshold)
        self.wei_obs = float(wei_obs)
        self.traj_resolution = int(traj_resolution)
        self.destraj_resolution = int(destraj_resolution)
        self.quad_threshold = float(quad_threshold)
        self.dist_cap = float(dist_cap)

        self.gdC = None
        self.gdT = None
        self.obs_cost = 0.0
        self.min_dist = np.inf   # 上次 addObstacleGradCost 采样到的最小 SDF，供日志读取

    def reset(self, coeffs: np.ndarray, piece_num: int):
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num, dtype=np.float64)
        self.obs_cost = 0.0
        self.min_dist = np.inf

    def addObstacleGradCost(
        self,
        coeffs: np.ndarray,
        T: np.ndarray,
        piece_num: int,
        grid_map,
    ):
        """计算避障代价与梯度。

        参数
        ----
        coeffs: shape (6*piece_num, 2)
        T: shape (piece_num,)
        piece_num: 段数
        grid_map: 提供 get_distance_and_gradient(pos)

        返回
        ----
        cost_obs: float
        """
        self.reset(coeffs, piece_num)

        # 检测 grid_map 是否支持批量查询（优先使用，快 10-50x）
        use_batch = hasattr(grid_map, 'get_distance_and_gradient_batch')

        for i in range(piece_num):
            # 采样分辨率：起止段更密
            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution

            c = coeffs[6 * i : 6 * (i + 1), :]  # (6,2)
            T_i = float(T[i])
            if T_i <= 0 or not np.isfinite(T_i):
                return 1e10

            step = T_i / K

            # ── 向量化 beta / pos / vel 计算 ──────────────────────────
            js = np.arange(K + 1, dtype=np.float64)
            s1 = step * js
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2
            s5 = s4 * s1
            # beta0 / beta1: (K+1, 6)
            beta0 = np.stack([np.ones(K + 1), s1, s2, s3, s4, s5], axis=1)
            beta1 = np.stack([np.zeros(K + 1), np.ones(K + 1),
                              2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4], axis=1)
            pos_all = beta0 @ c   # (K+1, 2)
            vel_all = beta1 @ c   # (K+1, 2)

            omg_vec = np.ones(K + 1)
            omg_vec[0] = 0.5
            omg_vec[K] = 0.5
            alpha_vec = js / K

            # ── ESDF 批量查询 ─────────────────────────────────────────
            if use_batch:
                dists, grad_sfds = grid_map.get_distance_and_gradient_batch(pos_all)
                # dists: (K+1,), grad_sfds: (K+1, 2)
            else:
                # 回退：逐点查询
                dists     = np.empty(K + 1)
                grad_sfds = np.zeros((K + 1, 2))
                for j in range(K + 1):
                    d, g = grid_map.get_distance_and_gradient(pos_all[j])
                    dists[j] = d
                    grad_sfds[j] = g

            # 记录最小距离（供日志复用）
            min_d = float(np.min(dists[np.isfinite(dists)]) if np.any(np.isfinite(dists)) else np.inf)
            if min_d < self.min_dist:
                self.min_dist = min_d

            # ── 向量化惩罚计算 ────────────────────────────────────────
            violas = self.safe_threshold - dists       # (K+1,)
            active = (violas > 0.0) & (dists < self.dist_cap)

            if not np.any(active):
                continue

            # penalty 和 penaD（二次/线性切换）
            penalty = np.where(violas < self.quad_threshold,
                               violas * violas,
                               violas)
            penaD   = np.where(violas < self.quad_threshold,
                               2.0 * violas,
                               np.ones_like(violas))
            penalty = np.where(active, penalty, 0.0)
            penaD   = np.where(active, penaD,   0.0)

            cost_c = self.wei_obs * penalty             # (K+1,)
            # grad_p: (K+1, 2)
            grad_p = self.wei_obs * penaD[:, None] * (-grad_sfds)

            # 累加代价
            weights = omg_vec * step                    # (K+1,)
            self.obs_cost += float(np.dot(weights, cost_c))

            # coeff 梯度: sum_j  weights[j] * outer(beta0[j], grad_p[j])
            # = beta0.T @ diag(weights) @ grad_p  → (6,2)
            self.gdC[6 * i : 6 * (i + 1), :] += beta0.T @ (weights[:, None] * grad_p)

            # 时间梯度: sum_j omg[j] * (cost_c[j]/K + step*alpha[j]*dot(grad_p[j], vel[j]))
            dot_gv = np.sum(grad_p * vel_all, axis=1)   # (K+1,)
            self.gdT[i] += float(np.dot(omg_vec,
                                        cost_c / K + step * alpha_vec * dot_gv))

        return float(self.obs_cost)

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_obs_cost(self):
        return float(self.obs_cost)


# ------------------------------------------------------------------
# GridMap2D (ESDF) and parameters
# ------------------------------------------------------------------


@dataclass
class GridMap2DParams:
    """Parameters-only constructor for GridMap2D (no MuJoCo)."""
    resolution: float = 0.1
    size_x: float = 20.0
    size_y: float = 20.0
    origin_at_center: bool = False
    robot_radius: float = 0.0
    margin: float = 0.0


class GridMap2D(GridMap):
    """GridMap extended with ESDF and continuous distance/gradient queries.

    This is effectively the content formerly in utils/gridmap_2d_v2.py but
    provided here so obstacle-related functionality lives under
    `m0.minco_planner.minco_obstacle` as requested.
    """

    def __init__(self, model_or_params=None, data=None, resolution=None,
                 width=None, height=None, robot_radius=None, margin=None,
                 *, model=None, origin_x: float = 0.0, origin_y: float = 0.0):
        # keep origin stored before parent's grid creation
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

        if isinstance(model_or_params, GridMap2DParams):
            p = model_or_params
            self.model = None
            self.data = None
            self.resolution = p.resolution
            self.width = p.size_x
            self.height = p.size_y
            # origin_at_center: 将地图中心对齐到世界原点
            if p.origin_at_center:
                self.origin_x = -p.size_x / 2.0
                self.origin_y = -p.size_y / 2.0
            self.grid_width = int(p.size_x / p.resolution)
            self.grid_height = int(p.size_y / p.resolution)
            self.robot_radius = p.robot_radius
            self.margin = p.margin
            self.inflation_radius = p.robot_radius + p.margin
            self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            self._params = p
        else:
            actual_model = model if model is not None else model_or_params
            super().__init__(actual_model, data, resolution, width, height,
                             robot_radius, margin)
            self._params = None

        # ESDF cache (lazy)
        self.esdf: np.ndarray | None = None
        self._esdf_valid: bool = False

    # ---- convenience aliases used elsewhere ----
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
    def min_boundary(self):
        return np.array([self.origin_x, self.origin_y])

    @property
    def max_boundary(self):
        return np.array([self.origin_x + self.width, self.origin_y + self.height])

    # override coordinate transforms to respect origin_x/origin_y
    def coor_to_index(self, coor):
        x, y = coor[0], coor[1]
        col = int((x - self.origin_x + self.resolution / 2) / self.resolution)
        row = int((y - self.origin_y + self.resolution / 2) / self.resolution)
        return row, col

    def index_to_coor(self, ind):
        row, col = ind[0], ind[1]
        x = self.origin_x + (col - self.resolution / 2) * self.resolution
        y = self.origin_y + (row - self.resolution / 2) * self.resolution
        return x, y

    def index_to_pos(self, idx_xy) -> np.ndarray:
        """兼容性别名：index_to_pos([col, row]) → np.array([x, y])。"""
        x, y = self.index_to_coor((int(idx_xy[1]), int(idx_xy[0])))
        return np.array([x, y], dtype=np.float64)

    # ------------------------------------------------------------------
    # 父类 _add_box / _add_sphere / _add_cylinder 重写（加入 origin 偏移）
    # ------------------------------------------------------------------

    def _add_box(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        lx, ly, _ = self.model.geom_size[geom_id]
        R = self.data.geom_xmat[geom_id].reshape(3, 3)
        inf = self.inflation_radius
        local_pts = np.array([
            [-lx - inf, -ly - inf],
            [ lx + inf, -ly - inf],
            [ lx + inf,  ly + inf],
            [-lx - inf,  ly + inf],
        ])
        world_pts = np.dot(local_pts, R[:2, :2].T) + center[:2]
        res = float(self.resolution)
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = self.origin_x + (i + 0.5) * res
                y = self.origin_y + (j + 0.5) * res
                if self._point_in_polygon(np.array([x, y]), world_pts):
                    self.grid[j, i] = 1

    def _add_sphere(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0]
        res = float(self.resolution)
        inf = self.inflation_radius
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = self.origin_x + (i + 0.5) * res
                y = self.origin_y + (j + 0.5) * res
                if (x - center[0])**2 + (y - center[1])**2 <= (radius + inf)**2:
                    self.grid[j, i] = 1

    def _add_cylinder(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0]
        res = float(self.resolution)
        inf = self.inflation_radius
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = self.origin_x + (i + 0.5) * res
                y = self.origin_y + (j + 0.5) * res
                if (x - center[0])**2 + (y - center[1])**2 <= (radius + inf)**2:
                    self.grid[j, i] = 1

    # deprecated: leave parent geometry rasterization methods in GridMap

    # occupancy helpers (write/update)
    def set_occupancy(self, occ: np.ndarray, *, update_esdf: bool = True) -> None:
        occ = np.asarray(occ, dtype=np.float32)
        if occ.shape == (self.grid_width, self.grid_height):
            self.grid = occ.T.copy()
        elif occ.shape == (self.grid_height, self.grid_width):
            self.grid = occ.copy()
        else:
            raise ValueError(f"occ shape {occ.shape} 与地图尺寸 ({self.grid_height},{self.grid_width}) 不匹配")
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_circle_obstacle(self, center: np.ndarray, radius: float, *, update_esdf: bool = True) -> None:
        cx, cy = float(center[0]), float(center[1])
        r_total = radius + self.inflation_radius
        res = float(self.resolution)
        col_min = max(0, int((cx - r_total - self.origin_x) / res))
        col_max = min(self.grid_width - 1,  int((cx + r_total - self.origin_x) / res) + 1)
        row_min = max(0, int((cy - r_total - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((cy + r_total - self.origin_y) / res) + 1)
        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)
        wx = self.origin_x + (cc + 0.5) * res
        wy = self.origin_y + (rr + 0.5) * res
        mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= r_total ** 2
        self.grid[row_min:row_max + 1, col_min:col_max + 1][mask] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_rectangle_obstacle(self, xmin: float, xmax: float, ymin: float, ymax: float, *, update_esdf: bool = True) -> None:
        inf = self.inflation_radius
        xmin -= inf; xmax += inf; ymin -= inf; ymax += inf
        res = float(self.resolution)
        col_min = max(0, int((xmin - self.origin_x) / res))
        col_max = min(self.grid_width - 1,  int((xmax - self.origin_x) / res) + 1)
        row_min = max(0, int((ymin - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((ymax - self.origin_y) / res) + 1)
        self.grid[row_min:row_max + 1, col_min:col_max + 1] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_polygon_obstacle(self, verts: np.ndarray, *, update_esdf: bool = True) -> None:
        verts = np.asarray(verts, dtype=np.float64)
        res = float(self.resolution)
        xs = verts[:, 0]; ys = verts[:, 1]
        col_min = max(0, int((xs.min() - self.origin_x) / res))
        col_max = min(self.grid_width - 1,  int((xs.max() - self.origin_x) / res) + 1)
        row_min = max(0, int((ys.min() - self.origin_y) / res))
        row_max = min(self.grid_height - 1, int((ys.max() - self.origin_y) / res) + 1)
        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)
        wx = self.origin_x + (cc + 0.5) * res
        wy = self.origin_y + (rr + 0.5) * res
        n = len(verts)
        inside = np.zeros(wx.shape, dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]; xj, yj = verts[j]
            cond = ((yi > wy) != (yj > wy)) & \
                   (wx < (xj - xi) * (wy - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            j = i
        self.grid[row_min:row_max + 1, col_min:col_max + 1][inside] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    # ---- ESDF construction & queries ----
    def update_esdf(self) -> None:
        occ_xy = np.asarray(self.grid, dtype=np.int8).T
        nx, ny = occ_xy.shape

        def _edt1d_sq(f: np.ndarray) -> np.ndarray:
            n = len(f)
            if not np.isfinite(f).any():
                return np.full(n, np.inf, dtype=np.float64)
            v = np.zeros(n, dtype=np.int32)
            z = np.zeros(n + 1, dtype=np.float64)
            d = np.zeros(n, dtype=np.float64)
            first = int(np.argmax(np.isfinite(f)))
            v[0] = first
            z[0] = -np.inf
            z[1] = np.inf
            k = 0
            for q in range(first + 1, n):
                if not np.isfinite(f[q]):
                    continue
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
                while s <= z[k]:
                    k -= 1
                    s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = np.inf
            k = 0
            for q in range(n):
                while z[k + 1] < q:
                    k += 1
                i = v[k]
                d[q] = (q - i) * (q - i) + f[i]
            return d

        def _edt2d(feature_mask: np.ndarray) -> np.ndarray:
            if not feature_mask.any():
                return np.full((nx, ny), np.inf, dtype=np.float64)
            INF = np.inf
            gx = np.empty((nx, ny), dtype=np.float64)
            for j in range(ny):
                f = np.where(feature_mask[:, j], 0.0, INF)
                gx[:, j] = _edt1d_sq(f)
            d2 = np.empty((nx, ny), dtype=np.float64)
            for i in range(nx):
                d2[i, :] = _edt1d_sq(gx[i, :])
            return d2

        d2_pos = _edt2d(occ_xy == 1)
        d2_neg = _edt2d(occ_xy == 0)

        res = float(self.resolution)
        dist_pos = np.sqrt(d2_pos) * res
        dist_neg = np.sqrt(d2_neg) * res

        esdf = dist_pos.copy()
        occupied = occ_xy == 1
        esdf[occupied] = -dist_neg[occupied] + res

        self.esdf = esdf
        self._esdf_valid = True

    def _ensure_esdf(self) -> None:
        if not self._esdf_valid or self.esdf is None:
            self.update_esdf()

    def get_distance(self, pos_xy) -> float:
        self._ensure_esdf()
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if x < self.origin_x or x > self.origin_x + self.width or \
           y < self.origin_y or y > self.origin_y + self.height:
            return float("inf")
        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)
        col_f = (x - self.origin_x) / res - 0.5
        row_f = (y - self.origin_y) / res - 0.5
        col0 = int(np.clip(int(col_f), 0, nx - 2))
        row0 = int(np.clip(int(row_f), 0, ny - 2))
        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)
        v00 = float(self.esdf[col0,     row0    ])
        v10 = float(self.esdf[col0 + 1, row0    ])
        v01 = float(self.esdf[col0,     row0 + 1])
        v11 = float(self.esdf[col0 + 1, row0 + 1])
        return float(
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )

    def get_distance_and_gradient(self, pos_xy) -> Tuple[float, np.ndarray]:
        self._ensure_esdf()
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if x < self.origin_x or x > self.origin_x + self.width or \
           y < self.origin_y or y > self.origin_y + self.height:
            return float("inf"), np.zeros(2, dtype=np.float64)
        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)
        col_f = (x - self.origin_x) / res - 0.5
        row_f = (y - self.origin_y) / res - 0.5
        col0 = int(np.clip(int(col_f), 0, nx - 2))
        row0 = int(np.clip(int(row_f), 0, ny - 2))
        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)
        v00 = float(self.esdf[col0,     row0    ])
        v10 = float(self.esdf[col0 + 1, row0    ])
        v01 = float(self.esdf[col0,     row0 + 1])
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
        N = len(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        distances = np.full(N, np.inf, dtype=np.float64)
        gradients = np.zeros((N, 2), dtype=np.float64)
        valid = (x >= self.origin_x) & (x <= self.origin_x + self.width) & \
                (y >= self.origin_y) & (y <= self.origin_y + self.height)
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
        v00 = self.esdf[col0,     row0    ]
        v10 = self.esdf[col0 + 1, row0    ]
        v01 = self.esdf[col0,     row0 + 1]
        v11 = self.esdf[col0 + 1, row0 + 1]
        dist_v = (v00 * (1 - dx) * (1 - dy)
                + v10 * dx       * (1 - dy)
                + v01 * (1 - dx) * dy
                + v11 * dx       * dy)
        grad_x = ((v10 - v00) * (1 - dy) + (v11 - v01) * dy) / res
        grad_y = ((v01 - v00) * (1 - dx) + (v11 - v10) * dx) / res
        distances[valid] = dist_v
        gradients[valid, 0] = grad_x
        gradients[valid, 1] = grad_y
        return distances, gradients


# ══════════════════════════════════════════════════════════════════════════════
# SFC-based obstacle constraint
# ══════════════════════════════════════════════════════════════════════════════

class SFCObstacleConstraint:
    """基于安全飞行走廊（Safe Flight Corridor, SFC）的静态障碍物约束

    设计思路（参考 Dftpav traj_optimizer.cpp::addPVAGradCost2CT 的 cfgHs 部分）
    -------------------------------------------------------------------------
    不依赖 ESDF，而是在优化前先生成一组凸多面体走廊（hPolys），
    每个走廊用若干半平面 n_i^T x <= b_i 表示自由空间。

    对轨迹每个采样点 p(t)，计算点到各半平面的 signed distance：

        d_hp = b_i - n_i^T p     (正值表示在安全侧，负值表示越界)

    取最小 d_hp（最近的走廊边界）作为该点到走廊的 signed distance：

        d_min = min_i(d_hp_i)
        viola = safe_margin - d_min

    若 viola > 0，施加惩罚（二次/线性切换，与 ESDF 方式一致）：

        penalty(viola) =
            { viola^2,                            0 < viola < quad_threshold
            { quad_threshold*(2*viola-quad_threshold),  viola >= quad_threshold

    梯度推导（d(cost)/dp）：
        viola = safe_margin - (b - n^T p)
        d(viola)/dp = n   （外法向量方向）
        d(cost)/dp  = wei_sfc * penaD * n

    走廊结构
    --------
    hPolys_per_piece: list of np.ndarray, len = piece_num
        每个元素 shape (K, 3) —— K 个半平面
        每行 [n_x, n_y, b]，约束为 n^T p <= b（外法向量朝外）
        若某段没有走廊约束，对应元素为 None。
    """

    def __init__(
        self,
        safe_margin: float = 0.0,
        wei_sfc: float = 1e4,
        traj_resolution: int = 16,
        destraj_resolution: int = 32,
        quad_threshold: float = 0.1,
    ):
        self.safe_margin = float(safe_margin)
        self.wei_sfc = float(wei_sfc)
        self.traj_resolution = int(traj_resolution)
        self.destraj_resolution = int(destraj_resolution)
        self.quad_threshold = float(quad_threshold)

        self.hPolys_per_piece: list = []

        self.gdC = None
        self.gdT = None
        self.sfc_cost = 0.0
        self.min_dist = np.inf   # 本轮最小 signed distance，供诊断

    # ------------------------------------------------------------------
    # 外部接口
    # ------------------------------------------------------------------

    def set_corridors(self, hPolys_per_piece: list):
        """设置每段的凸多面体走廊。

        参数
        ----
        hPolys_per_piece : list，长度需等于 piece_num
            每个元素为 np.ndarray shape (K, 3)，每行 [nx, ny, b]，约束 n^T p <= b；
            或 None（该段无约束）。
        """
        self.hPolys_per_piece = hPolys_per_piece

    def reset(self, coeffs: np.ndarray, piece_num: int):
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num)
        self.sfc_cost = 0.0
        self.min_dist = np.inf

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_sfc_cost(self):
        return float(self.sfc_cost)

    # 别名，与 ObstacleConstraint 接口对齐
    def get_obs_cost(self):
        return self.get_sfc_cost()

    # ------------------------------------------------------------------
    # 主计算接口
    # ------------------------------------------------------------------

    def addObstacleGradCost(
        self,
        coeffs: np.ndarray,
        T: np.ndarray,
        piece_num: int,
        grid_map=None,   # SFC 方法不需要 grid_map，保留参数兼容签名
    ) -> float:
        """计算 SFC 走廊约束代价与梯度（接口与 ObstacleConstraint 对齐）。

        参数
        ----
        coeffs   : shape (6*piece_num, 2)
        T        : shape (piece_num,)
        piece_num: 段数
        grid_map : 忽略（SFC 方法不使用）

        返回
        ----
        sfc_cost : float
        """
        self.reset(coeffs, piece_num)

        if not self.hPolys_per_piece or len(self.hPolys_per_piece) != piece_num:
            return 0.0

        for i in range(piece_num):
            hPoly = self.hPolys_per_piece[i]
            if hPoly is None or len(hPoly) == 0:
                continue

            hPoly = np.asarray(hPoly, dtype=np.float64)   # (K_planes, 3)
            normals = hPoly[:, :2]   # (K_planes, 2) 外法向量
            bs = hPoly[:, 2]         # (K_planes,)   偏移量 b

            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution
            c = coeffs[6 * i: 6 * (i + 1), :]   # (6, 2)
            T_i = float(T[i])
            if T_i <= 0 or not np.isfinite(T_i):
                return 1e10

            step = T_i / K

            js = np.arange(K + 1, dtype=np.float64)
            s1 = step * js
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2
            s5 = s4 * s1

            beta0 = np.stack([np.ones(K + 1), s1, s2, s3, s4, s5], axis=1)        # (K+1, 6)
            beta1 = np.stack([np.zeros(K + 1), np.ones(K + 1),
                              2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4], axis=1)   # (K+1, 6)

            pos_all = beta0 @ c    # (K+1, 2)
            vel_all = beta1 @ c    # (K+1, 2)

            omg_vec = np.ones(K + 1)
            omg_vec[0] = 0.5
            omg_vec[K] = 0.5

            # signed distance: d_hp[j, k] = b[k] - n[k]^T p[j]  (正值=安全)
            d_hp = bs[None, :] - pos_all @ normals.T   # (K+1, K_planes)

            d_min_per_pt = np.min(d_hp, axis=1)        # (K+1,)
            min_plane_idx = np.argmin(d_hp, axis=1)    # (K+1,)

            self.min_dist = min(self.min_dist, float(np.min(d_min_per_pt)))

            viola = self.safe_margin - d_min_per_pt    # (K+1,)
            active = viola > 0.0

            if not np.any(active):
                continue

            # 二次/线性惩罚
            penalty = np.where(
                viola < self.quad_threshold,
                viola ** 2,
                self.quad_threshold * (2.0 * viola - self.quad_threshold),
            )
            penaD = np.where(viola < self.quad_threshold, 2.0 * viola, 2.0 * self.quad_threshold)
            penalty = np.where(active, penalty, 0.0)
            penaD   = np.where(active, penaD,   0.0)

            cost_c = self.wei_sfc * penalty             # (K+1,)
            weights = omg_vec * step                    # (K+1,)

            # grad_p = d(cost)/dp = wei_sfc * penaD * n
            # viola = margin - (b - n^T p), d(viola)/dp = n (外法向量)
            active_normals = normals[min_plane_idx]                            # (K+1, 2)
            grad_p = self.wei_sfc * penaD[:, None] * active_normals           # (K+1, 2)

            self.sfc_cost += float(np.dot(weights, cost_c))

            self.gdC[6 * i: 6 * (i + 1), :] += beta0.T @ (weights[:, None] * grad_p)

            alpha_vec = js / K
            dot_gv = np.sum(grad_p * vel_all, axis=1)   # (K+1,)
            self.gdT[i] += float(np.dot(omg_vec, cost_c / K + step * alpha_vec * dot_gv))

        return float(self.sfc_cost)


# ------------------------------------------------------------------
# SFC / Corridor builder helpers
# ------------------------------------------------------------------

_N_BINS = 36


def _nearest_pt_on_segment(p0, p1, pts):
    """计算 pts 中每个点到线段 p0-p1 的最近点与距离。"""
    seg = p1 - p0
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 < 1e-12:
        nearest = np.tile(p0, (len(pts), 1))
        dists = np.linalg.norm(pts - p0, axis=1)
        return nearest, dists
    t = np.clip(((pts - p0) @ seg) / seg_len2, 0.0, 1.0)
    nearest = p0 + t[:, None] * seg
    dists = np.linalg.norm(pts - nearest, axis=1)
    return nearest, dists


def _map_bounds_to_hpoly(map_bounds, center, fallback_radius=5.0):
    """将地图边界转换为半平面表示。"""
    if map_bounds is not None:
        xmin, ymin, xmax, ymax = map_bounds
    else:
        xmin = center[0] - fallback_radius
        xmax = center[0] + fallback_radius
        ymin = center[1] - fallback_radius
        ymax = center[1] + fallback_radius
    return [
        [1.0, 0.0, xmax],
        [-1.0, 0.0, -xmin],
        [0.0, 1.0, ymax],
        [0.0, -1.0, -ymin],
    ]


def build_corridor_for_segment(p0, p1, obs_pts,
                                search_radius=6.0, n_bins=_N_BINS,
                                map_bounds=None):
    """为一段轨迹生成半平面走廊。返回 shape (K, 3) 的 hPoly。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = (p0 + p1) * 0.5
    half_planes = []
    if obs_pts is not None and len(obs_pts) > 0:
        nearest, dists = _nearest_pt_on_segment(p0, p1, obs_pts)
        mask = dists < search_radius
        nearby_obs = obs_pts[mask]
        nearby_nearest = nearest[mask]
        nearby_dists = dists[mask]
        if len(nearby_obs) > 0:
            dx = nearby_obs[:, 0] - center[0]
            dy = nearby_obs[:, 1] - center[1]
            angles = np.arctan2(dy, dx)
            bin_width = 2.0 * np.pi / n_bins
            bin_ids = ((angles + np.pi) / bin_width).astype(int) % n_bins
            best = {}
            for idx in range(len(nearby_obs)):
                bid = bin_ids[idx]
                d = nearby_dists[idx]
                if bid not in best or d < best[bid][2]:
                    best[bid] = (nearby_obs[idx], nearby_nearest[idx], d)
            for (q, nearest_pt, d) in best.values():
                if d < 1e-6:
                    continue
                n = (q - nearest_pt) / d
                b = float(np.dot(n, q))
                if np.dot(n, center) <= b + 1e-6:
                    half_planes.append([n[0], n[1], b])
    for hp in _map_bounds_to_hpoly(map_bounds, center):
        n = np.array(hp[:2])
        b = hp[2]
        if np.dot(n, center) <= b + 1e-6:
            half_planes.append(list(hp))
    if not half_planes:
        return np.array(_map_bounds_to_hpoly(map_bounds, center), dtype=np.float64)
    return np.array(half_planes, dtype=np.float64)


def build_corridors(waypoints, obs_pts,
                    search_radius=6.0, n_bins=_N_BINS, map_bounds=None,
                    traj_resolution=16, destraj_resolution=32, flip_radius=100.0):
    """为轨迹每段生成走廊，返回 list of hPoly（len = piece_num）。"""
    waypoints = np.asarray(waypoints, dtype=float)
    piece_num = len(waypoints) - 1
    if piece_num <= 0:
        return []
    return [
        build_corridor_for_segment(
            waypoints[i], waypoints[i + 1], obs_pts,
            search_radius=search_radius, n_bins=n_bins, map_bounds=map_bounds,
        )
        for i in range(piece_num)
    ]


def extract_obs_points_from_gridmap(grid_map, subsample=1):
    """从 GridMap2D 提取占据栅格的世界坐标（障碍物点云）。"""
    occ = np.asarray(grid_map.occ)
    ny, nx = occ.shape
    pts = []
    for r in range(0, ny, subsample):
        for c in range(0, nx, subsample):
            if occ[r, c] > 0:
                pts.append(grid_map.index_to_coor((r, c)))
    if not pts:
        return np.empty((0, 2))
    return np.array(pts, dtype=np.float64)
