import numpy as np


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
