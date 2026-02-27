import numpy as np


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
