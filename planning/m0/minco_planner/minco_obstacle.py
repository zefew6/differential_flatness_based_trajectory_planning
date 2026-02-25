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

    def reset(self, coeffs: np.ndarray, piece_num: int):
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num, dtype=np.float64)
        self.obs_cost = 0.0

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

        for i in range(piece_num):
            # 采样分辨率：起止段更密
            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution

            c = coeffs[6 * i : 6 * (i + 1), :]  # (6,2)
            T_i = float(T[i])
            if T_i <= 0 or not np.isfinite(T_i):
                # 不合理时间，直接给大惩罚
                return 1e10

            step = T_i / K

            for j in range(K + 1):
                omg = 0.5 if (j == 0 or j == K) else 1.0
                s1 = step * j
                s2 = s1 * s1
                s3 = s2 * s1
                s4 = s2 * s2
                s5 = s4 * s1
                alpha = j / K

                beta0 = np.array([1.0, s1, s2, s3, s4, s5])
                beta1 = np.array([0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4])

                pos = c.T @ beta0
                vel = c.T @ beta1

                # SDF & gradient
                dist, grad_sdf = grid_map.get_distance_and_gradient(pos)
                dist = float(dist)
                grad_sdf = np.asarray(grad_sdf, dtype=np.float64)

                viola = self.safe_threshold - dist
                if viola > 0.0 and dist < self.dist_cap:
                    # penalty + derivative wrt viola
                    if viola < self.quad_threshold:
                        penalty = viola * viola
                        penaD = 2.0 * viola
                    else:
                        penalty = viola
                        penaD = 1.0

                    cost_c = self.wei_obs * penalty

                    # d(cost)/dp = wei_obs * penaD * d(viola)/dp = wei_obs*penaD*(-grad_sdf)
                    grad_p = self.wei_obs * penaD * (-grad_sdf)

                    # accumulate cost
                    self.obs_cost += omg * step * cost_c

                    # coeff gradient: outer(beta0, grad_p)
                    self.gdC[6 * i : 6 * (i + 1), :] += omg * step * np.outer(beta0, grad_p)

                    # time gradient: align ST-opt-tools discrete form
                    # grad_time += omg * (cost/K + step * alpha * grad_p · v)
                    self.gdT[i] += omg * (cost_c / K + step * alpha * float(np.dot(grad_p, vel)))

        return float(self.obs_cost)

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_obs_cost(self):
        return float(self.obs_cost)
