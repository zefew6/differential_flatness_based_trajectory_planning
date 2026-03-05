"""
微分平坦轨迹跟踪控制器（Kanayama + 投影法）

适用于差速轮式机器人（unicycle model）。从 MINCO 平坦输出的一、二阶导
直接解析恢复参考状态，不依赖数值差分。

用法
----
    from m0.control.trajectory_follower import TrajectoryFollower

    follower = TrajectoryFollower(minco_traj)
    while not follower.done:
        v, w = follower.step(robot_pos_xy, robot_yaw, dt)
        robot.set_ctrl(v, w)
"""

import numpy as np
from m0.utils.utils import wrap_pi


class TrajectoryFollower:
    """差速机器人微分平坦轨迹跟踪控制器（投影法 + Kanayama）。

    控制律
    ------
        v = vd_eff + kx · ex
        ω = ωd + vd_eff · (ky · ey + kθ · sin(eθ))

    其中 vd, θd, ωd 由平坦输出的一、二阶导直接恢复（无数值差分）。
    """

    def __init__(self, minco_traj,
                 max_v: float = 2.0,
                 max_w: float = 4.0,
                 kx: float = 2.0,       # 纵向误差增益
                 ky: float = 2.0,       # 横向误差增益
                 ktheta: float = 3.0,   # 航向误差增益
                 proj_samples: int = 40,
                 proj_window: float = 2.0,
                 vd_min: float = 0.6,   # 巡航段最低推进速度
                 a_brake: float = 3.0,  # 制动减速度 (m/s²)，须 ≤ 物理可达值
                 goal_tol: float = 0.15):
        self.traj     = minco_traj
        self.t_total  = float(np.sum(minco_traj.T))
        self.max_v    = max_v
        self.max_w    = max_w
        self.kx       = kx
        self.ky       = ky
        self.ktheta   = ktheta
        self.proj_samples = proj_samples
        self.proj_window  = proj_window
        self.vd_min   = vd_min
        self.a_brake  = a_brake
        self.goal_tol = goal_tol

        self.t_proj = 0.0
        self.done   = False
        self.trail  = []   # 记录机器人实际轨迹，便于可视化

    def reset(self):
        """重置控制器状态，可用于同一轨迹的重跑。"""
        self.t_proj = 0.0
        self.done   = False
        self.trail  = []

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _project(self, robot_pos: np.ndarray) -> float:
        """在 [t_proj, t_proj+window] 内寻找距机器人最近的轨迹时刻。"""
        t_lo = self.t_proj
        t_hi = min(self.t_proj + self.proj_window, self.t_total)
        ts   = np.linspace(t_lo, t_hi, self.proj_samples)
        best_t, best_d2 = t_lo, np.inf
        for t in ts:
            p, _, _ = self.traj.eval(t)
            d2 = (p[0] - robot_pos[0]) ** 2 + (p[1] - robot_pos[1]) ** 2
            if d2 < best_d2:
                best_d2, best_t = d2, t
        return best_t

    @staticmethod
    def _flat_to_ref(pos, vel, acc):
        """从平坦输出（位置/速度/加速度）解析计算参考状态。

        返回 (xd, yd, θd, vd, ωd)
        """
        xdot, ydot   = vel
        xddot, yddot = acc
        vd      = np.hypot(xdot, ydot)
        theta_d = np.arctan2(ydot, xdot)
        omega_d = (xdot * yddot - ydot * xddot) / (vd ** 2) if vd > 1e-4 else 0.0
        return pos[0], pos[1], theta_d, vd, omega_d

    # ── 主控制步 ──────────────────────────────────────────────────────────

    def step(self, robot_pos: np.ndarray, robot_yaw: float, dt: float):
        """执行一步跟踪控制，返回 (v, ω) 指令。

        参数
        ----
        robot_pos : (2,) 世界系 xy 位置
        robot_yaw : 机器人当前偏航角（弧度）
        dt        : 控制周期（秒），当前版本仅预留，暂未使用

        返回
        ----
        v : 线速度指令 (m/s)，负值表示主动倒车制动
        w : 角速度指令 (rad/s)
        """
        self.trail.append(robot_pos.copy())

        # ── 终点距离 ─────────────────────────────────────────────────────
        final_pos, _, _ = self.traj.eval(self.t_total)
        dist_to_goal = np.hypot(robot_pos[0] - final_pos[0],
                                robot_pos[1] - final_pos[1])

        if dist_to_goal < self.goal_tol and self.t_proj >= self.t_total * 0.85:
            self.done = True
            return 0.0, 0.0

        # 轨迹投影（t_proj 单调递增）
        t_new       = self._project(robot_pos)
        self.t_proj = max(self.t_proj, t_new)

        # ── 参考状态恢复 ─────────────────────────────────────────────────
        ref_pos, ref_vel, ref_acc = self.traj.eval(self.t_proj)
        xd, yd, theta_d, vd, omega_d = self._flat_to_ref(ref_pos, ref_vel, ref_acc)

        # vd_min 在末段 (brake_window 秒) 内线性衰减至 0
        brake_window    = 2.0
        t_remaining     = self.t_total - self.t_proj
        brake_factor    = np.clip(t_remaining / brake_window, 0.0, 1.0)
        vd_min_tapered  = self.vd_min * brake_factor
        vd_eff          = max(vd, vd_min_tapered)

        # ── 机体系误差 ────────────────────────────────────────────────────
        dx = xd - robot_pos[0]
        dy = yd - robot_pos[1]
        cos_th, sin_th = np.cos(robot_yaw), np.sin(robot_yaw)
        ex      =  cos_th * dx + sin_th * dy
        ey      = -sin_th * dx + cos_th * dy
        e_theta = wrap_pi(theta_d - robot_yaw)

        # ── Kanayama 控制律 ───────────────────────────────────────────────
        v = vd_eff + self.kx * ex
        w = omega_d + vd_eff * (self.ky * ey + self.ktheta * np.sin(e_theta))

        # ── 运动学制动速度上限 v_cap = sqrt(2·a_brake·dist) ──────────────
        # 允许 v < 0（主动倒车），但幅值同样受 v_cap 约束
        v_cap = np.sqrt(max(2.0 * self.a_brake * dist_to_goal, 0.0))
        v = np.clip(v, -v_cap, min(v_cap, self.max_v))
        w = np.clip(w, -self.max_w, self.max_w)

        return float(v), float(w)

    @property
    def ref_point(self) -> np.ndarray:
        """返回当前投影点的 xy 坐标，用于可视化。"""
        pos, _, _ = self.traj.eval(self.t_proj)
        return pos
