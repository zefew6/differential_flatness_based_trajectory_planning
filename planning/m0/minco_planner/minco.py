"""
MINCO (Minimum Control) 轨迹优化 - 二维版本

包含:
- MINCO: 五次多项式轨迹生成（C3连续）
- MinJerkOpt: Jerk优化器
- PolyTrajOptimizer: 基于L-BFGS的轨迹优化器
"""

import numpy as np
import matplotlib.pyplot as plt


class MINCO:
    """五次多项式轨迹生成器（C3连续）
    
    p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
    """
    
    def __init__(self, head_pva, tail_pva, waypoints, durations):
        """初始化并求解轨迹
        
        参数:
            head_pva: 起始状态 [[px,py], [vx,vy], [ax,ay]], shape (3,2)
            tail_pva: 终止状态，同上
            waypoints: 中间航点列表 [[x1,y1], [x2,y2], ...]
            durations: 每段时间 [t1, t2, ...]
        """
        self.N = len(durations)
        self.head_pva = np.array(head_pva)
        self.tail_pva = np.array(tail_pva)
        self.waypoints = np.array(waypoints)
        self.T = np.array(durations)
        self.s = 3  # 控制量阶数
        self.m = 2  # 维数(x,y)
        
        # 时间幂次
        self.T2 = self.T ** 2
        self.T3 = self.T ** 3
        self.T4 = self.T ** 4
        self.T5 = self.T ** 5
        
        self.coeffs = self._solve()
    
    def _solve(self):
        """构造并求解线性系统 Ax = b"""
        s, m, M = self.s, self.m, self.N
        n = 2 * M * s
        A = np.zeros((n, n))
        b = np.zeros((n, m))

        # 初始位置和速度
        b[0] = self.head_pva[0]  # p(0)
        b[1] = self.head_pva[1]  # v(0)
        b[2] = self.head_pva[1]  # a(0)
        
        # 初始条件
        A[0, 0] = 1.0  # c0 = p(0)
        A[1, 1] = 1.0  # c1 = v(0)
        A[2, 2] = 2.0  # c2 = a(0)
        # 每段的约束
        for i in range(self.N - 1):
            # 三阶连续：  \beta`3(T)*c_{i} - \beta(0)`3*c_{i+1} = 0 
            A[6 * i + 3, 6 * i + 3] = 6.0
            A[6 * i + 3, 6 * i + 4] = 24.0 * self.T[i]
            A[6 * i + 3, 6 * i + 5] = 60.0 * self.T2[i]
            A[6 * i + 3, 6 * i + 9] = -6.0

            # 四阶连续：  \beta`4(T)*c_{i} - \beta(0)`4*c_{i+1} = 0  
            A[6 * i + 4, 6 * i + 4] = 24.0
            A[6 * i + 4, 6 * i + 5] = 120.0 * self.T[i]
            A[6 * i + 4, 6 * i + 10] = -24.0

            # 位置到达航点: \beta(T)*c_i = given point
            A[6 * i + 5, 6 * i] = 1.0
            A[6 * i + 5, 6 * i + 1] = self.T[i]
            A[6 * i + 5, 6 * i + 2] = self.T2[i]
            A[6 * i + 5, 6 * i + 3] = self.T3[i]
            A[6 * i + 5, 6 * i + 4] = self.T4[i]
            A[6 * i + 5, 6 * i + 5] = self.T5[i]

            # 位置连续: \beta(T)*c_{i} - \beta(0)*c_{i+1} = 0  
            A[6 * i + 6, 6 * i] = 1.0
            A[6 * i + 6, 6 * i + 1] = self.T[i]
            A[6 * i + 6, 6 * i + 2] = self.T2[i]
            A[6 * i + 6, 6 * i + 3] = self.T3[i]
            A[6 * i + 6, 6 * i + 4] = self.T4[i]
            A[6 * i + 6, 6 * i + 5] = self.T5[i]
            A[6 * i + 6, 6 * i + 6] = -1.0
            
            # 速度连续: \beta`1(T)*c_{i} - \beta`1(0)*c_{i+1} = 0  
            A[6 * i + 7, 6 * i + 1] = 1.0
            A[6 * i + 7, 6 * i + 2] = 2 * self.T[i]
            A[6 * i + 7, 6 * i + 3] = 3 * self.T2[i]
            A[6 * i + 7, 6 * i + 4] = 4 * self.T3[i]
            A[6 * i + 7, 6 * i + 5] = 5 * self.T4[i]
            A[6 * i + 7, 6 * i + 7] = -1.0

            # 加速度连续: \beta`2(T)*c_{i} - \beta`2(0)*c_{i+1} = 0  
            A[6 * i + 8, 6 * i + 2] = 2.0
            A[6 * i + 8, 6 * i + 3] = 6 * self.T[i]
            A[6 * i + 8, 6 * i + 4] = 12 * self.T2[i]
            A[6 * i + 8, 6 * i + 5] = 20 * self.T3[i]
            A[6 * i + 8, 6 * i + 8] = -2.0
            # 中间航点
            b[6*i + 5] = self.waypoints[i]
            
        # 终止条件
        # 位置
        A[n - 3, n - 6] = 1.0
        A[n - 3, n - 5] = self.T[-1]
        A[n - 3, n - 4] = self.T2[-1]
        A[n - 3, n - 3] = self.T3[-1]
        A[n - 3, n - 2] = self.T4[-1]
        A[n - 3, n - 1] = self.T5[-1]
        # 速度
        A[n - 2, n - 5] = 1.0
        A[n - 2, n - 4] = 2*self.T[-1]
        A[n - 2, n - 3] = 3*self.T2[-1]
        A[n - 2, n - 2] = 4*self.T3[-1]
        A[n - 2, n - 1] = 5*self.T4[-1]
        # 加速度
        A[n - 1, n - 4] = 2.0
        A[n - 1, n - 3] = 6.0 * self.T[-1]
        A[n - 1, n - 2] = 12.0 * self.T2[-1]
        A[n - 1, n - 1] = 20.0 * self.T3[-1]

        # 终止位置和速度
        b[n - 3] = self.tail_pva[0]  # p(T)
        b[n - 2] = self.tail_pva[1]  # v(T)
        b[n - 1] = self.tail_pva[2]  # a(T)
        # print(A)
        # print(b)
        # 求解
        coeffs = np.linalg.solve(A, b)
        # print('A:',A)
        # print('b:',b)
        # print('coeffs:',coeffs)
        return coeffs
    
    def eval(self, t):
        """
        评估轨迹在时刻 t
        
        参数:
            t: 时间 (从 0 开始)
        
        返回:
            pos: 位置 [x, y]
            vel: 速度 [vx, vy]
            acc: 加速度 [ax, ay]
        """
        # 找到对应的轨迹段
        t_cumsum = np.cumsum(self.T)
        piece_idx = np.searchsorted(t_cumsum, t)
        piece_idx = min(piece_idx, self.N - 1)
        
        # 计算段内时间
        if piece_idx == 0:
            t_local = t
        else:
            t_local = t - t_cumsum[piece_idx - 1]
        
        # 获取该段系数（每段6个系数：c0-c5）
        c0 = self.coeffs[6*piece_idx + 0, :]
        c1 = self.coeffs[6*piece_idx + 1, :]
        c2 = self.coeffs[6*piece_idx + 2, :]
        c3 = self.coeffs[6*piece_idx + 3, :]
        c4 = self.coeffs[6*piece_idx + 4, :]
        c5 = self.coeffs[6*piece_idx + 5, :]
        # 计算位置、速度、加速度
        t2 = t_local ** 2
        t3 = t2 * t_local
        t4 = t3 * t_local
        t5 = t4 * t_local
        # p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
        pos = c0 + c1*t_local + c2*t2 + c3*t3 + c4*t4 + c5*t5
        # v(t) = c1 + 2*c2*t + 3*c3*t² + 4*c4*t³ + 5*c5*t⁴
        vel = c1 + 2*c2*t_local + 3*c3*t2 + 4*c4*t3 + 5*c5*t4
        # a(t) = 2*c2 + 6*c3*t + 12*c4*t² + 20*c5*t³
        acc = 2*c2 + 6*c3*t_local + 12*c4*t2 + 20*c5*t3
        
        return pos, vel, acc
    
    def get_energy(self):
        """计算轨迹能量 (snap/四阶导数平方的积分)
        
        对于5阶多项式 p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
        四阶导数 p⁴(t) = 24*c4 + 120*c5*t
        能量 = ∫[p⁴(t)]² dt = ∫(24*c4 + 120*c5*t)² dt
             = ∫(576*c4² + 5760*c4*c5*t + 14400*c5²*t²) dt
             = 576*c4²*T + 2880*c4*c5*T² + 4800*c5²*T³
        """
        energy = 0.0
        for i in range(self.N):
            c4 = self.coeffs[6*i + 4, :]
            c5 = self.coeffs[6*i + 5, :]
            
            # snap 平方积分的系数
            energy += 576.0 * np.dot(c4, c4) * self.T[i]
            energy += 2880.0 * np.dot(c4, c5) * self.T2[i]
            energy += 4800.0 * np.dot(c5, c5) * self.T3[i]
        
        return energy

    def plot_trajectory(
        self,
        *,
        dt: float = 0.01,
        show_waypoints: bool = True,
        show_velocity: bool = True,
        show_acceleration: bool = True,
        title: str | None = None,
    ):
        """Plot trajectory position/velocity/acceleration in a single figure.

        Notes:
        - This method is used by `minco_test.py` for interactive visualization.
        - Returns (fig, axes) where axes is a 2x2 ndarray.
        """
        t_total = float(np.sum(self.T))
        if t_total <= 0:
            raise ValueError("Trajectory total time must be positive.")

        ts = np.arange(0.0, t_total + 1e-12, float(dt))
        ps = np.zeros((len(ts), 2), dtype=float)
        vs = np.zeros((len(ts), 2), dtype=float)
        accs = np.zeros((len(ts), 2), dtype=float)

        for k, t in enumerate(ts):
            p, v, a = self.eval(float(t))
            ps[k] = p
            vs[k] = v
            accs[k] = a

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_xy = axes[0, 0]
        ax_v = axes[0, 1]
        ax_a = axes[1, 0]
        ax_speed = axes[1, 1]

        # XY path
        ax_xy.plot(ps[:, 0], ps[:, 1], "b-", linewidth=2.0, label="Trajectory")
        ax_xy.plot(ps[0, 0], ps[0, 1], "go", label="Start")
        ax_xy.plot(ps[-1, 0], ps[-1, 1], "ro", label="Goal")
        if show_waypoints and getattr(self, "waypoints", None) is not None and len(self.waypoints) > 0:
            wpts = np.asarray(self.waypoints)
            ax_xy.plot(
                wpts[:, 0],
                wpts[:, 1],
                linestyle="none",
                marker="o",
                markersize=6,
                color="purple",
                label="Waypoints",
            )
        ax_xy.set_title("XY Path")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.axis("equal")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend(loc="best")

        # Velocity
        if show_velocity:
            ax_v.plot(ts, vs[:, 0], label="vx")
            ax_v.plot(ts, vs[:, 1], label="vy")
            ax_v.set_title("Velocity")
            ax_v.set_xlabel("t")
            ax_v.set_ylabel("v")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend(loc="best")
        else:
            ax_v.axis("off")

        # Acceleration
        if show_acceleration:
            ax_a.plot(ts, accs[:, 0], label="ax")
            ax_a.plot(ts, accs[:, 1], label="ay")
            ax_a.set_title("Acceleration")
            ax_a.set_xlabel("t")
            ax_a.set_ylabel("a")
            ax_a.grid(True, alpha=0.3)
            ax_a.legend(loc="best")
        else:
            ax_a.axis("off")

        # Speed magnitude
        speed = np.linalg.norm(vs, axis=1)
        ax_speed.plot(ts, speed, "m-", label="|v|")
        ax_speed.set_title("Speed")
        ax_speed.set_xlabel("t")
        ax_speed.set_ylabel("|v|")
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend(loc="best")

        if title:
            fig.suptitle(str(title))
        fig.tight_layout()
        return fig, axes
    
    def get_traj(self, dt=0.01, z_height=0.03):
        """
        根据系数和时间段生成完整轨迹
        
        参数:
            dt: 采样时间间隔 (秒)
            z_height: 可视化时的z坐标高度 (用于3D显示)
        
        返回:
            trajectory: dict包含
                'time': 时间数组
                'position': 位置数组 shape (n_samples, 3) - [x, y, z]
                'velocity': 速度数组 shape (n_samples, 2)
                'acceleration': 加速度数组 shape (n_samples, 2)
        """
        total_time = np.sum(self.T)
        time_samples = np.arange(0, total_time + dt, dt)
        
        n_samples = len(time_samples)
        positions = np.zeros((n_samples, 3))  # 改为3维，添加z坐标
        velocities = np.zeros((n_samples, self.m))
        accelerations = np.zeros((n_samples, self.m))
        
        for i, t in enumerate(time_samples):
            # 确保最后一个点不超过总时间
            t = min(t, total_time)
            pos, vel, acc = self.eval(t)
            positions[i, 0:2] = pos  # x, y
            positions[i, 2] = z_height  # z
            velocities[i] = vel
            accelerations[i] = acc
        
        trajectory = {
            'time': time_samples,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations
        }
        
        return trajectory
    
    # NOTE:
    # This class previously had a second `plot_trajectory` implementation below which
    # *overrode* the newer 2x2 kinematics version above and returned None.
    # That broke callers (e.g. `minco_test.py`) that expect `(fig, axes)`.
    # The duplicate implementation has been removed to keep a single, consistent API.
        if show_velocity:
            axes[ax_idx].plot(time_samples, velocities[:, 0], 'r-', label='Vx')
            axes[ax_idx].plot(time_samples, velocities[:, 1], 'b-', label='Vy')
            vel_magnitude = np.linalg.norm(velocities, axis=1)
            axes[ax_idx].plot(time_samples, vel_magnitude, 'k--', label='|V|', linewidth=2)
            axes[ax_idx].set_xlabel('Time (s)', fontsize=12)
            axes[ax_idx].set_ylabel('Velocity (m/s)', fontsize=12)
            axes[ax_idx].set_title('Velocity Profile', fontsize=14)
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1
        
        # 3. 绘制加速度曲线
        if show_acceleration:
            axes[ax_idx].plot(time_samples, accelerations[:, 0], 'r-', label='Ax')
            axes[ax_idx].plot(time_samples, accelerations[:, 1], 'b-', label='Ay')
            acc_magnitude = np.linalg.norm(accelerations, axis=1)
            axes[ax_idx].plot(time_samples, acc_magnitude, 'k--', label='|A|', linewidth=2)
            axes[ax_idx].set_xlabel('Time (s)', fontsize=12)
            axes[ax_idx].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[ax_idx].set_title('Acceleration Profile', fontsize=14)
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)  # 非阻塞模式，允许同时显示多个窗口
        plt.pause(0.1)  # 短暂暂停，确保窗口正确渲染

# ============================================================
# 测试代码
# ============================================================
