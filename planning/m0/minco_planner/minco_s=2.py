"""
MINCO 最小实现 - 用于 s=2 (三次多项式) 的最小控制轨迹优化

二维版本 (x, y)，简化版本，只包含核心功能：
- 设置边界条件和航点
- 求解多项式系数
- 评估轨迹
"""

import numpy as np


class MINCO:
    """
    MINCO for s=2 (三次多项式) 和非均匀时间 - 二维版本
    
    每段轨迹是三次多项式: p(t) = c0 + c1*t + c2*t^2 + c3*t^3
    确保位置、速度、加速度连续 (C2 连续)
    """
    
    def __init__(self, head_pv, tail_pv, waypoints, durations):
        """
        初始化并求解轨迹
        
        参数:
            head_pv: 起始状态 [pos, vel], shape (2, 2) - [[px,vx], [py,vy]]
            tail_pv: 终止状态 [pos, vel], shape (2, 2)
            waypoints: 中间航点, shape (2, N-1) - 每列是一个航点 [px, py]
            durations: 每段时间, shape (N,) - N 是轨迹段数
        """
        self.N = len(durations)  # 轨迹段数
        self.head_pv = np.array(head_pv)
        self.tail_pv = np.array(tail_pv)
        self.waypoints = np.array(waypoints)
        self.T = np.array(durations)
        
        # 计算时间的幂次
        self.T2 = self.T ** 2
        self.T3 = self.T ** 3
        # 求解系数
        self.coeffs = self._solve()
    
    def _solve(self):
        """构造并求解线性系统 Ax = b"""
        n = 2 * self.N * 2 # 总方程数 = 4 * 段数
        
        # 为每个维度 (x, y) 分别求解
        coeffs = np.zeros((n, 2)) #2Ms*m 维，其中M为段数 s为多项式阶速（2阶），m为轨迹个数（x和y）
        print(coeffs)
        for dim in range(2):
            # 构造右侧向量 b
            b = np.zeros(n)
            
            # 初始位置和速度
            b[0] = self.head_pv[dim, 0]  # p(0)
            b[1] = self.head_pv[dim, 1]  # v(0)
            
            # 中间航点
            for i in range(self.N - 1):
                b[4*i + 3] = self.waypoints[dim, i]
            
            # 终止位置和速度
            b[n - 2] = self.tail_pv[dim, 0]  # p(T)
            b[n - 1] = self.tail_pv[dim, 1]  # v(T)
            
            # 构造带状矩阵 A (带宽 = 4)
            A = np.zeros((n, n))
            
            # 初始条件
            A[0, 0] = 1.0  # c0 = p(0)
            A[1, 1] = 1.0  # c1 = v(0)
            
            # 每段的约束
            for i in range(self.N - 1):
                # 加速度连续: 2*c2_i + 6*c3_i*T_i = 2*c2_{i+1}
                A[4*i + 2, 4*i + 2] = 2.0
                A[4*i + 2, 4*i + 3] = 6.0 * self.T[i]
                A[4*i + 2, 4*i + 6] = -2.0
                
                # 位置到达航点: c0 + c1*T + c2*T^2 + c3*T^3 = waypoint
                A[4*i + 3, 4*i + 0] = 1.0
                A[4*i + 3, 4*i + 1] = self.T[i]
                A[4*i + 3, 4*i + 2] = self.T2[i]
                A[4*i + 3, 4*i + 3] = self.T3[i]
                
                # 位置连续: p_i(T_i) = p_{i+1}(0)
                A[4*i + 4, 4*i + 0] = 1.0
                A[4*i + 4, 4*i + 1] = self.T[i]
                A[4*i + 4, 4*i + 2] = self.T2[i]
                A[4*i + 4, 4*i + 3] = self.T3[i]
                A[4*i + 4, 4*i + 4] = -1.0
                
                # 速度连续: v_i(T_i) = v_{i+1}(0)
                A[4*i + 5, 4*i + 1] = 1.0
                A[4*i + 5, 4*i + 2] = 2.0 * self.T[i]
                A[4*i + 5, 4*i + 3] = 3.0 * self.T2[i]
                A[4*i + 5, 4*i + 5] = -1.0
            
            # 终止条件
            A[n - 2, n - 4] = 1.0
            A[n - 2, n - 3] = self.T[-1]
            A[n - 2, n - 2] = self.T2[-1]
            A[n - 2, n - 1] = self.T3[-1]
            
            A[n - 1, n - 3] = 1.0
            A[n - 1, n - 2] = 2.0 * self.T[-1]
            A[n - 1, n - 1] = 3.0 * self.T2[-1]
            
            # 求解
            coeffs[:, dim] = np.linalg.solve(A, b)
        
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
        
        # 获取该段系数
        c0 = self.coeffs[4*piece_idx + 0, :]
        c1 = self.coeffs[4*piece_idx + 1, :]
        c2 = self.coeffs[4*piece_idx + 2, :]
        c3 = self.coeffs[4*piece_idx + 3, :]
        
        # 计算位置、速度、加速度
        t2 = t_local ** 2
        t3 = t2 * t_local
        
        pos = c0 + c1*t_local + c2*t2 + c3*t3
        vel = c1 + 2*c2*t_local + 3*c3*t2
        acc = 2*c2 + 6*c3*t_local
        
        return pos, vel, acc
    
    def get_energy(self):
        """计算轨迹能量 (加加速度平方的积分)"""
        energy = 0.0
        for i in range(self.N):
            c2 = self.coeffs[4*i + 2, :]
            c3 = self.coeffs[4*i + 3, :]
            
            energy += 4.0 * np.dot(c2, c2) * self.T[i]
            energy += 12.0 * np.dot(c2, c3) * self.T2[i]
            energy += 12.0 * np.dot(c3, c3) * self.T3[i]
        
        return energy


# 使用示例
if __name__ == "__main__":
    # 定义起点和终点 (位置, 速度) - 二维
    head_pv = [
        [0.0, 0.0],  # x: 位置 0, 速度 0
        [0.0, 0.0],  # y: 位置 0, 速度 0
    ]
    
    tail_pv = [
        [5.0, 0.0],  # x: 位置 10, 速度 0
        [5.0, 0.0],  # y: 位置 10, 速度 0
    ]
    
    # 定义中间航点 (第一段结束时的位置)
    waypoints = [
        [2],   # x
        [2],   # y
    ]
    
    # 每段的时间
    durations = [0.2, 0.2]  # 2段，每段1秒
    
    # 创建轨迹
    traj = MINCO(head_pv, tail_pv, waypoints, durations)
    
    print(f"轨迹能量: {traj.get_energy():.4f}\n")
    
    # 评估几个时刻
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pos, vel, acc = traj.eval(t)
        print(f"t={t:.1f}s: pos={pos}, vel={vel}, acc={acc}")
