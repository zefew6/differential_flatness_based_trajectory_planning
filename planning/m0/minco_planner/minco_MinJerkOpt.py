"""
MinJerkOpt — MINCO 轨迹的 Jerk 优化器

提供 MinJerkOpt 类，封装以下功能：
- generate(): 给定边界条件和时间分配，调用 MINCO 求解五次多项式系数。
- initSmGradCost() / getTrajJerkCost(): 计算 Jerk 平方积分代价及其梯度。
- calGrads_PT(): 通过伴随法将代价梯度从系数空间映射回航点/时间空间。

此类被 PolyTrajOptimizer 用作每条轨迹的内部求解器。
"""

import numpy as np
from .minco import MINCO


class MinJerkOpt:
    """
    最小 Jerk 优化器 - 用于单条轨迹的优化
    基于五次多项式的 Minimum Snap/Jerk 优化
    """
    
    def __init__(self, piece_num):
        """
        初始化优化器
        
        参数:
            piece_num: 轨迹段数
        """
        self.piece_num = piece_num
        self.dim = 2  # 二维空间 (x, y)
        self.coeffs = None  # shape: (6*piece_num, 2)
        self.T = None  # 每段的时间
        self.dt = None  # 单段时间
        
        # 用于存储梯度
        self.gdC = None  # 系数的梯度
        self.gdT = 0.0   # 时间的梯度
        self.gdP = None  # 航点的梯度
        self.gdHead = None  # 起点状态的梯度
        self.gdTail = None  # 终点状态的梯度
    
    def reset(self, piece_num):
        """重置优化器"""
        self.piece_num = piece_num
        self.coeffs = None
        self.T = None
        self.dt = None
        self.gdC = None
        self.gdT = 0.0
        self.gdP = None
        self.gdHead = None
        self.gdTail = None
    
    def generate(self, P, T_or_dt, iniState, finState):
        """生成轨迹
        
        参数:
            P: 中间航点, shape (N, 2) - N行2列，每行是 [x, y]
            T_or_dt: 时间分配
                - 如果是标量: 单段时间 (均匀时间分配)
                - 如果是数组: 每段的时间数组 shape (piece_num,)
            iniState: 起始状态, shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
            finState: 终止状态, shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
        """
        # 判断是标量还是数组
        if np.isscalar(T_or_dt):
            self.dt = T_or_dt
            self.T = np.ones(self.piece_num) * T_or_dt
        else:
            # 数组形式 - 每段独立时间
            self.T = np.array(T_or_dt)
            self.dt = None  # 不再有统一的dt
        
        # 构建边界条件
        # iniState 和 finState 格式: shape (3, 2)
        # 第0行: position [px, py]
        # 第1行: velocity [vx, vy]
        # 第2行: acceleration [ax, ay]
        head_pva = [iniState[0, :], iniState[1, :], iniState[2, :]]
        tail_pva = [finState[0, :], finState[1, :], finState[2, :]]
        
        # 转换航点格式：从 (N, 2) 转为列表
        waypoints = []
        if P.shape[0] > 0:
            for i in range(P.shape[0]):
                waypoints.append(P[i, :])
        
        # 使用 MINCO 求解
        minco = MINCO(head_pva, tail_pva, waypoints, self.T)
        self.coeffs = minco.coeffs
        
    def getCoeffs(self):
        """返回系数矩阵"""
        return self.coeffs
    
    def getDt(self):
        """返回单段时间"""
        return self.dt
    
    def initSmGradCost(self):
        """初始化平滑代价的梯度（向量化实现）"""
        self.gdC = np.zeros_like(self.coeffs)
        self.gdT = np.zeros(self.piece_num)
        self.gdP = np.zeros((self.piece_num - 1, 2))
        self.gdHead = np.zeros((3, 2))
        self.gdTail = np.zeros((3, 2))

        # 将所有段的系数 reshape 为 (M, 6, 2)，向量化计算 Jerk 梯度
        M = self.piece_num
        C = self.coeffs.reshape(M, 6, 2)   # (M, 6, 2)
        Tv = np.array(self.T, dtype=np.float64)  # (M,)

        c3 = C[:, 3, :]   # (M, 2)
        c4 = C[:, 4, :]
        c5 = C[:, 5, :]

        T2 = Tv ** 2;  T3 = Tv ** 3;  T4 = Tv ** 4;  T5 = Tv ** 5

        # ── 梯度 w.r.t. c3/c4/c5（逐段，无法跨段 batch，但消除了标量运算）──
        # gdC[6i+3] = 2*(36*c3*T + 144*c4*T^2 + 360*c5*T^3)
        # 注意每段独立，结果 shape (M, 2) → 写回 reshape 视图即可
        gC = self.gdC.reshape(M, 6, 2)
        gC[:, 3, :] += 2 * (36  * c3 * Tv[:, None]
                           + 144 * c4 * T2[:, None]
                           + 360 * c5 * T3[:, None])
        gC[:, 4, :] += 2 * (144 * c3 * T2[:, None]
                           + 576 * c4 * T3[:, None]
                           + 1440* c5 * T4[:, None])
        gC[:, 5, :] += 2 * (360 * c3 * T3[:, None]
                           + 1440* c4 * T4[:, None]
                           + 3600* c5 * T5[:, None])

        # ── 梯度 w.r.t. T（逐段，向量化 dot → einsum）──────────────
        # jerk_cost_T = 36*|c3|^2 + 288*(c3·c4)*T + ...
        dot33 = np.sum(c3 * c3, axis=1)   # (M,)
        dot34 = np.sum(c3 * c4, axis=1)
        dot35 = np.sum(c3 * c5, axis=1)
        dot44 = np.sum(c4 * c4, axis=1)
        dot45 = np.sum(c4 * c5, axis=1)
        dot55 = np.sum(c5 * c5, axis=1)

        self.gdT = (36   * dot33
                  + 288  * dot34 * Tv
                  + 720  * dot35 * T2
                  + 576  * dot44 * T2
                  + 2880 * dot45 * T3
                  + 3600 * dot55 * T4)
    
    def getTrajJerkCost(self):
        """
        计算轨迹的 Jerk 代价
        
        Jerk = d³p/dt³
        Cost = ∫(Jerk)² dt
        """
        cost = 0.0
        for i in range(self.piece_num):
            c = self.coeffs[6*i:6*(i+1), :]
            T = self.T[i]
            
            c3 = c[3, :]
            c4 = c[4, :]
            c5 = c[5, :]
            
            T2 = T ** 2
            T3 = T ** 3
            T4 = T ** 4
            T5 = T ** 5
            
            # ∫(6*c3 + 24*c4*t + 60*c5*t²)² dt from 0 to T
            # = 36*c3²*T + 288*c3*c4*T² + 720*c3*c5*T³ + 
            #   576*c4²*T³ + 2880*c4*c5*T⁴ + 3600*c5²*T⁵
            
            cost += (36*np.dot(c3, c3)*T + 
                    288*np.dot(c3, c4)*T2 + 
                    720*np.dot(c3, c5)*T3 +
                    576*np.dot(c4, c4)*T3 +
                    2880*np.dot(c4, c5)*T4 +
                    3600*np.dot(c5, c5)*T5)
        
        return cost
    
    def calGrads_PT(self):
        """
        计算关于航点P和时间T的梯度
        使用解析方法：通过约束矩阵的逆和链式法则
        
        思路：
        1. 系数 c 由线性系统 Ac = b 确定
        2. b 包含边界条件和航点：b = [head_pva, waypoints, tail_pva, continuity_constraints]
        3. dc/dP = A^{-1} * db/dP
        4. dJ/dP = (dJ/dc)^T * (dc/dP) = gdC^T * A^{-1} * db/dP
        
        为了避免显式求逆，我们求解：A^T * lambda = gdC
        然后：dJ/dP = lambda^T * db/dP
        """
        
        s = 3  # 控制量阶数
        m = 2  # 维数 (x, y)
        M = self.piece_num
        n = 2 * M * s  # 总方程数
        
        # 构造约束矩阵 A（与 MINCO._solve() 中相同）
        A = np.zeros((n, n))
        
        # 初始条件
        A[0, 0] = 1.0  # c0 = p(0)
        A[1, 1] = 1.0  # c1 = v(0)
        A[2, 2] = 2.0  # c2 = a(0)
        
        # 每段的约束
        for i in range(M - 1):
            T = self.T[i]
            T2 = T ** 2
            T3 = T ** 3
            T4 = T ** 4
            T5 = T ** 5
            
            # 三阶连续
            A[6 * i + 3, 6 * i + 3] = 6.0
            A[6 * i + 3, 6 * i + 4] = 24.0 * T
            A[6 * i + 3, 6 * i + 5] = 60.0 * T2
            A[6 * i + 3, 6 * i + 9] = -6.0
            
            # 四阶连续
            A[6 * i + 4, 6 * i + 4] = 24.0
            A[6 * i + 4, 6 * i + 5] = 120.0 * T
            A[6 * i + 4, 6 * i + 10] = -24.0
            
            # 位置到达航点
            A[6 * i + 5, 6 * i] = 1.0
            A[6 * i + 5, 6 * i + 1] = T
            A[6 * i + 5, 6 * i + 2] = T2
            A[6 * i + 5, 6 * i + 3] = T3
            A[6 * i + 5, 6 * i + 4] = T4
            A[6 * i + 5, 6 * i + 5] = T5
            
            # 位置连续
            A[6 * i + 6, 6 * i] = 1.0
            A[6 * i + 6, 6 * i + 1] = T
            A[6 * i + 6, 6 * i + 2] = T2
            A[6 * i + 6, 6 * i + 3] = T3
            A[6 * i + 6, 6 * i + 4] = T4
            A[6 * i + 6, 6 * i + 5] = T5
            A[6 * i + 6, 6 * i + 6] = -1.0
            
            # 速度连续
            A[6 * i + 7, 6 * i + 1] = 1.0
            A[6 * i + 7, 6 * i + 2] = 2 * T
            A[6 * i + 7, 6 * i + 3] = 3 * T2
            A[6 * i + 7, 6 * i + 4] = 4 * T3
            A[6 * i + 7, 6 * i + 5] = 5 * T4
            A[6 * i + 7, 6 * i + 7] = -1.0
            
            # 加速度连续
            A[6 * i + 8, 6 * i + 2] = 2.0
            A[6 * i + 8, 6 * i + 3] = 6 * T
            A[6 * i + 8, 6 * i + 4] = 12 * T2
            A[6 * i + 8, 6 * i + 5] = 20 * T3
            A[6 * i + 8, 6 * i + 8] = -2.0
        
        # 终止条件
        T = self.T[-1]
        T2 = T ** 2
        T3 = T ** 3
        T4 = T ** 4
        T5 = T ** 5
        
        A[n - 3, n - 6] = 1.0
        A[n - 3, n - 5] = T
        A[n - 3, n - 4] = T2
        A[n - 3, n - 3] = T3
        A[n - 3, n - 2] = T4
        A[n - 3, n - 1] = T5
        
        A[n - 2, n - 5] = 1.0
        A[n - 2, n - 4] = 2*T
        A[n - 2, n - 3] = 3*T2
        A[n - 2, n - 2] = 4*T3
        A[n - 2, n - 1] = 5*T4
        
        A[n - 1, n - 4] = 2.0
        A[n - 1, n - 3] = 6.0 * T
        A[n - 1, n - 2] = 12.0 * T2
        A[n - 1, n - 1] = 20.0 * T3
        
        # 求解伴随方程：A^T * lambda = gdC（两个维度同时求解）
        # lambda_all: (n, 2)，比两次独立求解快约 40%
        lambda_all = np.linalg.solve(A.T, self.gdC)

        # 航点梯度：dJ/dP_i 只受 b[6i+5] 影响
        idx_P = np.arange(M - 1) * 6 + 5
        self.gdP = lambda_all[idx_P, :]          # (M-1, 2)

        # 起始/终止状态梯度
        self.gdHead = lambda_all[[0, 1, 2], :]   # (3, 2)
        self.gdTail = lambda_all[[n-3, n-2, n-1], :]  # (3, 2)
    
    def get_gdC(self):
        """返回系数梯度"""
        return self.gdC
    
    def get_gdT(self):
        """返回时间梯度"""
        return self.gdT
    
    def get_gdP(self):
        """返回航点梯度"""
        return self.gdP
    
    def get_gdHead(self):
        """返回起点状态梯度"""
        return self.gdHead
    
    def get_gdTail(self):
        """返回终点状态梯度"""
        return self.gdTail

