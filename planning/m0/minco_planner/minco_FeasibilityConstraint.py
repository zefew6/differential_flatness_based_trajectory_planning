import numpy as np

class FeasibilityConstraint:
    """
    动力学可行性约束类
    用于计算速度、加速度等约束的惩罚代价和梯度
    """
    
    def __init__(self, max_vel, max_acc, traj_resolution=16, destraj_resolution=32):
        """
        初始化约束
        
        参数:
            max_vel: 最大速度 (m/s)
            max_acc: 最大加速度 (m/s²)
            traj_resolution: 轨迹段采样分辨率
            destraj_resolution: 起止段采样分辨率
        """
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.traj_resolution = traj_resolution
        self.destraj_resolution = destraj_resolution
        
        # 用于存储梯度
        self.gdC = None  # 关于系数的梯度
        self.gdT = 0.0   # 关于时间的梯度
        
        # 代价统计
        self.vel_cost = 0.0
        self.acc_cost = 0.0
        self.total_cost = 0.0
    
    def reset(self):
        """重置梯度和代价"""
        self.gdC = None
        self.gdT = 0.0
        self.vel_cost = 0.0
        self.acc_cost = 0.0
        self.total_cost = 0.0
    
    @staticmethod
    def positiveSmoothedL1(x):
        """
        平滑的 L1 惩罚函数 (Positive Part Only)
        改进版：在大违反时使用平方根，避免梯度爆炸
        
        参数:
            x: 约束违反量
            
        返回:
            f: 惩罚值
            df: 惩罚梯度
        """
        pe = 1e-4
        half = 0.5 * pe
        f3c = 1.0 / (pe * pe)
        f4c = -0.5 * f3c / pe
        d2c = 3.0 * f3c
        d3c = 4.0 * f4c
        
        # 大违反阈值（超过此值使用平方根，更早启用以减小梯度）
        large_threshold = 0.1  # 降低阈值，更早使用平方根
        
        if x < pe:
            # 平滑区域：三次函数
            f = (f4c * x + f3c) * x * x * x
            df = (d3c * x + d2c) * x * x
        elif x < large_threshold:
            # 中等违反：线性
            f = x - half
            df = 1.0
        else:
            # 大违反：使用平方根缓解梯度爆炸
            # f = large_threshold + sqrt(x - large_threshold)
            delta = x - large_threshold
            sqrt_delta = np.sqrt(delta)
            f = large_threshold - half + sqrt_delta
            df = 0.5 / sqrt_delta  # d/dx[sqrt(x)] = 1/(2*sqrt(x))
        
        return f, df
    
    def addPVAGradCost(self, coeffs, T, piece_num, wei_feas):
        """
        添加位置-速度-加速度约束的梯度和代价（向量化实现）

        参数:
            coeffs: 多项式系数, shape (6*piece_num, 2)
            T: 每段时间, shape (piece_num,)
            piece_num: 段数
            wei_feas: 可行性权重

        返回:
            total_cost: 总约束代价
        """
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num)
        self.vel_cost = 0.0
        self.acc_cost = 0.0

        # positiveSmoothedL1 常数（与标量版本保持完全一致）
        pe   = 1e-4
        half = 0.5 * pe
        f3c  = 1.0 / (pe * pe)
        f4c  = -0.5 * f3c / pe
        d2c  = 3.0 * f3c
        d3c  = 4.0 * f4c
        lth  = 0.1          # large_threshold

        for i in range(piece_num):
            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution
            c   = coeffs[6 * i : 6 * (i + 1), :]   # (6, 2)
            T_i = float(T[i])
            step = T_i / K

            # ── 向量化 beta ────────────────────────────────────────────
            js  = np.arange(K + 1, dtype=np.float64)
            s1  = step * js
            s2  = s1 * s1;  s3 = s2 * s1;  s4 = s2 * s2;  s5 = s4 * s1

            beta1 = np.stack([np.zeros(K+1), np.ones(K+1),
                               2.0*s1, 3.0*s2, 4.0*s3, 5.0*s4], axis=1)          # (K+1,6)
            beta2 = np.stack([np.zeros(K+1), np.zeros(K+1),
                               2.0*np.ones(K+1), 6.0*s1, 12.0*s2, 20.0*s3], axis=1)
            beta3 = np.stack([np.zeros(K+1), np.zeros(K+1), np.zeros(K+1),
                               6.0*np.ones(K+1), 24.0*s1, 60.0*s2], axis=1)

            vel_all  = beta1 @ c   # (K+1, 2)
            acc_all  = beta2 @ c
            jerk_all = beta3 @ c

            omg_vec   = np.ones(K + 1); omg_vec[0] = 0.5; omg_vec[K] = 0.5
            alpha_vec = js / K

            # ── 向量化 positiveSmoothedL1(x) ─────────────────────────
            def _psL1_vec(x):
                """向量化版 positiveSmoothedL1，x 为 (N,) 数组"""
                # 各分支的 f / df（无论 x 正负都计算，最后 mask）
                # 分支1: x < pe
                f1  = (f4c * x + f3c) * x * x * x
                df1 = (d3c * x + d2c) * x * x
                # 分支2: pe <= x < lth
                f2  = x - half
                df2 = np.ones_like(x)
                # 分支3: x >= lth
                delta    = np.maximum(x - lth, 0.0)
                sqrt_d   = np.sqrt(np.where(delta > 0, delta, 1e-32))
                f3  = lth - half + sqrt_d
                df3 = 0.5 / sqrt_d
                # 按分支选择
                f  = np.where(x < pe,  f1,  np.where(x < lth, f2,  f3))
                df = np.where(x < pe,  df1, np.where(x < lth, df2, df3))
                return f, df

            # ── 速度约束 ────────────────────────────────────────────
            vel_sq  = np.sum(vel_all ** 2, axis=1)        # (K+1,)
            violaV  = vel_sq - self.max_vel ** 2
            activeV = violaV > 0.0

            if np.any(activeV):
                fV, dfV = _psL1_vec(violaV)
                fV  = np.where(activeV, fV,  0.0)
                dfV = np.where(activeV, dfV, 0.0)

                wfV  = omg_vec * step * wei_feas             # (K+1,)
                self.vel_cost += float(np.dot(wfV, fV))

                # gdC += 2 * beta1.T @ diag(wfV * dfV) @ vel_all
                wdV = wfV * dfV * 2.0                         # (K+1,)
                self.gdC[6*i : 6*(i+1), :] += beta1.T @ (wdV[:, None] * vel_all)

                # gdT: sum omg*(dfV * 2*alpha*dot(vel,acc)*step + fV/K) * wei_feas
                cross_va = np.sum(vel_all * acc_all, axis=1)
                self.gdT[i] += float(np.dot(omg_vec,
                    wei_feas * (dfV * 2.0 * alpha_vec * cross_va * step + fV / K)))

            # ── 加速度约束 ───────────────────────────────────────────
            acc_sq  = np.sum(acc_all ** 2, axis=1)
            violaA  = acc_sq - self.max_acc ** 2
            activeA = violaA > 0.0

            if np.any(activeA):
                fA, dfA = _psL1_vec(violaA)
                fA  = np.where(activeA, fA,  0.0)
                dfA = np.where(activeA, dfA, 0.0)

                wfA  = omg_vec * step * wei_feas
                self.acc_cost += float(np.dot(wfA, fA))

                wdA = wfA * dfA * 2.0
                self.gdC[6*i : 6*(i+1), :] += beta2.T @ (wdA[:, None] * acc_all)

                cross_aj = np.sum(acc_all * jerk_all, axis=1)
                self.gdT[i] += float(np.dot(omg_vec,
                    wei_feas * (dfA * 2.0 * alpha_vec * cross_aj * step + fA / K)))

        self.total_cost = self.vel_cost + self.acc_cost
        return self.total_cost
    
    def get_gdC(self):
        """返回关于系数的梯度"""
        return self.gdC
    
    def get_gdT(self):
        """返回关于时间的梯度"""
        return self.gdT
    
    def get_vel_cost(self):
        """返回速度约束代价"""
        return self.vel_cost
    
    def get_acc_cost(self):
        """返回加速度约束代价"""
        return self.acc_cost
