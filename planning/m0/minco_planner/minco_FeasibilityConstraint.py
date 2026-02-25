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
        添加位置-速度-加速度约束的梯度和代价
        
        参数:
            coeffs: 多项式系数, shape (6*piece_num, 2)
            T: 每段时间, shape (piece_num,)
            piece_num: 段数
            wei_feas: 可行性权重
            
        返回:
            total_cost: 总约束代价
        """
        # ⚠️ 重要：每次都必须重置梯度，否则会累积
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num)  # 每段独立的时间梯度
        
        self.vel_cost = 0.0
        self.acc_cost = 0.0
        
        # 对每一段进行采样
        for i in range(piece_num):
            # 确定采样分辨率
            if i == 0 or i == piece_num - 1:
                K = self.destraj_resolution  # 起止段更密集
            else:
                K = self.traj_resolution
            
            # 获取这一段的系数
            c = coeffs[6*i:6*(i+1), :]  # shape: (6, 2)
            T_i = T[i]
            step = T_i / K
            
            # 对这一段进行采样
            for j in range(K + 1):
                # 梯形积分权重
                omg = 0.5 if (j == 0 or j == K) else 1.0
                
                # 计算当前时间点
                s1 = step * j
                s2 = s1 * s1
                s3 = s2 * s1
                s4 = s2 * s2
                s5 = s4 * s1
                
                # 归一化时间（用于梯度计算）
                alpha = j / K
                
                # 基函数和导数
                beta0 = np.array([1.0, s1, s2, s3, s4, s5])
                beta1 = np.array([0.0, 1.0, 2.0*s1, 3.0*s2, 4.0*s3, 5.0*s4])
                beta2 = np.array([0.0, 0.0, 2.0, 6.0*s1, 12.0*s2, 20.0*s3])
                beta3 = np.array([0.0, 0.0, 0.0, 6.0, 24.0*s1, 60.0*s2])
                
                # 计算速度、加速度
                dsigma = c.T @ beta1
                ddsigma = c.T @ beta2
                dddsigma = c.T @ beta3
                
                # 速度约束 - 按照论文公式(16): G_v(σ̇) = σ̇ᵀσ̇ - v_m²
                vel_squared = np.dot(dsigma, dsigma)
                violaVel = vel_squared - self.max_vel ** 2
                
                if violaVel > 0.0:
                    violaVelPena, violaVelPenaD = self.positiveSmoothedL1(violaVel)
                    
                    # 梯度 - 按照论文公式(17): ∂G_v/∂σ̇ = 2σ̇
                    gradViolaVc = 2.0 * np.outer(beta1, dsigma)
                    gradViolaVt = 2.0 * alpha * np.dot(dsigma, ddsigma)
                    
                    # 累加梯度
                    self.gdC[6*i:6*(i+1), :] += omg * step * wei_feas * violaVelPenaD * gradViolaVc
                    self.gdT[i] += omg * wei_feas * (violaVelPenaD * gradViolaVt * step + violaVelPena / K)
                    
                    # 累加代价
                    self.vel_cost += omg * step * wei_feas * violaVelPena
                
                # 加速度约束 - 同样使用平方形式
                acc_squared = np.dot(ddsigma, ddsigma)
                violaAcc = acc_squared - self.max_acc ** 2
                
                if violaAcc > 0.0:
                    violaAccPena, violaAccPenaD = self.positiveSmoothedL1(violaAcc)
                    
                    # 梯度: ∂G_a/∂σ̈ = 2σ̈
                    gradViolaAc = 2.0 * np.outer(beta2, ddsigma)
                    gradViolaAt = 2.0 * alpha * np.dot(ddsigma, dddsigma)
                    
                    # 累加梯度
                    self.gdC[6*i:6*(i+1), :] += omg * step * wei_feas * violaAccPenaD * gradViolaAc
                    self.gdT[i] += omg * wei_feas * (violaAccPenaD * gradViolaAt * step + violaAccPena / K)
                    
                    # 累加代价
                    self.acc_cost += omg * step * wei_feas * violaAccPena
        
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
