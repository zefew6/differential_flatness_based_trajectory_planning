import numpy as np
from scipy.optimize import minimize
from .minco_MinJerkOpt import MinJerkOpt
from .minco_FeasibilityConstraint import FeasibilityConstraint
from .minco_obstacle import ObstacleConstraint
class PolyTrajOptimizer:
    """
    多项式轨迹优化器
    基于 L-BFGS 优化算法，优化航点位置和时间分配
    
    Cost Function:
        Total Cost = Smoothness Cost + Time Cost + Penalty Cost
        
        1. Smoothness Cost: 轨迹的 Jerk 平方积分
        2. Time Cost: 总时间 * wei_time
        3. Penalty Cost: 约束违反的惩罚（障碍物、动力学等）
    """
    
    def __init__(self):
        """初始化优化器"""
        # 优化参数 - 平衡收敛性和约束满足
        self.wei_time = 10.0        # 时间权重（低→允许时间延长）
        self.wei_feas = 500.0      # 可行性权重（适中，避免梯度爆炸）
        # 静态障碍物权重：对齐 ST-opt-tools 里常用的 rho_collision=1e5 量级
        self.wei_obs = 1e4
        self.wei_surround = 5000.0 # 动态障碍物权重
        # 障碍安全距离（米）：ESDF 小于此值的点会被惩罚。
        # 需要大于地图中自由点的最小 ESDF（典型值 0.02~0.05m），
        # 通常设为 robot_radius（不含 margin）以在膨胀边界内再留一圈安全余量。
        self.obs_safe_threshold = 0.05
        
        self.mini_T = 0.005         # 最小段时间（Dftpav: 0.1）
        
        # 动力学约束参数 - 参考 Dftpav
        self.max_vel = 1.5        # 最大速度 (m/s) - 提高限制
        self.max_acc = 1.0        # 最大加速度 (m/s²) - 提高限制
        
        # L-BFGS 优化器参数
        self.lbfgs_memsize = 256   # 内存大小（Dftpav: 256）
        self.lbfgs_past = 3        # 过去迭代数（Dftpav: 3）
        self.lbfgs_delta = 1e-4    # 收敛判据（函数值相对变化）
        self.lbfgs_g_epsilon = 1e-4  # 梯度收敛判据（投影梯度范数）
        self.lbfgs_max_iterations = 200  # 最大迭代次数
        
        # 轨迹分辨率
        self.traj_resolution = 16      # 轨迹段分辨率（Dftpav: 16）
        self.destraj_resolution = 32   # 起止段分辨率（Dftpav: 32）
        
        # 轨迹数据
        self.trajnum = 0
        self.piece_num_container = []
        self.jerkOpt_container = []
        self.feasConstr_container = []  # 可行性约束容器
        self.iniState_container = []
        self.finState_container = []
        
        # 优化变量
        self.variable_num = 0
        self.iter_num = 0

        # debug / diagnostics
        self.debug_print_every = 10  # print cost breakdown every N iterations (0 disables)
        self._last_cost_breakdown = None

        # --- obstacle related ---
        # 外部注入的地图对象（推荐 gridmap_2d_v2.GridMap2D），需要提供 get_distance_and_gradient(pos)
        self.grid_map = None
        # 静态障碍物约束模块
        self.obsConstr_container = []

    def setGridMap(self, grid_map):
        """设置用于避障项的 SDF/ESDF 地图。

        grid_map 需要提供：
            get_distance_and_gradient(pos)->(dist, grad)
        同时也应提供 coor_to_index / is_valid_index / is_occupied_index
        以支持 preprocessPath 的可见性剪枝。
        """
        self.grid_map = grid_map

    # ------------------------------------------------------------------
    # 路径预处理：可见性剪枝  （参考 ST-opt-tools AStar::optimizePath）
    # ------------------------------------------------------------------

    def _check_line_collision(self, p1, p2) -> bool:
        """Bresenham 直线检测：p1→p2 之间是否经过占据格。有碰撞返回 True。"""
        gm = self.grid_map
        r1, c1 = gm.coor_to_index(p1)
        r2, c2 = gm.coor_to_index(p2)
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        sr = 1 if r2 > r1 else -1
        sc = 1 if c2 > c1 else -1
        err = dr - dc
        r, c = r1, c1
        while True:
            idx = (r, c)
            if gm.is_valid_index(idx) and gm.is_occupied_index(idx):
                return True
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return False

    def preprocessPath(self, path) -> np.ndarray:
        """对 A* 原始路径做可见性剪枝，返回仅保留转角点的精简路径（含首尾）。

        参数
        ----
        path : np.ndarray, shape (N, 2)
            A* 输出的稠密路径（每步一个栅格）。

        返回
        ----
        pruned : np.ndarray, shape (M, 2)
            精简后的路径，M << N。内部路点 pruned[1:-1] 即为 MINCO 的 inner_pts。
        """
        if self.grid_map is None:
            raise RuntimeError("Call setGridMap() before preprocessPath().")
        path = np.asarray(path)
        if len(path) <= 2:
            return path
        pruned = [path[0]]
        prev = path[0]
        i = 1
        while i < len(path) - 1:
            # 若能从 prev 直线无碰到 path[i+1]，则跳过 path[i]
            if not self._check_line_collision(prev, path[i + 1]):
                i += 1
            else:
                pruned.append(path[i])
                prev = path[i]
                i += 1
        pruned.append(path[-1])
        return np.array(pruned)

    def resamplePath(self, path, max_seg_len: float = 1.5,
                     dense_path: np.ndarray = None) -> np.ndarray:
        """对路径按固定弧长重采样，使每段长度不超过 max_seg_len。

        目的：剪枝后路点数过少（段过长），MINCO 多项式会产生与 A* 无关的大弧。
        重采样后给 MINCO 足够的形状约束，让优化结果贴近原始路径。

        参数
        ----
        path : np.ndarray, shape (N, 2)
            待重采样路径（通常是 preprocessPath 剪枝后的稀疏路径）。
        max_seg_len : float
            最大段长（米），默认 1.5m
        dense_path : np.ndarray, shape (K, 2), optional
            原始稠密路径（如 A* 输出）。若提供，则在稀疏段之间沿稠密路径
            抽取中间路点，**避免直线插值穿越障碍边界** 而引发蛇形振荡。
            若不提供，则退化为直线插值。

        返回
        ----
        resampled : np.ndarray, shape (M, 2)，M >= N
        """
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return path

        # ── 若提供稠密路径，在稀疏段间沿稠密路径抽取中间点 ──────────────────
        if dense_path is not None:
            dense_path = np.asarray(dense_path, dtype=float)
            result = [path[0]]
            for i in range(len(path) - 1):
                p0, p1 = path[i], path[i + 1]
                seg_len = np.linalg.norm(p1 - p0)
                if seg_len > max_seg_len:
                    # 在 dense_path 中找到属于本段的区间（最近邻索引）
                    d0 = np.linalg.norm(dense_path - p0, axis=1)
                    d1 = np.linalg.norm(dense_path - p1, axis=1)
                    idx0 = int(np.argmin(d0))
                    idx1 = int(np.argmin(d1))
                    if idx0 > idx1:
                        idx0, idx1 = idx1, idx0
                    # 从稠密路径的对应区间中均匀抽取中间点
                    segment = dense_path[idx0:idx1 + 1]
                    if len(segment) > 2:
                        n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
                        # 按弧长均匀抽取 n_insert 个中间点
                        arc = np.cumsum(
                            np.r_[0, np.linalg.norm(np.diff(segment, axis=0), axis=1)]
                        )
                        total_arc = arc[-1]
                        for k in range(1, n_insert + 1):
                            s = total_arc * k / (n_insert + 1)
                            j = np.searchsorted(arc, s, side='right') - 1
                            j = min(j, len(segment) - 2)
                            alpha = (s - arc[j]) / (arc[j + 1] - arc[j] + 1e-12)
                            pt = segment[j] + alpha * (segment[j + 1] - segment[j])
                            result.append(pt)
                result.append(p1)
            return np.array(result)

        # ── 退化模式：直线插值（无稠密路径时使用）────────────────────────────
        result = [path[0]]
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            seg_len = np.linalg.norm(p1 - p0)
            if seg_len > max_seg_len:
                n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
                for k in range(1, n_insert + 1):
                    t = k / (n_insert + 1)
                    result.append(p0 + t * (p1 - p0))
            result.append(p1)
        return np.array(result)

    # ------------------------------------------------------------------
    # 时间分配：梯形速度曲线  （参考 ST-opt-tools AStar::evaluateDuration）
    # ------------------------------------------------------------------

    def allocateTime(self, waypoints) -> np.ndarray:
        """按梯形速度曲线为各段分配飞行时间。

        参数
        ----
        waypoints : np.ndarray, shape (K, 2)
            包含首尾的完整路点序列（head + inner_pts + tail）。

        返回
        ----
        durations : np.ndarray, shape (K-1,)
            每段的时间分配（秒）。
        """
        waypoints = np.asarray(waypoints)
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        durations = np.array([self._trapezoid_duration(d) for d in dists])
        # 保证不低于最小段时间
        return np.maximum(durations, self.mini_T)

    def _trapezoid_duration(self, length: float) -> float:
        """单段梯形速度曲线时间（起止速度均为 0）。"""
        v = self.max_vel
        a = self.max_acc
        if length < 1e-6:
            return self.mini_T
        critical_len = v * v / a          # 刚好能加速到 max_vel 再减速所需路程
        if length >= critical_len:
            return 2.0 * v / a + (length - critical_len) / v
        else:
            v_peak = np.sqrt(a * length)  # 三角形速度曲线
            return 2.0 * v_peak / a

    def setParam(self, wei_time=None, wei_feas=None, mini_T=None, 
                 lbfgs_memsize=None, lbfgs_delta=None, lbfgs_max_iterations=None,
                 wei_obs=None, obs_safe_threshold=None):
        """
        设置优化参数
        
        参数:
            wei_time: 时间权重
            wei_feas: 可行性权重
            mini_T: 最小段时间
            lbfgs_memsize: L-BFGS 内存大小
            lbfgs_delta: L-BFGS 收敛判据
            lbfgs_max_iterations: 最大迭代次数
        """
        if wei_time is not None:
            self.wei_time = wei_time
        if wei_feas is not None:
            self.wei_feas = wei_feas
        if mini_T is not None:
            self.mini_T = mini_T
        if lbfgs_memsize is not None:
            self.lbfgs_memsize = lbfgs_memsize
        if lbfgs_delta is not None:
            self.lbfgs_delta = lbfgs_delta
        if lbfgs_max_iterations is not None:
            self.lbfgs_max_iterations = lbfgs_max_iterations

        if wei_obs is not None:
            self.wei_obs = float(wei_obs)
            # propagate to already-created containers
            for obsConstr in getattr(self, 'obsConstr_container', []):
                obsConstr.wei_obs = float(wei_obs)

        if obs_safe_threshold is not None:
            self.obs_safe_threshold = float(obs_safe_threshold)  # 存到 self，供创建容器时使用
            # propagate to already-created containers
            for obsConstr in getattr(self, 'obsConstr_container', []):
                obsConstr.safe_threshold = float(obs_safe_threshold)

    def setDebugPrintEvery(self, n: int):
        """Set how often to print per-iteration cost diagnostics.

        n <= 0 disables printing.
        """
        self.debug_print_every = int(n)
    
    def OptimizeTrajectory(self, iniStates, finStates, initInnerPts, initTs,
                           initSegTs=None):
        """优化轨迹
        
        参数:
            iniStates: 起始状态列表，每个元素 shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
            finStates: 终止状态列表，每个元素 shape (3, 2)
            initInnerPts: 初始内部航点列表，每个元素 shape (N, 2) - N行2列
            initTs: 初始时间分配，shape (trajnum,)，每条轨迹的总时间
            initSegTs: 可选，按段时间列表，list of ndarray shape (piece_num,)
                       若提供则用于初始化各段时间，忽略 initTs 的等分逻辑
        
        返回:
            success: 是否优化成功
            final_cost: 最终代价值
        """
        self.trajnum = len(initInnerPts)
        self.iniState_container = []
        self.finState_container = []
        
        # 裁剪初始/终止状态以满足硬约束
        for i in range(self.trajnum):
            clipped_ini = self.clipStateConstraints(iniStates[i], self.max_vel, self.max_acc)
            clipped_fin = self.clipStateConstraints(finStates[i], self.max_vel, self.max_acc)
            self.iniState_container.append(clipped_ini)
            self.finState_container.append(clipped_fin)
        
        # 检查输入
        if len(initTs) != self.trajnum:
            print("Error: initTs size != trajnum")
            return False, np.inf
        
        if np.min(initTs) < self.mini_T:
            print("Error: mini segment T < mini_T!")
            return False, np.inf
        
        # 初始化优化器容器
        self.jerkOpt_container = []
        self.feasConstr_container = []  # 约束容器
        self.obsConstr_container = []   # 避障约束容器
        self.piece_num_container = []
        self.variable_num = 0
        
        for i in range(self.trajnum):
            if initInnerPts[i].shape[0] == 0:  # 检查行数（航点数量）
                print("Error: No inner points!")
                return False, np.inf
            
            piece_num = initInnerPts[i].shape[0] + 1  # 行数 + 1
            self.piece_num_container.append(piece_num)
            
            # 创建 MinJerkOpt
            jerkOpt = MinJerkOpt(piece_num)
            self.jerkOpt_container.append(jerkOpt)
            
            # 创建 FeasibilityConstraint
            feasConstr = FeasibilityConstraint(
                self.max_vel, 
                self.max_acc,
                self.traj_resolution,
                self.destraj_resolution
            )
            self.feasConstr_container.append(feasConstr)

            # 创建 ObstacleConstraint（避障项会在 cost callback 里按需启用）
            obsConstr = ObstacleConstraint(
                safe_threshold=self.obs_safe_threshold,  # 使用 setParam 设置的值
                wei_obs=self.wei_obs,
                traj_resolution=self.traj_resolution,
                destraj_resolution=self.destraj_resolution,
            )
            self.obsConstr_container.append(obsConstr)
            
            # 变量数：航点数 (2 * (piece_num - 1))
            self.variable_num += 2 * (piece_num - 1)
            
            # 变量数：每段独立的时间 (piece_num 个)
            self.variable_num += piece_num
        
        # 构建初始优化变量
        x0 = np.zeros(self.variable_num)
        offset = 0
        
        # 航点
        for i in range(self.trajnum):
            pts = initInnerPts[i]
            x0[offset:offset + pts.size] = pts.flatten()
            offset += pts.size
        
        # 时间（转换为虚拟时间）- 每段独立
        for i in range(self.trajnum):
            piece_num = self.piece_num_container[i]
            if initSegTs is not None and i < len(initSegTs):
                # 使用外部提供的按段时间（来自梯形速度分配）
                segment_times = np.array(initSegTs[i], dtype=float)
                if len(segment_times) != piece_num:
                    # 长度不匹配时退化为等分
                    dt_init = initTs[i] / piece_num
                    segment_times = np.full(piece_num, dt_init)
            else:
                # 将总时间平均分配给各段作为初始值
                dt_init = initTs[i] / piece_num
                segment_times = np.full(piece_num, dt_init)
            segment_times = np.maximum(segment_times, self.mini_T)

            VT = np.zeros(piece_num)
            self.RealT2VirtualT(segment_times, VT)
            x0[offset:offset + piece_num] = VT
            offset += piece_num
        
        # L-BFGS 优化（使用 Dftpav 的参数）
        self.iter_num = 0
        
        result = minimize(
            fun=self._cost_function_callback,
            x0=x0,
            method='L-BFGS-B',
            jac=True,  # 我们会返回梯度
            options={
                'maxiter': self.lbfgs_max_iterations,  # 12000（Dftpav）
                'ftol': self.lbfgs_delta,              # 1e-4（Dftpav）
                'gtol': self.lbfgs_g_epsilon,          # 1e-16（Dftpav）
                'maxls': 20,                            # 最大线搜索次数
                'disp': False                          # 不显示scipy优化器的详细信息
            }
        )
        
        success = result.success
        final_cost = result.fun
        
        # print(f"\nOptimization finished!")
        # print(f"Success: {success}")
        # print(f"Message: {result.message}")
        # print(f"Iterations: {result.nit}")
        # print(f"Function evaluations: {result.nfev}")
        # print(f"Final cost: {final_cost:.6f}")
        # print(f"Final gradient norm: {np.linalg.norm(result.jac):.6e}")
        # print(f"Iterations: {self.iter_num}")
        print(f"[MINCO] nit={result.nit}  nfev={result.nfev}  callback_calls={self.iter_num}  msg={result.message}")
        
        return success, final_cost
    
    def _cost_function_callback(self, x):
        """
        代价函数回调
        
        参数:
            x: 优化变量 [P0, P1, ..., t0, t1, ...]
        
        返回:
            cost: 总代价
            grad: 梯度
        """
        total_smcost = 0.0
        total_timecost = 0.0
        total_penalty = 0.0  # 约束惩罚代价

        # breakdown
        total_feas_cost = 0.0
        total_obs_cost = 0.0
        
        # 解析优化变量
        offset = 0
        P_container = []
        gradP_container = []
        
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            size = 2 * (piece_num - 1)
            
            # 重塑为 (N, 2) 格式：N行2列
            P = x[offset:offset + size].reshape(piece_num - 1, 2)
            gradP = np.zeros((piece_num - 1, 2))
            
            P_container.append(P)
            gradP_container.append(gradP)
            offset += size
        
        # 时间 - 每段独立的时间
        T_container = []
        gradT_container = []
        
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            
            # 提取每段的虚拟时间
            t_seg = x[offset:offset + piece_num]
            gradt_seg = np.zeros(piece_num)
            
            # 虚拟时间转真实时间
            T_seg = np.zeros(piece_num)
            self.VirtualT2RealT(t_seg, T_seg)
            
            T_container.append(T_seg)
            gradT_container.append(gradt_seg)
            offset += piece_num
        
    # 计算每条轨迹的代价
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            T_seg = T_container[trajid]
            
            # 检查所有段的时间合理性
            if np.any(T_seg < self.mini_T) or np.any(T_seg > 1000.0) or not np.all(np.isfinite(T_seg)):
                # 时间不合理，返回惩罚代价
                return np.inf, np.zeros_like(x)
            
            # 生成轨迹 - 使用每段独立的时间
            self.jerkOpt_container[trajid].generate(
                P_container[trajid],
                T_seg,  # 传入时间数组而非标量
                self.iniState_container[trajid],
                self.finState_container[trajid]
            )
            
            # 计算平滑代价
            self.jerkOpt_container[trajid].initSmGradCost()
            smoo_cost = self.jerkOpt_container[trajid].getTrajJerkCost()
            
            # 检查Jerk代价是否合理
            if not np.isfinite(smoo_cost) or smoo_cost > 1e8:
                # Jerk代价过大或无穷，返回惩罚
                return 1e10, np.zeros_like(x)
            
            total_smcost += smoo_cost
            
            # 计算动力学约束惩罚代价 (使用 FeasibilityConstraint)
            feasConstr = self.feasConstr_container[trajid]
            feas_cost = feasConstr.addPVAGradCost(
                self.jerkOpt_container[trajid].coeffs,
                self.jerkOpt_container[trajid].T,
                self.piece_num_container[trajid],
                self.wei_feas
            )
            total_penalty += feas_cost
            total_feas_cost += feas_cost
            
            # ⚠️ 重要：在 calGrads_PT() 之前累加约束梯度到 gdC
            # 因为 calGrads_PT() 会用 gdC 计算 gdP (通过伴随方法)
            self.jerkOpt_container[trajid].gdC += feasConstr.get_gdC()
            self.jerkOpt_container[trajid].gdT += feasConstr.get_gdT()

            # 计算避障惩罚代价（如果提供了地图）
            if self.grid_map is not None:
                obsConstr = self.obsConstr_container[trajid]
                obs_cost = obsConstr.addObstacleGradCost(
                    self.jerkOpt_container[trajid].coeffs,
                    self.jerkOpt_container[trajid].T,
                    self.piece_num_container[trajid],
                    self.grid_map,
                )
                total_penalty += obs_cost
                total_obs_cost += obs_cost

                # 累加到 gdC/gdT，等待 calGrads_PT() 统一回传到 P/T
                self.jerkOpt_container[trajid].gdC += obsConstr.get_gdC()
                self.jerkOpt_container[trajid].gdT += obsConstr.get_gdT()
        
        # 计算梯度
        offset_P_end = sum([2*(self.piece_num_container[i]-1) for i in range(self.trajnum)])
        
        for trajid in range(self.trajnum):
            # 计算 Jerk 梯度 (此时 gdC 已经包含了约束梯度)
            self.jerkOpt_container[trajid].calGrads_PT()
            
            # 航点梯度 (已经通过伴随方法包含了约束的影响)
            gradP_container[trajid][:] = self.jerkOpt_container[trajid].get_gdP()
            
            # 时间梯度 (包含 Jerk + Constraint) - 每段独立
            piece_num = self.piece_num_container[trajid]
            gdRT_vec = self.jerkOpt_container[trajid].get_gdT()  # 每段的时间梯度向量
            
            # 从优化变量中提取这个轨迹的时间部分
            offset_t = offset_P_end + sum([self.piece_num_container[i] for i in range(trajid)])
            t_seg = x[offset_t:offset_t + piece_num]
            T_seg = T_container[trajid]
            
            # 每段时间的虚拟梯度和时间代价
            for seg_id in range(piece_num):
                gradt_seg, time_cost_seg = self.VirtualTGradCost(
                    np.array([T_seg[seg_id]]), 
                    np.array([t_seg[seg_id]]), 
                    gdRT_vec[seg_id]
                )
                gradT_container[trajid][seg_id] = gradt_seg[0]
                total_timecost += time_cost_seg
        
        # 构建总梯度
        grad = np.zeros_like(x)
        offset = 0
        
        for trajid in range(self.trajnum):
            size = gradP_container[trajid].size
            grad[offset:offset + size] = gradP_container[trajid].flatten()
            offset += size

        # 时间梯度 - 每段独立
        for trajid in range(self.trajnum):
            piece_num = self.piece_num_container[trajid]
            grad[offset:offset + piece_num] = gradT_container[trajid]
            offset += piece_num

        # 梯度裁剪：防止数值爆炸
        max_grad_norm = 1e4  # 最大梯度范数
        grad_norm = np.linalg.norm(grad)
        grad_norm_scalar = float(grad_norm)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
        
        # 总代价 = 平滑代价 + 时间代价 + 惩罚代价
        # 注意：total_timecost 可能是 numpy array（由 VirtualTGradCost 返回），这里统一转 float
        total_smcost_f = float(total_smcost)
        total_timecost_f = float(np.sum(total_timecost))
        total_penalty_f = float(total_penalty)
        total_feas_cost_f = float(total_feas_cost)
        total_obs_cost_f = float(total_obs_cost)
        total_cost = float(total_smcost_f + total_timecost_f + total_penalty_f)

        # cache breakdown for external inspection (e.g., after optimize)
        self._last_cost_breakdown = {
            'iter': int(self.iter_num),
            'smooth': total_smcost_f,
            'time': total_timecost_f,
            'penalty': total_penalty_f,
            'feas': total_feas_cost_f,
            'obs': total_obs_cost_f,
            'total': float(total_cost),
            'grad_norm_raw': float(grad_norm_scalar),
        }

        # optional per-iteration print
        self.iter_num += 1
        if self.debug_print_every > 0 and (self.iter_num % self.debug_print_every == 0 or self.iter_num == 1):
            msg = (
                f"[MINCO opt] iter={self.iter_num:4d} "
                f"total={total_cost:.6e} "
                f"smooth={total_smcost_f:.3e} time={total_timecost_f:.3e} "
                f"penalty={total_penalty_f:.3e} (feas={total_feas_cost_f:.3e}, obs={total_obs_cost_f:.3e}) "
                f"|grad|={grad_norm_scalar:.3e}"
            )

            # try to report minimum SDF distance along current trajectory (diagnostic only)
            # 直接读 ObstacleConstraint 在本轮 addObstacleGradCost 中顺手记录的 min_dist，
            # 避免对轨迹再做一次完整重采样（之前是 200+ 次额外 ESDF 查询）。
            if self.grid_map is not None and len(self.obsConstr_container) > 0:
                try:
                    min_d = min(oc.min_dist for oc in self.obsConstr_container)
                    if np.isfinite(min_d):
                        msg += f" min_sdf={min_d:.3f}"
                except Exception:
                    pass

            print(msg)

        return total_cost, grad
    
    def RealT2VirtualT(self, RT, VT):
        """真实时间转虚拟时间（无约束优化）"""
        for i in range(len(RT)):
            if RT[i] > 1.0 + self.mini_T:
                VT[i] = np.sqrt(2.0 * RT[i] - 1.0 - 2 * self.mini_T) - 1.0
            else:
                VT[i] = 1.0 - np.sqrt(2.0 / (RT[i] - self.mini_T) - 1.0)
    
    def VirtualT2RealT(self, VT, RT):
        """虚拟时间转真实时间"""
        for i in range(len(VT)):
            if VT[i] > 0.0:
                RT[i] = (0.5 * VT[i] + 1.0) * VT[i] + 1.0 + self.mini_T
            else:
                RT[i] = 1.0 / ((0.5 * VT[i] - 1.0) * VT[i] + 1.0) + self.mini_T
    
    def VirtualTGradCost(self, RT, VT, gdRT):
        """计算虚拟时间的梯度和时间代价
        
        参数:
            RT: 真实时间
            VT: 虚拟时间
            gdRT: 关于真实时间的梯度
            
        返回:
            gdVT: 关于虚拟时间的梯度
            costT: 时间代价
        """
        # 计算 dRT/dVT
        if VT > 0:
            gdVT2Rt = VT + 1.0
        else:
            denSqrt = (0.5 * VT - 1.0) * VT + 1.0
            gdVT2Rt = (1.0 - VT) / (denSqrt * denSqrt)
        
        # 梯度 (链式法则)
        gdVT = (gdRT + self.wei_time) * gdVT2Rt
        
        # 时间代价
        costT = self.wei_time * RT
        
        return gdVT, costT
    
    def clipStateConstraints(self, state, max_vel, max_acc):
        """
        裁剪状态约束（用于初始/终止状态）
        
        参数:
            state: shape (3, 2) - [[px,py], [vx,vy], [ax,ay]]
            max_vel: 最大速度
            max_acc: 最大加速度
            
        返回:
            clipped_state: 裁剪后的状态
        """
        clipped_state = state.copy()
        
        # 裁剪速度
        vel_norm = np.linalg.norm(state[1, :])
        if vel_norm >= max_vel:
            clipped_state[1, :] = state[1, :] / vel_norm * (max_vel - 1e-2)
        
        # 裁剪加速度
        acc_norm = np.linalg.norm(state[2, :])
        if acc_norm >= max_acc:
            clipped_state[2, :] = state[2, :] / acc_norm * (max_acc - 1e-2)
        
        return clipped_state
    
    def getOptimizedTrajectories(self):
        """获取优化后的轨迹"""
        return self.jerkOpt_container