import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# 允许从 planning/m0 下导入 utils、minco_planner 等模块
_this_dir = os.path.dirname(os.path.abspath(__file__))
_m0_dir = os.path.dirname(_this_dir)
if _m0_dir not in sys.path:
    sys.path.insert(0, _m0_dir)

from minco import MINCO
from minco_Optimizer import PolyTrajOptimizer
from utils.gridmap_2d_v2 import GridMap2D, GridMap2DParams

def check_constraints(minco_traj, max_vel, max_acc, dt=0.01):
    """检查轨迹是否满足速度和加速度约束"""
    t_total = np.sum(minco_traj.T)
    t_samples = np.arange(0, t_total, dt)
    
    max_vel_violation = 0.0
    max_acc_violation = 0.0
    vel_violation_count = 0
    acc_violation_count = 0
    
    for t in t_samples:
        _, vel, acc = minco_traj.eval(t)
        vel_norm = np.linalg.norm(vel)
        acc_norm = np.linalg.norm(acc)
        
        if vel_norm > max_vel:
            max_vel_violation = max(max_vel_violation, vel_norm - max_vel)
            vel_violation_count += 1
        
        if acc_norm > max_acc:
            max_acc_violation = max(max_acc_violation, acc_norm - max_acc)
            acc_violation_count += 1
    
    return {
        'max_vel_violation': max_vel_violation,
        'max_acc_violation': max_acc_violation,
        'vel_violation_count': vel_violation_count,
        'acc_violation_count': acc_violation_count,
        'total_samples': len(t_samples)
    }
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib plotting (useful on headless machines / SSH).",
    )
    args = parser.parse_args()

    # 如果没有图形环境，默认关闭绘图，避免脚本卡在 plt.show() 或直接报错
    if os.environ.get("DISPLAY", "") == "":
        args.no_plot = True
    
    # 定义起点和终点 (位置, 速度) - 二维
    head_pva = np.array([
        [0, 0],  # 位置 x    y
        [1, 0],  # 速度 v_x v_y
        [0, 0]   # 加速度 a_x a_y
    ])
    
    tail_pva = np.array([
        [15, 15],  # 位置 x    y
        [1, 1],  # 速度 v_x v_y
        [0, 0]   # 加速度 a_x a_y
    ])
    
    # 定义中间航点 (第一段结束时的位置)
    waypoints = np.array([ # 位置 x y
        [2, 3],
        [6, 5],
        [8, 9],
        [10, 13],
        [13,14],
    ])

    # 每段的时间 - 设置合理的初始时间
    durations = np.array([3, 4, 5, 3, 4, 6])  # 总计39秒
    
    # 创建轨迹
    traj = MINCO(head_pva, tail_pva, waypoints, durations)
    
    # 准备优化器输入数据
    # 起始状态：shape (2, 3) - 行是 [p, v, a]，列是 [x, y]
    iniState = head_pva
    
    finState = tail_pva
    
    # 初始内部航点：N行2列，每行是一个航点 [x, y]
    initInnerPts = waypoints
    
    # 创建优化器（使用 Dftpav 默认参数）
    optimizer = PolyTrajOptimizer()

    # ------------------------------
    # 构造一个简单障碍物地图并注入给优化器
    # ------------------------------
    # 地图坐标范围（m）：[0, 20] x [0, 20]
    # 为了对齐原来想做的 [-2,18]，这里等价于把整个场景平移了 (+2,+2)
    # 放一个圆形障碍物在 (7.5, 7.5)（平移后是 (9.5, 9.5)），半径约 1.2m
    map_params = GridMap2DParams(
        resolution=0.1,
        size_x=20.0,
        size_y=20.0,
        origin_at_center=False,
    )
    grid_map = GridMap2D(map_params)

    # v2 地图在 origin_at_center=False 时，默认覆盖 [0,size_x]x[0,size_y]
    xs = np.arange(0.0, map_params.size_x, map_params.resolution)
    ys = np.arange(0.0, map_params.size_y, map_params.resolution)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    occ = ((X - 9.5) ** 2 + (Y - 9.5) ** 2) <= (1.2 ** 2)
    grid_map.set_occupancy(occ.astype(np.uint8))
    grid_map.update_esdf()

    optimizer.setGridMap(grid_map)
    
    # 初始时间分配 - 使用durations的总和
    total_dist = np.linalg.norm(tail_pva[0] - head_pva[0])  # 起点到终点的总距离
    initTs = np.array([np.sum(durations)])
    
    print(f"\n初始参数:")
    print(f"  总距离: {total_dist:.2f} m")
    print(f"  初始时间: {initTs[0]:.2f} s")
    print(f"  平均速度: {total_dist/initTs[0]:.2f} m/s")
    print(f"  Max constraints: vel={optimizer.max_vel:.1f} m/s, acc={optimizer.max_acc:.1f} m/s²")
    
    # 执行优化
    print("\n开始优化...")
    success, final_cost = optimizer.OptimizeTrajectory(
        iniStates=[iniState],
        finStates=[finState],
        initInnerPts=[initInnerPts],
        initTs=initTs
    )
    
    # 获取优化后的轨迹
    opt_traj = optimizer.getOptimizedTrajectories()[0]
    opt_jerk_cost = opt_traj.getTrajJerkCost()
    init_jerk_cost = traj.get_energy()
    
    # 检查约束违反情况
    
    # 检查优化前后的约束
    init_violations = check_constraints(traj, optimizer.max_vel, optimizer.max_acc)
    
    # 重建优化后的轨迹
    opt_coeffs = opt_traj.getCoeffs()
    opt_T = opt_traj.T
    
    # 调试：打印优化前后的变化
    print("\n优化前后对比:")
    print(f"  初始航点:\n{waypoints}")
    
    waypoints_opt = []
    for i in range(len(opt_T) - 1):
        c = opt_coeffs[6*i:6*(i+1), :]
        Ti = opt_T[i]
        pos = (c[0, :] + c[1, :]*Ti + c[2, :]*Ti**2 + 
               c[3, :]*Ti**3 + c[4, :]*Ti**4 + c[5, :]*Ti**5)
        waypoints_opt.append(pos)
    
    waypoints_opt_array = np.array(waypoints_opt)
    print(f"  优化后航点:\n{waypoints_opt_array}")
    print(f"\n  初始时间: {durations}")
    print(f"  优化后时间: {opt_T}")
    
    opt_minco = MINCO(head_pva, tail_pva, waypoints_opt, opt_T)
    opt_violations = check_constraints(opt_minco, optimizer.max_vel, optimizer.max_acc)

    # 统计轨迹与障碍物的最小 ESDF 距离（越大越安全）
    def min_esdf_along_traj(minco_traj, grid_map, dt=0.02):
        t_total = float(np.sum(minco_traj.T))
        ts = np.arange(0.0, t_total + 1e-9, dt)
        dmins = []
        for t in ts:
            pos, _, _ = minco_traj.eval(t)
            # 本例地图是 [0,20]x[0,20]；轨迹仍在原坐标系，所以同样平移 (+2,+2)
            pos_map = np.asarray(pos, dtype=float) + np.array([2.0, 2.0])
            dmins.append(grid_map.get_distance(pos_map))
        return float(np.min(dmins))

    init_min_dist = min_esdf_along_traj(traj, grid_map)
    opt_min_dist = min_esdf_along_traj(opt_minco, grid_map)
    print(f"\n[Obstacle check] min ESDF distance along trajectory:")
    print(f"  before opt: {init_min_dist:.3f} m")
    print(f"  after  opt: {opt_min_dist:.3f} m")
    
    # 显示对比结果
    print("\n" + "=" * 60)
    print("优化结果对比")
    print("=" * 60)
    print(f"优化状态: {'✓ 成功' if success else '⚠ 未完全收敛'}")
    print(f"最终代价: {final_cost:.6f}")
    print(f"\nJerk代价对比:")
    print(f"  优化前: {init_jerk_cost:.6f}")
    print(f"  优化后: {opt_jerk_cost:.6f}")
    
    improvement = (init_jerk_cost - opt_jerk_cost) / init_jerk_cost * 100
    if improvement > 0:
        print(f"  改进: {improvement:.2f}% ⬇")
    else:
        print(f"  变化: {improvement:.2f}% (可能权衡了时间)")
    
    print(f"\n时间对比:")
    print(f"  优化前: {initTs[0]:.2f}s")
    opt_total_time = np.sum(opt_traj.T)
    print(f"  优化后: {opt_total_time:.2f}s")
    time_change = (initTs[0] - opt_total_time) / initTs[0] * 100
    if time_change > 0:
        print(f"  减少: {time_change:.2f}% ⬇")
    else:
        print(f"  增加: {-time_change:.2f}% ⬆")
    
    # 显示约束违反情况
    print(f"\n约束违反统计 (max_vel={optimizer.max_vel:.2f} m/s, max_acc={optimizer.max_acc:.2f} m/s²):")
    print(f"  优化前:")
    print(f"    速度违反: {init_violations['vel_violation_count']}/{init_violations['total_samples']} 点, 最大超出: {init_violations['max_vel_violation']:.3f} m/s")
    print(f"    加速度违反: {init_violations['acc_violation_count']}/{init_violations['total_samples']} 点, 最大超出: {init_violations['max_acc_violation']:.3f} m/s²")
    print(f"  优化后:")
    print(f"    速度违反: {opt_violations['vel_violation_count']}/{opt_violations['total_samples']} 点, 最大超出: {opt_violations['max_vel_violation']:.3f} m/s")
    print(f"    加速度违反: {opt_violations['acc_violation_count']}/{opt_violations['total_samples']} 点, 最大超出: {opt_violations['max_acc_violation']:.3f} m/s²")
    
    if not args.no_plot:
        # 绘制优化前的轨迹（Figure 1）
        print("\n绘制优化前的轨迹...")
        traj.plot_trajectory(dt=0.01, show_waypoints=True, show_velocity=True, show_acceleration=True)

        # 绘制优化后的轨迹（Figure 2）
        print("绘制优化后的轨迹...")
        opt_minco.plot_trajectory(dt=0.01, show_waypoints=True, show_velocity=True, show_acceleration=True)

        # 保持所有窗口打开，直到用户关闭
        print("\n所有图形已显示，关闭窗口以结束程序...")
        plt.show()  # 阻塞模式，保持所有窗口打开
    else:
        print("\n[Plot] Disabled (--no-plot or no DISPLAY).")
    
    print("\n" + "=" * 60)