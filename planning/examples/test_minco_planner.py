"""
A minimal example to run the MINCO-based planner and visualize the optimized
trajectory in the same mujoco scene used by `test_astar.py`.

Usage:
  1) Activate the MINCO virtualenv in your shell:
       source /home/hac/Differential_Flatness/MAS/MINCO/bin/activate
  2) Run this script from anywhere (it will load the model using a relative path):
       python3 planning/examples/test_minco_planner.py

The script will:
  - load a mujoco model `m0/assets/test_world.xml`
  - construct a grid map and obstacle set (from m0/minco_planner/map_obstacles.py)
  - run the PolyTrajOptimizer to optimize a MINCO trajectory
  - render the optimized trajectory points as small spheres in Mujoco

This file intentionally keeps things simple and interactive (like `test_astar.py`).
"""

import os
import sys
import time
import numpy as np

try:
    import mujoco
except Exception as e:
    raise RuntimeError("Please activate the MINCO virtualenv where 'mujoco' is installed") from e

# Ensure repository packages/modules can be imported when running from repository root
_this_file = os.path.abspath(__file__)
# planning_root is the `planning` folder containing `m0`
planning_root = os.path.dirname(os.path.dirname(_this_file))
if planning_root not in sys.path:
    sys.path.insert(0, planning_root)

# Allow bare imports used inside m0/minco_planner (minco, minco_Optimizer, ...)
minco_planner_dir = os.path.join(planning_root, "m0", "minco_planner")
if minco_planner_dir not in sys.path:
    sys.path.insert(0, minco_planner_dir)

from minco import MINCO
from minco_Optimizer import PolyTrajOptimizer

# Import A* and GridMap2D (继承自 GridMap，同时支持 A* 和 MINCO 优化器)
from m0.planning.a_star import graph_search
from m0.utils.gridmap_2d_v2 import GridMap2D

# Import viewer via package path under planning_root
from m0.viewer.mujoco_visualization import MujocoViewer


def sample_traj_xy(minco_obj, n=800):
    t_total = float(np.sum(minco_obj.T))
    ts = np.linspace(0.0, t_total, int(n))
    pts = np.vstack([minco_obj.eval(t)[0] for t in ts])
    return ts, pts


def main():
    # load mujoco model (same xml as test_astar)
    xml_path = os.path.join(planning_root, "m0", "assets", "test_world.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"mujoco xml not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # viewer
    mjv = MujocoViewer(model, data)
    mjv.set_camera(distance=20.0, azimuth=0, elevation=-30, lookat=[5, 5, 0])
    # Problem definition: we'll generate an initial discrete path with A* and
    # then feed a downsampled version of that path (inner waypoints) to MINCO.
    head_pos = np.array([0, 0.2])
    tail_pos = np.array([9, 8.4])

    # Build the GridMap2D：继承自 GridMap，同时支持 A*（占用格、坐标转换）
    # 和 MINCO 优化器（get_distance_and_gradient），只需构建一次
    grid_map = GridMap2D(model=model, data=data, resolution=0.05, width=20.0, height=20.0,
                        robot_radius=0.3, margin=0.1)

    path = graph_search(start=head_pos, goal=tail_pos, gridmap=grid_map)
    if path is None:
        raise RuntimeError("A* failed to find a path")

    # grid_map 已经是 GridMap2D，同时具备 A* 接口和 ESDF 接口，直接传给优化器
    optimizer = PolyTrajOptimizer()
    # obs_safe_threshold 与 A* 的 inflation_radius（robot_radius+margin=0.4m）对齐，
    # 避免 MINCO 将轨迹推离 A* 路径太远
    optimizer.setGridMap(grid_map)

    # ── 路径后处理（由优化器完成）─────────────────────────────────────────
    # 1. 可见性剪枝：去掉可被直线跳过的冗余路点
    pruned_path = optimizer.preprocessPath(path)
    # 2. 重采样：每段不超过 1.5m，沿原始 A* 路径取中间点（避免直线插值引发蛇形振荡）
    resampled_path = optimizer.resamplePath(pruned_path, max_seg_len=1.5, dense_path=path)
    print(f"A* raw: {len(path)} pts → pruned: {len(pruned_path)} → resampled: {len(resampled_path)}")

    # 内部路点（去掉首尾）
    inner_pts = resampled_path[1:-1]  # shape (M, 2)

    # Build head/tail pva arrays (zero vel/acc at start and goal)
    head_pva = np.array([head_pos, [0.0, 0.0], [0.0, 0.0]])
    tail_pva = np.array([tail_pos, [0.0, 0.0], [0.0, 0.0]])

    # 2. 梯形速度曲线时间分配（由优化器完成，使用其 max_vel/max_acc 参数）
    full_pts = np.vstack([head_pos, inner_pts, tail_pos])
    durations = optimizer.allocateTime(full_pts)  # shape (piece_num,)

    initTs = np.array([np.sum(durations)])

    print("Starting MINCO optimization (A* initial path -> MINCO)...")
    success, final_cost = optimizer.OptimizeTrajectory(
        iniStates=[head_pva],
        finStates=[tail_pva],
        initInnerPts=[inner_pts],
        initTs=initTs,
        initSegTs=[durations],   # 按段时间，替代等分逻辑
    )

    if not success:
        print("Optimizer reported failure or did not fully converge, but will still try to visualize result.")

    opt_traj = optimizer.getOptimizedTrajectories()[0]

    # opt_traj is a MinJerkOpt-like object. It doesn't implement eval(), so
    # reconstruct a MINCO instance from optimized coefficients+T for sampling.
    opt_coeffs = opt_traj.getCoeffs()
    opt_T = opt_traj.T

    # compute optimized waypoints (positions at the end of each segment except last)
    waypoints_opt = []
    for i in range(len(opt_T) - 1):
        c = opt_coeffs[6 * i : 6 * (i + 1), :]
        Ti = opt_T[i]
        # evaluate polynomial at t=Ti
        pos = (c[0, :] + c[1, :] * Ti + c[2, :] * Ti ** 2 + c[3, :] * Ti ** 3 + c[4, :] * Ti ** 4 + c[5, :] * Ti ** 5)
        waypoints_opt.append(pos)
    waypoints_opt = np.array(waypoints_opt)

    # construct a MINCO object for easy evaluation/sampling
    opt_minco = MINCO(head_pva, tail_pva, waypoints_opt, opt_T)

    # sample optimized trajectory into XYZ for viewer (set small z)
    _, pts = sample_traj_xy(opt_minco, n=1000)
    xyz_opt = np.zeros((pts.shape[0], 3)) + 0.03
    xyz_opt[:, 0:2] = pts

    # A* 原始路径（红色）+ 重采样路点（橙色大球，即 MINCO 初始内部路点）
    xyz_astar = np.zeros((path.shape[0], 3)) + 0.03
    xyz_astar[:, 0:2] = path

    xyz_resampled = np.zeros((resampled_path.shape[0], 3)) + 0.03
    xyz_resampled[:, 0:2] = resampled_path

    print("Entering Mujoco viewer. Close the window to exit.")
    # interactive rendering loop
    try:
        while mjv.is_running():
            # reset geom write pointer so we don't overflow the pre-allocated buffer
            try:
                mjv.reset(0)
            except TypeError:
                # fallback for older viewer with reset() signature
                try:
                    mjv.reset()
                except Exception:
                    pass

            # 绘制 A* 原始路径（红色，细）
            mjv.draw_traj(xyz_astar, size=0.02, rgba=np.array([1.0, 0.2, 0.2, 1.0]))
            # 绘制重采样路点（橙色，显示 MINCO 初始导引点）
            mjv.draw_traj(xyz_resampled, size=0.04, rgba=np.array([1.0, 0.6, 0.0, 1.0]))
            # 绘制 MINCO 优化轨迹（绿色）
            mjv.draw_traj(xyz_opt, size=0.03, rgba=np.array([0.2, 0.8, 0.2, 1.0]))
            mjv.render()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        mjv.close()


if __name__ == "__main__":
    main()
