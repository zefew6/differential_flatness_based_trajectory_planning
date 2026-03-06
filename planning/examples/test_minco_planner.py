"""
MINCO 轨迹规划 + 微分平坦跟踪控制 一体化示例

用法
----
    source /home/hac/Differential_Flatness/MAS/MINCO/bin/activate
    # ESDF 方法（默认）
    python examples/test_minco_planner.py
    # SFC 凸走廊方法
    python examples/test_minco_planner.py --method sfc
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import mujoco
except Exception as e:
    raise RuntimeError("请先激活含有 mujoco 的 virtualenv") from e

# 将 planning 包根目录注入 sys.path
planning_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if planning_root not in sys.path:
    sys.path.insert(0, planning_root)

from m0.minco_planner import PolyTrajOptimizer
from m0.planning.a_star import graph_search
from m0.minco_planner.minco_obstacle import GridMap2D
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot
from m0.control import TrajectoryFollower   # ← 跟踪控制器


def sample_traj_xy(minco_obj, n=800):
    t_total = float(np.sum(minco_obj.T))
    ts  = np.linspace(0.0, t_total, int(n))
    pts = np.vstack([minco_obj.eval(t)[0] for t in ts])
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["esdf", "sfc"], default="esdf",
                        help="障碍物方法：esdf（距离场）或 sfc（凸走廊）")
    args   = parser.parse_args()
    method = args.method
    print(f"[MINCO planner] method = {method}")

    # ── MuJoCo ───────────────────────────────────────────────────────────
    xml_path = os.path.join(planning_root, "m0", "assets", "minco_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    robot = Robot(model, data, robot_body_name="pusher1",
                  actuator_names=["forward", "turn"], max_v=2.0, max_w=4.0)
    mjv   = MujocoViewer(model, data)
    mjv.set_camera(distance=20.0, azimuth=0, elevation=-30, lookat=[5, 5, 0])

    # ── 地图 & A* ────────────────────────────────────────────────────────
    head_pos = np.array([-4.0,  4.0])
    tail_pos = np.array([ 3.8,  0.0])

    grid_map = GridMap2D(model=model, data=data, resolution=0.05,
                         width=10.0, height=10.0, robot_radius=0.3,
                         margin=0.1, origin_x=-5.0, origin_y=-5.0)
    path = graph_search(start=head_pos, goal=tail_pos, gridmap=grid_map)
    if path is None:
        raise RuntimeError("A* 找不到路径")

    # ── 路径预处理 ────────────────────────────────────────────────────────
    optimizer = PolyTrajOptimizer(obstacle_method=method)
    optimizer.setGridMap(grid_map)

    pruned    = optimizer.preprocessPath(path)
    resampled = optimizer.resamplePath(pruned, max_seg_len=4, dense_path=path)
    inner_pts = resampled[1:-1]
    full_pts  = np.vstack([head_pos, inner_pts, tail_pos])
    print(f"A* {len(path)} pts → pruned {len(pruned)} → resampled {len(resampled)}")

    # ── MINCO 优化 ───────────────────────────────────────────────────────
    head_pva  = np.array([head_pos, [0.0, 0.0], [0.0, 0.0]])
    tail_pva  = np.array([tail_pos, [0.0, 0.0], [0.0, 0.0]])

    t0 = time.time()
    print("MINCO 优化中…")
    opt_minco, _ = optimizer.astar_path_to_follower_path(
        path,
        head_pva=head_pva,
        tail_pva=tail_pva,
        max_seg_len=4.0,
        waypoints=full_pts,
        sfc_push_to_clearance=False,
        sfc_build_method='cube',
        sfc_safe_margin=0.0,
        sfc_wei=1e5,
    )
    print(f"优化完成，耗时 {(time.time()-t0)*1000:.2f}ms")

    # ── 可视化用轨迹数据 ─────────────────────────────────────────────────
    def to_xyz(pts_2d, z=0.03):
        xyz = np.zeros((len(pts_2d), 3)) + z
        xyz[:, :2] = pts_2d
        return xyz

    xyz_astar     = to_xyz(path)
    xyz_resampled = to_xyz(resampled)
    xyz_opt       = to_xyz(sample_traj_xy(opt_minco, n=1000))

    # ── 机器人初始化 ──────────────────────────────────────────────────────
    _, first_vel, _ = opt_minco.eval(0.05)
    init_yaw = np.arctan2(first_vel[1] + 1e-9, first_vel[0] + 1e-9)
    robot.set_state([head_pos[0], head_pos[1], 0.03], init_yaw)
    mujoco.mj_forward(model, data)

    # ── 跟踪控制器 ────────────────────────────────────────────────────────
    follower = TrajectoryFollower(
        opt_minco,
        max_v=2.0, max_w=4.0,
        kx=2.0, ky=2.0, ktheta=3.0,
        proj_samples=40, proj_window=2.0,
        vd_min=0.6, a_brake=3.0, goal_tol=0.15,
    )
    frame_skip = 5
    dt_ctrl    = model.opt.timestep * frame_skip

    print("进入仿真，按 Ctrl-C 或关闭窗口退出。")
    try:
        while mjv.is_running():
            pos_xy = robot.get_pos()[:2]
            yaw    = robot.get_yaw()

            if follower.done:
                # 主动制动：data.cvel 格式 [ωx,ωy,ωz, vx,vy,vz]，取 [3:5] 为线速度
                vel6  = robot.get_v()
                v_now = np.hypot(vel6[3], vel6[4])
                if v_now > 0.08:
                    robot.set_ctrl(-np.clip(v_now * 2.0, 0.5, 2.0), 0.0)
                else:
                    robot.set_ctrl(0.0, 0.0)
            else:
                v, w = follower.step(pos_xy, yaw, dt_ctrl)
                robot.set_ctrl(v, w)

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

            # 渲染
            try:
                mjv.reset(0)
            except TypeError:
                mjv.reset()

            mjv.draw_traj(xyz_astar,     size=0.02, rgba=np.array([1.0, 0.2, 0.2, 1.0]))
            mjv.draw_traj(xyz_resampled, size=0.04, rgba=np.array([1.0, 0.6, 0.0, 1.0]))
            mjv.draw_traj(xyz_opt,       size=0.03, rgba=np.array([0.2, 0.8, 0.2, 1.0]))

            if len(follower.trail) > 1:
                trail = np.array(follower.trail)
                xyz_t = np.zeros((len(trail), 3)) + 0.05
                xyz_t[:, :2] = trail
                mjv.draw_traj(xyz_t, size=0.025, rgba=np.array([0.0, 0.9, 0.9, 0.9]))

            ref_p = follower.ref_point
            mjv.draw_point(np.array([ref_p[0], ref_p[1], 0.08]),
                           size=0.08, rgba=np.array([1.0, 1.0, 0.0, 1.0]))
            mjv.render()

    except KeyboardInterrupt:
        pass
    finally:
        mjv.close()


if __name__ == "__main__":
    main()
