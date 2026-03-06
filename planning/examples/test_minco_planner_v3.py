"""
MINCO 轨迹规划 + 微分平坦跟踪控制 —— 在线重规划动态避障测试（v3）

每次运行竹林都完全随机生成（位置 + 半径 + 高度），
起点在竹林西侧，终点在竹林东侧。

测试逻辑
--------
1) 先在随机竹林中做一次 A* + MINCO 初始规划并跟踪。
2) 机器人运动过程中，在当前参考路径前方“突然”注入一个圆柱障碍物
   （尺寸与竹子半径/高度同量级）。
3) 更新栅格地图 ESDF，并触发在线重规划（A* + MINCO），观察绕障效果。

用法
----
    source /home/hac/Differential_Flatness/MAS/MINCO/bin/activate
    python examples/test_minco_planner_v3.py
"""

import os
import sys
import time
import textwrap
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
from m0.minco_planner.minco_obstacle import GridMap2D
from m0.viewer.mujoco_visualization import MujocoViewer
from m0.robot.robot import Robot
from m0.control import TrajectoryFollower


# ══════════════════════════════════════════════════════════════════
#  随机竹林 XML 生成器
# ══════════════════════════════════════════════════════════════════
def _is_too_close(x, y, r, placed, min_gap=0.12):
    """检查新竹子 (x,y,r) 与已放置列表是否发生碰撞（含最小间隙 min_gap）。"""
    for (px, py, pr) in placed:
        if np.hypot(x - px, y - py) < r + pr + min_gap:
            return True
    return False


def build_bamboo_xml(
    n_bamboo: int = 80,
    seed: int | None = None,
    field_half: float = 4.6,          # 竹林区域半宽（m）
    r_min: float = 0.06,              # 竹竿最小半径
    r_max: float = 0.15,              # 竹竿最大半径
    h_min: float = 0.22,              # 竹竿最小半高（MuJoCo cylinder half-height）
    h_max: float = 0.45,              # 竹竿最大半高
    clearance_start: float = 1.0,     # 起点净空半径
    clearance_end: float = 1.0,       # 终点净空半径
    head_pos=(-4.5, 0.0),
    tail_pos=(4.3, 0.0),
    max_attempts: int = 5000,         # 最大采样次数（防止死循环）
) -> str:
    """返回完整的 MuJoCo XML 字符串，内含随机竹林。"""
    rng = np.random.default_rng(seed)

    # 竹竿颜色调色板（五种深浅青绿）
    colors = [
        ".28 .65 .18 1",
        ".35 .72 .22 1",
        ".22 .58 .14 1",
        ".40 .75 .12 1",
        ".18 .52 .16 1",
    ]

    placed = []     # 已放置竹子列表：(x, y, r)
    geom_lines = []

    attempts = 0
    while len(placed) < n_bamboo and attempts < max_attempts:
        attempts += 1
        # 随机半径、高度、位置
        r = float(rng.uniform(r_min, r_max))
        h = float(rng.uniform(h_min, h_max))
        x = float(rng.uniform(-field_half + r + 0.05, field_half - r - 0.05))
        y = float(rng.uniform(-field_half + r + 0.05, field_half - r - 0.05))

        # 起终点净空检查
        if np.hypot(x - head_pos[0], y - head_pos[1]) < clearance_start + r:
            continue
        if np.hypot(x - tail_pos[0], y - tail_pos[1]) < clearance_end + r:
            continue

        # 与已放置竹子的碰撞检查
        if _is_too_close(x, y, r, placed):
            continue

        idx = len(placed) + 1
        color = colors[idx % len(colors)]
        # z_pos = h（圆柱中心高度等于半高，底面恰好在地面）
        geom_lines.append(
            f'    <geom name="bam_{idx:03d}" type="cylinder" '
            f'pos="{x:.3f} {y:.3f} {h:.3f}" '
            f'size="{r:.3f} {h:.3f}" '
            f'rgba="{color}" material="bamboo_mat"/>'
        )
        placed.append((x, y, r))

    n_placed = len(placed)
    if n_placed < n_bamboo:
        print(f"[bamboo] 警告：仅放置了 {n_placed}/{n_bamboo} 根竹子（空间不足）")
    else:
        print(f"[bamboo] 随机生成 {n_placed} 根竹子，seed={seed}")

    bamboo_block = "\n".join(geom_lines)

    xml = textwrap.dedent(f"""\
    <mujoco>
      <compiler autolimits="true"/>

      <asset>
        <texture name="grass" type="2d" builtin="checker" width="512" height="512"
                 rgb1=".15 .28 .15" rgb2=".20 .35 .18"/>
        <material name="grass" texture="grass" texrepeat="6 6" texuniform="true" reflectance=".1"/>
        <texture name="bamboo_tex" type="2d" builtin="checker" width="64" height="256"
                 rgb1=".30 .62 .22" rgb2=".42 .72 .28"/>
        <material name="bamboo_mat" texture="bamboo_tex" texrepeat="1 4" reflectance=".05"/>
        <mesh name="chasis" scale=".01 .006 .0015"
          vertex=" 9   2   0
                  -10  10  10
                   9  -2   0
                   10  3  -10
                   10 -3  -10
                  -8   10 -10
                  -10 -10  10
                  -8  -10 -10
                  -5   0   20"/>
      </asset>

      <default>
        <joint damping=".008" actuatorfrcrange="-4 4"/>
        <default class="wheel">
          <geom type="cylinder" size=".03 .01" rgba=".5 .5 1 1"/>
        </default>
        <default class="decor">
          <site type="box" rgba=".5 1 .5 1"/>
        </default>
      </default>

      <worldbody>
        <geom name="floor" type="plane" size="6 6 .01" material="grass" pos="0 0 0"/>

        <!-- ── Robot ── -->
        <body name="pusher1" pos="0 0 .03">
          <freejoint/>
          <geom name="base" type="cylinder" pos="0 0 0.03" size="0.15 0.03" rgba="1.0 0.6 0.2 1"/>
          <geom type="box" pos="0.16 0 0.03" size="0.01 0.14 0.05" rgba="1 1 1 1"/>
          <geom name="front wheel" pos="0.12 0 -.015" type="sphere" size=".015" condim="1" priority="1"/>
          <body name="left wheel" pos="-.02 .16 0" zaxis="0 1 0">
            <joint name="left"/>
            <geom class="wheel"/>
            <site class="decor" size=".006 .025 .012"/>
            <site class="decor" size=".025 .006 .012"/>
          </body>
          <body name="right wheel" pos="-.02 -.16 0" zaxis="0 1 0">
            <joint name="right"/>
            <geom class="wheel"/>
            <site class="decor" size=".006 .025 .012"/>
            <site class="decor" size=".025 .006 .012"/>
          </body>
        </body>

        <!-- ── 四周围墙 ── -->
        <geom name="wall_north" type="box" pos=" 0.0  4.8 0.35" size="4.8 0.05 0.35" rgba=".55 .35 .15 1"/>
        <geom name="wall_south" type="box" pos=" 0.0 -4.8 0.35" size="4.8 0.05 0.35" rgba=".55 .35 .15 1"/>
        <geom name="wall_west"  type="box" pos="-4.8  0.0 0.35" size="0.05 4.8 0.35" rgba=".55 .35 .15 1"/>
        <geom name="wall_east"  type="box" pos=" 4.8  0.0 0.35" size="0.05 4.8 0.35" rgba=".55 .35 .15 1"/>

        <!-- ── 随机竹子（共 {n_placed} 根）── -->
    {bamboo_block}

      </worldbody>

      <tendon>
        <fixed name="forward">
          <joint joint="left"  coef=".5"/>
          <joint joint="right" coef=".5"/>
        </fixed>
        <fixed name="turn">
          <joint joint="left"  coef="-.5"/>
          <joint joint="right" coef=".5"/>
        </fixed>
      </tendon>

      <actuator>
        <motor name="forward" tendon="forward" ctrlrange="-2 2"/>
        <motor name="turn"    tendon="turn"    ctrlrange="-2 2"/>
      </actuator>

      <sensor>
        <jointactuatorfrc name="right" joint="right"/>
        <jointactuatorfrc name="left"  joint="left"/>
      </sensor>
    </mujoco>
    """)
    return xml

def sample_traj_xy(minco_obj, n=800):
    t_total = float(np.sum(minco_obj.T))
    ts  = np.linspace(0.0, t_total, int(n))
    pts = np.vstack([minco_obj.eval(t)[0] for t in ts])
    return pts


def build_circle_polyline(center_xy: np.ndarray, radius: float, n: int = 72, z: float = 0.08):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.zeros((n + 1, 3), dtype=float)
    pts[:-1, 0] = center_xy[0] + radius * np.cos(ang)
    pts[:-1, 1] = center_xy[1] + radius * np.sin(ang)
    pts[:-1, 2] = z
    pts[-1] = pts[0]
    return pts


def pick_injection_point_on_path(pos_xy: np.ndarray,
                                 goal_xy: np.ndarray,
                                 path_xy: np.ndarray,
                                 existing_centers: list,
                                 min_robot_dist: float = 1.0,
                                 min_goal_dist: float = 1.0,
                                 min_obs_spacing: float = 0.8):
    """从当前规划轨迹上选一个注入点（严格在 path_xy 上）。"""
    if path_xy is None or len(path_xy) < 20:
        return None

    d_robot = np.linalg.norm(path_xy - pos_xy[None, :], axis=1)
    near_idx = int(np.argmin(d_robot))

    start_idx = min(len(path_xy) - 1, near_idx + 35)  # 沿轨迹前方挑点
    end_idx = max(start_idx + 1, len(path_xy) - 15)   # 尾部留出终点缓冲

    for idx in range(start_idx, end_idx, 8):
        cand = path_xy[idx]
        if np.linalg.norm(cand - pos_xy) < min_robot_dist:
            continue
        if np.linalg.norm(cand - goal_xy) < min_goal_dist:
            continue
        if any(np.linalg.norm(cand - c) < min_obs_spacing for c in existing_centers):
            continue
        return cand.copy()

    return None
# ── 可视化用轨迹数据 ─────────────────────────────────────────────────
def to_xyz(pts_2d, z=0.06):
    xyz = np.zeros((len(pts_2d), 3)) + z
    xyz[:, :2] = pts_2d
    return xyz


# ══════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════
def main():
    seed = None
    n_bam = 80
    rng = np.random.default_rng(seed)
    print(f"[MINCO planner v3 | online replan] method=esdf  n_bamboo={n_bam}  seed={seed}")

    # ── 起终点（西侧 → 东侧，穿越整片竹林）──────────────────────────────
    head_pos = np.array([-4.5,  -4.5])   # 西侧中央，竹林入口
    tail_pos = np.array([ 4.3,  4.3])   # 东侧中央，竹林出口

    # ── 随机生成竹林 XML 并载入 MuJoCo ───────────────────────────────────
    xml_str = build_bamboo_xml(
        n_bamboo=n_bam,
        seed=seed,
        head_pos=tuple(head_pos),
        tail_pos=tuple(tail_pos),
    )
    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)
    robot = Robot(model, data, robot_body_name="pusher1",
                  actuator_names=["forward", "turn"], max_v=4.0, max_w=2.0)
    mjv   = MujocoViewer(model, data)
    mjv.set_camera(distance=22.0, azimuth=0, elevation=-35, lookat=[0, 0, 0])

    # ── 地图 & 首次规划 ───────────────────────────────────────────────────
    grid_map = GridMap2D(
        model=model, data=data,
        resolution=0.05,
        width=10.0, height=10.0,
        robot_radius=0.2,
        margin=0.1,
        origin_x=-5.0, origin_y=-5.0,
    )
    optimizer = PolyTrajOptimizer(obstacle_method="esdf")
    initial_plan = optimizer.online_replan_once(
        grid_map=grid_map,
        start_xy=head_pos,
        goal_xy=tail_pos,
        max_seg_len=1.2,
    )
    path = initial_plan["path"]
    resampled = initial_plan["resampled"]
    opt_minco = initial_plan["minco"]
    print(f"首次规划完成：A* {len(path)} pts → resampled {len(resampled)} pts, MINCO {initial_plan['cost_time_ms']:.1f} ms")

    xyz_astar = to_xyz(path, z=0.04)
    xyz_resampled = to_xyz(resampled, z=0.06)
    xyz_opt = to_xyz(sample_traj_xy(opt_minco, n=1000), z=0.08)

    # ── 在线重规划配置 ───────────────────────────────────────────────────
    replan_interval = 0.80
    next_replan_time = 0.0
    injected_obstacles = []  # list[ndarray(2,)]
    next_inject_time = 2.0
    inject_interval = 2.2
    inject_jitter = 0.8
    max_inject_count = 4
    injected_radius_min = 0.08
    injected_radius_max = 0.14
    injected_radii = []      # 每个障碍独立半径
    path_xy_for_inject = sample_traj_xy(opt_minco, n=1000)

    # ── 机器人初始化 ──────────────────────────────────────────────────────
    _, first_vel, _ = opt_minco.eval(0.05)
    init_yaw = np.arctan2(first_vel[1] + 1e-9, first_vel[0] + 1e-9)
    robot.set_state([head_pos[0], head_pos[1], 0.03], init_yaw)
    mujoco.mj_forward(model, data)

    # ── 跟踪控制器 ────────────────────────────────────────────────────────
    follower = TrajectoryFollower(
        opt_minco,
        max_v=4.0, max_w=2.0,
        kx=10.0, ky=10.0, ktheta=10.0,
        proj_samples=40, proj_window=2.0,
        vd_min=0.6, a_brake=3.0, goal_tol=0.15,
    )
    frame_skip = 5
    dt_ctrl    = model.opt.timestep * frame_skip
    sim_time = 0.0

    print("进入竹林仿真：将触发突发障碍并在线重规划。按 Ctrl-C 或关闭窗口退出。")
    try:
        while mjv.is_running():
            pos_xy = robot.get_pos()[:2]
            yaw    = robot.get_yaw()

            # 1) 多次障碍注入：每次都取当前规划轨迹上的点
            if (sim_time >= next_inject_time) and (len(injected_obstacles) < max_inject_count):
                cand = pick_injection_point_on_path(
                    pos_xy=pos_xy,
                    goal_xy=tail_pos,
                    path_xy=path_xy_for_inject,
                    existing_centers=injected_obstacles,
                )
                if cand is not None:
                    r_obs = float(rng.uniform(injected_radius_min, injected_radius_max))
                    grid_map.add_circle_obstacle(cand, r_obs, update_esdf=True)
                    injected_obstacles.append(cand)
                    injected_radii.append(r_obs)
                    next_replan_time = sim_time
                    next_inject_time = sim_time + inject_interval + float(rng.uniform(-inject_jitter, inject_jitter))
                    next_inject_time = max(next_inject_time, sim_time + 0.6)
                    print(f"[inject {len(injected_obstacles)}/{max_inject_count}] 路径上添加障碍: center={cand}, r={r_obs:.2f}, next_in={next_inject_time-sim_time:.2f}s")
                else:
                    next_inject_time = sim_time + 0.5
                    print("[inject] 当前轨迹上暂无合适注入点，稍后重试")

            # 2) 在线重规划：注入后按周期触发
            if (len(injected_obstacles) > 0) and (not follower.done) and (sim_time >= next_replan_time):
                try:
                    replan = optimizer.online_replan_once(
                        grid_map=grid_map,
                        start_xy=pos_xy.copy(),
                        goal_xy=tail_pos,
                        max_seg_len=1.2,
                    )
                    opt_minco = replan["minco"]
                    path = replan["path"]
                    resampled = replan["resampled"]
                    path_xy_for_inject = sample_traj_xy(opt_minco, n=1000)

                    follower = TrajectoryFollower(
                        opt_minco,
                        max_v=4.0, max_w=2.0,
                        kx=10.0, ky=10.0, ktheta=10.0,
                        proj_samples=40, proj_window=2.0,
                        vd_min=0.6, a_brake=3.0, goal_tol=0.15,
                    )

                    xyz_astar = to_xyz(path, z=0.04)
                    xyz_resampled = to_xyz(resampled, z=0.06)
                    xyz_opt = to_xyz(path_xy_for_inject, z=0.08)
                    next_replan_time = sim_time + replan_interval
                    print(f"[replan] 在线重规划成功, 耗时={replan['cost_time_ms']:.1f}ms, path_pts={len(path)}")
                except Exception as exc:
                    next_replan_time = sim_time + 0.4
                    print(f"[replan] 失败，稍后重试: {exc}")

            if follower.done:
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
            sim_time += dt_ctrl

            # 渲染
            try:
                mjv.reset(0)
            except TypeError:
                mjv.reset()

            mjv.draw_traj(xyz_astar,     size=0.02, rgba=np.array([1.0, 0.2, 0.2, 1.0]))
            mjv.draw_traj(xyz_resampled, size=0.04, rgba=np.array([1.0, 0.6, 0.0, 1.0]))
            mjv.draw_traj(xyz_opt,       size=0.03, rgba=np.array([0.0, 1.0, 0.5, 1.0]))

            for center, radius in zip(injected_obstacles, injected_radii):
                cyl_ring = build_circle_polyline(center, radius, n=72, z=0.10)
                mjv.draw_traj(cyl_ring, size=0.02, rgba=np.array([1.0, 0.1, 0.1, 1.0]))

            if len(follower.trail) > 1:
                trail = np.array(follower.trail)
                xyz_t = np.zeros((len(trail), 3)) + 0.05
                xyz_t[:, :2] = trail
                mjv.draw_traj(xyz_t, size=0.025, rgba=np.array([0.0, 0.9, 0.9, 0.9]))

            ref_p = follower.ref_point
            mjv.draw_point(np.array([ref_p[0], ref_p[1], 0.10]),
                           size=0.08, rgba=np.array([1.0, 1.0, 0.0, 1.0]))
            mjv.render()

    except KeyboardInterrupt:
        pass
    finally:
        mjv.close()


if __name__ == "__main__":
    main()
