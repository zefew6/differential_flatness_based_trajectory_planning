"""
MINCO 轨迹规划 + 微分平坦跟踪控制 —— 随机竹林场景（v2）

每次运行竹林都完全随机生成（位置 + 半径 + 高度），
起点在竹林西侧 (-4.5, 0.0)，终点在竹林东侧 (4.3, 0.0)。

用法
----
    source /home/hac/Differential_Flatness/MAS/MINCO/bin/activate
    # ESDF 方法（默认，每次随机）
    python examples/test_minco_planner_v2.py
    # SFC 凸走廊方法
    python examples/test_minco_planner_v2.py --method sfc
    # 固定随机种子（可复现）
    python examples/test_minco_planner_v2.py --seed 42
    # 指定竹子数量（默认 80）
    python examples/test_minco_planner_v2.py --n_bamboo 60
"""

import argparse
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

from m0.minco_planner import MINCO, PolyTrajOptimizer
from m0.planning.a_star import graph_search
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


# ══════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════
def uniform_resample_path(path: np.ndarray, max_seg_len: float = 3.0) -> np.ndarray:
    """Dftpav 风格：在原始 A* 路径上等弧长均匀采样，一步替代 preprocessPath + resamplePath。

    对应 Dftpav traj_manager.cpp L546-570：
        piece_nums = max(int(totalDuration / timePerPiece + 0.5), 2)
        ego_innerPs.col(i) = kino_path_finder_->evaluatePos(t).head(2)
    区别仅在于此处以弧长代替时间作为参数化依据。
    """
    path = np.asarray(path, dtype=float)
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_len = float(np.sum(seg_lens))
    piece_nums = max(int(total_len / max_seg_len + 0.5), 2)

    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    sample_dists = np.linspace(0.0, total_len, piece_nums + 1)

    result = []
    for d in sample_dists:
        idx = int(np.searchsorted(cum_dist, d, side='right')) - 1
        idx = np.clip(idx, 0, len(path) - 2)
        denom = cum_dist[idx + 1] - cum_dist[idx]
        alpha = (d - cum_dist[idx]) / (denom + 1e-12)
        result.append(path[idx] + alpha * (path[idx + 1] - path[idx]))
    return np.array(result)


def push_waypoints_to_clearance(waypoints: np.ndarray,
                                grid_map,
                                max_iters: int = 30,
                                step_size: float = 0.08,
                                target_clearance: float = 0.30) -> np.ndarray:
    """将内点沿 ESDF 梯度方向推离障碍物，使其位于更宽阔的自由空间中心。

    SFC 走廊质量完全取决于种子点位置：种子点越靠近竹子，走廊越窄，
    5 阶多项式越难约束在其中。此函数在建走廊前让种子点「逃」到安全位置。

    参数
    ----
    waypoints       : shape (N, 2)，含头尾
    grid_map        : GridMap2D，提供 get_distance_and_gradient()
    max_iters       : 最大推离迭代次数
    step_size       : 每步推离距离（m）
    target_clearance: 目标最小间隙（m），到达后停止推离

    返回
    ----
    新的 waypoints，头尾不变，内点已推离
    """
    pts = waypoints.copy()
    n_pts = len(pts)
    for k in range(1, n_pts - 1):  # 只移动内点，头尾固定
        for _ in range(max_iters):
            dist, grad = grid_map.get_distance_and_gradient(pts[k])
            if dist >= target_clearance:
                break
            # 梯度方向即远离最近障碍物的方向
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-8:
                break
            pts[k] += step_size * grad / gnorm
    return pts


def sample_traj_xy(minco_obj, n=800):
    t_total = float(np.sum(minco_obj.T))
    ts  = np.linspace(0.0, t_total, int(n))
    pts = np.vstack([minco_obj.eval(t)[0] for t in ts])
    return pts


# ══════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MINCO 规划随机竹林场景 (v2)")
    parser.add_argument("--method",   choices=["esdf", "sfc"], default="esdf",
                        help="障碍物方法：esdf 或 sfc")
    parser.add_argument("--seed",     type=int, default=None,
                        help="随机种子（不指定则每次不同）")
    parser.add_argument("--n_bamboo", type=int, default=80,
                        help="竹子数量（默认 80）")
    args   = parser.parse_args()
    method = args.method
    seed   = args.seed
    n_bam  = args.n_bamboo
    print(f"[MINCO planner v2 | bamboo scene] method={method}  n_bamboo={n_bam}  seed={seed}")

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

    # ── 地图 & A* ────────────────────────────────────────────────────────
    grid_map = GridMap2D(
        model=model, data=data,
        resolution=0.05,
        width=10.0, height=10.0,
        robot_radius=0.2,
        margin=0.1,
        origin_x=-5.0, origin_y=-5.0,
    )
    path = graph_search(start=head_pos, goal=tail_pos, gridmap=grid_map)
    if path is None:
        raise RuntimeError("A* 无法在竹林中找到路径，请换一个 seed 或减少 n_bamboo")

    # ── 路径预处理（Dftpav 风格：等弧长均匀采样，一步完成）────────────────
    # 对应 Dftpav traj_manager.cpp RunMINCOParking()：
    #   piece_nums = max(int(totalDuration / timePerPiece + 0.5), 2)
    #   ego_innerPs.col(i) = evaluatePos(t).head(2)
    # 几何 A* 以弧长代替时间参数化，等价地在原始路径上等间隔采样。
    optimizer = PolyTrajOptimizer(obstacle_method=method)
    optimizer.setGridMap(grid_map)

    resampled = uniform_resample_path(path, max_seg_len=1.2)
    inner_pts = resampled[1:-1]
    full_pts  = np.vstack([head_pos, inner_pts, tail_pos])
    print(f"A* {len(path)} pts → uniform resampled {len(resampled)} pts (Dftpav style)")

    # ── SFC 走廊（仅 sfc 方法）──────────────────────────────────────────
    if method == "sfc":
        # 先将内点推离障碍物：SFC 走廊质量由种子点位置决定，
        # 种子点越靠近竹子，走廊越窄，5 阶多项式越难约束在其中。
        full_pts_pushed = push_waypoints_to_clearance(
            full_pts, grid_map,
            max_iters=40,
            step_size=0.06,
            target_clearance=0.35,
        )
        n_moved = int(np.sum(
            np.linalg.norm(full_pts_pushed[1:-1] - full_pts[1:-1], axis=1) > 1e-3
        ))
        print(f"[SFC] 推离内点: {n_moved}/{len(inner_pts)} 个点被推离障碍物")
        optimizer.buildSFCCorridors(full_pts_pushed, search_radius=6.0, method='legacy')
        optimizer.setParam(sfc_safe_margin=0.05, wei_sfc=1e5)
        full_pts = full_pts_pushed
        inner_pts = full_pts_pushed[1:-1]

    # ── MINCO 优化 ───────────────────────────────────────────────────────
    head_pva  = np.array([head_pos, [0.0, 0.0], [0.0, 0.0]])
    tail_pva  = np.array([tail_pos, [0.0, 0.0], [0.0, 0.0]])
    durations = optimizer.allocateTime(full_pts)

    t0 = time.time()
    print("MINCO 竹林轨迹优化中…")
    # optimizer.setParam(wei_feas=1e3)
    optimizer.OptimizeTrajectory(
        iniStates=[head_pva], finStates=[tail_pva],
        initInnerPts=[inner_pts],
        initTs=np.array([np.sum(durations)]),
        initSegTs=[durations],
    )
    print(f"优化完成，耗时 {(time.time()-t0)*1000:.2f}ms")

    # 重建 MINCO 对象（支持 eval() 接口）
    opt_traj   = optimizer.getOptimizedTrajectories()[0]
    opt_coeffs = opt_traj.getCoeffs()
    opt_T      = opt_traj.T
    wpts = []
    for i in range(len(opt_T) - 1):
        c  = opt_coeffs[6*i:6*(i+1), :]
        Ti = opt_T[i]
        wpts.append(c[0] + c[1]*Ti + c[2]*Ti**2 + c[3]*Ti**3 + c[4]*Ti**4 + c[5]*Ti**5)
    opt_minco = MINCO(head_pva, tail_pva, np.array(wpts), opt_T)

    # ── 可视化用轨迹数据 ─────────────────────────────────────────────────
    def to_xyz(pts_2d, z=0.06):
        xyz = np.zeros((len(pts_2d), 3)) + z
        xyz[:, :2] = pts_2d
        return xyz

    xyz_astar     = to_xyz(path,      z=0.04)
    xyz_resampled = to_xyz(resampled, z=0.06)
    xyz_opt       = to_xyz(sample_traj_xy(opt_minco, n=1000), z=0.08)

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

    print("进入竹林仿真，按 Ctrl-C 或关闭窗口退出。")
    try:
        while mjv.is_running():
            pos_xy = robot.get_pos()[:2]
            yaw    = robot.get_yaw()

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

            # 渲染
            try:
                mjv.reset(0)
            except TypeError:
                mjv.reset()

            mjv.draw_traj(xyz_astar,     size=0.02, rgba=np.array([1.0, 0.2, 0.2, 1.0]))
            mjv.draw_traj(xyz_resampled, size=0.04, rgba=np.array([1.0, 0.6, 0.0, 1.0]))
            mjv.draw_traj(xyz_opt,       size=0.03, rgba=np.array([0.0, 1.0, 0.5, 1.0]))

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
