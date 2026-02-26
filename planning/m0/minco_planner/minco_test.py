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
from map_obstacles import CIRCULAR_OBSTACLES, OBSTACLES


def overlay_grid_map(ax, grid_map, *, safe_threshold: float = 0.5, circle=None, circles=None, rects=None, polys=None):
    """Overlay occupancy + ESDF safe contour + analytic circle on an XY axis.

    circle: (cx, cy, r) in the same frame as the axis.
    circles: list of (cx, cy, r) in the same frame as the axis (preferred).
    """
    # Occupancy grid
    try:
        occ = np.asarray(grid_map.occ)
        extent = [
            float(grid_map.min_boundary[0]),
            float(grid_map.max_boundary[0]),
            float(grid_map.min_boundary[1]),
            float(grid_map.max_boundary[1]),
        ]
        ax.imshow(
            occ.T,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            alpha=0.35,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            zorder=0,
        )
    except Exception:
        pass

    # ESDF safe-threshold contour
    try:
        esdf = np.asarray(grid_map.esdf)
        nx, ny = esdf.shape
        xs = np.linspace(float(grid_map.min_boundary[0]), float(grid_map.max_boundary[0]), nx)
        ys = np.linspace(float(grid_map.min_boundary[1]), float(grid_map.max_boundary[1]), ny)
        Xc, Yc = np.meshgrid(xs, ys, indexing="ij")
        ax.contour(
            Xc,
            Yc,
            esdf,
            levels=[float(safe_threshold)],
            colors=["r"],
            linewidths=1.2,
            linestyles="--",
            alpha=0.9,
            zorder=1,
        )
    except Exception:
        pass

    # Obstacle boundaries (analytic shapes; ESDF is from occupancy)
    if circles is None and circle is not None:
        circles = [circle]

    if circles is not None:
        th = np.linspace(0, 2 * np.pi, 200)
        for i, c in enumerate(list(circles)):
            cx, cy, r = c
            label = "Obstacle" if i == 0 else None
            ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), "k-", linewidth=1.5, alpha=0.9, label=label)

    if rects is not None:
        for rect in list(rects):
            xmin, xmax, ymin, ymax = rect
            xs = [xmin, xmax, xmax, xmin, xmin]
            ys = [ymin, ymin, ymax, ymax, ymin]
            ax.plot(xs, ys, "k-", linewidth=1.5, alpha=0.9)

    if polys is not None:
        for poly in list(polys):
            v = np.asarray(poly, dtype=float)
            if v.ndim != 2 or v.shape[0] < 3:
                continue
            vv = np.vstack([v, v[0]])
            ax.plot(vv[:, 0], vv[:, 1], "k-", linewidth=1.5, alpha=0.9)


def sample_traj_xy(minco_obj, *, n: int = 800):
    """Sample trajectory positions along time."""
    t_total = float(np.sum(minco_obj.T))
    ts = np.linspace(0.0, t_total, int(n))
    pts = np.vstack([minco_obj.eval(t)[0] for t in ts])
    return ts, pts


def compute_clearance_along_points(pts, grid_map, safe_threshold: float):
    """Compute clearance = d - safe_threshold for given points."""
    ds = np.array([float(grid_map.get_distance(p)) for p in pts], dtype=float)
    return ds - float(safe_threshold)


def plot_traj_mapframe(
    ax,
    *,
    pts_map,
    grid_map,
    safe_threshold: float,
    circle=None,
    start=None,
    goal=None,
    clearance=None,
    worst_case=None,
    title: str = "",
    show_colorbar: bool = True,
):
    """Plot trajectory in map frame with optional clearance coloring and worst-case marker."""
    overlay_grid_map(ax, grid_map, safe_threshold=safe_threshold, circle=circle)

    if clearance is None:
        ax.plot(pts_map[:, 0], pts_map[:, 1], "b-", linewidth=1.5, label="Trajectory")
    else:
        sc = ax.scatter(
            pts_map[:, 0],
            pts_map[:, 1],
            c=clearance,
            cmap="coolwarm",
            s=6,
            linewidths=0,
            label="Trajectory (colored by clearance)",
            zorder=3,
        )
        if show_colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("clearance = d - safe_threshold (m)")

    if worst_case is not None:
        wc = np.asarray(worst_case, dtype=float)
        ax.plot(wc[0], wc[1], marker="x", color="k", markersize=10, mew=2, label="Worst-case")

    if start is not None:
        ax.plot(start[0], start[1], "go", markersize=8, label="Start")
    if goal is not None:
        ax.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

    if title:
        ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def save_mapframe_figure(
    *,
    minco_obj,
    out_path: str,
    title: str,
    grid_map,
    safe_threshold: float,
    circle=None,
    start=None,
    goal=None,
    traj_to_map_offset=None,
    worst_case=None,
    n_samples: int = 800,
    dpi: int = 200,
    show_colorbar: bool = True,
):
    """Save a single trajectory figure in map frame (headless-friendly)."""
    if traj_to_map_offset is None:
        traj_to_map_offset = np.array([0.0, 0.0], dtype=float)

    _, pts = sample_traj_xy(minco_obj, n=int(n_samples))
    pts_map = pts + traj_to_map_offset
    clearance = compute_clearance_along_points(pts_map, grid_map, safe_threshold=safe_threshold)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_traj_mapframe(
        ax,
        pts_map=pts_map,
        grid_map=grid_map,
        safe_threshold=safe_threshold,
        circle=circle,
        start=start,
        goal=goal,
        clearance=clearance,
        worst_case=worst_case,
        title=title,
        show_colorbar=show_colorbar,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

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


def min_sdf_along_poly(coeffs, T, grid_map, *, traj_resolution: int, destraj_resolution: int):
    """Compute min ESDF distance along a piecewise polynomial trajectory.

    Sampling matches the optimizer's ObstacleConstraint convention:
    - Use `destraj_resolution` on first/last pieces
    - Use `traj_resolution` on middle pieces
    """
    piece_num = len(T)
    min_d = float("inf")
    min_pos = None
    min_piece = -1
    min_alpha = 0.0

    for i in range(piece_num):
        K = destraj_resolution if (i == 0 or i == piece_num - 1) else traj_resolution
        c = coeffs[6 * i : 6 * (i + 1), :]
        T_i = float(T[i])
        step = T_i / K
        for j in range(K + 1):
            s1 = step * j
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2
            s5 = s4 * s1
            beta0 = np.array([1.0, s1, s2, s3, s4, s5])
            pos = c.T @ beta0
            d = float(grid_map.get_distance(pos))
            if d < min_d:
                min_d = d
                min_pos = np.asarray(pos, dtype=float)
                min_piece = int(i)
                min_alpha = float(j / K)

    return float(min_d), min_pos, min_piece, min_alpha


def dense_clearance_check(coeffs, T, grid_map, *, safe_threshold: float, K_dense: int = 400, circle=None):
    """Stricter clearance check: fixed dense sampling per piece (for validation only)."""
    piece_num = len(T)
    min_d = float("inf")
    min_pos = None
    min_clear = float("inf")
    min_clear_pos = None

    # analytic circle check
    max_err_analytic = None
    min_d_analytic = None

    if circle is not None:
        cx, cy, r = circle
        max_err_analytic = 0.0
        min_d_analytic = float("inf")

    for i in range(piece_num):
        c = coeffs[6 * i : 6 * (i + 1), :]
        T_i = float(T[i])
        step = T_i / K_dense
        for j in range(K_dense + 1):
            s1 = step * j
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2
            s5 = s4 * s1
            beta0 = np.array([1.0, s1, s2, s3, s4, s5])
            pos = c.T @ beta0
            pos = np.asarray(pos, dtype=float)

            d = float(grid_map.get_distance(pos))
            if d < min_d:
                min_d = d
                min_pos = pos

            clear = d - float(safe_threshold)
            if clear < min_clear:
                min_clear = clear
                min_clear_pos = pos

            if circle is not None:
                d_an = float(np.hypot(pos[0] - cx, pos[1] - cy) - r)
                if d_an < min_d_analytic:
                    min_d_analytic = d_an
                err = abs(d - d_an)
                if err > max_err_analytic:
                    max_err_analytic = err

    return {
        "min_sdf": float(min_d),
        "min_sdf_pos": min_pos,
        "min_clear_to_safe": float(min_clear),
        "min_clear_pos": min_clear_pos,
        "min_sdf_analytic": None if circle is None else float(min_d_analytic),
        "max_err_vs_analytic": None if circle is None else float(max_err_analytic),
        "K_dense": int(K_dense),
    }
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Legacy flag: same as --plot-mode save.",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        choices=["auto", "show", "save"],
        default="auto",
        help=(
            "Plot behavior: auto=show figures if DISPLAY is available else save PNG; "
            "show=force plt.show(); save=force saving PNG (headless-friendly)."
        ),
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default="out/minco",
        help=(
            "Path prefix to save figures in map frame. Saves <prefix>_init.png and <prefix>_opt.png. "
            "Default: out/minco (relative to this script's directory)."
        ),
    )
    parser.add_argument(
        "--no-plot-kinematics",
        action="store_true",
        help=(
            "Disable the original MINCO kinematics figures (position/velocity/acceleration) in interactive mode. "
            "By default they are shown. Map-frame figures are always shown when plotting is enabled."
        ),
    )
    args = parser.parse_args()

    # Resolve plot mode (keep backward compatibility with --no-plot)
    plot_mode = str(args.plot_mode).lower()
    if args.no_plot:
        plot_mode = "save"

    has_display = os.environ.get("DISPLAY", "") != ""
    if plot_mode == "auto":
        plot_mode = "show" if has_display else "save"

    if plot_mode == "show" and (not has_display):
        raise RuntimeError(
            "plot-mode=show but DISPLAY is not set. "
            "Use --plot-mode save, or set up a graphical DISPLAY." 
        )

    # 本例地图是 [0,20]x[0,20]，轨迹也在该坐标系内，因此无需额外平移
    traj_to_map_offset = np.array([0.0, 0.0], dtype=float)
    
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

    # --- obstacle tuning (demo) ---
    # 对齐 ST-opt-tools：rho_collision ~ 1e5；safe_threshold 决定“安全圈”半径
    safe_th = 0.8
    optimizer.setParam(wei_obs=1e5, obs_safe_threshold=safe_th)
    optimizer.setDebugPrintEvery(10)

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
    # 障碍物在 `map_obstacles.py` 中集中配置，直接改源码即可（更工程化、可维护）。
    circles: list[tuple[float, float, float]] = []
    rects: list[tuple[float, float, float, float]] = []
    polys: list[np.ndarray] = []

    if OBSTACLES:
        for obs in list(OBSTACLES):
            t = str(obs.get("type", "")).lower()
            if t == "circle":
                circles.append((float(obs["cx"]), float(obs["cy"]), float(obs["r"])))
            elif t == "rect":
                rects.append((float(obs["xmin"]), float(obs["xmax"]), float(obs["ymin"]), float(obs["ymax"])))
            elif t in ("poly", "polygon"):
                polys.append(np.asarray(obs["verts"], dtype=float))
            else:
                raise ValueError(f"Unknown obstacle type: {t!r}")
    else:
        # Backward compatibility
        circles = list(CIRCULAR_OBSTACLES)

    if (len(circles) + len(rects) + len(polys)) == 0:
        raise RuntimeError("No obstacles configured. Please edit map_obstacles.py")

    # 清空 occupancy，然后把所有障碍物栅格化进去并更新 ESDF
    grid_map.set_occupancy(np.zeros((grid_map.nx, grid_map.ny), dtype=np.uint8), update_esdf=False)
    for (cx, cy, r) in circles:
        grid_map.add_circle_obstacle(np.array([cx, cy], dtype=float), float(r), update_esdf=False)
    for (xmin, xmax, ymin, ymax) in rects:
        grid_map.add_rectangle_obstacle(xmin, xmax, ymin, ymax, update_esdf=False)
    for v in polys:
        grid_map.add_polygon_obstacle(v, update_esdf=False)
    grid_map.update_esdf()

    # 现有 dense_clearance_check 的 analytic 只支持单圆，这里保留“第一个圆”的对照
    demo_circle_map = circles[0] if len(circles) > 0 else None

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

    # 统计轨迹与障碍物的最小 SDF/ESDF 距离（越大越安全）
    # 这里做“同口径”验收：
    # - 直接用多项式系数 + 分段时间采样（与优化器 ObstacleConstraint 一致）
    # - 使用同一个 grid_map
    # - 不引入额外的 (+2,+2) 平移（地图与轨迹均在 map frame 构造）

    # initial (use optimizer's internal polynomial after first generate)
    init_coeffs = optimizer.jerkOpt_container[0].coeffs
    init_T_seg = optimizer.jerkOpt_container[0].T

    init_min_dist, init_min_pos, init_min_piece, init_min_alpha = min_sdf_along_poly(
        init_coeffs,
        init_T_seg,
        grid_map,
        traj_resolution=optimizer.traj_resolution,
        destraj_resolution=optimizer.destraj_resolution,
    )

    opt_min_dist, opt_min_pos, opt_min_piece, opt_min_alpha = min_sdf_along_poly(
        opt_coeffs,
        opt_T,
        grid_map,
        traj_resolution=optimizer.traj_resolution,
        destraj_resolution=optimizer.destraj_resolution,
    )
    print(f"\n[Obstacle check] min ESDF distance along trajectory (optimizer-style sampling):")
    print(f"  before opt: {init_min_dist:.3f} m")
    print(f"  after  opt: {opt_min_dist:.3f} m")

    if opt_min_pos is not None:
        print(f"  argmin (opt): pos={opt_min_pos}, piece={opt_min_piece}, alpha={opt_min_alpha:.3f}")

    # 更严格的密集采样验收：确认不会在两采样点之间“钻”进障碍物/安全圈
    # 与 optimizer.setParam(obs_safe_threshold=...) 对齐
    dense = dense_clearance_check(
        opt_coeffs,
        opt_T,
        grid_map,
        safe_threshold=safe_th,
        K_dense=400,
        circle=demo_circle_map,
    )
    print(f"\n[Obstacle check] dense sampling (K={dense['K_dense']} per piece):")
    print(f"  min_sdf: {dense['min_sdf']:.3f} m at pos={dense['min_sdf_pos']}")
    print(f"  min(d-safe_threshold): {dense['min_clear_to_safe']:.3f} m (safe_threshold={safe_th:.2f}) at pos={dense['min_clear_pos']}")
    if dense['max_err_vs_analytic'] is not None:
        print(f"  analytic circle: min_d={dense['min_sdf_analytic']:.3f} m, max|grid-analytic|={dense['max_err_vs_analytic']:.3f} m")
    
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

    # Common plot inputs (map frame)
    sp = head_pva[0] + traj_to_map_offset
    gp = tail_pva[0] + traj_to_map_offset

    wc = None
    try:
        wc0 = dense.get("min_clear_pos", None)
        if wc0 is not None:
            wc = np.asarray(wc0, dtype=float) + traj_to_map_offset
    except Exception:
        wc = None
    
    if plot_mode == "show":
        print("\n绘制轨迹（仅保留 MINCO kinematics figures，并在 XY 子图叠加障碍物）...")

        # Figure A: initial kinematics
        fig0, axes0 = traj.plot_trajectory(
            dt=0.01,
            show_waypoints=True,
            show_velocity=True,
            show_acceleration=True,
        )
        try:
            ax_xy0 = axes0[0, 0]
            overlay_grid_map(ax_xy0, grid_map, safe_threshold=safe_th, circles=circles, rects=rects, polys=polys)
            ax_xy0.legend(loc="best")
        except Exception:
            pass

        # Figure B: optimized kinematics
        fig1, axes1 = opt_minco.plot_trajectory(
            dt=0.01,
            show_waypoints=True,
            show_velocity=True,
            show_acceleration=True,
        )
        try:
            ax_xy1 = axes1[0, 0]
            overlay_grid_map(ax_xy1, grid_map, safe_threshold=safe_th, circles=circles, rects=rects, polys=polys)
            # Mark worst-case (in map frame)
            if wc is not None:
                ax_xy1.plot(wc[0], wc[1], marker="x", color="k", markersize=10, mew=2, label="Worst-case")
            ax_xy1.legend(loc="best")
        except Exception:
            pass

        print("\n所有图形已显示，关闭窗口以结束程序...")
        plt.show()
    elif plot_mode == "save":
        print("\n[Plot] Save mode enabled (--plot-mode save).")

        # Headless: 仍然可以保存“地图坐标系”下的轨迹+障碍物
        if args.save_fig is not None:
            # If user passed a relative prefix, make it relative to this script directory
            save_prefix = str(args.save_fig)
            if not os.path.isabs(save_prefix):
                save_prefix = os.path.join(_this_dir, save_prefix)
            save_prefix = os.path.normpath(save_prefix)

            out_dir = os.path.dirname(save_prefix)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            init_path = f"{save_prefix}_init.png"
            opt_path = f"{save_prefix}_opt.png"
            save_mapframe_figure(
                minco_obj=traj,
                out_path=init_path,
                title="Initial Trajectory (map frame)",
                grid_map=grid_map,
                safe_threshold=safe_th,
                circle=demo_circle_map,
                start=sp,
                goal=gp,
                traj_to_map_offset=traj_to_map_offset,
                worst_case=None,
            )
            save_mapframe_figure(
                minco_obj=opt_minco,
                out_path=opt_path,
                title="Optimized Trajectory (map frame)",
                grid_map=grid_map,
                safe_threshold=safe_th,
                circle=demo_circle_map,
                start=sp,
                goal=gp,
                traj_to_map_offset=traj_to_map_offset,
                worst_case=wc,
            )
            print(f"[Plot] Saved: {init_path}")
            print(f"[Plot] Saved: {opt_path}")
    else:
        # Should never happen due to argparse choices, keep as a safe fallback.
        print(f"\n[Plot] Unknown plot_mode={plot_mode!r}, skipping plotting.")
    
    print("\n" + "=" * 60)