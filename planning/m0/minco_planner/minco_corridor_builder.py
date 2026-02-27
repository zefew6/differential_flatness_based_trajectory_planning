"""
minco_corridor_builder.py
=========================
为每段轨迹生成凸多面体走廊（Safe Flight Corridor, SFC）。

算法：直接半平面法（Direct Separating Hyperplane）
--------------------------------------------------
对每段轨迹 p0 → p1：
  1. 找到该段 search_radius 范围内的所有障碍物点
  2. 对每个障碍物点 q，计算它到线段上最近点 nearest 的向量 d = q - nearest
     - 外法向量 n = d / |d|（从路径指向障碍物）
     - 半平面偏移 b = n^T q  （约束 n^T p <= b，轨迹需在此平面内侧）
  3. 为避免半平面数量爆炸，按方向角分成 N_BINS 个桶，
     每桶只保留距路径最近的障碍物（最严格的约束）
  4. 加入地图边界约束（4个半平面）

半平面格式：每行 [nx, ny, b]，约束 n^T p <= b（外法向量朝外）
"""

import numpy as np

_N_BINS = 36


def _nearest_pt_on_segment(p0, p1, pts):
    seg = p1 - p0
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 < 1e-12:
        nearest = np.tile(p0, (len(pts), 1))
        dists = np.linalg.norm(pts - p0, axis=1)
        return nearest, dists
    t = np.clip(((pts - p0) @ seg) / seg_len2, 0.0, 1.0)
    nearest = p0 + t[:, None] * seg
    dists = np.linalg.norm(pts - nearest, axis=1)
    return nearest, dists


def _map_bounds_to_hpoly(map_bounds, center, fallback_radius=5.0):
    if map_bounds is not None:
        xmin, ymin, xmax, ymax = map_bounds
    else:
        xmin = center[0] - fallback_radius
        xmax = center[0] + fallback_radius
        ymin = center[1] - fallback_radius
        ymax = center[1] + fallback_radius
    return [
        [ 1.0,  0.0,  xmax],
        [-1.0,  0.0, -xmin],
        [ 0.0,  1.0,  ymax],
        [ 0.0, -1.0, -ymin],
    ]


def build_corridor_for_segment(p0, p1, obs_pts,
                                search_radius=6.0, n_bins=_N_BINS,
                                map_bounds=None):
    """为一段轨迹生成半平面走廊。返回 shape (K, 3) 的 hPoly。"""
    center = (p0 + p1) * 0.5
    half_planes = []

    if obs_pts is not None and len(obs_pts) > 0:
        nearest, dists = _nearest_pt_on_segment(p0, p1, obs_pts)
        mask = dists < search_radius
        nearby_obs = obs_pts[mask]
        nearby_nearest = nearest[mask]
        nearby_dists = dists[mask]

        if len(nearby_obs) > 0:
            dx = nearby_obs[:, 0] - center[0]
            dy = nearby_obs[:, 1] - center[1]
            angles = np.arctan2(dy, dx)
            bin_width = 2.0 * np.pi / n_bins
            bin_ids = ((angles + np.pi) / bin_width).astype(int) % n_bins

            best = {}
            for idx in range(len(nearby_obs)):
                bid = bin_ids[idx]
                d = nearby_dists[idx]
                if bid not in best or d < best[bid][2]:
                    best[bid] = (nearby_obs[idx], nearby_nearest[idx], d)

            for (q, nearest_pt, d) in best.values():
                if d < 1e-6:
                    continue
                n = (q - nearest_pt) / d
                b = float(np.dot(n, q))
                if np.dot(n, center) <= b + 1e-6:
                    half_planes.append([n[0], n[1], b])

    for hp in _map_bounds_to_hpoly(map_bounds, center):
        n = np.array(hp[:2])
        b = hp[2]
        if np.dot(n, center) <= b + 1e-6:
            half_planes.append(list(hp))

    if not half_planes:
        return np.array(_map_bounds_to_hpoly(map_bounds, center), dtype=np.float64)
    return np.array(half_planes, dtype=np.float64)


def build_corridors(waypoints, obs_pts,
                    search_radius=6.0, n_bins=_N_BINS, map_bounds=None,
                    traj_resolution=16, destraj_resolution=32, flip_radius=100.0):
    """为轨迹每段生成走廊，返回 list of hPoly（len = piece_num）。"""
    waypoints = np.asarray(waypoints, dtype=float)
    piece_num = len(waypoints) - 1
    if piece_num <= 0:
        return []
    return [
        build_corridor_for_segment(
            waypoints[i], waypoints[i + 1], obs_pts,
            search_radius=search_radius, n_bins=n_bins, map_bounds=map_bounds,
        )
        for i in range(piece_num)
    ]


def extract_obs_points_from_gridmap(grid_map, subsample=1):
    """从 GridMap2D 提取占据栅格的世界坐标（障碍物点云）。"""
    occ = np.asarray(grid_map.occ)
    ny, nx = occ.shape
    pts = []
    for r in range(0, ny, subsample):
        for c in range(0, nx, subsample):
            if occ[r, c] > 0:
                pts.append(grid_map.index_to_coor((r, c)))
    if not pts:
        return np.empty((0, 2))
    return np.array(pts, dtype=np.float64)
