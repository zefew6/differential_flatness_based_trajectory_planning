"""Safe corridor construction and visualization helpers."""

from __future__ import annotations

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
        [1.0, 0.0, xmax],
        [-1.0, 0.0, -xmin],
        [0.0, 1.0, ymax],
        [0.0, -1.0, -ymin],
    ]


def _idx_in_range(row: int, col: int, ny: int, nx: int) -> bool:
    return 0 <= row < ny and 0 <= col < nx


def _rect_is_free(occ: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    if r0 > r1 or c0 > c1:
        return False
    return not np.any(occ[r0 : r1 + 1, c0 : c1 + 1] > 0)


def _find_nearest_free_cell(occ: np.ndarray, r: int, c: int, max_radius: int = 8):
    ny, nx = occ.shape
    if _idx_in_range(r, c, ny, nx) and occ[r, c] <= 0:
        return r, c

    for rad in range(1, max_radius + 1):
        rr0 = max(0, r - rad)
        rr1 = min(ny - 1, r + rad)
        cc0 = max(0, c - rad)
        cc1 = min(nx - 1, c + rad)
        for cc in range(cc0, cc1 + 1):
            for rr in (rr0, rr1):
                if occ[rr, cc] <= 0:
                    return rr, cc
        for rr in range(rr0 + 1, rr1):
            for cc in (cc0, cc1):
                if occ[rr, cc] <= 0:
                    return rr, cc
    return None


def _inflate_rect_4dirs(
    occ: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    max_expand_cells: int,
    inflate_step_cells: int = 1,
):
    ny, nx = occ.shape
    up_done = down_done = left_done = right_done = False
    expanded = 0

    while not (up_done and down_done and left_done and right_done):
        if expanded >= max_expand_cells:
            break

        if not up_done:
            grew = False
            for _ in range(inflate_step_cells):
                nr1 = r1 + 1
                if nr1 >= ny or np.any(occ[nr1, c0 : c1 + 1] > 0):
                    up_done = True
                    break
                r1 = nr1
                expanded += 1
                grew = True
                if expanded >= max_expand_cells:
                    break
            if not grew and not up_done:
                up_done = True

        if not down_done:
            grew = False
            for _ in range(inflate_step_cells):
                nr0 = r0 - 1
                if nr0 < 0 or np.any(occ[nr0, c0 : c1 + 1] > 0):
                    down_done = True
                    break
                r0 = nr0
                expanded += 1
                grew = True
                if expanded >= max_expand_cells:
                    break
            if not grew and not down_done:
                down_done = True

        if not right_done:
            grew = False
            for _ in range(inflate_step_cells):
                nc1 = c1 + 1
                if nc1 >= nx or np.any(occ[r0 : r1 + 1, nc1] > 0):
                    right_done = True
                    break
                c1 = nc1
                expanded += 1
                grew = True
                if expanded >= max_expand_cells:
                    break
            if not grew and not right_done:
                right_done = True

        if not left_done:
            grew = False
            for _ in range(inflate_step_cells):
                nc0 = c0 - 1
                if nc0 < 0 or np.any(occ[r0 : r1 + 1, nc0] > 0):
                    left_done = True
                    break
                c0 = nc0
                expanded += 1
                grew = True
                if expanded >= max_expand_cells:
                    break
            if not grew and not left_done:
                left_done = True

    return r0, r1, c0, c1


def _rect_idx_to_hpoly(grid_map, r0: int, r1: int, c0: int, c1: int):
    res = float(grid_map.resolution)
    xmin = float(grid_map.origin_x + c0 * res)
    xmax = float(grid_map.origin_x + (c1 + 1) * res)
    ymin = float(grid_map.origin_y + r0 * res)
    ymax = float(grid_map.origin_y + (r1 + 1) * res)
    return np.array(
        [
            [1.0, 0.0, xmax],
            [-1.0, 0.0, -xmin],
            [0.0, 1.0, ymax],
            [0.0, -1.0, -ymin],
        ],
        dtype=np.float64,
    )


def build_corridor_for_segment(p0, p1, obs_pts, search_radius=6.0, n_bins=_N_BINS, map_bounds=None):
    """Build a legacy angular-bin half-plane corridor for one segment."""

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
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
            for q, nearest_pt, d in best.values():
                if d < 1e-6:
                    continue
                normal = (q - nearest_pt) / d
                offset = float(np.dot(normal, q))
                if np.dot(normal, center) <= offset + 1e-6:
                    half_planes.append([normal[0], normal[1], offset])

    for hp in _map_bounds_to_hpoly(map_bounds, center):
        normal = np.array(hp[:2])
        offset = hp[2]
        if np.dot(normal, center) <= offset + 1e-6:
            half_planes.append(list(hp))

    if not half_planes:
        return np.array(_map_bounds_to_hpoly(map_bounds, center), dtype=np.float64)
    return np.array(half_planes, dtype=np.float64)


def build_corridors(
    waypoints,
    obs_pts,
    search_radius=6.0,
    n_bins=_N_BINS,
    map_bounds=None,
    traj_resolution=16,
    destraj_resolution=32,
    flip_radius=100.0,
):
    waypoints = np.asarray(waypoints, dtype=float)
    piece_num = len(waypoints) - 1
    if piece_num <= 0:
        return []
    return [
        build_corridor_for_segment(
            waypoints[i],
            waypoints[i + 1],
            obs_pts,
            search_radius=search_radius,
            n_bins=n_bins,
            map_bounds=map_bounds,
        )
        for i in range(piece_num)
    ]


def build_corridors_inflated_cubes(grid_map, waypoints, search_radius: float = 6.0, inflate_step_cells: int = 1) -> list:
    """Build axis-aligned inflated-box corridors from occupancy cells."""

    waypoints = np.asarray(waypoints, dtype=float)
    piece_num = len(waypoints) - 1
    if piece_num <= 0:
        return []

    occ = np.asarray(grid_map.occ)
    ny, nx = occ.shape
    res = max(1e-6, float(grid_map.resolution))
    max_expand_cells = max(1, int(round(search_radius / res)))
    hpolys = []

    for i in range(piece_num):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        r0, c0 = grid_map.coor_to_index(p0)
        r1, c1 = grid_map.coor_to_index(p1)
        r0 = int(np.clip(r0, 0, ny - 1))
        r1 = int(np.clip(r1, 0, ny - 1))
        c0 = int(np.clip(c0, 0, nx - 1))
        c1 = int(np.clip(c1, 0, nx - 1))
        rl, ru = min(r0, r1), max(r0, r1)
        cl, cu = min(c0, c1), max(c0, c1)

        if not _rect_is_free(occ, rl, ru, cl, cu):
            mid = 0.5 * (p0 + p1)
            mr, mc = grid_map.coor_to_index(mid)
            mr = int(np.clip(mr, 0, ny - 1))
            mc = int(np.clip(mc, 0, nx - 1))
            free_idx = _find_nearest_free_cell(occ, mr, mc, max_radius=max(8, max_expand_cells // 2))
            if free_idx is None:
                center = mid
                map_bounds = (
                    float(grid_map.min_boundary[0]),
                    float(grid_map.min_boundary[1]),
                    float(grid_map.max_boundary[0]),
                    float(grid_map.max_boundary[1]),
                )
                hpolys.append(np.array(_map_bounds_to_hpoly(map_bounds, center), dtype=np.float64))
                continue
            rl = ru = int(free_idx[0])
            cl = cu = int(free_idx[1])

        rl, ru, cl, cu = _inflate_rect_4dirs(
            occ=occ,
            r0=rl,
            r1=ru,
            c0=cl,
            c1=cu,
            max_expand_cells=max_expand_cells,
            inflate_step_cells=max(1, int(inflate_step_cells)),
        )
        hpolys.append(_rect_idx_to_hpoly(grid_map, rl, ru, cl, cu))

    return hpolys


def extract_obs_points_from_gridmap(grid_map, subsample=1):
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


def build_sfc_from_gridmap(
    grid_map,
    waypoints,
    search_radius: float = 6.0,
    subsample: int = 2,
    n_bins: int = 36,
    method: str = "firi",
    inflate_step_cells: int = 1,
    firi_max_iter: int = 10,
    firi_convergence_rho: float = 0.02,
    robot_radius: float | None = None,
) -> list:
    """Build one SFC polytope per path segment.

    ``method`` may be ``"firi"``, ``"cube"``, or ``"legacy"``.
    """

    method_l = str(method).lower()
    if method_l == "cube":
        return build_corridors_inflated_cubes(
            grid_map=grid_map,
            waypoints=waypoints,
            search_radius=search_radius,
            inflate_step_cells=inflate_step_cells,
        )

    if method_l in ("firi", "rsi"):
        from .firi import build_firi_corridors

        return build_firi_corridors(
            grid_map=grid_map,
            waypoints=waypoints,
            search_radius=search_radius,
            max_iter=firi_max_iter,
            convergence_rho=firi_convergence_rho,
            robot_radius=robot_radius,
        )

    obs_pts = extract_obs_points_from_gridmap(grid_map, subsample=subsample)
    map_bounds = None
    if hasattr(grid_map, "min_boundary") and hasattr(grid_map, "max_boundary"):
        mn = grid_map.min_boundary
        mx = grid_map.max_boundary
        map_bounds = (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    return build_corridors(
        waypoints=waypoints,
        obs_pts=obs_pts,
        search_radius=search_radius,
        n_bins=n_bins,
        map_bounds=map_bounds,
    )


def _clip_polygon_with_halfplane(poly, normal, offset, eps=1e-9):
    if poly is None or len(poly) == 0:
        return []
    out = []
    pts = [tuple(p) for p in poly]
    count = len(pts)

    def inside(p):
        return (normal[0] * p[0] + normal[1] * p[1]) <= offset + eps

    for i in range(count):
        a = pts[i]
        bpt = pts[(i + 1) % count]
        a_in = inside(a)
        b_in = inside(bpt)
        denom = normal[0] * (bpt[0] - a[0]) + normal[1] * (bpt[1] - a[1])
        if a_in and b_in:
            out.append(bpt)
        elif a_in and not b_in:
            if abs(denom) > 1e-12:
                t = (offset - (normal[0] * a[0] + normal[1] * a[1])) / denom
                t = np.clip(t, 0.0, 1.0)
                out.append((a[0] + t * (bpt[0] - a[0]), a[1] + t * (bpt[1] - a[1])))
        elif (not a_in) and b_in:
            if abs(denom) > 1e-12:
                t = (offset - (normal[0] * a[0] + normal[1] * a[1])) / denom
                t = np.clip(t, 0.0, 1.0)
                out.append((a[0] + t * (bpt[0] - a[0]), a[1] + t * (bpt[1] - a[1])))
            out.append(bpt)
    return out


def _halfplanes_to_convex_polygon(hPoly, clip_box=None, fallback_radius=100.0):
    if hPoly is None or len(hPoly) == 0:
        return []
    hp = np.asarray(hPoly, dtype=np.float64)
    if clip_box is not None:
        xmin, ymin, xmax, ymax = clip_box
    else:
        xmin = -fallback_radius
        ymin = -fallback_radius
        xmax = fallback_radius
        ymax = fallback_radius

    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    for plane in hp:
        poly = _clip_polygon_with_halfplane(poly, plane[:2], float(plane[2]))
        if not poly:
            return []
    return np.array(poly, dtype=np.float64)


def draw_sfc_corridors(
    hPolys_per_piece,
    grid_map=None,
    ax=None,
    face_color="cyan",
    edge_color="k",
    alpha=0.25,
    zorder=2,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:
        raise RuntimeError("matplotlib is required for draw_sfc_corridors") from exc

    if ax is None:
        _, ax = plt.subplots()

    clip_box = None
    if grid_map is not None and hasattr(grid_map, "min_boundary") and hasattr(grid_map, "max_boundary"):
        mn = grid_map.min_boundary
        mx = grid_map.max_boundary
        clip_box = (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    polys = []
    for hpoly in hPolys_per_piece:
        if hpoly is None:
            continue
        poly = _halfplanes_to_convex_polygon(hpoly, clip_box=clip_box)
        if poly is None or len(poly) == 0:
            continue
        patch = Polygon(poly, closed=True, facecolor=face_color, edgecolor=edge_color, alpha=alpha, zorder=zorder)
        ax.add_patch(patch)
        polys.append(poly)
    return polys
