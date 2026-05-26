"""Path preprocessing helpers used before MINCO optimization."""

from __future__ import annotations

import numpy as np


def check_line_collision(grid_map, p1, p2) -> bool:
    """Bresenham grid check. Returns True if the segment touches occupancy."""

    r1, c1 = grid_map.coor_to_index(p1)
    r2, c2 = grid_map.coor_to_index(p2)
    dr, dc = abs(r2 - r1), abs(c2 - c1)
    sr = 1 if r2 > r1 else -1
    sc = 1 if c2 > c1 else -1
    err = dr - dc
    r, c = r1, c1
    while True:
        idx = (r, c)
        if grid_map.is_valid_index(idx) and grid_map.is_occupied_index(idx):
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


def preprocess_path(grid_map, path) -> np.ndarray:
    """Visibility-prune an A* path while preserving endpoints."""

    path = np.asarray(path)
    if len(path) <= 2:
        return path
    pruned = [path[0]]
    prev = path[0]
    i = 1
    while i < len(path) - 1:
        if not check_line_collision(grid_map, prev, path[i + 1]):
            i += 1
        else:
            pruned.append(path[i])
            prev = path[i]
            i += 1
    pruned.append(path[-1])
    return np.array(pruned)


def resample_path(path, max_seg_len: float = 1.5, dense_path: np.ndarray | None = None) -> np.ndarray:
    """Resample a path so each segment is no longer than ``max_seg_len``."""

    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path

    if dense_path is not None:
        dense_path = np.asarray(dense_path, dtype=float)
        result = [path[0]]
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len > max_seg_len:
                idx0 = int(np.argmin(np.linalg.norm(dense_path - p0, axis=1)))
                idx1 = int(np.argmin(np.linalg.norm(dense_path - p1, axis=1)))
                if idx0 > idx1:
                    idx0, idx1 = idx1, idx0
                segment = dense_path[idx0 : idx1 + 1]
                if len(segment) > 2:
                    n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
                    arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(segment, axis=0), axis=1)])
                    total_arc = arc[-1]
                    for k in range(1, n_insert + 1):
                        s = total_arc * k / (n_insert + 1)
                        j = np.searchsorted(arc, s, side="right") - 1
                        j = min(j, len(segment) - 2)
                        alpha = (s - arc[j]) / (arc[j + 1] - arc[j] + 1e-12)
                        result.append(segment[j] + alpha * (segment[j + 1] - segment[j]))
            result.append(p1)
        return np.array(result)

    result = [path[0]]
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len > max_seg_len:
            n_insert = int(np.ceil(seg_len / max_seg_len)) - 1
            for k in range(1, n_insert + 1):
                alpha = k / (n_insert + 1)
                result.append(p0 + alpha * (p1 - p0))
        result.append(p1)
    return np.array(result)


def uniform_resample_path(path: np.ndarray, max_seg_len: float = 3.0) -> np.ndarray:
    """Uniformly sample along path arc length."""

    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path

    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_len = float(np.sum(seg_lens))
    piece_nums = max(int(total_len / max_seg_len + 0.5), 2)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    sample_dists = np.linspace(0.0, total_len, piece_nums + 1)

    result = []
    for d in sample_dists:
        idx = int(np.searchsorted(cum_dist, d, side="right")) - 1
        idx = int(np.clip(idx, 0, len(path) - 2))
        denom = cum_dist[idx + 1] - cum_dist[idx]
        alpha = (d - cum_dist[idx]) / (denom + 1e-12)
        result.append(path[idx] + alpha * (path[idx + 1] - path[idx]))
    return np.array(result)


def push_waypoints_to_clearance(
    grid_map,
    waypoints: np.ndarray,
    max_iters: int = 30,
    step_size: float = 0.08,
    target_clearance: float = 0.30,
) -> np.ndarray:
    """Move interior waypoints along ESDF gradients until clearance improves."""

    pts = np.asarray(waypoints, dtype=float).copy()
    for k in range(1, len(pts) - 1):
        for _ in range(max_iters):
            dist, grad = grid_map.get_distance_and_gradient(pts[k])
            if dist >= target_clearance:
                break
            gnorm = float(np.linalg.norm(grad))
            if gnorm < 1e-8:
                break
            pts[k] += step_size * grad / gnorm
    return pts
