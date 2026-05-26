"""Initial segment-time allocation for MINCO paths."""

from __future__ import annotations

import numpy as np


def trapezoid_duration(length: float, max_vel: float, max_acc: float, min_time: float) -> float:
    if length < 1e-6:
        return float(min_time)
    critical_len = max_vel * max_vel / max_acc
    if length >= critical_len:
        return float(2.0 * max_vel / max_acc + (length - critical_len) / max_vel)
    v_peak = np.sqrt(max_acc * length)
    return float(2.0 * v_peak / max_acc)


def allocate_time(waypoints, max_vel: float, max_acc: float, min_time: float) -> np.ndarray:
    waypoints = np.asarray(waypoints)
    dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    durations = np.array([trapezoid_duration(d, max_vel, max_acc, min_time) for d in dists])
    return np.maximum(durations, min_time)
