"""ESDF obstacle cost for MINCO trajectories."""

from __future__ import annotations

import numpy as np


class ObstacleConstraint:
    """Static obstacle penalty sampled along a polynomial trajectory."""

    def __init__(
        self,
        safe_threshold: float = 0.5,
        wei_obs: float = 1000.0,
        traj_resolution: int = 16,
        destraj_resolution: int = 32,
        quad_threshold: float = 0.1,
        dist_cap: float = 5.0,
    ):
        self.safe_threshold = float(safe_threshold)
        self.wei_obs = float(wei_obs)
        self.traj_resolution = int(traj_resolution)
        self.destraj_resolution = int(destraj_resolution)
        self.quad_threshold = float(quad_threshold)
        self.dist_cap = float(dist_cap)

        self.gdC = None
        self.gdT = None
        self.obs_cost = 0.0
        self.min_dist = np.inf

    def reset(self, coeffs: np.ndarray, piece_num: int):
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num, dtype=np.float64)
        self.obs_cost = 0.0
        self.min_dist = np.inf

    def addObstacleGradCost(
        self,
        coeffs: np.ndarray,
        T: np.ndarray,
        piece_num: int,
        grid_map,
    ):
        """Accumulate ESDF collision cost and gradients."""

        self.reset(coeffs, piece_num)
        use_batch = hasattr(grid_map, "get_distance_and_gradient_batch")

        for i in range(piece_num):
            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution
            c = coeffs[6 * i : 6 * (i + 1), :]
            T_i = float(T[i])
            if T_i <= 0 or not np.isfinite(T_i):
                return 1e10

            step = T_i / K
            js = np.arange(K + 1, dtype=np.float64)
            s1 = step * js
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2
            s5 = s4 * s1
            beta0 = np.stack([np.ones(K + 1), s1, s2, s3, s4, s5], axis=1)
            beta1 = np.stack(
                [np.zeros(K + 1), np.ones(K + 1), 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4],
                axis=1,
            )
            pos_all = beta0 @ c
            vel_all = beta1 @ c

            omg_vec = np.ones(K + 1)
            omg_vec[0] = 0.5
            omg_vec[K] = 0.5
            alpha_vec = js / K

            if use_batch:
                dists, grad_sfds = grid_map.get_distance_and_gradient_batch(pos_all)
            else:
                dists = np.empty(K + 1)
                grad_sfds = np.zeros((K + 1, 2))
                for j in range(K + 1):
                    d, g = grid_map.get_distance_and_gradient(pos_all[j])
                    dists[j] = d
                    grad_sfds[j] = g

            finite = np.isfinite(dists)
            if np.any(finite):
                self.min_dist = min(self.min_dist, float(np.min(dists[finite])))

            violas = self.safe_threshold - dists
            active = (violas > 0.0) & (dists < self.dist_cap)
            if not np.any(active):
                continue

            penalty = np.where(violas < self.quad_threshold, violas * violas, violas)
            penaD = np.where(violas < self.quad_threshold, 2.0 * violas, np.ones_like(violas))
            penalty = np.where(active, penalty, 0.0)
            penaD = np.where(active, penaD, 0.0)

            cost_c = self.wei_obs * penalty
            grad_p = self.wei_obs * penaD[:, None] * (-grad_sfds)
            weights = omg_vec * step

            self.obs_cost += float(np.dot(weights, cost_c))
            self.gdC[6 * i : 6 * (i + 1), :] += beta0.T @ (weights[:, None] * grad_p)

            dot_gv = np.sum(grad_p * vel_all, axis=1)
            self.gdT[i] += float(np.dot(omg_vec, cost_c / K + step * alpha_vec * dot_gv))

        return float(self.obs_cost)

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_obs_cost(self):
        return float(self.obs_cost)
