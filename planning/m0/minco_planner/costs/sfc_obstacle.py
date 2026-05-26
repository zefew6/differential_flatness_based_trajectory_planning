"""Safe-flight-corridor half-plane cost for MINCO trajectories."""

from __future__ import annotations

import numpy as np


class SFCObstacleConstraint:
    """Penalty for leaving per-piece convex corridors.

    Each corridor is an array with rows ``[nx, ny, b]`` representing
    ``n.T @ p <= b``.
    """

    def __init__(
        self,
        safe_margin: float = 0.0,
        wei_sfc: float = 1e4,
        traj_resolution: int = 16,
        destraj_resolution: int = 32,
        quad_threshold: float = 0.1,
    ):
        self.safe_margin = float(safe_margin)
        self.wei_sfc = float(wei_sfc)
        self.traj_resolution = int(traj_resolution)
        self.destraj_resolution = int(destraj_resolution)
        self.quad_threshold = float(quad_threshold)

        self.hPolys_per_piece: list = []
        self.gdC = None
        self.gdT = None
        self.sfc_cost = 0.0
        self.min_dist = np.inf

    def set_corridors(self, hPolys_per_piece: list):
        self.hPolys_per_piece = hPolys_per_piece

    def reset(self, coeffs: np.ndarray, piece_num: int):
        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num)
        self.sfc_cost = 0.0
        self.min_dist = np.inf

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_sfc_cost(self):
        return float(self.sfc_cost)

    def get_obs_cost(self):
        return self.get_sfc_cost()

    def addObstacleGradCost(
        self,
        coeffs: np.ndarray,
        T: np.ndarray,
        piece_num: int,
        grid_map=None,
    ) -> float:
        self.reset(coeffs, piece_num)

        if not self.hPolys_per_piece or len(self.hPolys_per_piece) != piece_num:
            return 0.0

        for i in range(piece_num):
            hpoly = self.hPolys_per_piece[i]
            if hpoly is None or len(hpoly) == 0:
                continue

            hpoly = np.asarray(hpoly, dtype=np.float64)
            normals = hpoly[:, :2]
            bs = hpoly[:, 2]

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

            d_hp = bs[None, :] - pos_all @ normals.T
            d_min_per_pt = np.min(d_hp, axis=1)
            min_plane_idx = np.argmin(d_hp, axis=1)
            self.min_dist = min(self.min_dist, float(np.min(d_min_per_pt)))

            viola = self.safe_margin - d_min_per_pt
            active = viola > 0.0
            if not np.any(active):
                continue

            penalty = np.where(
                viola < self.quad_threshold,
                viola**2,
                self.quad_threshold * (2.0 * viola - self.quad_threshold),
            )
            penaD = np.where(viola < self.quad_threshold, 2.0 * viola, 2.0 * self.quad_threshold)
            penalty = np.where(active, penalty, 0.0)
            penaD = np.where(active, penaD, 0.0)

            cost_c = self.wei_sfc * penalty
            weights = omg_vec * step
            active_normals = normals[min_plane_idx]
            grad_p = self.wei_sfc * penaD[:, None] * active_normals

            self.sfc_cost += float(np.dot(weights, cost_c))
            self.gdC[6 * i : 6 * (i + 1), :] += beta0.T @ (weights[:, None] * grad_p)

            alpha_vec = js / K
            dot_gv = np.sum(grad_p * vel_all, axis=1)
            self.gdT[i] += float(np.dot(omg_vec, cost_c / K + step * alpha_vec * dot_gv))

        return float(self.sfc_cost)
