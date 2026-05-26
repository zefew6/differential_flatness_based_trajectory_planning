"""Velocity and acceleration feasibility cost for MINCO trajectories."""

from __future__ import annotations

import numpy as np


class FeasibilityConstraint:
    """Penalty term for velocity and acceleration limit violations."""

    def __init__(self, max_vel, max_acc, traj_resolution=16, destraj_resolution=32):
        self.max_vel = float(max_vel)
        self.max_acc = float(max_acc)
        self.traj_resolution = int(traj_resolution)
        self.destraj_resolution = int(destraj_resolution)

        self.gdC = None
        self.gdT = 0.0
        self.vel_cost = 0.0
        self.acc_cost = 0.0
        self.total_cost = 0.0

    def reset(self):
        self.gdC = None
        self.gdT = 0.0
        self.vel_cost = 0.0
        self.acc_cost = 0.0
        self.total_cost = 0.0

    @staticmethod
    def positiveSmoothedL1(x):
        pe = 1e-4
        half = 0.5 * pe
        f3c = 1.0 / (pe * pe)
        f4c = -0.5 * f3c / pe
        d2c = 3.0 * f3c
        d3c = 4.0 * f4c
        large_threshold = 0.1

        if x < pe:
            f = (f4c * x + f3c) * x * x * x
            df = (d3c * x + d2c) * x * x
        elif x < large_threshold:
            f = x - half
            df = 1.0
        else:
            delta = x - large_threshold
            sqrt_delta = np.sqrt(delta)
            f = large_threshold - half + sqrt_delta
            df = 0.5 / sqrt_delta
        return f, df

    def addPVAGradCost(self, coeffs, T, piece_num, wei_feas):
        """Accumulate cost and gradients for a coefficient matrix."""

        self.gdC = np.zeros_like(coeffs)
        self.gdT = np.zeros(piece_num)
        self.vel_cost = 0.0
        self.acc_cost = 0.0

        pe = 1e-4
        half = 0.5 * pe
        f3c = 1.0 / (pe * pe)
        f4c = -0.5 * f3c / pe
        d2c = 3.0 * f3c
        d3c = 4.0 * f4c
        large_threshold = 0.1

        def ps_l1_vec(x):
            f1 = (f4c * x + f3c) * x * x * x
            df1 = (d3c * x + d2c) * x * x
            f2 = x - half
            df2 = np.ones_like(x)
            delta = np.maximum(x - large_threshold, 0.0)
            sqrt_delta = np.sqrt(np.where(delta > 0, delta, 1e-32))
            f3 = large_threshold - half + sqrt_delta
            df3 = 0.5 / sqrt_delta
            f = np.where(x < pe, f1, np.where(x < large_threshold, f2, f3))
            df = np.where(x < pe, df1, np.where(x < large_threshold, df2, df3))
            return f, df

        for i in range(piece_num):
            K = self.destraj_resolution if (i == 0 or i == piece_num - 1) else self.traj_resolution
            c = coeffs[6 * i : 6 * (i + 1), :]
            T_i = float(T[i])
            step = T_i / K

            js = np.arange(K + 1, dtype=np.float64)
            s1 = step * js
            s2 = s1 * s1
            s3 = s2 * s1
            s4 = s2 * s2

            beta1 = np.stack(
                [np.zeros(K + 1), np.ones(K + 1), 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4],
                axis=1,
            )
            beta2 = np.stack(
                [
                    np.zeros(K + 1),
                    np.zeros(K + 1),
                    2.0 * np.ones(K + 1),
                    6.0 * s1,
                    12.0 * s2,
                    20.0 * s3,
                ],
                axis=1,
            )
            beta3 = np.stack(
                [
                    np.zeros(K + 1),
                    np.zeros(K + 1),
                    np.zeros(K + 1),
                    6.0 * np.ones(K + 1),
                    24.0 * s1,
                    60.0 * s2,
                ],
                axis=1,
            )

            vel_all = beta1 @ c
            acc_all = beta2 @ c
            jerk_all = beta3 @ c

            omg_vec = np.ones(K + 1)
            omg_vec[0] = 0.5
            omg_vec[K] = 0.5
            alpha_vec = js / K

            vel_sq = np.sum(vel_all**2, axis=1)
            viola_v = vel_sq - self.max_vel**2
            active_v = viola_v > 0.0
            if np.any(active_v):
                f_v, df_v = ps_l1_vec(viola_v)
                f_v = np.where(active_v, f_v, 0.0)
                df_v = np.where(active_v, df_v, 0.0)

                wf_v = omg_vec * step * wei_feas
                self.vel_cost += float(np.dot(wf_v, f_v))

                wd_v = wf_v * df_v * 2.0
                self.gdC[6 * i : 6 * (i + 1), :] += beta1.T @ (wd_v[:, None] * vel_all)

                cross_va = np.sum(vel_all * acc_all, axis=1)
                self.gdT[i] += float(
                    np.dot(omg_vec, wei_feas * (df_v * 2.0 * alpha_vec * cross_va * step + f_v / K))
                )

            acc_sq = np.sum(acc_all**2, axis=1)
            viola_a = acc_sq - self.max_acc**2
            active_a = viola_a > 0.0
            if np.any(active_a):
                f_a, df_a = ps_l1_vec(viola_a)
                f_a = np.where(active_a, f_a, 0.0)
                df_a = np.where(active_a, df_a, 0.0)

                wf_a = omg_vec * step * wei_feas
                self.acc_cost += float(np.dot(wf_a, f_a))

                wd_a = wf_a * df_a * 2.0
                self.gdC[6 * i : 6 * (i + 1), :] += beta2.T @ (wd_a[:, None] * acc_all)

                cross_aj = np.sum(acc_all * jerk_all, axis=1)
                self.gdT[i] += float(
                    np.dot(omg_vec, wei_feas * (df_a * 2.0 * alpha_vec * cross_aj * step + f_a / K))
                )

        self.total_cost = self.vel_cost + self.acc_cost
        return self.total_cost

    def get_gdC(self):
        return self.gdC

    def get_gdT(self):
        return self.gdT

    def get_vel_cost(self):
        return self.vel_cost

    def get_acc_cost(self):
        return self.acc_cost
