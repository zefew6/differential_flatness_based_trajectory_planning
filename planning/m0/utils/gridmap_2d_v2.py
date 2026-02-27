"""gridmap_2d_v2.py

在 gridmap_2d.py 中 GridMap 的基础上扩展，添加：
  - 2D ESDF（欧式有符号距离场）：自由空间为正，障碍内部为负
  - get_distance / get_distance_and_gradient：双线性插值连续查询

GridMap2D 继承 GridMap，所以：
  - 构造参数与 GridMap 完全相同（mujoco model/data + resolution/width/height/...）
  - 可以直接传给 A*（graph_search），因为 grid / coor_to_index / is_valid_index /
    is_occupied_index 全部继承自父类，无需任何改动
  - 也可以传给 MINCO PolyTrajOptimizer.setGridMap()，提供 get_distance_and_gradient

坐标约定
--------
GridMap 使用 [0, width] x [0, height] 的世界坐标。
网格索引为 (row, col)，row=y 方向，col=x 方向（见 GridMap.coor_to_index）。
本类内部的 esdf 数组使用 esdf[col, row] 存储（x 优先），
对外查询接口接受 pos_xy = [x, y] 的世界坐标。
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from .gridmap_2d import GridMap


class GridMap2D(GridMap):
    """GridMap 的扩展版本，增加 ESDF 和连续距离/梯度查询。

    额外属性
    --------
    esdf : np.ndarray  shape (grid_width, grid_height)
        有符号距离场（米）。自由空间为正值，障碍内部为负值。
        在第一次调用 get_distance / get_distance_and_gradient / update_esdf 时懒加载。
    """

    def __init__(self, model, data, resolution, width, height, robot_radius, margin):
        # 调用父类构造，建立 self.grid (grid_height × grid_width)
        super().__init__(model, data, resolution, width, height, robot_radius, margin)

        # ESDF 数组：懒初始化（第一次查询时计算）
        # shape: (grid_width, grid_height) = (x_cells, y_cells)
        self.esdf: np.ndarray | None = None
        self._esdf_valid: bool = False

    # ------------------------------------------------------------------
    # ESDF 构建
    # ------------------------------------------------------------------

    def update_esdf(self) -> None:
        """从当前 self.grid 计算有符号距离场并存入 self.esdf。

        self.grid shape = (grid_height, grid_width)，值为 1（占据）或 0（自由）。
        self.esdf shape = (grid_width, grid_height)，即 (x_cells, y_cells)，
        使用 esdf[col, row] 索引，与世界坐标 x=col*res, y=row*res 对应。
        """
        # 转换为 (x_cells, y_cells)，即 (col, row) 索引
        occ_xy = np.asarray(self.grid, dtype=np.int8).T  # shape: (grid_width, grid_height)
        nx, ny = occ_xy.shape  # nx=grid_width, ny=grid_height

        # ---- EDT：Felzenszwalb 1D 逐维扫描 ----

        def _edt1d_sq(f: np.ndarray) -> np.ndarray:
            """1D squared EDT（Felzenszwalb 2004）。f[i]=0 为特征点，其余为 +inf。"""
            n = len(f)
            if not np.isfinite(f).any():
                return np.full(n, np.inf, dtype=np.float64)
            v = np.zeros(n, dtype=np.int32)
            z = np.zeros(n + 1, dtype=np.float64)
            d = np.zeros(n, dtype=np.float64)
            # 找第一个有限点作为初始
            first = int(np.argmax(np.isfinite(f)))
            v[0] = first
            z[0] = -np.inf
            z[1] = np.inf
            k = 0
            for q in range(first + 1, n):
                if not np.isfinite(f[q]):
                    continue
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
                while s <= z[k]:
                    k -= 1
                    s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = np.inf
            k = 0
            for q in range(n):
                while z[k + 1] < q:
                    k += 1
                i = v[k]
                d[q] = (q - i) * (q - i) + f[i]
            return d

        def _edt2d(feature_mask: np.ndarray) -> np.ndarray:
            """2D squared EDT。feature_mask=True 处距离为 0。"""
            if not feature_mask.any():
                return np.full((nx, ny), np.inf, dtype=np.float64)
            INF = np.inf
            # Stage 1: along x (axis 0)
            gx = np.empty((nx, ny), dtype=np.float64)
            for j in range(ny):
                f = np.where(feature_mask[:, j], 0.0, INF)
                gx[:, j] = _edt1d_sq(f)
            # Stage 2: along y (axis 1)
            d2 = np.empty((nx, ny), dtype=np.float64)
            for i in range(nx):
                d2[i, :] = _edt1d_sq(gx[i, :])
            return d2

        # 自由→障碍 正距离（free cell 到最近障碍的距离）
        d2_pos = _edt2d(occ_xy == 1)
        # 障碍→自由 负距离（occupied cell 到最近自由区域的距离）
        d2_neg = _edt2d(occ_xy == 0)

        res = float(self.resolution)
        dist_pos = np.sqrt(d2_pos) * res
        dist_neg = np.sqrt(d2_neg) * res

        # signed distance: free=正, occupied=负（边界 ≈ 0）
        esdf = dist_pos.copy()
        occupied = occ_xy == 1
        esdf[occupied] = -dist_neg[occupied] + res

        self.esdf = esdf
        self._esdf_valid = True

    def _ensure_esdf(self) -> None:
        if not self._esdf_valid or self.esdf is None:
            self.update_esdf()

    # ------------------------------------------------------------------
    # 连续查询接口（双线性插值）
    # ------------------------------------------------------------------

    def get_distance(self, pos_xy) -> float:
        """查询世界坐标 pos_xy=[x,y] 处的 signed distance（米）。"""
        self._ensure_esdf()
        x, y = float(pos_xy[0]), float(pos_xy[1])

        # 边界外认为安全（距离很大）
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return float("inf")

        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)

        # 计算所在 cell（以 cell 中心为参考做双线性插值）
        # cell center: cx = (col+0.5)*res, cy = (row+0.5)*res
        # => col_f = x/res - 0.5, row_f = y/res - 0.5
        col_f = x / res - 0.5
        row_f = y / res - 0.5
        col0 = int(np.clip(int(col_f), 0, nx - 2))
        row0 = int(np.clip(int(row_f), 0, ny - 2))

        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)

        # esdf[col, row]
        v00 = float(self.esdf[col0,     row0    ])
        v10 = float(self.esdf[col0 + 1, row0    ])
        v01 = float(self.esdf[col0,     row0 + 1])
        v11 = float(self.esdf[col0 + 1, row0 + 1])

        return float(
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )

    def get_distance_and_gradient(self, pos_xy) -> Tuple[float, np.ndarray]:
        """查询 (distance, gradient)；gradient = [dd/dx, dd/dy]。"""
        self._ensure_esdf()
        x, y = float(pos_xy[0]), float(pos_xy[1])

        if x < 0 or x > self.width or y < 0 or y > self.height:
            return float("inf"), np.zeros(2, dtype=np.float64)

        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)

        col_f = x / res - 0.5
        row_f = y / res - 0.5
        col0 = int(np.clip(int(col_f), 0, nx - 2))
        row0 = int(np.clip(int(row_f), 0, ny - 2))

        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)

        v00 = float(self.esdf[col0,     row0    ])
        v10 = float(self.esdf[col0 + 1, row0    ])
        v01 = float(self.esdf[col0,     row0 + 1])
        v11 = float(self.esdf[col0 + 1, row0 + 1])

        dist = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )

        # 梯度（对双线性函数求偏导，再除以 res 换算到世界坐标单位）
        grad_x = ((v10 - v00) * (1 - dy) + (v11 - v01) * dy) / res
        grad_y = ((v01 - v00) * (1 - dx) + (v11 - v10) * dx) / res

        return float(dist), np.array([grad_x, grad_y], dtype=np.float64)
