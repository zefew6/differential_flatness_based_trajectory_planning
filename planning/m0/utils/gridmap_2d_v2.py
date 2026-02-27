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
from dataclasses import dataclass, field
import numpy as np
from .gridmap_2d import GridMap


@dataclass
class GridMap2DParams:
    """无 MuJoCo 模型的 GridMap2D 构造参数。"""
    resolution: float = 0.1
    size_x: float = 20.0
    size_y: float = 20.0
    origin_at_center: bool = False   # False: 原点在左下角 [0,size_x]x[0,size_y]
    robot_radius: float = 0.0
    margin: float = 0.0


class GridMap2D(GridMap):
    """GridMap 的扩展版本，增加 ESDF 和连续距离/梯度查询。

    支持两种构造方式：
      1. GridMap2D(model, data, resolution, width, height, robot_radius, margin)  -- 原有方式
      2. GridMap2D(params: GridMap2DParams)  -- 无 MuJoCo，纯几何地图

    额外属性
    --------
    esdf : np.ndarray  shape (grid_width, grid_height)
        有符号距离场（米）。自由空间为正值，障碍内部为负值。
        在第一次调用 get_distance / get_distance_and_gradient / update_esdf 时懒加载。
    """

    def __init__(self, model_or_params=None, data=None, resolution=None,
                 width=None, height=None, robot_radius=None, margin=None,
                 *, model=None):
        if isinstance(model_or_params, GridMap2DParams):
            # ── 无 MuJoCo 构造路径 ──────────────────────────────────────
            p = model_or_params
            # 用哨兵值调父类（不传 mujoco model），父类 __init__ 会 create_grid()
            # 但 self.model.ngeom 会报错，所以我们绕过父类，手动初始化必要字段
            self.model = None
            self.data = None
            self.resolution = p.resolution
            self.width = p.size_x
            self.height = p.size_y
            self.grid_width = int(p.size_x / p.resolution)
            self.grid_height = int(p.size_y / p.resolution)
            self.robot_radius = p.robot_radius
            self.margin = p.margin
            self.inflation_radius = p.robot_radius + p.margin
            # 占据格初始化为全空闲
            self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            self._params = p
        else:
            # ── 原有 MuJoCo 构造路径 ────────────────────────────────────
            # 兼容两种调用方式：
            #   GridMap2D(model, data, res, w, h, rr, m)          位置参数
            #   GridMap2D(model=model, data=data, resolution=...) 关键字参数
            actual_model = model if model is not None else model_or_params
            super().__init__(actual_model, data, resolution, width, height,
                             robot_radius, margin)
            self._params = None

        # ESDF 数组：懒初始化（第一次查询时计算）
        # shape: (grid_width, grid_height) = (x_cells, y_cells)
        self.esdf: np.ndarray | None = None
        self._esdf_valid: bool = False

    # ------------------------------------------------------------------
    # 别名属性（供 minco_test.py 等外部代码使用）
    # ------------------------------------------------------------------

    @property
    def nx(self) -> int:
        """x 方向格子数（= grid_width）。"""
        return int(self.grid_width)

    @property
    def ny(self) -> int:
        """y 方向格子数（= grid_height）。"""
        return int(self.grid_height)

    @property
    def occ(self) -> np.ndarray:
        """占据格（grid_height × grid_width），1=障碍，0=自由。"""
        return self.grid

    @property
    def min_boundary(self):
        """地图左下角世界坐标 [x_min, y_min]。"""
        return np.array([0.0, 0.0])

    @property
    def max_boundary(self):
        """地图右上角世界坐标 [x_max, y_max]。"""
        return np.array([self.width, self.height])

    # ------------------------------------------------------------------
    # 占据格写入 / 障碍物添加
    # ------------------------------------------------------------------

    def set_occupancy(self, occ: np.ndarray, *, update_esdf: bool = True) -> None:
        """用外部数组覆盖占据格。occ shape: (nx, ny) 或 (grid_height, grid_width)。"""
        occ = np.asarray(occ, dtype=np.float32)
        if occ.shape == (self.grid_width, self.grid_height):
            # (nx, ny) → 转置为 (grid_height, grid_width)
            self.grid = occ.T.copy()
        elif occ.shape == (self.grid_height, self.grid_width):
            self.grid = occ.copy()
        else:
            raise ValueError(f"occ shape {occ.shape} 与地图尺寸 "
                             f"({self.grid_height},{self.grid_width}) 不匹配")
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_circle_obstacle(self, center: np.ndarray, radius: float,
                            *, update_esdf: bool = True) -> None:
        """在占据格中光栅化一个圆形障碍（含 inflation_radius 膨胀）。"""
        cx, cy = float(center[0]), float(center[1])
        r_total = radius + self.inflation_radius
        res = float(self.resolution)
        # 受影响的格子范围
        col_min = max(0, int((cx - r_total) / res))
        col_max = min(self.grid_width - 1,  int((cx + r_total) / res) + 1)
        row_min = max(0, int((cy - r_total) / res))
        row_max = min(self.grid_height - 1, int((cy + r_total) / res) + 1)

        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)  # shape (rows, cols)
        wx = (cc + 0.5) * res
        wy = (rr + 0.5) * res
        mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= r_total ** 2
        self.grid[row_min:row_max + 1, col_min:col_max + 1][mask] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_rectangle_obstacle(self, xmin: float, xmax: float,
                                ymin: float, ymax: float,
                                *, update_esdf: bool = True) -> None:
        """在占据格中光栅化一个轴对齐矩形（含 inflation_radius 膨胀）。"""
        inf = self.inflation_radius
        xmin -= inf; xmax += inf; ymin -= inf; ymax += inf
        res = float(self.resolution)
        col_min = max(0, int(xmin / res))
        col_max = min(self.grid_width - 1,  int(xmax / res) + 1)
        row_min = max(0, int(ymin / res))
        row_max = min(self.grid_height - 1, int(ymax / res) + 1)
        self.grid[row_min:row_max + 1, col_min:col_max + 1] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_polygon_obstacle(self, verts: np.ndarray, *,
                             update_esdf: bool = True) -> None:
        """在占据格中光栅化一个多边形障碍（不含自动膨胀）。"""
        verts = np.asarray(verts, dtype=np.float64)
        res = float(self.resolution)
        xs = verts[:, 0]; ys = verts[:, 1]
        col_min = max(0, int(xs.min() / res))
        col_max = min(self.grid_width - 1,  int(xs.max() / res) + 1)
        row_min = max(0, int(ys.min() / res))
        row_max = min(self.grid_height - 1, int(ys.max() / res) + 1)
        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        cc, rr = np.meshgrid(cols, rows)
        wx = (cc + 0.5) * res
        wy = (rr + 0.5) * res
        # ray-casting 判断点在多边形内
        n = len(verts)
        inside = np.zeros(wx.shape, dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]; xj, yj = verts[j]
            cond = ((yi > wy) != (yj > wy)) & \
                   (wx < (xj - xi) * (wy - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            j = i
        self.grid[row_min:row_max + 1, col_min:col_max + 1][inside] = 1.0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

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

    def get_distance_and_gradient_batch(self, positions: np.ndarray):
        """批量查询 ESDF 距离与梯度（向量化实现，比逐点调用快 10-50x）。

        参数
        ----
        positions : (N, 2) 数组，每行为 (x, y) 世界坐标

        返回
        ----
        distances : (N,) 数组，越界点返回 inf
        gradients : (N, 2) 数组，越界点返回 (0, 0)
        """
        self._ensure_esdf()
        positions = np.asarray(positions, dtype=np.float64)
        N = len(positions)
        x = positions[:, 0]
        y = positions[:, 1]

        distances = np.full(N, np.inf, dtype=np.float64)
        gradients = np.zeros((N, 2), dtype=np.float64)

        valid = (x >= 0) & (x <= self.width) & (y >= 0) & (y <= self.height)
        if not np.any(valid):
            return distances, gradients

        res = float(self.resolution)
        nx, ny = int(self.grid_width), int(self.grid_height)

        xv = x[valid]
        yv = y[valid]

        col_f = xv / res - 0.5
        row_f = yv / res - 0.5
        col0 = np.clip(col_f.astype(np.int32), 0, nx - 2)
        row0 = np.clip(row_f.astype(np.int32), 0, ny - 2)

        dx = np.clip(col_f - col0, 0.0, 1.0)
        dy = np.clip(row_f - row0, 0.0, 1.0)

        # 四角 ESDF 值（fancy indexing，全向量化）
        v00 = self.esdf[col0,     row0    ]
        v10 = self.esdf[col0 + 1, row0    ]
        v01 = self.esdf[col0,     row0 + 1]
        v11 = self.esdf[col0 + 1, row0 + 1]

        dist_v = (v00 * (1 - dx) * (1 - dy)
                + v10 * dx       * (1 - dy)
                + v01 * (1 - dx) * dy
                + v11 * dx       * dy)

        grad_x = ((v10 - v00) * (1 - dy) + (v11 - v01) * dy) / res
        grad_y = ((v01 - v00) * (1 - dx) + (v11 - v10) * dx) / res

        distances[valid] = dist_v
        gradients[valid, 0] = grad_x
        gradients[valid, 1] = grad_y

        return distances, gradients

