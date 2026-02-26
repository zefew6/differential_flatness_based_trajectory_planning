"""gridmap_2d_v2.py

目标
----
这个版本是在 `planning/m0/utils/gridmap_2d.py` 的基础上重写的 2D 栅格地图工具，
并“学习/对齐”了 ST-opt-tools 里的 `grid_map::GridMap` 能力：

- 统一世界坐标 <-> 栅格索引转换（posToIndex/indexToPos）
- 占据栅格（occ）
- 2D ESDF 距离场（signed distance：自由空间为正，障碍内部为负）
- 连续查询：get_distance + get_distance_and_gradient（双线性插值 + 梯度）

注意
----
- 本文件是新建文件，不会修改旧的 `gridmap_2d.py`。
- 为了不强绑定 mujoco，本实现把“从 mujoco model/data 抽取障碍物”做成可选能力。
  如果你仍要从 mujoco 构图，可以用 `from_mujoco(...)` 或自行把 occupancy 填进来。

与 ST-opt-tools 的关键差异
---------------------------
- C++ 版本 `GridMap::init()` 默认以地图中心为原点，边界是 [-size/2, size/2]。
  这里也采用同样约定。
- 旧版 gridmap_2d.py 使用 [0,width]x[0,height] 的坐标更常见。
  如果你想兼容旧坐标系，可用 `origin_at_center=False` 初始化（见 __init__）。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GridMap2DParams:
    resolution: float
    size_x: float
    size_y: float
    # 安全膨胀半径（和 ST-opt-tools 的 safe_threshold 配合用）
    inflation_radius: float = 0.0

    # 是否采用“地图中心为 (0,0)” 的坐标系
    origin_at_center: bool = True


class GridMap2D:
    """2D occupancy + ESDF grid map (numpy implementation).

    索引约定
    --------
    - index / id: (ix, iy) 对应 C++ 里 Eigen::Vector2i(x, y)
    - 内部 buffer 展开地址: addr = ix * ny + iy
    - occ shape: (nx, ny)

    世界坐标约定
    -----------
    - 若 origin_at_center=True:
        min_boundary = (-size_x/2, -size_y/2)
        max_boundary = ( size_x/2,  size_y/2)
        map_origin = min_boundary
      这与 ST-opt-tools `grid_map.hpp` 一致。

    - 若 origin_at_center=False:
        min_boundary = (0, 0)
        max_boundary = (size_x, size_y)
        map_origin = (0, 0)
      更贴近旧版 `gridmap_2d.py` 的默认坐标。
    """

    def __init__(self, params: GridMap2DParams):
        if params.resolution <= 0:
            raise ValueError("resolution must be positive")
        if params.size_x <= 0 or params.size_y <= 0:
            raise ValueError("size_x/size_y must be positive")

        self.params = params
        self.resolution = float(params.resolution)
        self.resolution_inv = 1.0 / self.resolution

        self.map_size = np.array([params.size_x, params.size_y], dtype=np.float64)

        if params.origin_at_center:
            self.min_boundary = -0.5 * self.map_size
            self.max_boundary = 0.5 * self.map_size
            self.map_origin = self.min_boundary.copy()
        else:
            self.min_boundary = np.array([0.0, 0.0], dtype=np.float64)
            self.max_boundary = self.map_size.copy()
            self.map_origin = self.min_boundary.copy()

        # voxel_num 对齐 C++：ceil(size/res)
        self.voxel_num = np.ceil(self.map_size / self.resolution).astype(np.int32)
        self.nx = int(self.voxel_num[0])
        self.ny = int(self.voxel_num[1])

        # occupancy: 1=occupied, 0=free
        self.occ = np.zeros((self.nx, self.ny), dtype=np.int8)

        # esdf: signed distance (m)
        self.esdf = np.zeros((self.nx, self.ny), dtype=np.float64)

        # 缓存：是否已经更新
        self._esdf_valid = False

    # -----------------------
    # 坐标转换 / 边界判断
    # -----------------------

    def is_in_map(self, pos_xy: np.ndarray) -> bool:
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        return bool(np.all(pos_xy >= self.min_boundary) and np.all(pos_xy <= self.max_boundary))

    def bound_index(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int32)
        idx[0] = int(np.clip(idx[0], 0, self.nx - 1))
        idx[1] = int(np.clip(idx[1], 0, self.ny - 1))
        return idx

    def pos_to_index(self, pos_xy: np.ndarray) -> np.ndarray:
        """pos -> (ix, iy)"""
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        ix = int((pos_xy[0] - self.map_origin[0]) * self.resolution_inv)
        iy = int((pos_xy[1] - self.map_origin[1]) * self.resolution_inv)
        return np.array([ix, iy], dtype=np.int32)

    def index_to_pos(self, idx: np.ndarray) -> np.ndarray:
        """(ix, iy) -> cell center pos"""
        idx = np.asarray(idx, dtype=np.int32)
        x = (float(idx[0]) + 0.5) * self.resolution + self.map_origin[0]
        y = (float(idx[1]) + 0.5) * self.resolution + self.map_origin[1]
        return np.array([x, y], dtype=np.float64)

    # 兼容旧版命名（但注意旧版坐标系默认不同）
    def coor_to_index(self, coor: np.ndarray) -> Tuple[int, int]:
        idx = self.pos_to_index(coor)
        # 旧版返回 (row, col)；这里明确返回 (ix, iy) 以对齐 ST-opt-tools
        return int(idx[0]), int(idx[1])

    def index_to_coor(self, ind: Tuple[int, int]) -> Tuple[float, float]:
        pos = self.index_to_pos(np.array(ind, dtype=np.int32))
        return float(pos[0]), float(pos[1])

    # -----------------------
    # 占据操作
    # -----------------------

    def get_xy_meshgrid(self):
        """Return X,Y meshgrid of cell-center coordinates.

        Returns
        -------
        X, Y : ndarray
            Shape (nx, ny). X[ix,iy], Y[ix,iy] are the world coordinates (meters)
            of the center of the corresponding cell.
        """
        xs = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.resolution + float(self.map_origin[0])
        ys = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.resolution + float(self.map_origin[1])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X, Y

    def add_circle_obstacle(
        self,
        center_xy: np.ndarray,
        radius: float,
        *,
        value: int = 1,
        update_esdf: bool = True,
    ) -> None:
        """Rasterize a circular obstacle into occupancy grid.

        Parameters
        ----------
        center_xy: (2,) array-like
            Circle center in world/map frame.
        radius: float
            Circle radius in meters.
        value: int
            Occupancy value to write. 1 means occupied; 0 can be used to carve free space.
        update_esdf: bool
            If True, recompute ESDF after modification.
        """
        center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2)
        radius = float(radius)
        if radius <= 0:
            raise ValueError("radius must be positive")

        X, Y = self.get_xy_meshgrid()
        mask = (X - center_xy[0]) ** 2 + (Y - center_xy[1]) ** 2 <= radius ** 2
        self.occ[mask] = 1 if int(value) != 0 else 0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def add_rectangle_obstacle(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        *,
        value: int = 1,
        update_esdf: bool = True,
    ) -> None:
        """Axis-aligned rectangle obstacle rasterization (AABB)."""
        xmin = float(xmin)
        xmax = float(xmax)
        ymin = float(ymin)
        ymax = float(ymax)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        X, Y = self.get_xy_meshgrid()
        mask = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
        self.occ[mask] = 1 if int(value) != 0 else 0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    @staticmethod
    def _points_in_polygon(points_xy: np.ndarray, verts_xy: np.ndarray) -> np.ndarray:
        """Vectorized even-odd rule test.

        points_xy: (N,2)
        verts_xy: (M,2) polygon vertices (closed or open)
        returns mask: (N,) bool
        """
        pts = np.asarray(points_xy, dtype=np.float64)
        verts = np.asarray(verts_xy, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError("verts must be (M,2) with M>=3")

        x = pts[:, 0]
        y = pts[:, 1]
        xv = verts[:, 0]
        yv = verts[:, 1]

        inside = np.zeros(len(pts), dtype=bool)
        j = len(verts) - 1
        for i in range(len(verts)):
            xi, yi = xv[i], yv[i]
            xj, yj = xv[j], yv[j]
            # edge intersects horizontal ray?
            intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside ^= intersect
            j = i
        return inside

    def add_polygon_obstacle(
        self,
        verts_xy: np.ndarray,
        *,
        value: int = 1,
        update_esdf: bool = True,
    ) -> None:
        """Rasterize a polygon obstacle (triangle is a 3-vertex polygon)."""
        verts_xy = np.asarray(verts_xy, dtype=np.float64)
        X, Y = self.get_xy_meshgrid()
        pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        mask_flat = self._points_in_polygon(pts, verts_xy)
        mask = mask_flat.reshape(self.nx, self.ny)
        self.occ[mask] = 1 if int(value) != 0 else 0
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def set_occupancy(self, occ: np.ndarray, *, update_esdf: bool = True) -> None:
        """直接设置占据栅格。

        参数
        ----
        occ: shape (nx, ny), 值为 {0,1}
        """
        occ = np.asarray(occ)
        if occ.shape != (self.nx, self.ny):
            raise ValueError(f"occ shape must be {(self.nx, self.ny)}, got {occ.shape}")
        self.occ = (occ.astype(np.int8) != 0).astype(np.int8)
        self._esdf_valid = False
        if update_esdf:
            self.update_esdf()

    def is_occupied(self, idx: np.ndarray) -> bool:
        idx = np.asarray(idx, dtype=np.int32)
        if idx[0] < 0 or idx[0] >= self.nx or idx[1] < 0 or idx[1] >= self.ny:
            return True
        return bool(self.occ[idx[0], idx[1]] == 1)

    def is_collision(self, pos_xy: np.ndarray, safe_threshold: float = 0.0) -> bool:
        if not self.is_in_map(pos_xy):
            return True
        return self.get_distance(pos_xy) < float(safe_threshold)

    # -----------------------
    # ESDF / 距离场
    # -----------------------

    def update_esdf(self) -> None:
        """计算 signed distance field。

        逻辑对齐 ST-opt-tools:
        - 先算 free->occupied 的最近距离（正距离）
        - 再算 occupied->free 的最近距离（负距离）
        - 最终：occupied 内为负，free 内为正；边界约在 0 附近

        实现说明（严格欧式 EDT）
        ----------------------
        这里使用 Felzenszwalb & Huttenlocher (2004) 的 1D squared distance transform
        扩展到 2D（先对 x 方向，再对 y 方向），得到严格欧式距离。

        - dist_pos: free cell 到最近 occupied cell 的距离（正距离）
        - dist_neg: occupied cell 到最近 free cell 的距离（用于生成负距离）
        - signed ESDF: free 区域为正，occupied 区域为负

        该实现更接近 ST-opt-tools 中 `grid_map::GridMap::updateESDF()` 的结果，
        并且梯度场更平滑，有助于 L-BFGS 等梯度优化稳定收敛。
        """

        def _edt_1d_squared(f: np.ndarray) -> np.ndarray:
            """Felzenszwalb 1D squared distance transform.

            输入
            ----
            f: 1D array, f[i]=0 for feature points, +inf for others

            输出
            ----
            d: 1D array, d[i] = min_j ( (i-j)^2 + f[j] )
            """
            n = int(f.shape[0])
            finite = np.isfinite(f)
            # If there is no feature point in this 1D slice, the distance is +inf everywhere.
            # (This can happen if the whole map is free or whole map is occupied.)
            if not finite.any():
                return np.full(n, np.inf, dtype=np.float64)

            v = np.zeros(n, dtype=np.int32)
            z = np.zeros(n + 1, dtype=np.float64)
            d = np.zeros(n, dtype=np.float64)

            k = 0
            # Initialize with the first finite site; this avoids (+inf) - (+inf) when
            # many entries are +inf.
            v[0] = int(np.argmax(finite))
            z[0] = -np.inf
            z[1] = np.inf

            def _sep(i: int, u: int) -> float:
                # separation where parabola i and u intersect
                return ((f[u] + u * u) - (f[i] + i * i)) / (2.0 * (u - i))

            for q in range(v[0] + 1, n):
                if not np.isfinite(f[q]):
                    continue
                s = _sep(v[k], q)
                while s <= z[k]:
                    k -= 1
                    s = _sep(v[k], q)
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

        def _edt_2d_squared(feature_mask: np.ndarray) -> np.ndarray:
            """2D squared Euclidean distance transform (grid units).

            feature_mask: True 表示 feature(距离为 0) 的集合。
            返回每个 cell 到最近 feature 的 squared distance（以格子为单位）。
            """
            feature_mask = np.asarray(feature_mask, dtype=bool)
            if not feature_mask.any():
                # No feature in entire map -> distance is +inf everywhere
                return np.full((self.nx, self.ny), np.inf, dtype=np.float64)
            inf = np.inf

            # Stage 1: along x
            gx = np.empty((self.nx, self.ny), dtype=np.float64)
            for y in range(self.ny):
                f = np.where(feature_mask[:, y], 0.0, inf)
                gx[:, y] = _edt_1d_squared(f)

            # Stage 2: along y
            d2 = np.empty((self.nx, self.ny), dtype=np.float64)
            for x in range(self.nx):
                f = gx[x, :]
                d2[x, :] = _edt_1d_squared(f)
            return d2

        # free -> occupied 的距离（对 occupied 作为 feature）
        d2_pos = _edt_2d_squared(self.occ == 1)
        # occupied -> free 的距离（对 free 作为 feature）
        d2_neg = _edt_2d_squared(self.occ == 0)

        dist_pos_m = np.sqrt(d2_pos) * self.resolution
        dist_neg_m = np.sqrt(d2_neg) * self.resolution

        # 6) signed：occupied -> negative
        #    参考 C++：esdf = -neg_dist + resolution (让边界附近稍偏移)
        esdf = dist_pos_m.copy()
        inside_mask = self.occ == 1
        esdf[inside_mask] = -dist_neg_m[inside_mask] + self.resolution

        # 7) 可选：膨胀（把距离场减去 inflation，使“有效障碍”更大）
        if self.params.inflation_radius > 0:
            esdf = esdf - self.params.inflation_radius

        self.esdf = esdf
        self._esdf_valid = True

    def _ensure_esdf(self) -> None:
        if not self._esdf_valid:
            self.update_esdf()

    # -----------------------
    # 查询：distance + gradient（双线性插值对齐 C++ 思路）
    # -----------------------

    def get_distance(self, pos_xy: np.ndarray) -> float:
        """连续位置查询 signed distance（米）。

        对齐 C++ 版本做法：
        - 把 pos 平移 (0.5*res, 0.5*res)，并找到其左下角 cell idx
        - 对 esdf 的 2x2 邻域做双线性插值
        """
        if not self.is_in_map(pos_xy):
            return float("inf")
        self._ensure_esdf()

        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        pos_m = pos_xy.copy()
        pos_m[0] -= 0.5 * self.resolution
        pos_m[1] -= 0.5 * self.resolution

        idx = self.pos_to_index(pos_m)
        idx = self.bound_index(idx)
        idx_pos = self.index_to_pos(idx)

        diff = (pos_xy - idx_pos) * self.resolution_inv

        # 2x2 values
        def v(ix: int, iy: int) -> float:
            ix = int(np.clip(ix, 0, self.nx - 1))
            iy = int(np.clip(iy, 0, self.ny - 1))
            return float(self.esdf[ix, iy])

        v00 = v(idx[0] + 0, idx[1] + 0)
        v10 = v(idx[0] + 1, idx[1] + 0)
        v01 = v(idx[0] + 0, idx[1] + 1)
        v11 = v(idx[0] + 1, idx[1] + 1)

        v0 = v00 * (1.0 - diff[0]) + v10 * diff[0]
        v1 = v01 * (1.0 - diff[0]) + v11 * diff[0]
        return float(v0 * (1.0 - diff[1]) + v1 * diff[1])

    def get_distance_and_gradient(self, pos_xy: np.ndarray) -> Tuple[float, np.ndarray]:
        """返回 (distance, gradient)；gradient 是 dd/dx, dd/dy。"""
        if not self.is_in_map(pos_xy):
            return float("inf"), np.zeros(2, dtype=np.float64)
        self._ensure_esdf()

        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        pos_m = pos_xy.copy()
        pos_m[0] -= 0.5 * self.resolution
        pos_m[1] -= 0.5 * self.resolution

        idx = self.bound_index(self.pos_to_index(pos_m))
        idx_pos = self.index_to_pos(idx)
        diff = (pos_xy - idx_pos) * self.resolution_inv

        def v(ix: int, iy: int) -> float:
            ix = int(np.clip(ix, 0, self.nx - 1))
            iy = int(np.clip(iy, 0, self.ny - 1))
            return float(self.esdf[ix, iy])

        v00 = v(idx[0] + 0, idx[1] + 0)
        v10 = v(idx[0] + 1, idx[1] + 0)
        v01 = v(idx[0] + 0, idx[1] + 1)
        v11 = v(idx[0] + 1, idx[1] + 1)

        # bilinear distance
        v0 = v00 * (1.0 - diff[0]) + v10 * diff[0]
        v1 = v01 * (1.0 - diff[0]) + v11 * diff[0]
        dist = v0 * (1.0 - diff[1]) + v1 * diff[1]

        # gradient: 对齐 C++ 推导
        # d/dy: (v1 - v0) / res
        grad_y = (v1 - v0) * self.resolution_inv

        # d/dx: ((1-dy)*(v10-v00) + dy*(v11-v01)) / res
        grad_x = ((1.0 - diff[1]) * (v10 - v00) + diff[1] * (v11 - v01)) * self.resolution_inv

        grad = np.array([grad_x, grad_y], dtype=np.float64)
        return float(dist), grad

    def get_distance_and_gradient_batch(self, pos_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量查询 (distance, gradient)。

        参数
        ----
        pos_xy: shape (M, 2)

        返回
        ----
        dist: shape (M,)
        grad: shape (M, 2)

        说明
        ----
        MINCO 的避障项通常会在一个迭代中对轨迹进行多点采样，因此批量接口能减少 Python for-loop 开销。
        当前实现仍是逐点调用（保证一致性），但封装成批量入口，后续可替换为向量化实现。
        """
        pts = np.asarray(pos_xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"pos_xy must have shape (M,2), got {pts.shape}")

        dist = np.empty((pts.shape[0],), dtype=np.float64)
        grad = np.empty((pts.shape[0], 2), dtype=np.float64)
        for i in range(pts.shape[0]):
            d, g = self.get_distance_and_gradient(pts[i])
            dist[i] = d
            grad[i] = g
        return dist, grad

    # ST-opt-tools 风格别名
    def getDistance(self, pos_xy: np.ndarray) -> float:
        return self.get_distance(pos_xy)

    def getDistanceAndGradient(self, pos_xy: np.ndarray) -> Tuple[float, np.ndarray]:
        return self.get_distance_and_gradient(pos_xy)

    def isOccupied(self, idx_xy: Tuple[int, int] | np.ndarray) -> bool:
        idx = np.asarray(idx_xy, dtype=np.int32)
        return self.is_occupied(idx)

    def getResolution(self) -> float:
        return self.resolution

    def getMapSize(self) -> np.ndarray:
        return self.map_size.copy()

    def getOrigin(self) -> np.ndarray:
        return self.map_origin.copy()

    def getVoxelNum(self) -> np.ndarray:
        return self.voxel_num.copy()

    def posToIndex(self, pos_xy: np.ndarray) -> np.ndarray:
        return self.pos_to_index(pos_xy)

    def indexToPos(self, idx_xy: np.ndarray) -> np.ndarray:
        return self.index_to_pos(idx_xy)

    def isLineOccupancy(self, p1: np.ndarray, p2: np.ndarray, safe_threshold: float = 0.0) -> bool:
        """线段碰撞检测（对齐 C++ GridMap::isLineOccupancy 的用途）。

        用法
        ----
        - 用于 A* 的边可行性检测，或 MINCO 在优化中快速剔除明显碰撞的候选路径。
        - 通过沿线段均匀采样，检查 signed distance < safe_threshold。
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        diff = p2 - p1
        max_dist = float(np.linalg.norm(diff))
        if max_dist < 1e-9:
            return self.is_collision(p1, safe_threshold=safe_threshold)

        direction = diff / max_dist
        step = self.resolution * 0.1
        d = 0.0
        while d <= max_dist:
            pt = p1 + direction * d
            if self.is_collision(pt, safe_threshold=safe_threshold):
                return True
            d += step
        return False

    # -----------------------
    # 可选：从 mujoco model/data 生成 occupancy（保持与旧版兼容的入口）
    # -----------------------

    @classmethod
    def from_mujoco(
        cls,
        model,
        data,
        resolution: float,
        width: float,
        height: float,
        robot_radius: float,
        margin: float,
        origin_at_center: bool = False,
    ) -> "GridMap2D":
        """从 MuJoCo 几何体生成 occupancy（简单实现，覆盖 box/sphere/cylinder）。

        说明
        ----
        - 这是对旧版 `GridMap` 的功能补位，便于 planning 侧迁移。
        - 生成后会自动 update_esdf。
        """
        inflation = float(robot_radius) + float(margin)
        params = GridMap2DParams(
            resolution=resolution,
            size_x=width,
            size_y=height,
            inflation_radius=inflation,
            origin_at_center=origin_at_center,
        )
        gm = cls(params)

        # 这个实现不依赖 mujoco 常量（避免 import mujoco），直接用 model.geom_type 与 geom_size 的数值。
        # 若你的环境里 mujoco 可用，也可以在这里更精细区分类型。

        occ = np.zeros((gm.nx, gm.ny), dtype=np.int8)

        def world_cells():
            for ix in range(gm.nx):
                for iy in range(gm.ny):
                    pos = gm.index_to_pos(np.array([ix, iy], dtype=np.int32))
                    yield ix, iy, pos

        # 遍历几何体
        for gid in range(model.ngeom):
            gtype = int(model.geom_type[gid])
            center = np.array(data.geom_xpos[gid][:2], dtype=np.float64)
            size = np.array(model.geom_size[gid], dtype=np.float64)

            # 粗略：sphere/cylinder 认为是圆；box 认为是 AABB（不考虑旋转）
            # gtype 数值和 mujoco.mjtGeom 不完全绑定，但常见：sphere=2, cylinder=5, box=6
            if size.shape[0] >= 2 and gtype == 6:  # box
                lx, ly = float(size[0]), float(size[1])
                min_xy = center - np.array([lx, ly]) - inflation
                max_xy = center + np.array([lx, ly]) + inflation

                for ix, iy, pos in world_cells():
                    if (pos[0] >= min_xy[0] and pos[0] <= max_xy[0] and
                        pos[1] >= min_xy[1] and pos[1] <= max_xy[1]):
                        occ[ix, iy] = 1

            else:
                # sphere/cylinder/others: treat as circle radius=size[0]
                r = float(size[0]) + inflation
                r2 = r * r
                for ix, iy, pos in world_cells():
                    d2 = float((pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2)
                    if d2 <= r2:
                        occ[ix, iy] = 1

        gm.set_occupancy(occ, update_esdf=True)
        return gm


def _self_test() -> None:
    """不依赖项目其他模块的最小自测。"""
    params = GridMap2DParams(resolution=0.1, size_x=2.0, size_y=2.0, origin_at_center=True)
    gm = GridMap2D(params)

    # 在中心放一个 3x3 的障碍块
    occ = np.zeros((gm.nx, gm.ny), dtype=np.int8)
    cx, cy = gm.nx // 2, gm.ny // 2
    occ[cx - 1:cx + 2, cy - 1:cy + 2] = 1
    gm.set_occupancy(occ)

    # 取一个点查距离与梯度
    p = np.array([0.6, 0.0], dtype=np.float64)
    d, g = gm.get_distance_and_gradient(p)
    print("dist:", d)
    print("grad:", g)


if __name__ == "__main__":
    _self_test()
