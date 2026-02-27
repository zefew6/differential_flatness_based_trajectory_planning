"""map_obstacles.py

集中管理 minco_test 的地图障碍物配置（工程化、可维护）。

坐标系
------
- 与 `GridMap2DParams(origin_at_center=False)` 一致：地图覆盖 [0, size_x] x [0, size_y]
- 下面的圆心与半径都用这个 map frame（单位：米）

用法
----
在 `minco_test.py` 中导入：

    from map_obstacles import CIRCULAR_OBSTACLES

然后用这些圆去 rasterize occupancy 并 update ESDF。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

Obstacle = Dict[str, object]

# =============================
# 工程化障碍物配置（推荐）
# =============================
# 支持类型：
# - circle : {"type":"circle", "cx":.., "cy":.., "r":..}
# - rect   : {"type":"rect", "xmin":.., "xmax":.., "ymin":.., "ymax":..}
# - poly   : {"type":"poly", "verts":[(x1,y1),(x2,y2),...]}  # 三角形就是 3 个顶点
#
# 坐标系：map frame（与 minco_test.py 构建的 GridMap2D 一致）
OBSTACLES: List[Obstacle] = [
    {"type": "circle", "cx": 9.5, "cy": 9.5, "r": 2.1},
    # 矩形示例（轴对齐 bounding box）
    {"type": "rect", "xmin": 3.0, "xmax": 5.0, "ymin": 12.0, "ymax": 14.0},
    # 三角形示例（多边形）
    {"type": "poly", "verts": [(12.0, 3.0), (14.0, 3.0), (13.0, 5.0)]},
]


# 兼容旧接口：如果你只想维护“圆列表”，也可以用这个。
# minco_test.py 会优先读 OBSTACLES。
CIRCULAR_OBSTACLES: List[Tuple[float, float, float]] = [
    (9.5, 9.5, 1.2),
]
