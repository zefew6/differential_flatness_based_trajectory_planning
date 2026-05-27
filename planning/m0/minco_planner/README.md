# MINCO Planner

`m0.minco_planner` 是一个 2-D MINCO 轨迹规划模块。当前推荐入口是
`MincoPlanner`：它把 A* 几何路径搜索、路径重采样、时间分配、安全走廊构造、
MINCO 数值优化组织成一个可复用的类。

规划器核心不依赖 MuJoCo。MuJoCo 只通过 `adapters/mujoco.py` 中的
`MujocoGridMap2D` 适配进来。

## 快速使用

### 从栅格地图规划

```python
import numpy as np

from m0.minco_planner import GridMap2D, MincoPlanner, MincoPlannerConfig

grid_map = GridMap2D(
    resolution=0.05,
    width=10.0,
    height=10.0,
    origin_x=-5.0,
    origin_y=-5.0,
    robot_radius=0.2,
    margin=0.1,
)
grid_map.add_circle_obstacle(np.array([0.0, 0.0]), radius=0.5)

planner = MincoPlanner(MincoPlannerConfig(
    obstacle_method="esdf",
    max_seg_len=1.2,
))

result = planner.plan(
    grid_map=grid_map,
    start_xy=np.array([-4.0, -4.0]),
    goal_xy=np.array([4.0, 4.0]),
)

traj = result.minco
xy_samples = result.sample_xy(1000)
print(result.success, result.total_time, result.cost_time_ms)
```

### 从已有路径规划

```python
import numpy as np

from m0.minco_planner import MincoPlanner

path = np.array([
    [-4.0, -4.0],
    [-1.0,  0.5],
    [ 2.0,  1.0],
    [ 4.0,  4.0],
])

planner = MincoPlanner("esdf")
result = planner.plan(path=path)
traj = result.minco
```

## 使用 FIRI 安全走廊

SFC 模式下，规划器会先根据路径构造安全飞行走廊，再用走廊约束优化 MINCO。
`SFCOptions(build_method="firi")` 会启用移植自 ST-opt-tools 风格的 FIRI 走廊。

```python
from m0.minco_planner import MincoPlanner, MincoPlannerConfig, SFCOptions

planner = MincoPlanner(MincoPlannerConfig(
    obstacle_method="sfc",
    max_seg_len=1.2,
    sfc=SFCOptions(
        build_method="firi",
        search_radius=6.0,
        safe_margin=0.0,
        weight=5e5,
    ),
))

result = planner.plan(
    grid_map=grid_map,
    start_xy=start_xy,
    goal_xy=goal_xy,
)
```

可选走廊构造方式：

- `firi`：推荐默认值，使用 FIRI 迭代膨胀凸多面体。
- `cube`：较快的轴对齐走廊，适合调试。
- `legacy`：保留旧实现，主要用于对比。

命令行 demo：

```bash
cd /home/hac/Differential_Flatness/MAS/planning

python examples/test_minco_planner.py --method sfc --sfc-method firi
python examples/test_minco_planner_v2.py --method sfc --sfc-method firi
python examples/test_minco_planner_v3.py --method sfc --sfc-method firi
```

## 在线重规划

`replan()` 是在线重规划的推荐入口。地图更新后，传入当前机器人位置、目标点、
可选当前速度即可。

```python
result = planner.replan(
    grid_map,
    start_xy=robot_pos,
    goal_xy=goal_xy,
    start_vel=robot_vel,
    max_seg_len=0.8,
)

follower.set_trajectory(result.minco)
```

旧 demo 中的 `online_replan_once()` 仍然保留，但它返回的是兼容旧代码的 `dict`。
新代码建议直接使用 `plan()` / `replan()`，拿到结构化的 `MincoPlanResult`。

## MuJoCo 适配

纯规划器不导入 MuJoCo。如果需要从 MuJoCo 场景生成 ESDF 栅格，用：

```python
from m0.minco_planner import MincoPlanner, MujocoGridMap2D

grid_map = MujocoGridMap2D(
    model=model,
    data=data,
    resolution=0.05,
    width=10.0,
    height=10.0,
    robot_radius=0.2,
    margin=0.1,
    origin_x=-5.0,
    origin_y=-5.0,
)

planner = MincoPlanner("esdf")
result = planner.plan(grid_map=grid_map, start_xy=head_pos, goal_xy=tail_pos)
```

## 主要模块

- `planner.py`：高层规划入口，负责 A*、重采样、时间分配、SFC 构造和 optimizer 调用。
- `trajectory_optimizer.py`：低层数值优化器，负责 L-BFGS 变量打包、MINCO 系数生成、代价和梯度。
- `costs/feasibility.py`：速度、加速度等可行性代价。
- `costs/esdf_obstacle.py`：ESDF 障碍物代价。
- `costs/sfc_obstacle.py`：安全走廊约束代价。
- `corridor/firi.py`：FIRI 安全走廊实现。
- `corridor/sfc.py`：从 grid map 构造 SFC 的统一入口。
- `maps/grid_map.py`：纯 Python 2-D occupancy grid 和 ESDF 查询。
- `adapters/mujoco.py`：MuJoCo 场景到 `GridMap2D` 的适配层。
- `minco.py`、`minco_MinJerkOpt.py`：MINCO 轨迹表达和 minimum-jerk 求解核心。
- `minco_Optimizer.py`、`minco_obstacle.py`、`minco_FeasibilityConstraint.py`：旧接口兼容层。

## 常用配置

```python
planner = MincoPlanner(MincoPlannerConfig(
    obstacle_method="sfc",
    max_seg_len=1.0,
    sfc=SFCOptions(
        build_method="firi",
        search_radius=5.0,
        push_to_clearance=True,
        target_clearance=0.4,
        safe_margin=0.0,
        weight=5e5,
    ),
    optimizer_params={
        "max_vel": 3.5,
        "max_acc": 0.8,
        "wei_time": 1e2,
        "wei_feas": 2e4,
        "wei_obs": 2e4,
        "lbfgs_max_iterations": 200,
    },
    debug_print_every=10,
))
```

注意：`optimizer_params` 只能传 `TrajectoryOptimizer.set_params()` 支持的参数。

## 兼容接口

为了不破坏旧代码，下面这些接口仍然可用：

- `PolyTrajOptimizer`
- `minco_Optimizer.py`
- `setGridMap()` / `setParam()` / `OptimizeTrajectory()`
- `online_replan_once()`

新代码建议使用：

- `MincoPlanner`
- `MincoPlannerConfig`
- `SFCOptions`
- `MincoPlanResult`
- `plan()` / `replan()`

## 后续值得继续整理

- 给 `TrajectoryOptimizer` 增加更明确的收敛状态和失败原因。
- 把 A* 搜索后端做成可注入接口，方便替换成 JPS、Hybrid A* 或采样式路径搜索。
- 为 `plan(path=...)`、`plan(grid_map=...)`、`obstacle_method="sfc"` 增加纯 Python 单元测试。
- 逐步迁移旧 CamelCase API，最后把兼容层压到更薄。
- 给 FIRI corridor 增加可视化/调试导出，方便检查每段多面体边界。
