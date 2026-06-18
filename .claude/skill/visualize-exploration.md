# Skill: 探索过程可视化

## 触发条件
- 想直观理解策略行为
- Debug 训练问题
- 制作论文/报告的图

## 可视化类型

### 1. 探索轨迹 (多方法对比)

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

python -m fuel_rl.eval.vis_ordering
```

输出 4 张图:
- `vis_trajectories.png` — 各策略的完整探索轨迹
- `vis_curves.png` — 覆盖率 vs 步数 + 累计奖励 vs 步数
- `vis_selection.png` — 各策略在 step 0/5/15/40 时的前沿选择决策
- `vis_step_reward.png` — 每步奖励分解 (绿色=正, 红色=负)

### 2. 3D 前沿可视化

```bash
python tools/test_frontier_3d.py 42  # seed=42
```

### 3. 2D 占用地图 + 前沿

```bash
python tools/visualize_order_2d.py
```

## 使用 FuelVisualizer (复刻 FUEL 的 RViz 风格)

```python
from fuel_rl.visualizer import FuelVisualizer
from fuel_rl import FuelEnvCore
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.config import *

core = FuelEnvCore()
core.init(default_map_params(...), default_frontier_params(), ...)
pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
core.load_map_from_points(pts)
core.reset_map()
core.simulate_observation(np.array([0, 0, 1.5]), 0.0)

viz = FuelVisualizer(figsize=(10, 10))
frontiers = core.detect_frontiers(np.array([0, 0, 1.5]))
viz.render_2d(
    core,
    agent_pos=np.array([0, 0, 1.5]),
    agent_yaw=0.0,
    frontiers=frontiers,
    show_fov=True,
    show_viewpoints=True,
    save_path="exploration_step0.png"
)
```

## 颜色方案 (匹配 FUEL RViz)

| 元素 | 颜色 |
|------|------|
| 背景 | 淡黄 (255,253,224) |
| 未知 | 灰色 (128,128,128) |
| 自由 | 白色 |
| 障碍物 | 彩虹色 (按高度映射) |
| 前沿 | 彩虹色 (按索引映射) |
| 轨迹 | 红色线 |
| 视点 | 绿色圆点 |
| UAV | 绿色箭头 |
| FOV | 红色锥形 |
