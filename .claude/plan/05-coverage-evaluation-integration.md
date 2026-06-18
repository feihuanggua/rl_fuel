# 覆盖率评估方法论集成方案

## 目标

将 [search0614.md](C:\Users\jiading01\Documents\我的POPO\search0614.md) 中的 self-play 覆盖率评估方法
应用到 UAV 探索策略评估中。

## 方法论映射

### 1. 探索覆盖率量化

| search0614 方法 | UAV 应用 | 实现方式 |
|-----------------|----------|----------|
| Good-Turing 覆盖率 | 探索进度估计: 单例前沿占比预测未探索比例 | `C = 1 - f1/n`, f1=仅访问一次的前沿数 |
| Chao1 估计器 | 地图中可发现的总前沿数量下界 | `S_obs + f1²/2f2` |
| Rarefaction 曲线 | coverage vs steps 外推 → 判断采样饱和度 | 已在 experiment.py 中记录 progress_curve |

### 2. 策略多样性评估

| 方法 | UAV 应用 |
|------|----------|
| Gini 系数 | 前沿类型访问分布的均匀性 |
| Shannon 均匀度 J' | 综合丰富度 + 均匀度 |
| UMAP 降维 | 前沿 → 3D embedding → 2D 投影, 识别未覆盖区域 |

### 3. MAP-Elites 存档 (最核心)

定义**前沿类型行为描述符**:

```python
frontier_descriptor = (
    size_bucket,      # 前沿大小分桶: [small, medium, large]
    distance_bucket,  # 距离分桶: [near, mid, far]
    visib_bucket,     # 可见性分桶: [low, mid, high]
    cluster_bucket,   # 前沿聚集度: [isolated, clustered]
)
```

每个 grid cell 存档该类型前沿上表现最好的视点选择策略参数。
覆盖率的定义: `filled_cells / total_cells`。

### 4. 收益矩阵

构建不同策略在相同地图上的 pairwise 对比矩阵:

```
        BC   REINF  TSP   Closest
BC      -    0.52   0.38  0.61
REINF  0.48   -     0.35  0.58
TSP    0.62  0.65    -    0.72
Closest 0.39 0.42   0.28   -
```

## 实现优先级

| 优先级 | 功能 | 工作量 | 文件 |
|--------|------|--------|------|
| P0 | Rarefaction 曲线可视化 (已有 progress_curve 数据) | 小 | 新增 `fuel_rl/eval/rarefaction.py` |
| P1 | 前沿类型分桶 + Gini 系数 | 中 | 新增 `fuel_rl/eval/diversity_metrics.py` |
| P2 | MAP-Elites 存档 | 大 | 新增 `fuel_rl/eval/map_elites.py` |
| P2 | UMAP 可视化 | 中 | 新增 `fuel_rl/eval/embedding_viz.py` |
| P3 | 收益矩阵 | 中 | 扩展 `experiment.py` |

## 快速启动

```python
# Phase 0: 利用现有数据画 Rarefaction 曲线
import json, numpy as np
import matplotlib.pyplot as plt

data = json.load(open("/tmp/rl_fuel_experiment.json"))
for name, d in data.items():
    curves = d["progress_curve"]
    avg = np.mean(curves, axis=0)
    plt.plot(avg, label=name)
plt.xlabel("Steps")
plt.ylabel("Coverage")
plt.legend()
plt.savefig("rarefaction.png")
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `experiment/experiment.py --mode sequence` | 已产出 progress_curve 数据 |
| `fuel_rl/eval/plot_results.py` | 已有部分绘图 |
| 待新增: `fuel_rl/eval/rarefaction.py` | Rarefaction 曲线绘制 |
| 待新增: `fuel_rl/eval/diversity_metrics.py` | Gini / Shannon J' |
| 参考: `search0614.md` | 方法论来源 |
