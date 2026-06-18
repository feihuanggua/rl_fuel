# Skill: 收集 BC 专家数据

## 触发条件
- 首次训练 BC 模型
- 需要更多/更新训练数据
- 地图参数变更后

## 数据流程

```
200 张随机柱子地图
  → FuelEnvCore (每地图独立, 子进程)
    → 四方向初始观测
    → detectFrontiers
    → 对每个前沿: build_3channel_grid + get_expert_label
  → torch.save({inputs, targets}) → .pt 文件
```

## 执行步骤

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# 默认配置: 200 maps, 15 pillars, 32×32×10 grid
python -m fuel_rl.data.collector --num-maps 200 --grid-size 32 --grid-z 10

# 输出: ./fuel_rl_data/expert_data.pt
```

## 数据集结构

```python
{
    "inputs": torch.Tensor [N, 3, 32, 32, 10],  # float16
    "targets": torch.Tensor [N, 4],              # float32, [dx,dy,dz,dyaw] in [-1,1]
    "config": {"grid_size": 32, "grid_z": 10, "resolution": 0.2, "num_samples": 3539}
}
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `COLLECT_NUM_MAPS` | 200 | 地图数量 |
| `COLLECT_NUM_PILLARS` | 15 | 柱子数 |
| `GRID_SIZE` | 32 | XY 网格尺寸 |
| `GRID_Z` | 10 | Z 网格尺寸 |
| `VOXEL_RES` | 0.2 | 体素分辨率 (m) |
| `COLLECT_SAVE_PATH` | `./fuel_rl_data/expert_data.pt` | 输出路径 |

## 注意事项

- 使用 4 个子进程 (Pool(4)) 并行收集，避免 PCL 内存泄漏影响主进程
- 每个前沿的 label 如果超出 [-1,1] 范围会被过滤
- frontier_size < 5 的前沿被跳过 (碎片过滤)
- 数据保存为 float16 以节省磁盘 (~50MB vs ~100MB)
