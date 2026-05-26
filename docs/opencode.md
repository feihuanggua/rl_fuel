# FUEL RL 工作记录

## 目标

使用强化学习微调 FUEL 的视点选择策略，探索是否能超越行为克隆（BC）基线。

## 架构

```
rl_fuel/
├── src/standalone/          # C++ 独立核心 (无 ROS 依赖)
│   ├── sdf_map_standalone   # SDF 地图 (占据概率 + ESDF)
│   ├── frontier_finder_standalone  # 前沿检测与视点采样
│   ├── fuel_env_core        # 环境封装 (观测模拟、路径规划、地图管理)
│   ├── perception_utils     # 相机 FOV 计算
│   ├── raycast_standalone   # Bresenham 3D 光线投射
│   └── params.h             # 全局参数
├── fuel_rl/
│   ├── models/              # PyTorch 模型
│   │   ├── encoder.py       # 3D CNN 编码器 (Channel + Spatial 注意力)
│   │   ├── viewpoint_head.py # 解耦视点头 (位置流 + 偏航流)
│   │   └── sac_models.py    # SAC Actor + 双 Q 网络
│   ├── data/
│   │   ├── collector.py     # BC 专家数据收集 (子进程并行)
│   │   └── dataset.py       # PyTorch Dataset + 4倍旋转增强
│   ├── env/
│   │   └── viewpoint_env.py # 单步视点选择 Gym 环境
│   ├── train/
│   │   ├── train_bc.py      # BC 训练
│   │   ├── train_ppo.py     # PPO 微调 (自定义)
│   │   ├── train_sac.py     # SAC 训练
│   │   └── train_reinforce.py # REINFORCE 训练
│   └── config.py            # 全局配置
└── experiment.py            # 学术实验评估脚本
```

## 环境模拟修正

### C++ 核心 vs 原始 FUEL 差异修复

| 问题 | 原始 FUEL | 修复前 | 修复后 |
|------|-----------|--------|--------|
| **聚类参数** | cluster_min=100, min_visib_num=15, clearance=0.21 | 20, 1, 0.1 | 100, 15, 0.21 |
| **射线模型** | 深度相机 640×480 skip=2 (~75k rays) | 均匀角 80×60 (~4.8k rays) | Pinhole 相机 + cam02body 外参 |
| **自由空间射线** | depth_filter_maxdist=5.0m > max_ray_length=4.5m | free_pt 在 4.5m (等于 max_ray_length) | free_pt 改为 5.0m |
| **障碍物注入** | SDF map 初始全未知 | belief map 未注入障碍物 | loadMap 调用 injectObstacles |
| **地面补全** | PCD 自带地面点 | 无 | addGroundPlane(ground_z) 从底部填充 |
| **低高度过滤** | pos[2]<0.4, pt_w[2]<0.2 | ✅ 已有 | ✅ 已有 |
| **skip_pixel 可配** | ROS param | 硬编码 2 | PerceptionParams.skip_pixel |

### 关键修复细节

1. **自由空间射线 bug**: `inputPointCloud` 中 `length > max_ray_length(4.5m)` 才判定自由端点。修复前 free_pt 在 4.5m，不满足 `>`，导致所有自由射线被当成障碍物。

2. **resetMap 已同步**: 重置后自动重新注入 gt_cloud 障碍物，并应用地面补全（如启用）。

## BC 训练

### 数据

- 200 张随机柱子地图 (20×20×3m, 15 柱)
- 3539 个高质量前沿样本 (cluster_min=100 过滤碎片)
- 4 倍旋转增强 → 14,156 训练样本
- 3 通道输入: [障碍物, 前沿, 自由空间] × 32×32×10

### 模型

- Encoder: Conv3d[3→32→64→128] + ChannelAttention + SpatialAttention → 512d
- Head: 解耦位置流 (ResMLP×2) + 偏航流 (位置为条件)
- 输出: [dx, dy, dz, dyaw] ∈ [-1,1]

### 结果

| 指标 | 值 |
|------|-----|
| 参数 | 3.7M |
| Best val loss | 0.0539 |
| 确定性 reward | ~3.81 |

## RL 尝试

### PPO (自定义)

| 尝试 | 状态 | 结果 |
|------|------|------|
| PPO v2 | log_std=-2.0 (std=0.14), update_ts=2000 | reward 1.5 → 0.0 崩溃 |
| PPO v3 | log_std=-1.0 (std=0.37), KL penalty, entropy=0.1, update_ts=1000 | 训练中，ts=501/1000 时暂停 |

**问题**: 单步环境无法训练 Critic (explained_variance=0)，策略漂移后坍塌。

### SAC

| 阶段 | 结果 |
|------|------|
| BC 预热 (2000步确定性) | avg_r=+3.6 |
| SAC 主循环 (1000步) | avg_r=-1.04, alpha=0.93 |
| SAC 趋势 | reward 不增反降，无法超过 BC |

**问题**: 4 个 Q 网络各带独立 encoder (~15M 参数)，训练不稳定。

### REINFORCE

| 配置 | 结果 |
|------|------|
| std=0.14 (可学习) | reward 1.1→1.6，缓慢上升 |
| std=0.50 (大探索) | reward 0.3，80% 无效动作 |
| std=0.30 (中探索) | reward 0.2→0.0，std 膨胀到 0.48 后策略冻结 |
| **std=0.15 (固定)** | 单步确定性 eval: **4.08 vs BC 3.81** |

**发现**: 固定小 std 能超越 BC。可学习 std 会膨胀导致策略崩溃。

## 学术实验

### 设置

- 50 张随机地图 (20×20×3m, 15 柱)
- 每地图 50 步多步探索
- 4 种方法: BC, REINFORCE, Greedy (零偏移), Random

### 结果

| 方法 | 最终覆盖率 | 有效步 | 视点质量 |
|------|:---------:|:------:|:--------:|
| **BC** | **21.5%** | **38.6** | **3.00** |
| REINFORCE | 17.8% | 35.0 | 2.77 |
| Random | 20.3% | 15.9 | 1.14 |
| Greedy | 9.3% | 26.0 | 0.41 |

### 结论

BC 在所有多步探索指标上最优。REINFORCE 虽然单步确定性评估略胜 BC (4.08 vs 3.81)，但多步探索中因序列一致性不足而落后。

**核心洞察**: 单步视点选择任务，BC 已接近理论最优。RL 的探索噪声在多步序列中累积误差，导致整体覆盖率下降。要超越 BC，需要多步 RL 框架（如蒙特卡洛树搜索或序列级策略梯度）。

## 命令速查

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# BC 训练
python -m fuel_rl.train.train_bc --data-path ./fuel_rl_data/expert_data_v3.pt --save-dir ./fuel_rl_checkpoints/bc_v3

# REINFORCE 训练
python -c "import sys;sys.argv=['r','--total-steps','50000','--log-every','200'];from fuel_rl.train.train_reinforce import main;main()"

# 实验评估
python experiment.py --num-maps 50 --max-steps 50

# 3D 可视化
python test_frontier_3d.py 42

# 编译 C++ 核心
python setup.py build_ext --inplace
```

## 参数配置

```python
# config.py 关键参数
ENCODER_CHANNELS = [32, 64, 128]   # 3D CNN 通道
ENCODER_EMBED_DIM = 512            # 嵌入维度
GRID_SIZE, GRID_Z = 32, 10         # 输入网格
VOXEL_RES = 0.2                    # 网格分辨率

# params.h 关键参数
cluster_min = 100                  # 前沿最小体素
min_visib_num = 15                 # 最小可见体素
min_candidate_clearance = 0.21     # 视点最小间隙
skip_pixel = 2 (训练用4)           # 相机降采样
```
