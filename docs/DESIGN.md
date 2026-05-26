# FUEL RL 重构设计方案

## 问题诊断

当前方案训练失败的根本原因：

1. **MLP 无法处理 3D 体素数据** — 13k 维展平输入丢失空间结构，MlpPolicy 无法学到有效特征
2. **多步 episode (100步) 导致梯度信号稀疏** — 大部分 step 产生 no_path 无效动作，reward 被稀释
3. **没有预训练** — 纯 RL 从零探索，在稀疏奖励环境中几乎不可能收敛

## 设计目标

支持两个层次的决策任务：

| 层次 | 任务 | 输入 | 输出 | 训练方式 |
|------|------|------|------|----------|
| **低层：视点选择** | 给定一个前沿，选择最佳观测位姿 | 局部 3D 体素网格 | 4D 视点偏移 [dx,dy,dz,dyaw] | BC 预训练 + PPO 微调 |
| **高层：探索顺序** (后续) | 从多个前沿中选择下一个探索目标 | 所有前沿嵌入 + 全局状态 | 前沿选择 (离散/排序) | DQN / Attention Policy |

## 架构设计

### 总体架构

```
                    ┌─────────────────────────────┐
                    │        Global State          │
                    │  (agent_pos, yaw, progress)  │
                    └──────────┬──────────────────┘
                               │
┌──────────┐    ┌──────────┐   │   ┌──────────┐
│ Frontier  │    │ Frontier  │   │   │ Frontier  │
│   #0      │    │   #1      │   │   │   #N      │
│ [3,V,V,V] │    │ [3,V,V,V] │   │   │ [3,V,V,V] │
└────┬──────┘    └────┬──────┘   │   └────┬──────┘
     │                │           │        │
     ▼                ▼           │        ▼
┌─────────────────────────────────┤──────────────┐
│         3D CNN Encoder (共享)    │              │
│    Conv3d → BN → LeakyReLU ×4   │              │
│    + Spatial/Channel Attention   │              │
└──────────┬──────────────────────┘              │
           │ embedding_i                          │
           ▼                                      │
    ┌──────────────┐                              │
    │ Viewpoint    │◄──── embedding_i             │
    │ Head (低层)  │                              │
    │ ResMLP →     │                              │
    │ [dx,dy,dz,   │                              │
    │  dyaw]       │                              │
    └──────────────┘                              │
                                                  │
    ┌──────────────────────────────────────────────┐
    │ Exploration Order Head (高层, 后续实现)       │
    │ Attention Pooling(embeddings) + global_state  │
    │ → frontier scores → 选择                      │
    └──────────────────────────────────────────────┘
```

### 阶段一：视点选择 (当前实现)

#### 1. 数据收集

从 FUEL 内置规划器收集专家数据：

```
每条数据 = {
    voxel_grid: [3, 32, 32, 32],   # 3通道: 障碍/前沿/自由空间
    viewpoint:  [dx, dy, dz, dyaw]  # 归一化到 [-1, 1]
}
```

- 遍历环境中所有前沿
- 对每个前沿，用 FUEL 的 FrontierFinder 采样候选视点
- 用 FUEL 的可视性分析选出最佳视点作为标签
- 围绕前沿中心提取 32³ 局部体素网格

#### 2. 网络结构 (基于 RL_Viewpoint v2.0/v3.0 经验)

**Encoder (3D CNN)**:
```
Input: [B, 3, 32, 32, 32]
  → Conv3d(3→32, k=5, s=2, p=2) + BN + LeakyReLU     # [32, 16, 16, 16]
  → Conv3d(32→64, k=3, s=2, p=1) + BN + LeakyReLU     # [64, 8, 8, 8]
  → Conv3d(64→128, k=3, s=2, p=1) + BN + LeakyReLU    # [128, 4, 4, 4]
  → ChannelAttention3D(128) + SpatialAttention3D(128)
  → Flatten → Linear(8192→512) + LayerNorm + LeakyReLU
  → embedding (512-dim)
```

**Viewpoint Head (解耦)**:
```
embedding (512)
  → ResMlpBlock(512) × 2
  → Linear(512→3) + tanh → position [dx, dy, dz]
  → concat(embedding, pos.detach()) → ResMLP → Linear→1 + tanh → yaw
```

#### 3. 训练流程

```
阶段A: 行为克隆 (BC)
  - 用专家数据训练，MSE Loss
  - 100 epochs, AdamW, ReduceLROnPlateau
  - 4x 旋转数据增强

阶段B: PPO 微调
  - 单步 episode (1步 done)
  - 加载 BC 预训练权重
  - 自定义 PPO (非 SB3)
  - Reward: 覆盖率 + 体积 + 距离引导
```

#### 4. Reward 设计 (PPO 微调时)

```
reward = r_coverage + r_volume + r_dist_guide + r_penalty

r_coverage  = 可见前沿格子数 / 前沿总格子数 × 5.0
r_volume    = 新发现自由体积 × 0.005
r_dist_guide = 0.5 × exp(-(dist - 5)²/32)   # 鼓励合适距离
r_penalty   = 碰撞/出界 → -2.0, 一步终止
```

### 阶段二：探索顺序选择 (后续实现)

#### 输入设计

```
observation = {
    frontier_embeddings: [N_max, 512],   # 每个前沿的 CNN 编码
    frontier_features:   [N_max, 10],    # 手工特征 (中心/大小/距离等)
    mask:                [N_max],         # 有效前沿标记
    global_state:        [6],             # UAV 位置/朝向/进度
}
```

#### 网络结构

```
frontier_embeddings + frontier_features → per-frontier MLP → enhanced_i
all enhanced_i → Multi-Head Self-Attention → context_i
concat(context_i, global_state) → MLP → score_i
scores → softmax → frontier selection
```

#### 训练方式

- DQN 或 Attention-based Policy Gradient
- 可选择先用 BC 预训练（从 FUEL 全局 TSP 解收集标签）
- 奖励: episode 总探索时间 / 覆盖率

## 文件结构

```
rl_fuel/
├── fuel_rl/
│   ├── core/                      # C++ 核心封装 (已有)
│   │   └── fuel_rl_core.so
│   ├── models/                    # 神经网络模型
│   │   ├── __init__.py
│   │   ├── encoder.py             # 3D CNN 编码器 + 注意力
│   │   ├── viewpoint_head.py      # 解耦视点预测头
│   │   └── exploration_head.py    # 探索顺序选择头 (后续)
│   ├── data/                      # 数据收集与处理
│   │   ├── __init__.py
│   │   ├── collector.py           # 从 FUEL 收集专家数据
│   │   ├── dataset.py             # PyTorch Dataset + 数据增强
│   │   └── preprocess.py          # 体素网格预处理
│   ├── env/                       # 环境封装
│   │   ├── __init__.py
│   │   ├── viewpoint_env.py       # 单步视点选择环境
│   │   └── exploration_env.py     # 多步探索环境 (后续)
│   ├── train/                     # 训练脚本
│   │   ├── train_bc.py            # 行为克隆训练
│   │   ├── train_ppo.py           # PPO 微调训练
│   │   └── train_explore.py       # 探索顺序训练 (后续)
│   ├── callbacks.py               # 训练回调
│   ├── plot_metrics.py            # 可视化
│   ├── config.py                  # 全局配置
│   └── visualizer.py              # 环境可视化
├── src/                           # C++ 源码 (已有)
├── setup.py
├── COMMANDS.md
└── DESIGN.md                      # 本文件
```

## 实施优先级

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | 3D CNN 编码器 + 解耦视点头 | 待实现 |
| P0 | 专家数据收集 (从 FUEL C++ 核心采样) | 待实现 |
| P0 | BC 训练管线 | 待实现 |
| P1 | 单步 PPO 环境 + 微调 | 待实现 |
| P2 | 探索顺序选择头 + Attention | 后续 |
| P2 | 多步探索环境 | 后续 |
| P3 | 集成测试 + ROS 桥接 | 后续 |
