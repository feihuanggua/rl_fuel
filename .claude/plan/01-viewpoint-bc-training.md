# 视点选择 BC 训练方案

## 目标

用 FUEL 原生规划器产生专家数据，训练 3D CNN 模型做单步视点选择。
给定一个前沿的局部体素网格 (3×32×32×10)，预测最佳观测位姿 [dx, dy, dz, dyaw]。

## 当前状态

- ✅ Encoder: Conv3d[3→32→64→128] + Channel/Spatial Attention → 512d
- ✅ Head: 解耦位置流 (ResMLP×2) + 偏航流
- ✅ 数据: 200 maps, 3539 samples → 14,156 (×4旋转增强)
- ✅ 结果: Best val loss 0.0539, 确定性 reward ~3.81

## 架构

```
输入 [3, 32, 32, 10]
  → Conv3d(3→32, k=5, s=2) + BN + LeakyReLU  # [32, 16, 16, 5]
  → Conv3d(32→64, k=3, s=2) + BN + LeakyReLU  # [64, 8, 8, 3]
  → Conv3d(64→128, k=3, s=2) + BN + LeakyReLU  # [128, 4, 4, 2]
  → ChannelAttention + SpatialAttention
  → Flatten → Linear(4096→512) + LN + LeakyReLU
  → embedding [B, 512]
```

### 解耦头
```
位置流: embed → ResMLP(512)×2 → Linear(512,3) → tanh → [dx,dy,dz]
偏航流: embed ⊕ pos.detach() → Linear(512+3,256) → LN → LeakyReLU
       → ResMLP(256) → Linear(256,1) → tanh → [dyaw]
```

## 关键文件

| 文件 | 职责 |
|------|------|
| `fuel_rl/models/encoder.py` | 3D CNN + Channel/Spatial 注意力 |
| `fuel_rl/models/viewpoint_head.py` | ViewpointHead (BC推理用) |
| `fuel_rl/data/collector.py` | build_3channel_grid, get_expert_label |
| `fuel_rl/data/dataset.py` | ExpertDataset + 4倍旋转增强 |
| `fuel_rl/train/train_bc.py` | BC 训练主循环 |
| `fuel_rl/config.py` | BC_BATCH_SIZE, BC_LR, ENCODER_CHANNELS 等 |

## 后续改进

- [ ] P1: 增大训练数据 (200→500 maps), 验证是否饱和
- [ ] P2: encoder 增加 SE 通道注意力的 reduction ratio 消融 (当前=16)
- [ ] P3: 尝试 3D GroupNorm vs 原始 BatchNorm 的训练稳定性
