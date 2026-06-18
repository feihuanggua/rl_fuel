# Skill: 评估模型检查点

## 触发条件
- 训练中 / 训练完成
- 想对比不同 checkpoint 的表现
- 想查看模型在特定地图上的行为

## 快速评估

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# SAC 序列级模型
python -m fuel_rl.eval.quick_eval \
  --checkpoint fuel_rl_checkpoints/sac_seq5/best_model.pth

# BC 视点模型 (需要先加载 Encoder3D + ViewpointHead)
python -c "
import torch
import numpy as np
from fuel_rl.models.encoder import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointHead

model = ViewpointHead(
    Encoder3D(input_shape=(32,32,10), channels=[32,64,128], embed_dim=512),
    embed_dim=512
).cuda()
model.load_state_dict(torch.load('./fuel_rl_checkpoints/bc_v3/best_model.pth'))
model.eval()
# ... 对单前沿做推理
"
```

## 检查点对比

```bash
# 对比 BC 和 REINFORCE
python tools/compare_checkpoints.py \
  --model-a ./fuel_rl_checkpoints/bc_v3/best_model.pth \
  --model-b ./fuel_rl_checkpoints/reinforce/best_model.pth \
  --num-eps 50
```

## 输出解读

```
Policy          Cov      ±     Reward    Steps     Dist    Time
TSP_FUEL       0.312  0.120    285.3     42.5     85.3     12s
Closest        0.218  0.095    198.7     38.6     95.7      8s
```

- **Cov (覆盖率)**: 越高越好，核心指标
- **Steps**: 有效步数比例
- **Dist**: 总飞行距离，越短越好 (给定相同覆盖率)
- **Reward**: 环境累计奖励

## 单地图可视化

```bash
# 3D 可视化
python tools/test_frontier_3d.py 42

# 2D 轨迹可视化
python tools/visualize_order_2d.py
```
