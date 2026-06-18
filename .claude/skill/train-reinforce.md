# Skill: REINFORCE 微调训练

## 触发条件
- BC 训练完成后
- 想尝试超越 BC 基线

## 背景

REINFORCE 是当前唯一能微超 BC 的 RL 方法 (单步 eval: 4.08 vs BC 3.81)。
关键: **log_std 必须固定 (0.15)，不可学习！** 否则会膨胀导致崩溃。

## 执行步骤

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# 从 BC 权重开始 REINFORCE
python -c "
import sys
sys.argv = ['train_reinforce', '--total-steps', '50000', '--log-every', '200', '--num-pillars', '15']
from fuel_rl.train.train_reinforce import main
main()
"
```

## 关键配置

```python
class ReinforcePolicy:
    fixed_std = 0.15  # ← 核心! 不可改为可学习
    self.register_buffer("log_std", torch.full((1, 4), np.log(fixed_std)))

# 训练
lr = 1e-4
baseline_ema = 0.95  # EMA of reward
total_steps = 50000
log_every = 200
```

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| avg_r100 | > 1.5 | < 0 → 策略退化 |
| error_rate | < 10% | > 30% → 太多无效动作 |
| baseline | 2~4 | 逐步上升正常 |
| loss | 小幅波动 | 持续 > 0 → 梯度消失 |

## 实验历史

| std 配置 | 结果 |
|-----------|------|
| std=0.14 可学习 | 1.1→1.6 缓慢上升 |
| std=0.50 | 80% 无效动作 |
| std=0.30 可学习 | std 膨胀→0.48→策略冻结 |
| **std=0.15 固定** | ✅ 超越 BC (4.08 vs 3.81) |

## 评估

```bash
# 单步确定性评估
python -c "
from fuel_rl.train.train_reinforce import ReinforcePolicy
from fuel_rl.env.viewpoint_env import ViewpointEnv
import torch, numpy as np
model = ReinforcePolicy()
model.load_state_dict(torch.load('./fuel_rl_checkpoints/reinforce/best_model.pth'))
model.eval()
# ... 运行评估循环
"
```
