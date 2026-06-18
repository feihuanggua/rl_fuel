# Skill: 训练 BC 模型

## 触发条件
- 有新专家数据
- 调试编码器/视点头架构
- 准备 RL 微调的预训练权重

## 执行步骤

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# 标准训练
python -m fuel_rl.train.train_bc \
  --data-path ./fuel_rl_data/expert_data.pt \
  --save-dir ./fuel_rl_checkpoints/bc_v4 \
  --epochs 100 \
  --batch-size 128

# 断点续训
python -m fuel_rl.train.train_bc \
  --data-path ./fuel_rl_data/expert_data.pt \
  --save-dir ./fuel_rl_checkpoints/bc_v4 \
  --resume ./fuel_rl_checkpoints/bc_v4/latest_checkpoint.pth
```

## 输出

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最低 val loss 的模型 |
| `latest_checkpoint.pth` | 包含 optimizer/epoch/history (断点续训用) |
| `bc_loss.png` | 训练曲线 (log scale) |

## 关键配置

```python
# fuel_rl/config.py
BC_BATCH_SIZE = 128
BC_LR = 1e-3
BC_WEIGHT_DECAY = 1e-4
BC_EPOCHS = 100
BC_VAL_SPLIT = 0.1
ENCODER_CHANNELS = [32, 64, 128]
ENCODER_EMBED_DIM = 512
```

## 训练监控

- ReduceLROnPlateau: val loss 连续 5 epoch 不降则 lr 减半
- 每 5 epoch 打印 train/val loss
- 最佳模型的 val loss 通常 < 0.06

## 预期结果

| 指标 | 目标值 |
|------|--------|
| 训练样本 | ~14k (含旋转增强) |
| 参数量 | ~3.7M |
| Best val MSE | ~0.054 |
| 确定性 reward (单步) | ~3.81 |

## 注意事项

- `--grid-z 10` 必须与数据收集时一致 (32×32×10)
- resume 使用 `latest_checkpoint.pth` 而非 `best_model.pth` (后者不含 optimizer state)
- 数据增强在 Dataset 中通过切片实现 (augment=True: ×4)
