# Skill: 运行实验评估

## 触发条件
- 训练完成后
- 需要对比不同方法/检查点
- 论文写作时需要数据

## 执行步骤

### 视点质量实验 (单步)

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# 标准实验: 50 maps × 50 steps
python experiment/experiment.py --mode viewpoint --num-maps 50 --max-steps 50

# 快速测试: 10 maps × 20 steps
python experiment/experiment.py --mode viewpoint --num-maps 10 --max-steps 20
```

### 前沿排序实验 (序列级, 含 TSP)

```bash
# 核心实验! 对比 TSP vs 贪心
python experiment/experiment.py --mode sequence --num-maps 50 --max-steps 50
```

### 评估已训练的 SAC 检查点

```bash
python -m fuel_rl.eval.quick_eval --checkpoint fuel_rl_checkpoints/sac_seq5/best_model.pth
```

### 全面评估 (所有基线 + 所有检查点)

```bash
python -m fuel_rl.eval.run_eval
# 输出: fuel_rl_checkpoints/sac_seq/eval_results.json
```

## 输出指标

| 指标 | 含义 |
|------|------|
| `final_progress` | 最终探索覆盖率 (%) |
| `valid_steps` | 有效步数 (观测成功) |
| `avg_reward` | 平均每步奖励 |
| `total_distance` | 总飞行距离 (m) |
| `progress_curve` | 逐步覆盖率序列 |

## 汇总示例

```
Method          final_progress    valid_steps     avg_reward  total_distance
----------------------------------------------------------------------
TSP_FUEL            0.312±0.12      42.3±8.1      2.45±0.8      85.3±18.2
TSP_NN2OPT          0.295±0.11      40.1±7.8      2.31±0.7      88.1±19.1
Biggest             0.245±0.10      38.6±9.2      1.92±0.9      92.4±21.3
Closest             0.218±0.09      35.0±8.5      1.65±0.8      95.7±22.1
MostVisible         0.203±0.11      32.1±9.8      1.41±1.0     102.3±25.4
```

## 结果可视化

```bash
# 生成对比图表 (bar + boxplot)
python -m fuel_rl.eval.plot_results

# 生成轨迹可视化 (trajectories + coverage curves)
python -m fuel_rl.eval.vis_ordering
```
