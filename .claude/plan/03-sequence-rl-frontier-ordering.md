# 序列级 RL 前沿排序方案 (核心未完成目标)

## 目标

用 RL 决策多步探索中"选择访问哪个前沿"，超越 FUEL 原生 TSP 规划器。
这是项目中 RL 应有优势但尚未充分探索的关键问题。

## 为什么单步 RL 不够

单步视点选择任务中 BC 已接近最优 (val loss 0.0539)。
RL 真正的价值在于**序列级决策**——牺牲短期视点质量换取长期探索效率。
`ViewpointEnv` 的 `rollout_steps` 参数正是为此设计，但实验中没有充分消融。

## 架构 (已就位)

```
SequenceEnv:
  观测 → {frontiers[N,8], mask[N], global[2], map_img[3,64,64]}
  动作 → 离散选择 frontier index
  奖励 → delta_coverage * 200 - travel_dist * 3 + completion_bonus

OrderPolicy:
  MapCNN(3,64,64) → map_embed[64]
  每前沿: [feat8 ⊕ global2 ⊕ map_embed64] → MLP → score
  池化: masked average → [feat8 ⊕ global2 ⊕ map_embed64] → V(s)
```

## 当前状况

- ✅ `SequenceEnv` 实现完整，含 fallback 逻辑
- ✅ `OrderPolicy` (CNN + MLP) 实现完整
- ✅ TSP baseline (NN+2opt, FUEL-style) 已实现
- ✅ `experiment.py --mode sequence` 已新增
- ❌ 序列级 RL 训练脚本 (SAC/PPO for SequenceEnv): `train_sac_seq.py`/`train_seq_ppo.py` 存在但状态不明
- ❌ 无 TSP vs RL 的结论性对比实验

## 实现路径

### Phase 1: 基线确立
```bash
python experiment/experiment.py --mode sequence --num-maps 50 --max-steps 50
```
产出: Closest / Biggest / MostVisible / TSP_NN2OPT / TSP_FUEL 的覆盖率对比表。

### Phase 2: BC 预训练 OrderPolicy
- 用 TSP 解 (TSP_FUEL 或 TSP_NN2OPT) 作为专家标签
- 每步: 收集 (frontiers, mask, global, map_img) → (selected_index)
- 用 CrossEntropy 训练 OrderPolicy

### Phase 3: PPO 微调
- 从 BC 预训练权重初始化
- 序列级 PPO: GAE(λ) 处理可变长 episode
- 奖励: 覆盖率增量 + 距离惩罚
- 对比: TSP vs BC vs PPO

### Phase 4: 课程学习
- Start: 5 frontiers, 20 steps, small map
- Middle: 10 frontiers, 50 steps
- Target: 20 frontiers, 100 steps

## 关键文件

| 文件 | 职责 |
|------|------|
| `fuel_rl/env/sequence_env.py` | 序列级 Gym 环境 |
| `fuel_rl/models/order_policy.py` | 前沿排序策略网络 |
| `fuel_rl/eval/tsp_baseline.py` | TSP 求解器 (NN+2opt) |
| `experiment/experiment.py --mode sequence` | 排序策略对比实验 |
| `fuel_rl/eval/eval_policies.py` | 策略评估框架 |
| `fuel_rl/train/train_seq_ppo.py` | 序列级 PPO (待验证) |
| `fuel_rl/train/train_sac_seq.py` | 序列级 SAC (待验证) |

## 与 search0614.md 的交叉

序列探索生成的前沿访问序列可以用覆盖率评估方法论分析:
- Rarefaction 曲线: 不同策略的 coverage vs steps
- 收益矩阵: 各策略在不同地图类型上的 pairwise 胜率
- UMAP 可视化: 前沿类型的嵌入空间覆盖
