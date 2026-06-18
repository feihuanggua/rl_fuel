# REINFORCE / PPO 微调方案

## 目标

在 BC 预训练基础上用 RL 微调视点选择策略，尝试超越 BC 基线。

## 实验历史与结论

| 方法 | 配置 | 结果 | 结论 |
|------|------|------|------|
| PPO v2 | log_std=-2.0 (std=0.14), update=2000 | reward 1.5→0 崩溃 | 过于保守 + Critic 无法训练 |
| PPO v3 | log_std=-1.0 (std=0.37), KL penalty, entropy=0.1 | 训练中搁置 | explained_variance=0 |
| SAC | BC预热2000步 → SAC 1000步 | avg_r=-1.04, 无法超BC | 4 Q 网络太重 |
| REINFORCE std=0.14 | 可学习std | reward 1.1→1.6 → std膨胀0.48 | 可学习std会膨胀 |
| **REINFORCE std=0.15** | **std固定** | **单步eval 4.08 vs BC 3.81** | ✅ 固定小std能超越BC |

## 核心洞察

单步视点选择任务，BC 已接近理论最优 (val loss 0.0539)。
RL 的微小优势在单步确定性评估中可以体现 (4.08 vs 3.81)，
但在多步探索中由于序列累积误差反而落后 (BC 21.5% vs REINFORCE 17.8%)。

## 当前最佳实现

- 文件: `fuel_rl/train/train_reinforce.py`
- 策略: `ReinforcePolicy` with fixed `std=0.15` (register_buffer)
- baseline: EMA of reward (α=0.95)
- 关键: `log_std` 不可学习 !!! — 否则膨胀导致策略崩溃

## 关键文件

| 文件 | 职责 |
|------|------|
| `fuel_rl/train/train_reinforce.py` | REINFORCE 主训练 (最佳当前结果) |
| `fuel_rl/train/train_ppo.py` | PPO 微调 (已搁置) |
| `fuel_rl/train/train_sac.py` | SAC 训练 (未超越 BC) |
| `fuel_rl/env/viewpoint_env.py` | 单步视点 Gym 环境 |
| `fuel_rl/models/viewpoint_head.py` | ViewpointActorCritic (PPO用) |
| `fuel_rl/models/sac_models.py` | SACActor + SACQNetwork + SACAgent |

## 后续方向

- [ ] P0: 推进序列级 RL (见此方案的 rollback → 序列级 PPO)
- [ ] P1: 实现 `rollout_steps > 0` 奖励模式的ablation (已在 env 中实现但未消融)
- [ ] P2: SAC encoder 共享化 (减少 15M→4M 参数)
