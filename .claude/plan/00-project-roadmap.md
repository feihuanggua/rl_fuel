# RL-FUEL 项目总览与路线图

## 项目位置

`C:\marl\rl_fuel` / `/home/jd3/FUEL/rl_fuel`

## 项目目标

用强化学习优化 UAV 自主探索：替代 FUEL 的 TSP 启发式，实现更高效的未知环境探索。

## 架构概览

```
C++ 核心 (pybind11)              Python RL 框架
┌─────────────────────┐     ┌──────────────────────┐
│ SDF Map              │     │ ViewpointEnv          │
│ Frontier Finder      │◄───►│ SequenceEnv           │
│ A* Path Search       │     │ OrderPolicy (CNN+MLP) │
│ Raycast Sensing      │     │ ViewpointHead (3DCNN) │
│ Depth Rendering (GPU)│     │ PPO / SAC / REINFORCE │
└─────────────────────┘     └──────────────────────┘
```

## 目录结构

```
rl_fuel/
├── .claude/                   # ← 本次新增
│   ├── plan/                  # 功能方案 (6 个)
│   │   ├── 01-viewpoint-bc-training.md
│   │   ├── 02-reinforce-finetuning.md
│   │   ├── 03-sequence-rl-frontier-ordering.md
│   │   ├── 04-sac-architecture-optimization.md
│   │   ├── 05-coverage-evaluation-integration.md
│   │   └── 06-gpu-renderer-verification.md
│   └── skill/                 # 工具技能 (8 个)
│       ├── build-cpp-core.md
│       ├── collect-expert-data.md
│       ├── train-bc-model.md
│       ├── train-reinforce.md
│       ├── run-experiment.md
│       ├── evaluate-checkpoint.md
│       ├── visualize-exploration.md
│       └── diagnose-training.md
├── src/standalone/            # C++ 核心
├── src/bindings/              # pybind11 绑定
├── fuel_rl/                   # Python RL 框架
├── experiment/                # 实验脚本
├── tests/                     # 测试 (新增)
├── docs/                      # 文档
└── tools/                     # 辅助脚本
```

## 实验进展总结

| 阶段 | 方法 | 状态 | 结果 |
|------|------|------|------|
| 视点选择 | BC (行为克隆) | ✅ 完成 | val loss=0.054, 接近最优 |
| 视点选择 | REINFORCE 微调 | ✅ 完成 | 单步 4.08 vs BC 3.81, 多步落后 |
| 视点选择 | PPO 微调 | ❌ 搁置 | 策略坍塌, Critic 无法训练 |
| 视点选择 | SAC 微调 | ❌ 未超越 | 平均 reward 负值 |
| 前沿排序 | SequenceEnv + TSP 基线 | ✅ 刚实现 | 待实验 |
| 前沿排序 | 序列级 RL | ❌ 未实现 | 环境/模型已就位 |

## 优先级路线图

### 🔴 P0: 必须做 (解锁核心结论)

1. **TSP vs 贪心序列实验** (`experiment.py --mode sequence`)
   - 产出: 前沿排序 baselines 对比表
2. **序列级 PPO 训练** 
   - 从 TSP 预训练 → PPO 微调 OrderPolicy
   - 产出: RL 是否超越 TSP 的结论

### 🟡 P1: 应该做 (提升质量)

3. **SAC encoder 共享化** (`plan/04`)
   - 15M→4M 参数, 更快训练
4. **覆盖率评估集成** (`plan/05`)
   - Rarefaction 曲线 + Gini 系数
5. **GPU vs CPU 一致性测试** (`plan/06`)

### 🟢 P2: 可以做 (提升完整性)

6. Docker 构建
7. YAML 化配置
8. Type hints 完善
9. C++ 内存泄漏根因

## 关键文件快速索引

| 想做什么 | 文件 |
|----------|------|
| 编译 C++ | `setup.py` |
| 收集专家数据 | `fuel_rl/data/collector.py` |
| 训练 BC | `fuel_rl/train/train_bc.py` |
| RL 微调 | `fuel_rl/train/train_reinforce.py` |
| 视点实验 | `experiment/experiment.py --mode viewpoint` |
| 排序实验 | `experiment/experiment.py --mode sequence` |
| 评估模型 | `fuel_rl/eval/quick_eval.py` |
| TSP 基线 | `fuel_rl/eval/tsp_baseline.py` |
| 可视化 | `fuel_rl/visualizer.py` |
| 配置 | `fuel_rl/config.py` |
| 测试 | `tests/test_unit.py` |
