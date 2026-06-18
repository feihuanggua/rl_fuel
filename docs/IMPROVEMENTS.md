# RL-FUEL 代码修改与改进意见

> 修改日期: 2026-06-18

本文档记录对 `rl_fuel` 项目所做的代码修改、背后的设计意见，以及后续值得探索的方向。

---

## 一、已实施的修改

### 1.1 🔴 修复 GPU 深度渲染器相机参数 (Critical Bug)

**文件**: `fuel_rl/env/gpu_depth_renderer.py`

**问题**: GPU 渲染器使用了错误的内参 (`fx=381.0, fy=381.0, cx=320.0, cy=240.0`), 而 C++ `simulateObservation` 使用 FUEL 官方参数 (`fx=387.229, fy=387.229, cx=321.046, cy=243.449`)。这导致 GPU 路径和 CPU 路径产生不一致的观测，使用 GPU 加速训练时模型看到的"世界"与评估时不同。

**修改**: 将 GPU 渲染器的默认参数对齐到 C++ 侧:
```python
fx=387.229, fy=387.229, cx=321.046, cy=243.449
```

**影响范围**: 所有使用 `SequenceEnv` + GPU 渲染器的训练和评估。训练时 GPU 路径速度快 ~40 倍，这个 bug 意味着 GPU 训练的模型在 CPU 评估时可能表现不一致。

---

### 1.2 🟡 实验框架增加 TSP 基线 + 序列级评估模式

**文件**: `experiment/experiment.py`

**问题**: 原实验只评估单步视点质量 (选最近前沿 → BC/REINFORCE 选视点)。真正需要 RL 解决的是**前沿排序问题**——面对多个前沿时选择哪个去探索。且缺少 FUEL 原生 TSP 规划器作为 baseline。

**修改**:
1. 新增 `--mode sequence` 模式，对比 5 种前沿排序策略:
   - **Closest** (贪心最近) — 基础 baseline
   - **Biggest / MostVisible** (贪心最大/最可见) — 启发式
   - **TSP_NN2OPT** — Nearest-Neighbor + 2-opt = 原始 FUEL/TARE 方案
   - **TSP_FUEL** — FUEL 风格 TSP: 全局规划 + 局部 top3 选可见性最高

2. 旧 `--mode viewpoint` 保留，向后兼容。

3. 新增 `run_exploration_sequence()` 函数，直接操作 `FuelEnvCore` C++ 对象进行多步探索，每步策略从前沿列表中做离散选择。

**运行**:
```bash
# 视点质量对比 (原有)
python experiment/experiment.py --mode viewpoint --num-maps 50 --max-steps 50

# 前沿排序对比 (新增, 含 TSP)
python experiment/experiment.py --mode sequence --num-maps 50 --max-steps 50
```

**设计原理**: TSP 是 FUEL 原论文的核心。NN+2opt 在 Euclidean 距离矩阵上求解访问顺序，以总行程最小为目标。TSP_FUEL 变体更接近 FUEL 的实际实现——全局规划确定访问顺序，局部根据 visibility 微调第一个目标。不将 TSP 作为 baseline 就无法回答"RL 是否真正超越了传统方法"。

---

### 1.3 🟢 优化 `build_3channel_grid` 函数

**文件**: `fuel_rl/data/collector.py`

**问题**: 原始实现是 Python 三重 `for` 循环，每迭代一次就调用 `core.get_occupancy(np.array([wx, wy, wz]))` 一次。`32*32*10=10,240` 次 Python→C++ 边界穿越，每次还附带 `np.array()` 的内存分配。

**修改**: 使用 `np.mgrid` 一次性预计算所有 10,240 个体素的世界坐标，单层展平循环查询 occupancy。消除:
- 三重 Python 循环的 overhead
- 10,240 次 `np.array()` 临时对象创建

**注意**: 每个体素的 `get_occupancy()` C++ 调用仍然存在，这是进一步的瓶颈。终极优化应在 C++ 侧实现 `getLocalVoxelGrid` 的批量返回 (该函数已有声明，需要确认 SDF Map 实现中的输出格式)。

---

### 1.4 🟢 新增相机参数一致性验证

**文件**: `fuel_rl/config.py` (新增 `validate_camera_params()`), `fuel_rl/env/sequence_env.py` (调用)

**功能**: 当 `SequenceEnv` 创建 GPU 渲染器时，自动检测其内参是否与 C++ 核心一致。如果检测到 mismatch 会通过 Python warnings 发出告警。

这从根本上防止了类似 1.1 的 bug 再次发生。

---

### 1.5 🟢 新增测试基础设施

**文件**: 
- `tests/test_unit.py` — 可脱离 C++ 核心运行的纯 Python 测试
- `tests/test_integration.py` — 需要 C++ 编译的集成测试

**覆盖范围**:

| 测试类 | 覆盖内容 |
|--------|----------|
| `TestRotationAugment` | 4 倍旋转增强正确性: identity, 90°, wrap, 4×=一圈 |
| `TestEncoder3D` | 编码器输出形状、不同输入尺寸、flat_dim 计算 |
| `TestViewpointHead` | tanh 输出范围、确定性推理 |
| `TestViewpointActorCritic` | action/value/log_prob 形状, dterministic 一致性 |
| `TestTSPSolver` | 距离矩阵对称性, NN tour 完整性, 2opt 改进, 单点退化 |
| `TestConfigValidation` | 正确参数通过, 错配检测, 修前参数检测 |
| `TestOrderPolicy` | logits/value 形状, mask 正确性, act 返回有效索引 |
| `TestFuelEnvCore` | init, load, observation, frontier, occupancy, path, reuse |
| `TestViewpointEnv` | reset shape, step 不崩溃, 多 episode |
| `TestSequenceEnv` | obs dict shapes, random rollout |
| `TestBuild3ChannelGrid` | 输出形状、前沿通道非零 |
| `TestExpertLabel` | 标签维度与范围 |

**运行**:
```bash
# 纯 Python 测试 (不需要 C++ 编译)
pytest tests/test_unit.py -v

# 全量测试 (需要 C++ 编译)
pytest tests/ -v
```

---

## 二、未改但建议修改的部分 (后续工作)

### 2.1 SAC Encoder 架构简化

**文件**: `fuel_rl/models/sac_models.py`

**现状**: SAC 维护 5 个独立的 3D CNN encoder (actor + q1 + q2 + target_q1 + target_q2), ~15M 参数。

**建议**: 共享 encoder 为所有 head 提供特征。单步视点选择中，同一个体素网格的编码对所有 head 是一致的——没有理由各自独立编码。

```python
class SACAgentShared(nn.Module):
    def __init__(self):
        self.shared_encoder = Encoder3D(...)  # 唯一一份
        self.actor_head = ActorHead(embed_dim)
        self.q1_head = QHead(embed_dim)
        self.q2_head = QHead(embed_dim)
```

参数从 ~15M → ~4M, 训练更稳定。

---

### 2.2 推进序列级 RL 训练

**核心观点**: 单步视点选择任务 BC 已接近最优 (val loss 0.0539)。RL 的探索噪声在多步序列中累积误差，导致整体覆盖率下降。**RL 的价值应该在前沿排序决策中体现**。

`SequenceEnv` + `OrderPolicy` 的架构已经完备，但缺少:
1. **序列级 PPO**: 目前 SAC/PPO 训练脚本只支持单步 `ViewpointEnv`。需要实现支持可变长 episode + GAE 的 PPO 训练器。
2. **课程学习**: 从 {5 frontiers, 20 steps} → {20 frontiers, 100 steps} 渐进式训练。
3. **TSP 预训练**: 用 TSP 解作为 BC 标签预训练 `OrderPolicy`，再用 PPO 微调。

参考 [opencode.md](docs/opencode.md) 中结论: "REINFORCE 虽然单步确定性评估略胜 BC (4.08 vs 3.81)，但多步探索中因序列一致性不足而落后"。这正是序列级 RL 需要解决的。

---

### 2.3 与 self-play 覆盖率评估的交叉应用

将 [search0614.md](C:\Users\jiading01\Documents\我的POPO\search0614.md) 中的方法论应用到 UAV 探索:

| search0614 方法 | UAV 探索对应 | 应用 |
|-----------------|-------------|------|
| Good-Turing 覆盖率 | 探索进度 $1 - f_1/n$ 中的单例前沿比例 | 判断是否需要更长时间探索 |
| Rarefaction 曲线 | 不同地图的覆盖率 vs 步数曲线 | 判断探索策略的采样效率上限 |
| MAP-Elites 存档 | 按 (前沿大小, 距离, 可见性) 分桶的行为空间 | 确保训练数据覆盖各种前沿类型 |
| 收益矩阵 | 不同探索策略的 pairwise 胜率热力图 | 分析策略间互补关系 |
| UMAP 可视化 | 前沿-视点的 3D embedding 降至 2D | 识别策略未覆盖的前沿"空白区" |

特别是，可以定义**前沿类型**作为 MAP-Elites 的行为描述符: (`frontier_size_bucket`, `distance_bucket`, `visibility_bucket`)，记录 BC/RL 模型在不同类型前沿上的表现分布。

---

### 2.4 其他工程改进

| 优先级 | 项目 | 说明 |
|--------|------|------|
| P1 | 日志系统统一 | 目前 print/std::cerr/tensorboard/csv 四种日志方式混杂。建议用 Python logging + 统一的 metrics collector |
| P1 | 配置文件 YAML | `config.py` 的 Python module 方式不便于 sweep。改用 YAML + dataclass 验证 |
| P2 | Docker 构建 | setup.py 中 PCL/Eigen 路径硬编码 Ubuntu 路径。Dockerfile 可降低上手成本 |
| P2 | 结果可复现脚本 | 所有实验参数、随机种子、模型版本应一键可复现 |
| P3 | Type hints 完善 | `visualizer.py` 已经加了 type hints，但训练脚本和环境代码中缺失 |
| P3 | C++ 内存泄漏根因 | `collector.py` 提到 C++ 内存泄漏导致使用子进程。建议用 valgrind/ASAN 定位 |

---

## 三、项目整体评价

### 贡献

1. **FUEL-FUEL-RL 桥接**: 将 FUEL 的 C++ 核心解耦为 standalone 库并通过 pybind11 暴露给 Python RL 框架，这是非平凡的系统工程。
2. **诚实的实验记录**: PPO 策略坍塌、SAC 无法超越 BC、REINFORCE std 爆炸——这些都是对社区有价值的信息。
3. **GPU 深度渲染器**: 从 ~200ms → ~5ms 的加速使得 RL 训练成为可能。

### 结论

这是一个"基础设施完备但核心问题定义有偏差"的项目。**单步视点选择本质是监督学习问题**，BC 已接近最优；真正需要 RL 的**多步序列优化问题**虽然环境与模型架构已经就位，但尚未推进到有结论性的实验阶段。

建议的下一步优先级:
1. 序列级 PPO 训练 + TSP baseline 对比 (利用已修改的 `experiment.py --mode sequence`)
2. SAC encoder 共享化简化
3. 将 search0614.md 的覆盖率评估引入探索质量评估

---

## 四、如何运行测试

```bash
conda activate torch
cd /path/to/rl_fuel

# 单元测试 (无需 C++ 编译)
pytest tests/test_unit.py -v

# 集成测试 (需要先编译 C++)
python setup.py build_ext --inplace
pytest tests/test_integration.py -v

# 运行改进后的实验
python experiment/experiment.py --mode sequence --num-maps 20 --max-steps 50
```
