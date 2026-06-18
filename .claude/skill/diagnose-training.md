# Skill: 诊断训练问题

## 触发条件
- 训练 loss / reward 异常
- 模型不收敛或崩溃
- C++ 核心行为不符预期

## 诊断清单

### 1. 环境健康检查

```bash
conda activate torch
cd /home/jd3/FUEL/rl_fuel

# 快速测试 (需要 tools/test_env.py)
python tools/test_env.py

# 手动测试贪心基线 (零偏移策略)
python -c "
from fuel_rl.env import FuelRLEnvSingleFrontier
import numpy as np
env = FuelRLEnvSingleFrontier(num_pillars=15, max_steps=100)
for ep in range(3):
    obs, _ = env.reset()
    total = 0
    for i in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        total += r
        if term or trunc: break
    print(f'Ep {ep}: reward={total:.0f}, progress={info.get(\"exploration_progress\",0):.1%}')
"
```

### 2. 序列环境诊断

```bash
python -m fuel_rl.diagnose_env
```

检查项: 每步的 n_valid, reward, n_fails

### 3. 常见问题速查

| 现象 | 可能原因 | 检查方法 |
|------|----------|----------|
| 所有 step 返回 no_path | 地图边界过小 / agent 在障碍物中 | `core.get_occupancy(agent_pos)` 应为 1 |
| frontier_count=0 | cluster_min 过大 | 检查 `default_frontier_params()` 的 cluster_min |
| GPU/CPU 路径结果不同 | 相机参数不一致 (已修复) | `validate_camera_params()` 自动检查 |
| 训练中 reward 突然归零 | log_std 膨胀 → 动作全被 clamp | 打印 `torch.exp(model.log_std)` |
| C++ 段错误 | PCL 内存问题 | 用子进程包装 (collector 已做) |
| BC 模型输出始终接近 0 | 最后一层 mean bias 初始化过小 | 检查 `zeros_(self.pos_out.bias)` |

### 4. BC 训练不收敛

```bash
# 检查数据完整性
python -c "
import torch
data = torch.load('./fuel_rl_data/expert_data.pt', weights_only=False)
print(f'Samples: {len(data[\"inputs\"])}')
print(f'Input shape: {data[\"inputs\"].shape}')  # [N,3,32,32,10]
print(f'Target range: [{data[\"targets\"].min():.2f}, {data[\"targets\"].max():.2f}]')
print(f'Target std: {data[\"targets\"].std(dim=0)}')
"
```

### 5. RL 策略退化诊断

```python
# 在训练循环中插入
if step % log_every == 0:
    std = torch.exp(model.log_std).detach().cpu().numpy()
    print(f"std=[{std[0,0]:.3f},{std[0,1]:.3f},{std[0,2]:.3f},{std[0,3]:.3f}]")

    # 检查梯度
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    print(f"grad_norm: {total_norm**0.5:.4f}")
```

### 6. C++ 性能诊断

```bash
# 运行时会自动打印 SLOW 日志 (>500ms)
# 查看哪些步骤慢:
grep "SLOW" train.log
# [SLOW detect] total=850ms search=450ms compute=200ms getInfo=150ms nf=8
# [SLOW sim_obs >1s] total=1200ms raycast=900ms inputPC=100ms inflate=150ms esdf=50ms
```

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `tools/test_env.py` | 环境快速测试 |
| `tools/diagnose_order.py` | 排序策略诊断 |
| `tools/test_frontier_shape.py` | 前沿形状检查 |
| `tools/compare_checkpoints.py` | 检查点对比 |
| `fuel_rl/diagnose_env.py` | 序列环境诊断 |
