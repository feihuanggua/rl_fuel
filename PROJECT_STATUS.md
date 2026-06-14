# 项目进展文档

## 项目概述

基于深度强化学习（SAC算法）的无人机视角选择与自主探索系统。在ARiADNE大地图（最大40m）上训练frontier访问顺序策略，目标是用更短的路径完成地图探索。

- **训练环境**: WSL (Ubuntu 20.04, conda环境 `rl_fuel`, Python 3.11)
- **路径**: `/home/jdwsl/rl_fuel/`
- **GPU**: RTX 5070 (sm_120)
- **可视化**: Windows Open3D (conda环境 `pytorch`, Python 3.10)

---

## 地图配置

- **地图来源**: ARiADNE PNG地图，9张（1/2/3/4/5/10/20/50/100.png）
- **分辨率**: 0.1m/pixel
- **地图裁剪**: 最大40m×40m（400px），以机器人起点为中心
- **Ray距离**: 4m（GPU renderer max_range=4.0, free_dist=4.0）
- **地图高度**: 2m，agent z=1.0m
- **obstacles_inflation**: 0.099
- **corridor宽度**: 3.2m（32px）
- **机器人起点**: PNG中value=208的像素位置
- **排除地图**: 2.png、3.png（有大片不可达封闭区域，所有方法覆盖率上限66%/78%）
- **20.png**: 也有部分不可达区域（~79%覆盖率上限）

### ARiADNE地图加载流程 (`map_loader.py`)
1. 读取PNG，识别value 127=墙/194=free/208=robot
2. 裁剪到free-space边界，最大400px
3. 填充所有occupied cells为实体墙（向量化numpy，~100万点）
4. 返回 `(points, map_w, map_h, start_x, start_y)`

---

## 训练历史

### Seq8（已完成）
- 200k步，混合ARiADNE+随机地图，20m地图，3m ray
- eval cov=95.2%，旧版配置，与新地图不兼容

### Seq9（已完成）
- 300k步，课程学习（easy→medium→hard）
- eval avg=89.8%，无显著提升

### Seq10（已完成）
- 500k步，纯ARiADNE，从seq8微调，lr=5e-5
- 2.png/3.png卡在66%，诊断为不可达区域问题

### Seq11（已完成）
- 500k步，跳过2.png/3.png，lr=3e-5
- 最佳avg_dist=104m（优于Closest 105m），eval avg=96.0%
- 旧版模型，与新大地图配置不兼容

### Seq12（即将完成，~460k/500k步）
- **配置**: 全新训练，4m ray，40m ARiADNE地图，7张训练地图（排除2/3.png）
- **参数**: lr=1e-4, buffer=200k, max_steps=800
- **服务**: systemd `rl-fuel-train12`（PID 223819，运行~16h）
- **Checkpoint**: `/home/jdwsl/rl_fuel/fuel_rl_checkpoints/sac_seq12/`

#### Seq12 训练结果
- **覆盖率**: 始终在87-88%震荡，最高88.3%（step 445k），无实质提升
- **路径长度**: 有效地图均值从早期2500-3000m降到2000-2400m，但方差极大
- **策略不稳定**: 同一地图不同评估轮次距离从500m到6000m剧烈波动
- **中位数reward恒为-0.1**: 大部分episode卡住（无frontier/目标被占），reward信号几乎无区分度

#### Seq12 可视化结果（best_actor, 7张有效地图）
| 地图 | 步数 | 覆盖率 | 路径长度 |
|------|------|--------|----------|
| 1.png | 800 | 84.6% | 5627m |
| 4.png | 800 | 93.7% | 3262m |
| 5.png | 800 | 91.1% | 4392m |
| 10.png | 800 | 87.4% | 5751m |
| 20.png | 少 | 78.4% | 449m |
| 50.png | 800 | 90.0% | 5919m |
| 100.png | 800 | 87.8% | 3446m |

#### 核心问题诊断
1. **Reward设计缺陷**: 旧reward `delta*200 - eucl_dist*3.0`，距离惩罚(3.0)过重，覆盖增益无法补偿
2. **卡住反而"划算"**: 卡住只罚-0.1/step，正常移动罚-15~-60/step，agent倾向卡住
3. **无重复路惩罚**: agent可以反复折返而不受额外惩罚

---

## Reward设计演进

### 当前代码位置
`/home/jdwsl/rl_fuel/fuel_rl/env/sequence_env.py`，step()函数

### v1（旧版，已废弃）
```python
reward = delta * 200.0 - eucl_dist * 3.0
if delta > 0.01:
    reward += 2.0
# 卡住: return -0.1
```
问题：距离惩罚太重(3.0x)，卡住惩罚太轻(-0.1)

### v2（已废弃）
- 2m网格visited_cells追踪
- 新区域距离惩罚0.3x，重复区域2.0x
- 问题：2m网格太细，房间内正常移动也被判为重复

### v3（当前版本，已部署）
```python
# 基础奖励
reward = delta * 200.0 - 1.0          # 覆盖增益 - 每步成本
reward -= eucl_dist * 0.5             # 统一轻度距离惩罚

# 区域级重复路检测（5m网格）
CELL_SIZE = 5.0
new_cell = (int(vp[0] / CELL_SIZE), int(vp[1] / CELL_SIZE))
if new_cell in self._cell_last_visit:
    steps_since = self.step_count - self._cell_last_visit[new_cell]
    if steps_since > 30:                    # 离开>30步才回来 → 折返惩罚
        reward -= min(steps_since * 0.3, 8.0)
self._cell_last_visit[new_cell] = self.step_count

# 里程碑
if delta > 0.01:
    reward += 2.0
if delta < 0.003 and eucl_dist > 5.0:   # 长距离移动但无覆盖增益
    reward -= 3.0

# 卡住惩罚: -3.0 (3处)
# 达标奖励: +100.0
```

**设计意图**：
- 同一房间/区域内探索（≤30步内回来）：不额外惩罚
- 离开区域>30步后折返：惩罚随离开步数增长，封顶-8.0
- 必要的远距离新区域移动：只受轻度0.5x距离惩罚
- 卡住：-3.0/step（比正常移动更重，不再"划算"）

### 备份文件
- `/home/jdwsl/rl_fuel/fuel_rl/env/sequence_env.py.bak_v1` — 原始版本
- `/home/jdwsl/rl_fuel/fuel_rl/env/sequence_env.py.bak_v2` — v2版本

---

## 模型架构

### Actor (OrderPolicy)
- 输入: frontiers[B,N,8] + mask[B,N] + global[B,2] + map_img[B,3,64,64]
- frontier特征: center(3) + size(1) + eucl_dist(1) + visib(1) + direction(2)
- MapCNN提取map_img特征(64维)
- MLP: per-frontier评分 → logits + masked
- 输出: Categorical分布，act()返回action

### Critic (QNetwork)
- 与Actor结构相同，输出Q值而非logits
- Twin Q-networks (critic1, critic2) + target networks

### SAC训练
- Actor lr: 1e-4, Critic lr: 5e-5
- Replay buffer: 200k
- Alpha: 自动调节（初始0.01）
- max_steps per episode: 800

---

## 关键文件索引

### WSL端 (`/home/jdwsl/rl_fuel/`)
| 文件 | 说明 |
|------|------|
| `fuel_rl/env/sequence_env.py` | 环境核心（reward v3已部署） |
| `fuel_rl/map_loader.py` | ARiADNE地图加载 |
| `fuel_rl/models/order_policy.py` | Actor/Critic模型 |
| `fuel_rl/config.py` | 配置(obstacles_inflation=0.099, max_ray_length=6.0) |
| `fuel_rl/env/gpu_depth_renderer.py` | GPU深度渲染(max_range=4.0) |
| `fuel_rl/train/train_sac_seq12.py` | Seq12训练脚本 |
| `fuel_rl_checkpoints/sac_seq12/best_actor.pth` | 最佳模型 |

### Windows端 (`C:\Users\jd3\Desktop\code\bishe_rl\`)
| 文件 | 说明 |
|------|------|
| `sequence_env.py` | 环境代码同步副本 |
| `train_sac_seq12.py` | 训练脚本副本 |
| `vis_sac_seq12.py` | SAC可视化脚本 |
| `vis_4m_40m.py` | Closest baseline可视化 |
| `vis_closest_large.py` | Closest大地图可视化 |
| `view_3d.py` | Open3D 3D回放 |
| `eval_all_maps.py` | 全地图评估脚本 |
| `plot_train_curves.py` | 训练曲线绘制 |
| `map_loader_new.py` | 新地图加载器副本 |
| `rules/git-commit.md` | Git提交规范 |

### systemd服务
| 服务 | 状态 |
|------|------|
| `~/.config/systemd/user/rl-fuel-train12.service` | active (running), seq12训练 |

---

## Closest Baseline（大地图，4m ray）
- 有效地图覆盖率: ~89.8%
- 总距离: 4500-7900m（7张有效地图合计）
- 每步距离: 5.7-9.9m

---

## Git管理
- 仓库: `C:\Users\jd3\Desktop\code\bishe_rl\`
- 提交规范: `rules/git-commit.md`
- 前缀: feat/fix/refactor/docs/chore/test/perf/style
- 忽略: *.pth, *.npz, 临时脚本, __pycache__

---

## 下一步计划

1. **等seq12跑完**（~40k步剩余，约1小时）
2. **用reward v3重新训练seq13**:
   - 停掉seq12服务
   - 创建 `train_sac_seq13.py`（基于seq12，新reward）
   - 全新训练，lr可尝试3e-4（reward信号更强，可以更激进）
   - 评估对比seq12和Closest baseline
3. **可能进一步调整**:
   - 如果v3 reward仍不收敛，考虑curriculum learning
   - 调整网络结构（更大hidden dim）
   - 考虑per-episode reward normalization

---

## 常用命令

```bash
# 查看训练状态
wsl -d ubuntu2004_jd2 -- bash -c "systemctl --user status rl-fuel-train12"

# 查看训练日志
wsl -d ubuntu2004_jd2 -- bash -c "journalctl --user -u rl-fuel-train12 --no-pager | grep EVAL"

# 查看最新训练step
wsl -d ubuntu2004_jd2 -- bash -c "journalctl --user -u rl-fuel-train12 --no-pager -n 20"

# 运行可视化
wsl -d ubuntu2004_jd2 -- bash -c "source /home/jdwsl/miniconda3/etc/profile.d/conda.sh && conda activate rl_fuel && cd /home/jdwsl/rl_fuel && python vis_sac_seq12.py"

# 运行训练曲线
wsl -d ubuntu2004_jd2 -- bash -c "source /home/jdwsl/miniconda3/etc/profile.d/conda.sh && conda activate rl_fuel && python /mnt/c/Users/jd3/Desktop/code/bishe_rl/plot_train_curves.py"

# Git提交
cd C:\Users\jd3\Desktop\code\bishe_rl
git add <files>
git commit -m "feat: 描述"
```

---

## 已知限制

1. **2.png/3.png**: 有大片不可达封闭区域，所有方法覆盖率上限66%/78%
2. **20.png**: 部分不可达，~79%覆盖率上限，路径只有~450m就无frontier
3. **地图点云量大**: 85万-170万点/地图，reset约3s，每step约0.04-0.09s
4. **训练速度**: ~200k步/小时（大地图慢于旧20m配置）
5. **WSL无法运行Open3D**: libffi v7/v8冲突，3D可视化在Windows端
