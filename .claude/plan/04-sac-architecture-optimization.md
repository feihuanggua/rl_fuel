# SAC 架构优化方案

## 目标

简化 SAC 模型架构，减少参数量，提升训练稳定性。

## 问题诊断

### 当前架构

```
SACAgent:
  ├── actor: SACActor (encoder ~0.5M + head ~0.3M)
  ├── q1:    SACQNetwork (encoder ~0.5M + q_head ~0.2M)
  ├── q2:    SACQNetwork (encoder ~0.5M + q_head ~0.2M)
  ├── target_q1: SACQNetwork (encoder ~0.5M + q_head ~0.2M)
  └── target_q2: SACQNetwork (encoder ~0.5M + q_head ~0.2M)
────────────────────────────────────────────────
  Total: ~15M 参数, 5 个独立 encoder!
```

### 问题

1. **参数冗余**: 5 个独立 3D CNN encoder 编码同一个体素网格。Encoder 提取的是空间特征，对所有 head 应该是通用的。
2. **训练不稳定**: 多个独立 encoder 各自更新梯度方向可能冲突。
3. **内存占用**: batch_size=256 时 GPU 显存紧张。

### 目标架构

```
SACAgentShared:
  ├── shared_encoder: Encoder3D (唯一 ~0.5M)
  ├── actor_head: ActorHead (from embedding + optional action_input)
  ├── q1_head: QHead (embedding + action → Q)
  └── q2_head: QHead (embedding + action → Q)
────────────────────────────────────────────────
  Total: ~4M 参数, 1 个 encoder

训练时:
1. batch 通过 shared_encoder 一次得到 embeddings
2. actor_head(embeddings) → actions, log_probs
3. q1_head(embeddings, actions) → q1
4. q2_head(embeddings, actions) → q2
```

## 实现步骤

### Step 1: 重构 SACActor
```python
class SACActorHead(nn.Module):
    def __init__(self, embed_dim):
        # pos_net + pos_mean + yaw_net + yaw_mean (同原 actor 但不含 encoder)
        self.pos_net = nn.Sequential(ResMLP(embed_dim), ResMLP(embed_dim))
        self.pos_mean = nn.Linear(embed_dim, 3)
        self.yaw_net = nn.Sequential(...)
        self.yaw_mean = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((1,4), -1.0))

    def forward(self, embedding):  # 不再接收原始 voxel
        ...
```

### Step 2: 重构 SACQNetwork
```python
class SACQHead(nn.Module):
    def __init__(self, embed_dim):
        self.q_net = nn.Sequential(
            nn.Linear(embed_dim+4, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1),
            ResMLP(256), nn.Linear(256, 1),
        )

    def forward(self, embedding, action):  # embedding 预计算
        ...
```

### Step 3: 共享 Encoder 的 SACAgent
```python
class SACAgentShared(nn.Module):
    def __init__(self):
        self.encoder = Encoder3D(...)  # 仅此一份
        self.actor = SACActorHead(embed_dim)
        self.q1 = SACQHead(embed_dim)
        self.q2 = SACQHead(embed_dim)
        self.target_q1 = SACQHead(embed_dim)
        self.target_q2 = SACQHead(embed_dim)

    def encode(self, x):
        return self.encoder(x)
```

### Step 4: 训练循环修改
```python
# 前向一次 encoder
embeddings = agent.encode(obs_batch)
# 各 head 共享 embeddings
actions = agent.actor.sample(embeddings)
q1_pred = agent.q1(embeddings, actions)
q2_pred = agent.q2(embeddings, actions)
```

## 关键文件

| 文件 | 变更 |
|------|------|
| `fuel_rl/models/sac_models.py` | 重构为共享 encoder 架构 |
| `fuel_rl/train/train_sac.py` | 适配新的 forward 模式 |

## 预期收益

| 指标 | 当前 | 优化后 |
|------|------|--------|
| 参数 | ~15M | ~4M |
| GPU 显存 (batch=256) | ~3GB | ~1.2GB |
| 单步前向时间 | ~12ms | ~4ms |
| 训练稳定性 | 差 (多 encoder 梯度冲突) | 改善 |
