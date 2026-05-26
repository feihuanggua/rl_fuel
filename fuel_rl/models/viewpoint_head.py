"""解耦视点预测头: 位置流 + 偏航流."""
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_, zeros_
from fuel_rl.models.encoder import ResMLP


class ViewpointHead(nn.Module):
    """BC 推理用: 编码器 → 视点.

    Input:  [B, 3, V, V, V]
    Output: [B, 4]  (dx, dy, dz, dyaw) in [-1, 1]
    """

    def __init__(self, encoder: nn.Module, embed_dim=512):
        super().__init__()
        self.encoder = encoder

        # 位置流
        self.pos_net = nn.Sequential(
            ResMLP(embed_dim), ResMLP(embed_dim),
        )
        self.pos_out = nn.Linear(embed_dim, 3)

        # 偏航流 (以位置预测为条件)
        self.yaw_net = nn.Sequential(
            nn.Linear(embed_dim + 3, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1),
            ResMLP(256),
        )
        self.yaw_out = nn.Linear(256, 1)

        self._init_heads()

    def _init_heads(self):
        for m in [*self.pos_net.modules(), *self.yaw_net.modules()]:
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, (2 ** 0.5))
                if m.bias is not None:
                    zeros_(m.bias)
        orthogonal_(self.pos_out.weight, 0.01)
        orthogonal_(self.yaw_out.weight, 0.01)
        zeros_(self.pos_out.bias)
        zeros_(self.yaw_out.bias)

    def forward(self, x):
        feat = self.encoder(x)
        pos_feat = self.pos_net(feat)
        pos = torch.tanh(self.pos_out(pos_feat))

        yaw_in = torch.cat([feat, pos.detach()], dim=-1)
        yaw_feat = self.yaw_net(yaw_in)
        yaw = torch.tanh(self.yaw_out(yaw_feat))

        return torch.cat([pos, yaw], dim=-1)  # [B, 4]


class ViewpointActorCritic(nn.Module):
    """PPO 用: Actor-Critic 结构.

    Actor: 编码器 → 解耦头 → 高斯策略
    Critic: 编码器 → Value
    """

    def __init__(self, encoder: nn.Module, embed_dim=512):
        super().__init__()
        self.encoder = encoder

        # Actor: 位置流
        self.pos_net = nn.Sequential(ResMLP(embed_dim), ResMLP(embed_dim))
        self.pos_mean = nn.Linear(embed_dim, 3)

        # Actor: 偏航流
        self.yaw_net = nn.Sequential(
            nn.Linear(embed_dim + 3, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1),
            ResMLP(256),
        )
        self.yaw_mean = nn.Linear(256, 1)

        # Actor: log-std (initialized for more exploration)
        self.log_std = nn.Parameter(torch.full((1, 4), -1.0))  # std~0.37 (vs -2.0→0.14)

        # Critic
        self.critic_net = nn.Sequential(ResMLP(embed_dim), ResMLP(embed_dim))
        self.critic_out = nn.Linear(embed_dim, 1)

        self._init_heads()

    def _init_heads(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'encoder' not in name:
                orthogonal_(m.weight, (2 ** 0.5))
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        feat = self.encoder(x)

        # 位置
        pos_feat = self.pos_net(feat)
        pos_mean = torch.tanh(self.pos_mean(pos_feat))

        # 偏航
        yaw_in = torch.cat([feat, pos_mean.detach()], dim=-1)
        yaw_feat = self.yaw_net(yaw_in)
        yaw_mean = torch.tanh(self.yaw_mean(yaw_feat))

        mean = torch.cat([pos_mean, yaw_mean], dim=-1)  # [B, 4]
        std = torch.exp(self.log_std).expand_as(mean)

        # Value
        val = self.critic_out(self.critic_net(feat))

        return mean, std, val

    def get_action(self, x, deterministic=False):
        mean, std, val = self.forward(x)
        if deterministic:
            return mean, val
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(-1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, val

    def evaluate(self, x, action):
        mean, std, val = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, val
