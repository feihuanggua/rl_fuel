"""前沿排序策略: CNN(2D map) + MLP(frontier feats) → logits.

Architecture:
  - MapCNN: 2D 俯视图 [3, 64, 64] → embedding [64]
    - MLP: 每前沿特征 [6] + global [2] + map_embed [64] → score
  - Value: pooled frontiers + global + map_embed → V(s)
"""
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_, zeros_
from fuel_rl.config import DEVICE


class MapCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class OrderPolicy(nn.Module):
    """CNN + MLP → frontier logits + value.

    Input:
      frontiers: [B, N, 6]  center(3) + size(1) + eucl_dist(1) + visib(1)
      mask:      [B, N]     1=valid
      global:    [B, 2]     coverage, step
      map_img:   [B, 3, 64, 64]  2D occupancy map
    Output:
      logits:    [B, N]     masked
      value:     [B, 1]     state value
    """

    def __init__(self, d_frontier=8, d_global=2, d_hidden=128, d_map=64):
        super().__init__()
        self.map_cnn = MapCNN(out_dim=d_map)

        d_per_frontier = d_frontier + d_global + d_map
        self.feat_net = nn.Sequential(
            nn.Linear(d_per_frontier, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

        d_value = d_frontier + d_global + d_map
        self.value_net = nn.Sequential(
            nn.Linear(d_value, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, frontiers, mask, global_feat, map_img=None):
        B, N, _ = frontiers.shape

        map_embed = self.map_cnn(map_img) if map_img is not None else torch.zeros(B, 64, device=frontiers.device)

        g = global_feat.unsqueeze(1).expand(-1, N, -1)
        m = map_embed.unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([frontiers, g, m], dim=-1)

        logits = self.feat_net(feat).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, -1e9)

        pooled = (frontiers * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        v_in = torch.cat([pooled, global_feat, map_embed], dim=-1)
        value = self.value_net(v_in)

        return logits, value

    def act(self, frontiers, mask, global_feat, map_img=None, deterministic=False):
        logits, value = self.forward(frontiers, mask, global_feat, map_img)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, frontiers, mask, global_feat, map_img, action):
        logits, value = self.forward(frontiers, mask, global_feat, map_img)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)
