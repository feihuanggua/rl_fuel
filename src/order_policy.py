"""Frontier ordering policy with self-attention.

Architecture:
  MapCNN: [3,64,64] → [d_map]
  FrontierEncoder: per-frontier feat + map + global → self-attention → contextualized feats
  OrderPolicy: encoder + score head → logits; attention pooling → value
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_, zeros_
from fuel_rl.config import DEVICE


class MapCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim), nn.ReLU(),
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


class FrontierEncoder(nn.Module):
    """Shared encoder: MapCNN + self-attention over frontiers."""

    def __init__(self, d_frontier=8, d_global=2, d_map=70, d_hidden=256):
        super().__init__()
        self.map_cnn = MapCNN(out_dim=d_map)
        d_per = d_frontier + d_global + d_map
        self.d_per = d_per

        self.attn = nn.MultiheadAttention(
            embed_dim=d_per, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_per)
        self.ffn = nn.Sequential(
            nn.Linear(d_per, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_per),
        )
        self.ffn_norm = nn.LayerNorm(d_per)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, frontiers, mask, global_feat, map_img):
        B, N, _ = frontiers.shape
        map_embed = self.map_cnn(map_img)
        g = global_feat.unsqueeze(1).expand(-1, N, -1)
        m = map_embed.unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([frontiers, g, m], dim=-1)

        key_padding_mask = (mask < 0.5)
        attn_out, _ = self.attn(
            feat, feat, feat, key_padding_mask=key_padding_mask)
        feat = self.attn_norm(feat + attn_out)

        ffn_out = self.ffn(feat)
        feat = self.ffn_norm(feat + ffn_out)

        return feat, map_embed


class OrderPolicy(nn.Module):
    """Frontier ordering policy with self-attention.

    Input:
      frontiers: [B, N, 8]
      mask: [B, N]
      global_feat: [B, 2]
      map_img: [B, 3, 64, 64]
    Output:
      logits: [B, N] masked
      value: [B, 1]
    """

    def __init__(self, d_frontier=8, d_global=2, d_hidden=256, d_map=70):
        super().__init__()
        self.encoder = FrontierEncoder(d_frontier, d_global, d_map, d_hidden)
        d_per = self.encoder.d_per

        self.score_net = nn.Sequential(
            nn.Linear(d_per, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

        self.v_attn = nn.Linear(d_per, 1)
        self.value_net = nn.Sequential(
            nn.Linear(d_per, d_hidden), nn.ReLU(),
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
        feat, map_embed = self.encoder(frontiers, mask, global_feat, map_img)

        logits = self.score_net(feat).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, -1e9)

        v_scores = self.v_attn(feat).squeeze(-1)
        v_scores = v_scores.masked_fill(mask < 0.5, -1e9)
        v_weights = F.softmax(v_scores, dim=-1)
        pooled = (feat * v_weights.unsqueeze(-1)).sum(dim=1)
        value = self.value_net(pooled)

        return logits, value

    def act(self, frontiers, mask, global_feat, map_img=None,
            deterministic=False):
        logits, value = self.forward(frontiers, mask, global_feat, map_img)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, value.squeeze(-1)
