"""3D CNN 编码器 + 注意力模块."""
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_, zeros_


class ChannelAttention3D(nn.Module):
    """SE 风格通道注意力."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        w = x.mean(dim=(2, 3, 4))  # [B, C]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention3D(nn.Module):
    """空间注意力."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class ResMLP(nn.Module):
    """残差 MLP 块."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.LeakyReLU(0.1),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder3D(nn.Module):
    """3D CNN 编码器.

    Input:  [B, 3, V, V, V]  (obstacle, frontier, free)
    Output: [B, embed_dim]
    """

    def __init__(self, grid_size=32, channels=(32, 64, 128), embed_dim=512, input_shape=None):
        """input_shape: (D,H,W) 实际输入尺寸, 如 (32,32,10). None 则用 grid_size 立方体."""
        super().__init__()
        self.grid_size = grid_size
        self.input_shape = input_shape
        in_ch = 3
        layers = []
        for out_ch in channels:
            layers.append(nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # 注意力
        last_ch = channels[-1]
        self.ch_attn = ChannelAttention3D(last_ch)
        self.sp_attn = SpatialAttention3D()

        # 计算展平维度
        with torch.no_grad():
            if input_shape:
                dummy = torch.zeros(1, 3, *input_shape)
            else:
                dummy = torch.zeros(1, 3, grid_size, grid_size, grid_size)
            feat = self.conv(dummy)
            self._flat_dim = feat.numel()

        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                orthogonal_(m.weight, (2 ** 0.5))
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.ch_attn(x)
        x = self.sp_attn(x)
        x = x.flatten(1)
        return self.fc(x)
