"""SAC model — 复用 BC 编码器，双 Q 网络 + 自动熵调.

编码器不共享为子模块，避免 state_dict 重复。
"""
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_, zeros_
from fuel_rl.models.encoder import Encoder3D, ResMLP, ChannelAttention3D, SpatialAttention3D
from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE


class SACActor(nn.Module):
    """Actor: encoder → mean/std → tanh-squashed action."""

    def __init__(self, input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM):
        super().__init__()
        self.encoder = Encoder3D(input_shape=input_shape, channels=channels, embed_dim=embed_dim)
        self.pos_net = nn.Sequential(ResMLP(embed_dim), ResMLP(embed_dim))
        self.pos_mean = nn.Linear(embed_dim, 3)
        self.yaw_net = nn.Sequential(
            nn.Linear(embed_dim + 3, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), ResMLP(256),
        )
        self.yaw_mean = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((1, 4), -1.0))
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 1
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        feat = self.encoder(x)
        pos_feat = self.pos_net(feat)
        pos_mean = torch.tanh(self.pos_mean(pos_feat))
        yaw_in = torch.cat([feat, pos_mean.detach()], dim=-1)
        yaw_feat = self.yaw_net(yaw_in)
        yaw_mean = torch.tanh(self.yaw_mean(yaw_feat))
        mean = torch.cat([pos_mean, yaw_mean], dim=-1)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def encode(self, x):
        return self.encoder(x)


class SACQNetwork(nn.Module):
    """Q-network: encoder + action → Q(s,a)."""

    def __init__(self, input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM):
        super().__init__()
        self.encoder = Encoder3D(input_shape=input_shape, channels=channels, embed_dim=embed_dim)
        self.q_net = nn.Sequential(
            nn.Linear(embed_dim + 4, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1),
            ResMLP(256), nn.Linear(256, 1),
        )
        self._init()

    def _init(self):
        for m in self.q_net.modules():
            if isinstance(m, nn.Linear):
                orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x, action):
        feat = self.encoder(x)
        return self.q_net(torch.cat([feat, action], dim=-1))


class SACAgent(nn.Module):
    """SAC Agent: Actor + Twin Qs + Target Qs.

    Each network has its own encoder (copied from BC). Q encoders share with actor during BC load.
    """

    def __init__(self, input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM):
        super().__init__()
        self.actor = SACActor(input_shape, channels, embed_dim)
        self.q1 = SACQNetwork(input_shape, channels, embed_dim)
        self.q2 = SACQNetwork(input_shape, channels, embed_dim)
        self.target_q1 = SACQNetwork(input_shape, channels, embed_dim)
        self.target_q2 = SACQNetwork(input_shape, channels, embed_dim)

        # Init target networks
        for tq in [self.target_q1, self.target_q2]:
            tq.load_state_dict(self.q1.state_dict())  # copy from q1 initially

        for tq in [self.target_q1, self.target_q2]:
            for p in tq.parameters():
                p.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.target_entropy = -4.0
        self.to(DEVICE)

    def load_bc_pretrained(self, bc_path):
        """Load BC pretrained encoder + pos/yaw heads into actor and Q encoders."""
        bc_state = torch.load(bc_path, map_location="cpu", weights_only=False)
        model_state = self.state_dict()

        name_map = {
            "pos_out.weight": "actor.pos_mean.weight",
            "pos_out.bias": "actor.pos_mean.bias",
            "yaw_out.weight": "actor.yaw_mean.weight",
            "yaw_out.bias": "actor.yaw_mean.bias",
        }

        loaded = 0
        for k, v in bc_state.items():
            target_k = name_map.get(k, k)
            if k.startswith("encoder."):
                target_k = "actor." + k
            if k.startswith("pos_net"):
                target_k = "actor." + k
            if k.startswith("yaw_net"):
                target_k = "actor." + k
            if target_k in model_state and v.shape == model_state[target_k].shape:
                model_state[target_k] = v
                loaded += 1

        # Copy actor encoder to Q network encoders
        encoder_keys = [k for k in model_state if k.startswith("actor.encoder.")]
        for ek in encoder_keys:
            qk = "q1." + ek[len("actor."):]
            q2k = "q2." + ek[len("actor."):]
            t1k = "target_q1." + ek[len("actor."):]
            t2k = "target_q2." + ek[len("actor."):]
            if qk in model_state:
                model_state[qk] = model_state[ek].clone()
                model_state[q2k] = model_state[ek].clone()
                model_state[t1k] = model_state[ek].clone()
                model_state[t2k] = model_state[ek].clone()

        self.load_state_dict(model_state)
        print(f"Loaded {loaded}/{len(model_state)} params from BC checkpoint")
