"""REINFORCE 训练 — 纯策略梯度，无 Q/critic.

单步任务: s → π(a|s) → r(s,a) → ∇log π(a|s) * (r - baseline)
"""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from fuel_rl.models.encoder import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointHead, ResMLP
from fuel_rl.env.viewpoint_env import ViewpointEnv
from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE


class ReinforcePolicy(nn.Module):
    """策略: BC 编码器 + 解耦头 + 可学习的 log_std."""

    def __init__(self, input_shape=(32, 32, 10), channels=ENCODER_CHANNELS,
                 embed_dim=ENCODER_EMBED_DIM):
        super().__init__()
        self.encoder = Encoder3D(input_shape=input_shape, channels=channels, embed_dim=embed_dim)

        self.pos_net = nn.Sequential(ResMLP(embed_dim), ResMLP(embed_dim))
        self.pos_mean = nn.Linear(embed_dim, 3)

        self.yaw_net = nn.Sequential(
            nn.Linear(embed_dim + 3, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), ResMLP(256),
        )
        self.yaw_mean = nn.Linear(256, 1)

        # 固定 std ~0.15: 微小扰动微调 BC，不可学习防止膨胀
        fixed_std = 0.15
        self.register_buffer("log_std", torch.full((1, 4), np.log(fixed_std)))
        self.to(DEVICE)

    def forward(self, x):
        feat = self.encoder(x)
        pos_feat = self.pos_net(feat)
        pos_mean = torch.tanh(self.pos_mean(pos_feat))
        yaw_in = torch.cat([feat, pos_mean.detach()], dim=-1)
        yaw_feat = self.yaw_net(yaw_in)
        yaw_mean = torch.tanh(self.yaw_mean(yaw_feat))
        mean = torch.cat([pos_mean, yaw_mean], dim=-1)
        std = torch.exp(self.log_std)
        return mean, std

    def act(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample().clamp(-1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean

    def load_bc(self, bc_path):
        bc_state = torch.load(bc_path, map_location="cpu", weights_only=False)
        model_state = self.state_dict()
        name_map = {
            "pos_out.weight": "pos_mean.weight", "pos_out.bias": "pos_mean.bias",
            "yaw_out.weight": "yaw_mean.weight", "yaw_out.bias": "yaw_mean.bias",
        }
        loaded = 0
        for k, v in bc_state.items():
            target_k = name_map.get(k, k)
            if target_k in model_state and v.shape == model_state[target_k].shape:
                model_state[target_k] = v
                loaded += 1
        self.load_state_dict(model_state)
        print(f"Loaded {loaded}/{len(model_state)} params from BC")


def train_reinforce(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model = ReinforcePolicy()
    model.load_bc(args.bc_ckpt)

    env = ViewpointEnv(num_pillars=args.num_pillars)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.save_dir, "reinforce_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "avg_r100", "loss"])
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    baseline = 0.0
    best_avg = -float("inf")
    all_rewards = []

    for step in range(1, args.total_steps + 1):
        obs, _ = env.reset()
        grid = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        action_t, log_prob_t, mean_t = model.act(grid)
        action_np = action_t.detach().cpu().numpy().flatten()
        obs_next, reward, terminated, truncated, info = env.step(action_np)
        all_rewards.append(reward)

        # REINFORCE loss: -log_prob * (reward - baseline)
        advantage = reward - baseline
        loss = -log_prob_t.mean() * advantage

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Update baseline (EMA of reward)
        baseline = 0.95 * baseline + 0.05 * reward

        if step % args.log_every == 0:
            avg_r100 = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            std = torch.exp(model.log_std).detach().cpu().numpy().flatten()
            err_rate = np.mean([1 if r < 0 else 0 for r in all_rewards[-500:]]) if all_rewards else 0

            print(f"Step {step:6d}: avg_r100={avg_r100:+7.3f} loss={loss.item():.4f} "
                  f"err={err_rate:.0%} std={std} baseline={baseline:+.2f}")
            csv_w.writerow([step, avg_r100, loss.item()])
            csv_f.flush()
            tb.add_scalar("reward", avg_r100, step)
            tb.add_scalar("loss", loss.item(), step)
            tb.add_scalar("baseline", baseline, step)
            tb.add_scalar("std_pos", std[:3].mean(), step)
            tb.add_scalar("std_yaw", std[3], step)

            if avg_r100 > best_avg:
                best_avg = avg_r100
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        if step % 5000 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"step_{step}.pth"))

    csv_f.close()
    tb.close()
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print(f"Done. Best avg: {best_avg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="REINFORCE")
    parser.add_argument("--bc-ckpt", type=str, default="./fuel_rl_checkpoints/bc_v3/best_model.pth")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/reinforce")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()
    train_reinforce(args)


if __name__ == "__main__":
    main()
