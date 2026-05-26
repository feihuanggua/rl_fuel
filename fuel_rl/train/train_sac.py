"""SAC 训练 — 从 BC 预训练开始微调."""
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

from fuel_rl.models.sac_models import SACAgent
from fuel_rl.env.viewpoint_env import ViewpointEnv
from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE

# ── Replay Buffer ──
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs = torch.FloatTensor(np.stack([b[0] for b in batch])).to(DEVICE)
        action = torch.FloatTensor(np.stack([b[1] for b in batch])).to(DEVICE)
        reward = torch.FloatTensor(np.stack([b[2] for b in batch])).unsqueeze(1).to(DEVICE)
        next_obs = torch.FloatTensor(np.stack([b[3] for b in batch])).to(DEVICE)
        done = torch.FloatTensor(np.stack([b[4] for b in batch])).unsqueeze(1).to(DEVICE)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buf)


# ── SAC Trainer ──
def train_sac(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model = SACAgent()
    model.load_bc_pretrained(args.bc_ckpt)

    env = ViewpointEnv(num_pillars=args.num_pillars)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    tau = 0.005   # target network polyak
    gamma = 0.99  # discount

    q_opt = torch.optim.Adam(
        list(model.q1.parameters()) + list(model.q2.parameters()), lr=args.lr_critic)
    actor_opt = torch.optim.Adam(model.actor.parameters(), lr=args.lr_actor)
    alpha_opt = torch.optim.Adam([model.log_alpha], lr=args.lr_alpha)

    # CSV
    csv_path = os.path.join(args.save_dir, "sac_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if os.path.getsize(csv_path) == 0:
        csv_w.writerow(["episode", "avg_reward", "actor_loss", "q_loss", "alpha"])

    tb = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

    # ── Q 网络预热: 用 BC 策略收集经验，训练 Q 预测 reward ──
    if args.q_warmup_steps > 0:
        print(f"Q-network warmup: {args.q_warmup_steps} steps...")
        model.actor.eval()
        obs, _ = env.reset()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        for step in range(1, args.q_warmup_steps + 1):
            with torch.no_grad():
                mean, _ = model.actor(obs_t)  # deterministic BC action
                action = mean
            action_np = action.cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            buffer.add(obs, action_np, reward, next_obs, float(terminated or truncated))

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

            if step % 500 == 0:
                avg_r = np.mean([b[2] for b in list(buffer.buf)[-500:]])
                print(f"  Warmup {step}/{args.q_warmup_steps}: avg_r={avg_r:+7.3f} buffer={len(buffer)}")

        # Train Q-networks on collected experiences
        print("Training Q-networks on warmup data...")
        q_opt.zero_grad()
        for _ in range(500):
            s, a, r, ns, d = buffer.sample(args.batch_size)
            q1_pred = model.q1(s, a)
            q2_pred = model.q2(s, a)
            q_loss = nn.MSELoss()(q1_pred, r) + nn.MSELoss()(q2_pred, r)
            q_loss.backward()
            nn.utils.clip_grad_norm_(list(model.q1.parameters()) + list(model.q2.parameters()), 1.0)
            q_opt.step()
            q_opt.zero_grad()
        # Update targets
        for tq, q in [(model.target_q1, model.q1), (model.target_q2, model.q2)]:
            tq.load_state_dict(q.state_dict())
        print(f"  Q warmup done. alpha={model.log_alpha.exp().item():.4f}")
        model.actor.train()

    # ── 主训练循环 ──
    episode_rewards = []
    ep_reward = 0
    best_avg = -float("inf")

    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

    for global_step in range(1, args.total_steps + 1):
        # Sample action
        with torch.no_grad():
            action, _, _ = model.actor.sample(obs_t)
        action_np = action.cpu().numpy().flatten()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        buffer.add(obs, action_np, reward, next_obs, float(done))
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            ep_reward = 0
        else:
            obs = next_obs
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        # SAC update
        if len(buffer) >= args.batch_size and global_step % args.update_every == 0:
            for _ in range(args.updates_per_step):
                s, a, r, ns, d = buffer.sample(args.batch_size)

                # Compute target Q
                with torch.no_grad():
                    na, na_logp, _ = model.actor.sample(ns)
                    q1_t = model.target_q1(ns, na)
                    q2_t = model.target_q2(ns, na)
                    q_t = torch.min(q1_t, q2_t)
                    target = r + gamma * (1 - d) * (q_t - model.log_alpha.exp() * na_logp)

                # Q loss
                q1_pred = model.q1(s, a)
                q2_pred = model.q2(s, a)
                q_loss = nn.MSELoss()(q1_pred, target) + nn.MSELoss()(q2_pred, target)

                q_opt.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(list(model.q1.parameters()) + list(model.q2.parameters()), 1.0)
                q_opt.step()

                # Actor loss
                na, na_logp, _ = model.actor.sample(s)
                q1_a = model.q1(s, na)
                q2_a = model.q2(s, na)
                q_a = torch.min(q1_a, q2_a)
                actor_loss = (model.log_alpha.exp().detach() * na_logp - q_a).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
                actor_opt.step()

                # Alpha loss
                alpha_loss = -(model.log_alpha.exp() * (na_logp.detach() + model.target_entropy)).mean()

                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # Polyak update targets
                for tq, q in [(model.target_q1, model.q1), (model.target_q2, model.q2)]:
                    for tp, p in zip(tq.parameters(), q.parameters()):
                        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        # Logging
        if global_step % args.log_every == 0:
            avg_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            n_eps = len(episode_rewards)
            alpha = model.log_alpha.exp().item()

            print(f"Step {global_step:6d}: avg_r100={avg_r:+7.3f} eps={n_eps} alpha={alpha:.4f}")
            csv_w.writerow([global_step, avg_r, 0.0, 0.0, alpha])
            csv_f.flush()
            tb.add_scalar("reward", avg_r, global_step)
            tb.add_scalar("alpha", alpha, global_step)

            if avg_r > best_avg:
                best_avg = avg_r
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        if global_step % 5000 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "checkpoint.pth"))

    csv_f.close()
    tb.close()
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print(f"SAC done. Best avg reward: {best_avg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="SAC Fine-tuning")
    parser.add_argument("--bc-ckpt", type=str, default="./fuel_rl_checkpoints/bc_v3/best_model.pth")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/sac")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-alpha", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--q-warmup-steps", type=int, default=2000)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()
    train_sac(args)


if __name__ == "__main__":
    main()
