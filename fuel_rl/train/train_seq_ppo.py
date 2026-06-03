"""序列级 PPO: 优化前沿访问顺序，BC 负责视点生成."""
import os, csv, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.env.sequence_env import SequenceEnv, MAX_FRONTIERS
from fuel_rl.config import DEVICE


def _eval_coverage(model, env, seed=0, steps=10):
    """快速评估覆盖率 (确定性策略)."""
    import torch
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        frontiers_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
        mask_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
        global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            act, _, _ = model.act(frontiers_t, mask_t, global_t, deterministic=True)
        obs, _, done, _, info = env.step(act.item())
        if done:
            break
    return info.get("coverage", 0)


class RolloutBuffer:
    def __init__(self):
        self.obs = {"frontiers": [], "mask": [], "global": []}
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, logprob, reward, value, done):
        self.obs["frontiers"].append(obs["frontiers"])
        self.obs["mask"].append(obs["mask"])
        self.obs["global"].append(obs["global"])
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def to_tensors(self):
        frontiers = torch.FloatTensor(np.stack(self.obs["frontiers"])).to(DEVICE)
        mask = torch.FloatTensor(np.stack(self.obs["mask"])).to(DEVICE)
        global_f = torch.FloatTensor(np.stack(self.obs["global"])).to(DEVICE)
        actions = torch.LongTensor(self.actions).to(DEVICE)
        logprobs = torch.FloatTensor(self.logprobs).to(DEVICE)
        rewards = torch.FloatTensor(self.rewards).to(DEVICE)
        values = torch.FloatTensor(self.values).to(DEVICE)
        dones = torch.FloatTensor(self.dones).to(DEVICE)
        return frontiers, mask, global_f, actions, logprobs, rewards, values, dones


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE advantage + returns."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        gae = delta + gamma * lam * (1 - next_done) * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def train_sequence_ppo(args):
    os.makedirs(args.save_dir, exist_ok=True)

    env = SequenceEnv(max_steps=args.max_env_steps, num_pillars=args.num_pillars)
    model = OrderPolicy()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = RolloutBuffer()

    csv_path = os.path.join(args.save_dir, "seq_ppo_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["timesteps", "avg_reward", "avg_coverage", "loss", "entropy"])
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    timestep = 0
    best_coverage = -float("inf")

    for episode in range(args.max_episodes):
        obs, _ = env.reset()

        for step in range(env.max_steps):
            frontiers_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
            mask_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
            global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)

            # Debug NaN in input
            if torch.isnan(frontiers_t).any() or np.isnan(obs["frontiers"]).any():
                print(f"  NAN INPUT at ep={episode} step={step}", flush=True)
                break

            with torch.no_grad():
                action, logprob, value = model.act(frontiers_t, mask_t, global_t)
            action_i = action.item()

            next_obs, reward, done, truncated, info = env.step(action_i)
            buffer.add(obs, action_i, logprob.item(), reward, value.item(), done)

            obs = next_obs
            timestep += 1

            if done or truncated:
                break

        # 评估指标
        coverage = info.get("coverage", 0)
        avg_r = np.mean(buffer.rewards)
        csv_w.writerow([timestep, avg_r, coverage, 0.0, 0.0])
        csv_f.flush()
        tb.add_scalar("coverage", coverage, timestep)
        tb.add_scalar("reward", avg_r, timestep)

        if episode % args.log_every == 0:
            print(f"Ep {episode:5d}: ts={timestep} cov={coverage:.3f} avg_r={avg_r:+.4f} buf={len(buffer.rewards)}")

        if coverage > best_coverage:
            best_coverage = coverage
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        # PPO update with validation
        if len(buffer.rewards) >= args.update_timestep:
            # Save pre-update state
            pre_update = {k: v.clone() for k, v in model.state_dict().items()}

            # Compute pre-update coverage on a test map (same seed for consistency)
            test_env = SequenceEnv(max_steps=20, num_pillars=args.num_pillars)
            pre_cov = _eval_coverage(model, test_env, seed=99999)

            frontiers, mask, global_f, actions, logprobs_old, rewards, values, dones = buffer.to_tensors()
            advantages, returns = compute_gae(rewards, values, dones, args.gamma, args.lam)

            if torch.isnan(advantages).any() or torch.isnan(returns).any():
                print("  [NaN, skipping]", flush=True)
                buffer.clear()
                continue

            for _ in range(args.k_epochs):
                logprobs, entropy, values_new = model.evaluate(frontiers, mask, global_f, actions)
                if torch.isnan(logprobs).any():
                    break
                ratio = torch.exp(logprobs - logprobs_old)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * nn.MSELoss()(values_new, returns)
                loss = actor_loss + critic_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validate: revert if worse
            post_cov = _eval_coverage(model, test_env, seed=99999)
            if post_cov < pre_cov - 0.01:
                print(f"  [REVERT: cov {pre_cov:.3f}→{post_cov:.3f}]", flush=True)
                model.load_state_dict(pre_update)
                buffer.clear()  # 清除旧数据，全新开始
            else:
                print(f"  [UPDATE: cov {pre_cov:.3f}→{post_cov:.3f}]", flush=True)

            tb.add_scalar("loss", loss.item(), timestep)
            buffer.clear()

        if episode % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "checkpoint.pth"))

    csv_f.close()
    tb.close()
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print(f"Done. Best coverage: {best_coverage:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Sequence PPO for Frontier Ordering")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/seq_ppo")
    parser.add_argument("--max-episodes", type=int, default=10000)
    parser.add_argument("--max-env-steps", type=int, default=50)
    parser.add_argument("--update-timestep", type=int, default=2000)
    parser.add_argument("--k-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()
    train_sequence_ppo(args)


if __name__ == "__main__":
    main()
