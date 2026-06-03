"""SAC for discrete frontier ordering with CNN + MLP.

Actor: OrderPolicy (CNN map encoder + per-frontier MLP → logits)
Critic: QNetwork (same architecture, separate params)
Twin Q + target Q, delayed actor update, reward normalization.
"""
import os, sys, csv, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.env.sequence_env import SequenceEnv, MAX_FRONTIERS
from fuel_rl.config import DEVICE


class QNetwork(nn.Module):
    def __init__(self, d_frontier=8, d_global=2, d_hidden=128, d_map=64):
        super().__init__()
        from fuel_rl.models.order_policy import MapCNN
        self.map_cnn = MapCNN(out_dim=d_map)
        d_per = d_frontier + d_global + d_map
        self.feat_net = nn.Sequential(
            nn.Linear(d_per, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 2**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frontiers, mask, global_feat, map_img=None):
        B, N, _ = frontiers.shape
        map_embed = self.map_cnn(map_img) if map_img is not None else torch.zeros(B, 64, device=frontiers.device)
        g = global_feat.unsqueeze(1).expand(-1, N, -1)
        m = map_embed.unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([frontiers, g, m], dim=-1)
        q = self.feat_net(feat).squeeze(-1)
        q = q.masked_fill(mask < 0.5, 0.0)
        return q


class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    @staticmethod
    def _compress_obs(obs):
        return {
            "frontiers": obs["frontiers"],
            "mask": obs["mask"],
            "global": obs["global"],
            "map_img": (obs["map_img"] * 255).astype(np.uint8),
        }

    @staticmethod
    def _decompress_obs(obs):
        return {
            "frontiers": obs["frontiers"],
            "mask": obs["mask"],
            "global": obs["global"],
            "map_img": obs["map_img"].astype(np.float32) / 255.0,
        }

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((
            self._compress_obs(obs), action, reward,
            self._compress_obs(next_obs), done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        s0 = [self._decompress_obs(b[0]) for b in batch]
        s1 = [self._decompress_obs(b[3]) for b in batch]
        frontiers = torch.FloatTensor(np.stack([s["frontiers"] for s in s0])).to(DEVICE)
        mask = torch.FloatTensor(np.stack([s["mask"] for s in s0])).to(DEVICE)
        global_f = torch.FloatTensor(np.stack([s["global"] for s in s0])).to(DEVICE)
        map_imgs = torch.FloatTensor(np.stack([s["map_img"] for s in s0])).to(DEVICE)
        actions = torch.LongTensor([b[1] for b in batch]).to(DEVICE)
        rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(DEVICE)
        n_frontiers = torch.FloatTensor(np.stack([s["frontiers"] for s in s1])).to(DEVICE)
        n_mask = torch.FloatTensor(np.stack([s["mask"] for s in s1])).to(DEVICE)
        n_global = torch.FloatTensor(np.stack([s["global"] for s in s1])).to(DEVICE)
        n_map = torch.FloatTensor(np.stack([s["map_img"] for s in s1])).to(DEVICE)
        dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(DEVICE)
        return frontiers, mask, global_f, map_imgs, actions, rewards, n_frontiers, n_mask, n_global, n_map, dones

    def __len__(self):
        return len(self.buf)


def train_sac(args):
    os.makedirs(args.save_dir, exist_ok=True)
    env = SequenceEnv(max_steps=args.max_env_steps, num_pillars=args.num_pillars)

    start_step = 1
    csv_path = os.path.join(args.save_dir, "sac_log.csv")

    actor = OrderPolicy().to(DEVICE)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_lr = args.lr * 0.5

    critic1 = QNetwork().to(DEVICE)
    critic2 = QNetwork().to(DEVICE)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr)

    target1 = QNetwork().to(DEVICE)
    target2 = QNetwork().to(DEVICE)

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        if len(lines) > 1:
            try:
                start_step = int(float(lines[-1].strip().split(",")[0])) + 1
            except:
                start_step = 1
        ckpt = os.path.join(args.save_dir, f"actor_{start_step - 1}.pth")
        if not os.path.exists(ckpt):
            for i in range(start_step - 1, max(0, start_step - 5001), -200):
                ckpt = os.path.join(args.save_dir, f"actor_{i}.pth")
                if os.path.exists(ckpt):
                    start_step = i + 1
                    break
        latest = os.path.join(args.save_dir, "latest_checkpoint.pth")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location=DEVICE)
            actor.load_state_dict(ckpt["actor"])
            critic1.load_state_dict(ckpt["critic1"])
            critic2.load_state_dict(ckpt["critic2"])
            start_step = ckpt.get("step", start_step)
            print(f"Resumed from step {start_step}")

    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic1.state_dict())
    for t in [target1, target2]:
        for p in t.parameters():
            p.requires_grad_(False)

    alpha = 0.01
    buffer = ReplayBuffer(capacity=args.buffer_size)
    reward_rms = RunningMeanStd()
    tau = 0.01

    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if start_step == 1:
        csv_w.writerow(["step", "avg_r", "coverage", "alpha"])
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    obs, _ = env.reset()
    best_coverage = -float("inf")
    update_count = 0
    n_skip = 0
    n_ep = 0
    ep_rew = 0.0
    ep_cov = 0.0

    t0_act = t0_env = t0_upd = 0.0
    print(f"[DEBUG] start_step={start_step}, total_steps={args.total_steps}, range=({start_step}, {args.total_steps+1})", flush=True)
    for step in range(start_step, args.total_steps + 1):
        t_a = time.time()
        frontiers_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
        mask_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
        global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
        map_t = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = actor(frontiers_t, mask_t, global_t, map_t)
            logits = logits.squeeze(0)
            n_valid = int(mask_t.sum().item())
            if n_valid == 0:
                n_skip += 1
                obs, _ = env.reset()
                if ep_rew > 0:
                    ep_cov = max(ep_cov, 0.0)
                n_ep += 1
                ep_rew = 0.0
                continue
            valid_logits = logits[:n_valid]
            probs = F.softmax(valid_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        t0_act += time.time() - t_a

        action_np = action.item()
        t_e = time.time()
        next_obs, reward, done, truncated, info = env.step(action_np)
        ep_rew += reward
        buffer.add(obs, action_np, reward, next_obs, float(done))
        t0_env += time.time() - t_e

        if done:
            n_ep += 1
            ep_cov_final = info.get("coverage", 0)
            ep_rew = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if info.get("coverage", 0) > best_coverage:
            best_coverage = info["coverage"]

        t_u = time.time()
        if len(buffer) >= args.batch_size and step % 2 == 0:
            s_f, s_m, s_g, s_map, a, r, ns_f, ns_m, ns_g, ns_map, d = buffer.sample(args.batch_size)

            r_norm = r

            with torch.no_grad():
                ns_logits, _ = actor(ns_f, ns_m, ns_g, ns_map)
                ns_probs = F.softmax(ns_logits, dim=-1)
                ns_probs = ns_probs * ns_m
                ns_probs = ns_probs / (ns_probs.sum(dim=-1, keepdim=True) + 1e-8)
                ns_logprobs = torch.log(ns_probs + 1e-8)

                t1_q = target1(ns_f, ns_m, ns_g, ns_map)
                t2_q = target2(ns_f, ns_m, ns_g, ns_map)
                t_min = torch.min(t1_q, t2_q)
                v_next = (ns_probs * (t_min - alpha * ns_logprobs)).sum(dim=-1, keepdim=True)
                target_q = r_norm + args.gamma * (1 - d) * v_next

            c1_q = critic1(s_f, s_m, s_g, s_map).gather(1, a.unsqueeze(-1))
            c2_q = critic2(s_f, s_m, s_g, s_map).gather(1, a.unsqueeze(-1))
            critic_loss = F.mse_loss(c1_q, target_q) + F.mse_loss(c2_q, target_q)

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 1.0)
            critic_opt.step()

            update_count += 1

            if update_count % 4 == 0:
                a_logits, _ = actor(s_f, s_m, s_g, s_map)
                a_probs = F.softmax(a_logits, dim=-1)
                a_probs = a_probs * s_m
                a_probs = a_probs / (a_probs.sum(dim=-1, keepdim=True) + 1e-8)
                a_logprobs = torch.log(a_probs + 1e-8)

                with torch.no_grad():
                    c1_a = critic1(s_f, s_m, s_g, s_map)
                    c2_a = critic2(s_f, s_m, s_g, s_map)
                    c_min = torch.min(c1_a, c2_a)

                actor_loss = (a_probs * (alpha * a_logprobs - c_min)).sum(dim=-1).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()
            else:
                actor_loss = torch.tensor(0.0)

            for tp, p in zip(list(target1.parameters()) + list(target2.parameters()),
                             list(critic1.parameters()) + list(critic2.parameters())):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            if update_count % 20 == 0:
                tb.add_scalar("loss/critic", critic_loss.item(), step)
                tb.add_scalar("loss/actor", actor_loss.item(), step)
                with torch.no_grad():
                    q_mean = torch.min(c1_q, torch.min(
                        critic1(s_f, s_m, s_g, s_map).gather(1, a.unsqueeze(-1)),
                        critic2(s_f, s_m, s_g, s_map).gather(1, a.unsqueeze(-1))
                    )).mean().item()
                tb.add_scalar("q_value", q_mean, step)
        t0_upd += time.time() - t_u

        if step % args.log_every == 0:
            import resource
            mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            print(f"Step {step:6d}: cov={info.get('coverage',0):.3f} "
                  f"reward={reward:+.3f} buffer={len(buffer)} alpha={alpha:.4f} "
                  f"skip={n_skip} eps={n_ep} "
                  f"mem={mem_gb:.1f}GB "
                  f"time[act={t0_act:.1f}s env={t0_env:.1f}s upd={t0_upd:.1f}s]")
            sys.stdout.flush()
            t0_act = t0_env = t0_upd = 0.0
            csv_w.writerow([step, reward, info.get("coverage", 0), alpha])
            csv_f.flush()
            tb.add_scalar("train/coverage", info.get("coverage", 0), step)
            tb.add_scalar("train/reward", reward, step)

        if step % 5000 == 0 and step > 0:
            actor.eval()
            eval_covs = []
            for _ in range(5):
                eobs, _ = env.reset()
                for es in range(400):
                    nv = int(eobs["mask"].sum())
                    if nv == 0:
                        eval_covs.append(env.core.get_exploration_progress())
                        break
                    ef = torch.FloatTensor(eobs["frontiers"]).unsqueeze(0).to(DEVICE)
                    em = torch.FloatTensor(eobs["mask"]).unsqueeze(0).to(DEVICE)
                    eg = torch.FloatTensor(eobs["global"]).unsqueeze(0).to(DEVICE)
                    emap = torch.FloatTensor(eobs["map_img"]).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        el, _ = actor(ef, em, eg, emap)
                    ea = int(el[0][:nv].argmax().item())
                    eobs, er, ed, et, einfo = env.step(ea)
                    if ed:
                        eval_covs.append(einfo["coverage"])
                        break
            actor.train()
            obs, _ = env.reset()
            avg_eval = float(np.mean(eval_covs))
            print(f"  [EVAL step={step}] det_cov={avg_eval:.3f} over {len(eval_covs)} eps")
            tb.add_scalar("eval_coverage", avg_eval, step)

        if step % 1000 == 0:
            torch.save(actor.state_dict(), os.path.join(args.save_dir, f"actor_{step}.pth"))
            torch.save({
                "step": step, "actor": actor.state_dict(),
                "critic1": critic1.state_dict(), "critic2": critic2.state_dict(),
            }, os.path.join(args.save_dir, "latest_checkpoint.pth"))
            print(f"  [saved actor_{step}]")

    csv_f.close()
    torch.save(actor.state_dict(), os.path.join(args.save_dir, "final_actor.pth"))
    print(f"Done. Best cov: {best_coverage:.3f}")


def main():
    parser = argparse.ArgumentParser(description="SAC Discrete Frontier Ordering")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/sac_seq")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-env-steps", type=int, default=400)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()
    train_sac(args)


if __name__ == "__main__":
    main()
