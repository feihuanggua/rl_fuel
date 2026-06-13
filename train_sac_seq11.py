"""SAC Seq11: Fine-tune from seq8, skip unreachable maps (2,3), 
pure ARiADNE, 500k steps, lr=3e-5, curriculum-free."""
import os, sys, csv, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.env.sequence_env import SequenceEnv, MAX_FRONTIERS
from fuel_rl.config import DEVICE


ARIADNE_MAPS = ["1.png", "4.png", "5.png",
                "10.png", "20.png", "50.png", "100.png"]

EVAL_MAPS = ["1.png", "2.png", "3.png", "4.png", "5.png",
             "10.png", "20.png", "50.png", "100.png"]


def make_env_ariadne(max_env_steps=800):
    maps_dir = os.path.join(os.path.dirname(__file__), "..", "..", "maps", "ariadne")
    name = ARIADNE_MAPS[np.random.randint(len(ARIADNE_MAPS))]
    mp = os.path.join(maps_dir, name)
    return SequenceEnv(max_steps=max_env_steps, map_size=(20, 20, 2),
                       map_type='ariadne', map_path=mp)


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


class ReplayBuffer:
    def __init__(self, capacity=200000):
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


def train_sac_seq11(args):
    os.makedirs(args.save_dir, exist_ok=True)
    env = make_env_ariadne(args.max_env_steps)

    actor = OrderPolicy().to(DEVICE)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    critic1 = QNetwork().to(DEVICE)
    critic2 = QNetwork().to(DEVICE)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=args.lr * 0.5)

    target1 = QNetwork().to(DEVICE)
    target2 = QNetwork().to(DEVICE)

    start_step = 1
    latest = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if not os.path.exists(latest) and args.pretrained:
        latest = args.pretrained
    if os.path.exists(latest):
        ckpt = torch.load(latest, map_location=DEVICE)
        actor.load_state_dict(ckpt["actor"])
        critic1.load_state_dict(ckpt["critic1"])
        critic2.load_state_dict(ckpt["critic2"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Loaded from {latest}, step={start_step}")

    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    for t in [target1, target2]:
        for p in t.parameters():
            p.requires_grad_(False)

    alpha = 0.01
    buffer = ReplayBuffer(capacity=args.buffer_size)
    tau = 0.005

    csv_path = os.path.join(args.save_dir, "sac_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if start_step <= 1:
        csv_w.writerow(["step", "reward", "coverage", "alpha"])
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    obs, _ = env.reset()
    best_eval = -float("inf")
    update_count = 0
    n_skip = 0
    n_ep = 0
    ep_rew = 0.0
    t0_act = t0_env = t0_upd = 0.0

    print(f"[SEQ11] start={start_step}, total={args.total_steps}, ARiADNE (skip 2,3), lr={args.lr}", flush=True)

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
                env = make_env_ariadne(args.max_env_steps)
                obs, _ = env.reset()
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
            ep_rew = 0.0
            env = make_env_ariadne(args.max_env_steps)
            obs, _ = env.reset()
        else:
            obs = next_obs

        t_u = time.time()
        if len(buffer) >= args.batch_size and step % 2 == 0:
            s_f, s_m, s_g, s_map, a, r, ns_f, ns_m, ns_g, ns_map, d = buffer.sample(args.batch_size)

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
                target_q = r + args.gamma * (1 - d) * v_next

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

            for tp, p in zip(list(target1.parameters()) + list(target2.parameters()),
                             list(critic1.parameters()) + list(critic2.parameters())):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        t0_upd += time.time() - t_u

        if step % args.log_every == 0:
            import resource
            mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            print(f"Step {step:6d}: cov={info.get('coverage',0):.3f} "
                  f"reward={reward:+.3f} buffer={len(buffer)} alpha={alpha:.4f} "
                  f"skip={n_skip} eps={n_ep} mem={mem_gb:.1f}GB "
                  f"time[act={t0_act:.1f}s env={t0_env:.1f}s upd={t0_upd:.1f}s]")
            sys.stdout.flush()
            t0_act = t0_env = t0_upd = 0.0
            csv_w.writerow([step, reward, info.get("coverage", 0), alpha])
            csv_f.flush()
            tb.add_scalar("train/coverage", info.get("coverage", 0), step)
            tb.add_scalar("train/reward", reward, step)

        if step % 5000 == 0 and step > 0:
            actor.eval()
            maps_dir = os.path.join(os.path.dirname(__file__), "..", "..", "maps", "ariadne")
            eval_results = []
            for mp_name in EVAL_MAPS:
                mp = os.path.join(maps_dir, mp_name)
                eenv = SequenceEnv(max_steps=500, map_size=(20, 20, 2),
                                   map_type='ariadne', map_path=mp)
                eobs, _ = eenv.reset(seed=42)
                for es in range(500):
                    nv = int(eobs["mask"].sum())
                    if nv == 0:
                        eval_results.append((mp_name, eenv._get_exploration_progress(), eenv.total_distance))
                        break
                    ef = torch.FloatTensor(eobs["frontiers"]).unsqueeze(0).to(DEVICE)
                    em = torch.FloatTensor(eobs["mask"]).unsqueeze(0).to(DEVICE)
                    eg = torch.FloatTensor(eobs["global"]).unsqueeze(0).to(DEVICE)
                    emap = torch.FloatTensor(eobs["map_img"]).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        el, _ = actor(ef, em, eg, emap)
                    ea = int(el[0][:nv].argmax().item())
                    eobs, er, ed, et, einfo = eenv.step(ea)
                    if ed:
                        eval_results.append((mp_name, einfo.get("coverage", 0), eenv.total_distance))
                        break
                else:
                    eval_results.append((mp_name, eenv._get_exploration_progress(), eenv.total_distance))

            actor.train()
            env = make_env_ariadne(args.max_env_steps)
            obs, _ = env.reset()

            valid_covs = [c for n, c, d in eval_results if n not in ("2.png", "3.png")]
            avg_valid = np.mean(valid_covs)
            avg_all = np.mean([c for _, c, _ in eval_results])
            per_map = " ".join([f"{n}={c:.0%}/{d:.0f}m" for n, c, d in eval_results])
            print(f"  [EVAL step={step}] avg_valid={avg_valid:.3f} avg_all={avg_all:.3f} | {per_map}")
            tb.add_scalar("eval_avg_valid", avg_valid, step)
            tb.add_scalar("eval_avg_all", avg_all, step)

            if avg_valid > best_eval:
                best_eval = avg_valid
                torch.save(actor.state_dict(), os.path.join(args.save_dir, "best_actor.pth"))
                print(f"  [NEW BEST valid={best_eval:.3f}]")

        if step % 1000 == 0:
            torch.save(actor.state_dict(), os.path.join(args.save_dir, f"actor_{step}.pth"))
            torch.save({
                "step": step, "actor": actor.state_dict(),
                "critic1": critic1.state_dict(), "critic2": critic2.state_dict(),
            }, os.path.join(args.save_dir, "latest_checkpoint.pth"))

    csv_f.close()
    torch.save(actor.state_dict(), os.path.join(args.save_dir, "final_actor.pth"))
    print(f"Done. Best valid avg: {best_eval:.3f}")


def main():
    parser = argparse.ArgumentParser(description="SAC Seq11: skip unreachable maps")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/sac_seq11")
    parser.add_argument("--pretrained", type=str,
                        default="./fuel_rl_checkpoints/sac_seq8/latest_checkpoint.pth")
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-env-steps", type=int, default=800)
    args = parser.parse_args()
    train_sac_seq11(args)


if __name__ == "__main__":
    main()
