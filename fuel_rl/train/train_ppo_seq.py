"""PPO for discrete frontier ordering.

On-policy: collect a batch of trajectories, then do multiple epochs of updates.
No replay buffer needed.
"""
import argparse, os, sys, time, csv, resource
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.config import DEVICE


def collect_batch(env, actor, n_episodes, max_ep_steps):
    """Collect complete episodes."""
    all_obs, all_act, all_logp, all_rew, all_val, all_done = [], [], [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_act, ep_logp, ep_rew, ep_val, ep_done = [], [], [], [], [], []

        for step in range(max_ep_steps):
            f_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
            m_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
            g_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
            map_t = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, log_prob, value = actor.act(f_t, m_t, g_t, map_t)

            n_valid = int(m_t.sum().item())
            if n_valid == 0:
                ep_done.append(1.0)
                ep_rew.append(0.0)
                ep_val.append(0.0)
                ep_act.append(0)
                ep_logp.append(0.0)
                ep_obs.append({
                    "frontiers": obs["frontiers"].copy(),
                    "mask": obs["mask"].copy(),
                    "global": obs["global"].copy(),
                    "map_img": (obs["map_img"] * 255).astype(np.uint8),
                })
                break

            a_int = min(action.item(), n_valid - 1)
            next_obs, reward, done, truncated, info = env.step(a_int)

            ep_obs.append({
                "frontiers": obs["frontiers"].copy(),
                "mask": obs["mask"].copy(),
                "global": obs["global"].copy(),
                "map_img": (obs["map_img"] * 255).astype(np.uint8),
            })
            ep_act.append(a_int)
            ep_logp.append(log_prob.item())
            ep_rew.append(reward)
            ep_val.append(value.item())
            ep_done.append(1.0 if done else 0.0)

            if done:
                break
            obs = next_obs

        all_obs.extend(ep_obs)
        all_act.extend(ep_act)
        all_logp.extend(ep_logp)
        all_rew.extend(ep_rew)
        all_val.extend(ep_val)
        all_done.extend(ep_done)

    n = len(all_act)
    if n == 0:
        return [], [], [], np.zeros(0), np.zeros(0)

    # Per-episode GAE
    returns = np.zeros(n, dtype=np.float32)
    advantages = np.zeros(n, dtype=np.float32)
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    next_val = 0.0

    for t in reversed(range(n)):
        if all_done[t] > 0.5:
            gae = 0.0
            next_val = 0.0
        delta = all_rew[t] + gamma * next_val - all_val[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + all_val[t]
        next_val = all_val[t]

    return all_obs, all_act, all_logp, returns, advantages, all_done


def ppo_update(actor, optimizer, obs_list, act_list, old_logp,
               returns, advantages, ppo_epochs, mini_batch_size, clip_eps, ent_coef):
    n = len(act_list)
    adv_t = torch.FloatTensor(advantages).to(DEVICE)
    ret_t = torch.FloatTensor(returns).to(DEVICE)
    old_logp_t = torch.FloatTensor(old_logp).to(DEVICE)
    act_t = torch.LongTensor(act_list).to(DEVICE)

    # normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    all_f = torch.FloatTensor(np.stack([o["frontiers"] for o in obs_list])).to(DEVICE)
    all_m = torch.FloatTensor(np.stack([o["mask"] for o in obs_list])).to(DEVICE)
    all_g = torch.FloatTensor(np.stack([o["global"] for o in obs_list])).to(DEVICE)
    all_map = torch.FloatTensor(np.stack([o["map_img"].astype(np.float32) / 255.0 for o in obs_list])).to(DEVICE)

    total_loss_val = 0.0
    n_updates = 0

    for _ in range(ppo_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            if end > n:
                continue
            idx = indices[start:end]

            log_prob, entropy, value = actor.evaluate(
                all_f[idx], all_m[idx], all_g[idx], all_map[idx], act_t[idx])

            ratio = torch.exp(log_prob - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, ret_t[idx])

            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()

            total_loss_val += loss.item()
            n_updates += 1

    return total_loss_val / max(n_updates, 1)


def main():
    parser = argparse.ArgumentParser(description="PPO Frontier Ordering")
    parser.add_argument("--save-dir", type=str, default="./fuel_rl_checkpoints/ppo_seq")
    parser.add_argument("--total-iter", type=int, default=500)
    parser.add_argument("--episodes-per-batch", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--max-ep-steps", type=int, default=100)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tb = SummaryWriter(os.path.join(args.save_dir, "tb"))

    env = SequenceEnv(num_pillars=args.num_pillars, max_steps=args.max_ep_steps)
    actor = OrderPolicy(d_frontier=6).to(DEVICE)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)

    start_iter = 0
    latest = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if os.path.exists(latest):
        ckpt = torch.load(latest, map_location=DEVICE)
        actor.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0) + 1
        print(f"Resumed from iter {start_iter}")

    csv_path = os.path.join(args.save_dir, "ppo_log.csv")
    csv_f = open(csv_path, "a" if start_iter > 0 else "w", newline="")
    csv_w = csv.writer(csv_f)
    if start_iter == 0:
        csv_w.writerow(["iter", "avg_rew", "avg_cov", "loss", "n_steps"])
    csv_f.flush()

    best_cov = -float("inf")

    print(f"PPO: {args.total_iter} iters x {args.episodes_per_batch} eps/iter", flush=True)

    for it in range(start_iter, args.total_iter):
        t0 = time.time()

        t_collect = time.time()
        obs_list, act_list, logp_list, returns, advantages, all_done = collect_batch(
            env, actor, args.episodes_per_batch, args.max_ep_steps)
        t_collect = time.time() - t_collect

        n_collected = len(act_list)
        if n_collected < 10:
            continue

        avg_rew = float(np.mean(returns[:n_collected]))

        # coverage: average final coverage across episodes
        ep_coverages = []
        ep_cov = 0.0
        for i in range(n_collected):
            ep_cov = max(ep_cov, float(obs_list[i]["global"][0]))
            if all_done[i] > 0.5:
                ep_coverages.append(ep_cov)
                ep_cov = 0.0
        avg_cov = float(np.mean(ep_coverages)) if ep_coverages else 0.0
        cov = avg_cov

        t_update = time.time()
        loss = ppo_update(actor, optimizer, obs_list, act_list, logp_list,
                          returns, advantages, args.ppo_epochs, args.mini_batch,
                          args.clip_eps, args.ent_coef)
        t_update = time.time() - t_update

        elapsed = time.time() - t0
        mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        print(f"Iter {it:4d}: n={n_collected} cov={cov:.3f} "
              f"avg_rew={avg_rew:.3f} loss={loss:.4f} "
              f"mem={mem_gb:.1f}GB "
              f"time[{t_collect:.0f}s+{t_update:.1f}s={elapsed:.0f}s]",
              flush=True)

        csv_w.writerow([it, f"{avg_rew:.4f}", f"{cov:.4f}", f"{loss:.4f}", n_collected])
        csv_f.flush()

        tb.add_scalar("avg_reward", avg_rew, it)
        tb.add_scalar("coverage", cov, it)
        tb.add_scalar("loss", loss, it)

        if cov > best_cov:
            best_cov = cov

        if it % 10 == 0:
            torch.save(actor.state_dict(), os.path.join(args.save_dir, f"actor_{it}.pth"))
            torch.save({
                "iter": it, "model": actor.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, latest)
            print(f"  [saved iter_{it}]", flush=True)

    torch.save(actor.state_dict(), os.path.join(args.save_dir, "final_actor.pth"))
    print(f"Done. Best cov: {best_cov:.3f}")
    csv_f.close()
    tb.close()


if __name__ == "__main__":
    main()
