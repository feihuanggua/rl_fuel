"""Run full evaluation across checkpoints and baselines, save results for plotting."""
import numpy as np
import torch
import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy


def evaluate(policy_fn, env, n_episodes=20, max_steps=120):
    covs, rews, steps_list, dists = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_rew = 0.0
        last_info = {"coverage": 0.0, "total_dist": 0.0}
        for step in range(max_steps):
            n_valid = int(obs["mask"].sum())
            if n_valid == 0:
                last_info = {"coverage": env.core.get_exploration_progress(),
                             "total_dist": env.total_distance}
                break
            action = policy_fn(obs, n_valid)
            obs, rew, done, trunc, info = env.step(action)
            total_rew += rew
            last_info = info
            if done:
                break
        covs.append(last_info["coverage"])
        rews.append(total_rew)
        steps_list.append(step + 1)
        dists.append(last_info["total_dist"])
    return {
        "cov_mean": float(np.mean(covs)),
        "cov_std": float(np.std(covs)),
        "rew_mean": float(np.mean(rews)),
        "rew_std": float(np.std(rews)),
        "steps_mean": float(np.mean(steps_list)),
        "dist_mean": float(np.mean(dists)),
        "dist_std": float(np.std(dists)),
        "per_ep_cov": [float(c) for c in covs],
        "per_ep_rew": [float(r) for r in rews],
        "per_ep_dist": [float(d) for d in dists],
    }


def random_policy(obs, n_valid):
    return np.random.randint(n_valid)

def greedy_closest(obs, n_valid):
    return int(np.argmin(obs["frontiers"][:n_valid, 4]))

def greedy_biggest(obs, n_valid):
    return int(np.argmax(obs["frontiers"][:n_valid, 3]))

def greedy_visib(obs, n_valid):
    return int(np.argmax(obs["frontiers"][:n_valid, 5]))

def make_actor_policy(ckpt_path):
    actor = OrderPolicy().to("cuda")
    actor.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
    actor.eval()
    def policy_fn(obs, n_valid):
        f_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to("cuda")
        m_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to("cuda")
        g_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to("cuda")
        map_t = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to("cuda")
        with torch.no_grad():
            logits, _ = actor(f_t, m_t, g_t, map_t)
        return int(torch.argmax(logits[0][:n_valid]).item())
    return policy_fn


def main():
    n_ep = 20

    baselines = {
        "Random": random_policy,
        "Closest": greedy_closest,
        "Biggest": greedy_biggest,
        "Most Visible": greedy_visib,
    }

    results = {}
    for name, fn in baselines.items():
        env = SequenceEnv(max_steps=120, num_pillars=15)
        print(f"Evaluating {name}...", flush=True)
        t0 = time.time()
        results[name] = evaluate(fn, env, n_ep)
        print(f"  {name}: cov={results[name]['cov_mean']:.3f} rew={results[name]['rew_mean']:.1f} dist={results[name]['dist_mean']:.0f} ({time.time()-t0:.0f}s)", flush=True)

    ckpt_dir = "./fuel_rl_checkpoints/sac_seq"
    for ckpt_file in sorted(os.listdir(ckpt_dir)):
        if not ckpt_file.startswith("actor_") or not ckpt_file.endswith(".pth"):
            continue
        step = ckpt_file.replace("actor_", "").replace(".pth", "")
        name = f"SAC-{step}"
        path = os.path.join(ckpt_dir, ckpt_file)
        env = SequenceEnv(max_steps=120, num_pillars=15)
        print(f"Evaluating {name}...", flush=True)
        t0 = time.time()
        results[name] = evaluate(make_actor_policy(path), env, n_ep)
        print(f"  {name}: cov={results[name]['cov_mean']:.3f} rew={results[name]['rew_mean']:.1f} dist={results[name]['dist_mean']:.0f} ({time.time()-t0:.0f}s)", flush=True)

    out_path = os.path.join(ckpt_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
