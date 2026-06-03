"""Evaluate trained frontier ordering policies."""
import numpy as np
import torch
import sys, os, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.eval.tsp_baseline import tsp_policy, tsp_orienteering_policy, tsp_fuel_policy


def evaluate(policy_fn, env, n_episodes=20, max_steps=400):
    covs, rews, steps_list, dists = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_rew = 0.0
        last_info = {"coverage": 0.0, "total_dist": 0.0}
        for step in range(max_steps):
            n_valid = int(obs["mask"].sum())
            if n_valid == 0:
                break
            action = policy_fn(obs, n_valid)
            obs, rew, done, trunc, info = env.step(action)
            total_rew += rew
            last_info = info
            if done:
                break
        covs.append(last_info["coverage"])
        rews.append(total_rew)
        steps_list.append(step + 1 if n_valid > 0 else step)
        dists.append(last_info["total_dist"])
    return {
        "cov": np.mean(covs),
        "cov_std": np.std(covs),
        "rew": np.mean(rews),
        "steps": np.mean(steps_list),
        "dist": np.mean(dists),
    }


def random_policy(obs, n_valid):
    return np.random.randint(n_valid)


def greedy_closest(obs, n_valid):
    dists = obs["frontiers"][:n_valid, 4]
    return int(np.argmin(dists))


def greedy_biggest(obs, n_valid):
    sizes = obs["frontiers"][:n_valid, 3]
    return int(np.argmax(sizes))


def greedy_visib(obs, n_valid):
    visib = obs["frontiers"][:n_valid, 5]
    return int(np.argmax(visib))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-pillars", type=int, default=15)
    args = parser.parse_args()

    env = SequenceEnv(max_steps=350, num_pillars=args.num_pillars)

    baselines = {
        "random": random_policy,
        "closest": greedy_closest,
        "biggest": greedy_biggest,
        "most_visible": greedy_visib,
        "tsp_nn2opt": tsp_policy,
        "tsp_orient": tsp_orienteering_policy,
        "tsp_fuel": tsp_fuel_policy,
    }

    if args.checkpoint:
        baselines["SAC_trained"] = make_actor_policy(args.checkpoint)

    print(f"Evaluating {len(baselines)} policies x {args.episodes} episodes")
    print("-" * 75)
    print(f"{'Policy':15s} {'Cov':>8s} {'±':>4s} {'Reward':>8s} {'Steps':>7s} {'Dist':>8s}")
    print("-" * 75)

    for name, policy_fn in baselines.items():
        t0 = time.time()
        result = evaluate(policy_fn, env, args.episodes)
        elapsed = time.time() - t0
        print(f"{name:15s} {result['cov']:8.3f} {result['cov_std']:4.3f} "
              f"{result['rew']:8.1f} {result['steps']:7.1f} {result['dist']:8.1f}  "
              f"({elapsed:.0f}s)")
    print("-" * 75)


if __name__ == "__main__":
    main()
