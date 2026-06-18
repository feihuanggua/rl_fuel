"""2D可视化: SAC seq8 on ARiADNE map"""
import sys, os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl.env.sequence_env import SequenceEnv, MAX_FRONTIERS, FEAT_DIM
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.config import DEVICE


def run_with_env(env, max_steps, model=None, seed=0):
    obs, _ = env.reset(seed=seed)
    path = [env.agent_pos.copy()]
    snapshots = []

    for step in range(max_steps):
        n_valid = int(obs["mask"].sum())
        progress = env._get_exploration_progress()
        snapshots.append({"step": step, "progress": progress, "pos": env.agent_pos.copy()})
        if n_valid == 0:
            break
        if model is not None:
            f_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
            m_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
            g_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
            map_t = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(f_t, m_t, g_t, map_t)
            action = int(logits.squeeze(0)[:n_valid].argmax().item())
        else:
            action = int(obs["frontiers"][:n_valid, 4].argmin())
        obs, rew, done, trunc, info = env.step(action)
        path.append(env.agent_pos.copy())
        if done:
            break

    final_cov = env._get_exploration_progress()
    total_dist = info["total_dist"]
    return path, snapshots, final_cov, total_dist, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--map-type", type=str, default="ariadne")
    parser.add_argument("--map-path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = OrderPolicy().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=False))
    model.eval()

    results = {}
    for label, use_model in [("Closest", False), ("SAC", True)]:
        if args.map_type == "ariadne" and args.map_path:
            env = SequenceEnv(max_steps=args.max_steps, map_size=(20, 20, 2),
                              map_type="ariadne", map_path=args.map_path)
        else:
            env = SequenceEnv(max_steps=args.max_steps, num_pillars=15, map_size=(20, 20, 2))
        m = model if use_model else None
        path, snaps, cov, dist, env_final = run_with_env(env, args.max_steps, model=m, seed=args.seed)
        results[label] = {"path": path, "snaps": snaps, "cov": cov, "dist": dist, "env": env_final}
        print(f"{label}: {len(path)-1} steps, cov={cov:.1%}, dist={dist:.0f}m")

    cmap = mcolors.ListedColormap(["#f5f5f5", "#a8d8ea", "#ff6b6b", "#34495e"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors_line = {"Closest": "#2ecc71", "SAC": "#9b59b6"}

    for ax, (label, res) in zip(axes, results.items()):
        env_vis = res["env"]
        slice_2d = np.array(env_vis.core.get_occupancy_slice_2d(1.0)).reshape(200, 200)
        ax.imshow(slice_2d.T, origin="lower", cmap=cmap, norm=norm,
                  extent=[-10, 10, -10, 10], alpha=0.7)

        path = res["path"]
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color=colors_line[label], linewidth=2.0, alpha=0.8, zorder=4)
        ax.scatter(px, py, c=range(len(px)), cmap="viridis", s=25,
                   zorder=5, edgecolors="white", linewidths=0.3)
        ax.plot(px[0], py[0], "g*", markersize=18, zorder=6, label="Start")
        ax.plot(px[-1], py[-1], "r^", markersize=15, zorder=6, label="End")
        ax.set_title(f"{label}\ncov={res['cov']:.1%}, dist={res['dist']:.0f}m, {len(path)-1} steps",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.legend(fontsize=10, loc="upper right")

    plt.suptitle(f"SAC seq8 vs Closest (ARiADNE seed={args.seed})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(args.out, f"compare_seed{args.seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, res in results.items():
        steps = [s["step"] for s in res["snaps"]]
        covs = [s["progress"] for s in res["snaps"]]
        ax1.plot(steps, covs, label=label, linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for label, res in results.items():
        path = res["path"]
        dists = [0] + list(np.cumsum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1)))
        covs = [s["progress"] for s in res["snaps"]]
        min_len = min(len(dists), len(covs))
        ax2.plot(dists[:min_len], covs[:min_len], label=label, linewidth=2)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Coverage vs Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Exploration Curves (seed={args.seed})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(args.out, f"curves_seed{args.seed}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
