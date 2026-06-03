"""生成可视化对比图: SAC vs Closest vs TSP 前沿排序 (2D 俯视图)."""
import sys, os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl import FuelEnvCore
from fuel_rl.config import (default_map_params, default_frontier_params,
                            fast_perception_params, default_astar_params, DEVICE)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.eval.tsp_baseline import tsp_policy


MAX_FRONTIERS = 50


def _pos_key(avg):
    return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))


def _select_frontier(frontiers, agent_pos, progress, step, max_steps,
                     policy_type, model=None):
    if not frontiers:
        return None

    feats = np.zeros((MAX_FRONTIERS, 6), dtype=np.float32)
    mask = np.zeros(MAX_FRONTIERS, dtype=np.float32)
    vis_idx = 0
    visible_indices = []
    for i, f in enumerate(frontiers):
        if i >= MAX_FRONTIERS:
            break
        c = np.array(f.average)
        vp = np.array(f.best_viewpoint_pos)
        vp[2] = np.clip(vp[2], 0.5, 2.6)
        feats[vis_idx, 0:3] = c / 10.0
        feats[vis_idx, 3] = min(f.frontier_size / 2000.0, 1.0)
        feats[vis_idx, 4] = min(np.linalg.norm(vp - agent_pos) / 15.0, 1.0)
        feats[vis_idx, 5] = min(f.best_viewpoint_visib_num / 100.0, 1.0)
        mask[vis_idx] = 1.0
        visible_indices.append(i)
        vis_idx += 1

    n_valid = vis_idx
    if n_valid == 0:
        return None

    obs = {"frontiers": feats, "mask": mask,
           "global": np.array([progress * 2 - 1, step / max_steps * 2 - 1], dtype=np.float32)}

    if policy_type == "sac":
        ft = torch.FloatTensor(feats).unsqueeze(0).to(DEVICE)
        mt = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)
        gt = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(ft, mt, gt)
        act = logits.squeeze(0)[:n_valid].argmax().item()
    elif policy_type == "tsp":
        act = tsp_policy(obs, n_valid)
    else:
        act = int(np.argmin(feats[:n_valid, 4]))

    return frontiers[visible_indices[act]]


def run_exploration(core, agent_pos, max_steps, policy_type, model=None):
    path = [agent_pos.copy()]
    visited = set()
    snapshots = []
    consecutive_fails = 0
    cum_dist = 0.0

    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    snapshots.append({"step": -1, "progress": progress, "pos": agent_pos.copy(),
                       "frontiers": frontiers, "target": None, "cum_dist": 0.0})

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            snapshots.append({"step": step, "progress": progress, "pos": agent_pos.copy(),
                               "frontiers": [], "target": None, "cum_dist": cum_dist})
            break

        target = _select_frontier(frontiers, agent_pos, progress, step, max_steps,
                                  policy_type, model)
        if target is None:
            break

        vp = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw
        vp[2] = np.clip(vp[2], 0.5, 2.6)

        occ = core.get_occupancy(vp)
        if occ != 1:
            consecutive_fails += 1
            visited.add(_pos_key(target.average))
            if consecutive_fails > 5:
                break
            continue

        consecutive_fails = 0
        dist_step = np.linalg.norm(vp - agent_pos)
        cum_dist += dist_step
        core.simulate_observation(vp, vp_yaw)
        agent_pos = vp.copy()
        path.append(agent_pos.copy())

        snapshots.append({"step": step, "progress": progress, "pos": agent_pos.copy(),
                           "frontiers": frontiers,
                           "target": np.array(target.average), "cum_dist": cum_dist})

    final_cov = core.get_exploration_progress()
    total_dist = sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))
    return path, snapshots, final_cov, total_dist


def make_core(seed):
    core = FuelEnvCore()
    mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                            box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8))
    core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()
    return core


def plot_comparison(seeds, max_steps, model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model = None
    if model_path and os.path.exists(model_path):
        model = OrderPolicy().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()

    policies = [("Closest", "closest"), ("TSP", "tsp"), ("SAC", "sac")]
    if model is None:
        policies = [("Closest", "closest"), ("TSP", "tsp")]
    colors = {"Closest": "#2ecc71", "TSP": "#e67e22", "SAC": "#9b59b6"}

    cmap_occ = mcolors.ListedColormap(["#f5f5f5", "#a8d8ea", "#ff6b6b", "#34495e"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_occ = mcolors.BoundaryNorm(bounds, cmap_occ.N)

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        all_results = {}

        for label, ptype in policies:
            core = make_core(seed)
            agent_pos = np.array([0.0, 0.0, 1.5])
            for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
                core.simulate_observation(agent_pos, yaw)
            path, snaps, cov, dist = run_exploration(core, agent_pos, max_steps, ptype, model)
            all_results[label] = {"path": path, "snaps": snaps, "cov": cov, "dist": dist}
            print(f"  {label}: cov={cov:.1%}, dist={dist:.0f}m, {len(path) - 1} steps")

        # --- Path plot ---
        n_pol = len(policies)
        fig, axes = plt.subplots(1, n_pol, figsize=(7 * n_pol, 7))
        if n_pol == 1:
            axes = [axes]

        for ax, (label, _) in zip(axes, policies):
            res = all_results[label]
            core = make_core(seed)
            ap = np.array([0.0, 0.0, 1.5])
            for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
                core.simulate_observation(ap, yaw)
            m2 = model if label == "SAC" else None
            ptype = "sac" if label == "SAC" else label.lower()
            _, _, _, _ = run_exploration(core, ap, max_steps, ptype, m2)
            slice_2d = np.array(core.get_occupancy_slice_2d(1.5)).reshape(200, 200)

            ax.imshow(slice_2d.T, origin="lower", cmap=cmap_occ, norm=norm_occ,
                      extent=[-10, 10, -10, 10], alpha=0.7)

            path = res["path"]
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, color=colors[label], linewidth=2.2, alpha=0.85, zorder=4)
            sc = ax.scatter(px, py, c=range(len(px)), cmap="viridis", s=20,
                            zorder=5, edgecolors="white", linewidths=0.3)

            ax.plot(px[0], py[0], "g*", markersize=18, zorder=6, label="Start")
            ax.plot(px[-1], py[-1], "r^", markersize=14, zorder=6, label="End")

            for snap in res["snaps"][:25]:
                if snap["target"] is not None:
                    t = snap["target"]
                    ax.scatter(t[0], t[1], c="orange", marker="x", s=35,
                               zorder=5, alpha=0.4, linewidths=1.5)

            ax.set_title(f"{label}\ncov={res['cov']:.1%}, dist={res['dist']:.0f}m",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal")
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.legend(fontsize=9, loc="upper right")

        plt.suptitle(f"Frontier Ordering (seed={seed})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"paths_seed{seed}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # --- Curves plot ---
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for label, _ in policies:
            res = all_results[label]
            steps = [s["step"] for s in res["snaps"] if s["step"] >= 0]
            covs = [s["progress"] for s in res["snaps"] if s["step"] >= 0]
            ax1.plot(steps, covs, linewidth=2, label=label, color=colors[label])

            cum = [s["cum_dist"] for s in res["snaps"] if s["step"] >= 0]
            ax2.plot(cum, covs, linewidth=2, label=label, color=colors[label])

        ax1.set_xlabel("Step")
        ax1.set_ylabel("Coverage")
        ax1.set_title("Coverage vs Steps", fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.set_xlabel("Cumulative Distance (m)")
        ax2.set_ylabel("Coverage")
        ax2.set_title("Coverage vs Distance", fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle(f"Exploration Curves (seed={seed})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"curves_seed{seed}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved paths_seed{seed}.png + curves_seed{seed}.png")

    # --- Summary bar chart ---
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    labels_list = [l for l, _ in policies]
    dist_means, cov_means = [], []
    dist_stds, cov_stds = [], []

    for label, _ in policies:
        dists_s, covs_s = [], []
        for seed in seeds:
            core = make_core(seed)
            ap = np.array([0.0, 0.0, 1.5])
            for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
                core.simulate_observation(ap, yaw)
            ptype = "sac" if label == "SAC" else label.lower()
            m2 = model if label == "SAC" else None
            _, _, cov, dist = run_exploration(core, ap, max_steps, ptype, m2)
            dists_s.append(dist)
            covs_s.append(cov)
        dist_means.append(np.mean(dists_s))
        dist_stds.append(np.std(dists_s))
        cov_means.append(np.mean(covs_s))
        cov_stds.append(np.std(covs_s))

    x = np.arange(len(labels_list))
    bar_colors = [colors[l] for l in labels_list]

    ax3.bar(x, dist_means, yerr=dist_stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_list)
    ax3.set_ylabel("Total Distance (m)")
    ax3.set_title("Distance (lower is better)", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    ax4.bar(x, cov_means, yerr=cov_stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor="black")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_list)
    ax4.set_ylabel("Coverage")
    ax4.set_title("Coverage (higher is better)", fontweight="bold")
    ax4.set_ylim(0.5, 0.7)
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Multi-seed Comparison ({len(seeds)} seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(os.path.join(out_dir, "summary_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary_bars.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./fuel_rl_checkpoints/sac_seq/final_actor.pth")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out", type=str, default="./fuel_rl_checkpoints/sac_seq/vis")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    plot_comparison(seeds, args.max_steps, args.model, args.out)
