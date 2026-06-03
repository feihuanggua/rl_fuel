"""生成可视化对比图: SAC vs Closest 前沿排序 (2D 俯视图)."""
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


def run_exploration(core, agent_pos, agent_yaw, max_steps, model=None):
    path = [agent_pos.copy()]
    visited = set()
    snapshots = []
    consecutive_fails = 0

    def pos_key(avg):
        return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))

    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    snapshots.append({"step": -1, "progress": progress, "pos": agent_pos.copy(),
                       "frontiers": frontiers, "target": None})

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            snapshots.append({"step": step, "progress": progress, "pos": agent_pos.copy(),
                              "frontiers": frontiers, "target": None})
            break

        if model and len(frontiers) > 0:
            feats = np.zeros((50, 6), dtype=np.float32)
            mask = np.zeros(50, dtype=np.float32)
            vis_idx = 0
            visible_indices = []
            for i, f in enumerate(frontiers):
                if i >= 50: break
                c = np.array(f.average)
                key = pos_key(c)
                if key in visited: continue
                if vis_idx >= 50: break
                mask[vis_idx] = 1.0
                feats[vis_idx, 0:3] = c / 10.0
                feats[vis_idx, 3] = min(f.frontier_size / 2000.0, 1.0)
                feats[vis_idx, 4] = min(np.linalg.norm(c - agent_pos) / 15.0, 1.0)
                feats[vis_idx, 5] = min(f.best_viewpoint_visib_num / 100.0, 1.0)
                visible_indices.append(i)
                vis_idx += 1
            global_f = np.array([progress * 2 - 1, step / max_steps * 2 - 1], dtype=np.float32)
            ft = torch.FloatTensor(feats).unsqueeze(0).to(DEVICE)
            mt = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)
            gt = torch.FloatTensor(global_f).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(ft, mt, gt)
            n = vis_idx
            if n > 0:
                act = logits.squeeze(0)[:n].argmax().item()
                target = frontiers[visible_indices[act]]
            else:
                target = min(frontiers, key=lambda f: np.linalg.norm(np.array(f.average) - agent_pos))
        else:
            target = min(frontiers, key=lambda f: np.linalg.norm(np.array(f.average) - agent_pos))

        vp = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw
        vp[2] = np.clip(vp[2], 0.5, 2.6)

        occ = core.get_occupancy(vp)
        if occ != 1:
            consecutive_fails += 1
            visited.add(pos_key(target.average))
            if consecutive_fails > 5:
                break
            continue

        consecutive_fails = 0
        core.simulate_observation(vp, vp_yaw)
        agent_pos = vp.copy()
        agent_yaw = vp_yaw
        path.append(agent_pos.copy())

        snapshots.append({"step": step, "progress": progress, "pos": agent_pos.copy(),
                           "frontiers": frontiers,
                           "target": np.array(target.average) if target else None})

    final_cov = core.get_exploration_progress()
    return path, snapshots, final_cov


def get_map_2d(core, z=1.5):
    slice_2d = np.array(core.get_occupancy_slice_2d(z)).reshape(200, 200)
    return slice_2d


def plot_comparison(seed, max_steps, model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Setup two cores with same map
    def make_core():
        core = FuelEnvCore()
        mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                                box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8))
        core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
        pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
        core.load_map_from_points(pts)
        core.reset_map()
        return core, pts

    # Load model
    model = None
    if model_path:
        model = OrderPolicy().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()

    # Run both
    results = {}
    for label, use_model in [("Closest", False), ("SAC", True)]:
        core, pts = make_core()
        agent_pos = np.array([0.0, 0.0, 1.5])
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            core.simulate_observation(agent_pos, yaw)
        m = model if use_model else None
        path, snaps, final_cov = run_exploration(core, agent_pos, 0.0, max_steps, model=m)
        total_dist = sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))
        results[label] = {"path": path, "snaps": snaps, "cov": final_cov, "dist": total_dist}
        print(f"{label}: {len(path)-1} steps, cov={final_cov:.1%}, dist={total_dist:.0f}m")

    # Plot
    cmap = mcolors.ListedColormap(["#f5f5f5", "#a8d8ea", "#ff6b6b", "#34495e"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors_line = {"Closest": "#2ecc71", "SAC": "#9b59b6"}

    for ax, (label, res) in zip(axes, results.items()):
        core_plot, _ = make_core()
        agent_pos_init = np.array([0.0, 0.0, 1.5])
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            core_plot.simulate_observation(agent_pos_init, yaw)

        # Replay path to get final map state
        path = res["path"]
        for i in range(1, len(path)):
            pass  # map already explored during run

        # Use initial map for background (empty)
        slice_bg = np.array(core_plot.get_occupancy_slice_2d(1.5)).reshape(200, 200)
        # Actually use the explored map from the run core
        core_vis, _ = make_core()
        agent_vis = np.array([0.0, 0.0, 1.5])
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            core_vis.simulate_observation(agent_vis, yaw)

        m2 = model if label == "SAC" else None
        path2, _, _ = run_exploration(core_vis, agent_vis, 0.0, max_steps, model=m2)
        slice_2d = np.array(core_vis.get_occupancy_slice_2d(1.5)).reshape(200, 200)

        ax.imshow(slice_2d.T, origin="lower", cmap=cmap, norm=norm,
                  extent=[-10, 10, -10, 10], alpha=0.7)

        # Path
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color=colors_line[label], linewidth=2.0, alpha=0.8, zorder=4)
        sc = ax.scatter(px, py, c=range(len(px)), cmap="viridis", s=25,
                        zorder=5, edgecolors="white", linewidths=0.3)

        # Start / End
        ax.plot(px[0], py[0], "g*", markersize=18, zorder=6, label="Start")
        ax.plot(px[-1], py[-1], "r^", markersize=15, zorder=6, label="End")

        # Selected targets
        for snap in res["snaps"][:20]:
            if snap["target"] is not None:
                t = snap["target"]
                ax.scatter(t[0], t[1], c="orange", marker="x", s=40, zorder=5,
                           alpha=0.5, linewidths=1.5)

        ax.set_title(f"{label}\ncov={res['cov']:.1%}, dist={res['dist']:.0f}m, {len(path)-1} steps",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.legend(fontsize=10, loc="upper right")

    plt.suptitle(f"Frontier Ordering Comparison (seed={seed})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"compare_seed{seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Coverage curve
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for label, res in results.items():
        steps = [s["step"] for s in res["snaps"] if s["step"] >= 0]
        covs = [s["progress"] for s in res["snaps"] if s["step"] >= 0]
        ax1.plot(steps, covs, linewidth=2, label=label, color=colors_line[label])

        path_arr = np.array(res["path"])
        cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))])
        ax2.plot(cum_dist, covs + [res["cov"]], linewidth=2, label=label, color=colors_line[label])

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

    plt.tight_layout()
    out2 = os.path.join(out_dir, f"curves_seed{seed}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./fuel_rl_checkpoints/sac_seq/final_actor.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out", type=str, default="./fuel_rl_checkpoints/sac_seq/vis")
    args = parser.parse_args()
    plot_comparison(args.seed, args.max_steps, args.model, args.out)
