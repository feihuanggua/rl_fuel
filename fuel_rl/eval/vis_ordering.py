"""Visualize frontier ordering: trajectory + coverage curve + map snapshot."""
import numpy as np
import torch
import sys, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy


def run_episode(policy_fn, env, max_steps=120):
    obs, _ = env.reset()
    trajectory = [env.agent_pos.copy()]
    coverages = [env.core.get_exploration_progress()]
    rewards_list = [0.0]
    cum_rewards = [0.0]
    selected_frontiers = []
    all_frontiers_per_step = []
    total_rew = 0.0

    for step in range(max_steps):
        n_valid = int(obs["mask"].sum())
        if n_valid == 0:
            coverages.append(env.core.get_exploration_progress())
            cum_rewards.append(total_rew)
            break

        frontier_positions = obs["frontiers"][:n_valid, :3].copy() * 10.0
        all_frontiers_per_step.append(frontier_positions)

        action = policy_fn(obs, n_valid)
        sel_pos = obs["frontiers"][action, :3].copy() * 10.0
        selected_frontiers.append(sel_pos)

        obs, rew, done, trunc, info = env.step(action)
        total_rew += rew
        trajectory.append(env.agent_pos.copy())
        coverages.append(info["coverage"])
        rewards_list.append(rew)
        cum_rewards.append(total_rew)

        if done:
            break

    return {
        "trajectory": np.array(trajectory),
        "coverages": np.array(coverages),
        "cum_rewards": np.array(cum_rewards),
        "rewards": np.array(rewards_list),
        "selected": np.array(selected_frontiers) if selected_frontiers else np.zeros((0, 3)),
        "frontiers_per_step": all_frontiers_per_step,
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


def get_map_slice(env):
    slice_2d = np.array(env.core.get_occupancy_slice_2d(1.5)).reshape(200, 200)
    return slice_2d


def plot_map(ax, slice_2d, title=""):
    cmap = mcolors.ListedColormap(["#f0f0f0", "#a8d8ea", "#ff6b6b", "#2c3e50"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(slice_2d.T, origin="lower", cmap=cmap, norm=norm,
              extent=[-10, 10, -10, 10], alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")


def main():
    env = SequenceEnv(max_steps=120, num_pillars=15)

    policies = {
        "Random": random_policy,
        "Closest": greedy_closest,
        "Biggest": greedy_biggest,
        "SAC": make_actor_policy("./fuel_rl_checkpoints/sac_seq/actor_1000.pth"),
    }

    colors = {
        "Random": "#e74c3c",
        "Closest": "#2ecc71",
        "Biggest": "#3498db",
        "SAC": "#9b59b6",
    }

    np.random.seed(42)

    results = {}
    final_maps = {}

    for name, fn in policies.items():
        print(f"Running {name}...", flush=True)
        t0 = time.time()

        obs, _ = env.reset(seed=42)
        res = run_episode(fn, env, 120)
        results[name] = res
        final_maps[name] = get_map_slice(env)
        print(f"  {name}: {len(res['trajectory'])-1} steps, "
              f"cov={res['coverages'][-1]:.3f}, rew={res['cum_rewards'][-1]:.1f}, "
              f"dist={sum(np.linalg.norm(np.diff(res['trajectory'], axis=0), axis=1)):.0f}m "
              f"({time.time()-t0:.0f}s)")

    # ============ Figure 1: Trajectory on final map ============
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14))
    axes1 = axes1.flatten()

    for ax, (name, res) in zip(axes1, results.items()):
        plot_map(ax, final_maps[name], f"{name} — cov={res['coverages'][-1]:.3f}, "
                 f"dist={sum(np.linalg.norm(np.diff(res['trajectory'], axis=0), axis=1)):.0f}m")
        traj = res["trajectory"]
        n_pts = len(traj)

        sc = ax.scatter(traj[:, 0], traj[:, 1], c=range(n_pts), cmap="viridis",
                        s=20, zorder=5, edgecolors="white", linewidths=0.3)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[name], alpha=0.5, linewidth=1.2, zorder=4)

        ax.plot(traj[0, 0], traj[0, 1], "g*", markersize=15, zorder=6, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], "r^", markersize=12, zorder=6, label="End")

        if len(res["selected"]) > 0:
            sel = res["selected"]
            ax.scatter(sel[:, 0], sel[:, 1], c="orange", marker="x",
                       s=30, zorder=5, alpha=0.6, linewidths=1.5, label="Selected frontier")

        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Frontier Ordering — Exploration Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1.savefig("./fuel_rl_checkpoints/sac_seq/vis_trajectories.png", dpi=150)
    print("Saved vis_trajectories.png")

    # ============ Figure 2: Coverage & Reward curves ============
    fig2, (ax_cov, ax_rew) = plt.subplots(1, 2, figsize=(16, 5))

    for name, res in results.items():
        steps = np.arange(len(res["coverages"]))
        ax_cov.plot(steps, res["coverages"], color=colors[name], linewidth=2, label=name)
        ax_rew.plot(steps, res["cum_rewards"], color=colors[name], linewidth=2, label=name)

    ax_cov.set_xlabel("Step", fontsize=11)
    ax_cov.set_ylabel("Coverage", fontsize=11)
    ax_cov.set_title("Coverage over Steps", fontweight="bold")
    ax_cov.legend(fontsize=10)
    ax_cov.grid(alpha=0.3)
    ax_cov.set_ylim(0, 0.7)

    ax_rew.set_xlabel("Step", fontsize=11)
    ax_rew.set_ylabel("Cumulative Reward", fontsize=11)
    ax_rew.set_title("Cumulative Reward over Steps", fontweight="bold")
    ax_rew.legend(fontsize=10)
    ax_rew.grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig("./fuel_rl_checkpoints/sac_seq/vis_curves.png", dpi=150)
    print("Saved vis_curves.png")

    # ============ Figure 3: Step-by-step frontier selection ============
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 14))
    axes3 = axes3.flatten()

    snap_steps = [0, 5, 15, 40]

    for ax_idx, (name, res) in enumerate(results.items()):
        ax = axes3[ax_idx]
        traj = res["trajectory"]
        n_traj = len(traj)

        for snap in snap_steps:
            if snap >= len(res["frontiers_per_step"]):
                continue
            frontiers = res["frontiers_per_step"][snap]
            alpha = max(0.2, 1.0 - snap / 60.0)
            ax.scatter(frontiers[:, 0], frontiers[:, 1], c="gray", marker=".",
                       s=10, alpha=alpha, zorder=3)
            if snap < len(res["selected"]):
                sel = res["selected"][snap]
                ax.scatter(sel[0], sel[1], c=colors[name], marker="*",
                           s=100, zorder=5, edgecolors="black", linewidths=0.5)

        ax.plot(traj[:, 0], traj[:, 1], color=colors[name], alpha=0.4, linewidth=0.8)
        ax.plot(traj[0, 0], traj[0, 1], "g*", markersize=15, zorder=6)
        ax.plot(traj[-1, 0], traj[-1, 1], "r^", markersize=12, zorder=6)
        ax.set_title(f"{name} — frontier selection", fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(alpha=0.2)

    plt.suptitle("Frontier Selection at Steps 0/5/15/40", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig3.savefig("./fuel_rl_checkpoints/sac_seq/vis_selection.png", dpi=150)
    print("Saved vis_selection.png")

    # ============ Figure 4: Per-step reward ============
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
    axes4 = axes4.flatten()

    for ax, (name, res) in zip(axes4, results.items()):
        rews = res["rewards"][1:]
        colors_bar = ["#2ecc71" if r > 0 else "#e74c3c" for r in rews]
        ax.bar(range(len(rews)), rews, color=colors_bar, width=1.0, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"{name} — per-step reward", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig4.savefig("./fuel_rl_checkpoints/sac_seq/vis_step_reward.png", dpi=150)
    print("Saved vis_step_reward.png")

    print("\nDone. All figures saved to ./fuel_rl_checkpoints/sac_seq/")


if __name__ == "__main__":
    main()
