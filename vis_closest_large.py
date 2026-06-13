"""Test Closest baseline on new large ARiADNE maps + 2D visualization."""
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/home/jdwsl/rl_fuel')
from fuel_rl.env.sequence_env import SequenceEnv

MAPS_DIR = "/home/jdwsl/rl_fuel/maps/ariadne"


def run_closest(env, max_steps=800):
    obs, _ = env.reset(seed=42)
    path = [env.agent_pos.copy()]
    snapshots = []

    for step in range(max_steps):
        nv = int(obs["mask"].sum())
        progress = env._get_exploration_progress()
        snapshots.append({"step": step, "progress": progress, "pos": env.agent_pos.copy()})
        if nv == 0:
            break
        action = int(obs["frontiers"][:nv, 4].argmin())
        obs, _, done, _, info = env.step(action)
        path.append(env.agent_pos.copy())
        if done:
            break

    final_cov = env._get_exploration_progress()
    total_dist = env.total_distance
    return path, snapshots, final_cov, total_dist, env


def plot_map(map_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(MAPS_DIR, map_name)
    
    print(f"\n=== {map_name} ===")
    t0 = time.time()
    env = SequenceEnv(max_steps=800, map_size=(50, 40, 2),
                       map_type='ariadne', map_path=mp)
    path, snaps, cov, dist, env_final = run_closest(env)
    t1 = time.time()
    print(f"  Closest: {len(path)-1} steps, cov={cov:.1%}, dist={dist:.0f}m, time={t1-t0:.1f}s")

    cmap = mcolors.ListedColormap(["#f5f5f5", "#a8d8ea", "#ff6b6b", "#34495e"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    vn = env_final.core.get_map_voxel_num()
    nx, ny = vn[0], vn[1]
    slice_2d = np.array(env_final.core.get_occupancy_slice_2d(1.0)).reshape(nx, ny)
    
    sx_meters = nx * 0.1
    sy_meters = ny * 0.1
    half_x = sx_meters / 2.0
    half_y = sy_meters / 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left: map with path
    ax1.imshow(slice_2d.T, origin="lower", cmap=cmap, norm=norm,
               extent=[-half_x, half_x, -half_y, half_y], alpha=0.7)
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax1.plot(px, py, color="#2ecc71", linewidth=2.0, alpha=0.8, zorder=4)
    ax1.scatter(px, py, c=range(len(px)), cmap="viridis", s=20,
                zorder=5, edgecolors="white", linewidths=0.3)
    ax1.plot(px[0], py[0], "g*", markersize=18, zorder=6, label="Start")
    ax1.plot(px[-1], py[-1], "r^", markersize=15, zorder=6, label="End")
    ax1.set_title(f"Closest on {map_name}\n{len(path)-1} steps, cov={cov:.1%}, dist={dist:.0f}m",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right")

    # Right: coverage curve
    steps_arr = [s["step"] for s in snaps]
    covs_arr = [s["progress"] for s in snaps]
    dists_arr = [0] + list(np.cumsum(
        np.linalg.norm(np.diff(np.array(path)[:, :2], axis=0), axis=1)))
    min_len = min(len(dists_arr), len(covs_arr))

    ax2.plot(steps_arr, covs_arr, color="#2ecc71", linewidth=2, label="Coverage vs Steps")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Coverage", color="#2ecc71")
    ax2.tick_params(axis='y', labelcolor="#2ecc71")
    ax2.set_title("Coverage Progress")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(dists_arr[:min_len], covs_arr[:min_len], color="#e74c3c", linewidth=2, label="Coverage vs Distance")
    ax2b.set_xlabel("Distance (m)")
    ax2b.set_ylabel("Coverage", color="#e74c3c")
    ax2b.tick_params(axis='y', labelcolor="#e74c3c")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"closest_{map_name.replace('.png', '')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="+", default=["5.png", "50.png", "1.png"])
    parser.add_argument("--out", type=str, default="/home/jdwsl/rl_fuel/vis_large_maps")
    args = parser.parse_args()

    for mp_name in args.maps:
        plot_map(mp_name, args.out)
