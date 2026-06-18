"""Quick vis: Closest with 4m ray, 800 steps, 40m max crop."""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '/home/jdwsl/rl_fuel')
from fuel_rl.env.sequence_env import SequenceEnv

MAPS_DIR = "/home/jdwsl/rl_fuel/maps/ariadne"
OUT_DIR = "/home/jdwsl/rl_fuel/vis_4m_40m"
os.makedirs(OUT_DIR, exist_ok=True)

cmap = mcolors.ListedColormap(["#f5f5f5", "#a8d8ea", "#ff6b6b", "#34495e"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

for name in ["5.png", "50.png"]:
    mp = os.path.join(MAPS_DIR, name)
    env = SequenceEnv(max_steps=800, map_size=(50, 40, 2),
                       map_type='ariadne', map_path=mp)
    obs, _ = env.reset(seed=42)
    path = [env.agent_pos.copy()]
    snaps = []

    t0 = time.time()
    for step in range(800):
        nv = int(obs["mask"].sum())
        cov = env._get_exploration_progress()
        snaps.append({"step": step, "cov": cov})
        if nv == 0:
            break
        action = int(obs["frontiers"][:nv, 4].argmin())
        obs, _, done, _, info = env.step(action)
        path.append(env.agent_pos.copy())
        if done:
            break
    t1 = time.time()
    final_cov = env._get_exploration_progress()
    print(f"{name}: {step+1} steps, cov={final_cov:.1%}, dist={env.total_distance:.0f}m, time={t1-t0:.1f}s")

    vn = env.core.get_map_voxel_num()
    nx, ny = vn[0], vn[1]
    slice_2d = np.array(env.core.get_occupancy_slice_2d(1.0)).reshape(nx, ny)
    sx_m = nx * 0.1
    sy_m = ny * 0.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(slice_2d.T, origin="lower", cmap=cmap, norm=norm,
               extent=[-sx_m/2, sx_m/2, -sy_m/2, sy_m/2], alpha=0.7)
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax1.plot(px, py, color="#2ecc71", linewidth=1.5, alpha=0.7, zorder=4)
    ax1.scatter(px[::10], py[::10], c="blue", s=10, zorder=5)
    ax1.plot(px[0], py[0], "g*", markersize=18, zorder=6, label="Start")
    ax1.plot(px[-1], py[-1], "r^", markersize=15, zorder=6, label="End")
    ax1.set_title(f"Closest 4m ray - {name}\n{step+1} steps, cov={final_cov:.1%}, dist={env.total_distance:.0f}m")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.legend()

    steps_arr = [s["step"] for s in snaps]
    covs_arr = [s["cov"] for s in snaps]
    ax2.plot(steps_arr, covs_arr, linewidth=2, color="#2ecc71")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Coverage Progress")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"closest_4m_{name}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)
