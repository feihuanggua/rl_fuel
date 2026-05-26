"""验证前沿形状改进: 对比 pinhole 相机 vs 均匀角射线, 以及参数对齐后的效果."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
)
from fuel_rl.map_loader import generate_random_map_for_fuel

SEED = 42
MAP_SIZE = (20.0, 20.0, 3.0)
NUM_PILLARS = 15
AGENT_START = np.array([0.0, 0.0, 1.5])


def run_test(label):
    core = FuelEnvCore()

    # 加载相同地图
    box_margin = 1.0
    mp = default_map_params(
        size_x=MAP_SIZE[0], size_y=MAP_SIZE[1], size_z=MAP_SIZE[2],
        box_min=(-MAP_SIZE[0]/2 + box_margin, -MAP_SIZE[1]/2 + box_margin, 0.0),
        box_max=(MAP_SIZE[0]/2 - box_margin, MAP_SIZE[1]/2 - box_margin, MAP_SIZE[2] - 0.2),
    )
    fp = default_frontier_params()
    pp = default_perception_params()
    ap = default_astar_params()

    core.init(mp, fp, pp, ap)

    pts = generate_random_map_for_fuel(MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2], NUM_PILLARS, seed=SEED)
    core.load_map_from_points(pts)
    core.reset_map()

    # 模拟四个方向初始观测 (与 FUEL 行为一致)
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(AGENT_START, yaw)

    # 检测前沿
    frontiers = core.detect_frontiers(AGENT_START)

    print(f"[{label}] 前沿数={len(frontiers)}")
    stats = []
    for i, f in enumerate(frontiers):
        cells = np.array(f.cells)
        avg = np.array(f.average)
        size = f.frontier_size
        vp = np.array(f.best_viewpoint_pos)
        visib = f.best_viewpoint_visib_num
        stats.append((size, visib))
        x_range = cells[:, 0].max() - cells[:, 0].min()
        y_range = cells[:, 1].max() - cells[:, 1].min()
        print(f"  F{i}: size={size:5d}, cells_xy=({x_range:.1f},{y_range:.1f}), "
              f"visib={visib}, vp_dist_to_center={np.linalg.norm(vp - avg):.2f}")

    sizes = [s[0] for s in stats]
    if sizes:
        print(f"  平均大小: {np.mean(sizes):.0f}, 最小: {min(sizes)}, 最大: {max(sizes)}")
    print()

    return core, frontiers


def visualize(core, frontiers, title, fname):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 画占用栅格 (已知空间)
    occ = np.array(core.get_occupancy_slice_2d(1.5)).reshape(
        core.get_map_voxel_num()[1], core.get_map_voxel_num()[0]
    )
    # 画未知区域 (灰色)
    ax.imshow((occ == 0).T, extent=(-10, 10, -10, 10), origin="lower",
              cmap="gray_r", alpha=0.3, vmin=0, vmax=1)

    # 画已知 free 区域 (白色)
    free_mask = (occ == 1).T
    ax.contourf(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1),
                free_mask, levels=[0.5, 1], colors=["lightblue"], alpha=0.3)

    # 画障碍物 (黑色)
    obs_mask = ((occ == 2) | (occ == 3)).T
    ax.contourf(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1),
                obs_mask, levels=[0.5, 1], colors=["black"], alpha=0.5)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(frontiers), 1)))
    for i, f in enumerate(frontiers):
        cells = np.array(f.cells)
        if len(cells) == 0:
            continue
        ax.scatter(cells[:, 0], cells[:, 1], s=1, color=colors[i % len(colors)],
                   alpha=0.7, label=f"F{i}({f.frontier_size})")
        avg = np.array(f.average)
        vp = np.array(f.best_viewpoint_pos)
        ax.plot(avg[0], avg[1], "x", color=colors[i % len(colors)], markersize=8, mew=2)
        ax.plot([avg[0], vp[0]], [avg[1], vp[1]], "-", color=colors[i % len(colors)], alpha=0.5)

    ax.plot(AGENT_START[0], AGENT_START[1], "r*", markersize=15, label="Start")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_title(title)
    if len(frontiers) <= 20:
        ax.legend(loc="upper right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    print("=" * 60)
    print("前沿形状测试: 新参数 + pinhole 相机模型")
    print("=" * 60)
    core, frontiers = run_test("新版本")

    if frontiers:
        visualize(core, frontiers,
                  "Frontiers (pinhole camera + aligned params)",
                  "/tmp/frontier_test_new.png")
    else:
        print("WARNING: 未检测到前沿!")
