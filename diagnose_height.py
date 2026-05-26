"""诊断前沿 Z 方向分布 — 验证高度是否存在遗漏."""
import numpy as np
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
)
from fuel_rl.map_loader import generate_random_map_for_fuel


def diagnose_height(seed=42):
    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    print("=== 1. Z方向占用状态分布 ===")
    print(f"地图: 20x20x3m, 分辨率: 0.1m, box_z: [0, 2.8]")
    print(f"相机: z=1.5m, FOV_v=64°, forward=水平")
    print()

    # 逐层扫描占用状态
    resolution = 0.1
    z_stats = []
    for z_idx in range(31):  # 0m to 3.0m
        z = z_idx * resolution
        n_free, n_occ, n_unk = 0, 0, 0
        for x_idx in range(200):
            for y_idx in range(200):
                x = -10.0 + x_idx * resolution
                y = -10.0 + y_idx * resolution
                occ = core.get_occupancy(np.array([x, y, z]))
                if occ == 1:
                    n_free += 1
                elif occ == 2:
                    n_occ += 1
                else:
                    n_unk += 1
        z_stats.append((z, n_free, n_occ, n_unk))
        bar_free = "█" * (n_free // 200)
        bar_occ = "▓" * (n_occ // 200)
        bar_unk = "░" * min(n_unk // 200, 50)
        print(f"z={z:.1f}m  FREE={n_free:>6}({n_free/40000*100:>5.1f}%)  "
              f"OCC={n_occ:>5}({n_occ/40000*100:>4.1f}%)  "
              f"UNK={n_unk:>6}({n_unk/40000*100:>5.1f}%)  {bar_free}{bar_occ}{bar_unk}")

    print("\n=== 2. Z方向潜在前沿分析 ===")
    print("(FREE 体素的 ±Z 邻居中有 UNKNOWN 的数量)")
    # 采样检测（避免全量扫描太慢）
    step = 0.5  # 0.5m 间隔
    for z in np.arange(0.1, 2.8, step):
        z_up = z + resolution
        z_down = z - resolution
        potential_frontier_up = 0
        potential_frontier_down = 0
        # 采样 100x100 点
        for x_idx in range(0, 200, 2):
            for y_idx in range(0, 200, 2):
                x = -10.0 + x_idx * resolution
                y = -10.0 + y_idx * resolution
                pos = np.array([x, y, z])
                if core.get_occupancy(pos) == 1:  # FREE
                    if z_up <= 2.8:
                        up = np.array([x, y, z_up])
                        if core.get_occupancy(up) == 0:  # UNKNOWN
                            potential_frontier_up += 1
                    if z_down >= 0.0:
                        down = np.array([x, y, z_down])
                        if core.get_occupancy(down) == 0:  # UNKNOWN
                            potential_frontier_down += 1
        # 乘以 4（采样 2x2 取 1 个点）
        print(f"z={z:.1f}m: 上方边界={potential_frontier_up*4:>6}  下方边界={potential_frontier_down*4:>6}")

    # 前沿检测
    print("\n=== 3. 实际检测到的前沿 Z 分布 ===")
    frontiers = core.detect_frontiers(agent_pos)
    print(f"前沿总数: {len(frontiers)}")
    for f in frontiers:
        cells = np.array(f.cells)
        print(f"  前沿#{f.id:>2}: cells={f.frontier_size:>5}  "
              f"Z=[{cells[:,2].min():.1f}, {cells[:,2].max():.1f}]  "
              f"avg_z={np.array(f.average)[2]:.2f}")

    # 总体 Z 分布
    all_cells = np.concatenate([np.array(f.cells) for f in frontiers])
    print(f"\n所有前沿cells: {len(all_cells)}")
    print(f"Z范围: [{all_cells[:,2].min():.2f}, {all_cells[:,2].max():.2f}]")
    z_hist, z_edges = np.histogram(all_cells[:, 2], bins=np.arange(0, 3.1, 0.2))
    print("Z直方图 (0.2m bin):")
    for i, (count, edge) in enumerate(zip(z_hist, z_edges)):
        bar = "█" * (count // 50)
        print(f"  [{edge:.1f}, {edge+0.2:.1f})m: {count:>5}  {bar}")


if __name__ == "__main__":
    diagnose_height()
