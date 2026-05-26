"""深度诊断: 追踪前沿更新流程中 cells 的去向 — 新增/合并/丢失."""
import numpy as np
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
)
from fuel_rl.map_loader import generate_random_map_for_fuel


def count_potential_frontiers(core):
    """手动统计所有前沿体素 (FREE + UNKNOWN邻居)."""
    resolution = core.get_resolution()
    voxel_num = core.get_map_voxel_num()
    # 采样扫描 (2x 降采样避免太慢)
    step = 2
    n_potential = 0
    z_range = range(4, 28)  # z=0.4m to 2.7m (跳过地面)
    for x in range(0, 200, step):
        for y in range(0, 200, step):
            for z in z_range:
                wx = -10.0 + x * resolution
                wy = -10.0 + y * resolution
                wz = z * resolution
                pos = np.array([wx, wy, wz])
                if core.get_occupancy(pos) != 1:  # not FREE
                    continue
                # 检查6邻域
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nb = pos + np.array([dx, dy, dz]) * resolution
                    if core.get_occupancy(nb) == 0:  # UNKNOWN
                        n_potential += 1
                        break
    return n_potential * (step ** 2)  # 还原采样


def diagnose(seed=42, max_steps=6):
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

    print("=== 前沿更新流程深度诊断 ===\n")

    frontiers = core.detect_frontiers(agent_pos)
    detected_cells = sum(f.frontier_size for f in frontiers)
    potential = count_potential_frontiers(core)

    print(f"初始状态:")
    print(f"  覆盖率: {core.get_exploration_progress():.1%}")
    print(f"  检测到前沿: {len(frontiers)} 个, {detected_cells} cells")
    print(f"  全局潜在前沿cells (采样估算): ~{potential}")
    print(f"  检测/潜在比: {detected_cells/max(potential,1):.1%}")

    # 探索
    visited = []
    for step in range(max_steps):
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            break

        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)
                 if not any(np.linalg.norm(np.array(f.average) - vp) < 1.0 for vp in visited[-20:]) and d >= 0.3]
        if not new_f:
            new_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)]
        new_f.sort(key=lambda x: x[2])
        target = new_f[0][1]

        vp_pos = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw
        if core.get_occupancy(vp_pos) != 1:
            vp_pos = np.array(target.average)
            vp_pos[2] = 1.5
            vp_yaw = 0.0

        prev_detected = sum(f.frontier_size for f in frontiers)

        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = prev_unk - core.count_unknown_voxels()

        after_frontiers = core.detect_frontiers(vp_pos)
        after_detected = sum(f.frontier_size for f in after_frontiers)

        # 全局潜在前沿
        potential = count_potential_frontiers(core)

        print(f"\nStep {step}: 飞→({vp_pos[0]:+.1f},{vp_pos[1]:+.1f},{vp_pos[2]:+.1f})")
        print(f"  发现: {discovered} 体素, 覆盖率: {core.get_exploration_progress():.1%}")
        print(f"  检测前沿: {len(frontiers)}→{len(after_frontiers)} ({len(after_frontiers)-len(frontiers):+d})")
        print(f"  检测cells: {prev_detected}→{after_detected} ({after_detected-prev_detected:+d})")
        print(f"  全局潜在cells (采样): ~{potential}")
        print(f"  检测/潜在比: {after_detected/max(potential,1):.1%}")

        agent_pos = vp_pos.copy()
        visited.append(agent_pos.copy())


if __name__ == "__main__":
    diagnose()
