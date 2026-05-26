"""追踪前沿更新流程 — 每步观测后的前沿增删变化."""
import numpy as np
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
)
from fuel_rl.map_loader import generate_random_map_for_fuel


def trace_frontier_flow(seed=42, max_steps=15):
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

    # 初始 4 方向观测
    print("=== 初始 4 方向观测 ===")
    for yaw_name, yaw in [("0°", 0), ("90°", np.pi/2), ("180°", np.pi), ("270°", -np.pi/2)]:
        prev_frontiers = core.detect_frontiers(agent_pos)
        prev_ids = {f.id: (f.frontier_size, np.array(f.average).tolist()) for f in prev_frontiers}
        prev_total_cells = sum(f.frontier_size for f in prev_frontiers)

        core.simulate_observation(agent_pos, yaw)

        after_frontiers = core.detect_frontiers(agent_pos)
        after_ids = {f.id: (f.frontier_size, np.array(f.average).tolist()) for f in after_frontiers}
        after_total_cells = sum(f.frontier_size for f in after_frontiers)

        new_ids = set(after_ids.keys()) - set(prev_ids.keys())
        removed_ids = set(prev_ids.keys()) - set(after_ids.keys())
        kept_ids = set(prev_ids.keys()) & set(after_ids.keys())

        progress = core.get_exploration_progress()
        print(f"\n观测 yaw={yaw_name}:")
        print(f"  覆盖率: {progress:.1%}, 前沿: {len(prev_frontiers)} → {len(after_frontiers)}")
        print(f"  前沿cells: {prev_total_cells} → {after_total_cells} ({after_total_cells - prev_total_cells:+d})")
        print(f"  保留: {len(kept_ids)}, 新增: {len(new_ids)}, 删除: {len(removed_ids)}")
        if new_ids:
            for nid in sorted(new_ids):
                s, avg = after_ids[nid]
                print(f"    + 前沿#{nid}: cells={s}, avg=({avg[0]:+.1f},{avg[1]:+.1f},{avg[2]:+.1f})")
        if removed_ids:
            for rid in sorted(removed_ids):
                s, avg = prev_ids[rid]
                print(f"    - 前沿#{rid}: cells={s}, avg=({avg[0]:+.1f},{avg[1]:+.1f},{avg[2]:+.1f})")

    # 探索步骤
    print("\n\n=== 探索步骤 (最近新前沿策略) ===")
    visited = []

    for step in range(max_steps):
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            print(f"\nStep {step}: 无前沿，探索完成")
            break

        prev_total_cells = sum(f.frontier_size for f in frontiers)
        prev_ids = {f.id: f.frontier_size for f in frontiers}

        # 选最近的新前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)
                 if not any(np.linalg.norm(np.array(f.average) - vp) < 1.0 for vp in visited[-20:]) and d >= 0.3]
        if not new_f:
            new_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)]
        new_f.sort(key=lambda x: x[2])
        target = new_f[0][1]

        # 用 best_viewpoint 作为目标位置
        vp_pos = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw

        # 有效性检查
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            vp_pos = np.array(target.average)
            vp_pos[2] = 1.5
            vp_yaw = 0.0

        # 记录 before 状态
        progress_before = core.get_exploration_progress()

        # 执行观测
        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        new_unk = core.count_unknown_voxels()
        discovered = prev_unk - new_unk

        # 检测前沿变化
        after_frontiers = core.detect_frontiers(vp_pos)
        after_total_cells = sum(f.frontier_size for f in after_frontiers)
        after_ids = {f.id: f.frontier_size for f in after_frontiers}

        new_ids = set(after_ids.keys()) - set(prev_ids.keys())
        removed_ids = set(prev_ids.keys()) - set(after_ids.keys())
        kept_ids = set(prev_ids.keys()) & set(after_ids.keys())

        progress_after = core.get_exploration_progress()

        print(f"\nStep {step}: 飞→({vp_pos[0]:+.1f},{vp_pos[1]:+.1f},{vp_pos[2]:+.1f}) yaw={np.degrees(vp_yaw):.0f}°")
        print(f"  发现: {discovered} 体素, 覆盖率: {progress_before:.1%} → {progress_after:.1%}")
        print(f"  前沿数: {len(frontiers)} → {len(after_frontiers)} ({len(after_frontiers)-len(frontiers):+d})")
        print(f"  前沿cells: {prev_total_cells} → {after_total_cells} ({after_total_cells-prev_total_cells:+d})")
        print(f"  保留: {len(kept_ids)}, 新增: {len(new_ids)}, 删除: {len(removed_ids)}")

        # 大小变化的前沿
        for kid in kept_ids:
            old_s = prev_ids[kid]
            new_s = after_ids[kid]
            if abs(new_s - old_s) > 50:
                print(f"    ~ 前沿#{kid}: cells {old_s} → {new_s} ({new_s-old_s:+d})")

        if new_ids:
            for nid in sorted(new_ids)[:5]:
                print(f"    + 前沿#{nid}: cells={after_ids[nid]}")
        if removed_ids:
            for rid in sorted(removed_ids)[:5]:
                print(f"    - 前沿#{rid}: cells={prev_ids[rid]}")

        agent_pos = vp_pos.copy()
        visited.append(agent_pos.copy())

    # 最终状态
    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    total_cells = sum(f.frontier_size for f in frontiers)
    print(f"\n=== 最终: 覆盖率 {progress:.1%}, 前沿 {len(frontiers)}, cells {total_cells} ===")


if __name__ == "__main__":
    trace_frontier_flow()
