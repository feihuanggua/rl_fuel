"""诊断前沿更新问题: 逐步检查 simulate_observation 后前沿是否正确更新."""
import numpy as np


def debug_frontier_update():
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel

    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    pts = generate_random_map_for_fuel(20, 20, 3.0, 15, seed=42)
    core.load_map_from_points(pts)
    core.reset_map()

    # 初始观测
    agent_pos = np.array([0.0, 0.0, 1.5])
    print("=== 初始观测 ===")
    for i, yaw in enumerate([0, np.pi/2, np.pi, -np.pi/2]):
        before = core.count_unknown_voxels()
        core.simulate_observation(agent_pos, yaw)
        after = core.count_unknown_voxels()
        print(f"  Yaw {i}: unknown {before} -> {after} (discovered {before - after})")

    # 检测前沿
    frontiers = core.detect_frontiers(agent_pos)
    print(f"\n初始前沿: {len(frontiers)} 个")
    total_cells = sum(f.frontier_size for f in frontiers)
    print(f"前沿总格子数: {total_cells}")
    progress = core.get_exploration_progress()
    print(f"探索进度: {progress:.1%}")
    print(f"未知体素: {core.count_unknown_voxels()}")

    if not frontiers:
        print("ERROR: 初始前沿为空!")
        return

    # 打印前沿详情
    for i, f in enumerate(frontiers):
        avg = np.array(f.average)
        vp = np.array(f.best_viewpoint_pos)
        print(f"  F{i}: center=({avg[0]:.1f},{avg[1]:.1f},{avg[2]:.1f}) "
              f"size={f.frontier_size} "
              f"best_vp=({vp[0]:.1f},{vp[1]:.1f},{vp[2]:.1f}) "
              f"yaw={f.best_viewpoint_yaw:.2f} visib={f.best_viewpoint_visib_num}")

    # === 测试1: 用 expert viewpoint 观察 ===
    print("\n=== 测试1: 用最佳视点观察第1个前沿 ===")
    target = frontiers[0]
    vp_pos = np.array(target.best_viewpoint_pos)
    vp_yaw = target.best_viewpoint_yaw
    print(f"视点: pos=({vp_pos[0]:.2f},{vp_pos[1]:.2f},{vp_pos[2]:.2f}) yaw={vp_yaw:.2f}")

    occ = core.get_occupancy(vp_pos)
    print(f"视点占据状态: {occ} (1=free, 2=occupied, 0=unknown)")
    if occ != 1:
        print(f"WARNING: 视点不在自由空间!")
        # 尝试调整到最近的安全点
        print("尝试在附近寻找安全视点...")
        for dx in np.arange(-0.5, 0.6, 0.2):
            for dy in np.arange(-0.5, 0.6, 0.2):
                test_pos = vp_pos + np.array([dx, dy, 0])
                if core.get_occupancy(test_pos) == 1:
                    vp_pos = test_pos
                    print(f"  找到安全点: ({vp_pos[0]:.2f},{vp_pos[1]:.2f},{vp_pos[2]:.2f})")
                    break
            if core.get_occupancy(vp_pos) == 1:
                break

    before_unk = core.count_unknown_voxels()
    before_free = core.count_free_voxels()
    core.simulate_observation(vp_pos, vp_yaw)
    after_unk = core.count_unknown_voxels()
    after_free = core.count_free_voxels()
    print(f"未知: {before_unk} -> {after_unk} (发现 {before_unk - after_unk})")
    print(f"自由: {before_free} -> {after_free}")
    print(f"进度: {progress:.1%} -> {core.get_exploration_progress():.1%}")

    # 重新检测前沿
    agent_pos = vp_pos.copy()
    new_frontiers = core.detect_frontiers(agent_pos)
    print(f"\n观察后前沿: {len(new_frontiers)} 个")
    new_total = sum(f.frontier_size for f in new_frontiers)
    print(f"前沿总格子数: {new_total}")

    # 比较
    old_ids = set(tuple(np.round(np.array(f.average), 1)) for f in frontiers)
    new_ids = set(tuple(np.round(np.array(f.average), 1)) for f in new_frontiers)
    removed = old_ids - new_ids
    added = new_ids - old_ids
    kept = old_ids & new_ids
    print(f"\n变化: 保留={len(kept)} 移除={len(removed)} 新增={len(added)}")

    if removed:
        print("移除的前沿中心:")
        for c in removed:
            print(f"  ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
    if added:
        print("新增的前沿:")
        for i, f in enumerate(new_frontiers):
            c = tuple(np.round(np.array(f.average), 1))
            if c in added:
                vp = np.array(f.best_viewpoint_pos)
                print(f"  ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) size={f.frontier_size} "
                      f"vp=({vp[0]:.1f},{vp[1]:.1f},{vp[2]:.1f}) visib={f.best_viewpoint_visib_num}")

    # === 测试2: 持续探索多步 ===
    print("\n=== 测试2: 用 expert 视点持续探索5步 ===")
    for step in range(5):
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            print(f"Step {step}: 探索完成!")
            break

        # 选最近的前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        target = frontiers[int(np.argmin(dists))]

        vp_pos = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw

        # 安全检查
        if core.get_occupancy(vp_pos) != 1:
            safe = False
            for dx in np.arange(-0.5, 0.6, 0.2):
                for dy in np.arange(-0.5, 0.6, 0.2):
                    test_pos = vp_pos + np.array([dx, dy, 0])
                    if core.get_occupancy(test_pos) == 1:
                        vp_pos = test_pos
                        safe = True
                        break
                if safe:
                    break
            if not safe:
                print(f"Step {step}: 无法找到安全视点, skip")
                continue

        before_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = before_unk - core.count_unknown_voxels()
        agent_pos = vp_pos.copy()

        new_frontiers = core.detect_frontiers(agent_pos)
        print(f"Step {step}: frontiers {len(frontiers)} -> {len(new_frontiers)}, "
              f"discovered={discovered}, progress={core.get_exploration_progress():.1%}")


if __name__ == "__main__":
    debug_frontier_update()
