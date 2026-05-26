"""前沿检测链路诊断 — 验证每个环节的输出."""
import numpy as np
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
    GRID_SIZE, GRID_Z, VOXEL_RES,
)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid


def diagnose(seed=42):
    # 1. 初始化环境
    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    print("=== 环境初始化 ===")
    print(f"地图种子: {seed}, 障碍柱: 15")
    print(f"地图尺寸: 20x20x3m, 分辨率: 0.1m")

    # 2. 初始观测 (4方向)
    agent_pos = np.array([0.0, 0.0, 1.5])
    print(f"\n=== 初始观测 ===")
    print(f"起点: {agent_pos}, 4方向扫描 (0/90/180/270°)")

    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    progress = core.get_exploration_progress()
    unknown = core.count_unknown_voxels()
    print(f"初始覆盖率: {progress:.1%}, 未知体素: {unknown}")

    # 3. 前沿检测
    frontiers = core.detect_frontiers(agent_pos)
    print(f"\n=== 前沿检测 ===")
    print(f"检测到前沿数: {len(frontiers)}")

    if not frontiers:
        print("没有前沿! 前沿检测链路有问题.")
        return

    # 4. 前沿详细信息
    print(f"\n{'ID':>3} {'Cells':>6} {'AvgPos':>30} {'BBoxMin':>30} {'BBoxMax':>30} {'BestVP':>30} {'BestYaw':>8} {'BestVis':>8}")
    print("-" * 160)

    for i, f in enumerate(frontiers[:10]):
        avg = np.array(f.average)
        bmin = np.array(f.box_min)
        bmax = np.array(f.box_max)
        bvp = np.array(f.best_viewpoint_pos)
        print(f"{f.id:>3} {f.frontier_size:>6} "
              f"({avg[0]:+6.2f},{avg[1]:+6.2f},{avg[2]:+6.2f})     "
              f"({bmin[0]:+6.2f},{bmin[1]:+6.2f},{bmin[2]:+6.2f})     "
              f"({bmax[0]:+6.2f},{bmax[1]:+6.2f},{bmax[2]:+6.2f})     "
              f"({bvp[0]:+6.2f},{bvp[1]:+6.2f},{bvp[2]:+6.2f})     "
              f"{f.best_viewpoint_yaw:>+7.2f}° {f.best_viewpoint_visib_num:>8}")

    # 5. 构建网格并分析
    print(f"\n=== 3通道网格诊断 ===")
    for i, f in enumerate(frontiers[:5]):
        grid = build_3channel_grid(core, f, GRID_SIZE, GRID_Z, VOXEL_RES)
        ch_occ = grid[0]
        ch_frontier = grid[1]
        ch_free = grid[2]

        center = np.array(f.average)
        cells = np.array(f.cells)
        local = (cells - center) / VOXEL_RES
        half = np.array([GRID_SIZE / 2.0, GRID_SIZE / 2.0, GRID_Z / 2.0])

        # 统计前沿 cells 落入网格的情况
        idx = (local + half).astype(int)
        in_grid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < GRID_SIZE) &
            (idx[:, 1] >= 0) & (idx[:, 1] < GRID_SIZE) &
            (idx[:, 2] >= 0) & (idx[:, 2] < GRID_Z)
        )
        n_in_grid = in_grid.sum()
        n_total = len(cells)

        # 前沿 cells 在 Z 维度的分布
        z_dists = cells[:, 2] if len(cells) > 0 else np.array([])
        z_unique = np.unique(np.round(cells[:, 2], 1)) if len(cells) > 0 else np.array([])

        # 网格中前沿层在 Z 轴的分布
        frontier_z_counts = ch_frontier.sum(axis=(0, 1))

        print(f"\n前沿 #{f.id} (cells={n_total}):")
        print(f"  障碍物通道: {ch_occ.sum():.0f} 体素 ({ch_occ.mean() * 100:.1f}%)")
        print(f"  前沿通道:   {ch_frontier.sum():.0f} 体素 ({ch_frontier.mean() * 100:.2f}%)")
        print(f"  自由通道:   {ch_free.sum():.0f} 体素 ({ch_free.mean() * 100:.1f}%)")
        print(f"  空通道:     {(1 - (ch_occ + ch_frontier + ch_free).clip(0, 1)).sum():.0f} 体素 (UNKNOWN)")
        print(f"  前沿cells落入网格: {n_in_grid}/{n_total} ({n_in_grid / max(n_total, 1) * 100:.0f}%)")
        print(f"  前沿cells Z范围: [{z_dists.min():.2f}, {z_dists.max():.2f}] 均值={z_dists.mean():.2f}")
        print(f"  前沿cells Z唯一值: {z_unique.tolist()[:10]}")
        print(f"  网格前沿层Z分布: {frontier_z_counts.astype(int).tolist()}")
        print(f"  网格中心Z={center[2]:.2f}, 网格Z范围=[{center[2] - (GRID_Z/2)*VOXEL_RES:.2f}, {center[2] + (GRID_Z/2)*VOXEL_RES:.2f}]")

    # 6. 可视化：第一个前沿的 Z 中间层切片
    print(f"\n=== 可视化 (前沿 #{frontiers[0].id} Z 中间层切片) ===")
    f = frontiers[0]
    grid = build_3channel_grid(core, f)
    center_z = np.array(f.average)[2]
    z_mid = GRID_Z // 2
    slice_occ = grid[0, :, :, z_mid]
    slice_frt = grid[1, :, :, z_mid]
    slice_free = grid[2, :, :, z_mid]

    print(f"Z切片 index={z_mid} (世界Z≈{center_z + (z_mid - GRID_Z/2 + 0.5) * VOXEL_RES:.2f}m)")
    # 用不同字符表示
    symbols = {0: '.', 1: '#', 2: 'F', 3: '?'}  # 空, 障碍, 前沿, 自由
    for y in range(0, GRID_SIZE, 2):
        row = ""
        for x in range(GRID_SIZE):
            if slice_frt[x, y] > 0:
                row += "F"
            elif slice_occ[x, y] > 0:
                row += "#"
            elif slice_free[x, y] > 0:
                row += "."
            else:
                row += " "
        if y % 4 == 0:
            print(f"  y={y:2d} |{row}|")

    print(f"  图例: F=前沿 #=障碍 .=自由 ' '=未知")


if __name__ == "__main__":
    diagnose()
