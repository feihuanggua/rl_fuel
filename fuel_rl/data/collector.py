"""专家数据收集 — 单地图子进程版."""
import os
import numpy as np
import torch


def build_3channel_grid(core, frontier, grid_size=32, grid_z=10, resolution=0.2):
    """构建 [3, grid_size, grid_size, grid_z] 体素网格.

    优化: 预计算所有世界坐标 (numpy meshgrid), 用单次展平遍历替代三重 Python 循环,
    减少 Python→C++ 调用开销和中间 np.array() 分配.
    """
    center = np.array(frontier.average, dtype=np.float64)
    nx, ny, nz = grid_size, grid_size, grid_z
    half = np.array([nx / 2.0, ny / 2.0, nz / 2.0])

    # 预计算所有体素的世界坐标 [nx*ny*nz, 3]
    offsets = (np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T.astype(np.float64) - half + 0.5) * resolution
    world_coords = offsets + center  # broadcasting: center [3] + [N, 3]

    # 展平查询 occupancy, 一次 Python 循环 (每个 voxel 仍需 C++ 调用)
    ch_occ = np.zeros(nx * ny * nz, dtype=np.float32)
    ch_free = np.zeros(nx * ny * nz, dtype=np.float32)

    for i in range(len(world_coords)):
        occ = core.get_occupancy(world_coords[i])
        if occ == 2:
            ch_occ[i] = 1.0
        elif occ == 1:
            ch_free[i] = 1.0

    ch_occ = ch_occ.reshape(nx, ny, nz)
    ch_free = ch_free.reshape(nx, ny, nz)

    # 前沿通道 (向量化, 无变化)
    ch_frontier = np.zeros((nx, ny, nz), dtype=np.float32)
    cells = np.array(frontier.cells)
    if len(cells) > 0:
        local = (cells - center) / resolution
        idx = np.stack([
            (local[:, 0] + half[0]).astype(int),
            (local[:, 1] + half[1]).astype(int),
            (local[:, 2] + half[2]).astype(int),
        ], axis=1)
        valid = ((idx[:, 0] >= 0) & (idx[:, 0] < nx) &
                 (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
                 (idx[:, 2] >= 0) & (idx[:, 2] < nz))
        idx = idx[valid]
        if len(idx) > 0:
            ch_frontier[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    return np.stack([ch_occ, ch_frontier, ch_free], axis=0)


def get_expert_label(frontier, max_dist=4.0):
    center = np.array(frontier.average)
    vp_pos = np.array(frontier.best_viewpoint_pos)
    return np.array([
        (vp_pos[0] - center[0]) / max_dist,
        (vp_pos[1] - center[1]) / max_dist,
        (vp_pos[2] - center[2]) / max_dist,
        frontier.best_viewpoint_yaw / np.pi,
    ], dtype=np.float32)


def collect_single_map(seed, num_pillars=15, grid_size=32, grid_z=10, resolution=0.2):
    """收集单张地图的数据，用于子进程调用."""
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel

    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9.0, -9.0, 0.0), box_max=(9.0, 9.0, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    pts = generate_random_map_for_fuel(20, 20, 3, num_pillars, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    start = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(start, yaw)

    frontiers = core.detect_frontiers(start)

    grids, labels = [], []
    for f in frontiers:
        if f.frontier_size < 5:
            continue
        grid = build_3channel_grid(core, f, grid_size, grid_z, resolution)
        label = get_expert_label(f)
        if np.any(np.abs(label[:3]) > 1.0):
            continue
        grids.append(grid)
        labels.append(label)

    return grids, labels


def collect_expert_data(
    num_maps=200,
    num_pillars=15,
    grid_size=32,
    grid_z=10,
    resolution=0.2,
    save_path="./fuel_rl_data/expert_data.pt",
    seed_start=0,
):
    """使用子进程收集数据，避免 C++ 内存泄漏."""
    from multiprocessing import Pool
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    args = [(seed_start + i, num_pillars, grid_size, grid_z, resolution) for i in range(num_maps)]

    all_grids = []
    all_labels = []

    with Pool(processes=4) as pool:
        for i, (grids, labels) in enumerate(pool.starmap(collect_single_map, args)):
            all_grids.extend(grids)
            all_labels.extend(labels)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_maps} maps: {len(all_grids)} samples")

    if not all_grids:
        print("WARNING: No data collected!")
        return

    inputs = np.stack(all_grids).astype(np.float16)
    targets = np.stack(all_labels).astype(np.float32)

    torch.save({
        "inputs": torch.from_numpy(inputs),
        "targets": torch.from_numpy(targets),
        "config": {"grid_size": grid_size, "grid_z": grid_z,
                   "resolution": resolution, "num_samples": len(all_grids)},
    }, save_path)

    print(f"Collected {len(all_grids)} samples from {num_maps} maps")
    print(f"Saved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-maps", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--grid-z", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="./fuel_rl_data/expert_data.pt")
    args = parser.parse_args()
    collect_expert_data(num_maps=args.num_maps, grid_size=args.grid_size,
                        grid_z=args.grid_z, save_path=args.save_path)
