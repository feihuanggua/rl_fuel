"""追踪探索过程中前沿数量变化."""
import numpy as np
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
    GRID_SIZE, GRID_Z, VOXEL_RES, ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE,
)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid
from fuel_rl.models import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointHead
import torch


def trace_frontiers(seed=42, max_steps=50):
    # 加载 BC 模型（用 ViewpointHead + strict=False 跳过 BN running stats）
    encoder = Encoder3D(input_shape=(GRID_SIZE, GRID_SIZE, GRID_Z), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointHead(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    state = torch.load("./fuel_rl_checkpoints/bc_v2/best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("BC v2 model loaded\n")

    # 初始化环境
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

    max_dist = 4.0
    visited = []

    print(f"{'Step':>4} {'Frontiers':>10} {'New':>5} {'Old':>5} {'Target':>6} "
          f"{'Progress':>10} {'Action':>32} {'Result':>10} {'Cells':>6}")
    print("-" * 110)

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        n_frontiers = len(frontiers)

        if not frontiers:
            print(f"{step:4d} {n_frontiers:>10} {'---':>5} {'---':>5} {'---':>6} {progress:>9.1%}  {'---':>32} {'DONE':>10}")
            break

        # 分类前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)
                 if not any(np.linalg.norm(np.array(f.average) - vp) < 1.0 for vp in visited[-20:]) and d >= 0.3]
        old_f = [(i, f, d) for i, f, d in zip(range(len(frontiers)), frontiers, dists)
                 if any(np.linalg.norm(np.array(f.average) - vp) < 1.0 for vp in visited[-20:]) or d < 0.3]

        # 选择目标前沿
        if new_f:
            new_f.sort(key=lambda x: x[2])
            target = new_f[0][1]
            target_type = "new"
        else:
            old_f.sort(key=lambda x: -x[2])
            target = old_f[0][1]
            target_type = "old"

        # 模型推理
        grid = build_3channel_grid(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = model(grid_t).cpu().numpy().flatten()

        center = np.array(target.average)
        vp_pos = center + action[:3] * max_dist
        vp_pos[2] = np.clip(vp_pos[2], 0.5, 2.6)
        vp_yaw = action[3] * np.pi

        # 检查有效性
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            print(f"{step:4d} {n_frontiers:>10} {len(new_f):>5} {len(old_f):>5} {target.id:>6} "
                  f"{progress:>9.1%}  dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} yaw={action[3]:+.2f} "
                  f"{'INVALID':>10} {'---':>6}")
            continue

        path_cost = core.compute_path_cost(agent_pos, vp_pos)
        if path_cost < 0 or path_cost > 100:
            print(f"{step:4d} {n_frontiers:>10} {len(new_f):>5} {len(old_f):>5} {target.id:>6} "
                  f"{progress:>9.1%}  dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} yaw={action[3]:+.2f} "
                  f"{'NO_PATH':>10} {'---':>6}")
            continue

        # 执行
        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        new_unk = core.count_unknown_voxels()
        discovered = prev_unk - new_unk

        agent_pos = vp_pos.copy()
        visited.append(agent_pos.copy())

        # 重新检测前沿数（更新后）
        frontiers_after = core.detect_frontiers(agent_pos)
        delta = len(frontiers_after) - n_frontiers

        print(f"{step:4d} {n_frontiers:>10} {len(new_f):>5} {len(old_f):>5} {target.id:>6} "
              f"{progress:>9.1%}  dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} yaw={action[3]:+.2f} "
              f"{'OK':>10} {discovered:>6}  -> {len(frontiers_after):>3} ({delta:+d})")

    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    print(f"\n最终: 覆盖率 {progress:.1%}, 前沿 {len(frontiers)}")


if __name__ == "__main__":
    trace_frontiers()
