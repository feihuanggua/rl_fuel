"""对比 expert vs model 视点的前沿更新效果."""
import numpy as np
import torch


def debug_model_vs_expert():
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel
    from fuel_rl.models import ViewpointHead, Encoder3D
    from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE


    def build_grid_32x10(core, frontier):
        """构建 [3, 32, 32, 10] 体素网格 (BC v2 格式)."""
        center = np.array(frontier.average)
        nx, ny, nz = 32, 32, 10
        resolution = 0.2
        grid = np.zeros((3, nx, ny, nz), dtype=np.float32)
        for dz in range(nz):
            for dy in range(ny):
                for dx in range(nx):
                    wx = center[0] + (dx - 16 + 0.5) * resolution
                    wy = center[1] + (dy - 16 + 0.5) * resolution
                    wz = center[2] + (dz - 5 + 0.5) * resolution
                    occ = core.get_occupancy(np.array([wx, wy, wz]))
                    if occ == 2:
                        grid[0, dx, dy, dz] = 1.0
                    elif occ == 1:
                        grid[2, dx, dy, dz] = 1.0
        cells = np.array(frontier.cells)
        if len(cells) > 0:
            local = (cells - center) / resolution
            idx = np.stack([
                (local[:, 0] + 16).astype(int),
                (local[:, 1] + 16).astype(int),
                (local[:, 2] + 5).astype(int),
            ], axis=1)
            valid = ((idx[:, 0] >= 0) & (idx[:, 0] < 32) &
                     (idx[:, 1] >= 0) & (idx[:, 1] < 32) &
                     (idx[:, 2] >= 0) & (idx[:, 2] < 10))
            idx = idx[valid]
            if len(idx) > 0:
                grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        return grid

    # 加载 BC v2 模型 (input_shape=(32,32,10), 0.2m resolution)
    encoder = Encoder3D(input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointHead(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    ckpt_path = "./fuel_rl_checkpoints/bc_v2/best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"Model loaded from {ckpt_path}, device={DEVICE}")

    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    pts = generate_random_map_for_fuel(20, 20, 3.0, 15, seed=42)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    print(f"{'Step':>4} {'Type':>7} {'F_before':>8} {'F_after':>7} {'Disc':>6} {'Prog':>6} "
          f"{'VP_pos':>24} {'VP_yaw':>7} {'Valid':>5} {'Action':>24}")
    print("-" * 120)

    for step in range(15):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            print(f"\n探索完成! step={step}")
            break

        # 选最近的前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        target = frontiers[int(np.argmin(dists))]
        center = np.array(target.average)

        # Expert 视点
        expert_pos = np.array(target.best_viewpoint_pos)
        expert_yaw = target.best_viewpoint_yaw

        # Model 视点
        grid = build_grid_32x10(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = model(grid_t).cpu().numpy().flatten()
        max_dist = 4.0
        model_pos = center + action[:3] * max_dist
        model_yaw = action[3] * np.pi
        bmin_t, bmax_t = np.array(core.get_box()[0]), np.array(core.get_box()[1])
        model_pos[2] = np.clip(model_pos[2], 0.5, bmax_t[2] - 0.2)

        expert_valid = core.get_occupancy(expert_pos) == 1
        model_valid = core.get_occupancy(model_pos) == 1

        print(f"{step:4d} {'expert':>7} {len(frontiers):>8} ", end="")
        if expert_valid:
            before_unk = core.count_unknown_voxels()
            core.simulate_observation(expert_pos, expert_yaw)
            discovered = before_unk - core.count_unknown_voxels()
            new_f = core.detect_frontiers(expert_pos)
            agent_pos = expert_pos.copy()
            print(f"{len(new_f):>7} {discovered:>6} {core.get_exploration_progress():>5.1%} "
                  f"({expert_pos[0]:+.1f},{expert_pos[1]:+.1f},{expert_pos[2]:+.1f}) "
                  f"{expert_yaw:>+6.2f} {'OK':>5}")
        else:
            print(f"{'---':>7} {'---':>6} {'---':>5} "
                  f"({expert_pos[0]:+.1f},{expert_pos[1]:+.1f},{expert_pos[2]:+.1f}) "
                  f"{expert_yaw:>+6.2f} {'FAIL':>5}")

        frontiers2 = core.detect_frontiers(agent_pos)
        if not frontiers2:
            print("  探索完成 (after expert)")
            break

        # 再选最近的前沿给 model 测试
        dists2 = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers2]
        target2 = frontiers2[int(np.argmin(dists2))]
        center2 = np.array(target2.average)

        grid2 = build_grid_32x10(core, target2)
        grid2_t = torch.FloatTensor(grid2).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action2 = model(grid2_t).cpu().numpy().flatten()
        model_pos2 = center2 + action2[:3] * max_dist
        model_yaw2 = action2[3] * np.pi
        model_pos2[2] = np.clip(model_pos2[2], 0.5, bmax_t[2] - 0.2)

        model_valid2 = core.get_occupancy(model_pos2) == 1
        print(f"{'':>4} {'model':>7} {len(frontiers2):>8} ", end="")

        if model_valid2:
            before_unk = core.count_unknown_voxels()
            core.simulate_observation(model_pos2, model_yaw2)
            discovered = before_unk - core.count_unknown_voxels()
            new_f = core.detect_frontiers(model_pos2)
            print(f"{len(new_f):>7} {discovered:>6} {core.get_exploration_progress():>5.1%} "
                  f"({model_pos2[0]:+.1f},{model_pos2[1]:+.1f},{model_pos2[2]:+.1f}) "
                  f"{model_yaw2:>+6.2f} {'OK':>5} "
                  f"[{action2[0]:+.2f},{action2[1]:+.2f},{action2[2]:+.2f},{action2[3]:+.2f}]")
        else:
            print(f"{'---':>7} {'---':>6} {'---':>5} "
                  f"({model_pos2[0]:+.1f},{model_pos2[1]:+.1f},{model_pos2[2]:+.1f}) "
                  f"{model_yaw2:>+6.2f} {'FAIL':>5} "
                  f"[{action2[0]:+.2f},{action2[1]:+.2f},{action2[2]:+.2f},{action2[3]:+.2f}]")

    print(f"\n最终进度: {core.get_exploration_progress():.1%}")


if __name__ == "__main__":
    debug_model_vs_expert()
