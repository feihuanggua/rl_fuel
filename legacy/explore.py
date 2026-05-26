"""用训练好的模型在单张地图上执行完整探索."""
import os
import numpy as np
import torch
import argparse


def explore(args):
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel
    from fuel_rl.data.collector import build_3channel_grid
    from fuel_rl.models import ViewpointHead, Encoder3D
    from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE

    # 加载模型
    encoder = Encoder3D(input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointHead(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"Model loaded from {args.model}")

    # 初始化环境
    core = FuelEnvCore()
    mp = default_map_params(
        size_x=args.map_size, size_y=args.map_size, size_z=3.0,
        box_min=(-args.map_size/2+1, -args.map_size/2+1, 0.0),
        box_max=(args.map_size/2-1, args.map_size/2-1, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    # 加载地图
    if args.map_path:
        core.load_map_from_pcd(args.map_path)
    else:
        pts = generate_random_map_for_fuel(args.map_size, args.map_size, 3.0, args.num_pillars, seed=args.seed)
        core.load_map_from_points(pts)
    core.reset_map()

    # 初始观测
    agent_pos = np.array([0.0, 0.0, 1.5])
    agent_yaw = 0.0
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    max_dist = 4.0
    total_reward = 0
    path = [agent_pos.copy()]
    visited_positions = []

    print(f"\n{'Step':>4} {'Reward':>8} {'Progress':>10} {'Frontiers':>10} {'Action':>30} {'Info'}")
    print("-" * 90)

    for step in range(args.max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)

        if not frontiers:
            print(f"\n探索完成! {step} 步, 覆盖率 {progress:.1%}")
            break

        # 选择前沿: 优先选新的，实在没有就选最远的
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]

        new_frontiers = []
        old_frontiers = []
        for i, f in enumerate(frontiers):
            fpos = np.array(f.average)
            visited_near = any(np.linalg.norm(fpos - vp) < 1.0 for vp in visited_positions[-20:])
            if not visited_near and dists[i] >= 0.3:
                new_frontiers.append((i, f, dists[i]))
            else:
                old_frontiers.append((i, f, dists[i]))

        if new_frontiers:
            new_frontiers.sort(key=lambda x: x[2])
            target = new_frontiers[0][1]
        elif old_frontiers:
            old_frontiers.sort(key=lambda x: -x[2])
            target = old_frontiers[0][1]
        else:
            break

        # 构建输入 (32x32x10)
        grid = build_3channel_grid(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)

        # 模型推理
        with torch.no_grad():
            action = model(grid_t).cpu().numpy().flatten()

        # 解码动作
        center = np.array(target.average)
        vp_pos = center + action[:3] * max_dist
        vp_yaw = action[3] * np.pi

        # 边界约束
        bmin_t, bmax_t = np.array(core.get_box()[0]), np.array(core.get_box()[1])
        vp_pos[2] = np.clip(vp_pos[2], 0.5, bmax_t[2] - 0.2)

        # 检查有效性
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            print(f"{step:4d} {'---':>8} {progress:>9.1%} {len(frontiers):>10} "
                  f"INVALID(pos occ={occ})")
            continue

        path_cost = core.compute_path_cost(agent_pos, vp_pos)
        if path_cost < 0 or path_cost > 100:
            print(f"{step:4d} {'---':>8} {progress:>9.1%} {len(frontiers):>10} "
                  f"NO PATH (cost={path_cost:.1f})")
            continue

        # 执行
        prev_unknown = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        new_unknown = core.count_unknown_voxels()
        discovered = prev_unknown - new_unknown

        agent_pos = vp_pos.copy()
        agent_yaw = vp_yaw
        path.append(agent_pos.copy())
        visited_positions.append(agent_pos.copy())
        total_reward += discovered

        visible = core.count_visible_cells(vp_pos, vp_yaw, target.cells)
        coverage = visible / max(target.frontier_size, 1)

        if step % 5 == 0 or step < 3:
            print(f"{step:4d} {discovered:>8.0f} {progress:>9.1%} {len(frontiers):>10} "
                  f"dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} yaw={action[3]:+.2f} "
                  f"cov={coverage:.0%} dist={dists[int(np.argmin(dists))]:.1f}m")

    else:
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        print(f"\n达到最大步数 {args.max_steps}, 覆盖率 {progress:.1%}, 剩余前沿 {len(frontiers)}")

    print(f"\n总探索: {len(path)-1} 步, 发现 {total_reward} 个体素, 路径点 {len(path)}")

    # 保存路径
    if args.save_path:
        np.save(args.save_path, np.array(path))
        print(f"路径已保存到 {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./fuel_rl_checkpoints/bc_v2/best_model.pth")
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pillars", type=int, default=15)
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()
    explore(args)
