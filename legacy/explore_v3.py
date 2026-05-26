"""用 v3.0 已训练模型在 FUEL C++ 核心上执行完整探索."""
import os
import sys
import numpy as np
import torch
import argparse


def build_voxel_grid(core, frontier, voxel_res=0.2, roi_xy=3.2, roi_z=1.0):
    """构建 [3, 32, 32, 10] 体素网格 (v3.0 格式)."""
    center = np.array(frontier.average, dtype=np.float64)
    nx, ny, nz = 32, 32, 10

    # 获取 C++ 核心的占据数据
    raw = np.array(core.get_local_voxel_grid(center.tolist(), 11), dtype=np.int8)
    # 11³ 分辨率 0.1m，需要转换为 32x32x10 分辨率 0.2m
    # 用 map 的整体信息构建
    resolution = 0.2
    ch_occ = np.zeros((nx, ny, nz), dtype=np.float32)
    ch_frontier = np.zeros((nx, ny, nz), dtype=np.float32)
    ch_free = np.zeros((nx, ny, nz), dtype=np.float32)

    # 从 SDF map 获取占据信息
    cells = np.array(frontier.cells)

    # 遍历所有前沿 cells 以及周围的占据/自由格子
    # 用 C++ 核心的 get_occupancy 查询
    for dz_i in range(nz):
        for dy_i in range(ny):
            for dx_i in range(nx):
                wx = center[0] + (dx_i - nx/2.0 + 0.5) * resolution
                wy = center[1] + (dy_i - ny/2.0 + 0.5) * resolution
                wz = center[2] + (dz_i - nz/2.0 + 0.5) * resolution

                occ = core.get_occupancy(np.array([wx, wy, wz]))
                if occ == 2:  # occupied
                    ch_occ[dx_i, dy_i, dz_i] = 1.0
                elif occ == 1:  # free
                    ch_free[dx_i, dy_i, dz_i] = 1.0

    # 填充前沿通道
    if len(cells) > 0:
        local = (cells - center) / resolution
        idx = np.stack([
            (local[:, 0] + nx/2.0).astype(int),
            (local[:, 1] + ny/2.0).astype(int),
            (local[:, 2] + nz/2.0).astype(int),
        ], axis=1)
        valid = ((idx[:, 0] >= 0) & (idx[:, 0] < nx) &
                 (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
                 (idx[:, 2] >= 0) & (idx[:, 2] < nz))
        idx = idx[valid]
        ch_frontier[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    return np.stack([ch_occ, ch_frontier, ch_free], axis=0)


def load_v3_model(model_path):
    """加载 v3.0 ExplorationActorCritic 模型."""
    from fuel_rl.models.encoder import Encoder3D
    import torch.nn as nn

    class V3Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv3d(3, 32, 3, stride=(2,2,1), padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.1),
                nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.1),
                nn.Conv3d(64, 128, 3, stride=(2,2,1), padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.1),
                nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.1),
            )
            self.embedding = nn.Sequential(
                nn.Flatten(), nn.Linear(3072, 512), nn.LayerNorm(512), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            )
            self.actor_net = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(0.1), nn.Linear(128, 4), nn.Tanh())
            self.critic_net = nn.Sequential(nn.Linear(512, 64), nn.LeakyReLU(0.1), nn.Linear(64, 1))

        def forward(self, x):
            x = self.backbone(x)
            x = self.embedding(x)
            return self.actor_net(x), x  # also return embedding

    model = V3Model()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model


def explore(args):
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_v3_model(args.model).to(device)
    print(f"Model loaded from {args.model} (device={device})")

    core = FuelEnvCore()
    mp = default_map_params(
        size_x=args.map_size, size_y=args.map_size, size_z=3.0,
        box_min=(-args.map_size/2+1, -args.map_size/2+1, 0.0),
        box_max=(args.map_size/2-1, args.map_size/2-1, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    if args.map_path:
        core.load_map_from_pcd(args.map_path)
    else:
        pts = generate_random_map_for_fuel(args.map_size, args.map_size, 3.0, args.num_pillars, seed=args.seed)
        core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    visited = []
    path = [agent_pos.copy()]
    max_action_dist = 3.2  # ROI_XY

    print(f"\n{'Step':>4} {'Discovered':>10} {'Progress':>10} {'Frontiers':>10} {'Info'}")
    print("-" * 60)

    for step in range(args.max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)

        if not frontiers:
            print(f"\n探索完成! {step} 步, 覆盖率 {progress:.1%}")
            break

        # 选前沿: 新的优先，否则最远的
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i, f, d) for i, (f, d) in enumerate(zip(frontiers, dists))
                 if d >= 0.3 and not any(np.linalg.norm(np.array(f.average)-v) < 1.0 for v in visited[-30:])]
        if new_f:
            new_f.sort(key=lambda x: x[2])
            target = new_f[0][1]
        else:
            idx = int(np.argmax(dists))
            target = frontiers[idx]

        # 构建输入
        grid = build_voxel_grid(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(device)

        with torch.no_grad():
            action, emb = model(grid_t)
            action = action.cpu().numpy().flatten()
            emb = emb.cpu().numpy().flatten()

        # 解码
        center = np.array(target.average)
        vp_pos = center + action[:3] * max_action_dist
        vp_yaw = action[3] * np.pi

        # 约束
        bmin, bmax = np.array(core.get_box()[0]), np.array(core.get_box()[1])
        vp_pos[2] = np.clip(vp_pos[2], 0.5, bmax[2] - 0.2)

        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            if step % 10 == 0:
                print(f"{step:4d} {'---':>10} {progress:>9.1%} {len(frontiers):>10} INVALID")
            continue

        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = prev_unk - core.count_unknown_voxels()

        agent_pos = vp_pos.copy()
        path.append(agent_pos.copy())
        visited.append(agent_pos.copy())

        if step % 5 == 0 or step < 3:
            emb_norm = np.linalg.norm(emb)
            print(f"{step:4d} {discovered:>10.0f} {progress:>9.1%} {len(frontiers):>10} "
                  f"a=[{action[0]:+.2f},{action[1]:+.2f}] emb_norm={emb_norm:.2f}")
    else:
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        print(f"\n达到最大步数 {args.max_steps}, 覆盖率 {progress:.1%}, 前沿 {len(frontiers)}")

    print(f"总步数: {len(path)-1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/media/jd3/系统/Users/jd3/Desktop/code/RL_Viewpoint/checkpoints/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pillars", type=int, default=10)
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    explore(args)
