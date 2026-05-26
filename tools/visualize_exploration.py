"""可视化 PPO v2 best_model 探索过程 — 生成多步图片."""
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
    GRID_SIZE, GRID_Z, VOXEL_RES, ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE,
)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid
from fuel_rl.models import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointActorCritic


def visualize(seed=42, max_steps=200):
    # 加载模型
    encoder = Encoder3D(input_shape=(GRID_SIZE, GRID_SIZE, GRID_Z), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointActorCritic(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load('./fuel_rl_checkpoints/ppo_v2/best_model.pth', map_location=DEVICE, weights_only=False))
    model.eval()

    # 初始化环境
    core = FuelEnvCore()
    mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                            box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8))
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    out_dir = "/home/jd3/FUEL/rl_fuel/viz_frames"
    os.makedirs(out_dir, exist_ok=True)

    max_dist = 4.0
    visited = []
    path = [agent_pos.copy()]
    resolution = 0.1

    # 前沿访问计数：同一前沿探索3次后跳过
    from collections import defaultdict
    frontier_visits = defaultdict(int)
    MAX_VISIT = 2

    def pos_key(avg):
        return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        n_frontiers = len(frontiers)

        # --- 绘图 ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # 左图: XY 平面俯视图 (z=1.5m 切片)
        ax = axes[0]
        nx, ny = 200, 200
        occ_grid = np.zeros((nx, ny), dtype=np.int8)
        free_grid = np.zeros((nx, ny), dtype=np.float32)
        frontier_grid = np.zeros((nx, ny), dtype=np.float32)

        z_slice = 1.5
        for xi in range(0, nx, 2):
            for yi in range(0, ny, 2):
                wx = -10.0 + xi * resolution
                wy = -10.0 + yi * resolution
                occ = core.get_occupancy(np.array([wx, wy, z_slice]))
                if occ == 2:
                    occ_grid[xi, yi] = 1
                elif occ == 1:
                    free_grid[xi, yi] = 1

        # 前沿 cells 投影到 XY
        for f in frontiers:
            cells = np.array(f.cells)
            for c in cells:
                xi = int((c[0] + 10.0) / resolution)
                yi = int((c[1] + 10.0) / resolution)
                if 0 <= xi < nx and 0 <= yi < ny:
                    frontier_grid[xi, yi] = 1

        # RGB 合成
        rgb = np.ones((nx, ny, 3), dtype=np.float32) * 0.95  # 白色背景=未知
        rgb[free_grid > 0] = [0.85, 0.92, 0.85]  # 浅绿=自由
        rgb[occ_grid > 0] = [0.3, 0.3, 0.3]      # 深灰=障碍
        rgb[frontier_grid > 0] = [1.0, 0.2, 0.2]  # 红色=前沿

        ax.imshow(rgb.transpose(1, 0, 2), origin='lower', extent=[-10, 10, -10, 10])

        # 路径
        if len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, 'b-', linewidth=1.5, alpha=0.7)
            ax.plot(px, py, 'b.', markersize=2)

        # 当前位置
        ax.plot(agent_pos[0], agent_pos[1], 'ko', markersize=10)
        ax.plot(agent_pos[0], agent_pos[1], 'yo', markersize=6)

        # 前沿中心
        for f in frontiers:
            avg = np.array(f.average)
            ax.plot(avg[0], avg[1], 'r^', markersize=4, alpha=0.6)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step} | Coverage: {progress:.1%} | Frontiers: {n_frontiers}', fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # 右图: 3通道网格 (当前前沿)
        if not frontiers:
            axes[1].text(0.5, 0.5, 'No Frontiers!', ha='center', va='center', fontsize=20)
            axes[1].set_title('Frontier Grid (no frontier)')
            fig.savefig(f"{out_dir}/step_{step:03d}.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
            break

        # 选前沿: 跳过已访问 ≥3 次的，按距离排序
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        candidates = []
        for i, f in enumerate(frontiers):
            avg = np.array(f.average)
            dist = dists[i]
            key = pos_key(avg)
            visits = frontier_visits[key]
            visited_near = any(np.linalg.norm(avg - vp) < 1.0 for vp in path[-20:])
            candidates.append({
                "f": f, "dist": dist, "key": key,
                "visits": visits, "visited_near": visited_near,
                "skipped": visits >= MAX_VISIT,
            })

        # 优先选未跳过且未最近访问的，距离最近的
        active = [c for c in candidates if not c["skipped"]]
        if not active:
            active = [c for c in candidates if c["visits"] < MAX_VISIT + 3]
        if not active:
            active = candidates
        active.sort(key=lambda c: (c["visited_near"], c["dist"]))
        target = active[0]["f"]
        target_key = active[0]["key"]

        grid = build_3channel_grid(core, target)
        z_mid = GRID_Z // 2

        # 显示 Z 中间层切片
        ch_occ = grid[0, :, :, z_mid]
        ch_frt = grid[1, :, :, z_mid]
        ch_free = grid[2, :, :, z_mid]

        grid_rgb = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) * 0.9
        grid_rgb[ch_free > 0] = [0.7, 0.9, 0.7]
        grid_rgb[ch_occ > 0] = [0.2, 0.2, 0.2]
        grid_rgb[ch_frt > 0] = [1.0, 0.0, 0.0]

        axes[1].imshow(grid_rgb.transpose(1, 0, 2), origin='lower',
                       extent=[-3.2, 3.2, -3.2, 3.2])

        # 模型预测
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, std, val = model(grid_t)
            action = mean.cpu().numpy().flatten()

        center = np.array(target.average)
        vp_pos = center + action[:3] * max_dist
        vp_pos[2] = np.clip(vp_pos[2], 0.5, 2.6)

        # 标注预测视点
        axes[1].plot(0, 0, 'c+', markersize=15, markeredgewidth=2)  # 前沿中心
        axes[1].plot(action[0] * max_dist, action[1] * max_dist, 'y*', markersize=15)  # 预测偏移

        axes[1].set_title(f'Grid (z={z_slice:.1f}m) | Target: #{target.id} cells={target.frontier_size}\n'
                          f'Action: dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} yaw={action[3]:+.2f}',
                          fontsize=12)
        axes[1].set_xlabel('Local X (m)')
        axes[1].set_ylabel('Local Y (m)')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        fig.savefig(f"{out_dir}/step_{step:03d}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

        # 执行动作
        vp_yaw = action[3] * np.pi
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            frontier_visits[target_key] += 1
            skip_note = f" → SKIP({frontier_visits[target_key]}/{MAX_VISIT})" if frontier_visits[target_key] >= MAX_VISIT else ""
            print(f"Step {step}: INVALID{skip_note}")
            continue

        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = prev_unk - core.count_unknown_voxels()

        frontier_visits[target_key] += 1
        skip_note = ""
        if frontier_visits[target_key] >= MAX_VISIT:
            skip_note = f" → SKIP next({frontier_visits[target_key]} visits)"

        agent_pos = vp_pos.copy()
        path.append(agent_pos.copy())
        visited.append(agent_pos.copy())
        print(f"Step {step}: found={discovered} progress={progress:.1%} frontiers={n_frontiers} "
              f"action=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f},{action[3]:+.2f})"
              f"{skip_note}")

    progress = core.get_exploration_progress()
    print(f"\n最终: {len(path)-1}步, 覆盖率 {progress:.1%}")
    print(f"图片已保存到 {out_dir}/")


if __name__ == "__main__":
    visualize()
