"""纯模型探索测试: 用 BC v1 (11³) 和 BC v2 (32x32x10) 分别跑完整探索."""
import numpy as np
import torch


def build_grid_11(core, frontier):
    """构建 [3, 11, 11, 11] 体素网格."""
    from fuel_rl.data.collector import build_3channel_grid
    return build_3channel_grid(core, frontier, 11)


def build_grid_32x10(core, frontier):
    """构建 [3, 32, 32, 10] 体素网格."""
    center = np.array(frontier.average)
    nx, ny, nz = 32, 32, 10
    res = 0.2
    grid = np.zeros((3, nx, ny, nz), dtype=np.float32)
    for dz in range(nz):
        for dy in range(ny):
            for dx in range(nx):
                wx = center[0] + (dx - 16 + 0.5) * res
                wy = center[1] + (dy - 16 + 0.5) * res
                wz = center[2] + (dz - 5 + 0.5) * res
                occ = core.get_occupancy(np.array([wx, wy, wz]))
                if occ == 2: grid[0, dx, dy, dz] = 1.0
                elif occ == 1: grid[2, dx, dy, dz] = 1.0
    cells = np.array(frontier.cells)
    if len(cells) > 0:
        local = (cells - center) / res
        idx = np.stack([(local[:,0]+16).astype(int),(local[:,1]+16).astype(int),(local[:,2]+5).astype(int)],axis=1)
        v = ((idx[:,0]>=0)&(idx[:,0]<32)&(idx[:,1]>=0)&(idx[:,1]<32)&(idx[:,2]>=0)&(idx[:,2]<10))
        idx = idx[v]
        if len(idx)>0: grid[1,idx[:,0],idx[:,1],idx[:,2]] = 1.0
    return grid


def run_exploration(model, build_fn, max_dist, label, seed=42, max_steps=50):
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel

    core = FuelEnvCore()
    mp = default_map_params(size_x=20, size_y=20, size_z=3.0, box_min=(-9,-9,0), box_max=(9,9,2.8))
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    bmin_t, bmax_t = np.array(core.get_box()[0]), np.array(core.get_box()[1])
    visited = []
    invalid_count = 0
    total_discovered = 0

    for step in range(max_steps):
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            prog = core.get_exploration_progress()
            print(f"  [{label}] Step {step}: COMPLETE, progress={prog:.1%}, "
                  f"invalid={invalid_count}, discovered={total_discovered}")
            return prog, step

        # 选最近的新前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i,f,d) for i,(f,d) in enumerate(zip(frontiers, dists))
                 if d>=0.3 and not any(np.linalg.norm(np.array(f.average)-v)<1.0 for v in visited[-30:])]
        if new_f:
            new_f.sort(key=lambda x: x[2])
            target = new_f[0][1]
        else:
            target = frontiers[int(np.argmin(dists))]

        center = np.array(target.average)
        grid = build_fn(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0)

        device = next(model.parameters()).device
        grid_t = grid_t.to(device)

        with torch.no_grad():
            action = model(grid_t).cpu().numpy().flatten()

        vp_pos = center + action[:3] * max_dist
        vp_yaw = action[3] * np.pi
        vp_pos[2] = np.clip(vp_pos[2], 0.5, bmax_t[2] - 0.2)

        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            invalid_count += 1
            continue

        before = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = before - core.count_unknown_voxels()
        total_discovered += discovered

        agent_pos = vp_pos.copy()
        visited.append(agent_pos.copy())

    prog = core.get_exploration_progress()
    print(f"  [{label}] Max steps reached: progress={prog:.1%}, "
          f"invalid={invalid_count}/{max_steps}, discovered={total_discovered}")
    return prog, max_steps


if __name__ == "__main__":
    from fuel_rl.models import ViewpointHead, Encoder3D
    from fuel_rl.config import ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE

    # BC v1 (11³)
    enc1 = Encoder3D(grid_size=11, channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    m1 = ViewpointHead(enc1, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    m1.load_state_dict(torch.load('./fuel_rl_checkpoints/bc/best_model.pth', map_location=DEVICE, weights_only=False))
    m1.eval()

    # BC v2 (32x32x10)
    enc2 = Encoder3D(input_shape=(32,32,10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    m2 = ViewpointHead(enc2, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    m2.load_state_dict(torch.load('./fuel_rl_checkpoints/bc_v2/best_model.pth', map_location=DEVICE, weights_only=False))
    m2.eval()

    print("=== BC v1 (11³, max_dist=4.0) ===")
    p1, s1 = run_exploration(m1, build_grid_11, 4.0, "BCv1", seed=42)

    print("\n=== BC v2 (32x32x10, max_dist=4.0) ===")
    p2, s2 = run_exploration(m2, build_grid_32x10, 4.0, "BCv2", seed=42)

    print(f"\n{'Model':<15} {'Progress':>10} {'Steps':>6}")
    print(f"{'BC v1 (11³)':<15} {p1:>9.1%} {s1:>6}")
    print(f"{'BC v2 (32³¹⁰)':<15} {p2:>9.1%} {s2:>6}")
