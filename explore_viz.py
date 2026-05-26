"""可视化探索过程: 每步保存 2D 占据栅格 + 前沿 + 路径."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, os


def viz_step(core, agent_pos, agent_yaw, frontiers, planned_path, executed_path, step, save_dir, info=None):
    """保存单步可视化."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    origin = np.array(core.get_region()[0])
    size = np.array(core.get_region()[1])
    res = core.get_resolution()
    vn = np.array(core.get_map_voxel_num())

    # 2D 占据切片
    h = agent_pos[2]
    slice_2d = np.array(core.get_occupancy_slice_2d(h), dtype=np.int8)
    nx, ny = vn[0], vn[1]
    grid = slice_2d.reshape(nx, ny) if len(slice_2d) == nx * ny else np.zeros((nx, ny))

    # 颜色
    img = np.ones((nx, ny, 3)) * np.array([1.0, 0.99, 0.88])
    img[grid == 0] = [0.5, 0.5, 0.5]   # unknown: gray
    img[grid == 1] = [1.0, 1.0, 1.0]   # free: white
    img[grid == 2] = [0.0, 0.0, 1.0]   # occupied: blue

    extent = [origin[1], origin[1]+size[1], origin[0], origin[0]+size[0]]
    ax.imshow(img, origin='lower', extent=extent, aspect='equal')

    # 前沿 (彩色散点)
    if frontiers:
        for i, f in enumerate(frontiers):
            cells = np.array(f.cells)
            h_mask = np.abs(cells[:, 2] - h) < res * 3
            if np.any(h_mask):
                color = plt.cm.rainbow(i / max(len(frontiers), 1))
                ax.scatter(cells[h_mask, 1], cells[h_mask, 0], c=[color], s=1.5, zorder=3)

            # 最佳视点
            if f.best_viewpoint_visib_num > 0:
                vp = np.array(f.best_viewpoint_pos)
                ax.plot(vp[1], vp[0], 'o', color='lime', markersize=5, zorder=5, markeredgecolor='white')

    # 已执行路径 (蓝色)
    if executed_path and len(executed_path) > 1:
        pts = np.array(executed_path)
        ax.plot(pts[:, 1], pts[:, 0], '-', color='blue', linewidth=1.5, zorder=4)
        for p in executed_path:
            ax.plot(p[1], p[0], '.', color='blue', markersize=3, zorder=4)

    # 规划路径 (红色)
    if planned_path and len(planned_path) > 1:
        pts = np.array(planned_path)
        ax.plot(pts[:, 1], pts[:, 0], '-', color='red', linewidth=2, zorder=5)

    # UAV
    dx = 0.4 * np.cos(agent_yaw)
    dy = 0.4 * np.sin(agent_yaw)
    ax.annotate('', xy=(agent_pos[1]+dy, agent_pos[0]+dx), xytext=(agent_pos[1], agent_pos[0]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5), zorder=10)
    ax.plot(agent_pos[1], agent_pos[0], 'o', color='green', markersize=8, zorder=10, markeredgecolor='white')

    # 信息
    progress = core.get_exploration_progress()
    n_fr = len(frontiers) if frontiers else 0
    err = info.get('error', 'none') if info else 'none'
    disc = info.get('discovered', 0) if info else 0
    ax.set_title(f'Step {step} | Progress: {progress:.1%} | Frontiers: {n_fr} | '
                 f'Discovered: {disc} | {err}', fontsize=11)

    fig.savefig(os.path.join(save_dir, f'step_{step:04d}.png'), dpi=120, bbox_inches='tight')
    plt.close('all')


def run_exploration(args):
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载 v3 模型
    class V3Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv3d(3,32,3,stride=(2,2,1),padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.1),
                nn.Conv3d(32,64,3,stride=2,padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.1),
                nn.Conv3d(64,128,3,stride=(2,2,1),padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.1),
                nn.Conv3d(128,256,3,stride=2,padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.1))
            self.embedding = nn.Sequential(nn.Flatten(),nn.Linear(3072,512),nn.LayerNorm(512),nn.LeakyReLU(0.1),nn.Dropout(0.1))
            self.actor_net = nn.Sequential(nn.Linear(512,128),nn.LeakyReLU(0.1),nn.Linear(128,4),nn.Tanh())
            self.critic_net = nn.Sequential(nn.Linear(512,64),nn.LeakyReLU(0.1),nn.Linear(64,1))
        def forward(self, x):
            return self.actor_net(self.embedding(self.backbone(x)))

    model = V3Model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model.eval()
    print(f"Model loaded ({device})")

    core = FuelEnvCore()
    mp = default_map_params(size_x=20, size_y=20, size_z=3.0,
                            box_min=(-9,-9,0), box_max=(9,9,2.8))
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    pts = generate_random_map_for_fuel(20, 20, 3, args.num_pillars, seed=args.seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    agent_yaw = 0.0
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    executed = [agent_pos.copy()]
    planned = []
    visited = []

    # 初始状态可视化
    frontiers = core.detect_frontiers(agent_pos)
    viz_step(core, agent_pos, agent_yaw, frontiers, [], executed, -1, args.save_dir)

    resolution = 0.2
    max_action_dist = 3.2

    for step in range(args.max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)

        if not frontiers:
            viz_step(core, agent_pos, agent_yaw, [], [], executed, step, args.save_dir,
                     {'error': 'EXPLORATION COMPLETE'})
            print(f"Step {step}: DONE, progress={progress:.1%}")
            break

        # 选前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        new_f = [(i,f,d) for i,(f,d) in enumerate(zip(frontiers, dists))
                 if d>=0.3 and not any(np.linalg.norm(np.array(f.average)-v)<1.0 for v in visited[-30:])]
        if new_f:
            new_f.sort(key=lambda x: x[2])
            target = new_f[0][1]
        else:
            target = frontiers[int(np.argmax(dists))]

        # 构建体素
        center = np.array(target.average)
        nx, ny, nz = 32, 32, 10
        grid = np.zeros((3, nx, ny, nz), dtype=np.float32)
        for dz in range(nz):
            for dy in range(ny):
                for dx in range(nx):
                    wx = center[0]+(dx-16+0.5)*resolution
                    wy = center[1]+(dy-16+0.5)*resolution
                    wz = center[2]+(dz-5+0.5)*resolution
                    occ = core.get_occupancy(np.array([wx,wy,wz]))
                    if occ==2: grid[0,dx,dy,dz]=1
                    elif occ==1: grid[2,dx,dy,dz]=1
        cells = np.array(target.cells)
        if len(cells)>0:
            local=(cells-center)/resolution
            idx=np.stack([(local[:,0]+16).astype(int),(local[:,1]+16).astype(int),(local[:,2]+5).astype(int)],axis=1)
            v=((idx[:,0]>=0)&(idx[:,0]<32)&(idx[:,1]>=0)&(idx[:,1]<32)&(idx[:,2]>=0)&(idx[:,2]<10))
            idx=idx[v]
            if len(idx)>0: grid[1,idx[:,0],idx[:,1],idx[:,2]]=1

        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(grid_t).cpu().numpy().flatten()

        vp_pos = center + action[:3] * max_action_dist
        vp_yaw = action[3] * np.pi
        bmin, bmax = np.array(core.get_box()[0]), np.array(core.get_box()[1])
        vp_pos[2] = np.clip(vp_pos[2], 0.5, bmax[2]-0.2)

        info = {'discovered': 0, 'error': 'none'}
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            info['error'] = f'invalid(occ={occ})'
        else:
            prev = core.count_unknown_voxels()
            path = core.plan_path(agent_pos, vp_pos)
            if not path:
                info['error'] = 'no_path'
            else:
                planned = [np.array(p) for p in path]
                core.simulate_observation(vp_pos, vp_yaw)
                info['discovered'] = prev - core.count_unknown_voxels()
                agent_pos = vp_pos.copy()
                agent_yaw = vp_yaw
                executed.append(agent_pos.copy())
                visited.append(agent_pos.copy())

        viz_step(core, agent_pos, agent_yaw, frontiers, planned, executed, step, args.save_dir, info)
        planned = []

        if step % 3 == 0:
            print(f"Step {step:3d}: progress={progress:.1%} frontiers={len(frontiers)} discovered={info['discovered']} err={info['error']}")

    print(f"Images saved to {args.save_dir}/")
    print(f"View with: eog {args.save_dir}/step_*.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/media/jd3/系统/Users/jd3/Desktop/code/RL_Viewpoint/checkpoints/best_model.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-pillars", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="/tmp/fuel_rl_explore")
    args = p.parse_args()
    run_exploration(args)
