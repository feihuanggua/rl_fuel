"""Open3D 三维交互式探索可视化 — 基于v3.0 visualize_o3d方案."""
import numpy as np
import torch
import open3d as o3d
from collections import defaultdict

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


# ── 体素渲染工具 (复用v3.0方案) ──

def create_voxel_meshes(grid_indices, color, res=0.2, box_scale=0.95):
    """批量构建体素实体网格 + 线框."""
    if len(grid_indices) == 0:
        return None, None
    N = len(grid_indices)
    d = res * box_scale / 2.0
    base_v = np.array([[-d,-d,-d],[d,-d,-d],[d,d,-d],[-d,d,-d],
                        [-d,-d,d],[d,-d,d],[d,d,d],[-d,d,d]])
    base_tri = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                          [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                          [0,4,7],[0,7,3],[1,2,6],[1,6,5]])
    base_line = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                           [0,4],[1,5],[2,6],[3,7]])
    all_v = (base_v[None,:,:] + grid_indices[:,None,:]).reshape(-1,3)
    off = np.arange(N) * 8
    all_tri = (base_tri[None,:,:] + off[:,None,None]).reshape(-1,3)
    all_ln = (base_line[None,:,:] + off[:,None,None]).reshape(-1,2)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_v)
    mesh.triangles = o3d.utility.Vector3iVector(all_tri)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_v)
    ls.lines = o3d.utility.Vector2iVector(all_ln)
    ls.paint_uniform_color([0.15, 0.15, 0.15])
    return mesh, ls


def create_fog_cloud(grid_indices, color):
    """自由空间点云."""
    if len(grid_indices) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_indices.astype(float))
    pcd.paint_uniform_color(color)
    return pcd


def create_agent(pos, yaw, color, radius=0.3):
    """创建无人机: 球体 + 朝向箭头."""
    geoms = []
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(pos)
    geoms.append(sphere)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.08, cone_radius=0.15,
        cylinder_height=0.6, cone_height=0.25,
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)
    R_fix = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
    arrow.rotate(R_fix, center=[0,0,0])
    R_yaw = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
    arrow.rotate(R_yaw, center=[0,0,0])
    arrow.translate(pos)
    geoms.append(arrow)
    return geoms


def create_path_line(path_pts, color=[0.2, 0.6, 1.0]):
    """路径线."""
    if len(path_pts) < 2:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(path_pts))
    lines = [[i, i+1] for i in range(len(path_pts)-1)]
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


# ── 主探索逻辑 ──

def run_exploration(seed=42, max_steps=150, model_path='./fuel_rl_checkpoints/sac_seq2/actor_1000.pth'):
    """先跑探索，记录每步快照，再交互回放."""
    print(f"加载模型: {model_path}")
    encoder = Encoder3D(input_shape=(GRID_SIZE, GRID_SIZE, GRID_Z),
                        channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointActorCritic(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False), strict=False)
    model.eval()

    core = FuelEnvCore()
    mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                            box_min=(-9,-9,0.0), box_max=(9,9,2.8))
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    agent_yaw = 0.0
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    max_dist = 4.0
    res = VOXEL_RES
    path = [agent_pos.copy()]
    frontier_visits = defaultdict(int)
    MAX_VISIT = 2

    def pos_key(avg):
        return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))

    # 收集快照
    snapshots = []

    def take_snapshot(step, progress, frontiers, target=None, action=None, discovered=0, status="ok"):
        """采集当前全局地图体素 + 前沿 + 智能体状态."""
        # 采样地图体素（降采样以提高性能）
        occ_pts, free_pts = [], []
        step_size = 2
        for xi in range(0, 200, step_size):
            for yi in range(0, 200, step_size):
                wx = -10.0 + xi * 0.1
                wy = -10.0 + yi * 0.1
                for zi in range(0, 30, step_size):
                    wz = zi * 0.1
                    occ = core.get_occupancy(np.array([wx, wy, wz]))
                    if occ == 2:
                        occ_pts.append([wx, wy, wz])
                    elif occ == 1:
                        free_pts.append([wx, wy, wz])

        frt_pts = []
        if frontiers:
            for f in frontiers:
                cells = np.array(f.cells)
                if len(cells) > 0:
                    frt_pts.append(cells)

        snapshots.append({
            'step': step, 'progress': progress, 'n_frontiers': len(frontiers) if frontiers else 0,
            'agent_pos': agent_pos.copy(), 'agent_yaw': agent_yaw,
            'path': list(path),
            'occ': np.array(occ_pts) if occ_pts else np.empty((0,3)),
            'free': np.array(free_pts) if free_pts else np.empty((0,3)),
            'frontiers': frt_pts,
            'target': np.array(target.average) if target is not None else None,
            'action': action.copy() if action is not None else None,
            'discovered': discovered, 'status': status,
        })

    # 初始快照
    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    take_snapshot(-1, progress, frontiers)

    print(f"\n{'Step':>4} {'Reward':>8} {'Progress':>10} {'Frontiers':>10} {'Status'}")
    print("-" * 60)

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)

        if not frontiers:
            take_snapshot(step, progress, None, status="done")
            print(f"\n探索完成! {step}步, 覆盖率 {progress:.1%}")
            break

        # 选前沿
        dists = [np.linalg.norm(np.array(f.average) - agent_pos) for f in frontiers]
        candidates = []
        for i, f in enumerate(frontiers):
            avg = np.array(f.average)
            dist = dists[i]
            key = pos_key(avg)
            candidates.append({
                "f": f, "dist": dist, "key": key,
                "visits": frontier_visits[key],
                "skipped": frontier_visits[key] >= MAX_VISIT,
                "visited_near": any(np.linalg.norm(avg - p) < 1.0 for p in path[-20:]),
            })

        active = [c for c in candidates if not c["skipped"]]
        if not active:
            active = [c for c in candidates if c["visits"] < MAX_VISIT + 3]
        if not active:
            active = candidates
        active.sort(key=lambda c: (c["visited_near"], c["dist"]))
        target = active[0]["f"]
        target_key = active[0]["key"]

        # 模型推理
        grid = build_3channel_grid(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, std, val = model(grid_t)
            action = mean.cpu().numpy().flatten()

        center = np.array(target.average)
        vp_pos = center + action[:3] * max_dist
        vp_pos[2] = np.clip(vp_pos[2], 0.5, 2.6)
        vp_yaw = action[3] * np.pi

        # 执行
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            frontier_visits[target_key] += 1
            take_snapshot(step, progress, frontiers, target, action, status="invalid")
            print(f"{step:4d} {'---':>8} {progress:>9.1%} {len(frontiers):>10} INVALID")
            continue

        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        discovered = prev_unk - core.count_unknown_voxels()

        frontier_visits[target_key] += 1
        skip = frontier_visits[target_key] >= MAX_VISIT

        agent_pos = vp_pos.copy()
        agent_yaw = vp_yaw
        path.append(agent_pos.copy())

        take_snapshot(step, progress, frontiers, target, action, discovered,
                      status="skip" if skip else "ok")
        print(f"{step:4d} {discovered:>8.0f} {progress:>9.1%} {len(frontiers):>10} "
              f"{'SKIP' if skip else 'ok'}")
    else:
        progress = core.get_exploration_progress()
        take_snapshot(max_steps, progress, None, status="max_steps")

    print(f"\n探索结束: {len(path)-1}步, 覆盖率 {core.get_exploration_progress():.1%}")
    print(f"共 {len(snapshots)} 个快照，启动 3D 回放...\n")

    # ── Open3D 交互回放 ──
    play_3d(snapshots, res)


def play_3d(snapshots, res):
    """Open3D 交互式回放探索过程."""
    N = len(snapshots)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='FUEL 3D Exploration', width=1400, height=800)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.light_on = True
    opt.point_size = 2.0

    state = {'idx': 0, 'geoms': [], 'show_free': True, 'auto_play': False}

    def clear_geoms():
        for g in state['geoms']:
            vis.remove_geometry(g, reset_bounding_box=False)
        state['geoms'] = []

    def add_geom(g, reset=False):
        vis.add_geometry(g, reset_bounding_box=reset)
        state['geoms'].append(g)

    def render(idx):
        clear_geoms()
        snap = snapshots[idx]
        first = (idx == 0)

        # 标题
        vis.get_render_option()  # refresh
        print(f"\r快照 {idx+1}/{N} | Step {snap['step']} | "
              f"覆盖 {snap['progress']:.1%} | 前沿 {snap['n_frontiers']} | {snap['status']}", end='', flush=True)

        # 障碍物 (蓝色)
        if len(snap['occ']) > 0:
            m, l = create_voxel_meshes(snap['occ'], color=[0.2, 0.4, 0.85], res=res)
            add_geom(m, first)
            add_geom(l, first)

        # 自由空间 (绿色点云)
        if state['show_free'] and len(snap['free']) > 0:
            # 降采样
            free = snap['free']
            if len(free) > 5000:
                idx_sample = np.random.choice(len(free), 5000, replace=False)
                free = free[idx_sample]
            pcd = create_fog_cloud(free, color=[0.5, 0.85, 0.5])
            add_geom(pcd, first)

        # 前沿 (青色)
        for frt_cells in snap['frontiers']:
            if len(frt_cells) > 200:
                idx_s = np.random.choice(len(frt_cells), 200, replace=False)
                frt_cells = frt_cells[idx_s]
            if len(frt_cells) > 0:
                m_f, l_f = create_voxel_meshes(frt_cells, color=[0.0, 1.0, 1.0], res=res)
                if m_f is not None:
                    add_geom(m_f, first)
                    add_geom(l_f, first)

        # 路径 (蓝色线)
        if len(snap['path']) >= 2:
            line = create_path_line(snap['path'], color=[0.3, 0.6, 1.0])
            if line is not None:
                add_geom(line, first)

        # 目标前沿标记 (黄色球)
        if snap['target'] is not None:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            marker.compute_vertex_normals()
            marker.paint_uniform_color([1.0, 1.0, 0.0])
            marker.translate(snap['target'])
            add_geom(marker, first)

        # 无人机 (绿色)
        agent_geoms = create_agent(snap['agent_pos'], snap['agent_yaw'], color=[0.1, 1.0, 0.2])
        for g in agent_geoms:
            add_geom(g, first)

        # 预测视点 (如果有的话，紫红色)
        if snap['action'] is not None and snap['target'] is not None:
            vp = snap['target'] + snap['action'][:3] * 4.0
            vp[2] = np.clip(vp[2], 0.5, 2.6)
            vp_yaw = snap['action'][3] * np.pi
            vp_geoms = create_agent(vp, vp_yaw, color=[1.0, 0.0, 1.0], radius=0.2)
            for g in vp_geoms:
                add_geom(g, first)

    # 键盘回调
    def next_snap(vis_ref):
        state['idx'] = min(state['idx'] + 1, N - 1)
        render(state['idx'])
        return False

    def prev_snap(vis_ref):
        state['idx'] = max(state['idx'] - 1, 0)
        render(state['idx'])
        return False

    def jump_start(vis_ref):
        state['idx'] = 0
        render(state['idx'])
        return False

    def jump_end(vis_ref):
        state['idx'] = N - 1
        render(state['idx'])
        return False

    def toggle_free(vis_ref):
        state['show_free'] = not state['show_free']
        render(state['idx'])
        return False

    def quit_vis(vis_ref):
        vis_ref.close()
        return False

    vis.register_key_callback(ord('D'), next_snap)
    vis.register_key_callback(ord('A'), prev_snap)
    vis.register_key_callback(262, next_snap)   # Right arrow
    vis.register_key_callback(263, prev_snap)    # Left arrow
    vis.register_key_callback(ord('H'), jump_start)
    vis.register_key_callback(ord('E'), jump_end)
    vis.register_key_callback(ord('F'), toggle_free)
    vis.register_key_callback(ord('Q'), quit_vis)

    print("=== 图例 ===")
    print("  绿色球+箭头: 无人机位置和朝向")
    print("  紫色球+箭头: 模型预测视点")
    print("  黄色球: 目标前沿中心")
    print("  青色方块: 前沿体素")
    print("  蓝色方块: 障碍物")
    print("  蓝色线: 飞行路径")
    print("  绿色点云: 已探索自由空间")
    print("")
    print("=== 按键 ===")
    print("  A/D 或 ←/→: 前后切换快照")
    print("  H/E: 跳到开头/末尾")
    print("  F: 切换自由空间显示")
    print("  Q: 退出")

    render(0)
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./fuel_rl_checkpoints/sac_seq2/actor_4000.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()
    run_exploration(seed=args.seed, max_steps=args.max_steps, model_path=args.model)
