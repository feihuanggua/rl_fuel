"""3D 交互式可视化: FUEL 原生视点 + OrderPolicy 前沿排序."""
import sys, argparse
import numpy as np
import torch
import open3d as o3d

from fuel_rl import FuelEnvCore
from fuel_rl.config import (default_map_params, default_frontier_params,
                            fast_perception_params, default_astar_params, DEVICE)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.models.order_policy import OrderPolicy

# ── 渲染工具 ──

def create_voxel_meshes(grid_indices, color, res=0.2, box_scale=0.95):
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


def create_agent(pos, yaw, color, radius=0.3):
    geoms = []
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(pos)
    geoms.append(sphere)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.08, cone_radius=0.15, cylinder_height=0.6, cone_height=0.25)
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
    if len(path_pts) < 2:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(path_pts))
    ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path_pts)-1)])
    ls.paint_uniform_color(color)
    return ls


# ── 探索逻辑 ──

def run_exploration(seed=42, max_steps=150, model_path=None):
    # 加载策略
    if model_path:
        print(f"加载模型: {model_path}")
        model = OrderPolicy().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()
    else:
        model = None

    # 初始化环境
    core = FuelEnvCore()
    mp = default_map_params(size_x=20.0, size_y=20.0, size_z=3.0,
                            box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8))
    core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    agent_yaw = 0.0
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    path = [agent_pos.copy()]
    visited = set()
    snapshots = []
    consecutive_fails = 0
    total_astar_dist = 0.0

    def pos_key(avg):
        return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))

    def take_snapshot(step, progress, frontiers, target=None, status="ok"):
        occ_pts, free_pts = [], []
        step_size = 2
        for xi in range(0, 200, step_size):
            wx = -10.0 + xi * 0.1
            for yi in range(0, 200, step_size):
                wy = -10.0 + yi * 0.1
                for zi in range(0, 30, step_size):
                    wz = zi * 0.1
                    occ = core.get_occupancy(np.array([wx, wy, wz]))
                    if occ == 2: occ_pts.append([wx, wy, wz])
                    elif occ == 1: free_pts.append([wx, wy, wz])
        frt_pts = []
        if frontiers:
            for f in frontiers:
                cells = np.array(f.cells)
                if len(cells) > 0: frt_pts.append(cells)
        snapshots.append({
            'step': step, 'progress': progress,
            'agent_pos': agent_pos.copy(), 'agent_yaw': agent_yaw,
            'path': list(path), 'occ': np.array(occ_pts) if occ_pts else np.empty((0,3)),
            'free': np.array(free_pts) if free_pts else np.empty((0,3)),
            'frontiers': frt_pts,
            'target': np.array(target.average) if target is not None else None,
            'status': status,
        })

    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    take_snapshot(-1, progress, frontiers)

    print(f"\n{'Step':>4} {'Progress':>10} {'Frontiers':>10} {'Distance':>9} {'Status'}")
    print("-" * 55)

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            take_snapshot(step, progress, None, status="done")
            print(f"探索完成! {step}步, cov={progress:.1%}")
            break

        # 选前沿
        if model and len(frontiers) > 0:
            feats = np.zeros((50, 6), dtype=np.float32)
            mask = np.zeros(50, dtype=np.float32)
            vis_idx = 0
            visible_indices = []
            for i, f in enumerate(frontiers):
                if i >= 50: break
                c = np.array(f.average)
                key = pos_key(c)
                if key in visited: continue
                if vis_idx >= 50: break
                mask[vis_idx] = 1.0
                feats[vis_idx, 0:3] = c / 10.0
                feats[vis_idx, 3] = min(f.frontier_size / 2000.0, 1.0)
                feats[vis_idx, 4] = min(np.linalg.norm(c - agent_pos) / 15.0, 1.0)
                feats[vis_idx, 5] = min(f.best_viewpoint_visib_num / 100.0, 1.0)
                visible_indices.append(i)
                vis_idx += 1
            global_f = np.array([progress * 2 - 1, step / max_steps * 2 - 1], dtype=np.float32)

            ft = torch.FloatTensor(feats).unsqueeze(0).to(DEVICE)
            mt = torch.FloatTensor(mask).unsqueeze(0).to(DEVICE)
            gt = torch.FloatTensor(global_f).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(ft, mt, gt)
                probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
            n = vis_idx
            if n > 0:
                act = probs[:n].argmax()
                target = frontiers[visible_indices[act]]
            else:
                target = frontiers[0]
        else:
            target = min(frontiers, key=lambda f: np.linalg.norm(np.array(f.average) - agent_pos))

        # FUEL 原生视点
        vp = np.array(target.best_viewpoint_pos)
        vp_yaw = target.best_viewpoint_yaw
        vp[2] = np.clip(vp[2], 0.5, 2.6)

        occ = core.get_occupancy(vp)
        if occ != 1:
            consecutive_fails += 1
            visited.add(pos_key(target.average))
            status = f"INVALID#{consecutive_fails}"
            if consecutive_fails > 5:
                take_snapshot(step, progress, frontiers, target, status)
                break
            prev_frontiers = frontiers
            take_snapshot(step, progress, frontiers, target, status)
            continue

        consecutive_fails = 0
        astar_cost = core.compute_path_cost(agent_pos, vp)
        if astar_cost <= 0:
            astar_cost = np.linalg.norm(vp - agent_pos)
        total_astar_dist += astar_cost
        dist = astar_cost
        core.simulate_observation(vp, vp_yaw)
        agent_pos = vp.copy()
        agent_yaw = vp_yaw
        path.append(agent_pos.copy())

        take_snapshot(step, progress, frontiers, target, status="ok")
        if step % 5 == 0 or step < 3:
            print(f"{step:4d} {progress:>9.1%} {len(frontiers):>10} {dist:>8.1f}m  ok")

        prev_frontiers = frontiers
    else:
        progress = core.get_exploration_progress()
        take_snapshot(max_steps, progress, None, status="max_steps")

    final_cov = core.get_exploration_progress()
    print(f"\n结束: {len(path)-1}有效步, cov={final_cov:.1%}, astar总距离={total_astar_dist:.1f}m, {len(snapshots)} 快照")

    # ── 3D 回放 ──
    play_3d(snapshots)


def play_3d(snapshots):
    N = len(snapshots)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='FUEL 3D Exploration', width=1400, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.light_on = True
    opt.point_size = 2.0

    state = {'idx': 0, 'geoms': []}

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
        print(f"\r快照 {idx+1}/{N} | Step {snap['step']} | cov={snap['progress']:.1%} | {snap['status']}", end='', flush=True)

        if len(snap['occ']) > 0:
            m, l = create_voxel_meshes(snap['occ'], color=[0.2, 0.4, 0.85], res=0.1)
            add_geom(m, first); add_geom(l, first)

        for frt in snap['frontiers']:
            if len(frt) > 200:
                frt = frt[np.random.choice(len(frt), 200, replace=False)]
            if len(frt) > 0:
                m_f, l_f = create_voxel_meshes(frt, color=[0.0, 1.0, 1.0], res=0.1)
                if m_f is not None: add_geom(m_f, first); add_geom(l_f, first)

        if len(snap['path']) >= 2:
            line = create_path_line(snap['path'], [0.3, 0.6, 1.0])
            if line is not None: add_geom(line, first)

        if snap['target'] is not None:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            marker.compute_vertex_normals()
            marker.paint_uniform_color([1.0, 1.0, 0.0])
            marker.translate(snap['target'])
            add_geom(marker, first)

        for g in create_agent(snap['agent_pos'], snap['agent_yaw'], [0.1, 1.0, 0.2]):
            add_geom(g, first)

    def next_snap(vis_ref):
        state['idx'] = min(state['idx'] + 1, N - 1); render(state['idx']); return False
    def prev_snap(vis_ref):
        state['idx'] = max(state['idx'] - 1, 0); render(state['idx']); return False
    def quit_vis(vis_ref):
        vis_ref.close(); return False

    vis.register_key_callback(ord('D'), next_snap)
    vis.register_key_callback(ord('A'), prev_snap)
    vis.register_key_callback(262, next_snap)
    vis.register_key_callback(263, prev_snap)
    vis.register_key_callback(ord('Q'), quit_vis)

    print("\n=== 图例 ===")
    print("  绿色球+箭头: 无人机")
    print("  黄色球: 选中的前沿中心")
    print("  青色方块: 前沿体素")
    print("  蓝色方块: 障碍物")
    print("  蓝色线: 飞行路径")
    print("\n=== 按键 ===")
    print("  A/D 或 ←/→: 切换快照")
    print("  Q: 退出")

    render(0)
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="OrderPolicy checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()
    run_exploration(seed=args.seed, max_steps=args.max_steps, model_path=args.model)
