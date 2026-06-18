"""Windows: 读取快照 npz 文件, 用 Open3D 做 3D 交互式回放."""
import sys
import numpy as np
import open3d as o3d


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


def load_snapshots(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    n = 0
    while f's{n}_step' in data:
        n += 1
    print(f"Loaded {n} snapshots from {npz_path}")

    snapshots = []
    for i in range(n):
        prefix = f's{i}_'
        frt_count = int(data[prefix+'nfrt'])
        frontiers = []
        for j in range(frt_count):
            frt = data[prefix+f'frt{j}']
            frontiers.append(frt)
        target = data[prefix+'target']
        if target is not None and len(target) == 0:
            target = None
        snapshots.append({
            'step': int(data[prefix+'step']),
            'progress': float(data[prefix+'progress']),
            'agent_pos': data[prefix+'agent_pos'],
            'agent_yaw': float(data[prefix+'agent_yaw']),
            'path': data[prefix+'path'],
            'occ': data[prefix+'occ'],
            'free': data[prefix+'free'],
            'frontiers': frontiers,
            'target': target,
            'status': str(data[prefix+'status']),
        })
    return snapshots


def play_3d(snapshots):
    N = len(snapshots)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='FUEL 3D Exploration Replay', width=1400, height=800)
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
        print(f"\rSnapshot {idx+1}/{N} | Step {snap['step']} | cov={snap['progress']:.1%} | {snap['status']}", end='', flush=True)

        if len(snap['occ']) > 0:
            m, l = create_voxel_meshes(snap['occ'], color=[0.2, 0.4, 0.85], res=0.1)
            if m is not None:
                add_geom(m, first)
            if l is not None:
                add_geom(l, first)

        for frt in snap['frontiers']:
            if len(frt) > 200:
                frt = frt[np.random.choice(len(frt), 200, replace=False)]
            if len(frt) > 0:
                m_f, l_f = create_voxel_meshes(frt, color=[0.0, 1.0, 1.0], res=0.1)
                if m_f is not None:
                    add_geom(m_f, first)
                if l_f is not None:
                    add_geom(l_f, first)

        if len(snap['path']) >= 2:
            line = create_path_line(snap['path'], [0.3, 0.6, 1.0])
            if line is not None:
                add_geom(line, first)

        if snap['target'] is not None:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            marker.compute_vertex_normals()
            marker.paint_uniform_color([1.0, 1.0, 0.0])
            marker.translate(snap['target'])
            add_geom(marker, first)

        for g in create_agent(snap['agent_pos'], snap['agent_yaw'], [0.1, 1.0, 0.2]):
            add_geom(g, first)

    def next_snap(vis_ref):
        state['idx'] = min(state['idx'] + 1, N - 1)
        render(state['idx'])
        return False

    def prev_snap(vis_ref):
        state['idx'] = max(state['idx'] - 1, 0)
        render(state['idx'])
        return False

    def quit_vis(vis_ref):
        vis_ref.close()
        return False

    vis.register_key_callback(ord('D'), next_snap)
    vis.register_key_callback(ord('A'), prev_snap)
    vis.register_key_callback(262, next_snap)
    vis.register_key_callback(263, prev_snap)
    vis.register_key_callback(ord('Q'), quit_vis)

    print("\n=== Legend ===")
    print("  Green sphere+arrow: UAV agent")
    print("  Yellow sphere: selected frontier center")
    print("  Cyan cubes: frontier voxels")
    print("  Blue cubes: obstacles")
    print("  Blue line: flight path")
    print("\n=== Keys ===")
    print("  A/D or Left/Right: switch snapshot")
    print("  Q: quit")

    render(0)
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="snapshots.npz", help="Path to snapshots npz file")
    args = parser.parse_args()
    snapshots = load_snapshots(args.input)
    play_3d(snapshots)
