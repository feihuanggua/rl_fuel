"""3D 交互式前沿可视化测试 — 无需模型，直接查看新参数 + pinhole 相机效果."""
import sys
import numpy as np
import open3d as o3d

from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params, VOXEL_RES,
)
from fuel_rl.map_loader import generate_random_map_for_fuel

# ── 体素渲染 ──

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


def create_coordinate_axes(size=2.0):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])


# ── 主函数 ──

def main(seed=42):
    print(f"Seed={seed}, 初始化环境...")

    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    fp = default_frontier_params()
    pp = default_perception_params()
    ap = default_astar_params()

    core.init(mp, fp, pp, ap)

    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])

    # 四方向初始观测
    for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
        core.simulate_observation(agent_pos, yaw)

    # 检测前沿
    frontiers = core.detect_frontiers(agent_pos)
    print(f"检测到 {len(frontiers)} 个前沿")

    # 打印统计
    for i, f in enumerate(frontiers):
        cells = np.array(f.cells)
        avg = np.array(f.average)
        vp = np.array(f.best_viewpoint_pos)
        print(f"  F{i}: size={f.frontier_size:5d}, "
              f"visib={f.best_viewpoint_visib_num:3d}, "
              f"vp_dist={np.linalg.norm(vp-avg):.2f}, "
              f"avg={np.array2string(avg, precision=1)}")

    # ── 构建 3D 可视化 ──
    print("\n构建 3D 可视化...")

    # 采样地图体素 (障碍物高密度但分块查询，自由空间大幅降采样)
    occ_pts, free_pts = [], []
    res = VOXEL_RES  # 0.2m for visualization (matches frontier grid)
    step = int(res / 0.1)  # step=2 means 0.2m resolution
    for xi in range(0, 200, step):
        wx = -10.0 + xi * 0.1
        for yi in range(0, 200, step):
            wy = -10.0 + yi * 0.1
            for zi in range(0, 30, step):
                wz = zi * 0.1
                occ = core.get_occupancy(np.array([wx, wy, wz]))
                if occ == 2:
                    occ_pts.append([wx, wy, wz])
                elif occ == 1 and xi % 6 == 0 and yi % 6 == 0 and zi % 6 == 0:
                    free_pts.append([wx, wy, wz])

    occ_pts = np.array(occ_pts) if occ_pts else np.empty((0, 3))
    free_pts = np.array(free_pts) if free_pts else np.empty((0, 3))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='FUEL 3D Frontier Test (new params)', width=1400, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.12, 0.12, 0.14])
    opt.light_on = True
    opt.point_size = 2.0

    # 坐标轴
    vis.add_geometry(create_coordinate_axes(2.0), reset_bounding_box=True)

    # 障碍物 (蓝色)
    if len(occ_pts) > 0:
        # 降采样
        if len(occ_pts) > 3000:
            occ_pts = occ_pts[np.random.choice(len(occ_pts), 3000, replace=False)]
        m_obs, l_obs = create_voxel_meshes(occ_pts, color=[0.2, 0.4, 0.85], res=0.1)
        vis.add_geometry(m_obs)
        vis.add_geometry(l_obs)

    # 前沿 — 每个前沿不同颜色
    cmap = [
        [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
        [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        [0.8, 0.6, 0.2], [0.2, 0.8, 0.6], [0.8, 0.2, 0.6],
        [0.6, 0.8, 0.2], [0.2, 0.6, 0.8], [0.6, 0.2, 0.8],
        [0.5, 0.5, 0.5], [0.9, 0.5, 0.1], [0.1, 0.9, 0.5],
    ]
    for i, f in enumerate(frontiers):
        cells = np.array(f.cells)
        if len(cells) == 0:
            continue
        # 降采样前沿体素
        if len(cells) > 500:
            cells = cells[np.random.choice(len(cells), 500, replace=False)]
        color = cmap[i % len(cmap)]
        m_f, l_f = create_voxel_meshes(cells, color=color, res=0.1)
        if m_f is not None:
            vis.add_geometry(m_f)
            vis.add_geometry(l_f)

        # 前沿中心 (球)
        avg = np.array(f.average)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        sphere.translate(avg)
        vis.add_geometry(sphere)

        # 最佳视点 (小球)
        vp = np.array(f.best_viewpoint_pos)
        vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        vp_sphere.compute_vertex_normals()
        vp_sphere.paint_uniform_color([1.0, 0.5, 0.0])
        vp_sphere.translate(vp)
        vis.add_geometry(vp_sphere)

        # 中心到视点的连线
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.array([avg, vp]))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color(color)
        vis.add_geometry(line)

    # 无人机
    for g in create_agent(agent_pos, 0.0, color=[0.1, 1.0, 0.2]):
        vis.add_geometry(g)

    # 按键回调
    def toggle_obs(vis_ref):
        nonlocal m_obs
        if m_obs is not None:
            vis_ref.remove_geometry(m_obs, reset_bounding_box=False)
            m_obs = None
        else:
            m_obs, _ = create_voxel_meshes(occ_pts, color=[0.2, 0.4, 0.85], res=0.1)
            vis_ref.add_geometry(m_obs, reset_bounding_box=False)
        return False

    def quit_vis(vis_ref):
        vis_ref.close()
        return False

    vis.register_key_callback(ord('O'), toggle_obs)
    vis.register_key_callback(ord('Q'), quit_vis)

    print("\n=== 图例 ===")
    print("  彩色方块: 不同前沿簇 (每簇一种颜色)")
    print("  黄色球: 前沿中心")
    print("  橙色球: 最佳视点 (视点→中心的连线同色)")
    print("  蓝色方块: 障碍物")
    print("  绿色球+箭头: 无人机起始位置")
    print("\n=== 按键 ===")
    print("  O: 切换障碍物显示")
    print("  Q: 退出")
    print("  鼠标: 旋转/缩放/平移")
    print()

    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
