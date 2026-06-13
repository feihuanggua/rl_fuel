"""Map loading and random generation utilities."""
import numpy as np
import os
from typing import List, Optional, Tuple


def generate_random_map(
    x_range: Tuple[float, float] = (-10.0, 10.0),
    y_range: Tuple[float, float] = (-10.0, 10.0),
    z_range: Tuple[float, float] = (0.0, 2.5),
    num_pillars: int = 15,
    pillar_width_range: Tuple[float, float] = (0.3, 0.8),
    pillar_height_range: Tuple[float, float] = (1.5, 2.8),
    resolution: float = 0.1,
    seed: Optional[int] = None,
    add_ground: bool = True,
    ground_z: float = -0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obstacles = []
    for _ in range(num_pillars):
        cx = rng.uniform(*x_range)
        cy = rng.uniform(*y_range)
        w = rng.uniform(*pillar_width_range)
        h = rng.uniform(*pillar_height_range)
        h = min(h, z_range[1])
        for dx in np.arange(-w / 2, w / 2, resolution):
            for dy in np.arange(-w / 2, w / 2, resolution):
                for dz in np.arange(0, h, resolution):
                    obstacles.append([cx + dx, cy + dy, z_range[0] + dz])
    if add_ground:
        for x in np.arange(x_range[0], x_range[1], resolution * 3):
            for y in np.arange(y_range[0], y_range[1], resolution * 3):
                obstacles.append([x, y, ground_z])
    return np.array(obstacles) if obstacles else np.zeros((0, 3))


def generate_random_map_for_fuel(
    map_size_x: float = 20.0,
    map_size_y: float = 20.0,
    map_size_z: float = 3.0,
    num_pillars: int = 15,
    resolution: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    half_x = map_size_x / 2.0 - 1.0
    half_y = map_size_y / 2.0 - 1.0
    return generate_random_map(
        x_range=(-half_x, half_x),
        y_range=(-half_y, half_y),
        z_range=(0.0, map_size_z),
        num_pillars=num_pillars,
        resolution=resolution,
        seed=seed,
        add_ground=True,
        ground_z=-0.1,
    )


def pcd_file_list() -> List[str]:
    resource_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uav_simulator", "map_generator", "resource"
    )
    if os.path.isdir(resource_dir):
        pcds = [os.path.join(resource_dir, f) for f in os.listdir(resource_dir) if f.endswith(".pcd")]
        return sorted(pcds)
    return []


def load_ariadne_map(
    png_path: str,
    resolution: float = 0.1,
    wall_height: float = 1.8,
    **kwargs,
) -> Tuple[np.ndarray, float, float]:
    """Load ARiADNE PNG at full 0.1m resolution, trim whitespace borders,
    fill ALL occupied cells as solid walls.

    Returns (points Nx3, map_width_meters, map_height_meters).
    """
    try:
        from skimage import io
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")

    img = io.imread(png_path, as_gray=True)
    gt = (img * 255).astype(int)

    free_mask = (gt > 150) | ((gt <= 80) & (gt >= 50))

    rows_free = np.any(free_mask, axis=1)
    cols_free = np.any(free_mask, axis=0)
    rmin = int(np.argmax(rows_free))
    rmax = len(rows_free) - 1 - int(np.argmax(rows_free[::-1]))
    cmin = int(np.argmax(cols_free))
    cmax = len(cols_free) - 1 - int(np.argmax(cols_free[::-1]))

    # Cap crop to 40m (400 pixels at 0.1m), centered on robot start
    max_crop = 400
    robot_cells = np.nonzero(gt[rmin:rmax + 1, cmin:cmax + 1] == 208)
    if len(robot_cells[0]) > 0:
        rc_r = robot_cells[0][0]
        rc_c = robot_cells[1][0]
    else:
        rc_r = (rmax - rmin) // 2
        rc_c = (cmax - cmin) // 2

    crop_h_raw = rmax - rmin + 1
    crop_w_raw = cmax - cmin + 1
    if crop_w_raw > max_crop:
        new_cmin = cmin + max(0, rc_c - max_crop // 2)
        new_cmax = min(cmax + 1, new_cmin + max_crop)
        cmin = new_cmin
        cmax = new_cmax - 1
    if crop_h_raw > max_crop:
        new_rmin = rmin + max(0, rc_r - max_crop // 2)
        new_rmax = min(rmax + 1, new_rmin + max_crop)
        rmin = new_rmin
        rmax = new_rmax - 1

    crop_h = rmax - rmin + 1
    crop_w = cmax - cmin + 1
    map_w = crop_w * resolution
    map_h = crop_h * resolution

    cropped_free = free_mask[rmin:rmax + 1, cmin:cmax + 1]
    occ_mask = ~cropped_free

    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    z_levels = np.arange(0, wall_height, resolution)
    n_z = len(z_levels)

    # --- 1. ALL occupied cells extruded in z (vectorized) ---
    occ_ys, occ_xs = np.where(occ_mask)
    n_occ = len(occ_xs)

    wall_xs = np.tile(occ_xs.astype(np.float64), n_z)
    wall_ys = np.tile(occ_ys.astype(np.float64), n_z)
    wall_zs = np.repeat(z_levels, n_occ)
    wall_pts = np.column_stack([
        (wall_xs - half_w) * resolution,
        (wall_ys - half_h) * resolution,
        wall_zs,
    ])

    # --- 2. Thick perimeter walls (vectorized) ---
    perim_thick = 3
    z_perim = z_levels[::3]
    n_zp = len(z_perim)

    perim_px = []
    perim_py = []
    for t in range(perim_thick):
        xs_edge = np.arange(0, crop_w, 2)
        ys_edge = np.arange(0, crop_h, 2)
        perim_px.extend(list(xs_edge) + list(xs_edge) + [t] * len(ys_edge) + [crop_w - 1 - t] * len(ys_edge))
        perim_py.extend([crop_h - 1 - t] * len(xs_edge) + [t] * len(xs_edge) + list(ys_edge) + list(ys_edge))

    perim_px = np.array(perim_px, dtype=np.float64)
    perim_py = np.array(perim_py, dtype=np.float64)
    n_perim = len(perim_px)

    perim_xs = np.tile(perim_px, n_zp)
    perim_ys = np.tile(perim_py, n_zp)
    perim_zs = np.repeat(z_perim, n_perim)
    perim_pts = np.column_stack([
        (perim_xs - half_w) * resolution,
        (perim_ys - half_h) * resolution,
        perim_zs,
    ])

    # --- 3. Ceiling ---
    ceil_xs = np.arange(0, crop_w, 3)
    ceil_ys = np.arange(0, crop_h, 3)
    ceil_xx, ceil_yy = np.meshgrid(ceil_xs, ceil_ys)
    ceil_pts = np.column_stack([
        (ceil_xx.ravel() - half_w) * resolution,
        (ceil_yy.ravel() - half_h) * resolution,
        np.full(ceil_xx.size, wall_height),
    ])

    # --- 4. Ground ---
    gx = np.arange(-map_w / 2, map_w / 2, resolution * 3)
    gy = np.arange(-map_h / 2, map_h / 2, resolution * 3)
    gxx, gyy = np.meshgrid(gx, gy)
    ground_pts = np.column_stack([gxx.ravel(), gyy.ravel(), np.full(gxx.size, -0.1)])

    all_pts = np.vstack([wall_pts, perim_pts, ceil_pts, ground_pts])

    # Find robot start position (pixel value 208)
    robot_cells = np.nonzero(gt[rmin:rmax + 1, cmin:cmax + 1] == 208)
    if len(robot_cells[0]) > 0:
        idx = min(10, len(robot_cells[0]) - 1)
        start_px = robot_cells[1][idx]
        start_py = robot_cells[0][idx]
        start_x = (start_px - half_w) * resolution
        start_y = (start_py - half_h) * resolution
    else:
        start_x, start_y = 0.0, 0.0

    print(f"  [load_ariadne_map] cropped={crop_w}x{crop_h}px = {map_w:.1f}m x {map_h:.1f}m, "
          f"occ_cells={n_occ}, total_pts={len(all_pts)}, start=({start_x:.1f},{start_y:.1f})", flush=True)
    return all_pts, map_w, map_h, start_x, start_y


def get_ariadne_maps_dir() -> str:
    maps_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "maps", "ariadne"
    )
    return maps_dir if os.path.isdir(maps_dir) else ""


def list_ariadne_maps() -> List[str]:
    d = get_ariadne_maps_dir()
    if not d:
        return []
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.png')])
