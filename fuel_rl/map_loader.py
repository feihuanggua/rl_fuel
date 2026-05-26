"""Map loading and random generation utilities."""
import numpy as np
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
    """Generate random pillar obstacles.

    Ported from FUEL's random_forest_sensing.cpp.
    Returns Nx3 array of obstacle points.
    """
    rng = np.random.default_rng(seed)
    obstacles = []

    for _ in range(num_pillars):
        cx = rng.uniform(*x_range)
        cy = rng.uniform(*y_range)
        w = rng.uniform(*pillar_width_range)
        h = rng.uniform(*pillar_height_range)
        h = min(h, z_range[1])

        # Generate pillar as voxel grid
        for dx in np.arange(-w / 2, w / 2, resolution):
            for dy in np.arange(-w / 2, w / 2, resolution):
                for dz in np.arange(0, h, resolution):
                    obstacles.append([cx + dx, cy + dy, z_range[0] + dz])

    if add_ground:
        # Add ground plane
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
    """Generate random map aligned with FUEL's coordinate system.

    Map origin is at (-size/2, -size/2, ground_height).
    Exploration box slightly smaller than map size.
    """
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
    """Return list of available PCD files from FUEL's map resources."""
    import os
    resource_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uav_simulator", "map_generator", "resource"
    )
    if os.path.isdir(resource_dir):
        pcds = [os.path.join(resource_dir, f) for f in os.listdir(resource_dir) if f.endswith(".pcd")]
        return sorted(pcds)
    return []
