"""Visualization module replicating FUEL's RViz display using matplotlib."""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple, Dict

# FUEL color scheme (matching RViz config)
COLORS = {
    'background': np.array([255, 253, 224]) / 255.0,   # RViz background
    'occupied': None,        # Z-axis rainbow
    'free': np.array([1.0, 1.0, 1.0]),
    'unknown': np.array([0.5, 0.5, 0.5]),
    'frontier_alpha': 0.4,
    'trajectory': np.array([1.0, 0.0, 0.0]),       # Red B-spline
    'executed_traj': np.array([0.0, 0.0, 1.0]),     # Blue executed
    'fov': np.array([1.0, 0.0, 0.0]),               # Red FOV lines
    'viewpoint': np.array([0.0, 0.5, 0.0]),         # Dark green viewpoints
    'view_dir': np.array([0.0, 1.0, 0.5]),          # Teal view direction
    'yaw_traj': np.array([1.0, 0.5, 0.0]),          # Orange yaw
    'boundary': np.array([1.0, 0.0, 0.0]),           # Red boundary
    'ground_truth': np.array([0.63, 0.63, 0.63]),    # Light gray, alpha=0.02
    'uav': np.array([0.0, 0.5, 0.0]),                # Green UAV
}


def get_rainbow_color(h: float, alpha: float = 1.0) -> Tuple[float, ...]:
    """FUEL's rainbow color function (matching getColor in planning_visualization.cpp)."""
    h = h % 1.0
    r, g, b = 0.0, 0.0, 0.0
    if h < 1.0 / 6:
        r, g, b = 1.0, 0.0, h * 6
    elif h < 2.0 / 6:
        r, g, b = 1.0 - (h - 1.0 / 6) * 6, 0.0, 1.0
    elif h < 3.0 / 6:
        r, g, b = 0.0, (h - 2.0 / 6) * 6, 1.0
    elif h < 4.0 / 6:
        r, g, b = 0.0, 1.0, 1.0 - (h - 3.0 / 6) * 6
    elif h < 5.0 / 6:
        r, g, b = (h - 4.0 / 6) * 6, 1.0, 0.0
    else:
        r, g, b = 1.0, 1.0 - (h - 5.0 / 6) * 6, 0.0
    return (r, g, b, alpha)


class FuelVisualizer:
    """Replicates FUEL's RViz visualization using matplotlib."""

    def __init__(self, figsize: Tuple[int, int] = (10, 10), slice_height: Optional[float] = None):
        self.figsize = figsize
        self.slice_height = slice_height
        self.fig = None
        self.ax = None

    def render_2d(self, core, agent_pos: np.ndarray, agent_yaw: float,
                  frontiers: Optional[List] = None,
                  planned_path: Optional[List[np.ndarray]] = None,
                  executed_path: Optional[List[np.ndarray]] = None,
                  show_fov: bool = True,
                  show_viewpoints: bool = True,
                  show_boundary: bool = True,
                  show_ground_truth: bool = False,
                  title: str = '',
                  save_path: Optional[str] = None,
                  return_array: bool = False):
        """Render 2D occupancy grid slice matching FUEL's RViz top-down view.

        Args:
            core: FuelEnvCore instance
            agent_pos: [x, y, z] UAV position
            agent_yaw: UAV heading in radians
            frontiers: list of FrontierInfo objects
            planned_path: list of 3D points forming the planned path
            executed_path: list of 3D points forming the executed trajectory
            show_fov: show FOV frustum
            show_viewpoints: show viewpoint markers
            show_boundary: show exploration boundary
            show_ground_truth: show ground truth obstacle points
            title: plot title
            save_path: if set, save to this path
            return_array: if True, return image array instead of displaying
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)

        ax = self.ax
        ax.clear()
        ax.set_facecolor(COLORS['background'])

        # Get map dimensions (now returns tuples from pybind11)
        origin, size = core.get_region()
        origin = np.array(origin)
        size = np.array(size)
        resolution = core.get_resolution()
        voxel_num = np.array(core.get_map_voxel_num())

        # Slice height
        h = self.slice_height if self.slice_height is not None else agent_pos[2]

        # Get 2D occupancy slice
        slice_2d = np.array(core.get_occupancy_slice_2d(h), dtype=np.int8)
        nx, ny = voxel_num[0], voxel_num[1]

        if len(slice_2d) == nx * ny:
            grid = slice_2d.reshape(nx, ny)
        else:
            grid = np.zeros((nx, ny), dtype=np.int8)

        # Create RGB image matching FUEL's color scheme
        img = np.ones((nx, ny, 3)) * COLORS['background']

        # Unknown cells: gray
        mask_unknown = grid == 0
        img[mask_unknown] = COLORS['unknown']

        # Free cells: white
        mask_free = grid == 1
        img[mask_free] = COLORS['free']

        # Occupied cells: color by Z-height (rainbow, FUEL style)
        mask_occ = (grid == 2) | (grid == 3)
        if np.any(mask_occ):
            normalized_h = np.clip((h - origin[2]) / size[2], 0, 1)
            occ_color = np.array(get_rainbow_color(normalized_h)[:3])
            img[mask_occ] = occ_color

        # Ground truth overlay (very transparent, matching FUEL's alpha=0.02)
        if show_ground_truth:
            pass  # Would need gt_map access; skip for now

        # Compute extent for proper axis labels
        x_min, x_max = origin[0], origin[0] + size[0]
        y_min, y_max = origin[1], origin[1] + size[1]
        extent = [y_min, y_max, x_min, x_max]

        ax.imshow(img, origin='lower', extent=extent, aspect='equal', interpolation='nearest')

        # Draw exploration boundary (red dashed, matching FUEL's update range marker)
        if show_boundary:
            bmin, bmax = core.get_box()
            bmin = np.array(bmin)
            bmax = np.array(bmax)
            rect = patches.Rectangle(
                (bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                linewidth=1.5, edgecolor=(*COLORS['boundary'], 0.3),
                facecolor='none', linestyle='--')
            ax.add_patch(rect)

        # Draw frontiers (rainbow colored, alpha=0.4, matching FUEL)
        if frontiers:
            n_frontiers = len(frontiers)
            for i, f in enumerate(frontiers):
                color = get_rainbow_color(i / max(n_frontiers, 1), alpha=0.6)
                cells = np.array(f.cells)
                # Filter to slice near current height
                h_mask = np.abs(cells[:, 2] - h) < resolution * 3
                if np.any(h_mask):
                    ax.scatter(cells[h_mask, 1], cells[h_mask, 0],
                              c=[color], s=1.0, marker='s', zorder=3)

                # Draw viewpoints
                if show_viewpoints and f.best_viewpoint_visib_num > 0:
                    vp = np.array(f.best_viewpoint_pos)
                    ax.plot(vp[1], vp[0], 'o', color=COLORS['viewpoint'],
                            markersize=6, zorder=5, markeredgecolor='white', markeredgewidth=0.5)
                    # View direction line (teal, matching FUEL)
                    dx = 0.5 * np.cos(f.best_viewpoint_yaw)
                    dy = 0.5 * np.sin(f.best_viewpoint_yaw)
                    ax.plot([vp[1], vp[1] + dy], [vp[0], vp[0] + dx],
                            '-', color=COLORS['view_dir'], linewidth=1.5, zorder=5)

        # Draw FOV frustum (red lines, matching FUEL)
        if show_fov:
            fov_h = 1.4 / 2
            fov_v = 1.12 / 2
            max_dist = 4.5
            # 4 corners of FOV at max_dist
            corners = []
            for sh in [-1, 1]:
                for sv in [-1, 1]:
                    dx = max_dist * np.cos(agent_yaw + sh * fov_h) * np.cos(sv * fov_v)
                    dy = max_dist * np.sin(agent_yaw + sh * fov_h) * np.cos(sv * fov_v)
                    corners.append([agent_pos[0] + dx, agent_pos[1] + dy])
            # Draw FOV lines from UAV to corners
            for c in corners:
                ax.plot([agent_pos[1], c[1]], [agent_pos[0], c[0]],
                        '-', color=(*COLORS['fov'], 0.8), linewidth=1.0, zorder=4)
            # Draw connecting lines between corners
            for i in range(4):
                j = (i + 1) % 4 if i < 3 else 0
                if i < 3:
                    ax.plot([corners[i][1], corners[j][1]],
                            [corners[i][0], corners[j][0]],
                            '-', color=(*COLORS['fov'], 0.8), linewidth=1.0, zorder=4)

        # Draw executed trajectory (blue spheres, matching FUEL)
        if executed_path and len(executed_path) > 1:
            pts = np.array(executed_path)
            ax.plot(pts[:, 1], pts[:, 0], '.', color=COLORS['executed_traj'],
                    markersize=2, zorder=4)

        # Draw planned path (red line, matching FUEL's B-spline trajectory)
        if planned_path and len(planned_path) > 1:
            pts = np.array(planned_path)
            ax.plot(pts[:, 1], pts[:, 0], '-', color=COLORS['trajectory'],
                    linewidth=2.0, zorder=5)
            ax.plot(pts[:, 1], pts[:, 0], 'o', color=COLORS['trajectory'],
                    markersize=3, zorder=5)

        # Draw UAV position (green arrow, matching FUEL's robot model)
        uav_size = 0.4
        dx = uav_size * np.cos(agent_yaw)
        dy = uav_size * np.sin(agent_yaw)
        ax.annotate('', xy=(agent_pos[1] + dy, agent_pos[0] + dx),
                    xytext=(agent_pos[1], agent_pos[0]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['uav'], lw=2.5),
                    zorder=10)
        ax.plot(agent_pos[1], agent_pos[0], 'o', color=COLORS['uav'],
                markersize=8, zorder=10, markeredgecolor='white', markeredgewidth=1)

        # Axis labels and formatting
        ax.set_xlabel('Y (m)', fontsize=10)
        ax.set_ylabel('X (m)', fontsize=10)
        if title:
            ax.set_title(title, fontsize=11)

        # Progress info
        progress = core.get_exploration_progress()
        n_frontier = len(frontiers) if frontiers else 0
        info_text = f'Progress: {progress:.1%} | Frontiers: {n_frontier} | Pos: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f})'
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            self.fig.savefig(save_path, dpi=100, bbox_inches='tight')

        if return_array:
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

        return self.fig

    def render_episode_animation(self, frames: List[str], output_path: str, fps: int = 5):
        """Generate exploration animation from saved frame images."""
        import imageio
        images = []
        for path in frames:
            images.append(imageio.imread(path))
        if output_path.endswith('.gif'):
            imageio.mimsave(output_path, images, fps=fps)
        elif output_path.endswith('.mp4'):
            imageio.mimwrite(output_path, images, fps=fps)
        print(f'Animation saved to {output_path}')
