"""GPU-accelerated depth rendering using PyTorch.

Replaces the CPU ray-casting loop (~200ms) with GPU projection + z-buffer (~5ms).
"""
import torch
import torch.nn.functional as F
import numpy as np


class GPUDepthRenderer:
    def __init__(self, gt_points_np, device="cuda",
                 # Camera intrinsics match FUEL's camera.yaml & C++ simulateObservation
                 # fx=387.229, fy=387.229, cx=321.046, cy=243.449
                 fx=387.229, fy=387.229, cx=321.046, cy=243.449,
                 width=640, height=480, skip_pixel=4, margin=1,
                 max_range=4.5, min_range=0.2, free_dist=5.0):
        self.device = device
        self.width = width
        self.height = height
        self.skip_pixel = skip_pixel
        self.margin = margin
        self.max_range = max_range
        self.min_range = min_range
        self.free_dist = free_dist

        # Camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Upload GT points to GPU
        self.gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device)

        # Pre-compute pixel grid
        u_all = torch.arange(margin, width - margin, skip_pixel, device=device, dtype=torch.float32)
        v_all = torch.arange(margin, height - margin, skip_pixel, device=device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u_all, v_all, indexing="ij")
        self.n_rays = uu.numel()
        # ray directions in camera frame: [N, 3]
        dirs_cam = torch.stack([
            (uu - cx) / fx,
            (vv - cy) / fy,
            torch.ones_like(uu),
        ], dim=-1).reshape(-1, 3)
        self.ray_dirs_cam = F.normalize(dirs_cam, dim=-1)

        # Camera -> Body: X_cam = Z_body, Y_cam = -X_body, Z_cam = -Y_body
        self.R_bc = torch.tensor([
            [0.0,  0.0,  1.0],
            [-1.0, 0.0,  0.0],
            [0.0, -1.0,  0.0],
        ], dtype=torch.float32, device=device)

    def render(self, cam_pos, yaw):
        """Render depth from given camera pose. Returns hit_points as numpy [N, 3]."""
        # Body -> World rotation
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_wb = torch.tensor([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device)

        R_wc = R_wb @ self.R_bc  # [3, 3]

        # Transform GT points to camera frame
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=self.device)
        pts_world = self.gt_points - cam_pos_t  # [N, 3]
        R_wc_t = R_wc.T  # world -> camera
        pts_cam = pts_world @ R_wc_t.T  # [N, 3]

        # Filter: only points in front of camera and within range
        depth = pts_cam[:, 2]
        valid = (depth > self.min_range) & (depth < self.max_range)
        pts_cam_v = pts_cam[valid]
        depth_v = depth[valid]

        if depth_v.numel() == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # Project to image plane
        u = pts_cam_v[:, 0] * self.fx / depth_v + self.cx
        v = pts_cam_v[:, 1] * self.fy / depth_v + self.cy

        # Filter in-image
        margin = self.margin
        skip = self.skip_pixel
        in_img = (u >= margin) & (u < self.width - margin) & (v >= margin) & (v < self.height - margin)
        u = u[in_img]
        v = v[in_img]
        depth_v = depth_v[in_img]

        if depth_v.numel() == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # Z-buffer: for each pixel, keep minimum depth
        # Map (u, v) to pixel index in the downsampled grid
        u_idx = ((u - margin) / skip).long()
        v_idx = ((v - margin) / skip).long()
        n_u = (self.width - 2 * margin + skip - 1) // skip
        n_v = (self.height - 2 * margin + skip - 1) // skip
        flat_idx = u_idx * n_v + v_idx
        n_pixels = n_u * n_v

        # Scatter-reduce to get min depth per pixel
        min_depth = torch.full((n_pixels,), self.max_range + 1.0, device=self.device, dtype=torch.float32)
        min_depth.scatter_reduce_(0, flat_idx, depth_v, reduce="amin", include_self=True)

        # Get hit pixels
        hit_mask = min_depth <= self.max_range
        hit_flat = hit_mask.nonzero(as_tuple=True)[0]
        hit_depths = min_depth[hit_flat]

        # Recover (u, v) from flat index
        hit_u_idx = hit_flat // n_v
        hit_v_idx = hit_flat % n_v
        hit_u = hit_u_idx.float() * skip + margin
        hit_v = hit_v_idx.float() * skip + margin

        # Convert to 3D hit points in camera frame
        x_cam = (hit_u - self.cx) * hit_depths / self.fx
        y_cam = (hit_v - self.cy) * hit_depths / self.fy
        z_cam = hit_depths
        pts_cam_hit = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [M, 3]

        # Transform to world frame
        pts_world_hit = (R_wc @ pts_cam_hit.T).T + cam_pos_t

        return pts_world_hit.cpu().numpy().astype(np.float64)

    def render_with_free(self, cam_pos, yaw):
        """Render hit + free points. Matches CPU simulate_observation behavior."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_wb = torch.tensor([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device)
        R_wc = R_wb @ self.R_bc
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=self.device)

        # Ray directions in world frame [N_rays, 3]
        ray_dirs_world = (R_wc @ self.ray_dirs_cam.T).T

        # Transform GT points to camera frame for z-buffer
        pts_world = self.gt_points - cam_pos_t
        pts_cam = pts_world @ R_wc.T.T

        depth = pts_cam[:, 2]
        valid = (depth > self.min_range) & (depth < self.max_range)
        pts_cam_v = pts_cam[valid]
        depth_v = depth[valid]

        n_u = (self.width - 2 * self.margin + self.skip_pixel - 1) // self.skip_pixel
        n_v = (self.height - 2 * self.margin + self.skip_pixel - 1) // self.skip_pixel
        n_pixels = n_u * n_v

        min_depth = torch.full((n_pixels,), self.max_range + 1.0, device=self.device, dtype=torch.float32)

        if depth_v.numel() > 0:
            u = pts_cam_v[:, 0] * self.fx / depth_v + self.cx
            v = pts_cam_v[:, 1] * self.fy / depth_v + self.cy
            margin = self.margin
            skip = self.skip_pixel
            in_img = (u >= margin) & (u < self.width - margin) & (v >= margin) & (v < self.height - margin)
            u_idx = ((u[in_img] - margin) / skip).long()
            v_idx = ((v[in_img] - margin) / skip).long()
            flat_idx = u_idx * n_v + v_idx
            min_depth.scatter_reduce_(0, flat_idx, depth_v[in_img], reduce="amin", include_self=True)

        # Generate all observation points
        # For hit rays: endpoint at hit depth
        # For miss rays: endpoint at free_dist
        hit_mask = min_depth <= self.max_range
        ray_depths = torch.where(hit_mask, min_depth, torch.tensor(self.free_dist, device=self.device))

        # 3D points for all rays: cam_pos + depth * ray_dir
        all_pts = cam_pos_t.unsqueeze(0) + ray_depths.unsqueeze(1) * ray_dirs_world

        return all_pts.cpu().numpy().astype(np.float64)
