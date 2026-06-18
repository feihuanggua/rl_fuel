"""序列级环境: 多步探索，FUEL 原生方法生成各前沿视点，策略选择访问顺序.

观测包含:
  - frontiers [N, 6]: center(3) + size(1) + eucl_dist(1) + visib(1)
  - mask [N]
  - global [2]: coverage + step progress
  - map_img [3, H, W]: 2D 俯视图 (障碍/自由/未知 3通道)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import torch

from fuel_rl import FuelEnvCore
from fuel_rl.config import (default_map_params, default_frontier_params,
                            fast_perception_params, default_astar_params)
from fuel_rl.map_loader import generate_random_map_for_fuel, load_ariadne_map
from fuel_rl.env.gpu_depth_renderer import GPUDepthRenderer

MAX_FRONTIERS = 50
MAP_SIZE_2D = 64
FEAT_DIM = 8


class SequenceEnv(gym.Env):
    """多步探索: 每步从前沿列表中选一个 → FUEL 原生视点 → 观测 → 覆盖率奖励."""

    def __init__(self, max_steps=500, map_size=(20, 20, 2), num_pillars=15,
                  target_coverage=0.95, map_type='random', map_path=None):
        super().__init__()
        self.max_steps = max_steps
        self.map_size = map_size
        self.num_pillars = num_pillars
        self.target_coverage = target_coverage
        self.map_type = map_type
        self.map_path = map_path

        self.observation_space = spaces.Dict({
            "frontiers": spaces.Box(-1, 1, (MAX_FRONTIERS, FEAT_DIM), dtype=np.float32),
            "mask": spaces.Box(0, 1, (MAX_FRONTIERS,), dtype=np.float32),
            "global": spaces.Box(-1, 1, (2,), dtype=np.float32),
            "map_img": spaces.Box(0, 1, (3, MAP_SIZE_2D, MAP_SIZE_2D), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(MAX_FRONTIERS)

        self.core = FuelEnvCore()
        mp = default_map_params(
            size_x=map_size[0], size_y=map_size[1], size_z=map_size[2],
            box_min=(-map_size[0]/2+1, -map_size[1]/2+1, 0.0),
            box_max=(map_size[0]/2-1, map_size[1]/2-1, map_size[2]-0.2),
        )
        self.core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())

        self.agent_pos = np.zeros(3)
        self.step_count = 0
        self._cached_frontiers = []
        self._cached_obs = None
        self._gpu_renderer = None
        self._use_gpu = torch.cuda.is_available()
        self._initial_free_count = 0

    def _simulate_obs(self, pos, yaw):
        if self._use_gpu and self._gpu_renderer is not None:
            hit_pts, free_pts = self._gpu_renderer.render_with_free(pos, yaw)
            if len(hit_pts) > 0 or len(free_pts) > 0:
                self.core.input_hit_and_free_points(hit_pts, free_pts, pos)
        else:
            self.core.simulate_observation(pos, yaw)

    def _build_map_image(self):
        vn = self.core.get_map_voxel_num()
        nx, ny = vn[0], vn[1]
        slice_2d = np.array(self.core.get_occupancy_slice_2d(self.map_size[2] / 2.0)).reshape(nx, ny)
        from skimage.transform import resize as sk_resize
        img = np.zeros((3, MAP_SIZE_2D, MAP_SIZE_2D), dtype=np.float32)
        occ_3 = sk_resize((slice_2d == 3).astype(np.float32), (MAP_SIZE_2D, MAP_SIZE_2D),
                           order=0, preserve_range=True, anti_aliasing=False)
        occ_1 = sk_resize((slice_2d == 1).astype(np.float32), (MAP_SIZE_2D, MAP_SIZE_2D),
                           order=0, preserve_range=True, anti_aliasing=False)
        occ_0 = sk_resize((slice_2d == 0).astype(np.float32), (MAP_SIZE_2D, MAP_SIZE_2D),
                           order=0, preserve_range=True, anti_aliasing=False)
        img[0] = occ_3
        img[1] = occ_1
        img[2] = occ_0
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = self.np_random.integers(0, 100000) if seed is None else seed

        self.core = FuelEnvCore()
        if self.map_type == 'ariadne' and self.map_path:
            from fuel_rl.map_loader import load_ariadne_map
            pts, mw, mh, start_x, start_y = load_ariadne_map(
                self.map_path, resolution=0.1,
                wall_height=self.map_size[2] - 0.2)
            sx, sy = mw + 2.0, mh + 2.0
        else:
            pts = generate_random_map_for_fuel(
                self.map_size[0], self.map_size[1], self.map_size[2], self.num_pillars, seed=s,
            )
            sx, sy = self.map_size[0], self.map_size[1]
            start_x, start_y = 0.0, 0.0
        mp = default_map_params(
            size_x=sx, size_y=sy, size_z=self.map_size[2],
            box_min=(-sx/2+1, -sy/2+1, 0.0),
            box_max=(sx/2-1, sy/2-1, self.map_size[2]-0.2),
        )
        self.core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
        self.core.load_map_from_points(pts)
        self.core.reset_map()

        if self._use_gpu:
            self._gpu_renderer = GPUDepthRenderer(pts)

        agent_z = self.map_size[2] / 2.0
        self.agent_pos = np.array([start_x, start_y, agent_z])
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            self._simulate_obs(self.agent_pos, yaw)

        self.step_count = 0
        self.total_distance = 0.0
        self.consecutive_fails = 0
        self.last_progress = self._get_exploration_progress()
        self._cached_frontiers = []
        self._cached_obs = None
        self._skip_set = set()
        self._cell_last_visit = {}
        start_cell = (int(self.agent_pos[0] / 5.0), int(self.agent_pos[1] / 5.0))
        self._cell_last_visit[start_cell] = 0
        return self._get_obs(), {}

    def _get_exploration_progress(self):
        """Coverage = discovered_free / total_free."""
        if self.map_type == 'ariadne':
            vn = self.core.get_map_voxel_num()
            nx, ny = vn[0], vn[1]
            slice_2d = np.array(self.core.get_occupancy_slice_2d(
                self.map_size[2] / 2.0)).reshape(nx, ny)
            free_count = np.sum(slice_2d == 1)
            unknown_count = np.sum(slice_2d == 0)
            total_explorable = free_count + unknown_count
            if total_explorable == 0:
                return 1.0
            return float(free_count) / total_explorable
        else:
            return self.core.get_exploration_progress()

    def _get_obs(self):
        frontiers = self.core.detect_frontiers(self.agent_pos)
        self._cached_frontiers = frontiers
        progress = self._get_exploration_progress()

        frontier_feats = np.zeros((MAX_FRONTIERS, FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_FRONTIERS, dtype=np.float32)
        self._visible_indices = []

        vis_idx = 0
        for i, f in enumerate(frontiers):
            if i >= MAX_FRONTIERS:
                break
            center = np.array(f.average)
            if vis_idx >= MAX_FRONTIERS:
                break

            if i in self._skip_set:
                continue
            vp = np.array(f.best_viewpoint_pos)
            vp[2] = np.clip(vp[2], 0.5, 2.6)
            eucl_dist = np.linalg.norm(vp - self.agent_pos)
            direction = (center - self.agent_pos)
            dir_norm = np.linalg.norm(direction[:2])
            if dir_norm > 1e-6:
                direction[:2] /= dir_norm
            else:
                direction[:2] = 0.0

            mask[vis_idx] = 1.0
            rel_pos = center - self.agent_pos
            frontier_feats[vis_idx, 0] = np.clip(rel_pos[0] / 20.0, -1.0, 1.0)
            frontier_feats[vis_idx, 1] = np.clip(rel_pos[1] / 20.0, -1.0, 1.0)
            frontier_feats[vis_idx, 2] = np.clip(rel_pos[2] / 2.0, -1.0, 1.0)
            frontier_feats[vis_idx, 3] = min(f.frontier_size / 2000.0, 1.0)
            frontier_feats[vis_idx, 4] = min(eucl_dist / 25.0, 1.0)
            frontier_feats[vis_idx, 5] = min(f.best_viewpoint_visib_num / 100.0, 1.0)
            frontier_feats[vis_idx, 6:8] = direction[:2]
            self._visible_indices.append(i)
            vis_idx += 1

        if vis_idx == 0 and len(self._skip_set) > 0:
            self._skip_set.clear()
            return self._get_obs()

        self._n_visible = vis_idx

        global_feat = np.array([
            progress * 2 - 1,
            self.step_count / self.max_steps * 2 - 1,
        ], dtype=np.float32)

        map_img = self._build_map_image()

        obs = {"frontiers": frontier_feats, "mask": mask, "global": global_feat, "map_img": map_img}
        self._cached_obs = obs
        return obs

    def step(self, action):
        self.step_count += 1

        if self._n_visible == 0:
            obs = self._get_obs()
            return obs, -3.0, self.step_count >= self.max_steps, False, {"coverage": self.last_progress, "total_dist": self.total_distance}


        action = min(int(action), self._n_visible - 1)
        orig_idx = self._visible_indices[action]
        if orig_idx >= len(self._cached_frontiers):
            obs = self._get_obs()
            return obs, -3.0, self.step_count >= self.max_steps, False, {"coverage": self.last_progress, "total_dist": self.total_distance}

        f = self._cached_frontiers[orig_idx]

        vp = np.array(f.best_viewpoint_pos)
        vp_yaw = f.best_viewpoint_yaw
        vp[2] = np.clip(vp[2], 0.5, 2.6)

        occ = self.core.get_occupancy(vp)
        if occ != 1:
            fallback_dist = float('inf')
            fallback_f = None
            for j in range(self._n_visible):
                fi = self._visible_indices[j]
                if fi >= len(self._cached_frontiers):
                    continue
                fc = self._cached_frontiers[fi]
                vpc = np.array(fc.best_viewpoint_pos)
                vpc[2] = np.clip(vpc[2], 0.5, 2.6)
                if self.core.get_occupancy(vpc) == 1:
                    d = np.linalg.norm(vpc - self.agent_pos)
                    if d < fallback_dist:
                        fallback_dist = d
                        fallback_f = fc
            if fallback_f is not None:
                f = fallback_f
                vp = np.array(f.best_viewpoint_pos)
                vp[2] = np.clip(vp[2], 0.5, 2.6)
                vp_yaw = f.best_viewpoint_yaw
            else:
                obs = self._get_obs()
                return obs, -3.0, False, False, {"coverage": self.last_progress, "total_dist": self.total_distance}

        old_pos = self.agent_pos.copy()
        eucl_dist = np.linalg.norm(vp - self.agent_pos)
        self._simulate_obs(vp, vp_yaw)
        self.total_distance += eucl_dist
        self.consecutive_fails = 0
        self.agent_pos = vp.copy()

        new_progress = self._get_exploration_progress()
        delta = new_progress - self.last_progress

        if delta < 0.001:
            self._skip_set.add(orig_idx)
        else:
            self._skip_set.clear()

        self.last_progress = new_progress

        # --- Reward v3: region-level revisit ---
        reward = delta * 200.0 - 1.0
        reward -= eucl_dist * 0.5

        CELL_SIZE = 5.0
        new_cell = (int(vp[0] / CELL_SIZE), int(vp[1] / CELL_SIZE))
        if new_cell in self._cell_last_visit:
            steps_since = self.step_count - self._cell_last_visit[new_cell]
            if steps_since > 30:
                reward -= min(steps_since * 0.3, 8.0)
        self._cell_last_visit[new_cell] = self.step_count

        if delta > 0.01:
            reward += 2.0
        if delta < 0.003 and eucl_dist > 5.0:
            reward -= 3.0
        done = False

        if new_progress >= self.target_coverage:
            reward += 100.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, float(reward), done, False, {"coverage": new_progress, "total_dist": self.total_distance}
