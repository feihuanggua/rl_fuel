"""Gymnasium environment for FUEL reinforcement learning exploration."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any

from fuel_rl import FuelEnvCore, SDFMapParams, FrontierParams, PerceptionParams, AstarParams, FrontierInfo
from fuel_rl.config import default_map_params, default_frontier_params, default_perception_params, default_astar_params
from fuel_rl.map_loader import generate_random_map_for_fuel, pcd_file_list
from fuel_rl.visualizer import FuelVisualizer


class FuelRLEnv(gym.Env):
    """FUEL RL Exploration Environment.

    Simulates the full exploration process:
    1. Agent selects a frontier and generates viewpoint offset
    2. Environment plans path to viewpoint
    3. Simulates sensor observation
    4. Updates map and detects new frontiers
    5. Returns reward based on coverage and exploration progress
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    MAX_FRONTIERS = 10
    VOXEL_GRID_SIZE = 32

    def __init__(
        self,
        map_paths: Optional[List[str]] = None,
        map_size: Tuple[float, float, float] = (20.0, 20.0, 3.0),
        num_pillars: int = 15,
        max_steps: int = 200,
        visualize: bool = False,
        vis_mode: str = '2d',
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.map_paths = map_paths or []
        self.map_size = map_size
        self.num_pillars = num_pillars
        self.max_steps = max_steps

        # Reward weights
        self.rw = reward_weights or {
            'volume': 0.01,
            'coverage': 1.0,
            'path': 0.1,
            'frontier_elim': 2.0,
            'invalid': 20.0,
            'completion': 100.0,
            'progress': 5.0,
        }

        # Action space: frontier_idx + viewpoint offset
        self.action_space = spaces.Dict({
            'frontier_idx': spaces.Discrete(self.MAX_FRONTIERS),
            'viewpoint_offset': spaces.Box(
                low=np.array([-1.0, -1.0, -0.5, -np.pi/2], dtype=np.float32),
                high=np.array([1.0, 1.0, 0.5, np.pi/2], dtype=np.float32),
                shape=(4,), dtype=np.float32
            ),
        })

        voxel_feat_dim = self.VOXEL_GRID_SIZE ** 3
        frontier_feat_dim = 7  # center(3) + best_vp_pos(3) + frontier_size(1)
        # Note: vp_yaw and vp_visib are added to make it 7 features total
        # Actually: center(3) + best_vp_pos(3) + best_vp_yaw(1) + best_vp_visib(1) + frontier_size(1) + dist_to_agent(1) = 10
        frontier_feat_dim = 10

        self.observation_space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'frontiers': spaces.Box(
                low=-1, high=1e4, shape=(self.MAX_FRONTIERS, voxel_feat_dim + frontier_feat_dim),
                dtype=np.float32),
            'mask': spaces.Box(low=0, high=1, shape=(self.MAX_FRONTIERS,), dtype=bool),
        })

        # Core and visualizer
        self.core = FuelEnvCore()
        self.visualizer = FuelVisualizer(slice_height=1.0) if visualize else None
        self.vis_mode = vis_mode

        self._init_core()

        # Episode state
        self.agent_pos = np.zeros(3)
        self.episode_rewards = []
        self._current_episode_reward = 0.0
        self.agent_yaw = 0.0
        self.frontiers: List[FrontierInfo] = []
        self.step_count = 0
        self.prev_unknown = 0
        self.prev_n_frontiers = 0
        self.prev_progress = 0.0
        self.planned_path: List[np.ndarray] = []
        self.executed_path: List[np.ndarray] = []

    def _init_core(self):
        box_margin = 1.0
        mp = default_map_params(
            size_x=self.map_size[0], size_y=self.map_size[1], size_z=self.map_size[2],
            box_min=(-self.map_size[0]/2 + box_margin, -self.map_size[1]/2 + box_margin, 0.0),
            box_max=(self.map_size[0]/2 - box_margin, self.map_size[1]/2 - box_margin, self.map_size[2] - 0.2),
        )
        self.core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load map
        opts = options or {}
        map_path = opts.get('map_path')
        if map_path:
            self.core.load_map_from_pcd(map_path)
        elif self.map_paths:
            idx = self.np_random.integers(len(self.map_paths))
            self.core.load_map_from_pcd(self.map_paths[idx])
        else:
            seed = self.np_random.integers(0, 100000)
            obs_points = generate_random_map_for_fuel(
                map_size_x=self.map_size[0], map_size_y=self.map_size[1],
                map_size_z=self.map_size[2], num_pillars=self.num_pillars, seed=seed)
            self.core.load_map_from_points(obs_points)

        self.core.reset_map()

        # Start position
        self.agent_pos = np.array([0.0, 0.0, 1.5])
        self.agent_yaw = 0.0

        # Initial observation (4 directions)
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            self.core.simulate_observation(self.agent_pos, yaw)

        self.frontiers = self.core.detect_frontiers(self.agent_pos)
        self.step_count = 0
        self.prev_unknown = self.core.count_unknown_voxels()
        self.prev_n_frontiers = len(self.frontiers)
        self.prev_progress = self.core.get_exploration_progress()
        self.planned_path = []
        self.executed_path = [self.agent_pos.copy()]
        self._current_episode_reward = 0.0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        fidx = action['frontier_idx']
        offset = np.array(action['viewpoint_offset'], dtype=np.float64)

        # Validate frontier selection
        if fidx >= len(self.frontiers):
            obs = self._get_obs()
            return obs, -self.rw['invalid'], False, False, {'error': 'invalid_frontier_idx'}

        target = self.frontiers[fidx]
        prev_unknown = self.core.count_unknown_voxels()
        prev_n_frontiers = len(self.frontiers)
        prev_progress = self.core.get_exploration_progress()

        # Compute viewpoint
        vp_pos = np.array(target.best_viewpoint_pos) + offset[:3]
        vp_yaw = target.best_viewpoint_yaw + offset[3]

        # Validate viewpoint
        bmin, bmax = self.core.get_box()
        bmin = np.array(bmin)
        bmax = np.array(bmax)
        out_of_box = any(vp_pos[i] <= bmin[i] or vp_pos[i] >= bmax[i] for i in range(3))

        if out_of_box or self.core.get_occupancy(vp_pos) != 1:  # not free
            # Try clamping z to valid range
            vp_pos[2] = np.clip(vp_pos[2], 0.5, bmax[2] - 0.2)
            if self.core.get_occupancy(vp_pos) in [2, -1]:  # occupied or out of map
                obs = self._get_obs()
                return obs, -self.rw['invalid'] * 0.5, False, False, {'error': 'invalid_viewpoint'}

        # Plan path
        path = self.core.plan_path(self.agent_pos, vp_pos)
        if not path:
            obs = self._get_obs()
            return obs, -self.rw['invalid'] * 0.3, False, False, {'error': 'no_path'}

        self.planned_path = [np.array(p) for p in path]
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                         for i in range(len(path) - 1)) if len(path) > 1 else 0

        # Count expected visible cells
        visible_cells = self.core.count_visible_cells(vp_pos, vp_yaw, target.cells)

        # Move to viewpoint and simulate observation
        self.core.simulate_observation(vp_pos, vp_yaw)
        self.agent_pos = vp_pos.copy()
        self.agent_yaw = vp_yaw
        self.executed_path.append(self.agent_pos.copy())

        # Detect new frontiers
        self.frontiers = self.core.detect_frontiers(self.agent_pos)

        # Compute reward
        new_unknown = self.core.count_unknown_voxels()
        new_n_frontiers = len(self.frontiers)
        new_progress = self.core.get_exploration_progress()

        volume_reward = self.rw['volume'] * (prev_unknown - new_unknown)
        coverage_reward = self.rw['coverage'] * (visible_cells / max(target.frontier_size, 1))
        path_penalty = -self.rw['path'] * path_length
        frontier_elim = self.rw['frontier_elim'] * max(0, prev_n_frontiers - new_n_frontiers)
        progress_reward = self.rw['progress'] * (new_progress - prev_progress)

        reward = volume_reward + coverage_reward + path_penalty + frontier_elim + progress_reward

        # Termination
        terminated = len(self.frontiers) == 0
        if terminated:
            reward += self.rw['completion']

        truncated = self.step_count >= self.max_steps

        info = {
            'volume_reward': volume_reward,
            'coverage_reward': coverage_reward,
            'path_length': path_length,
            'frontier_elim': frontier_elim,
            'progress_reward': progress_reward,
            'exploration_progress': new_progress,
            'n_frontiers': new_n_frontiers,
            'visible_cells': visible_cells,
        }

        obs = self._get_obs()
        self._current_episode_reward += float(reward)
        if terminated or truncated:
            self.episode_rewards.append(self._current_episode_reward)
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        # Global state
        progress = self.core.get_exploration_progress()
        n_frontiers = len(self.frontiers)
        global_obs = np.array([
            self.agent_pos[0], self.agent_pos[1], self.agent_pos[2],
            self.agent_yaw, progress, n_frontiers
        ], dtype=np.float32)

        # Per-frontier observations
        voxel_dim = self.VOXEL_GRID_SIZE ** 3
        feat_dim = 10
        frontier_obs = np.zeros((self.MAX_FRONTIERS, voxel_dim + feat_dim), dtype=np.float32)
        mask = np.zeros(self.MAX_FRONTIERS, dtype=bool)

        for i in range(min(n_frontiers, self.MAX_FRONTIERS)):
            f = self.frontiers[i]
            mask[i] = True

            # Local voxel grid
            voxel = np.array(self.core.get_local_voxel_grid(f.average, self.VOXEL_GRID_SIZE), dtype=np.float32)
            frontier_obs[i, :voxel_dim] = voxel

            # Features
            avg = np.array(f.average)
            vp = np.array(f.best_viewpoint_pos)
            dist = np.linalg.norm(avg - self.agent_pos)
            feat = np.array([
                avg[0], avg[1], avg[2],
                vp[0], vp[1], vp[2],
                f.best_viewpoint_yaw,
                f.best_viewpoint_visib_num,
                f.frontier_size,
                dist,
            ], dtype=np.float32)
            frontier_obs[i, voxel_dim:] = feat

        return {
            'global': global_obs,
            'frontiers': frontier_obs,
            'mask': mask,
        }

    def render(self):
        if self.visualizer:
            return self.visualizer.render_2d(
                self.core, self.agent_pos, self.agent_yaw,
                frontiers=self.frontiers,
                planned_path=self.planned_path,
                executed_path=self.executed_path,
                return_array=True,
            )


class FuelRLEnvSingleFrontier(FuelRLEnv):
    """Simplified env: auto-selects nearest frontier, agent only outputs viewpoint offset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5, -np.pi/2], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5, np.pi/2], dtype=np.float32),
            shape=(4,), dtype=np.float32
        )

    def step(self, action):
        offset = np.array(action, dtype=np.float64)

        # Auto-select nearest frontier
        if not self.frontiers:
            obs = self._get_obs()
            return obs, 0.0, True, False, {'error': 'no_frontiers'}

        dists = [np.linalg.norm(np.array(f.average) - self.agent_pos) for f in self.frontiers]
        fidx = int(np.argmin(dists))

        full_action = {
            'frontier_idx': fidx,
            'viewpoint_offset': offset,
        }
        return super().step(full_action)
