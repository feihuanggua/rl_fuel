"""单步视点选择环境 (用于 PPO 微调)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fuel_rl import FuelEnvCore
from fuel_rl.config import default_map_params, default_frontier_params, fast_perception_params, default_astar_params
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid, get_expert_label

# 体素网格参数 (与 BC v2 训练时一致)
_GRID_XY = 32
_GRID_Z = 10
_VOXEL_RES = 0.2


class ViewpointEnv(gym.Env):
    """单步视点选择: 观测一个前沿的体素 → 输出视点 → 计算奖励 → done.

    rollout_steps > 0 时，执行视点后用 greedy 策略 rollout N 步，
    用累积覆盖率增量作为奖励，捕获视点的长期探索价值。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 32,
        grid_z: int = 10,
        resolution: float = 0.2,
        map_size=(20.0, 20.0, 3.0),
        num_pillars: int = 15,
        max_dist: float = 4.0,
        warmup_steps: int = 0,
        rollout_steps: int = 0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.grid_z = grid_z
        self.resolution = resolution
        self.map_size = map_size
        self.num_pillars = num_pillars
        self.max_dist = max_dist
        self.warmup_steps = warmup_steps
        self.rollout_steps = rollout_steps

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, grid_size, grid_size, grid_z), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        box_margin = 1.0
        mp = default_map_params(
            size_x=map_size[0], size_y=map_size[1], size_z=map_size[2],
            box_min=(-map_size[0]/2 + box_margin, -map_size[1]/2 + box_margin, 0.0),
            box_max=(map_size[0]/2 - box_margin, map_size[1]/2 - box_margin, map_size[2] - 0.2),
        )
        self.core = FuelEnvCore()
        self.core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())

        self.agent_pos = np.zeros(3)
        self.current_frontier = None
        self.current_grid = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 加载地图
        s = self.np_random.integers(0, 100000) if seed is None else seed
        obs_pts = generate_random_map_for_fuel(
            self.map_size[0], self.map_size[1], self.map_size[2],
            self.num_pillars, seed=s,
        )
        self.core.load_map_from_points(obs_pts)
        self.core.reset_map()

        # 初始观测扩展已知区域
        self.agent_pos = np.array([0.0, 0.0, 1.5])
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            self.core.simulate_observation(self.agent_pos, yaw)

        # 模拟几步产生更多前沿 (0 步 = 匹配 BC 训练分布)
        frontiers = self.core.detect_frontiers(self.agent_pos)
        pos = self.agent_pos.copy()
        for _ in range(self.warmup_steps):
            if not frontiers:
                break
            dists = [np.linalg.norm(np.array(f.average) - pos) for f in frontiers]
            f = frontiers[int(np.argmin(dists))]
            goal = np.array(f.best_viewpoint_pos)
            self.core.simulate_observation(goal, f.best_viewpoint_yaw)
            pos = goal.copy()
            frontiers = self.core.detect_frontiers(pos)

        if not frontiers:
            self.current_grid = np.zeros((3, self.grid_size, self.grid_size, self.grid_z), dtype=np.float32)
            self.current_frontier = None
            return self.current_grid, {}

        # 随机选一个前沿
        idx = self.np_random.integers(len(frontiers))
        self.current_frontier = frontiers[idx]
        self.current_grid = build_3channel_grid(self.core, self.current_frontier,
                                                 self.grid_size, self.grid_z, self.resolution)

        return self.current_grid, {}

    def _rollout(self, start_pos, n_steps):
        """Greedy rollout: 每步选最近前沿，用 best_viewpoint 执行观测.

        返回 rollout 结束后的覆盖率。
        """
        pos = start_pos.copy()
        for _ in range(n_steps):
            frontiers = self.core.detect_frontiers(pos)
            if not frontiers:
                break
            # 选最近前沿
            dists = [np.linalg.norm(np.array(f.average) - pos) for f in frontiers]
            f = frontiers[int(np.argmin(dists))]
            vp = np.array(f.best_viewpoint_pos)
            vy = f.best_viewpoint_yaw
            occ = self.core.get_occupancy(vp)
            if occ != 1:
                continue
            self.core.simulate_observation(vp, vy)
            pos = vp.copy()
        return self.core.get_exploration_progress()

    def _project_to_valid(self, center, offset_xyz):
        """将无效偏移投影到最近有效位置.

        从前沿中心沿 offset 方向逐步缩放，找到第一个自由空间点。
        返回 (valid_pos, scale)，scale ∈ [0, 1]。
        """
        direction = offset_xyz.copy()
        max_dist_vec = np.array([self.max_dist] * 3)
        target = center + direction * max_dist_vec

        # 先检查原始位置
        occ = self.core.get_occupancy(target)
        if occ == 1:
            bmin, bmax = np.array(self.core.get_box()[0]), np.array(self.core.get_box()[1])
            in_box = all(bmin[i] < target[i] < bmax[i] for i in range(3))
            if in_box:
                return target, 1.0, False

        # 沿方向二分搜索有效位置
        lo, hi = 0.0, 1.0
        best_pos, best_scale = center.copy(), 0.0
        for _ in range(10):  # 10 次二分，精度 ~0.1%
            mid = (lo + hi) / 2
            pos = center + direction * max_dist_vec * mid
            occ = self.core.get_occupancy(pos)
            bmin, bmax = np.array(self.core.get_box()[0]), np.array(self.core.get_box()[1])
            in_box = all(bmin[i] < pos[i] < bmax[i] for i in range(3))
            if occ == 1 and in_box:
                best_pos, best_scale = pos, mid
                lo = mid
            else:
                hi = mid
        return best_pos, best_scale, best_scale < 0.01  # too_close = 投影失败

    def step(self, action):
        frontier = self.current_frontier
        info = {"error": "none"}

        if frontier is None:
            return self.current_grid, 0.0, True, False, info

        center = np.array(frontier.average)
        offset_xyz = np.array([action[0], action[1], action[2]])
        target_yaw = action[3] * np.pi

        # 投影到有效位置
        target_pos, scale, too_close = self._project_to_valid(center, offset_xyz)

        if too_close:
            # 投影失败 (前沿中心附近无自由空间)，用 best_viewpoint 兜底
            target_pos = np.array(frontier.best_viewpoint_pos)
            target_yaw = frontier.best_viewpoint_yaw
            info["projected"] = "fallback"
        elif scale < 0.99:
            info["projected"] = f"scale={scale:.2f}"

        # 路径检测
        path = self.core.plan_path(self.agent_pos, target_pos)
        if not path:
            return self.current_grid, 0.0, True, False, {"error": "no_path"}

        # 计算奖励
        projection_penalty = -0.5 if "projected" in info else 0.0

        if self.rollout_steps > 0:
            prev_progress = self.core.get_exploration_progress()
            self.core.simulate_observation(target_pos, target_yaw)
            final_progress = self._rollout(target_pos, self.rollout_steps)
            reward = (final_progress - prev_progress) * 100.0 + projection_penalty

            return self.current_grid, float(reward), True, False, {
                "rollout_coverage_delta": final_progress - prev_progress,
                "rollout_steps": self.rollout_steps,
                "projected": info.get("projected", False),
            }
        else:
            # 即时奖励模式 (原始逻辑)
            visible = self.core.count_visible_cells(target_pos, target_yaw, frontier.cells)
            coverage = visible / max(frontier.frontier_size, 1)
            r_coverage = coverage * 5.0
            r_volume = visible * 0.005

            dist = np.linalg.norm(target_pos - center)
            r_dist = 0.5 * np.exp(-((dist - 2.0) ** 2) / (2 * 1.0 ** 2))

            reward = r_coverage + r_volume + r_dist

            return self.current_grid, float(reward), True, False, {
                "coverage": coverage,
                "visible": visible,
                "distance": dist,
            }
