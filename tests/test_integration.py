"""集成测试: 需要 C++ 核心 (fuel_rl_core) 已编译.

使用方法:
    # 纯 Python 单元测试 (无需 C++)
    pytest tests/test_unit.py -v

    # 集成测试 (需要编译 C++)
    pytest tests/test_integration.py -v

    # 全部
    pytest tests/ -v
"""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 检查 C++ 核心是否可用
try:
    from fuel_rl import FuelEnvCore
    from fuel_rl.config import default_map_params, default_frontier_params, fast_perception_params, default_astar_params
    from fuel_rl.map_loader import generate_random_map_for_fuel
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="C++ core not compiled")


# ═══════════════════════════════════════════════════════════════

class TestFuelEnvCore:
    """C++ 核心基本功能."""

    @pytest.fixture
    def core(self):
        c = FuelEnvCore()
        mp = default_map_params(
            size_x=20.0, size_y=20.0, size_z=3.0,
            box_min=(-9.0, -9.0, 0.0), box_max=(9.0, 9.0, 2.8),
        )
        c.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
        return c

    def test_init_success(self, core):
        assert core.get_resolution() > 0

    def test_load_random_map(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        # 应该有未知体素
        n_unk = core.count_unknown_voxels()
        assert n_unk > 0

    def test_initial_observation(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        start = np.array([0.0, 0.0, 1.5])
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            core.simulate_observation(start, yaw)
        # 四方向观测后应有自由和障碍物
        n_free = core.count_free_voxels()
        n_occ = core.count_occupied_voxels()
        assert n_free > 0
        assert n_occ > 0

    def test_frontier_detection(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
        frontiers = core.detect_frontiers(np.array([0.0, 0.0, 1.5]))
        assert isinstance(frontiers, list)
        for f in frontiers:
            assert f.frontier_size > 0
            assert len(f.cells) > 0
            assert abs(f.best_viewpoint_yaw) <= np.pi

    def test_occupancy_queries(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        # 地图中心应自由或未知
        occ = core.get_occupancy(np.array([0.0, 0.0, 1.5]))
        assert occ in (0, 1, 2)  # unknown=0, free=1, occupied=2

    def test_path_planning(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
        path = core.plan_path(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0.0, 1.5]))
        assert isinstance(path, list)

    def test_count_visible_cells(self, core):
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
        frontiers = core.detect_frontiers(np.array([0.0, 0.0, 1.5]))
        if frontiers:
            f = frontiers[0]
            visible = core.count_visible_cells(
                np.array(f.best_viewpoint_pos), f.best_viewpoint_yaw, f.cells,
            )
            assert visible >= 0

    def test_map_reuse(self, core):
        """验证 reset 后地图可以被重复使用而不会崩溃."""
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        for _ in range(3):
            core.reset_map()
            core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
            n = core.count_unknown_voxels()
            assert n > 0


class TestViewpointEnv:
    """视点选择环境."""

    @pytest.fixture
    def env(self):
        from fuel_rl.env.viewpoint_env import ViewpointEnv
        return ViewpointEnv(num_pillars=10)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (3, 32, 32, 10)
        assert 0 <= obs.min() <= obs.max() <= 1

    def test_step_with_zero_action(self, env):
        """零动作 (向中心看) 应产生有效结果."""
        obs, _ = env.reset(seed=42)
        obs, rew, done, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        # 可能因 no_path 失败, 但不应崩溃
        assert isinstance(rew, float)
        assert done in (True, False)

    def test_multiple_episodes(self, env):
        """多次 episode 不应崩溃 (验证 C++ 内存)."""
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            for _ in range(3):
                action = np.random.default_rng(ep).uniform(-0.5, 0.5, 4).astype(np.float32)
                obs, rew, done, trunc, info = env.step(action)
                if done:
                    break


class TestSequenceEnv:
    """序列探索环境."""

    @pytest.fixture
    def env(self):
        from fuel_rl.env.sequence_env import SequenceEnv
        return SequenceEnv(max_steps=20, num_pillars=10)

    def test_reset_shape(self, env):
        obs, _ = env.reset(seed=42)
        assert obs["frontiers"].shape == (50, 8)
        assert obs["mask"].shape == (50,)
        assert obs["global"].shape == (2,)
        assert obs["map_img"].shape == (3, 64, 64)

    def test_random_rollout(self, env):
        """随机策略 rollout 不应崩溃."""
        env.reset(seed=42)
        for step in range(20):
            n_valid = int(env._n_visible)
            if n_valid == 0:
                break
            action = np.random.randint(n_valid)
            obs, rew, done, trunc, info = env.step(action)
            assert isinstance(rew, float)
            if done:
                break


class TestBuild3ChannelGrid:
    """3 通道体素网格构建."""

    def test_output_shape(self):
        core = FuelEnvCore()
        mp = default_map_params(
            size_x=20.0, size_y=20.0, size_z=3.0,
            box_min=(-9.0, -9.0, 0.0), box_max=(9.0, 9.0, 2.8),
        )
        core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
        frontiers = core.detect_frontiers(np.array([0.0, 0.0, 1.5]))
        assert len(frontiers) > 0

        from fuel_rl.data.collector import build_3channel_grid
        grid = build_3channel_grid(core, frontiers[0])
        assert grid.shape == (3, 32, 32, 10)
        assert grid.dtype == np.float32
        # 前沿通道应非零
        assert grid[1].sum() > 0, "Frontier channel is empty"


class TestExpertLabel:
    """专家标签生成."""

    def test_label_range(self):
        core = FuelEnvCore()
        mp = default_map_params(
            size_x=20.0, size_y=20.0, size_z=3.0,
            box_min=(-9.0, -9.0, 0.0), box_max=(9.0, 9.0, 2.8),
        )
        core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
        pts = generate_random_map_for_fuel(20, 20, 3, 15, seed=42)
        core.load_map_from_points(pts)
        core.reset_map()
        core.simulate_observation(np.array([0.0, 0.0, 1.5]), 0.0)
        frontiers = core.detect_frontiers(np.array([0.0, 0.0, 1.5]))
        assert len(frontiers) > 0

        from fuel_rl.data.collector import get_expert_label
        label = get_expert_label(frontiers[0])
        assert label.shape == (4,)
        # dx, dy, dz 应在 [-1, 1] (距离不超过 max_dist=4.0)
        assert all(abs(label[i]) <= 1.0 for i in range(3)), f"label: {label}"
        # dyaw 应在 [-1, 1]
        assert abs(label[3]) <= 1.0


# ═══════════════════════════════════════════════════════════════
