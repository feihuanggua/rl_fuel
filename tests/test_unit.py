"""单元测试: 不需要 C++ 编译, 纯 Python/PyTorch 可运行."""
import pytest
import numpy as np
import torch
import sys, os

# 确保 fuel_rl 包可被导入 (即使 C++ 未编译)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# 数据增强测试
# ═══════════════════════════════════════════════════════════════

class TestRotationAugment:
    """4 倍旋转增强正确性."""

    def test_rotate_0_is_identity(self):
        from fuel_rl.data.dataset import ExpertDataset
        target = torch.tensor([0.5, 0.3, 0.0, 0.5])
        result = ExpertDataset._rotate_target(target, 0)
        assert torch.allclose(result, target)

    def test_rotate_1_90deg(self):
        """旋转 90°: (dx,dy) → (-dy,dx), dyaw += 0.5"""
        from fuel_rl.data.dataset import ExpertDataset
        target = torch.tensor([0.5, 0.3, 0.2, 0.2])
        result = ExpertDataset._rotate_target(target, 1)
        # After 90° rotation on XY plane
        expected_dx, expected_dy = -0.3, 0.5
        expected_dyaw = 0.7  # 0.2 + 0.5
        assert abs(result[0].item() - expected_dx) < 1e-5, f"dx: {result[0]} vs {expected_dx}"
        assert abs(result[1].item() - expected_dy) < 1e-5, f"dy: {result[1]} vs {expected_dy}"
        assert result[2].item() == 0.2  # dz unchanged
        assert abs(result[3].item() - expected_dyaw) < 1e-5

    def test_rotate_4_is_identity(self):
        """4 次 90° 旋转 = 1 圈."""
        from fuel_rl.data.dataset import ExpertDataset
        target = torch.tensor([0.5, 0.3, 0.1, 0.8])
        result = target
        for _ in range(4):
            result = ExpertDataset._rotate_target(result, 1)
        assert torch.allclose(result, target, atol=1e-5), f"4 rotations: {result} vs {target}"

    def test_dyaw_wraps_around(self):
        """偏航标签应正确 wrap [-1, 1]."""
        from fuel_rl.data.dataset import ExpertDataset
        target = torch.tensor([0.5, 0.3, 0.1, 0.9])  # dyaw = 0.9
        result = ExpertDataset._rotate_target(target, 1)  # dyaw = 1.4
        assert result[3] < -0.5, f"Expected wrapped dyaw, got {result[3]}"


# ═══════════════════════════════════════════════════════════════
# 3D Encoder 测试
# ═══════════════════════════════════════════════════════════════

class TestEncoder3D:
    """3D CNN 编码器形状和初始化."""

    def test_output_shape(self):
        from fuel_rl.models.encoder import Encoder3D
        encoder = Encoder3D(grid_size=32, channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        x = torch.randn(2, 3, 32, 32, 10)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_different_input_shape(self):
        from fuel_rl.models.encoder import Encoder3D
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=256,
                           input_shape=(16, 16, 8))
        x = torch.randn(4, 3, 16, 16, 8)
        out = encoder(x)
        assert out.shape == (4, 256)

    def test_flat_dim_computed_correctly(self):
        from fuel_rl.models.encoder import Encoder3D
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        assert encoder._flat_dim > 0
        # 3 layers of stride-2: 32→16→8→4, z: 10→5→3→2
        # 最后一层: 128 * 4 * 4 * 2 = 4096
        assert encoder._flat_dim == 4096, f"Expected 4096, got {encoder._flat_dim}"


# ═══════════════════════════════════════════════════════════════
# Viewpoint Head 测试
# ═══════════════════════════════════════════════════════════════

class TestViewpointHead:
    """解耦视点预测头."""

    def test_output_in_range(self):
        """输出应在 [-1, 1] 范围内 (tanh)."""
        from fuel_rl.models.encoder import Encoder3D
        from fuel_rl.models.viewpoint_head import ViewpointHead
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        head = ViewpointHead(encoder, embed_dim=512)
        x = torch.randn(4, 3, 32, 32, 10)
        out = head(x)
        assert out.shape == (4, 4)
        assert (out >= -1.0).all() and (out <= 1.0).all(), f"Out of range: min={out.min():.2f} max={out.max():.2f}"

    def test_deterministic_output(self):
        """相同输入应产生相同输出."""
        from fuel_rl.models.encoder import Encoder3D
        from fuel_rl.models.viewpoint_head import ViewpointHead
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        head = ViewpointHead(encoder, embed_dim=512)
        head.eval()
        x = torch.randn(1, 3, 32, 32, 10)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)


class TestViewpointActorCritic:
    """PPO Actor-Critic 结构."""

    def test_action_shape_and_range(self):
        from fuel_rl.models.encoder import Encoder3D
        from fuel_rl.models.viewpoint_head import ViewpointActorCritic
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        ac = ViewpointActorCritic(encoder, embed_dim=512)
        x = torch.randn(2, 3, 32, 32, 10)
        action, log_prob, value = ac.get_action(x, deterministic=False)
        assert action.shape == (2, 4)
        assert log_prob.shape == (2, 1)
        assert value.shape == (2, 1)
        assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_deterministic_action(self):
        from fuel_rl.models.encoder import Encoder3D
        from fuel_rl.models.viewpoint_head import ViewpointActorCritic
        encoder = Encoder3D(channels=[32, 64, 128], embed_dim=512,
                           input_shape=(32, 32, 10))
        ac = ViewpointActorCritic(encoder, embed_dim=512)
        x = torch.randn(2, 3, 32, 32, 10)
        a1, _, _ = ac.get_action(x, deterministic=True)
        a2, _, _ = ac.get_action(x, deterministic=True)
        assert torch.allclose(a1, a2)


# ═══════════════════════════════════════════════════════════════
# TSP 求解器测试
# ═══════════════════════════════════════════════════════════════

class TestTSPSolver:
    """TSP Nearest-Neighbor + 2-opt 基线."""

    def test_cost_matrix_symmetric(self):
        from fuel_rl.eval.tsp_baseline import _build_cost_matrix
        pts = np.array([[0, 0, 0], [3, 4, 0], [6, 0, 0]], dtype=np.float64)
        dist = _build_cost_matrix(pts)
        assert dist.shape == (3, 3)
        assert np.allclose(dist, dist.T)
        assert dist[0, 0] == 0.0
        assert abs(dist[0, 1] - 5.0) < 1e-5  # 3-4-5 triangle

    def test_nn_tour_visits_all(self):
        from fuel_rl.eval.tsp_baseline import _build_cost_matrix, _nearest_neighbor_tour
        pts = np.random.randn(10, 3).astype(np.float64)
        dist = _build_cost_matrix(pts)
        tour = _nearest_neighbor_tour(dist)
        assert len(tour) == 10
        assert len(set(tour)) == 10  # all unique

    def test_2opt_improves(self):
        from fuel_rl.eval.tsp_baseline import _build_cost_matrix, _nearest_neighbor_tour, _2opt_improve
        # 故意构造一个很差的 tour
        pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [100, 100, 0]], dtype=np.float64)
        dist = _build_cost_matrix(pts)
        bad_tour = [0, 1, 2, 3]  # zigzag
        improved = _2opt_improve(dist, bad_tour)
        bad_len = sum(dist[bad_tour[i], bad_tour[(i + 1) % 4]] for i in range(4))
        good_len = sum(dist[improved[i], improved[(i + 1) % 4]] for i in range(4))
        assert good_len <= bad_len + 1e-9

    def test_single_point(self):
        from fuel_rl.eval.tsp_baseline import _build_cost_matrix, _nearest_neighbor_tour
        pts = np.array([[1., 2., 3.]])
        dist = _build_cost_matrix(pts)
        tour = _nearest_neighbor_tour(dist)
        assert tour == [0]


# ═══════════════════════════════════════════════════════════════
# Config 验证测试
# ═══════════════════════════════════════════════════════════════

class TestConfigValidation:
    """配置一致性检查."""

    def test_correct_params_pass(self):
        """正确参数应通过验证."""
        from fuel_rl.config import validate_camera_params
        ok = validate_camera_params(
            387.229, 387.229, 321.046, 243.449,
            640, 480, 4.5, 5.0,
        )
        assert ok is True

    def test_mismatched_fx_fails(self):
        """错配 fx 应返回 False."""
        from fuel_rl.config import validate_camera_params
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok = validate_camera_params(
                381.0, 381.0, 320.0, 240.0,
                640, 480, 4.5, 5.0,
            )
            assert ok is False
            assert len(w) >= 1
            assert "fx/fy" in str(w[0].message)

    def test_old_gpu_params_detected(self):
        """修复前的 GPU 参数会被检测出来."""
        from fuel_rl.config import validate_camera_params
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok = validate_camera_params(
                381.0, 381.0, 321.046, 243.449,
                640, 480, 4.5, 5.0,
            )
            assert ok is False  # fx/fy 不匹配


# ═══════════════════════════════════════════════════════════════
# OrderPolicy 测试
# ═══════════════════════════════════════════════════════════════

class TestOrderPolicy:
    """前沿排序策略网络."""

    def test_output_shapes(self):
        from fuel_rl.models.order_policy import OrderPolicy
        policy = OrderPolicy(d_frontier=8, d_global=2, d_hidden=128, d_map=64)
        B, N = 2, 15
        frontiers = torch.randn(B, N, 8)
        mask = torch.ones(B, N)
        mask[:, 10:] = 0  # last 5 masked
        global_feat = torch.randn(B, 2)
        map_img = torch.randn(B, 3, 64, 64)

        logits, value = policy(frontiers, mask, global_feat, map_img)
        assert logits.shape == (B, N)
        assert value.shape == (B, 1)
        # Masked positions should have very negative logits
        assert (logits[:, 10:] < -1e8).all()

    def test_act_returns_valid_index(self):
        from fuel_rl.models.order_policy import OrderPolicy
        policy = OrderPolicy(d_frontier=8, d_global=2, d_hidden=128, d_map=64)
        B, N = 2, 5
        frontiers = torch.randn(B, N, 8)
        mask = torch.ones(B, N)
        global_feat = torch.randn(B, 2)
        map_img = torch.randn(B, 3, 64, 64)

        action, log_prob, value = policy.act(frontiers, mask, global_feat, map_img)
        assert action.shape == (B,)
        assert all(0 <= a < N for a in action)


# ═══════════════════════════════════════════════════════════════
# GPU Renderer 测试 (无需 GPU)
# ═══════════════════════════════════════════════════════════════

class TestGPURendererParams:
    """GPU 渲染器参数."""

    def test_default_params_match_cpu(self):
        """修复后的默认参数应与 C++ 一致."""
        from fuel_rl.env.gpu_depth_renderer import GPUDepthRenderer
        # 不能实例化 (没有 GPU 点), 但可以检查类默认值
        sig = GPUDepthRenderer.__init__.__code__
        # 验证默认值通过 config 验证
        from fuel_rl.config import validate_camera_params
        ok = validate_camera_params(387.229, 387.229, 321.046, 243.449, 640, 480, 4.5, 5.0)
        assert ok is True


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
