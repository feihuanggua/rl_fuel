"""学术论文风格实验: 评估 BC vs REINFORCE vs 基线方法.

支持两种评估模式:
  --mode viewpoint : 单步视点质量评估 (原有逻辑)
  --mode sequence  : 多步探索序列评估 (新增, 包含 TSP 基线)

指标: 逐步探索覆盖率、有效步比例、平均视点质量
"""
import json, sys, argparse
import numpy as np
import torch
from collections import defaultdict

from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    fast_perception_params, default_astar_params,
    ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE,
)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid
from fuel_rl.models.encoder import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointHead


def load_bc_model(path="./fuel_rl_checkpoints/bc_v3/best_model.pth"):
    encoder = Encoder3D(input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointHead(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model


def load_rf_model(path="./fuel_rl_checkpoints/reinforce/best_model.pth"):
    from fuel_rl.train.train_reinforce import ReinforcePolicy
    model = ReinforcePolicy()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model


# ── 视点级策略 (ViewpointEnv 用) ──

def get_action_bc(model, core, frontier):
    grid = build_3channel_grid(core, frontier)
    grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model(grid_t).cpu().numpy().flatten()


def get_action_rf(model, core, frontier):
    grid = build_3channel_grid(core, frontier)
    grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
    mean, _ = model(grid_t)
    return mean.detach().cpu().numpy().flatten()


def get_action_greedy(core, frontier):
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_action_random(rng, core, frontier):
    return rng.uniform(-1, 1, 4).astype(np.float32)


def run_exploration(core, get_action_fn, max_steps=50, max_dist=4.0,
                    agent_start=np.array([0.0, 0.0, 1.5])):
    """多步探索 (视点级): 每步选最好前沿 → 模型选视点 → 观测 → 重复."""
    agent_pos = agent_start.copy()
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    progress_curve = []
    valid_steps = 0
    total_reward = 0.0
    discovered_per_step = []

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            break

        frontier = min(frontiers, key=lambda f: np.linalg.norm(np.array(f.average) - agent_pos))

        action = get_action_fn(core, frontier)
        center = np.array(frontier.average)
        vp = center + action[:3] * max_dist
        vp[2] = np.clip(vp[2], 0.5, 2.6)
        vp_yaw = action[3] * np.pi

        occ = core.get_occupancy(vp)
        if occ != 1:
            progress_curve.append(progress)
            continue

        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp, vp_yaw)
        discovered = prev_unk - core.count_unknown_voxels()

        visible = core.count_visible_cells(vp, vp_yaw, frontier.cells)
        coverage = visible / max(frontier.frontier_size, 1)
        dist = np.linalg.norm(vp - center)
        r_coverage = coverage * 5.0
        r_volume = visible * 0.005
        r_dist = 0.5 * np.exp(-((dist - 2.0) ** 2) / 2.0)
        reward = r_coverage + r_volume + r_dist

        total_reward += reward
        valid_steps += 1
        discovered_per_step.append(discovered)
        agent_pos = vp.copy()
        progress_curve.append(progress)

    final_progress = core.get_exploration_progress()
    avg_reward = total_reward / valid_steps if valid_steps > 0 else 0
    total_discovered = sum(discovered_per_step)

    return {
        "final_progress": final_progress,
        "valid_steps": valid_steps,
        "total_steps": max_steps,
        "avg_reward": avg_reward,
        "total_discovered": total_discovered,
        "progress_curve": progress_curve,
        "discovered_per_step": discovered_per_step,
    }


# ── 序列级策略 (SequenceEnv / 前沿排序 用) ──

def run_exploration_sequence(core, policy_fn, max_steps=50,
                              agent_start=np.array([0.0, 0.0, 1.5])):
    """多步探索 (序列级): 每步检测所有前沿 → 策略选择 → FUEL 原生视点 → 观测.

    policy_fn(core, frontiers, agent_pos) -> int (选中的前沿索引)
    """
    agent_pos = agent_start.copy()
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    progress_curve = []
    valid_steps = 0
    total_reward = 0.0
    total_distance = 0.0
    discovered_per_step = []

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            break

        # 策略选择前沿
        idx = policy_fn(core, frontiers, agent_pos)
        if idx < 0 or idx >= len(frontiers):
            progress_curve.append(progress)
            continue

        f = frontiers[idx]
        vp = np.array(f.best_viewpoint_pos)
        vp[2] = np.clip(vp[2], 0.5, 2.6)
        vy = f.best_viewpoint_yaw

        occ = core.get_occupancy(vp)
        if occ != 1:
            # fallback: 找最近的有效前沿
            fallback = None
            fb_dist = float("inf")
            for fi in frontiers:
                fvp = np.array(fi.best_viewpoint_pos)
                fvp[2] = np.clip(fvp[2], 0.5, 2.6)
                if core.get_occupancy(fvp) == 1:
                    d = np.linalg.norm(fvp - agent_pos)
                    if d < fb_dist:
                        fb_dist = d
                        fallback = fi
            if fallback is None:
                progress_curve.append(progress)
                continue
            f = fallback
            vp = np.array(f.best_viewpoint_pos)
            vp[2] = np.clip(vp[2], 0.5, 2.6)
            vy = f.best_viewpoint_yaw

        dist = np.linalg.norm(vp - agent_pos)
        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp, vy)
        discovered = prev_unk - core.count_unknown_voxels()

        total_reward += discovered * 0.01 - dist * 0.05
        total_distance += dist
        valid_steps += 1
        discovered_per_step.append(discovered)
        agent_pos = vp.copy()
        progress_curve.append(progress)

    final_progress = core.get_exploration_progress()
    avg_reward = total_reward / valid_steps if valid_steps > 0 else 0

    return {
        "final_progress": final_progress,
        "valid_steps": valid_steps,
        "total_steps": max_steps,
        "avg_reward": avg_reward,
        "total_distance": total_distance,
        "total_discovered": sum(discovered_per_step),
        "progress_curve": progress_curve,
        "discovered_per_step": discovered_per_step,
    }


# ── 序列级探索策略 ──

def _seq_closest(core, frontiers, agent_pos):
    """贪心最近: 总是选 Euclidean 距离最近的前沿视点."""
    best_i, best_d = 0, float("inf")
    for i, f in enumerate(frontiers):
        vp = np.array(f.best_viewpoint_pos)
        d = np.linalg.norm(vp - agent_pos)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _seq_biggest(core, frontiers, agent_pos):
    """贪心最大: 总是选体素最多的前沿."""
    return max(range(len(frontiers)), key=lambda i: frontiers[i].frontier_size)


def _seq_most_visible(core, frontiers, agent_pos):
    """贪心最可见: 总是选最佳可见体素最多的前沿."""
    return max(range(len(frontiers)), key=lambda i: frontiers[i].best_viewpoint_visib_num)


def _make_seq_tsp_nn2opt():
    """TSP Nearest-Neighbor + 2-opt 全局规划, 选 tour 中第一个前沿."""
    from fuel_rl.eval.tsp_baseline import _build_cost_matrix, _nearest_neighbor_tour, _2opt_improve

    def policy(core, frontiers, agent_pos):
        n = len(frontiers)
        if n <= 1:
            return 0
        vp_positions = np.array([np.array(f.best_viewpoint_pos) for f in frontiers])
        vp_positions[:, 2] = np.clip(vp_positions[:, 2], 0.5, 2.6)
        dists_to_robot = np.array([np.linalg.norm(vp - agent_pos) for vp in vp_positions])

        # 构建增广距离矩阵 [robot + frontiers]
        dist = _build_cost_matrix(vp_positions)
        augmented = np.zeros((n + 1, n + 1))
        augmented[1:, 1:] = dist
        augmented[0, 1:] = dists_to_robot
        augmented[1:, 0] = 0.0

        tour = _nearest_neighbor_tour(augmented, start=0)
        if len(tour) > 3:
            tour = _2opt_improve(augmented, tour, max_iter=150)

        for node in tour:
            if node != 0:
                return node - 1
        return 0

    return policy


def _make_seq_tsp_fuel():
    """FUEL 风格 TSP: 全局规划后在前3个 frontier 中选可见性最高的."""
    from fuel_rl.eval.tsp_baseline import _build_cost_matrix, _nearest_neighbor_tour, _2opt_improve

    def policy(core, frontiers, agent_pos):
        n = len(frontiers)
        if n <= 1:
            return 0
        vp_positions = np.array([np.array(f.best_viewpoint_pos) for f in frontiers])
        vp_positions[:, 2] = np.clip(vp_positions[:, 2], 0.5, 2.6)

        visib = np.array([f.best_viewpoint_visib_num for f in frontiers])
        dists_to_robot = np.array([np.linalg.norm(vp - agent_pos) for vp in vp_positions])

        dist = _build_cost_matrix(vp_positions)
        augmented = np.zeros((n + 1, n + 1))
        augmented[1:, 1:] = dist
        augmented[0, 1:] = dists_to_robot
        augmented[1:, 0] = 0.0

        tour = _nearest_neighbor_tour(augmented, start=0)
        if len(tour) > 3:
            tour = _2opt_improve(augmented, tour, max_iter=150)

        # 在前 3 个 TSP 节点中选 visibility 最高的
        top_k = min(3, n)
        candidates = []
        for node in tour[1:]:
            fi = node - 1
            if fi >= n:
                continue
            candidates.append(fi)
            if len(candidates) >= top_k:
                break
        if candidates:
            return max(candidates, key=lambda i: visib[i])
        return 0

    return policy


# ── 实验主逻辑 ──

def run_viewpoint_experiment(num_maps=50, max_steps=50, map_size=(20, 20, 3), num_pillars=15):
    """视点质量对比实验 (原有逻辑)."""
    methods = {}
    try:
        methods["BC"] = load_bc_model()
        print("BC model loaded")
    except Exception as e:
        print(f"BC load failed: {e}")
    try:
        methods["REINFORCE"] = load_rf_model()
        print("REINFORCE model loaded")
    except Exception as e:
        print(f"REINFORCE load failed: {e}")
    methods["Greedy"] = "greedy"
    methods["Random"] = "random"

    results = {name: defaultdict(list) for name in methods}
    rng = np.random.default_rng(42)

    for map_idx in range(num_maps):
        seed = 1000 + map_idx
        pts = generate_random_map_for_fuel(map_size[0], map_size[1], map_size[2],
                                            num_pillars, seed=seed)

        for name, model_or_type in methods.items():
            core = FuelEnvCore()
            mp = default_map_params(
                size_x=map_size[0], size_y=map_size[1], size_z=map_size[2],
                box_min=(-map_size[0]/2+1, -map_size[1]/2+1, 0.0),
                box_max=(map_size[0]/2-1, map_size[1]/2-1, map_size[2]-0.2),
            )
            core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
            core.load_map_from_points(pts)
            core.reset_map()

            if name == "BC":
                m = model_or_type
                f = lambda c, ft: get_action_bc(m, c, ft)
            elif name == "REINFORCE":
                m = model_or_type
                f = lambda c, ft: get_action_rf(m, c, ft)
            elif name == "Greedy":
                f = lambda c, ft: get_action_greedy(c, ft)
            elif name == "Random":
                f = lambda c, ft: get_action_random(rng, c, ft)
            else:
                continue

            stats = run_exploration(core, f, max_steps=max_steps)
            for k, v in stats.items():
                if isinstance(v, list):
                    results[name][k].append(v)
                else:
                    results[name][k].append(v)

        if (map_idx + 1) % 10 == 0:
            print(f"  {map_idx+1}/{num_maps} maps done")

    return results


def run_sequence_experiment(num_maps=50, max_steps=50, map_size=(20, 20, 3), num_pillars=15):
    """前沿排序策略对比实验: 贪心 vs TSP vs RL (如有).

    核心问题: 选择访问哪个前沿才更高效？
    策略必须基于完整前沿列表做决策，而非只看最近的一个。
    """
    methods = {
        "Closest": _seq_closest,
        "Biggest": _seq_biggest,
        "MostVisible": _seq_most_visible,
        "TSP_NN2OPT": _make_seq_tsp_nn2opt(),
        "TSP_FUEL": _make_seq_tsp_fuel(),
    }

    print(f"Sequence experiment methods: {list(methods.keys())}")

    results = {name: defaultdict(list) for name in methods}

    for map_idx in range(num_maps):
        seed = 1000 + map_idx
        pts = generate_random_map_for_fuel(map_size[0], map_size[1], map_size[2],
                                            num_pillars, seed=seed)

        for name, policy_fn in methods.items():
            core = FuelEnvCore()
            mp = default_map_params(
                size_x=map_size[0], size_y=map_size[1], size_z=map_size[2],
                box_min=(-map_size[0]/2+1, -map_size[1]/2+1, 0.0),
                box_max=(map_size[0]/2-1, map_size[1]/2-1, map_size[2]-0.2),
            )
            core.init(mp, default_frontier_params(), fast_perception_params(), default_astar_params())
            core.load_map_from_points(pts)
            core.reset_map()

            stats = run_exploration_sequence(core, policy_fn, max_steps=max_steps)
            for k, v in stats.items():
                if isinstance(v, list):
                    results[name][k].append(v)
                else:
                    results[name][k].append(v)

        if (map_idx + 1) % 10 == 0:
            print(f"  {map_idx+1}/{num_maps} maps done")

    return results


# ── 汇总统计 ──

def summarize(results, title="实验结果汇总"):
    """汇总统计：均值±标准差."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    scalar_metrics = ["final_progress", "valid_steps", "avg_reward", "total_discovered", "total_distance"]
    # 过滤实际存在的指标
    first_key = next(iter(results.keys()))
    available = [m for m in scalar_metrics if m in results[first_key]]

    header = f"{'Method':<15}"
    for m in available:
        header += f" {m:>16}"
    print(header)
    print("-" * 70)

    # Sort by final_progress descending for clarity
    ordered = sorted(results.keys(), key=lambda n: np.mean(results[n].get("final_progress", [0])), reverse=True)

    for name in ordered:
        if name not in results:
            continue
        r = results[name]
        row = f"{name:<15}"
        for m in available:
            vals = np.array(r[m])
            row += f" {vals.mean():>8.3f}±{vals.std():<6.3f}"
        print(row)

    # 覆盖率曲线对比
    if "progress_curve" in results[first_key]:
        print(f"\n{'Method':<15} {'0%':>5} {'25%':>6} {'50%':>6} {'75%':>6} {'99%':>6}")
        print("-" * 50)
        for name in ordered:
            if name not in results:
                continue
            curves = results[name]["progress_curve"]
            max_len = min(50, max(len(c) for c in curves))
            avg_curve = np.zeros(max_len)
            for c in curves:
                for i in range(min(len(c), max_len)):
                    avg_curve[i] += c[i]
            avg_curve /= len(curves)

            def q(x):
                idx = int(x * (len(avg_curve) - 1))
                return avg_curve[min(idx, len(avg_curve)-1)]

            row = f"{name:<15}"
            for pct in [0, 0.25, 0.5, 0.75, 0.99]:
                row += f" {q(pct):>5.1%}"
            print(row)

    return results


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-FUEL Experiment Runner")
    parser.add_argument("--mode", type=str, default="viewpoint",
                        choices=["viewpoint", "sequence"],
                        help="viewpoint = 单步视点质量; sequence = 多步前沿排序 (含 TSP)")
    parser.add_argument("--num-maps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="/tmp/rl_fuel_experiment.json")
    args = parser.parse_args()

    if args.mode == "viewpoint":
        print(f"视点质量实验: {args.num_maps} maps, {args.max_steps} steps")
        results = run_viewpoint_experiment(num_maps=args.num_maps, max_steps=args.max_steps)
    else:
        print(f"前沿排序实验: {args.num_maps} maps, {args.max_steps} steps")
        results = run_sequence_experiment(num_maps=args.num_maps, max_steps=args.max_steps)

    # 保存原始数据
    json_data = {}
    for name, data in results.items():
        json_data[name] = {k: v for k, v in data.items() if not isinstance(v[0], (list, np.ndarray))}
    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2, default=lambda x: float(x))
    print(f"\nRaw data saved to {args.output}")

    title = "视点质量对比 (Viewpoint Quality)" if args.mode == "viewpoint" else "前沿排序策略对比 (Frontier Ordering)"
    summarize(results, title)
