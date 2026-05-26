"""学术论文风格实验: 评估 BC vs REINFORCE vs 基线方法.

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
    """多步探索: 每步选最好前沿 → 模型选视点 → 观测 → 重复."""
    agent_pos = agent_start.copy()
    # 四方向初始观测
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

        # 选最近的可能前沿
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

        # 计算视点质量奖励
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


def run_experiment(num_maps=50, max_steps=50, map_size=(20, 20, 3), num_pillars=15):
    """主实验: 多个地图, 多种方法, 收集统计数据."""
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


def summarize(results):
    """汇总统计：均值±标准差."""
    print("\n" + "=" * 70)
    print("实验结果汇总 (均值 ± 标准差)")
    print("=" * 70)

    scalar_metrics = ["final_progress", "valid_steps", "avg_reward", "total_discovered"]
    header = f"{'Method':<12}"
    for m in scalar_metrics:
        header += f" {m:>16}"
    print(header)
    print("-" * 70)

    for name in ["BC", "REINFORCE", "Greedy", "Random"]:
        if name not in results:
            continue
        r = results[name]
        row = f"{name:<12}"
        for m in scalar_metrics:
            vals = np.array(r[m])
            row += f" {vals.mean():>8.3f}±{vals.std():<6.3f}"
        print(row)

    # 覆盖率曲线对比
    print(f"\n{'Method':<12} {'0%':>5} {'25%':>6} {'50%':>6} {'75%':>6} {'99%':>6}")
    print("-" * 50)
    for name in ["BC", "REINFORCE", "Greedy", "Random"]:
        if name not in results:
            continue
        curves = results[name]["progress_curve"]
        # 对齐长度，取每步平均值
        max_len = min(50, max(len(c) for c in curves))
        avg_curve = np.zeros(max_len)
        for c in curves:
            for i in range(min(len(c), max_len)):
                avg_curve[i] += c[i]
        avg_curve /= len(curves)

        def q(x):
            # 插值取分位数步的覆盖率
            idx = int(x * (len(avg_curve) - 1))
            return avg_curve[min(idx, len(avg_curve)-1)]

        row = f"{name:<12}"
        for pct in [0, 0.25, 0.5, 0.75, 0.99]:
            row += f" {q(pct):>5.1%}"
        print(row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-maps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=str, default="/tmp/rl_fuel_experiment.json")
    args = parser.parse_args()

    print(f"实验设置: {args.num_maps} maps, {args.max_steps} steps")
    results = run_experiment(num_maps=args.num_maps, max_steps=args.max_steps)

    # 保存原始数据
    json_data = {}
    for name, data in results.items():
        json_data[name] = {k: v for k, v in data.items() if not isinstance(v[0], (list, np.ndarray))}
    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2, default=lambda x: float(x))
    print(f"\nRaw data saved to {args.output}")

    summarize(results)
