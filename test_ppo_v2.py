"""测试 PPO v2 模型探索效果 — 含前沿去重和跳过策略."""
import numpy as np
import torch
from collections import defaultdict
from fuel_rl import FuelEnvCore
from fuel_rl.config import (
    default_map_params, default_frontier_params,
    default_perception_params, default_astar_params,
    ENCODER_CHANNELS, ENCODER_EMBED_DIM, DEVICE,
)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.data.collector import build_3channel_grid
from fuel_rl.models import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointActorCritic


def match_frontier(frontier, prev_frontiers):
    """通过位置匹配前后两轮的同一个前沿 (ID 会变)."""
    avg = np.array(frontier.average)
    for pf in prev_frontiers:
        if np.linalg.norm(np.array(pf.average) - avg) < 1.5:
            return pf.id
    return None


def test_model(model_path, max_steps=50, seed=42,
               max_visit=3, stale_threshold=50):
    # max_visit: 同一前沿最多连续访问次数
    # stale_threshold: 连续访问发现的体素低于此值则标记为 stale

    # 加载模型
    encoder = Encoder3D(input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointActorCritic(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"Model loaded: {model_path}")

    # 初始化环境
    core = FuelEnvCore()
    mp = default_map_params(
        size_x=20.0, size_y=20.0, size_z=3.0,
        box_min=(-9, -9, 0.0), box_max=(9, 9, 2.8),
    )
    core.init(mp, default_frontier_params(), default_perception_params(), default_astar_params())
    pts = generate_random_map_for_fuel(20.0, 20.0, 3.0, 15, seed=seed)
    core.load_map_from_points(pts)
    core.reset_map()

    agent_pos = np.array([0.0, 0.0, 1.5])
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        core.simulate_observation(agent_pos, yaw)

    max_dist = 4.0
    total_discovered = 0
    path = [agent_pos.copy()]
    invalid_count = 0
    stale_skips = 0

    # 前沿追踪: key=位置哈希 → {visits, total_discovered}
    frontier_tracker = defaultdict(lambda: {"visits": 0, "discovered": 0})
    stale_frontiers = set()  # 位置元组集合
    prev_frontiers = []

    # 位置哈希函数 (四舍五入到 0.5m 网格)
    def pos_key(avg):
        return (round(avg[0], 0), round(avg[1], 0), round(avg[2], 0))

    hdr = f"{'Step':>4} {'Found':>7} {'Progress':>10} {'Frontiers':>10}  {'Target':>8} {'Action':>30} {'Note':>10}"
    print(f"\n{hdr}")
    print("-" * 110)

    for step in range(max_steps):
        progress = core.get_exploration_progress()
        frontiers = core.detect_frontiers(agent_pos)
        if not frontiers:
            print(f"探索完成! {step}步, 覆盖率 {progress:.1%}")
            break

        # 分类前沿
        candidates = []
        for i, f in enumerate(frontiers):
            avg = np.array(f.average)
            dist = np.linalg.norm(avg - agent_pos)
            key = pos_key(avg)

            # 检查是否 stale
            is_stale = key in stale_frontiers
            visited_near = any(np.linalg.norm(avg - vp) < 1.0 for vp in path[-20:])
            is_near = dist < 0.3

            candidates.append({
                "idx": i, "f": f, "dist": dist, "key": key,
                "is_stale": is_stale, "visited_near": visited_near,
                "is_near": is_near,
                "visits": frontier_tracker[key]["visits"],
            })

        # 优先级排序:
        # 1. 非stale、非近距离、未访问过的 (距离最近)
        # 2. 非stale、近距离但访问次数少的
        # 3. stale 的 (最后选择)
        def priority(c):
            if not c["is_stale"] and not c["visited_near"] and not c["is_near"]:
                return (0, c["dist"])           # 最好: 新前沿
            elif not c["is_stale"] and c["visits"] < max_visit:
                return (1, c["dist"])           # 次选: 可重试
            elif not c["is_stale"]:
                return (2, c["dist"])           # 访问多但未 stale
            else:
                return (3, c["dist"])           # stale, 最后选

        candidates.sort(key=priority)
        target = candidates[0]["f"]
        target_key = candidates[0]["key"]
        note = ""

        # 如果选了 stale 前沿, 提示
        if candidates[0]["is_stale"]:
            stale_skips += 1
            note = "STALE!"
        elif candidates[0]["visits"] > 0:
            note = f"retry#{candidates[0]['visits']}"

        # 模型推理
        grid = build_3channel_grid(core, target)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, _std, _val = model(grid_t)
            action = mean.cpu().numpy().flatten()

        center = np.array(target.average)
        vp_pos = center + action[:3] * max_dist
        vp_pos[2] = np.clip(vp_pos[2], 0.5, 2.6)
        vp_yaw = action[3] * np.pi

        # 有效性检查
        occ = core.get_occupancy(vp_pos)
        if occ != 1:
            invalid_count += 1
            frontier_tracker[target_key]["visits"] += 1
            print(f"{step:4d} {'---':>7} {progress:>9.1%} {len(frontiers):>10}  "
                  f"#{target.id:>5} dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} "
                  f"yaw={action[3]:+.2f}  INVALID")
            prev_frontiers = frontiers
            continue

        path_cost = core.compute_path_cost(agent_pos, vp_pos)
        if path_cost < 0 or path_cost > 100:
            invalid_count += 1
            frontier_tracker[target_key]["visits"] += 1
            print(f"{step:4d} {'---':>7} {progress:>9.1%} {len(frontiers):>10}  "
                  f"#{target.id:>5} dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} "
                  f"yaw={action[3]:+.2f}  NO_PATH")
            prev_frontiers = frontiers
            continue

        # 执行观测
        prev_unk = core.count_unknown_voxels()
        core.simulate_observation(vp_pos, vp_yaw)
        new_unk = core.count_unknown_voxels()
        discovered = prev_unk - new_unk
        total_discovered += discovered

        # 更新前沿追踪
        frontier_tracker[target_key]["visits"] += 1
        frontier_tracker[target_key]["discovered"] += discovered

        # 判断是否 stale: 连续访问 max_visit 次且平均发现 < stale_threshold
        tracker = frontier_tracker[target_key]
        if tracker["visits"] >= max_visit:
            avg_discovered = tracker["discovered"] / tracker["visits"]
            if avg_discovered < stale_threshold:
                stale_frontiers.add(target_key)
                note = f"→STALE(avg={avg_discovered:.0f})"
            else:
                # 重置计数（前沿仍然有效）
                frontier_tracker[target_key] = {"visits": 0, "discovered": 0}

        agent_pos = vp_pos.copy()
        path.append(agent_pos.copy())
        prev_frontiers = frontiers

        if step % 5 == 0 or step < 3 or note:
            print(f"{step:4d} {discovered:>7.0f} {progress:>9.1%} {len(frontiers):>10}  "
                  f"#{target.id:>5} dx={action[0]:+.2f} dy={action[1]:+.2f} dz={action[2]:+.2f} "
                  f"yaw={action[3]:+.2f}  {note}")

    progress = core.get_exploration_progress()
    frontiers = core.detect_frontiers(agent_pos)
    valid_steps = len(path) - 1
    print(f"\n结果: {valid_steps}有效步/{max_steps}步, {invalid_count}次无效, "
          f"{stale_skips}次stale跳过, "
          f"发现{total_discovered}体素, 覆盖率{progress:.1%}, 剩余前沿{len(frontiers)}")
    print(f"Stale前沿: {len(stale_frontiers)}个")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./fuel_rl_checkpoints/ppo_v2/best_model.pth")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    test_model(args.model, args.max_steps, args.seed)
