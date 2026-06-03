"""诊断 SAC 序列策略: 可视化每步选择了哪个前沿、效果如何."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from fuel_rl import FuelEnvCore
from fuel_rl.config import (default_map_params, default_frontier_params,
                            fast_perception_params, default_astar_params, DEVICE)
from fuel_rl.map_loader import generate_random_map_for_fuel
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.env.sequence_env import SequenceEnv


def diagnose(seed=42, steps=50):
    model = OrderPolicy().to(DEVICE)
    # 试加载 SAC 模型
    try:
        model.load_state_dict(torch.load("./fuel_rl_checkpoints/sac_seq/final_actor.pth",
                                        map_location=DEVICE, weights_only=False))
        print("Loaded SAC model")
    except Exception:
        print("No SAC model, using random init")

    env = SequenceEnv(max_steps=steps, num_pillars=15, target_coverage=0.60)
    obs, _ = env.reset(seed=seed)

    history = []
    agent_positions = [env.agent_pos.copy()]

    for step in range(steps):
        frontiers = env.core.detect_frontiers(env.agent_pos)
        if not frontiers:
            break

        # SAC 策略选择
        frontiers_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
        mask_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
        global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(frontiers_t, mask_t, global_t)
            probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()

        n_valid = int(obs["mask"].sum())
        chosen = probs[:n_valid].argmax()

        obs, reward, done, _, info = env.step(chosen)
        progress = info["coverage"]

        history.append({
            "step": step,
            "n_frontiers": n_valid,
            "chosen": chosen,
            "probs": probs[:n_valid].copy(),
            "reward": reward,
            "coverage": progress,
            "agent_pos": env.agent_pos.copy() if reward > -1 else None,
        })
        agent_positions.append(env.agent_pos.copy())
        if done:
            break

    # ── 文本报告 ──
    print(f"Seed {seed}: {len(history)} steps, final cov={info.get('coverage',0):.1%}")
    print(f"  Invalid rate: {np.mean([1 if h['reward'] <= -0.1 else 0 for h in history]):.0%}")

    # ── 图 1: 覆盖曲线 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    covs = [h["coverage"] for h in history]
    ax.plot(covs, "b-o", markersize=3)
    ax.axhline(0.60, color="r", linestyle="--", label="60% target")
    ax.set_xlabel("Step"); ax.set_ylabel("Coverage"); ax.set_title("Coverage Progression")
    ax.legend()

    # ── 图 2: 每步选择分布 ──
    ax = axes[1]
    n_f = [h["n_frontiers"] for h in history]
    chosen = [h["chosen"] for h in history]
    ax.bar(range(len(history)), n_f, alpha=0.3, label="Total frontiers")
    ax.plot(chosen, "rx", markersize=5, label="Chosen")
    ax.set_xlabel("Step"); ax.set_ylabel("Index"); ax.set_title("Frontier Selection")
    ax.legend()

    # ── 图 3: 每步各前沿概率 ──
    ax = axes[2]
    first_few = history[:15]
    for i, h in enumerate(first_few):
        probs = h["probs"]
        if len(probs) > 0:
            ax.plot(range(len(probs)), probs, "o-", alpha=0.5, markersize=2, label=f"step {i}" if i < 5 else "")
    ax.set_xlabel("Frontier rank"); ax.set_ylabel("Prob"); ax.set_title("Action Probabilities (first 15 steps)")
    if len(first_few) <= 5:
        ax.legend(fontsize=6)

    fig.tight_layout()
    out = f"/tmp/sac_diag_seed{seed}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close("all")


if __name__ == "__main__":
    for seed in [42, 99, 777]:
        diagnose(seed, steps=50)
