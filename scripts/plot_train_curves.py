"""Plot SAC seq12 training curves: reward, coverage, alpha."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "/home/jdwsl/rl_fuel/fuel_rl_checkpoints/sac_seq12/sac_log.csv"
out_dir = "/home/jdwsl/rl_fuel/vis_sac_seq12"

steps, rewards, coverages, alphas = [], [], [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        rewards.append(float(row["reward"]))
        coverages.append(float(row["coverage"]))
        alphas.append(float(row["alpha"]))

steps = np.array(steps)
rewards = np.array(rewards)
coverages = np.array(coverages)
alphas = np.array(alphas)

def smooth(arr, w=20):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode="valid")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Reward
axes[0].plot(steps, rewards, alpha=0.2, color="#e74c3c", linewidth=0.8)
s_rewards = smooth(rewards, 20)
axes[0].plot(steps[:len(s_rewards)], s_rewards, color="#e74c3c", linewidth=2, label="Smoothed (w=20)")
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_ylabel("Episode Reward")
axes[0].set_title("SAC Seq12 - Reward Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Coverage
axes[1].plot(steps, coverages, alpha=0.2, color="#3498db", linewidth=0.8)
s_cov = smooth(coverages, 20)
axes[1].plot(steps[:len(s_cov)], s_cov, color="#3498db", linewidth=2, label="Smoothed (w=20)")
axes[1].set_ylabel("Coverage")
axes[1].set_title("SAC Seq12 - Coverage Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Alpha
axes[2].plot(steps, alphas, color="#2ecc71", linewidth=1.5)
axes[2].set_ylabel("Alpha (entropy temp)")
axes[2].set_xlabel("Step")
axes[2].set_title("SAC Seq12 - Alpha Curve")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = f"{out_dir}/train_curves_seq12.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# Print stats
print(f"\nTotal episodes: {len(steps)}")
print(f"Steps range: {steps[0]} - {steps[-1]}")
print(f"\nReward stats:")
print(f"  Overall: mean={rewards.mean():.1f}, std={rewards.std():.1f}")
for label, mask in [("0-100k", steps <= 100000), ("100-200k", (steps > 100000) & (steps <= 200000)),
                     ("200-300k", (steps > 200000) & (steps <= 300000)), ("300-400k", (steps > 300000) & (steps <= 400000)),
                     ("400k+", steps > 400000)]:
    if mask.sum() > 0:
        r = rewards[mask]
        print(f"  {label}: mean={r.mean():.1f}, std={r.std():.1f}, median={np.median(r):.1f}")
print(f"\nCoverage stats:")
for label, mask in [("0-100k", steps <= 100000), ("100-200k", (steps > 100000) & (steps <= 200000)),
                     ("200-300k", (steps > 200000) & (steps <= 300000)), ("300-400k", (steps > 300000) & (steps <= 400000)),
                     ("400k+", steps > 400000)]:
    if mask.sum() > 0:
        c = coverages[mask]
        print(f"  {label}: mean={c.mean():.3f}, std={c.std():.3f}")
