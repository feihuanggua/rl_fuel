"""Plot evaluation results."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

data = json.load(open("./fuel_rl_checkpoints/sac_seq/eval_results.json"))

names_order = ["Random", "Closest", "Biggest", "Most Visible",
               "SAC-1000", "SAC-2000", "SAC-3000", "SAC-4000", "SAC-5000"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (metric, ylabel) in zip(axes, [
    ("cov_mean", "Coverage"),
    ("rew_mean", "Total Reward"),
    ("dist_mean", "Total Distance (m)"),
]):
    vals = [data[n][metric] for n in names_order]
    stds = [data[n].get(metric.replace("mean", "std"), 0) for n in names_order]
    colors = ["#888888"] * 4 + ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(range(len(names_order)), vals, yerr=stds, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names_order)))
    ax.set_xticklabels(names_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "./fuel_rl_checkpoints/sac_seq/eval_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Per-episode coverage box plot
all_covs = [data[n]["per_ep_cov"] for n in names_order]
bp = axes2[0].boxplot(all_covs, labels=names_order, patch_artist=True)
for patch, c in zip(bp["boxes"], ["#888888"]*4 + ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes2[0].set_ylabel("Coverage")
axes2[0].set_title("Coverage Distribution per Episode", fontweight="bold")
axes2[0].tick_params(axis="x", rotation=30)
axes2[0].grid(axis="y", alpha=0.3)

# Per-episode reward box plot
all_rews = [data[n]["per_ep_rew"] for n in names_order]
bp2 = axes2[1].boxplot(all_rews, labels=names_order, patch_artist=True)
for patch, c in zip(bp2["boxes"], ["#888888"]*4 + ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes2[1].set_ylabel("Total Reward")
axes2[1].set_title("Reward Distribution per Episode", fontweight="bold")
axes2[1].tick_params(axis="x", rotation=30)
axes2[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
out2 = "./fuel_rl_checkpoints/sac_seq/eval_boxplot.png"
plt.savefig(out2, dpi=150)
print(f"Saved to {out2}")

# Summary table
print("\n" + "="*80)
print(f"{'Policy':15s} {'Cov':>8s} {'±':>5s} {'Reward':>8s} {'±':>7s} {'Dist':>7s} {'±':>6s}")
print("-"*80)
for n in names_order:
    d = data[n]
    print(f"{n:15s} {d['cov_mean']:8.3f} {d['cov_std']:5.3f} "
          f"{d['rew_mean']:8.1f} {d['rew_std']:7.1f} "
          f"{d['dist_mean']:7.0f} {d.get('dist_std',0):6.0f}")
print("="*80)
