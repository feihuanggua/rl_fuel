"""Plot training metrics from metrics.json."""
import json
import argparse
import os
import numpy as np
from typing import Optional


def smooth(values, weight: float = 0.6):
    """Exponential moving average smoothing."""
    if not values or all(v is None for v in values):
        return values
    out = []
    last = None
    for v in values:
        if v is None:
            out.append(last)
            continue
        if last is None:
            last = v
        else:
            last = last * weight + v * (1 - weight)
        out.append(last)
    return out


def plot_metrics(metrics_path: str, output_dir: Optional[str] = None, smooth_weight: float = 0.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(metrics_path) as f:
        data = json.load(f)

    steps = data["timesteps"]
    if not steps:
        print("No data to plot")
        return

    base = os.path.dirname(metrics_path)
    out = output_dir or base

    # --- Reward & Episode Length ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ep_rew = data.get("ep_rew_mean", [])
    ep_len = data.get("ep_len_mean", [])

    valid = [(s, r) for s, r in zip(steps, ep_rew) if r is not None]
    if valid:
        ss, rr = zip(*valid)
        axes[0].plot(ss, rr, alpha=0.3, color="steelblue", label="raw")
        sm = smooth(rr, smooth_weight)
        axes[0].plot(ss, sm, color="steelblue", linewidth=2, label=f"EMA({smooth_weight})")
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Episode Reward (mean)")
    axes[0].set_xlabel("Timesteps")
    axes[0].legend()

    valid2 = [(s, l) for s, l in zip(steps, ep_len) if l is not None]
    if valid2:
        ss2, ll = zip(*valid2)
        axes[1].plot(ss2, ll, alpha=0.3, color="darkorange")
        axes[1].plot(ss2, smooth(ll, smooth_weight), color="darkorange", linewidth=2)
    axes[1].set_title("Episode Length (mean)")
    axes[1].set_xlabel("Timesteps")

    plt.tight_layout()
    fig.savefig(os.path.join(out, "reward_curve.png"), dpi=150)
    print(f"Saved: {os.path.join(out, 'reward_curve.png')}")

    # --- Loss curves ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    loss_pairs = [
        ("loss", "Total Loss", "crimson"),
        ("policy_gradient_loss", "Policy Gradient Loss", "royalblue"),
        ("value_loss", "Value Loss", "forestgreen"),
        ("entropy_loss", "Entropy Loss", "darkorchid"),
    ]

    for ax, (key, title, color) in zip(axes2.flat, loss_pairs):
        vals = data.get(key, [])
        valid_v = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if valid_v:
            sv, vv = zip(*valid_v)
            ax.plot(sv, vv, alpha=0.3, color=color)
            ax.plot(sv, smooth(vv, smooth_weight), color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")

    plt.tight_layout()
    fig2.savefig(os.path.join(out, "loss_curves.png"), dpi=150)
    print(f"Saved: {os.path.join(out, 'loss_curves.png')}")

    # --- KL & Clip fraction ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    kl_vals = data.get("approx_kl", [])
    clip_vals = data.get("clip_fraction", [])
    kl_valid = [(s, v) for s, v in zip(steps, kl_vals) if v is not None]
    clip_valid = [(s, v) for s, v in zip(steps, clip_vals) if v is not None]

    if kl_valid:
        sk, kv = zip(*kl_valid)
        axes3[0].plot(sk, kv, alpha=0.3, color="teal")
        axes3[0].plot(sk, smooth(kv, smooth_weight), color="teal", linewidth=2)
    axes3[0].set_title("Approx KL Divergence")
    axes3[0].set_xlabel("Timesteps")

    if clip_valid:
        sc, cv = zip(*clip_valid)
        axes3[1].plot(sc, cv, alpha=0.3, color="coral")
        axes3[1].plot(sc, smooth(cv, smooth_weight), color="coral", linewidth=2)
    axes3[1].axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range")
    axes3[1].set_title("Clip Fraction")
    axes3[1].set_xlabel("Timesteps")
    axes3[1].legend()

    plt.tight_layout()
    fig3.savefig(os.path.join(out, "kl_clip.png"), dpi=150)
    print(f"Saved: {os.path.join(out, 'kl_clip.png')}")

    plt.close("all")

    # Print summary
    if valid:
        print(f"\n--- Summary ({len(steps)} data points) ---")
        print(f"  Reward: first={valid[0][1]:.1f}  last={valid[-1][1]:.1f}  "
              f"best={max(r for _, r in valid):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Plot FUEL RL training metrics")
    parser.add_argument("--metrics", type=str, default="./fuel_rl_tensorboard/metrics.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smooth", type=float, default=0.6)
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output_dir, args.smooth)


if __name__ == "__main__":
    main()
