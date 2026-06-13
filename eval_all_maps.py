"""Evaluate SAC model on all ARiADNE maps: coverage + distance."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/home/jdwsl/rl_fuel')
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.config import DEVICE

ARIADNE_MAPS = ["1.png", "2.png", "3.png", "4.png", "5.png",
                "10.png", "20.png", "50.png", "100.png"]
MAPS_DIR = "/home/jdwsl/rl_fuel/maps/ariadne"

def eval_map(model, map_name, seed=42, max_steps=500):
    mp = os.path.join(MAPS_DIR, map_name)
    env = SequenceEnv(max_steps=max_steps, map_size=(20, 20, 2),
                       map_type='ariadne', map_path=mp)
    obs, _ = env.reset(seed=seed)
    for step in range(max_steps):
        nv = int(obs["mask"].sum())
        if nv == 0:
            cov = env._get_exploration_progress()
            return step, cov, env.total_distance
        f = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
        m = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
        g = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
        mi = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(f, m, g, mi)
        action = int(logits[0][:nv].argmax().item())
        obs, _, done, _, info = env.step(action)
        if done:
            cov = info.get("coverage", env._get_exploration_progress())
            return step + 1, cov, env.total_distance
    cov = env._get_exploration_progress()
    return max_steps, cov, env.total_distance

def eval_closest(map_name, seed=42, max_steps=500):
    mp = os.path.join(MAPS_DIR, map_name)
    env = SequenceEnv(max_steps=max_steps, map_size=(20, 20, 2),
                       map_type='ariadne', map_path=mp)
    obs, _ = env.reset(seed=seed)
    for step in range(max_steps):
        nv = int(obs["mask"].sum())
        if nv == 0:
            cov = env._get_exploration_progress()
            return step, cov, env.total_distance
        action = int(obs["frontiers"][:nv, 4].argmin())
        obs, _, done, _, info = env.step(action)
        if done:
            cov = info.get("coverage", env._get_exploration_progress())
            return step + 1, cov, env.total_distance
    cov = env._get_exploration_progress()
    return max_steps, cov, env.total_distance

models = {
    "seq8": "/home/jdwsl/rl_fuel/fuel_rl_checkpoints/sac_seq8/actor_200000.pth",
    "seq11_best": "/home/jdwsl/rl_fuel/fuel_rl_checkpoints/sac_seq11/best_actor.pth",
    "seq11_final": "/home/jdwsl/rl_fuel/fuel_rl_checkpoints/sac_seq11/final_actor.pth",
}

print(f"{'Map':<8} {'Closest':>20}    {'seq8':>20}    {'seq11_best':>20}")
print(f"{'':8} {'steps':>6} {'cov':>6} {'dist':>6}    {'steps':>6} {'cov':>6} {'dist':>6}    {'steps':>6} {'cov':>6} {'dist':>6}")

for mp in ARIADNE_MAPS:
    cs, cc, cd = eval_closest(mp)

    row = [f"{mp:<8} {cs:>6} {cc:>5.1%} {cd:>5.0f}m"]
    for label, path in models.items():
        model = OrderPolicy().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        model.eval()
        s, cov, dist = eval_map(model, mp)
        row.append(f"{s:>6} {cov:>5.1%} {dist:>5.0f}m")
    print("    ".join(row))

# Summary: average distance for maps that reach 90%+ (exclude 2,3)
print("\n=== Summary (excluding 2.png, 3.png unreachable) ===")
for label in ["Closest"] + list(models.keys()):
    dists = []
    for mp_name in ARIADNE_MAPS:
        if mp_name in ("2.png", "3.png"):
            continue
        if label == "Closest":
            s, c, d = eval_closest(mp_name)
        else:
            model = OrderPolicy().to(DEVICE)
            model.load_state_dict(torch.load(models[label], map_location=DEVICE, weights_only=False))
            model.eval()
            s, c, d = eval_map(model, mp_name)
        dists.append(d)
    print(f"  {label}: avg_dist={np.mean(dists):.0f}m")

print("\nDone.")
