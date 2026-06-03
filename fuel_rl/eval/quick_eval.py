"""Quick eval script with output to file."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.eval.eval_policies import evaluate, random_policy, greedy_closest, greedy_biggest, greedy_visib, make_actor_policy
from fuel_rl.eval.tsp_baseline import tsp_policy, tsp_orienteering_policy, tsp_fuel_policy

env = SequenceEnv(max_steps=400, num_pillars=15)
policies = {
    "closest": greedy_closest,
    "tsp_nn2opt": tsp_policy,
    "tsp_fuel": tsp_fuel_policy,
}

sac_path = os.path.join(os.path.dirname(__file__), "..", "..", "fuel_rl_checkpoints", "sac_seq", "final_actor.pth")
sac_path = os.path.abspath(sac_path)
if os.path.exists(sac_path):
    policies["SAC-100k"] = make_actor_policy(sac_path)

n_ep = 20
results = {}
for name, pfn in policies.items():
    sys.stderr.write(f"  Running {name}...\n")
    sys.stderr.flush()
    t0 = time.time()
    r = evaluate(pfn, env, n_ep)
    el = time.time() - t0
    results[name] = {**r, "time": el}
    sys.stderr.write(f"  {name}: cov={r['cov']:.3f} dist={r['dist']:.1f} ({el:.0f}s)\n")
    sys.stderr.flush()

print("=" * 85)
print(f"{'Policy':15s} {'Cov':>8s} {'±':>4s} {'Reward':>8s} {'Steps':>7s} {'Dist':>8s} {'Time':>6s}")
print("-" * 85)
for name, r in results.items():
    print(f"{name:15s} {r['cov']:8.3f} {r['cov_std']:4.3f} "
          f"{r['rew']:8.1f} {r['steps']:7.1f} {r['dist']:8.1f}  {r['time']:.0f}s")
print("=" * 85)
