"""Diagnose why episodes get shorter over training."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuel_rl.env.sequence_env import SequenceEnv

env = SequenceEnv(max_steps=100, num_pillars=15)

for ep in range(3):
    obs, _ = env.reset()
    total_rew = 0.0
    n_fails = 0
    for step in range(100):
        n_valid = int(obs["mask"].sum())
        if n_valid == 0:
            print(f"  ep{ep} step{step}: n_valid=0 -> done")
            break
        action = np.random.randint(n_valid)
        obs, rew, done, trunc, info = env.step(action)
        total_rew += rew
        if rew < -0.1:
            n_fails += 1
            if n_fails <= 3:
                print(f"    step{step}: FAIL rew={rew:.2f} n_valid={n_valid} action={action}")
        if done:
            print(f"  ep{ep} step{step}: done cov={info['coverage']:.3f} dist={info['total_dist']:.1f} n_fails={n_fails}")
            break
    print(f"  -> total_rew={total_rew:.2f} cov={info['coverage']:.3f} n_fails={n_fails}")
print(f"Random baseline done.")
