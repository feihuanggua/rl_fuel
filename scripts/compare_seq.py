import csv, numpy as np

for seq in ['sac_seq12', 'sac_seq13']:
    path = f'/home/jdwsl/rl_fuel/fuel_rl_checkpoints/{seq}/sac_log.csv'
    steps, rewards, covs = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            rewards.append(float(row['reward']))
            covs.append(float(row['coverage']))
    steps = np.array(steps)
    rewards = np.array(rewards)
    covs = np.array(covs)
    print(f"=== {seq} ===")
    for label, lo, hi in [("0-100k",0,100000),("100-250k",100000,250000),("250-400k",250000,400000),("400k+",400000,999999)]:
        mask = (steps > lo) & (steps <= hi)
        if mask.sum() > 0:
            r = rewards[mask]
            c = covs[mask]
            print(f"  {label}: reward mean={r.mean():.1f} med={np.median(r):.1f} std={r.std():.1f} | cov={c.mean():.3f}")
    print()
