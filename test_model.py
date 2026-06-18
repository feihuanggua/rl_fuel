"""Quick test: attention model + new obs encoding."""
import sys, os
sys.path.insert(0, '/home/jdwsl/rl_fuel')
import torch
import numpy as np
from fuel_rl.models.order_policy import OrderPolicy, FrontierEncoder
from fuel_rl.env.sequence_env import SequenceEnv

mp = '/home/jdwsl/rl_fuel/maps/ariadne/5.png'
env = SequenceEnv(max_steps=10, map_size=(50, 40, 2), map_type='ariadne', map_path=mp)
obs, _ = env.reset(seed=42)
print(f'Obs shapes: frontiers={obs["frontiers"].shape} mask={obs["mask"].shape} '
      f'global={obs["global"].shape} map_img={obs["map_img"].shape}')

model = OrderPolicy()
n_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {n_params:,}')

fr = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(model.encoder.map_cnn.net[0].weight.device)
ms = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(fr.device)
gf = torch.FloatTensor(obs["global"]).unsqueeze(0).to(fr.device)
mi = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(fr.device)

logits, value = model(fr, ms, gf, mi)
print(f'logits: {logits.shape}, value: {value.shape}')
print(f'n_valid: {int(ms.sum())}')

action, log_prob, val = model.act(fr, ms, gf, mi, deterministic=True)
print(f'action: {action.item()}, log_prob: {log_prob.item():.4f}, value: {val.item():.4f}')

for i in range(5):
    nv = int(obs["mask"].sum())
    if nv == 0:
        print('no frontiers')
        break
    action, _, _ = model.act(fr, ms, gf, mi, deterministic=True)
    obs, r, done, _, info = env.step(int(action.item()))
    fr = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(model.encoder.map_cnn.net[0].weight.device)
    ms = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(fr.device)
    gf = torch.FloatTensor(obs["global"]).unsqueeze(0).to(fr.device)
    mi = torch.FloatTensor(obs["map_img"]).unsqueeze(0).to(fr.device)
    print(f'step {i+1}: reward={r:.2f} cov={info["coverage"]:.3f} dist={info["total_dist"]:.1f}')
    if done:
        print('done')
        break

print('\nTEST PASSED')
