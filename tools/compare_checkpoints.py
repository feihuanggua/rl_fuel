"""对比各阶段 SAC 策略的表现."""
import numpy as np
import torch, sys
from fuel_rl.env.sequence_env import SequenceEnv
from fuel_rl.models.order_policy import OrderPolicy
from fuel_rl.config import DEVICE

def compare(checkpoint, steps=50, num_runs=3):
    model = OrderPolicy().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=False))
    model.eval()

    covs, invalids, n_actions = [], [], []
    for seed in range(num_runs):
        env = SequenceEnv(max_steps=steps, num_pillars=15, target_coverage=0.60)
        obs, _ = env.reset(seed=seed)
        invalid = 0
        for s in range(steps):
            frontiers_t = torch.FloatTensor(obs["frontiers"]).unsqueeze(0).to(DEVICE)
            mask_t = torch.FloatTensor(obs["mask"]).unsqueeze(0).to(DEVICE)
            global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(frontiers_t, mask_t, global_t)
                probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
            n = int(obs["mask"].sum())
            if n == 0: break
            act = probs[:n].argmax()
            f = env.core.detect_frontiers(env.agent_pos)[act]
            vp = np.array(f.best_viewpoint_pos)
            occ = env.core.get_occupancy(vp)
            obs, r, done, _, info = env.step(act)
            if occ != 1: invalid += 1
            if done: break
        covs.append(info["coverage"])
        invalids.append(invalid)
    print(f"{checkpoint.split('/')[-1]:>20s}: cov={np.mean(covs):.1%} invalid={np.mean(invalids):.0f}/{steps}")

if __name__ == "__main__":
    base = "./fuel_rl_checkpoints/sac_seq2"
    for step in [1000, 2000, 3000, 4000, 5000, 6000]:
        compare(f"{base}/actor_{step}.pth")
